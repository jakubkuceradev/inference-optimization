"""Bounded engineering probes. Scratch weights never enter study artifacts."""
import time
from pathlib import Path

import numpy as np
import torch

from ..common import environment, external, write_json
from ..model import MLPLOB
from .data import Development
from .plan import identity, model_config, positive
from .store import activity, provenance
from .training import device_check


def preflight(plan, data, output, device, seconds=60):
    positive(seconds); device_check(device)
    corpus = Development(data)
    if (plan["kind"]=="smoke") != (corpus.manifest["kind"]=="synthetic-smoke"):
        raise ValueError("Smoke plans and synthetic data must be paired")
    output = external(output); output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    batches = (8,16) if plan["kind"]=="smoke" else (16,32,64,128,256,512) if any(j["history"]>=256 for j in plan["jobs"]) else (256,512)
    measurements = []
    with activity():
        # Measure the largest <=128-history model for conservative small-shape estimates.
        small = [j for j in plan["jobs"] if j["history"]<=128]
        representative = max(small,key=lambda j:j["history"]*j["width"]*j["depth"]* (2 if j["expansion"]=="standard" else 1)) if small else dict(plan["jobs"][0],id="preflight-small-representative",history=64,width=64,depth=2)
        for batch in batches:
            if time.perf_counter()-started >= seconds: break
            model=optimizer=x=y=None
            try:
                model = MLPLOB(**model_config(representative)).to(device)
                optimizer = torch.optim.Adam(model.parameters(),lr=.0003,foreach=False,fused=False)
                x = torch.from_numpy(corpus.windows("train",np.arange(batch)%len(corpus.arrays["train"]["index"]),representative["history"],representative["features"])).to(device)
                y = torch.from_numpy(np.array(corpus.arrays["train"]["targets"][np.arange(batch)%len(corpus.arrays["train"]["index"])] )).to(device)
                if device=="cuda": torch.cuda.reset_peak_memory_stats()
                t=time.perf_counter()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(model(x),y)
                    if not torch.isfinite(loss): raise FloatingPointError("Preflight nonfinite loss")
                    loss.backward(); optimizer.step()
                if device=="cuda": torch.cuda.synchronize()
                step_seconds=(time.perf_counter()-t)/2
                model.eval(); t=time.perf_counter()
                with torch.inference_mode(): model(x)
                if device=="cuda": torch.cuda.synchronize()
                val_seconds=time.perf_counter()-t
                measurements.append(dict(job_id=representative["id"],batch=batch,status="supported",evidence="two scratch optimizer steps",step_seconds=step_seconds,
                    validation_batch_seconds=val_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None))
            except torch.OutOfMemoryError:
                measurements.append(dict(job_id=representative["id"],batch=batch,status="unsupported",evidence="scratch allocation failed"))
            del model,optimizer,x,y
            if device=="cuda": torch.cuda.empty_cache()
        rows=[]
        for job in plan["jobs"]:
            model = MLPLOB(**model_config(job)); parameters = sum(p.numel() for p in model.parameters()); del model
            for batch in batches:
                measured = next((m for m in measurements if m["batch"]==batch and m["status"]=="supported"),None)
                # Includes parameters/Adam/gradients and conservative activation/workspace allowance.
                estimate = parameters*32+batch*job["history"]*job["width"]*job["depth"]*256
                available = torch.cuda.mem_get_info()[0] if device=="cuda" else __import__('os').sysconf('SC_AVPHYS_PAGES')*__import__('os').sysconf('SC_PAGE_SIZE')
                row=dict(job_id=job["id"],batch=batch,status="supported" if measured and estimate<available*.7 else "unsupported",
                         evidence="measured representative plus conservative memory/work scaling; no reference fit",estimated_memory_bytes=estimate,available_memory_bytes=available)
                if measured:
                    scale=max(1., job["history"]*job["width"]**2*job["depth"]/(representative["history"]*representative["width"]**2*representative["depth"]))
                    row.update(estimated_epoch_fitting_seconds=measured["step_seconds"]*np.ceil(len(corpus.arrays['train']['index'])/batch)*scale,
                               estimated_full_validation_seconds=measured["validation_batch_seconds"]*np.ceil(len(corpus.arrays['validation']['index'])/batch)*scale,
                               measurement_calls=plan["timing"]["repeats"]*plan["timing"]["calls"]*2,
                               estimated_timing_seconds=measured["validation_batch_seconds"]*scale*plan["timing"]["calls"]*plan["timing"]["repeats"]*2,
                               timing_estimate_qualification="batched eager service time conservative proxy; runtime setup and actual native/graph latency must be measured")
                rows.append(row)
        result=dict(status="preflight-recorded",plan_hash=identity(plan),data_hash=corpus.binding,code=provenance(),device=device,
                    environment=environment(),seconds=time.perf_counter()-started,requested_probe_seconds=seconds,
                    bound="stop before next probe after deadline; current two-step probe completes",batches=rows,measurements=measurements,
                    scratch_only=True,selection="no validation F1 or learning-rate/model selection")
        write_json(output/"preflight.json",result)
        return result
