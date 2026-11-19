"""Frozen-runtime validation, timed-output audits and separate stage diagnostics."""
import copy
import time
from pathlib import Path

import numpy as np
import torch

from ..common import digest, environment, hardware_identity, read_json, write_json
from ..metrics import compare, latency_summary
from .data import Development
from .plan import identity
from .runtime import adapter, select_runtime, fingerprint, Unsupported
from .store import activity, job_path, lock, provenance
from .training import quality, selected, predict


def audit(reference,candidate,ids):
    result = compare(reference,candidate,3e-5,3e-4)
    passing = (np.isfinite(reference)&np.isfinite(candidate)&(np.abs(candidate-reference)<=3e-5+3e-4*np.abs(reference))).all(1)
    result.update(status="passed" if passing.all() else "numerical-check-failed",failing_ids=np.asarray(ids)[~passing].tolist())
    return result


def binding(study,job,checkpoint,device):
    return dict(job_key=job["key"],checkpoint_sha256=checkpoint["sha256"],data_hash=study["data_hash"],
                code=provenance(),device=device,hardware=hardware_identity(device),precision="FP32; TF32 off",timing=study["plan"]["timing"])


def canonical(study,job,model,checkpoint,corpus,create=False):
    """Freeze one CPU oracle; CUDA interpreters consume it without regenerating."""
    root=job_path(study,job)/"canonical"
    manifest_path=root/"reference.json"
    bound=dict(job_key=job["key"],checkpoint_sha256=checkpoint["sha256"],data_hash=corpus.binding,
               source_hashes=provenance()["source_hashes"],schedule=study["plan"]["timing"],threads=1)
    if not manifest_path.exists():
        if not create: raise ValueError("Run CPU evaluation first to publish the canonical CPU oracle")
        if torch.get_num_threads()!=1: raise ValueError("Canonical CPU oracle requires one thread")
        root.mkdir(parents=True,exist_ok=True)
        config=job["config"]; a=corpus.arrays["validation"]
        reference=predict(model,corpus,config,batch=256)
        schedule=study["plan"]["timing"]
        positions=np.random.default_rng(schedule["seed"]).integers(0,len(a["index"]),size=schedule["calls"])
        timed=np.empty((len(positions),3),np.float32)
        with torch.inference_mode():
            for i,p in enumerate(positions):
                timed[i]=model(torch.from_numpy(corpus.windows("validation",[p],config["history"],config["features"]))).numpy()[0]
        # A reference with nonfinites still remains evidence for the fixed gate.
        np.savez(root/"reference.npz",reference=reference,ids=a["index"],targets=a["targets"],
                 timed_reference=timed,positions=positions,timed_ids=a["index"][positions])
        write_json(manifest_path,dict(binding=bound,sha256=digest(root/"reference.npz"),environment=environment(),
                                     full_validation_batch=256,timed_batch=1,device="cpu"))
    metadata=read_json(manifest_path)
    if metadata["binding"]!=bound or digest(root/"reference.npz")!=metadata["sha256"]:
        raise ValueError("Canonical CPU oracle binding/hash mismatch")
    with np.load(root/"reference.npz",allow_pickle=False) as saved:
        result={k:saved[k].copy() for k in saved.files}
    if not np.array_equal(result["ids"],corpus.arrays["validation"]["index"]) or not np.array_equal(result["targets"],corpus.arrays["validation"]["targets"]):
        raise ValueError("Canonical oracle target mismatch")
    return result,metadata


def evaluate(study,job,device,include=None):
    directory=job_path(study,job)
    with activity(measurement=True),lock(directory/"mutation.lock"):
        model,checkpoint=selected(directory,job)
        corpus=Development(study["data"])
        a=corpus.arrays["validation"]; config=job["config"]
        output=directory/device; output.mkdir(parents=True,exist_ok=True)
        bound=binding(study,job,checkpoint,device)
        started=time.perf_counter()
        oracle,oracle_metadata=canonical(study,job,model,checkpoint,corpus,create=True) if device=="cpu" else (None,None)
        try:
            runtime,selection=select_runtime(device,model,config,output,corpus,include)
        except Unsupported as error:
            result=dict(status="unsupported",reason=str(error),binding=bound,environment=environment())
            write_json(output/"evaluation.json",result)
            return result
        except Exception as error:
            write_json(output/"evaluation.json",dict(status="failed",binding=bound,error=repr(error),environment=environment()))
            raise
        try:
            if oracle is None: oracle,oracle_metadata=canonical(study,job,model,checkpoint,corpus)
            reference=oracle["reference"]
            logits=np.empty((len(a["index"]),3),np.float32)
            for i in range(len(logits)):
                _,value,_=runtime.call(corpus.windows("validation",[i],config["history"],config["features"]))
                logits[i]=value[0]
            numerical=audit(reference,logits,a["index"])
            np.savez(output/"validation.npz",reference=reference,logits=logits,ids=a["index"],targets=a["targets"])
            result=dict(status="completed",numerical_status=numerical["status"],binding=bound,runtime=runtime.metadata,
                        numerical=numerical,quality=quality(a["targets"],logits,a["index"]) if np.isfinite(logits).all() else None,
                        canonical_reference=oracle_metadata,
                        runtime_selection=selection,
                        environment=environment(),seconds=time.perf_counter()-started,predictions_sha256=digest(output/"validation.npz"))
            write_json(output/"evaluation.json",result)
            return result
        except Exception as error:
            write_json(output/"evaluation.json",dict(status="failed",binding=bound,error=repr(error),environment=environment()))
            raise
        finally: runtime.close()


def benchmark(study,job,device,include=None):
    directory=job_path(study,job)
    with activity(measurement=True),lock(directory/"mutation.lock"):
        model,checkpoint=selected(directory,job)
        output=directory/device; output.mkdir(parents=True,exist_ok=True)
        bound=binding(study,job,checkpoint,device)
        evaluation=read_json(output/"evaluation.json")
        if evaluation["binding"]!=bound: raise ValueError("Stale runtime evaluation")
        if evaluation["status"]=="unsupported":
            result=dict(status="unsupported",reason=evaluation["reason"],binding=bound)
            write_json(output/"benchmark.json",result); return result
        if digest(output/"validation.npz")!=evaluation["predictions_sha256"]: raise ValueError("Validation artifact changed")
        corpus=Development(study["data"]); config=job["config"]; schedule=study["plan"]["timing"]
        oracle,oracle_metadata=canonical(study,job,model,checkpoint,corpus)
        positions=oracle["positions"]; ids=oracle["timed_ids"]; refs=oracle["timed_reference"]
        runtime=adapter(device,copy.deepcopy(model),config,output/"timing-runtime",include,runtime=evaluation["runtime"]["runtime"])
        try:
            if fingerprint(runtime.metadata)!=fingerprint(evaluation["runtime"]):
                raise ValueError("Runtime binary/export/provider changed since validation")
            # Deterministic boundary interleave, shared between devices and jobs.
            rng=np.random.default_rng(schedule["seed"])
            order=[(repeat,boundary) for repeat in range(schedule["repeats"]) for boundary in rng.permutation(["primary","prepared"]).tolist()]
            for boundary in ("primary","prepared"):
                for i in range(schedule["warmups"]):
                    runtime.call(corpus.windows("validation",[positions[i%len(positions)]],config["history"],config["features"]),boundary)
            # Setup/validation/warmup finished; all calls below are unprofiled.
            records=[]; gates=[]
            for repeat,boundary in order:
                ns=np.empty(len(positions),np.float64); logits=np.empty_like(refs); decisions=np.empty(len(positions),np.int64)
                for i,p in enumerate(positions):
                    # Array slicing/disk reads and layout preparation are explicitly outside the boundary.
                    x=corpus.windows("validation",[p],config["history"],config["features"])
                    ns[i],value,decisions[i]=runtime.call(x,boundary)
                    logits[i]=value[0]
                numerical=audit(refs,logits,ids); gates.append(numerical["status"])
                name=f"timed-{boundary}-{repeat}.npz"
                np.savez(output/name,ns=ns,logits=logits,decisions=decisions,ids=ids,reference=refs)
                records.append(dict(repeat=repeat,boundary=boundary,latency=latency_summary(ns),numerical=numerical,file=name,sha256=digest(output/name)))
            summaries={}
            for boundary in ("primary","prepared"):
                rows=[r for r in records if r["boundary"]==boundary]
                summaries[boundary]=dict(median_repeat_p50_us=float(np.median([r["latency"]["p50_us"] for r in rows])),worst_repeat_p99_us=max(r["latency"]["p99_us"] for r in rows))
            result=dict(status="completed",numerical_status="passed" if evaluation["numerical_status"]=="passed" and all(g=="passed" for g in gates) else "numerical-check-failed",
                        binding=bound,runtime=runtime.metadata,records=records,summary=summaries,order=order,
                        environment=environment(),canonical_reference=oracle_metadata,input_schedule_sha256=identity(ids.tolist()),
                        boundary=dict(primary="available contiguous full-window -> runtime input copy -> complete BiN/model -> CPU argmax; CUDA includes H2D/D2H/sync",prepared="runtime input copying before timer; CUDA output transfer and synchronization remain inside"),
                        limitations="Runner activity lock excludes our jobs only; unrelated host load uncontrolled")
            write_json(output/"benchmark.json",result); return result
        except Exception as error:
            write_json(output/"benchmark.json",dict(status="failed",binding=bound,error=repr(error),environment=environment()))
            raise
        finally: runtime.close()


def profile(study,job,device,calls=100):
    from .training import device_check
    directory=job_path(study,job)
    with activity(),lock(directory/"mutation.lock"):
        model,checkpoint=selected(directory,job)
        try: device_check(device)
        except RuntimeError as error:
            output=directory/device; output.mkdir(parents=True,exist_ok=True)
            result=dict(status="unsupported",binding=binding(study,job,checkpoint,device),reason=str(error),environment=environment())
            write_json(output/"profile.json",result); return result
        model=model.to(device)
        corpus=Development(study["data"]); config=job["config"]
        x=torch.from_numpy(corpus.windows("validation",[0],config["history"],config["features"])).to(device)
        def sync():
            if device=="cuda": torch.cuda.synchronize()
        samples={}; starts={}; handles=[]
        modules={"BiN":model.norm_layer,"input_projection":model.first_layer,"head":model.final_layers}
        # ModuleList has no forward; use first/last head layer hooks as one span.
        modules.pop("head")
        for i in range(config["depth"]):
            modules[f"feature_{i}"]=model.layers[2+2*i]; modules[f"time_{i}"]=model.layers[3+2*i]
        def pre(name):
            def hook(*args): sync(); starts[name]=time.perf_counter_ns()
            return hook
        def post(name):
            def hook(*args): sync(); samples.setdefault(name,[]).append(time.perf_counter_ns()-starts[name])
            return hook
        with torch.inference_mode():
            for _ in range(5): model(x)
            sync(); t=time.perf_counter_ns()
            for _ in range(calls): model(x)
            sync(); baseline=(time.perf_counter_ns()-t)/calls
            for name,module in modules.items():
                handles.extend((module.register_forward_pre_hook(pre(name)),module.register_forward_hook(post(name))))
            handles.extend((model.final_layers[0].register_forward_pre_hook(pre("head")),model.final_layers[-1].register_forward_hook(post("head"))))
            t=time.perf_counter_ns()
            try:
                for _ in range(calls): model(x)
                sync()
            finally:
                for handle in handles: handle.remove()
            profiled=(time.perf_counter_ns()-t)/calls
        output=directory/device; output.mkdir(parents=True,exist_ok=True)
        result=dict(status="completed",binding=binding(study,job,checkpoint,device),diagnostic="eager framework hooks; synchronization at boundaries",environment=environment(),
                    stages={k:latency_summary(v) for k,v in samples.items()},unhooked_mean_ns=baseline,hooked_mean_ns=profiled,overhead_ns=profiled-baseline,
                    warning="Overlapping/instrumented spans are not summed into unprofiled service latency")
        write_json(output/"profile.json",result); return result
