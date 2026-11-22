"""Deterministic epoch transactions, raw-weight selection and budget recovery."""
import copy
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ..common import digest, environment, read_json, write_json
from ..metrics import score, softmax
from ..model import MLPLOB
from .data import Development
from .plan import model_config
from .store import activity, atomic_torch, job_path, lock, provenance

TERMINAL = ("completed", "early-stopped", "budget-stopped")


def device_check(device):
    if device not in ("cpu","cuda") or device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested training device unavailable: {device}")


def quality(targets, logits, ids):
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite validation logits")
    pooled = score(targets,logits.argmax(1),softmax(logits))
    by_day = {str(day):score(targets[ids[:,1]==day],logits[ids[:,1]==day].argmax(1),softmax(logits[ids[:,1]==day])) for day in (6,7)}
    return dict(pooled=pooled,by_day=by_day,equal_day_macro_f1=float(np.mean([s["macro_f1"] for s in by_day.values()])))


def predict(model, corpus, job, batch=256):
    a = corpus.arrays["validation"]
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(a["index"]),batch):
            x = corpus.windows("validation",np.arange(start,min(start+batch,len(a["index"]))),job["history"],job["features"])
            outputs.append(model(torch.from_numpy(x).to(next(model.parameters()).device)).cpu().numpy())
    return np.concatenate(outputs)


def load_bundle(directory):
    directory = Path(directory)
    pointer = read_json(directory/"epoch.json")
    path = directory/pointer["file"]
    if digest(path) != pointer["sha256"]:
        raise ValueError("Published epoch hash mismatch")
    return torch.load(path,map_location="cpu",weights_only=False)


def primary_epoch(bundle):
    if bundle["binding"]["recipe"]["selection"] == "minimum-validation-cross-entropy-earliest-tie":
        return bundle["best_validation_loss_epoch"]
    return bundle["best_epoch"]


def publish_selected(directory, bundle):
    """Publish primary weights and link the diagnostic F1 checkpoint, if needed."""
    loss_selected = bundle["binding"]["recipe"]["selection"] == "minimum-validation-cross-entropy-earliest-tie"
    epoch = primary_epoch(bundle)
    def metadata(epoch, file, selection):
        record = next(r for r in bundle["history"] if r["epoch"] == epoch)
        return dict(epoch=epoch,score=record["validation"]["equal_day_macro_f1"],
                    equal_day_macro_f1=record["validation"]["equal_day_macro_f1"],
                    validation_loss=record["validation"]["pooled"]["log_loss"],
                    sha256=digest(directory/file),file=file,weights="raw",selection=selection)
    atomic_torch(directory/"selected.pt",bundle["best_loss_weights"] if loss_selected else bundle["best_weights"])
    selected_metadata = metadata(epoch,"selected.pt",bundle["binding"]["recipe"]["selection"])
    if loss_selected:
        alternate = "selected.pt"
        if bundle["best_epoch"] != epoch:
            alternate = "best-f1.pt"
            atomic_torch(directory/alternate,bundle["best_weights"])
        else:
            (directory/"best-f1.pt").unlink(missing_ok=True)
        selected_metadata["best_f1"] = metadata(bundle["best_epoch"],alternate,"equal-day-macro-f1-earliest-tie")
    write_json(directory/"selected.json",selected_metadata)


def selected(directory, job):
    directory = Path(directory)
    bundle = load_bundle(directory)
    if bundle["binding"] != job["binding"]:
        raise ValueError("Checkpoint binding mismatch")
    checkpoint = read_json(directory/"selected.json")
    if digest(directory/"selected.pt") != checkpoint["sha256"] or checkpoint["epoch"] != primary_epoch(bundle):
        raise ValueError("Selected checkpoint hash/epoch mismatch")
    if job["binding"]["recipe"]["selection"] == "minimum-validation-cross-entropy-earliest-tie":
        alternate = checkpoint.get("best_f1",{})
        expected_file = "selected.pt" if bundle["best_epoch"] == checkpoint["epoch"] else "best-f1.pt"
        if (checkpoint.get("selection") != job["binding"]["recipe"]["selection"] or
            alternate.get("epoch") != bundle["best_epoch"] or alternate.get("file") != expected_file or
            digest(directory/expected_file) != alternate.get("sha256")):
            raise ValueError("Best-F1 checkpoint binding/hash mismatch")
    model = MLPLOB(**model_config(job["config"]))
    model.load_state_dict(torch.load(directory/"selected.pt",map_location="cpu",weights_only=True),strict=True)
    model.eval()
    if any(p.item()<0 for p in (model.norm_layer.y1,model.norm_layer.y2)):
        raise ValueError("Frozen negative BiN coefficients require review")
    return model, checkpoint


def fit(study, job, device, resume=False, stop_after=None, campaign_remaining=None, fault=None):
    """stop_after/fault are engineering test hooks, never recipe overrides."""
    device_check(device)
    if device != job["binding"]["training_device"] or provenance() != job["binding"]["code"]:
        raise ValueError("Training device/code/dependency binding changed")
    directory = job_path(study,job); directory.mkdir(parents=True,exist_ok=True)
    with activity(), lock(directory/"mutation.lock"):
        corpus = Development(study["data"])
        if corpus.binding != job["binding"]["data"]:
            raise ValueError("Dataset binding changed")
        status_path = directory/"training.json"
        old = read_json(status_path) if status_path.exists() else None
        if old and old["binding"] != job["binding"]:
            raise ValueError("Existing incompatible training job")
        if old and old["status"] in TERMINAL:
            if (directory/"selected.json").exists(): selected(directory,job)
            return old
        if old and not resume:
            raise ValueError("Existing incomplete job; use --resume")
        config = job["config"]
        torch.manual_seed(config["seed"])
        if device == "cuda": torch.cuda.manual_seed_all(config["seed"])
        generator = torch.Generator().manual_seed(config["seed"])
        model = MLPLOB(**model_config(config)).to(device)
        recipe = job["binding"]["recipe"]
        loss_aware = recipe.get("early_stopping") == "f1-or-validation-loss"
        loss_selected = recipe["selection"] == "minimum-validation-cross-entropy-earliest-tie"
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"], foreach=False, fused=False)
        scheduler_config = recipe.get("scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**{k:v for k,v in scheduler_config.items() if k != "name"}) if scheduler_config else None
        state = dict(binding=job["binding"],epoch=0,best=-1.,best_epoch=None,
                     best_validation_loss=None,best_validation_loss_epoch=None,
                     patience=0,history=[],elapsed=0.,optimizer_steps=0,bin_repairs=0)
        if (directory/"epoch.json").exists():
            state = load_bundle(directory)
            if state["binding"] != job["binding"]: raise ValueError("Resume binding mismatch")
            model.load_state_dict(state["current"]); optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None: scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["cpu_rng"]); generator.set_state(state["shuffle_rng"])
            if device == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
            model.norm_layer.repairs = state["bin_repairs"]
        elapsed = max(state["elapsed"], old.get("elapsed",0.) if old else 0.)
        if old and old["status"] == "running":
            # Conservatively charge the lost attempt, including downtime after hard kill.
            elapsed = max(elapsed,old["elapsed"]+max(0,time.time()-old["started_unix"]))
        published_epoch = state["epoch"]
        started = time.perf_counter()
        limit = min(job["binding"]["run_seconds"], elapsed+campaign_remaining if campaign_remaining is not None else float("inf"))
        def spent(): return elapsed+time.perf_counter()-started
        def status(name, **extra):
            result = dict(status=name,binding=job["binding"],elapsed=spent(),epochs=published_epoch,
                          stopping=dict(criterion=recipe.get("early_stopping", "f1"),
                                        epoch_limit=recipe["epochs"],patience=state["patience"],
                                        learning_rate=optimizer.param_groups[0]["lr"]),
                          started_unix=time.time(),environment=environment(),unique_parameters=sum(p.numel() for p in model.parameters()),transfer_policy="batched raw windows, explicit host/device transfer",**extra)
            write_json(status_path,result)
            return result
        status("running")
        reason = "completed"
        try:
            while recipe["epochs"] is None or state["epoch"] < recipe["epochs"]:
                epoch = state["epoch"]+1
                if state["epoch"] >= recipe["minimum_epochs"] and state["patience"] >= recipe["patience"]:
                    reason = "early-stopped"; break
                if spent() >= limit:
                    reason = "budget-stopped"; break
                learning_rate = optimizer.param_groups[0]["lr"]
                model.train(); order = torch.randperm(len(corpus.arrays["train"]["index"]),generator=generator).numpy()
                training_start = time.perf_counter(); total = 0.
                batch = job["binding"]["batch"]
                for begin in range(0,len(order),batch):
                    if spent() >= limit:
                        reason = "budget-stopped"; break
                    pos = order[begin:begin+batch]
                    x = torch.from_numpy(corpus.windows("train",pos,config["history"],config["features"])).to(device)
                    y = torch.from_numpy(np.array(corpus.arrays["train"]["targets"][pos])).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x),y)
                    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite fitting loss")
                    loss.backward(); optimizer.step()
                    total += loss.item()*len(pos); state["optimizer_steps"] += 1
                if reason == "budget-stopped": break
                if device == "cuda": torch.cuda.synchronize()
                fit_seconds = time.perf_counter()-training_start
                val_start = time.perf_counter(); logits = predict(model,corpus,config)
                metrics = quality(corpus.arrays["validation"]["targets"],logits,corpus.arrays["validation"]["index"])
                validation_seconds = time.perf_counter()-val_start
                value = metrics["equal_day_macro_f1"]
                validation_loss = metrics["pooled"]["log_loss"]
                f1_improved = value > state["best"]
                previous_loss = state.get("best_validation_loss")
                loss_improved = previous_loss is None or validation_loss < previous_loss
                if f1_improved:
                    state.update(best=value,best_epoch=epoch,
                                 best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                if loss_improved:
                    state.update(best_validation_loss=validation_loss,best_validation_loss_epoch=epoch)
                    if loss_selected:
                        state["best_loss_weights"] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if scheduler is not None: scheduler.step(validation_loss)
                next_learning_rate = optimizer.param_groups[0]["lr"]
                lr_reduced = next_learning_rate < learning_rate
                progress = loss_improved if loss_selected else f1_improved or (loss_aware and loss_improved)
                state["patience"] = 0 if progress or lr_reduced else state["patience"]+1
                state["history"].append(dict(epoch=epoch,train_loss=total/len(order),validation=metrics,
                                             f1_improved=f1_improved,validation_loss_improved=loss_improved,
                                             stopping_progress=progress,stopping_patience=state["patience"],
                                             learning_rate=learning_rate,next_learning_rate=next_learning_rate,lr_reduced=lr_reduced,
                                             fitting_seconds=fit_seconds,validation_seconds=validation_seconds))
                state.update(epoch=epoch,elapsed=spent(),current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                             scheduler=copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
                             optimizer=copy.deepcopy(optimizer.state_dict()),cpu_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device=="cuda" else [],shuffle_rng=generator.get_state(),
                             bin_repairs=model.norm_layer.repairs)
                file = f"epoch-{epoch:03d}.pt"
                atomic_torch(directory/file,state)
                if fault: fault("before-publication",epoch)
                write_json(directory/"epoch.json",dict(file=file,sha256=digest(directory/file)))
                published_epoch = epoch
                if fault: fault("after-publication",epoch)
                status("running")
                if epoch >= recipe["minimum_epochs"] and state["patience"] >= recipe["patience"]:
                    reason = "early-stopped"; break
                if stop_after is not None and epoch >= stop_after:
                    reason = "interrupted"; break
            if (directory/"epoch.json").exists():
                # Publish selected from the committed bundle, even following a discarded partial epoch.
                committed = load_bundle(directory)
                publish_selected(directory,committed)
                selected(directory,job)
            elif reason in TERMINAL:
                return status("budget-stopped",selected_checkpoint=None)
            return status(reason,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=="cuda" else None)
        except BaseException as error:
            status("interrupted" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed",error=repr(error))
            raise
