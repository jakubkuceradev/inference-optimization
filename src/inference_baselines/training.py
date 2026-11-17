import copy
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import digest, environment, external, read_json, write_json
from .metrics import compare
from .data import Corpus, LIMITS
from .model import MLPLOB


def save_torch(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def ema_update(ema, model, decay):
    with torch.no_grad():
        # parameters() visits shared tensors once; state_dict() does not.
        for average, raw in zip(ema.parameters(), model.parameters(), strict=True):
            average.mul_(decay).add_(raw, alpha=1-decay)


def validation_loss(model, dataset, batch_size, limit=None):
    total, count = 0., 0
    model.eval()
    with torch.inference_mode():
        for i, (x, y) in enumerate(DataLoader(dataset, batch_size=batch_size)):
            if limit is not None and i >= limit:
                break
            device = next(model.parameters()).device
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            total += loss.item()
            count += len(y)
    return total/count


def load_selected(run):
    started = time.perf_counter()
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    selected = read_json(run / "selected.json")
    if digest(run / "selected.pt") != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    model = MLPLOB(**manifest["config"]["model"])
    model.load_state_dict(torch.load(run / "selected.pt", weights_only=True, map_location="cpu"), strict=True)
    model.eval()
    for value in (model.norm_layer.y1, model.norm_layer.y2):
        if value.item() < 0:
            raise ValueError("Negative selected BiN coefficient: inference semantics require review")
    selected["load_seconds"] = time.perf_counter()-started
    return model, manifest, selected


def train(data, output, config, seed, smoke=False, resume=False, allow_long_run=False, device="cpu"):
    output = external(output)
    if not smoke:
        smoke_root = output.parent / "smoke-42"
        smoke_status = read_json(smoke_root / "status.json")
        smoke_manifest = read_json(smoke_root / "manifest.json")
        if smoke_status["status"] != "completed" or smoke_manifest["dataset_manifest_sha256"] != digest(Path(data)/"manifest.json") or smoke_manifest["config"] != config or smoke_manifest.get("training_device", "cpu") != device:
            raise ValueError("A completed smoke with the same data/config is required")
        if not smoke_status["feasible"] and not allow_long_run:
            raise ValueError("Smoke exceeds the compute budget; select a host or explicitly use --allow-long-run")
    output.mkdir(parents=True, exist_ok=resume)
    config = copy.deepcopy(config)
    manifest = {"config": config, "seed": seed, "kind": "smoke" if smoke else "full",
                "dataset": str(Path(data).resolve()), "dataset_manifest_sha256": digest(Path(data)/"manifest.json"),
                "environment": environment(), "limitations": LIMITS,
                "upstream": read_json(Path(__file__).resolve().parent/"configs/source.json"),
                "allow_long_run": allow_long_run,
                "training_device": device,
                "selection": "minimum sample-weighted EMA validation cross-entropy; earliest exact tie",
                "recipe_deviations": ["fixed EMA after optimizer step, no warm-start", "constant learning rate",
                                      "device-independent in-place negative BiN repair retaining optimizer identity"],
                "optimizer": {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
                              "foreach": False, "fused": False}}
    if resume:
        old = read_json(output / "manifest.json")
        if any(old[k] != manifest[k] for k in ("config", "seed", "kind", "dataset_manifest_sha256", "training_device")):
            raise ValueError("Resume manifest mismatch")
        if old["environment"]["code_sha256"] != manifest["environment"]["code_sha256"]:
            raise ValueError("Resume source code differs from the recorded training implementation")
        if read_json(output / "status.json")["status"] == "completed":
            raise ValueError("Run already completed; reuse selected weights")
    else:
        write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"status": "running", "seed": seed})
    started = time.perf_counter()
    try:
        if seed not in config["seeds"]:
            raise ValueError("Seed is not scheduled")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but unavailable; no CPU fallback")
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        training, validation = Corpus(data, "train"), Corpus(data, "validation", verify=False)
        if training.manifest["protocol"] != config["protocol"]:
            raise ValueError("Protocol mismatch")
        model = MLPLOB(**config["model"]).to(device)
        average = copy.deepcopy(model).eval()
        for p in average.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
        generator = torch.Generator().manual_seed(seed)
        first_epoch, best, history = 0, math.inf, []
        if resume:
            state = torch.load(output / "resume.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(state["raw"])
            average.load_state_dict(state["ema"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            generator.set_state(state["shuffle_rng"])
            if device == "cuda":
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            first_epoch, best, history = state["next_epoch"], state["best"], state["history"]
            model.norm_layer.repairs = state["bin_repairs"]
        loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True, generator=generator)
        for epoch in range(first_epoch, 1 if smoke else config["epochs"]):
            model.train()
            total, count, steps, losses = 0., 0, 0, []
            epoch_started = time.perf_counter()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                ema_update(average, model, config["ema_decay"])
                total += loss.item()*len(y)
                losses.append(loss.item())
                count += len(y)
                steps += 1
                if smoke and steps >= 100:
                    break
            train_seconds = time.perf_counter()-epoch_started
            val_started = time.perf_counter()
            val = validation_loss(average, validation, config["evaluation_batch_size"], limit=10 if smoke else None)
            val_seconds = time.perf_counter()-val_started
            record = {"epoch": epoch, "train_loss": total/count, "validation_loss": val,
                      "train_count": count, "train_steps": steps, "train_seconds": train_seconds,
                      "validation_seconds": val_seconds, "bin_repairs": model.norm_layer.repairs,
                      "first_ten_batch_loss": float(np.mean(losses[:10])),
                      "last_ten_batch_loss": float(np.mean(losses[-10:]))}
            history.append(record)
            if val < best:
                best = val
                cpu_state = {key: value.detach().cpu() for key, value in average.state_dict().items()}
                save_torch(output / "selected.pt", cpu_state)
                with torch.inference_mode():
                    example = torch.from_numpy(validation[0][0][None])
                    training_probe = average(example.to(device)).cpu().numpy()
                    cpu_model = MLPLOB(**config["model"]).eval()
                    cpu_model.load_state_dict(cpu_state, strict=True)
                    probe = cpu_model(example).numpy()
                np.savez(output / "selected-validation.npz", logits=probe, index=validation.index[:1])
                write_json(output / "training-to-cpu.json", compare(training_probe, probe, config["atol"], config["rtol"]))
                write_json(output / "selected.json", {"epoch": epoch, "validation_loss": val,
                    "checkpoint_sha256": digest(output / "selected.pt"), "weights": "EMA", "ema_decay": config["ema_decay"]})
            save_torch(output / "resume.pt", {"raw": model.state_dict(), "ema": average.state_dict(),
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "shuffle_rng": generator.get_state(), "next_epoch": epoch+1, "best": best,
                "history": history, "bin_repairs": model.norm_layer.repairs})
            write_json(output / "history.json", history)
            print(record, flush=True)
        # Reloaded selected tensors must reproduce an actual saved validation prediction.
        selected_model, _, _ = load_selected(output)
        example = torch.from_numpy(validation[0][0][None])
        with torch.inference_mode():
            actual = selected_model(example)
        expected = np.load(output / "selected-validation.npz")["logits"]
        np.testing.assert_array_equal(expected, actual.numpy())
        np.savez(output / "reload-validation.npz", logits=actual.numpy(), index=validation.index[:1])
        write_json(output / "reload.json", {"exact": True, "partition": "validation", "count": 1})
        status = {"status": "completed", "seed": seed, "seconds": time.perf_counter()-started,
                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "parameters": sum(p.numel() for p in model.parameters())}
        if device == "cuda":
            status["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            status["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if smoke:
            last = history[-1]
            estimate = (last["train_seconds"]*len(training)/last["train_count"] + last["validation_seconds"]*len(validation)/(10*config["evaluation_batch_size"])) * config["epochs"] * len(config["seeds"])
            status.update({"estimated_full_schedule_hours": estimate/3600,
                           "feasible": estimate/3600 <= config["feasibility_max_hours"],
                           "feasibility_max_hours": config["feasibility_max_hours"],
                           "estimate_excludes": "full-test runtime evaluation, export/compile, and timing"})
        write_json(output / "status.json", status)
        return status
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "seed": seed, "reason": repr(error), "seconds": time.perf_counter()-started})
        raise
