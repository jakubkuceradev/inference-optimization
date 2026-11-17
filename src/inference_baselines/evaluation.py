from pathlib import Path

import numpy as np
import torch

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .metrics import compare, score, softmax
from .runtime import Runtime
from .training import load_selected


def reference(model, x):
    with torch.inference_mode():
        return model(torch.from_numpy(x)).numpy()


def binding(data, run, require_full=True):
    run = Path(run).resolve()
    model, manifest, selected = load_selected(run)
    if read_json(run / "status.json")["status"] != "completed":
        raise ValueError("Training is incomplete")
    if require_full and manifest["kind"] != "full":
        raise ValueError("Smoke checkpoints cannot supply final baseline results")
    if digest(Path(data)/"manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint/dataset manifest mismatch")
    return model, manifest, selected


def freeze_checks(model, runtime, validation, output, config):
    """No test partition is accessed until the diagnostic/tolerance record is sealed."""
    positions = np.linspace(0, len(validation)-1, 64, dtype=int)
    real = np.stack([validation[i][0] for i in positions])
    rng = np.random.default_rng(1729)
    random = rng.normal(size=(1,384,144)).astype(np.float32)
    diagnostic = np.concatenate([random, 3+random*1e-5, np.zeros_like(random),
                                 np.broadcast_to(random[:, :1, :], random.shape),
                                 np.broadcast_to(random[:, :, :1], random.shape)])
    checks = {}
    # Compare at batch one too; batching can change reduction/kernel choices.
    for name, x in (("validation_bulk", real), ("diagnostic_bulk", diagnostic)):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            checks[name] = compare(reference(model, x), runtime(x), config["atol"], config["rtol"])
    eager = np.concatenate([reference(model, x[None]) for x in real])
    candidate = np.concatenate([runtime(x[None]) for x in real])
    checks["validation_batch_one"] = compare(eager, candidate, config["atol"], config["rtol"])
    checks.update({"atol": config["atol"], "rtol": config["rtol"],
                   "policy": "strict logit gate retained even when diagnostics fail; never widened after test",
                   "diagnostic_order": ["random", "near_constant", "zero", "constant_time", "constant_feature"],
                   "dataset_manifest_sha256": validation.manifest_hash})
    write_json(output / "numerical-freeze.json", checks)
    return checks


def reload_check(model, run, validation, config):
    saved = np.load(Path(run)/"selected-validation.npz")
    index = saved["index"][0]
    file_id, edge = index
    x = np.array(validation.x[file_id][edge-383:edge+1][None], copy=True)
    return compare(saved["logits"], reference(model, x), config["atol"], config["rtol"])


def majority(data, output):
    training, test = Corpus(data, "train"), Corpus(data, "test", verify=False)
    counts = np.bincount(training.targets(), minlength=3)
    constant = int(counts.argmax())
    result = {"training_counts": counts.tolist(), "majority_class": constant,
              "tie_rule": "lowest mapped class", "dataset_manifest_sha256": training.manifest_hash,
              "metrics": score(test.targets(), np.full(len(test), constant)), "limitations": LIMITS}
    np.savez(output / "majority-predictions.npz", ids=test.ids(), targets=test.targets(),
             classes=np.full(len(test), constant))
    write_json(output / "majority.json", result)
    return result


def evaluate(data, run, output, name):
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "runtime": name})
    try:
        model, manifest, selected = binding(data, run)
        config = manifest["config"]
        validation = Corpus(data, "validation")
        runtime = Runtime(model, name, output)
        freeze = freeze_checks(model, runtime, validation, output, config)
        reloaded = reload_check(model, run, validation, config)
        write_json(output / "cpu-reload-check.json", reloaded)
        if reloaded["tolerance_pass_count"] != reloaded["count"]:
            raise ValueError("CPU reload differs from the saved checkpoint validation prediction")
        test = Corpus(data, "test", verify=False)
        canonical_hash = None
        canonical = None
        if name != "eager":
            canonical_root = Path(run).parent/f'evaluation-{manifest["seed"]}-eager'
            canonical_result = read_json(canonical_root/"result.json")
            if read_json(canonical_root/"status.json")["status"] != "completed" or canonical_result["checkpoint_sha256"] != selected["checkpoint_sha256"] or canonical_result["dataset_manifest_sha256"] != test.manifest_hash:
                raise ValueError("Complete the matching eager reference evaluation first")
            with np.load(canonical_root/"predictions.npz") as saved:
                np.testing.assert_array_equal(saved["ids"], test.ids())
                np.testing.assert_array_equal(saved["targets"], test.targets())
                canonical = saved["reference_logits"]
            canonical_hash = digest(canonical_root/"predictions.npz")
        refs, candidates = [], []
        size = config["evaluation_batch_size"]
        for start in range(0, len(test), size):
            x = np.stack([test[i][0] for i in range(start, min(start+size, len(test)))])
            if name == "eager":
                value = reference(model, x)
                refs.append(value)
                candidates.append(value)
            else:
                candidates.append(runtime(x))
        refs, candidates = np.concatenate(refs) if canonical is None else canonical, np.concatenate(candidates)
        targets = test.targets()
        np.savez(output / "predictions.npz", ids=test.ids(), targets=targets,
                 reference_logits=refs, runtime_logits=candidates,
                 reference_probabilities=softmax(refs), runtime_probabilities=softmax(candidates),
                 reference_classes=refs.argmax(1), runtime_classes=candidates.argmax(1))
        result = {"runtime": runtime.metadata(), "seed": manifest["seed"],
                  "checkpoint_load_seconds": selected["load_seconds"],
                  "cpu_reload": reloaded,
                  "canonical_prediction_sha256": canonical_hash,
                  "checkpoint_sha256": selected["checkpoint_sha256"], "dataset_manifest_sha256": test.manifest_hash,
                  "numerical_freeze_sha256": digest(output / "numerical-freeze.json"),
                  "environment": environment(), "limitations": LIMITS,
                  "numerical": compare(refs, candidates, config["atol"], config["rtol"]),
                  "diagnostics": freeze}
        for key, logits in (("reference", refs), ("candidate", candidates)):
            result[key] = score(targets, logits.argmax(1), softmax(logits)) if np.isfinite(logits).all() else {"status": "failed", "reason": "nonfinite logits; no financial metrics reported"}
        result["majority"] = majority(data, output)
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "completed", "runtime": name})
        return result
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "runtime": name, "reason": repr(error)})
        raise


def diagnose(data, run, output, name):
    """Validation-only integration check; never reports smoke weights as baselines."""
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "runtime": name})
    try:
        model, manifest, selected = binding(data, run, require_full=False)
        validation = Corpus(data, "validation")
        runtime = Runtime(model, name, output)
        checks = freeze_checks(model, runtime, validation, output, manifest["config"])
        result = {"kind": "validation-only tooling diagnostic; not baseline quality/performance",
                  "checkpoint_load_seconds": selected["load_seconds"],
                  "training_kind": manifest["kind"], "checkpoint_sha256": selected["checkpoint_sha256"],
                  "runtime": runtime.metadata(), "checks": checks, "environment": environment()}
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "completed", "runtime": name})
        return result
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "runtime": name, "reason": repr(error)})
        raise
