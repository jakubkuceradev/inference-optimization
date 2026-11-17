"""Fixed-recipe temporal tree comparison, with native completed-call timing."""
import importlib.metadata
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .metrics import compare, latency_summary, score
from .temporal import current_mean5
from .timing import measurement_order

CONFIG = Path(__file__).parent/"configs/tree288.json"


def probabilities(model, kind, x):
    if kind == "xgboost":
        value = model.inplace_predict(x, strict_shape=True)
    elif kind == "lightgbm":
        value = model.predict(x, num_threads=4)
    else:
        value = model.predict_proba(x, thread_count=4)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (len(x), 3) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Invalid tree probabilities")
    np.testing.assert_allclose(value.sum(1), 1, atol=1e-6, rtol=1e-6)
    return value


def load_model(root, record):
    path = Path(root)/record["file"]
    if digest(path) != record["sha256"]:
        raise ValueError(f"Model hash mismatch: {path}")
    if record["kind"] == "xgboost":
        import xgboost as xgb
        model = xgb.Booster(model_file=path)
        model.set_param({"nthread": 4, "device": "cpu"})
    elif record["kind"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
    else:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(path)
    return model


def fit(data, output):
    import catboost
    import lightgbm as lgb
    import xgboost as xgb
    from threadpoolctl import threadpool_limits
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        config = read_json(CONFIG)
        train, validation = Corpus(data, "train"), Corpus(data, "validation", verify=False)
        plan = {"config": config, "config_sha256": digest(CONFIG), "environment": environment(),
                "versions": {n: importlib.metadata.version(n) for n in ("xgboost-cpu", "lightgbm", "catboost")},
                "dataset_manifest_sha256": train.manifest_hash, "limitations": LIMITS,
                "normalization": "Supplied normalized features unchanged; no new normalization or relabeling",
                "scope": "Three fixed recipes, one seed; no feature selection or dataset tuning"}
        write_json(output/"plan.json", plan)
        x, y = current_mean5(train), train.targets()
        vx, vy = current_mean5(validation), validation.targets()
        positions = np.linspace(0, len(validation)-1, 256, dtype=np.int64)
        vx[positions].tofile(output/"validation-probe.f32")
        np.savez(output/"validation-probe-index.npz", positions=positions, index=validation.index[positions])
        models, training = {}, {}
        with threadpool_limits(limits=config["training_threads"]):
            for kind in ("xgboost", "lightgbm", "catboost"):
                start = time.perf_counter()
                history = {}
                if kind == "xgboost":
                    model = xgb.train({**config[kind], "eta": config["learning_rate"], "seed": config["seed"],
                        "nthread": config["training_threads"], "eval_metric": "mlogloss"},
                        xgb.DMatrix(x, label=y, nthread=4), num_boost_round=config["rounds"],
                        evals=[(xgb.DMatrix(vx, label=vy, nthread=4), "validation")],
                        evals_result=history, verbose_eval=False)
                elif kind == "lightgbm":
                    model = lgb.train({**config[kind], "learning_rate": config["learning_rate"],
                        "seed": config["seed"], "num_threads": config["training_threads"], "metric": "multi_logloss"},
                        lgb.Dataset(x, label=y), num_boost_round=config["rounds"],
                        valid_sets=[lgb.Dataset(vx, label=vy)], valid_names=["validation"],
                        callbacks=[lgb.record_evaluation(history)])
                else:
                    model = catboost.CatBoostClassifier(**config[kind], iterations=config["rounds"],
                        learning_rate=config["learning_rate"], random_seed=config["seed"], thread_count=4)
                    model.fit(x, y, eval_set=(vx, vy), use_best_model=False, verbose=False)
                    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
                    history = model.get_evals_result()
                training[kind] = {"seconds": time.perf_counter()-start, "history": history}
                for count in config["prefixes"]:
                    name = f"{kind}-{count}"
                    if kind == "xgboost":
                        prefix, suffix = model[:count], "ubj"
                        prefix.save_model(output/f"{name}.{suffix}")
                        tree_count = len(prefix.get_dump())
                    elif kind == "lightgbm":
                        suffix = "txt"
                        model.save_model(output/f"{name}.{suffix}", num_iteration=count)
                        prefix = lgb.Booster(model_file=output/f"{name}.{suffix}")
                        tree_count = prefix.num_trees()
                    else:
                        prefix, suffix = model.copy(), "cbm"
                        prefix.shrink(ntree_end=count)
                        prefix.save_model(output/f"{name}.{suffix}")
                        tree_count = prefix.tree_count_
                    value = probabilities(prefix, kind, vx)
                    value[positions].tofile(output/f"{name}-validation.f64")
                    path = output/f"{name}.{suffix}"
                    models[name] = {"kind": kind, "file": path.name, "sha256": digest(path),
                        "rounds": count, "tree_count": tree_count, "bytes": path.stat().st_size,
                        "validation": score(vy, value.argmax(1), value)}
                print(f"{kind} training complete: {training[kind]['seconds']:.1f}s", flush=True)
        # Recipes and all model identities are sealed before test scoring.
        write_json(output/"models.json", models)
        write_json(output/"training.json", training)
        del x, vx
        test = Corpus(data, "test", verify=False)
        tx, ty = current_mean5(test), test.targets()
        tx.tofile(output/"test-features.f32")
        np.save(output/"test-index.npy", test.index)
        np.save(output/"test-targets.npy", ty)
        for name, record in models.items():
            value = probabilities(load_model(output, record), record["kind"], tx)
            np.save(output/f"{name}-test-probabilities.npy", value)
            record["test"] = score(ty, value.argmax(1), value)
        result = {"plan": plan, "models": models,
                  "test_index_sha256": digest(output/"test-index.npy"),
                  "test_features_sha256": digest(output/"test-features.f32"),
                  "identity": "(source file ID, right edge) in test-index.npy, bound to prepared manifest"}
        write_json(output/"result.json", result)
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "models": len(models)}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise


def benchmark(data, run, output, catboost_library, cpu):
    import lightgbm
    import xgboost
    output, run = external(output), Path(run).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        if cpu is None or os.sched_getaffinity(0) != {cpu}:
            raise ValueError("Native timing requires --cpu with one pinned host core")
        if read_json(run/"status.json")["status"] != "completed":
            raise ValueError("Finish fixed tree training and quality scoring first")
        result = read_json(run/"result.json")
        config, models = result["plan"]["config"], result["models"]
        test = Corpus(data, "test")
        if result["plan"]["dataset_manifest_sha256"] != test.manifest_hash:
            raise ValueError("Dataset mismatch")
        np.testing.assert_array_equal(np.load(run/"test-index.npy"), test.index)
        if digest(run/"test-features.f32") != result["test_features_sha256"]:
            raise ValueError("Test feature hash mismatch")
        libraries = {"xgboost": Path(xgboost.__file__).parent/"lib/libxgboost.so",
                     "lightgbm": Path(lightgbm.__file__).parent/"lib/lib_lightgbm.so",
                     "catboost": Path(catboost_library).resolve()}
        lines = []
        for name, record in models.items():
            if digest(run/record["file"]) != record["sha256"]:
                raise ValueError("Model hash mismatch")
            fields = [name, record["kind"], str(libraries[record["kind"]]), str(run/record["file"])]
            if any("\t" in s or "\n" in s for s in fields):
                raise ValueError("Native manifest paths cannot contain tabs/newlines")
            lines.append("\t".join(fields))
        (output/"models.tsv").write_text("\n".join(lines)+"\n")
        source = Path(__file__).parent/"native/tree_runtime.cpp"
        binary = output/"tree-runtime"
        build = subprocess.run(["c++", "-O3", "-march=native", "-std=c++17", str(source), "-ldl", "-o", str(binary)],
                               capture_output=True, text=True)
        (output/"build.log").write_text(build.stdout+build.stderr)
        build.check_returncode()
        subprocess.run([str(binary), "selftest"], check=True)
        # Freeze numerical validation before invoking native prediction on test.
        validation = output/"validation"
        validation.mkdir()
        subprocess.run([str(binary), "predict", str(output/"models.tsv"), str(run/"validation-probe.f32"), str(validation), str(cpu)], check=True)
        checks = {}
        atol, rtol = config["native_probability_atol"], config["native_probability_rtol"]
        for name in models:
            expected = np.fromfile(run/f"{name}-validation.f64", np.float64).reshape(-1, 3)
            actual = np.fromfile(validation/f"{name}-probabilities.f64", np.float64).reshape(-1, 3)
            checks[name] = probability_check(expected, actual, atol, rtol)
            if checks[name]["tolerance_pass_count"] != len(expected):
                raise ValueError(f"Native validation probability check failed: {name}")
        write_json(output/"validation-freeze.json", {"checks": checks, "atol": atol, "rtol": rtol})
        native_test = output/"test"
        native_test.mkdir()
        subprocess.run([str(binary), "predict", str(output/"models.tsv"), str(run/"test-features.f32"), str(native_test), str(cpu)], check=True)
        ty = np.load(run/"test-targets.npy")
        full = {}
        for name in models:
            expected = np.load(run/f"{name}-test-probabilities.npy")
            actual = np.fromfile(native_test/f"{name}-probabilities.f64", np.float64).reshape(-1, 3)
            full[name] = {"numerical": probability_check(expected, actual, atol, rtol),
                          "metrics": score(ty, actual.argmax(1), actual)}
            if not np.isfinite(actual).all():
                raise ValueError("Nonfinite native financial probabilities")
        write_json(output/"native-quality.json", full)
        indices = np.arange(1, config["calls"]+1, dtype=np.int64)
        pairs = test.index[indices]
        f, edge = pairs[0]
        if not (pairs[:, 0] == f).all() or not (np.diff(pairs[:, 1]) == 1).all():
            raise ValueError("Timing requires a consecutive sequence within one known file")
        snapshots = np.array(test.x[f][pairs[:, 1]], copy=True)
        snapshots.tofile(output/"snapshots.f32")
        current_mean5(test, indices).tofile(output/"prepared.f32")
        np.array(test.x[f][edge-5:edge], copy=True).tofile(output/"prefill.f32")
        np.save(output/"input-indices.npy", indices)
        jobs = measurement_order(list(models), config["repeats"])
        (output/"schedule.tsv").write_text("".join(f"{r}\t{n}\t{b}\n" for r, assignments in enumerate(jobs) for n,b in assignments))
        plan = {"environment": environment(), "config": config, "models_sha256": digest(run/"models.json"),
            "quality_sha256": digest(run/"result.json"), "source_sha256": digest(source), "binary_sha256": digest(binary),
            "libraries": {n: {"path": str(p), "sha256": digest(p)} for n,p in libraries.items()},
            "input_order": "10000 consecutive eligible test windows in first file, starting at eligible index1",
            "prepared": "Resident FP32 current+mean5 through native probabilities and CPU argmax",
            "local_preparation": "New144-vector ready, preallocated five-row ring, FP64 mean roundedFP32, native probabilities and argmax",
            "excluded": ["raw supplied-feature generation", "disk", "network", "queueing", "prefill", "model loading", "compilation"],
            "notes": ["No Python per-call dispatch; all15 models/libraries resident; no global frequency/thermal control.",
                      "XGBoost C API still parses its fixed JSON interface/config internally.",
                      "Fixed first comparison, not feature selection or hyperparameter optimization."], "jobs": jobs}
        write_json(output/"plan.json", plan)
        write_json(output/"phase.json", {"phase": "timing", "time_unix": time.time()})
        subprocess.run([str(binary), "benchmark", str(output/"models.tsv"), str(output), str(cpu), str(config["warmup"])], check=True)
        observations = {n: [] for n in models}
        for repeat, assignments in enumerate(jobs):
            for name, boundary in assignments:
                stem = f"{name}-{boundary}-{repeat}"
                ns = np.fromfile(output/f"{stem}-ns.i64", np.int64)
                values = np.fromfile(output/f"{stem}-probabilities.f64", np.float64).reshape(-1, 3)
                if ns.shape != (config["calls"],) or not (ns > 0).all() or not np.isfinite(values).all():
                    raise ValueError("Incomplete/nonfinite timed native output")
                expected = np.load(run/f"{name}-test-probabilities.npy")[indices]
                observations[name].append({"repeat": repeat, "boundary": boundary, **latency_summary(ns),
                                          "numerical": probability_check(expected, values, atol, rtol)})
        write_json(output/"result.json", {"plan": plan, "observations": observations, "native_quality": full,
                                         "native_metadata": read_json(output/"native-metadata.json")})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "models": len(models), "calls": len(models)*2*config["repeats"]*config["calls"]}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise


def probability_check(expected, actual, atol, rtol):
    value = compare(expected, actual, atol, rtol)
    value.pop("probability_max_abs_error", None)
    value.pop("probability_rms_error", None)
    value["comparison_space"] = "probabilities, not logits; no second softmax"
    return value
