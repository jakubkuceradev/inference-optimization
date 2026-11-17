"""Bounded snapshot baselines; validation selects models before test access."""
import os
from pathlib import Path
import time

import numpy as np

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .metrics import compare, latency_summary, score
from .timing import measurement_order


def snapshot(corpus, i):
    file_id, edge = corpus.index[i]
    return np.array(corpus.x[file_id][edge], dtype=np.float32, copy=True)[None]


def matrix(corpus):
    result = np.empty((len(corpus), 144), np.float32)
    for f in np.unique(corpus.index[:, 0]):
        mask = corpus.index[:, 0] == f
        result[mask] = corpus.x[f][corpus.index[mask, 1]]
    return result


def ridge_fit(x, y, alpha):
    """Three regularized least-squares scores on +/-1 targets; no ordinal labels."""
    x = x.astype(np.float64)
    target = 2*np.eye(3, dtype=np.float64)[y]-1
    mean, target_mean = x.mean(0), target.mean(0)
    x -= mean
    target -= target_mean
    gram = x.T@x
    gram.flat[::len(gram)+1] += alpha
    coefficients = np.linalg.solve(gram, x.T@target)
    return coefficients.astype(np.float32), (target_mean-mean@coefficients).astype(np.float32)


class Predictor:
    def __init__(self, root, record, threads=1):
        self.kind = record["kind"]
        if self.kind == "ridge":
            with np.load(Path(root)/record["file"]) as weights:
                self.weight, self.bias = weights["weight"], weights["bias"]
        else:
            import xgboost as xgb
            self.model = xgb.Booster(model_file=Path(root)/record["file"])
            self.model.set_param({"nthread": threads, "device": "cpu"})

    def __call__(self, x):
        if self.kind == "ridge":
            return x@self.weight+self.bias
        return self.model.inplace_predict(x, validate_features=True, strict_shape=True)


def fit(data, output):
    import xgboost as xgb
    from threadpoolctl import threadpool_limits
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        train, validation = Corpus(data, "train"), Corpus(data, "validation", verify=False)
        plan = {"seed": 42, "dataset_manifest_sha256": train.manifest_hash,
                "features": "Current 144 supplied features at the same eligible target IDs; no 384-row history and no BiN",
                "normalization": "Supplied z-score unchanged; ridge centering fitted on training only",
                "ridge_alphas": [.1, 1., 10., 100.], "ridge_selection": "maximum validation macro-F1; first alpha on ties",
                "tree_depths": [2, 4, 6], "maximum_boosting_rounds": 300, "early_stopping_rounds": 20,
                "tree_prefixes": [1, 4, 16, 64, 128, "validation-selected last round"],
                "tree_parameters": {"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "device": "cpu",
                    "eta": .05, "max_bin": 256, "subsample": 1., "colsample_bytree": 1., "seed": 42,
                    "nthread": 4, "eval_metric": "mlogloss"},
                "tree_selection": "validation log loss early stopping per depth; all fixed prefix points retained, no test selection",
                "environment": environment(), "xgboost": xgb.__version__, "limitations": LIMITS}
        write_json(output/"plan.json", plan)
        x, y, vx, vy = matrix(train), train.targets(), matrix(validation), validation.targets()
        records, ridge_candidates = {}, []
        with threadpool_limits(limits=4):
            for alpha in plan["ridge_alphas"]:
                started = time.perf_counter()
                weight, bias = ridge_fit(x, y, alpha)
                value = vx@weight+bias
                metric = score(vy, value.argmax(1))
                ridge_candidates.append({"alpha": alpha, "metrics": metric, "seconds": time.perf_counter()-started,
                                         "weight": weight, "bias": bias})
        best = max(range(len(ridge_candidates)), key=lambda i: ridge_candidates[i]["metrics"]["macro_f1"])
        selected = ridge_candidates[best]
        np.savez(output/"ridge.npz", weight=selected["weight"], bias=selected["bias"])
        records["ridge"] = {"kind": "ridge", "file": "ridge.npz", "alpha": selected["alpha"],
                            "validation": selected["metrics"], "probabilities": "Not defined for least-squares scores"}
        write_json(output/"ridge-selection.json", [{k:v for k,v in c.items() if k not in ("weight", "bias")} for c in ridge_candidates])
        dtrain, dval = xgb.DMatrix(x, label=y, nthread=4), xgb.DMatrix(vx, label=vy, nthread=4)
        for depth in plan["tree_depths"]:
            started = time.perf_counter()
            history = {}
            model = xgb.train({**plan["tree_parameters"], "max_depth": depth}, dtrain,
                              num_boost_round=plan["maximum_boosting_rounds"], evals=[(dval, "validation")],
                              early_stopping_rounds=plan["early_stopping_rounds"], evals_result=history, verbose_eval=False)
            selected_rounds = model.best_iteration+1
            model.save_model(output/f"depth-{depth}-training.ubj")
            write_json(output/f"depth-{depth}-selection.json", {"best_iteration": model.best_iteration,
                       "rounds_executed": model.num_boosted_rounds(), "history": history, "seconds": time.perf_counter()-started})
            prefixes = sorted(set([k for k in plan["tree_prefixes"] if isinstance(k, int) and k <= selected_rounds]+[selected_rounds]))
            for rounds in prefixes:
                name = f"xgb-d{depth}-r{rounds}"
                prefix = model[:rounds]
                prefix.save_model(output/f"{name}.ubj")
                p = prefix.inplace_predict(vx, strict_shape=True)
                records[name] = {"kind": "xgboost", "file": f"{name}.ubj", "depth": depth, "rounds": rounds,
                                 "tree_count": len(prefix.get_dump()), "validation": score(vy, p.argmax(1), p),
                                 "validation_selected_round_count": selected_rounds}
            print(f"XGBoost depth {depth}: selected {selected_rounds} rounds", flush=True)
        # Seal choices and model hashes before accessing the test partition.
        for record in records.values():
            record["sha256"] = digest(output/record["file"])
        write_json(output/"models.json", records)
        test = Corpus(data, "test", verify=False)
        tx, ty = matrix(test), test.targets()
        np.savez(output/"test-identity.npz", ids=test.ids(), targets=ty, index=test.index)
        results = {}
        for name, record in records.items():
            with threadpool_limits(limits=4):
                value = Predictor(output, record, threads=4)(tx)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite scores: {name}")
            np.save(output/f"{name}-test-scores.npy", value)
            results[name] = {**record, "test": score(ty, value.argmax(1), value if record["kind"] == "xgboost" else None),
                             "bytes": (output/record["file"]).stat().st_size}
        write_json(output/"result.json", {"plan": plan, "models": results,
                    "test_identity_sha256": digest(output/"test-identity.npz")})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "models": len(records)}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise


def benchmark(data, run, output, config):
    from threadpoolctl import threadpool_info, threadpool_limits
    output, run = external(output), Path(run)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        if len(os.sched_getaffinity(0)) != 1:
            raise ValueError("Timing requires one pinned CPU")
        result = read_json(run/"result.json")
        if read_json(run/"status.json")["status"] != "completed":
            raise ValueError("Complete comparator training and quality checks first")
        test = Corpus(data, "test")
        if test.manifest_hash != result["plan"]["dataset_manifest_sha256"]:
            raise ValueError("Dataset binding mismatch")
        with np.load(run/"test-identity.npz") as saved:
            np.testing.assert_array_equal(saved["ids"], test.ids())
            np.testing.assert_array_equal(saved["targets"], test.targets())
        models = read_json(run/"models.json")
        for record in models.values():
            if digest(run/record["file"]) != record["sha256"]:
                raise ValueError("Model hash mismatch")
        names = list(models)
        order = np.linspace(0, len(test)-1, config["calls"], dtype=np.int64)
        jobs = measurement_order(names, config["repeats"])
        np.save(output/"input-indices.npy", order)
        write_json(output/"plan.json", {"environment": environment(), "order": jobs,
                   "calls": config["calls"], "repeats": config["repeats"], "warmup": config["warmup"],
                   "quality_sha256": digest(run/"result.json"), "models_sha256": digest(run/"models.json"),
                   "prepared": "Resident contiguous FP32 [1,144] through NumPy or XGBoost native Python API and CPU argmax",
                   "local_preparation": "Copy current 144-feature vector plus prepared boundary",
                   "excluded": ["generating supplied engineered features", "network", "queueing", "disk"],
                   "limitations": LIMITS+["Native Python API overhead included; no optimized tree code generation.", "Snapshot inputs differ from MLPLOB history inputs."]})
        predictors = {name: Predictor(run, record) for name, record in models.items()}
        observations = {name: [] for name in names}
        with threadpool_limits(limits=1):
            write_json(output/"threadpools.json", threadpool_info())
            for name, predictor in predictors.items():
                (output/name).mkdir()
                for _ in range(config["warmup"]):
                    predictor(snapshot(test, order[0]))
            write_json(output/"phase.json", {"phase": "timing", "time_unix": time.time()})
            for repeat, assignments in enumerate(jobs):
                for name, boundary in assignments:
                    predictor = predictors[name]
                    ns, scores = np.empty(len(order), np.int64), np.empty((len(order),3), np.float32)
                    for j, index in enumerate(order):
                        if boundary == "prepared":
                            value = snapshot(test, index)
                        start = time.perf_counter_ns()
                        if boundary == "local_preparation":
                            value = snapshot(test, index)
                        predicted = predictor(value)
                        int(predicted.argmax())
                        ns[j] = time.perf_counter_ns()-start
                        scores[j] = predicted
                    stem = f"{boundary}-{repeat}"
                    np.save(output/name/f"{stem}-ns.npy", ns)
                    np.save(output/name/f"{stem}-scores.npy", scores)
                    bulk = np.load(run/f"{name}-test-scores.npy", mmap_mode="r")[order]
                    numerical = compare(bulk, scores, config["atol"], config["rtol"])
                    # Inputs are affine decision scores or already-normalized
                    # tree probabilities; a second softmax is not meaningful.
                    numerical.pop("probability_max_abs_error", None)
                    numerical.pop("probability_rms_error", None)
                    row = {"repeat": repeat, "boundary": str(boundary), **latency_summary(ns),
                           "bulk_numerical": numerical,
                           "comparison_space": "least-squares scores" if models[name]["kind"] == "ridge" else "class probabilities"}
                    observations[name].append(row)
                    print(f"CPU {name} {boundary} {repeat}: p50 {row['p50_us']:.1f} us p99 {row['p99_us']:.1f} us", flush=True)
        write_json(output/"result.json", {"models": observations, "quality_sha256": digest(run/"result.json")})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "models": len(names)}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise
