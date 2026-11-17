"""Fixed, attributed feature recipes for native CatBoost and ridge comparisons."""
from pathlib import Path
import importlib.metadata
import os
import shutil
import subprocess
import time

import numpy as np

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .metrics import compare, latency_summary, score
from .timing import measurement_order

CONFIG = Path(__file__).parent/"configs/feature_comparison.json"
NATIVE = Path(__file__).parent/"native"


def provenance(data, run):
    corpus = Corpus(data, "train")
    return {"dataset_manifest_sha256": corpus.manifest_hash,
            "partition_index_sha256": {p: digest(Path(data)/f"{p}-index.npy") for p in ("train", "validation", "test")},
            "native_source_sha256": {p.name: digest(p) for p in (NATIVE/"features.h", NATIVE/"feature_runtime.cpp")},
            "binary_sha256": digest(Path(run)/"feature-runtime"),
            "versions": {p: importlib.metadata.version(p) for p in ("numpy", "catboost", "threadpoolctl")},
            "normalization": "Supplied source z-scores unchanged; original upstream normalization population unverified. Ridge additional scales use eligible training rows only, and fold into native weights.",
            "protocol": "fi2010-source-compatible-caveated-v1", "limitations": LIMITS}


def command(*args):
    subprocess.run([str(a) for a in args], check=True)


def export(binary, corpus, recipe, path):
    append = 0
    for f in np.unique(corpus.index[:, 0]):
        edges = corpus.index[corpus.index[:, 0] == f, 1]
        if not np.all(np.diff(edges) == 1):
            raise ValueError("Feature export requires contiguous known partitions")
        command(binary, "features", recipe, corpus.root/f"x-{f}.npy", corpus.x[f].offset,
                edges[0], edges[-1]+1, path, append)
        append = 1


def matrix(path, count, dimension):
    if path.stat().st_size != count*dimension*4:
        raise ValueError("Feature size mismatch")
    return np.memmap(path, mode="r", dtype=np.float32, shape=(count, dimension))


def ridge_predict(x, coefficients, bias):
    result = np.empty((len(x), 3), np.float64)
    for start in range(0, len(x), 4096):
        result[start:start+4096] = np.asarray(x[start:start+4096], np.float64)@coefficients+bias
    return result


def fit_ridge(x, y, vx, vy, config):
    """Train-only sufficient statistics; alpha/weight choices see validation only."""
    n, d = x.shape
    counts = np.bincount(y, minlength=3)
    grams, sums = [], []
    for c in range(3):
        block = np.asarray(x[y == c], np.float64)
        grams.append(block.T@block)
        sums.append(block.sum(axis=0))
        del block
    grams, sums = np.asarray(grams), np.asarray(sums)
    mean = sums.sum(0)/n
    deviation = np.sqrt(np.maximum(np.diag(grams.sum(0))/n-mean**2, 0))
    deviation[deviation < 1e-12] = 1
    candidates, best, coefficients, bias = [], -1., None, None
    for weighting in config["weights"]:
        weights = np.ones(3) if weighting == "uniform" else n/(3*counts)
        total = float(weights@counts)
        raw_sum = weights@sums
        center = raw_sum/total
        target_mean = 2*weights*counts/total-1
        gram = (np.einsum("c,cij->ij", weights, grams)-total*np.outer(center, center))/np.outer(deviation, deviation)
        rhs = (2*(weights[:, None]*sums).T-raw_sum[:, None]-total*center[:, None]*target_mean)/deviation[:, None]
        eigenvalues, vectors = np.linalg.eigh(gram)
        if eigenvalues.min() < -config["alphas"][0]*.01:
            raise ValueError("Ridge Gram lost positive semidefiniteness")
        projected = vectors.T@rhs
        for alpha in config["alphas"]:
            coefficient = (vectors@(projected/(eigenvalues[:, None]+alpha)))/deviation[:, None]
            intercept = target_mean-center@coefficient
            quality = score(vy, ridge_predict(vx, coefficient, intercept).argmax(1))
            candidate = {"alpha": alpha, "weighting": weighting, "validation": quality}
            candidates.append(candidate)
            if quality["macro_f1"] > best:
                best, selected = quality["macro_f1"], candidate
                coefficients, bias = coefficient, intercept
    return coefficients, bias, {"selected": selected, "candidates": candidates,
                                "training_mean": mean.tolist(), "training_scale": deviation.tolist()}


def fit(data, output, reuse):
    from catboost import CatBoostClassifier
    from threadpoolctl import threadpool_limits
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    config = read_json(CONFIG)
    write_json(output/"plan.json", {"config": config, "config_sha256": digest(CONFIG),
               "environment": environment(), "limitations": LIMITS, "reuse": str(reuse)})
    binary = output/"feature-runtime"
    command("g++", "-O3", "-std=c++17", "-ffp-contract=off", NATIVE/"feature_runtime.cpp", "-ldl", "-o", binary)
    write_json(output/"provenance.json", provenance(data, output))
    train, validation = Corpus(data, "train"), Corpus(data, "validation", verify=False)
    y, vy = train.targets(), validation.targets()
    registry = {}
    with threadpool_limits(limits=config["threads"]):
        for recipe, definition in config["recipes"].items():
            print(f"Fitting {recipe} ({definition['dimension']} features)", flush=True)
            directory = output/recipe
            directory.mkdir()
            for partition, corpus in (("train", train), ("validation", validation)):
                export(binary, corpus, recipe, directory/f"{partition}.f32")
            x = matrix(directory/"train.f32", len(train), definition["dimension"])
            vx = matrix(directory/"validation.f32", len(validation), definition["dimension"])
            started = time.perf_counter()
            coefficient, bias, selection = fit_ridge(x, y, vx, vy, config["ridge"])
            np.concatenate((coefficient.ravel(), bias)).tofile(directory/"ridge.f64")
            write_json(directory/"ridge-selection.json", selection)
            registry[f"ridge-{recipe}"] = {"kind": "ridge", "recipe": recipe, "file": f"{recipe}/ridge.f64",
                "fit_seconds": time.perf_counter()-started, "selection": selection["selected"]}
            started = time.perf_counter()
            model = CatBoostClassifier(**config["catboost"], iterations=config["catboost_rounds"])
            if recipe == "current_mean5":
                old_plan = read_json(reuse/"plan.json")["config"]
                if (old_plan["seed"] != config["seed"] or old_plan["rounds"] != config["catboost_rounds"]
                        or old_plan["learning_rate"] != config["catboost"]["learning_rate"]):
                    raise ValueError("Reused CatBoost round/seed/learning-rate configuration differs")
                for key, value in old_plan["catboost"].items():
                    if config["catboost"][key] != value:
                        raise ValueError(f"Reused CatBoost configuration differs: {key}")
                shutil.copyfile(reuse/"catboost-256.cbm", directory/"catboost.cbm")
                if digest(directory/"catboost.cbm") != read_json(reuse/"models.json")["catboost-256"]["sha256"]:
                    raise ValueError("Reused CatBoost checkpoint hash mismatch")
                model.load_model(directory/"catboost.cbm")
                from .temporal import current_mean5
                positions = np.linspace(0, len(validation)-1, 256, dtype=int)
                np.testing.assert_array_equal(vx[positions], current_mean5(validation, positions))
            else:
                model.fit(x, y, eval_set=(vx, vy), use_best_model=False, verbose=False)
                model.save_model(directory/"catboost.cbm")
            registry[f"catboost-{recipe}"] = {"kind": "catboost", "recipe": recipe, "file": f"{recipe}/catboost.cbm",
                "fit_seconds": time.perf_counter()-started, "reused": recipe == "current_mean5",
                "validation": score(vy, model.predict_proba(vx, thread_count=4).argmax(1))}
            for kind in ("ridge", "catboost"):
                record = registry[f"{kind}-{recipe}"]
                record["sha256"] = digest(output/record["file"])
            write_json(output/"progress.json", registry)
            del x, vx, model
    # Seal every model/selection before touching the test partition.
    write_json(output/"models.json", registry)
    test = Corpus(data, "test", verify=False)
    np.save(output/"test-index.npy", test.index)
    results = {}
    for recipe, definition in config["recipes"].items():
        directory = output/recipe
        export(binary, test, recipe, directory/"test.f32")
        x = matrix(directory/"test.f32", len(test), definition["dimension"])
        for kind in ("ridge", "catboost"):
            name = f"{kind}-{recipe}"
            if kind == "ridge":
                w = np.fromfile(output/registry[name]["file"], np.float64)
                predictions = ridge_predict(x, w[:-3].reshape(-1, 3), w[-3:])
            else:
                model = CatBoostClassifier()
                model.load_model(output/registry[name]["file"])
                predictions = model.predict_proba(x, thread_count=4)
            predictions.tofile(directory/f"{kind}-test.f64")
            results[name] = score(test.targets(), predictions.argmax(1), predictions if kind == "catboost" else None)
    write_json(output/"result.json", results)
    write_json(output/"status.json", {"status": "complete"})
    return results


def benchmark(data, run, output, library, cpu):
    import torch
    if os.sched_getaffinity(0) != {cpu} or torch.get_num_threads() != 1:
        raise ValueError("Measurement requires one pinned CPU and one Torch thread")
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    run, library = Path(run), Path(library)
    config, registry = read_json(CONFIG), read_json(run/"models.json")
    if read_json(run/"status.json")["status"] != "complete":
        raise ValueError("Training must be complete before measurement")
    if read_json(run/"plan.json")["config_sha256"] != digest(CONFIG):
        raise ValueError("Training and measurement configuration differ")
    binding = provenance(data, run)
    if binding != read_json(run/"provenance.json"):
        raise ValueError("Dataset, versions or native source/executable changed")
    binary = run/"feature-runtime"
    from catboost import CatBoostClassifier
    validation = Corpus(data, "validation")
    test = Corpus(data, "test")
    np.testing.assert_array_equal(test.index, np.load(run/"test-index.npy"))
    write_json(output/"plan.json", {"environment": environment(), "config": config, "cpu": cpu,
               "library_sha256": digest(library), "models_sha256": digest(run/"models.json"),
               "boundary": "Native completed call to three scores/probabilities and argmax; streaming includes state update; raw feature generation excluded"})
    results, checks = {}, {}
    count = config["calls"]
    first_file, first_edge = test.index[0]
    raw = test.x[first_file]
    raw[first_edge-383:first_edge+1].tofile(output/"prefill.f32")
    raw[first_edge+1:first_edge+1+count].tofile(output/"snapshots.f32")
    for recipe, definition in config["recipes"].items():
        x = matrix(run/recipe/"test.f32", len(test), definition["dimension"])
        x[1:count+1].tofile(output/f"{recipe}-timing.f32")
    validation_checks = {}
    for name, record in registry.items():
        recipe, kind = record["recipe"], record["kind"]
        if digest(run/record["file"]) != record["sha256"]:
            raise ValueError("Model hash mismatch")
        x = matrix(run/recipe/"validation.f32", len(validation), config["recipes"][recipe]["dimension"])
        positions = np.linspace(0, len(validation)-1, 256, dtype=int)
        probe = x[positions]
        probe.tofile(output/f"{name}-validation.f32")
        if kind == "ridge":
            w = np.fromfile(run/record["file"], np.float64)
            expected = ridge_predict(probe, w[:-3].reshape(-1, 3), w[-3:])
        else:
            model = CatBoostClassifier()
            model.load_model(run/record["file"])
            expected = model.predict_proba(probe, thread_count=1)
        command(binary, "run", recipe, kind, run/record["file"], library, output/f"{name}-validation.f32",
                output/f"{name}-validation.f64", "predict", "-", "-", len(probe))
        actual = np.fromfile(output/f"{name}-validation.f64", np.float64).reshape(-1, 3)
        np.testing.assert_allclose(actual, expected, atol=config[f"{kind}_atol"], rtol=config[f"{kind}_rtol"])
        np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
        validation_checks[name] = {"count": len(probe), "max_abs_error": float(np.abs(actual-expected).max())}
    write_json(output/"validation-checks.json", validation_checks)
    # Validate native quality before any latency run. Reference weights are never refit.
    for name, record in registry.items():
        recipe, kind = record["recipe"], record["kind"]
        if digest(run/record["file"]) != record["sha256"]:
            raise ValueError("Model hash mismatch")
        command(binary, "run", recipe, kind, run/record["file"], library, run/recipe/"test.f32",
                output/f"{name}-test.f64", "predict", "-", "-", len(test))
        actual = np.fromfile(output/f"{name}-test.f64", np.float64).reshape(-1, 3)
        expected = np.fromfile(run/recipe/f"{kind}-test.f64", np.float64).reshape(-1, 3)
        check = compare(expected, actual, config[f"{kind}_atol"], config[f"{kind}_rtol"])
        check = {k: v for k, v in check.items() if not k.startswith("probability_")}
        if check["tolerance_pass_count"] != len(test) or check["class_disagreements_finite"]:
            raise ValueError(f"Native quality gate failed: {name}: {check}")
        checks[name] = check
        results[name] = {"quality": score(test.targets(), actual.argmax(1), actual if kind == "catboost" else None)}
    write_json(output/"native-checks.json", checks)
    schedule = [{"name": name, "boundary": boundary, "repeat": repeat}
                for repeat, jobs in enumerate(measurement_order(list(registry), config["repeats"]))
                for name, boundary in jobs]
    write_json(output/"schedule.json", schedule)
    samples = {}
    for job in schedule:
        name, boundary, repeat = job["name"], job["boundary"], job["repeat"]
        record = registry[name]
        recipe, kind = record["recipe"], record["kind"]
        mode = "stream" if boundary == "local_preparation" else "prepared"
        path = output/f"{name}-{mode}-{repeat}.f64"
        command(binary, "run", recipe, kind, run/record["file"], library, output/f"{recipe}-timing.f32",
                path, mode, output/"snapshots.f32", output/"prefill.f32", count)
        actual = np.fromfile(path, np.float64).reshape(-1, 5)
        expected = np.fromfile(run/recipe/f"{kind}-test.f64", np.float64).reshape(-1, 3)[1:count+1]
        np.testing.assert_allclose(actual[:, 1:4], expected, atol=config[f"{kind}_atol"], rtol=config[f"{kind}_rtol"])
        np.testing.assert_array_equal(actual[:, 4], expected.argmax(1))
        samples.setdefault((name, mode), []).append(actual[:, 0])
    for (name, mode), values in samples.items():
        results[name][mode] = latency_summary(np.concatenate(values))
        results[name][mode+"_repeats"] = [latency_summary(v) for v in values]
    write_json(output/"result.json", results)
    lines = ["# CatBoost and ridge feature comparison", "",
             "Same FI-2010 source-compatible caveated protocol, supplied labels and 138,408 eligible test IDs. Seed42 only. "
             "Fixed depth4/256-round CatBoost; validation-selected ridge alpha/weighting with train-only scaling folded into FP64 weights. "
             "The completed288-feature CatBoost checkpoint is reused. No test selection or feature selection.", "",
             "Latency in microseconds: pooled quantiles over three seeded repeats of10,000 calls on one CPU. "
             "Streaming includes native rolling state update, feature assembly, prediction and argmax. "
             "Prepared excludes feature assembly. Raw feature generation and setup excluded. "
             "All full-test native predictions pass frozen score tolerances and preserve classes.", "",
             "| Model | Features | Accuracy | Macro-F1 | Prepared p50 µs | Prepared p99 µs | Streaming p50 µs | Streaming p99 µs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for recipe, definition in config["recipes"].items():
        for kind in ("ridge", "catboost"):
            row = results[f"{kind}-{recipe}"]
            lines.append(f"| {kind}/{recipe} | {definition['dimension']} | {row['quality']['accuracy']:.6f} | "
                         f"{row['quality']['macro_f1']:.6f} | {row['prepared']['p50_us']:.3f} | "
                         f"{row['prepared']['p99_us']:.3f} | {row['stream']['p50_us']:.3f} | {row['stream']['p99_us']:.3f} |")
    lines.extend(["", "## Feature provenance", "",
                  "Current: supplied FI-2010 vector. Mean5/current+mean5/concat5: "
                  "[Nousi et al., v2 §4.1/Figure1](https://arxiv.org/pdf/1809.07861), adapted to our unchanged labels/splits. "
                  "Concat ordering is current to oldest. Multiscale384 is our earlier proposed extension: current, "
                  "means5/20/100/384 and lags1/4/19/99/383; not a paper-tested or optimized feature set.", "",
                  "The upstream normalization population, internal stock/day boundaries and engineered-feature causality remain unverified. "
                  "This is a representative workflow comparison, not a leakage-free production or paper reproduction claim. "
                  "Laptop clocks, thermals and background activity are uncontrolled. "
                  "MLPLOB reference: accuracy0.888612, macro-F1 0.814709, streaming CUDA p50 497.418µs/p99 528.297µs from a separate round.", "",
                  "All model hashes, selection records, feature definitions, dataset binding, validation gates, full-test scores, "
                  "schedules and600,000 raw timing observations are retained in the adjacent JSON/binary artifacts."])
    (output/"report.md").write_text("\n".join(lines)+"\n")
    write_json(output/"status.json", {"status": "complete"})
    return results
