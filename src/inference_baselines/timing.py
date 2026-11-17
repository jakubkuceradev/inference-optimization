import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .evaluation import binding, freeze_checks
from .metrics import compare, latency_summary
from .runtime import Runtime


def measurement_order(names, repeats):
    rng = np.random.default_rng(42)
    return [[list(item) for item in rng.permutation([(name, boundary) for name in names for boundary in ("prepared", "local_preparation")])] for _ in range(repeats)]


def benchmark(data, run, output, name, config):
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    names = ["eager", "compiled", "ort"] if name == "all" else [name]
    write_json(output / "status.json", {"status": "running"})
    try:
        model, manifest, selected = binding(data, run)
        if manifest["seed"] != config["timing_seed"]:
            raise ValueError("Timing must use preselected seed 42")
        if torch.get_num_threads() != 1 or len(os.sched_getaffinity(0)) != 1:
            raise ValueError("Timing requires --cpu and one thread")
        for path in Path(run).parent.glob("seed-*/status.json"):
            if read_json(path)["status"] == "running":
                raise ValueError(f"Training must finish before timing: {path}")
        comparisons = {}
        for runtime_name in names:
            evaluation = Path(run).parent / f'evaluation-{manifest["seed"]}-{runtime_name}'
            if read_json(evaluation/"status.json")["status"] != "completed":
                raise ValueError(f"Complete runtime evaluation first: {evaluation}")
            comparison = read_json(evaluation/"result.json")
            if comparison["checkpoint_sha256"] != selected["checkpoint_sha256"] or comparison["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]:
                raise ValueError("Timing/evaluation binding mismatch")
            comparisons[runtime_name] = comparison
        test = Corpus(data, "test")
        order = np.linspace(0, len(test)-1, config["calls"], dtype=np.int64)
        jobs = measurement_order(names, config["repeats"])
        plan = {"checkpoint_sha256": selected["checkpoint_sha256"], "dataset_manifest_sha256": test.manifest_hash,
                "checkpoint_load_seconds": selected["load_seconds"],
                "calls": config["calls"], "repeats": config["repeats"], "warmup": config["warmup"],
                "input_order": "uniform eligible test indices, same fixed sequence per repeat/path/boundary",
                "measurement_order": jobs, "environment": environment(),
                "excluded": ["disk/decompression", "raw engineered-feature generation", "network", "queueing"],
                "output_retention": "argmax is consumed inside timing; copying three logits to audit storage happens after each timed call",
                "resident_runtime_count": len(names), "limitations": LIMITS + ["Laptop contention/frequency/thermals uncontrolled."]}
        write_json(output/"plan.json", plan)
        np.save(output/"input-indices.npy", order)
        runtimes, timing_diagnostics, diagnostic_seconds, records = {}, {}, {}, {n: [] for n in names}
        validation = Corpus(data, "validation", verify=False)
        for runtime_name in names:
            directory = output/runtime_name
            directory.mkdir()
            write_json(directory/"status.json", {"status": "running", "runtime": runtime_name})
            # Set up and warm every path before starting the interleaved schedule.
            runtime = Runtime(model, runtime_name, directory)
            runtime(test[order[0]][0][None])
            # Quality can use several CPU threads; verify the actual one-thread
            # timing path separately with the already frozen tolerances.
            diagnostic_started = time.perf_counter()
            timing_diagnostics[runtime_name] = freeze_checks(model, runtime, validation, directory, manifest["config"])
            diagnostic_seconds[runtime_name] = time.perf_counter()-diagnostic_started
            prepared = runtime.prepare(test[order[0]][0][None])
            with torch.inference_mode():
                for _ in range(config["warmup"]):
                    runtime.invoke(prepared)
            runtimes[runtime_name] = runtime
        with torch.inference_mode():
            for repeat, assignments in enumerate(jobs):
                for runtime_name, boundary in assignments:
                    runtime = runtimes[runtime_name]
                    observations = np.empty(len(order), dtype=np.int64)
                    logits = np.empty((len(order), 3), dtype=np.float32)
                    wall_start = time.perf_counter()
                    for j, index in enumerate(order):
                        if boundary == "prepared":
                            prepared = runtime.prepare(test[index][0][None])
                        before = time.perf_counter_ns()
                        if boundary == "local_preparation":
                            prepared = runtime.prepare(test[index][0][None])
                        value = runtime.invoke(prepared)
                        # CPU calls complete synchronously; consume output inside the boundary.
                        int(value.argmax())
                        observations[j] = time.perf_counter_ns()-before
                        logits[j] = value if isinstance(value, np.ndarray) else value.numpy()
                    wall_seconds = time.perf_counter()-wall_start
                    np.save(output/runtime_name/f'{boundary}-{repeat}-ns.npy', observations)
                    np.save(output/runtime_name/f'{boundary}-{repeat}-logits.npy', logits)
                    records[runtime_name].append({"repeat": repeat, "boundary": str(boundary),
                        "wall_seconds": wall_seconds, "wall_calls_per_second": len(order)/wall_seconds,
                        **latency_summary(observations)})
        for runtime_name in names:
            timed_numerical = []
            if "eager" not in names:
                with np.load(Path(run).parent/f'evaluation-{manifest["seed"]}-eager'/"predictions.npz") as saved:
                    bulk_reference = saved["reference_logits"][order]
            for record in records[runtime_name]:
                stem = f"{record['boundary']}-{record['repeat']}-logits.npy"
                expected = np.load(output/"eager"/stem) if "eager" in names else bulk_reference
                actual = np.load(output/runtime_name/stem)
                timed_numerical.append({"repeat": record["repeat"], "boundary": record["boundary"],
                    "reference": "one-thread eager batch-one timed outputs" if "eager" in names else "canonical eager bulk outputs",
                    **compare(expected, actual, manifest["config"]["atol"], manifest["config"]["rtol"])})
            result = {**plan, "runtime_setup": runtimes[runtime_name].metadata(), "observations": records[runtime_name],
                      "timed_numerical": timed_numerical,
                      "diagnostic_setup_seconds": diagnostic_seconds[runtime_name],
                      "numerical": comparisons[runtime_name]["numerical"], "diagnostics": timing_diagnostics[runtime_name],
                      "quality_environment": comparisons[runtime_name]["environment"],
                      "evaluation_result_sha256": digest(Path(run).parent/f'evaluation-{manifest["seed"]}-{runtime_name}'/"result.json"),
                      "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "memory_scope": "process peak includes all resident runtimes and setup/compiler allocations",
                      "checkpoint_bytes": (Path(run)/"selected.pt").stat().st_size,
                      "measurement": "completed-call service with output argmax; wall capacity includes loop and per-call assembly"}
            write_json(output/runtime_name/"result.json", result)
            write_json(output/runtime_name/"status.json", {"status": "completed", "runtime": runtime_name})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "runtimes": names}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        for path in output.glob("*/status.json"):
            if read_json(path)["status"] == "running":
                write_json(path, {"status": "failed", "reason": repr(error)})
        raise
