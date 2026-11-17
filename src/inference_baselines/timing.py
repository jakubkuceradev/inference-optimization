import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .evaluation import binding
from .metrics import latency_summary
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
                "resident_runtime_count": len(names), "limitations": LIMITS + ["Laptop contention/frequency/thermals uncontrolled."]}
        write_json(output/"plan.json", plan)
        np.save(output/"input-indices.npy", order)
        runtimes, records = {}, {n: [] for n in names}
        for runtime_name in names:
            directory = output/runtime_name
            directory.mkdir()
            write_json(directory/"status.json", {"status": "running", "runtime": runtime_name})
            # Set up and warm every path before starting the interleaved schedule.
            runtime = Runtime(model, runtime_name, directory)
            runtime(test[order[0]][0][None])
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
                    wall_seconds = time.perf_counter()-wall_start
                    np.save(output/runtime_name/f'{boundary}-{repeat}-ns.npy', observations)
                    records[runtime_name].append({"repeat": repeat, "boundary": str(boundary),
                        "wall_seconds": wall_seconds, "wall_calls_per_second": len(order)/wall_seconds,
                        **latency_summary(observations)})
        for runtime_name in names:
            result = {**plan, "runtime_setup": runtimes[runtime_name].metadata(), "observations": records[runtime_name],
                      "numerical": comparisons[runtime_name]["numerical"], "diagnostics": comparisons[runtime_name]["diagnostics"],
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
