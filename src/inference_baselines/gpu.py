"""Same-checkpoint CUDA checks and completed-call batch-one measurements."""
import copy
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import digest, environment, external, read_json, write_json
from .data import Corpus, LIMITS
from .evaluation import binding, freeze_checks
from .metrics import compare, latency_summary, score, softmax
from .timing import measurement_order

NAMES = ("eager", "graph", "compiled")


def supported_runtimes(model):
    runtimes, unsupported = {}, {}
    for name in NAMES:
        try:
            runtimes[name] = CudaRuntime(model, name)
        except RuntimeError as error:
            if name != "compiled" or "not supported on Python" not in str(error):
                raise
            unsupported[name] = str(error)
    return runtimes, unsupported


class CudaRuntime:
    """Own stable staging addresses, captures and compiled execution profiles."""
    def __init__(self, model, name):
        if name not in NAMES:
            raise ValueError(name)
        self.name, self.model, self.profiles = name, model, {}
        if name == "compiled":
            self.model = torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead")

    def profile(self, shape):
        shape = tuple(shape)
        if shape not in self.profiles:
            start = time.perf_counter()
            host = torch.zeros(shape, dtype=torch.float32, pin_memory=True)
            device = host.to("cuda", non_blocking=True)
            state = {"host": host, "input": device}
            if self.name == "graph":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), torch.inference_mode():
                    for _ in range(3):
                        self.model(device)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.inference_mode():
                    value = self.model(device)
                state.update(graph=graph, output=value)
            torch.cuda.synchronize()
            state["setup_seconds"] = time.perf_counter()-start
            self.profiles[shape] = state
        return self.profiles[shape]

    def prepare(self, x, wait=True):
        x = np.ascontiguousarray(x, dtype=np.float32)
        state = self.profile(x.shape)
        state["host"].copy_(torch.from_numpy(x))
        state["input"].copy_(state["host"], non_blocking=True)
        if wait:
            torch.cuda.current_stream().synchronize()
        return state

    def invoke(self, state):
        if self.name == "graph":
            state["graph"].replay()
            return state["output"]
        if self.name == "compiled":
            torch.compiler.cudagraph_mark_step_begin()
        return self.model(state["input"])

    def __call__(self, x):
        with torch.inference_mode():
            return self.invoke(self.prepare(x, wait=False)).cpu().numpy()

    def metadata(self):
        return {"name": self.name, "device": "cuda:0", "precision": "FP32; TF32 off",
                "compile_mode": "fullgraph, dynamic=False, reduce-overhead" if self.name == "compiled" else None,
                "graph_policy": "explicit CUDA Graph replay" if self.name == "graph" else "compiler managed when applicable" if self.name == "compiled" else "eager launches",
                "profiles": {str(k): {"setup_seconds": v["setup_seconds"]} for k, v in self.profiles.items()}}


def oracle(root, corpus, checkpoint):
    result = read_json(root/"result.json")
    if read_json(root/"status.json")["status"] != "completed" or result["checkpoint_sha256"] != checkpoint or result["dataset_manifest_sha256"] != corpus.manifest_hash:
        raise ValueError(f"Reference binding mismatch: {root}")
    with np.load(root/"predictions.npz") as saved:
        np.testing.assert_array_equal(saved["ids"], corpus.ids())
        np.testing.assert_array_equal(saved["targets"], corpus.targets())
        return saved["reference_logits"].copy()


def evaluate_gpu(data, run, output):
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable; CPU fallback is forbidden")
        model, manifest, selected = binding(data, run)
        config = manifest["config"]
        execution = environment()
        validation = Corpus(data, "validation")
        gpu_model = copy.deepcopy(model).cuda().eval()
        runtimes, unsupported = supported_runtimes(gpu_model)
        write_json(output/"runtime-support.json", {"available": list(runtimes), "unsupported": unsupported,
                   "ort": "Not installed: the preserved training environment has CPU ONNX Runtime only."})
        for name, runtime in runtimes.items():
            directory = output/name
            directory.mkdir()
            freeze_checks(model, runtime, validation, directory, config)
        # Expose the CPU package-version difference before test scoring.
        old = Path(run).parent/f'evaluation-{manifest["seed"]}-eager'
        with np.load(old/"validation-diagnostics.npz") as original, np.load(output/"eager/validation-diagnostics.npz") as current:
            np.testing.assert_array_equal(original["validation_index"], current["validation_index"])
            version_check = compare(original["validation_bulk_reference"], current["validation_bulk_reference"], config["atol"], config["rtol"])
        write_json(output/"cpu-version-check.json", {"saved_source": str(old), "saved_sha256": digest(old/"validation-diagnostics.npz"), **version_check})
        test = Corpus(data, "test", verify=False)
        expected = oracle(old, test, selected["checkpoint_sha256"])
        predictions = {name: [] for name in runtimes}
        start = time.perf_counter()
        size = config["evaluation_batch_size"]
        for offset in range(0, len(test), size):
            x = np.stack([test[i][0] for i in range(offset, min(offset+size, len(test)))])
            for name, runtime in runtimes.items():
                predictions[name].append(runtime(x))
            if offset % (size*500) == 0:
                print(f"GPU quality {offset}/{len(test)}", flush=True)
        predictions = {name: np.concatenate(parts) for name, parts in predictions.items()}
        targets = test.targets()
        for name, values in predictions.items():
            directory = output/name
            np.savez(directory/"predictions.npz", ids=test.ids(), targets=targets, reference_logits=values,
                     probabilities=softmax(values), classes=values.argmax(1))
            result = {"checkpoint_sha256": selected["checkpoint_sha256"], "dataset_manifest_sha256": test.manifest_hash,
                      "environment": execution, "runtime": runtimes[name].metadata(), "limitations": LIMITS,
                      "cpu_reference_sha256": digest(old/"predictions.npz"),
                      "cpu_numerical": compare(expected, values, config["atol"], config["rtol"]),
                      "cuda_eager_numerical": compare(predictions["eager"], values, config["atol"], config["rtol"]),
                      "metrics": score(targets, values.argmax(1), softmax(values)) if np.isfinite(values).all() else {"status": "nonfinite; financial metrics withheld"},
                      "total_quality_seconds_all_paths": time.perf_counter()-start}
            write_json(directory/"result.json", result)
            write_json(directory/"status.json", {"status": "completed"})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "runtimes": list(runtimes), "unsupported": unsupported}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise


def benchmark_gpu(data, run, evaluation, output, config):
    output, evaluation = external(output), Path(evaluation)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"status.json", {"status": "running"})
    try:
        if torch.get_num_threads() != 1 or len(os.sched_getaffinity(0)) != 1:
            raise ValueError("CUDA timing requires one pinned host CPU/thread")
        model, manifest, selected = binding(data, run)
        if manifest["seed"] != config["timing_seed"]:
            raise ValueError("Timing must use preselected seed 42")
        for path in Path(run).parent.glob("seed-*/status.json"):
            if read_json(path)["status"] == "running":
                raise ValueError("Finish training before timing")
        test = Corpus(data, "test")
        names = read_json(evaluation/"runtime-support.json")["available"]
        for name in names:
            oracle(evaluation/name, test, selected["checkpoint_sha256"])
        order = np.linspace(0, len(test)-1, config["calls"], dtype=np.int64)
        jobs = measurement_order(names, config["repeats"])
        np.save(output/"input-indices.npy", order)
        plan = {"checkpoint_sha256": selected["checkpoint_sha256"], "dataset_manifest_sha256": test.manifest_hash,
                "calls": config["calls"], "repeats": config["repeats"], "warmup": config["warmup"],
                "environment": environment(), "order": jobs,
                "prepared": "Resident GPU input through completed GPU logits and GPU argmax, host wall timer with stream synchronization; audit D2H after timer",
                "local_preparation": "CPU history copy, pinned staging, H2D, inference, D2H three logits, CPU argmax",
                "excluded": ["disk/decompression", "raw engineered-feature generation", "network", "queueing"],
                "limitations": LIMITS+["Laptop thermals/frequency/contention uncontrolled; all three paths resident.", "CPU reference uses torch 2.15; CUDA uses 2.9. Device and library differences are confounded."],
                "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR")}
        write_json(output/"plan.json", plan)
        gpu_model = copy.deepcopy(model).cuda().eval()
        runtimes, records = {}, {n: [] for n in names}
        validation = Corpus(data, "validation", verify=False)
        for name in names:
            directory = output/name
            directory.mkdir()
            runtime = CudaRuntime(gpu_model, name)
            freeze_checks(model, runtime, validation, directory, manifest["config"])
            with torch.inference_mode():
                state = runtime.prepare(test[order[0]][0][None])
                for _ in range(config["warmup"]):
                    value = runtime.invoke(state)
                    value.argmax()
                torch.cuda.synchronize()
            runtimes[name] = runtime
        write_json(output/"phase.json", {"phase": "timing", "time_unix": time.time()})
        with torch.inference_mode():
            for repeat, jobs_for_repeat in enumerate(jobs):
                for name, boundary in jobs_for_repeat:
                    runtime = runtimes[name]
                    ns, logits = np.empty(len(order), np.int64), np.empty((len(order),3), np.float32)
                    started = time.perf_counter()
                    for j, index in enumerate(order):
                        if boundary == "prepared":
                            state = runtime.prepare(test[index][0][None])
                        before = time.perf_counter_ns()
                        if boundary == "local_preparation":
                            state = runtime.prepare(test[index][0][None], wait=False)
                        value = runtime.invoke(state)
                        if boundary == "prepared":
                            decision = value.argmax()
                            torch.cuda.current_stream().synchronize()
                        else:
                            host_value = value.cpu().numpy()
                            decision = int(host_value.argmax())
                        ns[j] = time.perf_counter_ns()-before
                        logits[j] = value.cpu().numpy() if boundary == "prepared" else host_value
                    elapsed = time.perf_counter()-started
                    stem = f"{boundary}-{repeat}"
                    np.save(output/name/f"{stem}-ns.npy", ns)
                    np.save(output/name/f"{stem}-logits.npy", logits)
                    record = {"repeat": repeat, "boundary": str(boundary), "wall_seconds": elapsed, **latency_summary(ns)}
                    records[name].append(record)
                    print(f"GPU {name} {boundary} repeat {repeat}: p50 {record['p50_us']:.1f} us p99 {record['p99_us']:.1f} us", flush=True)
        for name in names:
            numerical = []
            for row in records[name]:
                stem = f"{row['boundary']}-{row['repeat']}-logits.npy"
                numerical.append({"repeat": row["repeat"], "boundary": row["boundary"],
                    **compare(np.load(output/"eager"/stem), np.load(output/name/stem), manifest["config"]["atol"], manifest["config"]["rtol"])})
            write_json(output/name/"result.json", {**plan, "runtime": runtimes[name].metadata(),
                "observations": records[name], "timed_numerical": numerical,
                "evaluation_sha256": digest(evaluation/name/"result.json"),
                "peak_all_paths_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_all_paths_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(output/"status.json", {"status": "completed"})
        return {"output": str(output), "runtimes": names}
    except Exception as error:
        write_json(output/"status.json", {"status": "failed", "reason": repr(error)})
        raise
