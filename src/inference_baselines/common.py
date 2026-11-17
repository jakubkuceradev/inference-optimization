import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def external(path):
    path = Path(path).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    checkouts = [p for p in (checkout, *checkout.parents) if (p / ".git").exists()]
    if any(path == root or root in path.parents for root in checkouts):
        raise ValueError("Experiment artifacts must be outside every checkout")
    return path


def configure(threads=1, cpu=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    # Imported numerical libraries can create worker pools immediately. Their
    # threads must inherit the timing affinity, not only the calling thread.
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment():
    import torch
    versions = {}
    for name in ("torch", "numpy", "onnx", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True)
    package = Path(__file__).resolve().parent
    return {
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        "affinity": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "precision": "FP32",
        "matmul_precision": torch.get_float32_matmul_precision(), "tf32": False,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "code_sha256": {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))},
        "lockfile_sha256": digest(root / "uv.lock") if (root / "uv.lock").exists() else None,
        "cplus_include_path": os.environ.get("CPLUS_INCLUDE_PATH"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "time_unix": time.time(), "limitations": "Laptop; thermals, frequency and background load uncontrolled",
    }
