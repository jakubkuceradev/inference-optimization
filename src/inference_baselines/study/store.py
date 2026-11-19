"""Atomic publication, process locks and immutable execution bindings."""
from contextlib import contextmanager
import fcntl
import importlib.metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

from ..common import digest, environment, external, read_json, write_json
from .plan import fields, identity, integer, positive, validate, describe
from .data import Development


def provenance():
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    checkout_layout = package.parent.name == "src" and ((root/".git").is_file() or (root/".git/HEAD").is_file())
    def git(*args):
        if not checkout_layout: return None
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ("torch","numpy","onnx","onnxruntime"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(git_revision=git("rev-parse","HEAD"), dirty=bool(git("status","--porcelain")),
                source_hashes={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob("*")) if p.suffix in (".py",".cpp",".h")},
                dependency_versions=versions, python=sys.version, lockfile_sha256=digest(root/"uv.lock") if checkout_layout and (root/"uv.lock").is_file() else None,
                source_context="checkout" if git("rev-parse","HEAD") is not None else "installed package; git/lockfile unavailable, actual package sources and dependencies hashed")


@contextmanager
def lock(path, exclusive=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Runner activity is locked: {path}") from None
        if exclusive:
            stream.seek(0); stream.truncate()
            stream.write(f"{socket.gethostname()} pid={os.getpid()}\n"); stream.flush()
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def activity(measurement=False):
    # Host-local namespace is common even when campaigns use different stores.
    return lock(Path("/tmp")/f"inference-study-{os.getuid()}"/"activity.lock", measurement)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(plan, data, preflight, execution, output):
    validate(plan)
    corpus = Development(data)
    fields(execution, ("batch_size","batch_exceptions","run_seconds","campaign_seconds","artifact_store","training_device"))
    integer(execution["batch_size"],1,512)
    positive(execution["run_seconds"]); positive(execution["campaign_seconds"])
    fields(execution["batch_exceptions"], (), [j["id"] for j in plan["jobs"]])
    for name,batch in execution["batch_exceptions"].items():
        integer(batch,1,512)
        if plan["name"] == "mlplob-limited":
            raise ValueError("Limited study requires a common batch")
    if execution["training_device"] not in ("cpu","cuda"):
        raise ValueError("Explicit training device required")
    if preflight["plan_hash"] != identity(plan) or preflight["data_hash"] != corpus.binding or preflight["status"] != "preflight-recorded":
        raise ValueError("Preflight does not bind this plan/data")
    if preflight["code"] != provenance() or preflight["device"] != execution["training_device"]:
        raise ValueError("Preflight code/dependency/device mismatch")
    if plan["kind"] != "smoke" and execution["batch_size"] not in (256,512):
        raise ValueError("Production batch must be 256 or 512")
    for job in plan["jobs"]:
        batch = execution["batch_exceptions"].get(job["id"],execution["batch_size"])
        if not any(row["job_id"] == job["id"] and row["batch"] == batch and row["status"] == "supported" for row in preflight["batches"]):
            raise ValueError(f"No successful preflight batch evidence for {job['id']}")
    store = external(execution["artifact_store"])
    output = external(output); output.mkdir(parents=True, exist_ok=False)
    jobs = []
    parameter_counts = {j["id"]:j["unique_parameters"] for j in describe(plan)}
    for job in plan["jobs"]:
        binding = dict(model={k:v for k,v in job.items() if k not in ("id","seed")},seed=job["seed"],
                       recipe=plan["recipe"],batch=execution["batch_exceptions"].get(job["id"],execution["batch_size"]),
                       data=corpus.binding,code=preflight["code"],training_device=execution["training_device"],
                       run_seconds=execution["run_seconds"])
        jobs.append(dict(config=job,binding=binding,key=identity(binding),unique_parameters=parameter_counts[job["id"]]))
    value = dict(state="frozen",plan=plan,data=str(Path(data).resolve()),data_hash=corpus.binding,
                 execution={**execution,"artifact_store":str(store)},preflight=preflight,jobs=jobs)
    write_json(output/"study.json",dict(value,sha256=identity(value)))
    return value


def study(path, check_data=True):
    path = Path(path)
    value = read_json(path/"study.json")
    expected = value.pop("sha256")
    if identity(value) != expected or value["state"] != "frozen":
        raise ValueError("Frozen execution manifest changed")
    if check_data and Development(value["data"]).binding != value["data_hash"]:
        raise ValueError("Dataset binding changed")
    return value


def job_path(study, job):
    return Path(study["execution"]["artifact_store"])/"jobs"/job["key"]
