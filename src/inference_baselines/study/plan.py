"""Strict explicit plans and stable training identities."""
import hashlib
import json
from pathlib import Path
import re

from ..common import read_json
from ..model import MLPLOB

LEGACY_RECIPE = dict(optimizer="Adam", learning_rate=0.0003, epochs=20, patience=5,
              minimum_epochs=8, precision="FP32", tf32=False, weights="raw",
              selection="equal-day-macro-f1-earliest-tie", horizon=10)
DUAL_METRIC_RECIPE = dict(LEGACY_RECIPE, epochs=None, early_stopping="f1-or-validation-loss")
RECIPE = dict(DUAL_METRIC_RECIPE, early_stopping="validation-loss",
              selection="minimum-validation-cross-entropy-earliest-tie",
              scheduler=dict(name="ReduceLROnPlateau", mode="min", factor=0.5,
                             patience=2, threshold=0.0, threshold_mode="abs",
                             cooldown=0, min_lr=0.000003, eps=1e-8))
TIMING = dict(warmups=200, repeats=3, calls=10000, seed=42)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"Expected fields {sorted(required)} (optional {sorted(optional)})")


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Expected integer in [{minimum}, {maximum}], got {value!r}")
    return value


def positive(value):
    if type(value) not in (int, float) or not 0 < value < float("inf"):
        raise ValueError("Expected finite positive budget")
    return value


def model_config(job):
    expansion = (4, 2) if job["expansion"] == "standard" else (2, 1)
    return dict(hidden_dim=job["width"], num_layers=job["depth"], seq_size=job["history"],
                num_features=job["features"], dataset_type="FI_2010",
                intermediate_expansion=expansion[0], final_expansion=expansion[1])


def validate(plan):
    fields(plan, ("version", "name", "kind", "support", "jobs", "recipe", "timing", "devices"), ("selection_provenance",))
    if type(plan["version"]) is not int or plan["version"] not in (1, 2, 3) or plan["kind"] not in ("screen", "followup", "smoke"):
        raise ValueError("Unsupported plan version/kind")
    if not isinstance(plan["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", plan["name"]):
        raise ValueError("Invalid plan name")
    recipe = {1: LEGACY_RECIPE, 2: DUAL_METRIC_RECIPE, 3: RECIPE}[plan["version"]]
    if plan["support"] != 384 or identity(plan["recipe"]) != identity(recipe):
        raise ValueError("Study requires declared history384 support and adopted recipe")
    fields(plan["timing"], TIMING)
    for k, v in plan["timing"].items():
        integer(v, 0 if k == "seed" else 1, 2**31-1)
    if plan["kind"] != "smoke" and plan["timing"] != TIMING:
        raise ValueError("Production timing schedule is frozen")
    if plan["devices"] != ["cpu", "cuda"]:
        raise ValueError("Both CPU and CUDA statuses are required")
    if not isinstance(plan["jobs"], list) or not plan["jobs"]:
        raise ValueError("Plan must list explicit jobs")
    names, shapes, references = set(), set(), 0
    for job in plan["jobs"]:
        fields(job, ("id", "history", "width", "depth", "features", "expansion", "seed"))
        if not isinstance(job["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]):
            raise ValueError("Invalid job ID")
        integer(job["history"], 4, 384); integer(job["width"], 4, 512)
        integer(job["depth"], 1, 4); integer(job["seed"], 0, 2**31-1)
        if job["history"] % 4 or job["width"] % 4 or job["features"] not in (40, 144) or job["expansion"] not in ("standard", "reduced"):
            raise ValueError("Unsupported model shape/expansion")
        if plan["kind"] == "screen" and job["seed"] != 42:
            raise ValueError("Screen uses seed42")
        if plan["kind"] == "followup" and (job["seed"] == 42 or job["history"] == 384):
            raise ValueError("Followup cannot repeat screen seed/reference")
        if job["history"] == 384:
            references += 1
            if (job["width"], job["depth"], job["features"], job["expansion"], job["seed"]) != (144, 3, 144, "standard", 42):
                raise ValueError("Only the single paper-sized reference is allowed")
        shape = identity({k:v for k,v in job.items() if k != "id"})
        if shape in shapes or job["id"] in names:
            raise ValueError("Duplicate job/configuration")
        names.add(job["id"]); shapes.add(shape)
    if references > 1:
        raise ValueError("Repeated history384 reference")
    if plan["kind"] == "followup":
        if not plan.get("selection_provenance"):
            raise ValueError("Followup requires selection provenance")
        groups = {}
        for job in plan["jobs"]:
            key = identity({k:v for k,v in job.items() if k not in ("id", "seed")})
            groups.setdefault(key, set()).add(job["seed"])
        if any(len(v) != 2 for v in groups.values()) or len(set.union(*groups.values())) != 2:
            raise ValueError("Followup declares exactly two common additional seeds")
    return plan


def compose(plans):
    jobs = []
    for plan in plans:
        jobs.extend(validate(plan)["jobs"])
        if identity(plan["recipe"]) != identity(plans[0]["recipe"]):
            raise ValueError("Composed plans require the same training recipe")
    combined = dict(plans[0], jobs=jobs)
    validate(combined)
    return jobs


def describe(plan):
    validate(plan)
    return [{**job, "unique_parameters": sum(p.numel() for p in MLPLOB(**model_config(job)).parameters())}
            for job in plan["jobs"]]


def load(path):
    return validate(read_json(path))
