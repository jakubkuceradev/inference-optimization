"""Named development arrays, verified target/segment binding and batched windows."""
from pathlib import Path
import shutil

import numpy as np

from ..common import digest, external, read_json, write_json
from .plan import fields, identity

PARTITIONS = ("train", "validation")
FILES = tuple(f"{part}-{kind}.npy" for part in PARTITIONS for kind in ("raw", "index", "targets"))


def expected_ids(segments):
    chunks = []
    end = 0
    seen = set()
    for segment in segments:
        fields(segment, ("group", "day", "start", "end", "reference_start"))
        if any(type(v) is not int for v in segment.values()):
            raise ValueError("Integer segment coordinates required")
        start, stop = segment["start"], segment["end"]
        key = segment["group"], segment["day"]
        if start != end or stop <= start or key in seen or segment["reference_start"] < 0:
            raise ValueError("Noncontiguous/duplicate/invalid segments")
        seen.add(key); end = stop
        edge = np.arange(start + 383, stop - 10, dtype=np.int64)
        chunks.append(np.column_stack((np.full(len(edge), key[0]), np.full(len(edge), key[1]),
                                      edge, edge-start+segment["reference_start"])))
    if not chunks:
        raise ValueError("Missing segments")
    return np.concatenate(chunks), end


class Development:
    def __init__(self, root, manifest=None):
        self.root = Path(root).resolve()
        self.manifest = manifest if manifest is not None else read_json(self.root / "manifest.json")
        m = self.manifest
        fields(m, ("version", "kind", "support", "horizon", "feature_order", "label_mapping",
                   "normalization", "segments", "files", "counts", "provenance"))
        if m["version"] != 1 or m["kind"] not in ("verified-cf5", "synthetic-smoke") or m["support"] != 384 or m["horizon"] != 10:
            raise ValueError("Verified development history384/horizon10 corpus required")
        if m["feature_order"] != list(range(1,145)) or m["label_mapping"] != {"1":"up:0", "2":"stable:1", "3":"down:2"}:
            raise ValueError("Unsupported feature order/label mapping")
        fields(m["normalization"], ("coordinates", "fit_days", "description"))
        if m["normalization"]["coordinates"] != "CF5" or m["normalization"]["fit_days"] != [1,2,3,4,5] or not m["normalization"]["description"]:
            raise ValueError("Training-only CF5 normalization provenance required")
        fields(m["segments"], PARTITIONS); fields(m["counts"], PARTITIONS); fields(m["files"], FILES)
        self.arrays = {}
        for part in PARTITIONS:
            segments = m["segments"][part]
            allowed = {1,2,3,4,5} if part == "train" else {6,7}
            if {s["day"] for s in segments} != allowed:
                raise ValueError("Development partitions must contain only the declared days")
            ids, size = expected_ids(segments)
            loaded = {}
            for kind in ("raw", "index", "targets"):
                name = f"{part}-{kind}.npy"
                path = self.root/name
                # Only these six files are opened, never a source-directory walk.
                if digest(path) != m["files"][name]:
                    raise ValueError(f"Array hash mismatch: {name}")
                loaded[kind] = np.load(path, mmap_mode="r", allow_pickle=False)
            raw, index, targets = (loaded[k] for k in ("raw", "index", "targets"))
            if raw.shape != (size,144) or raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError("Invalid raw features")
            if index.dtype != np.int64 or not np.array_equal(index, ids):
                raise ValueError("Target IDs disagree with segment/common support")
            if targets.shape != (len(ids),) or targets.dtype != np.int64 or not np.isin(targets,[0,1,2]).all():
                raise ValueError("Invalid targets")
            if len(ids) != m["counts"][part] or not len(ids):
                raise ValueError("Incorrect target count")
            self.arrays[part] = loaded
        if m["kind"] == "verified-cf5" and m["counts"] != {"train":168427,"validation":72568}:
            raise ValueError("Changed production comparison support")
        self.binding = identity(m)

    def windows(self, part, positions, history, features):
        if part not in PARTITIONS or type(history) is not int or not 4 <= history <= 384 or features not in (40,144):
            raise ValueError("Unsupported window request")
        positions = np.asarray(positions)
        a = self.arrays[part]
        edges = a["index"][positions,2]
        return np.ascontiguousarray(a["raw"][edges[:,None]+np.arange(1-history,1)[None,:], :features])


def prepare(source, output, source_manifest=None):
    source = Path(source).resolve()
    # The repository ships a pinned import descriptor for the already verified arrays.
    descriptor = Path(source_manifest) if source_manifest else Path(__file__).parents[1]/"configs/mlplob-development.json"
    manifest = read_json(descriptor)
    Development(source, manifest)
    output = external(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        shutil.copyfile(source/name, output/name)
    write_json(output/"manifest.json", manifest)
    return Development(output).manifest


def synthetic(output):
    output = external(output); output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(917)
    segments = {}
    counts = {}
    for part, days in (("train",range(1,6)),("validation",range(6,8))):
        segments[part] = [dict(group=1, day=d, start=i*400,end=(i+1)*400,reference_start=(d-1)*400) for i,d in enumerate(days)]
        index, size = expected_ids(segments[part])
        np.save(output/f"{part}-raw.npy",rng.normal(size=(size,144)).astype(np.float32))
        np.save(output/f"{part}-index.npy",index)
        np.save(output/f"{part}-targets.npy", (index[:,3]%3).astype(np.int64))
        counts[part] = len(index)
    manifest = dict(version=1,kind="synthetic-smoke",support=384,horizon=10,feature_order=list(range(1,145)),
                    label_mapping={"1":"up:0","2":"stable:1","3":"down:2"},
                    normalization=dict(coordinates="CF5",fit_days=[1,2,3,4,5],description="Synthetic independent Gaussian smoke; no market quality claims"),
                    segments=segments,counts=counts,files={n:digest(output/n) for n in FILES},provenance={"generator":"numpy default_rng(917)"})
    write_json(output/"manifest.json",manifest)
    return Development(output).manifest
