"""The supplied matrices are authoritative; unknown segment provenance stays unknown."""
from pathlib import Path
import zipfile

import numpy as np

from .common import digest, external, read_json, write_json

FILES = {
    "fi2010_Train_Dst_NoAuction_ZScore_CF_7.zip": (254750, "743694f68e830b0de5280c23a9088975c6a07d84259296a6df58253a4d4e0ea4"),
    "fi2010_Test_Dst_NoAuction_ZScore_CF_7.zip": (55478, "6b87c8f653aad689968fe086ceec751726e5dc6165421d2d5bfd86e43829bda8"),
    "fi2010_Test_Dst_NoAuction_ZScore_CF_8.zip": (52172, "78f9de735cbb53be54d805852a4f73d65be5a53890731700de439ca4f1033df3"),
    "fi2010_Test_Dst_NoAuction_ZScore_CF_9.zip": (31937, "53a222799f95486bfbea5356f4f62c36651c93c36e3217246c0917a976030e9b"),
}
PROTOCOL = "fi2010-source-compatible-caveated-v1"
LIMITS = [
    "Internal stock/session boundaries unavailable; histories may cross unknown boundaries.",
    "Supplied z-score fit population/statistics unavailable; validation contribution is unverified.",
    "Engineered-feature formulas are described in the dataset paper; implementation causality is unverified.",
    "Row 145 is TLOB horizon 10; original paper uses representation-event horizon nomenclature. No physical clock.",
    "Ten-column tail exclusion is conservative at known boundaries, not proof of internal target support.",
]


def eligible(start, stop, history=384, future=10):
    return np.arange(start + history - 1, max(start + history - 1, stop - future), dtype=np.int64)


def prepare(source, output):
    source, output = Path(source).resolve(), external(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    try:
        entries, partitions = [], {p: [] for p in ("train", "validation", "test")}
        for file_id, (name, (count, expected)) in enumerate(FILES.items()):
            path = source / name
            if digest(path) != expected:
                raise ValueError(f"SHA256 mismatch: {path}")
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"ZIP integrity failure: {path}")
                members = [m for m in archive.namelist() if not m.endswith("/")]
                if len(members) != 1:
                    raise ValueError(f"Expected one matrix member: {members}")
                with archive.open(members[0]) as stream:
                    matrix = np.loadtxt(stream, dtype=np.float64)
            if matrix.shape != (149, count) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid matrix shape or nonfinite values: {name}")
            if not np.isin(matrix[144:], [1, 2, 3]).all():
                raise ValueError(f"Invalid supplied labels: {name}")
            x = np.ascontiguousarray(matrix[:144].T, dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError("FP32 conversion overflow")
            y = matrix[144].astype(np.int64) - 1
            np.save(output / f"x-{file_id}.npy", x)
            np.save(output / f"y-{file_id}.npy", y)
            entries.append({"file_id": file_id, "source": str(path), "sha256": expected,
                            "member": members[0], "shape": list(matrix.shape),
                            "constant_feature_rows_1based": (np.flatnonzero(np.ptp(x, axis=0) == 0) + 1).tolist(),
                            "class_counts_all_columns": np.bincount(y, minlength=3).tolist()})
            spans = [("train", 0, 203800), ("validation", 203800, count)] if file_id == 0 else [("test", 0, count)]
            for partition, start, stop in spans:
                edges = eligible(start, stop)
                partitions[partition].append({"file_id": file_id, "start": start, "stop": stop,
                    "segment": f"file-{file_id}-{partition}", "segment_verified": "partition/file only",
                    "stock": None, "session": None, "raw_count": stop-start,
                    "history_exclusions": min(383, stop-start), "future_exclusions": min(10, max(0, stop-start-383)),
                    "eligible_count": len(edges)})
            del matrix, x, y
        for partition, spans in partitions.items():
            index = np.concatenate([np.column_stack((np.full(s["eligible_count"], s["file_id"], dtype=np.int64), eligible(s["start"], s["stop"]))) for s in spans])
            np.save(output / f"{partition}-index.npy", index)
        manifest = {"protocol": PROTOCOL, "history": 384, "future_tail_columns": 10,
                    "features": list(range(1, 145)), "label_row_1based": 145,
                    "classes": {"0": "up", "1": "stable", "2": "down"},
                    "normalization": "supplied z-score, no refit; BiN remains in model",
                    "limitations": LIMITS, "sources": entries, "partitions": partitions,
                    "license": "CC-BY-4.0; Ntakaris et al./BigDataFinance, FI-2010",
                    "provider_metadata_sha256": digest(source / "fi2010-provider-metadata.json"),
                    "arrays": {p.name: digest(p) for p in sorted(output.glob("*.npy"))}}
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", {"status": "completed", "manifest_sha256": digest(output / "manifest.json")})
        return manifest
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


class Corpus:
    def __init__(self, root, partition, verify=True):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        if read_json(self.root / "status.json")["status"] != "completed":
            raise ValueError("Incomplete dataset preparation")
        if verify:
            for name, expected in self.manifest["arrays"].items():
                if digest(self.root / name) != expected:
                    raise ValueError(f"Prepared array hash mismatch: {name}")
        self.manifest_hash = digest(self.root / "manifest.json")
        self.partition = partition
        self.index = np.load(self.root / f"{partition}-index.npy", mmap_mode="r")
        self.x = {s["file_id"]: np.load(self.root / f'x-{s["file_id"]}.npy', mmap_mode="r") for s in self.manifest["sources"]}
        self.y = {s["file_id"]: np.load(self.root / f'y-{s["file_id"]}.npy', mmap_mode="r") for s in self.manifest["sources"]}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        file_id, edge = self.index[i]
        return np.array(self.x[file_id][edge-383:edge+1], copy=True), int(self.y[file_id][edge])

    def targets(self):
        return np.array([self.y[f][e] for f, e in self.index], dtype=np.int64)

    def ids(self):
        sources = self.manifest["sources"]
        return np.array([f'{sources[f]["sha256"]}:{sources[f]["member"]}:{e}:{self.partition}:file-{f}-{self.partition}:internal-unverified' for f, e in self.index])
