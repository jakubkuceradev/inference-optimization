# Durable MLPLOB study

`inference-study` implements the development-only MLPLOB size study. The limited
manifest lists 44 seed42 models; the separate medium manifest adds three models,
including the only history384 reference. Both use identical history384/horizon10
support. The eight reduced-expansion models have exact standard counterparts.
The first 40 supplied features are a projection of the same 144-feature corpus.

Install the package with the CPU or CUDA extra in separate Python3.14 environments.
The optional `report` extra adds Matplotlib; JSON and CSV need no plotting extra.
Use absolute interpreter paths with `--python` when selecting the worker environment.
Every fit runs in a fresh sequential process. Missing CUDA never becomes CPU training.

The shipped import descriptor pins the six arrays from the verified CF5 development
preparation. Preparation opens only those named arrays and checks their hashes,
segments, IDs, targets, feature order and training-only coordinate provenance.
Days1–5 train, days6–7 validate; this CLI has no hold-out/test partition. Supplied
engineered-feature causality is not established by coordinate verification.
Artifacts and datasets must live outside checkouts.

```sh
inference-study prepare --development-source /data/verified-5-2 --output /runs/data
inference-study plan --config src/inference_baselines/configs/mlplob-limited.json --data /runs/data
inference-study preflight --config src/inference_baselines/configs/mlplob-limited.json --data /runs/data --output /runs/preflight --device cuda --seconds 60 --python /env/cuda/bin/python
```

Preflight runs two scratch optimizer steps at each candidate batch on a representative
small model. Other shapes get conservative memory/work estimates, with the basis
recorded. It does not fit another384 reference, select by validation F1, or launch a
campaign. The time bound stops starting probes after the deadline; a started probe
finishes. Estimates are engineering approximations, not measured full-epoch costs.
Use the recorded estimates to choose explicit budgets and batch settings. The
limited sweep needs one common256/512 batch. Medium preflight also assesses16/32/64/128 as candidates for declared exceptions.
Medium batch exceptions must have
successful preflight evidence. An execution settings JSON is required:

```json
{
  "batch_size": 256,
  "batch_exceptions": {},
  "run_seconds": 3600,
  "campaign_seconds": 86400,
  "artifact_store": "/runs/shared-artifacts",
  "training_device": "cuda"
}
```

These budgets are examples, not adopted campaign limits. Freeze binds settings to
preflight, data, source revision, actual source hashes, dirty state and dependency
versions. Use the same training interpreter for preflight, freeze and training.
Installed packages explicitly record unavailable checkout revision/lockfile while
retaining actual package hashes and dependency versions. Changing code/dependencies
after freeze requires new preflight/freeze. Plan
membership and CPU/GPU measurements do not change training identity. A later plan
can reuse the exact completed job in the shared store; changed bindings cannot
silently reuse weights. Follow-up manifests explicitly list selected models with
exactly two additional seeds and selection provenance; the384 reference is excluded.
The broader matrix remains unspecified. `compose()` rejects overlapping plans.

```sh
inference-study freeze --config src/inference_baselines/configs/mlplob-limited.json --data /runs/data --preflight /runs/preflight --execution /runs/execution.json --output /runs/limited --python /env/cuda/bin/python
# Run only when a campaign has been authorized:
inference-study train --study /runs/limited --device cuda --python /env/cuda/bin/python
inference-study train --study /runs/limited --device cuda --resume --python /env/cuda/bin/python
```

Adam uses lr0.0003, uniform cross-entropy, raw weights, deterministic FP32 and
TF32 off. The paper's printed0.003 discrepancy remains in provenance. Training
caps at20 epochs and stops after five non-improvements once eight epochs finish.
Highest equal-day mean day6/day7 macro-F1 selects weights; exact ties keep the
earlier epoch and increment patience. Reports also retain cross-entropy, pooled
accuracy, per-class and per-day metrics. GPU data transfer is batched and recorded.

Complete epoch bundles contain current/best tensors, optimizer, RNG/shuffle state,
selection/patience, steps, repairs, metrics and elapsed time. A flushed/fsynced file
is published by an atomic pointer. An interrupted epoch reruns from its preceding
published epoch. Recoverable files without a published pointer are ignored. Budgets
include prior attempts; a hard-killed process is conservatively charged from its
last ledger update through restart, including downtime. Budget exhaustion discards
an incomplete fitting epoch; validation/publication already started completes.
A zero-epoch budget stop has no selected checkpoint. Completed jobs are reused,
never overwritten; early-stop and budget-stop are separate terminal states.

Native CPU integration needs the official ONNX Runtime headers matching the installed
library. Set `ORT_INCLUDE_DIR` or use `--ort-include`. The native adapter compiles a
generic static-shape C++ service with preallocated IO, explicitly loads the installed
ORT library and validates the actual mapped library, version and model shape. It
records headers, compiler, library, binary and export hashes. No tiny-model kernel or
fixed384 export is reused. Missing headers/compiler/runtime are explicit unsupported
results. CPU selection records a bounded20-call native ORT/eager comparison under
the same completed-call boundary. Eager remains the numerical reference. CUDA uses
explicit graph replay based on established launch-overhead evidence, with stable
pinned host and device buffers. Capture failures are unsupported and retained.

```sh
inference-study evaluate --study /runs/limited --device cpu --python /env/cpu/bin/python --ort-include /opt/ort/include
inference-study evaluate --study /runs/limited --device cuda --python /env/cuda/bin/python
inference-study profile --study /runs/limited --model MODEL_ID --device cpu --python /env/cpu/bin/python
inference-study benchmark --study /runs/limited --device cpu --cpu 0 --python /env/cpu/bin/python --ort-include /opt/ort/include
inference-study benchmark --study /runs/limited --device cuda --cpu 0 --python /env/cuda/bin/python
inference-study report --study /runs/limited --output /runs/report --plots
```

Use a CPU from the process's allowed affinity set. Evaluation, profiling and exports
finish before benchmarking. OS-managed host locks exclude runner-owned fitting,
preflight, export and profiling during measurement, across plans/stores. They do not
control unrelated load or remote hosts. Per-job mutation locks release on exit.
`--model` selects one declared model without changing its identity.

CPU evaluation publishes one hash-bound canonical CPU oracle (full validation at
batch256 and the scheduled timed inputs at batch1, one thread). Run it before
CUDA evaluation. Both devices consume those same saved logits, even when their
worker interpreters use different PyTorch versions. Full validation outputs are
checked against that CPU FP32 eager oracle.
The immutable gate is `abs(error) <= 3e-5 + 3e-4*abs(reference)`. All scheduled timed
outputs are separately compared against CPU eager batch-one outputs outside timing.
Nonfinite counts, failing IDs and class disagreements are retained. A matching
class or F1 does not erase a failed logit gate. Benchmarking still records diagnostic
failed-gate points, excluded from the verified frontier.

The primary timer starts with an available contiguous full feature window, copies
it into preallocated runtime input, executes BiN and the complete model, and ends
with CPU-visible argmax. CUDA includes pinned staging, H2D, D2H and synchronization.
Window assembly, disk reads, export, setup, reference inference and audit retention
are outside timing. Prepared-input timing excludes runtime input copying; GPU return
transfer and synchronization remain included. This is warm service latency, not
exchange-to-decision time. Native timing is inside the C++ service boundary.

Each production path uses200 warmups and3×10000 calls per boundary, with deterministic
shared validation IDs and recorded interleaving. Raw durations, IDs, logits, reference
logits and decisions are retained for every call. Report p50/p99 per repeat, median
repeat p50 and worst-repeat p99. Stage hooks separately diagnose BiN, input projection,
each feature/time pair and head, including measured hook overhead. Instrumented
stage spans are never summed into an unprofiled total.

`report` only reads artifacts and validates hashes. JSON/CSV include explicit CPU/GPU
statuses, seed/features, numerical status and hardware/runtime/boundary groups.
Plots retain failed-gate points as diagnostic; verified frontiers never include
smoke points. Models exceeding100µs remain results. Seeds are separate; one seed
provides no confidence interval.

A reproducible CPU smoke uses only synthetic data, without market-data training:

```sh
inference-study synthetic --output /tmp/mlplob-smoke-data
inference-study preflight --config src/inference_baselines/configs/mlplob-smoke.json --data /tmp/mlplob-smoke-data --output /tmp/mlplob-smoke-preflight --device cpu --seconds 5
```

Use the same freeze/train/evaluate/benchmark/report flow with the smoke manifest,
`batch_size:8`, `training_device:"cpu"`, and separate external output paths. The smoke
has a short timing schedule and always stays outside the study frontier. For CUDA
inference smoke, evaluate and benchmark the same CPU-selected checkpoint using the
CUDA interpreter. If CUDA is unavailable its integration remains unverified.

Focused checks:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_study.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_study_integration.py -v
uv build --out-dir /tmp/mlplob-dist
```

The native integration test requires `ORT_INCLUDE_DIR`; the CUDA integration test
runs only when actual CUDA is accessible. Unit tests exercise invalid plans/data,
epoch-boundary determinism, interrupted publication, stale bindings and real process
locks. No full campaign is required for verification.
