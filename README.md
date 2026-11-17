# Inference optimization

Reproducible FI-2010 MLPLOB training and same-checkpoint CPU runtime comparisons.
This is a systems baseline, not a trading strategy or a paper reproduction.

## Protocol

`fi2010-source-compatible-caveated-v1` uses four pinned no-auction z-score ZIPs.
Development columns `[0,203800)` train; `[203800,254750)` validate. The three
test files stay separate. Inputs contain 384 observations and source feature
rows 1–144 in order. Row 145 supplies labels, mapped 1/2/3 → up/stable/down
(0/1/2). Each known partition/file loses its first 383 history edges and last
ten target edges.

| Partition | Raw columns | Eligible windows |
|---|---:|---:|
| Training | 203,800 | 203,407 |
| Validation | 50,950 | 50,557 |
| Test | 139,587 | 138,408 |

Internal stock/session boundaries, z-score fit populations and engineered-feature
implementation causality remain unverified. This is a caveated source-derived
variant, not a leakage-free evaluation or a six/one/three-session split. Upstream
also concatenates test files before windowing; this variant resets known file
boundaries. A verified session protocol requires richer pinned evidence and a
separately named manifest. No additional global normalization is fitted.

The original dataset paper describes representation-event horizons 1/2/3/5/10;
TLOB uses 10/20/30/50/100 for these label rows. We preserve the row-145 task and
claim no physical-time horizon. Ten-column trimming is conservative at known
ends; it cannot establish unknown internal target boundaries.

## Environment and artifacts

Run from this checkout with CPython 3.14 and `uv`. The lockfile pins transitive
dependencies; the explicit PyTorch index supplies the CPU wheel. No command
downloads datasets or source revisions.

```bash
export BASELINE_ARTIFACT_ROOT=/home/ashen/.local/share/dev/inference-optimization/experiments/baseline-models
export UV_PROJECT_ENVIRONMENT="$BASELINE_ARTIFACT_ROOT/venv"
uv sync --locked --extra cpu
```

Set the compiler cache location before starting a numerical CLI process:

```bash
export TORCHINDUCTOR_CACHE_DIR="$BASELINE_ARTIFACT_ROOT/inductor-cache"
```

PyTorch can initialize its default cache during imports, before the runtime
adapter selects a directory. The measured run reused `/tmp/torchinductor_ashen`;
its snapshot and exact cache-state notes are preserved with the experiment.
First-call numbers are observations with existing caches, not clean compilation
benchmarks. Global compiler and OS caches were not cleared.

Every output directory must be new. Preserve datasets, environments, checkpoints,
exports, compiler caches and generated reports outside checkouts. Attribution:
FI-2010, Ntakaris et al./BigDataFinance, CC-BY 4.0. Model adaptation and upstream
hashes are documented in [LICENSE-TLOB](LICENSE-TLOB) and
[source.json](src/inference_baselines/configs/source.json).

Compiled PyTorch needs a C++ compiler and matching Python development headers.
On the evidence host the preserved headers work without changing system packages:

```bash
export CPLUS_INCLUDE_PATH=/home/ashen/.local/share/dev/inference-optimization/research/mlp-focused-2026-11-12/environment/python-headers
```

## CUDA training with CPU inference

GPU training uses a separate environment; CPU inference retains the environment
above. Both environments are in `uv.lock`. The CUDA extra pins PyTorch 2.9.1+cu130;
the CPU extra pins 2.15.0+cpu. The same tensor-only selected checkpoint is loaded
into the CPU reference and all CPU runtime adapters. This is not a training
framework comparison. FP32 training disables TF32, enables deterministic
algorithms and records the GPU, CUDA runtime, cuBLAS workspace setting and memory.

```bash
UV_PROJECT_ENVIRONMENT="$BASELINE_ARTIFACT_ROOT/venv-cuda" uv sync --locked --extra cuda
```

Pass `--device cuda` to smoke and train using the CUDA environment's executable.
Save selected weights as CPU tensors. Preserve CUDA optimizer/RNG state separately
for resume, and record CUDA-to-CPU validation output differences plus an exact
CPU save/reload check. The original inference round stays on CPU; the separate
CUDA inference extension below reuses its selected checkpoint and saved outputs.
GPU devices may require host execution outside the tool filesystem sandbox.

The first experiment is scoped to seed 42; seeds 43/44 are explicitly deferred.
For a fresh experiment, copy
`src/inference_baselines/configs/schedule-seed42.json` to the artifact root as
`schedule.json` before starting. Do not overwrite a running schedule. This gives
the report its seed coverage; the original three-seed defaults remain available
in the training configuration. A single seed supports same-weight runtime
comparisons but does not estimate training variability.

## Audit and train

`audit` and `prepare` are aliases. Both verify hashes/ZIP CRCs, dimensions, all
five label domains, FP64/FP32 finiteness and constant features, then save hashed
arrays and example indices without materializing overlapping windows. Prediction
IDs encode source hash/member/column, partition, known file segment and unverified
internal-segment status.

```bash
"$BASELINE_ARTIFACT_ROOT/venv/bin/inference-baselines" prepare \
  --source /home/ashen/.local/share/dev/inference-optimization/research/free-extension-2026-11-17 \
  --output "$BASELINE_ARTIFACT_ROOT/data-v1"
"$BASELINE_ARTIFACT_ROOT/venv/bin/inference-baselines" control \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --output "$BASELINE_ARTIFACT_ROOT/control-v1"
"$BASELINE_ARTIFACT_ROOT/venv-cuda/bin/inference-baselines" smoke --device cuda \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --output "$BASELINE_ARTIFACT_ROOT/smoke-42"
```

The recipe in [fi2010.json](src/inference_baselines/configs/fi2010.json) fixes
width 144, three block pairs, FP32, Adam lr 0.0003, batch 32, available seeds
42/43/44 and ten epochs. This experiment runs seed 42 only. Fixed EMA decay
0.999 updates after each optimizer step. Select the
minimum sample-weighted EMA validation loss, earliest exact tie. Test outcomes
never select epochs, preprocessing, tolerances or hyperparameters.

Documented source deviations: fixed-decay post-step EMA rather than the engine's
pre-step/library behavior, constant learning rate instead of validation-based
halving, and device-independent in-place negative BiN coefficient reset to 0.01
retaining optimizer identity. BiN stays inside every runtime. Temporal standard
deviations below 1e-4 become one; the feature axis remains unguarded. Constant-axis
nonfinites remain visible. The paper's learning-rate table conflicts with its
search space/source; this recipe explicitly uses the source default.

Smoke runs at most 100 training steps and ten validation batches, saves a separate
diagnostic checkpoint and estimates the full schedule. Full training requires a
matching completed smoke. To exceed the configured 24-hour feasibility guard,
choose the compute plan and explicitly add `--allow-long-run`; epochs/seeds/data
are never silently reduced.

```bash
"$BASELINE_ARTIFACT_ROOT/venv-cuda/bin/inference-baselines" train --device cuda \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" \
  --seed 42 --output "$BASELINE_ARTIFACT_ROOT/seed-42"
```

Resume a failed/interrupted run using the same arguments plus `--resume`.
Resume restores raw/EMA tensors, Adam state, PyTorch RNG and the independent
shuffle-generator state from the last completed epoch; an interrupted epoch is
replayed. Completed seeds cannot be retrained in place. Selected weights/hash,
optimizer/EMA state, manifest, history, repair counts, reload evidence and status
stay together. Runtime comparisons always reuse those selected weights.

## Evaluate and measure

`diagnose` accepts smoke checkpoints for validation/synthetic tooling checks only;
it produces no financial quality or baseline latency. `evaluate` and `benchmark`
reject smoke checkpoints. Run every selected seed across runtimes:

```bash
for seed in 42; do
  for runtime in eager compiled ort; do
    "$BASELINE_ARTIFACT_ROOT/venv/bin/inference-baselines" evaluate \
      --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/seed-$seed" \
      --runtime "$runtime" --output "$BASELINE_ARTIFACT_ROOT/evaluation-$seed-$runtime"
  done
done
```

Validation and random/near-constant/constant-axis/zero diagnostics freeze the
strict logit gate (`atol=3e-5`, `rtol=3e-4`) before test access. Failures never
widen it. Full-test output saves IDs, targets, reference/runtime logits,
single-softmax probabilities and classes. Reports distinguish numerical errors,
tolerance passes, nonfinite masks, class disagreement and predictive metrics.
Metrics include macro-F1, accuracy, per-class precision/recall/F1, confusion,
support and log loss. Majority uses eligible training targets and lowest-class
tie-breaking, with no assumed probability convention.

Bulk accuracy uses four intra-op threads for every runtime, including ORT.
Timing uses one thread and reruns validation/synthetic checks for that actual
execution setting using the already frozen tolerances. Quality and timing thread
settings are recorded separately; bulk timings are not reported as batch-one latency.

Finish training before timing. Select an available logical CPU; one affinity
CPU and one intra/inter-op thread are enforced. Seed 42 is preselected without
test-quality selection. Matched runtime evaluations must already exist.

```bash
"$BASELINE_ARTIFACT_ROOT/venv/bin/inference-baselines" benchmark \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/seed-42" \
  --runtime all --cpu 0 --output "$BASELINE_ARTIFACT_ROOT/measurements-42"
"$BASELINE_ARTIFACT_ROOT/venv/bin/inference-baselines" report \
  --run "$BASELINE_ARTIFACT_ROOT" --output "$BASELINE_ARTIFACT_ROOT/report-v1"
```

The saved timing plan fixes 100 warmups, three repeats of 10,000 calls per
path/boundary, identical input sequences and seeded interleaved assignment order.
Prepared-input timing starts with a resident tensor/array; local preparation
includes window copying, contiguous FP32 conversion and invocation. Both end
after consuming output argmax. CPU calls complete synchronously. Loading/export/
compilation/first-call costs are separate. Disk/decompression, network, queueing
and raw generation of the 104 engineered features are excluded.

Retain every observation, including outliers. Outputs include repeat p50/p95/p99/
maximum, inverse-mean service capacity, wall-clock serial capacity, process peak
RSS (all resident runtimes/compiler setup), checkpoint size, versions, providers
and math settings. p99.9 is omitted given only ten observations beyond it per
default repeat. Laptop thermals, frequency and background contention are
uncontrolled. No failing numerical path is advertised as an equivalent speedup.

Timed calls also retain their three output logits after stopping the timer. The
`--runtime all` comparison checks them against eager batch-one outputs from the
same one-thread timing context, without running extra model invocations. This
complements the full-test bulk comparison; audit-output copying is outside the
call boundary and included in reported wall capacity.

Reports preserve missing/failed seeds and paths and show aggregate coverage.
Overlapping windows are dependent observations. No profitability, broad market
generalization, arrival-latency, deployment SLA or novel-optimization claim follows.

## CUDA inference and snapshot comparators

The CUDA follow-up does not retrain MLPLOB. `gpu-evaluate` checks validation and
synthetics before scoring the complete eligible test set with eager CUDA and
explicit CUDA Graph replay. It compares both with saved CPU predictions and
with CUDA eager, retaining differences between CPU and CUDA package versions.
The existing PyTorch 2.9/Python 3.14 environment rejects `torch.compile`; that
path is recorded as unsupported. Its ONNX Runtime package has only the CPU
provider, so this extension makes no exported-GPU claim.

```bash
"$BASELINE_ARTIFACT_ROOT/venv-cuda/bin/inference-baselines" gpu-evaluate \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/seed-42" \
  --output "$BASELINE_ARTIFACT_ROOT/gpu-evaluation-42-v2"
"$BASELINE_ARTIFACT_ROOT/venv-cuda/bin/inference-baselines" gpu-benchmark \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/seed-42" \
  --evaluation "$BASELINE_ARTIFACT_ROOT/gpu-evaluation-42-v2" \
  --cpu 0 --output "$BASELINE_ARTIFACT_ROOT/gpu-measurements-42"
```

CUDA prepared timing starts with resident GPU input and ends after completed
GPU logits/argmax, with explicit stream synchronization and a host wall timer.
Local preparation includes CPU history copying, pinned staging, H2D, inference,
D2H three logits and CPU argmax. Audit logits are retained after each timer.
CUDA Graph addresses remain stable; captures, diagnostics and warmup precede
timing. Both boundaries use the same three repeats of 10,000 inputs as CPU.
GPU-only latency is not host-to-decision latency.

The exploratory comparator uses each target's current 144 supplied features
at the identical eligible IDs, whereas MLPLOB consumes 384 rows. It preserves
the caveated split and supplied normalization. Ridge fits three +/-1
least-squares scores with an intercept; this avoids treating up/stable/down
as ordinal regression targets. Four ridge penalties select by validation
macro-F1; these scores have no declared probability calibration or log loss.
XGBoost CPU hist trains depth 2/4/6, at most 300 rounds, with validation-log-loss
early stopping (20 rounds). Fixed tree-count prefixes and the selected endpoint
form an initial curve. All choices and model hashes are sealed before test
access. This is a bounded exploratory curve, not an optimized frontier across
feature sets, MLPLOB architectures or compiled tree runtimes.

```bash
UV_PROJECT_ENVIRONMENT="$BASELINE_ARTIFACT_ROOT/venv-shallow" \
  uv sync --locked --extra cpu --extra shallow
"$BASELINE_ARTIFACT_ROOT/venv-shallow/bin/inference-baselines" shallow-fit \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --output "$BASELINE_ARTIFACT_ROOT/shallow-42"
"$BASELINE_ARTIFACT_ROOT/venv-shallow/bin/inference-baselines" shallow-benchmark \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/shallow-42" \
  --cpu 0 --output "$BASELINE_ARTIFACT_ROOT/shallow-measurements-42"
```

Comparator timing uses one pinned CPU and one numerical thread. Prepared input
is contiguous FP32 [1,144]; local preparation copies that vector. NumPy affine
scores or XGBoost `inplace_predict` plus CPU argmax finish the timed boundary.
Python/native API overhead is included. Model generation, training, tests and
builds must finish before any scientific timing; run GPU and CPU timing serially.
Raw timing arrays, actual scores, model hashes and full-test predictions remain
outside the checkout. Feature generation, queueing and network are excluded.

## CatBoost and ridge across temporal feature recipes

`feature-fit` compares five fixed representations: current144, mean5 (144),
current+mean5 (288), five concatenated vectors (720), and a project-defined
1440-feature bank spanning384 rows. The three five-vector recipes retain their
Nousi §4.1/Figure1 attribution in `configs/feature_comparison.json`; the longer
bank is an extension, not a published optimum. Labels, known partition bounds
and eligible sample IDs stay identical to the reference MLPLOB protocol.

CatBoost retains depth4/256 symmetric trees and seed42. The existing288-feature
checkpoint is reused byte-for-byte. Ridge is a three-score least-squares
classifier with an intercept, train-only standard deviations and validation-only
selection of alpha/class weighting from the frozen config. Its FP64 affine
coefficients include scaling, so inference does not separately normalize data.
This replaces the earlier weaker unscaled/unweighted ridge recipe; feature
effects should be compared within this round.

```sh
"$BASELINE_ARTIFACT_ROOT/venv-tree288/bin/inference-baselines" feature-fit \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" \
  --reuse "$BASELINE_ARTIFACT_ROOT/trees-288-42" \
  --output "$BASELINE_ARTIFACT_ROOT/cat-ridge-features-42"
"$BASELINE_ARTIFACT_ROOT/venv-tree288/bin/inference-baselines" feature-benchmark \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" \
  --run "$BASELINE_ARTIFACT_ROOT/cat-ridge-features-42" \
  --catboost-library "$BASELINE_ARTIFACT_ROOT/native-libraries/libcatboostmodel-linux-x86_64-1.2.10.so" \
  --cpu 0 --output "$BASELINE_ARTIFACT_ROOT/cat-ridge-features-measurements-42"
```

The C++ feature implementation serves both offline preparation and online state
updates. Known partitions initialize their own history. Five-vector means use
ordered FP64 sums; the longer bank uses FP64 rolling sums and FP32 means.
Native validation checks precede native full-test checks; both compare identical
saved weights with Python. Timing uses one CPU, 10,000 consecutive snapshots,
three repeats and seeded interleaving of prepared/streaming boundaries, retaining
every observation. Streaming includes state update, feature assembly, prediction
and argmax; raw feature generation and setup are outside the boundary. Scientific
timing must run after all controlled training, builds and tests have finished.

## Fixed 288-feature native tree workflow

`tree-fit` compares XGBoost 3.4.1, LightGBM 4.7.0 and CatBoost 1.2.10 with
one fixed seed-42/depth-4 recipe each. Its 288 features concatenate the current
144-vector with the mean of that vector and the four preceding vectors, from
[Nousi et al., §4.1/Figure 1](https://arxiv.org/pdf/1809.07861). The source,
version, formula, arithmetic and adaptation are retained in
[tree288.json](src/inference_baselines/configs/tree288.json), each run's plan
and reports. This is a representative workflow comparison, not a dataset
optimization exercise or reproduction of the paper's labels/normalization.
No feature selection or additional history variants run in this increment.

Each ensemble trains once for 256 rounds at learning rate .05. Saved prefixes
16/32/64/128/256 expose configuration changes without retraining. Boosting
rounds do not imply equal multiclass tree counts or execution cost. Labels,
eligible IDs and supplied normalization remain those of the original baseline.
All configurations are fixed before test scoring; no test-driven tuning.

```bash
UV_PROJECT_ENVIRONMENT="$BASELINE_ARTIFACT_ROOT/venv-tree288" \
  uv sync --locked --extra cpu --extra trees
"$BASELINE_ARTIFACT_ROOT/venv-tree288/bin/inference-baselines" tree-fit \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --output "$BASELINE_ARTIFACT_ROOT/trees-288-42"
"$BASELINE_ARTIFACT_ROOT/venv-tree288/bin/inference-baselines" tree-benchmark \
  --data "$BASELINE_ARTIFACT_ROOT/data-v1" --run "$BASELINE_ARTIFACT_ROOT/trees-288-42" \
  --catboost-library "$BASELINE_ARTIFACT_ROOT/native-libraries/libcatboostmodel-linux-x86_64-1.2.10.so" \
  --cpu 0 --output "$BASELINE_ARTIFACT_ROOT/trees-288-measurements-42"
```

The CatBoost native evaluator is the separately downloaded
[official 1.2.10 release](https://github.com/catboost/catboost/releases/tag/v1.2.10),
whose URL/hash are retained with the library. No command silently downloads it.
The benchmark compiles the packaged C++ runner outside the checkout, validates
the feature ring, checks native probabilities on validation, then compares all
eligible test predictions with the same saved model's Python predictions.
The frozen probability tolerance is atol=1e-6/rtol=1e-5, checked without a
second softmax. Full predictions, numerical errors and per-class scores remain.

XGBoost reuses its array-interface/config strings and input buffer; its C API
still parses JSON internally. LightGBM uses its preinitialized fast single-row
C API. CatBoost uses the native evaluation library. The C++ measurement loop
has no Python/ctypes calls or runner heap allocation per prediction; library
internal allocation is not claimed absent. All paths compute three probabilities
and consume argmax inside timing. Engine setup is outside measured calls.

Prepared timing begins with contiguous FP32[288]. Local preparation begins with
the newest FP32[144] supplied vector and updates a preallocated five-row ring,
computes the ordered FP64 mean rounded to FP32, predicts and consumes the
decision. Exact feature parity is checked over the entire stream before timing.
This fixed five-row mean avoids expensive general window/array construction.
Prefill and resets are outside warmed calls. Raw feature generation, network,
queueing and disk are excluded and remain material limitations.

Three seeded interleaved repeats of 10,000 consecutive target IDs run for all
15 points and both boundaries (900,000 retained samples and probabilities).
One pinned host core/thread, existing laptop thermal/background caveats, and
no concurrent controlled builds/tests/training. This input sequence differs
from the original uniformly spaced CPU timing; compare paired native paths
within this round. Peak native process memory includes all resident libraries
and models. Artifacts stay external and earlier runs remain unchanged.

## Verification

```bash
"$BASELINE_ARTIFACT_ROOT/venv/bin/python" -m unittest discover -s tests -v
```

Tests cover boundary accounting, label/window alignment, tamper detection,
metrics/numerics, shared-parameter EMA, BiN repair, optimizer/RNG restoration and
a tiny synthetic train/select/reload/evaluate flow. Exact upstream-forward
comparison runs when the pinned local snapshot is installed; otherwise it
explicitly skips. Synthetic tests verify tooling, not FI-2010 quality. The initial
repository had no configured CI.
