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
CPU save/reload check. GPU timing is excluded; inference comparisons stay on CPU.
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

Reports preserve missing/failed seeds and paths and show aggregate coverage.
Overlapping windows are dependent observations. No profitability, broad market
generalization, arrival-latency, deployment SLA or novel-optimization claim follows.

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
