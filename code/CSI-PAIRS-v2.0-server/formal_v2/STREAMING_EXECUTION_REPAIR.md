# Streaming execution repair

This patch changes evaluation execution, not the four-arm method. It must not
be hot-applied to an active c591701 process. Original NPZ, qualification,
checkpoint, configuration, localization and G5 records retain their identities.

## Protocol

- Production workers explicitly forward their logical CUDA device to all
  compatibility, shortcut and response probes. CPU remains an explicit reference
  option at the probe API; production device resolution rejects CPU and duplicate
  or unavailable logical devices.
- The original FP32 networks, seeds, optimizer, learning rate, AdamW decay,
  selection rule and 2,000 optimizer updates remain unchanged. No AMP/TF32.
- Prefer full-batch CUDA training, retaining its input tensors across updates.
  A conservative free-memory estimate admits full-batch execution; otherwise use
  bounded gradient accumulation. BCE/MSE sums decompose across independent rows,
  with the original global element-count denominator. GELU and Linear have no
  batch-dependent state or dropout. AdamW, including its decay, steps once per
  full-dataset update, not once per block. Resident inputs are cached when the
  memory budget permits; no OOM-and-retry policy is introduced.
- Initialization alone holds the RNG lock. It uses the original CPU generator
  initialization and seed mapping; fitting draws no further randomness. GPU
  fitting does not train or update the frozen encoder.
- Probe-corpus admission remains bounded, initially one complete build. Two
  builds require explicit selection and sufficient host/cgroup memory; the
  256 GiB per-build reservation is an admission estimate, not a measured guarantee.
  Production should remain at one until representative peak measurements allow
  and validate two. The file writer remains the coordinator only.
- AUROC remains unweighted binary positive-label-1 AUROC with average ties.
  Sort-and-group exact ranks replace repeated full scans. The existing rejection
  of empty, single-class, invalid labels and nonfinite inputs is retained.

## Progress and recovery

The official 1,489 work-unit definition is unchanged. `probe_progress.json`
reports each device's probe/family/phase, submitted optimizer steps, synchronized
training completions, elapsed times and memory-slot wait/hold measurements.
Heartbeat timestamps are distinct from actual worker callback timestamps. CUDA
is synchronized at training/measurement boundaries, not every step.

`formal_cli evaluation-status --output RUN` includes detailed probe telemetry.
Workers update memory only; the coordinator heartbeat atomically publishes it.
Only an authenticated atomic probe bundle increments the official unit count.
There is no intra-probe optimizer resume: interruption before its bundle commit
requires refitting that probe bundle. Completed authenticated bundles retain the
existing resume and quarantine checks.

## Version separation

`formal_cli run-evaluation-repair` uses a new output root, by default stops after
one original work unit, and supports a preceding `--representative-probe` check.
It authenticates the original run in its original clean c591701 source and exact
runtime, verifies input/checkpoint hashes and LIVE roles, then binds the new
evaluator separately in `evaluation_origin.json`. The source comparison rejects
changes to model/configuration/dependencies, unrelated metric/localization
functions and probe architectures. This is not legacy factorial migration and
does not rewrite any upstream identity. Repaired evaluation does not reapprove
training or make SOTA claims. Its receipt and execution identity must be retained
for downstream evidence integration; do not pass it off as unchanged c591701.

## Result dependencies

| Result | Generator | This probe/AUROC path | Disposition |
| --- | --- | --- | --- |
| 12 main weights and teacher | formal_factorial / formal_teacher | No | Preserve hashes; no retraining |
| Eight-city/budget cells (two cities, four budgets) | formal_factorial._run_localization | No | Preserve all four-arm meters and G5 FAIL |
| Compatibility/response/shortcut probes | formal_evaluation_streaming._fit_probe_bundle | Yes | Refit every affected seed/arm uniformly |
| Streaming CGS/response/shortcut statistics | formal_evaluation / formal_metrics | Yes | Uniform reevaluation; exact AUROC alone preserves fixed-score statistics |
| Risk/path/controls/claims | downstream evaluation consumers | Indirect | Require new evaluation bindings and ordinary gate assessment before reuse |

Full did not beat Response-only in the eight recorded meter cells. This does not
mean all eight differences were statistically significant. Execution acceleration
does not change that observation or establish SOTA.

## Validation

The operator evidence root for this execution is
`/root/xunlian/Futaoran/formal_external_inputs/supervision/c591701-streaming-gpu-fix-20260908`.
It contains scene preservation and stop records, command/exit logs, fixed-input
AUROC comparisons, shared-initialization numerical comparisons, and an actual
CUDA profiler trace. Synthetic smoke artifacts are explicitly NON_CLAIM and never
substitute for the formal NPZ. Representative and bounded complete-unit results
must be checked separately before resuming the full evaluation.
