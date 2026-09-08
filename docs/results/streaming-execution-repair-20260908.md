# Streaming execution repair: recorded acceptance

This report separates engineering acceptance from scientific claims. The
upstream c591701 four-arm localization results and G5 FAIL are unchanged.
Full did not beat Response-only in any of the eight recorded city/budget cells;
this statement does not assert statistically significant degradation in all cells.
There is no SOTA evidence.

## Identity

| Item | Value |
| --- | --- |
| Upstream training commit | c5917018610b262f5263ccaeba2f087b2eb96a96 |
| Evaluation code commit | 1738caea131b9ed0d28994c89d17c3730586d323 |
| Evaluation source SHA-256 | 98572664677378e50baead78828671d0cc92cc6e580809cb57b8e4c3ad8785eb |
| Original formal NPZ SHA-256 | 060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac |
| Runtime | Existing core venv, Python 3.12.13, torch 2.5.1+cu121 |
| Precision | FP32, no AMP, no TF32 |
| Scientific status | NON_CLAIM; sota_ready=false |

The code worktree stays on the recorded evaluation commit. This separate
evidence branch does not change the identity of an active evaluation process.
Original source, configuration, LIVE roles and all checkpoint hashes are
authenticated separately; NPZ provenance is not rewritten to the repair commit.

## Completed A/B Acceptance

The immutable [A/B manifest](../../artifacts/streaming_gpu_repair_20260908/AB-verified/manifest.json)
binds commands, exit codes, test inputs, shared initial states, numerical results,
CUDA trace, source diffs and the old-process stop receipt by file hashes.

| Check | Observed result | Scope |
| --- | --- | --- |
| A: exact binary AUROC | PASS: original, new, scipy 1.17.1 average ranks and independent small pair enumeration agree | Fixed scores; includes ties, strided inputs, supported tested dtypes and invalid-input rejection |
| A: 64,000 scores, three repetitions | Old 3.78730--3.90881 s; new 0.010995--0.013022 s | Function only, array generation excluded, internal conversions included; not end-to-end acceleration |
| B: response one-step CUDA trace | PASS: 124 CUDA kernel events; gradients and changed parameters verified | Shared actual CPU-created state, not independent seed calls |
| B: response complete 2,000 steps | PASS: max output error 3.815e-6 for GPU full vs CPU full | Synthetic smoke, not formal evidence |
| B: binary complete 2,000 steps | PASS: max probability error 4.769e-6 across linear/MLP | Same rankings, AUROC, threshold decisions and selected family in this smoke |
| Focused regression | 134 passed, exit 0 | Listed commands and warnings retained in the manifest |

Tolerances were written before comparisons: one-step rtol=1e-4, atol=5e-6;
complete-output rtol=2e-3, atol=2e-4. They were not widened after observing errors.
Response gradients had maximum absolute error 8.848e-9; maximum one-step updated
parameter error was 1.164e-10. Binary one-step parameter maximum was 1.193e-7.
GPU optimizer momentum/variance tensors and participating tensors were checked.

The larger legacy streaming test module is not universally green: the original
source and repair share ten observed failures (nine outdated mocked LIVE input
contracts and one exact/scene floating-point assertion). These were not bypassed
in production. The 134-test focused suite does not claim to be the entire repo.

## Formal Acceptance Still Separate

At this A/B snapshot, C is running in
`CSI_STREAMING_GPU_FIX_20260908/runs/representative-C-1738cae` and has not completed.
D (one complete original probe bundle) and E (formal dual-device acceptance) are
not passed. F (remaining full evaluation) has not started. No single probe,
synthetic test or process heartbeat counts as a completed original unit.

The official definition remains completed_units/1489. A complete probe-state
unit includes all 17 fitted candidates/models and a validated atomic bundle.
Concurrent synthetic tests are not formal E acceptance. The default memory
admission remains one complete probe build; two builds are not approved by an
idle GPU or a guessed memory estimate.

## Saved Scene and Old Task

PID 6306, its exact command, working directory, process start ticks, selected
non-secret environment, original hashes, outputs and limited procfs samples were
saved before an identity-checked SIGTERM. The old supervisor recorded exit -15.
No debugger was attached and no claim is made about unrecorded historical
function-level time. Original files were not overwritten.

The original 12 checkpoints and factorial results remain reusable under their
original identities. The incomplete old probe had no optimizer checkpoint or
committed probe bundle, so its former training step cannot be resumed losslessly.
Completed future bundles require identity and payload validation before reuse.

## Result Dependencies

| Result | Generator | Depends on repaired probe/AUROC | Disposition |
| --- | --- | --- | --- |
| Main weights / teacher | formal_factorial / formal_teacher | No | Preserve; no main-model retraining |
| Four-arm meter table | formal_factorial._run_localization | No | Preserve original eight cells and G5 FAIL |
| Compatibility / shortcut / response probes | formal_evaluation_streaming._fit_probe_bundle | Yes | Refit all affected arms and seeds uniformly |
| Streaming diagnostic statistics | formal_evaluation / formal_metrics | Yes | Uniform evaluation, retaining new source binding |
| Downstream risk / controls / claims | Original downstream consumers | Indirect | Need complete new evaluation and normal dependency checks |
