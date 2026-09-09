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
| C evaluation code commit | 1738caea131b9ed0d28994c89d17c3730586d323 |
| C evaluation source SHA-256 | 98572664677378e50baead78828671d0cc92cc6e580809cb57b8e4c3ad8785eb |
| D execution code commit | 2ce24729814d1d7357fafb280ab10192ee3ace6b |
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
The [original/repair comparison](../../artifacts/streaming_gpu_repair_20260908/streaming-regression-baseline/comparison.json)
records identical named failures, with 10 failed / 41 passed / 1 skipped on each
source version. Both exit codes are 1, not converted to a PASS.

## Formal Acceptance Still Separate

Telemetry-only follow-up `7613ac7` retains completed phase durations, brief
memory-admission events and lock wait/hold observations across heartbeat updates
and later checkpoints, and labels snapshots with the real run ID and original
total-unit count. Its [14 focused tests](../../artifacts/streaming_gpu_repair_20260908/progress-retention/manifest.json)
passed. This follow-up was isolated from C, whose source remained `1738cae` until
completion. After C exited and its artifacts passed independent checks, it was
cherry-picked as `27075c8`; full-source corpus capture and E acceptance tools were
then cherry-picked as `2ce2472`. C provenance was not rewritten.

C passed in `CSI_STREAMING_GPU_FIX_20260908/runs/representative-C-1738cae`.
Its [artifact manifest](../../artifacts/streaming_gpu_repair_20260908/C-formal-probe-artifacts/manifest.json)
contains the probe state, predictions, source receipt and independent validation.
The [command log](../../artifacts/streaming_gpu_repair_20260908/C-representative-formal/record.json)
and [independent check](../../artifacts/streaming_gpu_repair_20260908/C-artifact-validation/record.json)
both exited 0. This is representative probe acceptance, not a complete unit.

| C measurement | Actual value |
| --- | --- |
| Original train feature shape | 38,696 x 128 |
| Original selection feature shape | 42,520 x 128 |
| Full updates | 2,000 each for linear and MLP candidates; no accumulation observed |
| Train preparation phase | 2,966.423 s |
| Selection preparation phase | 3,212.926 s, including initialization before the first subsequent phase callback |
| Candidate fitting and selection wall time | 11.337 s, including cold initialization |
| Linear / MLP synchronized training phases | 2.340 s / 4.908 s |
| Representative total after origin authentication | 6,201.868 s |
| Peak process RSS | 154,314,981,376 bytes |
| Peak CUDA allocated / reserved | 1,203,760,640 / 1,233,125,376 bytes |
| Selected family | linear |
| Endpoint compatibility selection AUROC / NLL | 0.5000094405244746 / 0.6931731106398445 |

The CUDA event interval is 11,336.985 ms, not busy-kernel time. No old/new complete
unit comparison or end-to-end speed ratio is claimed. These endpoint probe
statistics are not Full localization results. Both completed training events,
file hashes, probability shape/range, finite FP32 state and unchanged backbone
were checked. Peak process RSS excludes separate authentication subprocess RSS;
the resource logs retain that sampling scope explicitly.

After the follow-ups, [139 focused regression tests](../../artifacts/streaming_gpu_repair_20260908/regression-before-D/record.json)
passed with 16 ordinary PyTorch warnings. The known legacy streaming-module
failure set above remains disclosed. A separate
[dual-device integration smoke](../../artifacts/streaming_gpu_repair_20260908/dual-execution-smoke/record.json)
recorded 56,029 / 92,081 kernels on devices 0 / 1, with 10,721.864 microseconds
of actual kernel interval overlap and identical serial/parallel smoke
probabilities. It is not formal E acceptance.

D started at 2026-09-08 18:12 Asia/Shanghai in
`CSI_STREAMING_GPU_FIX_20260908/runs/complete-D-2ce2472`, with
`--stop-after-units 1 --probe-build-limit 1 --capture-validation-corpus`.
It is started, not completed. E on authenticated full source input is not yet
passed; F is not started. No single probe, synthetic test or process heartbeat
counts as a completed original unit.

At 2026-09-08 22:32 Asia/Shanghai, D had completed 5 of 17 fitted models,
but still 0 of 1,489 original units. The map-only MLP was around full-update
step 421/2000 on cuda:0. D compatibility train/selection preparation took
3002.393 / 3403.469 seconds. Map-only linear training took 7056.739 seconds;
its much wider input triggered the conservative full-gradient-accumulation
admission path. Each optimizer update still covers all 38,696 training rows.
C's 11.337-second low-dimensional candidate fitting is not a timing estimate
for these wide map inputs, a complete bundle, or the entire evaluation.

The original D wrapper and resource sampler disappeared; their last resource
samples are at 19:12:51/52. D itself survived, with its original PID/start ticks,
working directory and stdout/stderr paths unchanged. A read-only observer was
started at 22:31:55, after the sandboxed nvidia-smi call returned status 9 and
the outside-sandbox launch was approved. This is a monitoring warning, not an
evaluation restart. Missing resource observations and the original parent's
eventual child wait status cannot be reconstructed or reported as exit 0.
The external receipt is
`formal_external_inputs/supervision/c591701-streaming-gpu-fix-20260908/checks/D-observer-resumed-20260908T1927/record.json`.
Its actual timestamp, not the directory label, identifies the resumed sampling.
The new observer binds PID 26999 and start ticks 9109609 and sends no signals.
At 22:32, the sampled GPU usage was 40% / 0%, with 7805 / 0 MiB allocated
as reported by nvidia-smi. D RSS was 241654398976 bytes and host MemAvailable
518335066112 bytes. These observations do not establish full-run resource
peaks, formal dual-GPU acceptance, or completion of the active unit.

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

## Existing SOTA Obstacles, 2026-09-09

The [summary-only audit](../../artifacts/streaming_gpu_repair_20260908/sota-obstacle-audit-20260909/report.md)
retains all eight city/budget cells and the original summary bytes. It authenticates
the factorial manifest and gate against D's upstream origin receipt, then checks
the actual localization/training CSV hashes against that original manifest.
The full NPZ is not rehashed by this small audit; its identity comes from D's
existing LIVE authentication receipt. Original and evaluation commits stay separate.

Full is strictly first among the four arms in 0/8 cells. Its mean bank-median
error exceeds Response-only by 0.376485--1.660415 meters, depending on the cell.
These are descriptive meter differences, not log-utility differences or a claim
of significance in every cell. The original assessed G4 conditions and G5 remain
FAIL; four G4 subconditions are still unassessed, not passed. The audit does not
certify a complete same-run Wi-GATr/PMNet comparison or an assembled claims package.

Full's original training gradient summaries are included as diagnostic leads.
Small alignment-gradient magnitudes and mean encoder cosine alone do not identify
the cause of the localization gap and do not establish a remedy. Method changes
and new main-model training are not part of this audit. Any separately authorized
method experiment must lock its selection protocol, preserve fair four-arm budgets,
exclude target query results from selection and disclose prior target exposure.

Commands executed from this evidence worktree using the existing core venv:

```text
python -B -m unittest discover -s artifacts/streaming_gpu_repair_20260908/tools -p test_audit_sota_obstacles.py -v
python -B artifacts/streaming_gpu_repair_20260908/tools/audit_sota_obstacles.py --origin-receipt /root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/runs/complete-D-2ce2472/evaluation_origin.json --output artifacts/streaming_gpu_repair_20260908/sota-obstacle-audit-20260909
```

Both commands exited 0. Six synthetic parser tests passed; they are not formal
experimental results. The actual audit status is `AUDIT_COMPLETE_NON_CLAIM`, with
`sota_ready=false`. Its JSON SHA-256 is
`dd2353e813afac27f0523cdd4f66a6f92d64d8f6f9b855ca48243e6c4252ab5f`.

## Authorized Source Pilot And D Validator, 2026-09-09

The user subsequently confirmed independent method research on existing hardware.
This does not change the scope of the execution-only Streaming repair or its gates.
The isolated source research worktree is
`/root/xunlian/Futaoran/CSI_METHOD_SOURCE_PILOT_20260909`, frozen commit
`e270230fa950337469f3f25340c22874767d2ac2`, branch
`experiment/source-method-pilot-20260909`. D remains untouched at `2ce2472`.

The pilot compares the existing fixed and effect-aware margin implementations:
two recipes, all four arms, 128 original training updates per arm, one shared
actual initialization and paired batch plans. It uses only source training and
source selection roles on the same real NPZ. All source cities and k=0/8/32/128
are covered; target query results do not select a candidate. It is NON_CLAIM and
adds zero formal units. The original formal config validator still rejects the
effect-aware candidate; it is permitted only by this separate, frozen research
protocol. No automatic 20,000-step formal retraining has been launched.

Evidence archive: [source-pilot-e270230-freeze](../../artifacts/streaming_gpu_repair_20260908/source-pilot-e270230-freeze/).
It retains the initial 3-fail/15-pass draft test, corrected tests, original-input
authentication, frozen-commit test outputs, and real supervisor start records.
Frozen-commit regression exited 0: 48 passed in 25.30 seconds, including actual
CPU/CUDA-1 synthetic training, paired control invariance, checkpoint restoration,
role/sibling isolation and CLI dispatch. These tests are not formal results.

The real NPZ was rehashed to
`060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`.
Original-source authentication exited 0 with both source roles verified PASS,
teacher and normalization manifest bindings checked, original factorial FAIL
preserved. The research never imports an old trained arm as its new model.

The research child PID is 218460 (start ticks 12191858); detached supervisor PID
218455. It started at 2026-09-08T18:45:53Z using the existing core venv and
logical cuda:1. CUDA properties independently mapped devices 0 and 1 to GPU UUIDs
`0e31845b-99e5-7784-4b0d-d37446c79176` and
`826ef21b-33e6-41a7-7a7b-c2e15cd2b96d`, respectively. NVML process IDs use the
host namespace and must not be confused with container PIDs. Two different
experiments using different GPUs are not formal dual-card acceptance E.

Live research output:
`/root/xunlian/Futaoran/CSI_METHOD_SOURCE_PILOT_20260909/runs/source-pilot-e270230-01`.
Shared endpoint calibration completed 128 synchronized CUDA updates at 18:52:32Z.
At 18:59:27Z the independent partial artifact audit exited 0 and verified one
complete arm (fixed-margin endpoint): 128 optimizer steps, changed model matching
its checkpoint, matching trace, and all 248 source validation rows. See
[the partial receipt](../../artifacts/streaming_gpu_repair_20260908/source-pilot-e270230-freeze/source-pilot-first-arm-audit.json).
Seven arm jobs were not yet validated by that snapshot. No Full comparison or
full-pilot completion is implied by this partial receipt.

A second [partial artifact audit](../../artifacts/streaming_gpu_repair_20260908/source-pilot-e270230-freeze/source-pilot-handoff-audit.json)
exited 0 at 19:05:02Z and independently validated 2/8 arm jobs (fixed-margin
endpoint and alignment), each with 128 optimizer updates and 248 validation rows.
At 19:07:29Z the live runner had saved 3/8 arms and was at Full step 64/128;
that later live count is separate from the two-arm independent audit snapshot.

`tools/validate_complete_D.py` is a new read-only artifact validator outside the
frozen runtime. It requires the bounded D receipt, official 1/1489 count, all 17
distinct model completions, authenticated captured corpus, restored full probe
bundle including response dimensions, and byte-identical deterministic contract
CSV replay. Its six count/identity mutation tests passed (exit 0). The actual D
invocation returned `NOT_READY` / exit 3 because the bounded result is absent;
it did not write a PASS receipt or modify D. Its full artifact path is therefore
still unverified. D had 12/17 models and 0/1489 units at the last observation.
Formal E and F have not been executed by this continuation.

The research supervisor can request a cooperative checkpoint-boundary pause
when D's bounded receipt appears, leaving GPU 1 available for formal E. It sends
no signals to D. This is a real process logger, not a promise of unattended
completion or a reliable full-chain ETA. Original c591701 eight-cell meter
conclusions, old 9850fff G5 FAIL, and lack of SOTA evidence remain unchanged.
