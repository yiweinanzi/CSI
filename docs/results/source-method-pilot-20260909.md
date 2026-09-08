# Independent Source Method Pilot, 2026-09-09

Status at protocol freeze: implementation and synthetic smoke verified; formal-NPZ
pilot not yet launched. This is separate from the Streaming execution repair.
No result in this document is SOTA evidence or a completed formal factorial run.

## Authorization And Scope

The user explicitly confirmed independent method research and new fair four-arm
training on existing hardware. No paid resources, target-driven tuning, altered
gates, or overwritten historical evidence are authorized. Streaming D remains
running from its own frozen `2ce2472` worktree and output directory.

The original upstream commit remains `c5917018610b262f5263ccaeba2f087b2eb96a96`.
The actual NPZ SHA-256 is
`060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`.
Existing core Python 3.12.13 / torch 2.5.1+cu121 is reused without upgrades.
Input authentication runs under original imports, checking LIVE role verification,
the qualification teacher, and the factorial manifest containing source-only
normalization. The old trained arm checkpoints are not reused for this pilot.
Original G5 FAIL does not prohibit this independent NON_CLAIM research task.

## Frozen Experiment

Machine-readable protocol:
`code/CSI-PAIRS-v2.0-server/formal_v2/configs/source_method_pilot_20260909.json`.

| Item | Declared Setting |
| --- | --- |
| Recipes | Existing fixed margin vs existing effect-aware margin |
| Arms per recipe | endpoint, alignment, response, full |
| Training | 128 original optimizer steps per arm, seed 20270001 |
| Pairing | One actual shared initial state and identical deterministic batch plans |
| Calibration | Shared endpoint-only 128-step fit; same source selection samples; recipe-specific alignment scale, shared response scale |
| Optimizer / architecture | Original AdamW, batches, losses except declared margin, weights and architecture |
| Precision | Original FP32; no AMP or TF32 added; runtime precision recorded |
| Source training | Austin/Chicago source_encoder_train scenes 0-31 |
| Source selection | Austin/Chicago source_method_selection scenes 32-39 |
| Evaluation grid | All source selection cities, k=0/8/32/128, 10 draws (k=0 once) |
| Metric | Equal city/budget macro mean of per-bank median meter errors |
| Device | Dedicated logical cuda:1, CUDA_VISIBLE_DEVICES=0,1 |
| Ceiling | NON_CLAIM, sota_ready=false, formal_units_added=0 |

The formal P0 validator still requires fixed margin. It is not changed or
disabled. The separate research entry validates the original base config and
allows exactly the two declared recipes. Candidate configs are not formal P0
approvals and cannot be passed to the formal runner as accepted evidence.

Original `source_method_selection` permitted kappa/scale calibration only. This
new research explicitly expands its use to candidate validation; it does not
retroactively change the old protocol. Existing aggregate target results have
already been seen, which limits claims of prospective hypothesis independence.
No new target metrics select this candidate.

Source validation positions have no original target support/query role. A
seeded half of unique physical positions forms a fixed support-only pool; the
rest form validation-only rows. ID matches and coordinate siblings (1e-9 m)
are excluded across banks. Support draws are nested across budgets and paired
across arms/recipes. The dataset loader validates the entire NPZ structure;
the checked computation view restricts all research fitting/selection reads
to the two source roles. This is an accidental-leakage guard, not a Python
security sandbox against malicious code.

Promotion only means eligibility for a larger source pilot: candidate Full must
be strictly best among its four arms in every source city/budget cell and must
improve the macro error over fixed-margin Full. There is no automatic long
formal retraining or SOTA claim based on a 128-step, one-seed pilot.

## Recovery And Resources

Single writer lock; new output root; original input read lock. The actual model
and head initial states are saved once and checked by value hashes. Arm training
uses the existing model+optimizer checkpoint implementation every 16 complete
steps. A result receipt is committed only after state, complete validation CSV,
and trace are saved. Reuse checks identity, step coverage, receipt and artifact
hashes. Mere file existence is not a completion condition.

Incomplete calibration restarts from its original initialization. Validation
interrupted after training restarts validation from that durable trained arm;
it does not redo completed optimizer updates or pretend to resume a head step.
Elapsed time stored inside a training checkpoint excludes the write latency of
that checkpoint. Parameter/trace equality, not identical wall time, is tested.

The research resource guard checks process peak RSS, host available RAM, and
the mounted cgroup memory headroom. Before launch the draft RSS envelope was
tightened to 64 GiB, with a 64 GiB available-memory floor. The measured container
limit is 369367187456 bytes; host RAM alone is not the admission criterion.
These are cooperative phase/step checks,
not hard memory isolation. It can pause this research at a durable arm boundary.
The supervisor records the real PID/start ticks/command/exit and GPU UUIDs,
and can request a research-only pause when D's bounded receipt becomes available
for formal E. It never signals D or other jobs. No promise of unattended
experiment completion follows from the supervisor being launched.

## Tests At Protocol Freeze

`python -B -m pytest -q -p no:cacheprovider formal_v2/tests/test_source_method_pilot.py`

Second run: 18 passed, 2 non-fatal existing Transformer warnings, 12.38 seconds,
exit 0. First run is retained: 3 failed / 15 passed. One failure exposed the
formal fixed-margin boundary; two were test assumptions (NumPy string truncation
and checkpoint time accounting). Neither was a method-performance result.

The source tests plus original factorial integrity tests subsequently passed
41 tests in 16.25 seconds (exit 0). Original-input authentication exited 0 at
2026-09-08T18:37:28Z: source roles verified PASS; original factorial status FAIL
preserved. A read-only real-NPZ metadata check found 512 support candidates per
source city. Source-validation counts per bank were Austin 123/124/135/130 and
Chicago 122/124/136/130; only scene rows 32-39 were used in this partition check.

The expanded CPU/CUDA-1 synthetic integration, factorial, and CLI regression
passed 48 tests in 25.48 seconds (exit 0). The GPU branch checks CUDA gradients
before optimizer steps and changed final parameters, and repeats authenticated
state/trace restores for both recipes and all four arms. It is not a formal
data result or a full CPU/GPU numerical-equivalence study for the main model.

Synthetic integration checks original corpus helpers through the source guard,
both calibration formulas, eight two-step arm executions, actual shared model
initialization, authenticated final checkpoint restore, unchanged endpoint and
response controls, and source-only localization. Synthetic tests do not use
the formal NPZ as a fixture and add zero official units.

## Historical Conclusions Remain

The original c591701 Full did not beat Response-only in any of eight target
city/budget cells. This is not a claim that every difference is statistically
significant. Old 9850fff and c591701 G5 FAIL remain unchanged. Small reported
alignment gradient norms were first/last-step samples, not proof of trajectory-
wide gradient conflict. They motivate a source experiment, not a causal result.

For this pilot, `identity.json`, `source_origin.json`, `protocol.json`, paired
initial states/plans, per-arm receipts, and supervisor start/exit records are
the authoritative execution evidence. Follow-up results must name their actual
run ID and commit and must not replace this protocol with observed outcomes.
