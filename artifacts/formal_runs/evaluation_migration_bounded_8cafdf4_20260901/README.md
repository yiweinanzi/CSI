# CSI-PAIRS V6 bounded evaluation migration snapshot

Snapshot time: 2026-09-01 09:55 +08:00

This directory records the small, reviewable evidence available while the
bounded evaluation migration was still running. The authoritative live evidence
root remains:

`/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901`

## Bound identities

- New commit: `8cafdf4a4c67d40c4fd41b7d71b6468df850cdd7`
- New source SHA-256: `a5e2658b88c051e7d83256d8b802b914c76659162464fb1e897f76f0473210f0`
- Legacy commit: `9850fffe0b34f16b45066973308f18b10555ca5d`
- Dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- New run ID: `csi-pairs-08828d387df6db2b`

## Status at snapshot

- All 12 formal training cells completed 20,000 steps before this migration.
- The factorial result remains `FAIL`, `G5=FAIL`, and
  `CANDIDATE_NOT_CLAIM`. No engineering result in this directory promotes that
  scientific state.
- Critical regression: 232 passed, 1 skipped, 0 failed. The 75 warnings are the
  known PyTorch nested-tensor warning.
- The legacy evaluation produced no accepted output and was frozen after a
  memory-cgroup OOM. The authenticated empty inventory and post-exit receipt are
  included under `post_exit_freeze/`.
- The first bounded performance attempt was intentionally rejected because a
  concurrent Wi-GATr preflight made GPU1 nonexclusive. Its traceback is retained
  under `failed_attempts/`; it is not valid performance evidence.
- A clean exclusive performance rerun and the cross-source real-subset check
  were still running. Their final reports are not represented by placeholders.

## Included

This snapshot includes source/run identities, the frozen compute plan, strict
receipts, the critical JUnit report, runtime provenance preflight, launch and
supervision scripts, the independent LLM judge prompt, nonempty diagnostic
logs available at snapshot time, and the complete base-to-new `formal_v2`
review patch.

## Deliberately excluded

- datasets, fixtures, checkpoints, model weights, CSV tables and run outputs;
- virtual environments and external runtime trees or links;
- active temporary files and empty wait logs.

Final performance, equivalence, controls, migration approval, evaluation and
downstream receipts must be added only after their official validators pass.
