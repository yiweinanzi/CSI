# Evaluation handoff hardening receipt (2026-09-01)

Scope: orchestration and evidence validation only. No dataset, checkpoint,
scientific configuration, metric, gate, or active scientific worker was changed.

## Authenticated script identities

- `supervise_all_streaming_8d489b2.sh`: `58c5d9696484532e18ca4a61673016b5b8cc5c0f3ec7093b04a5f548b03d8ba5`
- `validate_migration_gates_8d489b2.sh`: `de248fcc97cabd4b3979c8f39e39efb068ae0cc0eeffcae6de8d8b0e0606b7d1`
- `create_migration_request_8d489b2.sh`: `ae5fcb6833a8361046f903391392ebeeb06cb2517339772ba6ebcde83db2a417`
- `run_independent_migration_judge_8d489b2.sh`: `3d1dd3a0323f4e5769679474f6145c121dfca75debf664f59efc068c82cdd819`
- `accept_and_launch_after_judge_8d489b2.sh`: `1130689a0d2cda160dacf05cc408a77f462da49af6d1972f38504186fc2f477b`

## Fault tests

`SUPERVISOR_FAULT_TEST=PASS`

The isolated test verified that the launcher wrapper closes supervisor lock FD 9,
a restarted monitor releases a wrapper stranded between identity publication and
the `go` handshake, and an intentionally terminated test wrapper is reattached
through the launcher's authenticated session. The recovered receipt contained
exactly one matching `EXIT_CODE=0` log segment.

`COORDINATOR_REATTACH_TEST=PASS`

The isolated test held a synthetic supervisor lock with a matching PID/start-tick/
cmdline identity. The coordinator bypassed the idle-GPU wait, authenticated the
active supervisor, and emitted a `REATTACHED` launch receipt path. Its mocked
`nvidia-smi` function was not called.

All five scripts passed `bash -n`; the live and archived copies were byte-for-byte
identical; the embedded SHA chain passed. The tests used only temporary processes
and did not query, signal, pause, or modify protected scientific workers.

## Evidence validator additions

The strict validator now binds the performance report to all three observation
files and six stdout/stderr files in writer order, binds the GPU summary to the
4N observation, authenticates the runner runtime provenance against the deeply
validated candidate worker in the final real-subset report, binds controls to the
exact 4N evaluator output, and requires each control report result to equal its
raw observation result. The legacy runtime is separately checked for its frozen
requirements-lock identity; old and new source-bound runtime digests are not
incorrectly forced to be equal.
