# Source Publication, 2026-09-09

This publication integrates current source and small existing evidence onto the
previous GitHub main, `035d4079860e77a0a9062addfb7e6bded6ecf9e8`.
No training or evaluation task was started, stopped, restarted, or reconfigured
for publication. Running source worktrees were not edited.

## Current Source

| Branch | Frozen commit | Scope |
| --- | --- | --- |
| `fix/streaming-gpu-probes-20260908` | `2ce24729814d1d7357fafb280ab10192ee3ace6b` | Streaming execution repair |
| `experiment/source-method-pilot-20260909` | `e270230fa950337469f3f25340c22874767d2ac2` | Execution repair plus isolated source-only research |
| `evidence/streaming-gpu-repair-20260908` | `bf9abba789cb323c141043de70d23cd9d5aaa871` | Existing repair tests, bounded acceptance records and audits |

The integrated `code/` tree is byte-for-byte identical to `e270230`.
The single merge conflict was the source-pilot parser registration in
`formal_cli.py`; the complete frozen source-pilot entry was retained.
The existing streaming evidence directory and report are identical to `bf9abba`.

## Historical Source Archives

These branches preserve earlier or uncommitted code without replacing current
methods. Their existence is not a claim that the historical code is validated.

| Branch | Snapshot commit | Captured source |
| --- | --- | --- |
| `archive/local-main-source-20260909` | `d8e2974f436caca7ca4f558a17f8641d72daf2f4` | `CSI`, based on `11595f1`: 21 tracked changes and 21 untracked source files |
| `archive/cloud-source-20260909` | `89cd51b9cc6571e6c7fdd26f6b965ffd4dca1c1a` | `CSI_CLOUD_LATEST_3183664`, based on `5d068ff`: 20 tracked changes and 7 untracked source files |
| `archive/workspace-tools-20260909` | `8aa99c38d6785f6a8fdb0e0b509f6918aebd8473` | 457 loose helpers, source files, dependency lists and configurations under `workspace_source/` |

The last archive covers `formal_external_inputs`, `CSI_HANDOFF_20260809T032602Z`,
`baseline_repair_scratch_20260823T021700Z`, and
`formal_resume_repair_scratch_20260823T033000Z`, excluding vendor and installed
environment code. Historical local code branches are also preserved separately.
Snapshot manifests contain per-file SHA-256 values and original base commits.
Separate temporary Git indices preserved the original worktree files, HEADs,
uncommitted diffs, and real Git indices.

## Upload Boundary

New files are limited to 10 MiB each. The formal NPZ, main-model checkpoints,
virtual environments, generated datasets, runtime caches and large raw run
outputs are not added. Existing GitHub artifacts and LFS references are retained,
not deleted or re-uploaded. Small fixed smoke inputs and probe verification
outputs already committed on the evidence branch remain part of that evidence.

New Git objects are checked for size, new LFS pointers, and common credential
patterns before publication. This is a bounded secret scan, not a guarantee
against every possible secret encoding. The push disables LFS payload transfer
for this invocation only, after confirming there are no new LFS pointers.
No repository hook or scientific identity check is disabled or edited.

## Publication Checks

The merged `code/` and preserved evidence comparisons both exited 0.
`git diff --cached --check -- code docs .gitignore` exited 0. The unrestricted
whitespace check warns about spaces in historical patch/log evidence; those
bytes are preserved to avoid changing authenticated evidence.

Using the existing core Python environment, with `CUDA_VISIBLE_DEVICES` empty,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1`, the publication check ran:

```bash
python -B -m pytest -q -p no:cacheprovider \
  formal_v2/tests/test_metrics_auroc.py \
  formal_v2/tests/test_evaluation_repair.py \
  formal_v2/tests/test_formal_probes_devices.py \
  formal_v2/tests/test_probe_acceleration.py \
  formal_v2/tests/test_source_method_pilot.py
```

Observed result: **44 passed, 4 skipped, 1 failed**, exit 1, 15.68 seconds.
The failure is `test_mock_cuda_training_transfers_only_cpu_backed_batches`:
its mocked CUDA path still invokes real `torch.cuda.synchronize`, which fails
when GPUs are deliberately hidden. The same single test failed at the same
call on untouched frozen `e270230`, exit 1, 8.23 seconds. This is recorded as a
preexisting CPU-only test warning, not a passing test or a merge regression.
No GPU acceptance or full formal evaluation was rerun for this upload.

Machine-readable source manifests and JUnit records are in
[`artifacts/source_publication_20260909`](../../artifacts/source_publication_20260909).

## Scientific Status

This is a source publication, not experimental acceptance. The historical
`9850fff` negative result and G5 FAIL remain unchanged. The recorded `c591701`
Full result did not beat Response-only in the eight compared cells; this does
not assert significant degradation in every cell. The source pilot remains
NON_CLAIM. This upload supplies no new SOTA evidence and changes no gate.
See the [existing repair report](streaming-execution-repair-20260908.md) and
[source-only pilot record](source-method-pilot-20260909.md) for their separately
dated observations; the publication does not upgrade incomplete acceptance.
