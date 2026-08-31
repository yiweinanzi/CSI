# CSI audit handoff

Created: 2026-08-09 11:26 Asia/Shanghai

Source repository: `/root/xunlian/Futaoran/CSI`

Base branch/commit: `main` at
`11595f168f9fcf36dc178aaee927b348d7c6402a`

Current formal verdict: `FORMAL_RUN_NO_GO`

## Contents

- `project/`: current repository snapshot with all modified and untracked source,
  generated datasets, formal inputs, paper/docs, and formal-readiness evidence.
- `WORKTREE_CHANGES.patch`: binary-capable diff for tracked changes against the
  base checkout. Untracked files are present directly in `project/`.
- `GIT_STATUS.txt` and `UNTRACKED_FILES.txt`: handoff-time worktree inventories.
- `SHA256SUMS`: SHA-256 for every delivered file except the checksum file itself.

The source snapshot excludes only `.git`, `formal_envs`, project virtual
environments, installed Sionna/Wi-GATr runtimes, and Python bytecode caches.
These are reproducible from the included Conda, lock, wheel-manifest, setup, and
runtime-provenance files. No source code, dataset, formal input, or audit evidence
was excluded.

## Main dataset

```text
project/generated_datasets/csi_pairs_v2_1_v6_sionna_osm_candidate_20260809T091000Z_final_candidate/dataset.npz
SHA-256: 6534777ee33d6cc2f4b964b60e7a6fcd5f8bb14fd9b4714e19cb300c9a9bd5fc
Type: Sionna RT simulation, not measurement
fixture=false; scientific_use=CANDIDATE
34 banks; 6 cities; 4 worlds; 256 positions; 3 repeats
34,816 clean units; 104,448 repeated observations
```

This candidate is complete-scale but is not qualified paper data. Its frozen
CUDA exact-regeneration gate fails, and 16,124/34,816 clean units are no-path
all-zero units. See `project/artifacts/formal_readiness/PRE_RUN_AUDIT.md` and
`READINESS.json`.

## Verification state

- The server source inventory in
  `project/code/CSI-PAIRS-v2.0-server/SHA256SUMS` passed all entries immediately
  before this handoff was copied.
- The last completed clean full suite before the final metadata regression was
  333/333 PASS; its log SHA-256 is
  `5063377079f34a77cc4a22a7695f85495d4025ae110c5d440a3113aabe2e4647`.
- The new fail-closed Sionna metadata regression passed as a targeted test.
- A post-repair 334-test full suite was running when the user requested stop. It
  was interrupted with exit 130 and has no PASS/FAIL verdict. The partial log is
  `project/artifacts/formal_readiness/logs/unittest_full_post_replay_metadata_20260809T030500Z_v2.log`
  with SHA-256
  `219c548a72a11e5ebb78ec597519817b549a113250bdb61da782443210197714`.
- No qualification, formal-data pilot, `prepare-full-run`, or multi-day formal
  training was started.

All audit, test, render, and training processes were stopped before packaging;
both A100s had no active compute process at the stop check.
