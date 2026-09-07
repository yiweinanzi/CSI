# Fresh c591701 four-arm training snapshot

All 12 training cells are complete: endpoint, alignment, response and full for
seeds 20270001, 20270002 and 20270003, with 20,000 steps per cell (240,000 total).
The final checkpoints retain their original seed directories and SHA-256 values.

Training completion is not localization or SOTA evidence. Localization,
streaming evaluation, C1 with Wi-GATr/PMNet, controls and claims are not complete.
This archive is `scientific_use=CANDIDATE_NOT_CLAIM`, `fixture=false`, and
`sota_ready=false`. Training losses are not localization errors in meters.

## Identities

- Training code: `c5917018610b262f5263ccaeba2f087b2eb96a96`.
- Source SHA-256: `8e8eaf212578b3dc624e40d1d979f0c37787afa3e7e464eac52d5b68610dfe07`.
- Formal NPZ SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`.
- Config SHA-256: `3424a269a751f219f4897b969db7cc13d2236fc72e8e2ccb663dec8d7f8f477c`.
- Requirements lock SHA-256: `9548a24a4faf5db17b5251df407b4402de033ae8648ec321c850fe6105002cf1`.
- Run: `runs/c591701-formal-prepare-20260902T2040` in `CSI_MAIN_C591701_20260901`.

The upload commit archives these artifacts; it is not their training-code
identity. The original run and all signed or hashed provenance files remain
unchanged, including absolute paths recorded on the training machine.

## Contents

- `training_summary.csv`: original 12-row training table.
- `checkpoints/seed_*/`: all 12 original final checkpoints.
- `checkpoint_index.json`: original bindings and checkpoint SHA-256 values.
- `resume_index.json`: original recovery index; optimizer recovery slots are in
  the complete raw archive, not in the final-checkpoint directory.
- `normalization.json` and `frozen_pilot.json`: original frozen fit artifacts.
- `qualification/checkpoints/stage0_csi_teacher.pt`: bound frozen teacher.
- `snapshot/`: readable reports, logs, preflight, LIVE verification,
  qualification, approval and wrong-map diagnostics, including earlier failures.
- `provenance/LLM_JUDGE_APPROVAL.json`: original approval for the fresh run.
- `source_inventory.json`: original paths, lengths and hashes for all run files.
- `snapshot_status.json`: archive status, provenance and explicit omissions.
- `raw_snapshot/`: complete raw run files as five Git LFS archive parts,
  with a restoration script and file-by-file archive verification receipt.
- `SHA256SUMS`: integrity checks for the published snapshot.

## Progress

| Stage | Status |
| --- | --- |
| Operator preflight | Recorded |
| LIVE data verification | PASS, 227 banks |
| Qualification | QUALIFICATION_PASS |
| Fresh-run approval | ACCEPTED |
| Wrong-map diagnostic | DIAGNOSTIC_COMPLETE_NOT_DOMAIN_EVIDENCE |
| Four-arm training | 12/12 complete, 240,000/240,000 steps |
| Four-arm localization and G5 | Not complete |
| Streaming evaluation and C1 | Not complete |
| Wi-GATr/PMNet and controls | Not complete |
| Claims | Not complete; SOTA evidence not established |

The historical `9850fff` experiment remains a negative result: G5 FAIL, with
Response-only performing better. Earlier engineering-run failures are preserved
in their own snapshot directory and never promoted to fresh-run evidence.

## Verify

Run `sha256sum --check SHA256SUMS` from this directory. The complete raw archive
also retains the large regeneration files, per-sample wrong-map CSV, recovery
slots and earlier verification attempts. Consult its manifest for storage and
restoration details. Local environments, runtime links, unrelated worktrees and
the separate authoritative NPZ are excluded; the authoritative NPZ is bound by
its exact SHA above.
