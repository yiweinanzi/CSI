# Fixed Evaluation Performance Investigation

Recorded while PID 66858 was active. This note is outside the repository and
run root. It does not alter the fixed experiment.

## Fixed-run identity

- Git commit: `9850fffe0b34f16b45066973308f18b10555ca5d`
- Dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- Dataset archive: 2,006,664,110 bytes; approximately 18.2 GB uncompressed
- Active process identity SHA-256: `bcc357b38dcc6fc737a1b4a04c933663160e47713e53960831803845a96fbc09`

## Candidate hotspots

1. `run_formal_evaluation` loops over every one of the 12 checkpoint rows and
   reconstructs compatibility and response datasets for each checkpoint
   (`formal_v2/formal_evaluation.py:119`).
2. `_response_probe_dataset` nests scene, directed edge, eligible position, and
   query loops. It performs four separate one-sample `model.state` calls for
   each query (`formal_v2/formal_evaluation.py:1182-1260`).
3. `_native_mask_cover_metrics` has another scene/edge/position/query loop with
   one-sample state and prediction calls (`formal_v2/formal_evaluation.py:1601`).
4. `_compatibility_effect_rows` iterates unique pair IDs and performs a complete
   equality scan of the pair array for each ID (`formal_v2/formal_evaluation.py:1098`).
5. `_response_effect_rows` repeats the same full-array scan per unique pair ID
   (`formal_v2/formal_evaluation.py:1375`). With one unique ID per logical row,
   this is near O(N^2).
6. `_paired_score_differences` and `_active_effect_bin_rows` contain additional
   repeated equality scans (`formal_v2/formal_evaluation.py:963` and `:978`).
7. Stage files are written only after all checkpoints finish
   (`formal_v2/formal_evaluation.py:537`), so the fixed implementation has no
   durable intermediate progress or checkpoint recovery.

## Formal-scale count audit

This count uses only NPZ array headers and the small role arrays; it does not
load the large CSI or map arrays.

- Evaluation scenes contain 20,992 eligible positions: 10,496 from 41
  `source_final_unseen_bank` scenes and 10,496 target query positions from 82
  target scenes.
- The frozen world graph has 16 four-bit worlds and therefore 64 directed edit
  edges per scene. The frozen patch specification has 16 patches.
- `_response_probe_dataset(active_only=False)` therefore constructs up to
  `20,992 * 64 * 16 = 21,495,808` response units per checkpoint, or
  `257,949,696` units across 12 checkpoints.
- Response pair IDs include directed source/target worlds, position, and query,
  so each logical response unit has a unique pair ID. The legacy
  `_response_effect_rows` loop can consequently perform approximately
  `21,495,808^2 = 4.62e14` pair-array element comparisons per checkpoint.
- The alignment path has 28 eligible undirected hypercube edges after removing
  the four edges incident to the natural world. It constructs up to 2,351,104
  compatibility rows and 1,175,552 output pairs per checkpoint, or 14,106,624
  output pairs across 12 checkpoints.
- Both effect-row collections accumulate in Python lists across every
  checkpoint. `bind_rows` then copies each dictionary into another complete
  list before `write_csv` starts (`formal_v2/formal_evidence.py:782` and
  `formal_v2/formal_io.py:47`). This creates material memory and final-disk
  capacity risk in addition to the near-quadratic runtime.

At the 2026-08-30T18:42:48+08:00 supervisor sample, the fixed process used
141,891,813,376 bytes RSS and the filesystem had 168,734,830,592 bytes free.
These values are observations, not a completion estimate. The active process
must remain untouched under the fixed-run constraints.

## Candidate fixes to validate after the fixed pipeline stops

- Build one stable grouping/index pass for pair IDs and consume contiguous or
  indexed groups without repeated whole-array masks. Preserve `np.unique`
  ordering and current duplicate/error checks exactly.
- Batch model state/prediction calls over compatible query units. Precompute
  normalized map, radio, action, mask, and patch inputs keyed by immutable
  dataset identities. Verify every output field against the fixed code oracle.
- Persist the smallest independent checkpoint/checkpoint-row work units using
  temporary-file validation followed by atomic rename. Bind each shard to code,
  config, dataset, checkpoint, range, schema, and output hashes.
- Resume only authenticated COMPLETE shards; reject stale identities and
  recompute only damaged or partial shards. Use a writer lock.
- Stream or merge completed shards in the exact legacy sort order, preserving
  duplicate handling, tie behavior, floating-point order/tolerances, NaN rules,
  and output schema.

These are candidates only. The required isolated-worktree profile, oracle
comparison, resume tests, and N/2N/4N benchmark have not started and must not
replace or contaminate the active fixed-version result.
