# C13 external inputs still required

Checked: 2026-08-23T16:49:53Z

Current machine-verified inventory:

- Bound dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`.
- Bound final project source-tree SHA-256:
  `a6284fe400a3443b3805ca8896b04513c1d064d25f2461fdd304e1cf8a5efd7f`.
- Superseded source-tree SHA-256 values `ba16e7a...eec595` and
  `c640169b...e8c8` must not be used in final C13 evidence.
- Final-source tests: 571/571 PASS, with 0 failures, errors, or skips. The
  JUnit SHA-256 is
  `e82acb060d6cc05cf9680e2501c573b3db7b3f2a3747a24b0d4747114ebfb19a`.

- Crossref receipts: 3/3 frozen queries, each with 20 identified results.
- OpenAlex receipts: 3/3 frozen queries, each with 20 identified results.
- Authenticated paper PDFs: 7 files with hashes matching the project resource
  registry.
- Semantic Scholar receipts: 3/3 frozen queries, each with 20 identified
  results and nonempty `paperId` values. The incremental summary SHA-256 is
  `49ebedf7598c463351d3c82cbba991fcb513be6872698093fc6ab751ac6884cb`.
- A fresh frozen-URL-compatible Crossref/OpenAlex package is available at
  `packages/c13_machine_search_20260823T014300Z/`. All six receipts pass the
  project verifier. Final C13 material must use this package rather than the
  older OpenAlex `filter=fulltext.search:` URLs.
- Human novelty/license review: absent. This is the only remaining C13
  external input.

## Machine search complete

All nine official database/query pairs pass the frozen production receipt
rules. The complete window spans 52,534 seconds and is below the 24-hour
limit. Do not refetch or overwrite these receipts. Independent machine
validation is saved at
`artifacts/formal_readiness/c13_machine_validation_final_a628_20260823T164900Z/validation.json`.

## Missing human decisions

A real authorized reviewer must create `HUMAN_REVIEW.md`. It must identify the
reviewer and review time, list the reviewed records, record the license and
redistribution decision for every local PDF/source resource, assess direct
claim overlap, state the allowed novelty scope, and decide whether the RT, map,
and external-validity paths are ready. It must bind the current formal dataset
SHA-256 and the final frozen project source-tree SHA-256; any later source edit
invalidates the review and requires a new human review. The reviewer, not an AI process, must
then supply the corresponding values for:

- `licenses_reviewed`
- `decision.no_direct_overlap`
- `decision.rt_path_ready`
- `decision.map_path_ready`
- `decision.external_validity_path_ready`
- `decision.novelty_scope`

The machine-prepared evidence is in `C13_REVIEW_PACKET.md`. The reviewer may
start from `HUMAN_REVIEW_TEMPLATE.md`, but the template is explicitly invalid
until every placeholder is replaced after an actual review and the completed
document is saved as `HUMAN_REVIEW.md`.

## Final package and validation

After the real human review exists and the final compute plan passes, create
`literature_manifest.json` with schema
`csi-pairs-v6-literature-resource-manifest-v4`. It must contain exactly one
receipt for each of the 3 databases times 3 queries and bind every receipt and
PDF by SHA-256, plus `human_review_path` and `human_review_sha256` for the
completed `HUMAN_REVIEW.md`.

The real reviewer must also include the completed lines from
`HUMAN_REVIEW_STRUCTURED_APPENDIX_TEMPLATE.md` in `HUMAN_REVIEW.md`. After the
review is signed and the final eight-component compute plan passes, assemble
the manifest without manually copying receipt hashes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-core-formal-20260819T091049Z/bin/python -B \
  artifacts/formal_readiness/tools/build_c13_literature_manifest.py \
  --root /root/xunlian/Futaoran/formal_external_inputs/c13_literature \
  --config formal_v2/configs/formal_v2.json \
  --dataset '/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz' \
  --resource-registry formal_v2/configs/waibu_resources_v1.json \
  --crossref-openalex-summary \
    /root/xunlian/Futaoran/formal_external_inputs/c13_literature/packages/c13_machine_search_20260823T014300Z/crossref_openalex_receipts.json \
  --semantic-summary \
    /root/xunlian/Futaoran/formal_external_inputs/c13_literature/semantic_scholar_incremental_receipts.json \
  --human-review \
    /root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW.md \
  --compute-plan \
    artifacts/formal_readiness/formal_compute_plan_final_a628_20260823T065000Z/compute_plan.json \
  --output \
    /root/xunlian/Futaoran/formal_external_inputs/c13_literature/literature_manifest.json
```

The builder parses decisions from the genuine human review, validates all nine
official receipts, seven PDFs, the final dataset/source binding and compute
plan, and refuses to overwrite any existing manifest. It does not make or
default any novelty, license, overlap, readiness, reviewer or signature value.

Validate it in a fresh output root with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-core-formal-20260819T091049Z/bin/python -B \
  -m formal_v2.formal_cli run-literature-resources \
  --config formal_v2/configs/formal_v2.json \
  --dataset '/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz' \
  --output '<fresh-c13-validation-root>' \
  --literature-resource-manifest \
  /root/xunlian/Futaoran/formal_external_inputs/c13_literature/literature_manifest.json
```

Until that command issues a current `G0 PASS`, C13 remains an external P0
blocker and the verdict must remain `FORMAL_RUN_NO_GO`.
