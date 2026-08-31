# C13 machine-prepared review packet

Status: `FINAL_SOURCE_BOUND_MACHINE_EVIDENCE_COMPLETE_WAIT_FOR_HUMAN_REVIEW`

This packet binds the authenticated final source `a6284fe...a5efd7f` and its
571/571 passing test run. The prior `ba16e7a...eec595` and
`c640169b...e8c8` source bindings are superseded and must not be used for C13.
All nine genuine search receipts now exist and pass the frozen machine
verifier. This makes the packet ready for review; it does not make or imply a
novelty, overlap, readiness, license or redistribution decision.

This packet reduces clerical work but makes no novelty or license decision.
The authoritative reviewer must verify every URL and restriction before
creating `HUMAN_REVIEW.md` from `HUMAN_REVIEW_TEMPLATE.md`.

## Bound review target

- Formal dataset: `/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz`
- Dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- Project source-tree SHA-256: `a6284fe400a3443b3805ca8896b04513c1d064d25f2461fdd304e1cf8a5efd7f`
- Formal configuration SHA-256: `6842a565ba733ec6e6d2d3f96e99926cba75ea29dc87d46019bdea795d6def42`
- Frozen paper claim file: `paper_v2/main.tex`
- Frozen paper claim file SHA-256: `9a0d4aeb3655d16788c9d234cc454c8f6d22b45026de93877f96206e31379183`
- Final-source test result: `571/571 PASS` (`0` failures, errors, or skips)
- Test JUnit SHA-256: `e82acb060d6cc05cf9680e2501c573b3db7b3f2a3747a24b0d4747114ebfb19a`

The reviewer must inspect the abstract, the three contribution bullets in the
Introduction, the complete Related Work section, and the Artifact-to-Claim
Checklist. Any later `formal_v2` source/configuration or paper-claim change
invalidates this target and requires a new human review bound to the new hashes.

The frozen C13 contract is deliberately narrow: a current prior-work search
with auditable evidence is required, and even a machine G0 PASS leaves C13 as
`REVIEW_REQUIRED`. The reviewer must write the allowed novelty scope; this
packet does not assert priority, first-of-kind status, or absence of overlap.

## Frozen search scope

- Required databases: Crossref, OpenAlex, Semantic Scholar
- Frozen queries: `wireless channel foundation model`, `CSI map localization`,
  and `wireless scene representation`
- Crossref receipts: 3/3 present, 20 results each
- OpenAlex receipts: 3/3 present, 20 results each
- Semantic Scholar receipts: 3/3 present, 20 results each. Receipt SHA-256
  values are `01b6e5baba34e17c54c56ec4e17b37a2eba6528d3067777351a9208094134e63`
  for `wireless channel foundation model`,
  `e26d8d5ccd7ca8650d6f1d4214f3b1d6d48773e7081121b057448e54237dcc96`
  for `CSI map localization`, and
  `4737cfe0d9643e981a98c20c4491fcf3cf457a0c6fd3c1bd9921ef67a0d6452e`
  for `wireless scene representation`.

The frozen-verifier-compatible Crossref/OpenAlex records to use for the final
manifest are under `packages/c13_machine_search_20260823T014300Z/`. They were
captured at `2026-08-23T01:44:25Z` through `01:44:38Z`; all six exact URL,
query, result-identity and SHA-256 checks pass. The older OpenAlex
`filter=fulltext.search:` receipts must not be used for the final gate.

The final-source machine validator applied the production receipt rules to all
nine records and authenticated all seven PDFs. The receipt window is 52,534
seconds, below the frozen 24-hour limit. Evidence is
`artifacts/formal_readiness/c13_machine_validation_final_a628_20260823T164900Z/validation.json`.
Its status remains `INCOMPLETE` solely because a real `HUMAN_REVIEW.md` is
absent; `formal_gate_eligible=false` is intentional.

## Authenticated local papers

| File | SHA-256 | Registry title | Registry license | Registry redistribution |
| --- | --- | --- | --- | --- |
| `2406.14995v2.pdf` | `6c60e162f4de8a7523626730e3c1725c8d730c56dbe445712656cd6710d22201` | Differentiable and Learnable Wireless Simulation with Geometric Transformers | CC BY 4.0 | allowed |
| `2502.11965v2.pdf` | `1b6aac49344d04e60e07ce29f950593b5699daddaac04a01ab848e7be3426c2d` | A MIMO Wireless Channel Foundation Model via CIR-CSI Consistency | arXiv non-exclusive distribution | not established for redistribution |
| `2505.09160v2.pdf` | `b32ac897858212faf9cd7189f84f1bf94e522f8b8dfdaaa881b58cc798fde4d1` | A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning | arXiv non-exclusive distribution | not established for redistribution |
| `2601.03789v1.pdf` | `5ec21848bb1e82bd8b3331cea71e6663ce41f72cb410f6db72187c8088ae7d93` | CSI-MAE: A Masked Autoencoder-based Channel Foundation Model | arXiv non-exclusive distribution | not established for redistribution |
| `2603.25216v1.pdf` | `711304aed7766057263502d36af63927e97dff10d683dc1659e89e5a46b2b102` | A Wireless World Model for AI-Native 6G Networks | CC BY 4.0 | allowed |
| `2604.07086v1.pdf` | `24838f6a01eab58bf31d7fe54167dc13a245d3e6a6986da4e3023cafe2e0ff83` | Radio-Frequency Inverse Rendering for Wireless Environment Modeling | arXiv non-exclusive distribution | not established for redistribution |
| `2606.04770v1.pdf` | `97e368043d0c111987d26993ec88c295c5bb997126e21711223ca5448d1645ec` | WiSER: A Wireless Scene Encoder for Geometry-Grounded Multi-View Wireless Prediction | CC BY 4.0 | allowed |

The values above are copied from the authenticated external-resource inventory;
they are evidence for review, not a substitute for the reviewer's decision.

## Authenticated official source archives

| File | SHA-256 | Source revision | License | Registry redistribution |
| --- | --- | --- | --- | --- |
| `Wi-GATr-main.zip` | `b78be8ed14d16aab6c8fea54c2aa90a22bed492a12a6e892ee4c225c60a8138c` | `6daa5bd49d9499817d903ab335830221d01e9daa` | BSD-3-Clause-Clear | allowed |
| `PMNet-a0e0c592.zip` | `e48e283e596270a9932052cc41f213b0a22c0ebbf29ba9551c5562d8cb877331` | `a0e0c5926de721074beeb23f630f2d313f6508dd` | MIT | allowed |
| `sionna-main.zip` | `fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7` | `04ddb9312116b408093b9d3ad363a3df355093a6` | Apache-2.0 | allowed |
| `sionna-large-radio-maps-main.zip` | `694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b` | `1ba19ae1df1d26302fcfbaab14efc2347313da5d` | Apache-2.0 | allowed |

These values are copied from `formal_v2/configs/waibu_resources_v1.json`
(SHA-256 `29240d1d326208e9feb6482ad6cf40cb74a66470368b2b4ff32b5c587889b24c`).
The reviewer must verify the cited upstream license sources and actual archive
contents; registry text alone is not a human license decision.

## Submission safety

After personally completing the review, follow
`HUMAN_REVIEW_SUBMISSION_INSTRUCTIONS.md`: stage the result as
`HUMAN_REVIEW.candidate.md`, run the structural/frozen-binding linter, and only
then atomically submit `HUMAN_REVIEW.md`. The linter does not make or validate
scientific, license, identity, authorization, or signature decisions.
