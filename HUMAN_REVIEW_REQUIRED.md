# HUMAN REVIEW REQUIRED: C13 novelty and license review

This is the human-only review entry point for the CSI-PAIRS V6 formal
experiment. Machine evidence is complete, but it does not make a novelty,
overlap, licensing, redistribution, or scientific-readiness decision.

## Current binding

- Dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- Formal source-tree SHA-256: `a6284fe400a3443b3805ca8896b04513c1d064d25f2461fdd304e1cf8a5efd7f`
- Formal configuration SHA-256: `6842a565ba733ec6e6d2d3f96e99926cba75ea29dc87d46019bdea795d6def42`
- Paper claim file: `code/CSI-PAIRS-v2.0-server/paper_v2/main.tex`
- Paper claim SHA-256: `9a0d4aeb3655d16788c9d234cc454c8f6d22b45026de93877f96206e31379183`
- Frozen machine test evidence: `571/571 PASS`
- Search evidence: Crossref `3/3`, OpenAlex `3/3`, Semantic Scholar `3/3`
- Authenticated paper files: `7/7`

The formal source digest covers the executable/configuration files under
`formal_v2`. This review entry point and `.gitignore` are outside that digest.

## Read these project claims

Review the abstract, all three contribution bullets in the Introduction, the
complete Related Work section, and the Artifact-to-Claim Checklist in
`code/CSI-PAIRS-v2.0-server/paper_v2/main.tex`.

## Review these seven papers

The PDFs are deliberately excluded from Git history. Authenticated local
copies are in:

`/root/xunlian/Futaoran/formal_external_inputs/c13_literature/papers/`

| Paper | SHA-256 | License record | Redistribution record |
| --- | --- | --- | --- |
| `2406.14995v2.pdf` (Wi-GATr) | `6c60e162f4de8a7523626730e3c1725c8d730c56dbe445712656cd6710d22201` | CC BY 4.0 | allowed |
| `2502.11965v2.pdf` (CIR-CSI consistency foundation model) | `1b6aac49344d04e60e07ce29f950593b5699daddaac04a01ab848e7be3426c2d` | arXiv non-exclusive distribution | not established |
| `2505.09160v2.pdf` (contrastive/masked wireless representation model) | `b32ac897858212faf9cd7189f84f1bf94e522f8b8dfdaaa881b58cc798fde4d1` | arXiv non-exclusive distribution | not established |
| `2601.03789v1.pdf` (CSI-MAE) | `5ec21848bb1e82bd8b3331cea71e6663ce41f72cb410f6db72187c8088ae7d93` | arXiv non-exclusive distribution | not established |
| `2603.25216v1.pdf` (wireless world model) | `711304aed7766057263502d36af63927e97dff10d683dc1659e89e5a46b2b102` | CC BY 4.0 | allowed |
| `2604.07086v1.pdf` (RF inverse rendering) | `24838f6a01eab58bf31d7fe54167dc13a245d3e6a6986da4e3023cafe2e0ff83` | arXiv non-exclusive distribution | not established |
| `2606.04770v1.pdf` (WiSER) | `97e368043d0c111987d26993ec88c295c5bb997126e21711223ca5448d1645ec` | CC BY 4.0 | allowed |

For each paper, record its relevance, overlap category (`direct_overlap`,
`adjacent_nonoverlap`, `baseline`, or `facility`), implementation status,
license verified at the authoritative source, redistribution permission, and
any restrictions.

Also verify the official source archives and upstream license terms:

| Archive | Revision | License record |
| --- | --- | --- |
| Wi-GATr | `6daa5bd49d9499817d903ab335830221d01e9daa` | BSD-3-Clause-Clear |
| PMNet | `a0e0c5926de721074beeb23f630f2d313f6508dd` | MIT |
| Sionna | `04ddb9312116b408093b9d3ad363a3df355093a6` | Apache-2.0 |
| Sionna large radio maps | `1ba19ae1df1d26302fcfbaab14efc2347313da5d` | Apache-2.0 |

Do not infer a human licensing decision solely from the table. Verify the
upstream license and the actual resource before signing.

## Required human decisions

The completed review must explicitly state:

- Whether every local paper and source license was reviewed.
- Whether any prior work directly overlaps the frozen C13 claim.
- Whether the RT, map, and external-validity paths are ready.
- The precise, non-promotional novelty scope that remains supportable.
- Every conflict or unresolved restriction, or `none` only after review.
- A record-by-record decision for all seven papers and four source archives.

## Required output

After personally completing the review, create:

`/root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW.md`

Use the full local template at:

`/root/xunlian/Futaoran/formal_external_inputs/c13_literature/HUMAN_REVIEW_TEMPLATE.md`

The completed file must contain all of the following, with no placeholders:

```text
Reviewer name or authorized identity:
Affiliation or authorization basis:
Review completed UTC:
Dataset SHA-256:
Project source-tree SHA-256:
Project formal configuration SHA-256:
Paper claim SHA-256:

Record-by-record review:
  [all seven papers and all four source archives]

Licenses reviewed for every local PDF/source resource: true|false
No direct overlap with the frozen C13 claim: true|false
RT path ready: true|false
Map path ready: true|false
External-validity path ready: true|false
Allowed novelty scope:
Conflicts or unresolved restrictions:

I attest that I personally reviewed the listed resources and the frozen C13
claim, verified the recorded license/redistribution decisions from the cited
sources, and made the novelty and readiness decisions above.

Reviewer signature or authenticated identity:
Signed UTC:
```

An AI-generated signature or a renamed blank template is not valid evidence.
