# C13 human novelty and license review template

Status: `TEMPLATE_ONLY_NOT_REVIEWED`

Source-freeze status: this template binds authenticated final source
`a6284fe...a5efd7f`, for which 571/571 tests passed. The prior
`ba16e7a...eec595` and `c640169b...e8c8` bindings are superseded. All nine
genuine search receipts are now frozen and machine-validated. A completed
`HUMAN_REVIEW.md` may be saved only after a real authorized reviewer has
performed the review and replaced every placeholder.

This file is not an approval and must not be renamed without a real review.
The reviewer must inspect the cited papers, project claims, source resources,
and license terms before filling every field. An AI process must not sign it.
When creating `HUMAN_REVIEW.md`, replace the template status with
`HUMAN_REVIEW_COMPLETED_BY_AUTHORIZED_REVIEWER`, keep each labeled value on
one physical line, and remove every placeholder.

## Reviewer

- Reviewer name or authorized identity: `<REQUIRED>`
- Affiliation or authorization basis: `<REQUIRED>`
- Review completed UTC: `<YYYY-MM-DDTHH:MM:SSZ>`
- Project dataset SHA-256: `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
- Project source-tree SHA-256: `a6284fe400a3443b3805ca8896b04513c1d064d25f2461fdd304e1cf8a5efd7f`
- Project formal configuration SHA-256: `6842a565ba733ec6e6d2d3f96e99926cba75ea29dc87d46019bdea795d6def42`
- Final-source test result: `571/571 PASS`
- Test JUnit SHA-256: `e82acb060d6cc05cf9680e2501c573b3db7b3f2a3747a24b0d4747114ebfb19a`
- Project paper claim file: `paper_v2/main.tex`
- Project paper claim file SHA-256: `9a0d4aeb3655d16788c9d234cc454c8f6d22b45026de93877f96206e31379183`

## Record-by-record review

For every record below, state: relevance to C13, overlap category
(`direct_overlap`, `adjacent_nonoverlap`, `baseline`, or `facility`),
implementation status, license verified at source, redistribution allowed, and
any restriction that applies to the paper, code, model, or data.
Copy the fourteen exact `Record ... relation_to_claim` and
`Record ... implementation_status` lines from
`HUMAN_REVIEW_STRUCTURED_APPENDIX_TEMPLATE.md` into `HUMAN_REVIEW.md` exactly
once and replace every placeholder with a reviewed allowed value.

- `2406.14995v2.pdf` - Wi-GATr
- `2502.11965v2.pdf` - CIR-CSI consistency foundation model
- `2505.09160v2.pdf` - contrastive/masked wireless representation model
- `2601.03789v1.pdf` - CSI-MAE
- `2603.25216v1.pdf` - wireless world model
- `2604.07086v1.pdf` - RF inverse rendering
- `2606.04770v1.pdf` - WiSER

Also review the authenticated official source archives and their upstream
license terms: `Wi-GATr-main.zip`, `PMNet-a0e0c592.zip`, `sionna-main.zip`, and
`sionna-large-radio-maps-main.zip`. Record whether local use and redistribution
are allowed for each archive; do not infer this only from the registry label.

## Required decisions

- Licenses reviewed for every local PDF/source resource: `<true or false>`
- No direct overlap with the frozen C13 claim: `<true or false>`
- RT path ready: `<true or false>`
- Map path ready: `<true or false>`
- External-validity path ready: `<true or false>`
- Allowed novelty scope: `<REQUIRED; precise, non-promotional wording>`
- Conflicts or unresolved restrictions: `<REQUIRED; write none only if reviewed>`

## Attestation

`I attest that I personally reviewed the listed resources and the frozen C13
claim, verified the recorded license/redistribution decisions from the cited
sources, and made the novelty and readiness decisions above.`

- Reviewer signature or authenticated identity: `<REQUIRED>`
- Signed UTC: `<YYYY-MM-DDTHH:MM:SSZ>`
