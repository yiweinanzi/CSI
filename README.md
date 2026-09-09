# CSI-PAIRS

This repository contains the CSI-PAIRS V2.1 implementation of the frozen V6 research protocol.

The [2026-09-09 source publication index](docs/results/source-publication-20260909.md)
links the current execution repair, source-only research code, and separate historical
source archives. This source upload does not change experiment or SOTA conclusions.

The authoritative protocol is
[`Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md`](Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md).
The executable server package is under
[`code/CSI-PAIRS-v2.0-server`](code/CSI-PAIRS-v2.0-server); the directory name is retained as a compatibility path, while the runtime, schemas, paper draft, and generated bundle are V2.1.

Current status:

- package/software: verified on the audited code commit, with final bundle verification recorded in the traceability audit;
- protocol implementation: V2.1 frozen-V6 path;
- formal experiment readiness: `CODE_READY_FOR_FORMAL_INPUT`;
- factorial localization (commit `9850fff`): executed historical negative / ablation; the 32 bank-macro cells may be shown; G5 FAIL stands;
- fresh four-arm training (commit `c591701`): all 12 cells completed 20,000 steps; final checkpoints, training records and provenance are archived in [`artifacts/formal_factorial_c591701_20260902T2040`](artifacts/formal_factorial_c591701_20260902T2040); localization and downstream claims are not complete;
- SOTA or success claims: require same-data baseline meters (in-repo C1 adapters are wired, not yet run);
- fixture outputs: permanently `scientific_use=FORBIDDEN`.

Do not write that no results exist. Historical descriptive factorial meters are documented in [`docs/results/实验数据.md`](docs/results/实验数据.md). The fresh `c591701` training archive is `CANDIDATE_NOT_CLAIM`, with `sota_ready=false`; its training losses are not localization meters.

Start with the [server README](code/CSI-PAIRS-v2.0-server/README.md) and verify the package before use:

```bash
cd code/CSI-PAIRS-v2.0-server
sha256sum --check SHA256SUMS
python3 -m unittest discover -s formal_v2/tests -v
```

Historical V1 code, environments and unrelated local experiment runs remain excluded. The explicit fresh-training snapshot above preserves the completed training artifacts and their provenance.

The current tree includes an authenticated official-code adaptation of Wi-GATr under
`code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters`. It is implemented but not yet formally trained or evaluated; C1
remains blocked until Wi-GATr and at least one second map-conditioned model execute on the same
non-fixture six-condition unit registry and both satisfy the frozen C1 eligibility and gate rules.

The tree also ships executable, source-authenticated shuffled-pair, retained-representation,
scene-ID, equal-FLOP, parameter-matched concat, FLOP-matched concat, and generous 2x concat
controls. Their presence and fixture smoke runs establish software reachability only; no control
can support a claim until its non-fixture prerequisites and downstream gates pass.
