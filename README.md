# CSI-PAIRS

This repository contains the CSI-PAIRS V2.1 implementation of the frozen V6 research protocol.

The authoritative protocol is
[`Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md`](Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md).
The executable server package is under
[`code/CSI-PAIRS-v2.0-server`](code/CSI-PAIRS-v2.0-server); the directory name is retained as a compatibility path, while the runtime, schemas, paper draft, and generated bundle are V2.1.

Current status:

- package/software: verified on the audited code commit, with final bundle verification recorded in the traceability audit;
- protocol implementation: V2.1 frozen-V6 path;
- formal experiment readiness: `CODE_READY_FOR_FORMAL_INPUT`;
- factorial localization (commit `9850fff`): executed historical negative / ablation; the 32 bank-macro cells may be shown; G5 FAIL stands;
- SOTA or success claims: require same-data baseline meters (in-repo C1 adapters are wired, not yet run);
- fixture outputs: permanently `scientific_use=FORBIDDEN`.

Do not write that no results exist. Descriptive factorial meters are documented in [`docs/results/实验数据.md`](docs/results/实验数据.md). This Windows clone has no `runs/` and no `.npz`.

Start with the [server README](code/CSI-PAIRS-v2.0-server/README.md) and verify the package before use:

```bash
cd code/CSI-PAIRS-v2.0-server
sha256sum --check SHA256SUMS
python3 -m unittest discover -s formal_v2/tests -v
```

Historical V1 code, local experiment runs, external reference downloads, environments, and generated archives are intentionally excluded from the current-code repository.

The current tree includes an authenticated official-code adaptation of Wi-GATr under
`code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters`. It is implemented but not yet formally trained or evaluated; C1
remains blocked until Wi-GATr and at least one second map-conditioned model execute on the same
non-fixture six-condition unit registry and both satisfy the frozen C1 eligibility and gate rules.

The tree also ships executable, source-authenticated shuffled-pair, retained-representation,
scene-ID, equal-FLOP, parameter-matched concat, FLOP-matched concat, and generous 2x concat
controls. Their presence and fixture smoke runs establish software reachability only; no control
can support a claim until its non-fixture prerequisites and downstream gates pass.
