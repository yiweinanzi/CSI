# CSI-PAIRS

This repository contains CSI-PAIRS training, evaluation, baselines and paper-table export.
The active experiment workflow has been simplified; historical V6 certification workflows are retired.

The [2026-09-09 source publication index](docs/results/source-publication-20260909.md)
links the current execution repair, source-only research code, and separate historical
source archives. This source upload does not change experiment or SOTA conclusions.

The [six-table implementation status](docs/review/2026-09-09-paper-completion.md)
describes the simplified executable pipeline and what still needs server validation.

```bash
cd code/CSI-PAIRS-v2.0-server
bash formal_v2/scripts/run_paper.sh /path/to/formal.npz /path/to/new-run --device cuda:0
```

This runs source training, shared few-shot localization, independent probes, resource
and shuffled-pair controls, five map methods, risk calibration, and all six CSV/Markdown/LaTeX tables.
It uses the server's PyTorch environment and prepares the separate official Wi-GATr environment.
There are no LLM approvals, installation receipts or Python-environment closure checks.
Interrupted jobs can resume, missing data stays missing, and negative results are exported normally.
See the [server README](code/CSI-PAIRS-v2.0-server/README.md) for commands and protocol changes.

The original historical protocol is
[`Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md`](Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md).
The executable server package is under
[`code/CSI-PAIRS-v2.0-server`](code/CSI-PAIRS-v2.0-server); the directory name is retained as a compatibility path, while the runtime, schemas, paper draft, and generated bundle are V2.1.

Current status:

- package/software: current simplified workflow; see the cleanup report for validation;
- protocol implementation: new source-only loss normalization, shared standardized localization, minibatch probes/readout, and explicitly documented control adaptations;
- experiment readiness: six-table code paths implemented and locally tested; formal-data and full GPU validation remain server work;
- factorial localization (commit `9850fff`): executed historical negative / ablation; the 32 bank-macro cells may be shown; G5 FAIL stands;
- fresh four-arm training (commit `c591701`): all 12 cells completed 20,000 steps; final checkpoints, training records and provenance are archived in [`artifacts/formal_factorial_c591701_20260902T2040`](artifacts/formal_factorial_c591701_20260902T2040); the later saved localization audit reports Full worse than Response-only in all eight city/budget cells, while external comparisons and downstream claims remain incomplete;
- SOTA or success claims: require same-data baseline meters (in-repo C1 adapters are wired, not yet run);
- fixture outputs: permanently `scientific_use=FORBIDDEN`.

Do not write that no results exist. Historical descriptive factorial meters are documented in [`docs/results/实验数据.md`](docs/results/实验数据.md). The fresh `c591701` training archive is `CANDIDATE_NOT_CLAIM`, with `sota_ready=false`; its training losses are not localization meters.

The [server README](code/CSI-PAIRS-v2.0-server/README.md) describes the supported entry points.
The [code cleanup report](docs/review/2026-09-09-code-cleanup.md) records the removed workflows.
Historical certification code and its checksum inventory can be recovered from commit
`1121a0e045d147c56dbf9e513a32244ffc8ff358`; that inventory does not certify the current tree.

Historical V1 code, environments and unrelated local experiment runs remain excluded. The explicit fresh-training snapshot above preserves the completed training artifacts and their provenance.

The current tree includes an authenticated official-code adaptation of Wi-GATr under
`code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters`. It is implemented but not yet formally trained or evaluated; C1
remains blocked until Wi-GATr and at least one second map-conditioned model execute on the same
non-fixture six-condition unit registry and both satisfy the frozen C1 eligibility and gate rules.

The tree also ships executable, source-authenticated shuffled-pair, retained-representation,
scene-ID, equal-FLOP, parameter-matched concat, FLOP-matched concat, and generous 2x concat
controls. Their presence and fixture smoke runs establish software reachability only; no control
can support a claim until its non-fixture prerequisites and downstream gates pass.
