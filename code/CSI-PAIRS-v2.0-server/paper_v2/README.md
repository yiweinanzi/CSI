# CSI-PAIRS V2.1 V6 paper draft

This directory is independent of the frozen V1.26 paper. `main.tex` preserves the evidence-gated manuscript and includes protocol figures for the six-condition audit, model information flow, and fail-closed G0--G8 sequence. It states all five frozen research questions and uses the current G0--G8 meanings.

## What may appear in the draft

The executed factorial localization table may be shown as a descriptive negative result or ablation. The 32 bank-macro cells (4 arms × 2 cities × 4 budgets) come from `docs/results/实验数据.md` (2026-08-31), bound to commit `9850fffe0b34f16b45066973308f18b10555ca5d` and dataset SHA-256 `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`. The headline numbers are meters; utility is secondary. G5 remains FAIL (1/4). Do not relabel that run as PASS or as SOTA.

A SOTA or success claim still requires same-data baseline meters. In-repo C1-eligible adapters (Wi-GATr, PMNet) are wired and not yet run; those columns say “wired, not run,” never “not authorized.” SigMap, WiSER, RFIR, and representation baselines stay in a restricted descriptive column, not the strict C1 SOTA column. Do not copy foreign-paper meters (including 1.56 m / 60 cm) into the tables.

Do not write that “no results exist.” Evaluation CGS/NMSE, risk, path, and same-data baselines have not been run; that is “wired, not run,” not an empty science record.

## Release artifacts this clone does not have

This Windows clone has no `runs/` and no `.npz`. A result-bearing release bundle must include, with recorded SHA-256:

- `localization_summary.csv` (`c3db830a419136a359bcc2fc8143b3c61d352c2bd2486264778101d9711fcaa0`)
- `factorial_statistics.json` (`6dd2267170b4494b98f0974a14742e93c560f9e923c17a7a6acf84fb29afb918`)
- `factorial/gate.json` (`062cdfd1b4e05fcf0298ec83680c519ca84539573f5c4c52f10c6e7aff733de6`)
- `qualification/gate.json` (`ed425fc44be933449742ae565591b31525f409ef681e246867f254c6c48eca5c`)
- `training_summary.csv` (`83dd9c7ff480834b5321711b53550f700654dac0fb3179dd0e8c02293c4414d7`)

Do not add NPZ to this clone to satisfy that list.

## C13 / SUMS

Paper claim SHA-256 of the current `main.tex` (working-tree bytes):

`c6d899caba2865f1fa82c39857d2305ea262f19489426ed9355094335d5f96cb`

This digest is also written in repo-root `LLM_JUDGE_C13.md` and `code/CSI-PAIRS-v2.0-server/SHA256SUMS`. Recompute both if the tex changes again. The previous paper digest `9a0d4aeb…` is obsolete.

Do not populate a remaining `\wired` or `\planned` cell with fixture output, an expected value, or a foreign-paper meter. A formal comparison value must trace to a non-fixture per-sample row, frozen config hash, checkpoint hash, independent bank count, training seeds, label draws where applicable, interval, and pass threshold.

Build into an unused directory:

```bash
CSI_PAIRS_LATEXMK=/path/to/latexmk paper_v2/build_reproducible.sh /unused/build-directory
```

Page count will change after the 32-cell appendix is instantiated. Recheck the ICLR initial-submission nine-page main-text limit after the next authenticated comparison tables are filled.
