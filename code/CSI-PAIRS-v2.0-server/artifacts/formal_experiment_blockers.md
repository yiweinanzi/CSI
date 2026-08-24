# Formal experiment blockers

Date: 2026-08-09 (Asia/Shanghai)

Amended 2026-08-24: G0 literature receipts/PDF hashes/decision consistency,
independent data verification, C11 RT calibration, G8, and a
measurement-backed compute plan are **not launch blockers**. Missing or
failing those rows leave the matching claims `NOT_ASSESSED`/`BLOCKED`.
They do not set a hard `FORMAL_GO=NO-GO` stop on `prepare-full-run` or
`all`. This file remains a scientific-gap inventory, not a runner
kill-switch.

`PROTOCOL_READY=PASS`, `M4_DATA_PRODUCTION_READY=REPORTED`,
`EVIDENCE_REGISTRY_READY=YES`, and
`FORMAL_CANDIDATE_READY=EXTERNAL_DEEP_VERIFICATION_REQUIRED`.
`SCIENTIFIC_EVIDENCE=NOT_ASSESSED` remains the honest status for skipped
optional gates. The runner does not invent scientific PASS.

| ID | Severity/type | Missing decision or input | Acceptance check | Consequence |
|---|---|---|---|---|
| `INPUT-DATA-001` | P0 `EXTERNAL_DEEP_VERIFICATION_AND_LIVE_REGEN_REQUIRED` | The repository contains two immutable registries but omits both candidates and their original manifests. The older LLVM 18 candidate predates approval enforcement. The newer LLVM 22 registry reports an exact run, but its external bytes, origin bundle, and runtime evidence are absent; portable replay is content-integrity diagnosis only. | Deep mode authenticates every external root and required array. A fresh live independent regeneration binds the current source and preapproved runtime and passes all scene/role gates at zero tolerance. | Neither reported candidate may enter G1/G2 yet. |
| `INPUT-RT-001` | P0 `EXTERNAL_DATA_REQUIRED` | Independent RT calibration fit/validation/reference manifests, shared reference evidence, and raw rows. Candidate-data regeneration does not substitute for separate calibration partitions. | `run-rt-calibration` plus G1 four-statistic/noise-floor checks pass. | C11 and qualification remain blocked. |
| `MODEL-C1-001` | P0 `COMPUTE_REQUIRED` | PMNet and Wi-GATr formal non-fixture checkpoints and authenticated six-condition executions are unavailable. | External-baseline V3 passes in every city for both C1-eligible models. | C1 remains blocked. |
| `INPUT-G8-001` | P0 `EXTERNAL_DATA_REQUIRED` | Licensed independent-engine scenes or a controlled real paired intervention with registered active/null units. | G8 V4 re-probes an actually independent runtime and passes cluster intervals. | C12 and external-validity wording remain blocked. |
| `RESOURCE-001` | P0 `LICENSE_OR_ACCESS_REQUIRED` | Four nonredistributable papers must be fetched from registered URLs; selected assets/checkpoints need permission records. | `fetch_waibu_resources`, `verify-waibu-resources`, and license acknowledgements pass. | G0/full preflight remains blocked. |
| `COMPUTE-001` | P0 `COMPUTE_REQUIRED` | The final branch is not installed and preflighted on the destination two-A100 host with approved disk, wall-time, and GPU-hour budgets. | Follow `formal_v2/A100_RUNBOOK.md`; compute-plan preflight passes actual memory, driver, disk, and budget checks. | Formal training is not authorized. |
| `RESULTS-001` | P0 `EXTERNAL_DATA_REQUIRED` | Authenticated non-fixture Response qualification, four-arm, two-city, controls, and external runs. | Same-run gate chain and claim assembly pass; per-unit rows populate planned cells. | Scientific claims and submission-ready result panels remain absent. |

## Registered conditional data findings

| ID | Registered at | Evidence and boundary |
|---|---|---|
| `INPUT-DATA-001-HISTORICAL` | `artifacts/m4_formal_candidate_v2/` | Records hashes and sizes for candidate `e5ec3d32...7847`. Static mode reports `external_artifacts=NOT_VERIFIED`; the generator predates approval enforcement. |
| `INPUT-DATA-001-LATEST` | `artifacts/m4_llvm22_candidate_v1/` | Records a reported LLVM 22 candidate `e8903430...d2e`, receipt, and 34-scene inventory. The original external artifacts are not committed. Portable replay verifies registered content, not execution origin, and is `DIAGNOSTIC_NOT_CLAIM`. |

No candidate-data finding is closed by registry ingestion or portable replay.

## Closed author protocol decisions

| ID | Decision | Existing implementation evidence |
|---|---|---|
| `SC-GAUGE-001` | `A`: freeze `shared_complex_reference` for the formal study. | `formal_dataset.py`, `DATA_CONTRACT.md`, phase-gauge tests, and `paper_v2/main.tex` agree. |
| `SC-ROUTE-002` | `R1`: native full-channel Response remains on `r^A`; unified patch probe remains on `r^{R,q}`. | `formal_routing.py`, qualification/evaluation tests, formal config, and `paper_v2/main.tex` agree. |

## Closed code-controlled blockers

| ID | Closed at | Evidence |
|---|---|---|
| `TRACE-001` | current delivery head | The 2,124-row matrix rebuilds from the unique authority SHA and source CI compares every generated row with the tracked artifact. |
| `LOCK-001` | `6b0ccdcfb4f03909ed377151ff906d9f0c162adf` | CPython 3.12 platform locks bind reviewed wheels; evidence contexts validate installed bytes and reject untracked distributions, files, hooks, symlinks, and bytecode. |

## Permitted work before closure

- `SMOKE_GO=GO`: CPU fixture, static checks, packaging, and paper compilation only.
- `PILOT_GO=CONDITIONAL-GO`: disposable fixture/qualification diagnostics, no GPU, at most 30 minutes wall time and 5 GiB output per run. Stop on any nonzero exit or evidence mismatch. Every artifact remains `scientific_use=FORBIDDEN`.
- Optional scientific rows (G0/C13 receipts and hashes, data verification, C11, G8, measured compute plan) no longer hard-stop launch. Start the two-phase path when a loadable NPZ and G1/G2 plus LLM-as-judge approval are available; skipped rows stay `NOT_ASSESSED`.
