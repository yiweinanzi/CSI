# Current implementation verification

Date: 2026-08-08 (Asia/Shanghai)

- `BASE_SHA`: `bf5764afb52cc5c29fd41f230b66f9d869dfb8cb`
- Last completed exact-head CI before this delivery: `853a369447618a11c3384342ab38f8f53734dcb0`
- Branch: `codex/freeze-a-r1-protocols`
- Previous repair PR: `https://github.com/yiweinanzi/CSI/pull/4` (merged)
- Environment: CPython 3.12.10, locked packages, macOS arm64, CPU-only

## Code verification at the last completed exact head

| Check | Result |
|---|---|
| Full unittest discovery | `293/293 PASS` in 453.520 s |
| Python compilation | PASS for core, adapters and tests |
| Ruff 0.16.2 `E9,F` | PASS in an isolated audit-tool environment |
| Dependency health | `pip check` PASS |
| CLI help | `25/25 PASS` (top level plus 24 subcommands) |
| Strict config load | formal and smoke both `csi-pairs-formal-config-v2.3-v6` |
| Shell syntax | 8 scripts PASS |
| Wi-GATr vendor inventory | PASS |
| `git diff --check` | PASS |

Expected `error:` lines in the unittest log are failure-injection assertions. They do not denote
failed tests.

## Repairs covered by this record

- per-role source-city and global support/query isolation;
- dynamic per-city base-map-cluster qualification;
- physical-only primary routing and independent teacher strata;
- per-city Response, null and C1 decisions;
- exact action displacement and complete Holm families;
- safe, unique adapter identifiers and fail-closed resource exits;
- two-phase nonce/hash-bound llm-judge approval and full static compute preflight;
- main, Wi-GATr and Sionna runtime provenance, exact versions and RECORD authentication;
- complete stage inventories and same-run qualification/factorial/evaluation/control bindings;
- C8 G3/G4/G5 chain authentication and runtime re-probes;
- nonredistributable-resource exclusion and deterministic anonymous/internal deliveries;
- anonymous identity/Git/path scanner and CPU CI.
- anonymous export omits internal requirement-matrix tooling and proves the exported public test
  suite runs to completion after fresh extraction.
- bundle verification accepts only the fully authenticated expected fixture failure and never
  reclassifies it as a scientific PASS;
- pull-request merge-ref anonymity scanning ignores only GitHub's generic automation identity while
  retaining project author, committer, email, repository, SHA and personal-path checks.
- hosted Linux CPU nested integration bounds allow 300 seconds for the extracted public suite and
  complete fixture dry-run; assertions and executed coverage are unchanged, and both targeted
  regressions plus the complete source suite pass.
- GitHub Actions checkout and Python setup are pinned to the current official Node 24 release
  commits rather than floating legacy majors that emit a Node 20 deprecation annotation.
- formal installation is wheel-only and hash-locked for the supported macOS arm64 and Linux x86_64
  targets; runtime evidence authenticates the pip receipt, wheel/RECORD/file hashes, platform floor,
  source tree, CUDA inventory, and deterministic Torch state.
- all normative V6 rows resolve through an explicit semantic family and clause SHA without a
  section-level fallback; formal results and external/author boundaries remain unpromoted.

## Final local delivery checks

| Check | Result |
|---|---|
| Server bundle reproducibility | Two byte-identical builds and both sidecars pass in exact-head CI; temporary bundle digest is not retained by the workflow log |
| Anonymous bundle reproducibility | Two byte-identical builds and both sidecars pass in exact-head CI; temporary bundle digest is not retained by the workflow log |
| Bundle sidecars | All four generated sidecars verify |
| Fresh server extraction | Extracted server verifier passes in exact-head CI |
| Fresh anonymous extraction | Extracted anonymous full-suite replay and pre/post anonymity scans pass in exact-head CI |
| Anonymous exclusions | No internal matrix builder, internal audit test, audit artifacts, PDF, `.DS_Store`, bytecode or identity/path finding |
| Extracted server dry run | Outer verifier PASS after authenticating inner exit `1`, `passed=false`, `DRY_RUN_FAIL_NOT_EVIDENCE`, and fixture `scientific_use=FORBIDDEN` |
| Reproducible paper | Two byte-identical builds and tracked PDF share SHA-256 `35117a4a30261f7d9c04cdeedcf4edb0634722354509dc9b92da2f3d5acf2f3e` |
| ICLR preflight | Zero findings on the clean build directory containing the official style and build log |
| PDF visual/metadata review | 10/10 pages rendered; main text ends on page 8, references and appendix start on page 9; no clipping, overlap, identity metadata or author-bearing link |

The appendix's `PLANNED` cells are registered result-schema sentinels, not populated values or
claimed results. The startup guide, paper README and atomic matrix prohibit replacing them with
fixture, smoke, unit-test or expected values.

## Readiness verdict after remote-head replay

| Layer | Verdict | Boundary |
|---|---|---|
| `PACKAGE_INTEGRITY` | `PASS` | Deterministic local builds, sidecars and fresh extractions pass. |
| `SOFTWARE_READY` | `PASS` | No open reproduced code-level P0/P1 remains; clean-clone replay and hosted Linux CPU CI pass at `853a369447618a11c3384342ab38f8f53734dcb0`. |
| `V6_PROTOCOL_FIDELITY` | `PASS` | A freezes the shared complex reference and R1 preserves the two registered Response estimands. |
| `PAPER_PROTOCOL_READY` | `PASS` | Code, configuration, data contract, tests and paper already agree with A + R1. |
| `FORMAL_INPUT_READY` | `BLOCKED` | Formal data, RT/reference evidence, executed Wi-GATr/PMNet checkpoints, remaining licenses and authorized CUDA compute are absent. |
| `LAUNCH_READY` | `BLOCKED` | Software readiness alone cannot authorize formal execution. |
| `ANONYMOUS_RELEASE_READY` | `PASS` | Mechanical anonymous packaging passes; no scientific evidence is implied. |
| `SCIENTIFIC_EVIDENCE` | `NOT_ASSESSED` | No authenticated non-fixture result exists and no formal training was run. |

`SMOKE_GO=GO`, `PILOT_GO=CONDITIONAL-GO`, `FORMAL_GO=NO-GO`, and
`PAPER_PROTOCOL_GO=GO`. Remote PR-head clean-clone verification and hosted CI completed at the
last experiment-code-bearing head. Later ledger and CI-maintenance close commits do not alter
executable or paper content.
