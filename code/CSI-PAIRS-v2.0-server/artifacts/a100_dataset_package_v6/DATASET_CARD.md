# CSI-PAIRS A100 data suite V6 dataset card

## Plain-language summary

This delivery is a toolbox containing several different kinds of wireless
data. It is not one large dataset whose rows may be mixed together.

The main CSI-PAIRS research object is a paired-world dataset. For each physical
bank, it keeps the same transmitter, receiver positions, radio settings, and
base geometry while crossing at least two reversible interventions to make a
complete sibling-world hypercube. The independent experimental unit is the
bank, not a receiver row, edge, repeat, seed, or image patch.

The delivery also contains public datasets useful for comparison. Those public
datasets do not naturally have the same paired interventions or repeat
semantics, so mechanically concatenating them with CSI-PAIRS would change the
task and invalidate the paper design.

## Component table

| Component | What it is | Current status | Allowed | Forbidden |
| --- | --- | --- | --- | --- |
| Six-city OSM input | Frozen source responses for new scene generation | Authoritative raw input | A100 regeneration | Claiming it is CSI data |
| M4 CPU LLVM22 34-bank reference | 34 banks, 4 worlds, 256 positions, 16 CSI channels, 3 repeats | `CANDIDATE_NOT_CLAIM`; local bytes and 34/34 zero-tolerance replay exist, but destination deep verification and fresh live regeneration remain required | Diagnostic replay and comparison | Formal training or paper results |
| Dual-A100 Sionna data | 36-scene real GPU ray-tracing output on two A100s | `fixture=true`, `scientific_use=FORBIDDEN` | Loader, sharding, and GPU smoke tests | Empirical result tables or formal qualification |
| DeepMIMO | Public synthetic channel scenarios `asu_campus_3p5` and `i1_2p5` | Public core complete | Named baseline or adapter | Silent Stage-0 teacher or sibling-world main data |
| UrbanMIMOMap | Public map-0 set with 120 NPZ files | One complete public map, not the full 350-map collection | Single-map grouped baseline | Claiming full-dataset coverage or independent-map scale |
| RadioMapSeer | Main, Loc, ToA, RadioMap3DSeer, and RadioUNet assets | Public core complete | Radio-map, localization, ToA, and geometry baselines | Calling RadioMap3DSeer the paid `IRT2HighRes` archive |
| WWM / DeepSense | WWM public metadata plus DeepSense Scenarios 8 and 33 | Original WWM files unavailable; public substitute complete | Explicit DeepSense multimodal baseline | Calling DeepSense original WWM or reproducing WWM |

## Main paired-world shape

The CPU reference has 34 independent banks across six cities:

| Role | City | Banks |
| --- | --- | ---: |
| Source | Austin | 7 |
| Source | Chicago | 7 |
| Target | Boston | 8 |
| Target | Seattle | 8 |
| External validation | Denver | 2 |
| External validation | Miami | 2 |

Each bank contains four sibling worlds (`d=2`, `K=2^d=4`), 256 common
receiver positions, 16 real-then-imag CSI channels, and three observation
repeats. Repeats estimate observation variability; they are not independent
physical scenes. Boston and Seattle are the two target-city partitions inside
the same dataset.

## Provenance and licenses

- OpenStreetMap source responses: ODbL-1.0; retain OpenStreetMap contributor
  attribution and review derived-database obligations before public release.
- NVIDIA Sionna and Sionna RT engine source: Apache-2.0.
- Each external public source retains its own upstream terms. The package is an
  internal research handoff, not a declaration that all payloads may be
  republished as an anonymous supplement.
- Qualcomm Wi3R/WiPTR is not included because local provenance and
  redistribution permission remain unresolved.
- Original China Mobile WWM files and checkpoints are not included because
  they are not public. DeepSense is recorded as a non-equivalent substitute.

## Split and leakage rules

- The exact 34-bank assignment is frozen in
  `docs/FORMAL_MAIN_SPLIT_LEDGER.json`; the same rules are explained in Chinese
  in `docs/SPLIT_GUIDE.zh-CN.md`.
- CSI-PAIRS evaluation and uncertainty use bank/tile as the independent unit.
- Source roles remain mutually exclusive; target support/query is separated by
  receiver position. Label budgets `8/32/128` mean positions, not rows.
- Austin and Chicago bank indices `00..06` map respectively to encoder train,
  method selection, probe train, probe selection, calibration fit, calibration
  selection, and final unseen-bank test. Boston and Seattle use even position
  indices for support and odd indices for frozen queries.
- DeepSense must be grouped by acquisition sequence; row-wise random splitting
  can leak adjacent frames.
- RadioMapSeer is split by city map, DeepMIMO by scenario, and the current
  UrbanMIMOMap subset only within map 0 by grouped transmitter/configuration.
- External data may be used only through a named, preregistered baseline or
  ablation. It must not influence headline route selection, normalization, or
  Stage-0 teacher training unless that use is explicitly registered and the
  comparison remains fair.

## What package verification proves

The default verifier checks the exact file inventory and sizes, fixed status
boundaries, required raw inputs, absent forbidden payloads, public-core source
identities, and ZIP/NPZ central-directory readability. The 133 public-source
SHA-256 values are retained in the manifest and can be recomputed with the
optional `--deep-hash` mode. It also ensures `regenerated.npz` is absent so the
A100 host must create a fresh output.

These checks prove engineering integrity only. They do not establish measured
channel realism, independent-engine external validity, causal identification,
absence of benchmark leakage, trained-model performance, or acceptance-ready
paper evidence.

## Promotion gates

Formal training remains prohibited until all of the following are true:

1. Linux x86_64 libLLVM bytes and provenance are reviewed and committed before
   candidate generation.
2. The A100 host generates a fresh output from the frozen OSM inputs.
3. Fresh `inspect-data` and all-role `verify-data` pass 34/34 banks at
   `rtol=0`, `atol=0` in unused output directories.
4. Independent RT calibration/reference partitions and genuinely independent
   G8 evidence are available.
5. G1/G2, resource/license, two-A100 compute, and adapter gates pass.
6. `qualification/gate.json` says `QUALIFICATION_PASS` and
   `scientific_use=FORMAL_EXPERIMENT_ALLOWED`.
7. An allowed LLM-as-judge (`codex`, `claude-code`, or `cursor`) reviews and explicitly approves that exact qualified run.
