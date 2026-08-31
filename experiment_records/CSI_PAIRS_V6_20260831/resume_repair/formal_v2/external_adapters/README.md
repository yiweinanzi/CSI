# External Baseline Adapters

These adapters implement the frozen V6 Section 1.1 six-condition audit. An adapter is
scientific evidence only after its source, command, checkpoint, result file, dataset, config,
and stage manifest hashes all authenticate. C1 additionally needs two eligible models whose
outer-recomputed map/action inputs and cluster-level active/null decisions pass.

## Frozen adapter set

`all_map_adapters_v1.json` contains five distinct map-conditioned methods:

| Model | Implementation | Native training target | Six-condition localization |
|---|---|---|---|
| SigMap | V6 style-controlled map+CSI locator | source position | direct map-conditioned estimate |
| Wi-GATr | official-code adaptation | total received power | inverse coordinate optimization |
| PMNet | official-code adaptation | total received-power radiomap | nearest power-matched free cell |
| WiSER | style-controlled implementation | local 2D map/CSI surrogate objectives | frozen surrogate inverse search |
| RFIR | RFIR-inspired 2.5D controlled implementation | visibility-aware anisotropic Gaussian RF field + received power | frozen inverse-renderer search |

All five train on `source_encoder_train`, select on `source_method_selection`, and evaluate only
`source_final_unseen_bank` plus target `query` positions. Every adapter emits a source-only training
record, selected checkpoint, exact config, result rows, and hashes. Missing environments resolve to
`not_executed`; they do not abort into a false PASS.

The formal controlled-model configs preserve their effective batches while using microbatch 16:
SigMap uses BF16 autocast, while WiSER and RFIR remain float32 because their complex CSI paths are
not eligible for implicit reduced precision. RFIR uses an elementwise deterministic prefix-product
implementation for alpha transmittance because CUDA `cumprod` is unavailable under the frozen
deterministic-algorithm policy. These choices and their accumulation counts are checkpoint-bound.

WiSER and RFIR have no source archive in `waibu/`. The local WiSER code uses a dense 2D pooled
Transformer, scalar power prediction, IFFT-derived pseudo taps, and random initialization. Those
choices do not reproduce the paper's sparse 3D TRELLIS scene representation, dense receiver-plane
radiomap loss, physical unordered path targets, or pretrained checkpoint schedule. WiSER therefore
remains `style-controlled-implementation` and C1-ineligible, as do RFIR and SigMap. Wi-GATr and
PMNet are C1-eligible; the C1 gate still requires both non-fixture executions to pass every city's
cluster-level active/null decisions.

## Wi-GATr

`wigatr_adapter.py` adapts the official Wi-GATr implementation from Hehn et al. (ICLR 2025,
arXiv:2406.14995v2). Its evidence label is `official-code-adaptation`, not `faithful reproduction`:
the tokenizer and GATr architecture are official, while CSI-PAIRS supplies a deterministic 2.5D
grid-to-mesh conversion and a different source-domain dataset.

The retained paper semantics are:

- one token per compact triangular surface face, with material one-hot scalars;
- Tx, Rx, and Tx-Rx link tokens;
- the official `kitchen_sink_z` geometric-algebra embedding;
- 32 GATr blocks, 16 hidden multivector channels, 32 hidden scalar channels, 8 heads, and
  multi-query attention;
- scalar total received-power regression with MSE, Adam at `1e-3`, batch 64, cosine decay,
  and the paper's 200,000-step WiPTR dose;
- frozen-model inverse localization by optimizing receiver coordinates against observed power.

CSI-PAIRS-specific deterministic adaptations are:

- occupied 2.5D cells are extruded to their height; exactly coplanar, same-material top and side
  cells are merged into rectangles and triangulated. The full formal map bank must pass the
  independent directed-plane/material surface-ledger equivalence check before execution;
- real-then-imag CSI is reduced to
  `10 log10(mean(real^2 + imag^2))`; the affine scale is fitted on
  `source_encoder_train` only;
- training uses `source_encoder_train`; checkpoint selection uses
  `source_method_selection`; evaluation uses only `source_final_unseen_bank` and target query
  positions;
- target localization initializes from the public map extent, never from target coordinates,
  fingerprints, target normalization, support positions, or labels;
- each eligible observation has the same frozen CSI/radio context under `correct`,
  `paired_active_alternative`, `paired_null_alternative`, `wrong_city`,
  `geometry_destroyed`, and `empty` maps.

The adapter rejects varying carrier/antenna configurations because the paper's regression model
assumes one fixed wireless configuration. It reports localization error only. It does not add a CSI
prediction head and cannot support a full-channel response claim.

### Environment

The official snapshot requires Python 3.10 and git-pinned GATr/WiInSim dependencies:

```bash
formal_v2/external_adapters/setup_wigatr.sh
```

The setup is intentionally separate from the core V2.1 environment. If the pinned dependencies
cannot be installed, the adapter fails and remains `not_executed`; no substitute model is used.
The official GATr/xFormers forward requires NVIDIA CUDA. A CPU-only host fails before training with
an explicit prerequisite error rather than entering an unsupported fallback.
Setup records the actual Python 3.10 interpreter, `uv.lock` and vendor-manifest digests, every
installed distribution and RECORD digest, Torch/CUDA/cuDNN/driver/GPU identity, and deterministic
settings. The V3 execution manifest includes the same record. The core runner independently probes
that interpreter after execution and rejects package, driver, lock, or environment substitution.

### Formal execution

1. Run the complete `all_map_adapters_v1.json`; its Wi-GATr command points to the default environment.
2. Run `formal_cli all` or `formal_cli run-external-baselines` after data verification,
   qualification, factorial training, and evaluation have produced their authenticated artifacts.

Fixture runs remain `scientific_use=FORBIDDEN`, even if the adapter exits successfully.

## PMNet

`pmnet_adapter.py` adapts the official PMNet v3 source at revision
`a0e0c5926de721074beeb23f630f2d313f6508dd`. The vendored model and MIT license are authenticated
under `vendor/PMNet/`; upstream model bytes are unchanged. Its evidence label is
`official-code-adaptation`, not a reproduction of the USC/UCLA/Boston results.

The adapter retains the official `[3,3,27,3]` encoder blocks, ASPP rates `[6,12,18]`, multi-grid
`[1,2,4]`, output stride 8, decoder, Adam optimizer, and 30-epoch StepLR schedule. The input boundary
is expanded from building/Tx images to occupancy, height, material one-hot planes, and the fixed BS
transmitter raster. It predicts a total received-power radiomap and computes MSE only at registered
  receiver cells. Power normalization is fitted on `source_encoder_train`; checkpoint selection uses
  only `source_method_selection`. The effective batch remains 16 while gradients are accumulated from
  deterministic BF16 microbatches of two to satisfy the official BatchNorm shape and bound peak
  memory. The checkpoint and training record bind configured and executed precision, microbatch size,
  and accumulation count. Non-fixture execution fails before writing evidence
unless CUDA exposes at least 8 GiB; the PMNet subprocess enables deterministic Torch/cuDNN settings,
disables TF32, and binds its independently rechecked main-runtime provenance into the manifest.

For each frozen six-condition map, PMNet predicts a radiomap without a receiver coordinate. The
localizer returns the free grid cell whose predicted power is closest to the observed total power;
the true receiver position is read only afterward to compute localization error. A registered
source snapshot or successful fixture forward does not support C1 without formal per-city results.

## Representation baselines

The separate `formal_representation_baselines.py` stage runs CSI-MAE, CSI-CLIP, CSI-CLIP++,
ContraWiMAE, and WWM-inspired same-world matched prediction. These methods use a common source
probe and target label-draw budget, but preserve their distinct pretraining losses. They are not
inserted into the C1 adapter list because CSI/CIR consistency and same-world prediction are not
six-map interventions.

## Sionna

`setup_sionna.sh` authenticates and extracts both supplied official archives, copies the checked-in
`sionna_lrm_uv.lock`, installs it with `uv sync --locked`, freezes Sionna RT 1.2.1, and installs the
official tiling/scene/radio-map scripts. A fixed CPU PyTorch is installed because the G8 adapter reloads the frozen
Stage-0 teacher to reproduce route assignments; the unrelated CUDA dependency set is excluded. `sionna_facility.py`
exposes those operations and audits each `rm_*.npz` output.

`sionna_external_validity.py` is an internal G8 engine adapter. It requires every external-validation
sibling world to have a scene XML, canonical-map hash, asset manifest, per-asset hash, and license.
It retraces CFRs with `PathSolver` and emits paired active/null effects. A source archive alone cannot
pass G8; actual scene assets and a formal non-fixture run are mandatory.
The adapter emits a runtime record bound into the G8 V4 gate. The outer runner independently probes
the exact Python 3.12 interpreter before execution and rejects any differing lock, package inventory,
RECORD digest, Torch state, or interpreter prefix.

The standard outer adapter is `../configs/sionna_external_validity_adapter_v2.json`. It expects the
scene manifest at `RUN_ROOT/inputs/sionna_scene_manifest.json`; this fixed location keeps the exact
command hash and scene bundle inside the formal run tree.

When the primary dataset was generated by Sionna, do not use this adapter for
G8. The direct precomputed-archive schema documented in `../A100_RUNBOOK.md`
is diagnostic-only: it authenticates raw CSI, the independent-engine scene
manifest, and the engine configuration for first-party statistical checks, but
formal preflight and claim reauthentication reject it. A genuinely independent
executable adapter or controlled real intervention is still required for G8.

## DiffeRT independent G8

`differt_external_validity.py` is the reviewed independent executable G8
adapter for a Sionna primary dataset. It pins DiffeRT `0.10.0` at commit
`673cc58ef61906b8ab0869dd206b3d032dbc01b2`, runs exhaustive LOS and
first-order reflection paths with the frozen ITU/Fresnel material model, and
exports the formal two-antenna, four-subcarrier real/imag CSI layout. It is
CPU-isolated with JAX x64 enabled and CUDA hidden.

Create the runtime with `setup_differt.sh`. Prepare blind external scene assets
with `prepare_differt_external_scenes.py`; that preparer reads geometry, radio,
positions, pose, primitive identity, and metadata fields but never reads primary
CSI arrays. Every sibling world receives a distinct authenticated NPZ source
asset. Formal execution accepts only schema
`csi-pairs-v6-external-validity-independent-adapter-v1`, stages all inputs into
the G8 output directory, probes the interpreter before execution, and compares
the emitted runtime record with that probe. Claims reauthentication reloads the
staged raw CSI, assets, config, adapter, and runtime and recomputes all G8
statistics. The archive schema remains diagnostic-only even when its numbers
pass.

`PYTHONDONTWRITEBYTECODE=1 python -m formal_v2.formal_cli export-sionna-scenes` deterministically converts every formal
`external_validation` sibling world into material-separated PLY meshes and Sionna scene XML, writes
per-asset license/hash records, and emits the complete scene manifest. The command does not invent
an asset license: `--license-id` must match `metadata.assets.license_ids`; carrier frequency and
subcarrier spacing are also mandatory inputs.

## First-party V6 claim controls

`shuffled_pair_control.py` trains fresh Alignment and Response controls from independently
deranged pairing registries. It never reuses a matched factorial checkpoint as the shuffled model,
and the outer gate authenticates both checkpoint families, every active pair, both permutation
directions, training provenance, and per-unit Alignment/Response metrics.

`retention_control.py` freezes the Full retained encoder, fits source-only probes for compatibility,
response, and localization, and evaluates correct-map, map-swap, and map-removed conditions. Probe
checkpoints bind the Full checkpoint and reject target-trained or target-selected probes.

`scene_id_sigmap.py` reuses the source-trained, source-selected SigMap checkpoint from the external
baseline stage. The built-in V3 manifest constructs city prompts from `source_encoder_train` only
and evaluates map/prompt parity plus map-swap/ID-swap displacement-direction agreement on
`source_final_unseen_bank`. It requires at least two source cities. For software smoke only,
`make-fixture --source-banks-per-role 2` supplies that reachability; fixture output remains
scientifically forbidden.

## First-party V6 resource controls

`resource_control.py` implements equal-FLOP Alignment, equal-FLOP Response, parameter-matched
independent concat, FLOP-matched independent concat, and generous 2x concat. The shipped
`resource_controls_v3.json` authenticates the runner and five architecture specs. Every configured
seed produces a control checkpoint, component and localization loss traces, a training log,
operator-level profiler evidence, localization rows, a resource index, and a replay artifact.

The frozen accounting scope is
`representation_training_and_retained_inference_excluding_common_localization_head`. The same
localization head is excluded on both sides. Concat bottleneck parameters and measured bottleneck
training/inference FLOPs are included. Parameter/FLOP match errors are reported and must remain
within the preregistered tolerance; thresholds are never relaxed for a smoke run.
