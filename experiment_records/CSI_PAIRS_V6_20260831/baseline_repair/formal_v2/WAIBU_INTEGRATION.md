# V6 external-resource integration

This document maps every registry entry to executable V6 behavior. The authenticated registry is
`configs/waibu_resources_v1.json`. SHA-256 authentication proves local bytes, not paper fidelity or
scientific support.

The registry separately records `source_url`, `license_url`, and `redistribution_allowed`.
`2502.11965v2.pdf`, `2505.09160v2.pdf`, `2601.03789v1.pdf`, and `2604.07086v1.pdf` use the arXiv
non-exclusive distribution license and are local research inputs only. They are excluded from Git,
the internal delivery bundle, and the anonymous supplementary package; each user must obtain them
directly from the recorded source URL.

| Resource | Project role | Executable implementation | Allowed evidence label |
|---|---|---|---|
| `2406.14995v2.pdf`, `Wi-GATr-main.zip` | map-conditioned forward baseline | official tokenizer/GATr plus CSI-PAIRS mesh and inverse localization | Wi-GATr official-code adaptation |
| `PMNet-a0e0c592.zip` | pathloss-radiomap baseline | official PMNet v3 backbone plus CSI-PAIRS map/Tx raster, source-only masked radiomap training, and inverse localization | PMNet official-code adaptation |
| `2502.11965v2.pdf` | CIR/CSI consistency baseline | ResNet-50 CSI/CIR encoders and symmetric contrastive loss; ViT style extension | CSI-CLIP-style / CSI-CLIP++-style controlled implementation |
| `2505.09160v2.pdf` | reconstruction+contrastive baseline | reconstruction-only WiMAE warm-start followed by 2D MAE/independent noisy masked-view contrastive training | ContraWiMAE paper-spec controlled implementation |
| `2601.03789v1.pdf` | masked CSI baseline and Stage-0 neighbor | 75% per-sample random 2D patches, fixed 2D sine-cosine position, asymmetric ViT encoder/narrow decoder | CSI-MAE paper-spec controlled implementation |
| `2603.25216v1.pdf` | multimodal same-world baseline | CSI/map/context modality experts, online/EMA target JEPA, latent L1 | WWM-inspired same-world matched prediction |
| `2604.07086v1.pdf` | editable RF forward baseline | 2.5D anisotropic Gaussian primitives, occupancy visibility, alpha transmittance, learned RF material response | RFIR-inspired controlled implementation |
| `2606.04770v1.pdf` | geometry-grounded multi-view reference | local 2D map/CSI Transformer diagnostic; it omits the paper's sparse 3D TRELLIS model, dense receiver-plane radiomap objective, physical unordered path targets, and checkpoint warm starts | WiSER style-controlled implementation; C1-ineligible |
| `sionna-main.zip` | RT provenance package | authenticated local Sionna 2.0.1 source metadata plus official `sionna-rt` dependency; unrelated PHY/PyTorch CUDA stack is outside G8 | Sionna official-source RT facility |
| `sionna-large-radio-maps-main.zip` | tiling, scene build, large radio maps | official scripts and strict output audit | Sionna large-radio-map official facility |

SigMap is required by frozen V6 but has no paper/source file in `waibu/`. Its implementation is
therefore named only `style-controlled-implementation`; it cannot be upgraded by documentation.

`waibu/` contains no implementation for the shuffled-pair, retention, or resource controls because
those are CSI-PAIRS protocol controls rather than external prior work. Their first-party
implementations and authenticated manifests are shipped under `external_adapters/`. G4, C4, and C6
remain scientifically blocked until the controls run on qualified non-fixture data and their
upstream dependencies pass; resource authentication or source presence alone cannot promote them.

## Commands

Fetch omitted resources directly from their registered sources, then authenticate the complete
formal input set:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.fetch_waibu_resources \
  --registry formal_v2/configs/waibu_resources_v1.json \
  --waibu-root waibu
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli verify-waibu-resources \
  --registry formal_v2/configs/waibu_resources_v1.json \
  --waibu-root waibu \
  --output runs/waibu-auth-001
```

Run the five representation baselines after role-wise data verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli run-representation-baselines \
  --config formal_v2/configs/formal_v2.json \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --representation-baseline-config formal_v2/configs/representation_baselines_v1.json \
  --output /absolute/path/formal-run
```

Run the common C1 unit registry:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli run-external-baselines \
  --config formal_v2/configs/formal_v2.json \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --adapter-manifest formal_v2/external_adapters/all_map_adapters_v1.json \
  --output /absolute/path/formal-run
```

Run the shipped CSI-PAIRS protocol controls after factorial/evaluation artifacts exist:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli run-resource-controls \
  --config formal_v2/configs/formal_v2.json \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output /absolute/path/formal-run

PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli run-shuffled-pair-control \
  --config formal_v2/configs/formal_v2.json \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output /absolute/path/formal-run

PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli run-retention-audit \
  --config formal_v2/configs/formal_v2.json \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output /absolute/path/formal-run
```

Install and inspect Sionna facilities:

```bash
formal_v2/external_adapters/setup_sionna.sh
PYTHONDONTWRITEBYTECODE=1 formal_v2/external_adapters/.runtime-sionna/venv/bin/python -m formal_v2.sionna_facility \
  --runtime-root formal_v2/external_adapters/.runtime-sionna verify

PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_cli export-sionna-scenes \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output /absolute/path/formal-run/inputs \
  --license-id YOUR_SCENE_ASSET_LICENSE \
  --carrier-frequency-hz 3500000000 \
  --subcarrier-spacing-hz 30000
```

## Evidence boundary

- Representation-baseline PASS means source-only training, checkpoint selection, unified probe,
  and target query evaluation completed. It contributes comparison rows but never C1.
- External-baseline PASS requires at least two C1-eligible map-conditioned models over the exact same
  internally generated units. Every condition must report map/action digests that the outer runner
  recomputes, and each eligible model must pass cluster-macro active-effect and null-equivalence gates.
  The shipped manifest contains Wi-GATr and PMNet in that class. C1 remains blocked until both
  execute on qualified non-fixture data and pass every evaluation city's active/null gate.
- G8 requires actual independent Sionna scene XML/assets, hashes, licenses, and retraced CFRs.
- Shipped first-party control code closes the implementation gap, not the formal-execution gap.
- Formal profile execution on real, independently regenerated data is required for scientific use.
- Fixture, smoke profile, unavailable dependency, missing asset, or hash mismatch remains permanently
  non-scientific/fail-closed downstream.
