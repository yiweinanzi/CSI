# Experiment day one

Operator launch pack for a loadable formal NPZ. This is a command list,
not scientific evidence.

This Windows clone cannot execute the commands below as formal evidence.
Formal hosts are Linux x86_64 or macOS arm64 with CPython 3.12. Use
`operator-preflight` here only as an inventory.

G0 literature receipts/PDF hashes/decision consistency, independent data
verification, C11 RT calibration, G8, and a measurement-backed compute plan
are optional. Missing or failing them does not hard-stop `prepare-full-run`
or `all`. Those claims stay `NOT_ASSESSED`/`BLOCKED`. The runner does not
invent scientific PASS.

`formal_v2/configs/compute_plan_formal.template.json` is accepted as an
advisory budget. Zeros and `REPLACE_WITH_MEASURED_SHA256` no longer reject
launch.

## What this clone already has

- Protocol-aligned `formal_v2` package
- Unit tests in `formal_v2/tests/` (software contract only; not science)
- C1 adapters Wi-GATr + PMNet (`formal_v2/external_adapters/all_map_adapters_v1.json`)
- DiffeRT G8 adapter scripts (`setup_differt.sh`, `prepare_differt_external_scenes.py`, `differt_external_validity.py`)
- waibu fetch script `formal_v2/fetch_waibu_resources.py` (no separate literature fetch script)
- Two-phase CLI: `prepare-full-run` then `all`

## What is required to start

- A loadable NPZ on a formal host (Linux/macOS). Windows is not a formal host.
- `--adapter-manifest` plus the shipped control / scene-id / shuffled-pair /
  retention / representation defaults
- LLM-as-judge approval **outside** the run root (`codex`, `claude-code`, or
  `cursor`) before `all`

## Optional (do not block launch)

- Literature resource manifest (G0 / C13)
- RT calibration (C11). The adapter is not shipped.
- Independent data verifier manifest
- Independent G8 DiffeRT adapter
- Measurement-backed compute plan
- `CSI_PAIRS_DEVICES` / `CUDA_VISIBLE_DEVICES` for training, not as a
  compute-plan hard stop

Shipped adapter entry points under `formal_v2/external_adapters/`:

- `all_map_adapters_v1.json` (C1 / external baselines)
- `resource_controls_v3.json` (and the five `resource_*_v2.json` rows it binds)
- `shuffled_pair_control_v3.json`
- `retention_control_v3.json`
- scene-id default is `builtin:sigmap-scene-id-v1` (`scene_id_sigmap.py`)
- `wigatr_adapter_entry.json`

The unit-test fixture in
`formal_v2/tests/test_formal_v2.py::test_rt_calibration_stage_isolates_reference_and_binds_outputs_end_to_end`
is the **contract**, not a reusable shipped adapter. Do not copy it into a
formal run.

## Exact command sequence

Run from the extracted `CSI-PAIRS-v2.1-server` root on a formal host. This
Windows clone cannot produce formal evidence from these commands.

Optional inventory (any host):

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m formal_v2.formal_cli operator-preflight \
  --dataset /absolute/path/to/authenticated.npz \
  --config "$PWD/formal_v2/configs/formal_v2.json" \
  --output "$PWD/runs/operator-preflight-001"
```

### 1. Setup venvs that actually exist

```bash
formal_v2/scripts/setup_formal_v2.sh "$PWD/.venv"
formal_v2/external_adapters/setup_wigatr.sh
formal_v2/external_adapters/setup_sionna.sh
formal_v2/external_adapters/setup_differt.sh
```

### 2. `inspect-data`

```bash
export CSI_PAIRS_DEVICES=cuda:0,cuda:1
export CUDA_VISIBLE_DEVICES=0,1

PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B -m formal_v2.formal_cli inspect-data \
  --config "$PWD/formal_v2/configs/formal_v2.json" \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output "$PWD/runs/data-inspection-001"
```

### 3. waibu fetch (optional literature later)

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.fetch_waibu_resources \
  --registry "$PWD/formal_v2/configs/waibu_resources_v1.json" \
  --waibu-root "$PWD/waibu"

PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B -m formal_v2.formal_cli verify-waibu-resources \
  --registry "$PWD/formal_v2/configs/waibu_resources_v1.json" \
  --waibu-root "$PWD/waibu" \
  --output "$PWD/runs/resource-auth-001"
```

`--literature-resource-manifest` is optional. There is no literature fetch module.

### 4. DiffeRT scene prep (optional G8)

```bash
DIFFERT_RUNTIME="$PWD/formal_v2/external_adapters/.runtime-differt/venv/bin/python"
PYTHONPATH="$PWD" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES='' \
  "$DIFFERT_RUNTIME" -B \
  formal_v2/external_adapters/prepare_differt_external_scenes.py \
  --dataset "$CANDIDATE_ROOT/dataset.npz" \
  --asset-root "$CANDIDATE_ROOT/assets" \
  --engine-config "$PWD/formal_v2/configs/differt_external_engine_v1.json" \
  --minimum-external-clusters 32 \
  --output /new/unique/path/differt-g8-inputs
```

### 5. `prepare-full-run` (optional manifests omitted)

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B -m formal_v2.formal_cli prepare-full-run \
  --config "$PWD/formal_v2/configs/formal_v2.json" \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output "$PWD/runs/formal-001" \
  --adapter-manifest "$PWD/formal_v2/external_adapters/all_map_adapters_v1.json" \
  --control-manifest "$PWD/formal_v2/external_adapters/resource_controls_v3.json" \
  --scene-id-manifest builtin:sigmap-scene-id-v1 \
  --shuffled-pair-manifest "$PWD/formal_v2/external_adapters/shuffled_pair_control_v3.json" \
  --retention-manifest "$PWD/formal_v2/external_adapters/retention_control_v3.json" \
  --representation-baseline-config "$PWD/formal_v2/configs/representation_baselines_v1.json"
```

Add `--compute-plan`, `--verifier-manifest`, `--literature-resource-manifest`,
`--rt-calibration-manifest`, or `--external-validity-manifest` only when you
have them. They are not launch blockers.

The root `README.md` wrapper is `formal_v2/scripts/run_formal_v2.sh` with
`CSI_PAIRS_FULL_RUN_PHASE=prepare`. Optional `CSI_PAIRS_*` env names are
omitted unless set.

### 6. `create-run-approval` (LLM-as-judge file stays outside the run root)

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m formal_v2.formal_cli create-run-approval \
  --request "$PWD/runs/formal-001/approval/request.json" \
  --output /absolute/path/formal-001-llm-judge-approval.json \
  --judge codex:bound-session \
  --expires-utc 2027-01-01T00:00:00Z \
  --attest-llm-judged
```

### 7. `all --approval-manifest`

```bash
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -B -m formal_v2.formal_cli all \
  --config "$PWD/formal_v2/configs/formal_v2.json" \
  --dataset /absolute/path/csi_pairs_formal_v2_1_v6.npz \
  --output "$PWD/runs/formal-001" \
  --approval-manifest /absolute/path/formal-001-llm-judge-approval.json \
  --adapter-manifest "$PWD/formal_v2/external_adapters/all_map_adapters_v1.json" \
  --control-manifest "$PWD/formal_v2/external_adapters/resource_controls_v3.json" \
  --scene-id-manifest builtin:sigmap-scene-id-v1 \
  --shuffled-pair-manifest "$PWD/formal_v2/external_adapters/shuffled_pair_control_v3.json" \
  --retention-manifest "$PWD/formal_v2/external_adapters/retention_control_v3.json" \
  --representation-baseline-config "$PWD/formal_v2/configs/representation_baselines_v1.json"
```

Or `CSI_PAIRS_FULL_RUN_PHASE=run` plus `CSI_PAIRS_LLM_JUDGE_APPROVAL_MANIFEST=...`
through `run_formal_v2.sh`. `--approve-full-experiment` cannot authorize execution.

## Hash trap

Do not train on the dataset SHA recorded in `LLM_JUDGE_C13.md`
`060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`
unless it is independently authenticated against the formal registry.

The recorded 34-bank registry hash is a different binding:
`e5ec3d32bbb7c639f2fd6e6dcc23bc4bb6085cc76830a700e5b7103b17a37847`.
A later registry row records
`e89034302d6c4a64c838c244a60ad7c42b081e760ef2c9fc548440b0d5458d2e`.
Neither hash is mounted in this clone.

`formal_v2/configs/formal_v2.json` points at
`../data/csi_pairs_formal_v2_1_v6.npz` and this clone has no dataset bytes.
