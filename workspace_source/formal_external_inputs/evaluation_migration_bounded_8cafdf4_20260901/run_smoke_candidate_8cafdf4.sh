#!/bin/bash
set -euo pipefail

runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
fixture_dataset=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/runs/smoke-full-v6-p8-clusterfix-20260826T230312Z.fixture.npz
fixture_upstream=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/runs/smoke-full-v6-p8-clusterfix-20260826T230312Z
candidate_output=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/smoke_candidate_evaluation
run_nonce=d7f27e5a5a307b3ef45d0235202dfd3a14f07941c5c044063d8de4c7c5338591

cd "$runtime_root"
export CUDA_VISIBLE_DEVICES=0,1
export CSI_PAIRS_DEVICE=cuda:1
export CSI_PAIRS_DEVICES=cuda:0,cuda:1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

exec "$runtime_python" -B -c \
  'from formal_v2.formal_config import load_formal_config; from formal_v2.formal_dataset import FormalDataset; from formal_v2.formal_evaluation_identity import run_fixture_streaming_evaluation; config = load_formal_config("formal_v2/configs/formal_v2_smoke.json"); dataset = FormalDataset.load("'"$fixture_dataset"'"); run_fixture_streaming_evaluation(config, dataset, output_root="'"$candidate_output"'", upstream_root="'"$fixture_upstream"'", run_nonce="'"$run_nonce"'", execution_devices=("cuda:0", "cuda:1"), batch_size=1)'
