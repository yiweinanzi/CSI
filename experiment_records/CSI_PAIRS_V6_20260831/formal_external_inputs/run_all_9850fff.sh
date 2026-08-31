#!/usr/bin/env bash
set -uo pipefail

cd /root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server || exit 91

export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0,1
export CSI_PAIRS_DEVICE=cuda:1
export CSI_PAIRS_DEVICES=cuda:0,cuda:1
export CSI_PAIRS_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

approval=/root/xunlian/Futaoran/formal_external_inputs/run_approvals/formal-v6-gpu-9850fff-20260825T071800Z/LLM_JUDGE_APPROVAL.json
log=/root/xunlian/Futaoran/formal_external_inputs/logs/formal-v6-gpu-9850fff-20260825T071800Z.all.log

taskset -c 16-31 "$CSI_PAIRS_PYTHON" -B -m formal_v2.formal_cli all \
  --config formal_v2/configs/formal_v2.json \
  --dataset /root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz \
  --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
  --approval-manifest "$approval" \
  --adapter-manifest formal_v2/external_adapters/all_map_adapters_v1.json \
  --control-manifest formal_v2/external_adapters/resource_controls_v3.json \
  --scene-id-manifest builtin:sigmap-scene-id-v1 \
  --shuffled-pair-manifest formal_v2/external_adapters/shuffled_pair_control_v3.json \
  --retention-manifest formal_v2/external_adapters/retention_control_v3.json \
  --representation-baseline-config formal_v2/configs/representation_baselines_v1.json \
  >"$log" 2>&1
code=$?
printf 'EXIT_CODE=%s\n' "$code" >>"$log"
exit "$code"
