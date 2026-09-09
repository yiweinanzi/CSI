#!/bin/bash
set -uo pipefail

cd /root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server
export CUDA_VISIBLE_DEVICES=0,1
export CSI_PAIRS_DEVICE=cuda:1
export CSI_PAIRS_DEVICES=cuda:0,cuda:1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
export CSI_PAIRS_FIXTURE_RUNTIME_CACHE=1
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

.venv-core-formal-20260819T091049Z/bin/python -B -m formal_v2.formal_cli all \
  --config formal_v2/configs/formal_v2_smoke.json \
  --dataset runs/smoke-full-v6-p8-routefix-20260826T215954Z.fixture.npz \
  --output runs/smoke-full-v6-p8-routefix-20260826T215954Z \
  --allow-nonscientific-fixture \
  --adapter-manifest formal_v2/external_adapters/all_map_adapters_smoke_v1.json \
  --control-manifest formal_v2/external_adapters/resource_controls_v3.json \
  --scene-id-manifest builtin:sigmap-scene-id-v1 \
  --shuffled-pair-manifest formal_v2/external_adapters/shuffled_pair_control_v3.json \
  --retention-manifest formal_v2/external_adapters/retention_control_v3.json \
  --representation-baseline-config formal_v2/configs/representation_baselines_smoke_v1.json \
  --approval-manifest /root/xunlian/Futaoran/formal_external_inputs/run_approvals/smoke-full-v6-p8-routefix-20260826T215954Z/LLM_JUDGE_APPROVAL.json \
  >> /tmp/smoke-full-v6-p8-routefix-20260826T215954Z.all.log 2>&1
smoke_exit=$?
printf '%s\n' "$smoke_exit" > /tmp/smoke-full-v6-p8-routefix-20260826T215954Z.all.exit
exit "$smoke_exit"
