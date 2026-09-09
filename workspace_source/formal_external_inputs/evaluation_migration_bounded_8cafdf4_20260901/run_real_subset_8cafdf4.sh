#!/bin/bash
set -euo pipefail

exec env \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=0,1 \
  /root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python \
  -B -m formal_v2.formal_evaluation_subset_compare compare \
  --config /root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/formal_v2/configs/formal_v2.json \
  --dataset /root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz \
  --legacy-run /root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z \
  --report /root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/real_subset_report_8cafdf4.json \
  --frozen-source-root /root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server \
  --candidate-source-root /root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server \
  --device cuda:0 \
  --batch-size 1 \
  --source-scenes 1 \
  --target-scenes-per-city 1 \
  --positions-per-scene 4 \
  --checkpoint-count 1 \
  --checkpoint-arm endpoint
