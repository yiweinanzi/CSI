#!/bin/bash
set -euo pipefail

cd /root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
exec env \
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES=0,1 \
  /root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python \
  -B -m formal_v2.formal_migration_evidence_runner performance \
  --identity /root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/expected_identity.json \
  --config /root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/formal_v2/configs/formal_v2.json \
  --dataset /root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz \
  --output-dir /root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/performance \
  --device cuda:1 \
  --batch-size 1 \
  --base-checkpoint-count 1 \
  --base-positions-per-scene 4 \
  --source-scenes 1 \
  --target-scenes-per-city 1
