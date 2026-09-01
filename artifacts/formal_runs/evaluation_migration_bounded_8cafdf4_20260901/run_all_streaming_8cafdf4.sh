#!/bin/bash
# ARCHIVED Autodl snapshot — do not use as entry. Portable entry: formal_v2/scripts/run_formal_v2.sh (or launch_engineering_evaluation.sh for P0).
set -uo pipefail

runtime_root="${CSI_PAIRS_ROOT:-/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server}"
runtime_python="${CSI_PAIRS_PYTHON:-/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python}"
run_root="${CSI_PAIRS_FORMAL_OUTPUT:-runs/formal-v6-streaming-8cafdf4-08828d387df6db2b}"
accepted="$runtime_root/${run_root#/}/migration/accepted.json"
if [[ "$run_root" == /* ]]; then
  accepted="$run_root/migration/accepted.json"
fi
log="${CSI_PAIRS_LAUNCHER_LOG:-/root/xunlian/Futaoran/formal_external_inputs/logs/formal-v6-streaming-8cafdf4-08828d387df6db2b.all.log}"
dataset="${CSI_PAIRS_DATASET:-/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz}"

if [[ "${CSI_PAIRS_REQUIRE_SECOND_JUDGE:-0}" == "1" ]]; then
  if [[ ! -f "$accepted" || -L "$accepted" ]]; then
    printf 'REFUSAL=migration accepted receipt is missing or unsafe: %s\n' "$accepted" >&2
    exit 92
  fi
else
  printf 'SECOND_JUDGE=optional_publication_review; launching evaluation without accepted.json\n'
fi

cd "$runtime_root" || exit 91
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CSI_PAIRS_DEVICE="${CSI_PAIRS_DEVICE:-cuda:1}"
export CSI_PAIRS_DEVICES="${CSI_PAIRS_DEVICES:-cuda:0,cuda:1}"
export CSI_PAIRS_PYTHON="$runtime_python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

if [[ "${CSI_PAIRS_REQUIRE_SECOND_JUDGE:-0}" == "1" ]]; then
  command=(taskset -c 16-31 "$CSI_PAIRS_PYTHON" -B -m formal_v2.formal_cli all
    --config formal_v2/configs/formal_v2.json
    --dataset "$dataset"
    --output "$run_root"
    --adapter-manifest formal_v2/external_adapters/all_map_adapters_v1.json
    --control-manifest formal_v2/external_adapters/resource_controls_v3.json
    --scene-id-manifest builtin:sigmap-scene-id-v1
    --shuffled-pair-manifest formal_v2/external_adapters/shuffled_pair_control_v3.json
    --retention-manifest formal_v2/external_adapters/retention_control_v3.json
    --representation-baseline-config formal_v2/configs/representation_baselines_v1.json)
else
  command=("$CSI_PAIRS_PYTHON" -B -m formal_v2.formal_cli run-evaluation
    --config formal_v2/configs/formal_v2.json
    --dataset "$dataset"
    --output "$run_root")
fi
"${command[@]}" >>"$log" 2>&1
code=$?
printf 'EXIT_CODE=%s\n' "$code" >>"$log"
exit "$code"
