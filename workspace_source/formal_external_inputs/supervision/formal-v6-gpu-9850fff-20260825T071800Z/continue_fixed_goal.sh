#!/usr/bin/env bash
set -uo pipefail

AUDIT_DIR=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-gpu-9850fff-20260825T071800Z
PROJECT_ROOT=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
RUN_ROOT="$PROJECT_ROOT/runs/formal-v6-gpu-9850fff-20260825T071800Z"
DATASET=/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz
CONFIG="$PROJECT_ROOT/formal_v2/configs/formal_v2.json"
PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
ACCEPTOR="$AUDIT_DIR/accept_stage.py"
DRIVER_LOG="$AUDIT_DIR/driver.log"
LOCK_PATH="$AUDIT_DIR/driver.lock"

export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0,1
export CSI_PAIRS_DEVICE=cuda:1
export CSI_PAIRS_DEVICES=cuda:0,cuda:1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  printf '%s driver already active\n' "$(date --iso-8601=seconds)" >>"$DRIVER_LOG"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$DRIVER_LOG"
}

write_command_record() {
  local stage=$1
  shift
  local record="$AUDIT_DIR/$stage.command.txt"
  {
    printf 'recorded_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'cwd=%s\n' "$PROJECT_ROOT"
    printf 'PYTHONDONTWRITEBYTECODE=%s\n' "$PYTHONDONTWRITEBYTECODE"
    printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "$CUBLAS_WORKSPACE_CONFIG"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'CSI_PAIRS_DEVICE=%s\n' "$CSI_PAIRS_DEVICE"
    printf 'CSI_PAIRS_DEVICES=%s\n' "$CSI_PAIRS_DEVICES"
    printf 'OMP_NUM_THREADS=%s\n' "$OMP_NUM_THREADS"
    printf 'MKL_NUM_THREADS=%s\n' "$MKL_NUM_THREADS"
    printf 'argv='
    printf '%q ' "$@"
    printf '\n'
  } >"$record"
}

accept_stage() {
  local stage=$1
  local stage_root=$2
  local artifact=$3
  local schema=$4
  local exit_code=$5
  local extras=()
  if [[ -f "$AUDIT_DIR/$stage.timing.env" ]]; then
    extras+=(--timing-record "$AUDIT_DIR/$stage.timing.env")
  fi
  if [[ -f "$AUDIT_DIR/$stage.run.log" ]]; then
    extras+=(--run-log "$AUDIT_DIR/$stage.run.log" --log-location "$AUDIT_DIR/$stage.run.log")
  elif [[ "$stage" == evaluation ]]; then
    extras+=(--log-location EXEC_SESSION_60477_PIPE_NOT_PERSISTED)
  fi
  "$PYTHON" -B "$ACCEPTOR" \
    --stage "$stage" \
    --stage-root "$stage_root" \
    --artifact "$artifact" \
    --schema "$schema" \
    --project-root "$PROJECT_ROOT" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --receipt "$AUDIT_DIR/$stage.acceptance.json" \
    --command-record "$AUDIT_DIR/$stage.command.txt" \
    --exit-code "$exit_code" \
    "${extras[@]}" \
    >>"$AUDIT_DIR/$stage.acceptance.log" 2>&1
}

freeze_evaluation_site() {
  local target="$AUDIT_DIR/evaluation_exit_site.txt"
  {
    printf 'COMMAND_STARTED_AT=2026-08-29T10:52:49+08:00\n'
    printf 'PID_EXIT_DETECTED_AT=%s\n' "$(date --iso-8601=seconds)"
    printf 'PROCESS_EXIT_CODE=UNAVAILABLE_NON_SHELL_OBSERVER\n'
  } >"$AUDIT_DIR/evaluation.timing.env"
  {
    date --iso-8601=seconds
    printf 'git_head='
    git -C "$PROJECT_ROOT" rev-parse HEAD
    printf '%s\n' 'git_status_begin'
    git -C "$PROJECT_ROOT" status --short
    printf '%s\n' 'git_status_end'
    stat -Lc 'config bytes=%s mtime=%y inode=%i path=%n' "$CONFIG"
    stat -Lc 'dataset bytes=%s mtime=%y inode=%i path=%n' "$DATASET"
    sha256sum "$CONFIG" "$DATASET"
    printf '%s\n' 'evaluation_inventory_begin'
    find "$RUN_ROOT/evaluation" -maxdepth 1 -type f \
      -printf '%f\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
    printf '%s\n' 'evaluation_inventory_end'
    printf 'operation_lock='
    if [[ -e "$PROJECT_ROOT/runs/.formal-v6-gpu-9850fff-20260825T071800Z.csi-pairs-operation.lock" ]]; then
      printf 'PRESENT\n'
    else
      printf 'ABSENT\n'
    fi
    df -PB1 "$RUN_ROOT"
  } >"$target"
}

run_stage() {
  local stage=$1
  local relative=$2
  local artifact_name=$3
  local schema=$4
  shift 4
  local stage_root="$RUN_ROOT/$relative"
  local artifact="$stage_root/$artifact_name"
  local log_path="$AUDIT_DIR/$stage.run.log"
  local command=(taskset -c 16-31 "$PYTHON" -B -m formal_v2.formal_cli "$@")

  write_command_record "$stage" "${command[@]}"
  if [[ -d "$stage_root" ]]; then
    log "$stage output already exists; reauthenticating without overwrite"
    if accept_stage "$stage" "$stage_root" "$artifact" "$schema" EXISTING; then
      log "$stage existing output acceptance PASS"
      return 0
    fi
    log "$stage existing output acceptance FAIL; stopping"
    return 1
  fi

  log "$stage starting"
  printf 'COMMAND_STARTED_AT=%s\n' "$(date --iso-8601=seconds)" >"$AUDIT_DIR/$stage.timing.env"
  (
    cd "$PROJECT_ROOT" || exit 91
    "${command[@]}"
  ) >"$log_path" 2>&1
  local code=$?
  {
    printf 'COMMAND_FINISHED_AT=%s\n' "$(date --iso-8601=seconds)"
    printf 'PROCESS_EXIT_CODE=%s\n' "$code"
  } >>"$AUDIT_DIR/$stage.timing.env"
  printf 'EXIT_CODE=%s\n' "$code" >>"$log_path"
  log "$stage process exit_code=$code"

  if ! accept_stage "$stage" "$stage_root" "$artifact" "$schema" "$code"; then
    log "$stage output acceptance FAIL; stopping"
    return 1
  fi
  if [[ "$code" -eq 0 ]]; then
    log "$stage output acceptance PASS and process exit 0"
    return 0
  fi
  if [[ "$code" -eq 1 ]] && rg -q '"output_acceptance": "PASS"' "$AUDIT_DIR/$stage.acceptance.json"; then
    log "$stage produced authenticated scientific FAIL/BLOCKED evidence; preserving result and continuing per Goal"
    return 0
  fi
  log "$stage implementation/process failure exit_code=$code; stopping"
  return 1
}

main() {
  cd "$PROJECT_ROOT" || return 91
  log "fixed Goal continuation started"
  write_command_record evaluation \
    taskset -c 16-31 "$PYTHON" -B -m formal_v2.formal_cli run-evaluation \
    --config formal_v2/configs/formal_v2.json \
    --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z

  local operation_lock="$PROJECT_ROOT/runs/.formal-v6-gpu-9850fff-20260825T071800Z.csi-pairs-operation.lock"
  local waits=0
  while [[ -e "$operation_lock" && "$waits" -lt 60 ]]; do
    sleep 5
    waits=$((waits + 1))
  done
  freeze_evaluation_site
  if [[ -e "$operation_lock" ]]; then
    log "evaluation PID exited but operation lock remains; OUTPUT_ACCEPTANCE=INCOMPLETE"
    return 1
  fi
  if ! accept_stage evaluation "$RUN_ROOT/evaluation" "$RUN_ROOT/evaluation/gate.json" \
      csi-pairs-v6-evaluation-gate-v3 UNKNOWN; then
    log "evaluation OUTPUT_ACCEPTANCE=FAIL; stopping before risk"
    return 1
  fi
  log "evaluation OUTPUT_ACCEPTANCE=PASS"

  run_stage risk risk gate.json csi-pairs-v6-risk-gate-v2 \
    run-risk --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z || return 1
  run_stage path path gate.json csi-pairs-v6-path-gate-v3 \
    run-path --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z || return 1
  run_stage external_baselines external_baselines gate.json \
    csi-pairs-v6-external-baseline-gate-v3 \
    run-external-baselines --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --adapter-manifest formal_v2/external_adapters/all_map_adapters_v1.json || return 1
  run_stage representation_baselines representation_baselines gate.json \
    csi-pairs-v6-representation-baseline-gate-v2 \
    run-representation-baselines --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --representation-baseline-config formal_v2/configs/representation_baselines_v1.json || return 1
  run_stage resource_controls controls gate.json csi-pairs-v6-resource-control-gate-v3 \
    run-resource-controls --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --control-manifest formal_v2/external_adapters/resource_controls_v3.json || return 1
  run_stage scene_id scene_id gate.json csi-pairs-v6-scene-id-gate-v3 \
    run-scene-id-audit --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --scene-id-manifest builtin:sigmap-scene-id-v1 || return 1
  run_stage shuffled_pair controls/shuffled_pair gate.json csi-pairs-v6-shuffled-pair-gate-v3 \
    run-shuffled-pair-control --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --claim-control-manifest formal_v2/external_adapters/shuffled_pair_control_v3.json || return 1
  run_stage retention evaluation/retention gate.json csi-pairs-v6-retention-gate-v3 \
    run-retention-audit --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z \
    --claim-control-manifest formal_v2/external_adapters/retention_control_v3.json || return 1
  run_stage claims claims claim_evidence.json csi-pairs-v6-claim-evidence-v2 \
    assemble-claims --config formal_v2/configs/formal_v2.json --dataset "$DATASET" \
    --output runs/formal-v6-gpu-9850fff-20260825T071800Z || return 1
  log "fixed Goal downstream chain completed"
  return 0
}

main
code=$?
printf '%s DRIVER_EXIT_CODE=%s\n' "$(date --iso-8601=seconds)" "$code" >>"$DRIVER_LOG"
exit "$code"
