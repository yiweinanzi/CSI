#!/usr/bin/env bash
set -uo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
RUN_ROOT="$RUNTIME_ROOT/runs/formal-v6-streaming-8d489b2-d7d0b8affcd40f05"
LAUNCHER=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831/run_all_streaming_8d489b2.sh
SUPERVISION_ROOT=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-streaming-8d489b2-d7d0b8affcd40f05
EXPECTED_COMMIT=8d489b2387e7bb6c988a41d9e0d57b8a6cffc4d4
EXPECTED_SOURCE_SHA256=aa5b1d6a1062d68bb5de045b40f042e1a453d14a8d73632ad0be86dc7b07caff
EXPECTED_LAUNCHER_SHA256=399a848fc5bb31cbb7c71a857f10584655164fdb6ae2b9321c765f7a459e6e1b
WIGATR_LINK="$RUNTIME_ROOT/formal_v2/external_adapters/.venv-wigatr"
WIGATR_TARGET=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters/.venv-wigatr
RUNTIME_PROBE=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831/probe_wigatr_runtime_8d489b2.sh
EXPECTED_RUNTIME_PROBE_SHA256=0fd56ff2fb391d133861eb5986f01aa2ac3daba0ab719f6e41cbd5ebd1283d69
POLL_SECONDS=300

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$SUPERVISION_ROOT"
exec 9>"$SUPERVISION_ROOT/supervisor.lock"
if ! flock -n 9; then
  printf '%s SUPERVISOR_ALREADY_ACTIVE\n' "$(date --iso-8601=seconds)"
  exit 0
fi

SUPERVISOR_LOG="$SUPERVISION_ROOT/supervisor.log"
CHILD_IDENTITY="$SUPERVISION_ROOT/child_identity.env"
ACCEPTED="$RUN_ROOT/migration/accepted.json"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$SUPERVISOR_LOG"
}

refuse() {
  log "SUPERVISOR_REFUSAL=$1"
  return 1
}

preflight() {
  [[ -f "$ACCEPTED" && ! -L "$ACCEPTED" ]] || refuse ACCEPTED_RECEIPT_MISSING || return 1
  [[ -f "$LAUNCHER" && ! -L "$LAUNCHER" ]] || refuse LAUNCHER_MISSING || return 1
  [[ "$(sha256sum "$LAUNCHER" | awk '{print $1}')" == "$EXPECTED_LAUNCHER_SHA256" ]] \
    || refuse LAUNCHER_SHA_MISMATCH || return 1
  [[ "$(git -C "$RUNTIME_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] \
    || refuse COMMIT_MISMATCH || return 1
  [[ -z "$(git -C "$RUNTIME_ROOT" status --porcelain --untracked-files=no)" ]] \
    || refuse TRACKED_WORKTREE_DIRTY || return 1
  [[ "$(git -C "$RUNTIME_ROOT" ls-files --others --exclude-standard -- formal_v2)" == "formal_v2/external_adapters/.venv-wigatr" ]] \
    || refuse UNREVIEWED_UNTRACKED_FORMAL_SOURCE || return 1
  [[ -L "$WIGATR_LINK" && "$(readlink "$WIGATR_LINK")" == "$WIGATR_TARGET" ]] \
    || refuse WIGATR_RUNTIME_LINK_MISMATCH || return 1
  [[ -x "$WIGATR_LINK/bin/python" ]] \
    || refuse WIGATR_RUNTIME_EXECUTABLE_MISSING || return 1
  [[ -f "$RUNTIME_PROBE" && ! -L "$RUNTIME_PROBE" && -x "$RUNTIME_PROBE" ]] \
    || refuse WIGATR_RUNTIME_PROBE_MISSING || return 1
  [[ "$(sha256sum "$RUNTIME_PROBE" | awk '{print $1}')" == "$EXPECTED_RUNTIME_PROBE_SHA256" ]] \
    || refuse WIGATR_RUNTIME_PROBE_SHA_MISMATCH || return 1
  (
    cd "$RUNTIME_ROOT" || exit 1
    "$RUNTIME_PYTHON" -B -c \
      'from formal_v2.formal_migration import _source_identity; import sys; value=_source_identity("formal_v2"); sys.exit(0 if value["git_commit"]==sys.argv[1] and value["source_tree_sha256"]==sys.argv[2] else 1)' \
      "$EXPECTED_COMMIT" "$EXPECTED_SOURCE_SHA256"
  ) || refuse SOURCE_IDENTITY_MISMATCH || return 1
  "$RUNTIME_PROBE" validate >>"$SUPERVISOR_LOG" 2>&1 \
    || refuse WIGATR_RUNTIME_PROVENANCE_MISMATCH || return 1
}

refuse_duplicate_child() {
  [[ -f "$CHILD_IDENTITY" && ! -L "$CHILD_IDENTITY" ]] || return 0
  local recorded_pid
  local recorded_ticks
  recorded_pid=$(awk -F= '$1 == "pid" {print $2}' "$CHILD_IDENTITY")
  recorded_ticks=$(awk -F= '$1 == "start_ticks" {print $2}' "$CHILD_IDENTITY")
  if (
    [[ "$recorded_pid" =~ ^[1-9][0-9]*$ ]]
    [[ "$recorded_ticks" =~ ^[1-9][0-9]*$ ]]
    [[ -r "/proc/$recorded_pid/stat" ]]
    [[ "$(awk '{print $22}' "/proc/$recorded_pid/stat")" == "$recorded_ticks" ]]
  ); then
    log "SUPERVISOR_ALREADY_RUNNING_CHILD pid=$recorded_pid start_ticks=$recorded_ticks"
    return 1
  fi
  return 0
}

record_status() {
  local child_pid=$1
  local start_ticks=$2
  local observed_ticks
  if [[ -r "/proc/$child_pid/stat" ]]; then
    observed_ticks=$(awk '{print $22}' "/proc/$child_pid/stat")
    if [[ "$observed_ticks" != "$start_ticks" ]]; then
      log "CHILD_IDENTITY=PID_REUSED_OR_MISMATCH pid=$child_pid"
      return 1
    fi
  fi
  log "CHILD_IDENTITY=MATCH pid=$child_pid start_ticks=$start_ticks"
  if [[ -d "$RUN_ROOT/evaluation_state" && ! -L "$RUN_ROOT/evaluation_state" ]]; then
    (
      cd "$RUNTIME_ROOT" || exit 1
      "$RUNTIME_PYTHON" -B -m formal_v2.formal_cli evaluation-status \
        --output "$RUN_ROOT"
    ) >>"$SUPERVISOR_LOG" 2>&1 || log EVALUATION_STATUS_READ_FAILED
  else
    log EVALUATION_STATUS=NOT_STARTED
  fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits >>"$SUPERVISOR_LOG" 2>&1 \
    || log NVIDIA_SMI_READ_FAILED
  df -PB1 "$RUNTIME_ROOT" | tail -1 >>"$SUPERVISOR_LOG"
  log "PYC_COUNT=$(find "$RUNTIME_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | wc -l)"
}

main() {
  preflight || return 92
  refuse_duplicate_child || return 0
  log "SUPERVISOR_START accepted_sha256=$(sha256sum "$ACCEPTED" | awk '{print $1}')"
  bash "$LAUNCHER" &
  local child_pid=$!
  local start_ticks
  start_ticks=$(awk '{print $22}' "/proc/$child_pid/stat") || return 93
  {
    printf 'pid=%s\n' "$child_pid"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'launcher_sha256=%s\n' "$EXPECTED_LAUNCHER_SHA256"
    printf 'accepted_sha256=%s\n' "$(sha256sum "$ACCEPTED" | awk '{print $1}')"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$CHILD_IDENTITY"
  log "CHILD_STARTED pid=$child_pid start_ticks=$start_ticks"

  while kill -0 "$child_pid" 2>/dev/null; do
    record_status "$child_pid" "$start_ticks" || return 94
    sleep "$POLL_SECONDS"
  done
  wait "$child_pid"
  local code=$?
  log "CHILD_EXIT_CODE=$code"
  return "$code"
}

main
code=$?
log "SUPERVISOR_EXIT_CODE=$code"
exit "$code"
