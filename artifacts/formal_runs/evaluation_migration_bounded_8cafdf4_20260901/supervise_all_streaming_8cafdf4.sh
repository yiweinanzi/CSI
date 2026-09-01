#!/usr/bin/env bash
set -uo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
RUN_ROOT="$RUNTIME_ROOT/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b"
LAUNCHER=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/run_all_streaming_8cafdf4.sh
LAUNCHER_LOG=/root/xunlian/Futaoran/formal_external_inputs/logs/formal-v6-streaming-8cafdf4-08828d387df6db2b.all.log
SUPERVISION_ROOT=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-streaming-8cafdf4-08828d387df6db2b
EXPECTED_COMMIT=8cafdf4a4c67d40c4fd41b7d71b6468df850cdd7
EXPECTED_SOURCE_SHA256=a5e2658b88c051e7d83256d8b802b914c76659162464fb1e897f76f0473210f0
EXPECTED_LAUNCHER_SHA256=f7fb8a1013e33df821352a4a9e83ecf9ad3fd47712b65611098d6c683ae28fd2
WIGATR_LINK="$RUNTIME_ROOT/formal_v2/external_adapters/.venv-wigatr"
WIGATR_TARGET=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/formal_v2/external_adapters/.venv-wigatr
RUNTIME_PROBE=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901/probe_wigatr_runtime_8cafdf4.sh
EXPECTED_RUNTIME_PROBE_SHA256=1bb6de6c260f6c8f1625eb9188ca58bc27a8dc996996112ad7606b4a6ebe99d7
POLL_SECONDS=300
CHILD_CHECK_SECONDS=5
RETRY_SECONDS=60
MAX_EVALUATION_ATTEMPTS=3

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
STATE_ROOT="$SUPERVISION_ROOT/state"
CHILD_IDENTITY="$STATE_ROOT/child_identity.env"
ATTEMPT_COUNT="$STATE_ROOT/attempt_count"
SUPERVISOR_IDENTITY="$STATE_ROOT/supervisor_identity.env"
SUPERVISOR_EXIT="$STATE_ROOT/supervisor_exit.env"
ACCEPTED="$RUN_ROOT/migration/accepted.json"
mkdir -p "$STATE_ROOT"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$SUPERVISOR_LOG"
}

refuse() {
  log "SUPERVISOR_REFUSAL=$1"
  return 1
}

atomic_replace() {
  local source=$1
  local target=$2
  mv -f -- "$source" "$target"
}

env_value() {
  local path=$1
  local key=$2
  awk -F= -v expected="$key" '$1 == expected {print substr($0, length($1) + 2); found=1} END {if (!found) exit 1}' "$path"
}

process_identity_matches() {
  local pid=$1
  local expected_ticks=$2
  local expected_cmd_sha=$3
  local observed_ticks
  local observed_cmd_sha
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$expected_ticks" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "$expected_cmd_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
  observed_ticks=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [[ "$observed_ticks" == "$expected_ticks" ]] || return 1
  observed_cmd_sha=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | sha256sum | awk '{print $1}') || return 1
  [[ "$observed_cmd_sha" == "$expected_cmd_sha" ]]
}

write_supervisor_identity() {
  local self_ticks
  local self_cmd_sha
  local current_tmp
  local immutable
  self_ticks=$(awk '{print $22}' "/proc/$$/stat") || return 1
  self_cmd_sha=$(tr '\0' ' ' <"/proc/$$/cmdline" | sha256sum | awk '{print $1}') || return 1
  immutable="$STATE_ROOT/supervisor-start-$self_ticks.env"
  current_tmp="$SUPERVISOR_IDENTITY.tmp.$$"
  {
    printf 'schema=csi-pairs-v6-supervisor-identity-v1\n'
    printf 'pid=%s\n' "$$"
    printf 'start_ticks=%s\n' "$self_ticks"
    printf 'cmdline_sha256=%s\n' "$self_cmd_sha"
    printf 'supervisor_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'status=RUNNING\n'
  } >"$current_tmp"
  if [[ ! -e "$immutable" && ! -L "$immutable" ]]; then
    cp -- "$current_tmp" "$immutable"
  fi
  atomic_replace "$current_tmp" "$SUPERVISOR_IDENTITY"
}

write_supervisor_exit() {
  local code=$1
  local tmp="$SUPERVISOR_EXIT.tmp.$$"
  {
    printf 'schema=csi-pairs-v6-supervisor-exit-v1\n'
    printf 'pid=%s\n' "$$"
    printf 'start_ticks=%s\n' "$(awk '{print $22}' "/proc/$$/stat" 2>/dev/null || printf UNKNOWN)"
    printf 'exit_code=%s\n' "$code"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$SUPERVISOR_EXIT"
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

record_status() {
  local child_pid=$1
  local start_ticks=$2
  local cmd_sha=$3
  if process_identity_matches "$child_pid" "$start_ticks" "$cmd_sha"; then
    log "CHILD_IDENTITY=MATCH pid=$child_pid start_ticks=$start_ticks"
  elif [[ -e "/proc/$child_pid" ]]; then
    log "CHILD_IDENTITY=PID_REUSED_OR_MISMATCH pid=$child_pid"
    return 1
  else
    log "CHILD_IDENTITY=EXITED pid=$child_pid start_ticks=$start_ticks"
    return 0
  fi
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
  log "PYC_COUNT=$(find "$RUNTIME_ROOT/formal_v2" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | wc -l)"
}

downstream_started() {
  local relative
  for relative in \
    risk \
    path \
    external_baselines \
    representation_baselines \
    controls \
    scene_id \
    evaluation/retention \
    claims
  do
    if [[ -e "$RUN_ROOT/$relative" || -L "$RUN_ROOT/$relative" ]]; then
      log "DOWNSTREAM_STARTED=$relative"
      return 0
    fi
  done
  return 1
}

launcher_child() {
  local attempt_dir=$1
  local go_file="$attempt_dir/go"
  local ready="$attempt_dir/ready.env"
  local running="$attempt_dir/running.env"
  local exit_receipt="$attempt_dir/exit.env"
  local child_pid=$BASHPID
  local child_ticks
  local child_cmd_sha
  local launcher_pid
  local launcher_ticks
  local launcher_cmd_sha
  local launcher_session_id
  local launcher_cmdline
  local identity_loops=0
  local log_offset=0
  local log_size
  local code
  local segment_exit_code
  local segment_exit_count
  local tmp
  exec 9>&-
  child_ticks=$(awk '{print $22}' "/proc/$child_pid/stat") || exit 98
  child_cmd_sha=$(tr '\0' ' ' <"/proc/$child_pid/cmdline" | sha256sum | awk '{print $1}') || exit 98
  if [[ -f "$LAUNCHER_LOG" && ! -L "$LAUNCHER_LOG" ]]; then
    log_offset=$(stat -c '%s' "$LAUNCHER_LOG") || exit 98
  elif [[ -e "$LAUNCHER_LOG" || -L "$LAUNCHER_LOG" ]]; then
    exit 98
  fi
  tmp="$ready.tmp.$child_pid"
  {
    printf 'schema=csi-pairs-v6-launcher-child-ready-v1\n'
    printf 'pid=%s\n' "$child_pid"
    printf 'start_ticks=%s\n' "$child_ticks"
    printf 'cmdline_sha256=%s\n' "$child_cmd_sha"
    printf 'launcher_log_offset=%s\n' "$log_offset"
    printf 'ready_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$ready"
  while [[ ! -f "$go_file" || -L "$go_file" ]]; do
    sleep 0.1
  done
  setsid bash "$LAUNCHER" &
  launcher_pid=$!
  while true; do
    if [[ -r "/proc/$launcher_pid/stat" && -r "/proc/$launcher_pid/cmdline" ]]; then
      launcher_ticks=$(awk '{print $22}' "/proc/$launcher_pid/stat" 2>/dev/null || true)
      launcher_session_id=$(awk '{print $6}' "/proc/$launcher_pid/stat" 2>/dev/null || true)
      launcher_cmdline=$(tr '\0' ' ' <"/proc/$launcher_pid/cmdline" 2>/dev/null || true)
      if [[ "$launcher_session_id" == "$launcher_pid" && "$launcher_cmdline" == *"$LAUNCHER"* ]]; then
        launcher_cmd_sha=$(printf '%s' "$launcher_cmdline" | sha256sum | awk '{print $1}')
        break
      fi
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      wait "$launcher_pid"
      exit $?
    fi
    identity_loops=$((identity_loops + 1))
    (( identity_loops <= 300 )) || exit 98
    sleep 0.1
  done
  tmp="$running.tmp.$child_pid"
  {
    printf 'schema=csi-pairs-v6-launcher-running-v1\n'
    printf 'wrapper_pid=%s\n' "$child_pid"
    printf 'wrapper_start_ticks=%s\n' "$child_ticks"
    printf 'wrapper_cmdline_sha256=%s\n' "$child_cmd_sha"
    printf 'launcher_pid=%s\n' "$launcher_pid"
    printf 'launcher_start_ticks=%s\n' "$launcher_ticks"
    printf 'launcher_cmdline_sha256=%s\n' "$launcher_cmd_sha"
    printf 'launcher_session_id=%s\n' "$launcher_session_id"
    printf 'launcher_sha256=%s\n' "$EXPECTED_LAUNCHER_SHA256"
    printf 'launcher_log_offset=%s\n' "$log_offset"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$running"
  wait "$launcher_pid"
  code=$?
  log_size=$log_offset
  segment_exit_code=NONE
  segment_exit_count=0
  if [[ -f "$LAUNCHER_LOG" && ! -L "$LAUNCHER_LOG" ]]; then
    log_size=$(stat -c '%s' "$LAUNCHER_LOG" 2>/dev/null || printf '%s' "$log_offset")
    if (( log_size >= log_offset )); then
      segment_exit_count=$(tail -c "+$((log_offset + 1))" "$LAUNCHER_LOG" 2>/dev/null | awk '/^EXIT_CODE=[0-9]+$/ {count++} END {print count+0}')
      segment_exit_code=$(tail -c "+$((log_offset + 1))" "$LAUNCHER_LOG" 2>/dev/null | awk -F= '/^EXIT_CODE=[0-9]+$/ {value=$2} END {if (value != "") print value; else print "NONE"}')
    fi
  fi
  tmp="$exit_receipt.tmp.$child_pid"
  {
    printf 'schema=csi-pairs-v6-launcher-child-exit-v1\n'
    printf 'pid=%s\n' "$child_pid"
    printf 'start_ticks=%s\n' "$child_ticks"
    printf 'launcher_pid=%s\n' "$launcher_pid"
    printf 'launcher_start_ticks=%s\n' "$launcher_ticks"
    printf 'launcher_cmdline_sha256=%s\n' "$launcher_cmd_sha"
    printf 'launcher_session_id=%s\n' "$launcher_session_id"
    printf 'launcher_exit_code=%s\n' "$code"
    printf 'launcher_log_offset=%s\n' "$log_offset"
    printf 'launcher_log_size=%s\n' "$log_size"
    printf 'segment_exit_count=%s\n' "$segment_exit_count"
    printf 'segment_exit_code=%s\n' "$segment_exit_code"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$exit_receipt"
  exit "$code"
}

release_child() {
  local go_file=$1
  local tmp
  if [[ -e "$go_file" || -L "$go_file" ]]; then
    [[ -f "$go_file" && ! -L "$go_file" ]] || return 1
    return 0
  fi
  tmp="$go_file.tmp.$$"
  printf 'released_at=%s\n' "$(date --iso-8601=seconds)" >"$tmp"
  atomic_replace "$tmp" "$go_file"
}

next_attempt_number() {
  local current=0
  local next
  local tmp
  if [[ -e "$ATTEMPT_COUNT" || -L "$ATTEMPT_COUNT" ]]; then
    [[ -f "$ATTEMPT_COUNT" && ! -L "$ATTEMPT_COUNT" ]] || return 1
    current=$(<"$ATTEMPT_COUNT")
    [[ "$current" =~ ^[0-9]+$ ]] || return 1
  fi
  next=$((current + 1))
  (( next <= MAX_EVALUATION_ATTEMPTS )) || return 2
  tmp="$ATTEMPT_COUNT.tmp.$$"
  printf '%s\n' "$next" >"$tmp"
  atomic_replace "$tmp" "$ATTEMPT_COUNT"
  printf '%s\n' "$next"
}

publish_child_identity() {
  local attempt=$1
  local attempt_dir=$2
  local ready="$attempt_dir/ready.env"
  local tmp="$CHILD_IDENTITY.tmp.$$"
  {
    printf 'schema=csi-pairs-v6-supervised-child-v1\n'
    printf 'attempt=%s\n' "$attempt"
    printf 'attempt_dir=%s\n' "$attempt_dir"
    printf 'pid=%s\n' "$(env_value "$ready" pid)"
    printf 'start_ticks=%s\n' "$(env_value "$ready" start_ticks)"
    printf 'cmdline_sha256=%s\n' "$(env_value "$ready" cmdline_sha256)"
    printf 'launcher_sha256=%s\n' "$EXPECTED_LAUNCHER_SHA256"
    printf 'accepted_sha256=%s\n' "$(sha256sum "$ACCEPTED" | awk '{print $1}')"
    printf 'published_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$CHILD_IDENTITY"
}

start_child() {
  local attempt=$1
  local attempt_dir="$STATE_ROOT/attempt-$attempt"
  local child_pid
  local observed_pid
  local loops=0
  mkdir "$attempt_dir" || return 1
  launcher_child "$attempt_dir" &
  child_pid=$!
  while [[ ! -f "$attempt_dir/ready.env" || -L "$attempt_dir/ready.env" ]]; do
    if ! kill -0 "$child_pid" 2>/dev/null; then
      wait "$child_pid" 2>/dev/null
      log "CHILD_HANDSHAKE_FAILED pid=$child_pid attempt=$attempt"
      return 1
    fi
    loops=$((loops + 1))
    if (( loops > 300 )); then
      log "CHILD_HANDSHAKE_TIMEOUT pid=$child_pid attempt=$attempt"
      return 1
    fi
    sleep 0.1
  done
  observed_pid=$(env_value "$attempt_dir/ready.env" pid) || return 1
  [[ "$observed_pid" == "$child_pid" ]] || return 1
  process_identity_matches \
    "$child_pid" \
    "$(env_value "$attempt_dir/ready.env" start_ticks)" \
    "$(env_value "$attempt_dir/ready.env" cmdline_sha256)" || return 1
  publish_child_identity "$attempt" "$attempt_dir" || return 1
  release_child "$attempt_dir/go" || return 1
  log "CHILD_STARTED pid=$child_pid start_ticks=$(env_value "$attempt_dir/ready.env" start_ticks) attempt=$attempt"
}

load_child_identity() {
  [[ -f "$CHILD_IDENTITY" && ! -L "$CHILD_IDENTITY" ]] || return 1
  local required
  for required in attempt attempt_dir pid start_ticks cmdline_sha256; do
    env_value "$CHILD_IDENTITY" "$required" >/dev/null || return 1
  done
}

wait_for_exit_receipt() {
  local receipt=$1
  local loops=0
  while [[ ! -f "$receipt" || -L "$receipt" ]]; do
    loops=$((loops + 1))
    (( loops <= 100 )) || return 1
    sleep 0.1
  done
}

recover_detached_launcher() {
  local attempt_dir=$1
  local expected_wrapper_pid=$2
  local expected_wrapper_ticks=$3
  local expected_wrapper_cmd_sha=$4
  local running="$attempt_dir/running.env"
  local receipt="$attempt_dir/exit.env"
  local launcher_pid
  local launcher_ticks
  local launcher_cmd_sha
  local launcher_session_id
  local log_offset
  local log_size
  local exit_count
  local logged_code
  local state
  local checks=0
  local tmp
  [[ -f "$running" && ! -L "$running" ]] || return 1
  [[ "$(env_value "$running" wrapper_pid)" == "$expected_wrapper_pid" ]] || return 1
  [[ "$(env_value "$running" wrapper_start_ticks)" == "$expected_wrapper_ticks" ]] || return 1
  [[ "$(env_value "$running" wrapper_cmdline_sha256)" == "$expected_wrapper_cmd_sha" ]] || return 1
  [[ "$(env_value "$running" launcher_sha256)" == "$EXPECTED_LAUNCHER_SHA256" ]] || return 1
  launcher_pid=$(env_value "$running" launcher_pid) || return 1
  launcher_ticks=$(env_value "$running" launcher_start_ticks) || return 1
  launcher_cmd_sha=$(env_value "$running" launcher_cmdline_sha256) || return 1
  launcher_session_id=$(env_value "$running" launcher_session_id) || return 1
  log_offset=$(env_value "$running" launcher_log_offset) || return 1
  [[ "$log_offset" =~ ^[0-9]+$ && "$launcher_session_id" == "$launcher_pid" ]] || return 1

  if process_identity_matches "$launcher_pid" "$launcher_ticks" "$launcher_cmd_sha"; then
    log "LAUNCHER_REATTACH pid=$launcher_pid start_ticks=$launcher_ticks"
    while process_identity_matches "$launcher_pid" "$launcher_ticks" "$launcher_cmd_sha"; do
      state=$(awk '{print $3}' "/proc/$launcher_pid/stat" 2>/dev/null || printf X)
      [[ "$state" != Z ]] || break
      if (( checks == 0 || (checks * CHILD_CHECK_SECONDS) % POLL_SECONDS == 0 )); then
        record_status "$launcher_pid" "$launcher_ticks" "$launcher_cmd_sha" || return 1
      fi
      sleep "$CHILD_CHECK_SECONDS"
      checks=$((checks + 1))
    done
  elif [[ -e "/proc/$launcher_pid" ]]; then
    log "LAUNCHER_ORIGINAL_EXITED_PID_NOW_REUSED pid=$launcher_pid start_ticks=$launcher_ticks"
  else
    log "LAUNCHER_ALREADY_EXITED pid=$launcher_pid start_ticks=$launcher_ticks"
  fi

  while ps -e -o sid= 2>/dev/null | awk -v expected="$launcher_session_id" '$1 == expected {found=1} END {exit !found}'; do
    if (( checks == 0 || (checks * CHILD_CHECK_SECONDS) % POLL_SECONDS == 0 )); then
      log "LAUNCHER_SESSION_REATTACH session_id=$launcher_session_id"
    fi
    sleep "$CHILD_CHECK_SECONDS"
    checks=$((checks + 1))
  done

  wait_for_exit_receipt "$receipt" && return 0
  [[ -f "$LAUNCHER_LOG" && ! -L "$LAUNCHER_LOG" ]] || return 1
  log_size=$(stat -c '%s' "$LAUNCHER_LOG") || return 1
  (( log_size >= log_offset )) || return 1
  exit_count=$(tail -c "+$((log_offset + 1))" "$LAUNCHER_LOG" 2>/dev/null | awk '/^EXIT_CODE=[0-9]+$/ {count++} END {print count+0}')
  logged_code=$(tail -c "+$((log_offset + 1))" "$LAUNCHER_LOG" 2>/dev/null | awk -F= '/^EXIT_CODE=[0-9]+$/ {value=$2} END {if (value != "") print value; else print "NONE"}')
  [[ "$exit_count" == 1 && "$logged_code" =~ ^[0-9]+$ ]] || return 1
  tmp="$receipt.tmp.recovered.$$"
  {
    printf 'schema=csi-pairs-v6-launcher-child-exit-recovered-v1\n'
    printf 'pid=%s\n' "$expected_wrapper_pid"
    printf 'start_ticks=%s\n' "$expected_wrapper_ticks"
    printf 'launcher_pid=%s\n' "$launcher_pid"
    printf 'launcher_start_ticks=%s\n' "$launcher_ticks"
    printf 'launcher_cmdline_sha256=%s\n' "$launcher_cmd_sha"
    printf 'launcher_session_id=%s\n' "$launcher_session_id"
    printf 'launcher_exit_code=%s\n' "$logged_code"
    printf 'launcher_log_offset=%s\n' "$log_offset"
    printf 'launcher_log_size=%s\n' "$log_size"
    printf 'segment_exit_count=%s\n' "$exit_count"
    printf 'segment_exit_code=%s\n' "$logged_code"
    printf 'recovered_after_wrapper_exit=true\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$tmp"
  atomic_replace "$tmp" "$receipt"
  log "LAUNCHER_EXIT_RECEIPT_RECOVERED pid=$launcher_pid code=$logged_code evidence=$receipt"
}

monitor_current_child() {
  local child_pid
  local start_ticks
  local cmd_sha
  local attempt
  local attempt_dir
  local state
  local code
  local logged_code
  local exit_count
  local receipt
  local checks=0
  load_child_identity || return 97
  child_pid=$(env_value "$CHILD_IDENTITY" pid)
  start_ticks=$(env_value "$CHILD_IDENTITY" start_ticks)
  cmd_sha=$(env_value "$CHILD_IDENTITY" cmdline_sha256)
  attempt=$(env_value "$CHILD_IDENTITY" attempt)
  attempt_dir=$(env_value "$CHILD_IDENTITY" attempt_dir)
  [[ "$attempt" =~ ^[1-9][0-9]*$ ]] || return 97
  [[ "$attempt_dir" == "$STATE_ROOT/attempt-$attempt" && -d "$attempt_dir" && ! -L "$attempt_dir" ]] || return 97
  receipt="$attempt_dir/exit.env"

  if process_identity_matches "$child_pid" "$start_ticks" "$cmd_sha"; then
    if [[ ! -e "$attempt_dir/go" && ! -L "$attempt_dir/go" && ! -e "$receipt" && ! -L "$receipt" ]]; then
      release_child "$attempt_dir/go" || return 97
      log "CHILD_HANDSHAKE_RECOVERED pid=$child_pid attempt=$attempt"
    elif [[ -L "$attempt_dir/go" || ( -e "$attempt_dir/go" && ! -f "$attempt_dir/go" ) ]]; then
      return 97
    fi
    log "CHILD_REATTACH_OR_MONITOR pid=$child_pid start_ticks=$start_ticks attempt=$attempt"
    while process_identity_matches "$child_pid" "$start_ticks" "$cmd_sha"; do
      state=$(awk '{print $3}' "/proc/$child_pid/stat" 2>/dev/null || printf X)
      [[ "$state" != Z ]] || break
      if (( checks == 0 || (checks * CHILD_CHECK_SECONDS) % POLL_SECONDS == 0 )); then
        record_status "$child_pid" "$start_ticks" "$cmd_sha" || return 94
      fi
      sleep "$CHILD_CHECK_SECONDS"
      checks=$((checks + 1))
    done
  elif [[ -e "/proc/$child_pid" ]]; then
    log "CHILD_ORIGINAL_EXITED_PID_NOW_REUSED pid=$child_pid start_ticks=$start_ticks attempt=$attempt"
  else
    log "CHILD_ALREADY_EXITED pid=$child_pid start_ticks=$start_ticks attempt=$attempt"
  fi

  wait_for_exit_receipt "$receipt" || {
    recover_detached_launcher \
      "$attempt_dir" "$child_pid" "$start_ticks" "$cmd_sha" || {
        log "CHILD_EXIT_RECEIPT_MISSING pid=$child_pid attempt=$attempt"
        return 97
      }
  }
  code=$(env_value "$receipt" launcher_exit_code) || return 97
  logged_code=$(env_value "$receipt" segment_exit_code) || return 97
  exit_count=$(env_value "$receipt" segment_exit_count) || return 97
  [[ "$code" =~ ^[0-9]+$ ]] || return 97
  if [[ "$exit_count" != 1 || "$logged_code" != "$code" ]]; then
    log "CHILD_EXIT_EVIDENCE_INVALID pid=$child_pid attempt=$attempt launcher_code=$code logged_code=$logged_code count=$exit_count"
    return 97
  fi
  log "CHILD_EXIT_CODE=$code attempt=$attempt evidence=$receipt"
  return "$code"
}

retire_child_identity() {
  local attempt
  local archived
  load_child_identity || return 0
  attempt=$(env_value "$CHILD_IDENTITY" attempt) || return 1
  archived="$STATE_ROOT/child_identity.attempt-$attempt.handled.env"
  if [[ -e "$archived" || -L "$archived" ]]; then
    cmp -s "$CHILD_IDENTITY" "$archived" || return 1
    mv -- "$CHILD_IDENTITY" "$STATE_ROOT/child_identity.attempt-$attempt.handled-replay-$(date -u +%Y%m%dT%H%M%SZ)-$$.env"
  else
    mv -- "$CHILD_IDENTITY" "$archived"
  fi
}

main() {
  local attempt
  local code
  preflight || return 92
  log "SUPERVISOR_START accepted_sha256=$(sha256sum "$ACCEPTED" | awk '{print $1}')"

  while true; do
    if load_child_identity; then
      monitor_current_child
      code=$?
      if (( code == 0 )); then
        return 0
      fi
      if downstream_started; then
        log "AUTO_RESUME=REFUSED_AFTER_DOWNSTREAM_START child_exit_code=$code"
        return "$code"
      fi
      attempt=$(env_value "$CHILD_IDENTITY" attempt)
      if (( attempt >= MAX_EVALUATION_ATTEMPTS )); then
        log "AUTO_RESUME=EXHAUSTED child_exit_code=$code attempts=$attempt"
        return "$code"
      fi
      log "AUTO_RESUME=EVALUATION_ONLY attempt_next=$((attempt + 1)) wait_seconds=$RETRY_SECONDS"
      retire_child_identity || return 97
      sleep "$RETRY_SECONDS"
      preflight || return 92
      if downstream_started; then
        log "AUTO_RESUME=REFUSED_AFTER_DOWNSTREAM_START child_exit_code=$code"
        return "$code"
      fi
    fi

    if downstream_started; then
      log "AUTO_RESUME=REFUSED_WITHOUT_ACTIVE_CHILD"
      return 95
    fi
    attempt=$(next_attempt_number)
    code=$?
    if (( code == 2 )); then
      log "AUTO_RESUME=EXHAUSTED_PERSISTENT attempts=$MAX_EVALUATION_ATTEMPTS"
      return 96
    elif (( code != 0 )); then
      return 97
    fi
    start_child "$attempt" || {
      log "CHILD_START_FAILED attempt=$attempt"
      if (( attempt >= MAX_EVALUATION_ATTEMPTS )); then
        return 97
      fi
      sleep "$RETRY_SECONDS"
      continue
    }
  done
}

write_supervisor_identity || {
  log SUPERVISOR_REFUSAL=IDENTITY_WRITE_FAILED
  exit 97
}
main
code=$?
log "SUPERVISOR_EXIT_CODE=$code"
write_supervisor_exit "$code"
exit "$code"
