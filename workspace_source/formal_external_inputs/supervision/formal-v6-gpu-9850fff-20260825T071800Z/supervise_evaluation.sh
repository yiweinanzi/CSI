#!/usr/bin/env bash
set -uo pipefail

AUDIT_DIR=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-gpu-9850fff-20260825T071800Z
PROJECT_ROOT=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
RUN_ROOT="$PROJECT_ROOT/runs/formal-v6-gpu-9850fff-20260825T071800Z"
PID=66858
EXPECTED_BOOT_ID=fdc6d132-f41a-46ac-a732-c969575124c4
EXPECTED_START_TICKS=466542136
EXPECTED_CWD=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server
EXPECTED_EXE=/root/miniconda3/envs/csi-pairs-formal/bin/python3.12
EXPECTED_CMDLINE='/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python -B -m formal_v2.formal_cli run-evaluation --config formal_v2/configs/formal_v2.json --dataset /root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz --output runs/formal-v6-gpu-9850fff-20260825T071800Z'
EXPECTED_PID_IDENTITY=bcc357b38dcc6fc737a1b4a04c933663160e47713e53960831803845a96fbc09
POLL_SECONDS=600
MONITOR_LOG="$AUDIT_DIR/monitor.tsv"
LATEST_STATE="$AUDIT_DIR/latest_state.env"
SUPERVISOR_LOG="$AUDIT_DIR/supervisor.log"
LOCK_PATH="$AUDIT_DIR/supervisor.lock"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  printf '%s supervisor already active\n' "$(date --iso-8601=seconds)" >>"$SUPERVISOR_LOG"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$SUPERVISOR_LOG"
}

current_cmdline() {
  tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null | sed 's/ $//'
}

identity_status() {
  [[ -r "/proc/$PID/stat" ]] || return 2
  local state start_ticks boot_id cwd exe cmdline
  state=$(awk '{print $3}' "/proc/$PID/stat") || return 1
  [[ "$state" != Z ]] || return 2
  start_ticks=$(awk '{print $22}' "/proc/$PID/stat") || return 1
  boot_id=$(cat /proc/sys/kernel/random/boot_id) || return 1
  cwd=$(readlink -f "/proc/$PID/cwd") || return 1
  exe=$(readlink -f "/proc/$PID/exe") || return 1
  cmdline=$(current_cmdline) || return 1
  [[ "$start_ticks" == "$EXPECTED_START_TICKS" ]] || return 1
  [[ "$boot_id" == "$EXPECTED_BOOT_ID" ]] || return 1
  [[ "$cwd" == "$EXPECTED_CWD" ]] || return 1
  [[ "$exe" == "$EXPECTED_EXE" ]] || return 1
  [[ "$cmdline" == "$EXPECTED_CMDLINE" ]] || return 1
  return 0
}

write_latest() {
  local temporary="$LATEST_STATE.tmp.$$"
  printf '%s\n' "$@" >"$temporary"
  mv -f "$temporary" "$LATEST_STATE"
}

record_running() {
  local timestamp state elapsed cpu_ticks cpu_seconds rss_bytes ppid cwd exe cmdline
  local log_target log_size log_mtime artifact_count artifact_bytes disk_available anomaly
  timestamp=$(date --iso-8601=seconds)
  state=$(awk '{print $3}' "/proc/$PID/stat")
  ppid=$(awk '{print $4}' "/proc/$PID/stat")
  cpu_ticks=$(awk '{print $14 + $15}' "/proc/$PID/stat")
  cpu_seconds=$((cpu_ticks / $(getconf CLK_TCK)))
  elapsed=$(ps -p "$PID" -o etimes= | tr -d ' ')
  rss_bytes=$(awk '$1=="VmRSS:" {print $2 * 1024}' "/proc/$PID/status")
  cwd=$(readlink -f "/proc/$PID/cwd")
  exe=$(readlink -f "/proc/$PID/exe")
  cmdline=$(current_cmdline)
  log_target=$(readlink "/proc/$PID/fd/1" 2>/dev/null || printf 'UNAVAILABLE')
  log_size=0
  log_mtime=NOT_APPLICABLE_PIPE
  artifact_count=$(find "$RUN_ROOT/evaluation" -maxdepth 1 -type f 2>/dev/null | wc -l)
  artifact_bytes=$(find "$RUN_ROOT/evaluation" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  disk_available=$(df -B1 --output=avail "$RUN_ROOT" | tail -1 | tr -d ' ')
  anomaly=NONE
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" MATCH "$state" "$elapsed" "$cpu_seconds" "$rss_bytes" \
    "$log_target" "$artifact_count" "$artifact_bytes" "$disk_available" >>"$MONITOR_LOG"
  write_latest \
    "TIMESTAMP=$timestamp" \
    "PID_IDENTITY=MATCH" \
    "PID_IDENTITY_SHA256=$EXPECTED_PID_IDENTITY" \
    "CURRENT_EVALUATION=RUNNING" \
    "PID=$PID" \
    "PPID=$ppid" \
    "STATE=$state" \
    "ELAPSED_SECONDS=$elapsed" \
    "CPU_SECONDS=$cpu_seconds" \
    "RSS_BYTES=$rss_bytes" \
    "CWD=$cwd" \
    "EXE=$exe" \
    "CMDLINE=$cmdline" \
    "LOG_LOCATION=EXEC_SESSION_60477_PIPE" \
    "LOG_TARGET=$log_target" \
    "LOG_SIZE=$log_size" \
    "LOG_MTIME=$log_mtime" \
    "LAST_PROGRESS_EVIDENCE=CPU_TIME_AND_ACTIVE_PROCESS_STATE" \
    "EVALUATION_ARTIFACT_COUNT=$artifact_count" \
    "EVALUATION_ARTIFACT_BYTES=$artifact_bytes" \
    "DISK_AVAILABLE_BYTES=$disk_available" \
    "PROJECT_MINIMUM_FREE_DISK_BYTES=0" \
    "PROJECT_DISK_THRESHOLD_SOURCE=approval/preflight.json" \
    "ANOMALY=$anomaly"
}

write_identity_snapshot() {
  local target="$AUDIT_DIR/pid_identity.txt"
  {
    printf 'PID=%s\n' "$PID"
    printf 'PID_IDENTITY=MATCH\n'
    printf 'PID_IDENTITY_SHA256=%s\n' "$EXPECTED_PID_IDENTITY"
    printf 'BOOT_ID=%s\n' "$EXPECTED_BOOT_ID"
    printf 'START_TICKS=%s\n' "$EXPECTED_START_TICKS"
    printf 'START_ISO=2026-08-29T10:52:49+0800\n'
    printf 'CWD=%s\n' "$EXPECTED_CWD"
    printf 'EXE=%s\n' "$EXPECTED_EXE"
    printf 'CMDLINE=%s\n' "$EXPECTED_CMDLINE"
    printf 'LOG_LOCATION=EXEC_SESSION_60477_PIPE\n'
    printf 'STDOUT_FD_TARGET=%s\n' "$(readlink "/proc/$PID/fd/1" 2>/dev/null || true)"
    printf 'STDERR_FD_TARGET=%s\n' "$(readlink "/proc/$PID/fd/2" 2>/dev/null || true)"
    ps -p "$PID" -o pid,ppid,state,lstart,etime,time,%cpu,%mem,rss,args
  } >"$target"
}

main() {
  mkdir -p "$AUDIT_DIR"
  if [[ ! -e "$MONITOR_LOG" ]]; then
    printf 'timestamp\tidentity\tstate\telapsed_seconds\tcpu_seconds\trss_bytes\tlog_target\tartifact_count\tartifact_bytes\tdisk_available_bytes\n' >"$MONITOR_LOG"
  fi
  identity_status
  local status=$?
  if [[ "$status" -eq 1 ]]; then
    log "PID_REUSED_OR_MISMATCH pid=$PID; refusing to touch process or continue"
    write_latest \
      "TIMESTAMP=$(date --iso-8601=seconds)" \
      "PID_IDENTITY=PID_REUSED_OR_MISMATCH" \
      "CURRENT_EVALUATION=RUNNING" \
      "OUTPUT_ACCEPTANCE=INCOMPLETE" \
      "ANOMALY=PID_REUSED_OR_MISMATCH"
    return 3
  fi
  if [[ "$status" -eq 2 ]]; then
    log "target PID not present at supervisor start; entering acceptance driver"
    write_latest \
      "TIMESTAMP=$(date --iso-8601=seconds)" \
      "PID_IDENTITY=NOT_PRESENT_AT_START" \
      "CURRENT_EVALUATION=EXITED" \
      "OUTPUT_ACCEPTANCE=INCOMPLETE"
    "$AUDIT_DIR/continue_fixed_goal.sh" >>"$SUPERVISOR_LOG" 2>&1
    return $?
  fi
  write_identity_snapshot
  log "PID identity MATCH; monitoring every $POLL_SECONDS seconds"

  while true; do
    identity_status
    status=$?
    if [[ "$status" -eq 1 ]]; then
      log "PID_REUSED_OR_MISMATCH detected; refusing to touch process or continue"
      write_latest \
        "TIMESTAMP=$(date --iso-8601=seconds)" \
        "PID_IDENTITY=PID_REUSED_OR_MISMATCH" \
        "CURRENT_EVALUATION=RUNNING" \
        "OUTPUT_ACCEPTANCE=INCOMPLETE" \
        "ANOMALY=PID_REUSED_OR_MISMATCH"
      return 3
    fi
    if [[ "$status" -eq 2 ]]; then
      log "verified PID exited; entering evaluation acceptance"
      write_latest \
        "TIMESTAMP=$(date --iso-8601=seconds)" \
        "PID_IDENTITY=MATCH" \
        "PID_IDENTITY_SHA256=$EXPECTED_PID_IDENTITY" \
        "CURRENT_EVALUATION=EXITED" \
        "OUTPUT_ACCEPTANCE=INCOMPLETE"
      "$AUDIT_DIR/continue_fixed_goal.sh" >>"$SUPERVISOR_LOG" 2>&1
      return $?
    fi
    record_running
    sleep "$POLL_SECONDS"
  done
}

main
code=$?
printf '%s SUPERVISOR_EXIT_CODE=%s\n' "$(date --iso-8601=seconds)" "$code" >>"$SUPERVISOR_LOG"
exit "$code"
