#!/usr/bin/env bash
set -u

SUPERVISION_ROOT=/root/xunlian/Futaoran/formal_external_inputs/supervision/evaluation-migration-bounded-8cafdf4-20260901
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
SOURCE_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/formal_v2
REAL_PID=480098
REAL_START_TICKS=491876418
REAL_CMDLINE_SHA256=a8f827625397243045bdaf8ec4a97547483a6127b4fad1ba31a2c1e08b7f7b86
PERFORMANCE_PID=492822
PERFORMANCE_START_TICKS=492045496
PERFORMANCE_CMDLINE_SHA256=3a33e1af1ac04740181c2ea149c0c3dc8b099890b8eb1c5fefc2a9e4b28d9382
REAL_REPORT="$EVIDENCE_ROOT/real_subset_report_8cafdf4.json"
PERFORMANCE_REPORT="$EVIDENCE_ROOT/performance/performance.json"
POLL_SECONDS=300

mkdir -p "$SUPERVISION_ROOT"
exec 9>"$SUPERVISION_ROOT/monitor.lock"
if ! flock -n 9; then
  printf '%s MONITOR_ALREADY_ACTIVE\n' "$(date --iso-8601=seconds)"
  exit 0
fi

LOG="$SUPERVISION_ROOT/monitor.tsv"
IDENTITY="$SUPERVISION_ROOT/identity.env"
EXIT_RECEIPT="$SUPERVISION_ROOT/exit.env"

process_identity_matches() {
  local pid=$1
  local expected_ticks=$2
  local expected_cmdline_sha256=$3
  local observed_ticks
  local observed_cmdline_sha256
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
  observed_ticks=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null) || return 1
  [[ "$observed_ticks" == "$expected_ticks" ]] || return 1
  observed_cmdline_sha256=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | sha256sum | awk '{print $1}') || return 1
  [[ "$observed_cmdline_sha256" == "$expected_cmdline_sha256" ]]
}

write_identity() {
  local temp="$IDENTITY.tmp.$$"
  {
    printf 'schema=csi-pairs-v6-bounded-evidence-monitor-v1\n'
    printf 'real_pid=%s\n' "$REAL_PID"
    printf 'real_start_ticks=%s\n' "$REAL_START_TICKS"
    printf 'real_cmdline_sha256=%s\n' "$REAL_CMDLINE_SHA256"
    printf 'performance_pid=%s\n' "$PERFORMANCE_PID"
    printf 'performance_start_ticks=%s\n' "$PERFORMANCE_START_TICKS"
    printf 'performance_cmdline_sha256=%s\n' "$PERFORMANCE_CMDLINE_SHA256"
    printf 'monitor_pid=%s\n' "$$"
    printf 'monitor_start_ticks=%s\n' "$(awk '{print $22}' "/proc/$$/stat")"
    printf 'monitor_sha256=%s\n' "$(sha256sum "$0" | awk '{print $1}')"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$temp"
  mv -f -- "$temp" "$IDENTITY"
}

write_exit() {
  local status=$1
  local code=$2
  local temp="$EXIT_RECEIPT.tmp.$$"
  {
    printf 'schema=csi-pairs-v6-bounded-evidence-monitor-exit-v1\n'
    printf 'status=%s\n' "$status"
    printf 'exit_code=%s\n' "$code"
    printf 'real_report=%s\n' "$(test -f "$REAL_REPORT" && printf PRESENT || printf MISSING)"
    printf 'performance_report=%s\n' "$(test -f "$PERFORMANCE_REPORT" && printf PRESENT || printf MISSING)"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  } >"$temp"
  mv -f -- "$temp" "$EXIT_RECEIPT"
}

sample_process() {
  local pid=$1
  if [[ ! -r "/proc/$pid/stat" ]]; then
    printf 'EXITED\tNA\tNA'
    return
  fi
  awk '{printf "%s\t%s\t%s", $3, $14+$15, $24*4096}' "/proc/$pid/stat"
}

sample_children() {
  local parent=$1
  ps -e -o pid=,ppid=,stat=,time=,rss= 2>/dev/null \
    | awk -v expected="$parent" '$2 == expected {printf "%s:%s:%s:%sMiB,", $1,$3,$4,int($5/1024)}'
}

write_identity
if [[ ! -f "$LOG" ]]; then
  printf 'timestamp\treal_identity\treal_state\treal_ticks\treal_rss_bytes\treal_children\tperformance_identity\tperformance_state\tperformance_ticks\tperformance_rss_bytes\tperformance_children\treal_report\tperformance_report\tmemory_available_kib\tdisk_available_bytes\tpyc_count\tgpu_summary\n' >"$LOG"
fi

while true; do
  real_identity=MISSING
  performance_identity=MISSING
  real_sample=$(sample_process "$REAL_PID")
  performance_sample=$(sample_process "$PERFORMANCE_PID")
  IFS=$'\t' read -r real_state real_ticks real_rss_bytes <<<"$real_sample"
  IFS=$'\t' read -r performance_state performance_ticks performance_rss_bytes <<<"$performance_sample"
  real_children=$(sample_children "$REAL_PID")
  performance_children=$(sample_children "$PERFORMANCE_PID")
  process_identity_matches "$REAL_PID" "$REAL_START_TICKS" "$REAL_CMDLINE_SHA256" && real_identity=MATCH
  process_identity_matches "$PERFORMANCE_PID" "$PERFORMANCE_START_TICKS" "$PERFORMANCE_CMDLINE_SHA256" && performance_identity=MATCH
  real_report=$(test -f "$REAL_REPORT" && printf PRESENT || printf MISSING)
  performance_report=$(test -f "$PERFORMANCE_REPORT" && printf PRESENT || printf MISSING)
  memory_available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  disk_available=$(df -PB1 /root/xunlian/Futaoran | awk 'NR == 2 {print $4}')
  pyc_count=$(find "$SOURCE_ROOT" "$EVIDENCE_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | wc -l)
  gpu_summary=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')
  {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$real_identity" "$real_state" "$real_ticks" "$real_rss_bytes" "$real_children" \
      "$performance_identity" "$performance_state" "$performance_ticks" "$performance_rss_bytes" "$performance_children" \
      "$real_report" "$performance_report" "$memory_available" "$disk_available" "$pyc_count" "$gpu_summary"
  } >>"$LOG"

  if [[ -e "/proc/$REAL_PID" && "$real_identity" != MATCH ]]; then
    write_exit PID_REUSED_OR_MISMATCH 94
    exit 94
  fi
  if [[ -e "/proc/$PERFORMANCE_PID" && "$performance_identity" != MATCH ]]; then
    write_exit PID_REUSED_OR_MISMATCH 95
    exit 95
  fi
  if [[ "$real_identity" == MISSING && "$real_report" == MISSING ]]; then
    write_exit REAL_EXITED_WITHOUT_REPORT 93
    exit 93
  fi
  if [[ "$performance_identity" == MISSING && "$performance_report" == MISSING ]]; then
    write_exit PERFORMANCE_EXITED_WITHOUT_REPORT 96
    exit 96
  fi
  if [[ "$real_identity" == MISSING && "$performance_identity" == MISSING && "$real_report" == PRESENT && "$performance_report" == PRESENT ]]; then
    write_exit REPORTS_PRESENT 0
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
