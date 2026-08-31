#!/usr/bin/env bash
set -uo pipefail

EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
SUPERVISION_ROOT=/root/xunlian/Futaoran/formal_external_inputs/supervision/formal-v6-streaming-8d489b2-d7d0b8affcd40f05/migration-gates
RUNTIME_FORMAL=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server/formal_v2
FROZEN_FORMAL=/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/formal_v2
SUBSET_PID=11014
SUBSET_START_TICKS=484483454
SUBSET_CMD_SHA256=88389d2f73fc31e45824dda7f2012c3bf659236524ac410bd581d0ec1700cabd
PERFORMANCE_PID=225202
PERFORMANCE_START_TICKS=487473290
PERFORMANCE_CMD_SHA256=d1c584d5dbc8a66f6d0ea7069e66cc8fb5d10f939cf88a97f8700bdbcf165360
PRIOR_SUBSET=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_ac83473_20260831/real_subset_report.json
FINAL_SUBSET="$EVIDENCE_ROOT/real_subset_report_8d489b2.json"
PERFORMANCE_REPORT="$EVIDENCE_ROOT/performance/performance.json"
INTERVAL_SECONDS=${CSI_GATE_MONITOR_INTERVAL_SECONDS:-300}

mkdir -p "$SUPERVISION_ROOT"
exec 9>"$SUPERVISION_ROOT/monitor.lock"
if ! flock -n 9; then
  printf '%s MONITOR_ALREADY_ACTIVE\n' "$(date --iso-8601=seconds)"
  exit 0
fi

MONITOR_TSV="$SUPERVISION_ROOT/monitor.tsv"
LATEST="$SUPERVISION_ROOT/latest.env"
if [[ ! -e "$MONITOR_TSV" ]]; then
  printf 'timestamp\tsubset_parent\tprior_subset\tfinal_subset\tperformance_parent\tperformance_report\trefresh_processes\tgpu\tdisk_free_bytes\tpyc_count\tevidence_files\n' >"$MONITOR_TSV"
fi

regular_state() {
  local path=$1
  if [[ -f "$path" && ! -L "$path" ]]; then
    printf 'READY:%s' "$(stat -c '%s' "$path")"
  elif [[ -e "$path" || -L "$path" ]]; then
    printf 'UNSAFE'
  else
    printf 'ABSENT'
  fi
}

process_state() {
  local pid=$1
  local expected_ticks=$2
  local expected_cmd_sha=$3
  local observed_ticks
  local observed_cmd_sha
  local values
  local child
  local child_values
  local python_children=NONE
  if [[ ! -r "/proc/$pid/stat" || ! -r "/proc/$pid/cmdline" ]]; then
    printf 'EXITED'
    return 0
  fi
  observed_ticks=$(awk '{print $22}' "/proc/$pid/stat")
  if [[ "$observed_ticks" != "$expected_ticks" ]]; then
    printf 'PID_MISMATCH:start_ticks=%s' "$observed_ticks"
    return 0
  fi
  observed_cmd_sha=$(tr '\0' ' ' <"/proc/$pid/cmdline" | sha256sum | awk '{print $1}')
  if [[ "$observed_cmd_sha" != "$expected_cmd_sha" ]]; then
    printf 'PID_MISMATCH:cmd_sha256=%s' "$observed_cmd_sha"
    return 0
  fi
  values=$(awk '{printf "utime=%s,stime=%s,rss_pages=%s",$14,$15,$24}' "/proc/$pid/stat")
  while read -r child; do
    [[ -n "$child" && -r "/proc/$child/stat" && -r "/proc/$child/comm" ]] || continue
    [[ "$(<"/proc/$child/comm")" == python* ]] || continue
    child_values=$(awk '{printf "%s/utime=%s/stime=%s/rss_pages=%s",$1,$14,$15,$24}' "/proc/$child/stat")
    if [[ "$python_children" == NONE ]]; then
      python_children=$child_values
    else
      python_children="$python_children|$child_values"
    fi
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  printf 'RUNNING:%s,python_children=%s' "$values" "$python_children"
}

while true; do
  timestamp=$(date --iso-8601=seconds)
  subset_state=$(process_state "$SUBSET_PID" "$SUBSET_START_TICKS" "$SUBSET_CMD_SHA256")
  prior_state=$(regular_state "$PRIOR_SUBSET")
  final_state=$(regular_state "$FINAL_SUBSET")
  performance_state=$(process_state "$PERFORMANCE_PID" "$PERFORMANCE_START_TICKS" "$PERFORMANCE_CMD_SHA256")
  report_state=$(regular_state "$PERFORMANCE_REPORT")
  refresh_processes=$(pgrep -af 'formal_migration_evidence_runner refresh-subset' | awk '$2 ~ /python/ {print $1}' | paste -sd, -)
  [[ -n "$refresh_processes" ]] || refresh_processes=NONE
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n\t' '; ')
  [[ -n "$gpu" ]] || gpu=UNAVAILABLE
  disk_free=$(df -PB1 "$EVIDENCE_ROOT" | awk 'NR == 2 {print $4}')
  pyc_count=$(find "$RUNTIME_FORMAL" "$FROZEN_FORMAL" "$EVIDENCE_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print 2>/dev/null | wc -l)
  evidence_files=$(find "$EVIDENCE_ROOT" -maxdepth 2 -type f -print 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$timestamp" "$subset_state" "$prior_state" "$final_state" \
    "$performance_state" "$report_state" "$refresh_processes" "$gpu" \
    "$disk_free" "$pyc_count" "$evidence_files" >>"$MONITOR_TSV"
  latest_tmp="$LATEST.tmp.$$"
  printf 'timestamp=%q\nsubset_parent=%q\nprior_subset=%q\nfinal_subset=%q\nperformance_parent=%q\nperformance_report=%q\nrefresh_processes=%q\ngpu=%q\ndisk_free_bytes=%q\npyc_count=%q\nevidence_files=%q\n' \
    "$timestamp" "$subset_state" "$prior_state" "$final_state" \
    "$performance_state" "$report_state" "$refresh_processes" "$gpu" \
    "$disk_free" "$pyc_count" "$evidence_files" >"$latest_tmp"
  mv "$latest_tmp" "$LATEST"

  if [[ "$subset_state" == PID_MISMATCH:* || "$performance_state" == PID_MISMATCH:* ]]; then
    printf '%s MONITOR_REFUSAL=PID_IDENTITY_MISMATCH\n' "$timestamp"
    exit 95
  fi
  if [[ "$prior_state" == ABSENT && "$subset_state" == EXITED ]]; then
    printf '%s MONITOR_REFUSAL=SUBSET_EXITED_WITHOUT_PRIOR_REPORT\n' "$timestamp"
    exit 94
  fi
  if [[ "$report_state" == ABSENT && "$performance_state" == EXITED ]]; then
    printf '%s MONITOR_REFUSAL=PERFORMANCE_EXITED_WITHOUT_REPORT\n' "$timestamp"
    exit 94
  fi
  if [[ "$final_state" == READY:* && "$report_state" == READY:* ]]; then
    printf '%s MIGRATION_GATE_REPORTS=READY\n' "$timestamp"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
