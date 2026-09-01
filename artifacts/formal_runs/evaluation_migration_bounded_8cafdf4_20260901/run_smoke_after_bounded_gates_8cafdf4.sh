#!/bin/bash
set -euo pipefail

evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
real_subset_pid=480098
real_subset_start_ticks=491876418
performance_pid=492822
performance_start_ticks=492045496

same_process() {
  local pid=$1
  local expected_ticks=$2
  local observed_ticks
  [[ -r "/proc/$pid/stat" ]] || return 1
  observed_ticks=$(awk '{print $22}' "/proc/$pid/stat")
  [[ "$observed_ticks" == "$expected_ticks" ]]
}

while same_process "$real_subset_pid" "$real_subset_start_ticks" || \
      same_process "$performance_pid" "$performance_start_ticks"; do
  sleep 30
done

if [[ ! -f "$evidence_root/real_subset_report_8cafdf4.json" ]]; then
  printf 'SMOKE_WAITER_REFUSAL=real subset exited without report\n' >&2
  exit 93
fi
if [[ ! -f "$evidence_root/performance/performance.json" ]]; then
  printf 'SMOKE_WAITER_REFUSAL=performance exited without report\n' >&2
  exit 94
fi
if [[ -e "$evidence_root/smoke_candidate_evaluation" ]]; then
  printf 'SMOKE_WAITER_REFUSAL=candidate output already exists\n' >&2
  exit 95
fi

exec "$evidence_root/run_smoke_candidate_8cafdf4.sh"
