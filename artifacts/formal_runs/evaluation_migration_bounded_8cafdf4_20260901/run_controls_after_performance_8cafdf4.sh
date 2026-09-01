#!/bin/bash
set -euo pipefail

evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
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

while [[ ! -f "$evidence_root/performance/performance.json" ]]; do
  if ! same_process "$performance_pid" "$performance_start_ticks"; then
    printf 'CONTROL_WAITER_REFUSAL=performance exited without performance.json\n' >&2
    exit 93
  fi
  sleep 30
done

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime_python" -B -m formal_v2.formal_migration_evidence_runner controls \
  --identity "$evidence_root/expected_identity.json" \
  --config "$runtime_root/formal_v2/configs/formal_v2.json" \
  --dataset /root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz \
  --output-dir "$evidence_root/controls" \
  --performance-report "$evidence_root/performance/performance.json"
