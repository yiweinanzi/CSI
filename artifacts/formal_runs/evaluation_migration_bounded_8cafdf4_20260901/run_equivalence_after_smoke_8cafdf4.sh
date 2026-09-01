#!/bin/bash
set -euo pipefail

evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
oracle_root=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/runs/smoke-full-v6-p8-clusterfix-20260826T230312Z/evaluation
candidate_root="$evidence_root/smoke_candidate_evaluation"
smoke_report="$evidence_root/smoke_equivalence_8cafdf4.json"
subset_report="$evidence_root/real_subset_report_8cafdf4.json"

while [[ ! -f "$candidate_root/manifest.json" ]]; do
  sleep 30
done

if [[ ! -f "$subset_report" ]]; then
  printf 'EQUIVALENCE_WAITER_REFUSAL=real subset report is missing\n' >&2
  exit 93
fi

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
"$runtime_python" -B -m formal_v2.formal_evaluation_compare \
  "$oracle_root" "$candidate_root" \
  --absolute-tolerance 0 \
  --relative-tolerance 0 \
  --output "$smoke_report"

exec "$runtime_python" -B -c \
  'from formal_v2.formal_io import read_strict_json; from formal_v2.formal_migration_evidence import write_equivalence_report; root="/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901"; identity=read_strict_json(root+"/expected_identity.json"); result=write_equivalence_report(root+"/equivalence.json",expected_identity=identity,command="python -B formal_evaluation_compare zero tolerance; python -B write_equivalence_report",smoke_report_path=root+"/smoke_equivalence_8cafdf4.json",formal_subset_report_path=root+"/real_subset_report_8cafdf4.json"); print(result["status"])'
