#!/usr/bin/env bash
set -euo pipefail

runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
new_run_root="$runtime_root/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b"
request="$new_run_root/migration/request.json"
approval=/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8cafdf4-08828d387df6db2b/LLM_JUDGE_MIGRATION_APPROVAL.json
supervisor="$evidence_root/supervise_all_streaming_8cafdf4.sh"

while [[ ! -f "$approval" ]]; do
  sleep 30
done

if [[ ! -f "$request" || -L "$request" || -L "$approval" ]]; then
  printf 'ACCEPT_REFUSAL=request or approval is missing or unsafe\n' >&2
  exit 93
fi

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
"$runtime_python" -B - "$request" "$approval" <<'PY'
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_migration import accept_migration_request
import sys

config = load_formal_config("formal_v2/configs/formal_v2.json")
result = accept_migration_request(
    sys.argv[1],
    sys.argv[2],
    new_run_root="/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b",
    new_source_root="/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/formal_v2",
    protocol_path="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md",
    dataset_path="/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz",
    config_sha256=config_sha256(config),
    expected_seeds=config["seeds"],
)
print("MIGRATION_ACCEPTED=" + result["accepted_path"])
print("MIGRATION_ACCEPTED_SHA256=" + result["accepted_sha256"])
PY

exec "$supervisor"
