#!/usr/bin/env bash
# ARCHIVED Autodl snapshot — do not use as entry. Portable entry: formal_v2/scripts/run_formal_v2.sh (or launch_engineering_evaluation.sh for P0).
set -euo pipefail

runtime_root="${CSI_PAIRS_ROOT:-/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server}"
runtime_python="${CSI_PAIRS_PYTHON:-/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python}"
evidence_root="${CSI_PAIRS_EVIDENCE_ROOT:-/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901}"
new_run_root="${CSI_PAIRS_FORMAL_OUTPUT:-$runtime_root/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b}"
request="$new_run_root/migration/request.json"
approval="${CSI_PAIRS_SECOND_JUDGE_APPROVAL:-/root/xunlian/Futaoran/formal_external_inputs/migration_approvals/formal-v6-streaming-8cafdf4-08828d387df6db2b/LLM_JUDGE_MIGRATION_APPROVAL.json}"
supervisor="$evidence_root/supervise_all_streaming_8cafdf4.sh"

if [[ "${CSI_PAIRS_REQUIRE_SECOND_JUDGE:-0}" == "1" ]]; then
  while [[ ! -f "$approval" ]]; do
    sleep 30
  done
else
  printf 'SECOND_JUDGE=optional_publication_review skipped\n'
fi

export CSI_PAIRS_FORMAL_OUTPUT="$new_run_root"
export CSI_PAIRS_ROOT="$runtime_root"
export CSI_PAIRS_PYTHON="$runtime_python"
export CSI_PAIRS_DATASET="${CSI_PAIRS_DATASET:-/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz}"

if [[ "${CSI_PAIRS_REQUIRE_SECOND_JUDGE:-0}" == "1" ]]; then
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
import os
import sys

config = load_formal_config("formal_v2/configs/formal_v2.json")
result = accept_migration_request(
    sys.argv[1],
    sys.argv[2],
    new_run_root=os.environ["CSI_PAIRS_FORMAL_OUTPUT"],
    new_source_root=os.environ.get("CSI_PAIRS_NEW_SOURCE", os.environ["CSI_PAIRS_ROOT"] + "/formal_v2"),
    protocol_path=os.environ.get(
        "CSI_PAIRS_PROTOCOL",
        "/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md",
    ),
    dataset_path=os.environ["CSI_PAIRS_DATASET"],
    config_sha256=config_sha256(config),
    expected_seeds=config["seeds"],
)
print("MIGRATION_ACCEPTED=" + result["accepted_path"])
print("MIGRATION_ACCEPTED_SHA256=" + result["accepted_sha256"])
PY
fi

exec "$supervisor"
