#!/bin/bash
set -euo pipefail

evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901
runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
new_run_root="$runtime_root/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b"

required=(
  "$evidence_root/performance/performance.json"
  "$evidence_root/equivalence.json"
  "$evidence_root/controls/resume.json"
  "$evidence_root/controls/corrupt_checkpoint.json"
  "$evidence_root/controls/stale_checkpoint.json"
  "$evidence_root/controls/lock.json"
  "$evidence_root/controls/progress.json"
  "$evidence_root/base_to_new_diff.json"
  "$evidence_root/post_exit_freeze/legacy_evaluation_inventory.json"
  "$evidence_root/post_exit_freeze/legacy_evaluation_post_exit_freeze.json"
)

while true; do
  missing=0
  for artifact in "${required[@]}"; do
    if [[ ! -f "$artifact" || -L "$artifact" ]]; then
      missing=1
      break
    fi
  done
  [[ "$missing" == 0 ]] && break
  sleep 30
done

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime_python" -B - <<'PY'
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_io import read_strict_json
from formal_v2.formal_migration import write_migration_request
from formal_v2.formal_migration_evidence import validate_evidence_report

evidence_root = "/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901"
legacy_run = "/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z"
new_run = "/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b"
paths = {
    "performance": f"{evidence_root}/performance/performance.json",
    "equivalence": f"{evidence_root}/equivalence.json",
    "resume": f"{evidence_root}/controls/resume.json",
    "corrupt_checkpoint": f"{evidence_root}/controls/corrupt_checkpoint.json",
    "stale_checkpoint": f"{evidence_root}/controls/stale_checkpoint.json",
    "lock": f"{evidence_root}/controls/lock.json",
    "progress": f"{evidence_root}/controls/progress.json",
    "base_to_new_diff": f"{evidence_root}/base_to_new_diff.json",
    "legacy_evaluation_inventory": f"{evidence_root}/post_exit_freeze/legacy_evaluation_inventory.json",
    "legacy_evaluation_freeze": f"{evidence_root}/post_exit_freeze/legacy_evaluation_post_exit_freeze.json",
}
identity = read_strict_json(f"{evidence_root}/expected_identity.json")
for name, path in paths.items():
    validate_evidence_report(
        name,
        read_strict_json(path),
        expected_identity=identity,
        legacy_run_root=legacy_run,
        new_run_root=new_run,
    )
config = load_formal_config("formal_v2/configs/formal_v2.json")
result = write_migration_request(
    legacy_run_root=legacy_run,
    legacy_source_root="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/formal_v2",
    new_run_root=new_run,
    new_source_root="/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server/formal_v2",
    protocol_path="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md",
    dataset_path="/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz",
    config_sha256=config_sha256(config),
    expected_seeds=config["seeds"],
    migration_nonce="dad7559f9521ac56b2a59429a223d23c8f998cd137530360781a939c2e275669",
    new_run_nonce="08828d387df6db2b28b3f87003d67daca14ec154e54cb716540478b3a630e095",
    new_compute_plan_path=f"{evidence_root}/compute_plan.json",
    migration_evidence_paths=paths,
)
print("MIGRATION_REQUEST=" + result["request_path"])
print("MIGRATION_REQUEST_SHA256=" + result["request_sha256"])
PY
