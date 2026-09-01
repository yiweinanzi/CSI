#!/bin/bash
# ARCHIVED Autodl snapshot — do not use as entry. Portable entry: formal_v2/scripts/run_formal_v2.sh (or launch_engineering_evaluation.sh for P0).
set -euo pipefail

evidence_root="${CSI_PAIRS_EVIDENCE_ROOT:-/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_bounded_8cafdf4_20260901}"
runtime_root="${CSI_PAIRS_ROOT:-/root/xunlian/Futaoran/CSI_EVALUATION_BOUNDED_EVIDENCE_20260901/code/CSI-PAIRS-v2.0-server}"
runtime_python="${CSI_PAIRS_PYTHON:-/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python}"
new_run_root="${CSI_PAIRS_FORMAL_OUTPUT:-$runtime_root/runs/formal-v6-streaming-8cafdf4-08828d387df6db2b}"
launch_tier="${CSI_PAIRS_LAUNCH_TIER:-p0}"

p0_required=(
  "$evidence_root/expected_identity.json"
  "$evidence_root/equivalence.json"
  "$evidence_root/post_exit_freeze/legacy_evaluation_inventory.json"
)
p1_required=(
  "$evidence_root/performance/performance.json"
  "$evidence_root/controls/resume.json"
  "$evidence_root/controls/corrupt_checkpoint.json"
  "$evidence_root/controls/stale_checkpoint.json"
  "$evidence_root/controls/lock.json"
  "$evidence_root/controls/progress.json"
  "$evidence_root/base_to_new_diff.json"
  "$evidence_root/post_exit_freeze/legacy_evaluation_post_exit_freeze.json"
)
required=("${p0_required[@]}")
if [[ "${launch_tier}" == "p1" || "${CSI_PAIRS_REQUIRE_P1:-0}" == "1" ]]; then
  required+=("${p1_required[@]}")
fi

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

if [[ "${launch_tier}" != "p1" && "${CSI_PAIRS_REQUIRE_P1:-0}" != "1" ]]; then
  printf 'P0_READY=identity+equivalence+inventory\n'
  printf 'ENGINEERING_ENTRY=formal_v2/scripts/launch_engineering_evaluation.sh\n'
  printf 'FORMAL_ENTRY=formal_v2/scripts/run_formal_v2.sh\n'
  printf 'PUBLICATION_REQUEST=set CSI_PAIRS_LAUNCH_TIER=p1 to write a 10-receipt migration request\n'
  exit 0
fi

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
export CSI_PAIRS_EVIDENCE_ROOT="$evidence_root"
export CSI_PAIRS_FORMAL_OUTPUT="$new_run_root"
export CSI_PAIRS_DATASET="${CSI_PAIRS_DATASET:-/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz}"
export CSI_PAIRS_LEGACY_RUN="${CSI_PAIRS_LEGACY_RUN:-/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z}"
export CSI_PAIRS_LEGACY_SOURCE="${CSI_PAIRS_LEGACY_SOURCE:-/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/formal_v2}"
export CSI_PAIRS_NEW_SOURCE="${CSI_PAIRS_NEW_SOURCE:-$runtime_root/formal_v2}"
export CSI_PAIRS_PROTOCOL="${CSI_PAIRS_PROTOCOL:-/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md}"
export CSI_PAIRS_LAUNCH_TIER="$launch_tier"
exec "$runtime_python" -B - <<'PY'
import os

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_io import read_strict_json
from formal_v2.formal_migration import write_migration_request
from formal_v2.formal_migration_evidence import validate_evidence_report

evidence_root = os.environ["CSI_PAIRS_EVIDENCE_ROOT"]
legacy_run = os.environ["CSI_PAIRS_LEGACY_RUN"]
new_run = os.environ["CSI_PAIRS_FORMAL_OUTPUT"]
tier = os.environ.get("CSI_PAIRS_LAUNCH_TIER", "p0")
paths = {
    "equivalence": f"{evidence_root}/equivalence.json",
    "legacy_evaluation_inventory": f"{evidence_root}/post_exit_freeze/legacy_evaluation_inventory.json",
}
if tier == "p1" or os.environ.get("CSI_PAIRS_REQUIRE_P1") == "1":
    paths.update(
        {
            "performance": f"{evidence_root}/performance/performance.json",
            "resume": f"{evidence_root}/controls/resume.json",
            "corrupt_checkpoint": f"{evidence_root}/controls/corrupt_checkpoint.json",
            "stale_checkpoint": f"{evidence_root}/controls/stale_checkpoint.json",
            "lock": f"{evidence_root}/controls/lock.json",
            "progress": f"{evidence_root}/controls/progress.json",
            "base_to_new_diff": f"{evidence_root}/base_to_new_diff.json",
            "legacy_evaluation_freeze": f"{evidence_root}/post_exit_freeze/legacy_evaluation_post_exit_freeze.json",
        }
    )
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
    legacy_source_root=os.environ["CSI_PAIRS_LEGACY_SOURCE"],
    new_run_root=new_run,
    new_source_root=os.environ["CSI_PAIRS_NEW_SOURCE"],
    protocol_path=os.environ["CSI_PAIRS_PROTOCOL"],
    dataset_path=os.environ["CSI_PAIRS_DATASET"],
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
