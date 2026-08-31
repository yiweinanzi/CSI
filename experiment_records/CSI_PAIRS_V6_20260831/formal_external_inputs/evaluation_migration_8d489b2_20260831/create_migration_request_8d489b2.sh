#!/bin/bash
set -euo pipefail

evidence_root=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
runtime_root=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
runtime_python=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
runtime_probe="$evidence_root/probe_wigatr_runtime_8d489b2.sh"
runtime_report="$evidence_root/wigatr_runtime_preflight.json"
runtime_probe_sha256=099901cc3b68a7021728ce37c74a037e77f3047a24200da966e899b6f7debf0f
gate_validator="$evidence_root/validate_migration_gates_8d489b2.sh"
gate_validator_sha256=dbf185d225711803aeab886da1a1f56c267eb0af39c17d0912dc5c5357f65b05

required_evidence=(
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
for artifact in "${required_evidence[@]}"; do
  if [[ ! -f "$artifact" || -L "$artifact" ]]; then
    printf 'REFUSAL=required migration evidence is missing or unsafe: %s\n' "$artifact" >&2
    exit 93
  fi
done

if [[ ! -f "$gate_validator" || -L "$gate_validator" || ! -x "$gate_validator" ]]; then
  printf 'REFUSAL=migration gate validator is missing or unsafe: %s\n' "$gate_validator" >&2
  exit 94
fi
if [[ "$(sha256sum "$gate_validator" | awk '{print $1}')" != "$gate_validator_sha256" ]]; then
  printf 'REFUSAL=migration gate validator SHA-256 mismatch: %s\n' "$gate_validator" >&2
  exit 95
fi
"$gate_validator" all

if [[ ! -f "$runtime_probe" || -L "$runtime_probe" || ! -x "$runtime_probe" ]]; then
  printf 'REFUSAL=Wi-GATr runtime preflight is missing or unsafe: %s\n' "$runtime_probe" >&2
  exit 96
fi
if [[ "$(sha256sum "$runtime_probe" | awk '{print $1}')" != "$runtime_probe_sha256" ]]; then
  printf 'REFUSAL=Wi-GATr runtime preflight SHA-256 mismatch: %s\n' "$runtime_probe" >&2
  exit 97
fi
"$runtime_probe" write
if [[ ! -f "$runtime_report" || -L "$runtime_report" ]]; then
  printf 'REFUSAL=Wi-GATr runtime preflight report is missing or unsafe: %s\n' "$runtime_report" >&2
  exit 98
fi

cd "$runtime_root"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1

exec "$runtime_python" -B - <<'PY'
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_evidence import config_sha256
from formal_v2.formal_migration import write_migration_request

evidence_root = "/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831"
config = load_formal_config("formal_v2/configs/formal_v2.json")
result = write_migration_request(
    legacy_run_root="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/runs/formal-v6-gpu-9850fff-20260825T071800Z",
    legacy_source_root="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/code/CSI-PAIRS-v2.0-server/formal_v2",
    new_run_root="/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server/runs/formal-v6-streaming-8d489b2-d7d0b8affcd40f05",
    new_source_root="/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server/formal_v2",
    protocol_path="/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/Idea1-CSI-PAIRS-冻结版-零基础阅读稿-v6_VSCode兼容版.md",
    dataset_path="/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz",
    config_sha256=config_sha256(config),
    expected_seeds=config["seeds"],
    migration_nonce="fddf27b3145cc43bde6355b08136196ef34cb554cbdbb9f591b06e8daf192a90",
    new_run_nonce="d7d0b8affcd40f0503f800f7501d7da1573cc752460cda16e772f43af159ac2a",
    new_compute_plan_path=f"{evidence_root}/compute_plan.json",
    migration_evidence_paths={
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
    },
)
print(f"MIGRATION_REQUEST={result['request_path']}")
print(f"MIGRATION_REQUEST_SHA256={result['request_sha256']}")
PY
