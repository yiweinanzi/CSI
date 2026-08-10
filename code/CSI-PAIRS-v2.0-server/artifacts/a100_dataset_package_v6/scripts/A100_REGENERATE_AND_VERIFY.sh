#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT_INPUT="${1:?usage: A100_REGENERATE_AND_VERIFY.sh SERVER_ROOT CANDIDATE_ROOT INSPECTION_ROOT VERIFICATION_ROOT}"
CANDIDATE_ROOT_INPUT="${2:?candidate output root is required}"
INSPECTION_ROOT_INPUT="${3:?inspection output root is required}"
VERIFICATION_ROOT_INPUT="${4:?verification output root is required}"

SERVER_ROOT="$(cd "${SERVER_ROOT_INPUT}" && pwd)"
for path in "${CANDIDATE_ROOT_INPUT}" "${INSPECTION_ROOT_INPUT}" "${VERIFICATION_ROOT_INPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite output path: ${path}" >&2
    exit 2
  fi
  parent="$(cd "$(dirname "${path}")" && pwd)"
  case "${path}" in
    "${parent}"/*) ;;
    *) echo "output path could not be resolved safely: ${path}" >&2; exit 2 ;;
  esac
done
CANDIDATE_ROOT="$(cd "$(dirname "${CANDIDATE_ROOT_INPUT}")" && pwd)/$(basename "${CANDIDATE_ROOT_INPUT}")"
INSPECTION_ROOT="$(cd "$(dirname "${INSPECTION_ROOT_INPUT}")" && pwd)/$(basename "${INSPECTION_ROOT_INPUT}")"
VERIFICATION_ROOT="$(cd "$(dirname "${VERIFICATION_ROOT_INPUT}")" && pwd)/$(basename "${VERIFICATION_ROOT_INPUT}")"

"${PACKAGE_ROOT}/scripts/VERIFY_PACKAGE.sh"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "A100 regeneration requires reviewed Linux x86_64" >&2
  exit 3
fi
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required" >&2
  exit 4
}
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#GPU_NAMES[@]} < 2 )); then
  echo "two visible A100 GPUs are required; found ${#GPU_NAMES[@]}" >&2
  exit 5
fi
for index in 0 1; do
  if [[ "${GPU_NAMES[${index}]}" != *A100* ]]; then
    echo "GPU ${index} is not an A100: ${GPU_NAMES[${index}]}" >&2
    exit 5
  fi
done

REGISTRY="${SERVER_ROOT}/formal_v2/configs/sionna_llvm_approved_v1.json"
python3 - "${REGISTRY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
registry = json.loads(path.read_text(encoding="utf-8"))
matches = [
    item
    for item in registry.get("libraries", [])
    if item.get("platform_system") == "Linux"
    and item.get("platform_machine") == "x86_64"
]
if not matches:
    raise SystemExit(
        "BLOCKED: no pre-approved Linux x86_64 libLLVM is registered. "
        "Audit the destination bytes and provenance in a separate reviewed "
        "commit before running setup or generation."
    )
print(f"approved_linux_llvm_entries={len(matches)}")
PY

SIONNA_PYTHON="${SERVER_ROOT}/formal_v2/external_adapters/.runtime-sionna/venv/bin/python"
if [[ ! -x "${SIONNA_PYTHON}" ]]; then
  echo "reviewed Sionna runtime is absent; run formal_v2/external_adapters/setup_sionna.sh after registry approval" >&2
  exit 6
fi
CORE_PYTHON="${SERVER_ROOT}/.venv/bin/python"
if [[ ! -x "${CORE_PYTHON}" ]]; then
  echo "formal core runtime is absent; run formal_v2/scripts/setup_formal_v2.sh" >&2
  exit 7
fi

RAW_OSM="${PACKAGE_ROOT}/primary_raw_inputs/CSI-PAIRS-A100-input-v2/raw_osm"
CSI_PAIRS_SIONNA_PYTHON="${SIONNA_PYTHON}" \
  "${SERVER_ROOT}/formal_v2/scripts/generate_sionna_osm_formal_candidate.sh" \
  "${CANDIDATE_ROOT}" "${RAW_OSM}"

export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="${SERVER_ROOT}" "${CORE_PYTHON}" -B -m formal_v2.formal_cli inspect-data \
  --config "${SERVER_ROOT}/formal_v2/configs/formal_v2.json" \
  --dataset "${CANDIDATE_ROOT}/dataset.npz" \
  --output "${INSPECTION_ROOT}"

PYTHONPATH="${SERVER_ROOT}" "${CORE_PYTHON}" -B -m formal_v2.formal_cli verify-data \
  --config "${SERVER_ROOT}/formal_v2/configs/formal_v2.json" \
  --dataset "${CANDIDATE_ROOT}/dataset.npz" \
  --output "${VERIFICATION_ROOT}" \
  --verifier-manifest "${SERVER_ROOT}/formal_v2/configs/sionna_osm_verifier_v2.json"

python3 - "${VERIFICATION_ROOT}/data_verification/gate.json" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
required_roles = {
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
    "target",
    "external_validation",
}
roles = gate.get("role_status", {})
if (
    gate.get("status") != "PASS"
    or gate.get("passed") is not True
    or gate.get("fixture") is not False
    or gate.get("rtol") != 0.0
    or gate.get("atol") != 0.0
    or set(roles) != required_roles
    or any(value != "PASS" for value in roles.values())
):
    raise SystemExit("fresh A100 candidate failed the all-role zero-tolerance gate")
print("A100_CANDIDATE_REGENERATION=PASS")
print("A100_DATA_VERIFICATION=PASS")
print("FORMAL_TRAINING_READY=NO")
print("NEXT=run independent RT, G1/G2, resource, adapter, G8, qualification, and human approval gates")
PY
