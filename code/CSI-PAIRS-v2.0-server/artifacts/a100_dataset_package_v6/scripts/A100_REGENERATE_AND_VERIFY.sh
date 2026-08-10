#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT_INPUT="${1:?usage: A100_REGENERATE_AND_VERIFY.sh SERVER_ROOT CANDIDATE_ROOT INSPECTION_ROOT VERIFICATION_ROOT TRUSTED_SERVER_MANIFEST_SHA256}"
CANDIDATE_ROOT_INPUT="${2:?candidate output root is required}"
INSPECTION_ROOT_INPUT="${3:?inspection output root is required}"
VERIFICATION_ROOT_INPUT="${4:?verification output root is required}"
TRUSTED_SERVER_MANIFEST_SHA256="${5:?trusted server SHA256SUMS digest is required from the reviewed handoff sidecar}"

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

python3.12 - "${SERVER_ROOT}" "${TRUSTED_SERVER_MANIFEST_SHA256}" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
manifest_path = root / "SHA256SUMS"
if not manifest_path.is_file() or manifest_path.is_symlink():
    raise SystemExit("SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: regular root SHA256SUMS is required")
trusted_manifest_digest = sys.argv[2]
if not re.fullmatch(r"[0-9a-f]{64}", trusted_manifest_digest):
    raise SystemExit("SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: trusted manifest digest is invalid")
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != trusted_manifest_digest:
    raise SystemExit("SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: root SHA256SUMS trust anchor mismatch")

def runtime_path(path: Path) -> bool:
    parts = path.relative_to(root).parts
    return parts[:1] == (".venv",) or parts[:3] == (
        "formal_v2", "external_adapters", ".runtime-sionna"
    )

listed = {}
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  (\./[^\n]+)", line)
    if match is None or match.group(2) in listed:
        raise SystemExit(f"SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: invalid manifest line {line!r}")
    relative = match.group(2)
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise SystemExit(f"SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: unsafe file {relative}")
    listed[relative] = match.group(1)

static_symlinks = [path for path in root.rglob("*") if path.is_symlink() and not runtime_path(path)]
if static_symlinks:
    raise SystemExit(f"SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: static symlinks {static_symlinks[:5]}")
actual = {
    "./" + path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != manifest_path and not runtime_path(path)
}
if set(listed) != actual:
    raise SystemExit(
        "SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: inventory mismatch "
        f"missing={sorted(actual - set(listed))[:5]} unexpected={sorted(set(listed) - actual)[:5]}"
    )
for relative, expected in listed.items():
    digest = hashlib.sha256()
    with (root / relative).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"SERVER_BUNDLE_STATIC_INTEGRITY=FAIL: SHA-256 mismatch {relative}")
print(f"SERVER_BUNDLE_STATIC_INTEGRITY=PASS files={len(listed)}")
PY

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
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
echo "REGENERATION_BACKEND=llvm_ad_mono_polarized"
echo "GPU_ROLE=HOST_PROVENANCE_NOT_COMPUTE_CLAIM"

REGISTRY="${SERVER_ROOT}/formal_v2/configs/sionna_llvm_approved_v1.json"
python3.12 - "${REGISTRY}" <<'PY'
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

PYTHONPATH="${SERVER_ROOT}" "${CORE_PYTHON}" -B - \
  "${SERVER_ROOT}/formal_v2/configs/formal_v2.json" \
  "${CANDIDATE_ROOT}/dataset.npz" \
  "${VERIFICATION_ROOT}" <<'PY'
import sys
from pathlib import Path

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_data_verification import require_verified_roles_from_root
from formal_v2.formal_dataset import FormalDataset, SOURCE_ROLES

config = load_formal_config(Path(sys.argv[1]))
dataset = FormalDataset.load(
    Path(sys.argv[2]),
    require_clean_csi=bool(config["data"]["require_clean_csi"]),
)
roles = (*SOURCE_ROLES, "target", "external_validation")
require_verified_roles_from_root(Path(sys.argv[3]), config, dataset, roles)
print("A100_HOST_LLVM_CANDIDATE_REGENERATION=PASS")
print("A100_DATA_VERIFICATION=PASS")
print("FORMAL_TRAINING_READY=NO")
print("NEXT=run independent RT, G1/G2, resource, adapter, G8, qualification, and human approval gates")
PY
