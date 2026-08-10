#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_INPUT="${1:?usage: build_anonymous_supplement.sh UNUSED_OUTPUT_ZIP}"
OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_INPUT}")" && pwd)"
OUTPUT_ZIP="${OUTPUT_PARENT}/$(basename "${OUTPUT_INPUT}")"

if [[ -e "${OUTPUT_ZIP}" || -e "${OUTPUT_ZIP}.sha256" ]]; then
  echo "refusing to overwrite anonymous supplement output" >&2
  exit 2
fi
if ! command -v zip >/dev/null 2>&1; then
  echo "zip is required to build the anonymous supplement" >&2
  exit 3
fi

STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/csi-pairs-anonymous.XXXXXX")"
BUNDLE_ROOT="${STAGING_ROOT}/CSI-PAIRS-anonymous-supplement"
trap 'rm -rf "${STAGING_ROOT}"' EXIT

mkdir -p "${BUNDLE_ROOT}/paper/official_style"
(
  cd "${PROJECT_ROOT}"
  tar \
    --exclude='formal_v2/external_adapters/.venv-wigatr' \
    --exclude='formal_v2/external_adapters/.runtime-sionna' \
    --exclude='formal_v2/scripts/build_server_bundle.sh' \
    --exclude='formal_v2/scripts/build_anonymous_supplement.sh' \
    --exclude='formal_v2/scripts/build_v6_requirement_matrix.py' \
    --exclude='formal_v2/scripts/v6_trace_registry.py' \
    --exclude='formal_v2/anonymous_release.py' \
    --exclude='formal_v2/configs/a100_dataset_suite_v6.json' \
    --exclude='formal_v2/external_data_bundle' \
    --exclude='formal_v2/tests/test_anonymous_release.py' \
    --exclude='formal_v2/tests/test_a100_dataset_package_v6.py' \
    --exclude='formal_v2/tests/test_audit_artifacts.py' \
    --exclude='formal_v2/tests/test_m4_scene0_exact_gate.py' \
    --exclude='formal_v2/tests/test_m4_candidate_evidence.py' \
    --exclude='formal_v2/tests/test_sionna_visibility_one_factor_diagnostic.py' \
    -cf - formal_v2
) | (
  cd "${BUNDLE_ROOT}"
  tar -xf -
)
cp -R "${PROJECT_ROOT}/paper_v2" "${BUNDLE_ROOT}/"
cp -R "${PROJECT_ROOT}/paper/official_style/iclr2027" "${BUNDLE_ROOT}/paper/official_style/"
cp -p "${PROJECT_ROOT}/formal_v2/ANONYMOUS_SUPPLEMENT.md" "${BUNDLE_ROOT}/README.md"

find "${BUNDLE_ROOT}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${BUNDLE_ROOT}" -type d -name '*.egg-info' -prune -exec rm -rf {} +
find "${BUNDLE_ROOT}" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

python3 "${PROJECT_ROOT}/formal_v2/anonymous_release.py" \
  --tree "${BUNDLE_ROOT}" \
  --project-repository "${PROJECT_ROOT}"

cd "${BUNDLE_ROOT}"
find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | while IFS= read -r path; do
  shasum -a 256 "${path}"
done > SHA256SUMS
find "${BUNDLE_ROOT}" -type d -exec chmod 0755 {} +
find "${BUNDLE_ROOT}" -type f -exec chmod 0644 {} +
find "${BUNDLE_ROOT}" -type f -name '*.sh' -exec chmod 0755 {} +
find "${BUNDLE_ROOT}" -exec touch -t 198001010000.00 {} +

cd "${STAGING_ROOT}"
TZ=UTC find CSI-PAIRS-anonymous-supplement -type f -print | LC_ALL=C sort | TZ=UTC zip -X -q "${OUTPUT_ZIP}" -@
python3 "${PROJECT_ROOT}/formal_v2/anonymous_release.py" \
  --zip "${OUTPUT_ZIP}" \
  --project-repository "${PROJECT_ROOT}"
cd "${OUTPUT_PARENT}"
shasum -a 256 "$(basename "${OUTPUT_ZIP}")" > "${OUTPUT_ZIP}.sha256"

echo "${OUTPUT_ZIP}"
