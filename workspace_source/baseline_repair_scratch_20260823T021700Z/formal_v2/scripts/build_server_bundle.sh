#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_INPUT="${1:?usage: build_server_bundle.sh UNUSED_OUTPUT_ZIP}"
OUTPUT_PARENT="$(cd "$(dirname "${OUTPUT_INPUT}")" && pwd)"
OUTPUT_ZIP="${OUTPUT_PARENT}/$(basename "${OUTPUT_INPUT}")"

if [[ -e "${OUTPUT_ZIP}" || -e "${OUTPUT_ZIP}.sha256" || -e "${OUTPUT_ZIP}.manifest.sha256" ]]; then
  echo "refusing to overwrite server bundle output" >&2
  exit 2
fi
if ! command -v zip >/dev/null 2>&1; then
  echo "zip is required to build the server bundle" >&2
  exit 3
fi

STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/csi-pairs-v2-server.XXXXXX")"
BUNDLE_ROOT="${STAGING_ROOT}/CSI-PAIRS-v2.1-server"
trap 'rm -rf "${STAGING_ROOT}"' EXIT

echo "building internal research delivery; do not submit this bundle as anonymous supplementary" >&2

mkdir -p "${BUNDLE_ROOT}/artifacts" "${BUNDLE_ROOT}/output/pdf" "${BUNDLE_ROOT}/paper/official_style"
mkdir -p "${BUNDLE_ROOT}/artifacts/dataset_suite_v6/external_wireless_metadata"
(
  cd "${PROJECT_ROOT}"
  tar \
    --exclude='formal_v2/external_adapters/.venv-wigatr' \
    --exclude='formal_v2/external_adapters/.runtime-differt' \
    --exclude='formal_v2/external_adapters/.runtime-sionna' \
    --exclude='formal_v2/tests/test_m4_scene0_exact_gate.py' \
    --exclude='formal_v2/tests/test_sionna_visibility_one_factor_diagnostic.py' \
    -cf - formal_v2
) | (
  cd "${BUNDLE_ROOT}"
  tar -xf -
)
(
  cd "${PROJECT_ROOT}"
  PYTHONDONTWRITEBYTECODE=1 python3 -m formal_v2.formal_resources stage-redistributable \
    --registry "${PROJECT_ROOT}/formal_v2/configs/waibu_resources_v1.json" \
    --source-root "${PROJECT_ROOT}/waibu" \
    --destination-root "${BUNDLE_ROOT}/waibu"
)
cp -R "${PROJECT_ROOT}/paper_v2" "${BUNDLE_ROOT}/"
cp -R "${PROJECT_ROOT}/paper/official_style/iclr2027" "${BUNDLE_ROOT}/paper/official_style/"
cp -p "${PROJECT_ROOT}/CSI-PAIRS-startup-package-v2.0.md" \
  "${BUNDLE_ROOT}/CSI-PAIRS-startup-package-v2.0.md"
cp -p "${PROJECT_ROOT}/README.md" "${BUNDLE_ROOT}/README.md"
cp -p "${PROJECT_ROOT}/artifacts/code_to_paper_reverse_matrix.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/formal_experiment_blockers.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/paper_to_code_traceability.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/source_conflict_register.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/v2_0_claim_evidence_contract.json" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/v2_0_verification.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/v6_atomic_requirement_matrix_2026-08-08.csv" \
  "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/iclr2027_official_policy_recheck_2026-08-05.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/waibu_integration_audit_2026-08-06.md" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/v6_traceability_audit_2026-08-07.md" "${BUNDLE_ROOT}/artifacts/"
cp -R "${PROJECT_ROOT}/artifacts/m4_formal_candidate_v2" "${BUNDLE_ROOT}/artifacts/"
cp -R "${PROJECT_ROOT}/artifacts/m4_llvm22_candidate_v1" "${BUNDLE_ROOT}/artifacts/"
cp -R "${PROJECT_ROOT}/artifacts/a100_dataset_package_v6" "${BUNDLE_ROOT}/artifacts/"
cp -p "${PROJECT_ROOT}/artifacts/dataset_suite_v6/ROLE_ASSIGNMENTS.csv" \
  "${BUNDLE_ROOT}/artifacts/dataset_suite_v6/"
cp -p "${PROJECT_ROOT}/artifacts/dataset_suite_v6/DATA_AVAILABILITY.md" \
  "${BUNDLE_ROOT}/artifacts/dataset_suite_v6/"
cp -p "${PROJECT_ROOT}/artifacts/dataset_suite_v6/external_wireless_metadata/SHA256SUMS" \
  "${BUNDLE_ROOT}/artifacts/dataset_suite_v6/external_wireless_metadata/"
cp -p "${PROJECT_ROOT}/output/pdf/CSI-PAIRS-paper-v2.1-draft.pdf" "${BUNDLE_ROOT}/output/pdf/"

find "${BUNDLE_ROOT}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${BUNDLE_ROOT}" -type d -name '*.egg-info' -prune -exec rm -rf {} +
find "${BUNDLE_ROOT}/formal_v2/external_adapters" -maxdepth 1 -type d \
  \( -name '.venv-wigatr' -o -name '.runtime-differt' -o -name '.runtime-sionna' \) \
  -prune -exec rm -rf {} +
find "${BUNDLE_ROOT}" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

cd "${BUNDLE_ROOT}"
find . -type f ! -path ./SHA256SUMS -print | LC_ALL=C sort | while IFS= read -r path; do
  shasum -a 256 "${path}"
done > SHA256SUMS

find "${BUNDLE_ROOT}" -type d -exec chmod 0755 {} +
find "${BUNDLE_ROOT}" -type f -exec chmod 0644 {} +
find "${BUNDLE_ROOT}" -type f -name '*.sh' -exec chmod 0755 {} +
find "${BUNDLE_ROOT}" -exec touch -t 198001010000.00 {} +

cd "${STAGING_ROOT}"
TZ=UTC find CSI-PAIRS-v2.1-server -type f -print | LC_ALL=C sort | TZ=UTC zip -X -q "${OUTPUT_ZIP}" -@
cd "${OUTPUT_PARENT}"
shasum -a 256 "$(basename "${OUTPUT_ZIP}")" > "${OUTPUT_ZIP}.sha256"
MANIFEST_SHA256="$(shasum -a 256 "${BUNDLE_ROOT}/SHA256SUMS" | awk '{print $1}')"
printf '%s  SHA256SUMS\n' "${MANIFEST_SHA256}" > "${OUTPUT_ZIP}.manifest.sha256"

echo "${OUTPUT_ZIP}"
