#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/formal_v2/external_adapters/.runtime-differt"
VENV="${RUNTIME_ROOT}/venv"
REQUIREMENTS="${PROJECT_ROOT}/formal_v2/external_adapters/requirements-differt-runtime-linux-x86_64.txt"
ENGINE_CONFIG="${PROJECT_ROOT}/formal_v2/configs/differt_external_engine_v1.json"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "DiffeRT formal runtime is frozen only for Linux x86_64" >&2
  exit 2
fi
if [[ -e "${RUNTIME_ROOT}" ]]; then
  echo "refusing to overwrite DiffeRT runtime: ${RUNTIME_ROOT}" >&2
  exit 3
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the DiffeRT runtime" >&2
  exit 4
fi

uv venv --python 3.12.13 "${VENV}"
uv pip install \
  --python "${VENV}/bin/python" \
  --require-hashes \
  --only-binary :all: \
  -r "${REQUIREMENTS}"

PYTHONPATH="${PROJECT_ROOT}" \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
CUDA_VISIBLE_DEVICES='' \
"${VENV}/bin/python" -B \
  "${PROJECT_ROOT}/formal_v2/external_adapters/differt_external_validity.py" \
  --probe-runtime \
  --engine-config "${ENGINE_CONFIG}" \
  >"${RUNTIME_ROOT}/runtime_provenance.json"

printf 'DiffeRT runtime ready: %s\n' "${VENV}/bin/python"
