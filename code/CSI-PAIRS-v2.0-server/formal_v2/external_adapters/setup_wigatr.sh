#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_ROOT="${PROJECT_ROOT}/formal_v2/external_adapters/vendor/Wi-GATr"
ENV_DIR="${1:-${PROJECT_ROOT}/formal_v2/external_adapters/.venv-wigatr}"
PYTHON310="${CSI_PAIRS_PYTHON310:-}"

if [[ -e "${ENV_DIR}" && ! -f "${ENV_DIR}/pyvenv.cfg" ]]; then
  echo "refusing to reuse a non-venv Wi-GATr environment path" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to install the locked official Wi-GATr environment" >&2
  exit 3
fi
if [[ -n "${PYTHON310}" ]]; then
  if [[ ! -x "${PYTHON310}" ]]; then
    echo "CSI_PAIRS_PYTHON310 is not executable: ${PYTHON310}" >&2
    exit 4
  fi
else
  uv python install 3.10
  PYTHON310="$(uv python find 3.10)"
fi

if [[ ! -f "${ENV_DIR}/pyvenv.cfg" ]]; then
  uv venv --python "${PYTHON310}" "${ENV_DIR}"
fi
installed=false
for attempt in 1 2 3; do
  if UV_PROJECT_ENVIRONMENT="${ENV_DIR}" uv sync \
    --project "${VENDOR_ROOT}" \
    --locked \
    --no-dev; then
    installed=true
    break
  fi
  echo "Wi-GATr locked dependency sync failed (attempt ${attempt}/3); retrying" >&2
done
if [[ "${installed}" != true ]]; then
  echo "Wi-GATr locked dependency sync failed after 3 attempts" >&2
  exit 5
fi

"${ENV_DIR}/bin/python" - <<'PY'
import gatr
import torch
import torch_geometric
import wigatr
print("Wi-GATr environment installed")
print("cuda_available", torch.cuda.is_available())
print("formal_execution_ready", torch.cuda.is_available())
PY
