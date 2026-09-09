#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIBU_ROOT="${PROJECT_ROOT}/waibu"
RUNTIME_ROOT="${1:-${PROJECT_ROOT}/formal_v2/external_adapters/.runtime-sionna}"
ENV_DIR="${RUNTIME_ROOT}/venv"
SOURCE_DIR="${RUNTIME_ROOT}/src"
FROZEN_LOCK="${PROJECT_ROOT}/formal_v2/external_adapters/sionna_lrm_uv.lock"

if [[ -e "${RUNTIME_ROOT}" && ! -d "${RUNTIME_ROOT}" ]]; then
  echo "refusing to reuse a non-directory Sionna runtime: ${RUNTIME_ROOT}" >&2
  exit 2
fi
command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 3; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 4; }
uv python install 3.12.13
PYTHON312="$(uv python find --managed-python 3.12.13)"

if [[ -z "${DRJIT_LIBLLVM_PATH:-}" ]]; then
  DRJIT_LIBLLVM_PATH="$({ ldconfig -p 2>/dev/null || true; } \
    | awk '$1 ~ /^libLLVM-[0-9]+\.so$/ {print $NF}' \
    | sort -V \
    | tail -n 1)"
fi
if [[ -z "${DRJIT_LIBLLVM_PATH}" || ! -e "${DRJIT_LIBLLVM_PATH}" ]]; then
  echo "Dr.Jit requires libLLVM; set DRJIT_LIBLLVM_PATH to an installed shared library" >&2
  exit 6
fi
export DRJIT_LIBLLVM_PATH

(
  cd "${WAIBU_ROOT}"
  printf '%s  %s\n' \
    'fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7' 'sionna-main.zip' \
    '694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b' 'sionna-large-radio-maps-main.zip' \
    | sha256sum --check --strict
)

mkdir -p "${SOURCE_DIR}"
unzip -oq "${WAIBU_ROOT}/sionna-main.zip" -d "${SOURCE_DIR}"
unzip -oq "${WAIBU_ROOT}/sionna-large-radio-maps-main.zip" -d "${SOURCE_DIR}"
if [[ ! -f "${FROZEN_LOCK}" ]]; then
  echo "frozen Sionna LRM uv.lock is missing: ${FROZEN_LOCK}" >&2
  exit 7
fi
cp "${FROZEN_LOCK}" "${SOURCE_DIR}/sionna-large-radio-maps-main/uv.lock"
mkdir -p \
  "${RUNTIME_ROOT}/data/local/scenes" \
  "${RUNTIME_ROOT}/data/remote/scenes" \
  "${RUNTIME_ROOT}/data/remote/outputs" \
  "${RUNTIME_ROOT}/data/remote/transmitters"

# The GitHub source archive does not carry the sionna-rt git submodule. Both supplied
# projects declare the official wheel; large-radio-maps freezes it to 1.2.1.
if [[ -f "${ENV_DIR}/pyvenv.cfg" ]]; then
  ENV_BASE="$(${ENV_DIR}/bin/python -c 'import os,sys; print(os.path.realpath(sys._base_executable))')"
  if [[ "${ENV_BASE}" != "$(readlink -f "${PYTHON312}")" ]]; then
    uv venv --clear --python "${PYTHON312}" "${ENV_DIR}"
  fi
else
  uv venv --python "${PYTHON312}" "${ENV_DIR}"
fi
installed=false
for attempt in 1 2 3; do
  if UV_PROJECT_ENVIRONMENT="${ENV_DIR}" uv sync \
    --project "${SOURCE_DIR}/sionna-large-radio-maps-main" \
    --locked \
    --no-dev; then
    installed=true
    break
  fi
  echo "Sionna locked dependency sync failed (attempt ${attempt}/3); retrying" >&2
done
if [[ "${installed}" != true ]]; then
  echo "Sionna locked dependency sync failed after 3 attempts" >&2
  exit 5
fi
# G8 uses sionna.rt and also reloads the frozen Stage-0 teacher to reproduce the
# route ledger. Install a fixed CPU build of PyTorch without the unused CUDA stack.
TORCH_CPU_INDEX="${CSI_PAIRS_TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"
uv pip install --python "${ENV_DIR}/bin/python" \
  --index "${TORCH_CPU_INDEX}" \
  'torch==2.9.1+cpu'
uv pip install --python "${ENV_DIR}/bin/python" 'h5py==3.15.1'
# The top-level Sionna source package is installed for authenticated version and
# provenance metadata. RT/LRM dependencies were installed by the project above.
uv pip install --python "${ENV_DIR}/bin/python" --no-deps "${SOURCE_DIR}/sionna-main"
uv pip check --python "${ENV_DIR}/bin/python"

PYTHONPATH="${SOURCE_DIR}/sionna-large-radio-maps-main" \
SLRM_DATA_DIR="${RUNTIME_ROOT}/data" \
"${ENV_DIR}/bin/python" - <<'PY'
import importlib.metadata
import h5py
import sionna
import sionna.rt
import sionna_lrm
import torch
print("sionna", importlib.metadata.version("sionna"))
print("sionna-rt", importlib.metadata.version("sionna-rt"))
print("sionna-large-radio-maps", "source@1ba19ae1df1d26302fcfbaab14efc2347313da5d")
print("torch", torch.__version__)
print("h5py", h5py.__version__)
print("drjit-libllvm", __import__("os").environ["DRJIT_LIBLLVM_PATH"])
PY

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/formal_v2/formal_external_runtime.py" \
  --profile sionna \
  --project-root "${PROJECT_ROOT}" \
  --output "${RUNTIME_ROOT}/runtime_provenance.json" \
  --verify-existing

printf 'Sionna runtime ready: %s\n' "${RUNTIME_ROOT}"
