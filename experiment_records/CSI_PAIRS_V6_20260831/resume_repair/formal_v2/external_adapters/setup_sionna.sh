#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIBU_ROOT="${PROJECT_ROOT}/waibu"
RUNTIME_ROOT="${1:-${PROJECT_ROOT}/formal_v2/external_adapters/.runtime-sionna}"
ENV_DIR="${RUNTIME_ROOT}/venv"
SOURCE_DIR="${RUNTIME_ROOT}/src"
FROZEN_LOCK="${PROJECT_ROOT}/formal_v2/external_adapters/sionna_lrm_uv.lock"
SYSTEM="$(uname -s)"
MACHINE="$(uname -m)"

case "${SYSTEM}:${MACHINE}" in
  Darwin:arm64)
    SIONNA_RUNTIME_LOCK="${PROJECT_ROOT}/formal_v2/requirements-sionna-runtime-darwin-arm64.txt"
    ;;
  Linux:x86_64)
    SIONNA_RUNTIME_LOCK="${PROJECT_ROOT}/formal_v2/requirements-sionna-runtime-linux-x86_64.txt"
    ;;
  *)
    echo "unsupported Sionna runtime target: ${SYSTEM} ${MACHINE}" >&2
    exit 8
    ;;
esac

if [[ -e "${RUNTIME_ROOT}" && ! -d "${RUNTIME_ROOT}" ]]; then
  echo "refusing to reuse a non-directory Sionna runtime: ${RUNTIME_ROOT}" >&2
  exit 2
fi
command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 3; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 4; }
uv python install 3.12.13
PYTHON312="$(uv python find --managed-python 3.12.13)"

if [[ -z "${DRJIT_LIBLLVM_PATH:-}" ]]; then
  if [[ "${SYSTEM}" == "Darwin" ]]; then
    BREW_BIN="${CSI_PAIRS_BREW:-$(command -v brew || true)}"
    if [[ -z "${BREW_BIN}" || ! -x "${BREW_BIN}" ]]; then
      echo "macOS requires Homebrew llvm@18; set CSI_PAIRS_BREW to the brew executable" >&2
      exit 6
    fi
    LLVM_PREFIX="$("${BREW_BIN}" --prefix llvm@18)"
    DRJIT_LIBLLVM_PATH="$("${PYTHON312}" - "${LLVM_PREFIX}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "lib"
for candidate in sorted(root.glob("libLLVM*.dylib")):
    resolved = candidate.resolve()
    if resolved.is_file():
        print(resolved)
        break
PY
)"
  else
    DRJIT_LIBLLVM_PATH="$({ ldconfig -p 2>/dev/null || true; } \
      | awk '$1 ~ /^libLLVM-[0-9]+\.so$/ {print $NF}' \
      | sort -V \
      | tail -n 1)"
  fi
fi
if [[ -n "${DRJIT_LIBLLVM_PATH:-}" ]]; then
  DRJIT_LIBLLVM_PATH="$("${PYTHON312}" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${DRJIT_LIBLLVM_PATH}")"
fi
if [[ -z "${DRJIT_LIBLLVM_PATH:-}" || ! -f "${DRJIT_LIBLLVM_PATH}" ]]; then
  echo "Dr.Jit requires libLLVM; set DRJIT_LIBLLVM_PATH to an installed shared library" >&2
  exit 6
fi
export DRJIT_LIBLLVM_PATH
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON312}" -m formal_v2.sionna_runtime_lock \
  --project-root "${PROJECT_ROOT}" \
  --runtime-root "${RUNTIME_ROOT}" \
  register --libllvm "${DRJIT_LIBLLVM_PATH}"

"${PYTHON312}" - "${WAIBU_ROOT}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "sionna-main.zip": "fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7",
    "sionna-large-radio-maps-main.zip": "694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b",
}
for name, digest in expected.items():
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"missing regular Sionna source archive: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise SystemExit(f"Sionna source archive checksum mismatch: {path}")
    print(f"{name}: OK")
PY

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
  PYTHON312_REAL="$("${PYTHON312}" -c 'import os,sys; print(os.path.realpath(sys.executable))')"
  if [[ "${ENV_BASE}" != "${PYTHON312_REAL}" ]]; then
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
# G8 reloads the frozen Stage-0 teacher. Install the exact reviewed Torch and
# h5py wheels without allowing dependency resolution to select mutable bytes.
uv pip install --python "${ENV_DIR}/bin/python" \
  --require-hashes \
  --no-deps \
  --requirement "${SIONNA_RUNTIME_LOCK}"
# The top-level Sionna source package is installed for authenticated version and
# provenance metadata. RT/LRM dependencies were installed by the project above.
uv pip install --python "${ENV_DIR}/bin/python" --no-deps "${SOURCE_DIR}/sionna-main"
uv pip check --python "${ENV_DIR}/bin/python"

PYTHONPATH="${SOURCE_DIR}/sionna-large-radio-maps-main" \
SLRM_DATA_DIR="${RUNTIME_ROOT}/data" \
"${ENV_DIR}/bin/python" - <<'PY'
import importlib.metadata
import platform
import drjit
import h5py
import mitsuba
import sionna
import sionna.rt
import sionna_lrm
import torch

expected = {
    "sionna": "2.0.1",
    "sionna-rt": "1.2.1",
    "mitsuba": "3.7.1",
    "drjit": "1.2.0",
    "h5py": "3.15.1",
}
observed = {name: importlib.metadata.version(name) for name in expected}
if platform.python_version() != "3.12.13" or observed != expected:
    raise SystemExit(
        f"frozen Sionna runtime version mismatch: python={platform.python_version()} packages={observed}"
    )
mitsuba.set_variant("llvm_ad_mono_polarized")
drjit.set_thread_count(1)
probe = mitsuba.Float([1.0, 2.0, 3.0])
total = drjit.sum(probe)
drjit.eval(total)
if mitsuba.variant() != "llvm_ad_mono_polarized" or drjit.thread_count() != 1 or float(total[0]) != 6.0:
    raise SystemExit("frozen single-thread LLVM renderer self-test failed")

print("sionna", observed["sionna"])
print("sionna-rt", observed["sionna-rt"])
print("sionna-large-radio-maps", "source@1ba19ae1df1d26302fcfbaab14efc2347313da5d")
print("torch", torch.__version__)
print("h5py", observed["h5py"])
print("mitsuba-variant", mitsuba.variant())
print("drjit-threads", drjit.thread_count())
print("drjit-libllvm", __import__("os").environ["DRJIT_LIBLLVM_PATH"])
PY

PYTHONPATH="${PROJECT_ROOT}" "${ENV_DIR}/bin/python" -m formal_v2.formal_external_runtime \
  --profile sionna \
  --project-root "${PROJECT_ROOT}" \
  --output "${RUNTIME_ROOT}/runtime_provenance.json" \
  --verify-existing

printf 'Sionna runtime ready: %s\n' "${RUNTIME_ROOT}"
