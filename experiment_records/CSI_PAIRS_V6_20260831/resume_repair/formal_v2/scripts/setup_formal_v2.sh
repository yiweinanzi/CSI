#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CSI_PAIRS_BOOTSTRAP_PYTHON:-python3}"
ENVIRONMENT_PATH="${1:?usage: setup_formal_v2.sh UNUSED_ENVIRONMENT_PATH}"

if [[ -e "${ENVIRONMENT_PATH}" ]]; then
  echo "refusing to overwrite environment path: ${ENVIRONMENT_PATH}" >&2
  exit 2
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -B -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "Python 3.12 is required; ${PYTHON_BIN} reports ${PYTHON_VERSION}" >&2
  exit 3
fi

TARGET="$("${PYTHON_BIN}" -B - <<'PY'
import platform
import sys

system = platform.system()
machine = platform.machine()
if system == "Darwin" and machine == "arm64":
    version = platform.mac_ver()[0]
    if not version or int(version.split(".", 1)[0]) < 14:
        raise SystemExit(f"macOS 14 or newer is required; observed {version or 'unknown'}")
elif system == "Linux" and machine == "x86_64":
    libc, version = platform.libc_ver()
    parts = tuple(int(part) for part in version.split(".")[:2]) if version else ()
    if libc != "glibc" or parts < (2, 28):
        raise SystemExit(f"glibc 2.28 or newer is required; observed {libc} {version}")
else:
    raise SystemExit(
        "supported setup targets are macOS 14+ arm64 and glibc 2.28+ Linux x86_64; "
        f"observed {system} {machine}"
    )
print(f"{system}:{machine}")
PY
)"
echo "validated setup target: ${TARGET/:/ }"

PIP_DOWNLOAD_ARGS=(
  --require-hashes
  --only-binary=:all:
)
case "${TARGET}" in
  Darwin:arm64)
    REQUIREMENTS_LOCK="${PROJECT_ROOT}/formal_v2/requirements-lock.txt"
    ;;
  Linux:x86_64)
    REQUIREMENTS_LOCK="${PROJECT_ROOT}/formal_v2/requirements-lock-linux-x86_64-cu121.txt"
    PIP_DOWNLOAD_ARGS+=(--extra-index-url https://download.pytorch.org/whl/cu121)
    ;;
  *)
    echo "validated target has no requirements lock: ${TARGET}" >&2
    exit 5
    ;;
esac

"${PYTHON_BIN}" -B -m venv "${ENVIRONMENT_PATH}"
WHEELHOUSE_PATH="${ENVIRONMENT_PATH}/csi-pairs-reviewed-wheels"
WHEEL_MANIFEST_PATH="${ENVIRONMENT_PATH}/csi-pairs-reviewed-wheel-manifest.json"
mkdir -p "${WHEELHOUSE_PATH}"
if [[ -n "${CSI_PAIRS_PIP_CERT:-}" ]]; then
  if [[ ! -f "${CSI_PAIRS_PIP_CERT}" ]]; then
    echo "CSI_PAIRS_PIP_CERT is not a regular certificate file: ${CSI_PAIRS_PIP_CERT}" >&2
    exit 4
  fi
  PIP_DOWNLOAD_ARGS+=(--cert "${CSI_PAIRS_PIP_CERT}")
elif [[ "$(uname -s)" == "Darwin" && -f /etc/ssl/cert.pem ]]; then
  PIP_DOWNLOAD_ARGS+=(--cert /etc/ssl/cert.pem)
fi
"${ENVIRONMENT_PATH}/bin/python" -B -m pip download \
  "${PIP_DOWNLOAD_ARGS[@]}" \
  --dest "${WHEELHOUSE_PATH}" \
  --requirement "${REQUIREMENTS_LOCK}"
"${ENVIRONMENT_PATH}/bin/python" -B -m pip install \
  --no-index \
  --find-links "${WHEELHOUSE_PATH}" \
  --no-compile \
  --require-hashes \
  --only-binary=:all: \
  --report "${ENVIRONMENT_PATH}/csi-pairs-install-report.json" \
  --requirement "${REQUIREMENTS_LOCK}"
"${ENVIRONMENT_PATH}/bin/python" -B -m pip check
"${ENVIRONMENT_PATH}/bin/python" -B -c 'import numpy, torch; print("numpy", numpy.__version__, "torch", torch.__version__)'
PYTHONPATH="${PROJECT_ROOT}" "${ENVIRONMENT_PATH}/bin/python" -B - \
  "${WHEELHOUSE_PATH}" "${WHEEL_MANIFEST_PATH}" \
  "${REQUIREMENTS_LOCK}" <<'PY'
import sys
from pathlib import Path

from formal_v2.formal_evidence import _locked_requirement_records
from formal_v2.formal_io import sha256_file
from formal_v2.formal_runtime_integrity import write_reviewed_wheel_manifest

wheelhouse = Path(sys.argv[1])
manifest = Path(sys.argv[2])
requirements = Path(sys.argv[3])
write_reviewed_wheel_manifest(
    wheelhouse,
    manifest,
    _locked_requirement_records(requirements),
    sha256_file(requirements),
)
PY
"${ENVIRONMENT_PATH}/bin/python" -B -m pip uninstall --yes pip
find "${ENVIRONMENT_PATH}/lib/python3.12/site-packages" -type f \
  \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "${ENVIRONMENT_PATH}/lib/python3.12/site-packages" -type d \
  -name '__pycache__' -empty -delete
chmod 0444 "${ENVIRONMENT_PATH}/csi-pairs-install-report.json" "${WHEEL_MANIFEST_PATH}"
find "${WHEELHOUSE_PATH}" -type f -exec chmod 0444 {} +
find "${WHEELHOUSE_PATH}" -type d -exec chmod 0555 {} +
