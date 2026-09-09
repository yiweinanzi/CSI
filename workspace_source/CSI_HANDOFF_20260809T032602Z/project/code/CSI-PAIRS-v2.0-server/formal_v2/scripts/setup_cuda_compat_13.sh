#!/usr/bin/env bash
set -euo pipefail

PACKAGE_NAME="cuda-compat-13-0-580.105.08-1.el8.x86_64.rpm"
PACKAGE_SHA256="7d77eb1ed96bbc4f639f7b1acea4b66883ab57deedf4c55cf506d3e7dc03a0f3"
PACKAGE_URL="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/${PACKAGE_NAME}"
OUTPUT_INPUT="${1:?usage: setup_cuda_compat_13.sh UNUSED_ABSOLUTE_OUTPUT_PATH}"

if [[ "${OUTPUT_INPUT}" != /* ]]; then
  echo "CUDA compatibility output must be absolute: ${OUTPUT_INPUT}" >&2
  exit 2
fi
OUTPUT_PARENT="$(dirname "${OUTPUT_INPUT}")"
OUTPUT_NAME="$(basename "${OUTPUT_INPUT}")"
mkdir -p "${OUTPUT_PARENT}"
OUTPUT_PARENT="$(cd "${OUTPUT_PARENT}" && pwd)"
OUTPUT="${OUTPUT_PARENT}/${OUTPUT_NAME}"
if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite CUDA compatibility path: ${OUTPUT}" >&2
  exit 3
fi

CURL_BIN="${CSI_PAIRS_CURL:-$(command -v curl || true)}"
BSDTAR_BIN="${CSI_PAIRS_BSDTAR:-$(command -v bsdtar || true)}"
if [[ -z "${CURL_BIN}" || -z "${BSDTAR_BIN}" ]]; then
  echo "curl and bsdtar are required" >&2
  exit 4
fi

STAGING="$(mktemp -d "${OUTPUT_PARENT}/.cuda-compat-13-XXXXXX")"
cleanup() {
  rm -rf -- "${STAGING}"
}
trap cleanup EXIT

"${CURL_BIN}" --fail --location --silent --show-error \
  --output "${STAGING}/${PACKAGE_NAME}" "${PACKAGE_URL}"
echo "${PACKAGE_SHA256}  ${STAGING}/${PACKAGE_NAME}" | sha256sum -c -
"${BSDTAR_BIN}" -xf "${STAGING}/${PACKAGE_NAME}" -C "${STAGING}"

LIBRARY_DIR="${STAGING}/usr/local/cuda-13.0/compat"
test -f "${LIBRARY_DIR}/libcuda.so.580.105.08"
test "$(readlink "${LIBRARY_DIR}/libcuda.so.1")" = "libcuda.so.580.105.08"
echo "df9183549feb062f4195e6cf130e0ef372de4a59e59dbe51554ad8c3c5b167db  ${LIBRARY_DIR}/libcuda.so.580.105.08" | sha256sum -c -

mv "${STAGING}" "${OUTPUT}"
trap - EXIT
printf 'export CSI_PAIRS_CUDA_COMPAT_ROOT=%q\n' "${OUTPUT}"
printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' \
  "${OUTPUT}/usr/local/cuda-13.0/compat"
