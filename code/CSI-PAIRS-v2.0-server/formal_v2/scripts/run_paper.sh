#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CSI_PAPER_PYTHON:-python3}"
if [[ $# -lt 2 ]]; then
  echo "Usage: bash formal_v2/scripts/run_paper.sh DATASET.npz OUTPUT_DIR [paper_run options]" >&2
  exit 2
fi
DATASET="$("${PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")"
OUTPUT="$("${PYTHON}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$2")"
shift 2
cd "${PROJECT_ROOT}"

# Use the server's PyTorch environment. No installation receipts or run permits.
"${PYTHON}" -c 'import torch, numpy, scipy; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())'
WIGATR_ENV="${PROJECT_ROOT}/formal_v2/external_adapters/.venv-wigatr"
WIGATR_ARGS=()
if [[ -x "${WIGATR_ENV}/bin/python" ]] && "${WIGATR_ENV}/bin/python" -c 'import gatr, torch_geometric, wigatr' >/dev/null 2>&1; then
  WIGATR_ARGS=(--wigatr-python "${WIGATR_ENV}/bin/python")
else
  echo "Wi-GATr environment not found; continuing with other methods (Wi-GATr will be MISSING/N/A)." >&2
fi
exec "${PYTHON}" -B -m formal_v2.paper_run --dataset "${DATASET}" --output "${OUTPUT}" \
  "${WIGATR_ARGS[@]}" "$@"