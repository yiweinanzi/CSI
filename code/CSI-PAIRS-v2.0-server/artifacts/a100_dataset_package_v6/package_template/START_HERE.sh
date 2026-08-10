#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$ROOT/scripts/verify_package.py"

echo
echo "PACKAGE_ROOT=$ROOT"
echo "FORMAL_TRAINING_READY=NO"
echo "SCIENTIFIC_EVIDENCE=NOT_ASSESSED"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "A100_HOST_CHECK=SKIPPED_NO_NVIDIA_SMI"
fi

echo
echo "Run: source $ROOT/scripts/MOUNT_DATASETS.sh"
