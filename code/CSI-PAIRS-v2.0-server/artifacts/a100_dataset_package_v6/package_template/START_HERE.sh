#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$ROOT/scripts/VERIFY_PACKAGE.sh"

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
printf 'Run: source %q\n' "$ROOT/scripts/MOUNT_DATASETS.sh"
