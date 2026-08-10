#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v python3.12 >/dev/null 2>&1 || {
  echo "python3.12 is required to verify this package" >&2
  exit 3
}
exec python3.12 "$ROOT/scripts/verify_package.py" "$@"
