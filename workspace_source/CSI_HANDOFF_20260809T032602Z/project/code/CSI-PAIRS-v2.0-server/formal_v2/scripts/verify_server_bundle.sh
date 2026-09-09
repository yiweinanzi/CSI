#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CSI_PAIRS_PYTHON:-python3}"
OUTPUT="${1:?usage: verify_server_bundle.sh UNUSED_DRY_RUN_OUTPUT_DIRECTORY}"

# Verification must not mutate the authenticated bundle or make a second run
# fail its own checksum inventory.
export PYTHONDONTWRITEBYTECODE=1

cd "${PROJECT_ROOT}"
if [[ ! -f SHA256SUMS || -L SHA256SUMS ]]; then
  echo "regular SHA256SUMS is required to verify the bundle" >&2
  exit 4
fi
python3 - <<'PY'
from pathlib import Path
import re

root = Path.cwd().resolve()
lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("SHA256SUMS is empty")
listed = set()
for line in lines:
    match = re.fullmatch(r"([0-9a-f]{64})  (\./[^\n]+)", line)
    if match is None:
        raise SystemExit(f"invalid SHA256SUMS line: {line!r}")
    relative = match.group(2)
    if relative in listed:
        raise SystemExit(f"duplicate SHA256SUMS path: {relative}")
    listed.add(relative)
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise SystemExit(f"unsafe or missing SHA256SUMS path: {relative}")
symlinks = [path for path in root.rglob("*") if path.is_symlink()]
if symlinks:
    raise SystemExit(f"bundle contains symlinks: {symlinks[:5]}")
actual = {
    "./" + path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
}
if listed != actual:
    raise SystemExit(
        "SHA256SUMS inventory mismatch: "
        f"missing={sorted(actual - listed)[:5]}, unexpected={sorted(listed - actual)[:5]}"
    )
PY
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check SHA256SUMS
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 --check SHA256SUMS
else
  echo "sha256sum or shasum is required to verify the bundle" >&2
  exit 3
fi

"${PYTHON_BIN}" -m unittest discover -s formal_v2/tests -v
CSI_PAIRS_PYTHON="${PYTHON_BIN}" \
  formal_v2/scripts/run_formal_v2_dry_run.sh "${OUTPUT}"
