from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request as urlrequest

from .formal_io import read_strict_json, sha256_file
from .formal_resources import (
    REQUIRED_FILES,
    validate_resource_registry,
    validate_resource_registry_structure,
)


def fetch_missing_resources(
    registry_path: str | Path,
    waibu_root: str | Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict:
    registry_file = Path(registry_path).resolve()
    resources = validate_resource_registry_structure(read_strict_json(registry_file))
    supplied_root = Path(waibu_root)
    if supplied_root.exists() and (not supplied_root.is_dir() or supplied_root.is_symlink()):
        raise ValueError("waibu root must be a regular directory, not a file or symlink")
    supplied_root.mkdir(parents=True, exist_ok=True)
    root = supplied_root.resolve()
    unexpected = {
        path.name for path in root.iterdir() if path.is_file()
    } - REQUIRED_FILES
    if unexpected:
        raise ValueError(f"waibu directory contains unexpected files: {sorted(unexpected)}")

    fetched: list[str] = []
    authenticated: list[str] = []
    for row in resources:
        target = root / row["file"]
        if target.is_symlink():
            raise ValueError(f"refusing symlink resource target: {row['file']}")
        if target.exists():
            if not target.is_file() or sha256_file(target) != row["sha256"]:
                raise ValueError(
                    f"existing resource failed SHA-256 authentication; refusing overwrite: {row['file']}"
                )
            authenticated.append(row["file"])
            continue

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{row['file']}.",
                suffix=".download",
                dir=root,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                request = urlrequest.Request(
                    row["source_url"],
                    headers={"User-Agent": "CSI-PAIRS-resource-fetch/1"},
                )
                digest = hashlib.sha256()
                with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(
                    f"downloaded resource SHA-256 mismatch: {row['file']}"
                )
            os.replace(temporary_path, target)
            temporary_path = None
            fetched.append(row["file"])
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    rows = validate_resource_registry(
        read_strict_json(registry_file),
        root,
        mode="formal",
    )
    if not all(row["status"] == "PASS" for row in rows):
        raise ValueError("one or more fetched resources failed final authentication")
    return {
        "status": "PASS",
        "registry": str(registry_file),
        "waibu_root": str(root),
        "fetched": sorted(fetched),
        "already_authenticated": sorted(authenticated),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download missing CSI-PAIRS third-party inputs directly from their recorded sources "
            "and accept them only after SHA-256 authentication"
        )
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--waibu-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (0.0 < args.timeout_seconds <= 600.0):
        print("error: --timeout-seconds must be in (0, 600]", file=sys.stderr)
        return 2
    try:
        result = fetch_missing_resources(
            args.registry,
            args.waibu_root,
            timeout_seconds=float(args.timeout_seconds),
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
