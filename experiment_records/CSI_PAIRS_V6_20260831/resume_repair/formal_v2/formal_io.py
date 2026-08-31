from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


class StrictJsonError(ValueError):
    pass


def read_strict_json(path: str | Path) -> object:
    source = Path(path)
    try:
        payload = parse_strict_json(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, StrictJsonError) as error:
        raise StrictJsonError(f"{type(error).__name__}: {error}") from error
    return payload


def parse_strict_json(text: str) -> object:
    """Parse RFC JSON while rejecting duplicate keys and non-finite constants."""
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, json.JSONDecodeError, StrictJsonError) as error:
        raise StrictJsonError(f"{type(error).__name__}: {error}") from error
    _require_finite(payload)
    return payload


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(
    root: str | Path,
    *,
    evidence: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    base = Path(root)
    rows = []
    for path in sorted(
        item for item in base.rglob("*") if item.is_file() and item.name != "manifest.json"
    ):
        row: dict[str, object] = {
                "path": str(path.relative_to(base)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
        }
        if evidence:
            row.update(evidence)
        rows.append(row)
    return rows


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise StrictJsonError(f"non-standard numeric constant: {value}")


def _require_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJsonError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")
