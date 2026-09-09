#!/usr/bin/env python3
"""Compare a one-scene replay with its original Sionna render shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_sionna_regeneration import _confusion, _numeric, _path_metrics, _sha256, _visibility


def _scene_row(archive: dict[str, np.ndarray], scene_index: int) -> dict[str, np.ndarray]:
    matches = np.flatnonzero(np.asarray(archive["scene_indices"]) == scene_index)
    if matches.size != 1:
        raise RuntimeError(f"scene {scene_index} is not unique in shard")
    row = int(matches[0])
    return {
        name: np.asarray(value[row])
        for name, value in archive.items()
        if name != "scene_indices"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.original, allow_pickle=False) as archive:
        original = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(args.replay, allow_pickle=False) as archive:
        replay = {name: np.asarray(archive[name]) for name in archive.files}
    left = _scene_row(original, args.scene_index)
    right = _scene_row(replay, args.scene_index)
    if set(left) != set(right):
        raise RuntimeError("original and replay shard fields differ")
    arrays = {}
    for name in sorted(left):
        exact = bool(np.array_equal(left[name], right[name]))
        record = {
            "shape": list(left[name].shape),
            "dtype": str(left[name].dtype),
            "exact": exact,
        }
        if np.issubdtype(left[name].dtype, np.number) and left[name].dtype != np.bool_:
            record["numeric"] = _numeric(left[name], right[name])
        arrays[name] = record
    path_reports = {}
    for prefix in ("", "noop_"):
        label = "paths" if not prefix else "noop_paths"
        path_reports[label] = _path_metrics(
            left[f"{prefix}path_ids"], right[f"{prefix}path_ids"],
            left[f"{prefix}path_power"], right[f"{prefix}path_power"],
            left[f"{prefix}path_surface_ids"], right[f"{prefix}path_surface_ids"],
        )
    payload = {
        "schema_version": "csi-pairs-sionna-same-gpu-shard-replay-v1",
        "purpose": "diagnostic-only; the exact formal verifier remains authoritative",
        "status": "PASS" if all(row["exact"] for row in arrays.values()) else "FAIL",
        "scene_index": args.scene_index,
        "original": {"path": str(args.original.resolve()), "sha256": _sha256(args.original)},
        "replay": {"path": str(args.replay.resolve()), "sha256": _sha256(args.replay)},
        "exact_field_count": sum(row["exact"] for row in arrays.values()),
        "field_count": len(arrays),
        "clean_visibility": _confusion(_visibility(left["csi_clean"]), _visibility(right["csi_clean"])),
        "arrays": arrays,
        **path_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
