#!/usr/bin/env python3
"""Test repeated Sionna renders in one process and an alternate loop executor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from formal_v2.sionna_osm_candidate import ensure_sionna_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--original-shard", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    parser.add_argument("--drjit-threads", type=int)
    parser.add_argument("--loop-mode", choices=("symbolic", "evaluated"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.backend == "cuda":
        if args.physical_gpu_index is None or visible != str(args.physical_gpu_index):
            raise RuntimeError(
                f"expected CUDA_VISIBLE_DEVICES={args.physical_gpu_index}, got {visible!r}"
            )
    ensure_sionna_runtime([__file__, *sys.argv[1:]])

    import numpy as np
    import mitsuba as mi

    mi.set_variant(f"{args.backend}_ad_mono_polarized")
    import drjit as dr

    if args.drjit_threads is not None:
        if args.backend != "llvm" or args.drjit_threads < 1:
            raise ValueError("--drjit-threads requires LLVM and a positive count")
        dr.set_thread_count(args.drjit_threads)
    import sionna.rt

    from compare_sionna_regeneration import (
        _confusion,
        _numeric,
        _path_metrics,
        _sha256,
        _visibility,
    )
    from formal_v2.sionna_osm_candidate import (
        _read_json,
        load_asset_manifest,
        render_bank,
    )

    original_solver = sionna.rt.PathSolver

    class ConfiguredPathSolver(original_solver):
        def __init__(self):
            super().__init__()
            self.loop_mode = args.loop_mode

    sionna.rt.PathSolver = ConfiguredPathSolver
    root, manifest, config = load_asset_manifest(args.asset_root)
    bank_row = manifest["banks"][args.scene_index]
    bank = _read_json(root / str(bank_row["bank_record_path"]))
    first = render_bank(bank, root, config, args.scene_index)
    second = render_bank(bank, root, config, args.scene_index)
    with np.load(args.original_shard, allow_pickle=False) as archive:
        scene_indices = np.asarray(archive["scene_indices"])
        matches = np.flatnonzero(scene_indices == args.scene_index)
        if matches.size != 1:
            raise RuntimeError("original scene is absent or duplicated")
        row = int(matches[0])
        original = {
            name: np.asarray(archive[name][row])
            for name in archive.files
            if name != "scene_indices"
        }

    def comparison(left: dict, right: dict) -> dict:
        exact = {name: bool(np.array_equal(left[name], right[name])) for name in left}
        result = {
            "exact_field_count": sum(exact.values()),
            "field_count": len(exact),
            "exact_fields": exact,
            "clean_visibility": _confusion(
                _visibility(left["csi_clean"]), _visibility(right["csi_clean"])
            ),
            "csi_clean": _numeric(left["csi_clean"], right["csi_clean"]),
            "csi_repeat": _numeric(left["csi_repeat"], right["csi_repeat"]),
        }
        for prefix in ("", "noop_"):
            label = "paths" if not prefix else "noop_paths"
            result[label] = _path_metrics(
                left[f"{prefix}path_ids"], right[f"{prefix}path_ids"],
                left[f"{prefix}path_power"], right[f"{prefix}path_power"],
                left[f"{prefix}path_surface_ids"], right[f"{prefix}path_surface_ids"],
            )
        return result

    repeat = comparison(first, second)
    payload = {
        "schema_version": "csi-pairs-sionna-process-determinism-diagnostic-v1",
        "purpose": "diagnostic-only; no scientific protocol or verifier threshold changed",
        "status": "PASS" if repeat["exact_field_count"] == repeat["field_count"] else "FAIL",
        "scene_index": args.scene_index,
        "physical_gpu_index": args.physical_gpu_index,
        "cuda_visible_devices": visible,
        "backend": args.backend,
        "mitsuba_variant": mi.variant(),
        "drjit_thread_count": int(dr.thread_count()),
        "loop_mode": args.loop_mode,
        "asset_manifest_sha256": _sha256(args.asset_root / "asset_manifest.json"),
        "original_shard": {
            "path": str(args.original_shard.resolve()),
            "sha256": _sha256(args.original_shard),
        },
        "first_vs_second_same_process": repeat,
        "original_vs_first": comparison(original, first),
        "original_vs_second": comparison(original, second),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
