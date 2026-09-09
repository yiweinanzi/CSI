#!/usr/bin/env python3
"""Render one diagnostic Sionna bank under an explicitly recorded backend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--backend", choices=("cuda", "llvm"), required=True)
    parser.add_argument("--drjit-threads", type=int)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output {args.output}")
    if args.backend == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if args.physical_gpu_index is None or visible != str(args.physical_gpu_index):
            raise RuntimeError(
                f"expected CUDA_VISIBLE_DEVICES={args.physical_gpu_index}, got {visible!r}"
            )

    candidate.ensure_sionna_runtime([__file__, *sys.argv[1:]])
    import mitsuba as mi

    mi.set_variant(f"{args.backend}_ad_mono_polarized")
    import drjit as dr

    if args.drjit_threads is not None:
        if args.backend != "llvm" or args.drjit_threads < 1:
            raise ValueError("--drjit-threads requires LLVM and a positive count")
        dr.set_thread_count(args.drjit_threads)
    import sionna.rt

    root, manifest, config = candidate.load_asset_manifest(args.asset_root)
    if not 0 <= args.scene_index < len(manifest["banks"]):
        raise ValueError("scene index is out of range")
    bank_row = manifest["banks"][args.scene_index]
    bank = candidate._read_json(root / str(bank_row["bank_record_path"]))
    started = time.monotonic()
    rendered = candidate.render_bank(bank, root, config, args.scene_index)
    duration = time.monotonic() - started
    arrays = {
        "scene_indices": np.asarray((args.scene_index,), dtype=np.int64),
        **{name: np.asarray(value)[None, ...] for name, value in rendered.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate._write_npz_exclusive(args.output, arrays)
    tool_path = Path(__file__).resolve()
    payload = {
        "schema_version": "csi-pairs-sionna-bank-backend-diagnostic-v1",
        "status": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "simulation_not_measurement": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "mitsuba_variant": mi.variant(),
        "drjit_thread_count": int(dr.thread_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": args.physical_gpu_index,
        "scene_index": args.scene_index,
        "scene_id": bank_row["scene_id"],
        "duration_seconds": duration,
        "asset_root": str(root),
        "asset_manifest_sha256": candidate.sha256_file(root / "asset_manifest.json"),
        "output_path": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": candidate.sha256_file(args.output),
        "runtime": candidate._sionna_runtime_record(),
        "generator_path": str(Path(candidate.__file__).resolve()),
        "generator_sha256": candidate.sha256_file(Path(candidate.__file__).resolve()),
        "tool_path": str(tool_path),
        "tool_sha256": candidate.sha256_file(tool_path),
    }
    candidate._write_json_exclusive(args.output.with_suffix(".manifest.json"), payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
