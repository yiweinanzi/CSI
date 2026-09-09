#!/usr/bin/env python3
"""Diagnose repeatability for one Sionna receiver on CUDA or LLVM."""

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


def _compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    from compare_sionna_regeneration import _numeric, _path_metrics

    exact = {name: bool(np.array_equal(left[name], right[name])) for name in left}
    return {
        "exact_field_count": int(sum(exact.values())),
        "field_count": len(exact),
        "exact_fields": exact,
        "csi_clean": _numeric(left["csi_clean"], right["csi_clean"]),
        "paths": _path_metrics(
            left["path_ids"],
            right["path_ids"],
            left["path_power"],
            right["path_power"],
            left["path_surface_ids"],
            right["path_surface_ids"],
        ),
    }


def _original_unit(
    shard: Path, scene_index: int, world_index: int, receiver_index: int
) -> dict[str, np.ndarray]:
    with np.load(shard, allow_pickle=False) as archive:
        matches = np.flatnonzero(np.asarray(archive["scene_indices"]) == scene_index)
        if matches.size != 1:
            raise RuntimeError("original shard does not contain the scene exactly once")
        row = int(matches[0])
        return {
            "csi_clean": np.asarray(
                archive["csi_clean"][row, world_index, receiver_index]
            ),
            "path_ids": np.asarray(
                archive["path_ids"][row, world_index, receiver_index][None, :]
            ),
            "path_power": np.asarray(
                archive["path_power"][row, world_index, receiver_index][None, :]
            ),
            "path_surface_ids": np.asarray(
                archive["path_surface_ids"][row, world_index, receiver_index][None, :, :]
            ),
        }


def _render_once(
    scene,
    solver,
    frequencies,
    bandwidth_hz: float,
    position: np.ndarray,
    phase_value: complex,
    radio: dict,
    runtime_to_stable: dict[int, int],
    seed: int,
) -> tuple[dict[str, np.ndarray], float, int]:
    started = time.monotonic()
    paths = solver(
        scene,
        max_depth=int(radio["max_depth"]),
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        seed=seed,
    )
    raw_cfr = candidate._flatten_cfr(paths, frequencies, bandwidth_hz, 1, radio)
    clean = raw_cfr * np.conjugate(phase_value) / abs(phase_value)
    path_ids, path_power, path_surfaces = candidate._extract_paths(
        paths,
        runtime_to_stable,
        1,
        int(radio["max_stored_paths"]),
        int(radio["max_depth"]),
    )
    return (
        {
            "csi_clean": np.concatenate((clean.real, clean.imag), axis=1)[0],
            "path_ids": path_ids,
            "path_power": path_power,
            "path_surface_ids": path_surfaces,
        },
        time.monotonic() - started,
        int(np.count_nonzero(path_ids >= 0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--original-shard", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--receiver-index", type=int, required=True)
    parser.add_argument("--backend", choices=("cuda", "llvm"), required=True)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--drjit-threads", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.backend == "cuda":
        if args.physical_gpu_index is None:
            raise RuntimeError("CUDA diagnostics require --physical-gpu-index")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(args.physical_gpu_index):
            raise RuntimeError(
                f"expected CUDA_VISIBLE_DEVICES={args.physical_gpu_index}, got {visible!r}"
            )

    candidate.ensure_sionna_runtime([__file__, *sys.argv[1:]])
    import mitsuba as mi

    variant = f"{args.backend}_ad_mono_polarized"
    mi.set_variant(variant)
    import drjit as dr

    if args.drjit_threads is not None:
        if args.backend != "llvm" or args.drjit_threads < 1:
            raise ValueError("--drjit-threads requires LLVM and a positive count")
        dr.set_thread_count(args.drjit_threads)
    from sionna.rt import (
        ITURadioMaterial,
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
        subcarrier_frequencies,
    )

    root, manifest, config = candidate.load_asset_manifest(args.asset_root)
    if not 0 <= args.scene_index < len(manifest["banks"]):
        raise ValueError("scene index is out of range")
    bank_row = manifest["banks"][args.scene_index]
    bank = candidate._read_json(root / str(bank_row["bank_record_path"]))
    positions = candidate._receiver_positions(bank, config, args.scene_index)
    if not 0 <= args.receiver_index < positions.shape[0]:
        raise ValueError("receiver index is out of range")
    receiver_xy = positions[args.receiver_index]
    radio = config["radio"]
    scene = load_scene(root / str(bank_row["scene_xml_path"]))
    runtime_to_stable = {
        int(scene.objects[name].object_id): stable_id
        for name, stable_id in candidate.STABLE_SURFACE_IDS.items()
    }
    for material_type in ("glass", "metal"):
        name = f"itu_{material_type}"
        if name not in scene.radio_materials:
            scene.add(
                ITURadioMaterial(
                    name=name,
                    itu_type=material_type,
                    thickness=candidate.SIONNA_MATERIAL_THICKNESS_M,
                )
            )
    scene.frequency = float(radio["carrier_frequency_hz"])
    bandwidth_hz = float(radio["subcarrier_spacing_hz"]) * int(radio["subcarriers"])
    scene.bandwidth = bandwidth_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=int(radio["tx_antennas"]),
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(Transmitter("tx", position=[0.0, 0.0, float(radio["transmitter_z_m"])]))
    scene.add(
        Receiver(
            f"rx-{args.receiver_index}",
            position=[
                float(receiver_xy[0]),
                float(receiver_xy[1]),
                float(radio["receiver_z_m"]),
            ],
        )
    )
    worlds = np.asarray(config["world_bits"], dtype=np.int64)
    world_index = int(candidate._natural_world_index(args.scene_index, worlds))
    candidate._set_world_materials(
        scene,
        candidate._physical_states(
            worlds[world_index], candidate._primitive_mapping(args.scene_index)
        ),
    )
    frequencies = subcarrier_frequencies(
        int(radio["subcarriers"]), float(radio["subcarrier_spacing_hz"])
    )
    _, phase_values, _ = candidate._phase_references(bank, positions, config)
    seed = int(config["seed"]) + args.scene_index * 1000 + world_index
    solver = PathSolver()
    first, first_seconds, first_count = _render_once(
        scene,
        solver,
        frequencies,
        bandwidth_hz,
        receiver_xy,
        phase_values[args.receiver_index],
        radio,
        runtime_to_stable,
        seed,
    )
    second, second_seconds, second_count = _render_once(
        scene,
        solver,
        frequencies,
        bandwidth_hz,
        receiver_xy,
        phase_values[args.receiver_index],
        radio,
        runtime_to_stable,
        seed,
    )
    original = _original_unit(
        args.original_shard,
        args.scene_index,
        world_index,
        args.receiver_index,
    )
    repeat = _compare(first, second)
    payload = {
        "schema_version": "csi-pairs-sionna-single-receiver-backend-diagnostic-v1",
        "status": "PASS" if repeat["exact_field_count"] == repeat["field_count"] else "FAIL",
        "purpose": "diagnostic-only; does not change the frozen propagation protocol or exact verifier",
        "simulation_not_measurement": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "mitsuba_variant": mi.variant(),
        "drjit_thread_count": int(dr.thread_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": args.physical_gpu_index,
        "scene_index": args.scene_index,
        "scene_id": bank_row["scene_id"],
        "world_index": world_index,
        "receiver_index": args.receiver_index,
        "receiver_xy_m": receiver_xy.tolist(),
        "seed": seed,
        "first_duration_seconds": first_seconds,
        "second_duration_seconds": second_seconds,
        "first_path_count": first_count,
        "second_path_count": second_count,
        "asset_root": str(root),
        "asset_manifest_sha256": candidate.sha256_file(root / "asset_manifest.json"),
        "original_shard": {
            "path": str(args.original_shard.resolve()),
            "sha256": candidate.sha256_file(args.original_shard),
        },
        "runtime": candidate._sionna_runtime_record(),
        "first_vs_second_same_process": repeat,
        "original_multi_receiver_vs_first_single_receiver": _compare(original, first),
        "original_multi_receiver_vs_second_single_receiver": _compare(original, second),
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": candidate.sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
