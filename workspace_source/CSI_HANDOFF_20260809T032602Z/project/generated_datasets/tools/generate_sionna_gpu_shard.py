#!/usr/bin/env python3
"""Generate one GPU-backed Sionna RT shard for the V6 fixture dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import numpy as np


SEED = 20270809
SCENE_COUNT = 36
POSITIONS_PER_BANK = 30
WORLDS = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
REPEATS = 3
ANTENNAS = 2
SUBCARRIERS = 4
CHANNELS = 2 * ANTENNAS * SUBCARRIERS
MAX_PATHS = 32
MAX_DEPTH = 3
MAP_SIZE = 256
MAP_RESOLUTION_M = 1.0
MAP_ORIGIN_XY_M = np.asarray([-128.0, -128.0], dtype=np.float64)
BS_GLOBAL_XYZ_M = np.asarray([32.5, 10.5, 23.0], dtype=np.float64)
RECEIVER_Z_M = 1.5
CARRIER_FREQUENCY_HZ = 3.5e9
SUBCARRIER_SPACING_HZ = 30e3
BANDWIDTH_HZ = SUBCARRIERS * SUBCARRIER_SPACING_HZ
NOISE_STD = 1e-7
SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
SCENE_NAME = "simple_street_canyon"
MATERIAL_CATEGORY = {
    "free_space": 0,
    "glass": 1,
    "wood": 2,
    "marble": 3,
    "brick": 4,
    "concrete": 5,
}
STABLE_SURFACE_IDS = {
    "floor": 100,
    "building_2": 102,
    "no-name-1": 103,
    "building_6": 106,
    "no-name-2": 105,
}
PRIMITIVE_OBJECTS = ("building_2", "building_6")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primitive_mapping(scene_index: int) -> np.ndarray:
    return np.asarray([0, 1] if scene_index % 2 == 0 else [1, 0], dtype=np.int64)


def anchor_bits(scene_index: int) -> np.ndarray:
    return np.asarray([(scene_index // 2) % 2, scene_index % 2], dtype=np.int64)


def natural_world_index(scene_index: int) -> int:
    anchor = anchor_bits(scene_index)
    return next(index for index, row in enumerate(WORLDS) if np.array_equal(row, anchor))


def receiver_positions_global(scene_index: int) -> np.ndarray:
    """Return two bank-jittered street routes that expose both primitives."""
    y_offset = (scene_index - (SCENE_COUNT - 1) / 2.0) * 0.025
    x_offset = (scene_index % 6) * 0.031
    west_route = np.column_stack(
        (
            np.linspace(-45.0, 25.0, POSITIONS_PER_BANK // 2) + x_offset,
            np.full(POSITIONS_PER_BANK // 2, -4.0 + y_offset),
        )
    )
    east_route = np.column_stack(
        (
            np.linspace(35.0, 75.0, POSITIONS_PER_BANK // 2) + x_offset,
            np.full(POSITIONS_PER_BANK // 2, 4.0 + y_offset),
        )
    )
    return np.concatenate((west_route, east_route), axis=0).astype(np.float64)


def _gpu_snapshot(physical_index: int) -> dict[str, object]:
    fields = (
        "index,name,uuid,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "-i",
            str(physical_index),
        ],
        text=True,
    ).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 9:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    return {
        "physical_index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "memory_free_mib": int(values[5]),
        "utilization_percent": int(values[6]),
        "temperature_c": int(values[7]),
        "power_draw_w": float(values[8]),
    }


def _scene_asset_inventory(scene_xml: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for path in sorted(candidate for candidate in scene_xml.parent.rglob("*") if candidate.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(scene_xml.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows, hashlib.sha256(canonical_json(rows).encode("ascii")).hexdigest()


def _point(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _object_specs(rt_scene) -> tuple[list[dict[str, object]], dict[int, int]]:
    specs = []
    runtime_to_stable = {}
    unexpected = set(rt_scene.objects).difference(STABLE_SURFACE_IDS)
    missing = set(STABLE_SURFACE_IDS).difference(rt_scene.objects)
    if unexpected or missing:
        raise RuntimeError(
            f"Sionna object catalog changed: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name in sorted(rt_scene.objects):
        obj = rt_scene.objects[name]
        bbox = obj.mi_mesh.bbox()
        runtime_id = int(obj.object_id)
        stable_id = STABLE_SURFACE_IDS[name]
        runtime_to_stable[runtime_id] = stable_id
        specs.append(
            {
                "name": name,
                "runtime_object_id": runtime_id,
                "stable_surface_id": stable_id,
                "original_material": str(obj.radio_material.name),
                "bbox_min_xyz_m": _point(bbox.min),
                "bbox_max_xyz_m": _point(bbox.max),
            }
        )
    return specs, runtime_to_stable


def _physical_states(scene_index: int, bits: np.ndarray) -> np.ndarray:
    states = np.zeros(2, dtype=np.int64)
    for bit_index, value in enumerate(bits):
        states[int(primitive_mapping(scene_index)[bit_index])] = int(value)
    return states


def _set_world_materials(rt_scene, physical_states: np.ndarray) -> None:
    rt_scene.objects["building_2"].radio_material = (
        "brick" if int(physical_states[0]) == 0 else "glass"
    )
    rt_scene.objects["building_6"].radio_material = (
        "wood" if int(physical_states[1]) == 0 else "marble"
    )


def _bbox_cell_bounds(spec: dict[str, object]) -> tuple[int, int, int, int]:
    lower_global = np.asarray(spec["bbox_min_xyz_m"], dtype=np.float64)[:2]
    upper_global = np.asarray(spec["bbox_max_xyz_m"], dtype=np.float64)[:2]
    lower = lower_global - BS_GLOBAL_XYZ_M[:2]
    upper = upper_global - BS_GLOBAL_XYZ_M[:2]
    col0 = max(0, int(math.floor((lower[0] - MAP_ORIGIN_XY_M[0]) / MAP_RESOLUTION_M)))
    col1 = min(MAP_SIZE, int(math.ceil((upper[0] - MAP_ORIGIN_XY_M[0]) / MAP_RESOLUTION_M)))
    row0 = max(0, int(math.floor((lower[1] - MAP_ORIGIN_XY_M[1]) / MAP_RESOLUTION_M)))
    row1 = min(MAP_SIZE, int(math.ceil((upper[1] - MAP_ORIGIN_XY_M[1]) / MAP_RESOLUTION_M)))
    if row0 >= row1 or col0 >= col1:
        raise RuntimeError(f"object {spec['name']} lies outside the frozen map")
    return row0, row1, col0, col1


def _rasterize_world(
    scene_index: int,
    physical_states: np.ndarray,
    object_specs: list[dict[str, object]],
) -> np.ndarray:
    occupancy = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float64)
    height = np.zeros_like(occupancy)
    material = np.zeros_like(occupancy)
    tag_cell = None
    # AABBs overlap for some combined scene meshes. Draw intervention objects
    # last so their categorical state remains explicit in the map contract.
    ordered_specs = sorted(
        object_specs,
        key=lambda value: str(value["name"]) in PRIMITIVE_OBJECTS,
    )
    for spec in ordered_specs:
        name = str(spec["name"])
        if name == "floor":
            continue
        row0, row1, col0, col1 = _bbox_cell_bounds(spec)
        occupancy[row0:row1, col0:col1] = 1.0
        bounds_min = np.asarray(spec["bbox_min_xyz_m"], dtype=np.float64)
        bounds_max = np.asarray(spec["bbox_max_xyz_m"], dtype=np.float64)
        object_height = max(0.0, float(bounds_max[2] - bounds_min[2]))
        height[row0:row1, col0:col1] = object_height
        material_name = str(spec["original_material"])
        if name == "building_2":
            material_name = "brick" if int(physical_states[0]) == 0 else "glass"
        elif name == "building_6":
            material_name = "wood" if int(physical_states[1]) == 0 else "marble"
        material[row0:row1, col0:col1] = MATERIAL_CATEGORY[material_name]
        if name == "no-name-1":
            tag_cell = (row0, col0)
    if tag_cell is None:
        raise RuntimeError("could not select a stable occupied map tag cell")
    height[tag_cell] += (scene_index + 1) / 1000.0
    return np.stack((occupancy, height, material), axis=0)


def _phase_references(scene_index: int, positions_global: np.ndarray):
    references = np.empty(POSITIONS_PER_BANK, dtype=np.complex128)
    identifiers = np.empty(POSITIONS_PER_BANK, dtype="U96")
    digests = np.empty(POSITIONS_PER_BANK, dtype="U64")
    speed_of_light = 299_792_458.0
    for position_index, xy in enumerate(positions_global):
        receiver = np.asarray([xy[0], xy[1], RECEIVER_Z_M], dtype=np.float64)
        distance = float(np.linalg.norm(receiver - BS_GLOBAL_XYZ_M))
        phase = -2.0 * np.pi * CARRIER_FREQUENCY_HZ * distance / speed_of_light
        references[position_index] = np.exp(1j * phase)
        identifiers[position_index] = (
            f"free-space-geometric-phase:scene-{scene_index:02d}:p{position_index:04d}"
        )
        record = {
            "schema_version": "sionna-free-space-phase-reference-v1",
            "scene_index": scene_index,
            "position_index": position_index,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
            "speed_of_light_m_s": speed_of_light,
            "bs_global_xyz_m": BS_GLOBAL_XYZ_M.tolist(),
            "receiver_global_xyz_m": receiver.tolist(),
            "distance_m": distance,
        }
        digests[position_index] = hashlib.sha256(
            canonical_json(record).encode("ascii")
        ).hexdigest()
    return identifiers, references, digests


def _flatten_cfr(paths, frequencies: object) -> np.ndarray:
    values = np.asarray(
        paths.cfr(
            frequencies=frequencies,
            sampling_frequency=BANDWIDTH_HZ,
            num_time_steps=1,
            out_type="numpy",
        )
    )
    if values.shape != (POSITIONS_PER_BANK, 1, 1, 2, 1, SUBCARRIERS):
        raise RuntimeError(f"unexpected Sionna CFR shape: {values.shape}")
    return values[:, :, 0, :, 0, :].reshape(POSITIONS_PER_BANK, -1)


def _stable_path_id(surface_ids: np.ndarray, delay_s: float) -> int:
    record = np.concatenate(
        (
            np.asarray(surface_ids, dtype="<i8"),
            np.asarray([int(round(delay_s * 1e12))], dtype="<i8"),
        )
    )
    return int.from_bytes(hashlib.sha256(record.tobytes()).digest()[:8], "big") & ((1 << 63) - 1)


def _extract_paths(paths, runtime_to_stable: dict[int, int]):
    valid = np.asarray(paths.valid)
    delay = np.asarray(paths.tau)
    objects = np.asarray(paths.objects)
    real = np.asarray(paths.a[0])
    imag = np.asarray(paths.a[1])
    if valid.ndim != 3 or delay.shape != valid.shape or objects.shape[:3] != (
        MAX_DEPTH,
        POSITIONS_PER_BANK,
        1,
    ):
        raise RuntimeError(
            f"unexpected Sionna path tensors: valid={valid.shape}, "
            f"delay={delay.shape}, objects={objects.shape}"
        )
    power = np.sum(real * real + imag * imag, axis=(1, 2, 3))
    output_ids = np.full((POSITIONS_PER_BANK, MAX_PATHS), -1, dtype=np.int64)
    output_power = np.zeros((POSITIONS_PER_BANK, MAX_PATHS), dtype=np.float64)
    output_surfaces = np.full(
        (POSITIONS_PER_BANK, MAX_PATHS, MAX_DEPTH), -1, dtype=np.int64
    )
    counts = np.zeros(POSITIONS_PER_BANK, dtype=np.int64)
    invalid_shape = np.iinfo(np.uint32).max
    for receiver in range(POSITIONS_PER_BANK):
        entries = []
        valid_indices = np.flatnonzero(valid[receiver, 0])
        for path_index in valid_indices:
            raw_surfaces = objects[:, receiver, 0, path_index]
            stable_surfaces = np.full(MAX_DEPTH, -1, dtype=np.int64)
            for depth, raw_value in enumerate(raw_surfaces):
                raw_id = int(raw_value)
                if raw_id == invalid_shape:
                    continue
                if raw_id not in runtime_to_stable:
                    raise RuntimeError(f"unregistered Sionna object id in path: {raw_id}")
                stable_surfaces[depth] = runtime_to_stable[raw_id]
            path_id = _stable_path_id(stable_surfaces, float(delay[receiver, 0, path_index]))
            entries.append(
                (
                    path_id,
                    float(power[receiver, path_index]),
                    stable_surfaces,
                )
            )
        if len(entries) > MAX_PATHS:
            entries = sorted(entries, key=lambda row: row[1], reverse=True)[:MAX_PATHS]
        entries = sorted(entries, key=lambda row: row[0])
        if len({row[0] for row in entries}) != len(entries):
            raise RuntimeError("stable path identifier collision")
        counts[receiver] = len(entries)
        for output_index, (path_id, path_power, surfaces) in enumerate(entries):
            output_ids[receiver, output_index] = path_id
            output_power[receiver, output_index] = path_power
            output_surfaces[receiver, output_index] = surfaces
    return output_ids, output_power, output_surfaces, counts


def _write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite shard: {path}")
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def generate(args: argparse.Namespace) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu_index):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU "
            f"({args.physical_gpu_index}), got {visible!r}"
        )
    if not (0 <= args.scene_start < args.scene_end <= SCENE_COUNT):
        raise ValueError("scene range must be a nonempty subset of [0, 36)")

    from sionna.rt import (
        PathSolver,
        PlanarArray,
        Receiver,
        Transmitter,
        load_scene,
        scene,
        subcarrier_frequencies,
    )
    import drjit
    import mitsuba

    started_wall = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    gpu_before = _gpu_snapshot(args.physical_gpu_index)
    rt_scene = load_scene(scene.simple_street_canyon)
    scene_xml = Path(scene.simple_street_canyon).resolve()
    asset_rows, asset_tree_sha256 = _scene_asset_inventory(scene_xml)
    rt_scene.frequency = CARRIER_FREQUENCY_HZ
    rt_scene.bandwidth = BANDWIDTH_HZ
    rt_scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=2,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rt_scene.add(Transmitter("tx", position=BS_GLOBAL_XYZ_M.tolist()))
    first_positions = receiver_positions_global(args.scene_start)
    for position_index, xy in enumerate(first_positions):
        rt_scene.add(
            Receiver(
                f"rx-{position_index}",
                position=[float(xy[0]), float(xy[1]), RECEIVER_Z_M],
            )
        )
    object_specs, runtime_to_stable = _object_specs(rt_scene)
    solver = PathSolver()
    frequencies = subcarrier_frequencies(SUBCARRIERS, SUBCARRIER_SPACING_HZ)

    scene_indices = np.arange(args.scene_start, args.scene_end, dtype=np.int64)
    scene_rows = len(scene_indices)
    positions = np.zeros((scene_rows, POSITIONS_PER_BANK, 2), dtype=np.float64)
    clean = np.zeros((scene_rows, len(WORLDS), POSITIONS_PER_BANK, CHANNELS), dtype=np.float64)
    maps = np.zeros(
        (scene_rows, len(WORLDS), 3, MAP_SIZE, MAP_SIZE), dtype=np.float64
    )
    path_ids = np.full(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK, MAX_PATHS), -1, dtype=np.int64
    )
    path_power = np.zeros_like(path_ids, dtype=np.float64)
    path_surface_ids = np.full(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK, MAX_PATHS, MAX_DEPTH),
        -1,
        dtype=np.int64,
    )
    raw_path_counts = np.zeros(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK), dtype=np.int64
    )
    mappings = np.zeros((scene_rows, 2), dtype=np.int64)
    anchors = np.zeros((scene_rows, 2), dtype=np.int64)
    natural_worlds = np.zeros(scene_rows, dtype=np.int64)
    primitive_surface_ids = np.full((scene_rows, 2, 1), -1, dtype=np.int64)
    phase_ids = np.empty((scene_rows, POSITIONS_PER_BANK), dtype="U96")
    phase_values = np.empty((scene_rows, POSITIONS_PER_BANK), dtype=np.complex128)
    phase_source_sha256 = np.empty((scene_rows, POSITIONS_PER_BANK), dtype="U64")
    repeat_seeds = np.zeros(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK, REPEATS), dtype=np.int64
    )

    for row, scene_index_value in enumerate(scene_indices):
        scene_index = int(scene_index_value)
        positions_global = receiver_positions_global(scene_index)
        positions[row] = positions_global - BS_GLOBAL_XYZ_M[:2]
        for position_index, xy in enumerate(positions_global):
            rt_scene.receivers[f"rx-{position_index}"].position = [
                float(xy[0]),
                float(xy[1]),
                RECEIVER_Z_M,
            ]
        mappings[row] = primitive_mapping(scene_index)
        anchors[row] = anchor_bits(scene_index)
        natural_worlds[row] = natural_world_index(scene_index)
        for bit_index, primitive_index in enumerate(mappings[row]):
            primitive_surface_ids[row, bit_index, 0] = STABLE_SURFACE_IDS[
                PRIMITIVE_OBJECTS[int(primitive_index)]
            ]
        phase_ids[row], phase_values[row], phase_source_sha256[row] = _phase_references(
            scene_index, positions_global
        )

        for world_index, bits in enumerate(WORLDS):
            physical_states = _physical_states(scene_index, bits)
            _set_world_materials(rt_scene, physical_states)
            maps[row, world_index] = _rasterize_world(
                scene_index, physical_states, object_specs
            )
            traced = solver(
                rt_scene,
                max_depth=MAX_DEPTH,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
                synthetic_array=True,
                seed=SEED + scene_index * 100 + world_index,
            )
            raw_cfr = _flatten_cfr(traced, frequencies)
            gauge_fixed = raw_cfr * np.conjugate(phase_values[row])[:, None]
            clean[row, world_index] = np.concatenate(
                (gauge_fixed.real, gauge_fixed.imag), axis=1
            )
            (
                path_ids[row, world_index],
                path_power[row, world_index],
                path_surface_ids[row, world_index],
                raw_path_counts[row, world_index],
            ) = _extract_paths(traced, runtime_to_stable)
        print(
            canonical_json(
                {
                    "event": "bank_complete",
                    "physical_gpu_index": args.physical_gpu_index,
                    "scene_index": scene_index,
                    "completed_banks": row + 1,
                    "total_banks": scene_rows,
                    "elapsed_seconds": round(time.monotonic() - started_wall, 3),
                    "max_paths": int(np.max(raw_path_counts[row])),
                }
            ),
            flush=True,
        )

    free_space = np.ones(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK), dtype=np.bool_
    )
    for row in range(scene_rows):
        for position_index, xy in enumerate(positions[row]):
            column = int(math.floor((xy[0] - MAP_ORIGIN_XY_M[0]) / MAP_RESOLUTION_M))
            map_row = int(math.floor((xy[1] - MAP_ORIGIN_XY_M[1]) / MAP_RESOLUTION_M))
            if not (0 <= map_row < MAP_SIZE and 0 <= column < MAP_SIZE):
                raise RuntimeError("receiver route lies outside the map")
            if np.any(maps[row, :, 0, map_row, column] >= 0.5):
                raise RuntimeError("receiver route intersects a rasterized object")

    for row, scene_index_value in enumerate(scene_indices):
        scene_index = int(scene_index_value)
        for world_index in range(len(WORLDS)):
            for position_index in range(POSITIONS_PER_BANK):
                for repeat in range(REPEATS):
                    repeat_seeds[row, world_index, position_index, repeat] = (
                        SEED
                        + scene_index * 10_000_000
                        + world_index * 100_000
                        + position_index * 100
                        + repeat
                    )
    noise = np.empty(
        (scene_rows, len(WORLDS), POSITIONS_PER_BANK, REPEATS, CHANNELS),
        dtype=np.float64,
    )
    for row in range(scene_rows):
        for world_index in range(len(WORLDS)):
            for position_index in range(POSITIONS_PER_BANK):
                for repeat in range(REPEATS):
                    noise[row, world_index, position_index, repeat] = (
                        np.random.default_rng(
                            int(repeat_seeds[row, world_index, position_index, repeat])
                        ).normal(0.0, NOISE_STD, size=CHANNELS)
                    )
    repeated = clean[:, :, :, None, :] + noise
    radio_config = np.tile(
        np.asarray(
            [[CARRIER_FREQUENCY_HZ, ANTENNAS, SUBCARRIERS, SUBCARRIER_SPACING_HZ]],
            dtype=np.float64,
        ),
        (scene_rows, 1),
    )
    bs_pose = np.tile(
        np.asarray([[0.0, 0.0, BS_GLOBAL_XYZ_M[2], 1.0, 0.0, 0.0, 0.0]]),
        (scene_rows, 1),
    )
    arrays = {
        "scene_indices": scene_indices,
        "world_bits": WORLDS,
        "csi_repeat": repeated,
        "csi_clean": clean,
        "maps": maps,
        "positions": positions,
        "free_space": free_space,
        "radio_config": radio_config,
        "bs_pose": bs_pose,
        "repeat_seeds": repeat_seeds,
        "phase_reference_ids": phase_ids,
        "phase_reference_values": phase_values,
        "phase_reference_source_sha256": phase_source_sha256,
        "path_ids": path_ids,
        "path_power": path_power,
        "path_surface_ids": path_surface_ids,
        "primitive_surface_ids": primitive_surface_ids,
        "primitive_ids": mappings,
        "anchor_bits": anchors,
        "natural_world_index": natural_worlds,
        "raw_path_counts": raw_path_counts,
    }
    output = args.output.resolve()
    _write_npz_exclusive(output, arrays)
    gpu_after = _gpu_snapshot(args.physical_gpu_index)
    duration = time.monotonic() - started_wall
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    manifest = {
        "schema_version": "csi-pairs-sionna-gpu-shard-manifest-v1",
        "status": "PASS",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "coverage": {
            "scene_start_inclusive": args.scene_start,
            "scene_end_exclusive": args.scene_end,
            "scene_count": scene_rows,
            "worlds_per_scene": len(WORLDS),
            "positions_per_scene": POSITIONS_PER_BANK,
            "trace_calls": scene_rows * len(WORLDS),
            "max_paths_observed": int(np.max(raw_path_counts)),
        },
        "gpu": {
            "cuda_visible_devices": visible,
            "logical_device": 0,
            "physical_gpu_index": args.physical_gpu_index,
            "uuid": gpu_before["uuid"],
            "before": gpu_before,
            "after": gpu_after,
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "sionna": importlib.metadata.version("sionna"),
            "sionna_rt": importlib.metadata.version("sionna-rt"),
            "mitsuba": importlib.metadata.version("mitsuba"),
            "drjit": importlib.metadata.version("drjit"),
            "mitsuba_variant": mitsuba.variant(),
            "sionna_revision": SIONNA_REVISION,
        },
        "scene": {
            "name": SCENE_NAME,
            "xml_path": str(scene_xml),
            "xml_sha256": sha256_file(scene_xml),
            "asset_tree_sha256": asset_tree_sha256,
            "assets": asset_rows,
            "objects": object_specs,
        },
        "simulation": {
            "seed": SEED,
            "max_depth": MAX_DEPTH,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "synthetic_array": True,
            "carrier_frequency_hz": CARRIER_FREQUENCY_HZ,
            "bandwidth_hz": BANDWIDTH_HZ,
            "subcarrier_spacing_hz": SUBCARRIER_SPACING_HZ,
            "observation_noise_std": NOISE_STD,
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _write_json_exclusive(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-start", required=True, type=int)
    parser.add_argument("--scene-end", required=True, type=int)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    manifest = generate(parser.parse_args())
    print(canonical_json({"status": "PASS", "output": manifest["output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
