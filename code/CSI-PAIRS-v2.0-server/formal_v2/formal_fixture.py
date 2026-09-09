from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .formal_dataset import DATASET_SCHEMA_VERSION, SOURCE_ROLES, _array_sha256


def write_nonscientific_fixture(
    path: str | Path,
    seed: int = 20270805,
    scene_count: int | None = None,
    positions: int = 16,
    source_banks_per_role: int = 2,
) -> Path:
    """Create a V6-shaped deterministic code fixture with permanently forbidden use."""
    target = Path(path)
    if target.suffix != ".npz":
        target = Path(f"{target}.npz")
    if isinstance(source_banks_per_role, bool) or not isinstance(source_banks_per_role, int):
        raise ValueError("source_banks_per_role must be a positive integer")
    if source_banks_per_role < 1:
        raise ValueError("source_banks_per_role must be a positive integer")
    minimum_scenes = len(SOURCE_ROLES) * source_banks_per_role + 4
    if scene_count is None:
        scene_count = minimum_scenes
    if scene_count < minimum_scenes:
        raise ValueError(
            "fixture needs the requested banks for all seven source roles and "
            f"four target banks ({minimum_scenes} scenes)"
        )
    if positions < 8:
        raise ValueError("fixture needs at least eight positions")
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    bits = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    worlds, antennas, subcarriers, map_size, repeats = 4, 2, 4, 8, 3
    map_origin = np.asarray([-4.0, -4.0], dtype=np.float64)
    channels = 2 * antennas * subcarriers
    map_channel_names = np.asarray(["occupancy", "height", "material"], dtype="U16")
    maps = np.zeros((scene_count, worlds, 3, map_size, map_size), dtype=np.float64)
    coordinates = np.zeros((scene_count, positions, 2), dtype=np.float64)
    clean = np.zeros((scene_count, worlds, positions, channels), dtype=np.float64)
    primitive_ids = np.zeros((scene_count, 2), dtype=np.int64)
    anchor_bits = np.zeros((scene_count, 2), dtype=np.int64)
    natural = np.zeros(scene_count, dtype=np.int64)
    source_roles = [
        role
        for role in SOURCE_ROLES
        for _ in range(source_banks_per_role)
    ]
    roles = np.asarray(source_roles + ["target"] * (scene_count - len(source_roles)), dtype="U40")
    source_city_ids = [
        "source-a"
        if (bank_index if source_banks_per_role > 1 else role_index) % 2 == 0
        else "source-b"
        for role_index, _ in enumerate(SOURCE_ROLES)
        for bank_index in range(source_banks_per_role)
    ]
    target_count = scene_count - len(source_roles)
    target_city_ids = ["target-a" if index % 2 == 0 else "target-b" for index in range(target_count)]
    city_ids = np.asarray(source_city_ids + target_city_ids, dtype="U32")
    scene_ids = np.asarray([f"fixture-scene-{index:02d}" for index in range(scene_count)], dtype="U32")
    bank_ids = np.asarray([f"fixture-bank-{index:02d}" for index in range(scene_count)], dtype="U32")
    base_map_cluster_ids = np.asarray([f"fixture-base-{index:02d}" for index in range(scene_count)], dtype="U32")
    position_roles = np.full((scene_count, positions), "standard", dtype="U16")
    position_ids = np.empty((scene_count, positions), dtype="U64")
    free_space = np.ones((scene_count, worlds, positions), dtype=np.bool_)
    repeat_seeds = np.zeros((scene_count, worlds, positions, repeats), dtype=np.int64)
    phase_reference_ids = np.asarray(
        [[f"phase-reference:{scene}:{position}" for position in range(positions)] for scene in range(scene_count)],
        dtype="U64",
    )
    phase_reference_values = np.empty((scene_count, positions), dtype=np.complex128)
    phase_reference_source_sha256 = np.empty((scene_count, positions), dtype="U64")
    radio_config = np.tile(
        np.asarray([[3.5e9, float(antennas), float(subcarriers), 30e3]], dtype=np.float64),
        (scene_count, 1),
    )
    bs_pose = np.zeros((scene_count, 7), dtype=np.float64)
    bs_pose[:, 3] = 1.0
    path_count, interactions = 3, 2
    path_ids = np.full((scene_count, worlds, positions, path_count), -1, dtype=np.int64)
    path_power = np.zeros((scene_count, worlds, positions, path_count), dtype=np.float64)
    path_surface_ids = np.full(
        (scene_count, worlds, positions, path_count, interactions), -1, dtype=np.int64
    )
    primitive_surface_ids = np.full((scene_count, 2, 2), -1, dtype=np.int64)

    primitive_channel = np.zeros((2, channels), dtype=np.float64)
    primitive_channel[0, :8] = np.linspace(0.9, 1.5, 8)
    primitive_channel[0, 8:] = np.linspace(-0.6, 0.8, 8)
    primitive_channel[1, :8] = np.linspace(-1.1, 0.7, 8)
    primitive_channel[1, 8:] = np.linspace(1.3, -0.5, 8)

    for scene in range(scene_count):
        primitive_surface_ids[scene, 0, 0] = 1000 + scene * 10
        primitive_surface_ids[scene, 1, 0] = 1001 + scene * 10
        primitive_ids[scene] = np.asarray([0, 1] if scene % 2 == 0 else [1, 0])
        anchor_bits[scene] = np.asarray([(scene // 2) % 2, scene % 2])
        natural[scene] = next(index for index, row in enumerate(bits) if np.array_equal(row, anchor_bits[scene]))
        grid_centers_rc = _fixture_primitive_centers_rc(scene)
        centers_xy = grid_centers_rc[:, ::-1] + map_origin
        base_occupancy = np.zeros((map_size, map_size), dtype=np.float64)
        base_occupancy[0, :] = 1.0
        base_occupancy[:, 0] = 1.0
        base_height = np.zeros_like(base_occupancy)
        base_height[0, :] = 2.0
        base_height[:, 0] = 2.0
        base_material = np.zeros_like(base_occupancy)
        base_material[base_occupancy > 0] = 1.0
        primitive_masks = []
        for center in grid_centers_rc:
            mask = np.zeros_like(base_occupancy)
            row = int(round(center[0]))
            column = int(round(center[1]))
            mask[max(1, row - 1) : min(map_size, row + 2), max(1, column - 1) : min(map_size, column + 2)] = 1.0
            primitive_masks.append(mask)
        for world, world_bits in enumerate(bits):
            occupancy = base_occupancy.copy()
            height = base_height.copy()
            material = base_material.copy()
            for bit_index, enabled in enumerate(world_bits):
                primitive = int(primitive_ids[scene, bit_index])
                mask = primitive_masks[primitive] > 0
                if primitive == 0 and enabled:
                    occupancy[mask] = 1.0
                    height[mask] = 4.0
                    material[mask] = 2.0
                if primitive == 1:
                    occupancy[mask] = 1.0
                    height[mask] = 3.0 + 3.0 * float(enabled)
                    material[mask] = 1.0 + 2.0 * float(enabled)
            maps[scene, world] = np.stack((occupancy, height, material))

        common_free = np.all(maps[scene, :, 0] < 0.5, axis=0)
        free_cells = np.argwhere(common_free)
        if positions > len(free_cells):
            raise ValueError("fixture positions exceed common-free map cells")
        jitter = 0.25 + 0.5 * (scene + 1) / (scene_count + 1)
        selected = _stratified_fixture_cells(
            free_cells,
            grid_centers_rc,
            positions,
            jitter,
            rng,
        )
        coordinates[scene] = map_origin + np.stack(
            (selected[:, 1] + jitter, selected[:, 0] + jitter), axis=1
        )
        for position_index in range(positions):
            position_ids[scene, position_index] = f"{city_ids[scene]}:{bank_ids[scene]}:p{position_index:04d}"
            source_record = (
                "fixture-phase-reference-source-v1\0"
                f"{seed}\0{scene}\0{position_index}\0"
            ).encode("ascii") + np.asarray(
                coordinates[scene, position_index], dtype="<f8"
            ).tobytes()
            source_digest = hashlib.sha256(source_record).digest()
            phase_reference_source_sha256[scene, position_index] = (
                source_digest.hex()
            )
            phase_fraction = int.from_bytes(source_digest[:8], "big") / float(1 << 64)
            phase_reference_values[scene, position_index] = np.exp(
                2j * np.pi * phase_fraction
            )
        if roles[scene] == "target":
            support_count = positions // 2
            position_roles[scene, :support_count] = "support_pool"
            position_roles[scene, support_count:] = "query"

        city_shift = (0.03 * (scene % 4)) * np.linspace(-1.0, 1.0, channels)
        for position_index, coordinate in enumerate(coordinates[scene]):
            normalized = (coordinate - map_origin) / map_size
            base_csi = np.concatenate(
                (
                    np.sin(np.pi * normalized[0] * np.arange(1, 5)),
                    np.cos(np.pi * normalized[1] * np.arange(1, 5)),
                    np.sin(np.pi * (normalized[0] + normalized[1]) * np.arange(1, 5)),
                    np.cos(np.pi * (normalized[0] - normalized[1]) * np.arange(1, 5)),
                )
            )
            for world, world_bits in enumerate(bits):
                value = base_csi + city_shift
                for bit_index, enabled in enumerate(world_bits):
                    primitive = int(primitive_ids[scene, bit_index])
                    distance = float(np.linalg.norm(coordinate - centers_xy[primitive]))
                    amplitude = max(1e-4, 1.0 - distance / 3.0)
                    value = value + float(enabled) * amplitude * primitive_channel[primitive]
                complex_value = value[: channels // 2] + 1j * value[channels // 2 :]
                reference = phase_reference_values[scene, position_index]
                gauge_fixed = complex_value * np.conjugate(reference) / abs(reference)
                clean[scene, world, position_index] = np.concatenate(
                    (gauge_fixed.real, gauge_fixed.imag)
                )
                path_ids[scene, world, position_index] = np.asarray([0, 1, 2])
                path_power[scene, world, position_index] = np.asarray(
                    [1.0, 0.5 + 0.25 * float(world_bits[0]), 0.25 + 0.2 * float(world_bits[1])]
                )
                path_surface_ids[scene, world, position_index, 0, 0] = -1
                path_surface_ids[scene, world, position_index, 1, 0] = primitive_surface_ids[scene, 0, 0]
                path_surface_ids[scene, world, position_index, 2, 0] = primitive_surface_ids[scene, 1, 0]
                repeat_seeds[scene, world, position_index] = np.asarray(
                    [seed + scene * 1_000_000 + world * 100_000 + position_index * 100 + repeat for repeat in range(repeats)]
                )

    noise = np.empty((scene_count, worlds, positions, repeats, channels), dtype=np.float64)
    for scene in range(scene_count):
        for world in range(worlds):
            for position_index in range(positions):
                for repeat in range(repeats):
                    noise[scene, world, position_index, repeat] = np.random.default_rng(
                        int(repeat_seeds[scene, world, position_index, repeat])
                    ).normal(0.0, 0.001, size=channels)
    csi_repeat = clean[:, :, :, None, :] + noise
    canonical_map_sha256 = np.asarray(
        [[_array_sha256(maps[scene, world]) for world in range(worlds)] for scene in range(scene_count)],
        dtype="U64",
    )
    noop_maps = maps.copy()
    noop_map_sha256 = np.asarray(
        [[_array_sha256(noop_maps[scene, world]) for world in range(worlds)] for scene in range(scene_count)],
        dtype="U64",
    )
    engine_config = {
        "fixture": True,
        "map_channels": map_channel_names.tolist(),
        "seed": seed,
        "version": "2.1-v6",
    }
    engine_config_text = json.dumps(engine_config, sort_keys=True, separators=(",", ":"))
    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": "NONSCIENTIFIC-CODE-FIXTURE",
        "dataset_version": "2.1-v6",
        "scientific_use": "FORBIDDEN",
        "fixture": True,
        "engine": {
            "name": "deterministic-fixture-generator-not-rt",
            "version": "2.1-v6",
            "source_revision": "fixture-source-revision",
            "license_id": "GENERATED-FIXTURE-NO-EXTERNAL-ASSET",
            "config_sha256": hashlib.sha256(engine_config_text.encode("utf-8")).hexdigest(),
            "deterministic": True,
        },
        "representation": {
            "csi_layout": "real_then_imag",
            "csi_units": "arbitrary-fixture-units",
            "phase_gauge_rule": "shared_complex_reference",
            "coordinate_system": "bs_centered_right_handed_meters",
            "position_units": "m",
            "map_units": "m",
            "clean_target_definition": "deterministic fixture function before independent repeat noise",
            "antenna_count": antennas,
            "subcarrier_count": subcarriers,
            "patch_complex_size": 1,
            "patch_antenna_size": 1,
            "patch_subcarrier_size": 1,
            "map_resolution_m": 1.0,
            "map_origin_xy_m": map_origin.tolist(),
            "alignment_physical_representation": "complex_csi_plus_delay_angle_power",
        },
        "assets": {
            "license_ids": ["GENERATED-FIXTURE-NO-EXTERNAL-ASSET"],
            "provenance_uri": "local://formal_v2.formal_fixture",
            "material_library": "fixture-categorical-materials-v1",
            "material_category_count": 4,
            "redistribution_allowed": True,
        },
        "generation": {
            "created_utc": "2026-08-05T00:00:00Z",
            "generator_command": "python -m formal_v2.formal_cli make-fixture",
            "seed": seed,
        },
        "external_reference": {
            "available": False,
            "kind": "none-fixture-only",
            "dataset_id": "NOT-APPLICABLE",
            "pairing_rule": "NOT-APPLICABLE",
        },
    }
    try:
        with target.open("xb") as handle:
            np.savez_compressed(
                handle,
                csi_repeat=csi_repeat,
                csi_clean=clean,
                maps=maps,
                map_channel_names=map_channel_names,
                positions=coordinates,
                position_ids=position_ids,
                free_space=free_space,
                radio_config=radio_config,
                bs_pose=bs_pose,
                repeat_seeds=repeat_seeds,
                phase_reference_ids=phase_reference_ids,
                phase_reference_values=phase_reference_values,
                phase_reference_source_sha256=phase_reference_source_sha256,
                base_map_cluster_ids=base_map_cluster_ids,
                canonical_map_sha256=canonical_map_sha256,
                noop_maps=noop_maps,
                noop_map_sha256=noop_map_sha256,
                engine_config_json=np.asarray(engine_config_text),
                path_ids=path_ids,
                path_power=path_power,
                path_surface_ids=path_surface_ids,
                noop_path_ids=path_ids.copy(),
                noop_path_power=path_power.copy(),
                noop_path_surface_ids=path_surface_ids.copy(),
                primitive_surface_ids=primitive_surface_ids,
                world_bits=bits,
                primitive_ids=primitive_ids,
                anchor_bits=anchor_bits,
                natural_world_index=natural,
                scene_ids=scene_ids,
                city_ids=city_ids,
                bank_ids=bank_ids,
                scene_roles=roles,
                position_roles=position_roles,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite fixture: {target}") from error
    return target


def _stratified_fixture_cells(
    free_cells: np.ndarray,
    primitive_centers_rc: np.ndarray,
    position_count: int,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep both fixture support halves sensitive to both edit primitives."""
    cells = np.asarray(free_cells, dtype=np.int64)
    centers = np.asarray(primitive_centers_rc, dtype=np.float64)
    if cells.ndim != 2 or cells.shape[1] != 2 or centers.shape != (2, 2):
        raise ValueError("fixture cell geometry is malformed")
    if position_count < 8 or position_count > cells.shape[0]:
        raise ValueError("fixture position count is incompatible with free cells")

    selected: list[int] = []
    available = np.ones(cells.shape[0], dtype=np.bool_)
    block_sizes = (position_count // 2, position_count - position_count // 2)
    cell_centers = cells.astype(np.float64) + float(jitter)
    distances = np.linalg.norm(
        cell_centers[:, None, :] - centers[None, :, :], axis=-1
    )
    reserved_per_block = 2
    reserved: dict[int, list[int]] = {}
    primitive_order = sorted(
        range(centers.shape[0]),
        key=lambda primitive: int(np.sum(distances[:, primitive] < 3.0)),
    )
    for primitive in primitive_order:
        candidates = np.flatnonzero(available)
        ranked = candidates[np.argsort(distances[candidates, primitive], kind="stable")]
        chosen = ranked[: reserved_per_block * len(block_sizes)]
        if (
            chosen.size != reserved_per_block * len(block_sizes)
            or np.any(distances[chosen, primitive] >= 3.0)
        ):
            raise ValueError(
                "fixture geometry cannot cover both position halves for every primitive"
            )
        reserved[primitive] = chosen.astype(np.int64).tolist()
        available[chosen] = False

    for block_index, block_size in enumerate(block_sizes):
        block = [
            reserved[primitive][block_index * reserved_per_block + offset]
            for primitive in range(centers.shape[0])
            for offset in range(reserved_per_block)
        ]
        remaining = block_size - len(block)
        if remaining:
            candidates = np.flatnonzero(available)
            filler = rng.permutation(candidates)[:remaining].astype(np.int64).tolist()
            available[np.asarray(filler, dtype=np.int64)] = False
            block.extend(filler)
        selected.extend(block)
    return cells[np.asarray(selected, dtype=np.int64)]


def _fixture_primitive_centers_rc(scene: int) -> np.ndarray:
    scene_index = int(scene)
    foundation_variant = (scene_index % 2) ^ ((scene_index // 2) % 2)
    return np.asarray(
        [
            [2.0, 2.0 + float(scene_index % 2)],
            [5.0, 5.0 - float(foundation_variant)],
        ],
        dtype=np.float64,
    )
