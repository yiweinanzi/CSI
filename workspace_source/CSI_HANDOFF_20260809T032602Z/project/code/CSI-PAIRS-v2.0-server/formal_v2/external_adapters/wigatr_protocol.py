from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from formal_v2.formal_dataset import FormalDataset, _array_sha256
from formal_v2.formal_io import read_strict_json


WIGATR_SOURCE_REVISION = "6daa5bd49d9499817d903ab335830221d01e9daa"
WIGATR_ARCHIVE_SHA256 = "b78be8ed14d16aab6c8fea54c2aa90a22bed492a12a6e892ee4c225c60a8138c"
WIGATR_CONFIG_SCHEMA = "csi-pairs-wigatr-official-adapter-v1"
SIX_CONDITIONS = (
    "correct",
    "paired_active_alternative",
    "paired_null_alternative",
    "wrong_city",
    "geometry_destroyed",
    "empty",
)


@dataclass(frozen=True)
class SixConditionUnit:
    scene: int
    source_world: int
    active_world: int
    null_world: int
    wrong_city_scene: int
    wrong_city_world: int
    position: int
    unit_id: str
    csi_context_sha256: str


def load_wigatr_config(path: str | Path) -> dict:
    config = read_strict_json(path)
    required = {
        "schema_version",
        "profile",
        "source_revision",
        "source_roles",
        "model",
        "training",
        "inverse",
        "mesh",
        "power",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Wi-GATr adapter config fields must be exact")
    if config["schema_version"] != WIGATR_CONFIG_SCHEMA:
        raise ValueError("Wi-GATr adapter config schema mismatch")
    if config["profile"] != "formal-paper-dose":
        raise ValueError("scientific Wi-GATr execution requires the formal paper-dose profile")
    if config["source_revision"] != WIGATR_SOURCE_REVISION:
        raise ValueError("Wi-GATr source revision is not the frozen official snapshot")
    if config["source_roles"] != {
        "train": "source_encoder_train",
        "selection": "source_method_selection",
        "evaluation": ["source_final_unseen_bank", "target"],
    }:
        raise ValueError("Wi-GATr role ledger differs from the frozen adapter contract")
    _require_exact_numeric_section(
        config["model"],
        {
            "embedding",
            "hidden_mv_channels",
            "hidden_s_channels",
            "num_blocks",
            "num_heads",
            "multi_query",
        },
        "model",
    )
    model = config["model"]
    if model["embedding"] != "kitchen_sink_z" or model["multi_query"] is not True:
        raise ValueError("Wi-GATr must retain the official kitchen_sink_z multi-query architecture")
    for key in ("hidden_mv_channels", "hidden_s_channels", "num_blocks", "num_heads"):
        _positive_integer(model[key], f"model.{key}")
    _require_exact_numeric_section(
        config["training"],
        {
            "seed",
            "steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "clip_grad_norm",
            "selection_every_steps",
        },
        "training",
    )
    training = config["training"]
    for key in ("seed", "steps", "batch_size", "selection_every_steps"):
        _positive_integer(training[key], f"training.{key}")
    for key in ("learning_rate", "clip_grad_norm"):
        _positive_number(training[key], f"training.{key}")
    if not isinstance(training["weight_decay"], (int, float)) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    _require_exact_numeric_section(
        config["inverse"],
        {
            "seed",
            "receiver_z_m",
            "restarts",
            "steps",
            "learning_rate",
            "clip_grad_norm",
        },
        "inverse",
    )
    inverse = config["inverse"]
    for key in ("seed", "restarts", "steps"):
        _positive_integer(inverse[key], f"inverse.{key}")
    for key in ("learning_rate", "clip_grad_norm"):
        _positive_number(inverse[key], f"inverse.{key}")
    if not isinstance(inverse["receiver_z_m"], (int, float)):
        raise ValueError("inverse.receiver_z_m must be numeric")
    if set(config["mesh"]) != {"occupancy_threshold", "minimum_height_m"}:
        raise ValueError("mesh config fields must be exact")
    for key in config["mesh"]:
        _positive_number(config["mesh"][key], f"mesh.{key}")
    if set(config["power"]) != {"floor", "definition"}:
        raise ValueError("power config fields must be exact")
    _positive_number(config["power"]["floor"], "power.floor")
    if config["power"]["definition"] != "10log10(mean(real_csi^2+imag_csi^2))":
        raise ValueError("Wi-GATr power definition is not frozen")
    return config


def relative_total_power_db(csi: np.ndarray, floor: float) -> np.ndarray:
    values = np.asarray(csi, dtype=np.float64)
    if values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError("real-then-imag CSI must have an even nonzero channel axis")
    half = values.shape[-1] // 2
    power = np.mean(values[..., :half] ** 2 + values[..., half:] ** 2, axis=-1)
    return 10.0 * np.log10(np.maximum(power, float(floor)))


def grid_to_triangular_mesh(
    maps: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    occupancy_threshold: float,
    minimum_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Wi-GATr mesh conversion requires a square [channel,row,column] map")
    indices = {name: channel_names.index(name) for name in ("occupancy", "height", "material")}
    occupied = values[indices["occupancy"]] >= float(occupancy_threshold)
    heights = values[indices["height"]]
    materials = np.rint(values[indices["material"]]).astype(np.int64)
    if np.any(occupied & (heights < float(minimum_height_m))):
        raise ValueError("occupied Wi-GATr mesh cells must have positive physical height")
    if np.any(materials < 0):
        raise ValueError("Wi-GATr material IDs must be nonnegative")
    faces: list[np.ndarray] = []
    face_materials: list[int] = []
    rows, columns = occupied.shape
    origin_x, origin_y = (float(value) for value in origin_xy_m)
    resolution = float(resolution_m)

    def add_quad(a, b, c, d, material):
        faces.append(np.asarray((a, b, c), dtype=np.float32))
        faces.append(np.asarray((a, c, d), dtype=np.float32))
        face_materials.extend((int(material), int(material)))

    for row in range(rows):
        for column in range(columns):
            if not occupied[row, column]:
                continue
            x0 = origin_x + column * resolution
            x1 = x0 + resolution
            y0 = origin_y + row * resolution
            y1 = y0 + resolution
            top = float(heights[row, column])
            material = int(materials[row, column])
            add_quad((x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top), material)
            neighbours = (
                (row - 1, column, (x0, y0), (x1, y0)),
                (row, column + 1, (x1, y0), (x1, y1)),
                (row + 1, column, (x1, y1), (x0, y1)),
                (row, column - 1, (x0, y1), (x0, y0)),
            )
            for neighbour_row, neighbour_column, start, end in neighbours:
                if 0 <= neighbour_row < rows and 0 <= neighbour_column < columns:
                    lower = float(heights[neighbour_row, neighbour_column]) if occupied[neighbour_row, neighbour_column] else 0.0
                else:
                    lower = 0.0
                if lower + 1e-12 >= top:
                    continue
                add_quad(
                    (start[0], start[1], lower),
                    (end[0], end[1], lower),
                    (end[0], end[1], top),
                    (start[0], start[1], top),
                    material,
                )
    if not faces:
        return np.empty((0, 3, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    mesh = np.stack(faces)
    area = np.linalg.norm(np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]), axis=1)
    if np.any(area <= 0):
        raise RuntimeError("Wi-GATr map conversion produced a degenerate triangle")
    return mesh, np.asarray(face_materials, dtype=np.int64)


def geometry_destroyed_map(maps: np.ndarray, unit_id: str) -> np.ndarray:
    values = np.asarray(maps)
    if values.ndim != 3:
        raise ValueError("geometry-destroyed input must be [channel,row,column]")
    seed = int.from_bytes(hashlib.sha256(unit_id.encode("utf-8")).digest()[:8], "little")
    order = np.random.default_rng(seed).permutation(values.shape[1] * values.shape[2])
    return values.reshape(values.shape[0], -1)[:, order].reshape(values.shape).copy()


def build_six_condition_units(dataset: FormalDataset, routed) -> list[SixConditionUnit]:
    evaluation_scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    wrong_city = _wrong_city_lookup(dataset, tuple(int(value) for value in evaluation_scenes))
    units = []
    for scene_value in evaluation_scenes:
        scene = int(scene_value)
        positions = _eligible_positions(dataset, scene)
        neighbours: dict[int, list[int]] = {world: [] for world in range(dataset.world_count)}
        for edge in dataset.directed_edges(scene):
            neighbours[edge.source_world].append(edge.target_world)
        for source_world in range(dataset.world_count):
            for position in positions:
                active = sorted(
                    target
                    for target in neighbours[source_world]
                    if routed.alignment_route[(scene, source_world, target, int(position))] == 2
                )
                null = sorted(
                    target
                    for target in neighbours[source_world]
                    if routed.alignment_route[(scene, source_world, target, int(position))] == 0
                )
                if not active or not null:
                    continue
                unit_id = (
                    f"{dataset.bank_ids[scene]}:{source_world}:"
                    f"{dataset.position_ids[scene, position]}"
                )
                units.append(
                    SixConditionUnit(
                        scene=scene,
                        source_world=source_world,
                        active_world=active[0],
                        null_world=null[0],
                        wrong_city_scene=wrong_city[scene],
                        wrong_city_world=int(dataset.natural_world_index[wrong_city[scene]]),
                        position=int(position),
                        unit_id=unit_id,
                        csi_context_sha256=_csi_context_sha256(
                            dataset, scene, source_world, int(position)
                        ),
                    )
                )
    if not units:
        raise RuntimeError("Wi-GATr six-condition audit has no units with both active and null alternatives")
    return units


def condition_map(dataset: FormalDataset, unit: SixConditionUnit, condition: str) -> np.ndarray:
    if condition == "correct":
        return dataset.maps[unit.scene, unit.source_world]
    if condition == "paired_active_alternative":
        return dataset.maps[unit.scene, unit.active_world]
    if condition == "paired_null_alternative":
        return dataset.maps[unit.scene, unit.null_world]
    if condition == "wrong_city":
        return dataset.maps[unit.wrong_city_scene, unit.wrong_city_world]
    if condition == "geometry_destroyed":
        return geometry_destroyed_map(
            dataset.maps[unit.scene, unit.source_world], unit.unit_id
        )
    if condition == "empty":
        return np.zeros_like(dataset.maps[unit.scene, unit.source_world])
    raise ValueError(f"unknown Wi-GATr condition: {condition}")


def supplied_map_sha256(supplied_map: np.ndarray) -> str:
    """Digest the exact map array handed to an external model."""
    return _array_sha256(np.asarray(supplied_map))


def condition_action_sha256(
    dataset: FormalDataset, unit: SixConditionUnit, condition: str
) -> str:
    target_world = None
    if condition == "paired_active_alternative":
        target_world = int(unit.active_world)
    elif condition == "paired_null_alternative":
        target_world = int(unit.null_world)
    payload: dict[str, object] = {"condition": condition, "action": None}
    if target_world is not None:
        matches = [
            edge for edge in dataset.directed_edges(int(unit.scene))
            if int(edge.source_world) == int(unit.source_world)
            and int(edge.target_world) == target_world
        ]
        if len(matches) != 1:
            raise RuntimeError("six-condition input does not identify one directed action")
        edge = matches[0]
        payload["action"] = {
            "source_world": int(edge.source_world),
            "target_world": int(edge.target_world),
            "bit_index": int(edge.bit_index),
            "primitive_id": int(edge.primitive_id),
            "direction": int(edge.direction),
        }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def map_bounds(dataset: FormalDataset) -> tuple[tuple[float, float], tuple[float, float]]:
    representation = dataset.metadata["representation"]
    origin = representation["map_origin_xy_m"]
    resolution = float(representation["map_resolution_m"])
    extent = resolution * dataset.maps.shape[-1]
    return (
        (float(origin[0]), float(origin[0]) + extent),
        (float(origin[1]), float(origin[1]) + extent),
    )


def validate_fixed_radio_contract(dataset: FormalDataset) -> None:
    reference = dataset.radio_config[0]
    if not np.allclose(dataset.radio_config, reference[None, :], rtol=0.0, atol=0.0):
        raise RuntimeError(
            "official Wi-GATr assumes one frozen carrier/antenna configuration; "
            "the current dataset varies radio_config"
        )


def _eligible_positions(dataset: FormalDataset, scene: int) -> np.ndarray:
    if str(dataset.scene_roles[scene]) == "target":
        values = np.flatnonzero(dataset.position_roles[scene] == "query")
        if values.size == 0:
            raise RuntimeError("Wi-GATr target evaluation has no query positions")
        return values
    return np.arange(dataset.position_count, dtype=np.int64)


def _wrong_city_lookup(dataset: FormalDataset, scenes: tuple[int, ...]) -> dict[int, int]:
    result = {}
    for scene in scenes:
        candidates = sorted(
            (
                str(dataset.bank_ids[other]),
                other,
            )
            for other in scenes
            if str(dataset.city_ids[other]) != str(dataset.city_ids[scene])
        )
        if not candidates:
            raise RuntimeError("Wi-GATr wrong-city condition requires another evaluation city")
        result[scene] = int(candidates[0][1])
    return result


def _csi_context_sha256(
    dataset: FormalDataset, scene: int, world: int, position: int
) -> str:
    digest = hashlib.sha256()
    for value in (
        dataset.csi_clean[scene, world, position],
        dataset.radio_config[scene],
        dataset.bs_pose[scene],
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_exact_numeric_section(section, fields, name):
    if not isinstance(section, dict) or set(section) != fields:
        raise ValueError(f"{name} config fields must be exact")


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value, name):
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
