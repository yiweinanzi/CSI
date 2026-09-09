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
WIGATR_CONFIG_SCHEMA = "csi-pairs-wigatr-official-adapter-v2"
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
    if config["profile"] not in {"formal-paper-dose", "software-smoke-only"}:
        raise ValueError("Wi-GATr adapter profile is invalid")
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
            "microbatch_size",
            "learning_rate",
            "weight_decay",
            "clip_grad_norm",
            "selection_every_steps",
        },
        "training",
    )
    training = config["training"]
    for key in ("seed", "steps", "batch_size", "microbatch_size", "selection_every_steps"):
        _positive_integer(training[key], f"training.{key}")
    for key in ("learning_rate", "clip_grad_norm"):
        _positive_number(training[key], f"training.{key}")
    if not isinstance(training["weight_decay"], (int, float)) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    if training["batch_size"] % training["microbatch_size"]:
        raise ValueError("Wi-GATr microbatch must divide the effective batch")
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
    if set(config["mesh"]) != {
        "occupancy_threshold",
        "minimum_height_m",
        "preprocessing",
    }:
        raise ValueError("mesh config fields must be exact")
    for key in ("occupancy_threshold", "minimum_height_m"):
        _positive_number(config["mesh"][key], f"mesh.{key}")
    if (
        config["mesh"]["preprocessing"]
        != "surface-ledger-equivalent-rectangle-compaction-v1"
    ):
        raise ValueError("Wi-GATr mesh preprocessing is not frozen")
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


def grid_to_cellwise_triangular_mesh(
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


def grid_to_triangular_mesh(
    maps: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    occupancy_threshold: float,
    minimum_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an equivalent surface mesh with coplanar rectangles merged."""
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Wi-GATr mesh conversion requires a square [channel,row,column] map")
    indices = {
        name: channel_names.index(name)
        for name in ("occupancy", "height", "material")
    }
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

    used = np.zeros_like(occupied, dtype=bool)
    for row in range(rows):
        for column in range(columns):
            if not occupied[row, column] or used[row, column]:
                continue
            top = float(heights[row, column])
            material = int(materials[row, column])
            width = 1
            while column + width < columns:
                next_column = column + width
                if (
                    not occupied[row, next_column]
                    or used[row, next_column]
                    or float(heights[row, next_column]) != top
                    or int(materials[row, next_column]) != material
                ):
                    break
                width += 1
            depth = 1
            while row + depth < rows:
                next_row = row + depth
                if any(
                    not occupied[next_row, next_column]
                    or used[next_row, next_column]
                    or float(heights[next_row, next_column]) != top
                    or int(materials[next_row, next_column]) != material
                    for next_column in range(column, column + width)
                ):
                    break
                depth += 1
            used[row : row + depth, column : column + width] = True
            x0 = origin_x + column * resolution
            x1 = origin_x + (column + width) * resolution
            y0 = origin_y + row * resolution
            y1 = origin_y + (row + depth) * resolution
            add_quad(
                (x0, y0, top),
                (x1, y0, top),
                (x1, y1, top),
                (x0, y1, top),
                material,
            )

    def exposed(row, column, neighbour_row, neighbour_column):
        if not occupied[row, column]:
            return None
        top = float(heights[row, column])
        if 0 <= neighbour_row < rows and 0 <= neighbour_column < columns:
            lower = (
                float(heights[neighbour_row, neighbour_column])
                if occupied[neighbour_row, neighbour_column]
                else 0.0
            )
        else:
            lower = 0.0
        if lower + 1e-12 >= top:
            return None
        return lower, top, int(materials[row, column])

    for row in range(rows):
        for direction in ("north", "south"):
            column = 0
            while column < columns:
                neighbour_row = row - 1 if direction == "north" else row + 1
                descriptor = exposed(row, column, neighbour_row, column)
                if descriptor is None:
                    column += 1
                    continue
                end_column = column + 1
                while end_column < columns and exposed(
                    row, end_column, neighbour_row, end_column
                ) == descriptor:
                    end_column += 1
                lower, top, material = descriptor
                x0 = origin_x + column * resolution
                x1 = origin_x + end_column * resolution
                if direction == "north":
                    y = origin_y + row * resolution
                    add_quad(
                        (x0, y, lower),
                        (x1, y, lower),
                        (x1, y, top),
                        (x0, y, top),
                        material,
                    )
                else:
                    y = origin_y + (row + 1) * resolution
                    add_quad(
                        (x1, y, lower),
                        (x0, y, lower),
                        (x0, y, top),
                        (x1, y, top),
                        material,
                    )
                column = end_column

    for column in range(columns):
        for direction in ("east", "west"):
            row = 0
            while row < rows:
                neighbour_column = column + 1 if direction == "east" else column - 1
                descriptor = exposed(row, column, row, neighbour_column)
                if descriptor is None:
                    row += 1
                    continue
                end_row = row + 1
                while end_row < rows and exposed(
                    end_row, column, end_row, neighbour_column
                ) == descriptor:
                    end_row += 1
                lower, top, material = descriptor
                y0 = origin_y + row * resolution
                y1 = origin_y + end_row * resolution
                if direction == "east":
                    x = origin_x + (column + 1) * resolution
                    add_quad(
                        (x, y0, lower),
                        (x, y1, lower),
                        (x, y1, top),
                        (x, y0, top),
                        material,
                    )
                else:
                    x = origin_x + column * resolution
                    add_quad(
                        (x, y1, lower),
                        (x, y0, lower),
                        (x, y0, top),
                        (x, y1, top),
                        material,
                    )
                row = end_row

    if not faces:
        return np.empty((0, 3, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    mesh = np.stack(faces)
    area = np.linalg.norm(
        np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]),
        axis=1,
    )
    if np.any(area <= 0):
        raise RuntimeError("Wi-GATr map conversion produced a degenerate triangle")
    return mesh, np.asarray(face_materials, dtype=np.int64)


def canonical_surface_ledger(
    mesh: np.ndarray,
    face_materials: np.ndarray,
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
) -> tuple[tuple[object, ...], ...]:
    """Expand paired axis-aligned triangles into canonical grid surface cells."""
    triangles = np.asarray(mesh, dtype=np.float32)
    materials = np.asarray(face_materials, dtype=np.int64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Wi-GATr surface ledger requires [face,vertex,xyz]")
    if materials.shape != (triangles.shape[0],) or triangles.shape[0] % 2:
        raise ValueError("Wi-GATr surface ledger requires paired face materials")
    resolution = float(resolution_m)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Wi-GATr surface ledger resolution must be positive")
    origin_x, origin_y = (float(value) for value in origin_xy_m)
    tolerance = max(1e-5, resolution * 1e-5)
    counts: dict[tuple[object, ...], int] = {}

    def grid_index(value: float, origin: float) -> int:
        index = int(round((float(value) - origin) / resolution))
        reconstructed = origin + index * resolution
        if abs(float(value) - reconstructed) > tolerance:
            raise RuntimeError("Wi-GATr surface vertex is off the frozen map grid")
        return index

    def height_token(value: float) -> str:
        return np.float32(value).tobytes().hex()

    def add(key: tuple[object, ...]) -> None:
        counts[key] = counts.get(key, 0) + 1

    for face in range(0, triangles.shape[0], 2):
        first = triangles[face]
        second = triangles[face + 1]
        if materials[face] != materials[face + 1]:
            raise RuntimeError("Wi-GATr paired triangles differ in material")
        if not (
            np.array_equal(first[0], second[0])
            and np.array_equal(first[2], second[1])
        ):
            raise RuntimeError("Wi-GATr triangles do not preserve add_quad pairing")
        vertices = np.stack((first[0], first[1], first[2], second[2]))
        normal = np.cross(first[1] - first[0], first[2] - first[0])
        active_axes = np.flatnonzero(np.abs(normal) > 1e-7)
        if active_axes.size != 1:
            raise RuntimeError("Wi-GATr surface ledger found a non-axis-aligned face")
        axis = int(active_axes[0])
        if float(np.ptp(vertices[:, axis])) > tolerance:
            raise RuntimeError("Wi-GATr paired triangles are not coplanar")
        direction = 1 if normal[axis] > 0 else -1
        material = int(materials[face])

        if axis == 2:
            x = sorted({grid_index(value, origin_x) for value in vertices[:, 0]})
            y = sorted({grid_index(value, origin_y) for value in vertices[:, 1]})
            if len(x) != 2 or len(y) != 2 or x[0] == x[1] or y[0] == y[1]:
                raise RuntimeError("Wi-GATr top surface has invalid grid bounds")
            plane = height_token(vertices[0, 2])
            for column in range(x[0], x[1]):
                for row in range(y[0], y[1]):
                    add((axis, direction, plane, column, row, material))
        elif axis == 0:
            plane = grid_index(vertices[0, 0], origin_x)
            y = sorted({grid_index(value, origin_y) for value in vertices[:, 1]})
            z = sorted({height_token(value) for value in vertices[:, 2]})
            if len(y) != 2 or len(z) != 2 or y[0] == y[1] or z[0] == z[1]:
                raise RuntimeError("Wi-GATr x-normal surface has invalid bounds")
            for row in range(y[0], y[1]):
                add((axis, direction, plane, row, z[0], z[1], material))
        else:
            plane = grid_index(vertices[0, 1], origin_y)
            x = sorted({grid_index(value, origin_x) for value in vertices[:, 0]})
            z = sorted({height_token(value) for value in vertices[:, 2]})
            if len(x) != 2 or len(z) != 2 or x[0] == x[1] or z[0] == z[1]:
                raise RuntimeError("Wi-GATr y-normal surface has invalid bounds")
            for column in range(x[0], x[1]):
                add((axis, direction, plane, column, z[0], z[1], material))
    return tuple((*key, count) for key, count in sorted(counts.items()))


def require_surface_ledger_equivalence(
    reference_mesh: np.ndarray,
    reference_materials: np.ndarray,
    compact_mesh: np.ndarray,
    compact_materials: np.ndarray,
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
) -> tuple[tuple[object, ...], ...]:
    reference = canonical_surface_ledger(
        reference_mesh,
        reference_materials,
        resolution_m=resolution_m,
        origin_xy_m=origin_xy_m,
    )
    compact = canonical_surface_ledger(
        compact_mesh,
        compact_materials,
        resolution_m=resolution_m,
        origin_xy_m=origin_xy_m,
    )
    if compact != reference:
        raise RuntimeError("Wi-GATr compact mesh changed the canonical surface ledger")
    return compact


def require_compact_mesh_matches_map_surface(
    maps: np.ndarray,
    channel_names: tuple[str, ...],
    compact_mesh: np.ndarray,
    compact_materials: np.ndarray,
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
    occupancy_threshold: float,
    minimum_height_m: float,
) -> tuple[str, int]:
    """Compare a compact mesh against an independent grid-derived surface ledger."""
    expected = _map_surface_arrays(
        maps,
        channel_names,
        occupancy_threshold=occupancy_threshold,
        minimum_height_m=minimum_height_m,
    )
    observed = _mesh_surface_arrays(
        compact_mesh,
        compact_materials,
        expected["top"]["valid"].shape,
        resolution_m=resolution_m,
        origin_xy_m=origin_xy_m,
    )
    digest = hashlib.sha256()
    atomic_quad_count = 0
    for surface in ("top", "x_positive", "x_negative", "y_positive", "y_negative"):
        for field in ("valid", "lower", "upper", "material"):
            left = expected[surface][field]
            right = observed[surface][field]
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Wi-GATr compact mesh changed map surface {surface}.{field}"
                )
            name = f"{surface}.{field}".encode("ascii")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(left.dtype.str.encode("ascii"))
            digest.update(np.asarray(left.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(left).tobytes())
        atomic_quad_count += int(np.sum(expected[surface]["valid"]))
    return digest.hexdigest(), atomic_quad_count


def _surface_layer(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "valid": np.zeros(shape, dtype=bool),
        "lower": np.zeros(shape, dtype=np.float32),
        "upper": np.zeros(shape, dtype=np.float32),
        "material": np.zeros(shape, dtype=np.int64),
    }


def _assign_surface(
    layer: dict[str, np.ndarray],
    selection,
    mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    material: np.ndarray,
) -> None:
    valid_view = layer["valid"][selection]
    if valid_view.shape != mask.shape or np.any(valid_view & mask):
        raise RuntimeError("Wi-GATr map surface ledger contains overlapping faces")
    valid_view[mask] = True
    layer["lower"][selection][mask] = np.asarray(lower, dtype=np.float32)[mask]
    layer["upper"][selection][mask] = np.asarray(upper, dtype=np.float32)[mask]
    layer["material"][selection][mask] = np.asarray(material, dtype=np.int64)[mask]


def _map_surface_arrays(
    maps: np.ndarray,
    channel_names: tuple[str, ...],
    *,
    occupancy_threshold: float,
    minimum_height_m: float,
) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Wi-GATr surface audit requires a square channel map")
    indices = {
        name: channel_names.index(name)
        for name in ("occupancy", "height", "material")
    }
    occupied = values[indices["occupancy"]] >= float(occupancy_threshold)
    heights = values[indices["height"]]
    materials = np.rint(values[indices["material"]]).astype(np.int64)
    if np.any(occupied & (heights < float(minimum_height_m))):
        raise ValueError("occupied Wi-GATr mesh cells must have positive physical height")
    if np.any(materials < 0):
        raise ValueError("Wi-GATr material IDs must be nonnegative")
    rows, columns = occupied.shape
    layers = {
        "top": _surface_layer((rows, columns)),
        "x_positive": _surface_layer((rows, columns + 1)),
        "x_negative": _surface_layer((rows, columns + 1)),
        "y_positive": _surface_layer((rows + 1, columns)),
        "y_negative": _surface_layer((rows + 1, columns)),
    }
    zeros = np.zeros_like(heights)
    _assign_surface(
        layers["top"],
        np.s_[:, :],
        occupied,
        zeros,
        heights,
        materials,
    )

    neighbour_occupied = np.zeros_like(occupied)
    neighbour_height = np.zeros_like(heights)
    neighbour_occupied[:, :-1] = occupied[:, 1:]
    neighbour_height[:, :-1] = heights[:, 1:]
    lower = np.where(neighbour_occupied, neighbour_height, 0.0)
    exposed = occupied & (lower + 1e-12 < heights)
    _assign_surface(
        layers["x_positive"], np.s_[:, 1:], exposed, lower, heights, materials
    )

    neighbour_occupied.fill(False)
    neighbour_height.fill(0.0)
    neighbour_occupied[:, 1:] = occupied[:, :-1]
    neighbour_height[:, 1:] = heights[:, :-1]
    lower = np.where(neighbour_occupied, neighbour_height, 0.0)
    exposed = occupied & (lower + 1e-12 < heights)
    _assign_surface(
        layers["x_negative"], np.s_[:, :-1], exposed, lower, heights, materials
    )

    neighbour_occupied.fill(False)
    neighbour_height.fill(0.0)
    neighbour_occupied[:-1, :] = occupied[1:, :]
    neighbour_height[:-1, :] = heights[1:, :]
    lower = np.where(neighbour_occupied, neighbour_height, 0.0)
    exposed = occupied & (lower + 1e-12 < heights)
    _assign_surface(
        layers["y_positive"], np.s_[1:, :], exposed, lower, heights, materials
    )

    neighbour_occupied.fill(False)
    neighbour_height.fill(0.0)
    neighbour_occupied[1:, :] = occupied[:-1, :]
    neighbour_height[1:, :] = heights[:-1, :]
    lower = np.where(neighbour_occupied, neighbour_height, 0.0)
    exposed = occupied & (lower + 1e-12 < heights)
    _assign_surface(
        layers["y_negative"], np.s_[:-1, :], exposed, lower, heights, materials
    )
    return layers


def _mesh_surface_arrays(
    mesh: np.ndarray,
    face_materials: np.ndarray,
    map_shape: tuple[int, int],
    *,
    resolution_m: float,
    origin_xy_m: tuple[float, float],
) -> dict[str, dict[str, np.ndarray]]:
    triangles = np.asarray(mesh, dtype=np.float32)
    materials = np.asarray(face_materials, dtype=np.int64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or materials.shape != (triangles.shape[0],)
        or triangles.shape[0] % 2
    ):
        raise ValueError("Wi-GATr compact surface audit requires paired triangles")
    rows, columns = map_shape
    layers = {
        "top": _surface_layer((rows, columns)),
        "x_positive": _surface_layer((rows, columns + 1)),
        "x_negative": _surface_layer((rows, columns + 1)),
        "y_positive": _surface_layer((rows + 1, columns)),
        "y_negative": _surface_layer((rows + 1, columns)),
    }
    resolution = float(resolution_m)
    origin_x, origin_y = (float(value) for value in origin_xy_m)
    tolerance = max(1e-5, resolution * 1e-5)

    def grid_index(value: float, origin: float) -> int:
        index = int(round((float(value) - origin) / resolution))
        if abs(float(value) - (origin + index * resolution)) > tolerance:
            raise RuntimeError("Wi-GATr compact surface vertex is off-grid")
        return index

    def fill(layer, selection, lower, upper, material):
        if np.any(layer["valid"][selection]):
            raise RuntimeError("Wi-GATr compact surface contains overlapping quads")
        layer["valid"][selection] = True
        layer["lower"][selection] = np.float32(lower)
        layer["upper"][selection] = np.float32(upper)
        layer["material"][selection] = int(material)

    for face in range(0, triangles.shape[0], 2):
        first = triangles[face]
        second = triangles[face + 1]
        if materials[face] != materials[face + 1] or not (
            np.array_equal(first[0], second[0])
            and np.array_equal(first[2], second[1])
        ):
            raise RuntimeError("Wi-GATr compact surface lost quad pairing")
        vertices = np.stack((first[0], first[1], first[2], second[2]))
        normal = np.cross(first[1] - first[0], first[2] - first[0])
        active_axes = np.flatnonzero(np.abs(normal) > 1e-7)
        if active_axes.size != 1:
            raise RuntimeError("Wi-GATr compact surface contains a non-axis face")
        axis = int(active_axes[0])
        direction = 1 if normal[axis] > 0 else -1
        material = int(materials[face])
        if axis == 2:
            if direction != 1:
                raise RuntimeError("Wi-GATr compact mesh contains an unexpected bottom face")
            x = sorted({grid_index(value, origin_x) for value in vertices[:, 0]})
            y = sorted({grid_index(value, origin_y) for value in vertices[:, 1]})
            if len(x) != 2 or len(y) != 2:
                raise RuntimeError("Wi-GATr compact top face has invalid bounds")
            fill(
                layers["top"],
                np.s_[y[0] : y[1], x[0] : x[1]],
                0.0,
                vertices[0, 2],
                material,
            )
        elif axis == 0:
            plane = grid_index(vertices[0, 0], origin_x)
            y = sorted({grid_index(value, origin_y) for value in vertices[:, 1]})
            z = sorted({float(value) for value in vertices[:, 2]})
            if len(y) != 2 or len(z) != 2:
                raise RuntimeError("Wi-GATr compact x-normal face has invalid bounds")
            layer = layers["x_positive" if direction > 0 else "x_negative"]
            fill(layer, np.s_[y[0] : y[1], plane], z[0], z[1], material)
        else:
            plane = grid_index(vertices[0, 1], origin_y)
            x = sorted({grid_index(value, origin_x) for value in vertices[:, 0]})
            z = sorted({float(value) for value in vertices[:, 2]})
            if len(x) != 2 or len(z) != 2:
                raise RuntimeError("Wi-GATr compact y-normal face has invalid bounds")
            layer = layers["y_positive" if direction > 0 else "y_negative"]
            fill(layer, np.s_[plane, x[0] : x[1]], z[0], z[1], material)
    return layers


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
