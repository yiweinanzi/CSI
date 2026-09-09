from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .formal_io import read_strict_json, sha256_file
from .formal_physics_response import (
    action_rectangle_xy,
    complex_csi,
    direct_path_basis,
    material_direction,
    normalized_patch_delta,
    reflection_path,
    visible_complex_mask,
)
from .formal_protocol import PatchSpec, frozen_mask_query_bank


PHYSICAL_RESPONSE_CONFIG_SCHEMA = "csi-pairs-physical-response-config-v1"
PHYSICAL_RESPONSE_CHECKPOINT_SCHEMA = "csi-pairs-formal-physical-response-v3"
PHYSICAL_RESPONSE_CONTRACT = (
    "source-csi-source-map-id-free-action-radio-mask-query-"
    "physical-inverse-response-with-fit-only-mean-gate-ensemble-shared-train-amplitude-"
    "and-oracle-position-control-v3"
)


class PhysicalResponseError(RuntimeError):
    pass


class PhysicalNullGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


@dataclass
class PhysicalResponseModel:
    coefficient_calibration: dict[tuple[int, int], np.ndarray]
    gates: list[PhysicalNullGate]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    thresholds: dict[str, float]
    response_scale: float
    metadata: dict


def load_physical_response_config(path: str | Path | None = None) -> dict:
    source = (
        Path(__file__).resolve().parent / "configs" / "physical_response_v1.json"
        if path is None
        else Path(path).resolve()
    )
    config = read_strict_json(source)
    required = {
        "schema_version",
        "calibration_ridge",
        "decomposition_ridge",
        "fit_position_stride",
        "gate_batch_size",
        "gate_ensemble_size",
        "gate_seed_offset",
        "gate_seed_stride",
        "gate_steps_per_member",
        "grid_spacing_m",
        "maximum_fit_bank_group_false_positive_rate",
        "selection_position_stride",
        "world_scope",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("physical response config fields differ from the frozen schema")
    if config["schema_version"] != PHYSICAL_RESPONSE_CONFIG_SCHEMA:
        raise ValueError("physical response config schema is unsupported")
    for key in (
        "fit_position_stride",
        "gate_batch_size",
        "gate_ensemble_size",
        "gate_seed_offset",
        "gate_seed_stride",
        "gate_steps_per_member",
        "selection_position_stride",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"physical response config {key} must be a positive integer")
    for key in (
        "calibration_ridge",
        "decomposition_ridge",
        "grid_spacing_m",
        "maximum_fit_bank_group_false_positive_rate",
    ):
        if not isinstance(config[key], (int, float)) or not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"physical response config {key} must be positive and finite")
    if not 0.0 < float(config["maximum_fit_bank_group_false_positive_rate"]) < 1.0:
        raise ValueError("physical response false-positive rate must be in (0, 1)")
    if config["world_scope"] != "all" or int(config["selection_position_stride"]) != 1:
        raise ValueError("formal physical response must evaluate all worlds and positions")
    return config


def load_bound_candidate_config(dataset) -> tuple[dict, Path]:
    path = Path(dataset.source_path).resolve().parent / "assets" / "generator_config.json"
    if path.is_symlink() or not path.is_file():
        raise PhysicalResponseError(
            f"formal physical response requires a regular generator snapshot: {path}"
        )
    from .sionna_osm_candidate import load_config

    config = load_config(path)
    representation = dataset.metadata["representation"]
    if (
        int(config["radio"]["tx_antennas"]) != int(representation["antenna_count"])
        or int(config["radio"]["subcarriers"]) != int(representation["subcarrier_count"])
        or float(config["map"]["resolution_m"])
        != float(representation["map_resolution_m"])
        or list(config["map"]["origin_xy_m"])
        != list(representation["map_origin_xy_m"])
    ):
        raise PhysicalResponseError("generator snapshot disagrees with formal dataset metadata")
    return config, path


def _candidate_positions(
    source_map: np.ndarray,
    *,
    origin_xy_m: np.ndarray,
    resolution_m: float,
    grid_spacing_m: float,
    minimum_bs_distance_m: float,
    transmitter_z_m: float,
    receiver_z_m: float,
    minimum_vertical_clearance_m: float,
) -> np.ndarray:
    size = int(source_map.shape[-1])
    origin = np.asarray(origin_xy_m, dtype=np.float64)
    spacing = float(grid_spacing_m)
    x = np.arange(origin[0] + 0.5 * spacing, origin[0] + size * resolution_m, spacing)
    y = np.arange(origin[1] + 0.5 * spacing, origin[1] + size * resolution_m, spacing)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    candidates = np.stack((xx.ravel(), yy.ravel()), axis=1)
    columns = np.floor((candidates[:, 0] - origin[0]) / resolution_m).astype(int)
    rows = np.floor((candidates[:, 1] - origin[1]) / resolution_m).astype(int)
    inside = (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size)
    free = np.zeros(candidates.shape[0], dtype=np.bool_)
    free[inside] = source_map[0, rows[inside], columns[inside]] <= 0.0
    distance = np.linalg.norm(candidates, axis=1)
    keep = free & (distance >= float(minimum_bs_distance_m))
    for index in np.flatnonzero(keep):
        receiver = candidates[index]
        sample_count = max(2, int(np.ceil(distance[index] / resolution_m)) + 1)
        fraction = np.linspace(0.0, 1.0, sample_count)
        points = fraction[:, None] * receiver[None, :]
        sample_columns = np.floor((points[:, 0] - origin[0]) / resolution_m).astype(int)
        sample_rows = np.floor((points[:, 1] - origin[1]) / resolution_m).astype(int)
        sample_inside = (
            (sample_rows >= 0)
            & (sample_rows < size)
            & (sample_columns >= 0)
            & (sample_columns < size)
        )
        heights = np.zeros(sample_count, dtype=np.float64)
        heights[sample_inside] = source_map[
            1, sample_rows[sample_inside], sample_columns[sample_inside]
        ]
        ray_height = transmitter_z_m + fraction * (receiver_z_m - transmitter_z_m)
        if np.any(
            (heights > 0.0)
            & (ray_height - heights <= float(minimum_vertical_clearance_m))
        ):
            keep[index] = False
    return candidates[keep]


def _candidate_dictionary(
    candidates: np.ndarray,
    rectangle: tuple[float, float, float, float],
    candidate_config: dict,
    spec: PatchSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radio = candidate_config["radio"]
    intervention = candidate_config["interventions"]
    map_config = candidate_config["map"]
    positions = []
    direct = []
    reflected = []
    for candidate in candidates:
        path = reflection_path(
            rectangle,
            candidate,
            primitive_height_m=float(intervention["height_m"]),
            transmitter_z_m=float(radio["transmitter_z_m"]),
            receiver_z_m=float(radio["receiver_z_m"]),
            carrier_frequency_hz=float(radio["carrier_frequency_hz"]),
            subcarrier_spacing_hz=float(radio["subcarrier_spacing_hz"]),
            antennas=spec.antennas,
            subcarriers=spec.subcarriers,
            minimum_incidence_cosine=float(map_config["minimum_reflection_incidence_cosine"]),
        )
        if path is None:
            continue
        positions.append(candidate)
        direct.append(
            direct_path_basis(
                candidate,
                transmitter_z_m=float(radio["transmitter_z_m"]),
                receiver_z_m=float(radio["receiver_z_m"]),
                antennas=spec.antennas,
                subcarriers=spec.subcarriers,
            ).reshape(-1)
        )
        reflected.append(path.basis.reshape(-1))
    if not positions:
        raise PhysicalResponseError("action geometry has no valid receiver candidates")
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(direct, dtype=np.complex128),
        np.asarray(reflected, dtype=np.complex128),
    )


def _pair_fit(
    source_csi: np.ndarray,
    visible: np.ndarray,
    direct: np.ndarray,
    reflected: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.asarray(visible, dtype=np.bool_).reshape(-1)
    target = np.asarray(source_csi, dtype=np.complex128).reshape(-1)[keep]
    first = direct[:, keep]
    second = reflected[:, keep]
    g00 = np.sum(np.abs(first) ** 2, axis=1) + ridge
    g11 = np.sum(np.abs(second) ** 2, axis=1) + ridge
    g01 = np.sum(np.conjugate(first) * second, axis=1)
    h0 = np.sum(np.conjugate(first) * target[None, :], axis=1)
    h1 = np.sum(np.conjugate(second) * target[None, :], axis=1)
    determinant = g00 * g11 - np.abs(g01) ** 2
    stable = determinant > max(ridge**2, 1e-15)
    c0 = np.zeros_like(h0)
    c1 = np.zeros_like(h1)
    c0[stable] = (
        g11[stable] * h0[stable] - g01[stable] * h1[stable]
    ) / determinant[stable]
    c1[stable] = (
        -np.conjugate(g01[stable]) * h0[stable] + g00[stable] * h1[stable]
    ) / determinant[stable]
    prediction = c0[:, None] * first + c1[:, None] * second
    residual = np.sum(np.abs(prediction - target[None, :]) ** 2, axis=1)
    residual[~stable] = np.inf
    return residual / max(float(np.sum(np.abs(target) ** 2)), 1e-15), c1


def _scene_dictionaries(
    dataset,
    scene: int,
    candidate_config: dict,
    spec: PatchSpec,
    grid_spacing_m: float,
    *,
    infer_receiver: bool,
):
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(dataset.world_bits)}
    base_world = lookup[(0,) * dataset.bit_count]
    source_map = dataset.maps[scene, base_world]
    map_config = candidate_config["map"]
    radio = candidate_config["radio"]
    candidates = None
    if infer_receiver:
        candidates = _candidate_positions(
            source_map,
            origin_xy_m=np.asarray(map_config["origin_xy_m"]),
            resolution_m=float(map_config["resolution_m"]),
            grid_spacing_m=float(grid_spacing_m),
            minimum_bs_distance_m=float(map_config["receiver_minimum_bs_distance_m"]),
            transmitter_z_m=float(radio["transmitter_z_m"]),
            receiver_z_m=float(radio["receiver_z_m"]),
            minimum_vertical_clearance_m=float(
                map_config["receiver_minimum_direct_path_vertical_clearance_m"]
            ),
        )
    dictionaries = {}
    for bit in range(dataset.bit_count):
        target_bits = np.zeros(dataset.bit_count, dtype=np.int64)
        target_bits[bit] = 1
        target_world = lookup[tuple(target_bits.tolist())]
        rectangle = action_rectangle_xy(
            source_map,
            dataset.maps[scene, target_world],
            origin_xy_m=np.asarray(map_config["origin_xy_m"]),
            resolution_m=float(map_config["resolution_m"]),
        )
        dictionaries[bit] = {
            "rectangle": rectangle,
            "dictionary": (
                _candidate_dictionary(candidates, rectangle, candidate_config, spec)
                if candidates is not None
                else None
            ),
        }
    return dictionaries


def _oracle_receiver_dictionary(
    receiver_xy: np.ndarray,
    rectangle: tuple[float, float, float, float],
    candidate_config: dict,
    spec: PatchSpec,
):
    receiver = np.asarray(receiver_xy, dtype=np.float64)
    radio = candidate_config["radio"]
    intervention = candidate_config["interventions"]
    map_config = candidate_config["map"]
    path = reflection_path(
        rectangle,
        receiver,
        primitive_height_m=float(intervention["height_m"]),
        transmitter_z_m=float(radio["transmitter_z_m"]),
        receiver_z_m=float(radio["receiver_z_m"]),
        carrier_frequency_hz=float(radio["carrier_frequency_hz"]),
        subcarrier_spacing_hz=float(radio["subcarrier_spacing_hz"]),
        antennas=spec.antennas,
        subcarriers=spec.subcarriers,
        minimum_incidence_cosine=float(map_config["minimum_reflection_incidence_cosine"]),
    )
    if path is None:
        return None
    direct = direct_path_basis(
        receiver,
        transmitter_z_m=float(radio["transmitter_z_m"]),
        receiver_z_m=float(radio["receiver_z_m"]),
        antennas=spec.antennas,
        subcarriers=spec.subcarriers,
    )
    return receiver[None, :], direct.reshape(1, -1), path.basis.reshape(1, -1), path


def build_physical_response_rows(
    dataset,
    scenes: Iterable[int],
    formal_config: dict,
    candidate_config: dict,
    patch_scale: np.ndarray,
    route_lookup,
    *,
    position_stride: int,
    grid_spacing_m: float,
    decomposition_ridge: float,
    include_targets: bool,
    receiver_mode: str = "inferred",
) -> list[dict]:
    spec = PatchSpec.from_metadata(dataset.metadata)
    if spec.patch_count != 16:
        raise PhysicalResponseError(
            "physical response v2 requires the frozen 16-query patch grid"
        )
    if receiver_mode not in {"inferred", "oracle"}:
        raise ValueError("physical response receiver_mode must be inferred or oracle")
    if include_targets and receiver_mode != "inferred":
        raise PhysicalResponseError("physical response fit targets require inferred receivers")
    infer_receiver = receiver_mode == "inferred"
    masks = {
        int(entry.query): visible_complex_mask(entry.mask, spec)
        for entry in frozen_mask_query_bank(spec, int(formal_config["model"]["mask_bank_seed"]))
        if entry.mode == "random_75"
    }
    if set(masks) != set(range(spec.patch_count)):
        raise PhysicalResponseError("physical response requires every frozen random query")
    positions = np.arange(0, dataset.position_count, int(position_stride), dtype=np.int64)
    radio = candidate_config["radio"]
    intervention = candidate_config["interventions"]
    map_config = candidate_config["map"]
    rows = []
    for scene_value in scenes:
        scene = int(scene_value)
        dictionaries = _scene_dictionaries(
            dataset,
            scene,
            candidate_config,
            spec,
            float(grid_spacing_m),
            infer_receiver=infer_receiver,
        )
        for edge in dataset.directed_edges(scene):
            detail = dictionaries[int(edge.bit_index)]
            if infer_receiver:
                candidate_xy, direct, reflected = detail["dictionary"]
            material = material_direction(
                dataset.maps[scene, edge.source_world],
                dataset.maps[scene, edge.target_world],
            )
            for position_value in positions:
                position = int(position_value)
                source_csi = complex_csi(
                    dataset.csi[scene, edge.source_world, position]
                ).reshape(spec.antennas, spec.subcarriers)
                normalized_target = None
                raw_target = None
                if include_targets:
                    raw_target = complex_csi(
                        dataset.csi[scene, edge.target_world, position]
                        - dataset.csi[scene, edge.source_world, position]
                    ).reshape(spec.antennas, spec.subcarriers)
                    normalized_target = normalized_patch_delta(raw_target, patch_scale, spec)
                oracle_dictionary = None
                if not infer_receiver:
                    oracle_dictionary = _oracle_receiver_dictionary(
                        dataset.positions[scene, position],
                        detail["rectangle"],
                        candidate_config,
                        spec,
                    )
                for query, visible in masks.items():
                    if not infer_receiver and oracle_dictionary is None:
                        path = None
                        receiver_xy = np.asarray(
                            dataset.positions[scene, position], dtype=np.float64
                        )
                        path_features = np.zeros(7, dtype=np.float64)
                        design = np.zeros(15, dtype=np.complex128)
                        basis = np.zeros(
                            (spec.antennas, spec.subcarriers), dtype=np.complex128
                        )
                        visible_fit_nmse = 1.0e12
                    else:
                        if infer_receiver:
                            active_candidate_xy = candidate_xy
                            active_direct = direct
                            active_reflected = reflected
                            path = None
                        else:
                            assert oracle_dictionary is not None
                            (
                                active_candidate_xy,
                                active_direct,
                                active_reflected,
                                path,
                            ) = oracle_dictionary
                        residual, reflected_coefficient = _pair_fit(
                            source_csi,
                            visible,
                            active_direct,
                            active_reflected,
                            ridge=float(decomposition_ridge),
                        )
                        best = int(np.argmin(residual))
                        receiver_xy = active_candidate_xy[best]
                        if path is None:
                            path = reflection_path(
                                detail["rectangle"],
                                receiver_xy,
                                primitive_height_m=float(intervention["height_m"]),
                                transmitter_z_m=float(radio["transmitter_z_m"]),
                                receiver_z_m=float(radio["receiver_z_m"]),
                                carrier_frequency_hz=float(radio["carrier_frequency_hz"]),
                                subcarrier_spacing_hz=float(radio["subcarrier_spacing_hz"]),
                                antennas=spec.antennas,
                                subcarriers=spec.subcarriers,
                                minimum_incidence_cosine=float(
                                    map_config["minimum_reflection_incidence_cosine"]
                                ),
                            )
                        if path is None:
                            raise AssertionError("selected receiver candidate lost its path")
                        path_features = path.features
                        feature = np.concatenate(
                            (np.asarray((1.0,)), path_features, path_features**2)
                        )
                        design = reflected_coefficient[best] * feature
                        basis = path.basis
                        visible_fit_nmse = float(residual[best])
                    row = {
                        "key": (
                            scene,
                            int(edge.source_world),
                            int(edge.target_world),
                            int(edge.bit_index),
                            position,
                            int(query),
                        ),
                        "scene": scene,
                        "bank_id": str(dataset.bank_ids[scene]),
                        "source_world": int(edge.source_world),
                        "target_world": int(edge.target_world),
                        "bit": int(edge.bit_index),
                        "position": position,
                        "query": int(query),
                        "material": material,
                        "design": design,
                        "basis": basis,
                        "path_features": path_features,
                        "inferred_xy": receiver_xy,
                        "receiver_position_source": receiver_mode,
                        "path_available": path is not None,
                        "route": int(
                            route_lookup[
                                (
                                    scene,
                                    int(edge.source_world),
                                    int(edge.target_world),
                                    position,
                                    int(query),
                                )
                            ]
                        ),
                        "visible_fit_nmse": visible_fit_nmse,
                    }
                    if include_targets:
                        assert raw_target is not None and normalized_target is not None
                        row["coefficient_target"] = complex(
                            np.vdot(path.basis, raw_target)
                            / max(float(np.vdot(path.basis, path.basis).real), 1e-15)
                        )
                        row["target"] = normalized_target[int(query)]
                    rows.append(row)
    return rows


def _fit_coefficient_calibration(rows: list[dict], ridge: float):
    output = {}
    for material in sorted({tuple(row["material"]) for row in rows}):
        selected = [row for row in rows if tuple(row["material"]) == material]
        design = np.stack([row["design"] for row in selected])
        target = np.asarray([row["coefficient_target"] for row in selected])
        gram = design.conj().T @ design
        output[material] = np.linalg.solve(
            gram + float(ridge) * np.eye(gram.shape[0]),
            design.conj().T @ target,
        )
    return output


def _raw_predictions(rows: list[dict], calibration, patch_scale, spec: PatchSpec):
    output = []
    for row in rows:
        coefficient = row["design"] @ calibration[tuple(row["material"])]
        normalized = normalized_patch_delta(coefficient * row["basis"], patch_scale, spec)
        output.append(normalized[int(row["query"])])
    return np.asarray(output)


def _fit_shared_response_scale(
    rows: list[dict], gated_prediction: np.ndarray
) -> tuple[float, dict]:
    prediction = np.asarray(gated_prediction, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[0] != len(rows):
        raise ValueError("physical response scale prediction shape is invalid")
    active_indices = np.asarray(
        [index for index, row in enumerate(rows) if int(row["route"]) == 2],
        dtype=np.int64,
    )
    if active_indices.size == 0:
        raise PhysicalResponseError("physical response scale fit requires active rows")
    if any("target" not in rows[index] for index in active_indices):
        raise PhysicalResponseError("physical response scale fit requires train targets")
    target = np.stack(
        [np.asarray(rows[index]["target"], dtype=np.float64) for index in active_indices]
    )
    active_prediction = prediction[active_indices]
    if target.shape != active_prediction.shape:
        raise ValueError("physical response scale target shape is invalid")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(active_prediction)):
        raise PhysicalResponseError("physical response scale fit received non-finite values")
    denominator = float(np.sum(active_prediction**2, dtype=np.float64))
    numerator = float(np.sum(target * active_prediction, dtype=np.float64))
    if denominator <= 1e-15 or not np.isfinite(denominator):
        raise PhysicalResponseError("physical response scale fit has zero prediction energy")
    response_scale = numerator / denominator
    if not np.isfinite(response_scale) or response_scale <= 0.0:
        raise PhysicalResponseError("physical response scale fit is not positive and finite")
    before = float(np.sum((target - active_prediction) ** 2, dtype=np.float64))
    after = float(
        np.sum((target - response_scale * active_prediction) ** 2, dtype=np.float64)
    )
    if after > before + 1e-10 * max(before, 1.0):
        raise AssertionError("least-squares response scale increased train squared error")
    return float(response_scale), {
        "strategy": "source_encoder_train_active_gated_scalar_least_squares",
        "fit_active_rows": int(active_indices.size),
        "numerator": numerator,
        "denominator": denominator,
        "response_scale": float(response_scale),
        "active_squared_error_before": before,
        "active_squared_error_after": after,
        "selection_targets_read": False,
        "shared_by_inferred_oracle_and_action_swap": True,
    }


def _gate_group(row: dict) -> str:
    source_material, target_material = row["material"]
    return f"q{int(row['query'])}:m{int(source_material)}-{int(target_material)}"


def _gate_features(rows: list[dict], prediction: np.ndarray) -> np.ndarray:
    output = []
    for row, response in zip(rows, prediction, strict=True):
        design = np.asarray(row["design"], dtype=np.complex128)
        query = np.zeros(16, dtype=np.float64)
        query[int(row["query"])] = 1.0
        source_material, target_material = row["material"]
        response_values = np.asarray(response, dtype=np.float64)
        output.append(
            np.concatenate(
                (
                    design.real,
                    design.imag,
                    np.log1p(np.abs(design)),
                    np.asarray(row["path_features"], dtype=np.float64),
                    np.asarray(row["inferred_xy"], dtype=np.float64) / 128.0,
                    np.asarray(
                        (
                            np.log1p(float(row["visible_fit_nmse"])),
                            np.sqrt(np.mean(response_values**2)),
                        )
                    ),
                    response_values,
                    np.asarray(
                        (
                            source_material / 4.0,
                            target_material / 4.0,
                            float(source_material == 1),
                            float(source_material == 4),
                        )
                    ),
                    query,
                )
            )
        )
    return np.asarray(output, dtype=np.float32)


def _gate_index_pools(labels: np.ndarray, bank_ids: np.ndarray):
    banks = sorted(set(bank_ids.tolist()))
    pools = {
        label: {
            str(bank): np.flatnonzero((labels == label) & (bank_ids == bank))
            for bank in banks
        }
        for label in (0.0, 1.0)
    }
    missing = {
        int(label): [bank for bank, indices in by_bank.items() if not indices.size]
        for label, by_bank in pools.items()
        if any(not indices.size for indices in by_bank.values())
    }
    if missing:
        raise PhysicalResponseError(f"physical gate fit banks lack route rows: {missing}")
    return pools


def _bank_balanced_indices(pools, batch_size: int, generator: np.random.Generator):
    class_sizes = (batch_size // 2, batch_size - batch_size // 2)
    output = []
    for label, size in zip((0.0, 1.0), class_sizes, strict=True):
        by_bank = pools[label]
        assignments = np.resize(np.asarray(sorted(by_bank), dtype=object), size)
        generator.shuffle(assignments)
        for bank in sorted(by_bank):
            count = int(np.sum(assignments == bank))
            if count:
                output.extend(
                    generator.choice(by_bank[bank], size=count, replace=True).tolist()
                )
    indices = np.asarray(output, dtype=np.int64)
    generator.shuffle(indices)
    return indices


def calibrate_gate_thresholds(
    probability: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    bank_ids: np.ndarray,
    *,
    maximum_false_positive_rate: float,
) -> tuple[dict[str, float], dict]:
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float32)
    group_ids = np.asarray(group_ids)
    bank_ids = np.asarray(bank_ids)
    if not (probability.shape == labels.shape == group_ids.shape == bank_ids.shape):
        raise ValueError("physical gate calibration arrays must have identical shapes")
    null = labels == 0.0
    if not np.any(null):
        raise PhysicalResponseError("physical gate calibration requires null rows")
    quantile = 1.0 - float(maximum_false_positive_rate)
    thresholds = {}
    boundaries_by_group = {}
    banks = sorted(set(bank_ids.tolist()))
    groups = sorted(set(group_ids.tolist()))
    for group in groups:
        boundaries = {}
        for bank in banks:
            selected = probability[null & (group_ids == group) & (bank_ids == bank)]
            if not selected.size:
                raise PhysicalResponseError(
                    f"physical gate bank/group {bank}/{group} has no null rows"
                )
            boundaries[str(bank)] = float(
                np.quantile(selected, quantile, method="higher")
            )
        boundary = max(boundaries.values())
        thresholds[str(group)] = float(np.nextafter(boundary, np.inf))
        boundaries_by_group[str(group)] = boundaries
    false_positive = {
        f"{bank}/{group}": float(
            np.mean(
                probability[null & (group_ids == group) & (bank_ids == bank)]
                >= thresholds[str(group)]
            )
        )
        for bank in banks
        for group in groups
    }
    return thresholds, {
        "strategy": "maximum_fit_bank_query_material_conditional_null_quantile",
        "maximum_false_positive_rate": float(maximum_false_positive_rate),
        "thresholds_by_group": thresholds,
        "bank_group_quantile_boundaries": boundaries_by_group,
        "false_positive_rate_by_bank_group": false_positive,
        "maximum_fit_bank_group_false_positive_rate": max(false_positive.values()),
    }


def _fit_gate_member(
    normalized: np.ndarray,
    labels: np.ndarray,
    bank_ids: np.ndarray,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    device: torch.device,
):
    torch.manual_seed(int(seed))
    model = PhysicalNullGate(normalized.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(int(seed))
    pools = _gate_index_pools(labels, bank_ids)
    values = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    final_loss = None
    model.train()
    for _step in range(int(steps)):
        indices = _bank_balanced_indices(pools, int(batch_size), generator)
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(values[index]), target[index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(values)).cpu().numpy()
    return model, probability, final_loss


def fit_physical_response_model(
    rows: list[dict],
    patch_scale: np.ndarray,
    spec: PatchSpec,
    config: dict,
    *,
    seed: int,
    device: str | torch.device,
) -> PhysicalResponseModel:
    calibration = _fit_coefficient_calibration(rows, float(config["calibration_ridge"]))
    raw = _raw_predictions(rows, calibration, patch_scale, spec)
    routes = np.asarray([row["route"] for row in rows])
    selected = (routes == 0) | (routes == 2)
    labels = (routes[selected] == 2).astype(np.float32)
    bank_ids = np.asarray([str(row["bank_id"]) for row in rows])[selected]
    group_ids = np.asarray([_gate_group(row) for row in rows])[selected]
    features = _gate_features(rows, raw)[selected]
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    execution_device = torch.device(device)
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise PhysicalResponseError("physical gate requested unavailable CUDA")
    gates = []
    probabilities = []
    members = []
    for member in range(int(config["gate_ensemble_size"])):
        member_seed = int(seed) + member * int(config["gate_seed_stride"])
        gate, probability, loss = _fit_gate_member(
            normalized,
            labels,
            bank_ids,
            seed=member_seed,
            steps=int(config["gate_steps_per_member"]),
            batch_size=int(config["gate_batch_size"]),
            device=execution_device,
        )
        gates.append(gate)
        probabilities.append(probability)
        members.append({"member": member, "seed": member_seed, "final_bce": loss})
    ensemble_probability = np.mean(np.stack(probabilities), axis=0)
    thresholds, threshold_audit = calibrate_gate_thresholds(
        ensemble_probability,
        labels,
        group_ids,
        bank_ids,
        maximum_false_positive_rate=float(
            config["maximum_fit_bank_group_false_positive_rate"]
        ),
    )
    row_threshold = np.asarray([thresholds[value] for value in group_ids])
    provisional_model = PhysicalResponseModel(
        coefficient_calibration=calibration,
        gates=gates,
        feature_mean=mean,
        feature_scale=scale,
        thresholds=thresholds,
        response_scale=1.0,
        metadata={},
    )
    gated_fit_prediction, _fit_probability = predict_physical_response(
        rows, provisional_model, patch_scale, spec
    )
    response_scale, response_scale_audit = _fit_shared_response_scale(
        rows, gated_fit_prediction
    )
    metadata = {
        "contract": PHYSICAL_RESPONSE_CONTRACT,
        "fit_rows": int(labels.size),
        "fit_null_rows": int(np.sum(labels == 0.0)),
        "fit_active_rows": int(np.sum(labels == 1.0)),
        "feature_count": int(mean.size),
        "member_audits": members,
        "threshold_calibration": threshold_audit,
        "fit_null_false_positive_rate": float(
            np.mean(
                ensemble_probability[labels == 0.0]
                >= row_threshold[labels == 0.0]
            )
        ),
        "fit_active_recall": float(
            np.mean(
                ensemble_probability[labels == 1.0]
                >= row_threshold[labels == 1.0]
            )
        ),
        "response_scale_calibration": response_scale_audit,
    }
    return PhysicalResponseModel(
        coefficient_calibration=calibration,
        gates=gates,
        feature_mean=mean,
        feature_scale=scale,
        thresholds=thresholds,
        response_scale=response_scale,
        metadata=metadata,
    )


def predict_physical_response(
    rows: list[dict],
    model: PhysicalResponseModel,
    patch_scale: np.ndarray,
    spec: PatchSpec,
) -> tuple[np.ndarray, np.ndarray]:
    raw = _raw_predictions(rows, model.coefficient_calibration, patch_scale, spec)
    normalized = (_gate_features(rows, raw) - model.feature_mean) / model.feature_scale
    probabilities = []
    for gate in model.gates:
        device = next(gate.parameters()).device
        with torch.no_grad():
            probabilities.append(
                torch.sigmoid(
                    gate(torch.as_tensor(normalized, dtype=torch.float32, device=device))
                ).cpu().numpy()
            )
    probability = np.mean(np.stack(probabilities), axis=0)
    threshold = np.asarray([model.thresholds[_gate_group(row)] for row in rows])
    prediction = raw.copy()
    prediction[probability < threshold] = 0.0
    if not np.isfinite(model.response_scale) or model.response_scale <= 0.0:
        raise PhysicalResponseError("physical response scale is not positive and finite")
    prediction *= float(model.response_scale)
    return prediction, probability


def physical_action_swap_predictions(rows, prediction, dataset, wrong_action_plan):
    index = {row["key"]: offset for offset, row in enumerate(rows)}
    target_lookup = {
        tuple(int(value) for value in bits): world
        for world, bits in enumerate(dataset.world_bits)
    }
    output = np.zeros_like(prediction)
    statuses = []
    for offset, row in enumerate(rows):
        _, status, wrong_world = wrong_action_plan["selections"][
            (row["scene"], row["source_world"], row["bit"], row["position"])
        ]
        statuses.append(status)
        if wrong_world is None:
            continue
        changed = np.flatnonzero(
            dataset.world_bits[row["source_world"]]
            != dataset.world_bits[int(wrong_world)]
        )
        if changed.size != 1:
            raise PhysicalResponseError("wrong action does not toggle exactly one bit")
        target_bits = dataset.world_bits[row["source_world"]].copy()
        target_bits[int(changed[0])] = 1 - target_bits[int(changed[0])]
        target_world = target_lookup[tuple(int(value) for value in target_bits)]
        swap_key = (
            row["scene"],
            row["source_world"],
            target_world,
            int(changed[0]),
            row["position"],
            row["query"],
        )
        output[offset] = prediction[index[swap_key]]
    return output, np.asarray(statuses)


def fit_and_predict_qualification_response(
    dataset,
    train_scenes,
    selection_scenes,
    train_routed,
    selection_routed,
    formal_config: dict,
    train_wrong_actions,
    selection_wrong_actions,
    *,
    seed: int,
    device: str | torch.device,
    physical_config: dict | None = None,
):
    config = load_physical_response_config() if physical_config is None else physical_config
    candidate_config, candidate_path = load_bound_candidate_config(dataset)
    spec = PatchSpec.from_metadata(dataset.metadata)
    from .formal_protocol import patchify_csi

    patch_scale = patchify_csi(train_routed.normalization.channel_scale, spec)
    fit_rows = build_physical_response_rows(
        dataset,
        train_scenes,
        formal_config,
        candidate_config,
        patch_scale,
        train_routed.response_route,
        position_stride=int(config["fit_position_stride"]),
        grid_spacing_m=float(config["grid_spacing_m"]),
        decomposition_ridge=float(config["decomposition_ridge"]),
        include_targets=True,
    )
    print(
        json.dumps(
            {
                "stage": "physical_response_fit_rows_ready",
                "fit_banks": int(len(train_scenes)),
                "fit_rows": int(len(fit_rows)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model = fit_physical_response_model(
        fit_rows,
        patch_scale,
        spec,
        config,
        seed=int(seed) + int(config["gate_seed_offset"]),
        device=device,
    )
    print(
        json.dumps(
            {
                "stage": "physical_response_gate_fit_complete",
                "fit_active_rows": int(model.metadata["fit_active_rows"]),
                "fit_null_rows": int(model.metadata["fit_null_rows"]),
                "gate_members": int(len(model.gates)),
                "response_scale": float(model.response_scale),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    fit_row_count = len(fit_rows)
    del fit_rows
    gc.collect()
    key_chunks = []
    prediction_chunks = []
    oracle_chunks = []
    swap_chunks = []
    status_chunks = []
    probability_quantiles = []
    for scene_value in selection_scenes:
        scene = int(scene_value)
        rows = build_physical_response_rows(
            dataset,
            (scene,),
            formal_config,
            candidate_config,
            patch_scale,
            selection_routed.response_route,
            position_stride=int(config["selection_position_stride"]),
            grid_spacing_m=float(config["grid_spacing_m"]),
            decomposition_ridge=float(config["decomposition_ridge"]),
            include_targets=False,
        )
        prediction, probability = predict_physical_response(rows, model, patch_scale, spec)
        swap, statuses = physical_action_swap_predictions(
            rows, prediction, dataset, selection_wrong_actions
        )
        key_chunks.extend(row["key"] for row in rows)
        scene_keys = [row["key"] for row in rows]
        prediction_chunks.append(prediction)
        swap_chunks.append(swap)
        status_chunks.append(statuses)
        probability_quantiles.append(
            {
                "scene": scene,
                "bank_id": str(dataset.bank_ids[scene]),
                "quantiles": {
                    str(value): float(np.quantile(probability, value))
                    for value in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
                },
            }
        )
        del rows, prediction, swap, probability, statuses
        gc.collect()
        oracle_rows = build_physical_response_rows(
            dataset,
            (scene,),
            formal_config,
            candidate_config,
            patch_scale,
            selection_routed.response_route,
            position_stride=int(config["selection_position_stride"]),
            grid_spacing_m=float(config["grid_spacing_m"]),
            decomposition_ridge=float(config["decomposition_ridge"]),
            include_targets=False,
            receiver_mode="oracle",
        )
        if [row["key"] for row in oracle_rows] != scene_keys:
            raise PhysicalResponseError(
                "oracle-position rows do not align with inferred-position rows"
            )
        oracle_prediction, _oracle_probability = predict_physical_response(
            oracle_rows, model, patch_scale, spec
        )
        oracle_chunks.append(oracle_prediction)
        print(
            json.dumps(
                {
                    "stage": "physical_response_selection_bank_complete",
                    "scene": scene,
                    "bank_id": str(dataset.bank_ids[scene]),
                    "rows": int(len(scene_keys)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del oracle_rows, oracle_prediction, _oracle_probability, scene_keys
        gc.collect()
    audit = {
        **model.metadata,
        "fit_row_count_including_gray": fit_row_count,
        "selection_row_count": len(key_chunks),
        "physical_config": config,
        "physical_config_sha256": sha256_file(
            Path(__file__).resolve().parent / "configs" / "physical_response_v1.json"
        ),
        "candidate_config": str(candidate_path),
        "candidate_config_sha256": sha256_file(candidate_path),
        "selection_probability_quantiles": probability_quantiles,
        "oracle_position_control": {
            "receiver_position_source": "frozen_ground_truth_receiver_xy",
            "target_csi_read": False,
            "route_label_read_by_predictor": False,
            "shared_fit_model_and_gate": True,
        },
        "train_wrong_action_plan_bound": train_wrong_actions is not None,
    }
    return (
        model,
        key_chunks,
        np.concatenate(prediction_chunks, axis=0),
        np.concatenate(swap_chunks, axis=0),
        np.concatenate(oracle_chunks, axis=0),
        np.concatenate(status_chunks, axis=0),
        audit,
    )


def save_physical_response_checkpoint(
    path: str | Path,
    model: PhysicalResponseModel,
    bindings: dict,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite physical response checkpoint: {target}")
    torch.save(
        {
            "schema_version": PHYSICAL_RESPONSE_CHECKPOINT_SCHEMA,
            "contract": PHYSICAL_RESPONSE_CONTRACT,
            "coefficient_calibration": model.coefficient_calibration,
            "state_dicts": [
                {name: value.detach().cpu() for name, value in gate.state_dict().items()}
                for gate in model.gates
            ],
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "thresholds": model.thresholds,
            "response_scale": float(model.response_scale),
            "metadata": model.metadata,
            "bindings": dict(bindings),
        },
        target,
    )
    return target


def load_physical_response_checkpoint(path: str | Path) -> tuple[PhysicalResponseModel, dict]:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PHYSICAL_RESPONSE_CHECKPOINT_SCHEMA
        or payload.get("contract") != PHYSICAL_RESPONSE_CONTRACT
    ):
        raise PhysicalResponseError("physical response checkpoint schema/contract mismatch")
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    scale = np.asarray(payload["feature_scale"], dtype=np.float32)
    gates = []
    for state in payload["state_dicts"]:
        gate = PhysicalNullGate(mean.size)
        gate.load_state_dict(state, strict=True)
        gate.eval()
        gates.append(gate)
    model = PhysicalResponseModel(
        coefficient_calibration=payload["coefficient_calibration"],
        gates=gates,
        feature_mean=mean,
        feature_scale=scale,
        thresholds={str(key): float(value) for key, value in payload["thresholds"].items()},
        response_scale=float(payload["response_scale"]),
        metadata=dict(payload["metadata"]),
    )
    if not np.isfinite(model.response_scale) or model.response_scale <= 0.0:
        raise PhysicalResponseError("physical response checkpoint scale is invalid")
    return model, dict(payload["bindings"])
