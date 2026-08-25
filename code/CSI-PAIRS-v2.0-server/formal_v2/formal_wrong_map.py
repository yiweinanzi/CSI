from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path

import numpy as np

from .formal_baselines import RidgeRegressor
from .formal_dataset import FormalDataset
from .formal_evidence import bind_rows, evidence_context, require_manifested_formal_qualification
from .formal_features import map_pair_features, response_features
from .formal_io import artifact_manifest, write_csv, write_json
from .formal_routing import fit_route_normalization, route_dataset
from .formal_teacher import load_teacher_bundle


CONDITIONS = (
    "correct",
    "paired_active_alternative",
    "paired_null_alternative",
    "wrong_city",
    "geometry_destroyed",
    "empty",
)


def run_formal_wrong_map(
    config: dict,
    dataset: FormalDataset,
    output_root: str | Path,
    qualification_gate: dict,
) -> dict:
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        ("source_encoder_train", "source_method_selection", "target"),
    )
    qualification_gate = require_manifested_formal_qualification(
        qualification_gate,
        config,
        dataset,
        allow_nonscientific_fixture=True,
    )
    output_dir = Path(output_root) / "wrong_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_scenes = dataset.indices_for_role("source_encoder_train")
    evaluation_scenes = np.concatenate(
        (dataset.indices_for_role("source_method_selection"), dataset.indices_for_role("target"))
    )
    train_x = []
    train_y = []
    for scene_value in train_scenes:
        scene = int(scene_value)
        for world in range(dataset.world_count):
            map_features = _localization_map_features(dataset.maps[scene, world])
            csi = np.asarray(dataset.csi[scene, world], dtype=np.float64).reshape(
                dataset.position_count, -1
            )
            train_x.append(
                np.concatenate(
                    (
                        csi,
                        np.broadcast_to(
                            map_features,
                            (dataset.position_count, map_features.size),
                        ),
                    ),
                    axis=1,
                )
            )
            train_y.append(dataset.positions[scene])
    model = RidgeRegressor(float(config["localization"]["ridge"])).fit(
        np.vstack(train_x), np.vstack(train_y)
    )
    model.save(
        output_dir / "checkpoints" / "controlled_relative_map_ridge.npz",
        {
            "model_name": "controlled-relative-map-ridge-diagnostic",
            "implementation_status": "local-controlled-diagnostic-not-external-baseline",
            "scientific_use": "FORBIDDEN_FOR_DOMAIN_LEVEL_CLAIMS",
        },
    )
    teacher = load_teacher_bundle(qualification_gate["teacher_checkpoint"], config)
    route_normalization = fit_route_normalization(dataset, teacher)
    routed = route_dataset(
        dataset,
        teacher,
        config,
        evaluation_scenes,
        normalization=route_normalization,
    )
    rows = _evaluation_rows(
        config, dataset, evaluation_scenes, routed, model
    )
    if not rows:
        raise ValueError("no evaluation units have both active and null paired alternatives")
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_DOMAIN_EVIDENCE"
    )
    write_csv(output_dir / "per_sample_results.csv", bind_rows(rows, evidence))
    summary = _summary(rows)
    write_csv(output_dir / "summary.csv", bind_rows(summary, evidence))
    status = {
        "schema_version": "csi-pairs-formal-wrong-map-status-v2.1-v6",
        "status": "DIAGNOSTIC_COMPLETE_NOT_DOMAIN_EVIDENCE",
        "scientific_use": "FORBIDDEN_FOR_EXTERNAL_MODEL_OR_EFFICACY_CLAIMS",
        "fixture": dataset.is_fixture,
        **evidence,
        "model_name": "controlled-relative-map-ridge-diagnostic",
        "teacher_checkpoint": qualification_gate["teacher_checkpoint"],
        "teacher_checkpoint_sha256": qualification_gate["teacher_checkpoint_sha256"],
        "conditions": list(CONDITIONS),
        "paired_units": len({row["unit_id"] for row in rows}),
        "independent_banks": len({row["bank_id"] for row in rows}),
        "required_upgrade": (
            "Run the same exported condition contract on at least two accurately named, licensed, "
            "map-conditioned external models before making a domain-level wrong-map claim."
        ),
    }
    write_json(output_dir / "status.json", status)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return status


def _evaluation_rows(
    config: dict,
    dataset: FormalDataset,
    scenes: np.ndarray,
    routed,
    model: RidgeRegressor,
) -> list[dict]:
    scene_tuple = tuple(int(value) for value in scenes)
    canonical_units = {
        (scene, world)
        for scene in scene_tuple
        for world in range(dataset.world_count)
    }
    for scene in scene_tuple:
        wrong_scene = _wrong_city_scene(dataset, scene)
        canonical_units.add(
            (wrong_scene, int(dataset.natural_world_index[wrong_scene]))
        )
    workers = _worker_count(len(scene_tuple))
    ordered_units = sorted(canonical_units)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        feature_values = executor.map(
            _localization_map_features,
            (dataset.maps[scene, world] for scene, world in ordered_units),
        )
        map_feature_cache = dict(zip(ordered_units, feature_values, strict=True))
        scene_rows = executor.map(
            lambda scene: _evaluation_rows_for_scene(
                dataset, scene, routed, model, map_feature_cache
            ),
            scene_tuple,
        )
        return [row for rows in scene_rows for row in rows]


def _evaluation_rows_for_scene(dataset, scene, routed, model, map_feature_cache):
    rows: list[dict] = []
    allowed_positions = (
        np.flatnonzero(dataset.position_roles[scene] == "query")
        if dataset.scene_roles[scene] == "target"
        else np.arange(dataset.position_count)
    )
    wrong_scene = _wrong_city_scene(dataset, scene)
    wrong_world = int(dataset.natural_world_index[wrong_scene])
    empty_features = _localization_map_features(
        np.zeros_like(dataset.maps[scene, 0])
    )
    scene_edges = tuple(dataset.directed_edges(scene))
    for source_world in range(dataset.world_count):
        candidates = [
            edge for edge in scene_edges if edge.source_world == source_world
        ]
        for position in allowed_positions:
            active = []
            null = []
            for edge in candidates:
                route = routed.alignment_route[
                    (scene, source_world, edge.target_world, int(position))
                ]
                if route == 2:
                    active.append(edge.target_world)
                elif route == 0:
                    null.append(edge.target_world)
            if not active or not null:
                continue
            correct_map = dataset.maps[scene, source_world]
            supplied_features = {
                "correct": map_feature_cache[(scene, source_world)],
                "paired_active_alternative": map_feature_cache[(scene, active[0])],
                "paired_null_alternative": map_feature_cache[(scene, null[0])],
                "wrong_city": map_feature_cache[(wrong_scene, wrong_world)],
                "geometry_destroyed": _localization_map_features(
                    _geometry_destroyed(
                        correct_map,
                        f"{dataset.bank_ids[scene]}:{source_world}:{position}",
                    )
                ),
                "empty": empty_features,
            }
            csi = dataset.csi[scene, source_world, position]
            unit_id = f"{dataset.bank_ids[scene]}:{source_world}:{position}"
            for condition in CONDITIONS:
                prediction = model.predict(
                    _assemble_localization_features(
                        csi, supplied_features[condition]
                    )[None, :]
                )[0]
                error = float(
                    np.linalg.norm(prediction - dataset.positions[scene, position])
                )
                rows.append(
                    {
                        "unit_id": unit_id,
                        "scene_id": str(dataset.scene_ids[scene]),
                        "city_id": str(dataset.city_ids[scene]),
                        "bank_id": str(dataset.bank_ids[scene]),
                        "source_world": source_world,
                        "position_index": int(position),
                        "condition": condition,
                        "prediction_x": float(prediction[0]),
                        "prediction_y": float(prediction[1]),
                        "true_x": float(dataset.positions[scene, position, 0]),
                        "true_y": float(dataset.positions[scene, position, 1]),
                        "localization_error_m": error,
                    }
                )
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["city_id"], row["bank_id"], row["condition"]) for row in rows})
    correct_lookup = {
        row["unit_id"]: row["localization_error_m"]
        for row in rows
        if row["condition"] == "correct"
    }
    for city, bank, condition in keys:
        selected = [
            row for row in rows
            if (row["city_id"], row["bank_id"], row["condition"]) == (city, bank, condition)
        ]
        values = np.asarray([row["localization_error_m"] for row in selected])
        paired_delta = np.asarray(
            [row["localization_error_m"] - correct_lookup[row["unit_id"]] for row in selected]
        )
        output.append(
            {
                "model": "controlled-relative-map-ridge-diagnostic",
                "city_id": city,
                "bank_id": bank,
                "condition": condition,
                "n": len(selected),
                "median_error_m": float(np.median(values)),
                "p90_error_m": float(np.percentile(values, 90)),
                "median_paired_change_vs_correct_m": float(np.median(paired_delta)),
            }
        )
    return output


def _localization_features(csi: np.ndarray, supplied_map: np.ndarray) -> np.ndarray:
    return response_features(
        np.asarray(csi, dtype=np.float64),
        np.asarray(supplied_map, dtype=np.float64),
        np.asarray(supplied_map, dtype=np.float64),
        include_action=False,
    )


def _localization_map_features(supplied_map: np.ndarray) -> np.ndarray:
    features, _zero_edit = map_pair_features(supplied_map, supplied_map)
    return features


def _assemble_localization_features(
    csi: np.ndarray, map_features: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(csi, dtype=np.float64).ravel(),
            np.asarray(map_features, dtype=np.float64).ravel(),
        )
    )


def _worker_count(item_count: int) -> int:
    try:
        configured = int(os.environ.get("OMP_NUM_THREADS", "1"))
    except ValueError:
        configured = 1
    return max(1, min(int(item_count), configured))


def _wrong_city_scene(dataset: FormalDataset, scene: int) -> int:
    for candidate in range(dataset.scene_count):
        if dataset.city_ids[candidate] != dataset.city_ids[scene]:
            return candidate
    raise ValueError("wrong-city stress test requires at least two cities")


def _geometry_destroyed(world_map: np.ndarray, token: str) -> np.ndarray:
    array = np.asarray(world_map, dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    if array.ndim == 2:
        return array.ravel()[rng.permutation(array.size)].reshape(array.shape)
    if array.ndim != 3:
        raise ValueError("geometry_destroyed expects [channel,row,column]")
    return np.stack(
        [channel.ravel()[rng.permutation(channel.size)].reshape(channel.shape) for channel in array]
    )
