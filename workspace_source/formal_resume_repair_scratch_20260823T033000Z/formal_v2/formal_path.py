from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .formal_dataset import FormalDataset
from .formal_evidence import bind_rows, evidence_context
from .formal_io import (
    artifact_manifest,
    read_strict_json,
    sha256_file,
    write_csv,
    write_json,
)
from .formal_protocol import PatchSpec, typed_signed_edit
from .formal_factorial import _canonical_bank_digest


def _path_provenance(config, dataset, power_coverage):
    engine = dataset.metadata.get("engine", {})
    engine_bound = bool(
        engine.get("deterministic") is True
        and str(engine.get("source_revision", "")).strip()
        and str(engine.get("license_id", "")).strip()
        and str(engine.get("config_sha256", "")).strip()
    )
    persistent = True
    for scene in range(dataset.path_ids.shape[0]):
        if any(np.all(dataset.primitive_surface_ids[scene, bit] < 0) for bit in range(dataset.bit_count)):
            persistent = False
            break
        for position in range(dataset.path_ids.shape[2]):
            registry = {}
            for world in range(dataset.path_ids.shape[1]):
                for index, value in enumerate(dataset.path_ids[scene, world, position]):
                    path_id = int(value)
                    if path_id < 0:
                        continue
                    surfaces = tuple(
                        sorted(
                            int(item)
                            for item in dataset.path_surface_ids[scene, world, position, index]
                            if int(item) >= 0
                        )
                    )
                    if path_id in registry and registry[path_id] != surfaces:
                        persistent = False
                        break
                    registry[path_id] = surfaces
                if not persistent:
                    break
            if not persistent:
                break
        if not persistent:
            break
    convergence = _power_coverage_convergence(config, dataset, power_coverage)
    payload = {
        "schema_version": "csi-pairs-v6-path-provenance-v1",
        "engine_name": str(engine.get("name", "")),
        "engine_version": str(engine.get("version", "")),
        "engine_source_revision": str(engine.get("source_revision", "")),
        "engine_license_id": str(engine.get("license_id", "")),
        "engine_config_sha256": str(engine.get("config_sha256", "")),
        "engine_bound_and_deterministic": engine_bound,
        "persistent_path_and_surface_ids": persistent,
        "power_coverage_convergence": convergence,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for values in (
        dataset.path_ids,
        dataset.path_surface_ids,
        dataset.path_power,
        dataset.primitive_surface_ids,
        dataset.noop_path_ids,
        dataset.noop_path_surface_ids,
        dataset.noop_path_power,
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    payload["registry_sha256"] = digest.hexdigest()
    payload["passed"] = bool(engine_bound and persistent and convergence["passed"])
    return payload


def _power_coverage_convergence(config, dataset, power_coverage):
    coverages = sorted(
        {
            float(power_coverage),
            float((power_coverage + 1.0) / 2.0),
            1.0,
        }
    )
    frozen_role = "source_method_selection"
    values = []
    for scene in range(len(dataset.scene_roles)):
        if str(dataset.scene_roles[scene]) != frozen_role:
            continue
        for edge in dataset.directed_edges(scene):
            if edge.source_world >= edge.target_world:
                continue
            changed = np.flatnonzero(
                dataset.world_bits[edge.source_world] != dataset.world_bits[edge.target_world]
            )
            if changed.size != 1:
                continue
            for position in range(dataset.position_count):
                curve = [
                    path_incidence(
                        dataset,
                        scene,
                        edge.source_world,
                        edge.target_world,
                        position,
                        int(changed[0]),
                        coverage,
                    )
                    for coverage in coverages
                ]
                values.append(curve)
    if not values:
        return {
            "passed": False,
            "coverages": coverages,
            "unit_count": 0,
            "frozen_role": frozen_role,
            "maximum_absolute_change": None,
            "tolerance": float(config["path"]["equivalence_margin"]),
        }
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(np.abs(array[:, 1:] - array[:, :-1])))
    tolerance = float(config["path"]["equivalence_margin"])
    return {
        "passed": bool(maximum <= tolerance),
        "coverages": coverages,
        "unit_count": len(values),
        "frozen_role": frozen_role,
        "maximum_absolute_change": maximum,
        "tolerance": tolerance,
    }


def path_incidence(
    dataset: FormalDataset,
    scene: int,
    source_world: int,
    target_world: int,
    position: int,
    bit_index: int,
    power_coverage: float = 1.0,
) -> float:
    surfaces = {
        int(value) for value in dataset.primitive_surface_ids[scene, bit_index] if int(value) >= 0
    }
    source_ids = dataset.path_ids[scene, source_world, position]
    target_ids = dataset.path_ids[scene, target_world, position]
    source_selected = _selected_paths(
        source_ids, dataset.path_power[scene, source_world, position], power_coverage
    )
    target_selected = _selected_paths(
        target_ids, dataset.path_power[scene, target_world, position], power_coverage
    )
    source_valid = {int(source_ids[index]) for index in source_selected}
    target_valid = {int(target_ids[index]) for index in target_selected}
    numerator = 0.0
    denominator = 0.0
    for world, ids, indices, other_ids in (
        (source_world, source_ids, source_selected, target_valid),
        (target_world, target_ids, target_selected, source_valid),
    ):
        for path_index in indices:
            path_id = int(ids[path_index])
            power = float(dataset.path_power[scene, world, position, path_index])
            denominator += power
            interactions = {
                int(value)
                for value in dataset.path_surface_ids[scene, world, position, path_index]
                if int(value) >= 0
            }
            if path_id not in other_ids or interactions.intersection(surfaces):
                numerator += power
    if denominator <= 0:
        raise RuntimeError("path incidence denominator is empty")
    value = numerator / denominator
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise RuntimeError("path incidence is outside [0,1]")
    return float(np.clip(value, 0.0, 1.0))


def localization_path_incidence(
    dataset: FormalDataset,
    scene: int,
    world: int,
    position: int,
    power_coverage: float = 1.0,
) -> float:
    values = [
        path_incidence(
            dataset,
            scene,
            edge.source_world,
            edge.target_world,
            position,
            edge.bit_index,
            power_coverage,
        )
        for edge in dataset.directed_edges(scene)
        if edge.source_world == world
    ]
    if not values:
        raise RuntimeError("localization world has no registered direct edits")
    return float(np.mean(values))


def _require_manifested_stage_artifact(path, evidence):
    artifact_path = Path(path)
    manifest_path = artifact_path.parent / "manifest.json"
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise RuntimeError(f"path input artifact is missing or symlinked: {artifact_path}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"path input has no regular stage manifest: {artifact_path}")
    manifest = read_strict_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "csi-pairs-formal-stage-manifest-v2.1-v6"
    ):
        raise RuntimeError(f"path input stage manifest schema mismatch: {manifest_path}")
    for key, value in evidence.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"path input stage manifest {key} mismatch: {manifest_path}")
    entries = manifest.get("files")
    relative = str(artifact_path.relative_to(artifact_path.parent))
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("path") == relative
    ] if isinstance(entries, list) else []
    if len(matches) != 1:
        raise RuntimeError(
            f"path input artifact is absent from or duplicated in its stage manifest: {artifact_path}"
        )
    entry = matches[0]
    for key, value in evidence.items():
        if key in entry and entry[key] != value:
            raise RuntimeError(
                f"path input artifact manifest entry {key} mismatch: {artifact_path}"
            )
    observed_sha256 = sha256_file(artifact_path)
    if entry.get("sha256") != observed_sha256:
        raise RuntimeError(f"path input artifact sha256 mismatch: {artifact_path}")
    if entry.get("bytes") != artifact_path.stat().st_size:
        raise RuntimeError(f"path input artifact byte count mismatch: {artifact_path}")
    return entry


def run_path_audit(
    config: dict,
    dataset: FormalDataset,
    factorial_root: str | Path,
    output_root: str | Path,
) -> dict:
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        ("source_method_selection", "source_final_unseen_bank", "target"),
    )
    root = Path(output_root)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    compatibility_path = root / "evaluation" / "compatibility_pair_effects.csv"
    response_path = root / "evaluation" / "response_pair_effects.csv"
    localization_path = Path(factorial_root) / "localization_per_sample.csv"
    for path in (compatibility_path, response_path, localization_path):
        _require_manifested_stage_artifact(path, evidence)

    output_dir = root / "path"
    output_dir.mkdir(parents=True, exist_ok=True)
    power_coverage = float(config["path"]["power_coverage"])
    epsilon = _noop_path_threshold(config, dataset, power_coverage)
    provenance = _path_provenance(config, dataset, power_coverage)
    write_json(output_dir / "path_provenance.json", provenance)
    compatibility = _join_effects(
        compatibility_path,
        dataset,
        "matched_minus_alternative",
        epsilon,
        power_coverage,
        evidence,
    )
    response = _join_effects(
        response_path,
        dataset,
        "response_advantage",
        epsilon,
        power_coverage,
        evidence,
    )
    matched_compatibility = _exact_path_match(compatibility)
    matched_response = _exact_path_match(response)
    localization = _localization_rows(
        localization_path,
        dataset,
        epsilon,
        power_coverage,
        evidence,
    )
    write_csv(output_dir / "compatibility_path_effects.csv", bind_rows(compatibility, evidence))
    write_csv(output_dir / "response_path_effects.csv", bind_rows(response, evidence))
    write_csv(
        output_dir / "matched_compatibility_path_effects.csv",
        bind_rows(matched_compatibility, evidence),
    )
    write_csv(
        output_dir / "matched_response_path_effects.csv",
        bind_rows(matched_response, evidence),
    )
    write_csv(output_dir / "localization_path_incidence.csv", bind_rows(localization, evidence))
    summaries = _bin_summaries(matched_compatibility, matched_response)
    write_csv(output_dir / "bin_summaries.csv", bind_rows(summaries, evidence))
    balance = _balance_report(matched_compatibility)
    write_csv(output_dir / "matching_balance.csv", bind_rows(balance, evidence))
    mechanism = _mechanism_gate(
        config,
        matched_compatibility,
        matched_response,
        balance,
        compatibility,
        response,
        localization,
        provenance,
    )
    if dataset.is_fixture:
        mechanism["passed"] = False
        mechanism["software_only_qualification_bypass"] = True
    status = {
        "schema_version": "csi-pairs-v6-path-gate-v3",
        "status": "PASS" if mechanism["passed"] else "FAIL",
        "passed": mechanism["passed"],
        **evidence,
        "gate": "G7",
        "epsilon_path": epsilon,
        "epsilon_source": "source_method_selection canonical no-edit path retraces",
        "matching_rule": "exact seed/arm/route/scene/edit-family/bit strata plus nearest standardized delta-map, BS-UE-distance, UE-edit-distance and LoS covariates",
        "matching_strata": ["seed", "arm", "route", "bank_id", "bit_index", "edit_family"],
        "a_path_loc_aggregation": "unweighted mean over all registered adjacent edits from the natural world",
        "power_coverage": power_coverage,
        "path_provenance_sha256": provenance["registry_sha256"],
        **mechanism,
    }
    write_json(output_dir / "gate.json", status)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return status


def _selected_paths(path_ids, powers, coverage):
    valid = np.flatnonzero((np.asarray(path_ids) >= 0) & (np.asarray(powers) > 0))
    if valid.size == 0:
        raise RuntimeError("path record has no positive-power persistent path")
    order = valid[np.argsort(-np.asarray(powers)[valid], kind="stable")]
    total = float(np.sum(np.asarray(powers)[valid]))
    cumulative = np.cumsum(np.asarray(powers)[order]) / total
    count = int(np.searchsorted(cumulative, float(coverage), side="left")) + 1
    return order[: min(count, order.size)]


def _noop_path_threshold(config, dataset, power_coverage):
    scenes = dataset.indices_for_role("source_method_selection")
    values = []
    for scene_value in scenes:
        scene = int(scene_value)
        for world in range(dataset.world_count):
            for position in range(dataset.position_count):
                values.append(_path_record_change(dataset, scene, world, position, power_coverage))
    if not values:
        raise RuntimeError("no canonical no-edit path retraces are available")
    return float(np.quantile(values, float(config["path"]["zero_quantile"])))


def _path_record_change(dataset, scene, world, position, power_coverage):
    left_ids = dataset.path_ids[scene, world, position]
    right_ids = dataset.noop_path_ids[scene, world, position]
    left_power = dataset.path_power[scene, world, position]
    right_power = dataset.noop_path_power[scene, world, position]
    left = _selected_paths(left_ids, left_power, power_coverage)
    right = _selected_paths(right_ids, right_power, power_coverage)
    left_set = {int(left_ids[index]) for index in left}
    right_set = {int(right_ids[index]) for index in right}
    denominator = float(np.sum(left_power[left]) + np.sum(right_power[right]))
    changed = float(
        np.sum([left_power[index] for index in left if int(left_ids[index]) not in right_set])
        + np.sum([right_power[index] for index in right if int(right_ids[index]) not in left_set])
    )
    return changed / max(denominator, 1e-12)


def _effect_row_identity(row, dataset, path, metric, canonical_bank_cache):
    required = {
        "seed",
        "arm",
        "pair_id",
        "scene_index",
        "bank_id",
        "base_map_cluster_id",
        "canonical_base_map_digest",
        "canonical_bank_digest",
        "city_id",
        "source_world",
        "target_world",
        "position_index",
        "route",
        metric,
    }
    if metric == "response_advantage":
        required.update(("query_index", "wrong_action_match_status"))
    missing = sorted(name for name in required if row.get(name) in (None, ""))
    if missing:
        raise RuntimeError(f"path effect row is missing identity fields {missing}: {path}")
    try:
        seed = int(row["seed"])
        scene = int(row["scene_index"])
        source = int(row["source_world"])
        target = int(row["target_world"])
        position = int(row["position_index"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"path effect row has non-integer identity: {path}") from error
    if not 0 <= scene < len(dataset.bank_ids):
        raise RuntimeError(f"path effect row scene_index is out of range: {path}")
    if not 0 <= source < len(dataset.world_bits) or not 0 <= target < len(dataset.world_bits):
        raise RuntimeError(f"path effect row world index is out of range: {path}")
    if not 0 <= position < dataset.position_ids.shape[1]:
        raise RuntimeError(f"path effect row position_index is out of range: {path}")
    if str(dataset.scene_roles[scene]) not in {"source_final_unseen_bank", "target"}:
        raise RuntimeError(f"path effect row uses a non-evaluation scene role: {path}")
    if (
        str(dataset.scene_roles[scene]) == "target"
        and str(dataset.position_roles[scene, position]) != "query"
    ):
        raise RuntimeError("target support_pool leaked into path denominator")
    if scene not in canonical_bank_cache:
        canonical_bank_cache[scene] = _canonical_bank_digest(dataset, scene)
    expected = {
        "bank_id": str(dataset.bank_ids[scene]),
        "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
        "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
        "canonical_bank_digest": canonical_bank_cache[scene],
        "city_id": str(dataset.city_ids[scene]),
    }
    for name, value in expected.items():
        if str(row.get(name, "")) != str(value):
            raise RuntimeError(
                f"path effect row {name} does not match scene_index={scene}: {path}"
            )
    changed = np.flatnonzero(dataset.world_bits[source] != dataset.world_bits[target])
    if changed.size != 1:
        raise RuntimeError("path effect row is not a direct edit")
    if str(row["arm"]) not in {"endpoint", "alignment", "response", "full"}:
        raise RuntimeError(f"path effect row has an unknown arm: {path}")
    if str(row["route"]) not in {"active", "gray", "null"}:
        raise RuntimeError(f"path effect row has an unknown route: {path}")
    if metric == "matched_minus_alternative":
        left, right = sorted((source, target))
        expected_pair_id = (
            f"{expected['bank_id']}:{left}:{right}:{position}:{source}"
        )
        query = None
    elif metric == "response_advantage":
        try:
            query = int(row["query_index"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"path response row has non-integer query_index: {path}") from error
        if not 0 <= query < PatchSpec.from_metadata(dataset.metadata).patch_count:
            raise RuntimeError(f"path response row query_index is out of range: {path}")
        expected_pair_id = (
            f"{expected['bank_id']}:{source}:{target}:{position}:{query}"
        )
        status = str(row["wrong_action_match_status"])
        if status not in {"exact", "fallback", "failed"}:
            raise RuntimeError(f"path response row has invalid wrong-action status: {path}")
        has_swap_value = row.get("response_advantage_vs_action_swap") not in (None, "")
        if (status == "exact") != has_swap_value:
            raise RuntimeError(
                f"path response row exact status and action-swap metric disagree: {path}"
            )
    else:
        raise ValueError(f"unsupported path effect metric {metric!r}")
    if str(row["pair_id"]) != expected_pair_id:
        raise RuntimeError(
            f"path effect row pair_id does not match its dataset identity: {path}"
        )
    return {
        "seed": seed,
        "scene": scene,
        "source": source,
        "target": target,
        "position": position,
        "query": query,
        "bit_index": int(changed[0]),
        **expected,
    }


def _join_effects(path, dataset, metric, epsilon, power_coverage, evidence):
    if not path.is_file():
        raise RuntimeError(f"path audit requires evaluation artifact {path}")
    output = []
    canonical_bank_cache = {}
    observed_rows = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            _validate_row_evidence(row, evidence, path)
            identity = _effect_row_identity(
                row, dataset, path, metric, canonical_bank_cache
            )
            scene = identity["scene"]
            source = identity["source"]
            target = identity["target"]
            position = identity["position"]
            row_key = (identity["seed"], str(row["arm"]), str(row["pair_id"]))
            if row_key in observed_rows:
                raise RuntimeError(f"path effect table contains duplicate pair identity: {path}")
            observed_rows.add(row_key)
            value = path_incidence(
                dataset,
                scene,
                source,
                target,
                position,
                identity["bit_index"],
                power_coverage,
            )
            covariates = _path_covariates(
                dataset, scene, source, target, position, power_coverage
            )
            metric_value = float(row[metric])
            if not np.isfinite(metric_value):
                raise RuntimeError(f"path effect metric is non-finite: {path}")
            auxiliary = {}
            for name in (
                "response_advantage_vs_action_swap",
                "response_advantage_vs_no_action",
            ):
                if name in row and row[name] != "":
                    auxiliary[name] = float(row[name])
                    if not np.isfinite(auxiliary[name]):
                        raise RuntimeError(f"path auxiliary effect is non-finite: {path}")
            output.append(
                {
                    "seed": identity["seed"],
                    "arm": row["arm"],
                    "pair_id": row["pair_id"],
                    "scene_index": scene,
                    "bank_id": identity["bank_id"],
                    "base_map_cluster_id": identity["base_map_cluster_id"],
                    "canonical_base_map_digest": identity[
                        "canonical_base_map_digest"
                    ],
                    "canonical_bank_digest": identity["canonical_bank_digest"],
                    "city_id": identity["city_id"],
                    "source_world": source,
                    "target_world": target,
                    "position_index": position,
                    **(
                        {"query_index": identity["query"]}
                        if identity["query"] is not None
                        else {}
                    ),
                    "route": row["route"],
                    **(
                        {"wrong_action_match_status": row["wrong_action_match_status"]}
                        if row.get("wrong_action_match_status")
                        else {}
                    ),
                    "bit_index": identity["bit_index"],
                    "a_path": value,
                    "a_path_bin": _path_bin(value, epsilon),
                    **covariates,
                    metric: metric_value,
                    **auxiliary,
                }
            )
    if not output:
        raise RuntimeError("path audit effect table is empty")
    return output


def _path_covariates(dataset, scene, source, target, position, power_coverage):
    action = typed_signed_edit(
        dataset.maps[scene, source],
        dataset.maps[scene, target],
        dataset.map_channel_names,
        int(dataset.metadata["assets"]["material_category_count"]),
    )
    changed = np.any(np.abs(dataset.maps[scene, target] - dataset.maps[scene, source]) > 1e-12, axis=0)
    cells = np.argwhere(changed)
    if cells.size == 0:
        raise RuntimeError("direct path edit has no changed map cell")
    origin = np.asarray(dataset.metadata["representation"]["map_origin_xy_m"], dtype=np.float64)
    resolution = float(dataset.metadata["representation"]["map_resolution_m"])
    centroid_rc = cells.mean(axis=0)
    edit_xy = origin + resolution * np.asarray([centroid_rc[1] + 0.5, centroid_rc[0] + 0.5])
    receiver = np.asarray(dataset.positions[scene, position], dtype=np.float64)
    source_map = np.asarray(dataset.maps[scene, source], dtype=np.float64)
    target_map = np.asarray(dataset.maps[scene, target], dtype=np.float64)
    names = [str(value) for value in dataset.map_channel_names.tolist()]
    occupancy = names.index("occupancy")
    height = names.index("height")
    material = names.index("material")
    cell_count = float(source_map.shape[1] * source_map.shape[2])
    occupancy_fraction = float(
        np.sum(np.abs(target_map[occupancy] - source_map[occupancy]) > 1e-12)
        / cell_count
    )
    height_delta = target_map[height] - source_map[height]
    source_height = dataset.maps[
        dataset.indices_for_role("source_method_selection"), :, height
    ]
    positive_height = source_height[source_height > 0]
    height_reference = float(np.median(positive_height)) if positive_height.size else 1.0
    height_rms = float(np.sqrt(np.mean(height_delta**2)) / max(height_reference, 1e-12))
    material_fraction = float(
        np.sum(np.abs(target_map[material] - source_map[material]) > 1e-12)
        / cell_count
    )
    changed_families = []
    if occupancy_fraction > 0:
        changed_families.append("occupancy")
    if height_rms > 0:
        changed_families.append("height")
    if material_fraction > 0:
        changed_families.append("material")
    return {
        "edit_magnitude_l2": float(np.linalg.norm(action)),
        "delta_map": occupancy_fraction + height_rms + material_fraction,
        "edit_family": "+".join(changed_families),
        "bs_ue_distance_m": float(np.linalg.norm(receiver)),
        "ue_edit_distance_m": float(np.linalg.norm(receiver - edit_xy)),
        "los_indicator": int(
            _has_los_path(dataset, scene, source, position, power_coverage)
            and _has_los_path(dataset, scene, target, position, power_coverage)
        ),
    }


def _has_los_path(dataset, scene, world, position, power_coverage):
    selected = _selected_paths(
        dataset.path_ids[scene, world, position],
        dataset.path_power[scene, world, position],
        power_coverage,
    )
    return any(
        np.all(dataset.path_surface_ids[scene, world, position, index] < 0)
        for index in selected
    )


def _localization_rows(path, dataset, epsilon, power_coverage, evidence):
    if not path.is_file():
        raise RuntimeError("path audit requires localization per-sample results")
    rows = []
    canonical_bank_cache = {}
    observed_rows = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            _validate_row_evidence(row, evidence, path)
            required = {
                "seed",
                "arm",
                "city_id",
                "bank_id",
                "base_map_cluster_id",
                "canonical_base_map_digest",
                "canonical_bank_digest",
                "budget",
                "draw",
                "position_id",
                "true_x",
                "true_y",
                "error_m",
            }
            missing = sorted(name for name in required if row.get(name) in (None, ""))
            if missing:
                raise RuntimeError(
                    f"path localization row is missing identity fields {missing}: {path}"
                )
            clean_row = {
                key: value
                for key, value in row.items()
                if key not in evidence
            }
            scenes = np.flatnonzero(dataset.bank_ids == row["bank_id"])
            if scenes.size != 1:
                raise RuntimeError("localization bank join is not one-to-one")
            scene = int(scenes[0])
            if str(dataset.scene_roles[scene]) != "target":
                raise RuntimeError("localization path input must use target banks")
            positions = np.flatnonzero(dataset.position_ids[scene] == row["position_id"])
            if positions.size != 1:
                raise RuntimeError("localization position join is not one-to-one")
            position = int(positions[0])
            if str(dataset.position_roles[scene, position]) != "query":
                raise RuntimeError("target support_pool leaked into localization path denominator")
            if scene not in canonical_bank_cache:
                canonical_bank_cache[scene] = _canonical_bank_digest(dataset, scene)
            expected = {
                "city_id": str(dataset.city_ids[scene]),
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
                "canonical_bank_digest": canonical_bank_cache[scene],
            }
            for name, expected_value in expected.items():
                if str(row.get(name, "")) != str(expected_value):
                    raise RuntimeError(
                        f"path localization row {name} does not match bank_id={row['bank_id']}: {path}"
                    )
            if not np.allclose(
                [float(row["true_x"]), float(row["true_y"])],
                dataset.positions[scene, position],
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError("path localization truth does not match its position identity")
            error_m = float(row["error_m"])
            if not np.isfinite(error_m) or error_m < 0:
                raise RuntimeError("path localization error must be finite and nonnegative")
            row_key = (
                int(row["seed"]),
                str(row["arm"]),
                str(row["bank_id"]),
                int(row["budget"]),
                int(row["draw"]),
                str(row["position_id"]),
            )
            if row_key in observed_rows:
                raise RuntimeError("path localization table contains duplicate sample identity")
            observed_rows.add(row_key)
            value = localization_path_incidence(
                dataset,
                scene,
                int(dataset.natural_world_index[scene]),
                position,
                power_coverage,
            )
            clean_row.update(expected)
            clean_row["error_m"] = error_m
            rows.append(
                {
                    **clean_row,
                    "a_path_loc": value,
                    "a_path_bin": _path_bin(value, epsilon),
                }
            )
    return rows


def _validate_row_evidence(row, evidence, path):
    expected = {
        key: ("True" if value else "False") if isinstance(value, bool) else str(value)
        for key, value in evidence.items()
        if not isinstance(value, (dict, list))
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"path input {key} mismatch: {path}")


def _bin_summaries(compatibility, response):
    rows = []
    for source, metric in (
        (compatibility, "matched_minus_alternative"),
        (response, "response_advantage"),
    ):
        keys = sorted({(row["arm"], row["route"], row["a_path_bin"]) for row in source})
        for arm, route, bin_name in keys:
            selected = [
                row[metric]
                for row in source
                if (row["arm"], row["route"], row["a_path_bin"]) == (arm, route, bin_name)
            ]
            rows.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "route": route,
                    "a_path_bin": bin_name,
                    "n": len(selected),
                    "mean": float(np.mean(selected)),
                    "median": float(np.median(selected)),
                }
            )
    return rows


def _exact_path_match(rows):
    bins = ("zero", "low", "medium", "high")
    strata = sorted(
        {
            (
                int(row["seed"]),
                row["arm"],
                row["route"],
                row["bank_id"],
                row["bit_index"],
                row["edit_family"],
            )
            for row in rows
        }
    )
    matched = []
    for stratum in strata:
        groups = {
            bin_name: [
                row
                for row in rows
                if (
                    int(row["seed"]),
                    row["arm"],
                    row["route"],
                    row["bank_id"],
                    row["bit_index"],
                    row["edit_family"],
                )
                == stratum
                and row["a_path_bin"] == bin_name
            ]
            for bin_name in bins
        }
        count = min(len(groups[bin_name]) for bin_name in bins)
        if count == 0:
            continue
        covariates = (
            "delta_map",
            "bs_ue_distance_m",
            "ue_edit_distance_m",
            "los_indicator",
        )
        pooled = np.asarray(
            [[float(row[name]) for name in covariates] for group in groups.values() for row in group]
        )
        scale = np.std(pooled, axis=0)
        scale[scale < 1e-9] = 1.0
        anchors = sorted(
            groups["zero"], key=lambda row: (row["bank_id"], row["pair_id"])
        )[:count]
        available = {name: list(groups[name]) for name in bins if name != "zero"}
        for match_index, anchor in enumerate(anchors):
            reference = np.asarray([float(anchor[name]) for name in covariates])
            selected = {"zero": anchor}
            for bin_name in bins[1:]:
                distances = [
                    float(
                        np.linalg.norm(
                            (np.asarray([float(row[name]) for name in covariates]) - reference)
                            / scale
                        )
                    )
                    for row in available[bin_name]
                ]
                best = int(np.argmin(distances))
                selected[bin_name] = available[bin_name].pop(best)
            match_id = f"{'|'.join(map(str, stratum))}|{match_index}"
            for bin_name in bins:
                matched.append(
                    {
                        **selected[bin_name],
                        "matching_stratum": "|".join(map(str, stratum)),
                        "match_set_id": match_id,
                    }
                )
    return matched


def _balance_report(rows):
    selected = [row for row in rows if row["arm"] == "full" and row["route"] == "active"]
    output = []
    bins = ("zero", "low", "medium", "high")
    for field in ("delta_map", "bs_ue_distance_m", "ue_edit_distance_m", "los_indicator"):
        reference = np.asarray(
            [float(row[field]) for row in selected if row["a_path_bin"] == "zero"]
        )
        for bin_name in bins:
            group = np.asarray(
                [float(row[field]) for row in selected if row["a_path_bin"] == bin_name]
            )
            if group.size == 0 or reference.size == 0:
                smd = float("inf")
            else:
                pooled = np.sqrt(max((np.var(group) + np.var(reference)) / 2.0, 1e-12))
                smd = float(abs(np.mean(group) - np.mean(reference)) / pooled)
            output.append(
                {
                    "covariate": field,
                    "level": "continuous_or_binary",
                    "a_path_bin": bin_name,
                    "standardized_mean_difference": smd,
                }
            )
    return output


def _localization_mechanism_intervals(rows, resamples):
    keyed = {}
    row_by_key = {}
    for row in rows:
        key = (
            str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            ),
            str(row.get("canonical_bank_digest") or row["bank_id"]),
            int(row["seed"]),
            int(row["budget"]),
            int(row["draw"]),
            str(row["position_id"]),
        )
        keyed.setdefault((key, str(row["arm"])), []).append(float(row["error_m"]))
        row_by_key.setdefault(key, row)
    output = {}
    for offset, baseline in enumerate(("endpoint", "alignment", "response")):
        selected = []
        for key in sorted(row_by_key):
            if (key, "full") not in keyed or (key, baseline) not in keyed:
                continue
            source = row_by_key[key]
            selected.append(
                {
                    "base_map_cluster_id": key[0],
                    "canonical_base_map_digest": key[0],
                    "bank_id": key[1],
                    "canonical_bank_digest": key[1],
                    "seed": key[2],
                    "a_path": float(source["a_path_loc"]),
                    "localization_advantage": float(
                        np.mean(keyed[(key, baseline)]) - np.mean(keyed[(key, "full")])
                    ),
                }
            )
        if not selected:
            raise RuntimeError(f"localization path audit has no Full-vs-{baseline} pairs")
        output[baseline] = _cluster_slope_interval(
            selected,
            "localization_advantage",
            resamples,
            120100 + offset,
        )
    return output


def _mechanism_gate(
    config,
    compatibility,
    response,
    balance,
    original_compatibility,
    original_response,
    localization,
    provenance,
):
    minimum = float(config["path"]["minimum_trend_slope"])
    margin = float(config["path"]["equivalence_margin"])
    balance_max = float(config["path"]["balance_smd_max"])
    def slope_pass(interval):
        return bool(
            isinstance(interval, dict)
            and interval.get("assessed", True) is True
            and interval.get("ci95_low") is not None
            and float(interval["ci95_low"]) > minimum
        )

    def headline(rows):
        return [
            row
            for row in rows
            if row["arm"] == "full" and row["route"] == "active"
        ]

    def exact_response(rows):
        return [
            row
            for row in headline(rows)
            if row.get("wrong_action_match_status") == "exact"
            and "response_advantage_vs_action_swap" in row
        ]

    comp = headline(compatibility)
    resp = headline(response)
    original_comp = headline(original_compatibility)
    original_resp = headline(original_response)
    original_exact_resp = exact_response(original_response)
    swap_resp = exact_response(response)
    compatibility_overlap_consistent = len(comp) <= len(original_comp)
    exact_response_overlap_consistent = len(swap_resp) <= len(original_exact_resp)
    compatibility_overlap = (
        len(comp) / len(original_comp)
        if original_comp and compatibility_overlap_consistent
        else 0.0
    )
    exact_response_overlap = (
        len(swap_resp) / len(original_exact_resp)
        if original_exact_resp and exact_response_overlap_consistent
        else 0.0
    )
    original_exact_response_coverage = (
        len(original_exact_resp) / len(original_resp) if original_resp else 0.0
    )
    if not comp or not resp:
        return {
            "passed": False,
            "g7_subgates": {
                "1_registered_path_provenance_and_noop_epsilon": "PASS"
                if provenance.get("passed") is True
                else "FAIL",
                "2_covariate_matching_balance_overlap_ess": "FAIL",
                "3_compatibility_trend_cluster_ci": "FAIL",
                "4_response_trend_cluster_ci": "FAIL",
                "5_zero_path_bank_equivalence": "FAIL",
            },
            "failure_reason": "no matched full-active strata spanning zero/low/medium/high path bins",
            "balance_passed": False,
            "matched_overlap_fraction": 0.0,
            "compatibility_matched_overlap_fraction": compatibility_overlap,
            "exact_response_matched_overlap_fraction": exact_response_overlap,
            "original_exact_response_coverage": original_exact_response_coverage,
            "effective_base_map_clusters": 0,
            "compatibility_effective_base_map_clusters": 0,
            "exact_response_effective_base_map_clusters": 0,
            "path_provenance": provenance,
        }
    resamples = int(config["path"]["bootstrap_resamples"])
    comp_slope = _safe_cluster_slope_interval(
        comp, "matched_minus_alternative", resamples, 120000
    )
    response_slope = _safe_cluster_slope_interval(
        resp, "response_advantage", resamples, 120001
    )
    swap_coverage = len(swap_resp) / len(resp)
    minimum_swap_coverage = float(
        config["qualification"]["minimum_geometry_matched_wrong_action_fraction"]
    )
    response_swap_slope = _safe_cluster_slope_interval(
        swap_resp, "response_advantage_vs_action_swap", resamples, 120011
    )
    response_no_action_slope = _safe_cluster_slope_interval(
        resp, "response_advantage_vs_no_action", resamples, 120012
    )
    try:
        localization_slopes = _localization_mechanism_intervals(
            localization, resamples
        )
    except RuntimeError as error:
        localization_slopes = {
            arm: _unassessed_interval(str(error))
            for arm in ("endpoint", "alignment", "response")
        }
    zero_comp = _bank_bootstrap_equivalence(
        [row for row in compatibility if row["arm"] == "full" and row["a_path_bin"] == "zero"],
        "matched_minus_alternative",
        margin,
        resamples,
        120002,
    )
    zero_response = _bank_bootstrap_equivalence(
        [row for row in response if row["arm"] == "full" and row["a_path_bin"] == "zero"],
        "response_advantage",
        margin,
        resamples,
        120003,
    )
    balance_pass = bool(
        balance and max(row["standardized_mean_difference"] for row in balance) <= balance_max
    )
    compatibility_effective_clusters = len(
        {
            str(row.get("canonical_base_map_digest") or row["base_map_cluster_id"])
            for row in comp
        }
    )
    exact_response_effective_clusters = len(
        {
            str(row.get("canonical_base_map_digest") or row["base_map_cluster_id"])
            for row in swap_resp
        }
    )
    minimum_effective = int(config["path"]["minimum_effective_sample_size"])
    minimum_overlap = float(config["path"]["covariate_overlap_minimum"])
    match_pass = bool(
        balance_pass
        and compatibility_overlap_consistent
        and exact_response_overlap_consistent
        and compatibility_effective_clusters >= minimum_effective
        and exact_response_effective_clusters >= minimum_effective
        and compatibility_overlap >= minimum_overlap
        and exact_response_overlap >= minimum_overlap
        and original_exact_response_coverage >= minimum_swap_coverage
    )
    g7_subgates = {
        "1_registered_path_provenance_and_noop_epsilon": "PASS"
        if provenance.get("passed") is True
        else "FAIL",
        "2_covariate_matching_balance_overlap_ess": "PASS" if match_pass else "FAIL",
        "3_compatibility_trend_cluster_ci": "PASS"
        if slope_pass(comp_slope)
        else "FAIL",
        "4_response_trend_cluster_ci": "PASS"
        if slope_pass(response_slope)
        and swap_coverage >= minimum_swap_coverage
        and slope_pass(response_swap_slope)
        and slope_pass(response_no_action_slope)
        and all(slope_pass(value) for value in localization_slopes.values())
        else "FAIL",
        "5_zero_path_bank_equivalence": "PASS"
        if zero_comp["passed"] and zero_response["passed"]
        else "FAIL",
    }
    passed = all(value == "PASS" for value in g7_subgates.values())
    return {
        "passed": passed,
        "g7_subgates": g7_subgates,
        "compatibility_trend_slope": comp_slope,
        "response_trend_slope": response_slope,
        "response_vs_action_swap_trend_slope": response_swap_slope,
        "response_action_swap_exact_coverage": swap_coverage,
        "response_action_swap_original_exact_coverage": original_exact_response_coverage,
        "response_action_swap_minimum_exact_coverage": minimum_swap_coverage,
        "response_vs_no_action_trend_slope": response_no_action_slope,
        "localization_full_advantage_trend_slopes": localization_slopes,
        "minimum_trend_slope": minimum,
        "zero_path_compatibility_equivalence": zero_comp,
        "zero_path_response_equivalence": zero_response,
        "balance_passed": balance_pass,
        "balance_smd_max": balance_max,
        "matched_overlap_fraction": min(
            compatibility_overlap, exact_response_overlap
        ),
        "compatibility_matched_overlap_fraction": compatibility_overlap,
        "exact_response_matched_overlap_fraction": exact_response_overlap,
        "compatibility_overlap_count": len(comp),
        "compatibility_overlap_denominator": len(original_comp),
        "exact_response_overlap_count": len(swap_resp),
        "exact_response_overlap_denominator": len(original_exact_resp),
        "compatibility_effective_base_map_clusters": compatibility_effective_clusters,
        "exact_response_effective_base_map_clusters": exact_response_effective_clusters,
        "effective_base_map_clusters": min(
            compatibility_effective_clusters, exact_response_effective_clusters
        ),
        "path_provenance": provenance,
    }


def _slope(rows, metric):
    if len(rows) < 2 or np.std([row["a_path"] for row in rows]) <= 1e-12:
        raise RuntimeError(f"path trend for {metric} has insufficient variation")
    return float(np.polyfit([row["a_path"] for row in rows], [row[metric] for row in rows], 1)[0])


def _seed_bank_foundation_interval(cells, resamples, seed):
    foundations = sorted(cells)
    if len(foundations) < 2:
        raise RuntimeError("path interval requires two canonical base-map foundations")
    seed_sets = []
    canonical_banks = set()
    for foundation in foundations:
        if not cells[foundation]:
            raise RuntimeError("path interval contains an empty foundation layer")
        for bank, seed_values in cells[foundation].items():
            canonical_banks.add(bank)
            seed_sets.append(set(seed_values))
            if not seed_values or not all(np.isfinite(value) for value in seed_values.values()):
                raise RuntimeError("path interval contains an empty or non-finite seed cell")
    seeds = sorted(seed_sets[0])
    if not seeds or any(set(values) != set(seeds) for values in seed_sets):
        raise RuntimeError(
            "path interval requires the same complete paired training-seed layer in every bank"
        )

    def aggregate(selected_foundations, selected_seeds, rng=None):
        foundation_values = []
        for foundation in selected_foundations:
            banks = sorted(cells[foundation])
            selected_banks = (
                [banks[index] for index in rng.integers(0, len(banks), size=len(banks))]
                if rng is not None
                else banks
            )
            foundation_values.append(
                float(
                    np.mean(
                        [
                            np.mean(
                                [cells[foundation][bank][seed_value] for seed_value in selected_seeds]
                            )
                            for bank in selected_banks
                        ]
                    )
                )
            )
        return float(np.mean(foundation_values))

    estimate = aggregate(foundations, seeds)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        selected_foundations = [
            foundations[value]
            for value in rng.integers(0, len(foundations), size=len(foundations))
        ]
        selected_seeds = [
            seeds[value] for value in rng.integers(0, len(seeds), size=len(seeds))
        ]
        samples[index] = aggregate(selected_foundations, selected_seeds, rng)
    return {
        "estimate": estimate,
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "base_map_cluster_count": len(foundations),
        "canonical_bank_count": len(canonical_banks),
        "training_seed_count": len(seeds),
        "aggregation": "per-seed slope/value then equal canonical-bank and canonical-foundation macro",
        "resampling_layers": [
            "canonical_foundation",
            "canonical_bank_within_foundation",
            "paired_training_seed_across_all_banks",
        ],
    }


def _cluster_slope_interval(rows, metric, resamples, seed):
    grouped = {}
    all_seeds = {int(row["seed"]) for row in rows}
    for row in rows:
        foundation = str(
            row.get("canonical_base_map_digest") or row["base_map_cluster_id"]
        )
        bank = str(row.get("canonical_bank_digest") or row["bank_id"])
        grouped.setdefault(foundation, {}).setdefault(bank, {}).setdefault(
            int(row["seed"]), []
        ).append(row)
    cells = {}
    for foundation, banks in grouped.items():
        for bank, seed_rows in banks.items():
            if set(seed_rows) != all_seeds:
                raise RuntimeError(
                    f"path trend for {metric} has an incomplete paired seed layer"
                )
            variable = {
                seed_value: len(values) >= 2
                and np.std([row["a_path"] for row in values]) > 1e-12
                for seed_value, values in seed_rows.items()
            }
            if any(variable.values()) and not all(variable.values()):
                raise RuntimeError(
                    f"path trend for {metric} has seed-selective within-bank variation"
                )
            if not all(variable.values()):
                continue
            cells.setdefault(foundation, {})[bank] = {
                seed_value: _slope(values, metric)
                for seed_value, values in seed_rows.items()
            }
    try:
        return _seed_bank_foundation_interval(cells, resamples, seed)
    except RuntimeError as error:
        raise RuntimeError(f"path trend for {metric}: {error}") from error


def _unassessed_interval(reason):
    return {
        "assessed": False,
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "base_map_cluster_count": 0,
        "reason": str(reason),
    }


def _safe_cluster_slope_interval(rows, metric, resamples, seed):
    try:
        result = _cluster_slope_interval(rows, metric, resamples, seed)
    except (KeyError, RuntimeError, ValueError) as error:
        return _unassessed_interval(str(error))
    return {"assessed": True, **result}


def _bank_bootstrap_equivalence(rows, metric, margin, resamples, seed):
    grouped = {}
    for row in rows:
        if metric not in row:
            continue
        foundation = str(
            row.get("canonical_base_map_digest") or row["base_map_cluster_id"]
        )
        bank = str(row.get("canonical_bank_digest") or row["bank_id"])
        grouped.setdefault(foundation, {}).setdefault(bank, {}).setdefault(
            int(row["seed"]), []
        ).append(float(row[metric]))
    cells = {
        foundation: {
            bank: {
                seed_value: float(np.mean(values))
                for seed_value, values in seed_rows.items()
            }
            for bank, seed_rows in banks.items()
        }
        for foundation, banks in grouped.items()
    }
    try:
        interval = _seed_bank_foundation_interval(cells, resamples, seed)
    except RuntimeError as error:
        return {
            "passed": False,
            "base_map_cluster_count": len(grouped),
            "canonical_bank_count": len(
                {bank for banks in grouped.values() for bank in banks}
            ),
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "margin": margin,
            "reason": f"zero-path equivalence for {metric}: {error}",
        }
    return {
        **interval,
        "mean": interval["estimate"],
        "margin": margin,
        "passed": bool(
            interval["ci95_low"] >= -margin
            and interval["ci95_high"] <= margin
        ),
    }


def _path_bin(value: float, epsilon: float = 0.0) -> str:
    if value <= epsilon + 1e-12:
        return "zero"
    first = epsilon + (1.0 - epsilon) / 3.0
    second = epsilon + 2.0 * (1.0 - epsilon) / 3.0
    if value <= first:
        return "low"
    if value <= second:
        return "medium"
    return "high"
