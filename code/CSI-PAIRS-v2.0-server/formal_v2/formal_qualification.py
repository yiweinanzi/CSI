from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .formal_baselines import RidgeRegressor, normalized_mse
from .formal_config import public_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import QUALIFICATION_SCHEMA, bind_rows, complete_gate_vector, evidence_context
from .formal_features import protocol_response_features, variant_features
from .formal_io import artifact_manifest, sha256_file, write_csv, write_json
from .formal_model import module_device, resolve_execution_device
from .formal_protocol import PatchSpec, delay_angle_power, patchify_csi, typed_signed_edit, zero_typed_edit
from .formal_routing import (
    PRIMARY_ROUTE_CONTRACT,
    RouteNormalization,
    fit_route_normalization,
    route_coverage,
    route_dataset,
)
from .formal_teacher import (
    TeacherBundle,
    masked_reconstruction_nmse,
    save_teacher_bundle,
    teacher_targets,
    train_teacher_bundle,
)


QUALIFICATION_METHODS = (
    "no_x",
    "oracle_x",
    "no_action",
    "map_edit_only",
    "csi_only",
    "variant_id_only",
)
BASELINES = ("copy", "no_action", "action_swap")


def run_formal_qualification(
    config: dict,
    dataset: FormalDataset,
    output_root: str | Path,
    data_verification_gate: dict,
    data_verification_gate_path: str | Path | None = None,
) -> dict:
    from .formal_data_verification import require_data_verification

    data_verification_gate = require_data_verification(
        data_verification_gate,
        config,
        dataset,
        gate_path=data_verification_gate_path,
    )
    output_dir = Path(output_root) / "qualification"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    dataset.validate(
        require_clean_csi=bool(data_config["require_clean_csi"]),
        minimum_repeats=int(data_config["minimum_repeats"]),
        minimum_target_cities=int(data_config["minimum_target_cities"]),
        minimum_source_cities=int(data_config["minimum_source_cities"]),
        minimum_banks_per_target_city=int(data_config["minimum_banks_per_target_city"]),
        minimum_independent_base_map_clusters_per_target_city=int(
            data_config["minimum_independent_base_map_clusters_per_target_city"]
        ),
        minimum_banks_per_source_role=int(data_config["minimum_banks_per_source_role"]),
    )
    provisional_evidence = evidence_context(config, dataset, "FORBIDDEN")
    contract = {**dataset.contract_report(), **provisional_evidence}
    write_json(output_dir / "data_contract.json", contract)

    blocking = qualification_blocking_scenes(dataset)
    train_scenes = blocking["teacher_train"]
    selection_scenes = blocking["method_selection"]
    patch_spec = PatchSpec.from_metadata(dataset.metadata)
    execution_device = resolve_execution_device(dataset)
    teacher_bundle = train_teacher_bundle(
        dataset.csi[train_scenes],
        patch_spec,
        config,
        seed=int(config["seeds"][0]) + 1009,
        device=execution_device,
    )
    teacher_checkpoint = output_dir / "checkpoints" / "stage0_csi_teacher.pt"
    save_teacher_bundle(teacher_checkpoint, teacher_bundle, config, int(config["seeds"][0]) + 1009)
    route_normalization = fit_route_normalization(dataset, teacher_bundle)
    routed_selection = route_dataset(
        dataset,
        teacher_bundle,
        config,
        selection_scenes,
        normalization=route_normalization,
    )
    train_routed = route_dataset(
        dataset,
        teacher_bundle,
        config,
        train_scenes,
        normalization=route_normalization,
    )

    repeat_rows = _repeat_rows(dataset, selection_scenes, route_normalization.channel_scale, config)
    route_noise_floor_rows = _route_noise_floor_rows(
        dataset,
        teacher_bundle,
        selection_scenes,
        route_normalization,
        config,
    )
    selection_coverage_rows = route_coverage(dataset, routed_selection, selection_scenes)
    branch_rows = {
        row["scene_id"]: row
        for row in _branch_coverage(dataset, routed_selection, selection_scenes)
    }
    qualification = config["qualification"]
    for row in selection_coverage_rows:
        row.update(branch_rows[row["scene_id"]])
        row["coverage_scope"] = "source_method_selection_with_wrong_action_audit"
        row["passed"] = bool(
            _route_coverage_passed(row, qualification)
            and row["active_branching_fraction"]
            >= float(qualification["minimum_active_branching_fraction"])
            and row["geometry_matched_wrong_action_fraction"]
            >= float(qualification["minimum_geometry_matched_wrong_action_fraction"])
        )
    train_coverage_rows = route_coverage(dataset, train_routed, train_scenes)
    for row in train_coverage_rows:
        row["coverage_scope"] = "source_encoder_train_per_bank"
        row["passed"] = _route_coverage_passed(row, qualification)
    coverage_rows = train_coverage_rows + selection_coverage_rows
    teacher_rows, teacher_passed = _teacher_qualification(
        dataset, teacher_bundle, routed_selection, selection_scenes, config
    )

    train_records = _build_records(dataset, train_scenes, train_routed, teacher_bundle.mask_bank)
    selection_records = _build_records(
        dataset, selection_scenes, routed_selection, teacher_bundle.mask_bank
    )
    if not train_records or not selection_records:
        raise ValueError("V6 qualification requires nonempty train and selection records")
    targets = np.vstack([row["target_delta"] for row in train_records])
    models: dict[str, RidgeRegressor] = {}
    for method in QUALIFICATION_METHODS:
        features = np.vstack([row[method] for row in train_records])
        model = RidgeRegressor(float(qualification["ridge"])).fit(features, targets)
        model.save(
            output_dir / "checkpoints" / f"{method}.npz",
            {
                "schema_version": "csi-pairs-v6-qualification-probe-v1",
                "input_contract": _method_contract(method),
                "target": "source-normalized clean physical CSI patch delta",
                "source_roles": ["source_encoder_train"],
                "status": "PRETRAINING_QUALIFICATION_PROBE_NOT_HEADLINE_MODEL",
                **provisional_evidence,
            },
        )
        models[method] = model
    selection_target = np.vstack([row["target_delta"] for row in selection_records])
    predictions = {
        method: model.predict(np.vstack([row[method] for row in selection_records]))
        for method, model in models.items()
    }
    predictions["copy"] = np.zeros_like(predictions["no_x"])
    predictions["action_swap"] = models["no_x"].predict(
        np.vstack([row["action_swap"] for row in selection_records])
    )
    response_rows, null_rows, response_gate_rows = _response_gate_rows(
        dataset, selection_records, selection_target, predictions, qualification
    )
    warnings = _shortcut_warnings(selection_records, selection_target, predictions, qualification)

    repeat_passed = all(bool(row["passed"]) for row in repeat_rows)
    route_noise_floor_passed = all(bool(row["passed"]) for row in route_noise_floor_rows)
    coverage_passed = all(bool(row["passed"]) for row in coverage_rows)
    response_passed = all(bool(row["passed"]) for row in response_gate_rows)
    randomization_passed = bool(warnings["shortcut_gate_passed"])
    g1_passed = bool(repeat_passed and route_noise_floor_passed and coverage_passed)
    g2_passed = bool(teacher_passed and response_passed and randomization_passed)
    passed = bool(g1_passed and g2_passed)
    scientific_use = (
        "FORMAL_EXPERIMENT_ALLOWED"
        if passed and not dataset.is_fixture and dataset.metadata["scientific_use"] == "QUALIFIED"
        else "FORBIDDEN"
    )
    evidence = evidence_context(config, dataset, scientific_use)
    for path, rows in (
        ("repeat_noise.csv", repeat_rows),
        ("route_noise_floor.csv", route_noise_floor_rows),
        ("route_coverage.csv", coverage_rows),
        ("teacher_qualification.csv", teacher_rows),
        ("response_scene_metrics.csv", response_rows),
        ("null_safety.csv", null_rows),
        ("response_gate.csv", response_gate_rows),
    ):
        write_csv(output_dir / path, bind_rows(rows, evidence))
    warnings.update(evidence)
    write_json(output_dir / "shortcut_audit.json", warnings)

    gate_vector = complete_gate_vector(
        {
            "G1": "PASS" if g1_passed else "FAIL",
            "G2": "PASS" if g2_passed else "FAIL",
        }
    )
    status = (
        "DRY_RUN_PASS_NOT_EVIDENCE"
        if dataset.is_fixture and passed
        else "DRY_RUN_FAIL_NOT_EVIDENCE"
        if dataset.is_fixture
        else "QUALIFICATION_PASS"
        if scientific_use == "FORMAL_EXPERIMENT_ALLOWED"
        else "QUALIFICATION_NO_GO"
    )
    gate = {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": status,
        "passed": passed,
        **evidence,
        "upstream_gates": gate_vector,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
        "external_validity": {
            "available": bool(dataset.metadata["external_reference"]["available"]),
            "status": "NOT_ASSESSED",
            "blocking_for_real_causal_claim": True,
        },
        "data_verification_gate_schema": data_verification_gate["schema_version"],
        "data_verification_blocking_roles": data_verification_gate["blocking_roles"],
        "g1_components": {
            "repeat_noise": "PASS" if repeat_passed else "FAIL",
            "native_route_noise_floor": "PASS" if route_noise_floor_passed else "FAIL",
            "route_coverage": "PASS" if coverage_passed else "FAIL",
        },
        "route_noise_floor_contract": {
            "source_role": "source_method_selection",
            "quantile": float(qualification["noise_floor_quantile"]),
            "replicate_contrast": "all unordered pairs of independent repeats for the same scene/world/position",
            "normalization_source_role": "source_encoder_train",
            "threshold_rule": "each configured null threshold must cover its same-unit per-bank repeat-pair quantile",
        },
        "decision_rule": "Source-encoder-train banks determine route normalization and per-bank train-route coverage. Alignment primary inclusion uses full-channel physical distance only; Response primary inclusion uses query-patch physical distance only. Teacher sensitivity is an independent audit stratum and auxiliary G2 qualification condition, never a primary inclusion rule. Source-method-selection banks determine repeat noise, same-unit noise floors, branch, teacher, and response qualification. Target, external, probe, calibration, and final-unseen banks are unread.",
        "config": public_formal_config(config),
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def _repeat_rows(
    dataset: FormalDataset,
    scenes: np.ndarray,
    scale: np.ndarray,
    config: dict,
) -> list[dict]:
    rows = []
    for scene_value in scenes:
        scene = int(scene_value)
        repeated = dataset.csi_repeat[scene]
        center = dataset.csi[scene]
        repeat_nmse = float(np.mean(((repeated - center[:, :, None, :]) / scale) ** 2))
        clean_repeat_nmse = float(np.mean(((repeated.mean(axis=2) - center) / scale) ** 2))
        passed = bool(
            repeat_nmse <= float(config["qualification"]["repeat_noise_nmse_max"])
            and clean_repeat_nmse <= float(config["qualification"]["clean_repeat_nmse_max"])
        )
        rows.append(
            {
                "scene_id": str(dataset.scene_ids[scene]),
                "bank_id": str(dataset.bank_ids[scene]),
                "role": str(dataset.scene_roles[scene]),
                "repeat_noise_nmse": repeat_nmse,
                "clean_repeat_nmse": clean_repeat_nmse,
                "passed": passed,
            }
        )
    return rows


def _route_noise_floor_rows(
    dataset: FormalDataset,
    bundle: TeacherBundle,
    scenes: np.ndarray,
    normalization: RouteNormalization,
    config: dict,
) -> list[dict]:
    """Audit every route null threshold against repeat noise in the same norm."""
    spec = PatchSpec.from_metadata(dataset.metadata)
    patch_scale = patchify_csi(normalization.channel_scale, spec)
    by_bank: dict[str, dict[str, object]] = {}
    for scene_value in np.asarray(scenes):
        scene = int(scene_value)
        role = str(dataset.scene_roles[scene])
        if role != "source_method_selection":
            raise ValueError("route noise floors may only read source_method_selection banks")
        bank_id = str(dataset.bank_ids[scene])
        values = by_bank.setdefault(
            bank_id,
            {
                "scene_ids": [],
                "alignment_physical": [],
                "alignment_latent": [],
                "response_physical": [],
                "response_latent": [],
            },
        )
        values["scene_ids"].append(str(dataset.scene_ids[scene]))
        repeats = dataset.csi_repeat[scene]
        repeat_patches = patchify_csi(repeats, spec)
        repeat_latent = teacher_targets(bundle, repeats)
        repeat_delay_angle = delay_angle_power(repeats, spec)
        for first in range(dataset.repeat_count):
            for second in range(first + 1, dataset.repeat_count):
                complex_difference = (
                    repeats[:, :, second] - repeats[:, :, first]
                ) / normalization.channel_scale
                delay_angle_difference = (
                    repeat_delay_angle[:, :, second]
                    - repeat_delay_angle[:, :, first]
                ) / normalization.delay_angle_scale
                alignment_physical = np.sqrt(
                    np.mean(
                        np.concatenate(
                            (complex_difference, delay_angle_difference), axis=-1
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                latent_difference = (
                    repeat_latent[:, :, second] - repeat_latent[:, :, first]
                ) / normalization.latent_scale
                alignment_latent = np.sqrt(
                    np.mean(latent_difference**2, axis=(-2, -1))
                )
                response_physical = np.sqrt(
                    np.mean(
                        (
                            (repeat_patches[:, :, second] - repeat_patches[:, :, first])
                            / patch_scale
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                response_latent = np.sqrt(np.mean(latent_difference**2, axis=-1))
                values["alignment_physical"].extend(alignment_physical.ravel().tolist())
                values["alignment_latent"].extend(alignment_latent.ravel().tolist())
                values["response_physical"].extend(response_physical.ravel().tolist())
                values["response_latent"].extend(response_latent.ravel().tolist())

    qualification = config["qualification"]
    quantile = float(qualification["noise_floor_quantile"])
    metric_thresholds = {
        "alignment_physical": "physical_null_rms_max",
        "alignment_latent": "latent_null_rms_max",
        "response_physical": "response_physical_null_rms_max",
        "response_latent": "response_latent_null_rms_max",
    }
    rows = []
    for bank_id in sorted(by_bank):
        values = by_bank[bank_id]
        row: dict[str, object] = {
            "bank_id": bank_id,
            "scene_ids": ";".join(sorted(values["scene_ids"])),
            "source_role": "source_method_selection",
            "quantile": quantile,
            "quantile_method": "higher",
            "repeat_pair_count": len(values["alignment_physical"]),
            "response_pair_patch_count": len(values["response_physical"]),
        }
        metric_passes = []
        for metric, threshold_key in metric_thresholds.items():
            samples = np.asarray(values[metric], dtype=np.float64)
            if samples.size == 0 or not np.all(np.isfinite(samples)):
                raise ValueError(f"route noise floor {metric} is empty or nonfinite")
            floor = float(np.quantile(samples, quantile, method="higher"))
            threshold = float(qualification[threshold_key])
            prefix = f"{metric}_noise_floor_rms"
            row[prefix] = floor
            row[f"{metric}_null_threshold_rms"] = threshold
            row[f"{metric}_threshold_covers_noise"] = bool(floor <= threshold)
            metric_passes.append(bool(floor <= threshold))
        row["passed"] = bool(all(metric_passes))
        rows.append(row)
    if not rows:
        raise ValueError("route noise floor audit requires source_method_selection banks")
    return rows


def qualification_blocking_scenes(dataset: FormalDataset) -> dict[str, np.ndarray]:
    """The only banks allowed to affect the qualification startup decision."""
    return {
        "teacher_train": dataset.indices_for_role("source_encoder_train"),
        "method_selection": dataset.indices_for_role("source_method_selection"),
    }


def _route_coverage_passed(row: dict, qualification: dict) -> bool:
    return bool(
        row["alignment_active_units"]
        >= int(qualification["minimum_active_units_per_bank"])
        and row["alignment_null_units"]
        >= int(qualification["minimum_null_units_per_bank"])
        and row["response_patch_active_units"]
        >= int(qualification["minimum_active_units_per_bank"])
        and row["response_patch_null_units"]
        >= int(qualification["minimum_null_units_per_bank"])
    )


def _teacher_qualification(dataset, bundle, routed, scenes, config):
    rows = []
    passed = True
    q = config["qualification"]
    for scene_value in scenes:
        scene = int(scene_value)
        patches = routed.physical_patches[scene]
        latent = routed.teacher_latent[scene]
        with torch.no_grad():
            prediction = (
                bundle.readout(
                    torch.as_tensor(
                        latent,
                        dtype=torch.float32,
                        device=module_device(bundle.readout),
                    )
                )
                .cpu()
                .numpy()
            )
        readout_nmse = normalized_mse(patches, prediction)
        alignment_items = [
            distances
            for key, distances in routed.alignment_distances.items()
            if key[0] == scene
        ]
        physical_active = [value for value in alignment_items if value[0] >= float(q["physical_active_rms_min"])]
        physical_null = [value for value in alignment_items if value[0] <= float(q["physical_null_rms_max"])]
        active_agreement = (
            float(np.mean([value[1] >= float(q["latent_active_rms_min"]) for value in physical_active]))
            if physical_active
            else 0.0
        )
        null_agreement = (
            float(np.mean([value[1] <= float(q["latent_null_rms_max"]) for value in physical_null]))
            if physical_null
            else 0.0
        )
        reconstruction_nmse = masked_reconstruction_nmse(
            bundle, dataset.csi[scene], audit=True
        )
        scene_passed = bool(
            reconstruction_nmse <= float(q["teacher_reconstruction_nmse_max"])
            and readout_nmse <= float(q["teacher_reconstruction_nmse_max"])
            and active_agreement >= float(q["teacher_physical_active_agreement_min"])
            and null_agreement >= float(q["teacher_physical_null_agreement_min"])
        )
        passed = passed and scene_passed
        rows.append(
            {
                "scene_id": str(dataset.scene_ids[scene]),
                "bank_id": str(dataset.bank_ids[scene]),
                "masked_teacher_reconstruction_nmse": reconstruction_nmse,
                "mask_bank": "B_audit_hold",
                "qualification_role": "source_method_selection",
                "frozen_readout_nmse": readout_nmse,
                "physical_active_teacher_sensitive_agreement": active_agreement,
                "physical_null_teacher_null_agreement": null_agreement,
                "passed": scene_passed,
            }
        )
    return rows, bool(passed)


def _build_records(dataset, scenes, routed, mask_bank):
    records = []
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    for scene_value in scenes:
        scene = int(scene_value)
        position_center = dataset.positions[scene].mean(axis=0)
        position_scale = dataset.positions[scene].std(axis=0)
        position_scale[position_scale < 1e-9] = 1.0
        for edge in dataset.directed_edges(scene):
            source_map = dataset.maps[scene, edge.source_world]
            target_map = dataset.maps[scene, edge.target_world]
            action = typed_signed_edit(source_map, target_map, dataset.map_channel_names, material_categories)
            zero_action = zero_typed_edit((), source_map.shape[-1], material_categories)
            for position in range(dataset.position_count):
                swap_action, swap_status, swap_world = _select_wrong_action(
                    dataset,
                    scene,
                    edge.source_world,
                    edge.bit_index,
                    action,
                    material_categories,
                    receiver_position=dataset.positions[scene, position],
                )
                source_patches = routed.physical_patches[scene][edge.source_world, position]
                target_patches = routed.physical_patches[scene][edge.target_world, position]
                normalized_position = (dataset.positions[scene, position] - position_center) / position_scale
                for query in range(source_patches.shape[0]):
                    entry = next(item for item in mask_bank if item.mode == "random_75" and item.query == query)
                    visible = source_patches.copy()
                    visible[entry.mask] = 0.0
                    complete_context = np.concatenate(
                        (dataset.radio_config[scene], dataset.bs_pose[scene])
                    )
                    common = dict(
                        visible_patches=visible,
                        source_map=source_map,
                        radio_config=complete_context,
                        mask=entry.mask,
                        query=query,
                    )
                    route = routed.response_route[(scene, edge.source_world, edge.target_world, position, query)]
                    records.append(
                        {
                            "scene": scene,
                            "bank_id": str(dataset.bank_ids[scene]),
                            "route": route,
                            "target_delta": target_patches[query] - source_patches[query],
                            "no_x": protocol_response_features(**common, typed_action=action),
                            "oracle_x": protocol_response_features(
                                **common, typed_action=action, position=normalized_position
                            ),
                            "no_action": protocol_response_features(**common, typed_action=zero_action),
                            "map_edit_only": protocol_response_features(
                                **common, typed_action=action, include_csi=False
                            ),
                            "csi_only": protocol_response_features(
                                **common, typed_action=zero_action, include_action=False, include_map=False
                            ),
                            "variant_id_only": variant_features(
                                dataset.world_bits[edge.source_world], dataset.world_bits[edge.target_world]
                            ),
                            "action_swap": protocol_response_features(**common, typed_action=swap_action),
                            "wrong_action_match_status": swap_status,
                            "wrong_action_world": swap_world,
                        }
                    )
    return records


def _response_gate_rows(dataset, records, target, predictions, config):
    metric_rows = []
    null_rows = []
    gate_rows = []
    routes = np.asarray([row["route"] for row in records])
    wrong_action_status = np.asarray([row["wrong_action_match_status"] for row in records])
    for scene in sorted({int(row["scene"]) for row in records}):
        scene_mask = np.asarray([int(row["scene"]) == scene for row in records])
        active = scene_mask & (routes == 2)
        null = scene_mask & (routes == 0)
        active_scores = {}
        for method, prediction in predictions.items():
            for stratum, mask in (("all", scene_mask), ("active", active), ("null", null)):
                score = normalized_mse(target[mask], prediction[mask]) if np.any(mask) else None
                metric_rows.append(
                    {
                        "scene_id": str(dataset.scene_ids[scene]),
                        "bank_id": str(dataset.bank_ids[scene]),
                        "method": method,
                        "stratum": stratum,
                        "n": int(np.sum(mask)),
                        "physical_patch_delta_nmse": score,
                    }
                )
                if stratum == "active" and score is not None:
                    active_scores[method] = score
        if not active_scores or not np.any(null):
            raise RuntimeError("empty active/null route in a qualified selection bank")
        null_norm = np.sqrt(np.mean(predictions["no_x"][null] ** 2, axis=1))
        violation = float(np.mean(null_norm > float(config["response_physical_null_rms_max"])))
        improvements = {
            baseline: _relative_improvement(active_scores["no_x"], active_scores[baseline])
            for baseline in BASELINES[:2]
        }
        exact_wrong_action = active & (wrong_action_status == "exact")
        if np.any(exact_wrong_action):
            exact_no_x = normalized_mse(target[exact_wrong_action], predictions["no_x"][exact_wrong_action])
            exact_swap = normalized_mse(
                target[exact_wrong_action], predictions["action_swap"][exact_wrong_action]
            )
            improvements["action_swap"] = _relative_improvement(exact_no_x, exact_swap)
        else:
            improvements["action_swap"] = None
        oracle = _relative_improvement(active_scores["oracle_x"], active_scores["copy"])
        passed = bool(
            improvements["action_swap"] is not None
            and all(
                value >= float(config["no_x_min_relative_improvement"])
                for value in improvements.values()
                if value is not None
            )
            and oracle >= float(config["oracle_min_relative_improvement"])
            and violation <= float(config["null_violation_rate_max"])
        )
        null_rows.append(
            {
                "scene_id": str(dataset.scene_ids[scene]),
                "bank_id": str(dataset.bank_ids[scene]),
                "null_units": int(np.sum(null)),
                "prediction_rms_median": float(np.median(null_norm)),
                "violation_rate": violation,
                "passed": violation <= float(config["null_violation_rate_max"]),
            }
        )
        gate_rows.append(
            {
                "scene_id": str(dataset.scene_ids[scene]),
                "bank_id": str(dataset.bank_ids[scene]),
                "relative_improvement_vs_copy": improvements["copy"],
                "relative_improvement_vs_no_action": improvements["no_action"],
                "relative_improvement_vs_action_swap": improvements["action_swap"],
                "action_swap_exact_common_denominator": int(np.sum(exact_wrong_action)),
                "oracle_relative_improvement_vs_copy": oracle,
                "null_violation_rate": violation,
                "passed": passed,
            }
        )
    return metric_rows, null_rows, gate_rows


def _branch_coverage(dataset, routed, scenes):
    rows = []
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    for scene_value in scenes:
        scene = int(scene_value)
        active_states = 0
        active_branching = 0
        active_units = 0
        match_units = {"exact": 0, "fallback": 0, "failed": 0}
        match_actions = {"exact": 0, "fallback": 0, "failed": 0}
        for source in range(dataset.world_count):
            edges = [edge for edge in dataset.directed_edges(scene) if edge.source_world == source]
            for position in range(dataset.position_count):
                for query in range(routed.physical_patches[scene].shape[-2]):
                    active_targets = [
                        edge.target_world
                        for edge in edges
                        if routed.response_route[(scene, source, edge.target_world, position, query)] == 2
                    ]
                    if active_targets:
                        active_states += 1
                        active_branching += int(len(active_targets) >= 2)
                for edge in edges:
                    correct = typed_signed_edit(
                        dataset.maps[scene, source],
                        dataset.maps[scene, edge.target_world],
                        dataset.map_channel_names,
                        material_categories,
                    )
                    _, status, _ = _select_wrong_action(
                        dataset,
                        scene,
                        source,
                        edge.bit_index,
                        correct,
                        material_categories,
                        receiver_position=dataset.positions[scene, position],
                    )
                    match_actions[status] += 1
                    routes = [
                        routed.response_route[(scene, source, edge.target_world, position, query)]
                        for query in range(routed.physical_patches[scene].shape[-2])
                    ]
                    units = int(np.sum(np.asarray(routes) == 2))
                    active_units += units
                    match_units[status] += units
        if active_states == 0 or active_units == 0:
            raise RuntimeError("branch coverage has an empty active denominator")
        if sum(match_units.values()) != active_units:
            raise AssertionError("wrong-action coverage does not partition the active denominator")
        rows.append(
            {
                "scene_id": str(dataset.scene_ids[scene]),
                "active_source_state_units": active_states,
                "active_branching_units": active_branching,
                "active_branching_fraction": active_branching / active_states,
                "active_response_units_for_wrong_action": active_units,
                "wrong_action_exact_units": match_units["exact"],
                "wrong_action_fallback_units": match_units["fallback"],
                "wrong_action_failed_units": match_units["failed"],
                "wrong_action_exact_actions": match_actions["exact"],
                "wrong_action_fallback_actions": match_actions["fallback"],
                "wrong_action_failed_actions": match_actions["failed"],
                "wrong_action_exact_fraction": match_units["exact"] / active_units,
                "wrong_action_fallback_fraction": match_units["fallback"] / active_units,
                "wrong_action_failed_fraction": match_units["failed"] / active_units,
                "geometry_matched_wrong_action_units": match_units["exact"],
                "geometry_matched_wrong_action_fraction": match_units["exact"] / active_units,
            }
        )
    return rows


def _select_wrong_action(
    dataset,
    scene: int,
    source_world: int,
    correct_bit: int,
    correct_action: np.ndarray,
    material_categories: int,
    *,
    receiver_position: np.ndarray | None = None,
) -> tuple[np.ndarray, str, int | None]:
    """Choose a position-specific different primitive and report match quality."""
    lookup = {tuple(row): index for index, row in enumerate(dataset.world_bits.tolist())}
    representation = dataset.metadata.get("representation", {})
    origin = representation.get("map_origin_xy_m")
    resolution = representation.get("map_resolution_m")
    receiver_geometry_available = bool(
        receiver_position is not None
        and origin is not None
        and resolution is not None
        and np.isfinite(float(resolution))
        and float(resolution) > 0.0
    )
    candidates = []
    for bit_index in range(dataset.bit_count):
        if bit_index == int(correct_bit):
            continue
        bits = dataset.world_bits[int(source_world)].copy()
        bits[bit_index] = 1 - bits[bit_index]
        target_world = int(lookup[tuple(int(value) for value in bits)])
        action = typed_signed_edit(
            dataset.maps[int(scene), int(source_world)],
            dataset.maps[int(scene), target_world],
            dataset.map_channel_names,
            material_categories,
        )
        candidates.append(
            (
                action,
                target_world,
                _wrong_action_distance(
                    correct_action,
                    action,
                    receiver_position=(receiver_position if receiver_geometry_available else None),
                    map_origin_xy_m=(origin if receiver_geometry_available else None),
                    map_resolution_m=(resolution if receiver_geometry_available else None),
                ),
            )
        )
    if not candidates:
        return np.zeros_like(correct_action), "failed", None
    exact = [
        candidate
        for candidate in candidates
        if receiver_geometry_available
        and candidate[2][0] == 0
        and candidate[2][1] == 0
    ]
    if exact:
        action, target_world, _ = min(exact, key=lambda candidate: candidate[2])
        return action, "exact", target_world
    fallback = [candidate for candidate in candidates if candidate[2][0] == 0]
    if fallback:
        action, target_world, _ = min(fallback, key=lambda candidate: candidate[2])
        return action, "fallback", target_world
    action, target_world, _ = min(candidates, key=lambda candidate: candidate[2])
    return action, "failed", target_world


def _wrong_action_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    receiver_position: np.ndarray | None = None,
    map_origin_xy_m: np.ndarray | None = None,
    map_resolution_m: float | None = None,
) -> tuple:
    reference_profile = _action_geometry_profile(
        reference,
        receiver_position=receiver_position,
        map_origin_xy_m=map_origin_xy_m,
        map_resolution_m=map_resolution_m,
    )
    candidate_profile = _action_geometry_profile(
        candidate,
        receiver_position=receiver_position,
        map_origin_xy_m=map_origin_xy_m,
        map_resolution_m=map_resolution_m,
    )
    family_mismatch = int(reference_profile["family"] != candidate_profile["family"])
    area_relatives = []
    norm_relatives = []
    bbox_distance = 0
    pattern_mismatch = 0
    for reference_area, candidate_area, reference_norm, candidate_norm, reference_bbox, candidate_bbox, reference_pattern, candidate_pattern in zip(
        reference_profile["plane_areas"],
        candidate_profile["plane_areas"],
        reference_profile["plane_norms"],
        candidate_profile["plane_norms"],
        reference_profile["plane_bbox_shapes"],
        candidate_profile["plane_bbox_shapes"],
        reference_profile["plane_canonical_patterns"],
        candidate_profile["plane_canonical_patterns"],
    ):
        if reference_area == candidate_area == 0:
            continue
        area_relatives.append(
            abs(int(candidate_area) - int(reference_area)) / max(int(reference_area), 1)
        )
        norm_relatives.append(
            abs(float(candidate_norm) - float(reference_norm))
            / max(float(reference_norm), np.finfo(np.float64).eps)
        )
        bbox_distance += sum(
            abs(int(left) - int(right))
            for left, right in zip(reference_bbox, candidate_bbox)
        )
        if reference_pattern.shape != candidate_pattern.shape or not np.allclose(
            reference_pattern,
            candidate_pattern,
            rtol=0.05,
            atol=1e-12,
        ):
            pattern_mismatch += 1
    area_relative = max(area_relatives, default=0.0)
    norm_relative = max(norm_relatives, default=0.0)
    receiver_relative = 0.0
    reference_distances = reference_profile["receiver_plane_distances_m"]
    candidate_distances = candidate_profile["receiver_plane_distances_m"]
    if reference_distances is not None and candidate_distances is not None:
        receiver_relative = max(
            (
                abs(float(candidate_distance) - float(reference_distance))
                / max(
                    float(reference_distance),
                    float(map_resolution_m),
                    np.finfo(np.float64).eps,
                )
                for reference_distance, candidate_distance, active in zip(
                    reference_distances,
                    candidate_distances,
                    reference_profile["family"],
                )
                if active
            ),
            default=0.0,
        )
    geometry_mismatch = int(
        area_relative > 0.05
        or norm_relative > 0.05
        or bbox_distance != 0
        or pattern_mismatch != 0
        or receiver_relative > 0.05
    )
    return (
        family_mismatch,
        geometry_mismatch,
        area_relative + norm_relative + receiver_relative,
        bbox_distance,
    )


def _action_geometry_profile(
    action: np.ndarray,
    *,
    receiver_position: np.ndarray | None = None,
    map_origin_xy_m: np.ndarray | None = None,
    map_resolution_m: float | None = None,
) -> dict:
    values = np.asarray(action, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 5:
        raise ValueError("typed action must have shape [channel,row,column]")
    active_planes = np.any(np.abs(values) > 1e-12, axis=(1, 2))
    family = tuple(bool(value) for value in active_planes.tolist())
    support = np.any(np.abs(values) > 1e-12, axis=0)
    coordinates = np.argwhere(support)
    if coordinates.size:
        extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        bbox_shape = tuple(int(value) for value in extent)
    else:
        bbox_shape = (0, 0)
    plane_areas = []
    plane_norms = []
    plane_bbox_shapes = []
    plane_canonical_patterns = []
    receiver_plane_distances = []
    geometry_available = bool(
        receiver_position is not None
        and map_origin_xy_m is not None
        and map_resolution_m is not None
    )
    receiver_xy = (
        np.asarray(receiver_position, dtype=np.float64).reshape(-1)[:2]
        if geometry_available
        else None
    )
    origin = (
        np.asarray(map_origin_xy_m, dtype=np.float64).reshape(-1)[:2]
        if geometry_available
        else None
    )
    resolution = float(map_resolution_m) if geometry_available else None
    for plane in values:
        plane_support = np.abs(plane) > 1e-12
        plane_coordinates = np.argwhere(plane_support)
        plane_areas.append(int(np.sum(plane_support)))
        plane_norms.append(float(np.linalg.norm(plane)))
        if plane_coordinates.size:
            lower = plane_coordinates.min(axis=0)
            upper = plane_coordinates.max(axis=0) + 1
            plane_extent = plane_coordinates.max(axis=0) - plane_coordinates.min(axis=0) + 1
            plane_bbox_shapes.append(tuple(int(value) for value in plane_extent))
            cropped = plane[lower[0] : upper[0], lower[1] : upper[1]]
            plane_canonical_patterns.append(
                cropped / max(float(np.max(np.abs(cropped))), np.finfo(np.float64).eps)
            )
            if geometry_available:
                centroid_rc = plane_coordinates.mean(axis=0)
                centroid_xy = origin + resolution * np.asarray(
                    [centroid_rc[1] + 0.5, centroid_rc[0] + 0.5]
                )
                receiver_plane_distances.append(
                    float(np.linalg.norm(receiver_xy - centroid_xy))
                )
            else:
                receiver_plane_distances.append(0.0)
        else:
            plane_bbox_shapes.append((0, 0))
            plane_canonical_patterns.append(np.zeros((0, 0), dtype=np.float64))
            receiver_plane_distances.append(0.0)
    return {
        "family": family,
        "area": int(np.sum(support)),
        "norm": float(np.linalg.norm(values)),
        "bbox_shape": bbox_shape,
        "plane_areas": tuple(plane_areas),
        "plane_norms": tuple(plane_norms),
        "plane_bbox_shapes": tuple(plane_bbox_shapes),
        "plane_canonical_patterns": tuple(plane_canonical_patterns),
        "receiver_plane_distances_m": (
            tuple(receiver_plane_distances) if geometry_available else None
        ),
    }


def _shortcut_warnings(records, target, predictions, config):
    routes = np.asarray([row["route"] for row in records])
    active = routes == 2
    copy = normalized_mse(target[active], predictions["copy"][active])
    variant = normalized_mse(target[active], predictions["variant_id_only"][active])
    csi_only = normalized_mse(target[active], predictions["csi_only"][active])
    map_edit_only = normalized_mse(target[active], predictions["map_edit_only"][active])
    no_x = normalized_mse(target[active], predictions["no_x"][active])
    oracle = normalized_mse(target[active], predictions["oracle_x"][active])
    threshold = float(config["shortcut_relative_improvement_max"])
    variant_signal = bool(_relative_improvement(variant, copy) > threshold)
    csi_signal = bool(_relative_improvement(csi_only, copy) > threshold)
    edit_signal = bool(_relative_improvement(map_edit_only, copy) > threshold)
    map_hurts = bool(_relative_improvement(csi_only, no_x) > threshold)
    return {
        "schema_version": "csi-pairs-v6-shortcut-audit-v1",
        "inherited_failures_remain_binding": True,
        "checks": {
            "map_feature_hurts": {
                "triggered": map_hurts,
                "csi_only_relative_improvement_vs_no_x": _relative_improvement(csi_only, no_x),
            },
            "variant_id_signal": {
                "triggered": variant_signal,
                "relative_improvement_vs_copy": _relative_improvement(variant, copy),
            },
            "csi_only_shortcut_signal": {
                "triggered": csi_signal,
                "relative_improvement_vs_copy": _relative_improvement(csi_only, copy),
            },
            "map_edit_only_shortcut_signal": {
                "triggered": edit_signal,
                "relative_improvement_vs_copy": _relative_improvement(map_edit_only, copy),
            },
            "oracle_x_not_helpful": {
                "triggered": bool(
                    _relative_improvement(oracle, copy) < float(config["oracle_min_relative_improvement"])
                ),
                "relative_improvement_vs_copy": _relative_improvement(oracle, copy),
            },
            "beam_readout_insensitive_to_copy": {
                "triggered": None,
                "status": "NOT_ASSESSED_WITHOUT_COMMUNICATION_READOUT",
            },
        },
        "shortcut_gate_passed": not any(
            (variant_signal, csi_signal, edit_signal, map_hurts)
        ),
        "interpretation": "NOT_ASSESSED is never PASS; inherited V1 failures are not overwritten by fixture or proxy output.",
    }


def _relative_improvement(method_score: float, baseline_score: float) -> float:
    if not np.isfinite(method_score) or not np.isfinite(baseline_score) or baseline_score <= 1e-15:
        return float("-inf")
    return float((baseline_score - method_score) / baseline_score)


def _method_contract(method: str) -> list[str]:
    contracts = {
        "no_x": ["visible_source_csi_patches", "source_map", "radio_config", "bs_pose", "mask", "query", "typed_signed_edit"],
        "oracle_x": ["visible_source_csi_patches", "source_map", "radio_config", "bs_pose", "mask", "query", "typed_signed_edit", "receiver_position"],
        "no_action": ["visible_source_csi_patches", "source_map", "radio_config", "bs_pose", "mask", "query", "zero_action"],
        "map_edit_only": ["source_map", "radio_config", "bs_pose", "mask", "query", "typed_signed_edit"],
        "csi_only": ["visible_source_csi_patches", "radio_config", "bs_pose", "mask", "query"],
        "variant_id_only": ["source_bits", "target_bits"],
    }
    return contracts[method]
