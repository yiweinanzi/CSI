from __future__ import annotations

import ctypes
import gc
import hashlib
from pathlib import Path

import numpy as np

from .formal_dataset import FormalDataset
from .formal_evidence import (
    FACTORIAL_SCHEMA,
    bind_rows,
    complete_gate_vector,
    evidence_context,
    require_manifested_formal_qualification,
    require_stage_manifested_gate,
)
from .formal_factorial import (
    _canonical_bank_digest,
    _normalized_action,
    _normalized_map,
    _normalized_radio,
    _training_normalization,
)
from .formal_features import multichannel_spatial_features
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
from .formal_metrics import binary_auroc, spearman_correlation
from .formal_model import (
    CSIPairsFormalModel,
    endpoint_per_sample,
    resolve_execution_device,
    tensor_for_module,
    torch,
)
from .formal_probes import (
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)
from .formal_protocol import (
    headline_alignment_edge,
    patchify_csi,
    typed_signed_edit,
    unpatchify_csi,
    zero_typed_edit,
)
from .formal_routing import ROUTE_NAMES, fit_route_normalization, route_dataset
from .formal_statistics import (
    holm_adjust,
    interval_decision,
    paired_cluster_interval,
    paired_sign_flip_test,
)
from .formal_teacher import load_teacher_bundle


def run_formal_evaluation(
    config: dict,
    dataset: FormalDataset,
    output_root: str | Path,
    qualification_gate: dict,
    factorial_gate: dict,
) -> dict:
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        (
            "source_encoder_train",
            "source_probe_train",
            "source_probe_selection",
            "source_final_unseen_bank",
            "target",
        ),
    )
    qualification_gate = require_manifested_formal_qualification(
        qualification_gate,
        config,
        dataset,
        allow_nonscientific_fixture=True,
    )
    root = Path(output_root)
    _validate_factorial_gate(config, dataset, factorial_gate, root / "factorial" / "gate.json")
    output_dir = root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_device = resolve_execution_device(dataset)
    teacher = load_teacher_bundle(
        qualification_gate["teacher_checkpoint"],
        config,
        device=execution_device,
    )
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    checkpoint_index = read_strict_json(root / "factorial" / "checkpoint_index.json")
    checkpoint_rows = _validate_checkpoint_index(
        checkpoint_index, config, dataset, qualification_gate
    )
    cgs_rows = []
    response_rows = []
    probe_contract_rows = []
    response_probe_contract_rows = []
    route_rows = []
    effect_bin_rows = []
    shortcut_rows = []
    compatibility_effect_rows = []
    response_effect_rows = []
    for checkpoint_row in checkpoint_rows:
        seed = int(checkpoint_row["seed"])
        arm = str(checkpoint_row["arm"])
        model = _load_model(
            root / "factorial",
            checkpoint_row,
            qualification_gate,
            config,
            dataset,
            device=execution_device,
        )
        probe_train = _compatibility_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            dataset.indices_for_role("source_probe_train"),
        )
        probe_selection = _compatibility_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            dataset.indices_for_role("source_probe_selection"),
        )
        probe, selection_record = fit_select_compatibility_probe(
            probe_train["features"],
            probe_train["labels"],
            probe_selection["features"],
            probe_selection["labels"],
            config,
            seed=seed + 31001,
        )
        probe_contract_rows.append({"seed": seed, "arm": arm, **selection_record})
        evaluation_scenes = np.concatenate(
            (
                dataset.indices_for_role("source_final_unseen_bank"),
                dataset.indices_for_role("target"),
            )
        )
        evaluated = _compatibility_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            evaluation_scenes,
            active_only=False,
        )
        probabilities = predict_binary_probe(probe, evaluated["features"])
        compatibility_effect_rows.extend(
            _compatibility_effect_rows(seed, arm, evaluated, probabilities)
        )
        for bank in sorted(set(evaluated["bank_ids"].tolist())):
            bank_mask = evaluated["bank_ids"] == bank
            mask = bank_mask & (evaluated["routes"] == "active")
            if not np.any(mask):
                raise RuntimeError(f"evaluation bank {bank!r} has no active CGS quartet")
            scene = int(evaluated["scene_indices"][mask][0])
            evaluation_scope = _evaluation_scope(dataset, scene)
            cgs_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "bank_id": bank,
                    "base_map_cluster_id": str(
                        dataset.base_map_cluster_ids[int(evaluated["scene_indices"][mask][0])]
                    ),
                    "canonical_base_map_digest": dataset.canonical_base_map_digest(
                        int(evaluated["scene_indices"][mask][0])
                    ),
                    "canonical_bank_digest": _canonical_bank_digest(
                        dataset, int(evaluated["scene_indices"][mask][0])
                    ),
                    "city_id": str(evaluated["city_ids"][mask][0]),
                    "evaluation_scope": evaluation_scope,
                    "route": "active",
                    "probe_family": selection_record["selected_family"],
                    "cgs_auroc": binary_auroc(evaluated["labels"][mask], probabilities[mask]),
                    "native_energy_auroc": binary_auroc(
                        evaluated["labels"][mask], evaluated["native_training_scores"][mask]
                    ),
                    "native_audit_hold_auroc": binary_auroc(
                        evaluated["labels"][mask], evaluated["native_scores"][mask]
                    ),
                    "native_probe_spearman": spearman_correlation(
                        evaluated["native_training_scores"][mask], probabilities[mask]
                    ),
                    "native_audit_hold_probe_spearman": spearman_correlation(
                        evaluated["native_scores"][mask], probabilities[mask]
                    ),
                    "n": int(np.sum(mask)),
                }
            )
            shortcut_rows.extend(
                _alignment_shortcut_rows(
                    seed,
                    arm,
                    bank,
                    evaluation_scope,
                    probe_train,
                    probe_selection,
                    evaluated,
                    mask,
                    config,
                )
            )
            effect_bin_rows.extend(
                _active_effect_bin_rows(
                    seed,
                    arm,
                    bank,
                    str(evaluated["city_ids"][mask][0]),
                    evaluation_scope,
                    probabilities[mask],
                    evaluated["labels"][mask],
                    evaluated["pair_ids"][mask],
                    evaluated["physical_distances"][mask],
                )
            )
            for route in ROUTE_NAMES.tolist():
                route_mask = bank_mask & (evaluated["routes"] == route)
                if not np.any(route_mask):
                    route_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "bank_id": bank,
                            "base_map_cluster_id": str(
                                dataset.base_map_cluster_ids[
                                    int(evaluated["scene_indices"][bank_mask][0])
                                ]
                            ),
                            "canonical_base_map_digest": dataset.canonical_base_map_digest(
                                int(evaluated["scene_indices"][bank_mask][0])
                            ),
                            "canonical_bank_digest": _canonical_bank_digest(
                                dataset, int(evaluated["scene_indices"][bank_mask][0])
                            ),
                            "city_id": str(evaluated["city_ids"][bank_mask][0]),
                            "evaluation_scope": evaluation_scope,
                            "route": route,
                            "condition_status": "MISSING",
                            "pair_count": 0,
                            "matched_minus_alternative_mean": None,
                            "matched_minus_alternative_median": None,
                            "absolute_difference_p90": None,
                            "overclassification_rate": None,
                        }
                    )
                    continue
                differences = _paired_score_differences(
                    probabilities[route_mask],
                    evaluated["labels"][route_mask],
                    evaluated["pair_ids"][route_mask],
                )
                margin = float(config["evaluation"]["null_score_equivalence_margin"])
                route_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "bank_id": bank,
                        "base_map_cluster_id": str(
                            dataset.base_map_cluster_ids[
                                int(evaluated["scene_indices"][route_mask][0])
                            ]
                        ),
                        "canonical_base_map_digest": dataset.canonical_base_map_digest(
                            int(evaluated["scene_indices"][route_mask][0])
                        ),
                        "canonical_bank_digest": _canonical_bank_digest(
                            dataset, int(evaluated["scene_indices"][route_mask][0])
                        ),
                        "city_id": str(evaluated["city_ids"][route_mask][0]),
                        "evaluation_scope": evaluation_scope,
                        "route": route,
                        "condition_status": "ASSESSED",
                        "pair_count": int(differences.size),
                        "matched_minus_alternative_mean": float(np.mean(differences)),
                        "matched_minus_alternative_median": float(np.median(differences)),
                        "absolute_difference_p90": float(np.percentile(np.abs(differences), 90)),
                        "overclassification_rate": (
                            float(np.mean(np.abs(differences) > margin)) if route == "null" else None
                        ),
                    }
                )

        response_train = _response_probe_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            dataset.indices_for_role("source_probe_train"),
            active_only=False,
        )
        response_probe = fit_action_response_probe(
            response_train["features"],
            response_train["targets"] - response_train["source_targets"],
            config,
            seed=seed + 32001,
            zero_action_x=response_train["no_action_features"],
        )
        response_probe_contract_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "probe": "main_masked_state_map_action_query",
                "steps": int(config["evaluation"]["probe_steps"]),
                "hidden_dim": int(config["evaluation"]["probe_hidden_dim"]),
            }
        )
        variant_probes = {}
        contrast_variants = {"without_map", "edit_only", "oracle_x"}
        for offset, name in enumerate(
            ("without_map", "edit_only", "csi_only", "oracle_x"), start=1
        ):
            variant_probes[name] = fit_action_response_probe(
                response_train[f"{name}_features"],
                response_train["targets"] - response_train["source_targets"],
                config,
                seed=seed + 32001 + offset,
                zero_action_x=(
                    response_train[f"{name}_zero_action_features"]
                    if name in contrast_variants
                    else None
                ),
            )
            response_probe_contract_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "probe": name,
                    "steps": int(config["evaluation"]["probe_steps"]),
                    "hidden_dim": int(config["evaluation"]["probe_hidden_dim"]),
                }
            )
        response_eval = _response_probe_dataset(
            model,
            dataset,
            teacher,
            config,
            normalization,
            evaluation_scenes,
            active_only=False,
        )
        source_target = response_eval["source_targets"]
        probe_prediction = source_target + predict_response_probe(
            response_probe,
            response_eval["features"],
            response_eval["no_action_features"],
        )
        action_swap_prediction = source_target + predict_response_probe(
            response_probe,
            response_eval["action_swap_features"],
            response_eval["no_action_features"],
        )
        no_action_prediction = source_target + predict_response_probe(
            response_probe,
            response_eval["no_action_features"],
            response_eval["no_action_features"],
        )
        variant_predictions = {}
        for name, probe in variant_probes.items():
            zero_features = (
                response_eval[f"{name}_zero_action_features"]
                if name in contrast_variants
                else None
            )
            variant_predictions[name] = source_target + predict_response_probe(
                probe,
                response_eval[f"{name}_features"],
                zero_features,
            )
        response_effect_rows.extend(
            _response_effect_rows(
                seed,
                arm,
                response_eval,
                probe_prediction,
                action_swap_prediction,
                no_action_prediction,
            )
        )
        for bank in sorted(set(response_eval["bank_ids"].tolist())):
            mask = (response_eval["bank_ids"] == bank) & (response_eval["routes"] == "active")
            if not np.any(mask):
                raise RuntimeError(f"evaluation bank {bank!r} has no active response patches")
            numerator = np.sum((probe_prediction[mask] - response_eval["targets"][mask]) ** 2)
            denominator = max(float(np.sum(response_eval["targets"][mask] ** 2)), 1e-12)
            copy_numerator = np.sum(
                (response_eval["source_targets"][mask] - response_eval["targets"][mask]) ** 2
            )
            swap_numerator = np.sum(
                (
                    action_swap_prediction[
                        mask
                        & (response_eval["wrong_action_match_status"] == "exact")
                    ]
                    - response_eval["targets"][
                        mask
                        & (response_eval["wrong_action_match_status"] == "exact")
                    ]
                )
                ** 2
            )
            exact_swap_mask = mask & (
                response_eval["wrong_action_match_status"] == "exact"
            )
            exact_swap_denominator = max(
                float(np.sum(response_eval["targets"][exact_swap_mask] ** 2)),
                1e-12,
            )
            variant_nmse = {
                f"probe_{name}_active_patch_nmse": float(
                    np.sum(
                        (prediction[mask] - response_eval["targets"][mask]) ** 2
                    )
                    / denominator
                )
                for name, prediction in variant_predictions.items()
            }
            native = _native_mask_cover_metrics(
                model,
                dataset,
                teacher,
                config,
                normalization,
                evaluation_scenes,
                bank,
            )
            response_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "bank_id": bank,
                    "base_map_cluster_id": str(
                        dataset.base_map_cluster_ids[
                            int(response_eval["scene_indices"][mask][0])
                        ]
                    ),
                    "canonical_base_map_digest": dataset.canonical_base_map_digest(
                        int(response_eval["scene_indices"][mask][0])
                    ),
                    "canonical_bank_digest": _canonical_bank_digest(
                        dataset, int(response_eval["scene_indices"][mask][0])
                    ),
                    "city_id": str(response_eval["city_ids"][mask][0]),
                    "evaluation_scope": _evaluation_scope(
                        dataset, int(response_eval["scene_indices"][mask][0])
                    ),
                    "unified_response_probe_active_patch_nmse": float(numerator / denominator),
                    "probe_copy_active_patch_nmse": float(copy_numerator / denominator),
                    "unified_response_probe_action_swap_exact_patch_nmse": (
                        float(
                            np.sum(
                                (
                                    probe_prediction[exact_swap_mask]
                                    - response_eval["targets"][exact_swap_mask]
                                )
                                ** 2
                            )
                            / exact_swap_denominator
                        )
                        if np.any(exact_swap_mask)
                        else None
                    ),
                    "probe_action_swap_active_patch_nmse": (
                        float(swap_numerator / exact_swap_denominator)
                        if np.any(exact_swap_mask)
                        else None
                    ),
                    "probe_action_swap_exact_count": int(np.sum(exact_swap_mask)),
                    "probe_action_swap_fallback_count": int(
                        np.sum(
                            mask
                            & (response_eval["wrong_action_match_status"] == "fallback")
                        )
                    ),
                    "probe_action_swap_failed_count": int(
                        np.sum(
                            mask
                            & (response_eval["wrong_action_match_status"] == "failed")
                        )
                    ),
                    "probe_action_swap_exact_fraction": float(
                        np.mean(
                            response_eval["wrong_action_match_status"][mask] == "exact"
                        )
                    ),
                    **variant_nmse,
                    **native,
                    "n_active_patches": int(np.sum(mask)),
                }
            )
        del (
            model,
            probe,
            probe_train,
            probe_selection,
            evaluated,
            probabilities,
            response_train,
            response_probe,
            variant_probes,
            response_eval,
            probe_prediction,
            action_swap_prediction,
            no_action_prediction,
            variant_predictions,
        )
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass
    write_csv(output_dir / "compatibility_probe_contract.csv", bind_rows(probe_contract_rows, evidence))
    write_csv(
        output_dir / "response_probe_contract.csv",
        bind_rows(response_probe_contract_rows, evidence),
    )
    write_csv(output_dir / "cgs_per_bank.csv", bind_rows(cgs_rows, evidence))
    write_csv(
        output_dir / "alignment_shortcut_baselines.csv",
        bind_rows(shortcut_rows, evidence),
    )
    write_csv(output_dir / "compatibility_route_distributions.csv", bind_rows(route_rows, evidence))
    write_csv(output_dir / "cgs_active_effect_bins.csv", bind_rows(effect_bin_rows, evidence))
    write_csv(
        output_dir / "compatibility_pair_effects.csv",
        bind_rows(compatibility_effect_rows, evidence),
    )
    write_csv(
        output_dir / "response_pair_effects.csv",
        bind_rows(response_effect_rows, evidence),
    )
    write_csv(output_dir / "response_per_bank.csv", bind_rows(response_rows, evidence))
    gate = _evaluation_gate(
        config,
        dataset,
        cgs_rows,
        route_rows,
        effect_bin_rows,
        response_rows,
        shortcut_rows,
        factorial_gate,
        evidence,
        qualification_gate_sha256=sha256_file(
            root / "qualification" / "gate.json"
        ),
        factorial_gate_sha256=sha256_file(root / "factorial" / "gate.json"),
    )
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


def _validate_factorial_gate(config, dataset, gate, gate_path):
    expected = evidence_context(config, dataset, str(gate.get("scientific_use", "")))
    if gate.get("schema_version") != FACTORIAL_SCHEMA:
        raise RuntimeError("evaluation requires a V6 factorial gate")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if gate.get(key) != expected[key]:
            raise RuntimeError(f"factorial gate {key} mismatch")
    require_stage_manifested_gate(
        gate_path,
        gate,
        config,
        dataset,
        schema_version=FACTORIAL_SCHEMA,
    )


def _load_model(
    factorial_root,
    checkpoint_row,
    qualification_gate,
    config,
    dataset,
    *,
    device="cpu",
):
    root = Path(factorial_root).resolve()
    path = (root / checkpoint_row["path"]).resolve()
    if root not in path.parents:
        raise RuntimeError("checkpoint index path escapes the factorial directory")
    if sha256_file(path) != checkpoint_row["sha256"]:
        raise RuntimeError("factorial checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "csi-pairs-formal-checkpoint-v2.1-v6":
        raise RuntimeError("checkpoint schema is not V6-compatible")
    expected = evidence_context(config, dataset, str(payload.get("scientific_use", "")))
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if payload.get(key) != expected[key]:
            raise RuntimeError(f"factorial checkpoint {key} mismatch")
    if payload.get("teacher_checkpoint_sha256") != qualification_gate["teacher_checkpoint_sha256"]:
        raise RuntimeError("factorial checkpoint uses a different frozen teacher")
    if payload.get("seed") != int(checkpoint_row["seed"]) or payload.get("arm") != checkpoint_row["arm"]:
        raise RuntimeError("factorial checkpoint seed/arm identity mismatch")
    model = CSIPairsFormalModel(**payload["model_spec"]).to(torch.device(device))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _validate_checkpoint_index(index, config, dataset, qualification_gate):
    if not isinstance(index, dict) or index.get("schema_version") != "csi-pairs-formal-checkpoint-index-v2.1-v6":
        raise RuntimeError("checkpoint index schema mismatch")
    expected = evidence_context(config, dataset, str(index.get("scientific_use", "")))
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if index.get(key) != expected[key]:
            raise RuntimeError(f"checkpoint index {key} mismatch")
    rows = index.get("checkpoints")
    expected_cells = {(int(seed), arm) for seed in config["seeds"] for arm in config["factorial"]["arms"]}
    actual_cells = (
        {(int(row["seed"]), str(row["arm"])) for row in rows}
        if isinstance(rows, list)
        else set()
    )
    if actual_cells != expected_cells or len(rows) != len(expected_cells):
        raise RuntimeError("checkpoint index must contain every seed/arm exactly once")
    for row in rows:
        if row.get("teacher_checkpoint_sha256") != qualification_gate["teacher_checkpoint_sha256"]:
            raise RuntimeError("checkpoint index teacher hash mismatch")
    return rows


def _compatibility_dataset(
    model, dataset, teacher, config, normalization, scenes, active_only=True
):
    route_norm = fit_route_normalization(dataset, teacher)
    routed = route_dataset(dataset, teacher, config, scenes, normalization=route_norm)
    features = []
    without_map_features = []
    labels = []
    native_scores = []
    native_training_scores = []
    bank_ids = []
    city_ids = []
    base_map_cluster_ids = []
    canonical_base_map_digests = []
    canonical_bank_digests = []
    routes = []
    pair_ids = []
    physical_distances = []
    scene_indices = []
    edge_sources = []
    edge_targets = []
    positions = []
    csi_worlds = []
    supplied_worlds = []
    csi_only_shortcut_scores = []
    map_only_shortcut_scores = []
    constant_shortcut_features = []
    csi_only_shortcut_features = []
    map_only_shortcut_features = []
    scene_id_only_shortcut_features = []
    edit_status_xor_shortcut_features = []
    variant_id_match_shortcut_features = []
    zero = zero_typed_edit((1,), dataset.maps.shape[-1], int(dataset.metadata["assets"]["material_category_count"]))
    zero_tensor = tensor_for_module(
        model,
        _normalized_action(normalization, zero),
        dtype=torch.float32,
    )
    for scene_value in scenes:
        scene = int(scene_value)
        for edge in dataset.directed_edges(scene):
            if edge.source_world >= edge.target_world:
                continue
            if not headline_alignment_edge(dataset, scene, edge):
                continue
            for position in _eligible_evaluation_positions(dataset, scene):
                key = (scene, edge.source_world, edge.target_world, position)
                route_code = routed.alignment_route[key]
                if active_only and route_code != 2:
                    continue
                for csi_world, matched_map, alternative_map in (
                    (edge.source_world, edge.source_world, edge.target_world),
                    (edge.target_world, edge.target_world, edge.source_world),
                ):
                    for supplied_world, label in ((matched_map, 1), (alternative_map, 0)):
                        patches = _normalized_scene_patches(
                            dataset, normalization, teacher.patch_spec, scene, csi_world, position
                        )
                        maps = _normalized_map(normalization, dataset.maps[scene, supplied_world])[None, ...]
                        radio = _normalized_radio(
                            normalization, dataset.radio_config[scene], dataset.bs_pose[scene]
                        )[None, ...]
                        with torch.no_grad():
                            patch_tensor = tensor_for_module(
                                model, patches[None, ...], dtype=torch.float32
                            )
                            map_tensor = tensor_for_module(model, maps, dtype=torch.float32)
                            radio_tensor = tensor_for_module(model, radio, dtype=torch.float32)
                            representation, training_score = _masked_alignment_state_and_score(
                                model,
                                patch_tensor,
                                map_tensor,
                                radio_tensor,
                                zero_tensor,
                                tuple(
                                    entry
                                    for entry in teacher.mask_bank
                                    if entry.mode == "random_75"
                                ),
                                routed.teacher_latent[scene][csi_world, position],
                                patches,
                                normalization,
                            )
                            no_map_representation, _ = _masked_alignment_state_and_score(
                                model,
                                patch_tensor,
                                torch.zeros_like(map_tensor),
                                radio_tensor,
                                zero_tensor,
                                tuple(
                                    entry
                                    for entry in teacher.mask_bank
                                    if entry.mode == "random_75"
                                ),
                                routed.teacher_latent[scene][csi_world, position],
                                patches,
                                normalization,
                            )
                            _, audit_score = _masked_alignment_state_and_score(
                                model,
                                patch_tensor,
                                map_tensor,
                                radio_tensor,
                                zero_tensor,
                                tuple(
                                    entry
                                    for entry in teacher.audit_mask_bank
                                    if entry.mode == "random_75"
                                ),
                                routed.teacher_latent[scene][csi_world, position],
                                patches,
                                normalization,
                            )
                        features.append(representation)
                        without_map_features.append(no_map_representation)
                        labels.append(label)
                        native_scores.append(audit_score)
                        native_training_scores.append(training_score)
                        bank_ids.append(str(dataset.bank_ids[scene]))
                        city_ids.append(str(dataset.city_ids[scene]))
                        base_map_cluster_ids.append(str(dataset.base_map_cluster_ids[scene]))
                        canonical_base_map_digests.append(
                            dataset.canonical_base_map_digest(scene)
                        )
                        canonical_bank_digests.append(
                            _canonical_bank_digest(dataset, scene)
                        )
                        routes.append(str(ROUTE_NAMES[route_code]))
                        pair_ids.append(
                            f"{dataset.bank_ids[scene]}:{edge.source_world}:{edge.target_world}:"
                            f"{position}:{csi_world}"
                        )
                        physical_distances.append(routed.alignment_distances[key][0])
                        scene_indices.append(scene)
                        edge_sources.append(edge.source_world)
                        edge_targets.append(edge.target_world)
                        positions.append(position)
                        csi_worlds.append(csi_world)
                        supplied_worlds.append(supplied_world)
                        csi_only_shortcut_scores.append(float(np.linalg.norm(patches)))
                        map_only_shortcut_scores.append(
                            float(np.linalg.norm(dataset.maps[scene, supplied_world]))
                        )
                        scene_features, edit_features, variant_features = (
                            _shortcut_metadata_features(
                                dataset,
                                scene,
                                csi_world,
                                supplied_world,
                            )
                        )
                        constant_shortcut_features.append(np.zeros(1, dtype=np.float64))
                        csi_only_shortcut_features.append(
                            np.concatenate((patches.reshape(-1), radio.reshape(-1)))
                        )
                        map_only_shortcut_features.append(
                            np.concatenate((maps.reshape(-1), radio.reshape(-1)))
                        )
                        scene_id_only_shortcut_features.append(
                            scene_features
                        )
                        edit_status_xor_shortcut_features.append(
                            edit_features
                        )
                        variant_id_match_shortcut_features.append(
                            variant_features
                        )
    if not features:
        raise RuntimeError("compatibility probe dataset has no active quartets")
    return {
        "features": np.asarray(features, dtype=np.float64),
        "without_map_features": np.asarray(
            without_map_features, dtype=np.float64
        ),
        "labels": np.asarray(labels, dtype=np.int64),
        "native_scores": np.asarray(native_scores, dtype=np.float64),
        "native_training_scores": np.asarray(native_training_scores, dtype=np.float64),
        "bank_ids": np.asarray(bank_ids),
        "city_ids": np.asarray(city_ids),
        "base_map_cluster_ids": np.asarray(base_map_cluster_ids),
        "canonical_base_map_digests": np.asarray(canonical_base_map_digests),
        "canonical_bank_digests": np.asarray(canonical_bank_digests),
        "routes": np.asarray(routes),
        "pair_ids": np.asarray(pair_ids),
        "physical_distances": np.asarray(physical_distances, dtype=np.float64),
        "scene_indices": np.asarray(scene_indices, dtype=np.int64),
        "edge_sources": np.asarray(edge_sources, dtype=np.int64),
        "edge_targets": np.asarray(edge_targets, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.int64),
        "csi_worlds": np.asarray(csi_worlds, dtype=np.int64),
        "supplied_worlds": np.asarray(supplied_worlds, dtype=np.int64),
        "constant_shortcut_scores": np.zeros(len(labels), dtype=np.float64),
        "csi_only_shortcut_scores": np.asarray(csi_only_shortcut_scores, dtype=np.float64),
        "map_only_shortcut_scores": np.asarray(map_only_shortcut_scores, dtype=np.float64),
        "constant_shortcut_features": np.asarray(constant_shortcut_features),
        "csi_only_shortcut_features": np.asarray(csi_only_shortcut_features),
        "map_only_shortcut_features": np.asarray(map_only_shortcut_features),
        "scene_id_only_shortcut_features": np.asarray(scene_id_only_shortcut_features),
        "edit_status_xor_shortcut_features": np.asarray(
            edit_status_xor_shortcut_features
        ),
        "variant_id_match_shortcut_features": np.asarray(
            variant_id_match_shortcut_features
        ),
        "edge_scope": "non-natural-incident direct edits only",
    }


def _shortcut_metadata_features(dataset, scene, csi_world, supplied_world):
    """Return separate metadata tokens; never construct the match label itself."""
    bank = str(dataset.bank_ids[int(scene)])
    source_bits = np.asarray(dataset.world_bits[int(csi_world)], dtype=np.float64).ravel()
    supplied_bits = np.asarray(
        dataset.world_bits[int(supplied_world)], dtype=np.float64
    ).ravel()
    natural = int(dataset.natural_world_index[int(scene)])
    scene_features = _stable_token_features(f"bank:{bank}")
    edit_features = np.concatenate(
        (
            source_bits,
            supplied_bits,
            np.asarray(
                [
                    float(int(csi_world) == natural),
                    float(int(supplied_world) == natural),
                ],
                dtype=np.float64,
            ),
        )
    )
    variant_features = np.concatenate(
        (
            _stable_token_features(f"bank:{bank}:world:{int(csi_world)}"),
            _stable_token_features(f"bank:{bank}:world:{int(supplied_world)}"),
        )
    )
    return scene_features, edit_features, variant_features


def _stable_token_features(value, width=16):
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    values = np.frombuffer(digest, dtype=np.uint8)[: int(width)].astype(np.float64)
    return values / 127.5 - 1.0


def _masked_alignment_state_and_score(
    model,
    patch_tensor,
    map_tensor,
    radio_tensor,
    zero_tensor,
    mask_bank,
    latent_targets,
    physical_targets,
    normalization,
):
    representations = []
    errors = []
    for entry in mask_bank:
        visible = patch_tensor.clone()
        visible[:, entry.mask] = 0.0
        masks = tensor_for_module(model, entry.mask[None, :], dtype=torch.bool)
        state = model.state(visible, map_tensor, radio_tensor, masks)
        query = tensor_for_module(model, [entry.query], dtype=torch.long)
        prediction_z, prediction_y = model.predict(state, zero_tensor, query)
        target_z = (
            latent_targets[entry.query] - normalization.latent_mean
        ) / normalization.latent_scale
        errors.append(
            endpoint_per_sample(
                prediction_z,
                prediction_y,
                tensor_for_module(model, target_z[None, :], dtype=torch.float32),
                tensor_for_module(
                    model,
                    physical_targets[entry.query][None, :],
                    dtype=torch.float32,
                ),
                1.0,
            )[0]
        )
        representations.append(state[0].mean(dim=0))
    if {int(entry.query) for entry in mask_bank} != set(range(patch_tensor.shape[1])):
        raise RuntimeError("alignment mask bank must cover every query exactly once")
    return (
        torch.mean(torch.stack(representations), dim=0).cpu().numpy(),
        -float(torch.mean(torch.stack(errors)).cpu()),
    )


def _eligible_evaluation_positions(dataset, scene):
    if str(dataset.scene_roles[scene]) == "target":
        selected = np.flatnonzero(dataset.position_roles[scene] == "query")
        if selected.size == 0:
            raise RuntimeError("target evaluation has no query positions after support exclusion")
        return selected.astype(np.int64)
    return np.arange(dataset.position_count, dtype=np.int64)


def _evaluation_scope(dataset, scene):
    role = str(dataset.scene_roles[int(scene)])
    if role == "source_final_unseen_bank":
        return role
    if role == "target":
        return f"target:{dataset.city_ids[int(scene)]}"
    raise RuntimeError(f"unsupported evaluation scene role: {role}")


def _paired_score_differences(scores, labels, pair_ids):
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    identifiers = np.asarray(pair_ids).astype(str)
    differences = []
    for pair_id in np.unique(identifiers):
        mask = identifiers == pair_id
        if int(np.sum(mask)) != 2 or set(targets[mask].tolist()) != {0, 1}:
            raise RuntimeError("compatibility pair must contain one matched and one alternative score")
        differences.append(float(values[mask & (targets == 1)][0] - values[mask & (targets == 0)][0]))
    if not differences:
        raise RuntimeError("compatibility route has no complete paired scores")
    return np.asarray(differences, dtype=np.float64)


def _active_effect_bin_rows(
    seed, arm, bank, city, evaluation_scope, scores, labels, pair_ids, distances
):
    identifiers = np.asarray(pair_ids).astype(str)
    pair_order = np.unique(identifiers)
    if pair_order.size < 4:
        raise RuntimeError("active CGS effect-bin report requires at least four paired units per bank")
    pair_distance = np.asarray(
        [float(np.mean(np.asarray(distances)[identifiers == pair])) for pair in pair_order]
    )
    ranked = np.argsort(np.argsort(pair_distance, kind="stable"), kind="stable")
    bins = np.minimum(3, (4 * ranked) // pair_order.size)
    rows = []
    for bin_index, label in enumerate(("low", "medium_low", "medium_high", "high")):
        selected_pairs = pair_order[bins == bin_index]
        selected = np.isin(identifiers, selected_pairs)
        if not np.any(selected):
            raise RuntimeError("active CGS effect bin is empty")
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "bank_id": str(bank),
                "city_id": str(city),
                "evaluation_scope": str(evaluation_scope),
                "effect_bin": label,
                "pair_count": int(selected_pairs.size),
                "physical_distance_min": float(np.min(np.asarray(distances)[selected])),
                "physical_distance_max": float(np.max(np.asarray(distances)[selected])),
                "cgs_auroc": binary_auroc(np.asarray(labels)[selected], np.asarray(scores)[selected]),
            }
        )
    return rows


def _alignment_shortcut_rows(
    seed,
    arm,
    bank,
    evaluation_scope,
    source_train,
    source_selection,
    evaluated,
    mask,
    config,
):
    definitions = {
        "constant": ("constant_shortcut_features", "legal_no_input_control"),
        "csi_only": ("csi_only_shortcut_features", "legal_single_modality_control"),
        "map_only": ("map_only_shortcut_features", "legal_single_modality_control"),
        "scene_id_only": (
            "scene_id_only_shortcut_features",
            "forbidden_bank_identity_probe",
        ),
        "edit_status_xor": (
            "edit_status_xor_shortcut_features",
            "forbidden_separate_edit_metadata_probe",
        ),
        "variant_id_matcher": (
            "variant_id_match_shortcut_features",
            "forbidden_separate_variant_token_probe",
        ),
    }
    rows = []
    for offset, (name, (field, input_class)) in enumerate(definitions.items()):
        if name == "constant":
            source_scores = np.zeros(len(source_train["labels"]), dtype=np.float64)
            unseen_scores = np.zeros(int(np.sum(mask)), dtype=np.float64)
            selected_family = "constant"
        else:
            probe, selection = fit_select_compatibility_probe(
                source_train[field],
                source_train["labels"],
                source_selection[field],
                source_selection["labels"],
                config,
                seed=int(seed) + 33001 + offset,
            )
            source_scores = predict_binary_probe(probe, source_train[field])
            unseen_features = evaluated[field][mask]
            unseen_scores = predict_binary_probe(probe, unseen_features)
            selected_family = selection["selected_family"]
        source_auroc = binary_auroc(source_train["labels"], source_scores)
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "bank_id": str(bank),
                "base_map_cluster_id": str(evaluated["base_map_cluster_ids"][mask][0]),
                "canonical_base_map_digest": str(
                    evaluated["canonical_base_map_digests"][mask][0]
                ),
                "canonical_bank_digest": str(
                    evaluated["canonical_bank_digests"][mask][0]
                ),
                "city_id": str(evaluated["city_ids"][mask][0]),
                "evaluation_scope": str(evaluation_scope),
                "baseline": name,
                "input_class": input_class,
                "source_train_auroc": source_auroc,
                "unseen_bank_auroc": binary_auroc(
                    evaluated["labels"][mask], unseen_scores
                ),
                "auroc": binary_auroc(evaluated["labels"][mask], unseen_scores),
                "identity_token_contract": (
                    {
                        "scene_id_only": "stable_hashed_bank_token_no_label_feature",
                        "edit_status_xor": "separate_world_bits_and_natural_flags_no_xor",
                        "variant_id_matcher": "separate_stable_hashed_variant_tokens_no_match_or_unk_override",
                    }.get(name, "not_applicable")
                ),
                "probe_family": selected_family,
                "fit_role": "source_probe_train",
                "selection_role": "source_probe_selection",
                "n": int(np.sum(mask)),
            }
        )
    return rows


def _compatibility_effect_rows(seed, arm, evaluated, probabilities):
    rows = []
    pair_ids = evaluated["pair_ids"]
    for pair_id in np.unique(pair_ids):
        mask = pair_ids == pair_id
        difference = _paired_score_differences(
            probabilities[mask], evaluated["labels"][mask], pair_ids[mask]
        )[0]
        first = int(np.flatnonzero(mask)[0])
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "pair_id": str(pair_id),
                "scene_index": int(evaluated["scene_indices"][first]),
                "bank_id": str(evaluated["bank_ids"][first]),
                "base_map_cluster_id": str(
                    evaluated.get("base_map_cluster_ids", evaluated["bank_ids"])[first]
                ),
                "canonical_base_map_digest": str(
                    evaluated["canonical_base_map_digests"][first]
                ),
                "canonical_bank_digest": str(
                    evaluated["canonical_bank_digests"][first]
                ),
                "city_id": str(evaluated["city_ids"][first]),
                "source_world": int(evaluated["csi_worlds"][first]),
                "target_world": int(
                    evaluated["edge_targets"][first]
                    if evaluated["csi_worlds"][first] == evaluated["edge_sources"][first]
                    else evaluated["edge_sources"][first]
                ),
                "position_index": int(evaluated["positions"][first]),
                "route": str(evaluated["routes"][first]),
                "physical_distance": float(evaluated["physical_distances"][first]),
                "matched_minus_alternative": float(difference),
            }
        )
    return rows


def _unified_response_route_is_active(routed, key) -> bool:
    """Select unified query probes with the query-level Response estimand."""
    return routed.response_route[key] == 2


def _native_response_route_is_active(routed, key) -> bool:
    """Select native full-channel transitions with the Alignment estimand."""
    return routed.alignment_route[key] == 2


def _response_probe_dataset(model, dataset, teacher, config, normalization, scenes, active_only):
    from .formal_qualification import _select_wrong_action

    route_norm = fit_route_normalization(dataset, teacher)
    routed = route_dataset(dataset, teacher, config, scenes, normalization=route_norm)
    features = []
    action_swap_features = []
    no_action_features = []
    without_map_features = []
    map_swap_features = []
    edit_only_features = []
    csi_only_features = []
    oracle_x_features = []
    without_map_zero_action_features = []
    edit_only_zero_action_features = []
    oracle_x_zero_action_features = []
    targets = []
    bank_ids = []
    city_ids = []
    source_targets = []
    routes = []
    pair_ids = []
    scene_indices = []
    source_worlds = []
    target_worlds = []
    positions = []
    queries = []
    cluster_ids = []
    canonical_cluster_ids = []
    canonical_bank_ids = []
    wrong_action_match_statuses = []
    wrong_action_worlds = []
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    for scene_value in scenes:
        scene = int(scene_value)
        for edge in dataset.directed_edges(scene):
            action = typed_signed_edit(
                dataset.maps[scene, edge.source_world],
                dataset.maps[scene, edge.target_world],
                dataset.map_channel_names,
                material_categories,
            )
            action_features = multichannel_spatial_features(
                _normalized_action(normalization, action)
            )
            no_action_vector = np.zeros_like(action_features)
            for position in _eligible_evaluation_positions(dataset, scene):
                swap_action, swap_status, swap_world = _select_wrong_action(
                    dataset,
                    scene,
                    edge.source_world,
                    edge.bit_index,
                    action,
                    material_categories,
                    receiver_position=dataset.positions[scene, position],
                )
                swap_action_features = multichannel_spatial_features(
                    _normalized_action(normalization, swap_action)
                )
                source_patches = _normalized_scene_patches(
                    dataset,
                    normalization,
                    teacher.patch_spec,
                    scene,
                    edge.source_world,
                    int(position),
                )
                maps = _normalized_map(
                    normalization, dataset.maps[scene, edge.source_world]
                )[None, ...]
                swapped_maps = _normalized_map(
                    normalization, dataset.maps[scene, edge.target_world]
                )[None, ...]
                radio = _normalized_radio(
                    normalization, dataset.radio_config[scene], dataset.bs_pose[scene]
                )[None, ...]
                for query in range(teacher.patch_spec.patch_count):
                    key = (scene, edge.source_world, edge.target_world, position, query)
                    if active_only and not _unified_response_route_is_active(routed, key):
                        continue
                    query_onehot = np.zeros(teacher.patch_spec.patch_count)
                    query_onehot[query] = 1.0
                    entry = next(item for item in teacher.mask_bank if item.query == query)
                    visible = source_patches.copy()
                    visible[entry.mask] = 0.0
                    with torch.no_grad():
                        state = model.state(
                            tensor_for_module(model, visible[None, ...], dtype=torch.float32),
                            tensor_for_module(model, maps, dtype=torch.float32),
                            tensor_for_module(model, radio, dtype=torch.float32),
                            tensor_for_module(model, entry.mask[None, :], dtype=torch.bool),
                        )[0, query].cpu().numpy()
                        no_map_state = model.state(
                            tensor_for_module(model, visible[None, ...], dtype=torch.float32),
                            torch.zeros_like(tensor_for_module(model, maps, dtype=torch.float32)),
                            tensor_for_module(model, radio, dtype=torch.float32),
                            tensor_for_module(model, entry.mask[None, :], dtype=torch.bool),
                        )[0, query].cpu().numpy()
                        map_swap_state = model.state(
                            tensor_for_module(model, visible[None, ...], dtype=torch.float32),
                            tensor_for_module(model, swapped_maps, dtype=torch.float32),
                            tensor_for_module(model, radio, dtype=torch.float32),
                            tensor_for_module(model, entry.mask[None, :], dtype=torch.bool),
                        )[0, query].cpu().numpy()
                        map_only_state = model.state(
                            torch.zeros_like(
                                tensor_for_module(model, visible[None, ...], dtype=torch.float32)
                            ),
                            tensor_for_module(model, maps, dtype=torch.float32),
                            tensor_for_module(model, radio, dtype=torch.float32),
                            tensor_for_module(model, entry.mask[None, :], dtype=torch.bool),
                        )[0, query].cpu().numpy()
                    features.append(np.concatenate((state, action_features, query_onehot)))
                    action_swap_features.append(
                        np.concatenate((state, swap_action_features, query_onehot))
                    )
                    no_action_features.append(
                        np.concatenate((state, no_action_vector, query_onehot))
                    )
                    without_map_features.append(
                        np.concatenate((no_map_state, action_features, query_onehot))
                    )
                    without_map_zero_action_features.append(
                        np.concatenate((no_map_state, no_action_vector, query_onehot))
                    )
                    map_swap_features.append(
                        np.concatenate((map_swap_state, action_features, query_onehot))
                    )
                    edit_only_features.append(
                        np.concatenate((map_only_state, action_features, query_onehot))
                    )
                    edit_only_zero_action_features.append(
                        np.concatenate((map_only_state, no_action_vector, query_onehot))
                    )
                    csi_only_features.append(
                        np.concatenate((no_map_state, no_action_vector, query_onehot))
                    )
                    normalized_position = (
                        dataset.positions[scene, position] - normalization.position_mean
                    ) / normalization.position_scale
                    oracle_x_features.append(
                        np.concatenate(
                            (state, action_features, query_onehot, normalized_position)
                        )
                    )
                    oracle_x_zero_action_features.append(
                        np.concatenate(
                            (state, no_action_vector, query_onehot, normalized_position)
                        )
                    )
                    targets.append(
                        _normalized_scene_patches(
                            dataset,
                            normalization,
                            teacher.patch_spec,
                            scene,
                            edge.target_world,
                            position,
                        )[query]
                    )
                    source_targets.append(
                        _normalized_scene_patches(
                            dataset,
                            normalization,
                            teacher.patch_spec,
                            scene,
                            edge.source_world,
                            position,
                        )[query]
                    )
                    bank_ids.append(str(dataset.bank_ids[scene]))
                    city_ids.append(str(dataset.city_ids[scene]))
                    routes.append(str(ROUTE_NAMES[routed.response_route[key]]))
                    pair_ids.append(
                        f"{dataset.bank_ids[scene]}:{edge.source_world}:{edge.target_world}:{position}:{query}"
                    )
                    scene_indices.append(scene)
                    source_worlds.append(edge.source_world)
                    target_worlds.append(edge.target_world)
                    positions.append(position)
                    queries.append(query)
                    cluster_ids.append(str(dataset.base_map_cluster_ids[scene]))
                    canonical_cluster_ids.append(dataset.canonical_base_map_digest(scene))
                    canonical_bank_ids.append(_canonical_bank_digest(dataset, scene))
                    wrong_action_match_statuses.append(swap_status)
                    wrong_action_worlds.append(-1 if swap_world is None else int(swap_world))
    if not features:
        raise RuntimeError("response probe dataset has no eligible patches")
    return {
        "features": np.asarray(features),
        "action_swap_features": np.asarray(action_swap_features),
        "no_action_features": np.asarray(no_action_features),
        "without_map_features": np.asarray(without_map_features),
        "map_swap_features": np.asarray(map_swap_features),
        "edit_only_features": np.asarray(edit_only_features),
        "csi_only_features": np.asarray(csi_only_features),
        "oracle_x_features": np.asarray(oracle_x_features),
        "without_map_zero_action_features": np.asarray(
            without_map_zero_action_features
        ),
        "edit_only_zero_action_features": np.asarray(
            edit_only_zero_action_features
        ),
        "oracle_x_zero_action_features": np.asarray(
            oracle_x_zero_action_features
        ),
        "targets": np.asarray(targets),
        "bank_ids": np.asarray(bank_ids),
        "city_ids": np.asarray(city_ids),
        "source_targets": np.asarray(source_targets),
        "routes": np.asarray(routes),
        "pair_ids": np.asarray(pair_ids),
        "scene_indices": np.asarray(scene_indices, dtype=np.int64),
        "source_worlds": np.asarray(source_worlds, dtype=np.int64),
        "target_worlds": np.asarray(target_worlds, dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.int64),
        "queries": np.asarray(queries, dtype=np.int64),
        "base_map_cluster_ids": np.asarray(cluster_ids),
        "canonical_base_map_digests": np.asarray(canonical_cluster_ids),
        "canonical_bank_digests": np.asarray(canonical_bank_ids),
        "wrong_action_match_status": np.asarray(wrong_action_match_statuses),
        "wrong_action_world": np.asarray(wrong_action_worlds, dtype=np.int64),
        "input_contract": "masked_F_query_state_plus_typed_action_plus_query; target patch is supervision-only",
    }


def _response_effect_rows(
    seed, arm, evaluated, prediction, action_swap_prediction, no_action_prediction
):
    rows = []
    for pair_id in np.unique(evaluated["pair_ids"]):
        mask = evaluated["pair_ids"] == pair_id
        first = int(np.flatnonzero(mask)[0])
        prediction_error = float(np.mean((prediction[mask] - evaluated["targets"][mask]) ** 2))
        copy_error = float(
            np.mean((evaluated["source_targets"][mask] - evaluated["targets"][mask]) ** 2)
        )
        swap_status = str(evaluated["wrong_action_match_status"][first])
        action_swap_error = (
            float(
                np.mean(
                    (action_swap_prediction[mask] - evaluated["targets"][mask]) ** 2
                )
            )
            if swap_status == "exact"
            else None
        )
        no_action_error = float(
            np.mean((no_action_prediction[mask] - evaluated["targets"][mask]) ** 2)
        )
        rows.append(
            {
                "seed": int(seed),
                "arm": str(arm),
                "pair_id": str(pair_id),
                "scene_index": int(evaluated["scene_indices"][first]),
                "bank_id": str(evaluated["bank_ids"][first]),
                "base_map_cluster_id": str(evaluated["base_map_cluster_ids"][first]),
                "canonical_base_map_digest": str(
                    evaluated["canonical_base_map_digests"][first]
                ),
                "canonical_bank_digest": str(
                    evaluated["canonical_bank_digests"][first]
                ),
                "city_id": str(evaluated["city_ids"][first]),
                "source_world": int(evaluated["source_worlds"][first]),
                "target_world": int(evaluated["target_worlds"][first]),
                "position_index": int(evaluated["positions"][first]),
                "query_index": int(evaluated["queries"][first]),
                "route": str(evaluated["routes"][first]),
                "wrong_action_match_status": swap_status,
                "wrong_action_world": int(evaluated["wrong_action_world"][first]),
                "prediction_mse": prediction_error,
                "copy_mse": copy_error,
                "action_swap_mse": action_swap_error,
                "no_action_mse": no_action_error,
                "response_advantage": copy_error - prediction_error,
                "response_advantage_vs_action_swap": (
                    action_swap_error - prediction_error
                    if action_swap_error is not None
                    else None
                ),
                "response_advantage_vs_no_action": no_action_error - prediction_error,
            }
        )
    return rows


def _complex_csi_from_patches(patches, normalization, spec):
    raw = np.asarray(patches) * normalization.patch_scale + normalization.patch_mean
    csi = unpatchify_csi(raw, spec)
    count = spec.complex_values
    return (csi[..., :count] + 1j * csi[..., count:]).reshape(
        *csi.shape[:-1], spec.antennas, spec.subcarriers
    )


def _periodic_power_spread(
    power: np.ndarray,
    axis: np.ndarray,
    *,
    period: float,
) -> float:
    weights = np.asarray(power, dtype=np.float64)
    coordinates = np.asarray(axis, dtype=np.float64)
    if (
        weights.ndim != 1
        or coordinates.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(coordinates))
        or np.any(weights < 0.0)
        or not np.isfinite(period)
        or float(period) <= 0.0
    ):
        raise ValueError("periodic power spread inputs are invalid")
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.0
    raw_distance = np.abs(coordinates[:, None] - coordinates[None, :])
    wrapped_distance = np.minimum(
        np.mod(raw_distance, float(period)),
        float(period) - np.mod(raw_distance, float(period)),
    )
    pair_weights = weights[:, None] * weights[None, :]
    variance = 0.5 * float(
        np.sum(pair_weights * wrapped_distance**2) / (total * total)
    )
    return float(np.sqrt(max(variance, 0.0)))


def _channel_summary(channel):
    values = np.asarray(channel, dtype=np.complex128)
    power = np.abs(values) ** 2
    received_power_db = float(10.0 * np.log10(max(float(np.mean(power)), 1e-12)))
    delay_power = np.mean(np.abs(np.fft.ifft(values, axis=-1)) ** 2, axis=-2)
    delay_axis = np.linspace(0.0, 1.0, delay_power.shape[-1], endpoint=False)
    delay_total = max(float(np.sum(delay_power)), 1e-12)
    delay_mean = float(np.sum(delay_power * delay_axis) / delay_total)
    delay_spread = float(
        np.sqrt(np.sum(delay_power * (delay_axis - delay_mean) ** 2) / delay_total)
    )
    angle_power = np.mean(
        np.abs(np.fft.fftshift(np.fft.fft(values, axis=-2), axes=-2)) ** 2,
        axis=-1,
    )
    angle_axis = np.linspace(-1.0, 1.0, angle_power.shape[-1], endpoint=False)
    angular_spread = _periodic_power_spread(
        angle_power,
        angle_axis,
        period=2.0,
    )
    return {
        "path_loss": -received_power_db,
        "delay_spread": delay_spread,
        "angular_spread": angular_spread,
    }


def _transition_metrics(
    prediction, zero_action_prediction, source, target, normalization, spec
):
    predicted = _complex_csi_from_patches(prediction, normalization, spec)
    zero_action = _complex_csi_from_patches(
        zero_action_prediction, normalization, spec
    )
    source_csi = _complex_csi_from_patches(source, normalization, spec)
    target_csi = _complex_csi_from_patches(target, normalization, spec)
    true_delta = target_csi - source_csi
    predicted_delta = predicted - zero_action
    flat_predicted_delta = predicted_delta.reshape(predicted_delta.shape[0], -1)
    flat_true_delta = true_delta.reshape(true_delta.shape[0], -1)
    numerator = np.abs(
        np.sum(np.conj(flat_predicted_delta) * flat_true_delta, axis=1)
    ) ** 2
    denominator = np.sum(np.abs(flat_predicted_delta) ** 2, axis=1) * np.sum(
        np.abs(flat_true_delta) ** 2, axis=1
    )
    sgcs = numerator / np.maximum(denominator, 1e-12)
    errors = {name: [] for name in ("path_loss", "delay_spread", "angular_spread")}
    directions = {name: [] for name in errors}
    for index in range(predicted.shape[0]):
        source_summary = _channel_summary(source_csi[index])
        zero_action_summary = _channel_summary(zero_action[index])
        target_summary = _channel_summary(target_csi[index])
        prediction_summary = _channel_summary(predicted[index])
        for name in errors:
            true_change = target_summary[name] - source_summary[name]
            predicted_change = prediction_summary[name] - zero_action_summary[name]
            errors[name].append(abs(predicted_change - true_change))
            directions[name].append(
                1.0
                if abs(true_change) <= 1e-12 and abs(predicted_change) <= 1e-12
                else float(np.sign(true_change) == np.sign(predicted_change))
            )
    return {
        "native_sgcs": float(np.mean(sgcs)),
        **{
            f"native_{name}_change_mae": float(np.mean(values))
            for name, values in errors.items()
        },
        **{
            f"native_{name}_direction_accuracy": float(np.mean(values))
            for name, values in directions.items()
        },
    }


def _protocol_transition_skill(latent_prediction, latent_source, latent_target, include):
    """Bank-level teacher-latent TransitionSkill. None means N/A."""
    include = np.asarray(include, dtype=bool)
    if not np.any(include):
        return None
    predicted = np.asarray(latent_prediction, dtype=np.float64)[include]
    source = np.asarray(latent_source, dtype=np.float64)[include]
    target = np.asarray(latent_target, dtype=np.float64)[include]
    axes = tuple(range(1, predicted.ndim))
    numer = float(np.sum(np.sqrt(np.mean((predicted - target) ** 2, axis=axes))))
    denom = float(np.sum(np.sqrt(np.mean((source - target) ** 2, axis=axes))))
    if denom <= 1e-12:
        return None
    return float(1.0 - numer / denom)


def _native_mask_cover_metrics(model, dataset, teacher, config, normalization, scenes, bank_id):
    from .formal_qualification import _select_wrong_action

    scene_matches = [int(scene) for scene in scenes if str(dataset.bank_ids[int(scene)]) == bank_id]
    if len(scene_matches) != 1:
        raise RuntimeError("native response bank join is not unique")
    scene = scene_matches[0]
    route_norm = fit_route_normalization(dataset, teacher)
    routed = route_dataset(dataset, teacher, config, np.asarray([scene]), normalization=route_norm)
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    predictions = []
    latent_predictions = []
    no_action_predictions = []
    latent_no_action_predictions = []
    action_swap_predictions = []
    latent_action_swap_predictions = []
    sources = []
    latent_sources = []
    targets = []
    latent_targets = []
    teacher_sensitive = []
    wrong_action_statuses = []
    direction_cosines = []
    magnitude_errors = []
    null_delta_norms = []
    latent_null_delta_norms = []
    audit_entries = [
        entry for entry in teacher.audit_mask_bank if entry.mode == "random_75"
    ]
    for edge in dataset.directed_edges(scene):
        action = typed_signed_edit(
            dataset.maps[scene, edge.source_world],
            dataset.maps[scene, edge.target_world],
            dataset.map_channel_names,
            material_categories,
        )
        action_tensor = tensor_for_module(
            model,
            _normalized_action(normalization, action[None, ...]),
            dtype=torch.float32,
        )
        zero_action_tensor = torch.zeros_like(action_tensor)
        for position in _eligible_evaluation_positions(dataset, scene):
            akey = (scene, edge.source_world, edge.target_world, position)
            if not _native_response_route_is_active(routed, akey):
                continue
            swap_action, swap_status, _ = _select_wrong_action(
                dataset,
                scene,
                edge.source_world,
                edge.bit_index,
                action,
                material_categories,
                receiver_position=dataset.positions[scene, position],
            )
            swap_action_tensor = tensor_for_module(
                model,
                _normalized_action(normalization, swap_action[None, ...]),
                dtype=torch.float32,
            )
            source = _normalized_scene_patches(
                dataset, normalization, teacher.patch_spec, scene, edge.source_world, position
            )
            target = _normalized_scene_patches(
                dataset, normalization, teacher.patch_spec, scene, edge.target_world, position
            )
            predicted = np.zeros_like(target)
            predicted_latent = np.zeros(
                (teacher.patch_spec.patch_count, normalization.latent_mean.size),
                dtype=np.float64,
            )
            predicted_no_action = np.zeros_like(target)
            predicted_latent_no_action = np.zeros_like(predicted_latent)
            predicted_action_swap = np.zeros_like(target)
            predicted_latent_action_swap = np.zeros_like(predicted_latent)
            for query in range(teacher.patch_spec.patch_count):
                entry = next(item for item in audit_entries if item.query == query)
                visible = source.copy()
                visible[entry.mask] = 0.0
                maps = _normalized_map(normalization, dataset.maps[scene, edge.source_world])[None, ...]
                radio = _normalized_radio(
                    normalization, dataset.radio_config[scene], dataset.bs_pose[scene]
                )[None, ...]
                with torch.no_grad():
                    state = model.state(
                        tensor_for_module(model, visible[None, ...], dtype=torch.float32),
                        tensor_for_module(model, maps, dtype=torch.float32),
                        tensor_for_module(model, radio, dtype=torch.float32),
                        tensor_for_module(model, entry.mask[None, :], dtype=torch.bool),
                    )
                    latent_value, value = model.predict(
                        state, action_tensor, tensor_for_module(model, [query], dtype=torch.long)
                    )
                    latent_no_action, no_action_value = model.predict(
                        state,
                        zero_action_tensor,
                        tensor_for_module(model, [query], dtype=torch.long),
                    )
                    latent_swap, swap_value = model.predict(
                        state,
                        swap_action_tensor,
                        tensor_for_module(model, [query], dtype=torch.long),
                    )
                predicted[query] = value[0].cpu().numpy()
                predicted_latent[query] = latent_value[0].cpu().numpy()
                predicted_no_action[query] = no_action_value[0].cpu().numpy()
                predicted_latent_no_action[query] = latent_no_action[0].cpu().numpy()
                predicted_action_swap[query] = swap_value[0].cpu().numpy()
                predicted_latent_action_swap[query] = latent_swap[0].cpu().numpy()
                rkey = (scene, edge.source_world, edge.target_world, int(position), query)
                if routed.response_route[rkey] == 0:
                    null_delta_norms.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    (
                                        predicted[query]
                                        - predicted_no_action[query]
                                    )
                                    ** 2
                                )
                            )
                        )
                    )
                    latent_null_delta_norms.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    (
                                        predicted_latent[query]
                                        - predicted_latent_no_action[query]
                                    )
                                    ** 2
                                )
                            )
                        )
                    )
            predictions.append(predicted)
            latent_predictions.append(predicted_latent)
            no_action_predictions.append(predicted_no_action)
            latent_no_action_predictions.append(predicted_latent_no_action)
            action_swap_predictions.append(predicted_action_swap)
            latent_action_swap_predictions.append(predicted_latent_action_swap)
            sources.append(source)
            latent_source = (
                routed.teacher_latent[scene][edge.source_world, position]
                - normalization.latent_mean
            ) / normalization.latent_scale
            latent_target = (
                routed.teacher_latent[scene][edge.target_world, position]
                - normalization.latent_mean
            ) / normalization.latent_scale
            latent_sources.append(latent_source)
            targets.append(target)
            latent_targets.append(latent_target)
            teacher_sensitive.append(routed.alignment_teacher_stratum[akey] == 2)
            wrong_action_statuses.append(swap_status)
            true_delta = (target - source).reshape(-1)
            predicted_delta = (predicted - predicted_no_action).reshape(-1)
            denominator = float(np.linalg.norm(true_delta) * np.linalg.norm(predicted_delta))
            direction_cosines.append(
                float(np.dot(true_delta, predicted_delta) / denominator) if denominator > 1e-12 else 0.0
            )
            magnitude_errors.append(
                float(
                    abs(np.linalg.norm(predicted_delta) - np.linalg.norm(true_delta))
                    / max(float(np.linalg.norm(true_delta)), 1e-12)
                )
            )
    if not predictions:
        raise RuntimeError("native response audit has no alignment-active transition")
    prediction = np.asarray(predictions)
    latent_prediction = np.asarray(latent_predictions)
    no_action_prediction = np.asarray(no_action_predictions)
    latent_no_action_prediction = np.asarray(latent_no_action_predictions)
    action_swap_prediction = np.asarray(action_swap_predictions)
    latent_action_swap_prediction = np.asarray(latent_action_swap_predictions)
    source = np.asarray(sources)
    latent_source = np.asarray(latent_sources)
    target = np.asarray(targets)
    latent_target = np.asarray(latent_targets)
    wrong_action_status = np.asarray(wrong_action_statuses)
    exact_swap = wrong_action_status == "exact"
    target_energy = max(float(np.sum(target**2)), 1e-12)
    latent_target_energy = max(float(np.sum(latent_target**2)), 1e-12)
    exact_target_energy = (
        max(float(np.sum(target[exact_swap] ** 2)), 1e-12)
        if np.any(exact_swap)
        else None
    )
    exact_latent_target_energy = (
        max(float(np.sum(latent_target[exact_swap] ** 2)), 1e-12)
        if np.any(exact_swap)
        else None
    )
    null_threshold = float(config["qualification"]["response_physical_null_rms_max"])
    transition_metrics = _transition_metrics(
        prediction,
        no_action_prediction,
        source,
        target,
        normalization,
        teacher.patch_spec,
    )
    native_transition_skill = _protocol_transition_skill(
        latent_prediction,
        latent_source,
        latent_target,
        np.asarray(teacher_sensitive, dtype=bool),
    )
    return {
        "native_target_free_full_channel_nmse": float(
            np.sum((prediction - target) ** 2) / target_energy
        ),
        "native_copy_full_channel_nmse": float(np.sum((source - target) ** 2) / target_energy),
        "native_no_action_full_channel_nmse": float(
            np.sum((no_action_prediction - target) ** 2) / target_energy
        ),
        "native_target_free_action_swap_exact_full_channel_nmse": (
            float(
                np.sum((prediction[exact_swap] - target[exact_swap]) ** 2)
                / exact_target_energy
            )
            if np.any(exact_swap)
            else None
        ),
        "native_action_swap_full_channel_nmse": (
            float(
                np.sum(
                    (action_swap_prediction[exact_swap] - target[exact_swap]) ** 2
                )
                / exact_target_energy
            )
            if np.any(exact_swap)
            else None
        ),
        "native_latent_nmse": float(
            np.sum((latent_prediction - latent_target) ** 2) / latent_target_energy
        ),
        "native_latent_copy_nmse": float(
            np.sum((latent_source - latent_target) ** 2) / latent_target_energy
        ),
        "native_latent_no_action_nmse": float(
            np.sum((latent_no_action_prediction - latent_target) ** 2)
            / latent_target_energy
        ),
        "native_latent_action_swap_exact_target_nmse": (
            float(
                np.sum(
                    (latent_prediction[exact_swap] - latent_target[exact_swap]) ** 2
                )
                / exact_latent_target_energy
            )
            if np.any(exact_swap)
            else None
        ),
        "native_latent_action_swap_nmse": (
            float(
                np.sum(
                    (
                        latent_action_swap_prediction[exact_swap]
                        - latent_target[exact_swap]
                    )
                    ** 2
                )
                / exact_latent_target_energy
            )
            if np.any(exact_swap)
            else None
        ),
        "native_action_swap_exact_count": int(np.sum(exact_swap)),
        "native_action_swap_fallback_count": int(
            np.sum(wrong_action_status == "fallback")
        ),
        "native_action_swap_failed_count": int(
            np.sum(wrong_action_status == "failed")
        ),
        "native_action_swap_exact_fraction": float(np.mean(exact_swap)),
        "native_delta_direction_cosine": float(np.mean(direction_cosines)),
        "native_delta_relative_magnitude_error": float(np.mean(magnitude_errors)),
        **transition_metrics,
        "native_transition_skill": native_transition_skill,
        "native_null_patch_count": len(null_delta_norms),
        "native_null_delta_rms_mean": (
            float(np.mean(null_delta_norms)) if null_delta_norms else None
        ),
        "native_null_violation_rate": (
            float(np.mean(np.asarray(null_delta_norms) > null_threshold))
            if null_delta_norms
            else None
        ),
        "native_latent_null_delta_rms_mean": (
            float(np.mean(latent_null_delta_norms)) if latent_null_delta_norms else None
        ),
        "native_latent_null_violation_rate": (
            float(
                np.mean(
                    np.asarray(latent_null_delta_norms)
                    > float(config["qualification"]["response_latent_null_rms_max"])
                )
            )
            if latent_null_delta_norms
            else None
        ),
        "native_mask_bank": "B_audit_hold",
        "native_input_contract": "masked source F-state plus supplied map/c/action/query; target CSI supervision-only",
    }


def _normalized_scene_patches(dataset, normalization, spec, scene, world, position):
    raw = patchify_csi(dataset.csi[scene, world, position], spec)
    return (raw - normalization.patch_mean) / normalization.patch_scale


def _evaluation_gate(
    config,
    dataset,
    cgs_rows,
    route_rows,
    effect_bin_rows,
    response_rows,
    shortcut_rows,
    factorial_gate,
    evidence,
    *,
    qualification_gate_sha256,
    factorial_gate_sha256,
):
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    alignment_margin = float(
        config["evaluation"]["minimum_alignment_superiority"]
    )
    response_margin = float(config["evaluation"]["minimum_response_superiority"])
    alignment_superiority = _paired_arm_comparison(
        cgs_rows,
        "cgs_auroc",
        "alignment",
        "endpoint",
        higher_is_better=True,
        resamples=resamples,
        seed=81101,
        null_threshold=alignment_margin,
    )
    response_superiority = _paired_arm_comparison(
        response_rows,
        "native_target_free_full_channel_nmse",
        "response",
        "endpoint",
        higher_is_better=False,
        resamples=resamples,
        seed=81102,
        null_threshold=response_margin,
    )
    response_copy = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_copy_full_channel_nmse",
        "native_target_free_full_channel_nmse",
        resamples,
        81103,
        null_threshold=response_margin,
    )
    response_swap = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_action_swap_full_channel_nmse",
        "native_target_free_action_swap_exact_full_channel_nmse",
        resamples,
        81104,
        null_threshold=response_margin,
    )
    response_no_action = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_no_action_full_channel_nmse",
        "native_target_free_full_channel_nmse",
        resamples,
        81106,
        null_threshold=response_margin,
    )
    latent_copy = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_latent_copy_nmse",
        "native_latent_nmse",
        resamples,
        81107,
        null_threshold=response_margin,
    )
    latent_no_action = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_latent_no_action_nmse",
        "native_latent_nmse",
        resamples,
        81108,
        null_threshold=response_margin,
    )
    latent_swap = _within_arm_advantage_interval(
        response_rows,
        "response",
        "native_latent_action_swap_nmse",
        "native_latent_action_swap_exact_target_nmse",
        resamples,
        81109,
        null_threshold=response_margin,
    )
    probe_response_swap = _within_arm_advantage_interval(
        response_rows,
        "response",
        "probe_action_swap_active_patch_nmse",
        "unified_response_probe_action_swap_exact_patch_nmse",
        resamples,
        81110,
        null_threshold=response_margin,
    )
    shortcut_intervals = {
        name: _within_arm_advantage_interval(
            response_rows,
            "response",
            f"probe_{name}_active_patch_nmse",
            "unified_response_probe_active_patch_nmse",
            resamples,
            81120 + offset,
            null_threshold=response_margin,
        )
        for offset, name in enumerate(("without_map", "edit_only", "csi_only"))
    }
    direction = _within_arm_level_interval(
        response_rows, "response", "native_delta_direction_cosine", resamples, 81105
    )
    expected_scopes = ["source_final_unseen_bank"] + [
        f"target:{city}"
        for city in sorted(
            set(dataset.city_ids[dataset.scene_roles == "target"].tolist())
        )
    ]
    scope_intervals = _g3_primary_scope_intervals(
        cgs_rows,
        response_rows,
        expected_scopes,
        resamples,
        alignment_superiority_margin=alignment_margin,
        response_superiority_margin=response_margin,
    )
    null_safety = _null_safety_by_arm(config, route_rows)
    scope_null_safety = {
        scope: _null_safety_by_arm(
            config,
            [row for row in route_rows if row.get("evaluation_scope") == scope],
        )
        for scope in expected_scopes
    }
    response_null_safety = {
        scope: _response_null_safety(
            config,
            [row for row in response_rows if row.get("evaluation_scope") == scope],
        )
        for scope in expected_scopes
    }
    null_safe = all(
        scope_null_safety[scope][arm]["passed"]
        for scope in expected_scopes
        for arm in ("alignment", "full")
    )
    response_null_safe = all(
        response_null_safety[scope]["passed"] for scope in expected_scopes
    )
    family = [
        alignment_superiority,
        response_superiority,
        response_copy,
        response_no_action,
        response_swap,
        latent_copy,
        latent_no_action,
        latent_swap,
        probe_response_swap,
        *shortcut_intervals.values(),
        direction,
        *[
            interval
            for scope in expected_scopes
            for interval in scope_intervals[scope].values()
        ],
    ]
    adjusted = holm_adjust([float(row["p_value_two_sided"]) for row in family])
    for row, value in zip(family, adjusted):
        row["holm_adjusted_p"] = float(value)
    alpha = float(config["evaluation"]["familywise_alpha"])
    scope_primary_passed = all(
        interval_decision(
            scope_intervals[scope]["alignment_superiority"],
            threshold=float(config["evaluation"]["minimum_alignment_superiority"]),
            relation="superiority",
        )
        and scope_intervals[scope]["alignment_superiority"]["holm_adjusted_p"]
        < alpha
        and interval_decision(
            scope_intervals[scope]["response_superiority"],
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and scope_intervals[scope]["response_superiority"]["holm_adjusted_p"]
        < alpha
        and all(
            interval_decision(
                scope_intervals[scope][name],
                threshold=float(config["evaluation"]["minimum_response_superiority"]),
                relation="superiority",
            )
            and scope_intervals[scope][name]["holm_adjusted_p"] < alpha
            for name in (
                "response_vs_copy",
                "response_vs_no_action",
                "response_vs_action_swap",
            )
        )
        and float(scope_intervals[scope]["response_direction"]["ci95_low"]) > 0
        and scope_intervals[scope]["response_direction"]["holm_adjusted_p"]
        < alpha
        for scope in expected_scopes
    )
    effect_bins_complete = _effect_bins_complete(cgs_rows, effect_bin_rows)
    evaluation_scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    canonical_banks = {
        _canonical_bank_digest(dataset, int(scene)) for scene in evaluation_scenes
    }
    expected_cells = {
        (int(seed), str(arm), str(bank))
        for seed in config["seeds"]
        for arm in config["factorial"]["arms"]
        for bank in canonical_banks
    }
    gray_complete = _gray_cells_complete(route_rows, expected_cells)
    correlation_macros = {
        arm: _native_probe_correlation_macro(cgs_rows, arm, expected_cells)
        for arm in ("alignment", "full")
    }
    correlation_complete = all(
        np.isfinite(value)
        and value >= float(config["evaluation"]["minimum_native_probe_correlation"])
        for value in correlation_macros.values()
    )
    expected_shortcuts = {
        "constant",
        "csi_only",
        "map_only",
        "scene_id_only",
        "edit_status_xor",
        "variant_id_matcher",
    }
    shortcut_groups = {}
    shortcut_duplicate = False
    for row in shortcut_rows:
        key = (
            int(row["seed"]),
            str(row["arm"]),
            str(row["canonical_bank_digest"]),
        )
        group = shortcut_groups.setdefault(key, {})
        if row["baseline"] in group:
            shortcut_duplicate = True
        group[row["baseline"]] = row
    shortcut_complete = bool(
        shortcut_groups
        and not shortcut_duplicate
        and set(shortcut_groups) == expected_cells
        and all(set(rows) == expected_shortcuts for rows in shortcut_groups.values())
    )
    legal_shortcut_rows = [
        row
        for row in shortcut_rows
        if row["baseline"] in {"constant", "csi_only", "map_only"}
    ]
    metadata_probe_rows = [
        row
        for row in shortcut_rows
        if row["baseline"]
        in {"scene_id_only", "edit_status_xor", "variant_id_matcher"}
    ]
    metadata_contracts = {
        "scene_id_only": "stable_hashed_bank_token_no_label_feature",
        "edit_status_xor": "separate_world_bits_and_natural_flags_no_xor",
        "variant_id_matcher": "separate_stable_hashed_variant_tokens_no_match_or_unk_override",
    }
    shortcut_margin = float(config["evaluation"]["null_score_equivalence_margin"])
    shortcut_passed = bool(
        shortcut_complete
        and legal_shortcut_rows
        and metadata_probe_rows
        and all(
            abs(float(row["unseen_bank_auroc"]) - 0.5) <= shortcut_margin
            for row in legal_shortcut_rows + metadata_probe_rows
        )
        and all(
            row["identity_token_contract"] == metadata_contracts[row["baseline"]]
            for row in metadata_probe_rows
        )
    )
    minimum_swap_fraction = float(
        config["qualification"]["minimum_geometry_matched_wrong_action_fraction"]
    )
    action_swap_coverage_passed = bool(
        response_rows
        and all(
            int(row.get("native_action_swap_exact_count", 0)) > 0
            and float(row.get("native_action_swap_exact_fraction", 0.0))
            >= minimum_swap_fraction
            and int(row.get("probe_action_swap_exact_count", 0)) > 0
            and float(row.get("probe_action_swap_exact_fraction", 0.0))
            >= minimum_swap_fraction
            for row in response_rows
            if row["arm"] in {"response", "full"}
        )
    )
    g3_subgates = {
        "1_alignment_active_cgs_superiority_ci": "PASS"
        if interval_decision(
            alignment_superiority,
            threshold=float(config["evaluation"]["minimum_alignment_superiority"]),
            relation="superiority",
        )
        and alignment_superiority["holm_adjusted_p"] < alpha
        else "FAIL",
        "2_response_active_native_superiority_ci": "PASS"
        if interval_decision(
            response_superiority,
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and response_superiority["holm_adjusted_p"] < alpha
        else "FAIL",
        "3_response_physical_and_latent_baselines_ci": "PASS"
        if interval_decision(
            response_copy,
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and interval_decision(
            response_swap,
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and interval_decision(
            response_no_action,
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and all(
            interval_decision(
                value,
                threshold=float(config["evaluation"]["minimum_response_superiority"]),
                relation="superiority",
            )
            for value in (latent_copy, latent_no_action, latent_swap)
        )
        and interval_decision(
            probe_response_swap,
            threshold=float(config["evaluation"]["minimum_response_superiority"]),
            relation="superiority",
        )
        and probe_response_swap["holm_adjusted_p"] < alpha
        and all(
            interval_decision(
                value,
                threshold=float(config["evaluation"]["minimum_response_superiority"]),
                relation="superiority",
            )
            and value["holm_adjusted_p"] < alpha
            for value in shortcut_intervals.values()
        )
        and response_copy["holm_adjusted_p"] < alpha
        and response_no_action["holm_adjusted_p"] < alpha
        and response_swap["holm_adjusted_p"] < alpha
        and all(
            value["holm_adjusted_p"] < alpha
            for value in (latent_copy, latent_no_action, latent_swap)
        )
        and all(
            np.isfinite(row["probe_oracle_x_active_patch_nmse"])
            for row in response_rows
            if row["arm"] == "response"
        )
        and action_swap_coverage_passed
        else "FAIL",
        "4_response_direction_and_magnitude": "PASS"
        if float(direction["ci95_low"]) > 0
        and direction["holm_adjusted_p"] < alpha
        and all(
            np.isfinite(row["native_delta_relative_magnitude_error"])
            for row in response_rows
            if row["arm"] == "response"
        )
        and all(
            all(
                np.isfinite(row[name])
                for name in (
                    "native_sgcs",
                    "native_path_loss_change_mae",
                    "native_delay_spread_change_mae",
                    "native_angular_spread_change_mae",
                    "native_path_loss_direction_accuracy",
                    "native_delay_spread_direction_accuracy",
                    "native_angular_spread_direction_accuracy",
                )
            )
            and (
                row["native_transition_skill"] is None
                or np.isfinite(row["native_transition_skill"])
            )
            for row in response_rows
            if row["arm"] == "response"
        )
        else "FAIL",
        "5_four_active_effect_bins": "PASS" if effect_bins_complete else "FAIL",
        "6_gray_distributions_reported": "PASS" if gray_complete else "FAIL",
        "7_null_equivalence_and_overclassification": "PASS"
        if null_safe and response_null_safe
        else "FAIL",
        "8_native_probe_correlation": "PASS"
        if correlation_complete and shortcut_passed
        else "FAIL",
        "9_unpooled_source_and_target_primary_metrics": "PASS"
        if scope_primary_passed
        else "FAIL",
    }
    c3_keys = (
        "1_alignment_active_cgs_superiority_ci",
        "5_four_active_effect_bins",
        "6_gray_distributions_reported",
        "7_null_equivalence_and_overclassification",
        "8_native_probe_correlation",
        "9_unpooled_source_and_target_primary_metrics",
    )
    c5_keys = (
        "2_response_active_native_superiority_ci",
        "3_response_physical_and_latent_baselines_ci",
        "4_response_direction_and_magnitude",
        "7_null_equivalence_and_overclassification",
        "9_unpooled_source_and_target_primary_metrics",
    )
    c3_complete = bool(
        not dataset.is_fixture
        and all(g3_subgates[key] == "PASS" for key in c3_keys)
    )
    c5_complete = bool(
        not dataset.is_fixture
        and all(g3_subgates[key] == "PASS" for key in c5_keys)
    )
    g3_pass = bool(c3_complete and c5_complete)
    g4 = dict(factorial_gate["g4_subgates"])
    cgs_noninferiority_margin = abs(
        float(config["evaluation"]["minimum_cgs_noninferiority"])
    )
    response_noninferiority_margin = abs(
        float(config["evaluation"]["minimum_response_noninferiority"])
    )
    full_cgs = _paired_arm_comparison(
        cgs_rows,
        "cgs_auroc",
        "full",
        "alignment",
        True,
        resamples,
        81201,
        null_threshold=-cgs_noninferiority_margin,
    )
    full_response = _paired_arm_comparison(
        response_rows,
        "native_target_free_full_channel_nmse",
        "full",
        "response",
        False,
        resamples,
        81202,
        null_threshold=-response_noninferiority_margin,
    )
    for interval, adjusted_p in zip(
        (full_cgs, full_response),
        holm_adjust(
            [full_cgs["p_value_two_sided"], full_response["p_value_two_sided"]]
        ),
    ):
        interval["holm_adjusted_p"] = float(adjusted_p)
    g4["2_cgs_noninferior_to_alignment"] = "PASS" if interval_decision(
        full_cgs,
        threshold=cgs_noninferiority_margin,
        relation="noninferiority",
    ) and full_cgs["holm_adjusted_p"] < alpha else "FAIL"
    g4["3_native_response_noninferior_to_response"] = "PASS" if interval_decision(
        full_response,
        threshold=response_noninferiority_margin,
        relation="noninferiority",
    ) and full_response["holm_adjusted_p"] < alpha else "FAIL"
    g4_status = "PASS" if all(value == "PASS" for value in g4.values()) else (
        "FAIL" if any(value == "FAIL" for value in g4.values()) else "NOT_ASSESSED"
    )
    if dataset.is_fixture:
        g4_status = "FAIL"
    vector = complete_gate_vector(
        {
            "G1": "FAIL" if dataset.is_fixture else "PASS",
            "G2": "FAIL" if dataset.is_fixture else "PASS",
            "G3": "PASS" if g3_pass else "FAIL",
            "G4": g4_status,
            "G5": factorial_gate.get("gate_vector", {}).get("G5", "NOT_ASSESSED"),
        }
    )
    return {
        "schema_version": "csi-pairs-v6-evaluation-gate-v3",
        "status": "PASS" if g3_pass else "FAIL",
        "passed": bool(g3_pass),
        "scientific_claim_status": "SOFTWARE_ONLY" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
        **evidence,
        "qualification_gate_sha256": qualification_gate_sha256,
        "factorial_gate_sha256": factorial_gate_sha256,
        "gate_vector": vector,
        "g3_subgates": g3_subgates,
        "g3_intervals": {
            "alignment_superiority": alignment_superiority,
            "response_superiority": response_superiority,
            "response_vs_copy": response_copy,
            "response_vs_no_action": response_no_action,
            "response_vs_action_swap": response_swap,
            "latent_vs_copy": latent_copy,
            "latent_vs_no_action": latent_no_action,
            "latent_vs_action_swap": latent_swap,
            "unified_probe_vs_action_swap": probe_response_swap,
            "shortcut_probe_intervals": shortcut_intervals,
            "response_direction": direction,
        },
        "g3_scope_intervals": scope_intervals,
        "c3_evidence_complete": c3_complete,
        "c5_evidence_complete": c5_complete,
        "g4_subgates": g4,
        "g4_intervals": {"full_cgs": full_cgs, "full_response": full_response},
        "null_compatibility_safety": null_safety,
        "null_compatibility_safety_by_scope": scope_null_safety,
        "response_null_safety_by_scope": response_null_safety,
        "native_probe_correlation_macro": {
            "hierarchy": "canonical_unit_then_seed_to_bank_to_foundation_equal_macro",
            "values": correlation_macros,
        },
        "alignment_shortcut_audit": {
            "passed": shortcut_passed,
            "complete": shortcut_complete,
            "required_baselines": sorted(expected_shortcuts),
            "legal_control_chance_margin": shortcut_margin,
            "source_trained_frozen_probes": True,
            "label_derived_match_features_forbidden": True,
            "forced_unk_override": False,
            "metadata_leakage_probes_must_be_chance": True,
            "headline_edge_scope": "non-natural-incident direct edits only",
        },
        "geometry_matched_action_swap_audit": {
            "passed": action_swap_coverage_passed,
            "minimum_exact_fraction": minimum_swap_fraction,
            "denominator": (
                "exact_same_typed_planes_per_plane_geometry_and_receiver_relative_distance_only"
            ),
        },
        "claim_boundary": "G4 remains NOT_ASSESSED until equal-FLOP and both matched-concat controls are present.",
    }


def _gray_cells_complete(route_rows, expected_cells):
    rows = [row for row in route_rows if row.get("route") == "gray"]
    cells = [
        (
            int(row["seed"]),
            str(row["arm"]),
            str(row["canonical_bank_digest"]),
        )
        for row in rows
        if row.get("condition_status") == "ASSESSED"
    ]
    return bool(
        expected_cells
        and len(rows) == len(expected_cells)
        and len(cells) == len(expected_cells)
        and len(cells) == len(set(cells))
        and set(cells) == set(expected_cells)
    )


def _native_probe_correlation_macro(cgs_rows, arm, expected_cells=None):
    units = {}
    for row in cgs_rows:
        if str(row.get("arm")) != str(arm):
            continue
        required = (
            "seed",
            "canonical_base_map_digest",
            "canonical_bank_digest",
            "native_probe_spearman",
        )
        if any(row.get(field) is None for field in required):
            return float("nan")
        key = (int(row["seed"]), str(row["canonical_bank_digest"]))
        foundation = str(row["canonical_base_map_digest"])
        value = float(row["native_probe_spearman"])
        if not np.isfinite(value):
            return float("nan")
        if key in units:
            previous_foundation, previous_value = units[key]
            if previous_foundation != foundation or not np.isclose(
                previous_value, value, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    "native-probe correlation has conflicting duplicate canonical units"
                )
            continue
        units[key] = (foundation, value)
    if expected_cells is not None:
        expected = {
            (int(seed), str(bank))
            for seed, expected_arm, bank in expected_cells
            if str(expected_arm) == str(arm)
        }
        if set(units) != expected:
            return float("nan")
    if not units:
        return float("nan")
    bank_values = {}
    for (seed, bank), (foundation, value) in units.items():
        bank_values.setdefault((foundation, bank), {})[seed] = value
    foundation_values = {}
    for (foundation, _), seed_values in bank_values.items():
        foundation_values.setdefault(foundation, []).append(
            float(np.mean(list(seed_values.values())))
        )
    return float(
        np.mean(
            [
                np.mean(foundation_values[foundation])
                for foundation in sorted(foundation_values)
            ]
        )
    )


def _paired_arm_comparison(
    rows,
    metric,
    first_arm,
    second_arm,
    higher_is_better,
    resamples,
    seed,
    null_threshold=0.0,
):
    grouped = {}
    for row in rows:
        if row["arm"] not in {first_arm, second_arm} or row.get(metric) is None:
            continue
        key = (
            str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            ),
            int(row["seed"]),
        )
        bank = str(row.get("canonical_bank_digest") or row["bank_id"])
        grouped.setdefault((key, row["arm"]), {}).setdefault(bank, []).append(
            float(row[metric])
        )
    cells = sorted(
        key for key in {item[0] for item in grouped}
        if (key, first_arm) in grouped and (key, second_arm) in grouped
    )
    if not cells:
        raise RuntimeError(f"paired comparison has no complete cells for {metric}")
    first = np.asarray(
        [
            np.mean([np.mean(values) for values in grouped[(key, first_arm)].values()])
            for key in cells
        ]
    )
    second = np.asarray(
        [
            np.mean([np.mean(values) for values in grouped[(key, second_arm)].values()])
            for key in cells
        ]
    )
    advantage = first - second if higher_is_better else second - first
    clusters = np.asarray([key[0] for key in cells])
    result = paired_cluster_interval(clusters, advantage, np.zeros_like(advantage), resamples, seed)
    test = paired_sign_flip_test(
        clusters,
        advantage,
        np.zeros_like(advantage),
        seed + 1,
        null_difference=float(null_threshold),
    )
    return {
        **result,
        "null_difference": float(null_threshold),
        "p_value_two_sided": test["p_value_two_sided"],
    }


def _g3_primary_scope_intervals(
    cgs_rows,
    response_rows,
    scopes,
    resamples,
    *,
    alignment_superiority_margin=0.0,
    response_superiority_margin=0.0,
):
    result = {}
    for index, scope in enumerate(scopes):
        scoped_cgs = [
            row for row in cgs_rows if row.get("evaluation_scope") == scope
        ]
        scoped_response = [
            row for row in response_rows if row.get("evaluation_scope") == scope
        ]
        if not scoped_cgs or not scoped_response:
            raise RuntimeError(f"G3 evaluation scope is incomplete: {scope}")
        result[str(scope)] = {
            "alignment_superiority": _paired_arm_comparison(
                scoped_cgs,
                "cgs_auroc",
                "alignment",
                "endpoint",
                higher_is_better=True,
                resamples=resamples,
                seed=81300 + 2 * index,
                null_threshold=alignment_superiority_margin,
            ),
            "response_superiority": _paired_arm_comparison(
                scoped_response,
                "native_target_free_full_channel_nmse",
                "response",
                "endpoint",
                higher_is_better=False,
                resamples=resamples,
                seed=81301 + 2 * index,
                null_threshold=response_superiority_margin,
            ),
            "response_vs_copy": _within_arm_advantage_interval(
                scoped_response,
                "response",
                "native_copy_full_channel_nmse",
                "native_target_free_full_channel_nmse",
                resamples,
                81400 + 5 * index,
                null_threshold=response_superiority_margin,
            ),
            "response_vs_no_action": _within_arm_advantage_interval(
                scoped_response,
                "response",
                "native_no_action_full_channel_nmse",
                "native_target_free_full_channel_nmse",
                resamples,
                81401 + 5 * index,
                null_threshold=response_superiority_margin,
            ),
            "response_vs_action_swap": _within_arm_advantage_interval(
                scoped_response,
                "response",
                "native_action_swap_full_channel_nmse",
                "native_target_free_action_swap_exact_full_channel_nmse",
                resamples,
                81402 + 5 * index,
                null_threshold=response_superiority_margin,
            ),
            "response_direction": _within_arm_level_interval(
                scoped_response,
                "response",
                "native_delta_direction_cosine",
                resamples,
                81403 + 5 * index,
            ),
        }
    return result


def _within_arm_advantage_interval(
    rows,
    arm,
    baseline_metric,
    method_metric,
    resamples,
    seed,
    *,
    null_threshold=0.0,
):
    selected = [
        row
        for row in rows
        if row["arm"] == arm
        and row.get(baseline_metric) is not None
        and row.get(method_metric) is not None
    ]
    grouped = {}
    for row in selected:
        key = (
            str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            ),
            int(row["seed"]),
            str(row.get("canonical_bank_digest") or row["bank_id"]),
        )
        grouped.setdefault(key, [[], []])
        grouped[key][0].append(float(row[baseline_metric]))
        grouped[key][1].append(float(row[method_metric]))
    keys = sorted(grouped)
    if len({key[0] for key in keys}) < 2:
        return _unassessed_comparison(
            f"{baseline_metric} vs {method_metric} lacks two independent clusters"
        )
    clusters = np.asarray([key[0] for key in keys])
    baseline = np.asarray(
        [np.mean(grouped[key][0]) for key in keys], dtype=np.float64
    )
    method = np.asarray(
        [np.mean(grouped[key][1]) for key in keys], dtype=np.float64
    )
    result = paired_cluster_interval(clusters, baseline, method, resamples, seed)
    test = paired_sign_flip_test(
        clusters,
        baseline,
        method,
        seed + 1,
        null_difference=float(null_threshold),
    )
    return {
        **result,
        "null_difference": float(null_threshold),
        "p_value_two_sided": test["p_value_two_sided"],
    }


def _within_arm_level_interval(rows, arm, metric, resamples, seed):
    selected = [row for row in rows if row["arm"] == arm and row.get(metric) is not None]
    grouped = {}
    for row in selected:
        key = (
            str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            ),
            int(row["seed"]),
            str(row.get("canonical_bank_digest") or row["bank_id"]),
        )
        grouped.setdefault(key, []).append(float(row[metric]))
    keys = sorted(grouped)
    if len({key[0] for key in keys}) < 2:
        return _unassessed_comparison(
            f"{metric} lacks two independent clusters"
        )
    clusters = np.asarray([key[0] for key in keys])
    values = np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)
    zeros = np.zeros_like(values)
    result = paired_cluster_interval(clusters, values, zeros, resamples, seed)
    test = paired_sign_flip_test(clusters, values, zeros, seed + 1)
    return {**result, "p_value_two_sided": test["p_value_two_sided"]}


def _unassessed_comparison(reason):
    floor = -float(np.finfo(np.float64).max)
    return {
        "assessed": False,
        "cluster_count": 0,
        "paired_mean_difference": floor,
        "ci95_low": floor,
        "ci95_high": floor,
        "confidence_level": 0.95,
        "p_value_two_sided": 1.0,
        "reason": str(reason),
    }


def _effect_bins_complete(cgs_rows, effect_bin_rows):
    expected = {
        (int(row["seed"]), str(row["arm"]), str(row["bank_id"]))
        for row in cgs_rows
    }
    bins = ("low", "medium_low", "medium_high", "high")
    return bool(expected) and all(
        {
            row["effect_bin"]
            for row in effect_bin_rows
            if (int(row["seed"]), str(row["arm"]), str(row["bank_id"])) == key
        }
        == set(bins)
        for key in expected
    )


def _null_safety_by_arm(config, route_rows):
    margin = float(config["evaluation"]["null_score_equivalence_margin"])
    maximum_rate = float(config["evaluation"]["null_overclassification_rate_max"])
    resamples = int(config["qualification"]["bootstrap_resamples"])
    result = {}
    for arm_index, arm in enumerate(("endpoint", "alignment", "response", "full")):
        rows = [
            row
            for row in route_rows
            if row["route"] == "null"
            and row["arm"] == arm
            and row.get("condition_status") == "ASSESSED"
        ]
        cluster_ids = {
            str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            )
            for row in rows
        }
        if len(cluster_ids) < 2:
            result[arm] = {
                "base_map_cluster_count": len(cluster_ids),
                "passed": False,
                "reason": "fewer than two assessed independent null clusters",
            }
            continue
        bank_values = {}
        bank_rates = {}
        for row in rows:
            cluster = str(
                row.get("canonical_base_map_digest")
                or row["base_map_cluster_id"]
            )
            bank = str(row.get("canonical_bank_digest") or row["bank_id"])
            cell = (cluster, bank, int(row["seed"]))
            bank_values.setdefault(cell, []).append(
                row["matched_minus_alternative_mean"]
            )
            bank_rates.setdefault(cell, []).append(row["overclassification_rate"])
        clusters = sorted(cluster_ids)
        values = np.asarray(
            [
                np.mean(
                    [
                        np.mean(cell_values)
                        for (cluster, _, _), cell_values in bank_values.items()
                        if cluster == cluster_id
                    ]
                )
                for cluster_id in clusters
            ],
            dtype=np.float64,
        )
        rates = np.asarray(
            [
                np.mean(
                    [
                        np.mean(cell_values)
                        for (cluster, _, _), cell_values in bank_rates.items()
                        if cluster == cluster_id
                    ]
                )
                for cluster_id in clusters
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(91001 + arm_index)
        samples = np.empty(resamples, dtype=np.float64)
        rate_samples = np.empty(resamples, dtype=np.float64)
        for index in range(resamples):
            selected = rng.integers(0, len(clusters), size=len(clusters))
            samples[index] = float(np.mean(values[selected]))
            rate_samples[index] = float(np.mean(rates[selected]))
        low = float(np.percentile(samples, 2.5))
        high = float(np.percentile(samples, 97.5))
        rate = float(np.mean(rates))
        rate_high = float(np.percentile(rate_samples, 97.5))
        result[arm] = {
            "base_map_cluster_count": len(clusters),
            "mean_score_difference": float(np.mean(values)),
            "ci95_low": low,
            "ci95_high": high,
            "equivalence_margin": margin,
            "overclassification_rate": rate,
            "overclassification_rate_ci95_high": rate_high,
            "overclassification_rate_max": maximum_rate,
            "passed": bool(
                low >= -margin
                and high <= margin
                and rate_high <= maximum_rate
            ),
        }
    return result


def _response_null_safety(config, rows):
    selected = [row for row in rows if row.get("arm") in {"response", "full"}]
    observed_arms = {row.get("arm") for row in selected}
    maximum_rate = float(config["evaluation"]["response_null_violation_rate_max"])
    margin = float(config["evaluation"]["response_null_equivalence_margin"])
    passed = bool(
        observed_arms == {"response", "full"}
        and selected
        and all(
            row.get("native_null_violation_rate") is not None
            and row.get("native_latent_null_violation_rate") is not None
            and row.get("native_null_delta_rms_mean") is not None
            and row.get("native_latent_null_delta_rms_mean") is not None
            and float(row["native_null_violation_rate"]) <= maximum_rate
            and float(row["native_latent_null_violation_rate"]) <= maximum_rate
            and float(row["native_null_delta_rms_mean"]) <= margin
            and float(row["native_latent_null_delta_rms_mean"]) <= margin
            for row in selected
        )
    )
    return {
        "passed": passed,
        "row_count": len(selected),
        "observed_arms": sorted(str(arm) for arm in observed_arms),
        "required_arms": ["full", "response"],
        "null_violation_rate_max": maximum_rate,
        "null_equivalence_margin": margin,
    }
