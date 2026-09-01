from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .formal_baselines import (
    RidgeRegressor,
    RidgeSufficientStatistics,
    ZeroPreservingRidgeRegressor,
    ZeroPreservingRidgeSufficientStatistics,
    normalized_mse,
)
from .formal_config import public_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import QUALIFICATION_SCHEMA, bind_rows, complete_gate_vector, evidence_context
from .formal_features import (
    assemble_protocol_response_feature_matrix,
    assemble_protocol_response_features,
    multichannel_spatial_features,
    variant_features,
)
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
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
    load_teacher_bundle,
    masked_reconstruction_nmse,
    normalized_teacher_patches,
    save_teacher_bundle,
    teacher_targets,
    train_teacher_bundle,
)


QUALIFICATION_METHODS = (
    "no_x",
    "oracle_x",
    "map_edit_only",
    "csi_only",
    "variant_id_only",
)
ZERO_PRESERVING_METHODS = ("no_x", "oracle_x", "map_edit_only")
BASELINES = ("copy", "no_action", "action_swap")
QUALIFICATION_STREAM_BATCH_ROWS = 2048
QUALIFICATION_FINAL_ARTIFACT_NAMES = (
    "repeat_noise.csv",
    "route_noise_floor.csv",
    "route_coverage.csv",
    "teacher_qualification.csv",
    "response_scene_metrics.csv",
    "null_safety.csv",
    "response_gate.csv",
    "shortcut_audit.json",
    "gate.json",
    "manifest.json",
)
QUALIFICATION_RESUME_REQUIRED_FILES = (
    "data_contract.json",
    "teacher_resume.json",
    "checkpoints/stage0_csi_teacher.pt",
)
TEACHER_RESUME_RECEIPT_SCHEMA = "csi-pairs-stage0-resume-receipt-v1-v6"


LIVE_DATA_VERIFICATION_SCHEMA = "csi-pairs-v6-data-verification-gate-v1"
LIVE_DATA_VERIFICATION_MODE = "live_independent_regeneration"


def _is_live_data_verification(gate: dict | None) -> bool:
    return (
        isinstance(gate, dict)
        and gate.get("schema_version") == LIVE_DATA_VERIFICATION_SCHEMA
        and gate.get("verification_mode") == LIVE_DATA_VERIFICATION_MODE
        and gate.get("passed") is True
        and gate.get("blocking_passed") is True
    )


def _qualification_scientific_use(
    dataset: FormalDataset, passed: bool, data_verification_gate: dict | None
) -> str:
    if not passed or dataset.is_fixture:
        return "FORBIDDEN"
    if not _is_live_data_verification(data_verification_gate):
        return "FORBIDDEN"
    if dataset.metadata["scientific_use"] not in {"CANDIDATE", "QUALIFIED"}:
        return "FORBIDDEN"
    return "FORMAL_EXPERIMENT_ALLOWED"


SKIPPED_DATA_VERIFICATION = {
    "schema_version": "csi-pairs-v6-data-verification-skipped-v1",
    "status": "NOT_ASSESSED",
    "passed": None,
    "blocking_roles": [],
}


def run_formal_qualification(
    config: dict,
    dataset: FormalDataset,
    output_root: str | Path,
    data_verification_gate: dict | None = None,
    data_verification_gate_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    from .formal_data_verification import require_data_verification

    output_dir = Path(output_root) / "qualification"
    _prepare_qualification_output(output_dir, resume=resume)
    if data_verification_gate is None:
        data_verification_gate = dict(SKIPPED_DATA_VERIFICATION)
    elif data_verification_gate.get("passed") is True:
        data_verification_gate = require_data_verification(
            data_verification_gate,
            config,
            dataset,
            gate_path=data_verification_gate_path,
        )
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
        minimum_independent_source_final_unseen_clusters=int(
            data_config["minimum_independent_source_final_unseen_clusters"]
        ),
        minimum_independent_external_validation_clusters=int(
            data_config["minimum_independent_external_validation_clusters"]
        ),
    )
    provisional_evidence = evidence_context(config, dataset, "FORBIDDEN")
    contract = {**dataset.contract_report(), **provisional_evidence}
    contract_path = output_dir / "data_contract.json"
    if resume:
        if read_strict_json(contract_path) != contract:
            raise RuntimeError("qualification resume data contract no longer matches")
    else:
        write_json(contract_path, contract)

    blocking = qualification_blocking_scenes(dataset)
    train_scenes = blocking["teacher_train"]
    selection_scenes = blocking["method_selection"]
    patch_spec = PatchSpec.from_metadata(dataset.metadata)
    execution_device = resolve_execution_device(dataset)
    teacher_checkpoint = output_dir / "checkpoints" / "stage0_csi_teacher.pt"
    teacher_seed = int(config["seeds"][0]) + 1009
    teacher_receipt_path = output_dir / "teacher_resume.json"
    if resume:
        teacher_bundle = _load_resumed_teacher(
            teacher_checkpoint,
            teacher_receipt_path,
            config,
            provisional_evidence,
            teacher_seed,
            execution_device,
        )
    else:
        teacher_bundle = train_teacher_bundle(
            dataset.csi[train_scenes],
            patch_spec,
            config,
            seed=teacher_seed,
            device=execution_device,
        )
        save_teacher_bundle(teacher_checkpoint, teacher_bundle, config, teacher_seed)
        write_json(
            teacher_receipt_path,
            _teacher_resume_receipt(
                teacher_checkpoint, provisional_evidence, teacher_seed
            ),
        )
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
    selection_wrong_actions = _build_wrong_action_plan(
        dataset, selection_scenes, int(dataset.metadata["assets"]["material_category_count"])
    )
    selection_coverage_rows = route_coverage(dataset, routed_selection, selection_scenes)
    branch_rows = {
        row["scene_id"]: row
        for row in _branch_coverage(
            dataset,
            routed_selection,
            selection_scenes,
            wrong_action_plan=selection_wrong_actions,
        )
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

    train_wrong_actions = _build_wrong_action_plan(
        dataset, train_scenes, int(dataset.metadata["assets"]["material_category_count"])
    )
    probe_attempt = _reserve_qualification_probe_attempt(output_dir)
    models = _fit_qualification_probes_streaming(
        dataset,
        train_scenes,
        train_routed,
        teacher_bundle.mask_bank,
        train_wrong_actions,
        qualification,
    )
    for method, model in models.items():
        model.save(
            probe_attempt / f"{method}.npz",
            {
                "schema_version": "csi-pairs-v6-qualification-probe-v2",
                "input_contract": _method_contract(method),
                "target": "source-normalized clean physical CSI patch delta",
                "source_roles": ["source_encoder_train"],
                "status": "PRETRAINING_QUALIFICATION_PROBE_NOT_HEADLINE_MODEL",
                **provisional_evidence,
            },
        )
    selection_records, selection_target, predictions = (
        _predict_qualification_probes_streaming(
            dataset,
            selection_scenes,
            routed_selection,
            teacher_bundle.mask_bank,
            selection_wrong_actions,
            models,
        )
    )
    physical_response_checkpoint = None
    if dataset.is_fixture:
        physical_response_audit = {
            "status": "NOT_ASSESSED_FIXTURE_FORBIDDEN",
            "contract": "ridge-fixture-engineering-path-only",
            "formal_physical_response_required_for_nonfixture": True,
        }
    else:
        from .formal_action_inverse_response import (
            PHYSICAL_RESPONSE_CONTRACT,
            fit_and_predict_qualification_response,
            load_physical_response_checkpoint,
            save_physical_response_checkpoint,
        )

        (
            physical_response_model,
            physical_keys,
            physical_no_x,
            physical_action_swap,
            physical_oracle_x,
            physical_wrong_action_status,
            physical_response_audit,
        ) = fit_and_predict_qualification_response(
            dataset,
            train_scenes,
            selection_scenes,
            train_routed,
            routed_selection,
            config,
            train_wrong_actions,
            selection_wrong_actions,
            seed=int(config["seeds"][0]),
            device=execution_device,
        )
        expected_keys = list(
            zip(
                selection_records["scene"].astype(np.int64).tolist(),
                selection_records["source_world"].astype(np.int64).tolist(),
                selection_records["target_world"].astype(np.int64).tolist(),
                selection_records["bit"].astype(np.int64).tolist(),
                selection_records["position"].astype(np.int64).tolist(),
                selection_records["query"].astype(np.int64).tolist(),
                strict=True,
            )
        )
        if physical_keys != expected_keys:
            raise RuntimeError(
                "physical response rows do not align with qualification records"
            )
        if not np.array_equal(
            physical_wrong_action_status,
            selection_records["wrong_action_match_status"],
        ):
            raise RuntimeError(
                "physical response wrong-action statuses do not align with qualification"
            )
        if (
            physical_no_x.shape != selection_target.shape
            or physical_action_swap.shape != selection_target.shape
            or physical_oracle_x.shape != selection_target.shape
        ):
            raise RuntimeError(
                "physical response predictions differ from qualification target shape"
            )
        predictions["ridge_no_x_diagnostic"] = predictions["no_x"]
        predictions["ridge_action_swap_diagnostic"] = predictions["action_swap"]
        predictions["ridge_oracle_x_diagnostic"] = predictions["oracle_x"]
        predictions["no_x"] = physical_no_x
        predictions["action_swap"] = physical_action_swap
        predictions["oracle_x"] = physical_oracle_x
        physical_response_checkpoint = probe_attempt / "physical_response.pt"
        checkpoint_bindings = {
            "dataset": str(dataset.source_path),
            "dataset_sha256": sha256_file(dataset.source_path),
            "formal_config_sha256": provisional_evidence["config_sha256"],
            "train_bank_ids": [
                str(dataset.bank_ids[int(scene)]) for scene in train_scenes
            ],
            "selection_bank_ids": [
                str(dataset.bank_ids[int(scene)]) for scene in selection_scenes
            ],
            "physical_config_sha256": physical_response_audit[
                "physical_config_sha256"
            ],
            "candidate_config": physical_response_audit["candidate_config"],
            "candidate_config_sha256": physical_response_audit[
                "candidate_config_sha256"
            ],
            "model_source": str(
                (
                    Path(__file__).resolve().parent
                    / "formal_action_inverse_response.py"
                ).resolve()
            ),
            "model_source_sha256": sha256_file(
                Path(__file__).resolve().parent
                / "formal_action_inverse_response.py"
            ),
        }
        save_physical_response_checkpoint(
            physical_response_checkpoint,
            physical_response_model,
            checkpoint_bindings,
        )
        reloaded_model, reloaded_bindings = load_physical_response_checkpoint(
            physical_response_checkpoint
        )
        if (
            reloaded_bindings != checkpoint_bindings
            or len(reloaded_model.gates) != len(physical_response_model.gates)
            or reloaded_model.thresholds != physical_response_model.thresholds
        ):
            raise RuntimeError("physical response checkpoint round trip failed")
        physical_response_audit.update(
            {
                "status": "FORMAL_QUALIFICATION_MODEL_FIT",
                "contract": PHYSICAL_RESPONSE_CONTRACT,
                "checkpoint": str(physical_response_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(physical_response_checkpoint),
                "checkpoint_round_trip_valid": True,
                "bindings": checkpoint_bindings,
            }
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
    scientific_use = _qualification_scientific_use(
        dataset, passed, data_verification_gate
    )
    evidence = evidence_context(config, dataset, scientific_use)
    final_paths = _qualification_final_paths(output_dir)
    if any(path.exists() or path.is_symlink() for path in final_paths):
        raise FileExistsError("qualification refuses to overwrite partial final evidence")
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
        "qualification_probe_attempt": str(probe_attempt.relative_to(output_dir)),
        "qualification_probe_checkpoint_sha256": {
            method: sha256_file(probe_attempt / f"{method}.npz")
            for method in QUALIFICATION_METHODS
        },
        "physical_response": physical_response_audit,
        "primary_route_contract": PRIMARY_ROUTE_CONTRACT,
        "external_validity": {
            "available": bool(dataset.metadata["external_reference"]["available"]),
            "status": "NOT_ASSESSED",
            "blocking_for_real_causal_claim": True,
        },
        "data_verification_gate_schema": data_verification_gate["schema_version"],
        "data_verification_blocking_roles": data_verification_gate["blocking_roles"],
        "data_verification_mode": data_verification_gate.get("verification_mode"),
        "data_verification_passed": data_verification_gate.get("passed"),
        "data_verification_gate_sha256": (
            sha256_file(data_verification_gate_path)
            if data_verification_gate_path
            and Path(data_verification_gate_path).is_file()
            and _is_live_data_verification(data_verification_gate)
            else None
        ),
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
        "decision_rule": "Source-encoder-train banks determine route normalization, the frozen teacher/readout, physical inverse-response coefficient/gate fitting, and per-bank train-route coverage. Alignment primary inclusion uses full-channel physical distance only; Response primary inclusion uses query-patch physical distance only. The formal no-X response uses visible source CSI, source map, ID-free action geometry, radio/BS context, mask, and query; its internal receiver hypothesis is inferred from source CSI and is never supplied ground-truth receiver position. Teacher sensitivity is measured after the source-train-frozen readout decodes normalized CSI; it is an independent audit stratum and auxiliary G2 qualification condition, never a primary inclusion rule. Source-method-selection banks determine repeat noise, same-unit noise floors, branch, teacher, and full-position response qualification. Target, external, probe, calibration, and final-unseen banks are unread.",
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


def _qualification_final_paths(output_dir: Path) -> tuple[Path, ...]:
    return tuple(output_dir / name for name in QUALIFICATION_FINAL_ARTIFACT_NAMES)


def _prepare_qualification_output(output_dir: Path, *, resume: bool) -> None:
    """Fail closed before training can overwrite an earlier qualification attempt."""
    if output_dir.is_symlink():
        raise RuntimeError("qualification output directory cannot be a symbolic link")
    if resume:
        if not output_dir.is_dir():
            raise RuntimeError("qualification resume requires an existing qualification directory")
        if any(path.exists() or path.is_symlink() for path in _qualification_final_paths(output_dir)):
            raise RuntimeError("qualification resume refuses completed or partial final evidence")
        for relative in QUALIFICATION_RESUME_REQUIRED_FILES:
            path = output_dir / relative
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"qualification resume requires a regular authenticated artifact: {relative}"
                )
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise RuntimeError("qualification resume checkpoint directory is invalid")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    owned_paths = (
        output_dir / "data_contract.json",
        output_dir / "teacher_resume.json",
        output_dir / "checkpoints",
        *_qualification_final_paths(output_dir),
    )
    if any(path.exists() or path.is_symlink() for path in owned_paths):
        raise FileExistsError("qualification refuses to overwrite an earlier attempt")


def _reserve_qualification_probe_attempt(output_dir: Path) -> Path:
    probe_root = output_dir / "checkpoints" / "qualification_probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    if probe_root.is_symlink() or not probe_root.is_dir():
        raise RuntimeError("qualification probe directory is invalid")
    attempt_index = 1
    while True:
        attempt = probe_root / f"attempt_{attempt_index:04d}"
        try:
            attempt.mkdir(exist_ok=False)
        except FileExistsError:
            attempt_index += 1
            continue
        return attempt


def _teacher_resume_receipt(
    teacher_checkpoint: Path,
    evidence: dict[str, object],
    teacher_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": TEACHER_RESUME_RECEIPT_SCHEMA,
        **evidence,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_seed": int(teacher_seed),
    }


def _load_resumed_teacher(
    teacher_checkpoint: Path,
    teacher_receipt_path: Path,
    config: dict,
    evidence: dict[str, object],
    teacher_seed: int,
    execution_device: str | torch.device,
) -> TeacherBundle:
    receipt = read_strict_json(teacher_receipt_path)
    expected = _teacher_resume_receipt(
        teacher_checkpoint, evidence, teacher_seed
    )
    if receipt != expected:
        raise RuntimeError("qualification resume teacher receipt no longer matches")
    teacher_bundle = load_teacher_bundle(
        teacher_checkpoint, config, device=execution_device
    )
    if teacher_bundle.seed != int(teacher_seed):
        raise RuntimeError("qualification resume teacher seed no longer matches")
    return teacher_bundle


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
        decoded_distances = _frozen_readout_alignment_distances(
            dataset, bundle, routed, scene
        )
        patches = normalized_teacher_patches(bundle, dataset.csi[scene])
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
            (
                float(distances[0]),
                float(decoded_distances[key]),
            )
            for key, distances in routed.alignment_distances.items()
            if key[0] == scene
        ]
        physical_active = [
            value
            for value in alignment_items
            if value[0] >= float(q["physical_active_rms_min"])
        ]
        physical_null = [
            value
            for value in alignment_items
            if value[0] <= float(q["physical_null_rms_max"])
        ]
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
                "teacher_sensitivity_representation": (
                    "source-train-frozen-readout-decoded-normalized-csi"
                ),
                "physical_active_teacher_sensitive_agreement": active_agreement,
                "physical_null_teacher_null_agreement": null_agreement,
                "passed": scene_passed,
            }
        )
    return rows, bool(passed)


def _frozen_readout_alignment_distances(
    dataset,
    bundle: TeacherBundle,
    routed,
    scene: int,
    *,
    batch_rows: int = 32_768,
) -> dict[tuple[int, int, int, int], float]:
    """Measure recoverable CSI change without fitting on selection banks."""
    latent = np.asarray(routed.teacher_latent[int(scene)], dtype=np.float32)
    if latent.ndim != 4:
        raise ValueError("routed teacher latent must be [world,position,patch,dim]")
    flat = latent.reshape(-1, latent.shape[-1])
    decoded = []
    device = module_device(bundle.readout)
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(batch_rows)):
            stop = min(start + int(batch_rows), flat.shape[0])
            decoded.append(
                bundle.readout(
                    torch.as_tensor(
                        flat[start:stop], dtype=torch.float32, device=device
                    )
                )
                .cpu()
                .numpy()
            )
    reconstructed = np.concatenate(decoded, axis=0).reshape(
        *latent.shape[:-1], -1
    )
    distances = {}
    for edge in dataset.directed_edges(int(scene)):
        delta = (
            reconstructed[int(edge.target_world)]
            - reconstructed[int(edge.source_world)]
        )
        values = np.sqrt(np.mean(delta.astype(np.float64) ** 2, axis=(-2, -1)))
        for position, value in enumerate(values.tolist()):
            distances[
                (
                    int(scene),
                    int(edge.source_world),
                    int(edge.target_world),
                    int(position),
                )
            ] = float(value)
    expected = {
        key
        for key in routed.alignment_distances
        if int(key[0]) == int(scene)
    }
    if set(distances) != expected:
        raise RuntimeError(
            "frozen teacher readout rows do not align with routed qualification rows"
        )
    if not all(np.isfinite(value) for value in distances.values()):
        raise RuntimeError("frozen teacher readout produced nonfinite distances")
    return distances


def _fit_qualification_probes_streaming(
    dataset,
    scenes,
    routed,
    mask_bank,
    wrong_action_plan,
    qualification,
) -> dict[str, RidgeRegressor | ZeroPreservingRidgeRegressor]:
    alpha = float(qualification["ridge"])
    statistics = {
        method: (
            ZeroPreservingRidgeSufficientStatistics(alpha)
            if method in ZERO_PRESERVING_METHODS
            else RidgeSufficientStatistics(alpha)
        )
        for method in QUALIFICATION_METHODS
    }
    observed = 0
    for batch in _iter_qualification_record_batches(
        dataset,
        scenes,
        routed,
        mask_bank,
        wrong_action_plan=wrong_action_plan,
    ):
        target = batch["target_delta"]
        observed += int(target.shape[0])
        for method, accumulator in statistics.items():
            if method in ZERO_PRESERVING_METHODS:
                accumulator.update(
                    batch[method], batch[f"{method}_zero_action"], target
                )
            else:
                accumulator.update(batch[method], target)
    expected = _qualification_record_count(dataset, scenes, mask_bank)
    if observed != expected or observed <= 0:
        raise RuntimeError(
            f"qualification stream emitted {observed} records; expected {expected}"
        )
    return {method: accumulator.finalize() for method, accumulator in statistics.items()}


def _predict_qualification_probes_streaming(
    dataset,
    scenes,
    routed,
    mask_bank,
    wrong_action_plan,
    models,
):
    index_chunks = {
        name: []
        for name in (
            "scene",
            "source_world",
            "target_world",
            "bit",
            "position",
            "query",
            "route",
            "wrong_action_match_status",
        )
    }
    target_chunks = []
    prediction_chunks = {
        name: [] for name in (*QUALIFICATION_METHODS, *BASELINES)
    }
    observed = 0
    for batch in _iter_qualification_record_batches(
        dataset,
        scenes,
        routed,
        mask_bank,
        wrong_action_plan=wrong_action_plan,
    ):
        target = batch["target_delta"]
        observed += int(target.shape[0])
        target_chunks.append(target)
        for name in index_chunks:
            index_chunks[name].append(batch[name])
        for method, model in models.items():
            if method in ZERO_PRESERVING_METHODS:
                prediction = model.predict_contrast(
                    batch[method], batch[f"{method}_zero_action"]
                )
            else:
                prediction = model.predict(batch[method])
            prediction_chunks[method].append(prediction)
        zeros = np.zeros_like(prediction_chunks["no_x"][-1])
        prediction_chunks["copy"].append(zeros)
        prediction_chunks["no_action"].append(zeros.copy())
        prediction_chunks["action_swap"].append(
            models["no_x"].predict_contrast(
                batch["action_swap"], batch["no_x_zero_action"]
            )
        )
    expected = _qualification_record_count(dataset, scenes, mask_bank)
    if observed != expected or observed <= 0:
        raise RuntimeError(
            f"qualification prediction stream emitted {observed} records; expected {expected}"
        )
    records = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in index_chunks.items()
    }
    target = np.concatenate(target_chunks, axis=0)
    predictions = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in prediction_chunks.items()
    }
    return records, target, predictions


def _qualification_record_count(dataset, scenes, mask_bank) -> int:
    query_count = len({int(entry.query) for entry in mask_bank if entry.mode == "random_75"})
    if query_count <= 0:
        return 0
    edge_count = sum(
        1 for scene in np.asarray(scenes) for _edge in dataset.directed_edges(int(scene))
    )
    return int(edge_count * dataset.position_count * query_count)


def _iter_qualification_record_batches(
    dataset,
    scenes,
    routed,
    mask_bank,
    *,
    wrong_action_plan=None,
    batch_rows: int = QUALIFICATION_STREAM_BATCH_ROWS,
):
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError("qualification stream batch_rows must be a positive integer")
    pending: dict[str, list[np.ndarray]] = {}
    pending_rows = 0
    for unit in _iter_qualification_record_units(
        dataset,
        scenes,
        routed,
        mask_bank,
        wrong_action_plan=wrong_action_plan,
    ):
        if not pending:
            pending = {name: [] for name in unit}
        if set(unit) != set(pending):
            raise RuntimeError("qualification record unit fields changed")
        for name, values in unit.items():
            pending[name].append(values)
        pending_rows += int(unit["target_delta"].shape[0])
        if pending_rows >= batch_rows:
            yield {
                name: np.concatenate(chunks, axis=0)
                for name, chunks in pending.items()
            }
            pending = {}
            pending_rows = 0
    if pending_rows:
        yield {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in pending.items()
        }


def _iter_qualification_record_units(
    dataset,
    scenes,
    routed,
    mask_bank,
    *,
    wrong_action_plan=None,
):
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    wrong_actions = wrong_action_plan or _build_wrong_action_plan(
        dataset, scenes, material_categories
    )
    patch_spec = PatchSpec.from_metadata(dataset.metadata)
    patch_mean = patchify_csi(routed.normalization.channel_mean, patch_spec)
    patch_scale = patchify_csi(routed.normalization.channel_scale, patch_spec)
    query_entries = {
        int(entry.query): entry for entry in mask_bank if entry.mode == "random_75"
    }
    if set(query_entries) != set(range(patch_spec.patch_count)):
        raise RuntimeError("qualification mask bank must cover every response query")
    queries = np.arange(patch_spec.patch_count, dtype=np.int64)
    mask_rows = np.vstack([query_entries[int(query)].mask for query in queries])
    zero_action = zero_typed_edit((), dataset.maps.shape[-1], material_categories)
    zero_action_features = multichannel_spatial_features(zero_action)
    for scene_value in scenes:
        scene = int(scene_value)
        normalized_patches = (
            routed.physical_patches[scene] - patch_mean
        ) / patch_scale
        position_center = dataset.positions[scene].mean(axis=0)
        position_scale = dataset.positions[scene].std(axis=0)
        position_scale[position_scale < 1e-9] = 1.0
        complete_context = np.concatenate(
            (dataset.radio_config[scene], dataset.bs_pose[scene])
        )
        map_features = {
            world: multichannel_spatial_features(dataset.maps[scene, world])
            for world in range(dataset.world_count)
        }
        for edge in dataset.directed_edges(scene):
            action_features = wrong_actions["action_features"][
                (scene, edge.source_world, edge.bit_index)
            ]
            variant = variant_features(
                dataset.world_bits[edge.source_world],
                dataset.world_bits[edge.target_world],
            )
            for position in range(dataset.position_count):
                swap_action_features, swap_status, _swap_world = wrong_actions[
                    "selections"
                ][(scene, edge.source_world, edge.bit_index, position)]
                source_patches = normalized_patches[edge.source_world, position]
                target_patches = normalized_patches[edge.target_world, position]
                visible_rows = np.broadcast_to(
                    source_patches,
                    (patch_spec.patch_count, *source_patches.shape),
                ).copy()
                for query in queries:
                    visible_rows[int(query), mask_rows[int(query)]] = 0.0
                normalized_position = (
                    dataset.positions[scene, position] - position_center
                ) / position_scale
                common = (visible_rows, map_features[edge.source_world])
                no_x = assemble_protocol_response_feature_matrix(
                    *common,
                    action_features,
                    complete_context,
                    mask_rows,
                    queries,
                )
                no_x_zero = assemble_protocol_response_feature_matrix(
                    *common,
                    zero_action_features,
                    complete_context,
                    mask_rows,
                    queries,
                )
                yield {
                    "scene": np.full(patch_spec.patch_count, scene, dtype=np.int64),
                    "source_world": np.full(
                        patch_spec.patch_count,
                        int(edge.source_world),
                        dtype=np.int64,
                    ),
                    "target_world": np.full(
                        patch_spec.patch_count,
                        int(edge.target_world),
                        dtype=np.int64,
                    ),
                    "bit": np.full(
                        patch_spec.patch_count,
                        int(edge.bit_index),
                        dtype=np.int64,
                    ),
                    "position": np.full(
                        patch_spec.patch_count,
                        int(position),
                        dtype=np.int64,
                    ),
                    "query": queries.copy(),
                    "route": np.asarray(
                        [
                            routed.response_route[
                                (
                                    scene,
                                    edge.source_world,
                                    edge.target_world,
                                    position,
                                    int(query),
                                )
                            ]
                            for query in queries
                        ],
                        dtype=np.int8,
                    ),
                    "wrong_action_match_status": np.full(
                        patch_spec.patch_count, swap_status, dtype="U8"
                    ),
                    "target_delta": target_patches - source_patches,
                    "no_x": no_x,
                    "no_x_zero_action": no_x_zero,
                    "oracle_x": assemble_protocol_response_feature_matrix(
                        *common,
                        action_features,
                        complete_context,
                        mask_rows,
                        queries,
                        position=normalized_position,
                    ),
                    "oracle_x_zero_action": assemble_protocol_response_feature_matrix(
                        *common,
                        zero_action_features,
                        complete_context,
                        mask_rows,
                        queries,
                        position=normalized_position,
                    ),
                    "map_edit_only": assemble_protocol_response_feature_matrix(
                        *common,
                        action_features,
                        complete_context,
                        mask_rows,
                        queries,
                        include_csi=False,
                    ),
                    "map_edit_only_zero_action": assemble_protocol_response_feature_matrix(
                        *common,
                        zero_action_features,
                        complete_context,
                        mask_rows,
                        queries,
                        include_csi=False,
                    ),
                    "csi_only": assemble_protocol_response_feature_matrix(
                        visible_rows,
                        None,
                        None,
                        complete_context,
                        mask_rows,
                        queries,
                    ),
                    "variant_id_only": np.broadcast_to(
                        variant, (patch_spec.patch_count, variant.size)
                    ).copy(),
                    "action_swap": assemble_protocol_response_feature_matrix(
                        *common,
                        swap_action_features,
                        complete_context,
                        mask_rows,
                        queries,
                    ),
                }


def _build_records(dataset, scenes, routed, mask_bank, *, wrong_action_plan=None):
    records = []
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    wrong_actions = wrong_action_plan or _build_wrong_action_plan(
        dataset, scenes, material_categories
    )
    patch_spec = PatchSpec.from_metadata(dataset.metadata)
    patch_mean = patchify_csi(routed.normalization.channel_mean, patch_spec)
    patch_scale = patchify_csi(routed.normalization.channel_scale, patch_spec)
    query_masks = {
        int(entry.query): entry for entry in mask_bank if entry.mode == "random_75"
    }
    for scene_value in scenes:
        scene = int(scene_value)
        normalized_patches = (
            routed.physical_patches[scene] - patch_mean
        ) / patch_scale
        position_center = dataset.positions[scene].mean(axis=0)
        position_scale = dataset.positions[scene].std(axis=0)
        position_scale[position_scale < 1e-9] = 1.0
        complete_context = np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene]))
        map_features = {
            world: multichannel_spatial_features(dataset.maps[scene, world])
            for world in range(dataset.world_count)
        }
        zero_action = zero_typed_edit(
            (), dataset.maps.shape[-1], material_categories
        )
        zero_action_features = multichannel_spatial_features(zero_action)
        for edge in dataset.directed_edges(scene):
            source_map = dataset.maps[scene, edge.source_world]
            target_map = dataset.maps[scene, edge.target_world]
            action_features = wrong_actions["action_features"][
                (scene, edge.source_world, edge.bit_index)
            ]
            variant_id = variant_features(
                dataset.world_bits[edge.source_world], dataset.world_bits[edge.target_world]
            )
            for position in range(dataset.position_count):
                swap_action_features, swap_status, swap_world = wrong_actions[
                    "selections"
                ][(scene, edge.source_world, edge.bit_index, position)]
                source_patches = normalized_patches[edge.source_world, position]
                target_patches = normalized_patches[edge.target_world, position]
                normalized_position = (dataset.positions[scene, position] - position_center) / position_scale
                for query in range(source_patches.shape[0]):
                    entry = query_masks[query]
                    visible = source_patches.copy()
                    visible[entry.mask] = 0.0
                    route = routed.response_route[(scene, edge.source_world, edge.target_world, position, query)]
                    records.append(
                        {
                            "scene": scene,
                            "bank_id": str(dataset.bank_ids[scene]),
                            "route": route,
                            "target_delta": target_patches[query] - source_patches[query],
                            "no_x": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], action_features,
                                complete_context, entry.mask, query
                            ),
                            "no_x_zero_action": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], zero_action_features,
                                complete_context, entry.mask, query
                            ),
                            "oracle_x": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], action_features,
                                complete_context, entry.mask, query,
                                position=normalized_position,
                            ),
                            "oracle_x_zero_action": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], zero_action_features,
                                complete_context, entry.mask, query,
                                position=normalized_position,
                            ),
                            "no_action": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], zero_action_features,
                                complete_context, entry.mask, query
                            ),
                            "map_edit_only": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], action_features,
                                complete_context, entry.mask, query, include_csi=False
                            ),
                            "map_edit_only_zero_action": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], zero_action_features,
                                complete_context, entry.mask, query, include_csi=False
                            ),
                            "csi_only": assemble_protocol_response_features(
                                visible, None, None, complete_context, entry.mask, query
                            ),
                            "variant_id_only": variant_id,
                            "action_swap": assemble_protocol_response_features(
                                visible, map_features[edge.source_world], swap_action_features,
                                complete_context, entry.mask, query
                            ),
                            "wrong_action_match_status": swap_status,
                            "wrong_action_world": swap_world,
                        }
                    )
    return records


def _response_gate_rows(dataset, records, target, predictions, config):
    metric_rows = []
    null_rows = []
    gate_rows = []
    routes = _record_column(records, "route")
    wrong_action_status = _record_column(records, "wrong_action_match_status")
    scene_values = _record_column(records, "scene").astype(np.int64, copy=False)
    for scene in sorted(set(int(value) for value in scene_values)):
        scene_mask = scene_values == scene
        active = scene_mask & (routes == 2)
        null = scene_mask & (routes == 0)
        active_scores = {}
        for method, prediction in predictions.items():
            for stratum, mask in (("all", scene_mask), ("active", active), ("null", null)):
                score = _optional_descriptive_nmse(target[mask], prediction[mask])
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


def _optional_descriptive_nmse(
    target: np.ndarray, prediction: np.ndarray
) -> float | None:
    """Return N/A for an empty or zero-energy descriptive denominator."""
    target_values = np.asarray(target)
    prediction_values = np.asarray(prediction)
    if target_values.shape != prediction_values.shape:
        raise ValueError("descriptive NMSE target/prediction shapes differ")
    if target_values.size == 0:
        return None
    denominator = float(np.sum(target_values**2))
    if not np.isfinite(denominator):
        raise RuntimeError("descriptive NMSE target energy is non-finite")
    if denominator <= 1e-15:
        return None
    score = normalized_mse(target_values, prediction_values)
    if not np.isfinite(score):
        raise RuntimeError("descriptive NMSE is non-finite on a positive denominator")
    return float(score)


def _branch_coverage(dataset, routed, scenes, *, wrong_action_plan=None):
    rows = []
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    wrong_actions = wrong_action_plan or _build_wrong_action_plan(
        dataset, scenes, material_categories
    )
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
                    _, status, _ = wrong_actions["selections"][
                        (scene, source, edge.bit_index, position)
                    ]
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


def _build_wrong_action_plan(
    dataset,
    scenes,
    material_categories: int,
    *,
    positions: np.ndarray | tuple[int, ...] | list[int] | None = None,
) -> dict:
    """Precompute equivalent compact wrong-action choices once per scene."""
    position_values = (
        tuple(range(dataset.position_count))
        if positions is None
        else tuple(int(value) for value in np.asarray(positions).reshape(-1))
    )
    if (
        not position_values
        or len(set(position_values)) != len(position_values)
        or min(position_values) < 0
        or max(position_values) >= dataset.position_count
    ):
        raise ValueError("wrong-action plan positions must be unique in-range indices")
    representation = dataset.metadata.get("representation", {})
    origin = representation.get("map_origin_xy_m")
    resolution = representation.get("map_resolution_m")
    receiver_geometry_available = bool(
        origin is not None
        and resolution is not None
        and np.isfinite(float(resolution))
        and float(resolution) > 0.0
    )
    action_features = {}
    selections = {}
    for scene_value in np.asarray(scenes):
        scene = int(scene_value)
        actions = {}
        for edge in dataset.directed_edges(scene):
            key = (int(edge.source_world), int(edge.bit_index))
            action = typed_signed_edit(
                dataset.maps[scene, edge.source_world],
                dataset.maps[scene, edge.target_world],
                dataset.map_channel_names,
                material_categories,
            )
            actions[key] = {
                "features": multichannel_spatial_features(action),
                "profile": _action_geometry_profile(action),
                "target_world": int(edge.target_world),
            }
            action_features[(scene, *key)] = actions[key]["features"]
        for (source_world, correct_bit), reference in actions.items():
            candidates = [
                candidate
                for (candidate_source, candidate_bit), candidate in actions.items()
                if candidate_source == source_world and candidate_bit != correct_bit
            ]
            if not candidates:
                zero = np.zeros_like(reference["features"])
                for position in position_values:
                    selections[(scene, source_world, correct_bit, position)] = (
                        zero,
                        "failed",
                        None,
                    )
                continue
            for position in position_values:
                receiver = (
                    dataset.positions[scene, position]
                    if receiver_geometry_available
                    else None
                )
                scored = [
                    (
                        candidate,
                        _wrong_action_profile_distance(
                            reference["profile"],
                            candidate["profile"],
                            receiver_position=receiver,
                            map_origin_xy_m=(origin if receiver is not None else None),
                            map_resolution_m=(resolution if receiver is not None else None),
                        ),
                    )
                    for candidate in candidates
                ]
                exact = [
                    item
                    for item in scored
                    if receiver is not None and item[1][0] == 0 and item[1][1] == 0
                ]
                fallback = [item for item in scored if item[1][0] == 0]
                if exact:
                    selected, _ = min(exact, key=lambda item: item[1])
                    status = "exact"
                elif fallback:
                    selected, _ = min(fallback, key=lambda item: item[1])
                    status = "fallback"
                else:
                    selected, _ = min(scored, key=lambda item: item[1])
                    status = "failed"
                selections[(scene, source_world, correct_bit, position)] = (
                    selected["features"],
                    status,
                    selected["target_world"],
                )
    return {"action_features": action_features, "selections": selections}


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
    return _wrong_action_profile_distance(
        reference_profile,
        candidate_profile,
        receiver_position=receiver_position,
        map_origin_xy_m=map_origin_xy_m,
        map_resolution_m=map_resolution_m,
    )


def _wrong_action_profile_distance(
    reference_profile: dict,
    candidate_profile: dict,
    *,
    receiver_position: np.ndarray | None = None,
    map_origin_xy_m: np.ndarray | None = None,
    map_resolution_m: float | None = None,
) -> tuple:
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
    reference_distances = _profile_receiver_distances(
        reference_profile,
        receiver_position,
        map_origin_xy_m,
        map_resolution_m,
    )
    candidate_distances = _profile_receiver_distances(
        candidate_profile,
        receiver_position,
        map_origin_xy_m,
        map_resolution_m,
    )
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
                    reference_profile["active_planes"],
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


def _profile_receiver_distances(
    profile: dict,
    receiver_position: np.ndarray | None,
    map_origin_xy_m: np.ndarray | None,
    map_resolution_m: float | None,
) -> tuple[float, ...] | None:
    cached = profile["receiver_plane_distances_m"]
    if cached is not None:
        return cached
    if (
        receiver_position is None
        or map_origin_xy_m is None
        or map_resolution_m is None
    ):
        return None
    receiver_xy = np.asarray(receiver_position, dtype=np.float64).reshape(-1)[:2]
    origin = np.asarray(map_origin_xy_m, dtype=np.float64).reshape(-1)[:2]
    resolution = float(map_resolution_m)
    return tuple(
        0.0
        if centroid is None
        else float(
            np.linalg.norm(
                receiver_xy
                - (
                    origin
                    + resolution
                    * np.asarray([centroid[1] + 0.5, centroid[0] + 0.5])
                )
            )
        )
        for centroid in profile["plane_centroids_rc"]
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
    material_categories = (values.shape[0] - 4) // 2
    if values.shape[0] != 4 + 2 * material_categories:
        raise ValueError("typed action channel count is inconsistent")
    raw_active = np.any(np.abs(values) > 1e-12, axis=(1, 2))
    source_materials = tuple(
        int(index) for index in np.flatnonzero(raw_active[4 : 4 + material_categories])
    )
    target_materials = tuple(
        int(index) for index in np.flatnonzero(raw_active[4 + material_categories :])
    )
    family = (
        bool(raw_active[0] or raw_active[1]),
        bool(raw_active[2] or raw_active[3]),
        source_materials,
        target_materials,
    )
    geometry_planes = (
        np.maximum(np.abs(values[0]), np.abs(values[1])),
        np.maximum(np.abs(values[2]), np.abs(values[3])),
        np.max(np.abs(values[4:]), axis=0),
    )
    active_planes = np.asarray(
        [np.any(plane > 1e-12) for plane in geometry_planes], dtype=np.bool_
    )
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
    plane_centroids_rc = []
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
    for plane in geometry_planes:
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
            centroid_rc = plane_coordinates.mean(axis=0)
            plane_centroids_rc.append(
                tuple(float(value) for value in centroid_rc.tolist())
            )
            if geometry_available:
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
            plane_centroids_rc.append(None)
            receiver_plane_distances.append(0.0)
    return {
        "family": family,
        "active_planes": tuple(bool(value) for value in active_planes.tolist()),
        "area": int(np.sum(support)),
        "norm": float(np.linalg.norm(values)),
        "bbox_shape": bbox_shape,
        "plane_areas": tuple(plane_areas),
        "plane_norms": tuple(plane_norms),
        "plane_bbox_shapes": tuple(plane_bbox_shapes),
        "plane_canonical_patterns": tuple(plane_canonical_patterns),
        "plane_centroids_rc": tuple(plane_centroids_rc),
        "receiver_plane_distances_m": (
            tuple(receiver_plane_distances) if geometry_available else None
        ),
    }


def _shortcut_warnings(records, target, predictions, config):
    routes = _record_column(records, "route")
    active = routes == 2
    copy = normalized_mse(target[active], predictions["copy"][active])
    variant = normalized_mse(target[active], predictions["variant_id_only"][active])
    csi_only = normalized_mse(target[active], predictions["csi_only"][active])
    map_edit_only = normalized_mse(target[active], predictions["map_edit_only"][active])
    no_x = normalized_mse(target[active], predictions["no_x"][active])
    oracle = normalized_mse(target[active], predictions["oracle_x"][active])
    threshold = float(config["shortcut_relative_improvement_max"])
    variant_signal = bool(_relative_improvement(variant, copy) > threshold)
    conditional_gain = _relative_improvement(no_x, csi_only)
    csi_signal = bool(conditional_gain <= threshold)
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
                "no_x_relative_improvement_vs_csi_only": conditional_gain,
                "csi_only_relative_improvement_vs_copy": _relative_improvement(csi_only, copy),
                "criterion": "full map-and-action response must improve over CSI-only on the same active denominator",
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


def _record_column(records, name: str) -> np.ndarray:
    if isinstance(records, dict):
        if name not in records:
            raise KeyError(f"qualification record index is missing {name}")
        values = np.asarray(records[name])
    else:
        values = np.asarray([row[name] for row in records])
    if values.ndim != 1:
        raise ValueError(f"qualification record column {name} must be one-dimensional")
    return values


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
