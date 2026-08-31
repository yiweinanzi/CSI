from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import (
    EVIDENCE_AUTH_KEYS,
    RUNTIME_PROVENANCE_FIELDS,
    RUNTIME_PROVENANCE_SCHEMA,
    TORCH_RUNTIME_FIELDS,
    evidence_context,
)
from .formal_config import ARMS
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_json
from .formal_statistics import (
    holm_adjust,
    interval_decision,
    paired_cluster_interval,
    paired_sign_flip_test,
)


SHUFFLED_SYSTEMS = (
    "matched_model",
    "shuffled_model",
    "constant",
    "csi_only",
    "map_only",
    "scene_id_only",
    "edit_status_xor",
    "variant_id_matcher",
)
SHUFFLED_METRICS = ("alignment_cgs", "response_probe")
PAIR_LABELS = ("positive", "negative")
RETENTION_CONDITIONS = ("correct", "map_swap", "map_removed")

_CHECKPOINT_EVIDENCE_LEGACY_V1 = "legacy-v1"
_CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1 = "migrated-legacy-v1"
_CHECKPOINT_EVIDENCE_RECORDED_RUNTIME_V3 = "recorded-runtime-v3"
_CHECKPOINT_COMMON_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_label",
        "dataset_sha256",
        "config_sha256",
        "fixture",
        "scientific_use",
    }
)
_CHECKPOINT_PROVENANCE_BINDING_FIELDS = frozenset(
    {
        "source_tree_sha256",
        "requirements_lock_sha256",
        "runtime_provenance_sha256",
    }
)
_CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS = frozenset(
    {*_CHECKPOINT_PROVENANCE_BINDING_FIELDS, "runtime_provenance"}
)
_CHECKPOINT_EVIDENCE_FIELDS_BY_VERSION = {
    _CHECKPOINT_EVIDENCE_LEGACY_V1: frozenset(),
    _CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1: _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS,
    _CHECKPOINT_EVIDENCE_RECORDED_RUNTIME_V3: (
        _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS
    ),
}
_FORMAL_CHECKPOINT_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "arm",
        "seed",
        "model_spec",
        "normalization",
        "teacher_checkpoint_sha256",
        "checkpoint_rule",
        "state_dict",
        *_CHECKPOINT_COMMON_EVIDENCE_FIELDS,
    }
)
_SHUFFLED_CHECKPOINT_BASE_FIELDS = frozenset(
    {
        *_FORMAL_CHECKPOINT_BASE_FIELDS,
        "pairing_breaks",
        "training_provenance_sha256",
    }
)
_FORMAL_CHECKPOINT_INDEX_FIELDS = frozenset(
    {"schema_version", "scientific_use", "checkpoints", *EVIDENCE_AUTH_KEYS}
)
_FORMAL_CHECKPOINT_ROW_BASE_FIELDS = frozenset(
    {
        "seed",
        "arm",
        "path",
        "sha256",
        "parameters",
        "measured_flops_per_step",
        "execution_device",
        "teacher_checkpoint_sha256",
    }
)


def run_shuffled_pair_control(config, dataset, manifest_path, output_root):
    from .formal_upstream import resolve_authenticated_upstream

    upstream = resolve_authenticated_upstream(config, dataset, output_root)
    qualification_teacher_sha256 = _authenticated_qualification_gate(
        config, dataset, upstream
    )["teacher_checkpoint_sha256"]
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        ("source_probe_train", "source_probe_selection", "source_final_unseen_bank", "target"),
    )
    output_dir = Path(output_root) / "controls" / "shuffled_pair"
    result, manifest, source_hash = _run_adapter(
        config,
        dataset,
        manifest_path,
        output_dir,
        "csi-pairs-v6-shuffled-pair-adapter-v3",
        "results.json",
        output_run_root=upstream.run_root,
        upstream_root=_upstream_run_root(upstream),
    )
    required = {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "per_unit_results_path", "per_unit_results_sha256",
        "pair_registry_sha256", "checkpoint_index_sha256", "adapter_source_sha256",
        "shuffled_checkpoint_index_path", "shuffled_checkpoint_index_sha256",
        "training_provenance_path", "training_provenance_sha256",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise RuntimeError("shuffled-pair result fields must be exact")
    if result["schema_version"] != "csi-pairs-v6-shuffled-pair-results-v3":
        raise RuntimeError("shuffled-pair result schema mismatch")
    evidence = _validate_result_evidence(config, dataset, result)
    if result["adapter_source_sha256"] != source_hash:
        raise RuntimeError("shuffled-pair result is not bound to the authenticated adapter source")
    shortcut_binding = _validate_evaluation_shortcut_binding(
        config, dataset, output_root, evidence
    )
    checkpoints = _verify_checkpoint_index_binding(config, dataset, output_root, result)
    shuffled_checkpoints = _verify_shuffled_training_binding(
        config,
        dataset,
        output_dir,
        result,
        checkpoints,
        manifest["control_seed"],
        qualification_teacher_sha256,
    )
    registry = _read_active_pair_registry(output_root, result, evidence, dataset)
    rows = _read_bound_rows(output_dir, result, "per_unit_results", _shuffled_columns())
    _validate_shuffled_rows(
        rows, registry, checkpoints, shuffled_checkpoints, dataset
    )
    assessment = _shuffled_assessment(config, rows, registry)
    passed = assessment["passed"]
    gate = {
        "schema_version": "csi-pairs-v6-shuffled-pair-gate-v3",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "claim": "C4",
        **assessment,
        "pairing_permutation_seed": manifest["control_seed"],
        "checkpoint_hashes_verified": True,
        "independently_trained_shuffled_checkpoints_verified": True,
        "alignment_and_response_pairing_breaks_verified": True,
        "training_provenance_sha256": result["training_provenance_sha256"],
        "shuffled_checkpoint_index_sha256": result["shuffled_checkpoint_index_sha256"],
        "per_unit_rows_verified": True,
        "adapter_source_sha256": source_hash,
        **shortcut_binding,
        "input_manifest_path": "adapter_manifest.json",
        "input_manifest_sha256": sha256_file(output_dir / "adapter_manifest.json"),
    }
    write_json(output_dir / "gate.json", gate)
    _write_manifest(output_dir, evidence)
    return gate


def run_retention_audit(config, dataset, manifest_path, output_root):
    from .formal_data_verification import require_verified_roles_from_root
    from .formal_upstream import resolve_authenticated_upstream

    upstream = resolve_authenticated_upstream(config, dataset, output_root)
    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        ("source_final_unseen_bank", "target"),
    )
    output_dir = Path(output_root) / "evaluation" / "retention"
    result, _, source_hash = _run_adapter(
        config,
        dataset,
        manifest_path,
        output_dir,
        "csi-pairs-v6-retention-adapter-v3",
        "results.json",
        output_run_root=upstream.run_root,
        upstream_root=_upstream_run_root(upstream),
    )
    required = {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "per_unit_results_path", "per_unit_results_sha256",
        "pair_registry_sha256", "checkpoint_index_sha256", "adapter_source_sha256",
        "probe_checkpoint_index_path", "probe_checkpoint_index_sha256",
        "probe_training_provenance_path", "probe_training_provenance_sha256",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise RuntimeError("retention result fields must be exact")
    if result["schema_version"] != "csi-pairs-v6-retention-results-v3":
        raise RuntimeError("retention result schema mismatch")
    evidence = _validate_result_evidence(config, dataset, result)
    if result["adapter_source_sha256"] != source_hash:
        raise RuntimeError("retention result is not bound to the authenticated adapter source")
    checkpoints = _verify_checkpoint_index_binding(config, dataset, output_root, result)
    probes = _verify_retention_probe_binding(
        config, dataset, output_dir, result, checkpoints
    )
    registry = _read_active_pair_registry(output_root, result, evidence, dataset)
    rows = _read_bound_rows(output_dir, result, "per_unit_results", _retention_columns())
    _validate_retention_rows(rows, registry, checkpoints, probes)
    assessment = _retention_assessment(config, rows, registry)
    passed = assessment["passed"]
    gate = {
        "schema_version": "csi-pairs-v6-retention-gate-v3",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "claim": "C6",
        **assessment,
        "checkpoint_hashes_verified": True,
        "source_only_probe_training_verified": True,
        "probe_checkpoint_index_sha256": result["probe_checkpoint_index_sha256"],
        "probe_training_provenance_sha256": result["probe_training_provenance_sha256"],
        "per_unit_rows_verified": True,
        "downstream_f_only_verified": True,
        "disposable_heads_absent_verified": True,
        "adapter_source_sha256": source_hash,
        "input_manifest_path": "adapter_manifest.json",
        "input_manifest_sha256": sha256_file(output_dir / "adapter_manifest.json"),
    }
    write_json(output_dir / "gate.json", gate)
    _write_manifest(output_dir, evidence)
    return gate


def _run_adapter(
    config,
    dataset,
    manifest_path,
    output_dir,
    schema,
    result_name,
    *,
    output_run_root=None,
    upstream_root=None,
):
    from .formal_config import public_formal_config

    manifest_path = Path(manifest_path).resolve()
    manifest = read_strict_json(manifest_path)
    required = {
        "schema_version", "command", "implementation_revision", "control_seed",
        "adapter_source_path", "adapter_source_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("claim-control manifest fields must be exact")
    if manifest["schema_version"] != schema:
        raise ValueError("claim-control manifest schema mismatch")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValueError("claim-control command must be nonempty argv")
    if not isinstance(manifest["implementation_revision"], str) or not manifest["implementation_revision"]:
        raise ValueError("claim-control implementation revision must be nonempty")
    if type(manifest["control_seed"]) is not int or manifest["control_seed"] < 0:
        raise ValueError("claim-control seed must be nonnegative integer")
    source = (manifest_path.parent / manifest["adapter_source_path"]).resolve()
    if manifest_path.parent not in source.parents or not source.is_file():
        raise ValueError("claim-control adapter source is missing or escapes the manifest directory")
    source_hash = sha256_file(source)
    if source_hash != manifest["adapter_source_sha256"]:
        raise ValueError("claim-control adapter source hash mismatch")
    if manifest["implementation_revision"] != source_hash:
        raise ValueError(
            "claim-control implementation revision must equal the authenticated source hash"
        )
    if manifest["command"][:2] != ["{python}", "{adapter_source}"]:
        raise ValueError(
            "claim-control command must execute the authenticated adapter source directly"
        )
    _require_explicit_root_bindings(manifest["command"])
    if output_run_root is None or upstream_root is None:
        local_root = Path(output_dir).parents[1].resolve()
        output_run_root = local_root
        upstream_root = local_root
    output_dir.mkdir(parents=True, exist_ok=True)
    source_copy = output_dir / f"adapter_source{source.suffix or '.bin'}"
    shutil.copyfile(source, source_copy)
    bound_manifest = dict(manifest)
    bound_manifest["adapter_source_path"] = source_copy.name
    write_json(output_dir / "adapter_manifest.json", bound_manifest)
    context_path = output_dir / "control_context.json"
    write_json(
        context_path,
        {
            "schema_version": "csi-pairs-v6-claim-control-context-v1",
            "config": public_formal_config(config),
            **evidence_context(
                config,
                dataset,
                "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
            ),
        },
    )
    command = [
        value.format(
            dataset=str(dataset.source_path), output=str(output_dir), python=sys.executable,
            adapter_source=str(source),
            output_run_root=str(Path(output_run_root).resolve()),
            upstream_root=str(Path(upstream_root).resolve()),
            context=str(context_path),
        )
        for value in manifest["command"]
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    path = output_dir / result_name
    if completed.returncode != 0 or not path.is_file():
        raise RuntimeError("claim-control adapter failed")
    return read_strict_json(path), manifest, source_hash


def _upstream_run_root(upstream):
    if upstream.migrated:
        return upstream.migration.legacy_run_root
    return upstream.run_root


def _require_explicit_root_bindings(command):
    bindings = {
        "--output-run-root": "{output_run_root}",
        "--upstream-root": "{upstream_root}",
    }
    for option, placeholder in bindings.items():
        positions = [index for index, value in enumerate(command) if value == option]
        if (
            len(positions) != 1
            or positions[0] + 1 >= len(command)
            or command[positions[0] + 1] != placeholder
            or command.count(placeholder) != 1
        ):
            raise ValueError(
                "claim-control command must bind output and upstream roots explicitly"
            )
    if "--run-root" in command or "{run_root}" in command:
        raise ValueError("claim-control command cannot use the ambiguous run root")


def _validate_result_evidence(config, dataset, result):
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if result[key] != evidence[key]:
            raise RuntimeError(f"claim-control result {key} mismatch")
    return evidence


def _authenticated_qualification_gate(config, dataset, upstream):
    gate_path = Path(upstream.qualification_gate)
    if gate_path.is_symlink() or not gate_path.is_file():
        raise RuntimeError("claim-control qualification gate is missing or invalid")
    gate = read_strict_json(gate_path)
    if upstream.migrated:
        authenticated_evidence = upstream.qualification_evidence
        if not isinstance(authenticated_evidence, dict):
            raise RuntimeError(
                "claim-control migrated qualification evidence is unavailable"
            )
        for key in (*EVIDENCE_AUTH_KEYS, "scientific_use"):
            if gate.get(key) != authenticated_evidence.get(key):
                raise RuntimeError(
                    f"claim-control migrated qualification {key} mismatch"
                )
    else:
        from .formal_evidence import (
            QUALIFICATION_SCHEMA,
            require_formal_qualification,
            require_stage_manifested_gate,
        )

        gate = require_formal_qualification(
            gate,
            config,
            dataset,
            allow_nonscientific_fixture=True,
        )
        gate = require_stage_manifested_gate(
            gate_path,
            gate,
            config,
            dataset,
            schema_version=QUALIFICATION_SCHEMA,
        )

    teacher_sha256 = gate.get("teacher_checkpoint_sha256")
    teacher_source = Path(str(gate.get("teacher_checkpoint", "")))
    teacher_path = teacher_source.resolve()
    qualification_root = Path(upstream.qualification_root).resolve()
    if (
        not _lower_sha256(teacher_sha256)
        or teacher_source.is_symlink()
        or qualification_root not in teacher_path.parents
        or not teacher_path.is_file()
        or sha256_file(teacher_path) != teacher_sha256
    ):
        raise RuntimeError(
            "claim-control qualification teacher checkpoint is not authenticated"
        )
    return gate


def _verify_checkpoint_index_binding(config, dataset, output_root, result):
    from .formal_upstream import resolve_authenticated_upstream

    upstream = resolve_authenticated_upstream(config, dataset, output_root)
    path = upstream.checkpoint_index
    if not path.is_file() or result["checkpoint_index_sha256"] != sha256_file(path):
        raise RuntimeError("claim-control result does not bind the executed checkpoint index")
    index = read_strict_json(path)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    if not isinstance(index, dict) or set(index) != _FORMAL_CHECKPOINT_INDEX_FIELDS:
        raise RuntimeError("claim-control checkpoint index fields must be exact")
    if index["schema_version"] != "csi-pairs-formal-checkpoint-index-v2.1-v6":
        raise RuntimeError("claim-control checkpoint index schema mismatch")
    authenticated_evidence = (
        upstream.factorial_evidence if upstream.migrated else evidence
    )
    if not isinstance(authenticated_evidence, dict):
        raise RuntimeError("claim-control checkpoint index evidence is unavailable")
    for key in (*EVIDENCE_AUTH_KEYS, "scientific_use"):
        if index[key] != authenticated_evidence.get(key):
            raise RuntimeError(f"claim-control checkpoint index {key} mismatch")
    for key in _CHECKPOINT_COMMON_EVIDENCE_FIELDS:
        if index[key] != evidence[key]:
            raise RuntimeError(f"claim-control checkpoint index {key} mismatch")

    scalar_evidence = {
        key: index[key]
        for key in (*EVIDENCE_AUTH_KEYS, "scientific_use")
        if not isinstance(index[key], (dict, list))
    }
    expected_row_fields = set(_FORMAL_CHECKPOINT_ROW_BASE_FIELDS).union(
        scalar_evidence
    )
    rows = index["checkpoints"]
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != expected_row_fields
        for row in rows
    ):
        raise RuntimeError("claim-control checkpoint index row fields must be exact")
    qualification_teacher_sha256 = _authenticated_qualification_gate(
        config, dataset, upstream
    )["teacher_checkpoint_sha256"]
    expected_cells = [
        (int(seed), arm) for seed in config["seeds"] for arm in ARMS
    ]
    if len(rows) != len(expected_cells):
        raise RuntimeError("claim-control checkpoint index cell inventory is incomplete")
    for row, (expected_seed, expected_arm) in zip(rows, expected_cells):
        if type(row["seed"]) is not int or (
            row["seed"], row["arm"]
        ) != (expected_seed, expected_arm):
            raise RuntimeError("claim-control checkpoint index cell order changed")
        for key, value in scalar_evidence.items():
            if row[key] != value:
                raise RuntimeError(
                    f"claim-control checkpoint row {key} evidence mismatch"
                )
        if (
            not isinstance(row["path"], str)
            or not row["path"]
            or not _lower_sha256(row["sha256"])
            or type(row["parameters"]) is not int
            or row["parameters"] <= 0
            or type(row["measured_flops_per_step"]) is not int
            or row["measured_flops_per_step"] <= 0
            or not isinstance(row["execution_device"], str)
            or not row["execution_device"]
            or not _lower_sha256(row["teacher_checkpoint_sha256"])
        ):
            raise RuntimeError("claim-control checkpoint index row metadata is invalid")
        if row["teacher_checkpoint_sha256"] != qualification_teacher_sha256:
            raise RuntimeError(
                "claim-control checkpoint index teacher hash mismatch"
            )

    root = path.parent.resolve()
    full_by_seed = {}
    for row in rows:
        checkpoint_path = (root / row["path"]).resolve()
        if root not in checkpoint_path.parents or not checkpoint_path.is_file():
            raise RuntimeError("claim-control checkpoint index contains a missing checkpoint")
        digest = sha256_file(checkpoint_path)
        if digest != row["sha256"]:
            raise RuntimeError("claim-control checkpoint index contains a mismatched checkpoint")
        if row.get("arm") != "full":
            continue
        payload = _validate_formal_checkpoint(
            checkpoint_path,
            row,
            evidence,
            teacher_checkpoint_sha256=qualification_teacher_sha256,
            legacy_runtime=upstream.migrated,
        )
        seed = int(payload["seed"])
        if seed in full_by_seed:
            raise RuntimeError("claim-control checkpoint index duplicates a full-arm seed")
        full_by_seed[seed] = digest
    if not full_by_seed:
        raise RuntimeError("claim-control checkpoint index has no authenticated full-arm checkpoint")
    return full_by_seed


def _verify_shuffled_training_binding(
    config,
    dataset,
    output_dir,
    result,
    matched_checkpoints,
    control_seed,
    teacher_checkpoint_sha256,
):
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    provenance_path = _bound_output_file(
        output_dir,
        result["training_provenance_path"],
        result["training_provenance_sha256"],
        "shuffled training provenance",
    )
    provenance = read_strict_json(provenance_path)
    required = {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "control_seed", "training_roles", "selection_role", "target_roles_used",
        "pairing_breaks", "pairing_strata", "loss_form_preserved", "training_steps",
        "permutation_registry_path", "permutation_registry_sha256",
        "training_summary_path", "training_summary_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise RuntimeError("shuffled training provenance fields must be exact")
    if (
        provenance["schema_version"]
        != "csi-pairs-v6-shuffled-training-provenance-v1"
        or provenance["control_seed"] != int(control_seed)
        or provenance["training_roles"] != ["source_encoder_train"]
        or provenance["selection_role"] != "source_method_selection"
        or provenance["target_roles_used"] != []
        or set(provenance["pairing_breaks"])
        != {"alignment_h_map_edge", "response_action_target"}
        or set(provenance["pairing_strata"])
        != {"scene", "edit_family", "effect_bucket"}
        or provenance["loss_form_preserved"] is not True
        or provenance["training_steps"] != int(config["factorial"]["steps"])
    ):
        raise RuntimeError("shuffled training provenance violates the frozen V6 control")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if provenance[key] != evidence[key]:
            raise RuntimeError(f"shuffled training provenance {key} mismatch")
    permutation_path = _bound_output_file(
        output_dir,
        provenance["permutation_registry_path"],
        provenance["permutation_registry_sha256"],
        "shuffled permutation registry",
    )
    _validate_pairing_permutations(permutation_path, set(matched_checkpoints))
    training_summary_path = _bound_output_file(
        output_dir,
        provenance["training_summary_path"],
        provenance["training_summary_sha256"],
        "shuffled training summary",
    )
    _validate_shuffled_training_summary(
        training_summary_path,
        set(matched_checkpoints),
        int(config["factorial"]["steps"]),
    )

    index_path = _bound_output_file(
        output_dir,
        result["shuffled_checkpoint_index_path"],
        result["shuffled_checkpoint_index_sha256"],
        "shuffled checkpoint index",
    )
    index = read_strict_json(index_path)
    if not isinstance(index, dict) or set(index) != {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "training_provenance_sha256", "checkpoints",
    }:
        raise RuntimeError("shuffled checkpoint index fields must be exact")
    if (
        index["schema_version"] != "csi-pairs-v6-shuffled-checkpoint-index-v1"
        or index["training_provenance_sha256"]
        != result["training_provenance_sha256"]
    ):
        raise RuntimeError("shuffled checkpoint index identity mismatch")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if index[key] != evidence[key]:
            raise RuntimeError(f"shuffled checkpoint index {key} mismatch")
    rows = index["checkpoints"]
    if not isinstance(rows, list):
        raise RuntimeError("shuffled checkpoint index checkpoints must be a list")
    shuffled = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "seed", "path", "sha256", "matched_checkpoint_sha256"
        }:
            raise RuntimeError("shuffled checkpoint index row fields must be exact")
        seed = int(row["seed"])
        if seed in shuffled or seed not in matched_checkpoints:
            raise RuntimeError("shuffled checkpoint index has an unexpected or duplicate seed")
        if row["matched_checkpoint_sha256"] != matched_checkpoints[seed]:
            raise RuntimeError("shuffled checkpoint does not bind its matched full checkpoint")
        path = _bound_output_file(
            output_dir, row["path"], row["sha256"], "shuffled checkpoint"
        )
        _require_distinct_checkpoint_hash(
            matched_checkpoints[seed], row["sha256"]
        )
        _validate_shuffled_checkpoint(
            path,
            row,
            evidence,
            result["training_provenance_sha256"],
            teacher_checkpoint_sha256,
        )
        shuffled[seed] = row["sha256"]
    if set(shuffled) != set(matched_checkpoints):
        raise RuntimeError("shuffled checkpoints do not cover every full-arm seed")
    return shuffled


def _require_distinct_checkpoint_hash(matched_sha256, shuffled_sha256):
    if not _lower_sha256(matched_sha256) or not _lower_sha256(shuffled_sha256):
        raise RuntimeError("shuffled checkpoint digest is invalid")
    if shuffled_sha256 == matched_sha256:
        raise RuntimeError("shuffled control reused the matched full checkpoint")


def _validate_pairing_permutations(path, expected_seeds):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "seed", "branch", "scene_id", "edit_family", "effect_bucket",
        "original_pair_id", "permuted_pair_id",
    }
    if not rows or any(set(row) != required for row in rows):
        raise RuntimeError("shuffled permutation registry columns must be exact")
    if {int(row["seed"]) for row in rows} != set(expected_seeds):
        raise RuntimeError("shuffled permutation registry does not cover every seed")
    groups = {}
    for row in rows:
        if row["branch"] not in {"alignment_h_map_edge", "response_action_target"}:
            raise RuntimeError("shuffled permutation registry has an unknown branch")
        if row["original_pair_id"] == row["permuted_pair_id"]:
            raise RuntimeError("shuffled permutation registry contains an unshuffled pair")
        group = (
            int(row["seed"]), row["branch"], row["scene_id"],
            row["edit_family"], row["effect_bucket"],
        )
        groups.setdefault(group, []).append(row)
    branches = {group[1] for group in groups}
    if branches != {"alignment_h_map_edge", "response_action_target"}:
        raise RuntimeError("shuffled permutation registry must break both V6 branches")
    for rows_in_group in groups.values():
        original = [row["original_pair_id"] for row in rows_in_group]
        permuted = [row["permuted_pair_id"] for row in rows_in_group]
        if len(original) < 2 or len(original) != len(set(original)) or set(original) != set(permuted):
            raise RuntimeError("shuffled permutation stratum is not a complete derangement")


def _validate_shuffled_training_summary(path, expected_seeds, expected_steps):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "seed", "arm", "steps", "parameters", "measured_flops_per_step",
        "alignment_gradient_norm_mean", "response_gradient_norm_mean",
        "raw_alignment_gradient_norm_mean", "raw_response_gradient_norm_mean",
        "checkpoint_rule",
    }
    if (
        not rows
        or any(not required.issubset(row) for row in rows)
        or {int(row["seed"]) for row in rows} != set(expected_seeds)
        or len(rows) != len(expected_seeds)
    ):
        raise RuntimeError("shuffled training summary does not cover every seed exactly once")
    for row in rows:
        if (
            row["arm"] != "full"
            or int(row["steps"]) != int(expected_steps)
            or row["checkpoint_rule"] != "fixed_final_step_no_target_selection"
            or int(row["parameters"]) <= 0
            or not np.isfinite(float(row["measured_flops_per_step"]))
            or float(row["measured_flops_per_step"]) <= 0
            or any(
                not np.isfinite(float(row[key])) or float(row[key]) <= 0
                for key in (
                    "alignment_gradient_norm_mean", "response_gradient_norm_mean",
                    "raw_alignment_gradient_norm_mean", "raw_response_gradient_norm_mean",
                )
            )
        ):
            raise RuntimeError("shuffled training summary violates the full-arm contract")


def _checkpoint_evidence_version(binding, label, *, legacy_runtime=False):
    if not isinstance(binding, dict):
        raise RuntimeError(f"{label} provenance binding must be an object")
    present = _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS.intersection(binding)
    if not present:
        return _CHECKPOINT_EVIDENCE_LEGACY_V1
    if not _CHECKPOINT_PROVENANCE_BINDING_FIELDS.issubset(binding):
        raise RuntimeError(f"{label} provenance binding is incomplete")
    if legacy_runtime:
        return _CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1
    return _CHECKPOINT_EVIDENCE_RECORDED_RUNTIME_V3


def _recorded_runtime_sha256(runtime, label):
    try:
        encoded = json.dumps(
            runtime,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} runtime provenance is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_recorded_runtime_provenance(runtime, label):
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_PROVENANCE_FIELDS
        or runtime.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA
    ):
        raise RuntimeError(f"{label} runtime provenance fields are not exact")
    for key in (
        "source_tree_sha256",
        "requirements_lock_sha256",
        "installer_report_sha256",
        "reviewed_wheelhouse_sha256",
        "reviewed_wheel_manifest_sha256",
    ):
        if not _lower_sha256(runtime.get(key)):
            raise RuntimeError(f"{label} runtime provenance {key} is invalid")
    if runtime.get("python_dont_write_bytecode") is not True:
        raise RuntimeError(f"{label} runtime provenance permits Python bytecode")
    torch_record = runtime.get("torch")
    if not isinstance(torch_record, dict) or set(torch_record) != TORCH_RUNTIME_FIELDS:
        raise RuntimeError(f"{label} runtime torch provenance fields are not exact")
    installed = runtime.get("installed_distributions")
    if not isinstance(installed, dict):
        raise RuntimeError(f"{label} runtime distribution provenance is invalid")
    for name, record in installed.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record, dict)
            or set(record) != {"version", "record_sha256", "wheel_sha256"}
            or not isinstance(record["version"], str)
            or not record["version"]
            or not _lower_sha256(record["record_sha256"])
            or not _lower_sha256(record["wheel_sha256"])
        ):
            raise RuntimeError(f"{label} runtime distribution provenance is invalid")


def _validate_migrated_legacy_runtime(runtime, label):
    """Validate the less-populated runtime record accepted by migration v2."""
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_PROVENANCE_FIELDS
        or runtime.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA
    ):
        raise RuntimeError(f"{label} legacy runtime provenance fields are not exact")
    if runtime.get("python_implementation") != "CPython":
        raise RuntimeError(f"{label} legacy runtime implementation is invalid")
    if runtime.get("python_dont_write_bytecode") is not True:
        raise RuntimeError(f"{label} legacy runtime permits Python bytecode")


def _validate_checkpoint_evidence(
    payload,
    *,
    base_fields,
    provenance_binding,
    evidence,
    label,
    legacy_runtime=False,
):
    version = _checkpoint_evidence_version(
        provenance_binding, label, legacy_runtime=legacy_runtime
    )
    required = set(base_fields).union(
        _CHECKPOINT_EVIDENCE_FIELDS_BY_VERSION[version]
    )
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError(
            f"{label} fields are not exact for evidence version {version}"
        )
    if not isinstance(evidence, dict) or not _CHECKPOINT_COMMON_EVIDENCE_FIELDS.issubset(
        evidence
    ):
        raise RuntimeError(f"{label} outer evidence is incomplete")
    for key in _CHECKPOINT_COMMON_EVIDENCE_FIELDS:
        if payload[key] != evidence[key]:
            raise RuntimeError(f"{label} {key} mismatch")
    if version == _CHECKPOINT_EVIDENCE_LEGACY_V1:
        return version

    for key in _CHECKPOINT_PROVENANCE_BINDING_FIELDS:
        if not _lower_sha256(payload[key]) or payload[key] != provenance_binding[key]:
            raise RuntimeError(f"{label} {key} provenance mismatch")
    runtime = payload["runtime_provenance"]
    if version == _CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1:
        _validate_migrated_legacy_runtime(runtime, label)
    else:
        _validate_recorded_runtime_provenance(runtime, label)
    if (
        runtime["source_tree_sha256"] != payload["source_tree_sha256"]
        or runtime["requirements_lock_sha256"]
        != payload["requirements_lock_sha256"]
        or _recorded_runtime_sha256(runtime, label)
        != payload["runtime_provenance_sha256"]
    ):
        raise RuntimeError(f"{label} runtime provenance digest or identity mismatch")
    if (
        "runtime_provenance" in provenance_binding
        and runtime != provenance_binding["runtime_provenance"]
    ):
        raise RuntimeError(f"{label} runtime provenance object mismatch")
    return version


def _validate_shuffled_checkpoint(
    path,
    row,
    evidence,
    provenance_sha256,
    teacher_checkpoint_sha256=None,
):
    try:
        import torch
        from .formal_model import CSIPairsFormalModel

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("shuffled checkpoint is unreadable") from error
    _validate_checkpoint_evidence(
        payload,
        base_fields=_SHUFFLED_CHECKPOINT_BASE_FIELDS,
        provenance_binding=evidence,
        evidence=evidence,
        label="shuffled checkpoint",
    )
    if (
        payload["schema_version"] != "csi-pairs-v6-shuffled-formal-checkpoint-v1"
        or payload["arm"] != "full"
        or int(payload["seed"]) != int(row["seed"])
        or payload["checkpoint_rule"] != "fixed_final_step_no_target_selection"
        or set(payload["pairing_breaks"])
        != {"alignment_h_map_edge", "response_action_target"}
        or payload["training_provenance_sha256"] != provenance_sha256
        or (
            teacher_checkpoint_sha256 is not None
            and (
                not _lower_sha256(teacher_checkpoint_sha256)
                or payload["teacher_checkpoint_sha256"]
                != teacher_checkpoint_sha256
            )
        )
    ):
        raise RuntimeError("shuffled checkpoint identity mismatch")
    try:
        model = CSIPairsFormalModel(**payload["model_spec"])
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as error:
        raise RuntimeError("shuffled checkpoint state is not the frozen formal model") from error


def _verify_retention_probe_binding(
    config, dataset, output_dir, result, full_checkpoints
):
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    provenance_path = _bound_output_file(
        output_dir,
        result["probe_training_provenance_path"],
        result["probe_training_provenance_sha256"],
        "retention probe training provenance",
    )
    provenance = read_strict_json(provenance_path)
    required = {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "fit_role", "selection_role", "evaluation_roles",
        "formal_model_parameters_updated", "target_labels_used",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise RuntimeError("retention probe training provenance fields must be exact")
    if (
        provenance["schema_version"]
        != "csi-pairs-v6-retention-probe-training-v1"
        or provenance["fit_role"] != "source_probe_train"
        or provenance["selection_role"] != "source_probe_selection"
        or provenance["evaluation_roles"] != ["source_final_unseen_bank", "target"]
        or provenance["formal_model_parameters_updated"] is not False
        or provenance["target_labels_used"] is not False
    ):
        raise RuntimeError("retention probes violate the frozen source-only role ledger")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if provenance[key] != evidence[key]:
            raise RuntimeError(f"retention probe training provenance {key} mismatch")

    index_path = _bound_output_file(
        output_dir,
        result["probe_checkpoint_index_path"],
        result["probe_checkpoint_index_sha256"],
        "retention probe checkpoint index",
    )
    index = read_strict_json(index_path)
    if not isinstance(index, dict) or set(index) != {
        "schema_version", "dataset_sha256", "config_sha256", "fixture",
        "training_provenance_sha256", "probes",
    }:
        raise RuntimeError("retention probe checkpoint index fields must be exact")
    if (
        index["schema_version"] != "csi-pairs-v6-retention-probe-index-v1"
        or index["training_provenance_sha256"]
        != result["probe_training_provenance_sha256"]
    ):
        raise RuntimeError("retention probe checkpoint index identity mismatch")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if index[key] != evidence[key]:
            raise RuntimeError(f"retention probe checkpoint index {key} mismatch")
    probes = {}
    for row in index["probes"] if isinstance(index["probes"], list) else []:
        if not isinstance(row, dict) or set(row) != {"seed", "kind", "path", "sha256"}:
            raise RuntimeError("retention probe index row fields must be exact")
        seed = int(row["seed"])
        kind = row["kind"]
        if kind not in {"compatibility", "response"} or kind in probes.setdefault(seed, {}):
            raise RuntimeError("retention probe index has a duplicate or unknown probe")
        path = _bound_output_file(output_dir, row["path"], row["sha256"], "retention probe")
        _validate_retention_probe_checkpoint(
            path,
            seed,
            kind,
            evidence,
            result["probe_training_provenance_sha256"],
            full_checkpoints.get(seed),
        )
        probes[seed][kind] = row["sha256"]
    if set(probes) != set(map(int, config["seeds"])) or any(
        set(value) != {"compatibility", "response"} for value in probes.values()
    ):
        raise RuntimeError("retention probes do not cover both probes for every seed")
    return probes


def _validate_retention_probe_checkpoint(
    path,
    seed,
    kind,
    evidence,
    provenance_sha256,
    full_checkpoint_sha256,
):
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("retention probe checkpoint is unreadable") from error
    required = {
        "schema_version", "seed", "kind", "dataset_sha256", "config_sha256",
        "fixture", "fit_role", "selection_role", "training_provenance_sha256",
        "full_checkpoint_sha256", "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("retention probe checkpoint fields must be exact")
    if (
        payload["schema_version"] != "csi-pairs-v6-retention-probe-checkpoint-v1"
        or int(payload["seed"]) != int(seed)
        or payload["kind"] != kind
        or payload["fit_role"] != "source_probe_train"
        or payload["selection_role"] != "source_probe_selection"
        or payload["training_provenance_sha256"] != provenance_sha256
        or not _lower_sha256(full_checkpoint_sha256)
        or payload["full_checkpoint_sha256"] != full_checkpoint_sha256
        or not isinstance(payload["state_dict"], dict)
        or not payload["state_dict"]
    ):
        raise RuntimeError("retention probe checkpoint identity mismatch")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if payload[key] != evidence[key]:
            raise RuntimeError(f"retention probe checkpoint {key} mismatch")
    if any(
        not isinstance(value, torch.Tensor)
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
        for value in payload["state_dict"].values()
    ):
        raise RuntimeError("retention probe checkpoint contains invalid tensors")


def _bound_output_file(output_dir, relative, digest, label):
    if not isinstance(relative, str) or not relative or not _lower_sha256(digest):
        raise RuntimeError(f"{label} binding is invalid")
    root = Path(output_dir).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"{label} is missing, escapes output, or is hash-mismatched")
    return path


def _validate_formal_checkpoint(
    path,
    row,
    evidence,
    *,
    teacher_checkpoint_sha256=None,
    legacy_runtime=False,
):
    try:
        import torch
        from .formal_model import CSIPairsFormalModel

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("claim-control checkpoint is unreadable") from error
    _validate_checkpoint_evidence(
        payload,
        base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
        provenance_binding=row,
        evidence=evidence,
        label="claim-control checkpoint",
        legacy_runtime=legacy_runtime,
    )
    if (
        payload["schema_version"] != "csi-pairs-formal-checkpoint-v2.1-v6"
        or payload["arm"] != "full"
        or int(payload["seed"]) != int(row["seed"])
        or payload["checkpoint_rule"] != "fixed_final_step_no_target_selection"
        or payload["teacher_checkpoint_sha256"]
        != row["teacher_checkpoint_sha256"]
        or (
            teacher_checkpoint_sha256 is not None
            and (
                not _lower_sha256(teacher_checkpoint_sha256)
                or payload["teacher_checkpoint_sha256"]
                != teacher_checkpoint_sha256
            )
        )
    ):
        raise RuntimeError("claim-control checkpoint identity mismatch")
    try:
        model = CSIPairsFormalModel(**payload["model_spec"])
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as error:
        raise RuntimeError("claim-control checkpoint state is not the frozen formal model") from error
    if set(payload["state_dict"]) != set(model.state_dict()):
        raise RuntimeError("claim-control checkpoint contains disposable or missing heads")
    return payload


def _read_active_pair_registry(output_root, result, evidence, dataset):
    from .formal_factorial import _canonical_bank_digest

    path = Path(output_root) / "evaluation" / "compatibility_pair_effects.csv"
    if not path.is_file() or result["pair_registry_sha256"] != sha256_file(path):
        raise RuntimeError("claim-control result does not bind the evaluation pair registry")
    _require_stage_artifact(path)
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = [row for row in raw if row.get("arm") == "full" and row.get("route") == "active"]
    if not rows:
        raise RuntimeError("claim-control pair registry has no full-arm active rows")
    required = {
        "seed", "pair_id", "scene_index", "bank_id", "base_map_cluster_id",
        "canonical_base_map_digest", "canonical_bank_digest", "source_world",
        "target_world", "position_index", "route", "arm", "dataset_sha256",
        "config_sha256",
    }
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("claim-control pair registry lacks canonical hierarchy fields")
    registry = {}
    for row in rows:
        for key in ("dataset_sha256", "config_sha256"):
            if row.get(key) != str(evidence[key]):
                raise RuntimeError(f"claim-control pair registry {key} mismatch")
        key = (int(row["seed"]), row["pair_id"])
        if key in registry:
            raise RuntimeError("claim-control pair registry duplicates a seed/pair")
        scene = int(row["scene_index"])
        if scene < 0 or scene >= int(dataset.scene_count):
            raise RuntimeError("claim-control pair registry has an invalid scene index")
        expected_foundation = str(dataset.canonical_base_map_digest(scene))
        expected_bank = str(_canonical_bank_digest(dataset, scene))
        if (
            row["canonical_base_map_digest"] != expected_foundation
            or row["canonical_bank_digest"] != expected_bank
        ):
            raise RuntimeError(
                "claim-control pair registry canonical hierarchy differs from outer recomputation"
            )
        normalized = dict(row)
        for field in (
            "seed", "scene_index", "source_world", "target_world", "position_index"
        ):
            normalized[field] = int(row[field])
        registry[key] = normalized
    _require_unique_canonical_units(registry)
    if len({row["canonical_base_map_digest"] for row in registry.values()}) < 2:
        raise RuntimeError("claim-control pair registry has fewer than two independent clusters")
    return registry


def _validate_evaluation_shortcut_binding(config, dataset, output_root, evidence):
    from .formal_evidence import require_stage_manifested_gate

    evaluation_dir = Path(output_root) / "evaluation"
    gate_path = evaluation_dir / "gate.json"
    rows_path = evaluation_dir / "alignment_shortcut_baselines.csv"
    if not gate_path.is_file() or not rows_path.is_file():
        raise RuntimeError("shuffled-pair control requires the evaluation shortcut audit")
    gate = read_strict_json(gate_path)
    require_stage_manifested_gate(
        gate_path,
        gate,
        config,
        dataset,
        schema_version="csi-pairs-v6-evaluation-gate-v3",
    )
    audit = gate.get("alignment_shortcut_audit")
    required_baselines = set(SHUFFLED_SYSTEMS[2:])
    if (
        not isinstance(audit, dict)
        or audit.get("passed") is not True
        or audit.get("complete") is not True
        or set(audit.get("required_baselines", [])) != required_baselines
    ):
        raise RuntimeError("evaluation alignment shortcut audit is incomplete or failed")
    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {row.get("baseline") for row in rows} != required_baselines:
        raise RuntimeError("evaluation shortcut rows do not cover the frozen baseline set")
    _require_stage_artifact(rows_path)
    for row in rows:
        for key in ("dataset_sha256", "config_sha256"):
            if row.get(key) != str(evidence[key]):
                raise RuntimeError(f"evaluation shortcut row {key} mismatch")
        if not np.isfinite(float(row["auroc"])):
            raise RuntimeError("evaluation shortcut AUROC must be finite")
    return {
        "evaluation_alignment_shortcut_audit_verified": True,
        "evaluation_gate_sha256": sha256_file(gate_path),
        "alignment_shortcut_rows_sha256": sha256_file(rows_path),
    }


def _require_stage_artifact(path):
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"stage artifact has no manifest: {path}")
    manifest = read_strict_json(manifest_path)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    matches = [
        row for row in files
        if isinstance(row, dict) and row.get("path") == path.name
    ] if isinstance(files, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(path):
        raise RuntimeError(f"stage artifact is absent from or mismatched with its manifest: {path}")


def _read_bound_rows(output_dir, result, prefix, required_columns):
    path = (Path(output_dir) / result[f"{prefix}_path"]).resolve()
    if Path(output_dir).resolve() not in path.parents or not path.is_file():
        raise RuntimeError(f"claim-control {prefix} is missing or escapes adapter output")
    if sha256_file(path) != result[f"{prefix}_sha256"]:
        raise RuntimeError(f"claim-control {prefix} hash mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(set(row) != required_columns for row in rows):
        raise RuntimeError(f"claim-control {prefix} columns must be exact")
    return rows


def _shuffled_columns():
    return {
        "seed", "pair_id", "metric", "system", "pair_label", "score",
        "matched_checkpoint_sha256", "shuffled_checkpoint_sha256", "action_sha256",
    }


def _retention_columns():
    return {
        "seed", "pair_id", "condition", "cgs_score", "response_score",
        "checkpoint_sha256", "compatibility_probe_sha256", "response_probe_sha256",
        "representation_scope",
    }


def _validate_shuffled_rows(
    rows, registry, checkpoints, shuffled_checkpoints, dataset
):
    expected = {
        (seed, pair_id, metric, system, label)
        for seed, pair_id in registry
        for metric in SHUFFLED_METRICS
        for system in (
            SHUFFLED_SYSTEMS if metric == "alignment_cgs" else SHUFFLED_SYSTEMS[:2]
        )
        for label in PAIR_LABELS
    }
    seen = set()
    for row in rows:
        key = (int(row["seed"]), row["pair_id"])
        expanded = (*key, row["metric"], row["system"], row["pair_label"])
        if expanded in seen:
            raise RuntimeError("shuffled-pair rows duplicate a pair/system/label")
        seen.add(expanded)
        allowed_systems = (
            SHUFFLED_SYSTEMS
            if row["metric"] == "alignment_cgs"
            else SHUFFLED_SYSTEMS[:2]
            if row["metric"] == "response_probe"
            else ()
        )
        if key not in registry or row["system"] not in allowed_systems or row["pair_label"] not in PAIR_LABELS:
            raise RuntimeError("shuffled-pair rows do not match the frozen pair registry")
        value = float(row["score"])
        if not np.isfinite(value):
            raise RuntimeError("shuffled-pair scores must be finite")
        if row["matched_checkpoint_sha256"] != checkpoints.get(key[0]):
            raise RuntimeError("shuffled-pair row is not bound to its matched full-arm checkpoint")
        if row["shuffled_checkpoint_sha256"] != shuffled_checkpoints.get(key[0]):
            raise RuntimeError("shuffled-pair row is not bound to its independently trained shuffled checkpoint")
        expected_action = _registry_action_sha256(dataset, registry[key])
        if row["action_sha256"] != expected_action:
            raise RuntimeError("shuffled-pair row action digest differs from outer recomputation")
    if seen != expected:
        raise RuntimeError("shuffled-pair rows do not cover the complete active pair registry")


def _validate_retention_rows(rows, registry, checkpoints, probes):
    expected = {
        (seed, pair_id, condition)
        for seed, pair_id in registry
        for condition in RETENTION_CONDITIONS
    }
    seen = set()
    for row in rows:
        key = (int(row["seed"]), row["pair_id"])
        expanded = (*key, row["condition"])
        if expanded in seen:
            raise RuntimeError("retention rows duplicate a pair/condition")
        seen.add(expanded)
        if key not in registry or row["condition"] not in RETENTION_CONDITIONS:
            raise RuntimeError("retention rows do not match the frozen pair registry")
        if row["representation_scope"] != "formal_model.retained_representation":
            raise RuntimeError("retention rows do not evaluate the downstream retained F representation")
        if row["checkpoint_sha256"] != checkpoints.get(key[0]):
            raise RuntimeError("retention row is not bound to its full-arm seed checkpoint")
        probe_hashes = probes.get(key[0])
        if (
            probe_hashes is None
            or row["compatibility_probe_sha256"] != probe_hashes["compatibility"]
            or row["response_probe_sha256"] != probe_hashes["response"]
        ):
            raise RuntimeError("retention row is not bound to its source-only frozen probes")
        for field in ("cgs_score", "response_score"):
            if not np.isfinite(float(row[field])):
                raise RuntimeError("retention scores must be finite")
    if seen != expected:
        raise RuntimeError("retention rows do not cover the complete active pair registry")


def _registry_action_sha256(dataset, row):
    scene = int(row["scene_index"])
    source = int(row["source_world"])
    target = int(row["target_world"])
    matches = [
        edge for edge in dataset.directed_edges(scene)
        if edge.source_world == source and edge.target_world == target
    ]
    if len(matches) != 1:
        raise RuntimeError("claim-control pair does not identify one directed action")
    edge = matches[0]
    payload = {
        "scene_index": scene,
        "source_world": source,
        "target_world": target,
        "bit_index": int(edge.bit_index),
        "primitive_id": int(edge.primitive_id),
        "direction": int(edge.direction),
        "position_index": int(row["position_index"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _shuffled_assessment(config, rows, registry):
    _require_unique_canonical_units(registry)
    lookup = {
        (
            int(row["seed"]), row["pair_id"], row.get("metric", "alignment_cgs"),
            row["system"], row["pair_label"],
        ): float(row.get("score", row.get("alignment_score")))
        for row in rows
    }
    keys = sorted(registry)
    metric_gains = {
        metric: {
            system: {
                key: lookup[(*key, metric, system, "positive")]
                - lookup[(*key, metric, system, "negative")]
                for key in keys
            }
            for system in (
                SHUFFLED_SYSTEMS if metric == "alignment_cgs" else SHUFFLED_SYSTEMS[:2]
            )
        }
        for metric in SHUFFLED_METRICS
    }
    clusters = None
    gains = {}
    for system in SHUFFLED_SYSTEMS:
        system_clusters, system_values = _canonical_cluster_macro(
            registry, metric_gains["alignment_cgs"][system]
        )
        if clusters is None:
            clusters = system_clusters
        elif not np.array_equal(clusters, system_clusters):
            raise RuntimeError("shuffled-pair systems do not share canonical support")
        gains[system] = system_values
    if clusters is None:
        raise RuntimeError("shuffled-pair assessment has no canonical support")
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    matched_interval = paired_cluster_interval(
        clusters,
        gains["matched_model"],
        np.zeros_like(gains["matched_model"]),
        resamples,
        86201,
    )
    retained_interval = paired_cluster_interval(
        clusters,
        gains["shuffled_model"],
        np.zeros_like(gains["shuffled_model"]),
        resamples,
        86202,
    )
    shortcut_intervals = {
        system: {
            **paired_cluster_interval(
                clusters, gains["matched_model"], gains[system], resamples, 86210 + index
            ),
            "p_value_two_sided": paired_sign_flip_test(
                clusters, gains["matched_model"], gains[system], 86240 + index
            )["p_value_two_sided"],
        }
        for index, system in enumerate(SHUFFLED_SYSTEMS[2:])
    }
    for system, adjusted in zip(
        SHUFFLED_SYSTEMS[2:],
        holm_adjust([
            shortcut_intervals[system]["p_value_two_sided"]
            for system in SHUFFLED_SYSTEMS[2:]
        ]),
    ):
        shortcut_intervals[system]["holm_adjusted_p"] = float(adjusted)
    gain = float(matched_interval["paired_mean_difference"])
    shuffled = float(retained_interval["paired_mean_difference"])
    maximum_fraction = float(config["evaluation"]["shuffled_gain_fraction_max"])
    suppression_interval = paired_cluster_interval(
        clusters,
        maximum_fraction * gains["matched_model"],
        gains["shuffled_model"],
        resamples,
        86203,
    )
    alpha = float(config["evaluation"]["familywise_alpha"])
    shortcut_passed = all(
        interval_decision(value, threshold=0.0, relation="superiority")
        and value["holm_adjusted_p"] < alpha
        for value in shortcut_intervals.values()
    )
    response_clusters, response_matched = _canonical_cluster_macro(
        registry, metric_gains["response_probe"]["matched_model"]
    )
    response_shuffled_clusters, response_shuffled = _canonical_cluster_macro(
        registry, metric_gains["response_probe"]["shuffled_model"]
    )
    if not np.array_equal(response_clusters, response_shuffled_clusters):
        raise RuntimeError("shuffled-pair response systems do not share canonical support")
    response_matched_interval = paired_cluster_interval(
        response_clusters,
        response_matched,
        np.zeros_like(response_matched),
        resamples,
        86204,
    )
    response_shuffled_interval = paired_cluster_interval(
        response_clusters,
        response_shuffled,
        np.zeros_like(response_shuffled),
        resamples,
        86205,
    )
    response_suppression_interval = paired_cluster_interval(
        response_clusters,
        maximum_fraction * response_matched,
        response_shuffled,
        resamples,
        86206,
    )
    alignment_passed = bool(
        interval_decision(matched_interval, threshold=0.0, relation="superiority")
        and interval_decision(suppression_interval, threshold=0.0, relation="superiority")
        and shortcut_passed
    )
    response_passed = bool(
        interval_decision(response_matched_interval, threshold=0.0, relation="superiority")
        and interval_decision(
            response_suppression_interval, threshold=0.0, relation="superiority"
        )
    )
    return {
        "base_map_cluster_count": int(np.unique(clusters).size),
        "alignment_gain": gain,
        "alignment_gain_interval": matched_interval,
        "shuffled_alignment_gain": shuffled,
        "shuffled_alignment_gain_interval": retained_interval,
        "shuffled_suppression_interval": suppression_interval,
        "maximum_retained_fraction": maximum_fraction,
        "shortcut_superiority_intervals": shortcut_intervals,
        "shortcut_familywise_method": "Holm",
        "shortcut_familywise_alpha": alpha,
        "shortcut_baselines_passed": shortcut_passed,
        "alignment_pairing_break_passed": alignment_passed,
        "response_gain": float(response_matched_interval["paired_mean_difference"]),
        "response_gain_interval": response_matched_interval,
        "shuffled_response_gain": float(response_shuffled_interval["paired_mean_difference"]),
        "shuffled_response_gain_interval": response_shuffled_interval,
        "shuffled_response_suppression_interval": response_suppression_interval,
        "response_pairing_break_passed": response_passed,
        "passed": bool(alignment_passed and response_passed),
    }


def _retention_assessment(config, rows, registry):
    _require_unique_canonical_units(registry)
    lookup = {
        (int(row["seed"]), row["pair_id"], row["condition"]): row
        for row in rows
    }
    keys = sorted(registry)
    metric_conditions = {
        "cgs_map_swap_effect": ("cgs_score", "map_swap"),
        "cgs_map_removal_effect": ("cgs_score", "map_removed"),
        "response_map_swap_effect": ("response_score", "map_swap"),
    }
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    intervals = {}
    for index, (name, (metric, condition)) in enumerate(metric_conditions.items()):
        unit_effects = {
            key: float(lookup[(*key, "correct")][metric])
            - float(lookup[(*key, condition)][metric])
            for key in keys
        }
        clusters, effects = _canonical_cluster_macro(registry, unit_effects)
        intervals[name] = paired_cluster_interval(
            clusters, effects, np.zeros_like(effects), resamples, 86301 + index
        )
    minimum = float(config["evaluation"]["retention_minimum_effect"])
    effects = {name: float(value["paired_mean_difference"]) for name, value in intervals.items()}
    passed = all(
        interval_decision(value, threshold=minimum, relation="superiority")
        for value in intervals.values()
    )
    return {
        "base_map_cluster_count": int(np.unique(clusters).size),
        **effects,
        "effect_intervals": intervals,
        "minimum_effect": minimum,
        "passed": bool(passed),
    }


def _require_unique_canonical_units(registry):
    seen = {}
    for key, row in registry.items():
        unit = (
            int(row["seed"]),
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
            int(row["source_world"]),
            int(row["target_world"]),
            int(row["position_index"]),
        )
        if unit in seen:
            raise RuntimeError(
                "claim-control pair registry duplicates a canonical evaluation unit"
            )
        seen[unit] = key


def _canonical_cluster_macro(registry, unit_values):
    if set(unit_values) != set(registry):
        raise RuntimeError("claim-control statistic does not cover the canonical registry")
    _require_unique_canonical_units(registry)
    bank_seed_values = {}
    for key, row in registry.items():
        value = float(unit_values[key])
        if not np.isfinite(value):
            raise RuntimeError("claim-control canonical unit statistic must be finite")
        cell = (
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
            int(row["seed"]),
        )
        bank_seed_values.setdefault(cell, []).append(value)

    bank_values = {}
    for (cluster, bank, _), values in bank_seed_values.items():
        bank_values.setdefault((cluster, bank), []).append(float(np.mean(values)))
    cluster_values = {}
    for (cluster, _), seed_values in bank_values.items():
        cluster_values.setdefault(cluster, []).append(float(np.mean(seed_values)))
    clusters = np.asarray(sorted(cluster_values))
    if clusters.size < 2:
        raise RuntimeError("claim-control statistic has fewer than two canonical clusters")
    values = np.asarray(
        [np.mean(cluster_values[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    return clusters, values


def _write_manifest(output_dir, evidence):
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
