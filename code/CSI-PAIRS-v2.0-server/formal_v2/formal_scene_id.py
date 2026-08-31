from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import bind_rows, evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
from .formal_statistics import paired_cluster_interval


CONDITIONS = ("map", "scene_id", "map_swap", "id_swap")
BUILTIN_SCENE_ID_MANIFEST = "builtin:sigmap-scene-id-v1"


def run_scene_id_audit(config, dataset, manifest_path, output_root):
    from .formal_data_verification import require_verified_roles_from_root
    from .formal_upstream import resolve_authenticated_upstream

    upstream = resolve_authenticated_upstream(config, dataset, output_root)
    require_verified_roles_from_root(
        output_root, config, dataset, ("source_final_unseen_bank",)
    )
    output_dir = Path(output_root) / "scene_id"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path in {None, BUILTIN_SCENE_ID_MANIFEST}:
        manifest_file = _materialize_builtin_manifest(
            config, dataset, output_root, output_dir
        )
    else:
        manifest_file = Path(manifest_path).resolve()
    manifest = read_strict_json(manifest_file)
    _validate_manifest(manifest)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    verified = []
    bound_payload = {**manifest, "adapters": [dict(value) for value in manifest["adapters"]]}
    for adapter, bound_adapter in zip(manifest["adapters"], bound_payload["adapters"]):
        adapter_source, checkpoint, training_provenance = _verify_adapter_files(
            adapter, manifest_file.parent
        )
        _validate_training_provenance(
            adapter, training_provenance, evidence
        )
        bound_adapter["adapter_source_path"] = str(adapter_source)
        bound_adapter["model_checkpoint_path"] = str(checkpoint)
        bound_adapter["training_provenance_path"] = str(training_provenance)
        verified.append((adapter, adapter_source, checkpoint, training_provenance))
    bound_manifest = output_dir / "adapter_manifest.json"
    write_json(bound_manifest, bound_payload)
    all_rows = []
    model_rows = []
    for adapter, adapter_source, checkpoint, training_provenance in verified:
        target = output_dir / "adapters" / adapter["adapter_id"]
        target.mkdir(parents=True, exist_ok=True)
        command = [
            value.format(
                dataset=str(dataset.source_path),
                output=str(target),
                checkpoint=str(checkpoint),
                adapter_source=str(adapter_source),
                python=sys.executable,
                output_run_root=str(upstream.run_root),
                upstream_root=str(_upstream_run_root(upstream)),
            )
            for value in adapter["command"]
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=_adapter_environment(Path(__file__).resolve().parents[1]),
        )
        (target / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (target / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result = target / "scene_id_results.csv"
        if completed.returncode != 0 or not result.is_file():
            raise RuntimeError(f"scene-ID adapter {adapter['adapter_id']} failed")
        with result.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        _validate_rows(adapter, rows, dataset, evidence)
        all_rows.extend(rows)
        assessment = _model_assessment(config, adapter, rows)
        assessment["implementation_revision"] = adapter["implementation_revision"]
        assessment["adapter_source_sha256"] = adapter["adapter_source_sha256"]
        assessment["model_checkpoint_sha256"] = adapter["model_checkpoint_sha256"]
        assessment["training_provenance_sha256"] = adapter[
            "training_provenance_sha256"
        ]
        model_rows.append(assessment)
    passed = bool(model_rows and all(row["passed"] for row in model_rows))
    write_csv(output_dir / "per_unit.csv", bind_rows(all_rows, evidence))
    write_csv(output_dir / "per_model.csv", bind_rows(model_rows, evidence))
    gate = {
        "schema_version": "csi-pairs-v6-scene-id-gate-v3",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "claim": "C2",
        "models_assessed": len(model_rows),
        "model_assessments": model_rows,
        "input_manifest_path": bound_manifest.name,
        "input_manifest_sha256": sha256_file(bound_manifest),
        "scope": "held-out source positions only",
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


def _materialize_builtin_manifest(config, dataset, output_root, output_dir):
    from .formal_external import _validate_execution_manifest, _validate_manifest as validate_external_manifest

    root = Path(output_root).resolve()
    external_root = root / "external_baselines"
    external_manifest_path = external_root / "adapter_manifest.json"
    if not external_manifest_path.is_file():
        raise RuntimeError("built-in scene-ID audit requires the external-baseline stage")
    external_manifest = read_strict_json(external_manifest_path)
    validate_external_manifest(external_manifest)
    matches = [
        adapter
        for adapter in external_manifest["adapters"]
        if adapter["adapter_id"] == "sigmap-controlled-csi-pairs-v1"
    ]
    if len(matches) != 1:
        raise RuntimeError("built-in scene-ID audit requires the frozen SigMap adapter")
    external_adapter = matches[0]
    adapter_root = external_root / "adapters" / external_adapter["adapter_id"]
    results_path = adapter_root / "six_condition_results.csv"
    _validate_execution_manifest(
        external_adapter, adapter_root, results_path, dataset
    )
    execution_path = adapter_root / "execution_manifest.json"
    execution = read_strict_json(execution_path)
    checkpoint = (adapter_root / execution["checkpoint_path"]).resolve()
    training_record = (adapter_root / execution["training_record_path"]).resolve()
    source = (
        Path(__file__).resolve().parent
        / "external_adapters"
        / "scene_id_sigmap.py"
    ).resolve()
    if not source.is_file():
        raise RuntimeError("built-in scene-ID adapter source is missing")
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    inputs = output_dir / "builtin_inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    provenance_path = inputs / "sigmap_training_provenance.json"
    write_json(
        provenance_path,
        {
            "schema_version": "csi-pairs-v6-scene-id-training-provenance-v1",
            "adapter_id": "sigmap-scene-id-v1",
            "model_name": "SigMap",
            "dataset_sha256": evidence["dataset_sha256"],
            "config_sha256": evidence["config_sha256"],
            "fixture": evidence["fixture"],
            "training_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "evaluation_role": "source_final_unseen_bank",
            "target_data_used": False,
            "checkpoint_rule": "source_selection_then_frozen",
            "adapter_source_sha256": sha256_file(source),
            "model_checkpoint_sha256": sha256_file(checkpoint),
            "training_record_path": str(training_record),
            "training_record_sha256": sha256_file(training_record),
            "training_command_sha256": execution["command_sha256"],
            "external_execution_manifest_sha256": sha256_file(execution_path),
        },
    )
    manifest_path = inputs / "scene_id_manifest.json"
    source_hash = sha256_file(source)
    write_json(
        manifest_path,
        {
            "schema_version": "csi-pairs-v6-scene-id-adapters-v3",
            "adapters": [
                {
                    "adapter_id": "sigmap-scene-id-v1",
                    "model_name": "SigMap",
                    "implementation_revision": source_hash,
                    "adapter_source_path": str(source),
                    "adapter_source_sha256": source_hash,
                    "model_checkpoint_path": str(checkpoint),
                    "model_checkpoint_sha256": sha256_file(checkpoint),
                    "training_provenance_path": str(provenance_path),
                    "training_provenance_sha256": sha256_file(provenance_path),
                    "command": [
                        "{python}", "{adapter_source}",
                        "--dataset", "{dataset}",
                        "--output", "{output}",
                        "--checkpoint", "{checkpoint}",
                        "--output-run-root", "{output_run_root}",
                        "--upstream-root", "{upstream_root}",
                    ],
                }
            ],
        },
    )
    return manifest_path


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "adapters"}:
        raise ValueError("scene-ID manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-scene-id-adapters-v3":
        raise ValueError("scene-ID manifest schema mismatch")
    if not isinstance(manifest["adapters"], list) or not manifest["adapters"]:
        raise ValueError("scene-ID audit requires at least one map-conditioned model adapter")
    seen = set()
    for adapter in manifest["adapters"]:
        if set(adapter) != {
            "adapter_id", "model_name", "implementation_revision", "adapter_source_path",
            "adapter_source_sha256", "model_checkpoint_path", "model_checkpoint_sha256",
            "training_provenance_path", "training_provenance_sha256", "command",
        }:
            raise ValueError("scene-ID adapter fields must be exact")
        if not isinstance(adapter["command"], list) or not adapter["command"]:
            raise ValueError("scene-ID adapter command must be a nonempty argv list")
        adapter_id = adapter["adapter_id"]
        if (
            not isinstance(adapter_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", adapter_id)
            is None
            or adapter_id in {".", ".."}
        ):
            raise ValueError("scene-ID adapter_id must be a safe path component")
        if adapter_id in seen:
            raise ValueError("scene-ID adapter IDs must be unique")
        seen.add(adapter_id)
        if adapter["implementation_revision"] != adapter["adapter_source_sha256"]:
            raise ValueError(
                "scene-ID implementation revision must equal the authenticated source hash"
            )
        if not _command_executes_adapter_source(
            adapter["command"], adapter["adapter_source_path"]
        ):
            raise ValueError(
                "scene-ID command must execute the authenticated adapter module"
            )
        _require_explicit_root_bindings(adapter["command"])
        for key in (
            "adapter_id", "model_name", "implementation_revision", "adapter_source_path",
            "model_checkpoint_path", "training_provenance_path",
        ):
            if not isinstance(adapter[key], str) or not adapter[key].strip():
                raise ValueError(f"scene-ID adapter {key} must be nonempty")
        for key in (
            "adapter_source_sha256", "model_checkpoint_sha256",
            "training_provenance_sha256",
        ):
            digest = adapter[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError(f"scene-ID adapter {key} must be lowercase SHA-256")


def _command_executes_adapter_source(command, adapter_source_path):
    return bool(
        Path(adapter_source_path).suffix == ".py"
        and len(command) >= 2
        and command[:2] == ["{python}", "{adapter_source}"]
    )


def _upstream_run_root(upstream):
    if upstream.migrated:
        return upstream.migration.legacy_run_root
    return upstream.run_root


def _require_explicit_root_bindings(command):
    for option, placeholder in (
        ("--output-run-root", "{output_run_root}"),
        ("--upstream-root", "{upstream_root}"),
    ):
        positions = [index for index, value in enumerate(command) if value == option]
        if (
            len(positions) != 1
            or positions[0] + 1 >= len(command)
            or command[positions[0] + 1] != placeholder
            or command.count(placeholder) != 1
        ):
            raise ValueError(
                "scene-ID command must bind output and upstream roots explicitly"
            )
    if "--run-root" in command or "{run_root}" in command:
        raise ValueError("scene-ID command cannot use the ambiguous run root")


def _adapter_environment(project_root):
    root = str(Path(project_root).resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
    }


def _verify_adapter_files(adapter, manifest_root):
    paths = []
    for prefix in ("adapter_source", "model_checkpoint", "training_provenance"):
        path = Path(adapter[f"{prefix}_path"])
        if not path.is_absolute():
            path = Path(manifest_root) / path
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != adapter[f"{prefix}_sha256"]:
            raise RuntimeError(f"scene-ID {prefix} is missing or hash-mismatched")
        paths.append(path)
    return tuple(paths)


def _validate_training_provenance(adapter, path, evidence):
    payload = read_strict_json(path)
    required = {
        "schema_version", "adapter_id", "model_name", "dataset_sha256",
        "config_sha256", "fixture", "training_role", "selection_role",
        "evaluation_role", "target_data_used", "checkpoint_rule",
        "adapter_source_sha256", "model_checkpoint_sha256",
        "training_record_path", "training_record_sha256",
        "training_command_sha256", "external_execution_manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("scene-ID training provenance fields must be exact")
    if (
        payload["schema_version"] != "csi-pairs-v6-scene-id-training-provenance-v1"
        or payload["adapter_id"] != adapter["adapter_id"]
        or payload["model_name"] != adapter["model_name"]
        or payload["training_role"] != "source_encoder_train"
        or payload["selection_role"] != "source_method_selection"
        or payload["evaluation_role"] != "source_final_unseen_bank"
        or payload["target_data_used"] is not False
        or payload["checkpoint_rule"] != "source_selection_then_frozen"
        or payload["adapter_source_sha256"] != adapter["adapter_source_sha256"]
        or payload["model_checkpoint_sha256"] != adapter["model_checkpoint_sha256"]
        or not _lower_sha256(payload["training_command_sha256"])
        or not _lower_sha256(payload["external_execution_manifest_sha256"])
    ):
        raise RuntimeError("scene-ID checkpoint training provenance violates V6")
    for key in ("dataset_sha256", "config_sha256", "fixture"):
        if payload[key] != evidence[key]:
            raise RuntimeError(f"scene-ID training provenance {key} mismatch")
    training_record = Path(payload["training_record_path"]).resolve()
    if (
        not training_record.is_file()
        or not _lower_sha256(payload["training_record_sha256"])
        or sha256_file(training_record) != payload["training_record_sha256"]
    ):
        raise RuntimeError("scene-ID training record is missing or hash-mismatched")


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_rows(adapter, rows, dataset, evidence):
    from .formal_factorial import _canonical_bank_digest

    required = {
        "unit_id", "model_name", "condition", "bank_id", "position_id",
        "localization_error_m", "prediction_x", "prediction_y", "source_role",
        "dataset_sha256", "config_sha256",
    }
    if not rows or any(set(row) != required for row in rows):
        raise RuntimeError("scene-ID result columns must be exact")
    by_unit = {}
    final_scenes = dataset.indices_for_role("source_final_unseen_bank")
    bank_to_scene = {str(dataset.bank_ids[int(scene)]): int(scene) for scene in final_scenes}
    if len(bank_to_scene) != len(final_scenes):
        raise RuntimeError("scene-ID held-out bank join is not unique")
    expected_identities = {
        (str(dataset.bank_ids[int(scene)]), str(dataset.position_ids[int(scene), int(position)]))
        for scene in final_scenes
        for position in np.flatnonzero(
            dataset.position_roles[int(scene)] == "standard"
        )
    }
    if not expected_identities:
        raise RuntimeError("scene-ID held-out registry has no standard positions")
    observed_pairs = set()
    identity_to_unit = {}
    for row in rows:
        if row["model_name"] != adapter["model_name"]:
            raise RuntimeError("scene-ID result model name mismatch")
        if row["source_role"] != "source_final_unseen_bank" or row["bank_id"] not in bank_to_scene:
            raise RuntimeError("scene-ID audit may only use held-out source banks")
        if row["dataset_sha256"] != str(evidence["dataset_sha256"]) or row["config_sha256"] != str(evidence["config_sha256"]):
            raise RuntimeError("scene-ID result evidence hash mismatch")
        if (
            not np.isfinite(float(row["localization_error_m"]))
            or float(row["localization_error_m"]) < 0.0
            or not np.isfinite(float(row["prediction_x"]))
            or not np.isfinite(float(row["prediction_y"]))
        ):
            raise RuntimeError(
                "scene-ID localization error must be finite and nonnegative; "
                "prediction coordinates must be finite"
            )
        if row["condition"] not in CONDITIONS:
            raise RuntimeError("scene-ID result contains an unknown condition")
        scene = bank_to_scene[row["bank_id"]]
        positions = np.flatnonzero(dataset.position_ids[scene] == row["position_id"])
        if positions.size != 1 or str(dataset.position_roles[scene, int(positions[0])]) != "standard":
            raise RuntimeError("scene-ID result is not bound to one held-out standard position")
        pair = (row["unit_id"], row["condition"])
        if pair in observed_pairs:
            raise RuntimeError("scene-ID result duplicates a unit/condition cell")
        observed_pairs.add(pair)
        identity = (row["model_name"], row["bank_id"], row["position_id"])
        registry_identity = (row["bank_id"], row["position_id"])
        if registry_identity not in expected_identities:
            raise RuntimeError("scene-ID row is outside the held-out position registry")
        prior_unit = identity_to_unit.setdefault(registry_identity, row["unit_id"])
        if prior_unit != row["unit_id"]:
            raise RuntimeError("scene-ID registry identity is duplicated under another unit ID")
        unit = by_unit.setdefault(row["unit_id"], {"conditions": set(), "identity": identity})
        if unit["identity"] != identity:
            raise RuntimeError("scene-ID four-condition unit changes bank or position")
        unit["conditions"].add(row["condition"])
        row["base_map_cluster_id"] = str(dataset.base_map_cluster_ids[scene])
        row["canonical_base_map_digest"] = str(
            dataset.canonical_base_map_digest(scene)
        )
        row["canonical_bank_digest"] = str(_canonical_bank_digest(dataset, scene))
        row["canonical_unit_id"] = (
            f"{row['canonical_bank_digest']}:{row['position_id']}"
        )
        row["scene_index"] = scene
        for key in ("dataset_sha256", "config_sha256"):
            row.pop(key)
    if any(value["conditions"] != set(CONDITIONS) for value in by_unit.values()):
        raise RuntimeError("each scene-ID unit must contain the exact four conditions")
    if set(identity_to_unit) != expected_identities:
        raise RuntimeError("scene-ID adapter omitted held-out bank/position units")
    _canonical_units(rows)


def _model_assessment(config, adapter, rows):
    units = _canonical_units(rows)
    clusters, map_values = _canonical_cluster_metric(
        units, "map", "localization_error_m"
    )
    id_clusters, id_values = _canonical_cluster_metric(
        units, "scene_id", "localization_error_m"
    )
    direction_clusters, direction_values = _canonical_cluster_direction_cosines(
        units
    )
    if not (
        np.array_equal(clusters, id_clusters)
        and np.array_equal(clusters, direction_clusters)
    ):
        raise RuntimeError("scene-ID conditions do not share canonical support")
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    noninferiority = paired_cluster_interval(
        clusters, id_values, map_values, resamples, 86001
    )
    direction = paired_cluster_interval(
        direction_clusters,
        direction_values,
        np.zeros_like(direction_values),
        resamples,
        86002,
    )
    margin = float(config["evaluation"]["scene_id_error_noninferiority_m"])
    minimum_direction_cosine = float(
        config["evaluation"]["scene_id_swap_direction_cosine_min"]
    )
    passed = bool(
        noninferiority["ci95_high"] <= margin
        and direction["ci95_low"] >= minimum_direction_cosine
    )
    return {
        "adapter_id": adapter["adapter_id"],
        "model_name": adapter["model_name"],
        "unit_count": len(units),
        "base_map_cluster_count": int(noninferiority["cluster_count"]),
        "map_mean_error_m": float(np.mean(map_values)),
        "scene_id_mean_error_m": float(np.mean(id_values)),
        "scene_id_minus_map_error_m": noninferiority["paired_mean_difference"],
        "scene_id_minus_map_ci95_low": noninferiority["ci95_low"],
        "scene_id_minus_map_ci95_high": noninferiority["ci95_high"],
        "scene_id_error_noninferiority_margin_m": margin,
        "map_swap_id_swap_direction_cosine": direction["paired_mean_difference"],
        "map_swap_id_swap_direction_cosine_ci95_low": direction["ci95_low"],
        "map_swap_id_swap_direction_cosine_ci95_high": direction["ci95_high"],
        "minimum_swap_direction_cosine": minimum_direction_cosine,
        "passed": passed,
    }


def _canonical_units(rows):
    by_raw_unit = {}
    for row in rows:
        required = {
            "unit_id",
            "condition",
            "canonical_unit_id",
            "canonical_base_map_digest",
            "canonical_bank_digest",
            "localization_error_m",
            "prediction_x",
            "prediction_y",
        }
        if not required.issubset(row):
            raise RuntimeError("scene-ID assessment lacks canonical unit fields")
        by_raw_unit.setdefault(str(row["unit_id"]), {})[str(row["condition"])] = row
    canonical = {}
    for conditions in by_raw_unit.values():
        if set(conditions) != set(CONDITIONS):
            raise RuntimeError("scene-ID canonical unit is condition-incomplete")
        reference = conditions["map"]
        key = str(reference["canonical_unit_id"])
        signature = {
            condition: (
                float(row["localization_error_m"]),
                float(row["prediction_x"]),
                float(row["prediction_y"]),
            )
            for condition, row in conditions.items()
        }
        if key in canonical:
            prior_conditions, prior_signature = canonical[key]
            if (
                str(reference["canonical_base_map_digest"])
                != str(prior_conditions["map"]["canonical_base_map_digest"])
                or str(reference["canonical_bank_digest"])
                != str(prior_conditions["map"]["canonical_bank_digest"])
                or any(
                    not np.allclose(signature[name], prior_signature[name])
                    for name in CONDITIONS
                )
            ):
                raise RuntimeError(
                    "copied canonical scene-ID unit has inconsistent results"
                )
            continue
        canonical[key] = (conditions, signature)
    return [canonical[key][0] for key in sorted(canonical)]


def _canonical_cluster_metric(units, condition, metric):
    bank_values = {}
    for conditions in units:
        row = conditions[condition]
        key = (
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
        )
        value = float(row[metric])
        if not np.isfinite(value):
            raise RuntimeError("scene-ID canonical metric must be finite")
        bank_values.setdefault(key, []).append(value)
    cluster_values = {}
    for (cluster, _), values in bank_values.items():
        cluster_values.setdefault(cluster, []).append(float(np.mean(values)))
    clusters = np.asarray(sorted(cluster_values))
    if clusters.size < 2:
        raise RuntimeError("scene-ID inference requires two canonical foundations")
    values = np.asarray(
        [np.mean(cluster_values[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    return clusters, values


def _canonical_cluster_direction_cosines(units):
    bank_values = {}
    for conditions in units:
        map_prediction = _prediction(conditions["map"])
        scene_id_prediction = _prediction(conditions["scene_id"])
        map_displacement = _prediction(conditions["map_swap"]) - map_prediction
        id_displacement = _prediction(conditions["id_swap"]) - scene_id_prediction
        denominator = float(
            np.linalg.norm(map_displacement) * np.linalg.norm(id_displacement)
        )
        if not np.isfinite(denominator) or denominator <= 1e-12:
            raise RuntimeError(
                "scene-ID swap direction is undefined for a zero-displacement unit"
            )
        cosine = float(np.dot(map_displacement, id_displacement) / denominator)
        row = conditions["map"]
        key = (
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
        )
        bank_values.setdefault(key, []).append(cosine)
    cluster_values = {}
    for (cluster, _), values in bank_values.items():
        cluster_values.setdefault(cluster, []).append(float(np.mean(values)))
    clusters = np.asarray(sorted(cluster_values))
    if clusters.size < 2:
        raise RuntimeError("scene-ID inference requires two canonical foundations")
    values = np.asarray(
        [np.mean(cluster_values[cluster]) for cluster in clusters], dtype=np.float64
    )
    return clusters, values


def _prediction(row):
    value = np.asarray(
        [float(row["prediction_x"]), float(row["prediction_y"])],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(value)):
        raise RuntimeError("scene-ID prediction vector must be finite")
    return value
