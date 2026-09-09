from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import bind_rows, evidence_context
from .formal_dataset import _array_sha256
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
from .formal_statistics import interval_decision, paired_cluster_interval
from .formal_resources import validate_resource_registry
from .external_adapters.wigatr_protocol import SIX_CONDITIONS as CONDITIONS


ALLOWED_IMPLEMENTATION_STATUS = {
    "official-code-adaptation",
    "paper-spec-controlled-implementation",
    "style-controlled-implementation",
}
REQUIRED_BASELINE_NAMES = {
    "CSI-MAE",
    "CSI-CLIP",
    "CSI-CLIP++",
    "ContraWiMAE",
    "WWM",
    "SigMap",
    "Wi-GATr",
    "PMNet",
    "WiSER",
    "CSI-only",
    "oracle-x",
    "RFIR",
}
BASELINE_STATUSES = {"executed", "not_executed", "not_applicable", "oracle_only"}
C1_ELIGIBLE_STATUSES = {
    "official-code-adaptation",
    "paper-spec-controlled-implementation",
}


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def run_external_baselines(config, dataset, manifest_path, output_root):
    from .formal_data_verification import require_verified_roles_from_root

    project_root = Path(__file__).resolve().parents[1]
    resource_registry_path = Path(__file__).resolve().parent / "configs/waibu_resources_v1.json"
    validate_resource_registry(
        read_strict_json(resource_registry_path), project_root / "waibu"
    )
    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        (
            "source_encoder_train",
            "source_method_selection",
            "source_final_unseen_bank",
            "target",
        ),
    )
    manifest = read_strict_json(manifest_path)
    _validate_manifest(manifest)
    output_dir = Path(output_root) / "external_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_manifest_copy = output_dir / "adapter_manifest.json"
    write_json(adapter_manifest_copy, manifest)
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    expected_units = _expected_six_condition_units(
        config, dataset, output_root
    )
    expected_unit_ids = {unit.unit_id for unit in expected_units}
    expected_condition_contract = _expected_condition_contract(dataset, expected_units)
    unit_rows = []
    for unit in expected_units:
        unit_rows.append(
            {
                "unit_id": unit.unit_id,
                "city_id": str(dataset.city_ids[unit.scene]),
                "bank_id": str(dataset.bank_ids[unit.scene]),
                "position_id": str(dataset.position_ids[unit.scene, unit.position]),
                "source_world": unit.source_world,
                "active_world": unit.active_world,
                "null_world": unit.null_world,
                "wrong_city_bank_id": str(dataset.bank_ids[unit.wrong_city_scene]),
                "csi_context_sha256": unit.csi_context_sha256,
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[unit.scene]),
                "query_count": 1,
            }
        )
    write_csv(
        output_dir / "external_unit_registry.csv",
        bind_rows(unit_rows, evidence),
    )
    condition_registry_path = output_dir / "external_condition_registry.csv"
    write_csv(
        condition_registry_path,
        bind_rows(
            [
                {
                    "unit_id": unit_id,
                    "condition": condition,
                    **contract,
                }
                for (unit_id, condition), contract in sorted(
                    expected_condition_contract.items()
                )
            ],
            evidence,
        ),
    )
    status_rows = []
    all_rows = []
    model_assessments = []
    for adapter in manifest["adapters"]:
        adapter_output = output_dir / "adapters" / adapter["adapter_id"]
        adapter_output.mkdir(parents=True, exist_ok=True)
        _, command = _resolve_adapter_command(
            adapter,
            dataset_path=dataset.source_path,
            output_path=adapter_output,
            run_root=output_root,
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
                env=_adapter_environment(Path(__file__).resolve().parents[1]),
            )
        except OSError as error:
            completed = subprocess.CompletedProcess(command, 127, "", f"{type(error).__name__}: {error}")
        (adapter_output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (adapter_output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result_path = adapter_output / "six_condition_results.csv"
        if completed.returncode != 0 or not result_path.is_file():
            status_rows.append(
                {
                    "adapter_id": adapter["adapter_id"],
                    "model_name": adapter["model_name"],
                    "implementation_status": adapter["implementation_status"],
                    "c1_eligible": adapter["c1_eligible"],
                    "status": "FAIL",
                    "return_code": completed.returncode,
                }
            )
            continue
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        try:
            _validate_six_condition_rows(
                adapter,
                rows,
                dataset,
                expected_unit_ids=expected_unit_ids,
                expected_unit_contract={unit.unit_id: unit for unit in expected_units},
                expected_condition_contract=expected_condition_contract,
            )
            _validate_execution_manifest(
                adapter, adapter_output, result_path, dataset
            )
            assessment = _c1_model_assessment(
                config, adapter, rows, dataset, expected_unit_contract={
                    unit.unit_id: unit for unit in expected_units
                }
            )
        except (ValueError, RuntimeError) as error:
            status_rows.append(
                {
                    "adapter_id": adapter["adapter_id"],
                    "model_name": adapter["model_name"],
                    "implementation_status": adapter["implementation_status"],
                    "c1_eligible": adapter["c1_eligible"],
                    "status": "FAIL",
                    "return_code": completed.returncode,
                    "reason": f"evidence contract failed: {type(error).__name__}: {error}",
                }
            )
            continue
        all_rows.extend(rows)
        model_assessments.append(assessment)
        status_rows.append(
            {
                "adapter_id": adapter["adapter_id"],
                "model_name": adapter["model_name"],
                "implementation_status": adapter["implementation_status"],
                "c1_eligible": adapter["c1_eligible"],
                "status": "PASS",
                "return_code": completed.returncode,
                "reason": "authenticated execution and six-condition input contract",
            }
        )
    write_csv(output_dir / "adapter_status.csv", bind_rows(status_rows, evidence))
    resolved_registry = []
    passed_adapter_ids = {
        row["adapter_id"] for row in status_rows if row["status"] == "PASS"
    }
    for row in manifest["literature_registry"]:
        resolved = dict(row)
        if resolved["status"] == "executed" and resolved["adapter_id"] not in passed_adapter_ids:
            resolved["status"] = "not_executed"
            resolved["reason"] = resolved["reason"] + " Adapter execution did not authenticate in this run."
            resolved["adapter_id"] = ""
        resolved_registry.append(resolved)
    write_csv(output_dir / "literature_baseline_registry.csv", bind_rows(resolved_registry, evidence))
    write_csv(output_dir / "six_condition_results.csv", bind_rows(all_rows, evidence))
    passed_models = {
        row["model_name"] for row in status_rows if row["status"] == "PASS"
    }
    assessment_by_model = {row["model_name"]: row for row in model_assessments}
    c1_eligible_models = {
        row["model_name"] for row in status_rows
        if row["status"] == "PASS"
        and row["c1_eligible"] is True
        and assessment_by_model[row["model_name"]]["passed"] is True
    }
    passed = len(c1_eligible_models) >= 2
    gate = {
        "schema_version": "csi-pairs-v6-external-baseline-gate-v3",
        "status": "PASS" if passed else "BLOCKED",
        "passed": passed,
        **evidence,
        "gate_scope": "C1 six-condition domain evidence",
        "passing_map_conditioned_models": len(passed_models),
        "unique_passing_model_count": len(passed_models),
        "unique_passing_models": sorted(passed_models),
        "c1_eligible_model_count": len(c1_eligible_models),
        "c1_eligible_models": sorted(c1_eligible_models),
        "c1_required_eligible_model_count": 2,
        "c1_city_gate_contract": "all-evaluation-cities-must-pass-v1",
        "model_assessments": model_assessments,
        "condition_input_contract": "outer-recomputed-map-and-action-sha256-v1",
        "condition_registry_path": condition_registry_path.name,
        "condition_registry_sha256": sha256_file(condition_registry_path),
        "adapter_manifest_sha256": sha256_file(adapter_manifest_copy),
        "adapter_manifest_path": adapter_manifest_copy.name,
        "resource_registry_sha256": sha256_file(resource_registry_path),
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


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "adapters", "literature_registry"}:
        raise ValueError("external adapter manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-external-adapters-v2":
        raise ValueError("external adapter manifest schema mismatch")
    adapters = manifest["adapters"]
    if not isinstance(adapters, list) or len(adapters) < 2:
        raise ValueError("at least two external map-conditioned adapters are required")
    seen = set()
    eligible_models = set()
    for adapter in adapters:
        required = {
            "adapter_id",
            "model_name",
            "implementation_status",
            "license_id",
            "citation_key",
            "source_revision",
            "adapter_source_path",
            "adapter_source_sha256",
            "adapter_config_path",
            "adapter_config_sha256",
            "map_conditioned",
            "c1_eligible",
            "command",
        }
        if not isinstance(adapter, dict) or set(adapter) != required:
            raise ValueError("external adapter fields must be exact")
        adapter_id = adapter["adapter_id"]
        if (
            not isinstance(adapter_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", adapter_id)
            is None
            or adapter_id in {".", ".."}
        ):
            raise ValueError("external adapter_id must be a safe path component")
        if adapter_id in seen:
            raise ValueError("external adapter IDs must be unique")
        seen.add(adapter_id)
        if adapter["implementation_status"] not in ALLOWED_IMPLEMENTATION_STATUS:
            raise ValueError("external adapter implementation status is inaccurate or unsupported")
        if adapter["map_conditioned"] is not True:
            raise ValueError("C1 adapters must actually be map-conditioned")
        if not isinstance(adapter["c1_eligible"], bool):
            raise ValueError("external adapter c1_eligible must be boolean")
        if adapter["c1_eligible"] and adapter["implementation_status"] == "style-controlled-implementation":
            raise ValueError("style-controlled adapters cannot support C1")
        if adapter["c1_eligible"]:
            if adapter["implementation_status"] not in C1_ELIGIBLE_STATUSES:
                raise ValueError("C1-eligible adapter must be official-code or paper-spec controlled")
            if adapter["model_name"] in eligible_models:
                raise ValueError("C1-eligible model identities must be unique")
            eligible_models.add(adapter["model_name"])
        if (
            not isinstance(adapter["command"], list)
            or not adapter["command"]
            or not all(
                isinstance(value, str) and value for value in adapter["command"]
            )
        ):
            raise ValueError("external adapter command must be a nonempty argv list")
        if not _command_executes_adapter_source(
            adapter["command"], adapter["adapter_source_path"]
        ):
            raise ValueError(
                "external adapter command must execute the authenticated source directly"
            )
        if not _command_binds_adapter_config(adapter["command"]):
            raise ValueError(
                "external adapter command must contain exactly one --config {adapter_config} binding"
            )
        if not all(isinstance(adapter[key], str) and adapter[key].strip() for key in ("citation_key", "source_revision", "license_id")):
            raise ValueError("external adapter provenance fields must be nonempty")
        _verified_project_file(
            adapter["adapter_source_path"],
            adapter["adapter_source_sha256"],
            "source",
        )
        _verified_project_file(
            adapter["adapter_config_path"],
            adapter["adapter_config_sha256"],
            "config",
        )
    registry = manifest["literature_registry"]
    if not isinstance(registry, list) or {row.get("baseline_name") for row in registry} != REQUIRED_BASELINE_NAMES:
        raise ValueError("literature registry must contain every frozen V6 baseline name")
    for row in registry:
        if set(row) != {"baseline_name", "status", "adapter_id", "reason"}:
            raise ValueError("literature baseline registry fields must be exact")
        if row["status"] not in BASELINE_STATUSES:
            raise ValueError("literature baseline registry has an invalid status")
        if row["status"] == "executed" and row["adapter_id"] not in seen:
            raise ValueError("executed literature baseline must reference an adapter")
        if row["status"] != "executed" and row["adapter_id"] != "":
            raise ValueError("unexecuted literature baseline may not claim an adapter")


def _command_executes_adapter_source(command, adapter_source_path):
    source = Path(adapter_source_path)
    if source.suffix != ".py":
        return False
    allowed_interpreters = {
        "{python}",
        "{project_root}/formal_v2/external_adapters/.venv-wigatr/bin/python",
    }
    return bool(
        len(command) >= 2
        and command[0] in allowed_interpreters
        and command[1] == "{adapter_source}"
    )


def _command_binds_adapter_config(command):
    config_options = [
        index
        for index, value in enumerate(command)
        if value == "--config" or value.startswith("--config=")
    ]
    return bool(
        len(config_options) == 1
        and command[config_options[0]] == "--config"
        and config_options[0] + 1 < len(command)
        and command[config_options[0] + 1] == "{adapter_config}"
        and command.count("{adapter_config}") == 1
    )


def _verified_project_file(relative_path, expected_sha256, label):
    if (
        not isinstance(relative_path, str)
        or not relative_path.strip()
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
        or not _lower_sha256(expected_sha256)
    ):
        raise ValueError(
            f"external adapter {label} path/hash is missing or mismatched"
        )
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / relative_path
    cursor = project_root
    for part in Path(relative_path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"external adapter {label} path/hash is missing or mismatched"
            )
    resolved = candidate.resolve()
    if (
        project_root not in resolved.parents
        or not candidate.is_file()
        or sha256_file(candidate) != expected_sha256
    ):
        raise ValueError(
            f"external adapter {label} path/hash is missing or mismatched"
        )
    return resolved


def _adapter_environment(project_root):
    root = str(Path(project_root).resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
    }


def _resolve_adapter_command(adapter, *, dataset_path, output_path, run_root):
    adapter_source = _verified_project_file(
        adapter["adapter_source_path"],
        adapter["adapter_source_sha256"],
        "source",
    )
    adapter_config = _verified_project_file(
        adapter["adapter_config_path"],
        adapter["adapter_config_sha256"],
        "config",
    )
    command_digest = hashlib.sha256(
        json.dumps(
            adapter["command"], separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    command = [
        value.format(
            dataset=str(Path(dataset_path).resolve()),
            output=str(Path(output_path).resolve()),
            run_root=str(Path(run_root).resolve()),
            project_root=str(Path(__file__).resolve().parents[1]),
            adapter_command_sha256=command_digest,
            adapter_id=adapter["adapter_id"],
            model_name=adapter["model_name"],
            source_revision=adapter["source_revision"],
            adapter_source=str(adapter_source),
            adapter_config=str(adapter_config),
            python=sys.executable,
        )
        for value in adapter["command"]
    ]
    return command_digest, command


def _validate_six_condition_rows(
    adapter,
    rows,
    dataset,
    *,
    expected_unit_ids=None,
    expected_unit_contract=None,
    expected_condition_contract=None,
):
    if not rows:
        raise ValueError("external adapter emitted no result rows")
    by_unit = {}
    required_columns = {
        "unit_id",
        "model_name",
        "condition",
        "city_id",
        "bank_id",
        "position_id",
        "localization_error_m",
        "csi_context_sha256",
        "base_map_cluster_id",
        "map_sha256",
        "action_sha256",
        "query_count",
    }
    seen_rows = set()
    for row in rows:
        if set(row) != required_columns:
            raise ValueError("external result columns must be exact")
        if row.get("model_name") != adapter["model_name"]:
            raise ValueError("external result model name does not match its manifest")
        if not row.get("unit_id") or not row.get("bank_id") or not row.get("position_id"):
            raise ValueError("external result unit keys must be nonempty")
        scene_matches = [
            index for index, value in enumerate(dataset.bank_ids.tolist()) if str(value) == row["bank_id"]
        ]
        if len(scene_matches) != 1:
            raise ValueError("external result bank_id is not unique in the current dataset")
        scene = scene_matches[0]
        if row["city_id"] != str(dataset.city_ids[scene]):
            raise ValueError("external result city_id does not match its bank")
        if row["position_id"] not in set(dataset.position_ids[scene].tolist()):
            raise ValueError("external result position_id is absent from its bank")
        position = int(np.flatnonzero(dataset.position_ids[scene] == row["position_id"])[0])
        if str(dataset.scene_roles[scene]) == "target" and str(dataset.position_roles[scene, position]) != "query":
            raise ValueError("external baseline includes target support_pool in its denominator")
        try:
            value = float(row["localization_error_m"])
        except ValueError as error:
            raise ValueError("external localization error must be numeric") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError("external localization error must be nonnegative")
        key = (row.get("unit_id"), row.get("condition"))
        by_unit.setdefault(row.get("unit_id"), set()).add(row.get("condition"))
        if key in seen_rows:
            raise ValueError("external result duplicates a unit-condition row")
        seen_rows.add(key)
    required = set(CONDITIONS)
    if any(conditions != required for conditions in by_unit.values()):
        raise ValueError("every external unit must contain exactly the six frozen conditions")
    if expected_unit_ids is not None and set(by_unit) != set(expected_unit_ids):
        missing = sorted(set(expected_unit_ids).difference(by_unit))
        unexpected = sorted(set(by_unit).difference(expected_unit_ids))
        raise ValueError(
            "external adapter does not cover the frozen common unit registry: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if expected_unit_contract is not None:
        for unit_id, unit in expected_unit_contract.items():
            selected = [row for row in rows if row["unit_id"] == unit_id]
            expected = {
                "city_id": str(dataset.city_ids[unit.scene]),
                "bank_id": str(dataset.bank_ids[unit.scene]),
                "position_id": str(dataset.position_ids[unit.scene, unit.position]),
                "csi_context_sha256": unit.csi_context_sha256,
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[unit.scene]),
                "query_count": "1",
            }
            for key, value in expected.items():
                if any(str(row[key]) != value for row in selected):
                    raise ValueError(
                        f"external unit {unit_id!r} changes frozen {key}"
                    )
    if expected_condition_contract is not None:
        for row in rows:
            key = (row["unit_id"], row["condition"])
            expected = expected_condition_contract.get(key)
            if expected is None:
                raise ValueError("external result has no outer-generated condition contract")
            for field in ("map_sha256", "action_sha256", "base_map_cluster_id"):
                if str(row[field]) != str(expected[field]):
                    raise ValueError(
                        f"external {key!r} {field} differs from outer recomputation"
                    )
    for unit_id in by_unit:
        selected = [row for row in rows if row["unit_id"] == unit_id]
        if len({row["csi_context_sha256"] for row in selected}) != 1:
            raise ValueError("six-condition unit changes the frozen CSI/radio context")
        if len({row["query_count"] for row in selected}) != 1:
            raise ValueError("six-condition unit changes the evaluation denominator")
        digest = selected[0]["csi_context_sha256"]
        if not _lower_sha256(digest):
            raise ValueError("external CSI context digest must be lowercase SHA-256")
        if any(not _lower_sha256(row[field]) for row in selected for field in ("map_sha256", "action_sha256")):
            raise ValueError("external map/action digests must be lowercase SHA-256")
        try:
            query_count = int(selected[0]["query_count"])
        except ValueError as error:
            raise ValueError("external query_count must be an integer") from error
        if query_count <= 0:
            raise ValueError("external query_count must be positive")


def _expected_condition_contract(dataset, units):
    from .external_adapters.wigatr_protocol import condition_map

    result = {}
    for unit in units:
        cluster = str(dataset.base_map_cluster_ids[unit.scene])
        for condition in CONDITIONS:
            result[(unit.unit_id, condition)] = {
                "base_map_cluster_id": cluster,
                "map_sha256": _array_sha256(condition_map(dataset, unit, condition)),
                "action_sha256": _condition_action_sha256(dataset, unit, condition),
            }
    return result


def _condition_action_sha256(dataset, unit, condition):
    target_world = None
    if condition == "paired_active_alternative":
        target_world = int(unit.active_world)
    elif condition == "paired_null_alternative":
        target_world = int(unit.null_world)
    payload = {"condition": condition, "action": None}
    if target_world is not None:
        matches = [
            edge for edge in dataset.directed_edges(int(unit.scene))
            if int(edge.source_world) == int(unit.source_world)
            and int(edge.target_world) == target_world
        ]
        if len(matches) != 1:
            raise RuntimeError("six-condition action is not a unique directed edit")
        edge = matches[0]
        payload["action"] = {
            "source_world": int(edge.source_world),
            "target_world": int(edge.target_world),
            "bit_index": int(edge.bit_index),
            "primitive_id": int(edge.primitive_id),
            "direction": int(edge.direction),
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _c1_model_assessment(config, adapter, rows, dataset, *, expected_unit_contract):
    from .formal_factorial import _canonical_bank_digest

    lookup = {(row["unit_id"], row["condition"]): row for row in rows}
    units = sorted(expected_unit_contract)
    grouped_units = {}
    for unit in units:
        scene = int(expected_unit_contract[unit].scene)
        foundation_method = getattr(dataset, "canonical_base_map_digest", None)
        foundation = (
            foundation_method(scene)
            if callable(foundation_method)
            else str(dataset.base_map_cluster_ids[scene])
        )
        bank = (
            _canonical_bank_digest(dataset, scene)
            if all(
                hasattr(dataset, field)
                for field in ("maps", "csi", "positions", "radio_config", "bs_pose")
            )
            else str(dataset.bank_ids[scene])
        )
        key = (
            str(dataset.city_ids[scene]),
            foundation,
            bank,
        )
        grouped_units.setdefault(key, []).append(unit)
    cells = sorted(grouped_units)
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    active_minimum = float(config["evaluation"]["c1_active_error_minimum_m"])
    margin = float(config["evaluation"]["c1_null_error_equivalence_margin_m"])
    maximum_rate = float(config["evaluation"]["null_overclassification_rate_max"])

    def assess(selected_cells, seed):
        selected_cells = sorted(selected_cells)
        clusters = np.asarray([key[1] for key in selected_cells])
        cluster_count = int(np.unique(clusters).size)
        if cluster_count < 2:
            return {
                "base_map_cluster_count": cluster_count,
                "active_effect_passed": False,
                "null_safety_passed": False,
                "passed": False,
                "reason": "fewer than two independent base-map clusters",
            }

        def collapsed(condition):
            return np.asarray(
                [
                    np.mean(
                        [
                            float(
                                lookup[(unit, condition)]["localization_error_m"]
                            )
                            for unit in grouped_units[key]
                        ]
                    )
                    for key in selected_cells
                ],
                dtype=np.float64,
            )

        correct = collapsed("correct")
        active = collapsed("paired_active_alternative")
        null = collapsed("paired_null_alternative")
        active_interval = paired_cluster_interval(
            clusters, active, correct, resamples, seed
        )
        null_interval = paired_cluster_interval(
            clusters, null, correct, resamples, seed + 1
        )
        stress_intervals = {
            condition: paired_cluster_interval(
                clusters,
                collapsed(condition),
                correct,
                resamples,
                seed + 10 + index,
            )
            for index, condition in enumerate(
                ("wrong_city", "geometry_destroyed", "empty")
            )
        }
        cluster_rates = []
        for cluster in sorted(set(clusters.tolist())):
            selected = clusters == cluster
            cluster_rates.append(
                float(np.mean((null[selected] - correct[selected]) > margin))
            )
        overclassification_rate = float(np.mean(cluster_rates))
        rate_rng = np.random.default_rng(seed + 2)
        rates = np.asarray(cluster_rates)
        rate_samples = np.asarray(
            [
                np.mean(
                    rates[
                        rate_rng.integers(0, len(cluster_rates), size=len(cluster_rates))
                    ]
                )
                for _ in range(resamples)
            ],
            dtype=np.float64,
        )
        overclassification_rate_high = float(np.percentile(rate_samples, 97.5))
        active_passed = interval_decision(
            active_interval, threshold=active_minimum, relation="superiority"
        )
        null_passed = bool(
            interval_decision(null_interval, threshold=margin, relation="equivalence")
            and overclassification_rate_high <= maximum_rate
        )
        return {
            "base_map_cluster_count": cluster_count,
            "active_error_minus_correct": active_interval,
            "active_error_minimum_m": active_minimum,
            "active_effect_passed": active_passed,
            "null_error_minus_correct": null_interval,
            "null_equivalence_margin_m": margin,
            "null_overclassification_rate": overclassification_rate,
            "null_overclassification_rate_ci95_high": overclassification_rate_high,
            "null_overclassification_rate_max": maximum_rate,
            "null_safety_passed": null_passed,
            "stress_condition_error_minus_correct": stress_intervals,
            "passed": bool(active_passed and null_passed),
        }

    pooled = assess(cells, 86101)
    cities = sorted({key[0] for key in cells})
    city_assessments = {
        city: {
            "city_id": city,
            **assess(
                [key for key in cells if key[0] == city],
                86201 + 20 * index,
            ),
        }
        for index, city in enumerate(cities)
    }
    all_cities_passed = bool(
        city_assessments
        and all(row["passed"] is True for row in city_assessments.values())
    )
    return {
        "adapter_id": adapter["adapter_id"],
        "model_name": adapter["model_name"],
        "c1_eligible": adapter["c1_eligible"],
        **pooled,
        "aggregation": "pooled-report-plus-simultaneous-per-city-gate",
        "evaluation_city_count": len(cities),
        "evaluation_cities": cities,
        "city_assessments": city_assessments,
        "all_cities_passed": all_cities_passed,
        "passed": bool(pooled["passed"] and all_cities_passed),
    }


def _validate_execution_manifest(adapter, output_dir, result_path, dataset):
    path = output_dir / "execution_manifest.json"
    if not path.is_file():
        raise RuntimeError("external adapter omitted execution_manifest.json")
    payload = read_strict_json(path)
    required = {
        "schema_version",
        "adapter_id",
        "model_name",
        "implementation_status",
        "source_revision",
        "dataset_sha256",
        "adapter_config_path",
        "adapter_config_sha256",
        "training_record_path",
        "training_record_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "command_sha256",
        "results_sha256",
    }
    external_wigatr_runtime = ".venv-wigatr" in adapter["command"][0]
    main_pmnet_runtime = adapter["model_name"] == "PMNet"
    if external_wigatr_runtime or main_pmnet_runtime:
        required.update(
            {
                "runtime_provenance_path",
                "runtime_provenance_sha256",
            }
        )
    if external_wigatr_runtime:
        required.add("runtime_environment_sha256")
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("external execution manifest fields must be exact")
    if external_wigatr_runtime:
        expected_schema = "csi-pairs-v6-external-execution-v3"
    elif main_pmnet_runtime:
        expected_schema = "csi-pairs-v6-external-execution-v4"
    else:
        expected_schema = "csi-pairs-v6-external-execution-v2"
    if payload["schema_version"] != expected_schema:
        raise RuntimeError("external execution manifest schema mismatch")
    for key in (
        "adapter_id",
        "model_name",
        "implementation_status",
        "source_revision",
    ):
        if payload[key] != adapter[key]:
            raise RuntimeError(f"external execution {key} mismatch")
    if payload["dataset_sha256"] != sha256_file(dataset.source_path):
        raise RuntimeError("external execution dataset hash mismatch")
    command_digest = hashlib.sha256(
        json.dumps(adapter["command"], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if payload["command_sha256"] != command_digest:
        raise RuntimeError("external execution command hash mismatch")
    if payload["adapter_config_sha256"] != adapter["adapter_config_sha256"]:
        raise RuntimeError(
            "external execution adapter config differs from the outer frozen hash"
        )
    for prefix in ("adapter_config", "training_record", "checkpoint"):
        artifact = (output_dir / payload[f"{prefix}_path"]).resolve()
        if output_dir.resolve() not in artifact.parents or not artifact.is_file():
            raise RuntimeError(f"external {prefix} is missing or escapes adapter output")
        if sha256_file(artifact) != payload[f"{prefix}_sha256"]:
            raise RuntimeError(f"external {prefix} hash mismatch")
    adapter_config = read_strict_json(output_dir / payload["adapter_config_path"])
    if not dataset.is_fixture and adapter_config.get("profile") != "formal-paper-dose":
        raise RuntimeError("scientific external adapter did not use its formal-paper-dose profile")
    training_record = read_strict_json(output_dir / payload["training_record_path"])
    if (
        training_record.get("train_role") != "source_encoder_train"
        or training_record.get("selection_role") != "source_method_selection"
        or training_record.get("target_roles_read") != []
    ):
        raise RuntimeError("external training record violates the source-only role ledger")
    if sha256_file(result_path) != payload["results_sha256"]:
        raise RuntimeError("external result hash mismatch")
    if external_wigatr_runtime:
        from .formal_external_runtime import (
            probe_external_runtime,
            validate_external_runtime,
        )

        runtime_path = (output_dir / payload["runtime_provenance_path"]).resolve()
        if output_dir.resolve() not in runtime_path.parents or not runtime_path.is_file():
            raise RuntimeError(
                "external runtime provenance is missing or escapes adapter output"
            )
        if sha256_file(runtime_path) != payload["runtime_provenance_sha256"]:
            raise RuntimeError("external runtime provenance hash mismatch")
        runtime_record = read_strict_json(runtime_path)
        project_root = Path(__file__).resolve().parents[1]
        executable = adapter["command"][0].replace(
            "{project_root}", str(project_root)
        )
        validate_external_runtime(
            runtime_record,
            profile="wigatr",
            executable=executable,
            require_execution_ready=True,
        )
        independently_probed = probe_external_runtime(
            executable,
            "wigatr",
            project_root,
            require_execution_ready=True,
        )
        if runtime_record != independently_probed:
            raise RuntimeError(
                "external runtime provenance differs from the independently probed interpreter"
            )
        if payload["runtime_environment_sha256"] != runtime_record["environment_sha256"]:
            raise RuntimeError("external runtime environment digest mismatch")
    elif main_pmnet_runtime:
        from .formal_evidence import runtime_provenance, validate_runtime_provenance

        runtime_path = (output_dir / payload["runtime_provenance_path"]).resolve()
        if output_dir.resolve() not in runtime_path.parents or not runtime_path.is_file():
            raise RuntimeError("PMNet runtime provenance is missing or escapes adapter output")
        if sha256_file(runtime_path) != payload["runtime_provenance_sha256"]:
            raise RuntimeError("PMNet runtime provenance hash mismatch")
        runtime_record = validate_runtime_provenance(read_strict_json(runtime_path))
        independently_probed = validate_runtime_provenance(runtime_provenance())
        if runtime_record != independently_probed:
            raise RuntimeError(
                "PMNet runtime provenance differs from the independently probed interpreter"
            )


def _expected_six_condition_units(config, dataset, output_root):
    from .external_adapters.wigatr_protocol import build_six_condition_units
    from .formal_evidence import require_manifested_formal_qualification
    from .formal_routing import fit_route_normalization, route_dataset
    from .formal_teacher import load_teacher_bundle

    qualification = read_strict_json(
        Path(output_root) / "qualification" / "gate.json"
    )
    qualification = require_manifested_formal_qualification(
        qualification,
        config,
        dataset,
        allow_nonscientific_fixture=True,
    )
    teacher = load_teacher_bundle(qualification["teacher_checkpoint"], config)
    normalization = fit_route_normalization(dataset, teacher)
    scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    routed = route_dataset(
        dataset, teacher, config, scenes, normalization=normalization
    )
    return build_six_condition_units(dataset, routed)
