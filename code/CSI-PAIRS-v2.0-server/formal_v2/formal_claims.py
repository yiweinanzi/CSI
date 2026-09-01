from __future__ import annotations

import csv
from pathlib import Path

from .formal_evidence import (
    CLAIM_IDS,
    GATE_IDS,
    QUALIFICATION_SCHEMA,
    blocked_claim_vector,
    evidence_context,
    require_stage_manifested_gate,
)
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json


OPTIONAL_CLAIM_GATES = frozenset({"G0", "G8"})
OPTIONAL_CLAIM_STAGES = frozenset({"G0", "G8", "rt_calibration", "wrong_map"})
DESCRIPTIVE_PUBLISHABLE = "DESCRIPTIVE_PUBLISHABLE"
NON_CLAIM = "NON_CLAIM"
ENGINEERING_UNAPPROVED_NAME = "ENGINEERING_UNAPPROVED.json"
_CORRUPT_STAGE_STATUSES = frozenset({"INVALID"})
_SCIENTIFIC_FAIL_STATUSES = frozenset({"FAIL"})
_INCOMPLETE_ASSESSMENT = "INCOMPLETE"
_INCOMPLETE_PAYLOAD_STATUSES = frozenset(
    {"INCOMPLETE_FAIL_CLOSED", "ERROR", "PARTIAL"}
)
_CLI_ASSEMBLE_CLAIMS_EXIT_NOTE = (
    "formal_cli._stage_exit_code uses assemble_claims_exit_code for "
    "assemble-claims/all: scientific FAIL is not a process failure. "
    "Exit 1 only on incomplete or corrupt required evidence."
)
_DESCRIPTIVE_ARTIFACT_NAMES = (
    ("factorial_gate", "gate.json"),
    ("factorial_statistics", "factorial_statistics.json"),
    ("localization_summary", "localization_summary.csv"),
    ("localization_per_bank", "localization_per_bank.csv"),
)
_EVALUATION_ARTIFACT_NAMES = (
    ("evaluation_gate", "evaluation/gate.json"),
    ("evaluation_manifest", "evaluation/manifest.json"),
)
_MASTER_SLICE_KEY = "city × k × method"
_SOTA_COMPARISON_METHODS = (
    ("full", "factorial_arm"),
    ("alignment", "factorial_arm"),
    ("response", "factorial_arm"),
    ("endpoint", "factorial_arm"),
    ("Wi-GATr", "c1_external"),
    ("PMNet", "c1_external"),
)
_DESCRIPTIVE_ONLY_METHODS = (
    ("representation", "representation_baseline"),
    ("resource_control", "resource_control"),
)
_MASTER_COMPARISON_METHODS = _SOTA_COMPARISON_METHODS + _DESCRIPTIVE_ONLY_METHODS
_SOTA_METHOD_NAMES = frozenset(name for name, _family in _SOTA_COMPARISON_METHODS)
_STAGE_FAILURES_NAME = "stage_failures.json"
MIGRATED_LEGACY_FACTORIAL_NAME = "MIGRATED_LEGACY_FACTORIAL.json"

CLAIM_DEPENDENCIES = {
    "C1": ("external_baselines",),
    "C2": ("scene_id_mechanism",),
    "C3": ("G1_G2", "G3_C3"),
    "C4": ("G1_G2", "G3_C3", "shuffled_pair"),
    "C5": ("G1_G2", "G3_C5"),
    "C6": ("G1_G2", "G3_C5", "retention"),
    "C7": ("G1_G2", "G3", "G4", "G5"),
    "C8": ("G1_G2", "G3", "G4", "G5"),
    "C9": ("G1_G2", "G3", "G6"),
    "C10": ("G1_G2", "G3", "G7"),
    "C11": ("rt_calibration",),
    "C12": ("G8",),
    "C13": ("G0",),
}


STAGE_SPECS = {
    "G0": ("literature_resources/gate.json", "csi-pairs-v6-literature-resource-gate-v4"),
    "G1_G2": ("qualification/gate.json", QUALIFICATION_SCHEMA),
    "G3": ("evaluation/gate.json", "csi-pairs-v6-evaluation-gate-v3"),
    "G3_C3": ("evaluation/gate.json", "csi-pairs-v6-evaluation-gate-v3"),
    "G3_C5": ("evaluation/gate.json", "csi-pairs-v6-evaluation-gate-v3"),
    "G4": ("controls/gate.json", "csi-pairs-v6-resource-control-gate-v3"),
    "G5": ("factorial/gate.json", "csi-pairs-formal-factorial-gate-v2.1-v6"),
    "G6": ("risk/gate.json", "csi-pairs-v6-risk-gate-v2"),
    "G7": ("path/gate.json", "csi-pairs-v6-path-gate-v3"),
    "G8": ("external_validity/gate.json", "csi-pairs-v6-external-validity-gate-v4"),
    "external_baselines": (
        "external_baselines/gate.json",
        "csi-pairs-v6-external-baseline-gate-v3",
    ),
    "scene_id_mechanism": ("scene_id/gate.json", "csi-pairs-v6-scene-id-gate-v3"),
    "shuffled_pair": (
        "controls/shuffled_pair/gate.json",
        "csi-pairs-v6-shuffled-pair-gate-v3",
    ),
    "retention": (
        "evaluation/retention/gate.json",
        "csi-pairs-v6-retention-gate-v3",
    ),
    "rt_calibration": (
        "qualification/rt_calibration/gate.json",
        "csi-pairs-v6-rt-calibration-gate-v6",
    ),
    "wrong_map": (
        "wrong_map/gate.json",
        "csi-pairs-formal-wrong-map-status-v2.1-v6",
    ),
}


def assemble_claim_evidence(config, dataset, output_root):
    from .formal_upstream import resolve_authenticated_upstream

    root = Path(output_root)
    upstream = resolve_authenticated_upstream(config, dataset, root)
    output_dir = root / "claims"
    output_dir.mkdir(parents=True, exist_ok=True)
    nonclaim_reasons = _nonclaim_reasons(root, dataset)
    evidence = evidence_context(
        config, dataset, _publication_scientific_use(dataset, nonclaim_reasons)
    )
    assessments = {}
    errors = {}
    for name, (relative, schema) in STAGE_SPECS.items():
        if relative == "qualification/gate.json":
            path = upstream.qualification_gate
        elif relative.startswith("qualification/"):
            path = upstream.qualification_path(relative.removeprefix("qualification/"))
        elif relative == "factorial/gate.json":
            path = upstream.factorial_gate
        else:
            path = root / relative
        status, error = _assess_stage(
            path,
            schema,
            name,
            config,
            dataset,
            output_root=root,
            authenticated_legacy=bool(
                upstream.migrated
                and (
                    relative.startswith("qualification/")
                    or relative.startswith("factorial/")
                )
            ),
        )
        assessments[name] = status
        if error is not None:
            errors[name] = error

    gates = {gate_id: "NOT_ASSESSED" for gate_id in GATE_IDS}
    gates["G0"] = _gate_state(assessments["G0"])
    qualification = assessments["G1_G2"]
    gates["G1"] = _gate_state(qualification)
    gates["G2"] = _gate_state(qualification)
    for gate_id in ("G3", "G4", "G5", "G6", "G7", "G8"):
        gates[gate_id] = _gate_state(assessments[gate_id])

    claims = blocked_claim_vector()
    for claim_id in CLAIM_IDS:
        statuses = [
            assessments[value] if value in assessments else "NOT_ASSESSED"
            for value in CLAIM_DEPENDENCIES[claim_id]
        ]
        claims[claim_id] = _claim_state(claim_id, statuses, dataset.is_fixture)
    if nonclaim_reasons:
        claims = {claim_id: NON_CLAIM for claim_id in CLAIM_IDS}

    required_cities, required_budgets, min_cities = _sota_required_grid(
        config, dataset
    )
    descriptive = _assemble_descriptive_results(
        root=root,
        upstream=upstream,
        gates=gates,
        fixture=dataset.is_fixture or bool(nonclaim_reasons),
        required_cities=required_cities,
        required_budgets=required_budgets,
        min_cities=min_cities,
    )
    write_csv(
        output_dir / "comparison_master_table.csv",
        descriptive["comparison_master_table"]["rows"],
    )
    descriptive["artifacts"]["comparison_master_table"] = _describe_existing_artifact(
        output_dir / "comparison_master_table.csv"
    )
    stage_failures = _read_stage_failures(root)
    package_status = _claim_package_status(
        gates,
        errors,
        assessments=assessments,
        nonclaim=bool(nonclaim_reasons),
        stage_failures=stage_failures,
    )
    master = descriptive["comparison_master_table"]
    sota_report = _sota_readiness_report(
        package_status=package_status,
        nonclaim=bool(nonclaim_reasons),
        master=master,
        fixture=dataset.is_fixture,
        required_cities=required_cities,
        required_budgets=required_budgets,
        min_cities=min_cities,
        legacy_factorial_reused=_legacy_factorial_reused(root)
        or bool(upstream.migrated),
    )
    primary_table_complete = bool(sota_report["primary_table_complete"])
    sota_ready = bool(sota_report["sota_ready"])
    result = {
        "schema_version": "csi-pairs-v6-claim-evidence-v2",
        "status": package_status,
        **evidence,
        "gate_vector": gates,
        "claim_vector": claims,
        "stage_assessments": assessments,
        "evidence_errors": errors,
        "claim_dependencies": {key: list(value) for key, value in CLAIM_DEPENDENCIES.items()},
        "descriptive_results": descriptive,
        "primary_table_complete": primary_table_complete,
        "sota_ready": sota_ready,
        "sota_publication_status": "SOTA_READY" if sota_ready else "SOTA_BLOCKED",
        "sota_block_reasons": list(sota_report["block_reasons"]),
        "sota_grid_status": sota_report["sota_grid_status"],
        "legacy_factorial_reused": bool(sota_report["legacy_factorial_reused"]),
        "success_claim_upgrade": "NOT_PERMITTED",
        "nonclaim_reasons": list(nonclaim_reasons),
        "assemble_claims_exit_code": None,
        "cli_exit_note": _CLI_ASSEMBLE_CLAIMS_EXIT_NOTE,
        "rule": (
            "Only exact-schema, stage-manifested, semantically complete evidence can become PASS. "
            "Scientific FAIL is FAILED and remains descriptively publishable. "
            "INVALID/BLOCKED are reserved for missing or corrupt evidence. "
            "DESCRIPTIVE_PUBLISHABLE tables may be shown; success claims may not be upgraded. "
            "Missing, malformed, BLOCKED, or NOT_ASSESSED evidence cannot support a claim. "
            "COMPLETE means required gates were assessed; it is not a SOTA draft. "
            "SOTA publication requires sota_ready: a complete unique city × k grid "
            "for four arms plus Wi-GATr/PMNet, and Full strictly first on every cell. "
            "Restricted/representation/resource-control rows are descriptive only."
        ),
    }
    result["assemble_claims_exit_code"] = assemble_claims_exit_code(result)
    write_json(output_dir / "claim_evidence.json", result)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return result


def _nonclaim_reasons(root, dataset) -> list[str]:
    del dataset
    reasons = []
    marker = Path(root) / ENGINEERING_UNAPPROVED_NAME
    if marker.is_file():
        reasons.append("ENGINEERING_UNAPPROVED")
    return reasons


def _legacy_factorial_reused(root) -> bool:
    return (Path(root) / MIGRATED_LEGACY_FACTORIAL_NAME).is_file()


def _read_stage_failures(root) -> list[dict]:
    path = Path(root) / _STAGE_FAILURES_NAME
    if not path.is_file():
        return []
    try:
        payload = read_strict_json(path)
    except Exception:
        return [{"stage": _STAGE_FAILURES_NAME, "error": "unreadable"}]
    if not isinstance(payload, dict):
        return [{"stage": _STAGE_FAILURES_NAME, "error": "invalid"}]
    failures = payload.get("failures")
    if not isinstance(failures, list):
        return []
    return [row for row in failures if isinstance(row, dict)]


def _claim_package_status(
    gates, errors, *, assessments=None, nonclaim=False, stage_failures=None
):
    if nonclaim:
        return NON_CLAIM
    if stage_failures:
        return "INCOMPLETE_FAIL_CLOSED"
    if isinstance(assessments, dict) and any(
        assessments.get(name) == _INCOMPLETE_ASSESSMENT
        for name in assessments
        if name not in OPTIONAL_CLAIM_STAGES
    ):
        return "INCOMPLETE_FAIL_CLOSED"
    required_assessed = all(
        value in {"PASS", "FAIL"}
        for gate_id, value in gates.items()
        if gate_id not in OPTIONAL_CLAIM_GATES
    )
    blocking_errors = {
        name: error
        for name, error in errors.items()
        if name not in OPTIONAL_CLAIM_STAGES
    }
    return "COMPLETE" if required_assessed and not blocking_errors else "INCOMPLETE_FAIL_CLOSED"


def _publication_scientific_use(dataset, nonclaim_reasons=None):
    if getattr(dataset, "is_fixture", False) or nonclaim_reasons:
        return "FORBIDDEN"
    return DESCRIPTIVE_PUBLISHABLE


def assemble_claims_exit_code(result):
    """Exit 0 when the descriptive report was written, including honest FAIL.

    Non-zero only for missing/corrupt required inputs or a missing report.
    Scientific gate FAIL is not a process failure. CLI should call this
    helper; formal_cli._stage_exit_code uses this for assemble-claims/all.
    """
    if not isinstance(result, dict):
        return 1
    if not isinstance(result.get("descriptive_results"), dict):
        return 1
    errors = result.get("evidence_errors")
    if not isinstance(errors, dict):
        return 1
    blocking_errors = {
        name: error
        for name, error in errors.items()
        if name not in OPTIONAL_CLAIM_STAGES
    }
    if blocking_errors or result.get("status") in {
        "INCOMPLETE_FAIL_CLOSED",
        NON_CLAIM,
    }:
        return 1
    return 0


def _claim_state(_claim_id, statuses, fixture):
    # Missing/corrupt (hash, schema, exception) stays INVALID/BLOCKED.
    # Scientific FAIL is FAILED so C7/C8 numbers remain exportable.
    if any(value in _CORRUPT_STAGE_STATUSES for value in statuses):
        return "INVALID"
    if any(value in _SCIENTIFIC_FAIL_STATUSES for value in statuses):
        return "FAILED"
    if not all(value == "PASS" for value in statuses):
        return "BLOCKED"
    if fixture:
        return "SOFTWARE_ONLY"
    return "SUPPORTED"


def _assemble_descriptive_results(
    *,
    root,
    upstream,
    gates,
    fixture,
    required_cities=None,
    required_budgets=None,
    min_cities=2,
):
    artifacts = {}
    for name, relative in _DESCRIPTIVE_ARTIFACT_NAMES:
        artifacts[name] = _describe_existing_artifact(upstream.factorial_path(relative))
    for name, relative in _EVALUATION_ARTIFACT_NAMES:
        artifacts[name] = _describe_existing_artifact(root / relative)

    loc_artifact = artifacts["localization_summary"]
    localization = _read_localization_cells(loc_artifact)
    extra_rows = _collect_nonfactorial_master_rows(root)
    statistics = _read_optional_json(artifacts["factorial_statistics"])
    factorial_gate = _read_optional_json(artifacts["factorial_gate"])
    evaluation_gate = _read_optional_json(artifacts["evaluation_gate"])
    return {
        "claim_scope": (
            "descriptive localization/factorial/evaluation comparison only; "
            "this package cannot upgrade a success claim"
        ),
        "scientific_use": "FORBIDDEN" if fixture else DESCRIPTIVE_PUBLISHABLE,
        "success_claim_upgrade": "NOT_PERMITTED",
        "artifacts": artifacts,
        "localization_cells": localization,
        "confidence_intervals": _extract_descriptive_intervals(
            statistics, factorial_gate, evaluation_gate
        ),
        "g5_four_cells": _extract_g5_four_cells(factorial_gate),
        "comparison_master_table": _assemble_comparison_master_table(
            localization,
            extra_rows=extra_rows,
            factorial_source_path=loc_artifact.get("path"),
            factorial_source_sha256=loc_artifact.get("sha256"),
            required_cities=required_cities,
            required_budgets=required_budgets,
            min_cities=min_cities,
        ),
        "gate_vector": dict(gates),
    }


def _primary_table_complete(master) -> bool:
    if not isinstance(master, dict):
        return False
    return master.get("sota_grid_status") == "PRESENT"


def _sota_ready(
    *,
    package_status,
    nonclaim,
    master,
    fixture=False,
    required_cities=None,
    required_budgets=None,
    min_cities=2,
    legacy_factorial_reused=False,
) -> bool:
    return bool(
        _sota_readiness_report(
            package_status=package_status,
            nonclaim=nonclaim,
            master=master,
            fixture=fixture,
            required_cities=required_cities,
            required_budgets=required_budgets,
            min_cities=min_cities,
            legacy_factorial_reused=legacy_factorial_reused,
        )["sota_ready"]
    )


def _sota_required_grid(config, dataset):
    min_cities = 2
    required_budgets = None
    required_cities = None
    if isinstance(config, dict):
        try:
            min_cities = int(config["data"]["minimum_target_cities"])
        except (KeyError, TypeError, ValueError):
            min_cities = 2
        try:
            required_budgets = tuple(
                int(value) for value in config["localization"]["label_budgets"]
            )
        except (KeyError, TypeError, ValueError):
            required_budgets = None
    city_ids = getattr(dataset, "city_ids", None)
    roles = getattr(dataset, "scene_roles", None)
    if city_ids is not None and roles is not None:
        try:
            required_cities = tuple(
                sorted(
                    {
                        str(city)
                        for city, role in zip(city_ids.tolist(), roles.tolist())
                        if str(role) == "target"
                    }
                )
            )
        except Exception:
            required_cities = None
    return required_cities, required_budgets, min_cities


def _grid_city(value):
    if value in {None, ""}:
        return None
    return str(value)


def _grid_budget(value):
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_meter(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number != float("inf") and number != float("-inf")


def _sota_readiness_report(
    *,
    package_status,
    nonclaim,
    master,
    fixture=False,
    required_cities=None,
    required_budgets=None,
    min_cities=2,
    legacy_factorial_reused=False,
) -> dict:
    reasons = []
    if nonclaim:
        reasons.append("nonclaim_or_engineering_marker")
    if fixture:
        reasons.append("fixture")
    if package_status != "COMPLETE":
        reasons.append(f"package_status={package_status}")
    if legacy_factorial_reused:
        reasons.append("migrated_legacy_factorial_reused")
    rows = list(master.get("rows") or []) if isinstance(master, dict) else []
    keyed = {}
    duplicates = []
    inferred_cities = set()
    inferred_budgets = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        method = row.get("method")
        family = row.get("family")
        if method not in _SOTA_METHOD_NAMES or family not in {
            "factorial_arm",
            "c1_external",
        }:
            continue
        city = _grid_city(row.get("city"))
        budget = _grid_budget(row.get("k"))
        if family == "factorial_arm" and row.get("status") == "PRESENT":
            if city is not None:
                inferred_cities.add(city)
            if budget is not None:
                inferred_budgets.add(budget)
        if (
            row.get("status") != "PRESENT"
            or row.get("join_ready") is not True
            or city is None
            or budget is None
            or not _finite_meter(row.get("median_error_m"))
        ):
            continue
        key = (city, budget, method)
        if key in keyed:
            duplicates.append(key)
            continue
        keyed[key] = float(row["median_error_m"])
    if duplicates:
        reasons.append("duplicate_city_k_method_keys")
    if required_cities:
        cities = {_grid_city(city) for city in required_cities}
        cities.discard(None)
    else:
        cities = inferred_cities
    if required_budgets:
        budgets = set()
        for budget in required_budgets:
            normalized = _grid_budget(budget)
            if normalized is not None:
                budgets.add(normalized)
    else:
        budgets = inferred_budgets
    if len(cities) < int(min_cities):
        reasons.append("insufficient_target_cities")
    if not cities or not budgets:
        reasons.append("missing_factorial_city_k_grid")
    expected = {
        (city, budget, method)
        for city in cities
        for budget in budgets
        for method in _SOTA_METHOD_NAMES
    }
    missing = sorted(
        expected - set(keyed),
        key=lambda item: (str(item[0]), int(item[1]) if item[1] is not None else -1, str(item[2])),
    )
    if missing:
        reasons.append(f"incomplete_sota_grid:{len(missing)}")
    rank_failures = []
    for city in cities:
        for budget in budgets:
            full_key = (city, budget, "full")
            if full_key not in keyed:
                continue
            full_error = keyed[full_key]
            for method in _SOTA_METHOD_NAMES:
                if method == "full":
                    continue
                other = keyed.get((city, budget, method))
                if other is None or not (full_error < other):
                    rank_failures.append((city, budget, method))
    if rank_failures:
        reasons.append("full_not_strictly_first_on_every_cell")
    grid_complete = (
        not missing
        and not duplicates
        and bool(cities)
        and bool(budgets)
        and len(cities) >= int(min_cities)
    )
    sota_ready = not reasons and grid_complete
    return {
        "sota_ready": sota_ready,
        "primary_table_complete": grid_complete,
        "sota_grid_status": "PRESENT" if grid_complete else "PARTIAL" if keyed else "NOT_ASSESSED",
        "block_reasons": reasons,
        "legacy_factorial_reused": bool(legacy_factorial_reused),
        "missing_cells": missing[:32],
    }


def _master_row(
    *,
    city,
    k,
    method,
    family,
    median_error_m,
    status,
    join_ready,
    missing_join_fields,
    source,
    source_sha256=None,
):
    return {
        "slice_key": _MASTER_SLICE_KEY,
        "city": city,
        "k": k,
        "method": method,
        "family": family,
        "median_error_m": median_error_m,
        "status": status,
        "join_ready": join_ready,
        "missing_join_fields": list(missing_join_fields),
        "source": source,
        "source_sha256": source_sha256,
    }


def _collect_nonfactorial_master_rows(root) -> list[dict]:
    root = Path(root)
    rows = []
    rows.extend(
        _rows_from_external_csv(root / "external_baselines" / "six_condition_results.csv")
    )
    rows.extend(
        _rows_from_representation_csv(
            root / "representation_baselines" / "localization_across_seed_summary.csv"
        )
    )
    rows.extend(
        _rows_from_resource_csv(root / "controls" / "localization_per_bank.csv")
    )
    return rows


def _rows_from_external_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    digest = sha256_file(path)
    grouped = {}
    methods = set()
    for row in _read_csv_rows(path):
        if str(row.get("condition") or "") != "correct":
            continue
        method = row.get("model_name")
        if method in {None, ""}:
            continue
        methods.add(method)
        city = row.get("city_id")
        budget = row.get("budget", row.get("k"))
        error = row.get("localization_error_m")
        if city in {None, ""} or budget in {None, ""} or not _finite_meter(error):
            continue
        grouped.setdefault((city, budget, method), []).append(float(error))
    rows = []
    present = set()
    for key, values in grouped.items():
        city, budget, method = key
        present.add(method)
        rows.append(
            _master_row(
                city=city,
                k=budget,
                method=method,
                family="c1_external",
                median_error_m=float(sum(values) / len(values)),
                status="PRESENT",
                join_ready=True,
                missing_join_fields=[],
                source=str(path),
                source_sha256=digest,
            )
        )
    for method in methods - present:
        rows.append(
            _master_row(
                city=None,
                k=None,
                method=method,
                family="c1_external",
                median_error_m=None,
                status="NOT_ASSESSED",
                join_ready=False,
                missing_join_fields=["k"],
                source=str(path),
                source_sha256=digest,
            )
        )
    return rows


def _rows_from_representation_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    digest = sha256_file(path)
    rows = []
    for row in _read_csv_rows(path):
        if str(row.get("split_role") or "") != "target":
            continue
        method = row.get("model_name") or row.get("paper_label")
        city = row.get("city_id")
        budget = row.get("budget")
        error = row.get("cluster_macro_across_seed_mean_error_m") or row.get(
            "median_error_m"
        )
        if method in {None, ""}:
            continue
        join_ready = (
            city not in {None, ""}
            and budget not in {None, ""}
            and _finite_meter(error)
        )
        rows.append(
            _master_row(
                city=city if join_ready else None,
                k=budget if join_ready else None,
                method=method,
                family="representation_baseline",
                median_error_m=float(error) if join_ready else None,
                status="PRESENT" if join_ready else "NOT_ASSESSED",
                join_ready=join_ready,
                missing_join_fields=[] if join_ready else ["city", "k"],
                source=str(path),
                source_sha256=digest,
            )
        )
    return rows


def _rows_from_resource_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    digest = sha256_file(path)
    grouped = {}
    for row in _read_csv_rows(path):
        method = row.get("arm")
        city = row.get("city_id")
        budget = row.get("budget")
        if method in {None, ""} or city in {None, ""} or budget in {None, ""}:
            continue
        error = row.get("median_error_m")
        if not _finite_meter(error) and _finite_meter(row.get("utility_neg_log_median")):
            error = None
        if not _finite_meter(error):
            continue
        grouped.setdefault((city, budget, method), []).append(float(error))
    return [
        _master_row(
            city=city,
            k=budget,
            method=method,
            family="resource_control",
            median_error_m=float(sum(values) / len(values)),
            status="PRESENT",
            join_ready=True,
            missing_join_fields=[],
            source=str(path),
            source_sha256=digest,
        )
        for (city, budget, method), values in grouped.items()
    ]


def _read_csv_rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [_coerce_descriptive_row(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _assemble_comparison_master_table(
    localization,
    extra_rows=None,
    *,
    factorial_source_path=None,
    factorial_source_sha256=None,
    required_cities=None,
    required_budgets=None,
    min_cities=2,
):
    rows = []
    present_methods = set()
    source = factorial_source_path or "factorial/localization_summary.csv"
    if localization.get("status") == "PRESENT":
        for cell in localization.get("rows") or []:
            if not isinstance(cell, dict):
                continue
            method = cell.get("arm")
            if method in {None, ""}:
                continue
            present_methods.add(method)
            rows.append(
                _master_row(
                    city=cell.get("city_id"),
                    k=cell.get("budget"),
                    method=method,
                    family="factorial_arm",
                    median_error_m=cell.get("mean_bank_median_error_m"),
                    status="PRESENT",
                    join_ready=True,
                    missing_join_fields=[],
                    source=source,
                    source_sha256=factorial_source_sha256,
                )
            )
    for extra in extra_rows or []:
        method = extra.get("method")
        if method in {None, ""}:
            continue
        if extra.get("status") == "PRESENT":
            present_methods.add(method)
        rows.append(extra)
    for method, family in _MASTER_COMPARISON_METHODS:
        if method in present_methods:
            continue
        missing_fields = [] if family == "factorial_arm" else ["city", "k"]
        rows.append(
            _master_row(
                city=None,
                k=None,
                method=method,
                family=family,
                median_error_m=None,
                status="NOT_ASSESSED",
                join_ready=False,
                missing_join_fields=missing_fields,
                source=None,
            )
        )
    present = any(row["status"] == "PRESENT" for row in rows)
    missing = any(row["status"] == "NOT_ASSESSED" for row in rows)
    if present and missing:
        status = "PARTIAL"
    elif present:
        status = "PRESENT"
    else:
        status = "NOT_ASSESSED"
    grid = _sota_readiness_report(
        package_status="COMPLETE",
        nonclaim=False,
        master={"rows": rows},
        fixture=False,
        required_cities=required_cities,
        required_budgets=required_budgets,
        min_cities=min_cities,
    )
    return {
        "status": status,
        "sota_grid_status": grid["sota_grid_status"],
        "slice_key": _MASTER_SLICE_KEY,
        "sota_methods": [name for name, _family in _SOTA_COMPARISON_METHODS],
        "descriptive_only_methods": [name for name, _family in _DESCRIPTIVE_ONLY_METHODS],
        "join_rule": (
            "SOTA slice is city × k × method for four arms plus Wi-GATr/PMNet. "
            "Restricted/representation/resource-control rows are descriptive and "
            "never invent meters. External rows join only when they expose k."
        ),
        "success_claim_upgrade": "NOT_PERMITTED",
        "rows": rows,
    }


def _describe_existing_artifact(path):
    candidate = Path(path)
    record = {
        "path": str(candidate),
        "status": "NOT_ASSESSED",
        "sha256": None,
    }
    if not candidate.is_file():
        return record
    record["status"] = "PRESENT"
    record["sha256"] = sha256_file(candidate)
    return record


def _read_optional_json(artifact):
    if artifact.get("status") != "PRESENT":
        return {"status": "NOT_ASSESSED", "payload": None, "error": None}
    try:
        payload = read_strict_json(artifact["path"])
    except Exception as error:
        return {
            "status": "INVALID",
            "payload": None,
            "error": f"{type(error).__name__}: {error}",
        }
    if not isinstance(payload, dict):
        return {"status": "INVALID", "payload": None, "error": "payload is not an object"}
    return {"status": "PRESENT", "payload": payload, "error": None}


def _read_localization_cells(artifact):
    if artifact.get("status") != "PRESENT":
        return {"status": "NOT_ASSESSED", "row_count": 0, "rows": [], "error": None}
    try:
        with Path(artifact["path"]).open(newline="", encoding="utf-8") as handle:
            rows = [_coerce_descriptive_row(row) for row in csv.DictReader(handle)]
    except Exception as error:
        return {
            "status": "INVALID",
            "row_count": 0,
            "rows": [],
            "error": f"{type(error).__name__}: {error}",
        }
    return {"status": "PRESENT", "row_count": len(rows), "rows": rows, "error": None}


def _coerce_descriptive_row(row):
    coerced = {}
    for key, value in row.items():
        if value is None or value == "":
            coerced[key] = value
            continue
        coerced[key] = _maybe_number(value)
    return coerced


def _maybe_number(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped in {"", "NOT_ASSESSED"}:
        return stripped
    try:
        if any(character in stripped for character in ".eE"):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return value


def _extract_g5_four_cells(factorial_gate):
    if factorial_gate.get("status") != "PRESENT":
        return {
            "status": factorial_gate.get("status", "NOT_ASSESSED"),
            "subgates": None,
            "localization_city_budget_checks": [],
            "error": factorial_gate.get("error"),
        }
    payload = factorial_gate.get("payload") or {}
    subgates = payload.get("g5_subgates")
    checks = payload.get("localization_city_budget_checks")
    return {
        "status": "PRESENT" if isinstance(subgates, dict) else "NOT_ASSESSED",
        "subgates": dict(subgates) if isinstance(subgates, dict) else None,
        "localization_city_budget_checks": (
            list(checks) if isinstance(checks, list) else []
        ),
        "error": None,
    }


def _extract_descriptive_intervals(statistics, factorial_gate, evaluation_gate):
    intervals = {
        "status": "NOT_ASSESSED",
        "factorial_statistics": None,
        "evaluation_intervals": None,
        "error": None,
    }
    present = False
    if statistics.get("status") == "PRESENT":
        payload = statistics.get("payload") or {}
        intervals["factorial_statistics"] = {
            "exact": payload.get("exact") if "exact" in payload else None,
            "hierarchical_bootstrap": (
                payload.get("hierarchical_bootstrap")
                if "hierarchical_bootstrap" in payload
                else None
            ),
            "bank_only_bootstrap": (
                payload.get("bank_only_bootstrap")
                if "bank_only_bootstrap" in payload
                else None
            ),
        }
        present = True
    elif statistics.get("status") == "INVALID":
        intervals["error"] = statistics.get("error")
    if factorial_gate.get("status") == "PRESENT":
        payload = factorial_gate.get("payload") or {}
        if "localization_city_budget_checks" in payload:
            intervals.setdefault("factorial_statistics", {})
            if intervals["factorial_statistics"] is None:
                intervals["factorial_statistics"] = {}
            intervals["factorial_statistics"]["localization_city_budget_checks"] = (
                payload.get("localization_city_budget_checks")
            )
            present = True
    if evaluation_gate.get("status") == "PRESENT":
        payload = evaluation_gate.get("payload") or {}
        extracted = {}
        for key in ("g3_intervals", "g3_scope_intervals", "g4_intervals"):
            if key in payload:
                extracted[key] = payload.get(key)
        if extracted:
            intervals["evaluation_intervals"] = extracted
            present = True
    elif evaluation_gate.get("status") == "INVALID" and intervals["error"] is None:
        intervals["error"] = evaluation_gate.get("error")
    if present:
        intervals["status"] = "PRESENT"
    elif statistics.get("status") == "INVALID" or evaluation_gate.get("status") == "INVALID":
        intervals["status"] = "INVALID"
    return intervals


def _assess_stage(
    path,
    schema,
    name,
    config,
    dataset,
    *,
    output_root=None,
    authenticated_legacy=False,
):
    if not path.is_file():
        return "NOT_ASSESSED", None
    try:
        payload = read_strict_json(path)
        if not authenticated_legacy:
            require_stage_manifested_gate(
                path,
                payload,
                config,
                dataset,
                schema_version=schema,
            )
        elif payload.get("schema_version") != schema:
            raise RuntimeError("authenticated legacy stage schema mismatch")
        if name == "external_baselines":
            _validate_external_manifest_binding(path, payload, dataset, config)
        if name == "G6":
            _validate_risk_mixture_binding(
                path, payload, config, dataset, output_root=output_root
            )
        if name in {
            "G0", "G4", "G8", "scene_id_mechanism", "rt_calibration",
            "shuffled_pair", "retention",
        }:
            _validate_stage_bound_input(path, payload, config, name, dataset)
        if name == "shuffled_pair":
            _validate_shuffled_evaluation_binding(path, payload, config, dataset)
        if name in {"G3", "G3_C3", "G3_C5", "G4", "G5"}:
            _validate_critical_chain_binding(
                path, payload, name, config, dataset, output_root=output_root
            )
        return _semantic_status(name, payload), None
    except Exception as error:
        return "INVALID", f"{type(error).__name__}: {error}"


def _payload_is_incomplete(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get("status") in _INCOMPLETE_PAYLOAD_STATUSES:
        return True
    if payload.get("engineering_complete") is False:
        return True
    return False


def _semantic_status(name, payload):
    if _payload_is_incomplete(payload):
        return _INCOMPLETE_ASSESSMENT
    if name == "G0":
        decision = payload.get("decision")
        if (
            not isinstance(decision, dict)
            or set(decision) != {
                "no_direct_overlap", "rt_path_ready", "map_path_ready",
                "external_validity_path_ready", "novelty_scope",
            }
            or any(
                decision.get(key) is not True
                for key in (
                    "no_direct_overlap", "rt_path_ready", "map_path_ready",
                    "external_validity_path_ready",
                )
            )
            or not isinstance(decision.get("novelty_scope"), str)
            or not decision["novelty_scope"].strip()
            or not _lower_sha256(payload.get("input_manifest_sha256"))
            or payload.get("search_receipts_verified") is not True
            or int(payload.get("query_count", 0)) < 1
            or not isinstance(payload.get("databases"), list)
            or int(payload.get("search_receipt_count", 0))
            != int(payload.get("query_count", 0)) * len(payload.get("databases", []))
            or payload.get("c13_requires_llm_judge") is not True
        ):
            return "FAIL"
    elif name == "G1_G2":
        from .formal_routing import PRIMARY_ROUTE_CONTRACT

        vector = payload.get("upstream_gates")
        if not isinstance(vector, dict):
            raise RuntimeError("qualification gate lacks upstream_gates")
        return (
            "PASS"
            if vector.get("G1") == vector.get("G2") == "PASS"
            and payload.get("passed") is True
            and payload.get("primary_route_contract") == PRIMARY_ROUTE_CONTRACT
            else "FAIL"
        )
    if name == "G3_C3":
        return "PASS" if payload.get("c3_evidence_complete") is True else "FAIL"
    if name == "G3_C5":
        return "PASS" if payload.get("c5_evidence_complete") is True else "FAIL"
    if name == "G3":
        subgates = payload.get("g3_subgates")
        if not isinstance(subgates, dict) or len(subgates) != 9:
            raise RuntimeError("g3_subgates must contain exactly 9 entries")
        if any(value not in {"PASS", "FAIL"} for value in subgates.values()):
            return _INCOMPLETE_ASSESSMENT
        if (
            payload.get("c3_evidence_complete") is not True
            or payload.get("c5_evidence_complete") is not True
        ):
            return "FAIL"
    elif name == "G4":
        if not _require_pass_subgates(payload, "g4_subgates", 7):
            return "FAIL"
        generous = payload.get("generous_2x_report_only")
        if (
            payload.get("resource_integrity_verified") is not True
            or payload.get("seed_complete_resource_evidence") is not True
            or not _lower_sha256(payload.get("input_manifest_sha256"))
            or not isinstance(generous, dict)
            or generous.get("included_in_g4_subgate_7") is not False
        ):
            return "FAIL"
    elif name == "G5":
        if not _require_pass_subgates(payload, "g5_subgates", 4):
            return "FAIL"
    elif name == "G6":
        if (
            not _require_pass_subgates(payload, "c9_subgates", 4)
            or payload.get("feature_generation")
            != "first-party checkpoint/data/proposal replay"
            or payload.get("risk_score_source")
            != "source-trained frozen unified compatibility probe"
            or payload.get("native_energy_role") != "diagnostic_only"
            or payload.get("proposal_count_contract_verified") is not True
            or payload.get("mixture_freeze_path") != "mixture_freeze.json"
            or not _lower_sha256(payload.get("mixture_freeze_sha256"))
            or payload.get("replay_binding_path") != "replay_binding.json"
            or not _lower_sha256(payload.get("replay_binding_sha256"))
            or not isinstance(payload.get("common_support_interval"), dict)
            or not isinstance(payload.get("outside_support_noninferiority"), dict)
        ):
            return "FAIL"
    elif name == "G7":
        provenance = payload.get("path_provenance")
        localization = payload.get("localization_full_advantage_trend_slopes")
        if (
            not _require_pass_subgates(payload, "g7_subgates", 5)
            or not isinstance(provenance, dict)
            or provenance.get("passed") is not True
            or not _lower_sha256(provenance.get("registry_sha256"))
            or not isinstance(provenance.get("power_coverage_convergence"), dict)
            or provenance["power_coverage_convergence"].get("passed") is not True
            or not isinstance(localization, dict)
            or set(localization) != {"endpoint", "alignment", "response"}
        ):
            return "FAIL"
    elif name == "G8":
        null = payload.get("null_equivalence")
        execution_mode = payload.get("execution_mode")
        if (
            not isinstance(null, dict)
            or null.get("passed") is not True
            or int(null.get("base_map_cluster_count", 0)) < 2
            or int(payload.get("active_direction_cluster_count", 0)) < 2
            or float(payload.get("active_direction_agreement_ci95_low", -1.0))
            < float(payload.get("minimum_active_direction_agreement", 1.0))
            or payload.get("rt_scene_manifest_path") != "rt_scene_manifest.json"
            or not _lower_sha256(payload.get("rt_scene_manifest_sha256"))
            or payload.get("external_csi_path") != "external_csi.npz"
            or not _lower_sha256(payload.get("external_csi_sha256"))
            or payload.get("external_csi_contract")
            != "outer-recomputed-direction-and-effect-from-raw-csi-v1"
            or int(payload.get("external_scene_count", 0)) < 1
        ):
            return "FAIL"
        if execution_mode == "authenticated_sionna_adapter":
            if (
                payload.get("engine_family") != "sionna"
                or not _lower_sha256(payload.get("adapter_source_sha256"))
                or payload.get("external_engine_config_path") is not None
                or payload.get("external_engine_config_sha256") is not None
                or payload.get("external_runtime_provenance_path")
                != "runtime_provenance.json"
                or not _lower_sha256(
                    payload.get("external_runtime_provenance_sha256")
                )
                or not _lower_sha256(
                    payload.get("external_runtime_environment_sha256")
                )
                or not isinstance(payload.get("external_runtime_provenance"), dict)
                or payload["external_runtime_provenance"].get("environment_sha256")
                != payload.get("external_runtime_environment_sha256")
            ):
                return "FAIL"
        elif execution_mode == "authenticated_independent_rt_adapter":
            if (
                not isinstance(payload.get("engine_family"), str)
                or not payload["engine_family"].strip()
                or payload["engine_family"].strip().lower() == "sionna"
                or not _lower_sha256(payload.get("adapter_source_sha256"))
                or payload.get("external_engine_config_path")
                != "external_engine_config.bin"
                or not _lower_sha256(
                    payload.get("external_engine_config_sha256")
                )
                or payload.get("external_runtime_provenance_path")
                != "runtime_provenance.json"
                or not _lower_sha256(
                    payload.get("external_runtime_provenance_sha256")
                )
                or not _lower_sha256(
                    payload.get("external_runtime_environment_sha256")
                )
                or not isinstance(payload.get("external_runtime_provenance"), dict)
                or payload["external_runtime_provenance"].get("environment_sha256")
                != payload.get("external_runtime_environment_sha256")
            ):
                return "FAIL"
        elif execution_mode == "authenticated_precomputed_rt_archive":
            return "FAIL"
        else:
            return "FAIL"
    elif name == "external_baselines":
        eligible = payload.get("c1_eligible_models")
        assessments = payload.get("model_assessments")
        if (
            not isinstance(eligible, list)
            or len(eligible) != len(set(eligible))
            or int(payload.get("c1_eligible_model_count", 0)) != len(eligible)
            or int(payload.get("c1_required_eligible_model_count", 0)) != 2
            or not isinstance(assessments, list)
            or payload.get("c1_city_gate_contract")
            != "all-evaluation-cities-must-pass-v1"
            or payload.get("condition_input_contract")
            != "outer-recomputed-map-and-action-sha256-v1"
            or payload.get("condition_registry_path") != "external_condition_registry.csv"
            or not _lower_sha256(payload.get("condition_registry_sha256"))
            or not _lower_sha256(payload.get("adapter_manifest_sha256"))
            or payload.get("adapter_manifest_path") != "adapter_manifest.json"
        ):
            return "FAIL"
        by_model = {
            row.get("model_name"): row
            for row in assessments
            if isinstance(row, dict) and isinstance(row.get("model_name"), str)
        }
        if set(eligible).difference(by_model):
            return "FAIL"
        for model in eligible:
            row = by_model[model]
            cities = row.get("evaluation_cities")
            city_assessments = row.get("city_assessments")
            if (
                row.get("c1_eligible") is not True
                or row.get("passed") is not True
                or row.get("active_effect_passed") is not True
                or row.get("null_safety_passed") is not True
                or int(row.get("base_map_cluster_count", 0)) < 2
                or float(
                    row.get("null_overclassification_rate_ci95_high", float("inf"))
                )
                > float(row.get("null_overclassification_rate_max", -1.0))
                or row.get("aggregation")
                != "pooled-report-plus-simultaneous-per-city-gate"
                or row.get("all_cities_passed") is not True
                or not isinstance(cities, list)
                or len(cities) != len(set(cities))
                or int(row.get("evaluation_city_count", 0)) != len(cities)
                or not isinstance(city_assessments, dict)
                or set(city_assessments) != set(cities)
            ):
                return "FAIL"
            for city in cities:
                assessment = city_assessments[city]
                if (
                    not isinstance(assessment, dict)
                    or assessment.get("city_id") != city
                    or assessment.get("passed") is not True
                    or assessment.get("active_effect_passed") is not True
                    or assessment.get("null_safety_passed") is not True
                    or int(assessment.get("base_map_cluster_count", 0)) < 2
                    or float(
                        assessment.get(
                            "null_overclassification_rate_ci95_high", float("inf")
                        )
                    )
                    > float(assessment.get("null_overclassification_rate_max", -1.0))
                ):
                    return "FAIL"
        if len(eligible) < 2:
            return (
                "BLOCKED"
                if payload.get("status") == "BLOCKED" and payload.get("passed") is False
                else "FAIL"
            )
        return (
            "PASS"
            if payload.get("status") == "PASS" and payload.get("passed") is True
            else "FAIL"
        )
    elif name in {"shuffled_pair", "retention"}:
        if (
            payload.get("checkpoint_hashes_verified") is not True
            or payload.get("per_unit_rows_verified") is not True
            or not _lower_sha256(payload.get("adapter_source_sha256"))
            or not _lower_sha256(payload.get("input_manifest_sha256"))
        ):
            return "FAIL"
        if name == "shuffled_pair" and (
            payload.get("shortcut_baselines_passed") is not True
            or payload.get("independently_trained_shuffled_checkpoints_verified") is not True
            or payload.get("alignment_and_response_pairing_breaks_verified") is not True
            or payload.get("alignment_pairing_break_passed") is not True
            or payload.get("response_pairing_break_passed") is not True
            or payload.get("evaluation_alignment_shortcut_audit_verified") is not True
            or not _lower_sha256(payload.get("evaluation_gate_sha256"))
            or not _lower_sha256(payload.get("alignment_shortcut_rows_sha256"))
            or int(payload.get("base_map_cluster_count", 0)) < 2
            or not isinstance(payload.get("alignment_gain_interval"), dict)
            or not isinstance(payload.get("shuffled_alignment_gain_interval"), dict)
            or not isinstance(payload.get("shuffled_suppression_interval"), dict)
            or not isinstance(payload.get("response_gain_interval"), dict)
            or not isinstance(payload.get("shuffled_response_gain_interval"), dict)
            or not isinstance(payload.get("shuffled_response_suppression_interval"), dict)
            or not _lower_sha256(payload.get("training_provenance_sha256"))
            or not _lower_sha256(payload.get("shuffled_checkpoint_index_sha256"))
            or payload.get("shortcut_familywise_method") != "Holm"
        ):
            return "FAIL"
        if name == "retention" and (
            payload.get("downstream_f_only_verified") is not True
            or payload.get("disposable_heads_absent_verified") is not True
            or payload.get("source_only_probe_training_verified") is not True
            or not _lower_sha256(payload.get("probe_checkpoint_index_sha256"))
            or not _lower_sha256(payload.get("probe_training_provenance_sha256"))
            or int(payload.get("base_map_cluster_count", 0)) < 2
            or not isinstance(payload.get("effect_intervals"), dict)
            or set(payload["effect_intervals"]) != {
                "cgs_map_swap_effect", "cgs_map_removal_effect", "response_map_swap_effect"
            }
        ):
            return "FAIL"
    elif name == "scene_id_mechanism":
        if (
            payload.get("status") == "BLOCKED"
            and payload.get("engineering_complete") is True
            and isinstance(payload.get("skip_reason"), str)
            and payload["skip_reason"].strip()
        ):
            return "BLOCKED"
        assessments = payload.get("model_assessments")
        if (
            not isinstance(assessments, list)
            or not assessments
            or len(assessments) != int(payload.get("models_assessed", 0))
            or not _lower_sha256(payload.get("input_manifest_sha256"))
        ):
            return "FAIL"
        for row in assessments:
            if (
                not isinstance(row, dict)
                or row.get("passed") is not True
                or int(row.get("base_map_cluster_count", 0)) < 2
                or float(row.get("scene_id_minus_map_ci95_high", float("inf")))
                > float(row.get("scene_id_error_noninferiority_margin_m", -1.0))
                or float(
                    row.get("map_swap_id_swap_direction_cosine_ci95_low", -1.0)
                )
                < float(row.get("minimum_swap_direction_cosine", 1.0))
                or not _lower_sha256(row.get("adapter_source_sha256"))
                or not _lower_sha256(row.get("model_checkpoint_sha256"))
                or not _lower_sha256(row.get("training_provenance_sha256"))
            ):
                return "FAIL"
    elif name == "rt_calibration":
        statistics = payload.get("statistics")
        independence = payload.get("fit_validation_independence")
        independence_fields = {
            "verified",
            "rule",
            "unit_id_overlap",
            "scene_id_overlap",
            "source_asset_sha256_overlap",
            "source_record_identity_overlap",
            "raw_unit_identity_overlap",
            "canonical_payload_sha256_overlap_count",
        }
        if (
            not isinstance(statistics, dict)
            or set(statistics) != {"path_loss", "delay_spread", "angular_spread", "visible_path_count"}
            or any(not isinstance(value, dict) or value.get("passed") is not True for value in statistics.values())
            or not isinstance(independence, dict)
            or set(independence) != independence_fields
            or independence.get("verified") is not True
            or independence.get("rule")
            != "raw_partition_unit_scene_source_and_raw_unit_disjoint_v2"
            or any(
                independence.get(key) != []
                for key in (
                    "unit_id_overlap",
                    "scene_id_overlap",
                    "source_asset_sha256_overlap",
                    "source_record_identity_overlap",
                    "raw_unit_identity_overlap",
                )
            )
            or not isinstance(
                independence.get("canonical_payload_sha256_overlap_count"), int
            )
            or independence["canonical_payload_sha256_overlap_count"] < 0
            or payload.get("aggregation") != "mean_absolute_error_per_unit"
            or int(payload.get("fit_unit_count", 0)) < 1
            or int(payload.get("fit_scene_count", 0)) < 1
            or int(payload.get("validation_unit_count", 0)) < 2
            or int(payload.get("validation_scene_count", 0)) < 1
            or len({
                payload.get("fit_dataset_sha256"),
                payload.get("validation_inputs_sha256"),
                payload.get("validation_reference_sha256"),
            }) != 3
            or payload.get("simulated_statistics_path") != "simulated_statistics.csv"
            or payload.get("validated_statistics_path") != "validated_statistics.csv"
            or any(
                not _lower_sha256(payload.get(key))
                for key in (
                    "protocol_sha256", "input_manifest_sha256", "adapter_source_sha256",
                    "fit_dataset_sha256", "validation_inputs_sha256",
                    "validation_reference_sha256", "fitted_parameters_sha256",
                    "simulated_statistics_sha256", "validated_statistics_sha256",
                )
            )
        ):
            return "FAIL"
    if payload.get("passed") is True and payload.get("status") == "PASS":
        return "PASS"
    if payload.get("status") in _INCOMPLETE_PAYLOAD_STATUSES:
        return _INCOMPLETE_ASSESSMENT
    if payload.get("passed") is False or payload.get("status") in {"FAIL", "BLOCKED"}:
        return "FAIL"
    raise RuntimeError(f"stage {name} has no unambiguous PASS/FAIL result")


def _require_pass_subgates(payload, field, count):
    values = payload.get(field)
    if not isinstance(values, dict) or len(values) != int(count):
        raise RuntimeError(f"{field} must contain exactly {count} entries")
    if any(value not in {"PASS", "FAIL"} for value in values.values()):
        raise RuntimeError(f"{field} contains an unassessed state")
    if any(value != "PASS" for value in values.values()):
        return False
    return True


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_external_manifest_binding(
    gate_path, payload, dataset=None, config=None
):
    from .formal_external import (
        _c1_model_assessment,
        _expected_condition_contract,
        _expected_six_condition_units,
        _validate_execution_manifest,
        _validate_manifest,
        _validate_six_condition_rows,
    )

    manifest_path = gate_path.parent / str(payload.get("adapter_manifest_path", ""))
    if not manifest_path.is_file():
        raise RuntimeError("external-baseline gate has no bound adapter manifest")
    if sha256_file(manifest_path) != payload.get("adapter_manifest_sha256"):
        raise RuntimeError("external-baseline adapter manifest hash mismatch")
    stage_manifest_path = gate_path.parent / "manifest.json"
    stage_manifest = read_strict_json(stage_manifest_path)
    entries = stage_manifest.get("files") if isinstance(stage_manifest, dict) else None
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("path") == manifest_path.name
    ] if isinstance(entries, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != payload.get("adapter_manifest_sha256"):
        raise RuntimeError("external adapter manifest is absent from or mismatched with the stage manifest")
    adapter_manifest = read_strict_json(manifest_path)
    _validate_manifest(adapter_manifest)
    condition_path = gate_path.parent / str(payload.get("condition_registry_path", ""))
    condition_digest = payload.get("condition_registry_sha256")
    if not condition_path.is_file() or sha256_file(condition_path) != condition_digest:
        raise RuntimeError("external condition registry is missing or hash-mismatched")
    condition_matches = [
        row for row in entries
        if isinstance(row, dict) and row.get("path") == condition_path.name
    ] if isinstance(entries, list) else []
    if len(condition_matches) != 1 or condition_matches[0].get("sha256") != condition_digest:
        raise RuntimeError("external condition registry is absent from or mismatched with the stage manifest")
    if dataset is None:
        return
    status_path = gate_path.parent / "adapter_status.csv"
    status_matches = [
        row for row in entries
        if isinstance(row, dict) and row.get("path") == status_path.name
    ] if isinstance(entries, list) else []
    if not status_path.is_file() or len(status_matches) != 1:
        raise RuntimeError("external adapter status is not stage-authenticated")
    if status_matches[0].get("sha256") != sha256_file(status_path):
        raise RuntimeError("external adapter status hash mismatch")
    with status_path.open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))
    passed_ids = [
        row.get("adapter_id") for row in status_rows if row.get("status") == "PASS"
    ]
    if len(passed_ids) != len(set(passed_ids)):
        raise RuntimeError("external adapter status duplicates a passing adapter")
    adapters = {row["adapter_id"]: row for row in adapter_manifest["adapters"]}
    if set(passed_ids).difference(adapters):
        raise RuntimeError("external adapter status names an unknown passing adapter")
    passed_adapters = [
        adapter
        for adapter in adapter_manifest["adapters"]
        if adapter["adapter_id"] in passed_ids
    ]
    passed_models = [adapter["model_name"] for adapter in passed_adapters]
    if len(passed_models) != len(set(passed_models)):
        raise RuntimeError("external adapter status duplicates a passing model")
    for row in status_rows:
        if row.get("status") == "PASS" and row.get("model_name") != adapters[
            str(row.get("adapter_id"))
        ]["model_name"]:
            raise RuntimeError("external adapter status model mismatch")
    claimed_eligible = payload.get("c1_eligible_models")
    if not isinstance(claimed_eligible, list) or any(
        model not in passed_models
        or not next(
            adapter for adapter in passed_adapters if adapter["model_name"] == model
        )["c1_eligible"]
        for model in claimed_eligible
    ):
        raise RuntimeError(
            "C1 gate names a model without a passing raw adapter"
        )
    if not passed_adapters:
        if (
            payload.get("model_assessments") not in ([], None)
            or claimed_eligible != []
            or int(payload.get("c1_eligible_model_count", -1)) != 0
            or payload.get("unique_passing_models") != []
            or int(payload.get("unique_passing_model_count", -1)) != 0
            or int(payload.get("passing_map_conditioned_models", -1)) != 0
        ):
            raise RuntimeError(
                "C1 gate model counts require passing raw adapters"
            )
        return
    if config is None:
        raise RuntimeError("C1 outer recomputation requires the formal config")
    expected_units = _expected_six_condition_units(
        config, dataset, gate_path.parent.parent
    )
    expected_unit_ids = {unit.unit_id for unit in expected_units}
    expected_unit_contract = {unit.unit_id: unit for unit in expected_units}
    expected_condition_contract = _expected_condition_contract(
        dataset, expected_units
    )
    recomputed_assessments = []
    for adapter in passed_adapters:
        adapter_id = adapter["adapter_id"]
        adapter_output = gate_path.parent / "adapters" / str(adapter_id)
        result_path = adapter_output / "six_condition_results.csv"
        if not result_path.is_file():
            raise RuntimeError("passing external adapter has no raw six-condition rows")
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        _validate_six_condition_rows(
            adapter,
            rows,
            dataset,
            expected_unit_ids=expected_unit_ids,
            expected_unit_contract=expected_unit_contract,
            expected_condition_contract=expected_condition_contract,
        )
        _validate_execution_manifest(
            adapter, adapter_output, result_path, dataset
        )
        recomputed_assessments.append(
            _c1_model_assessment(
                config,
                adapter,
                rows,
                dataset,
                expected_unit_contract=expected_unit_contract,
            )
        )
    if payload.get("model_assessments") != recomputed_assessments:
        raise RuntimeError("C1 gate assessments differ from raw adapter results")
    recomputed_eligible = sorted(
        row["model_name"]
        for row in recomputed_assessments
        if row["c1_eligible"] is True and row["passed"] is True
    )
    if (
        claimed_eligible != recomputed_eligible
        or int(payload.get("c1_eligible_model_count", -1))
        != len(recomputed_eligible)
        or payload.get("unique_passing_models") != sorted(passed_models)
        or int(payload.get("unique_passing_model_count", -1))
        != len(passed_models)
        or int(payload.get("passing_map_conditioned_models", -1))
        != len(passed_models)
    ):
        raise RuntimeError("C1 gate model counts differ from raw adapter results")


def _validate_risk_mixture_binding(
    gate_path, payload, config, dataset, *, output_root=None
):
    if payload.get("passed") is not True:
        return
    relative = payload.get("mixture_freeze_path")
    digest = payload.get("mixture_freeze_sha256")
    if relative != "mixture_freeze.json" or not _lower_sha256(digest):
        raise RuntimeError("risk gate has no authenticated mixture freeze")
    path = gate_path.parent / relative
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("risk mixture freeze is missing or hash-mismatched")
    freeze = read_strict_json(path)
    if (
        freeze.get("schema_version") != "csi-pairs-v6-risk-mixture-freeze-v1"
        or freeze.get("frozen_role") != "source_method_selection"
        or not freeze.get("bank_ids")
    ):
        raise RuntimeError("risk mixture freeze contract is invalid")
    replay_relative = payload.get("replay_binding_path")
    replay_digest = payload.get("replay_binding_sha256")
    if replay_relative != "replay_binding.json" or not _lower_sha256(replay_digest):
        raise RuntimeError("risk gate has no authenticated first-party replay binding")
    replay_path = gate_path.parent / replay_relative
    if not replay_path.is_file() or sha256_file(replay_path) != replay_digest:
        raise RuntimeError("risk replay binding is missing or hash-mismatched")
    stage_manifest = read_strict_json(gate_path.parent / "manifest.json")
    entries = stage_manifest.get("files") if isinstance(stage_manifest, dict) else None
    for relative_path, expected_digest in (
        (relative, digest),
        (replay_relative, replay_digest),
    ):
        matches = [
            row
            for row in entries
            if isinstance(row, dict) and row.get("path") == relative_path
        ] if isinstance(entries, list) else []
        if len(matches) != 1 or matches[0].get("sha256") != expected_digest:
            raise RuntimeError(
                "risk binding artifact is absent from or mismatched with the stage manifest"
            )
    binding = read_strict_json(replay_path)
    from .formal_evidence import config_sha256
    from . import formal_risk

    root = gate_path.parent.parent if output_root is None else Path(output_root)
    from .formal_upstream import resolve_authenticated_upstream

    checkpoint_index = resolve_authenticated_upstream(
        config, dataset, root
    ).checkpoint_index
    if set(binding) != {
        "schema_version",
        "dataset_sha256",
        "config_sha256",
        "checkpoint_index_sha256",
        "implementation_source_sha256",
        "payload_sha256",
    }:
        raise RuntimeError("risk replay binding fields are not exact")
    if (
        binding["schema_version"] != "csi-pairs-v6-risk-replay-binding-v1"
        or binding["dataset_sha256"] != sha256_file(dataset.source_path)
        or binding["config_sha256"] != config_sha256(config)
        or not checkpoint_index.is_file()
        or binding["checkpoint_index_sha256"] != sha256_file(checkpoint_index)
        or binding["implementation_source_sha256"]
        != sha256_file(Path(formal_risk.__file__).resolve())
        or not _lower_sha256(binding["payload_sha256"])
    ):
        raise RuntimeError("risk replay binding does not match executed first-party inputs")


def _validate_critical_chain_binding(
    gate_path, payload, stage_name, config=None, dataset=None, *, output_root=None
):
    root = gate_path.parent.parent if output_root is None else Path(output_root)
    if config is None or dataset is None:
        qualification_path = root / "qualification" / "gate.json"
        factorial_path = root / "factorial" / "gate.json"
    else:
        from .formal_upstream import resolve_authenticated_upstream

        upstream = resolve_authenticated_upstream(config, dataset, root)
        qualification_path = upstream.qualification_gate
        factorial_path = upstream.factorial_gate
    evaluation_path = root / "evaluation" / "gate.json"

    def require_digest(field, path, label):
        digest = payload.get(field)
        if not _lower_sha256(digest) or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"{label} is missing or hash-mismatched")

    if stage_name == "G5":
        require_digest(
            "qualification_gate_sha256",
            qualification_path,
            "G5 qualification binding",
        )
    elif stage_name in {"G3", "G3_C3", "G3_C5"}:
        require_digest(
            "qualification_gate_sha256",
            qualification_path,
            "G3 qualification binding",
        )
        require_digest(
            "factorial_gate_sha256",
            factorial_path,
            "G3 factorial binding",
        )
    elif stage_name == "G4":
        require_digest(
            "factorial_gate_sha256",
            factorial_path,
            "G4 factorial binding",
        )
        require_digest(
            "evaluation_gate_sha256",
            evaluation_path,
            "G4 evaluation binding",
        )


def _validate_shuffled_evaluation_binding(gate_path, payload, config, dataset):
    root = gate_path.parent.parent.parent
    evaluation_gate_path = root / "evaluation" / "gate.json"
    shortcut_rows_path = root / "evaluation" / "alignment_shortcut_baselines.csv"
    if (
        not evaluation_gate_path.is_file()
        or sha256_file(evaluation_gate_path) != payload.get("evaluation_gate_sha256")
        or not shortcut_rows_path.is_file()
        or sha256_file(shortcut_rows_path) != payload.get("alignment_shortcut_rows_sha256")
    ):
        raise RuntimeError("shuffled-pair gate does not bind the executed evaluation shortcut audit")
    evaluation_gate = read_strict_json(evaluation_gate_path)
    require_stage_manifested_gate(
        evaluation_gate_path,
        evaluation_gate,
        config,
        dataset,
        schema_version="csi-pairs-v6-evaluation-gate-v3",
    )
    audit = evaluation_gate.get("alignment_shortcut_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise RuntimeError("bound evaluation shortcut audit did not pass")


def _validate_stage_bound_input(
    gate_path, payload, config=None, stage_name=None, dataset=None
):
    relative = payload.get("input_manifest_path")
    digest = payload.get("input_manifest_sha256")
    if not isinstance(relative, str) or Path(relative).name != relative or not _lower_sha256(digest):
        raise RuntimeError("stage gate has an invalid bound input-manifest reference")
    path = gate_path.parent / relative
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("stage input manifest is missing or hash-mismatched")
    stage_manifest = read_strict_json(gate_path.parent / "manifest.json")
    files = stage_manifest.get("files") if isinstance(stage_manifest, dict) else None
    matches = [
        row for row in files
        if isinstance(row, dict) and row.get("path") == relative
    ] if isinstance(files, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != digest:
        raise RuntimeError("bound input manifest is absent from or mismatched with the stage manifest")
    if stage_name is None:
        return
    manifest = read_strict_json(path)
    if stage_name == "G0":
        from .formal_literature import _validate_manifest

        review = _validate_manifest(config, manifest, path.parent, dataset)
        for key, expected in (
            ("llm_judge_path", str(review["path"])),
            ("llm_judge_sha256", manifest["llm_judge_sha256"]),
            ("llm_judge", review["judge"]),
            ("llm_judge_family", review["judge_family"]),
            ("llm_judge_completed_utc", review["completed_utc"]),
            ("llm_judge_signature", review["signature"]),
            ("llm_judge_signed_utc", review["signed_utc"]),
        ):
            if payload.get(key) != expected:
                raise RuntimeError(f"G0 gate {key} differs from its llm-judge review")
    elif stage_name == "G4":
        from .formal_controls import _validate_manifest

        _validate_manifest(manifest, path.parent)
    elif stage_name == "G8":
        from .formal_external_validity import (
            _cluster_direction_interval,
            _execution_mode,
            _expected_external_registry,
            _load_external_csi,
            _load_rt_scene_manifest,
            _paired_bank_equivalence,
            _probe_independent_runtime,
            _rows_from_external_csi,
            _validate_manifest,
            _verify_adapter_source,
            require_claim_eligible_manifest,
            require_independent_primary_engine,
        )

        _validate_manifest(manifest)
        if dataset is None:
            raise RuntimeError("G8 reauthentication requires the formal dataset")
        require_independent_primary_engine(dataset, manifest)
        execution_mode = _execution_mode(manifest)
        require_claim_eligible_manifest(manifest)
        adapter_source = None
        if execution_mode in {
            "authenticated_sionna_adapter",
            "authenticated_independent_rt_adapter",
        }:
            adapter_source = _verify_adapter_source(manifest)
            if (
                payload.get("adapter_source_sha256")
                != manifest.get("adapter_source_sha256")
                or payload.get("adapter_source_path") != str(adapter_source)
            ):
                raise RuntimeError(
                    "G8 gate adapter source differs from its authenticated manifest"
                )
        if payload.get("execution_mode") != execution_mode:
            raise RuntimeError("G8 gate execution mode differs from its input manifest")
        scene_relative = payload.get("rt_scene_manifest_path")
        scene_digest = payload.get("rt_scene_manifest_sha256")
        if scene_relative != "rt_scene_manifest.json" or not _lower_sha256(
            scene_digest
        ):
            raise RuntimeError("G8 gate has no authenticated RT scene manifest")
        scene_path = gate_path.parent / scene_relative
        if not scene_path.is_file() or sha256_file(scene_path) != scene_digest:
            raise RuntimeError("G8 RT scene manifest is missing or hash-mismatched")
        scene_matches = [
            row
            for row in files
            if isinstance(row, dict) and row.get("path") == scene_relative
        ] if isinstance(files, list) else []
        if len(scene_matches) != 1 or scene_matches[0].get("sha256") != scene_digest:
            raise RuntimeError(
                "G8 RT scene manifest is absent from or mismatched with the stage manifest"
            )
        scene_manifest = _load_rt_scene_manifest(
            scene_path,
            dataset,
            (
                manifest
                if execution_mode
                in {
                    "authenticated_independent_rt_adapter",
                    "authenticated_precomputed_rt_archive",
                }
                else None
            ),
        )
        if execution_mode in {
            "authenticated_independent_rt_adapter",
            "authenticated_precomputed_rt_archive",
        }:
            for entry in scene_manifest["worlds"]:
                source_relative = entry["source_asset_path"]
                source_digest = entry["source_asset_sha256"]
                source_matches = [
                    row
                    for row in files
                    if isinstance(row, dict) and row.get("path") == source_relative
                ] if isinstance(files, list) else []
                if (
                    len(source_matches) != 1
                    or source_matches[0].get("sha256") != source_digest
                ):
                    raise RuntimeError(
                        "G8 source asset is absent from or mismatched with the stage manifest"
                    )
        raw_relative = payload.get("external_csi_path")
        raw_digest = payload.get("external_csi_sha256")
        if raw_relative != "external_csi.npz" or not _lower_sha256(raw_digest):
            raise RuntimeError("G8 gate has no authenticated raw external CSI")
        raw_path = gate_path.parent / raw_relative
        raw_matches = [
            row
            for row in files
            if isinstance(row, dict) and row.get("path") == raw_relative
        ] if isinstance(files, list) else []
        if (
            not raw_path.is_file()
            or raw_path.is_symlink()
            or sha256_file(raw_path) != raw_digest
            or len(raw_matches) != 1
            or raw_matches[0].get("sha256") != raw_digest
        ):
            raise RuntimeError("G8 raw external CSI is not stage-authenticated")
        input_hashes_changed = bool(
            manifest.get("rt_scene_manifest_sha256") != scene_digest
            or (
                execution_mode == "authenticated_precomputed_rt_archive"
                and manifest.get("external_csi_sha256") != raw_digest
            )
        )
        if execution_mode in {
            "authenticated_independent_rt_adapter",
            "authenticated_precomputed_rt_archive",
        } and input_hashes_changed:
            raise RuntimeError("G8 input hashes differ from the stage-authenticated inputs")
        if execution_mode in {
            "authenticated_independent_rt_adapter",
            "authenticated_precomputed_rt_archive",
        }:
            config_relative = payload.get("external_engine_config_path")
            config_digest = payload.get("external_engine_config_sha256")
            if (
                config_relative != "external_engine_config.bin"
                or not _lower_sha256(config_digest)
                or manifest.get("engine_config_sha256") != config_digest
                or scene_manifest.get("configuration_sha256") != config_digest
            ):
                raise RuntimeError("G8 has no authenticated engine configuration")
            config_path = gate_path.parent / config_relative
            config_matches = [
                row
                for row in files
                if isinstance(row, dict) and row.get("path") == config_relative
            ] if isinstance(files, list) else []
            if (
                config_path.is_symlink()
                or not config_path.is_file()
                or sha256_file(config_path) != config_digest
                or len(config_matches) != 1
                or config_matches[0].get("sha256") != config_digest
            ):
                raise RuntimeError("G8 engine configuration is not stage-authenticated")
        elif (
            payload.get("external_engine_config_path") is not None
            or payload.get("external_engine_config_sha256") is not None
        ):
            raise RuntimeError("G8 Sionna adapter cannot claim an archive engine configuration")
        external_csi = _load_external_csi(raw_path, dataset)
        registry = _expected_external_registry(
            config, dataset, gate_path.parent.parent, scene_manifest
        )
        rows = _rows_from_external_csi(external_csi, registry)
        active = [row for row in rows if row["route"] == "active"]
        null_rows = [row for row in rows if row["route"] == "null"]
        agreement = _cluster_direction_interval(
            active, int(config["external_validity"]["bootstrap_resamples"])
        )
        equivalence = _paired_bank_equivalence(
            null_rows,
            float(config["external_validity"]["null_equivalence_margin"]),
            int(config["external_validity"]["bootstrap_resamples"]),
        )
        if (
            payload.get("active_direction_agreement") != agreement["estimate"]
            or payload.get("active_direction_agreement_ci95_low") != agreement["ci95_low"]
            or payload.get("active_direction_agreement_ci95_high") != agreement["ci95_high"]
            or payload.get("active_direction_cluster_count") != agreement["base_map_cluster_count"]
            or payload.get("null_equivalence") != equivalence
            or payload.get("external_scene_count") != external_csi.shape[0]
        ):
            raise RuntimeError("G8 gate statistics differ from raw-CSI outer recomputation")
        if execution_mode in {
            "authenticated_sionna_adapter",
            "authenticated_independent_rt_adapter",
        }:
            runtime_relative = payload.get("external_runtime_provenance_path")
            runtime_digest = payload.get("external_runtime_provenance_sha256")
            if runtime_relative != "runtime_provenance.json" or not _lower_sha256(
                runtime_digest
            ):
                raise RuntimeError("G8 gate has no authenticated runtime provenance")
            runtime_path = gate_path.parent / runtime_relative
            runtime_matches = [
                row
                for row in files
                if isinstance(row, dict) and row.get("path") == runtime_relative
            ] if isinstance(files, list) else []
            if (
                runtime_path.is_symlink()
                or not runtime_path.is_file()
                or sha256_file(runtime_path) != runtime_digest
                or len(runtime_matches) != 1
                or runtime_matches[0].get("sha256") != runtime_digest
            ):
                raise RuntimeError("G8 runtime provenance is not stage-authenticated")
            project_root = Path(__file__).resolve().parents[1]
            executable = manifest["command"][0].replace(
                "{project_root}", str(project_root)
            )
            if execution_mode == "authenticated_sionna_adapter":
                from .formal_external_runtime import probe_external_runtime
                from .formal_external_validity import _authenticate_sionna_runtime

                independently_probed = probe_external_runtime(
                    executable,
                    "sionna",
                    project_root,
                    require_execution_ready=True,
                )
                _, runtime_record = _authenticate_sionna_runtime(
                    [executable], gate_path.parent, independently_probed
                )
            else:
                from .formal_external_validity import (
                    _authenticate_independent_runtime,
                )

                independently_probed = _probe_independent_runtime(
                    [executable, str(adapter_source)], config_path
                )
                _, runtime_record = _authenticate_independent_runtime(
                    [executable, str(adapter_source)],
                    gate_path.parent,
                    independently_probed,
                    config_path,
                )
            if (
                payload.get("external_runtime_provenance") != runtime_record
                or payload.get("external_runtime_environment_sha256")
                != runtime_record["environment_sha256"]
            ):
                raise RuntimeError("G8 gate runtime fields differ from the probed interpreter")
        elif any(
            payload.get(key) is not None
            for key in (
                "adapter_source_path",
                "adapter_source_sha256",
                "external_runtime_provenance_path",
                "external_runtime_provenance_sha256",
                "external_runtime_environment_sha256",
                "external_runtime_provenance",
            )
        ):
            raise RuntimeError("G8 precomputed archive cannot claim adapter runtime provenance")
    elif stage_name == "scene_id_mechanism":
        if (
            payload.get("status") == "BLOCKED"
            and payload.get("engineering_complete") is True
            and isinstance(payload.get("skip_reason"), str)
            and payload["skip_reason"].strip()
        ):
            if manifest.get("schema_version") != "csi-pairs-v6-scene-id-skip-v1":
                raise RuntimeError("scene-ID skip receipt schema mismatch")
            return
        from .formal_scene_id import _validate_manifest, _verify_adapter_files

        _validate_manifest(manifest)
        for adapter in manifest["adapters"]:
            _verify_adapter_files(adapter, path.parent)
    elif stage_name == "rt_calibration":
        from .formal_rt_calibration import (
            _bound_input,
            _join_and_assess,
            _read_partition_contract,
            _read_statistics_csv,
            _validate_partition_independence,
            _validate_manifest,
            _validate_protocol,
        )

        _validate_manifest(manifest)
        protocol = _bound_input(
            manifest["protocol_path"], manifest["protocol_sha256"], path.parent, "protocol"
        )
        fit = _bound_input(
            manifest["fit_dataset_path"], manifest["fit_dataset_sha256"], path.parent, "fit dataset"
        )
        validation_inputs = _bound_input(
            manifest["validation_inputs_path"],
            manifest["validation_inputs_sha256"],
            path.parent,
            "validation inputs",
        )
        validation_reference = _bound_input(
            manifest["validation_reference_path"],
            manifest["validation_reference_sha256"],
            path.parent,
            "validation reference",
        )
        _bound_input(
            manifest["adapter_source_path"],
            manifest["adapter_source_sha256"],
            path.parent,
            "adapter source",
        )
        design_record = _bound_input(
            manifest["design_record_path"],
            manifest["design_record_sha256"],
            path.parent,
            "calibration design record",
        )
        license_review = _bound_input(
            manifest["license_review_path"],
            manifest["license_review_sha256"],
            path.parent,
            "license review record",
        )
        from .formal_rt_calibration import _validate_review_record

        _validate_review_record(design_record, "design")
        _validate_review_record(license_review, "license")
        if (
            payload.get("design_record_path") != str(design_record)
            or payload.get("license_review_path") != str(license_review)
        ):
            raise RuntimeError(
                "RT calibration gate review-record paths differ from its bound manifest"
            )
        if len({fit, validation_inputs, validation_reference}) != 3:
            raise RuntimeError("RT calibration fit and validation paths are not distinct")
        for key in (
            "protocol_sha256",
            "fit_dataset_sha256",
            "validation_inputs_sha256",
            "validation_reference_sha256",
            "adapter_source_sha256",
            "design_record_sha256",
            "license_review_sha256",
        ):
            if payload.get(key) != manifest[key]:
                raise RuntimeError(f"RT calibration gate {key} differs from its bound manifest")
        protocol_payload = read_strict_json(protocol)
        _validate_protocol(protocol_payload)
        simulated_name = payload.get("simulated_statistics_path")
        validated_name = payload.get("validated_statistics_path")
        for name, digest in (
            (simulated_name, payload.get("simulated_statistics_sha256")),
            (validated_name, payload.get("validated_statistics_sha256")),
        ):
            if not isinstance(name, str) or Path(name).name != name or not _lower_sha256(digest):
                raise RuntimeError("RT calibration gate has an invalid statistics artifact")
            artifact = gate_path.parent / name
            matches = [
                row for row in files
                if isinstance(row, dict) and row.get("path") == name
            ]
            if (
                not artifact.is_file()
                or sha256_file(artifact) != digest
                or len(matches) != 1
                or matches[0].get("sha256") != digest
            ):
                raise RuntimeError("RT calibration statistics artifact is not stage-authenticated")
        reference_rows = _read_statistics_csv(validation_reference, "validation reference")
        fit_partition = _read_partition_contract(fit, "fit")
        validation_partition = _read_partition_contract(validation_inputs, "validation")
        independence = _validate_partition_independence(
            fit_partition, validation_partition, set(reference_rows)
        )
        simulated_rows = _read_statistics_csv(
            gate_path.parent / simulated_name, "simulated statistics"
        )
        validated_rows, assessments = _join_and_assess(
            reference_rows,
            simulated_rows,
            protocol_payload["absolute_tolerances"],
        )
        if (
            len(validated_rows) != payload.get("validation_unit_count")
            or assessments != payload.get("statistics")
            or independence != payload.get("fit_validation_independence")
            or len(fit_partition) != payload.get("fit_unit_count")
            or len({row["scene_id"] for row in fit_partition.values()})
            != payload.get("fit_scene_count")
            or len({row["scene_id"] for row in validation_partition.values()})
            != payload.get("validation_scene_count")
        ):
            raise RuntimeError("RT calibration gate statistics differ from outer recomputation")
    elif stage_name in {"shuffled_pair", "retention"}:
        expected_schema = {
            "shuffled_pair": "csi-pairs-v6-shuffled-pair-adapter-v3",
            "retention": "csi-pairs-v6-retention-adapter-v3",
        }[stage_name]
        required = {
            "schema_version", "command", "implementation_revision", "control_seed",
            "adapter_source_path", "adapter_source_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise RuntimeError("claim-control bound manifest fields are not exact")
        if manifest["schema_version"] != expected_schema:
            raise RuntimeError("claim-control bound manifest schema mismatch")
        source = path.parent / manifest["adapter_source_path"]
        if (
            Path(manifest["adapter_source_path"]).name != manifest["adapter_source_path"]
            or not source.is_file()
            or sha256_file(source) != manifest["adapter_source_sha256"]
            or manifest["implementation_revision"]
            != manifest["adapter_source_sha256"]
            or manifest["command"][:2] != ["{python}", "{adapter_source}"]
        ):
            raise RuntimeError("claim-control bound adapter source is missing or hash-mismatched")


def _gate_state(status):
    if status == "PASS":
        return "PASS"
    if status == "FAIL":
        return "FAIL"
    if status == _INCOMPLETE_ASSESSMENT:
        return "NOT_ASSESSED"
    if status in {"INVALID", "BLOCKED"}:
        return "BLOCKED"
    return "NOT_ASSESSED"
