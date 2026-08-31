from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from .formal_io import read_strict_json, sha256_file


REPORT_SCHEMA = "csi-pairs-v6-evaluation-equivalence-report-v1"
MAX_DIFFERENCE_EXAMPLES = 100
EVIDENCE_FIELDS = (
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "scientific_use",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
)
FROZEN_IDENTITY_FIELDS = (
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "scientific_use",
    "requirements_lock_sha256",
)
IMPLEMENTATION_IDENTITY_FIELDS = (
    "source_tree_sha256",
    "runtime_provenance_sha256",
    "runtime_provenance",
)
CSV_IMPLEMENTATION_IDENTITY_FIELDS = (
    "source_tree_sha256",
    "runtime_provenance_sha256",
)
GATE_IMPLEMENTATION_EVIDENCE_KEYS = IMPLEMENTATION_IDENTITY_FIELDS
MANIFEST_ARTIFACT_CONTENT_FIELDS = ("bytes", "sha256")


@dataclass(frozen=True)
class CsvContract:
    fields: tuple[str, ...]
    primary_key: tuple[str, ...]
    float_fields: tuple[str, ...] = ()
    structured_fields: tuple[str, ...] = ()


def _csv_contract(
    fields,
    primary_key,
    *,
    float_fields=(),
    structured_fields=(),
):
    return CsvContract(
        fields=tuple(fields) + EVIDENCE_FIELDS,
        primary_key=tuple(primary_key),
        float_fields=tuple(float_fields),
        structured_fields=tuple(structured_fields),
    )


CSV_CONTRACTS = {
    "alignment_shortcut_baselines.csv": _csv_contract(
        (
            "seed", "arm", "bank_id", "base_map_cluster_id",
            "canonical_base_map_digest", "canonical_bank_digest", "city_id",
            "evaluation_scope", "baseline", "input_class", "source_train_auroc",
            "unseen_bank_auroc", "auroc", "identity_token_contract", "probe_family",
            "fit_role", "selection_role", "n",
        ),
        ("seed", "arm", "bank_id", "baseline"),
        float_fields=("source_train_auroc", "unseen_bank_auroc", "auroc"),
    ),
    "cgs_active_effect_bins.csv": _csv_contract(
        (
            "seed", "arm", "bank_id", "city_id", "evaluation_scope", "effect_bin",
            "pair_count", "physical_distance_min", "physical_distance_max", "cgs_auroc",
        ),
        ("seed", "arm", "bank_id", "effect_bin"),
        float_fields=("physical_distance_min", "physical_distance_max", "cgs_auroc"),
    ),
    "cgs_per_bank.csv": _csv_contract(
        (
            "seed", "arm", "bank_id", "base_map_cluster_id",
            "canonical_base_map_digest", "canonical_bank_digest", "city_id",
            "evaluation_scope", "route", "probe_family", "cgs_auroc",
            "native_energy_auroc", "native_audit_hold_auroc", "native_probe_spearman",
            "native_audit_hold_probe_spearman", "n",
        ),
        ("seed", "arm", "bank_id"),
        float_fields=(
            "cgs_auroc", "native_energy_auroc", "native_audit_hold_auroc",
            "native_probe_spearman", "native_audit_hold_probe_spearman",
        ),
    ),
    "compatibility_pair_effects.csv": _csv_contract(
        (
            "seed", "arm", "pair_id", "scene_index", "bank_id",
            "base_map_cluster_id", "canonical_base_map_digest", "canonical_bank_digest",
            "city_id", "source_world", "target_world", "position_index", "route",
            "physical_distance", "matched_minus_alternative",
        ),
        ("seed", "arm", "pair_id"),
        float_fields=("physical_distance", "matched_minus_alternative"),
    ),
    "compatibility_probe_contract.csv": _csv_contract(
        (
            "seed", "arm", "selected_family", "selection_nll", "selection_auroc",
            "candidate_metrics",
        ),
        ("seed", "arm"),
        float_fields=("selection_nll", "selection_auroc"),
        structured_fields=("candidate_metrics",),
    ),
    "compatibility_route_distributions.csv": _csv_contract(
        (
            "seed", "arm", "bank_id", "base_map_cluster_id",
            "canonical_base_map_digest", "canonical_bank_digest", "city_id",
            "evaluation_scope", "route", "condition_status", "pair_count",
            "matched_minus_alternative_mean", "matched_minus_alternative_median",
            "absolute_difference_p90", "overclassification_rate",
        ),
        ("seed", "arm", "bank_id", "route"),
        float_fields=(
            "matched_minus_alternative_mean", "matched_minus_alternative_median",
            "absolute_difference_p90", "overclassification_rate",
        ),
    ),
    "response_pair_effects.csv": _csv_contract(
        (
            "seed", "arm", "pair_id", "scene_index", "bank_id",
            "base_map_cluster_id", "canonical_base_map_digest", "canonical_bank_digest",
            "city_id", "source_world", "target_world", "position_index", "query_index",
            "route", "wrong_action_match_status", "wrong_action_world", "prediction_mse",
            "copy_mse", "action_swap_mse", "no_action_mse", "response_advantage",
            "response_advantage_vs_action_swap", "response_advantage_vs_no_action",
        ),
        ("seed", "arm", "pair_id"),
        float_fields=(
            "prediction_mse", "copy_mse", "action_swap_mse", "no_action_mse",
            "response_advantage", "response_advantage_vs_action_swap",
            "response_advantage_vs_no_action",
        ),
    ),
    "response_per_bank.csv": _csv_contract(
        (
            "seed", "arm", "bank_id", "base_map_cluster_id",
            "canonical_base_map_digest", "canonical_bank_digest", "city_id",
            "evaluation_scope", "unified_response_probe_active_patch_nmse",
            "probe_copy_active_patch_nmse", "unified_response_probe_action_swap_exact_patch_nmse",
            "probe_action_swap_active_patch_nmse", "probe_action_swap_exact_count",
            "probe_action_swap_fallback_count", "probe_action_swap_failed_count",
            "probe_action_swap_exact_fraction", "probe_without_map_active_patch_nmse",
            "probe_edit_only_active_patch_nmse", "probe_csi_only_active_patch_nmse",
            "probe_oracle_x_active_patch_nmse", "native_target_free_full_channel_nmse",
            "native_copy_full_channel_nmse", "native_no_action_full_channel_nmse",
            "native_target_free_action_swap_exact_full_channel_nmse",
            "native_action_swap_full_channel_nmse", "native_latent_nmse",
            "native_latent_copy_nmse", "native_latent_no_action_nmse",
            "native_latent_action_swap_exact_target_nmse", "native_latent_action_swap_nmse",
            "native_action_swap_exact_count", "native_action_swap_fallback_count",
            "native_action_swap_failed_count", "native_action_swap_exact_fraction",
            "native_delta_direction_cosine", "native_delta_relative_magnitude_error",
            "native_sgcs", "native_path_loss_change_mae", "native_delay_spread_change_mae",
            "native_angular_spread_change_mae", "native_path_loss_direction_accuracy",
            "native_delay_spread_direction_accuracy", "native_angular_spread_direction_accuracy",
            "native_transition_skill", "native_null_patch_count", "native_null_delta_rms_mean",
            "native_null_violation_rate", "native_latent_null_delta_rms_mean",
            "native_latent_null_violation_rate", "native_mask_bank", "native_input_contract",
            "n_active_patches",
        ),
        ("seed", "arm", "bank_id"),
        float_fields=(
            "unified_response_probe_active_patch_nmse", "probe_copy_active_patch_nmse",
            "unified_response_probe_action_swap_exact_patch_nmse",
            "probe_action_swap_active_patch_nmse", "probe_action_swap_exact_fraction",
            "probe_without_map_active_patch_nmse", "probe_edit_only_active_patch_nmse",
            "probe_csi_only_active_patch_nmse", "probe_oracle_x_active_patch_nmse",
            "native_target_free_full_channel_nmse", "native_copy_full_channel_nmse",
            "native_no_action_full_channel_nmse",
            "native_target_free_action_swap_exact_full_channel_nmse",
            "native_action_swap_full_channel_nmse", "native_latent_nmse",
            "native_latent_copy_nmse", "native_latent_no_action_nmse",
            "native_latent_action_swap_exact_target_nmse", "native_latent_action_swap_nmse",
            "native_action_swap_exact_fraction", "native_delta_direction_cosine",
            "native_delta_relative_magnitude_error", "native_sgcs",
            "native_path_loss_change_mae", "native_delay_spread_change_mae",
            "native_angular_spread_change_mae", "native_path_loss_direction_accuracy",
            "native_delay_spread_direction_accuracy", "native_angular_spread_direction_accuracy",
            "native_transition_skill", "native_null_delta_rms_mean",
            "native_null_violation_rate", "native_latent_null_delta_rms_mean",
            "native_latent_null_violation_rate",
        ),
    ),
    "response_probe_contract.csv": _csv_contract(
        (
            "seed", "arm", "probe", "steps", "hidden_dim",
        ),
        ("seed", "arm", "probe"),
    ),
}

REQUIRED_ARTIFACTS = tuple(CSV_CONTRACTS) + ("gate.json",)


def _validated_tolerance(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _numeric_difference(reference, candidate, absolute_tolerance, relative_tolerance):
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("numeric comparison requires finite values")
    absolute = abs(reference - candidate)
    scale = max(abs(reference), abs(candidate))
    relative = 0.0 if absolute == 0.0 else absolute / max(scale, 1e-300)
    within = absolute <= absolute_tolerance + relative_tolerance * scale
    return absolute, relative, within


def _append_example(report, example):
    report["difference_count"] += 1
    if len(report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
        report["differences"].append(example)
    else:
        report["differences_truncated"] = True


def _pointer(path, value):
    escaped = str(value).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def compare_json_values(
    reference,
    candidate,
    *,
    absolute_tolerance=0.0,
    relative_tolerance=0.0,
):
    absolute_tolerance = _validated_tolerance(
        absolute_tolerance, "absolute_tolerance"
    )
    relative_tolerance = _validated_tolerance(
        relative_tolerance, "relative_tolerance"
    )
    report = {
        "equivalent": True,
        "exact": True,
        "path_count": 0,
        "float_path_count": 0,
        "changed_float_path_count": 0,
        "float_outside_tolerance_count": 0,
        "nonfloat_mismatch_count": 0,
        "structural_mismatch_count": 0,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "worst_abs_path": None,
        "worst_rel_path": None,
        "difference_count": 0,
        "differences": [],
        "differences_truncated": False,
    }

    def mismatch(kind, path, left, right):
        report["equivalent"] = False
        report["exact"] = False
        if kind == "structural":
            report["structural_mismatch_count"] += 1
        else:
            report["nonfloat_mismatch_count"] += 1
        _append_example(
            report,
            {
                "path": path or "/",
                "kind": kind,
                "reference": left,
                "candidate": right,
                "within_tolerance": False,
            },
        )

    def visit(left, right, path):
        report["path_count"] += 1
        if type(left) is not type(right):
            mismatch("structural", path, left, right)
            return
        if isinstance(left, dict):
            left_keys = set(left)
            right_keys = set(right)
            for key in sorted(left_keys - right_keys):
                mismatch("structural", _pointer(path, key), left[key], "<MISSING>")
            for key in sorted(right_keys - left_keys):
                mismatch("structural", _pointer(path, key), "<MISSING>", right[key])
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], _pointer(path, key))
            return
        if isinstance(left, list):
            if len(left) != len(right):
                mismatch("structural", path, len(left), len(right))
            for index, (left_value, right_value) in enumerate(zip(left, right)):
                visit(left_value, right_value, _pointer(path, index))
            return
        if isinstance(left, float):
            report["float_path_count"] += 1
            try:
                absolute, relative, within = _numeric_difference(
                    left, right, absolute_tolerance, relative_tolerance
                )
            except ValueError:
                mismatch("nonfinite_float", path, left, right)
                return
            if absolute > report["max_abs_diff"]:
                report["max_abs_diff"] = absolute
                report["worst_abs_path"] = path or "/"
            if relative > report["max_rel_diff"]:
                report["max_rel_diff"] = relative
                report["worst_rel_path"] = path or "/"
            if absolute:
                report["exact"] = False
                report["changed_float_path_count"] += 1
                if not within:
                    report["equivalent"] = False
                    report["float_outside_tolerance_count"] += 1
                _append_example(
                    report,
                    {
                        "path": path or "/",
                        "kind": "float",
                        "reference": left,
                        "candidate": right,
                        "abs_diff": absolute,
                        "rel_diff": relative,
                        "within_tolerance": within,
                    },
                )
            return
        if left != right:
            mismatch("nonfloat", path, left, right)

    visit(reference, candidate, "")
    return report


def _float_field_report():
    return {
        "compared_count": 0,
        "null_count": 0,
        "changed_count": 0,
        "outside_tolerance_count": 0,
        "parse_error_count": 0,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "worst_abs": None,
        "worst_rel": None,
        "differences": [],
        "differences_truncated": False,
    }


def _identity_binding_report(identity, fields):
    checked = isinstance(identity, dict)
    missing = [field for field in fields if not checked or field not in identity]
    return {
        "checked": checked,
        "valid": checked and not missing,
        "missing_identity_fields": missing,
        "compared_row_count": 0,
        "mismatch_count": 0,
        "differences": [],
        "differences_truncated": False,
    }


def _record_row_identity_binding(
    report,
    row,
    identity,
    fields,
    row_index,
    primary_key,
):
    if not report["checked"]:
        return
    report["compared_row_count"] += 1
    for field in fields:
        expected = identity.get(field, "<MISSING>")
        actual = row.get(field, "<MISSING>")
        if actual == str(expected):
            continue
        report["valid"] = False
        report["mismatch_count"] += 1
        example = {
            "row_index": row_index,
            "primary_key": list(primary_key),
            "field": field,
            "manifest": expected,
            "row": actual,
        }
        if len(report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
            report["differences"].append(example)
        else:
            report["differences_truncated"] = True


def _implementation_field_report():
    return {
        "match": True,
        "mismatch_count": 0,
        "differences": [],
        "differences_truncated": False,
    }


def _record_implementation_field(
    report,
    field,
    reference,
    candidate,
    row_index,
    primary_key,
):
    if reference == candidate:
        return
    field_report = report["fields"][field]
    field_report["match"] = False
    field_report["mismatch_count"] += 1
    report["match"] = False
    example = {
        "row_index": row_index,
        "primary_key": list(primary_key),
        "reference": reference,
        "candidate": candidate,
    }
    if len(field_report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
        field_report["differences"].append(example)
    else:
        field_report["differences_truncated"] = True


def _record_float_field(
    report,
    reference,
    candidate,
    row_index,
    primary_key,
    absolute_tolerance,
    relative_tolerance,
):
    if reference == "" and candidate == "":
        report["null_count"] += 1
        return True, True
    if reference == "" or candidate == "":
        report["outside_tolerance_count"] += 1
        example = {
            "row_index": row_index,
            "primary_key": list(primary_key),
            "reference": reference,
            "candidate": candidate,
            "kind": "null_mismatch",
        }
        if len(report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
            report["differences"].append(example)
        else:
            report["differences_truncated"] = True
        return False, False
    try:
        left = float(reference)
        right = float(candidate)
        absolute, relative, within = _numeric_difference(
            left, right, absolute_tolerance, relative_tolerance
        )
    except (ValueError, OverflowError):
        report["parse_error_count"] += 1
        report["outside_tolerance_count"] += 1
        example = {
            "row_index": row_index,
            "primary_key": list(primary_key),
            "reference": reference,
            "candidate": candidate,
            "kind": "invalid_float",
        }
        if len(report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
            report["differences"].append(example)
        else:
            report["differences_truncated"] = True
        return False, False
    report["compared_count"] += 1
    if absolute > report["max_abs_diff"]:
        report["max_abs_diff"] = absolute
        report["worst_abs"] = {
            "row_index": row_index,
            "primary_key": list(primary_key),
            "reference": left,
            "candidate": right,
        }
    if relative > report["max_rel_diff"]:
        report["max_rel_diff"] = relative
        report["worst_rel"] = {
            "row_index": row_index,
            "primary_key": list(primary_key),
            "reference": left,
            "candidate": right,
        }
    if not absolute:
        return True, True
    report["changed_count"] += 1
    if not within:
        report["outside_tolerance_count"] += 1
    example = {
        "row_index": row_index,
        "primary_key": list(primary_key),
        "reference": left,
        "candidate": right,
        "abs_diff": absolute,
        "rel_diff": relative,
        "within_tolerance": within,
    }
    if len(report["differences"]) < MAX_DIFFERENCE_EXAMPLES:
        report["differences"].append(example)
    else:
        report["differences_truncated"] = True
    return within, False


def compare_csv_artifact(
    reference_path,
    candidate_path,
    contract,
    *,
    reference_identity=None,
    candidate_identity=None,
    absolute_tolerance=0.0,
    relative_tolerance=0.0,
):
    absolute_tolerance = _validated_tolerance(
        absolute_tolerance, "absolute_tolerance"
    )
    relative_tolerance = _validated_tolerance(
        relative_tolerance, "relative_tolerance"
    )
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    report = {
        "reference_path": str(reference_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "primary_key": list(contract.primary_key),
        "expected_fields": list(contract.fields),
        "reference_fields": None,
        "candidate_fields": None,
        "schema_match": False,
        "reference_row_count": 0,
        "candidate_row_count": 0,
        "row_count_match": False,
        "row_order_match": True,
        "row_order_mismatch_count": 0,
        "primary_key_set_match": False,
        "reference_duplicate_key_count": 0,
        "candidate_duplicate_key_count": 0,
        "duplicate_key_examples": [],
        "nonfloat_mismatch_count": 0,
        "nonfloat_differences": [],
        "nonfloat_differences_truncated": False,
        "identity_binding": {
            "reference": _identity_binding_report(
                reference_identity, EVIDENCE_FIELDS
            ),
            "candidate": _identity_binding_report(
                candidate_identity, EVIDENCE_FIELDS
            ),
        },
        "implementation_identity": {
            "match": True,
            "fields": {
                field: _implementation_field_report()
                for field in CSV_IMPLEMENTATION_IDENTITY_FIELDS
            },
        },
        "float_fields": {
            field: _float_field_report() for field in contract.float_fields
        },
        "structured_fields": {
            field: {
                "compared_count": 0,
                "parse_error_count": 0,
                "equivalent": True,
                "exact": True,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "difference_count": 0,
                "differences": [],
                "differences_truncated": False,
            }
            for field in contract.structured_fields
        },
        "errors": [],
        "equivalent": False,
        "exact": False,
    }
    if not reference_path.is_file() or not candidate_path.is_file():
        if not reference_path.is_file():
            report["errors"].append("reference CSV is missing or not a regular file")
        if not candidate_path.is_file():
            report["errors"].append("candidate CSV is missing or not a regular file")
        return report

    reference_keys = set()
    candidate_keys = set()
    float_equivalent = True
    float_exact = True
    structured_equivalent = True
    structured_exact = True
    try:
        with reference_path.open("r", encoding="utf-8", newline="") as left_handle, candidate_path.open(
            "r", encoding="utf-8", newline=""
        ) as right_handle:
            left_reader = csv.DictReader(left_handle, strict=True)
            right_reader = csv.DictReader(right_handle, strict=True)
            report["reference_fields"] = left_reader.fieldnames
            report["candidate_fields"] = right_reader.fieldnames
            report["schema_match"] = (
                tuple(left_reader.fieldnames or ()) == contract.fields
                and tuple(right_reader.fieldnames or ()) == contract.fields
            )
            if not report["schema_match"]:
                report["errors"].append("CSV fields do not match the frozen contract")
                return report
            sentinel = object()
            for row_index, pair in enumerate(
                zip_longest(left_reader, right_reader, fillvalue=sentinel), start=1
            ):
                left_row, right_row = pair
                if left_row is not sentinel:
                    report["reference_row_count"] += 1
                if right_row is not sentinel:
                    report["candidate_row_count"] += 1
                if left_row is sentinel or right_row is sentinel:
                    report["row_order_match"] = False
                    report["row_order_mismatch_count"] += 1
                    continue
                if None in left_row or None in right_row:
                    report["errors"].append(
                        f"CSV row {row_index} contains fields beyond its header"
                    )
                    continue
                left_key = tuple(left_row[field] for field in contract.primary_key)
                right_key = tuple(right_row[field] for field in contract.primary_key)
                if not all(left_key) or not all(right_key):
                    report["errors"].append(
                        f"CSV row {row_index} contains an empty primary-key component"
                    )
                if left_key in reference_keys:
                    report["reference_duplicate_key_count"] += 1
                    if len(report["duplicate_key_examples"]) < MAX_DIFFERENCE_EXAMPLES:
                        report["duplicate_key_examples"].append(
                            {"side": "reference", "row_index": row_index, "key": list(left_key)}
                        )
                if right_key in candidate_keys:
                    report["candidate_duplicate_key_count"] += 1
                    if len(report["duplicate_key_examples"]) < MAX_DIFFERENCE_EXAMPLES:
                        report["duplicate_key_examples"].append(
                            {"side": "candidate", "row_index": row_index, "key": list(right_key)}
                        )
                reference_keys.add(left_key)
                candidate_keys.add(right_key)
                if left_key != right_key:
                    report["row_order_match"] = False
                    report["row_order_mismatch_count"] += 1
                _record_row_identity_binding(
                    report["identity_binding"]["reference"],
                    left_row,
                    reference_identity,
                    EVIDENCE_FIELDS,
                    row_index,
                    left_key,
                )
                _record_row_identity_binding(
                    report["identity_binding"]["candidate"],
                    right_row,
                    candidate_identity,
                    EVIDENCE_FIELDS,
                    row_index,
                    right_key,
                )
                for field in contract.fields:
                    if field in CSV_IMPLEMENTATION_IDENTITY_FIELDS:
                        _record_implementation_field(
                            report["implementation_identity"],
                            field,
                            left_row[field],
                            right_row[field],
                            row_index,
                            left_key,
                        )
                    elif field in contract.float_fields:
                        within, exact = _record_float_field(
                            report["float_fields"][field],
                            left_row[field],
                            right_row[field],
                            row_index,
                            left_key,
                            absolute_tolerance,
                            relative_tolerance,
                        )
                        float_equivalent = float_equivalent and within
                        float_exact = float_exact and exact
                    elif field in contract.structured_fields:
                        field_report = report["structured_fields"][field]
                        try:
                            left_value = ast.literal_eval(left_row[field])
                            right_value = ast.literal_eval(right_row[field])
                        except (ValueError, SyntaxError):
                            field_report["parse_error_count"] += 1
                            field_report["equivalent"] = False
                            field_report["exact"] = False
                            structured_equivalent = False
                            structured_exact = False
                            continue
                        comparison = compare_json_values(
                            left_value,
                            right_value,
                            absolute_tolerance=absolute_tolerance,
                            relative_tolerance=relative_tolerance,
                        )
                        field_report["compared_count"] += 1
                        field_report["equivalent"] = (
                            field_report["equivalent"] and comparison["equivalent"]
                        )
                        field_report["exact"] = field_report["exact"] and comparison["exact"]
                        field_report["max_abs_diff"] = max(
                            field_report["max_abs_diff"], comparison["max_abs_diff"]
                        )
                        field_report["max_rel_diff"] = max(
                            field_report["max_rel_diff"], comparison["max_rel_diff"]
                        )
                        field_report["difference_count"] += comparison["difference_count"]
                        available = MAX_DIFFERENCE_EXAMPLES - len(field_report["differences"])
                        field_report["differences"].extend(
                            {
                                "row_index": row_index,
                                "primary_key": list(left_key),
                                **difference,
                            }
                            for difference in comparison["differences"][:available]
                        )
                        if comparison["differences_truncated"] or comparison["difference_count"] > available:
                            field_report["differences_truncated"] = True
                        structured_equivalent = structured_equivalent and comparison["equivalent"]
                        structured_exact = structured_exact and comparison["exact"]
                    elif left_row[field] != right_row[field]:
                        report["nonfloat_mismatch_count"] += 1
                        if len(report["nonfloat_differences"]) < MAX_DIFFERENCE_EXAMPLES:
                            report["nonfloat_differences"].append(
                                {
                                    "row_index": row_index,
                                    "primary_key": list(left_key),
                                    "field": field,
                                    "reference": left_row[field],
                                    "candidate": right_row[field],
                                }
                            )
                        else:
                            report["nonfloat_differences_truncated"] = True
    except (OSError, UnicodeError, csv.Error) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        return report

    report["row_count_match"] = (
        report["reference_row_count"] == report["candidate_row_count"]
    )
    report["primary_key_set_match"] = reference_keys == candidate_keys
    structural = (
        report["schema_match"]
        and report["row_count_match"]
        and report["row_order_match"]
        and report["primary_key_set_match"]
        and report["reference_duplicate_key_count"] == 0
        and report["candidate_duplicate_key_count"] == 0
        and report["identity_binding"]["reference"]["valid"]
        and report["identity_binding"]["candidate"]["valid"]
        and not report["errors"]
    )
    nonfloat_exact = report["nonfloat_mismatch_count"] == 0
    report["equivalent"] = (
        structural and nonfloat_exact and float_equivalent and structured_equivalent
    )
    report["exact"] = (
        structural
        and nonfloat_exact
        and float_exact
        and structured_exact
        and report["implementation_identity"]["match"]
    )
    return report


def _runtime_provenance_digest(runtime):
    payload = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_side(root):
    root = Path(root).resolve()
    path = root / "manifest.json"
    report = {
        "path": str(path),
        "valid": False,
        "errors": [],
        "identity": {},
        "required_file_records": {},
        "inventory_paths": [],
    }
    try:
        payload = read_strict_json(path)
    except (OSError, ValueError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        return None, {}, report
    if not isinstance(payload, dict):
        report["errors"].append("manifest must be a JSON object")
        return payload, {}, report
    report["identity"] = {
        field: payload.get(field)
        for field in FROZEN_IDENTITY_FIELDS + IMPLEMENTATION_IDENTITY_FIELDS
    }
    missing_identity = [
        field
        for field in ("schema_version",) + FROZEN_IDENTITY_FIELDS + IMPLEMENTATION_IDENTITY_FIELDS
        if field not in payload
    ]
    if missing_identity:
        report["errors"].append(
            "manifest is missing identity fields: " + ", ".join(missing_identity)
        )
    runtime = payload.get("runtime_provenance")
    if not isinstance(runtime, dict):
        report["errors"].append("manifest runtime_provenance must be an object")
    else:
        try:
            runtime_digest = _runtime_provenance_digest(runtime)
        except (TypeError, ValueError) as error:
            report["errors"].append(
                f"manifest runtime_provenance is not canonical JSON: {error}"
            )
        else:
            if payload.get("runtime_provenance_sha256") != runtime_digest:
                report["errors"].append(
                    "manifest runtime_provenance_sha256 does not bind runtime_provenance"
                )
        for field in ("source_tree_sha256", "requirements_lock_sha256"):
            if runtime.get(field) != payload.get(field):
                report["errors"].append(
                    f"manifest runtime_provenance {field} does not match manifest identity"
                )
    files = payload.get("files")
    if not isinstance(files, list):
        report["errors"].append("manifest files must be a list")
        return payload, {}, report
    index = {}
    for position, record in enumerate(files):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            report["errors"].append(f"manifest file record {position} is malformed")
            continue
        relative = record["path"]
        if relative in index:
            report["errors"].append(f"manifest contains duplicate path: {relative}")
            continue
        artifact = (root / relative).resolve()
        if root not in artifact.parents:
            report["errors"].append(f"manifest path escapes evaluation root: {relative}")
            continue
        index[relative] = record
    report["inventory_paths"] = list(index)
    for relative in REQUIRED_ARTIFACTS:
        record_report = {"present": relative in index, "valid": False, "errors": []}
        report["required_file_records"][relative] = record_report
        if relative not in index:
            record_report["errors"].append("required manifest record is missing")
            continue
        record = index[relative]
        artifact = root / relative
        if not artifact.is_file():
            record_report["errors"].append("required artifact is missing")
            continue
        actual_bytes = artifact.stat().st_size
        actual_sha256 = sha256_file(artifact)
        record_report.update(
            {
                "recorded_bytes": record.get("bytes"),
                "actual_bytes": actual_bytes,
                "recorded_sha256": record.get("sha256"),
                "actual_sha256": actual_sha256,
            }
        )
        if type(record.get("bytes")) is not int or record.get("bytes") != actual_bytes:
            record_report["errors"].append("artifact byte count does not match manifest")
        if record.get("sha256") != actual_sha256:
            record_report["errors"].append("artifact SHA-256 does not match manifest")
        for field in FROZEN_IDENTITY_FIELDS + IMPLEMENTATION_IDENTITY_FIELDS:
            if record.get(field) != payload.get(field):
                record_report["errors"].append(
                    f"artifact record {field} does not match manifest identity"
                )
        record_report["valid"] = not record_report["errors"]
    if any(not value["valid"] for value in report["required_file_records"].values()):
        report["errors"].append("one or more required artifact records are invalid")
    report["valid"] = not report["errors"]
    return payload, index, report


def _identity_layer(reference, candidate, fields):
    values = {}
    match = True
    for field in fields:
        left = reference.get(field) if isinstance(reference, dict) else None
        right = candidate.get(field) if isinstance(candidate, dict) else None
        same = left == right
        values[field] = {"match": same, "reference": left, "candidate": right}
        match = match and same
    return {"match": match, "fields": values}


def _without_keys(value, excluded):
    return {key: item for key, item in value.items() if key not in excluded}


def compare_manifests(reference_root, candidate_root):
    left, left_index, left_report = _manifest_side(reference_root)
    right, right_index, right_report = _manifest_side(candidate_root)
    report = {
        "reference": left_report,
        "candidate": right_report,
        "schema_identity": {"match": False, "reference": None, "candidate": None},
        "frozen_experiment_identity": {"match": False, "fields": {}},
        "implementation_identity": {"match": False, "fields": {}},
        "scientific_metadata": {"equivalent": False, "exact": False},
        "artifact_record_metadata": {},
        "inventory": {
            "match": False,
            "order_match": False,
            "reference_paths": left_report["inventory_paths"],
            "candidate_paths": right_report["inventory_paths"],
            "only_reference": [],
            "only_candidate": [],
        },
        "required_artifact_hashes": {},
        "comparable": False,
        "exact": False,
    }
    if not isinstance(left, dict) or not isinstance(right, dict):
        return report
    report["schema_identity"] = {
        "match": left.get("schema_version") == right.get("schema_version"),
        "reference": left.get("schema_version"),
        "candidate": right.get("schema_version"),
    }
    report["frozen_experiment_identity"] = _identity_layer(
        left, right, FROZEN_IDENTITY_FIELDS
    )
    report["implementation_identity"] = _identity_layer(
        left, right, IMPLEMENTATION_IDENTITY_FIELDS
    )
    report["scientific_metadata"] = compare_json_values(
        _without_keys(left, {"files", *IMPLEMENTATION_IDENTITY_FIELDS}),
        _without_keys(right, {"files", *IMPLEMENTATION_IDENTITY_FIELDS}),
    )
    left_paths = list(left_index)
    right_paths = list(right_index)
    report["inventory"].update(
        {
            "match": set(left_paths) == set(right_paths),
            "order_match": left_paths == right_paths,
            "only_reference": sorted(set(left_paths) - set(right_paths)),
            "only_candidate": sorted(set(right_paths) - set(left_paths)),
        }
    )
    required_hashes_exact = True
    for relative in REQUIRED_ARTIFACTS:
        left_record = left_index.get(relative, {})
        right_record = right_index.get(relative, {})
        value = {
            "bytes_match": left_record.get("bytes") == right_record.get("bytes"),
            "sha256_match": left_record.get("sha256") == right_record.get("sha256"),
            "reference_sha256": left_record.get("sha256"),
            "candidate_sha256": right_record.get("sha256"),
        }
        report["required_artifact_hashes"][relative] = value
        required_hashes_exact = (
            required_hashes_exact and value["bytes_match"] and value["sha256_match"]
        )
    artifact_record_metadata_equivalent = True
    artifact_record_metadata_exact = True
    excluded_record_fields = {
        *MANIFEST_ARTIFACT_CONTENT_FIELDS,
        *IMPLEMENTATION_IDENTITY_FIELDS,
    }
    for relative in sorted(set(left_paths) | set(right_paths)):
        left_record = left_index.get(relative)
        right_record = right_index.get(relative)
        if left_record is None or right_record is None:
            comparison = {"equivalent": False, "exact": False}
        else:
            comparison = compare_json_values(
                _without_keys(left_record, excluded_record_fields),
                _without_keys(right_record, excluded_record_fields),
            )
        report["artifact_record_metadata"][relative] = comparison
        artifact_record_metadata_equivalent = (
            artifact_record_metadata_equivalent and comparison["equivalent"]
        )
        artifact_record_metadata_exact = (
            artifact_record_metadata_exact and comparison["exact"]
        )
    report["comparable"] = (
        left_report["valid"]
        and right_report["valid"]
        and report["schema_identity"]["match"]
        and report["frozen_experiment_identity"]["match"]
        and report["inventory"]["match"]
        and report["inventory"]["order_match"]
        and report["scientific_metadata"]["equivalent"]
        and artifact_record_metadata_equivalent
    )
    report["exact"] = (
        report["comparable"]
        and report["implementation_identity"]["match"]
        and report["inventory"]["match"]
        and report["inventory"]["order_match"]
        and report["scientific_metadata"]["exact"]
        and artifact_record_metadata_exact
        and required_hashes_exact
    )
    return report


def _read_gate(path):
    try:
        payload = read_strict_json(path)
    except (OSError, ValueError) as error:
        return None, {"valid": False, "error": f"{type(error).__name__}: {error}"}
    if not isinstance(payload, dict):
        return payload, {"valid": False, "error": "gate must be a JSON object"}
    return payload, {"valid": True, "error": None}


def _gate_identity_binding(gate, manifest_identity):
    fields = FROZEN_IDENTITY_FIELDS + IMPLEMENTATION_IDENTITY_FIELDS
    report = {
        "checked": isinstance(gate, dict) and isinstance(manifest_identity, dict),
        "valid": False,
        "fields": {},
    }
    if not report["checked"]:
        return report
    report["fields"] = {
        field: {
            "match": gate.get(field) == manifest_identity.get(field),
            "gate": gate.get(field),
            "manifest": manifest_identity.get(field),
        }
        for field in fields
    }
    report["valid"] = all(value["match"] for value in report["fields"].values())
    return report


def _gate_payload_layers(gate):
    scientific = _without_keys(gate, set(GATE_IMPLEMENTATION_EVIDENCE_KEYS))
    implementation = {
        field: gate[field]
        for field in GATE_IMPLEMENTATION_EVIDENCE_KEYS
        if field in gate
    }
    return scientific, implementation


def _write_report_atomic(path, report):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def compare_evaluation_artifacts(
    reference_root,
    candidate_root,
    *,
    absolute_tolerance=0.0,
    relative_tolerance=0.0,
    output_path=None,
):
    absolute_tolerance = _validated_tolerance(
        absolute_tolerance, "absolute_tolerance"
    )
    relative_tolerance = _validated_tolerance(
        relative_tolerance, "relative_tolerance"
    )
    reference_root = Path(reference_root).resolve()
    candidate_root = Path(candidate_root).resolve()
    if reference_root == candidate_root:
        raise ValueError("reference and candidate evaluation roots must be distinct")
    if output_path is not None and (
        _inside(output_path, reference_root) or _inside(output_path, candidate_root)
    ):
        raise ValueError("equivalence report must be outside both evaluation roots")
    report = {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "tolerance": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
            "rule": "abs(a-b) <= absolute + relative * max(abs(a), abs(b))",
        },
        "required_artifacts": list(REQUIRED_ARTIFACTS) + ["manifest.json"],
        "csv": {},
        "gate": None,
        "manifest": None,
        "summary": {},
        "equivalent": False,
        "exact_match": False,
        "status": "FAIL",
    }
    report["manifest"] = compare_manifests(reference_root, candidate_root)
    reference_identity = report["manifest"]["reference"].get("identity")
    candidate_identity = report["manifest"]["candidate"].get("identity")
    for name, contract in CSV_CONTRACTS.items():
        report["csv"][name] = compare_csv_artifact(
            reference_root / name,
            candidate_root / name,
            contract,
            reference_identity=reference_identity,
            candidate_identity=candidate_identity,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    left_gate, left_gate_status = _read_gate(reference_root / "gate.json")
    right_gate, right_gate_status = _read_gate(candidate_root / "gate.json")
    if left_gate_status["valid"] and right_gate_status["valid"]:
        left_scientific_gate, left_implementation_gate = _gate_payload_layers(
            left_gate
        )
        right_scientific_gate, right_implementation_gate = _gate_payload_layers(
            right_gate
        )
        gate_comparison = compare_json_values(
            left_scientific_gate,
            right_scientific_gate,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        gate_implementation_comparison = compare_json_values(
            left_implementation_gate,
            right_implementation_gate,
        )
    else:
        gate_comparison = {"equivalent": False, "exact": False}
        gate_implementation_comparison = {"equivalent": False, "exact": False}
    reference_gate_binding = _gate_identity_binding(left_gate, reference_identity)
    candidate_gate_binding = _gate_identity_binding(right_gate, candidate_identity)
    gate_equivalent = (
        gate_comparison["equivalent"]
        and reference_gate_binding["valid"]
        and candidate_gate_binding["valid"]
    )
    gate_exact = gate_equivalent and gate_comparison["exact"] and (
        gate_implementation_comparison["exact"]
    )
    report["gate"] = {
        "reference": left_gate_status,
        "candidate": right_gate_status,
        "comparison": gate_comparison,
        "implementation_evidence": {
            "keys": list(GATE_IMPLEMENTATION_EVIDENCE_KEYS),
            "comparison": gate_implementation_comparison,
        },
        "identity_binding": {
            "reference": reference_gate_binding,
            "candidate": candidate_gate_binding,
        },
        "equivalent": gate_equivalent,
        "exact": gate_exact,
    }
    csv_equivalent = all(value["equivalent"] for value in report["csv"].values())
    csv_exact = all(value["exact"] for value in report["csv"].values())
    report["equivalent"] = (
        csv_equivalent
        and gate_equivalent
        and report["manifest"]["comparable"]
    )
    report["exact_match"] = (
        csv_exact
        and gate_exact
        and report["manifest"]["exact"]
    )
    report["status"] = "PASS" if report["equivalent"] else "FAIL"
    report["summary"] = {
        "csv_passed": sum(value["equivalent"] for value in report["csv"].values()),
        "csv_total": len(report["csv"]),
        "csv_exact": sum(value["exact"] for value in report["csv"].values()),
        "gate_equivalent": gate_equivalent,
        "gate_exact": gate_exact,
        "manifest_comparable": report["manifest"]["comparable"],
        "manifest_exact": report["manifest"]["exact"],
    }
    if output_path is not None:
        _write_report_atomic(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare complete CSI-PAIRS V6 evaluation artifacts."
    )
    parser.add_argument("reference_root")
    parser.add_argument("candidate_root")
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    report = compare_evaluation_artifacts(
        arguments.reference_root,
        arguments.candidate_root,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
        output_path=arguments.output,
    )
    if arguments.output:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "equivalent": report["equivalent"],
                    "exact_match": report["exact_match"],
                    "report": str(Path(arguments.output).resolve()),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
