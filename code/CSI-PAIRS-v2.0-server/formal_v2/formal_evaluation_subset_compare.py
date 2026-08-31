from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import struct
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


BOUND_SOURCE_ROOT_ENV = "CSI_PAIRS_SUBSET_WORKER_SOURCE_ROOT"
_BOUND_SOURCE_ROOT = os.environ.get(BOUND_SOURCE_ROOT_ENV)
if _BOUND_SOURCE_ROOT:
    _bound_root = Path(_BOUND_SOURCE_ROOT).resolve()
    sys.path.insert(0, str(_bound_root))
    if __name__ == "__main__":
        __package__ = "formal_v2"


from . import formal_evaluation as legacy
try:
    from . import formal_evaluation_streaming as streaming
except ImportError:
    if (Path(legacy.__file__).resolve().parent / "formal_evaluation_streaming.py").exists():
        raise
    streaming = None
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import (
    FACTORIAL_SCHEMA,
    QUALIFICATION_SCHEMA,
    bind_rows,
    config_sha256,
    configure_reproducible_runtime,
    evidence_context,
)
from .formal_factorial import _training_normalization
from .formal_io import read_strict_json, sha256_file
from .formal_model import resolve_execution_device, torch
from .formal_probes import (
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)
from .formal_routing import ROUTE_NAMES, fit_route_normalization, route_dataset
from .formal_teacher import load_teacher_bundle


REPORT_SCHEMA = "csi-pairs-v6-real-evaluation-subset-equivalence-report-v1"
WORKER_FRAGMENT_SCHEMA = "csi-pairs-v6-real-evaluation-subset-worker-fragment-v1"
WORKER_REQUEST_SCHEMA = "csi-pairs-v6-real-evaluation-subset-worker-request-v1"
FROZEN_LEGACY_COMMIT = "9850fffe0b34f16b45066973308f18b10555ca5d"
FROZEN_LEGACY_EVALUATION_SHA256 = (
    "259478e4acf6285cb1b00c2d3d02510f88e48caaf6ed26705fe2abf5d4eeddbb"
)
SELECTION_RULE = (
    "source:lexicographic(city_id,bank_id,canonical_bank_digest,scene_id,scene_index);"
    "target:one-or-more-per-city-by-the-same-key;execution:lexicographic-bank_id;v1"
)
CHECKPOINT_SELECTION_RULE = "seed-ascending,frozen-arm-order,checkpoint-sha256;v1"
GATE_REQUIRED_TABLES = (
    "cgs_per_bank.csv",
    "alignment_shortcut_baselines.csv",
    "compatibility_route_distributions.csv",
    "cgs_active_effect_bins.csv",
    "response_per_bank.csv",
)
MAX_DIFFERENCE_EXAMPLES = 20
IMPLEMENTATION_EVIDENCE_FIELDS = (
    "source_tree_sha256",
    "runtime_provenance_sha256",
)
SOURCE_IDENTITY_SUFFIXES = frozenset(
    {".py", ".json", ".sh", ".txt", ".toml", ".lock", ".yaml", ".yml"}
)
SOURCE_IDENTITY_EXCLUDED_PREFIXES = (".runtime-", ".venv-")


def _field_tuple(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


_FROZEN_EVALUATION_TABLE_FIELDS = {
    "compatibility_probe_contract.csv": _field_tuple(
        "seed,arm,selected_family,selection_nll,selection_auroc,candidate_metrics,"
        "artifact_label,dataset_sha256,config_sha256,fixture,scientific_use,"
        "source_tree_sha256,requirements_lock_sha256,runtime_provenance_sha256"
    ),
    "response_probe_contract.csv": _field_tuple(
        "seed,arm,probe,steps,hidden_dim,artifact_label,dataset_sha256,config_sha256,"
        "fixture,scientific_use,source_tree_sha256,requirements_lock_sha256,"
        "runtime_provenance_sha256"
    ),
    "cgs_per_bank.csv": _field_tuple(
        "seed,arm,bank_id,base_map_cluster_id,canonical_base_map_digest,"
        "canonical_bank_digest,city_id,evaluation_scope,route,probe_family,cgs_auroc,"
        "native_energy_auroc,native_audit_hold_auroc,native_probe_spearman,"
        "native_audit_hold_probe_spearman,n,artifact_label,dataset_sha256,config_sha256,"
        "fixture,scientific_use,source_tree_sha256,requirements_lock_sha256,"
        "runtime_provenance_sha256"
    ),
    "alignment_shortcut_baselines.csv": _field_tuple(
        "seed,arm,bank_id,base_map_cluster_id,canonical_base_map_digest,"
        "canonical_bank_digest,city_id,evaluation_scope,baseline,input_class,"
        "source_train_auroc,unseen_bank_auroc,auroc,identity_token_contract,"
        "probe_family,fit_role,selection_role,n,artifact_label,dataset_sha256,"
        "config_sha256,fixture,scientific_use,source_tree_sha256,"
        "requirements_lock_sha256,runtime_provenance_sha256"
    ),
    "compatibility_route_distributions.csv": _field_tuple(
        "seed,arm,bank_id,base_map_cluster_id,canonical_base_map_digest,"
        "canonical_bank_digest,city_id,evaluation_scope,route,condition_status,"
        "pair_count,matched_minus_alternative_mean,matched_minus_alternative_median,"
        "absolute_difference_p90,overclassification_rate,artifact_label,dataset_sha256,"
        "config_sha256,fixture,scientific_use,source_tree_sha256,"
        "requirements_lock_sha256,runtime_provenance_sha256"
    ),
    "cgs_active_effect_bins.csv": _field_tuple(
        "seed,arm,bank_id,city_id,evaluation_scope,effect_bin,pair_count,"
        "physical_distance_min,physical_distance_max,cgs_auroc,artifact_label,"
        "dataset_sha256,config_sha256,fixture,scientific_use,source_tree_sha256,"
        "requirements_lock_sha256,runtime_provenance_sha256"
    ),
    "compatibility_pair_effects.csv": _field_tuple(
        "seed,arm,pair_id,scene_index,bank_id,base_map_cluster_id,"
        "canonical_base_map_digest,canonical_bank_digest,city_id,source_world,"
        "target_world,position_index,route,physical_distance,matched_minus_alternative,"
        "artifact_label,dataset_sha256,config_sha256,fixture,scientific_use,"
        "source_tree_sha256,requirements_lock_sha256,runtime_provenance_sha256"
    ),
    "response_pair_effects.csv": _field_tuple(
        "seed,arm,pair_id,scene_index,bank_id,base_map_cluster_id,"
        "canonical_base_map_digest,canonical_bank_digest,city_id,source_world,"
        "target_world,position_index,query_index,route,wrong_action_match_status,"
        "wrong_action_world,prediction_mse,copy_mse,action_swap_mse,no_action_mse,"
        "response_advantage,response_advantage_vs_action_swap,"
        "response_advantage_vs_no_action,artifact_label,dataset_sha256,config_sha256,"
        "fixture,scientific_use,source_tree_sha256,requirements_lock_sha256,"
        "runtime_provenance_sha256"
    ),
    "response_per_bank.csv": _field_tuple(
        "seed,arm,bank_id,base_map_cluster_id,canonical_base_map_digest,"
        "canonical_bank_digest,city_id,evaluation_scope,"
        "unified_response_probe_active_patch_nmse,probe_copy_active_patch_nmse,"
        "unified_response_probe_action_swap_exact_patch_nmse,"
        "probe_action_swap_active_patch_nmse,probe_action_swap_exact_count,"
        "probe_action_swap_fallback_count,probe_action_swap_failed_count,"
        "probe_action_swap_exact_fraction,probe_without_map_active_patch_nmse,"
        "probe_edit_only_active_patch_nmse,probe_csi_only_active_patch_nmse,"
        "probe_oracle_x_active_patch_nmse,native_target_free_full_channel_nmse,"
        "native_copy_full_channel_nmse,native_no_action_full_channel_nmse,"
        "native_target_free_action_swap_exact_full_channel_nmse,"
        "native_action_swap_full_channel_nmse,native_latent_nmse,"
        "native_latent_copy_nmse,native_latent_no_action_nmse,"
        "native_latent_action_swap_exact_target_nmse,native_latent_action_swap_nmse,"
        "native_action_swap_exact_count,native_action_swap_fallback_count,"
        "native_action_swap_failed_count,native_action_swap_exact_fraction,"
        "native_delta_direction_cosine,native_delta_relative_magnitude_error,native_sgcs,"
        "native_path_loss_change_mae,native_delay_spread_change_mae,"
        "native_angular_spread_change_mae,native_path_loss_direction_accuracy,"
        "native_delay_spread_direction_accuracy,native_angular_spread_direction_accuracy,"
        "native_transition_skill,native_null_patch_count,native_null_delta_rms_mean,"
        "native_null_violation_rate,native_latent_null_delta_rms_mean,"
        "native_latent_null_violation_rate,native_mask_bank,native_input_contract,"
        "n_active_patches,artifact_label,dataset_sha256,config_sha256,fixture,"
        "scientific_use,source_tree_sha256,requirements_lock_sha256,"
        "runtime_provenance_sha256"
    ),
}

EVALUATION_TABLE_FIELDS = (
    streaming.EVALUATION_TABLE_FIELDS
    if streaming is not None
    else _FROZEN_EVALUATION_TABLE_FIELDS
)
if EVALUATION_TABLE_FIELDS != _FROZEN_EVALUATION_TABLE_FIELDS:
    raise RuntimeError("current streaming table contract differs from the frozen V6 contract")
SCENE_TABLES = (
    streaming.SCENE_TABLES
    if streaming is not None
    else (
        "cgs_per_bank.csv",
        "alignment_shortcut_baselines.csv",
        "compatibility_route_distributions.csv",
        "cgs_active_effect_bins.csv",
        "compatibility_pair_effects.csv",
        "response_pair_effects.csv",
        "response_per_bank.csv",
    )
)
TABLE_PRIMARY_KEYS = {
    "alignment_shortcut_baselines.csv": ("seed", "arm", "bank_id", "baseline"),
    "cgs_active_effect_bins.csv": ("seed", "arm", "bank_id", "effect_bin"),
    "cgs_per_bank.csv": ("seed", "arm", "bank_id"),
    "compatibility_pair_effects.csv": ("seed", "arm", "pair_id"),
    "compatibility_probe_contract.csv": ("seed", "arm"),
    "compatibility_route_distributions.csv": ("seed", "arm", "bank_id", "route"),
    "response_pair_effects.csv": ("seed", "arm", "pair_id"),
    "response_per_bank.csv": ("seed", "arm", "bank_id"),
    "response_probe_contract.csv": ("seed", "arm", "probe"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_git_commit(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _inside(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_report_destination(
    report_path: str | Path,
    legacy_run_root: str | Path,
    *,
    source_root: str | Path | None = None,
    protected_files: Iterable[str | Path] = (),
) -> Path:
    """Require the sole output to live outside both source and the legacy run."""

    target = Path(report_path).resolve()
    legacy_root = Path(legacy_run_root).resolve()
    implementation_root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parent
    )
    if _inside(target, legacy_root):
        raise ValueError("subset comparison report must be outside the legacy run")
    if _inside(target, implementation_root):
        raise ValueError("subset comparison report must be outside the source tree")
    protected = {Path(path).resolve() for path in protected_files}
    if target in protected:
        raise ValueError("subset comparison report would overwrite an authenticated input")
    if os.path.lexists(os.fspath(target)) and (target.is_symlink() or not target.is_file()):
        raise ValueError("subset comparison report target must be a regular file")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_report_atomic(path: str | Path, report: object) -> Path:
    """Publish strict JSON by fsync plus atomic replacement."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("subset comparison report parent must be a regular directory")
    if os.path.lexists(os.fspath(target)) and (target.is_symlink() or not target.is_file()):
        raise ValueError("subset comparison report target must be a regular file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                # Worker table field order is part of the scientific contract.
                sort_keys=False,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def _scene_selection_key(dataset, scene: int) -> tuple[str, str, str, str, int]:
    index = int(scene)
    return (
        str(dataset.city_ids[index]),
        str(dataset.bank_ids[index]),
        str(legacy._canonical_bank_digest(dataset, index)),
        str(dataset.scene_ids[index]),
        index,
    )


def _scene_record(dataset, scene: int) -> dict[str, object]:
    index = int(scene)
    return {
        "scene_index": index,
        "scene_id": str(dataset.scene_ids[index]),
        "scene_role": str(dataset.scene_roles[index]),
        "city_id": str(dataset.city_ids[index]),
        "bank_id": str(dataset.bank_ids[index]),
        "base_map_cluster_id": str(dataset.base_map_cluster_ids[index]),
        "canonical_base_map_digest": dataset.canonical_base_map_digest(index),
        "canonical_bank_digest": legacy._canonical_bank_digest(dataset, index),
    }


def select_real_evaluation_subset(
    dataset,
    *,
    source_scenes: int = 1,
    target_scenes_per_city: int = 1,
) -> tuple[dict[str, object], ...]:
    """Select real banks deterministically without slicing or rewriting the dataset."""

    if bool(dataset.is_fixture):
        raise ValueError("real-subset comparison rejects synthetic/fixture datasets")
    for value, label in (
        (source_scenes, "source_scenes"),
        (target_scenes_per_city, "target_scenes_per_city"),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")

    source_candidates = sorted(
        (int(value) for value in dataset.indices_for_role("source_final_unseen_bank")),
        key=lambda scene: _scene_selection_key(dataset, scene),
    )
    if len(source_candidates) < source_scenes:
        raise RuntimeError("formal dataset has too few source_final_unseen_bank scenes")
    target_candidates = [
        int(value) for value in dataset.indices_for_role("target")
    ]
    target_cities = sorted({str(dataset.city_ids[scene]) for scene in target_candidates})
    if not target_cities:
        raise RuntimeError("formal dataset has no target scenes")

    chosen = source_candidates[:source_scenes]
    for city in target_cities:
        city_candidates = sorted(
            (
                scene
                for scene in target_candidates
                if str(dataset.city_ids[scene]) == city
            ),
            key=lambda scene: _scene_selection_key(dataset, scene),
        )
        if len(city_candidates) < target_scenes_per_city:
            raise RuntimeError(
                f"formal dataset target city {city!r} has too few scenes"
            )
        chosen.extend(city_candidates[:target_scenes_per_city])

    bank_ids = [str(dataset.bank_ids[scene]) for scene in chosen]
    if len(bank_ids) != len(set(bank_ids)):
        raise RuntimeError("real-subset selection requires unique bank IDs")
    chosen.sort(key=lambda scene: (str(dataset.bank_ids[scene]), int(scene)))
    return tuple(_scene_record(dataset, scene) for scene in chosen)


def _frozen_subset_evaluation_scenes(
    dataset, scenes: Iterable[int]
) -> tuple[int, ...]:
    """Restore the frozen source-then-target matrix order for a selected subset."""

    selected = tuple(int(scene) for scene in scenes)
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("subset evaluation scenes must be nonempty and unique")
    ordered = tuple(
        scene
        for role in ("source_final_unseen_bank", "target")
        for scene in selected
        if str(dataset.scene_roles[scene]) == role
    )
    roles = {str(dataset.scene_roles[scene]) for scene in selected}
    if (
        len(ordered) != len(selected)
        or roles != {"source_final_unseen_bank", "target"}
    ):
        raise RuntimeError(
            "subset evaluation must contain only source unseen and target scenes"
        )
    return ordered


def select_legacy_checkpoints(
    checkpoint_rows: Iterable[Mapping[str, object]],
    arm_order: Iterable[str],
    *,
    checkpoint_count: int = 1,
    checkpoint_arm: str | None = None,
) -> tuple[dict[str, object], ...]:
    if type(checkpoint_count) is not int or checkpoint_count < 1:
        raise ValueError("checkpoint_count must be a positive integer")
    arms = tuple(str(value) for value in arm_order)
    if len(arms) != len(set(arms)) or not arms:
        raise ValueError("checkpoint arm order must be nonempty and unique")
    if checkpoint_arm is not None and checkpoint_arm not in arms:
        raise ValueError(f"unknown checkpoint arm: {checkpoint_arm!r}")
    arm_rank = {arm: index for index, arm in enumerate(arms)}
    rows = [dict(row) for row in checkpoint_rows]
    if checkpoint_arm is not None:
        rows = [row for row in rows if str(row.get("arm")) == checkpoint_arm]
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            arm_rank.get(str(row["arm"]), len(arms)),
            str(row["sha256"]),
        )
    )
    if len(rows) < checkpoint_count:
        raise RuntimeError("legacy checkpoint inventory is smaller than requested subset")
    selected = rows[:checkpoint_count]
    for row in selected:
        if str(row.get("arm")) not in arm_rank or not _valid_sha256(row.get("sha256")):
            raise RuntimeError("selected legacy checkpoint identity is malformed")
    return tuple(selected)


def _normalize_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest_value(digest, value: object) -> None:
    value = _normalize_scalar(value)
    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        encoded = str(value).encode("ascii")
        digest.update(b"I" + len(encoded).to_bytes(8, "big") + encoded)
    elif type(value) is float:
        digest.update(b"F" + struct.pack(">d", value))
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value):
            _digest_value(digest, str(key))
            _digest_value(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _digest_value(digest, item)
        digest.update(b"]")
    else:
        raise TypeError(f"unsupported scientific row value: {type(value).__name__}")


def _table_sha256(rows: Iterable[Mapping[str, object]], fields: Iterable[str]) -> str:
    digest = hashlib.sha256()
    field_tuple = tuple(fields)
    for field in field_tuple:
        _digest_value(digest, field)
    for row in rows:
        digest.update(b"R")
        for field in field_tuple:
            _digest_value(digest, row.get(field))
    return digest.hexdigest()


def _json_safe_value(value: object) -> object:
    value = _normalize_scalar(value)
    if type(value) is float:
        return {
            "decimal": repr(value),
            "hex": value.hex(),
            "bits": struct.pack(">d", value).hex(),
        }
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return repr(value)


def _append_difference(state: dict[str, object], example: dict[str, object]) -> None:
    examples = state["difference_examples"]
    if len(examples) < MAX_DIFFERENCE_EXAMPLES:
        examples.append(example)


def _compare_value(
    reference: object,
    candidate: object,
    *,
    path: str,
    state: dict[str, object],
) -> bool:
    reference = _normalize_scalar(reference)
    candidate = _normalize_scalar(candidate)
    if type(reference) is float and type(candidate) is float:
        state["float_comparisons"] += 1
        if not math.isfinite(reference) or not math.isfinite(candidate):
            _append_difference(
                state,
                {
                    "path": path,
                    "reason": "nonfinite_float",
                    "legacy": _json_safe_value(reference),
                    "streaming": _json_safe_value(candidate),
                },
            )
            return False
        difference = abs(reference - candidate)
        state["max_absolute_difference"] = max(
            state["max_absolute_difference"], difference
        )
        denominator = max(abs(reference), abs(candidate))
        relative = difference / denominator if denominator else 0.0
        state["max_relative_difference"] = max(
            state["max_relative_difference"], relative
        )
        same_bits = struct.pack(">d", reference) == struct.pack(">d", candidate)
        if same_bits:
            state["bitwise_equal_float_count"] += 1
            return True
        _append_difference(
            state,
            {
                "path": path,
                "reason": "float_bits",
                "absolute_difference": difference,
                "relative_difference": relative,
                "legacy": _json_safe_value(reference),
                "streaming": _json_safe_value(candidate),
            },
        )
        return False
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        if list(reference) != list(candidate):
            _append_difference(
                state,
                {
                    "path": path,
                    "reason": "mapping_keys_or_order",
                    "legacy": list(reference),
                    "streaming": list(candidate),
                },
            )
            return False
        return all(
            _compare_value(
                reference[key], candidate[key], path=f"{path}/{key}", state=state
            )
            for key in reference
        )
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if type(reference) is not type(candidate) or len(reference) != len(candidate):
            _append_difference(
                state,
                {
                    "path": path,
                    "reason": "sequence_shape_or_type",
                    "legacy": _json_safe_value(reference),
                    "streaming": _json_safe_value(candidate),
                },
            )
            return False
        return all(
            _compare_value(left, right, path=f"{path}/{index}", state=state)
            for index, (left, right) in enumerate(zip(reference, candidate))
        )
    if type(reference) is type(candidate) and reference == candidate:
        return True
    _append_difference(
        state,
        {
            "path": path,
            "reason": "value_or_type",
            "legacy_type": type(reference).__name__,
            "streaming_type": type(candidate).__name__,
            "legacy": _json_safe_value(reference),
            "streaming": _json_safe_value(candidate),
        },
    )
    return False


def _row_keys(
    rows: Iterable[Mapping[str, object]], primary_key: Iterable[str]
) -> list[tuple[object, ...]]:
    fields = tuple(primary_key)
    return [tuple(_normalize_scalar(row.get(field)) for field in fields) for row in rows]


def compare_evaluation_tables(
    legacy_tables: Mapping[str, Iterable[Mapping[str, object]]],
    streaming_tables: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    implementation_fields: Iterable[str] = (),
) -> dict[str, object]:
    expected_tables = tuple(EVALUATION_TABLE_FIELDS)
    excluded = tuple(implementation_fields)
    if any(
        field not in fields
        for field in excluded
        for fields in EVALUATION_TABLE_FIELDS.values()
    ):
        raise ValueError("implementation comparison field is not present in every table")
    if set(legacy_tables) != set(expected_tables) or set(streaming_tables) != set(
        expected_tables
    ):
        raise ValueError("comparison requires exactly the complete nine-table inventory")

    reports: dict[str, object] = {}
    total_rows = 0
    total_float_comparisons = 0
    total_bitwise_equal_floats = 0
    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    all_exact = True
    all_fields = True
    all_order = True
    all_unique = True
    for table in expected_tables:
        fields = EVALUATION_TABLE_FIELDS[table]
        legacy_rows = [dict(row) for row in legacy_tables[table]]
        streaming_rows = [dict(row) for row in streaming_tables[table]]
        legacy_field_order = all(tuple(row) == fields for row in legacy_rows)
        streaming_field_order = all(tuple(row) == fields for row in streaming_rows)
        field_sets_valid = all(set(row) == set(fields) for row in legacy_rows) and all(
            set(row) == set(fields) for row in streaming_rows
        )
        scientific_fields = tuple(field for field in fields if field not in excluded)
        primary_key = TABLE_PRIMARY_KEYS[table]
        legacy_keys = _row_keys(legacy_rows, primary_key)
        streaming_keys = _row_keys(streaming_rows, primary_key)
        row_order_equal = legacy_keys == streaming_keys
        primary_keys_unique = (
            len(legacy_keys) == len(set(legacy_keys))
            and len(streaming_keys) == len(set(streaming_keys))
        )
        state: dict[str, object] = {
            "float_comparisons": 0,
            "bitwise_equal_float_count": 0,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
            "difference_examples": [],
        }
        exact = len(legacy_rows) == len(streaming_rows)
        if not exact:
            _append_difference(
                state,
                {
                    "path": f"/{table}",
                    "reason": "row_count",
                    "legacy": len(legacy_rows),
                    "streaming": len(streaming_rows),
                },
            )
        for row_index, (legacy_row, streaming_row) in enumerate(
            zip(legacy_rows, streaming_rows)
        ):
            for field in fields:
                if field not in legacy_row or field not in streaming_row:
                    exact = False
                    continue
                if field in excluded:
                    continue
                if not _compare_value(
                    legacy_row[field],
                    streaming_row[field],
                    path=f"/{table}/{row_index}/{field}",
                    state=state,
                ):
                    exact = False
        fields_exact = bool(
            field_sets_valid and legacy_field_order and streaming_field_order
        )
        exact = bool(exact and fields_exact and row_order_equal and primary_keys_unique)
        report = {
            "exact": exact,
            "expected_fields": list(fields),
            "field_order_equal": bool(legacy_field_order and streaming_field_order),
            "field_sets_valid": field_sets_valid,
            "primary_key": list(primary_key),
            "primary_keys_unique": primary_keys_unique,
            "row_order_equal": row_order_equal,
            "legacy_row_count": len(legacy_rows),
            "streaming_row_count": len(streaming_rows),
            "scientific_fields": list(scientific_fields),
            "implementation_fields": list(excluded),
            "legacy_scientific_ordered_sha256": _table_sha256(
                legacy_rows, scientific_fields
            ),
            "streaming_scientific_ordered_sha256": _table_sha256(
                streaming_rows, scientific_fields
            ),
            "legacy_full_ordered_sha256": _table_sha256(legacy_rows, fields),
            "streaming_full_ordered_sha256": _table_sha256(streaming_rows, fields),
            "implementation_values": {
                field: {
                    "legacy": sorted({str(row[field]) for row in legacy_rows}),
                    "streaming": sorted({str(row[field]) for row in streaming_rows}),
                }
                for field in excluded
            },
            **state,
        }
        reports[table] = report
        total_rows += len(legacy_rows)
        total_float_comparisons += int(state["float_comparisons"])
        total_bitwise_equal_floats += int(state["bitwise_equal_float_count"])
        maximum_absolute_difference = max(
            maximum_absolute_difference, float(state["max_absolute_difference"])
        )
        maximum_relative_difference = max(
            maximum_relative_difference, float(state["max_relative_difference"])
        )
        all_exact = all_exact and exact
        all_fields = all_fields and fields_exact
        all_order = all_order and row_order_equal
        all_unique = all_unique and primary_keys_unique
    return {
        "exact": all_exact,
        "bitwise_float_equal": (
            total_float_comparisons == total_bitwise_equal_floats
        ),
        "field_contract_equal": all_fields,
        "row_order_equal": all_order,
        "primary_keys_unique": all_unique,
        "comparison_profile": (
            "cross_source_scientific_payload"
            if excluded
            else "same_source_complete_rows"
        ),
        "implementation_fields_compared_separately": list(excluded),
        "table_count": len(expected_tables),
        "legacy_total_row_count": total_rows,
        "float_comparisons": total_float_comparisons,
        "bitwise_equal_float_count": total_bitwise_equal_floats,
        "max_absolute_difference": maximum_absolute_difference,
        "max_relative_difference": maximum_relative_difference,
        "tables": reports,
    }


def gate_aggregate_report(
    legacy_tables: Mapping[str, Iterable[Mapping[str, object]]],
    streaming_tables: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    implementation_fields: Iterable[str] = (),
) -> dict[str, object]:
    excluded = set(implementation_fields)
    tables = {}
    exact = True
    for table in GATE_REQUIRED_TABLES:
        fields = tuple(
            field for field in EVALUATION_TABLE_FIELDS[table] if field not in excluded
        )
        legacy_rows = list(legacy_tables[table])
        streaming_rows = list(streaming_tables[table])
        legacy_digest = _table_sha256(legacy_rows, fields)
        streaming_digest = _table_sha256(streaming_rows, fields)
        table_exact = bool(
            len(legacy_rows) == len(streaming_rows)
            and legacy_digest == streaming_digest
        )
        exact = exact and table_exact
        tables[table] = {
            "exact": table_exact,
            "legacy_row_count": len(legacy_rows),
            "streaming_row_count": len(streaming_rows),
            "legacy_ordered_sha256": legacy_digest,
            "streaming_ordered_sha256": streaming_digest,
        }
    return {
        "exact": exact,
        "table_count": len(GATE_REQUIRED_TABLES),
        "tables": tables,
        "formal_gate_executed": False,
        "formal_gate_reason": (
            "a bounded checkpoint/scene subset is incomplete formal evidence; only the "
            "exact ordered inputs consumed by _evaluation_gate are compared"
        ),
    }


def _bind_complete_tables(
    tables: Mapping[str, Iterable[Mapping[str, object]]], evidence: dict[str, object]
) -> dict[str, list[dict[str, object]]]:
    if set(tables) != set(EVALUATION_TABLE_FIELDS):
        raise RuntimeError("evaluation subset produced an incomplete table inventory")
    bound = {}
    for table, fields in EVALUATION_TABLE_FIELDS.items():
        rows = bind_rows(tables[table], evidence)
        canonical = []
        for row in rows:
            if set(row) != set(fields):
                missing = sorted(set(fields) - set(row))
                extra = sorted(set(row) - set(fields))
                raise RuntimeError(
                    f"evaluation table {table} fields mismatch: missing={missing}, extra={extra}"
                )
            canonical.append({field: row[field] for field in fields})
        bound[table] = canonical
    return bound


def _empty_tables() -> dict[str, list[dict[str, object]]]:
    return {table: [] for table in EVALUATION_TABLE_FIELDS}


def _extend_tables(
    destination: dict[str, list[dict[str, object]]],
    source: Mapping[str, Iterable[Mapping[str, object]]],
) -> None:
    if set(source) != set(destination):
        raise RuntimeError("evaluation table inventory changed during comparison")
    for table in destination:
        destination[table].extend(dict(row) for row in source[table])


def _evaluate_legacy_merged_checkpoint(
    model,
    probes,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    scenes: Iterable[int],
    *,
    seed: int,
    arm: str,
    batch_size: int,
) -> dict[str, list[dict[str, object]]]:
    """Execute the legacy all-scenes array path for one bounded checkpoint."""

    scene_array = np.asarray(tuple(int(scene) for scene in scenes), dtype=np.int64)
    routed = route_dataset(
        dataset,
        teacher,
        config,
        scene_array,
        normalization=route_normalization,
    )
    evaluated = legacy._compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        scene_array,
        active_only=False,
        route_normalization=route_normalization,
        routed=routed,
        batch_size=batch_size,
    )
    probabilities = predict_binary_probe(
        probes.compatibility_probe, evaluated["features"]
    )
    cgs_rows = []
    shortcut_rows = []
    route_rows = []
    effect_rows = []
    compatibility_effect_rows = legacy._compatibility_effect_rows(
        seed, arm, evaluated, probabilities
    )
    for bank in sorted(set(evaluated["bank_ids"].tolist())):
        bank_mask = evaluated["bank_ids"] == bank
        active_mask = bank_mask & (evaluated["routes"] == "active")
        if not np.any(active_mask):
            raise RuntimeError(f"evaluation bank {bank!r} has no active CGS quartet")
        scene = int(evaluated["scene_indices"][active_mask][0])
        evaluation_scope = legacy._evaluation_scope(dataset, scene)
        cgs_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "bank_id": bank,
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
                "canonical_bank_digest": legacy._canonical_bank_digest(dataset, scene),
                "city_id": str(evaluated["city_ids"][active_mask][0]),
                "evaluation_scope": evaluation_scope,
                "route": "active",
                "probe_family": probes.compatibility_selection["selected_family"],
                "cgs_auroc": legacy.binary_auroc(
                    evaluated["labels"][active_mask], probabilities[active_mask]
                ),
                "native_energy_auroc": legacy.binary_auroc(
                    evaluated["labels"][active_mask],
                    evaluated["native_training_scores"][active_mask],
                ),
                "native_audit_hold_auroc": legacy.binary_auroc(
                    evaluated["labels"][active_mask],
                    evaluated["native_scores"][active_mask],
                ),
                "native_probe_spearman": legacy.spearman_correlation(
                    evaluated["native_training_scores"][active_mask],
                    probabilities[active_mask],
                ),
                "native_audit_hold_probe_spearman": legacy.spearman_correlation(
                    evaluated["native_scores"][active_mask],
                    probabilities[active_mask],
                ),
                "n": int(np.sum(active_mask)),
            }
        )
        shortcut_rows.extend(
            legacy._alignment_shortcut_rows(
                seed,
                arm,
                bank,
                evaluation_scope,
                {},
                {},
                evaluated,
                active_mask,
                config,
                prepared=probes.shortcut_probes,
            )
        )
        effect_rows.extend(
            legacy._active_effect_bin_rows(
                seed,
                arm,
                bank,
                str(evaluated["city_ids"][active_mask][0]),
                evaluation_scope,
                probabilities[active_mask],
                evaluated["labels"][active_mask],
                evaluated["pair_ids"][active_mask],
                evaluated["physical_distances"][active_mask],
            )
        )
        for route in ROUTE_NAMES.tolist():
            route_mask = bank_mask & (evaluated["routes"] == route)
            common = {
                "seed": seed,
                "arm": arm,
                "bank_id": bank,
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
                "canonical_bank_digest": legacy._canonical_bank_digest(dataset, scene),
                "city_id": str(evaluated["city_ids"][bank_mask][0]),
                "evaluation_scope": evaluation_scope,
                "route": route,
            }
            if not np.any(route_mask):
                route_rows.append(
                    {
                        **common,
                        "condition_status": "MISSING",
                        "pair_count": 0,
                        "matched_minus_alternative_mean": None,
                        "matched_minus_alternative_median": None,
                        "absolute_difference_p90": None,
                        "overclassification_rate": None,
                    }
                )
                continue
            differences = legacy._paired_score_differences(
                probabilities[route_mask],
                evaluated["labels"][route_mask],
                evaluated["pair_ids"][route_mask],
            )
            margin = float(config["evaluation"]["null_score_equivalence_margin"])
            route_rows.append(
                {
                    **common,
                    "condition_status": "ASSESSED",
                    "pair_count": int(differences.size),
                    "matched_minus_alternative_mean": float(np.mean(differences)),
                    "matched_minus_alternative_median": float(np.median(differences)),
                    "absolute_difference_p90": float(
                        np.percentile(np.abs(differences), 90)
                    ),
                    "overclassification_rate": (
                        float(np.mean(np.abs(differences) > margin))
                        if route == "null"
                        else None
                    ),
                }
            )

    response_eval = legacy._response_probe_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        scene_array,
        active_only=False,
        route_normalization=route_normalization,
        routed=routed,
        batch_size=batch_size,
    )
    source_target = response_eval["source_targets"]
    prediction = source_target + predict_response_probe(
        probes.response_probe,
        response_eval["features"],
        response_eval["no_action_features"],
    )
    swap_prediction = source_target + predict_response_probe(
        probes.response_probe,
        response_eval["action_swap_features"],
        response_eval["no_action_features"],
    )
    no_action_prediction = source_target + predict_response_probe(
        probes.response_probe,
        response_eval["no_action_features"],
        response_eval["no_action_features"],
    )
    contrast_variants = {"without_map", "edit_only", "oracle_x"}
    variant_predictions = {
        name: source_target
        + predict_response_probe(
            probe,
            response_eval[f"{name}_features"],
            (
                response_eval[f"{name}_zero_action_features"]
                if name in contrast_variants
                else None
            ),
        )
        for name, probe in probes.variant_probes.items()
    }
    response_effect_rows = legacy._response_effect_rows(
        seed,
        arm,
        response_eval,
        prediction,
        swap_prediction,
        no_action_prediction,
    )
    response_rows = []
    for bank in sorted(set(response_eval["bank_ids"].tolist())):
        response_mask = (response_eval["bank_ids"] == bank) & (
            response_eval["routes"] == "active"
        )
        if not np.any(response_mask):
            raise RuntimeError(f"evaluation bank {bank!r} has no active response patches")
        scene = int(response_eval["scene_indices"][response_mask][0])
        exact_swap_mask = response_mask & (
            response_eval["wrong_action_match_status"] == "exact"
        )
        denominator = max(
            float(np.sum(response_eval["targets"][response_mask] ** 2)), 1e-12
        )
        exact_denominator = max(
            float(np.sum(response_eval["targets"][exact_swap_mask] ** 2)), 1e-12
        )
        native = legacy._native_mask_cover_metrics(
            model,
            dataset,
            teacher,
            config,
            normalization,
            scene_array,
            bank,
            route_normalization=route_normalization,
            routed=routed,
            batch_size=batch_size,
        )
        response_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "bank_id": bank,
                "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
                "canonical_bank_digest": legacy._canonical_bank_digest(dataset, scene),
                "city_id": str(response_eval["city_ids"][response_mask][0]),
                "evaluation_scope": legacy._evaluation_scope(dataset, scene),
                "unified_response_probe_active_patch_nmse": float(
                    np.sum(
                        (
                            prediction[response_mask]
                            - response_eval["targets"][response_mask]
                        )
                        ** 2
                    )
                    / denominator
                ),
                "probe_copy_active_patch_nmse": float(
                    np.sum(
                        (
                            response_eval["source_targets"][response_mask]
                            - response_eval["targets"][response_mask]
                        )
                        ** 2
                    )
                    / denominator
                ),
                "unified_response_probe_action_swap_exact_patch_nmse": (
                    float(
                        np.sum(
                            (
                                prediction[exact_swap_mask]
                                - response_eval["targets"][exact_swap_mask]
                            )
                            ** 2
                        )
                        / exact_denominator
                    )
                    if np.any(exact_swap_mask)
                    else None
                ),
                "probe_action_swap_active_patch_nmse": (
                    float(
                        np.sum(
                            (
                                swap_prediction[exact_swap_mask]
                                - response_eval["targets"][exact_swap_mask]
                            )
                            ** 2
                        )
                        / exact_denominator
                    )
                    if np.any(exact_swap_mask)
                    else None
                ),
                "probe_action_swap_exact_count": int(np.sum(exact_swap_mask)),
                "probe_action_swap_fallback_count": int(
                    np.sum(
                        response_mask
                        & (response_eval["wrong_action_match_status"] == "fallback")
                    )
                ),
                "probe_action_swap_failed_count": int(
                    np.sum(
                        response_mask
                        & (response_eval["wrong_action_match_status"] == "failed")
                    )
                ),
                "probe_action_swap_exact_fraction": float(
                    np.mean(
                        response_eval["wrong_action_match_status"][response_mask]
                        == "exact"
                    )
                ),
                **{
                    f"probe_{name}_active_patch_nmse": float(
                        np.sum(
                            (
                                variant_prediction[response_mask]
                                - response_eval["targets"][response_mask]
                            )
                            ** 2
                        )
                        / denominator
                    )
                    for name, variant_prediction in variant_predictions.items()
                },
                **native,
                "n_active_patches": int(np.sum(response_mask)),
            }
        )
    return {
        "cgs_per_bank.csv": cgs_rows,
        "alignment_shortcut_baselines.csv": shortcut_rows,
        "compatibility_route_distributions.csv": route_rows,
        "cgs_active_effect_bins.csv": effect_rows,
        "compatibility_pair_effects.csv": compatibility_effect_rows,
        "response_pair_effects.csv": response_effect_rows,
        "response_per_bank.csv": response_rows,
    }


def _evaluate_streaming_checkpoint(
    model,
    probes,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    scenes: Iterable[int],
    *,
    seed: int,
    arm: str,
    batch_size: int,
) -> dict[str, list[dict[str, object]]]:
    output_scenes = tuple(int(scene) for scene in scenes)
    # Validate the frozen role/order contract, but retain no matrix spanning
    # multiple scenes.  Each payload is consumed and released before the next
    # scene is prepared so restart and memory behavior are scene-bounded.
    _frozen_subset_evaluation_scenes(dataset, output_scenes)
    tables = {table: [] for table in SCENE_TABLES}
    for scene in output_scenes:
        rows = streaming._prepare_and_evaluate_scene(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene=int(scene),
            seed=seed,
            arm=arm,
            batch_size=batch_size,
        )
        if set(rows) != set(SCENE_TABLES):
            raise RuntimeError("streaming scene returned an incomplete table inventory")
        for table in SCENE_TABLES:
            tables[table].extend(rows[table])
    return tables


def _manifested_file(stage_root: Path, relative: str) -> dict[str, object]:
    manifest_path = stage_root / "manifest.json"
    manifest = read_strict_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "csi-pairs-formal-stage-manifest-v2.1-v6"
        or not isinstance(manifest.get("files"), list)
    ):
        raise RuntimeError(f"invalid legacy stage manifest: {manifest_path}")
    matches = [
        row
        for row in manifest["files"]
        if isinstance(row, dict) and row.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"legacy stage manifest does not uniquely bind {relative}")
    path = (stage_root / relative).resolve()
    if stage_root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"legacy stage input is missing, linked, or escapes: {path}")
    row = matches[0]
    if row.get("bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
        raise RuntimeError(f"legacy manifested input changed: {path}")
    return dict(row)


def _validate_legacy_inputs(
    config: dict,
    dataset,
    legacy_root: Path,
) -> tuple[dict, dict, list[dict], dict[str, Path]]:
    qualification_root = legacy_root / "qualification"
    factorial_root = legacy_root / "factorial"
    if legacy_root.is_symlink() or not legacy_root.is_dir():
        raise RuntimeError("legacy run root must be a regular directory")
    for stage in (qualification_root, factorial_root):
        if stage.is_symlink() or not stage.is_dir():
            raise RuntimeError(f"legacy stage must be a regular directory: {stage}")
    qualification_path = qualification_root / "gate.json"
    factorial_path = factorial_root / "gate.json"
    index_path = factorial_root / "checkpoint_index.json"
    qualification_gate = read_strict_json(qualification_path)
    factorial_gate = read_strict_json(factorial_path)
    checkpoint_index = read_strict_json(index_path)
    if (
        not isinstance(qualification_gate, dict)
        or qualification_gate.get("schema_version") != QUALIFICATION_SCHEMA
        or qualification_gate.get("passed") is not True
        or qualification_gate.get("fixture") is not False
        or qualification_gate.get("scientific_use") != "FORMAL_EXPERIMENT_ALLOWED"
    ):
        raise RuntimeError("legacy qualification is not an allowed formal-data gate")
    if (
        not isinstance(factorial_gate, dict)
        or factorial_gate.get("schema_version") != FACTORIAL_SCHEMA
        or factorial_gate.get("fixture") is not False
    ):
        raise RuntimeError("legacy factorial gate is not a formal-data V6 gate")
    if (
        not isinstance(checkpoint_index, dict)
        or checkpoint_index.get("schema_version")
        != "csi-pairs-formal-checkpoint-index-v2.1-v6"
        or checkpoint_index.get("fixture") is not False
    ):
        raise RuntimeError("legacy checkpoint index is not a formal-data V6 index")

    dataset_digest = sha256_file(dataset.source_path)
    configuration_digest = config_sha256(config)
    for name, payload in (
        ("qualification", qualification_gate),
        ("factorial", factorial_gate),
        ("checkpoint index", checkpoint_index),
    ):
        if payload.get("dataset_sha256") != dataset_digest:
            raise RuntimeError(f"legacy {name} dataset SHA mismatch")
        if payload.get("config_sha256") != configuration_digest:
            raise RuntimeError(f"legacy {name} config SHA mismatch")
    _manifested_file(qualification_root, "gate.json")
    _manifested_file(factorial_root, "gate.json")
    _manifested_file(factorial_root, "checkpoint_index.json")

    teacher_path = Path(str(qualification_gate.get("teacher_checkpoint", ""))).resolve()
    if (
        qualification_root.resolve() not in teacher_path.parents
        or teacher_path.is_symlink()
        or not teacher_path.is_file()
    ):
        raise RuntimeError("legacy teacher checkpoint is missing or escapes qualification")
    teacher_relative = teacher_path.relative_to(qualification_root.resolve()).as_posix()
    _manifested_file(qualification_root, teacher_relative)
    if sha256_file(teacher_path) != qualification_gate.get("teacher_checkpoint_sha256"):
        raise RuntimeError("legacy teacher checkpoint SHA mismatch")

    rows = checkpoint_index.get("checkpoints")
    if not isinstance(rows, list):
        raise RuntimeError("legacy checkpoint inventory is not a list")
    expected_cells = {
        (int(seed), str(arm))
        for seed in config["seeds"]
        for arm in config["factorial"]["arms"]
    }
    actual_cells = {
        (int(row.get("seed", -1)), str(row.get("arm", "")))
        for row in rows
        if isinstance(row, dict)
    }
    if actual_cells != expected_cells or len(rows) != len(expected_cells):
        raise RuntimeError("legacy checkpoint inventory does not cover the formal design")
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("teacher_checkpoint_sha256")
            != qualification_gate["teacher_checkpoint_sha256"]
            or not _valid_sha256(row.get("sha256"))
        ):
            raise RuntimeError("legacy checkpoint inventory identity is malformed")
    return (
        qualification_gate,
        factorial_gate,
        [dict(row) for row in rows],
        {
            "qualification_gate": qualification_path,
            "qualification_manifest": qualification_root / "manifest.json",
            "factorial_gate": factorial_path,
            "factorial_manifest": factorial_root / "manifest.json",
            "checkpoint_index": index_path,
            "teacher_checkpoint": teacher_path,
        },
    )


def _selected_checkpoint_records(
    factorial_root: Path, rows: Iterable[Mapping[str, object]]
) -> tuple[dict[str, object], ...]:
    records = []
    root = factorial_root.resolve()
    for row in rows:
        relative = str(row["path"])
        path = (root / relative).resolve()
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise RuntimeError("selected legacy checkpoint is missing, linked, or escapes")
        _manifested_file(root, relative)
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise RuntimeError("selected legacy checkpoint SHA mismatch")
        records.append(
            {
                "seed": int(row["seed"]),
                "arm": str(row["arm"]),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "recorded_source_tree_sha256": row.get("source_tree_sha256"),
                "teacher_checkpoint_sha256": row["teacher_checkpoint_sha256"],
            }
        )
    return tuple(records)


def _input_record(
    path: Path, *, identity_sha256: str | None = None
) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    record = {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }
    if identity_sha256 is not None:
        record["identity_sha256"] = identity_sha256
    return record


def _assert_inputs_unchanged(records: Iterable[Mapping[str, object]]) -> None:
    for record in records:
        path = Path(str(record["path"]))
        stat = path.stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"authenticated input changed during comparison: {path}")


def _read_only_audit(
    records: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    audit = []
    for before in records:
        after = _input_record(Path(str(before["path"])))
        unchanged = all(
            before.get(field) == after.get(field)
            for field in ("path", "bytes", "mtime_ns", "sha256")
        )
        audit.append(
            {
                "path": before["path"],
                "before": dict(before),
                "after": after,
                "unchanged": unchanged,
            }
        )
    if not all(row["unchanged"] for row in audit):
        raise RuntimeError("an authenticated input changed during subset comparison")
    return tuple(audit)


def _implementation_sources(evidence: Mapping[str, object]) -> dict[str, object]:
    files = {}
    paths = [Path(__file__).resolve(), Path(legacy.__file__).resolve()]
    if streaming is not None:
        paths.append(Path(streaming.__file__).resolve())
    for path in paths:
        files[path.name] = {"path": str(path), "sha256": sha256_file(path)}
    return {
        "comparator_source_tree_sha256": evidence["source_tree_sha256"],
        "files": files,
    }


def _git_output(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _git_tracked_files(git_root: Path) -> set[Path]:
    completed = subprocess.run(
        ("git", "-C", str(git_root), "ls-files", "-z"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        (git_root / os.fsdecode(relative)).resolve()
        for relative in completed.stdout.split(b"\0")
        if relative
    }


def _source_file_record(path: Path, tracked_files: set[Path]) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"worker source file is missing or linked: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "git_tracked": resolved in tracked_files,
    }


def _untracked_scientific_files(
    source_root: Path, tracked_files: set[Path]
) -> list[str]:
    formal_root = source_root / "formal_v2"
    untracked = []
    for path in formal_root.rglob("*"):
        relative = path.relative_to(source_root)
        if (
            not path.is_file()
            or path.suffix not in SOURCE_IDENTITY_SUFFIXES
            or "__pycache__" in relative.parts
            or any(
                part.startswith(SOURCE_IDENTITY_EXCLUDED_PREFIXES)
                for part in relative.parts
            )
        ):
            continue
        if path.resolve() not in tracked_files:
            untracked.append(relative.as_posix())
    return sorted(untracked)


def source_identity(
    source_root: str | Path,
    *,
    require_streaming: bool,
) -> dict[str, object]:
    supplied_root = Path(source_root)
    if supplied_root.is_symlink():
        raise RuntimeError("worker source root must not be a symbolic link")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise RuntimeError("worker source root must be a regular directory")
    git_root = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked_status = _git_output(root, "status", "--porcelain", "--untracked-files=no")
    tracked_files = _git_tracked_files(git_root)
    untracked_scientific = _untracked_scientific_files(root, tracked_files)
    formal_evaluation = root / "formal_v2" / "formal_evaluation.py"
    streaming_path = root / "formal_v2" / "formal_evaluation_streaming.py"
    if formal_evaluation.is_symlink() or not formal_evaluation.is_file():
        raise RuntimeError("worker source has no regular formal_evaluation.py")
    if require_streaming and (
        streaming_path.is_symlink() or not streaming_path.is_file()
    ):
        raise RuntimeError("candidate worker source has no streaming implementation")
    comparator_path = Path(__file__).resolve()
    comparator_git_root = Path(
        _git_output(comparator_path.parent, "rev-parse", "--show-toplevel")
    ).resolve()
    comparator_tracked_files = (
        tracked_files
        if comparator_git_root == git_root
        else _git_tracked_files(comparator_git_root)
    )
    files = {
        "formal_evaluation_subset_compare.py": _source_file_record(
            comparator_path, comparator_tracked_files
        ),
        "formal_evaluation.py": _source_file_record(formal_evaluation, tracked_files),
    }
    if streaming_path.is_file() and not streaming_path.is_symlink():
        files["formal_evaluation_streaming.py"] = _source_file_record(
            streaming_path, tracked_files
        )
    diff = subprocess.run(
        ("git", "-C", str(root), "diff", "--binary", "HEAD", "--"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "source_root": str(root),
        "git_root": str(git_root),
        "git_commit": commit,
        "tracked_tree_clean": tracked_status == "" and not untracked_scientific,
        "tracked_status": tracked_status.splitlines(),
        "untracked_scientific_paths": untracked_scientific,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": files,
    }


def _validate_loaded_source(
    identity: Mapping[str, object],
    *,
    expected_commit: str,
    expected_evaluation_sha256: str,
    require_streaming: bool,
) -> None:
    root = Path(str(identity["source_root"])).resolve()
    loaded_evaluation = Path(legacy.__file__).resolve()
    expected_evaluation = root / "formal_v2" / "formal_evaluation.py"
    if loaded_evaluation != expected_evaluation:
        raise RuntimeError("worker imported formal_evaluation from the wrong source root")
    if identity.get("git_commit") != expected_commit:
        raise RuntimeError("worker source commit does not match its request")
    evaluation_record = identity["files"]["formal_evaluation.py"]
    if evaluation_record.get("git_tracked") is not True:
        raise RuntimeError("worker formal_evaluation.py is not committed source")
    if evaluation_record.get("sha256") != expected_evaluation_sha256:
        raise RuntimeError("worker formal_evaluation.py SHA does not match its request")
    comparator_record = identity["files"].get("formal_evaluation_subset_compare.py")
    if (
        not isinstance(comparator_record, dict)
        or Path(str(comparator_record.get("path"))).resolve() != Path(__file__).resolve()
        or comparator_record.get("git_tracked") is not True
        or comparator_record.get("bytes") != Path(__file__).resolve().stat().st_size
        or comparator_record.get("sha256") != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("worker comparator harness is not committed source")
    if require_streaming:
        if streaming is None:
            raise RuntimeError("candidate worker did not import streaming code")
        expected_streaming = root / "formal_v2" / "formal_evaluation_streaming.py"
        if Path(streaming.__file__).resolve() != expected_streaming:
            raise RuntimeError("worker imported streaming code from the wrong source root")
        if (
            identity["files"]["formal_evaluation_streaming.py"].get("git_tracked")
            is not True
        ):
            raise RuntimeError("worker streaming implementation is not committed source")


def _selection_sha256(
    scenes: Iterable[Mapping[str, object]],
    checkpoints: Iterable[Mapping[str, object]],
) -> str:
    payload = {
        "scene_rule": SELECTION_RULE,
        "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
        "scenes": list(scenes),
        "checkpoints": [
            {
                "seed": int(row["seed"]),
                "arm": str(row["arm"]),
                "sha256": str(row["sha256"]),
            }
            for row in checkpoints
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _worker_input_bindings(
    config_file: Path,
    dataset_file: Path,
    config: Mapping[str, object],
    upstream_paths: Mapping[str, Path],
    checkpoint_records: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = [
        _input_record(config_file, identity_sha256=config_sha256(dict(config))),
        _input_record(dataset_file, identity_sha256=sha256_file(dataset_file)),
        *(_input_record(path) for path in upstream_paths.values()),
        *(
            _input_record(
                Path(str(record["path"])),
                identity_sha256=str(record["sha256"]),
            )
            for record in checkpoint_records
        ),
    ]
    by_path = {str(record["path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("worker authenticated input inventory contains duplicate paths")
    return (
        {
            "config_sha256": config_sha256(dict(config)),
            "dataset_sha256": sha256_file(dataset_file),
            "files": [by_path[path] for path in sorted(by_path)],
        },
        records,
    )


def _directory_tree_snapshot(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"read-only snapshot root is missing or linked: {path}")
    root = path.resolve()
    root_stat = root.stat()
    entries = []
    total_file_bytes = 0
    for entry in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise RuntimeError(f"read-only snapshot rejects symbolic links: {entry}")
        entry_stat = entry.stat()
        if entry.is_dir():
            record = {
                "path": relative,
                "kind": "directory",
                "mode": entry_stat.st_mode,
                "mtime_ns": entry_stat.st_mtime_ns,
            }
        elif entry.is_file():
            total_file_bytes += entry_stat.st_size
            record = {
                "path": relative,
                "kind": "file",
                "mode": entry_stat.st_mode,
                "bytes": entry_stat.st_size,
                "mtime_ns": entry_stat.st_mtime_ns,
                "sha256": sha256_file(entry),
            }
        else:
            raise RuntimeError(f"read-only snapshot rejects special files: {entry}")
        entries.append(record)
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "root": str(root),
        "root_mode": root_stat.st_mode,
        "root_mtime_ns": root_stat.st_mtime_ns,
        "entry_count": len(entries),
        "total_file_bytes": total_file_bytes,
        "entries_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _run_frozen_original_oracle(
    config,
    dataset,
    legacy_root: Path,
    qualification_gate: dict,
    factorial_gate: dict,
    selected_scenes: Iterable[Mapping[str, object]],
    selected_checkpoints: Iterable[Mapping[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Capture the exact frozen run body while bounding only its outer inventories."""

    if streaming is not None:
        raise RuntimeError("frozen oracle worker unexpectedly imported streaming code")
    evaluation_dir = legacy_root / "evaluation"
    if evaluation_dir.is_symlink() or not evaluation_dir.is_dir():
        raise RuntimeError(
            "read-only frozen oracle requires the legacy evaluation directory to exist"
        )
    evaluation_before = _directory_tree_snapshot(evaluation_dir)
    from . import formal_data_verification

    required_roles = (
        "source_encoder_train",
        "source_probe_train",
        "source_probe_selection",
        "source_final_unseen_bank",
        "target",
    )
    formal_data_verification.require_verified_roles_from_root(
        legacy_root, config, dataset, required_roles
    )
    selected_by_role = {
        role: np.asarray(
            [
                int(row["scene_index"])
                for row in selected_scenes
                if str(row["scene_role"]) == role
            ],
            dtype=np.int64,
        )
        for role in ("source_final_unseen_bank", "target")
    }
    if any(values.size == 0 for values in selected_by_role.values()):
        raise RuntimeError("frozen oracle subset must cover source unseen and target")
    selected_cells = {
        (int(row["seed"]), str(row["arm"])) for row in selected_checkpoints
    }
    original_indices = dataset.indices_for_role
    original_validate_index = legacy._validate_checkpoint_index
    original_write_csv = legacy.write_csv
    original_write_json = legacy.write_json
    original_evaluation_gate = legacy._evaluation_gate
    original_verify_roles = formal_data_verification.require_verified_roles_from_root
    captured: dict[str, list[dict[str, object]]] = {}

    def restricted_indices(role):
        if role in selected_by_role:
            return selected_by_role[role].copy()
        return original_indices(role)

    def restricted_checkpoint_index(index, selected_config, selected_dataset, gate):
        authenticated = original_validate_index(
            index, selected_config, selected_dataset, gate
        )
        selected = [
            row
            for row in authenticated
            if (int(row["seed"]), str(row["arm"])) in selected_cells
        ]
        if len(selected) != len(selected_cells):
            raise RuntimeError("frozen oracle checkpoint subset is incomplete")
        return selected

    def capture_csv(path, rows):
        name = Path(path).name
        if name in captured:
            raise RuntimeError(f"frozen oracle wrote table twice: {name}")
        captured[name] = [dict(row) for row in rows]

    try:
        dataset.indices_for_role = restricted_indices
        legacy._validate_checkpoint_index = restricted_checkpoint_index
        legacy.write_csv = capture_csv
        legacy.write_json = lambda _path, _payload: None
        legacy._evaluation_gate = lambda *_args, **_kwargs: {
            "status": "SUBSET_GATE_NOT_EXECUTED"
        }
        formal_data_verification.require_verified_roles_from_root = (
            lambda *_args, **_kwargs: None
        )
        legacy.run_formal_evaluation(
            config,
            dataset,
            legacy_root,
            qualification_gate,
            factorial_gate,
        )
    finally:
        dataset.indices_for_role = original_indices
        legacy._validate_checkpoint_index = original_validate_index
        legacy.write_csv = original_write_csv
        legacy.write_json = original_write_json
        legacy._evaluation_gate = original_evaluation_gate
        formal_data_verification.require_verified_roles_from_root = original_verify_roles
    evaluation_after = _directory_tree_snapshot(evaluation_dir)
    if evaluation_before != evaluation_after:
        raise RuntimeError("frozen oracle changed the legacy evaluation directory")
    if set(captured) != set(EVALUATION_TABLE_FIELDS):
        raise RuntimeError("frozen original run did not emit the complete nine tables")
    if any(not captured[table] for table in EVALUATION_TABLE_FIELDS):
        raise RuntimeError("frozen original run emitted an empty subset table")
    return captured, {
        "root": str(evaluation_dir.resolve()),
        "before": evaluation_before,
        "after": evaluation_after,
        "unchanged": True,
    }


def _run_candidate_tables(
    config,
    dataset,
    legacy_root: Path,
    qualification_gate: dict,
    upstream_paths: Mapping[str, Path],
    selected_scenes: Iterable[Mapping[str, object]],
    selected_checkpoints: Iterable[Mapping[str, object]],
    evidence: dict[str, object],
    *,
    execution_device,
    batch_size: int,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
]:
    if streaming is None:
        raise RuntimeError("candidate worker has no streaming implementation")
    if execution_device.type == "cuda":
        torch.cuda.set_device(execution_device)
        torch.cuda.reset_peak_memory_stats(execution_device)
    teacher = load_teacher_bundle(
        upstream_paths["teacher_checkpoint"], config, device=execution_device
    )
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )
    scene_indices = tuple(int(row["scene_index"]) for row in selected_scenes)
    frozen_scene_indices = _frozen_subset_evaluation_scenes(dataset, scene_indices)
    merged_tables = _empty_tables()
    per_scene_tables = _empty_tables()
    for checkpoint in selected_checkpoints:
        seed = int(checkpoint["seed"])
        arm = str(checkpoint["arm"])
        model = legacy._load_model(
            legacy_root / "factorial",
            checkpoint,
            qualification_gate,
            config,
            dataset,
            device=execution_device,
        )
        probes = streaming._fit_probe_bundle(
            model,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            seed=seed,
            batch_size=batch_size,
        )
        contracts = streaming._probe_contract_rows(
            probes, config, seed=seed, arm=arm
        )
        for destination in (merged_tables, per_scene_tables):
            for table, rows in contracts.items():
                destination[table].extend(dict(row) for row in rows)
        merged = _evaluate_legacy_merged_checkpoint(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            frozen_scene_indices,
            seed=seed,
            arm=arm,
            batch_size=batch_size,
        )
        per_scene = _evaluate_streaming_checkpoint(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene_indices,
            seed=seed,
            arm=arm,
            batch_size=batch_size,
        )
        for table in SCENE_TABLES:
            merged_tables[table].extend(merged[table])
            per_scene_tables[table].extend(per_scene[table])
        del model, probes, contracts, merged, per_scene
        gc.collect()
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        _bind_complete_tables(merged_tables, evidence),
        _bind_complete_tables(per_scene_tables, evidence),
    )


def run_worker_request(request_path: str | Path) -> dict[str, object]:
    started = time.monotonic()
    request_file = Path(request_path).resolve()
    request = read_strict_json(request_file)
    if not isinstance(request, dict) or request.get("schema_version") != WORKER_REQUEST_SCHEMA:
        raise RuntimeError("subset worker request schema mismatch")
    role = request.get("role")
    if role not in {"frozen_legacy", "streaming_candidate"}:
        raise RuntimeError("subset worker role is invalid")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or request.get("device") != "cuda:0"
    ):
        raise RuntimeError(
            "formal subset worker requires CUDA_VISIBLE_DEVICES=0 and device cuda:0"
        )
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("formal subset worker requires isolated python -I -B")
    if not _BOUND_SOURCE_ROOT:
        raise RuntimeError("subset worker has no source-root process binding")
    source_root = Path(_BOUND_SOURCE_ROOT).resolve()
    require_streaming = role == "streaming_candidate"
    identity = source_identity(source_root, require_streaming=require_streaming)
    _validate_loaded_source(
        identity,
        expected_commit=str(request["expected_git_commit"]),
        expected_evaluation_sha256=str(request["expected_evaluation_sha256"]),
        require_streaming=require_streaming,
    )
    if not identity["tracked_tree_clean"]:
        raise RuntimeError("formal subset worker source has tracked changes")
    config_file = Path(str(request["config_path"])).resolve()
    dataset_file = Path(str(request["dataset_path"])).resolve()
    legacy_root = Path(str(request["legacy_run_root"])).resolve()
    fragment_path = validate_report_destination(
        str(request["fragment_path"]),
        legacy_root,
        source_root=source_root,
        protected_files=(config_file, dataset_file, request_file, Path(__file__)),
    )
    configure_reproducible_runtime()
    config = load_formal_config(config_file)
    dataset = FormalDataset.load(dataset_file)
    if dataset.is_fixture:
        raise ValueError("dual-source formal worker rejects synthetic/fixture datasets")
    execution_device = resolve_execution_device(dataset, str(request["device"]))
    os.environ["CSI_PAIRS_DEVICE"] = str(execution_device)
    (
        qualification_gate,
        factorial_gate,
        checkpoint_rows,
        upstream_paths,
    ) = _validate_legacy_inputs(config, dataset, legacy_root)
    selected_scenes = select_real_evaluation_subset(
        dataset,
        source_scenes=int(request["source_scenes"]),
        target_scenes_per_city=int(request["target_scenes_per_city"]),
    )
    selected_checkpoints = select_legacy_checkpoints(
        checkpoint_rows,
        config["factorial"]["arms"],
        checkpoint_count=int(request["checkpoint_count"]),
        checkpoint_arm=request.get("checkpoint_arm"),
    )
    checkpoint_records = _selected_checkpoint_records(
        legacy_root / "factorial", selected_checkpoints
    )
    input_bindings, input_records = _worker_input_bindings(
        config_file,
        dataset_file,
        config,
        upstream_paths,
        checkpoint_records,
    )
    selection = {
        "scene_rule": SELECTION_RULE,
        "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
        "scenes_in_execution_order": list(selected_scenes),
        "checkpoints_in_execution_order": [
            {
                "seed": int(row["seed"]),
                "arm": str(row["arm"]),
                "sha256": str(row["sha256"]),
            }
            for row in selected_checkpoints
        ],
        "sha256": _selection_sha256(selected_scenes, selected_checkpoints),
    }
    evidence = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
    if role == "frozen_legacy":
        tables, legacy_evaluation_audit = _run_frozen_original_oracle(
            config,
            dataset,
            legacy_root,
            qualification_gate,
            factorial_gate,
            selected_scenes,
            selected_checkpoints,
        )
        tables = _bind_complete_tables(
            {
                table: [
                    {
                        field: row[field]
                        for field in EVALUATION_TABLE_FIELDS[table]
                        if field in row
                    }
                    for row in rows
                ]
                for table, rows in tables.items()
            },
            evidence,
        )
        same_source_merged_tables = None
    else:
        legacy_evaluation_audit = None
        same_source_merged_tables, tables = _run_candidate_tables(
            config,
            dataset,
            legacy_root,
            qualification_gate,
            upstream_paths,
            selected_scenes,
            selected_checkpoints,
            evidence,
            execution_device=execution_device,
            batch_size=int(request["batch_size"]),
        )
    read_only = _read_only_audit(input_records)
    identity["source_tree_sha256"] = evidence["source_tree_sha256"]
    identity["runtime_provenance_sha256"] = evidence["runtime_provenance_sha256"]
    request_record = _input_record(request_file)
    fragment = {
        "schema_version": WORKER_FRAGMENT_SCHEMA,
        "role": role,
        "created_utc": _utc_now(),
        "request": {
            "path": request_record["path"],
            "bytes": request_record["bytes"],
            "sha256": request_record["sha256"],
        },
        "source_identity": identity,
        "runtime_provenance": evidence["runtime_provenance"],
        "input_bindings": input_bindings,
        "selection": selection,
        "tables": tables,
        "same_source_merged_tables": same_source_merged_tables,
        "read_only": {
            "passed": all(row["unchanged"] for row in read_only),
            "files": list(read_only),
            "legacy_evaluation": legacy_evaluation_audit,
        },
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "maximum_resident_set_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "maximum_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(execution_device))
                if execution_device.type == "cuda"
                else 0
            ),
        },
    }
    write_report_atomic(fragment_path, fragment)
    return fragment


def _runtime_digest(runtime: object) -> str:
    encoded = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fragment_tables(fragment: Mapping[str, object]) -> None:
    tables = fragment.get("tables")
    if not isinstance(tables, dict) or set(tables) != set(EVALUATION_TABLE_FIELDS):
        raise RuntimeError("worker fragment does not contain exactly nine tables")
    source = fragment.get("source_identity")
    inputs = fragment.get("input_bindings")
    runtime = fragment.get("runtime_provenance")
    if not isinstance(source, dict) or not isinstance(inputs, dict) or not isinstance(runtime, dict):
        raise RuntimeError("worker fragment identity bindings are malformed")
    runtime_digest = _runtime_digest(runtime)
    if (
        source.get("runtime_provenance_sha256") != runtime_digest
        or not _valid_sha256(source.get("source_tree_sha256"))
        or runtime.get("source_tree_sha256") != source.get("source_tree_sha256")
        or not _valid_sha256(runtime.get("requirements_lock_sha256"))
    ):
        raise RuntimeError("worker runtime provenance digest is invalid")
    if (
        not _valid_sha256(inputs.get("dataset_sha256"))
        or not _valid_sha256(inputs.get("config_sha256"))
        or not isinstance(inputs.get("files"), list)
        or not inputs["files"]
    ):
        raise RuntimeError("worker formal input bindings are invalid")
    source_files = source.get("files")
    if not isinstance(source_files, dict) or not source_files:
        raise RuntimeError("worker source file bindings are missing")
    for name, record in source_files.items():
        if not isinstance(name, str) or not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "git_tracked",
        }:
            raise RuntimeError("worker source file binding fields are invalid")
        source_path = Path(str(record["path"]))
        if (
            not source_path.is_absolute()
            or source_path.is_symlink()
            or not source_path.is_file()
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or source_path.stat().st_size != record["bytes"]
            or not _valid_sha256(record["sha256"])
            or sha256_file(source_path) != record["sha256"]
            or record["git_tracked"] is not True
        ):
            raise RuntimeError(f"worker source file binding is invalid: {name}")
    evidence = {
        "dataset_sha256": inputs.get("dataset_sha256"),
        "config_sha256": inputs.get("config_sha256"),
        "fixture": False,
        "scientific_use": "CANDIDATE_NOT_CLAIM",
        "source_tree_sha256": source.get("source_tree_sha256"),
        "requirements_lock_sha256": runtime.get("requirements_lock_sha256"),
        "runtime_provenance_sha256": runtime_digest,
    }
    for table, fields in EVALUATION_TABLE_FIELDS.items():
        rows = tables[table]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"worker fragment table is empty: {table}")
        primary_keys = []
        for row in rows:
            if not isinstance(row, dict) or tuple(row) != fields:
                raise RuntimeError(f"worker fragment table fields/order mismatch: {table}")
            for field, expected in evidence.items():
                if row.get(field) != expected:
                    raise RuntimeError(
                        f"worker fragment row evidence mismatch: {table}/{field}"
                    )
            primary_keys.append(tuple(row[field] for field in TABLE_PRIMARY_KEYS[table]))
        if len(primary_keys) != len(set(primary_keys)):
            raise RuntimeError(f"worker fragment primary key is not unique: {table}")


def validate_worker_pair(
    frozen_fragment: Mapping[str, object],
    candidate_fragment: Mapping[str, object],
    *,
    expected_frozen_root: str | Path,
    expected_candidate_root: str | Path,
    expected_frozen_commit: str,
    expected_candidate_commit: str,
    expected_frozen_evaluation_sha256: str,
) -> None:
    for fragment, role in (
        (frozen_fragment, "frozen_legacy"),
        (candidate_fragment, "streaming_candidate"),
    ):
        if (
            fragment.get("schema_version") != WORKER_FRAGMENT_SCHEMA
            or fragment.get("role") != role
        ):
            raise RuntimeError(f"{role} worker fragment identity is invalid")
        request_binding = fragment.get("request")
        if not isinstance(request_binding, dict) or set(request_binding) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise RuntimeError(f"{role} worker request binding is invalid")
        request_file = Path(str(request_binding["path"]))
        if (
            not request_file.is_absolute()
            or request_file.is_symlink()
            or not request_file.is_file()
            or type(request_binding["bytes"]) is not int
            or request_file.stat().st_size != request_binding["bytes"]
            or not _valid_sha256(request_binding["sha256"])
            or sha256_file(request_file) != request_binding["sha256"]
        ):
            raise RuntimeError(f"{role} worker request binding changed")
        resources = fragment.get("resources")
        elapsed = (
            float(resources.get("elapsed_seconds", 0.0))
            if isinstance(resources, dict)
            else 0.0
        )
        if (
            not isinstance(resources, dict)
            or not math.isfinite(elapsed)
            or elapsed <= 0.0
            or int(resources.get("maximum_resident_set_bytes", 0)) <= 0
            or int(resources.get("maximum_cuda_allocated_bytes", 0)) <= 0
        ):
            raise RuntimeError(f"{role} worker resource evidence is incomplete")
        read_only = fragment.get("read_only")
        if (
            not isinstance(read_only, dict)
            or set(read_only) != {"passed", "files", "legacy_evaluation"}
            or read_only.get("passed") is not True
            or not isinstance(read_only.get("files"), list)
            or not read_only["files"]
        ):
            raise RuntimeError(f"{role} worker did not prove read-only inputs")
        for audit in read_only["files"]:
            if (
                not isinstance(audit, dict)
                or audit.get("unchanged") is not True
                or not isinstance(audit.get("before"), dict)
                or not isinstance(audit.get("after"), dict)
                or any(
                    audit["before"].get(field) != audit["after"].get(field)
                    for field in ("path", "bytes", "mtime_ns", "sha256")
                )
            ):
                raise RuntimeError(f"{role} worker input audit is invalid")
        input_files = fragment.get("input_bindings", {}).get("files", [])
        bound_by_path = {
            str(record.get("path")): record
            for record in input_files
            if isinstance(record, dict)
        }
        audited_by_path = {
            str(audit["before"].get("path")): audit["before"]
            for audit in read_only["files"]
        }
        if (
            len(bound_by_path) != len(input_files)
            or len(audited_by_path) != len(read_only["files"])
            or bound_by_path != audited_by_path
        ):
            raise RuntimeError(f"{role} worker input audit does not bind its inputs")
        _validate_fragment_tables(fragment)
    frozen_source = frozen_fragment["source_identity"]
    candidate_source = candidate_fragment["source_identity"]
    if Path(str(frozen_source.get("source_root"))).resolve() != Path(
        expected_frozen_root
    ).resolve():
        raise RuntimeError("frozen worker source root mismatch")
    if Path(str(candidate_source.get("source_root"))).resolve() != Path(
        expected_candidate_root
    ).resolve():
        raise RuntimeError("candidate worker source root mismatch")
    if frozen_source.get("git_commit") != expected_frozen_commit:
        raise RuntimeError("frozen worker commit mismatch")
    if candidate_source.get("git_commit") != expected_candidate_commit:
        raise RuntimeError("candidate worker commit mismatch")
    if not _valid_git_commit(expected_frozen_commit) or not _valid_git_commit(
        expected_candidate_commit
    ):
        raise RuntimeError("cross-source worker commit identity is malformed")
    if expected_frozen_commit == expected_candidate_commit:
        raise RuntimeError("cross-source workers must use distinct commits")
    if Path(expected_frozen_root).resolve() == Path(expected_candidate_root).resolve():
        raise RuntimeError("cross-source workers must use distinct source roots")
    if (
        frozen_source["files"]["formal_evaluation.py"].get("sha256")
        != expected_frozen_evaluation_sha256
    ):
        raise RuntimeError("frozen worker evaluation source SHA mismatch")
    if frozen_source.get("tracked_tree_clean") is not True:
        raise RuntimeError("frozen worker tracked tree is dirty")
    if candidate_source.get("tracked_tree_clean") is not True:
        raise RuntimeError("candidate worker tracked tree is dirty")
    if frozen_source.get("untracked_scientific_paths") != []:
        raise RuntimeError("frozen worker has untracked scientific source")
    if candidate_source.get("untracked_scientific_paths") != []:
        raise RuntimeError("candidate worker has untracked scientific source")
    if set(frozen_source.get("files", {})) != {
        "formal_evaluation_subset_compare.py",
        "formal_evaluation.py",
    }:
        raise RuntimeError("frozen worker source file inventory is invalid")
    if set(candidate_source.get("files", {})) != {
        "formal_evaluation_subset_compare.py",
        "formal_evaluation.py",
        "formal_evaluation_streaming.py",
    }:
        raise RuntimeError("candidate worker source file inventory is invalid")
    if (
        frozen_source["files"]["formal_evaluation_subset_compare.py"]
        != candidate_source["files"]["formal_evaluation_subset_compare.py"]
    ):
        raise RuntimeError("cross-source workers used different comparator harnesses")
    if frozen_fragment.get("input_bindings") != candidate_fragment.get(
        "input_bindings"
    ):
        raise RuntimeError("cross-source workers did not bind identical formal inputs")
    if frozen_fragment.get("selection") != candidate_fragment.get("selection"):
        raise RuntimeError("cross-source workers selected different scenes/checkpoints")
    selection = frozen_fragment["selection"]
    if (
        not isinstance(selection, dict)
        or set(selection)
        != {
            "scene_rule",
            "checkpoint_rule",
            "scenes_in_execution_order",
            "checkpoints_in_execution_order",
            "sha256",
        }
        or selection.get("scene_rule") != SELECTION_RULE
        or selection.get("checkpoint_rule") != CHECKPOINT_SELECTION_RULE
        or not isinstance(selection.get("scenes_in_execution_order"), list)
        or not selection["scenes_in_execution_order"]
        or not isinstance(selection.get("checkpoints_in_execution_order"), list)
        or not selection["checkpoints_in_execution_order"]
    ):
        raise RuntimeError("cross-source worker selection contract is invalid")
    if selection.get("sha256") != _selection_sha256(
        selection.get("scenes_in_execution_order", ()),
        selection.get("checkpoints_in_execution_order", ()),
    ):
        raise RuntimeError("cross-source worker selection digest is invalid")
    if frozen_fragment.get("same_source_merged_tables") is not None:
        raise RuntimeError("frozen fragment contains a candidate-only table layer")
    frozen_evaluation = frozen_fragment["read_only"].get("legacy_evaluation")
    if (
        not isinstance(frozen_evaluation, dict)
        or frozen_evaluation.get("unchanged") is not True
        or frozen_evaluation.get("before") != frozen_evaluation.get("after")
    ):
        raise RuntimeError("frozen worker did not prove legacy evaluation read-only")
    if candidate_fragment["read_only"].get("legacy_evaluation") is not None:
        raise RuntimeError("candidate worker contains a frozen-only read-only audit")
    candidate_merged = candidate_fragment.get("same_source_merged_tables")
    if not isinstance(candidate_merged, dict):
        raise RuntimeError("candidate fragment lacks same-source decomposition rows")
    synthetic_candidate = dict(candidate_fragment)
    synthetic_candidate["tables"] = candidate_merged
    _validate_fragment_tables(synthetic_candidate)


def _worker_artifact_path(report: Path, label: str) -> Path:
    return report.with_name(f"{report.name}.{label}.json")


def _run_bound_worker(
    *,
    source_root: Path,
    request_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[BOUND_SOURCE_ROOT_ENV] = str(source_root)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "worker",
            "--request",
            str(request_path),
        ),
        cwd=source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        tail = completed.stderr[-4000:]
        raise RuntimeError(
            f"subset worker failed for {source_root} with exit "
            f"{completed.returncode}: {tail}"
        )


def run_cross_source_subset_comparison(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    legacy_run_root: str | Path,
    report_path: str | Path,
    frozen_source_root: str | Path,
    candidate_source_root: str | Path,
    frozen_commit: str = FROZEN_LEGACY_COMMIT,
    frozen_evaluation_sha256: str = FROZEN_LEGACY_EVALUATION_SHA256,
    device: str = "cuda:0",
    batch_size: int = 1,
    source_scenes: int = 1,
    target_scenes_per_city: int = 1,
    checkpoint_count: int = 1,
    checkpoint_arm: str | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or device != "cuda:0":
        raise RuntimeError(
            "formal dual-source subset comparison requires CUDA_VISIBLE_DEVICES=0 "
            "and --device cuda:0"
        )
    legacy_root = Path(legacy_run_root).resolve()
    frozen_root = Path(frozen_source_root).resolve()
    candidate_root = Path(candidate_source_root).resolve()
    target = validate_report_destination(
        report_path,
        legacy_root,
        source_root=candidate_root,
        protected_files=(config_path, dataset_path),
    )
    validate_report_destination(
        target,
        legacy_root,
        source_root=frozen_root,
        protected_files=(config_path, dataset_path),
    )
    frozen_identity = source_identity(frozen_root, require_streaming=False)
    candidate_identity = source_identity(candidate_root, require_streaming=True)
    if frozen_identity["git_commit"] != frozen_commit:
        raise RuntimeError("requested frozen source root is not the frozen commit")
    if (
        frozen_identity["files"]["formal_evaluation.py"]["sha256"]
        != frozen_evaluation_sha256
    ):
        raise RuntimeError("requested frozen source root has the wrong evaluation SHA")
    if frozen_identity["tracked_tree_clean"] is not True:
        raise RuntimeError("frozen source root has tracked changes")
    if candidate_identity["tracked_tree_clean"] is not True:
        raise RuntimeError("candidate source root has tracked changes")
    if frozen_identity["git_commit"] == candidate_identity["git_commit"]:
        raise RuntimeError("candidate source must be committed after the frozen source")
    for identity, required in (
        (
            frozen_identity,
            ("formal_evaluation_subset_compare.py", "formal_evaluation.py"),
        ),
        (
            candidate_identity,
            (
                "formal_evaluation_subset_compare.py",
                "formal_evaluation.py",
                "formal_evaluation_streaming.py",
            ),
        ),
    ):
        if any(identity["files"][name].get("git_tracked") is not True for name in required):
            raise RuntimeError("worker scientific implementation is not committed")

    common = {
        "schema_version": WORKER_REQUEST_SCHEMA,
        "config_path": str(Path(config_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "legacy_run_root": str(legacy_root),
        "device": device,
        "batch_size": batch_size,
        "source_scenes": source_scenes,
        "target_scenes_per_city": target_scenes_per_city,
        "checkpoint_count": checkpoint_count,
        "checkpoint_arm": checkpoint_arm,
    }
    artifacts = {}
    for role, root, identity in (
        ("frozen_legacy", frozen_root, frozen_identity),
        ("streaming_candidate", candidate_root, candidate_identity),
    ):
        request_path = _worker_artifact_path(target, f"{role}.request")
        fragment_path = _worker_artifact_path(target, f"{role}.fragment")
        request = {
            **common,
            "role": role,
            "fragment_path": str(fragment_path),
            "expected_git_commit": identity["git_commit"],
            "expected_evaluation_sha256": identity["files"][
                "formal_evaluation.py"
            ]["sha256"],
        }
        write_report_atomic(request_path, request)
        _run_bound_worker(source_root=root, request_path=request_path)
        fragment = read_strict_json(fragment_path)
        if not isinstance(fragment, dict):
            raise RuntimeError("subset worker fragment is not a JSON object")
        artifacts[role] = {
            "request_path": request_path,
            "fragment_path": fragment_path,
            "fragment": fragment,
        }
    frozen_fragment = artifacts["frozen_legacy"]["fragment"]
    candidate_fragment = artifacts["streaming_candidate"]["fragment"]
    validate_worker_pair(
        frozen_fragment,
        candidate_fragment,
        expected_frozen_root=frozen_root,
        expected_candidate_root=candidate_root,
        expected_frozen_commit=frozen_identity["git_commit"],
        expected_candidate_commit=candidate_identity["git_commit"],
        expected_frozen_evaluation_sha256=frozen_evaluation_sha256,
    )
    cross_comparison = compare_evaluation_tables(
        frozen_fragment["tables"],
        candidate_fragment["tables"],
        implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
    )
    cross_gate = gate_aggregate_report(
        frozen_fragment["tables"],
        candidate_fragment["tables"],
        implementation_fields=IMPLEMENTATION_EVIDENCE_FIELDS,
    )
    same_comparison = compare_evaluation_tables(
        candidate_fragment["same_source_merged_tables"],
        candidate_fragment["tables"],
    )
    same_gate = gate_aggregate_report(
        candidate_fragment["same_source_merged_tables"],
        candidate_fragment["tables"],
    )
    passed = bool(
        cross_comparison["exact"]
        and cross_gate["exact"]
        and same_comparison["exact"]
        and same_gate["exact"]
    )
    worker_bindings = {}
    for role in ("frozen_legacy", "streaming_candidate"):
        artifact = artifacts[role]
        fragment = artifact["fragment"]
        worker_bindings["legacy" if role == "frozen_legacy" else "new"] = {
            "role": role,
            "source_identity": fragment["source_identity"],
            "runtime_provenance": fragment["runtime_provenance"],
            "request": _input_record(artifact["request_path"]),
            "fragment": _input_record(artifact["fragment_path"]),
            "resources": fragment["resources"],
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "created_utc": _utc_now(),
        "authoritative_layer": "layers.cross_source_end_to_end",
        "contract": {
            "formal_dataset_only": True,
            "full_legacy_evaluation_executed": False,
            "legacy_run_read_only": True,
            "frozen_legacy_run_body_executed": True,
            "cross_source_expected_implementation_fields": list(
                IMPLEMENTATION_EVIDENCE_FIELDS
            ),
            "scientific_float_requirement": "IEEE-754 binary64 bitwise equality",
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
        },
        "workers": worker_bindings,
        "shared_inputs": frozen_fragment["input_bindings"],
        "selection": frozen_fragment["selection"],
        "layers": {
            "cross_source_end_to_end": {
                "authoritative": True,
                "legacy_source_role": "frozen_legacy_oracle",
                "candidate_source_role": "streaming_candidate",
                "comparison": cross_comparison,
                "gate_required_aggregates": cross_gate,
            },
            "same_source_decomposition": {
                "authoritative": False,
                "description": (
                    "candidate-source merged-scene decomposition versus candidate-source "
                    "per-scene streaming; this layer is not legacy-source evidence"
                ),
                "comparison": same_comparison,
                "gate_required_aggregates": same_gate,
            },
        },
        "read_only": {
            "passed": bool(
                frozen_fragment["read_only"]["passed"]
                and candidate_fragment["read_only"]["passed"]
            ),
            "legacy": frozen_fragment["read_only"],
            "new": candidate_fragment["read_only"],
        },
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "legacy": frozen_fragment["resources"],
            "new": candidate_fragment["resources"],
        },
        "report_path": str(target),
    }
    write_report_atomic(target, report)
    return report


def run_same_source_decomposition_comparison(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    legacy_run_root: str | Path,
    report_path: str | Path,
    device: str = "cuda:0",
    batch_size: int = 1,
    source_scenes: int = 1,
    target_scenes_per_city: int = 1,
    checkpoint_count: int = 1,
    checkpoint_arm: str | None = None,
) -> dict[str, object]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    configure_reproducible_runtime()
    started = time.monotonic()
    config_file = Path(config_path).resolve()
    dataset_file = Path(dataset_path).resolve()
    legacy_root = Path(legacy_run_root).resolve()
    target = validate_report_destination(
        report_path,
        legacy_root,
        source_root=Path(__file__).resolve().parent.parent,
        protected_files=(config_file, dataset_file),
    )
    if config_file.is_symlink() or not config_file.is_file():
        raise RuntimeError("formal config must be a regular file")
    if dataset_file.is_symlink() or not dataset_file.is_file():
        raise RuntimeError("formal dataset must be a regular file")
    config = load_formal_config(config_file)
    dataset = FormalDataset.load(dataset_file)
    if dataset.is_fixture:
        raise ValueError("real-subset comparison rejects synthetic/fixture datasets")
    execution_device = resolve_execution_device(dataset, device)
    (
        qualification_gate,
        factorial_gate,
        checkpoint_rows,
        upstream_paths,
    ) = _validate_legacy_inputs(config, dataset, legacy_root)
    del factorial_gate
    selected_scenes = select_real_evaluation_subset(
        dataset,
        source_scenes=source_scenes,
        target_scenes_per_city=target_scenes_per_city,
    )
    selected_checkpoints = select_legacy_checkpoints(
        checkpoint_rows,
        config["factorial"]["arms"],
        checkpoint_count=checkpoint_count,
        checkpoint_arm=checkpoint_arm,
    )
    checkpoint_records = _selected_checkpoint_records(
        legacy_root / "factorial", selected_checkpoints
    )
    input_records = [
        _input_record(config_file, identity_sha256=config_sha256(config)),
        _input_record(dataset_file, identity_sha256=sha256_file(dataset_file)),
        *(_input_record(path) for path in upstream_paths.values()),
        *(
            _input_record(
                Path(record["path"]), identity_sha256=str(record["sha256"])
            )
            for record in checkpoint_records
        ),
    ]
    protected = [record["path"] for record in input_records]
    target = validate_report_destination(
        target,
        legacy_root,
        protected_files=protected,
    )
    evidence = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
    scene_indices = tuple(int(record["scene_index"]) for record in selected_scenes)

    if execution_device.type == "cuda":
        torch.cuda.set_device(execution_device)
        torch.cuda.reset_peak_memory_stats(execution_device)
    teacher = load_teacher_bundle(
        upstream_paths["teacher_checkpoint"], config, device=execution_device
    )
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset,
        dataset.indices_for_role("source_encoder_train"),
        route_normalization,
        teacher.patch_spec,
    )
    merged_tables = _empty_tables()
    per_scene_tables = _empty_tables()
    for checkpoint in selected_checkpoints:
        seed = int(checkpoint["seed"])
        arm = str(checkpoint["arm"])
        model = legacy._load_model(
            legacy_root / "factorial",
            checkpoint,
            qualification_gate,
            config,
            dataset,
            device=execution_device,
        )
        probes = streaming._fit_probe_bundle(
            model,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            seed=seed,
            batch_size=batch_size,
        )
        contracts = streaming._probe_contract_rows(
            probes, config, seed=seed, arm=arm
        )
        for destination in (merged_tables, per_scene_tables):
            destination["compatibility_probe_contract.csv"].extend(
                contracts["compatibility_probe_contract.csv"]
            )
            destination["response_probe_contract.csv"].extend(
                contracts["response_probe_contract.csv"]
            )
        merged = _evaluate_legacy_merged_checkpoint(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene_indices,
            seed=seed,
            arm=arm,
            batch_size=batch_size,
        )
        per_scene = _evaluate_streaming_checkpoint(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene_indices,
            seed=seed,
            arm=arm,
            batch_size=batch_size,
        )
        _extend_tables(
            merged_tables,
            {
                **{
                    "compatibility_probe_contract.csv": [],
                    "response_probe_contract.csv": [],
                },
                **merged,
            },
        )
        _extend_tables(
            per_scene_tables,
            {
                **{
                    "compatibility_probe_contract.csv": [],
                    "response_probe_contract.csv": [],
                },
                **per_scene,
            },
        )
        del model, probes, contracts, merged, per_scene
        gc.collect()
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()

    merged_bound = _bind_complete_tables(merged_tables, evidence)
    per_scene_bound = _bind_complete_tables(per_scene_tables, evidence)
    comparison = compare_evaluation_tables(merged_bound, per_scene_bound)
    gate_aggregates = gate_aggregate_report(merged_bound, per_scene_bound)
    _assert_inputs_unchanged(input_records)
    elapsed = time.monotonic() - started
    maximum_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(execution_device))
        if execution_device.type == "cuda"
        else 0
    )
    legacy_source_shas = sorted(
        {
            str(value)
            for value in (
                qualification_gate.get("source_tree_sha256"),
                *(row.get("source_tree_sha256") for row in selected_checkpoints),
            )
            if _valid_sha256(value)
        }
    )
    exact = bool(comparison["exact"] and gate_aggregates["exact"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS" if exact else "FAIL",
        "passed": exact,
        "created_utc": _utc_now(),
        "contract": {
            "authoritative": False,
            "evidence_scope": "candidate_source_decomposition_only",
            "legacy_source_equivalence_assessed": False,
            "dataset_kind": "formal_nonfixture_real_banks",
            "merged_path": "all_selected_scenes_in_one_merged_array",
            "streaming_path": "one_scene_at_a_time_then_checkpoint_bank_order_merge",
            "shared_teacher_model_and_probe_objects": True,
            "scientific_float_requirement": "IEEE-754 binary64 bitwise equality",
            "absolute_tolerance": 0.0,
            "relative_tolerance": 0.0,
            "legacy_inputs_read_only": True,
            "full_legacy_evaluation_executed": False,
        },
        "inputs": {
            "config": input_records[0],
            "dataset": {
                **input_records[1],
                "schema_version": dataset.metadata.get("schema_version"),
                "fixture": False,
                "scene_count": dataset.scene_count,
            },
            "legacy_run_root": str(legacy_root),
            "qualification_gate": _input_record(upstream_paths["qualification_gate"]),
            "factorial_gate": _input_record(upstream_paths["factorial_gate"]),
            "checkpoint_index": _input_record(upstream_paths["checkpoint_index"]),
            "teacher_checkpoint": _input_record(
                upstream_paths["teacher_checkpoint"],
                identity_sha256=qualification_gate["teacher_checkpoint_sha256"],
            ),
            "checkpoints": list(checkpoint_records),
            "legacy_recorded_source_tree_sha256": legacy_source_shas,
            "implementation": _implementation_sources(evidence),
        },
        "selection": {
            "scene_rule": SELECTION_RULE,
            "checkpoint_rule": CHECKPOINT_SELECTION_RULE,
            "requested_source_scenes": source_scenes,
            "requested_target_scenes_per_city": target_scenes_per_city,
            "requested_checkpoint_count": checkpoint_count,
            "requested_checkpoint_arm": checkpoint_arm,
            "scene_count": len(selected_scenes),
            "checkpoint_count": len(selected_checkpoints),
            "scenes_in_execution_order": list(selected_scenes),
            "checkpoints_in_execution_order": [
                {
                    "seed": int(row["seed"]),
                    "arm": str(row["arm"]),
                    "sha256": str(row["sha256"]),
                }
                for row in selected_checkpoints
            ],
        },
        "comparison": comparison,
        "gate_required_aggregates": gate_aggregates,
        "execution": {
            "device": str(execution_device),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed,
            "maximum_resident_set_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "maximum_cuda_allocated_bytes": maximum_gpu_bytes,
            "python_dont_write_bytecode": bool(os.environ.get("PYTHONDONTWRITEBYTECODE")),
            "report_path": str(target),
        },
    }
    write_report_atomic(target, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen legacy and candidate streaming evaluation sources on a "
            "deterministic real-data subset."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", required=True)
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--legacy-run", required=True)
    compare.add_argument("--report", required=True)
    compare.add_argument("--frozen-source-root", required=True)
    compare.add_argument(
        "--frozen-commit", default=FROZEN_LEGACY_COMMIT
    )
    compare.add_argument(
        "--frozen-evaluation-sha256",
        default=FROZEN_LEGACY_EVALUATION_SHA256,
    )
    compare.add_argument(
        "--candidate-source-root",
        default=str(Path(__file__).resolve().parent.parent),
    )
    compare.add_argument("--device", default="cuda:0")
    compare.add_argument("--batch-size", type=int, default=1)
    compare.add_argument("--source-scenes", type=int, default=1)
    compare.add_argument("--target-scenes-per-city", type=int, default=1)
    compare.add_argument("--checkpoint-count", type=int, default=1)
    compare.add_argument("--checkpoint-arm")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--request", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        fragment = run_worker_request(args.request)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "role": fragment["role"],
                    "source_root": fragment["source_identity"]["source_root"],
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        )
        return 0
    report = run_cross_source_subset_comparison(
        config_path=args.config,
        dataset_path=args.dataset,
        legacy_run_root=args.legacy_run,
        report_path=args.report,
        frozen_source_root=args.frozen_source_root,
        candidate_source_root=args.candidate_source_root,
        frozen_commit=args.frozen_commit,
        frozen_evaluation_sha256=args.frozen_evaluation_sha256,
        device=args.device,
        batch_size=args.batch_size,
        source_scenes=args.source_scenes,
        target_scenes_per_city=args.target_scenes_per_city,
        checkpoint_count=args.checkpoint_count,
        checkpoint_arm=args.checkpoint_arm,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "report": report["report_path"],
                "table_count": report["layers"]["cross_source_end_to_end"][
                    "comparison"
                ]["table_count"],
                "max_absolute_difference": report["layers"][
                    "cross_source_end_to_end"
                ]["comparison"]["max_absolute_difference"],
            },
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
