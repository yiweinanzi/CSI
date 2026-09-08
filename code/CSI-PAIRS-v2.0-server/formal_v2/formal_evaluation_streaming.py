from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import json
import math
import os
import queue
import tempfile
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Iterable, Mapping

import numpy as np

from . import formal_evaluation as legacy
from .formal_evaluation_fragments import (
    CsvFragmentError,
    inspect_csv_fragment,
    merge_csv_fragments,
    visit_csv_fragment_rows,
    write_csv_fragment_stream,
)
from .formal_evaluation_inventory import (
    PairInventoryError,
    ScenePairUniverse,
    build_scene_pair_universe,
)
from .formal_evaluation_resume import (
    EvaluationExecutionProfile,
    EvaluationResumeStore,
    EvaluationRunIdentity,
    EvaluationShardIdentity,
    StaleResumeError,
    require_safe_directory_prefix,
    write_atomic_json,
)
from .formal_evidence import evidence_context
from .formal_io import StrictJsonError, read_strict_json, sha256_file
from .formal_metrics import binary_auroc
from .formal_model import torch
from .formal_probes import (
    ActionResponseProbe,
    CompatibilityProbe,
    fit_action_response_probe,
    fit_select_compatibility_probe,
    predict_binary_probe,
    predict_response_probe,
)
from .formal_routing import ROUTE_NAMES, fit_route_normalization, route_dataset
from .formal_teacher import load_teacher_bundle


STREAMING_EVALUATION_SCHEMA = "csi-pairs-v6-streaming-evaluation-v1"
PROBE_BUNDLE_SCHEMA = "csi-pairs-v6-evaluation-probe-bundle-v1"
SCENE_BUNDLE_SCHEMA = "csi-pairs-v6-evaluation-scene-bundle-v3"
PAIR_INVENTORY_SCHEMA = "csi-pairs-v6-evaluation-pair-inventory-v1"
FINALIZATION_SCHEMA = "csi-pairs-v6-evaluation-finalization-v1"
STATE_DIRECTORY = "evaluation_state"
DEFAULT_PROGRESS_HEARTBEAT_SECONDS = 30.0
DEFAULT_EVALUATION_WORKER_DEVICES = ("cuda:0", "cuda:1")
PROBE_TRAIN_BATCH_ROWS = 4096
PROBE_BUILD_RESERVATION_BYTES = 256 * 1024**3


def _available_probe_memory() -> int | None:
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0]) * 1024
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _hierarchy, controllers, relative = line.split(":", 2)
            if not controllers:
                root = Path("/sys/fs/cgroup")
                limit_name, usage_name = "memory.max", "memory.current"
            elif "memory" in controllers.split(","):
                root = Path("/sys/fs/cgroup/memory")
                limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
            else:
                continue
            nested = root / relative.lstrip("/")
            # A namespaced cgroup mount can already point at the current group.
            current = nested if (nested / limit_name).is_file() else root
            for group in (current, *current.parents):
                if group != root and root not in group.parents:
                    break
                limit = (group / limit_name).read_text().strip()
                if limit != "max":
                    available = min(available, int(limit) - int((group / usage_name).read_text()))
        return max(0, available)
    except (OSError, ValueError, KeyError):
        return None


def _probe_build_capacity(worker_count: int, available_bytes: int | None) -> int:
    if available_bytes is None:
        return 1
    # Reserve headroom for the dataset, scene workers and allocator temporaries.
    capacity = (available_bytes - 64 * 1024**3) // PROBE_BUILD_RESERVATION_BYTES
    return max(1, min(worker_count, int(capacity)))


class _ProbeProgress:
    """Workers update memory only; the coordinator heartbeat publishes telemetry."""

    def __init__(self, capacity: int, available_bytes: int | None):
        self._lock = threading.Lock()
        self._workers = {}
        self.capacity = capacity
        self.available_bytes = available_bytes
        self._started = {}
        self._phases = {}
        self._completed = {}

    def callback(self, unit):
        def update(event):
            with self._lock:
                key = (unit.execution_device, unit.checkpoint_index)
                now = time.monotonic()
                self._started.setdefault(key, now)
                phase = (event.get("probe"), event.get("family"), event.get("phase"))
                if phase[-1] == "accumulating":
                    phase = (*phase[:-1], "training")
                previous_phase, phase_start = self._phases.get(key, (phase, now))
                if previous_phase != phase:
                    phase_start = now
                self._phases[key] = (phase, phase_start)
                completed = self._completed.setdefault(key, set())
                if event.get("phase") == "training_complete":
                    completed.add((event.get("probe"), event.get("family")))
                self._workers[unit.execution_device] = {
                    "seed": unit.seed, "arm": unit.arm,
                    "checkpoint_index": unit.checkpoint_index,
                    "unit_index": getattr(unit, "canonical_index", unit.checkpoint_index),
                    "device": unit.execution_device,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "last_actual_callback_at": datetime.now(timezone.utc).isoformat(),
                    "unit_elapsed_seconds": now - self._started[key],
                    "stage_elapsed_seconds": now - phase_start,
                    "completed_training_models": len(completed),
                    "total_training_models": 17,
                    "step_note": "optimizer submissions during training; CUDA synchronized at training_complete",
                    **event,
                }
        return update

    def snapshot(self):
        with self._lock:
            return {
                "schema_version": "csi-pairs-probe-progress-v1",
                "note": "telemetry only; not completed shards or scientific evidence",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "build_capacity": self.capacity,
                "available_memory_at_start_bytes": self.available_bytes,
                "reservation_per_build_bytes": PROBE_BUILD_RESERVATION_BYTES,
                "train_batch_rows": PROBE_TRAIN_BATCH_ROWS,
                "workers": {key: dict(value) for key, value in self._workers.items()},
            }


@dataclass(frozen=True)
class EvaluationWorkUnit:
    """One canonical compute-only unit assigned to a deterministic device."""

    canonical_index: int
    checkpoint_index: int
    checkpoint_sha256: str
    seed: int
    arm: str
    substage: str
    shard: str
    scene: int | None
    execution_device: str
    batch_size: int = 1


def _validated_worker_devices(devices: Iterable[str]) -> tuple[str, ...]:
    values = tuple(devices)
    if not values:
        raise ValueError("evaluation worker devices must not be empty")
    for device in values:
        if (
            type(device) is not str
            or not device.startswith("cuda:")
            or not device[5:].isdigit()
            or str(int(device[5:])) != device[5:]
        ):
            raise ValueError(f"invalid evaluation worker device: {device!r}")
    if len(values) != len(set(values)):
        raise ValueError("evaluation worker devices must be unique")
    return values


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_evaluation_work_plan(
    checkpoint_rows: Iterable[Mapping[str, object]],
    scenes: Iterable[int],
    *,
    devices: Iterable[str] = DEFAULT_EVALUATION_WORKER_DEVICES,
    batch_size: int = 1,
) -> tuple[EvaluationWorkUnit, ...]:
    """Build a canonical checkpoint-grouped plan without inspecting GPU state."""

    worker_devices = _validated_worker_devices(devices)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("evaluation worker batch_size must be a positive integer")
    scene_rows = tuple(scenes)
    if any(type(scene) is not int or scene < 0 for scene in scene_rows):
        raise ValueError("evaluation work-plan scenes must be nonnegative integers")
    if len(scene_rows) != len(set(scene_rows)):
        raise ValueError("evaluation work-plan scenes must be unique")

    work: list[EvaluationWorkUnit] = []
    canonical_index = 0
    for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
        if not isinstance(checkpoint, Mapping):
            raise TypeError("evaluation work-plan checkpoint must be a mapping")
        checkpoint_sha256 = checkpoint.get("sha256")
        seed = checkpoint.get("seed")
        arm = checkpoint.get("arm")
        if not _valid_sha256(checkpoint_sha256):
            raise ValueError("evaluation work-plan checkpoint SHA-256 is invalid")
        if type(seed) is not int or seed < 0:
            raise ValueError("evaluation work-plan checkpoint seed is invalid")
        if type(arm) is not str or not arm:
            raise ValueError("evaluation work-plan checkpoint arm is invalid")
        device = worker_devices[checkpoint_index % len(worker_devices)]
        work.append(
            EvaluationWorkUnit(
                canonical_index=canonical_index,
                checkpoint_index=checkpoint_index,
                checkpoint_sha256=checkpoint_sha256,
                seed=seed,
                arm=arm,
                substage="probe_state",
                shard=f"checkpoint-{checkpoint_index:02d}",
                scene=None,
                execution_device=device,
                batch_size=batch_size,
            )
        )
        canonical_index += 1
        for scene in scene_rows:
            work.append(
                EvaluationWorkUnit(
                    canonical_index=canonical_index,
                    checkpoint_index=checkpoint_index,
                    checkpoint_sha256=checkpoint_sha256,
                    seed=seed,
                    arm=arm,
                    substage="scene",
                    shard=f"scene-{scene:03d}",
                    scene=scene,
                    execution_device=device,
                    batch_size=batch_size,
                )
            )
            canonical_index += 1
    return tuple(work)


def _validate_evaluation_work_plan(
    work_units: tuple[EvaluationWorkUnit, ...],
) -> tuple[str, ...]:
    if any(not isinstance(unit, EvaluationWorkUnit) for unit in work_units):
        raise TypeError("evaluation work plan contains a non-work-unit value")
    canonical_indices = [unit.canonical_index for unit in work_units]
    if canonical_indices != list(range(len(work_units))):
        raise ValueError("evaluation work plan canonical indices are invalid")
    if not work_units:
        return ()

    devices = _validated_worker_devices(
        dict.fromkeys(unit.execution_device for unit in work_units)
    )
    checkpoint_indices = sorted({unit.checkpoint_index for unit in work_units})
    if checkpoint_indices != list(range(len(checkpoint_indices))):
        raise ValueError("evaluation work plan checkpoint indices are invalid")
    previous_checkpoint = -1
    identities: dict[int, tuple[object, ...]] = {}
    substages: dict[int, list[str]] = {}
    scenes: dict[int, set[int]] = {}
    for unit in work_units:
        if type(unit.canonical_index) is not int:
            raise ValueError("evaluation work-unit canonical index is invalid")
        if type(unit.checkpoint_index) is not int or unit.checkpoint_index < 0:
            raise ValueError("evaluation work-unit checkpoint index is invalid")
        if unit.checkpoint_index < previous_checkpoint:
            raise ValueError("evaluation work plan is not checkpoint-canonical")
        previous_checkpoint = unit.checkpoint_index
        if not _valid_sha256(unit.checkpoint_sha256):
            raise ValueError("evaluation work-unit checkpoint SHA-256 is invalid")
        if type(unit.seed) is not int or unit.seed < 0:
            raise ValueError("evaluation work-unit seed is invalid")
        if type(unit.arm) is not str or not unit.arm:
            raise ValueError("evaluation work-unit arm is invalid")
        if type(unit.shard) is not str or not unit.shard:
            raise ValueError("evaluation work-unit shard is invalid")
        if type(unit.batch_size) is not int or unit.batch_size < 1:
            raise ValueError("evaluation work-unit batch_size is invalid")
        identity = (
            unit.checkpoint_sha256,
            unit.seed,
            unit.arm,
            unit.execution_device,
            unit.batch_size,
        )
        previous_identity = identities.setdefault(unit.checkpoint_index, identity)
        if previous_identity != identity:
            raise ValueError("one checkpoint spans multiple worker identities")
        expected_device = devices[unit.checkpoint_index % len(devices)]
        if unit.execution_device != expected_device:
            raise ValueError("evaluation checkpoint device assignment is not round-robin")
        checkpoint_substages = substages.setdefault(unit.checkpoint_index, [])
        checkpoint_scenes = scenes.setdefault(unit.checkpoint_index, set())
        checkpoint_substages.append(unit.substage)
        if unit.substage == "probe_state":
            if unit.scene is not None or checkpoint_substages != ["probe_state"]:
                raise ValueError("evaluation probe work-unit order is invalid")
        elif unit.substage == "scene":
            if type(unit.scene) is not int or unit.scene < 0:
                raise ValueError("evaluation scene work-unit identity is invalid")
            if unit.scene in checkpoint_scenes:
                raise ValueError("evaluation work plan contains a duplicate scene")
            checkpoint_scenes.add(unit.scene)
        else:
            raise ValueError("evaluation work-unit substage is invalid")
    for checkpoint_substages in substages.values():
        if not checkpoint_substages or checkpoint_substages[0] != "probe_state":
            raise ValueError("evaluation checkpoint is missing its probe work-unit")
    return devices


def run_evaluation_worker_coordinator(
    work_units: Iterable[EvaluationWorkUnit],
    worker: Callable[[EvaluationWorkUnit], object],
    coordinator: Callable[[EvaluationWorkUnit, object], object],
    *,
    wait_context_factory: (
        Callable[[EvaluationWorkUnit], ContextManager[object]] | None
    ) = None,
) -> tuple[object, ...]:
    """Compute on device workers and publish in deterministic device-round order.

    Each device owns one serial queue, so work for a checkpoint never overlaps on
    the same GPU. Interleaving one result from each device prevents a checkpoint-
    major output order from starving the other GPU. Workers only return values;
    the caller-thread coordinator is the sole publication hook, and receipts are
    returned in canonical-index order.
    """

    plan = tuple(work_units)
    devices = _validate_evaluation_work_plan(plan)
    if not callable(worker) or not callable(coordinator):
        raise TypeError("evaluation worker and coordinator must be callable")
    if not plan:
        return ()

    if wait_context_factory is not None and not callable(wait_context_factory):
        raise TypeError("evaluation wait_context_factory must be callable")
    device_work = {
        device: tuple(unit for unit in plan if unit.execution_device == device)
        for device in devices
    }
    coordination_plan = tuple(
        device_work[device][offset]
        for offset in range(max(len(units) for units in device_work.values()))
        for device in devices
        if offset < len(device_work[device])
    )
    if len(coordination_plan) != len(plan) or set(coordination_plan) != set(plan):
        raise RuntimeError("evaluation coordination plan is incomplete")
    result_queues = {device: queue.Queue(maxsize=1) for device in devices}
    stop = threading.Event()

    def publish(device: str, item: tuple[str, EvaluationWorkUnit, object]) -> bool:
        while not stop.is_set():
            try:
                result_queues[device].put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def compute_device_queue(device: str) -> None:
        for unit in device_work[device]:
            if stop.is_set():
                return
            try:
                result = worker(unit)
            except BaseException as error:
                publish(device, ("error", unit, error))
                return
            if not publish(device, ("result", unit, result)):
                return

    caller_thread = threading.get_ident()
    receipts: dict[int, object] = {}
    with ThreadPoolExecutor(
        max_workers=len(device_work), thread_name_prefix="evaluation-worker"
    ) as executor:
        futures = {
            executor.submit(compute_device_queue, device): device for device in devices
        }
        try:
            for expected in coordination_plan:
                context = (
                    wait_context_factory(expected)
                    if wait_context_factory is not None
                    else nullcontext()
                )
                with context:
                    kind, unit, payload = result_queues[
                        expected.execution_device
                    ].get()
                    if unit != expected:
                        raise RuntimeError(
                            "evaluation worker returned a noncanonical or missing result"
                        )
                    if kind == "error":
                        if not isinstance(payload, BaseException):
                            raise RuntimeError("evaluation worker error payload is invalid")
                        raise payload
                    if kind != "result":
                        raise RuntimeError("evaluation worker result kind is invalid")
                    if threading.get_ident() != caller_thread:
                        raise RuntimeError(
                            "evaluation coordinator left the caller thread"
                        )
                    receipts[unit.canonical_index] = coordinator(unit, payload)
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    if len(receipts) != len(plan):
        raise RuntimeError("evaluation coordinator did not publish every work unit")
    return tuple(receipts[index] for index in range(len(plan)))


def _field_tuple(value: str) -> tuple[str, ...]:
    return tuple(value.split(","))


EVALUATION_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
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

SCENE_TABLES = (
    "cgs_per_bank.csv",
    "alignment_shortcut_baselines.csv",
    "compatibility_route_distributions.csv",
    "cgs_active_effect_bins.csv",
    "compatibility_pair_effects.csv",
    "response_pair_effects.csv",
    "response_per_bank.csv",
)

CSV_EVIDENCE_FIELDS = (
    "artifact_label",
    "dataset_sha256",
    "config_sha256",
    "fixture",
    "scientific_use",
    "source_tree_sha256",
    "requirements_lock_sha256",
    "runtime_provenance_sha256",
)
PAIR_SCENE_TABLES = (
    "compatibility_pair_effects.csv",
    "response_pair_effects.csv",
)
SMALL_SCENE_TABLES = tuple(
    table for table in SCENE_TABLES if table not in PAIR_SCENE_TABLES
)

SCENE_TEXT_FIELDS = {
    "arm",
    "bank_id",
    "base_map_cluster_id",
    "canonical_base_map_digest",
    "canonical_bank_digest",
    "city_id",
    "evaluation_scope",
    "route",
    "probe_family",
    "baseline",
    "input_class",
    "identity_token_contract",
    "fit_role",
    "selection_role",
    "condition_status",
    "effect_bin",
    "pair_id",
    "wrong_action_match_status",
    "native_mask_bank",
    "native_input_contract",
    *CSV_EVIDENCE_FIELDS,
}
SCENE_INTEGER_FIELDS = {
    "seed",
    "steps",
    "hidden_dim",
    "n",
    "pair_count",
    "scene_index",
    "source_world",
    "target_world",
    "position_index",
    "query_index",
    "wrong_action_world",
    "probe_action_swap_exact_count",
    "probe_action_swap_fallback_count",
    "probe_action_swap_failed_count",
    "native_action_swap_exact_count",
    "native_action_swap_fallback_count",
    "native_action_swap_failed_count",
    "native_null_patch_count",
    "n_active_patches",
}
SCENE_NONNEGATIVE_FLOAT_FIELDS = {
    field
    for fields in EVALUATION_TABLE_FIELDS.values()
    for field in fields
    if field.endswith(("_mse", "_nmse", "_mae", "_rms_mean"))
} | {
    "physical_distance",
    "physical_distance_min",
    "physical_distance_max",
    "absolute_difference_p90",
    "native_delta_relative_magnitude_error",
}
OPTIONAL_SCENE_NUMERIC_FIELDS = frozenset(
    {
        "matched_minus_alternative_mean",
        "matched_minus_alternative_median",
        "absolute_difference_p90",
        "overclassification_rate",
        "action_swap_mse",
        "response_advantage_vs_action_swap",
        "unified_response_probe_action_swap_exact_patch_nmse",
        "probe_action_swap_active_patch_nmse",
        "native_target_free_action_swap_exact_full_channel_nmse",
        "native_action_swap_full_channel_nmse",
        "native_latent_action_swap_exact_target_nmse",
        "native_latent_action_swap_nmse",
        "native_transition_skill",
        "native_null_delta_rms_mean",
        "native_null_violation_rate",
        "native_latent_null_delta_rms_mean",
        "native_latent_null_violation_rate",
    }
)


def evaluation_output_schema_sha256() -> str:
    payload = {
        "schema_version": STREAMING_EVALUATION_SCHEMA,
        "tables": {
            name: list(fields) for name, fields in EVALUATION_TABLE_FIELDS.items()
        },
        "json_outputs": ["gate.json", "manifest.json"],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _ProbeBundle:
    compatibility_probe: CompatibilityProbe
    compatibility_selection: dict
    shortcut_probes: dict[str, dict]
    response_probe: ActionResponseProbe
    variant_probes: dict[str, ActionResponseProbe]


@dataclass
class _ProbeWorkResult:
    probes: _ProbeBundle
    requires_commit: bool


@dataclass
class _SceneWorkResult:
    completed_bundle: dict[str, object] | None
    rows_by_table: dict[str, list[dict]] | None


@dataclass(frozen=True)
class _SceneInputs:
    scene: int
    routed: object
    compatibility: dict[str, object]
    response: dict[str, object]


@dataclass(frozen=True)
class _SceneEvaluation:
    scene: int
    routed: object
    compatibility: dict[str, object]
    compatibility_probabilities: np.ndarray
    response: dict[str, object]
    response_prediction: np.ndarray
    response_swap_prediction: np.ndarray
    response_no_action_prediction: np.ndarray
    response_variant_predictions: dict[str, np.ndarray]


class _SceneBundleSemanticError(RuntimeError):
    """A hash-valid scene artifact failed its payload-level contract."""

    def __init__(self, message: str, *, table_identity=None) -> None:
        super().__init__(message)
        self.table_identity = table_identity


def _fit_probe_bundle(
    model,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    seed: int,
    batch_size: int,
    rng_lock=None,
    device=None,
    progress_callback=None,
) -> _ProbeBundle:
    def report(probe, event=None):
        if progress_callback is not None:
            progress_callback({"probe": probe, **(event or {"phase": "preparing"})})

    probe_options = {
        "device": device,
        "train_batch_rows": PROBE_TRAIN_BATCH_ROWS if device is not None else None,
        "rng_lock": rng_lock,
        "prefer_full_batch": device is not None,
    }
    report("routing")
    train_scenes = dataset.indices_for_role("source_probe_train")
    selection_scenes = dataset.indices_for_role("source_probe_selection")
    routed_train = route_dataset(
        dataset,
        teacher,
        config,
        train_scenes,
        normalization=route_normalization,
    )
    routed_selection = route_dataset(
        dataset,
        teacher,
        config,
        selection_scenes,
        normalization=route_normalization,
    )
    report("compatibility_train_features")
    compatibility_train = legacy._compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        train_scenes,
        route_normalization=route_normalization,
        routed=routed_train,
        batch_size=batch_size,
    )
    report("compatibility_selection_features")
    compatibility_selection = legacy._compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        selection_scenes,
        route_normalization=route_normalization,
        routed=routed_selection,
        batch_size=batch_size,
    )
    compatibility_probe, selection_record = fit_select_compatibility_probe(
        compatibility_train["features"],
        compatibility_train["labels"],
        compatibility_selection["features"],
        compatibility_selection["labels"],
        config,
        seed=seed + 31001,
        **probe_options,
        progress_callback=lambda event: report("compatibility", event),
    )
    shortcut_probes = legacy._prepare_alignment_shortcut_probes(
        seed,
        compatibility_train,
        compatibility_selection,
        config,
        **probe_options,
        progress_callback=lambda event: report("shortcut", event),
    )
    del compatibility_train, compatibility_selection, routed_selection
    gc.collect()
    report("response_train_features")
    response_train = legacy._response_probe_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        train_scenes,
        active_only=False,
        route_normalization=route_normalization,
        routed=routed_train,
        batch_size=batch_size,
    )
    response_target = response_train["targets"] - response_train["source_targets"]
    response_probe = fit_action_response_probe(
        response_train["features"],
        response_target,
        config,
        seed=seed + 32001,
        zero_action_x=response_train["no_action_features"],
        **probe_options,
        progress_callback=lambda event: report("response", event),
    )
    variants: dict[str, ActionResponseProbe] = {}
    contrast_variants = {"without_map", "edit_only", "oracle_x"}
    for offset, name in enumerate(
        ("without_map", "edit_only", "csi_only", "oracle_x"), start=1
    ):
        variants[name] = fit_action_response_probe(
            response_train[f"{name}_features"],
            response_target,
            config,
            seed=seed + 32001 + offset,
            zero_action_x=(
                response_train[f"{name}_zero_action_features"]
                if name in contrast_variants
                else None
            ),
            **probe_options,
            progress_callback=lambda event, name=name: report(name, event),
        )
    del response_train, response_target, routed_train
    gc.collect()
    if device is not None:
        # Preserve identical scene evaluation placement on fresh and resumed bundles.
        for probe in (
            compatibility_probe, response_probe, *variants.values(),
            *(record["probe"] for record in shortcut_probes.values()),
        ):
            if probe is not None:
                probe.cpu()
    report("bundle", {"phase": "ready_to_commit"})
    return _ProbeBundle(
        compatibility_probe=compatibility_probe,
        compatibility_selection=selection_record,
        shortcut_probes=shortcut_probes,
        response_probe=response_probe,
        variant_probes=variants,
    )


def _fit_probe_bundle_exclusive(probe_build_lock, *args, **kwargs) -> _ProbeBundle:
    """Admit only the number of full probe corpora allowed by the memory budget."""

    callback = kwargs.get("progress_callback")
    if callback is not None:
        callback({"phase": "waiting_for_memory_slot", "probe": "bundle"})
    waiting_since = time.monotonic()
    with probe_build_lock:
        acquired_at = time.monotonic()
        if callback is not None:
            callback({"phase": "memory_slot_acquired", "probe": "bundle", "lock_wait_seconds": acquired_at - waiting_since})
        result = _fit_probe_bundle(*args, **kwargs)
        if callback is not None:
            callback({"phase": "ready_to_commit", "probe": "bundle", "lock_wait_seconds": acquired_at - waiting_since, "lock_held_seconds": time.monotonic() - acquired_at})
        return result


def _require_prediction_shape(
    values, expected_shape: tuple[int, ...], label: str
) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != expected_shape:
        raise RuntimeError(
            f"scene-local {label} shape is invalid: "
            f"expected {expected_shape}, observed {result.shape}"
        )
    return result


def _require_scene_rows(
    payload: Mapping[str, object], scene: int, label: str
) -> int:
    scene_indices = np.asarray(payload.get("scene_indices"))
    if (
        scene_indices.ndim != 1
        or scene_indices.size == 0
        or not np.all(scene_indices == int(scene))
    ):
        raise RuntimeError(f"scene-local {label} rows have an invalid scene identity")
    row_count = int(scene_indices.shape[0])
    for name, value in payload.items():
        if isinstance(value, np.ndarray) and (
            value.ndim < 1 or value.shape[0] != row_count
        ):
            raise RuntimeError(f"scene-local {label} field is not row aligned: {name}")
    return row_count


def _prepare_scene_inputs(
    model,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    scene: int,
    batch_size: int,
) -> _SceneInputs:
    """Materialize exactly one scene without changing probe batch semantics."""

    if type(scene) is not int or scene < 0:
        raise ValueError("scene-local evaluation scene must be a nonnegative integer")
    scene_values = np.asarray([scene], dtype=np.int64)
    routed = route_dataset(
        dataset,
        teacher,
        config,
        scene_values,
        normalization=route_normalization,
    )
    evaluated = legacy._compatibility_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        scene_values,
        active_only=False,
        route_normalization=route_normalization,
        routed=routed,
        batch_size=batch_size,
    )
    _require_scene_rows(evaluated, scene, "compatibility")

    response = legacy._response_probe_dataset(
        model,
        dataset,
        teacher,
        config,
        normalization,
        scene_values,
        active_only=False,
        route_normalization=route_normalization,
        routed=routed,
        batch_size=batch_size,
    )
    _require_scene_rows(response, scene, "response")
    source_target = np.asarray(response["source_targets"])
    target = np.asarray(response["targets"])
    if source_target.ndim != 2 or source_target.shape != target.shape:
        raise RuntimeError("scene-local response target shape is invalid")
    return _SceneInputs(
        scene=scene,
        routed=routed,
        compatibility=evaluated,
        response=response,
    )


def _split_checkpoint_prediction(
    values: np.ndarray,
    row_counts: tuple[int, ...],
    *,
    label: str,
) -> tuple[np.ndarray, ...]:
    expected_rows = sum(row_counts)
    result = np.asarray(values)
    if result.ndim < 1 or result.shape[0] != expected_rows:
        raise RuntimeError(
            f"checkpoint-global {label} row count is invalid: "
            f"expected {expected_rows}, observed {result.shape}"
        )
    offsets = np.cumsum((0, *row_counts), dtype=np.int64)
    return tuple(
        np.array(result[int(start) : int(stop)], copy=True)
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )


def _checkpoint_probe_prediction(
    probe,
    scenes: tuple[_SceneInputs, ...],
    feature_name: str,
    *,
    zero_feature_name: str | None = None,
    binary: bool = False,
    label: str,
) -> tuple[np.ndarray, ...]:
    """Run one probe per scene. Do not concatenate a checkpoint-global matrix."""

    if not scenes:
        return ()
    source = "compatibility" if binary else "response"
    predictions = []
    for scene in scenes:
        payload = getattr(scene, source)
        features = np.asarray(payload[feature_name])
        if features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError(f"checkpoint-global {label} features are invalid")
        if binary:
            predicted = np.asarray(predict_binary_probe(probe, features))
            expected_shape = (int(features.shape[0]),)
        else:
            zero = (
                np.asarray(payload[zero_feature_name])
                if zero_feature_name is not None
                else None
            )
            predicted = np.asarray(
                predict_response_probe(probe, features, zero)
            )
            target_width = np.asarray(payload["source_targets"]).shape[1:]
            expected_shape = (int(features.shape[0]), *target_width)
        if predicted.shape != expected_shape:
            raise RuntimeError(
                f"checkpoint-global {label} shape is invalid: "
                f"expected {expected_shape}, observed {predicted.shape}"
            )
        predictions.append(predicted)
    return tuple(predictions)


def _predict_checkpoint_scenes(
    probes: _ProbeBundle,
    scene_inputs: tuple[_SceneInputs, ...],
) -> dict[int, _SceneEvaluation]:
    """Preserve the legacy all-scene probe matrix while retaining scene shards."""

    if not scene_inputs:
        return {}
    scene_ids = tuple(item.scene for item in scene_inputs)
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("checkpoint-global prediction scenes must be unique")
    compatibility = _checkpoint_probe_prediction(
        probes.compatibility_probe,
        scene_inputs,
        "features",
        binary=True,
        label="compatibility prediction",
    )
    response = _checkpoint_probe_prediction(
        probes.response_probe,
        scene_inputs,
        "features",
        zero_feature_name="no_action_features",
        label="response prediction",
    )
    response_swap = _checkpoint_probe_prediction(
        probes.response_probe,
        scene_inputs,
        "action_swap_features",
        zero_feature_name="no_action_features",
        label="response action-swap prediction",
    )
    response_no_action = _checkpoint_probe_prediction(
        probes.response_probe,
        scene_inputs,
        "no_action_features",
        zero_feature_name="no_action_features",
        label="response no-action prediction",
    )
    contrast_variants = {"without_map", "edit_only", "oracle_x"}
    variants = {
        name: _checkpoint_probe_prediction(
            probe,
            scene_inputs,
            f"{name}_features",
            zero_feature_name=(
                f"{name}_zero_action_features"
                if name in contrast_variants
                else None
            ),
            label=f"response {name} prediction",
        )
        for name, probe in probes.variant_probes.items()
    }
    prepared: dict[int, _SceneEvaluation] = {}
    for index, item in enumerate(scene_inputs):
        source_target = np.asarray(item.response["source_targets"])
        expected_response_shape = source_target.shape
        compatibility_rows = int(np.asarray(item.compatibility["scene_indices"]).shape[0])
        probabilities = _require_prediction_shape(
            compatibility[index],
            (compatibility_rows,),
            "compatibility prediction",
        )
        prediction = source_target + _require_prediction_shape(
            response[index], expected_response_shape, "response prediction"
        )
        swap_prediction = source_target + _require_prediction_shape(
            response_swap[index],
            expected_response_shape,
            "response action-swap prediction",
        )
        no_action_prediction = source_target + _require_prediction_shape(
            response_no_action[index],
            expected_response_shape,
            "response no-action prediction",
        )
        variant_predictions = {
            name: source_target
            + _require_prediction_shape(
                values[index],
                expected_response_shape,
                f"response {name} prediction",
            )
            for name, values in variants.items()
        }
        prepared[item.scene] = _SceneEvaluation(
            scene=item.scene,
            routed=item.routed,
            compatibility=item.compatibility,
            compatibility_probabilities=probabilities,
            response=item.response,
            response_prediction=prediction,
            response_swap_prediction=swap_prediction,
            response_no_action_prediction=no_action_prediction,
            response_variant_predictions=variant_predictions,
        )
    return prepared


def _prepare_checkpoint_scene_evaluations(
    model,
    probes: _ProbeBundle,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    scenes: tuple[int, ...],
    batch_size: int,
) -> dict[int, _SceneEvaluation]:
    inputs = tuple(
        _prepare_scene_inputs(
            model,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene=scene,
            batch_size=batch_size,
        )
        for scene in scenes
    )
    return _predict_checkpoint_scenes(probes, inputs)


def _prepare_scene_evaluation(
    model,
    probes: _ProbeBundle,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    scene: int,
    batch_size: int,
) -> _SceneEvaluation:
    """Compatibility wrapper for one-scene tests and diagnostic callers."""

    prepared = _prepare_checkpoint_scene_evaluations(
        model,
        probes,
        dataset,
        teacher,
        config,
        normalization,
        route_normalization,
        scenes=(scene,),
        batch_size=batch_size,
    )
    return prepared[scene]


def _evaluate_scene(
    model,
    probes: _ProbeBundle,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    scene: int,
    seed: int,
    arm: str,
    batch_size: int,
    scene_evaluation: _SceneEvaluation,
) -> dict[str, list[dict]]:
    if not isinstance(scene_evaluation, _SceneEvaluation):
        raise TypeError("scene evaluation requires a scene-local prediction payload")
    prepared = scene_evaluation
    if prepared.scene != scene:
        raise RuntimeError("scene-local evaluation identity mismatch")
    evaluated = prepared.compatibility
    probabilities = prepared.compatibility_probabilities
    response_eval = prepared.response
    prediction = prepared.response_prediction
    swap_prediction = prepared.response_swap_prediction
    no_action_prediction = prepared.response_no_action_prediction
    variant_predictions = prepared.response_variant_predictions
    banks = sorted(set(np.asarray(evaluated["bank_ids"]).tolist()))
    expected_bank = str(dataset.bank_ids[scene])
    if banks != [expected_bank]:
        raise RuntimeError("streaming compatibility scene must contain exactly its bank")
    cgs_rows: list[dict] = []
    route_rows: list[dict] = []
    effect_rows: list[dict] = []
    shortcut_rows: list[dict] = []
    bank = expected_bank
    bank_mask = evaluated["bank_ids"] == bank
    active_mask = bank_mask & (evaluated["routes"] == "active")
    if not np.any(active_mask):
        raise RuntimeError(f"evaluation bank {bank!r} has no active CGS quartet")
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
            "cgs_auroc": binary_auroc(
                evaluated["labels"][active_mask], probabilities[active_mask]
            ),
            "native_energy_auroc": binary_auroc(
                evaluated["labels"][active_mask],
                evaluated["native_training_scores"][active_mask],
            ),
            "native_audit_hold_auroc": binary_auroc(
                evaluated["labels"][active_mask],
                evaluated["native_scores"][active_mask],
            ),
            "native_probe_spearman": legacy.spearman_correlation(
                evaluated["native_training_scores"][active_mask],
                probabilities[active_mask],
            ),
            "native_audit_hold_probe_spearman": legacy.spearman_correlation(
                evaluated["native_scores"][active_mask], probabilities[active_mask]
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

    response_effect_rows = legacy._response_effect_rows(
        seed,
        arm,
        response_eval,
        prediction,
        swap_prediction,
        no_action_prediction,
    )
    response_mask = (response_eval["bank_ids"] == bank) & (
        response_eval["routes"] == "active"
    )
    if not np.any(response_mask):
        raise RuntimeError(f"evaluation bank {bank!r} has no active response patches")
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
        np.asarray([prepared.scene], dtype=np.int64),
        bank,
        route_normalization=route_normalization,
        routed=prepared.routed,
        batch_size=batch_size,
    )
    response_rows = [
        {
            "seed": seed,
            "arm": arm,
            "bank_id": bank,
            "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
            "canonical_base_map_digest": dataset.canonical_base_map_digest(scene),
            "canonical_bank_digest": legacy._canonical_bank_digest(dataset, scene),
            "city_id": str(response_eval["city_ids"][response_mask][0]),
            "evaluation_scope": evaluation_scope,
            "unified_response_probe_active_patch_nmse": float(
                np.sum(
                    (prediction[response_mask] - response_eval["targets"][response_mask])
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
                    response_eval["wrong_action_match_status"][response_mask] == "exact"
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
    ]
    compatibility_effect_rows = legacy._compatibility_effect_rows(
        seed, arm, evaluated, probabilities
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


def _prepare_and_evaluate_scene(
    model,
    probes: _ProbeBundle,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    *,
    scene: int,
    seed: int,
    arm: str,
    batch_size: int,
) -> dict[str, list[dict]]:
    prepared = _prepare_scene_evaluation(
        model,
        probes,
        dataset,
        teacher,
        config,
        normalization,
        route_normalization,
        scene=scene,
        batch_size=batch_size,
    )
    try:
        return _evaluate_scene(
            model,
            probes,
            dataset,
            teacher,
            config,
            normalization,
            route_normalization,
            scene=scene,
            seed=seed,
            arm=arm,
            batch_size=batch_size,
            scene_evaluation=prepared,
        )
    finally:
        del prepared
        gc.collect()


def _cpu_state_dict(module) -> dict[str, object]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _linear_dimensions(module) -> tuple[int, int]:
    layers = [child for child in module.modules() if isinstance(child, torch.nn.Linear)]
    if not layers:
        raise RuntimeError("evaluation probe contains no linear layer")
    return int(layers[0].in_features), int(layers[-1].out_features)


def _compatibility_record(probe: CompatibilityProbe) -> dict[str, object]:
    input_dim, output_dim = _linear_dimensions(probe)
    if output_dim != 1:
        raise RuntimeError("compatibility probe output dimension is invalid")
    return {
        "kind": "compatibility",
        "input_dim": input_dim,
        "family": str(probe.family),
        "state_dict": _cpu_state_dict(probe),
    }


def _response_record(probe: ActionResponseProbe) -> dict[str, object]:
    input_dim, output_dim = _linear_dimensions(probe)
    return {
        "kind": "response",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "zero_preserving": bool(probe.zero_preserving),
        "state_dict": _cpu_state_dict(probe),
    }


def _probe_bundle_payload(
    probes: _ProbeBundle,
    config: dict,
    *,
    seed: int,
    arm: str,
) -> dict[str, object]:
    shortcuts = {}
    for name, record in probes.shortcut_probes.items():
        probe = record["probe"]
        shortcuts[name] = {
            "probe": _compatibility_record(probe) if probe is not None else None,
            "selected_family": str(record["selected_family"]),
            "source_train_auroc": float(record["source_train_auroc"]),
        }
    return {
        "schema_version": PROBE_BUNDLE_SCHEMA,
        "seed": int(seed),
        "arm": str(arm),
        "probe_hidden_dim": int(config["evaluation"]["probe_hidden_dim"]),
        "compatibility_probe": _compatibility_record(probes.compatibility_probe),
        "compatibility_selection": probes.compatibility_selection,
        "shortcut_probes": shortcuts,
        "response_probe": _response_record(probes.response_probe),
        "variant_probes": {
            name: _response_record(probe)
            for name, probe in probes.variant_probes.items()
        },
    }


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RuntimeError(f"persisted {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"persisted {label} must be finite")
    if minimum is not None and result < minimum:
        raise RuntimeError(f"persisted {label} is below its valid range")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"persisted {label} is above its valid range")
    return result


def _require_finite_probe_parameters(probe, label: str) -> None:
    for name, value in probe.state_dict().items():
        if not torch.is_tensor(value) or not bool(torch.all(torch.isfinite(value))):
            raise RuntimeError(f"persisted {label} parameter is invalid: {name}")


def _restore_compatibility_probe(record: object, hidden_dim: int) -> CompatibilityProbe:
    required = {"kind", "input_dim", "family", "state_dict"}
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("persisted compatibility probe fields are invalid")
    if (
        record["kind"] != "compatibility"
        or type(record["input_dim"]) is not int
        or record["input_dim"] < 1
        or record["family"] not in {"linear", "mlp2"}
        or not isinstance(record["state_dict"], Mapping)
    ):
        raise RuntimeError("persisted compatibility probe kind is invalid")
    probe = CompatibilityProbe(
        record["input_dim"], record["family"], int(hidden_dim)
    )
    probe.load_state_dict(record["state_dict"], strict=True)
    if _linear_dimensions(probe) != (record["input_dim"], 1):
        raise RuntimeError("persisted compatibility probe dimensions are invalid")
    _require_finite_probe_parameters(probe, "compatibility probe")
    probe.eval()
    return probe


def _restore_response_probe(record: object, hidden_dim: int) -> ActionResponseProbe:
    required = {
        "kind",
        "input_dim",
        "output_dim",
        "zero_preserving",
        "state_dict",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("persisted response probe fields are invalid")
    if (
        record["kind"] != "response"
        or type(record["input_dim"]) is not int
        or record["input_dim"] < 1
        or type(record["output_dim"]) is not int
        or record["output_dim"] < 1
        or type(record["zero_preserving"]) is not bool
        or not isinstance(record["state_dict"], Mapping)
    ):
        raise RuntimeError("persisted response probe identity is invalid")
    probe = ActionResponseProbe(
        record["input_dim"],
        record["output_dim"],
        int(hidden_dim),
    )
    probe.load_state_dict(record["state_dict"], strict=True)
    if _linear_dimensions(probe) != (record["input_dim"], record["output_dim"]):
        raise RuntimeError("persisted response probe dimensions are invalid")
    _require_finite_probe_parameters(probe, "response probe")
    probe.zero_preserving = record["zero_preserving"]
    probe.eval()
    return probe


def _restore_compatibility_selection(
    record: object, *, selected_probe_family: str
) -> dict[str, object]:
    required = {
        "selected_family",
        "selection_nll",
        "selection_auroc",
        "candidate_metrics",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("persisted compatibility selection fields are invalid")
    selected_family = record["selected_family"]
    if (
        selected_family not in {"linear", "mlp2"}
        or selected_family != selected_probe_family
    ):
        raise RuntimeError("persisted compatibility selection family is invalid")
    selection_nll = _finite_number(
        record["selection_nll"], "compatibility selection NLL", minimum=0.0
    )
    selection_auroc = _finite_number(
        record["selection_auroc"],
        "compatibility selection AUROC",
        minimum=0.0,
        maximum=1.0,
    )
    candidates = record["candidate_metrics"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise RuntimeError("persisted compatibility candidates are invalid")
    restored_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "family",
            "selection_nll",
            "selection_auroc",
        }:
            raise RuntimeError("persisted compatibility candidate fields are invalid")
        family = candidate["family"]
        if family not in {"linear", "mlp2"}:
            raise RuntimeError("persisted compatibility candidate family is invalid")
        restored_candidates.append(
            {
                "family": family,
                "selection_nll": _finite_number(
                    candidate["selection_nll"],
                    "compatibility candidate NLL",
                    minimum=0.0,
                ),
                "selection_auroc": _finite_number(
                    candidate["selection_auroc"],
                    "compatibility candidate AUROC",
                    minimum=0.0,
                    maximum=1.0,
                ),
            }
        )
    if {candidate["family"] for candidate in restored_candidates} != {
        "linear",
        "mlp2",
    } or restored_candidates != sorted(
        restored_candidates,
        key=lambda candidate: (candidate["selection_nll"], candidate["family"]),
    ):
        raise RuntimeError("persisted compatibility candidate order is invalid")
    selected = restored_candidates[0]
    if (
        selected["family"] != selected_family
        or selected["selection_nll"] != selection_nll
        or selected["selection_auroc"] != selection_auroc
    ):
        raise RuntimeError("persisted compatibility selection winner is inconsistent")
    return {
        "selected_family": selected_family,
        "selection_nll": selection_nll,
        "selection_auroc": selection_auroc,
        "candidate_metrics": restored_candidates,
    }


def _restore_probe_bundle(
    path: Path,
    config: dict,
    *,
    seed: int,
    arm: str,
    expected_response_output_dim: int | None = None,
) -> _ProbeBundle:
    if expected_response_output_dim is not None and (
        type(expected_response_output_dim) is not int
        or expected_response_output_dim < 1
    ):
        raise ValueError("expected response probe output dimension is invalid")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version",
        "seed",
        "arm",
        "probe_hidden_dim",
        "compatibility_probe",
        "compatibility_selection",
        "shortcut_probes",
        "response_probe",
        "variant_probes",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("persisted evaluation probe bundle fields are invalid")
    hidden_dim = int(config["evaluation"]["probe_hidden_dim"])
    if (
        payload["schema_version"] != PROBE_BUNDLE_SCHEMA
        or payload["seed"] != seed
        or payload["arm"] != arm
        or payload["probe_hidden_dim"] != hidden_dim
    ):
        raise RuntimeError("persisted evaluation probe bundle identity is stale")
    shortcut_payload = payload["shortcut_probes"]
    expected_shortcuts = tuple(legacy._alignment_shortcut_definitions())
    if not isinstance(shortcut_payload, dict) or tuple(shortcut_payload) != expected_shortcuts:
        raise RuntimeError("persisted shortcut probe contract is invalid")
    compatibility_probe = _restore_compatibility_probe(
        payload["compatibility_probe"], hidden_dim
    )
    compatibility_selection = _restore_compatibility_selection(
        payload["compatibility_selection"],
        selected_probe_family=compatibility_probe.family,
    )
    shortcuts: dict[str, dict] = {}
    for name in expected_shortcuts:
        record = shortcut_payload[name]
        if not isinstance(record, dict) or set(record) != {
            "probe",
            "selected_family",
            "source_train_auroc",
        }:
            raise RuntimeError("persisted shortcut probe fields are invalid")
        probe_record = record["probe"]
        selected_family = record["selected_family"]
        source_train_auroc = _finite_number(
            record["source_train_auroc"],
            f"{name} shortcut source-train AUROC",
            minimum=0.0,
            maximum=1.0,
        )
        if name == "constant":
            if probe_record is not None or selected_family != "constant":
                raise RuntimeError("persisted constant shortcut contract is invalid")
            restored_probe = None
        else:
            if probe_record is None or selected_family not in {"linear", "mlp2"}:
                raise RuntimeError(f"persisted {name} shortcut contract is invalid")
            restored_probe = _restore_compatibility_probe(probe_record, hidden_dim)
            if restored_probe.family != selected_family:
                raise RuntimeError(f"persisted {name} shortcut family is inconsistent")
        shortcuts[name] = {
            "probe": restored_probe,
            "selected_family": selected_family,
            "source_train_auroc": source_train_auroc,
        }
    variants = payload["variant_probes"]
    expected_variants = ("without_map", "edit_only", "csi_only", "oracle_x")
    if not isinstance(variants, dict) or tuple(variants) != expected_variants:
        raise RuntimeError("persisted response variant probe contract is invalid")
    response_probe = _restore_response_probe(payload["response_probe"], hidden_dim)
    if not response_probe.zero_preserving:
        raise RuntimeError("persisted main response probe must be zero-preserving")
    variant_probes = {
        name: _restore_response_probe(variants[name], hidden_dim)
        for name in expected_variants
    }
    expected_zero_preserving = {
        "without_map": True,
        "edit_only": True,
        "csi_only": False,
        "oracle_x": True,
    }
    output_dim = _linear_dimensions(response_probe)[1]
    if (
        expected_response_output_dim is not None
        and output_dim != expected_response_output_dim
    ):
        raise RuntimeError("persisted response probe output dimension is invalid")
    for name, probe in variant_probes.items():
        if (
            probe.zero_preserving is not expected_zero_preserving[name]
            or _linear_dimensions(probe)[1] != output_dim
        ):
            raise RuntimeError(
                f"persisted {name} response variant contract is invalid"
            )
    return _ProbeBundle(
        compatibility_probe=compatibility_probe,
        compatibility_selection=compatibility_selection,
        shortcut_probes=shortcuts,
        response_probe=response_probe,
        variant_probes=variant_probes,
    )


def _dataset_response_output_dim(dataset) -> int | None:
    """Resolve the frozen patch target width when a real dataset is supplied."""

    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise RuntimeError("evaluation dataset metadata is invalid")
    representation = metadata.get("representation")
    if not isinstance(representation, Mapping):
        raise RuntimeError("evaluation dataset representation metadata is invalid")
    patch_complex_size = representation.get("patch_complex_size")
    if type(patch_complex_size) is not int or patch_complex_size < 1:
        raise RuntimeError("evaluation dataset patch size metadata is invalid")
    return 2 * patch_complex_size


def _shard_identity(
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    *,
    substage: str,
    shard_id: str,
    range_start: int,
    range_stop: int,
) -> EvaluationShardIdentity:
    return EvaluationShardIdentity(
        run=run_identity,
        checkpoint_sha256=str(checkpoint["sha256"]),
        seed=int(checkpoint["seed"]),
        arm=str(checkpoint["arm"]),
        substage=substage,
        shard_id=shard_id,
        range_start=range_start,
        range_stop=range_stop,
    )


def _probe_state_identity(
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    checkpoint_index: int,
) -> EvaluationShardIdentity:
    return _shard_identity(
        run_identity,
        checkpoint,
        substage="probe_state",
        shard_id=f"checkpoint-{checkpoint_index:02d}",
        range_start=checkpoint_index,
        range_stop=checkpoint_index + 1,
    )


def _scene_bundle_identity(
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    scene: int,
) -> EvaluationShardIdentity:
    return _shard_identity(
        run_identity,
        checkpoint,
        substage="scene_bundle",
        shard_id=f"scene-{scene:03d}",
        range_start=scene,
        range_stop=scene + 1,
    )


def _table_identity(
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    table: str,
    scene: int,
) -> EvaluationShardIdentity:
    return _shard_identity(
        run_identity,
        checkpoint,
        substage=table.removesuffix(".csv"),
        shard_id=f"scene-{scene:03d}",
        range_start=scene,
        range_stop=scene + 1,
    )


def _contract_identity(
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    table: str,
    checkpoint_index: int,
) -> EvaluationShardIdentity:
    return _shard_identity(
        run_identity,
        checkpoint,
        substage=table.removesuffix(".csv"),
        shard_id=f"checkpoint-{checkpoint_index:02d}",
        range_start=checkpoint_index,
        range_stop=checkpoint_index + 1,
    )


def _finalization_identity(
    run_identity: EvaluationRunIdentity,
) -> EvaluationShardIdentity:
    return EvaluationShardIdentity(
        run=run_identity,
        checkpoint_sha256=run_identity.legacy_checkpoint_inventory_sha256,
        seed=0,
        arm="finalization",
        substage="finalization",
        shard_id="canonical-output",
        range_start=0,
        range_stop=1,
    )


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _commit_probe_bundle(
    store: EvaluationResumeStore,
    identity: EvaluationShardIdentity,
    probes: _ProbeBundle,
    config: dict,
) -> Path:
    payload = _probe_bundle_payload(
        probes,
        config,
        seed=identity.seed,
        arm=identity.arm,
    )
    result = store.commit_shard(
        identity,
        lambda handle: torch.save(payload, handle),
        suffix=".pt",
    )
    return result.payload_path


def _probe_contract_rows(
    probes: _ProbeBundle,
    config: dict,
    *,
    seed: int,
    arm: str,
) -> dict[str, list[dict]]:
    compatibility = [{"seed": seed, "arm": arm, **probes.compatibility_selection}]
    response = [
        {
            "seed": seed,
            "arm": arm,
            "probe": "main_masked_state_map_action_query",
            "steps": int(config["evaluation"]["probe_steps"]),
            "hidden_dim": int(config["evaluation"]["probe_hidden_dim"]),
        }
    ]
    for name in ("without_map", "edit_only", "csi_only", "oracle_x"):
        response.append(
            {
                "seed": seed,
                "arm": arm,
                "probe": name,
                "steps": int(config["evaluation"]["probe_steps"]),
                "hidden_dim": int(config["evaluation"]["probe_hidden_dim"]),
            }
        )
    return {
        "compatibility_probe_contract.csv": compatibility,
        "response_probe_contract.csv": response,
    }


def _scene_table_core_fields(table: str) -> tuple[str, ...]:
    fields = EVALUATION_TABLE_FIELDS[table]
    if fields[-len(CSV_EVIDENCE_FIELDS) :] != CSV_EVIDENCE_FIELDS:
        raise RuntimeError(f"evaluation table evidence suffix is invalid: {table}")
    return fields[: -len(CSV_EVIDENCE_FIELDS)]


def _csv_evidence_values(evidence: Mapping[str, object]) -> dict[str, str]:
    """Return the exact CSV spellings emitted for the authenticated evidence."""

    if not isinstance(evidence, Mapping):
        raise RuntimeError("evaluation CSV evidence must be a mapping")
    values: dict[str, str] = {}
    for field in CSV_EVIDENCE_FIELDS:
        if field not in evidence:
            raise RuntimeError(f"evaluation CSV evidence is missing: {field}")
        value = evidence[field]
        if isinstance(value, (dict, list, tuple, set)) or value is None:
            raise RuntimeError(f"evaluation CSV evidence value is not scalar: {field}")
        values[field] = str(value)
    return values


def _validate_scene_scalars(
    table: str, row: Mapping[str, object], *, persisted: bool
) -> None:
    """Reject malformed, non-finite, or out-of-domain evaluation metrics."""

    numeric_fields = set(EVALUATION_TABLE_FIELDS[table]) - SCENE_TEXT_FIELDS
    for field in numeric_fields:
        value = row[field]
        if value is None or (persisted and value == ""):
            if field not in OPTIONAL_SCENE_NUMERIC_FIELDS:
                raise RuntimeError(
                    f"evaluation required scalar is missing: {table}.{field}"
                )
            continue
        if field in SCENE_INTEGER_FIELDS:
            if persisted:
                if type(value) is not str:
                    raise RuntimeError(
                        f"persisted evaluation scalar is not text: {table}.{field}"
                    )
                try:
                    parsed_integer = int(value, 10)
                except ValueError as error:
                    raise RuntimeError(
                        f"persisted evaluation integer is invalid: {table}.{field}"
                    ) from error
                if str(parsed_integer) != value:
                    raise RuntimeError(
                        f"persisted evaluation integer is non-canonical: {table}.{field}"
                    )
            elif isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RuntimeError(
                    f"evaluation integer is invalid: {table}.{field}"
                )
            else:
                parsed_integer = int(value)
            if field != "wrong_action_world" and parsed_integer < 0:
                raise RuntimeError(
                    f"persisted evaluation count is negative: {table}.{field}"
                )
            continue
        if persisted and type(value) is not str:
            raise RuntimeError(
                f"persisted evaluation scalar is not text: {table}.{field}"
            )
        if not persisted and (
            isinstance(value, (bool, str, bytes))
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise RuntimeError(f"evaluation float is invalid: {table}.{field}")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"persisted evaluation float is invalid: {table}.{field}"
            ) from error
        if not math.isfinite(parsed):
            raise RuntimeError(
                f"persisted evaluation float is non-finite: {table}.{field}"
            )
        if field in SCENE_NONNEGATIVE_FLOAT_FIELDS and parsed < 0.0:
            raise RuntimeError(
                f"persisted evaluation nonnegative metric is invalid: {table}.{field}"
            )
        if field.endswith("_auroc") and not 0.0 <= parsed <= 1.0:
            raise RuntimeError(
                f"persisted evaluation AUROC is outside [0,1]: {table}.{field}"
            )
        if field.endswith("_spearman") and not -1.0 <= parsed <= 1.0:
            raise RuntimeError(
                f"persisted evaluation correlation is outside [-1,1]: {table}.{field}"
            )
        if field.endswith("_cosine") and not -1.0 <= parsed <= 1.0:
            raise RuntimeError(
                f"persisted evaluation cosine is outside [-1,1]: {table}.{field}"
            )
        if field.endswith(("_rate", "_fraction", "_accuracy")) and not (
            0.0 <= parsed <= 1.0
        ):
            raise RuntimeError(
                f"persisted evaluation rate is outside [0,1]: {table}.{field}"
            )
    minimum = row.get("physical_distance_min")
    maximum = row.get("physical_distance_max")
    missing = (None, "") if persisted else (None,)
    if minimum not in missing and maximum not in missing and float(minimum) > float(maximum):
        raise RuntimeError(
            f"persisted evaluation distance bounds are reversed: {table}"
        )


def _missing_scene_value(value: object, *, persisted: bool) -> bool:
    """Return whether a scene scalar is represented as protocol N/A."""

    return value == "" if persisted else value is None


def _require_scene_value(
    row: Mapping[str, object],
    field: str,
    *,
    expect_missing: bool,
    table: str,
    persisted: bool,
) -> None:
    missing = _missing_scene_value(row[field], persisted=persisted)
    if missing != expect_missing:
        state = "must be missing" if expect_missing else "is missing"
        raise RuntimeError(f"evaluation conditional scalar {state}: {table}.{field}")


def _validate_persisted_optional_fields(
    table: str, row: Mapping[str, object], *, persisted: bool
) -> None:
    """Allow only protocol-defined N/A fields and enforce their condition guards.

    The same condition matrix is applied before CSV serialization and after CSV
    deserialization.  This prevents a fresh in-memory row with an impossible
    blank metric from becoming an apparently valid persisted fragment.
    """

    def require(field: str, *, expect_missing: bool) -> None:
        _require_scene_value(
            row,
            field,
            expect_missing=expect_missing,
            table=table,
            persisted=persisted,
        )

    def integer(field: str) -> int:
        value = row[field]
        if persisted:
            if not isinstance(value, str):
                raise RuntimeError(f"evaluation condition integer is not text: {table}.{field}")
            try:
                return int(value, 10)
            except ValueError as error:
                raise RuntimeError(
                    f"evaluation condition integer is invalid: {table}.{field}"
                ) from error
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(f"evaluation condition integer is invalid: {table}.{field}")
        return int(value)

    if table == "compatibility_route_distributions.csv":
        status = row.get("condition_status")
        route = str(row.get("route", ""))
        if status not in {"MISSING", "ASSESSED"}:
            raise RuntimeError(f"evaluation route condition status is invalid: {table}")
        if status == "MISSING":
            if integer("pair_count") != 0:
                raise RuntimeError(f"missing route must have zero pair count: {table}")
            for field in (
                "matched_minus_alternative_mean",
                "matched_minus_alternative_median",
                "absolute_difference_p90",
                "overclassification_rate",
            ):
                require(field, expect_missing=True)
        else:
            if integer("pair_count") <= 0:
                raise RuntimeError(f"assessed route must have positive pair count: {table}")
            for field in (
                "matched_minus_alternative_mean",
                "matched_minus_alternative_median",
                "absolute_difference_p90",
            ):
                require(field, expect_missing=False)
            require("overclassification_rate", expect_missing=route != "null")
        return

    if table == "response_pair_effects.csv":
        status = row.get("wrong_action_match_status")
        if status not in {"exact", "fallback", "failed"}:
            raise RuntimeError(f"evaluation response match status is invalid: {table}")
        exact = status == "exact"
        require("action_swap_mse", expect_missing=not exact)
        require("response_advantage_vs_action_swap", expect_missing=not exact)
        return

    if table == "response_per_bank.csv":
        probe_exact = integer("probe_action_swap_exact_count")
        native_exact = integer("native_action_swap_exact_count")
        for field in (
            "unified_response_probe_action_swap_exact_patch_nmse",
            "probe_action_swap_active_patch_nmse",
        ):
            require(field, expect_missing=probe_exact == 0)
        for field in (
            "native_target_free_action_swap_exact_full_channel_nmse",
            "native_action_swap_full_channel_nmse",
            "native_latent_action_swap_exact_target_nmse",
            "native_latent_action_swap_nmse",
        ):
            require(field, expect_missing=native_exact == 0)
        null_count = integer("native_null_patch_count")
        for field in (
            "native_null_delta_rms_mean",
            "native_null_violation_rate",
            "native_latent_null_delta_rms_mean",
            "native_latent_null_violation_rate",
        ):
            require(field, expect_missing=null_count == 0)
        return

    # Every other numeric field, including pair coordinates and identifiers, is required.
    for field in set(EVALUATION_TABLE_FIELDS[table]) - SCENE_TEXT_FIELDS:
        if field == "native_transition_skill":
            continue
        require(field, expect_missing=False)


def _pair_id_integer(value: str, *, table: str, field: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise RuntimeError(f"evaluation pair ID is invalid: {table}.{field}") from error
    if str(parsed) != value or parsed < 0:
        raise RuntimeError(f"evaluation pair ID is non-canonical: {table}.{field}")
    return parsed


def _validate_pair_row_identity(table: str, row: Mapping[str, object]) -> None:
    pair_id = str(row["pair_id"])
    parts = pair_id.rsplit(":", 4)
    if len(parts) != 5 or not parts[0] or parts[0] != str(row["bank_id"]):
        raise RuntimeError(f"evaluation pair ID bank identity is invalid: {table}")
    first = _pair_id_integer(parts[1], table=table, field="first_world")
    second = _pair_id_integer(parts[2], table=table, field="second_world")
    position = _pair_id_integer(parts[3], table=table, field="position_index")
    final = _pair_id_integer(parts[4], table=table, field="final_index")
    source = int(row["source_world"])
    target = int(row["target_world"])
    if source < 0 or target < 0 or source == target or position != int(row["position_index"]):
        raise RuntimeError(f"evaluation pair ID coordinate identity is invalid: {table}")
    if table == "compatibility_pair_effects.csv":
        if (
            first >= second
            or (first, second) != (min(source, target), max(source, target))
            or final != source
        ):
            raise RuntimeError(
                f"evaluation compatibility pair ID identity is invalid: {table}"
            )
    elif table == "response_pair_effects.csv":
        if (first, second, final) != (source, target, int(row["query_index"])):
            raise RuntimeError(
                f"evaluation response pair ID identity is invalid: {table}"
            )
    else:
        raise RuntimeError(f"evaluation pair ID table is unsupported: {table}")


def _canonical_pair_integer(value: object, *, table: str, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"evaluation pair inventory integer is invalid: {table}.{field}")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError as error:
            raise RuntimeError(
                f"evaluation pair inventory integer is invalid: {table}.{field}"
            ) from error
        if str(parsed) != value:
            raise RuntimeError(
                f"evaluation pair inventory integer is non-canonical: {table}.{field}"
            )
    else:
        raise RuntimeError(f"evaluation pair inventory integer is invalid: {table}.{field}")
    return parsed


def _pair_identity_record(table: str, row: Mapping[str, object]) -> dict[str, object]:
    fields = [
        "pair_id",
        "scene_index",
        "bank_id",
        "source_world",
        "target_world",
        "position_index",
    ]
    if table == "response_pair_effects.csv":
        fields.append("query_index")
    if table not in PAIR_SCENE_TABLES or any(field not in row for field in fields):
        raise RuntimeError(f"evaluation pair inventory fields are invalid: {table}")
    record: dict[str, object] = {
        "pair_id": str(row["pair_id"]),
        "scene_index": _canonical_pair_integer(
            row["scene_index"], table=table, field="scene_index"
        ),
        "bank_id": str(row["bank_id"]),
        "source_world": _canonical_pair_integer(
            row["source_world"], table=table, field="source_world"
        ),
        "target_world": _canonical_pair_integer(
            row["target_world"], table=table, field="target_world"
        ),
        "position_index": _canonical_pair_integer(
            row["position_index"], table=table, field="position_index"
        ),
    }
    if table == "response_pair_effects.csv":
        record["query_index"] = _canonical_pair_integer(
            row["query_index"], table=table, field="query_index"
        )
    return record


def _pair_inventory_summary(
    table: str, rows: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Summarize pair identities without duplicating the full pair table."""

    digest = hashlib.sha256()
    row_count = 0
    previous_pair_id: str | None = None
    for row in rows:
        record = _pair_identity_record(table, row)
        pair_id = str(record["pair_id"])
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise RuntimeError(
                f"evaluation pair inventory order is not canonical: {table}"
            )
        previous_pair_id = pair_id
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest.update(encoded)
        digest.update(b"\n")
        row_count += 1
    if row_count == 0:
        raise RuntimeError(f"evaluation pair inventory is empty: {table}")
    return {
        "schema_version": PAIR_INVENTORY_SCHEMA,
        "row_count": row_count,
        "sha256": digest.hexdigest(),
    }


def _pair_inventory(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]]
) -> dict[str, dict[str, object]]:
    return {
        table: _pair_inventory_summary(table, rows_by_table[table])
        for table in PAIR_SCENE_TABLES
    }


def _dataset_pair_inventory(
    pair_universe: ScenePairUniverse,
) -> dict[str, dict[str, object]]:
    """Return the compact, dataset-derived identity for both pair tables."""

    if not isinstance(pair_universe, ScenePairUniverse):
        raise TypeError("dataset pair inventory requires a ScenePairUniverse")
    result: dict[str, dict[str, object]] = {}
    for table in PAIR_SCENE_TABLES:
        summary = pair_universe.summary(table)
        result[table] = {
            "schema_version": PAIR_INVENTORY_SCHEMA,
            "row_count": int(summary["row_count"]),
            "sha256": str(summary["sha256"]),
            "identity_schema": str(summary["schema"]),
        }
    return result


def _validate_pair_inventory_against_dataset(
    table: str,
    observed: Mapping[str, object],
    pair_universe: ScenePairUniverse | None,
) -> None:
    """Reject a pair inventory that is not independently derived from dataset metadata."""

    if pair_universe is None:
        return
    if not isinstance(pair_universe, ScenePairUniverse):
        raise RuntimeError("evaluation dataset pair universe is invalid")
    expected = _dataset_pair_inventory(pair_universe).get(table)
    if expected is None or dict(observed) != expected:
        raise RuntimeError(
            f"evaluation pair inventory differs from the dataset: {table}"
        )


def _validate_scene_row(
    table: str,
    row: object,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    persisted: bool,
    evidence: Mapping[str, object] | None = None,
) -> None:
    """Validate one row without imposing a table-level cardinality/order rule."""

    if table not in SCENE_TABLES or not isinstance(row, dict):
        raise RuntimeError("evaluation scene row is invalid")
    expected_fields = (
        EVALUATION_TABLE_FIELDS[table]
        if persisted
        else _scene_table_core_fields(table)
    )
    if set(row) != set(expected_fields):
        raise RuntimeError(f"evaluation scene table row fields are invalid: {table}")
    expected_evidence = (
        _csv_evidence_values(evidence) if persisted and evidence is not None else None
    )
    if persisted and expected_evidence is None:
        raise RuntimeError("persisted evaluation scene rows require evidence")
    if expected_evidence is not None:
        for field, expected_value in expected_evidence.items():
            if row.get(field) != expected_value:
                raise RuntimeError(
                    f"evaluation scene table evidence identity is invalid: {table}"
                )
    _validate_scene_scalars(table, row, persisted=persisted)
    expected_seed = str(int(checkpoint["seed"]))
    expected_arm = str(checkpoint["arm"])
    expected_bank = str(bank_id)
    if str(row["seed"]) != expected_seed:
        raise RuntimeError(f"evaluation scene table seed identity is invalid: {table}")
    if str(row["arm"]) != expected_arm:
        raise RuntimeError(f"evaluation scene table arm identity is invalid: {table}")
    if str(row["bank_id"]) != expected_bank:
        raise RuntimeError(f"evaluation scene table bank identity is invalid: {table}")
    if table in PAIR_SCENE_TABLES and str(row["scene_index"]) != str(int(scene)):
        raise RuntimeError(f"evaluation scene table scene identity is invalid: {table}")
    if table in PAIR_SCENE_TABLES:
        _validate_pair_row_identity(table, row)
    _validate_persisted_optional_fields(table, row, persisted=persisted)


def _validate_scene_table_rows(
    table: str,
    rows: object,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    persisted: bool,
    evidence: Mapping[str, object] | None = None,
    pair_universe: ScenePairUniverse | None = None,
) -> dict[str, object] | None:
    if table not in SCENE_TABLES or not isinstance(rows, list):
        raise RuntimeError("evaluation scene table rows are invalid")
    if table in PAIR_SCENE_TABLES:
        if not rows:
            raise RuntimeError(f"evaluation scene pair table is empty: {table}")
    elif table in {"cgs_per_bank.csv", "response_per_bank.csv"}:
        if len(rows) != 1:
            raise RuntimeError(
                f"evaluation scene table must contain exactly one bank row: {table}"
            )
    expected_sequences = {
        "alignment_shortcut_baselines.csv": (
            "baseline",
            tuple(legacy._alignment_shortcut_definitions()),
        ),
        "compatibility_route_distributions.csv": (
            "route",
            tuple(str(value) for value in ROUTE_NAMES.tolist()),
        ),
        "cgs_active_effect_bins.csv": (
            "effect_bin",
            ("low", "medium_low", "medium_high", "high"),
        ),
    }
    sequence = expected_sequences.get(table)
    if sequence is not None:
        field, expected_values = sequence
        observed_values = tuple(
            str(row.get(field, "")) if isinstance(row, dict) else "" for row in rows
        )
        if observed_values != expected_values:
            raise RuntimeError(
                f"evaluation scene table row order or count is invalid: {table}"
            )
    pair_validator = (
        pair_universe.validator(table)
        if table in PAIR_SCENE_TABLES and pair_universe is not None
        else None
    )
    for row in rows:
        _validate_scene_row(
            table,
            row,
            checkpoint,
            scene=scene,
            bank_id=bank_id,
            persisted=persisted,
            evidence=evidence,
        )
        if pair_validator is not None:
            pair_validator.feed(row)
    if table in PAIR_SCENE_TABLES:
        pair_ids = [str(row["pair_id"]) for row in rows]
        if (
            any(not pair_id for pair_id in pair_ids)
            or len(pair_ids) != len(set(pair_ids))
            or pair_ids != sorted(pair_ids)
        ):
            raise RuntimeError(
                f"evaluation scene pair IDs must be unique canonical text order: {table}"
            )
        if pair_validator is not None:
            summary = pair_validator.finish()
            observed = {
                "schema_version": PAIR_INVENTORY_SCHEMA,
                "row_count": int(summary["row_count"]),
                "sha256": str(summary["sha256"]),
                "identity_schema": str(summary["schema"]),
            }
            _validate_pair_inventory_against_dataset(
                table, observed, pair_universe
            )
            return observed
    return None


def _validate_scene_rows(
    rows_by_table: object,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    tables: tuple[str, ...],
    persisted: bool,
    evidence: Mapping[str, object] | None = None,
    pair_universe: ScenePairUniverse | None = None,
) -> dict[str, dict[str, object]]:
    if not isinstance(rows_by_table, dict) or set(rows_by_table) != set(tables):
        raise RuntimeError("evaluation scene table inventory is invalid")
    pair_inventory: dict[str, dict[str, object]] = {}
    for table in tables:
        summary = _validate_scene_table_rows(
            table,
            rows_by_table[table],
            checkpoint,
            scene=scene,
            bank_id=bank_id,
            persisted=persisted,
            evidence=evidence,
            pair_universe=pair_universe,
        )
        if summary is not None:
            pair_inventory[table] = summary
    return pair_inventory


def _fragment_fingerprint(
    store: EvaluationResumeStore,
    rows: list[dict],
    *,
    fields: tuple[str, ...],
    evidence: Mapping[str, object],
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(mode="w+b", dir=store.root) as expected:
        row_count = write_csv_fragment_stream(
            expected,
            rows,
            fieldnames=fields,
            evidence=evidence,
        )
        expected.flush()
        byte_count = expected.tell()
        expected.seek(0)
        for chunk in iter(lambda: expected.read(1024 * 1024), b""):
            digest.update(chunk)
    return row_count, byte_count, digest.hexdigest()


def _read_fragment_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with gzip.open(path, mode="rt", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle, dialect="excel", strict=True)
            if tuple(reader.fieldnames or ()) != fields:
                raise RuntimeError("evaluation fragment header differs from its table")
            rows = list(reader)
    except (csv.Error, EOFError, OSError, UnicodeError) as error:
        raise RuntimeError(f"evaluation fragment cannot be decoded: {path}") from error
    if any(None in row or None in row.values() for row in rows):
        raise RuntimeError(f"evaluation fragment row width is invalid: {path}")
    return rows


def _visit_validated_scene_fragment(
    path: Path,
    table: str,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    evidence: Mapping[str, object],
    pair_universe: ScenePairUniverse | None,
):
    """Authenticate one scene fragment with bounded pair-table memory."""

    retained_rows: list[dict[str, str]] | None = (
        [] if table in SMALL_SCENE_TABLES or pair_universe is None else None
    )
    pair_validator = (
        pair_universe.validator(table)
        if table in PAIR_SCENE_TABLES and pair_universe is not None
        else None
    )

    def validate(row: dict[str, str]) -> None:
        _validate_scene_row(
            table,
            row,
            checkpoint,
            scene=scene,
            bank_id=bank_id,
            persisted=True,
            evidence=evidence,
        )
        if pair_validator is not None:
            pair_validator.feed(row)
        if retained_rows is not None:
            retained_rows.append(row)

    receipt = visit_csv_fragment_rows(
        path,
        fieldnames=EVALUATION_TABLE_FIELDS[table],
        visitor=validate,
    )
    if pair_validator is not None:
        summary = pair_validator.finish()
        observed_inventory = {
            "schema_version": PAIR_INVENTORY_SCHEMA,
            "row_count": int(summary["row_count"]),
            "sha256": str(summary["sha256"]),
            "identity_schema": str(summary["schema"]),
        }
        _validate_pair_inventory_against_dataset(
            table, observed_inventory, pair_universe
        )
    else:
        if retained_rows is None:
            raise RuntimeError("evaluation fragment validation lost its rows")
        observed_inventory = _validate_scene_table_rows(
            table,
            retained_rows,
            checkpoint,
            scene=scene,
            bank_id=bank_id,
            persisted=True,
            evidence=evidence,
        )
        if table in PAIR_SCENE_TABLES:
            observed_inventory = _pair_inventory_summary(table, retained_rows)
    return receipt, retained_rows, observed_inventory


def _commit_table_fragment(
    store: EvaluationResumeStore,
    identity: EvaluationShardIdentity,
    rows: list[dict],
    *,
    table: str,
    evidence: Mapping[str, object],
    repair_reused_mismatch: bool = False,
    repair_reused_malformed: bool = False,
):
    if (
        type(repair_reused_mismatch) is not bool
        or type(repair_reused_malformed) is not bool
    ):
        raise TypeError("fragment repair policy must be boolean")
    fields = EVALUATION_TABLE_FIELDS[table]
    result = store.load_or_quarantine_shard(identity, suffix=".csv.gz")
    if result is None:
        result = store.commit_shard(
            identity,
            lambda handle: write_csv_fragment_stream(
                handle,
                rows,
                fieldnames=fields,
                evidence=evidence,
            ),
            suffix=".csv.gz",
        )
    try:
        receipt = inspect_csv_fragment(result.payload_path, fieldnames=fields)
    except CsvFragmentError as error:
        if not result.reused or not (
            repair_reused_malformed or repair_reused_mismatch
        ):
            raise
        store.quarantine_completed_shard(
            identity,
            suffix=".csv.gz",
            reason=(
                "persisted derived evaluation fragment failed CSV validation; "
                f"recomputing the isolated shard: {table} "
                f"seed={identity.seed} arm={identity.arm} shard={identity.shard_id}; "
                f"{type(error).__name__}"
            ),
        )
        result = store.commit_shard(
            identity,
            lambda handle: write_csv_fragment_stream(
                handle,
                rows,
                fieldnames=fields,
                evidence=evidence,
            ),
            suffix=".csv.gz",
        )
        receipt = inspect_csv_fragment(result.payload_path, fieldnames=fields)
    if result.reused:
        expected_rows, expected_bytes, expected_sha256 = _fragment_fingerprint(
            store,
            rows,
            fields=fields,
            evidence=evidence,
        )
        if (
            expected_rows != receipt.row_count
            or expected_bytes != receipt.bytes
            or expected_sha256 != receipt.sha256
        ):
            if not repair_reused_mismatch:
                raise RuntimeError(
                    "persisted evaluation fragment differs from deterministic "
                    f"replay: {table} seed={identity.seed} arm={identity.arm} "
                    f"shard={identity.shard_id}"
                )
            store.quarantine_completed_shard(
                identity,
                suffix=".csv.gz",
                reason=(
                    "persisted derived evaluation fragment failed deterministic "
                    "semantic validation; recomputing the isolated shard: "
                    f"{table} seed={identity.seed} arm={identity.arm} "
                    f"shard={identity.shard_id}"
                ),
            )
            result = store.commit_shard(
                identity,
                lambda handle: write_csv_fragment_stream(
                    handle,
                    rows,
                    fieldnames=fields,
                    evidence=evidence,
                ),
                suffix=".csv.gz",
            )
            receipt = inspect_csv_fragment(result.payload_path, fieldnames=fields)
    return result, receipt


def _commit_scene_bundle(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    rows_by_table: dict[str, list[dict]],
    evidence: Mapping[str, object],
    pair_universe: ScenePairUniverse | None = None,
) -> dict[str, object]:
    validated_pair_inventory = _validate_scene_rows(
        rows_by_table,
        checkpoint,
        scene=scene,
        bank_id=bank_id,
        tables=SCENE_TABLES,
        persisted=False,
        pair_universe=pair_universe,
    )
    table_receipts: dict[str, dict[str, object]] = {}
    for table in SCENE_TABLES:
        result, receipt = _commit_table_fragment(
            store,
            _table_identity(run_identity, checkpoint, table, scene),
            rows_by_table[table],
            table=table,
            evidence=evidence,
            repair_reused_malformed=True,
        )
        table_receipts[table] = {
            "path": result.payload_path.relative_to(store.root).as_posix(),
            "manifest_path": result.manifest_path.relative_to(store.root).as_posix(),
            "bytes": receipt.bytes,
            "sha256": receipt.sha256,
            "row_count": receipt.row_count,
            "fieldnames": list(receipt.fieldnames),
        }
    small_rows = {table: rows_by_table[table] for table in SMALL_SCENE_TABLES}
    bundle = {
        "schema_version": SCENE_BUNDLE_SCHEMA,
        "seed": int(checkpoint["seed"]),
        "arm": str(checkpoint["arm"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "scene": int(scene),
        "bank_id": str(bank_id),
        "tables": table_receipts,
        "pair_inventory": (
            validated_pair_inventory
            if pair_universe is not None
            else _pair_inventory(rows_by_table)
        ),
        "small_rows": small_rows,
    }
    bundle_identity = _scene_bundle_identity(run_identity, checkpoint, scene)
    encoded_bundle = _json_bytes(bundle)
    completed_bundle = store.load_or_quarantine_shard(
        bundle_identity, suffix=".json"
    )
    if completed_bundle is None:
        store.commit_shard_bytes(
            bundle_identity,
            encoded_bundle,
            suffix=".json",
        )
    elif completed_bundle.payload_path.read_bytes() != encoded_bundle:
        raise RuntimeError(
            "resumed evaluation scene bundle differs from deterministic recomputation"
        )
    return bundle


def _preflight_scene_table_fragments(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    evidence: Mapping[str, object],
    pair_universe: ScenePairUniverse | None = None,
) -> None:
    """Authenticate surviving table fragments before rebuilding a missing bundle.

    A missing bundle is recoverable state, but a malformed table must not be
    allowed to reach deterministic replay.  Quarantine only the malformed
    table; valid siblings remain reusable and replay mismatches still fail
    closed in ``_commit_table_fragment``.
    """

    for table in SCENE_TABLES:
        table_identity = _table_identity(run_identity, checkpoint, table, scene)
        shard = store.load_or_quarantine_shard(table_identity, suffix=".csv.gz")
        if shard is None:
            continue
        try:
            _visit_validated_scene_fragment(
                shard.payload_path,
                table,
                checkpoint,
                scene=scene,
                bank_id=bank_id,
                evidence=evidence,
                pair_universe=pair_universe,
            )
        except StaleResumeError:
            raise
        except (CsvFragmentError, RuntimeError, StrictJsonError, TypeError, ValueError) as error:
            store.quarantine_completed_shard(
                table_identity,
                suffix=".csv.gz",
                reason=(
                    "persisted evaluation table failed preflight while scene bundle "
                    "was missing; recomputing isolated table: "
                    f"{table} ({type(error).__name__})"
                ),
            )


def _load_scene_bundle(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    *,
    scene: int,
    bank_id: str,
    evidence: Mapping[str, object],
    pair_universe: ScenePairUniverse | None = None,
) -> dict[str, object] | None:
    bundle_identity = _scene_bundle_identity(run_identity, checkpoint, scene)
    result = store.load_or_quarantine_shard(bundle_identity, suffix=".json")
    if result is None:
        _preflight_scene_table_fragments(
            store,
            run_identity,
            checkpoint,
            scene=scene,
            bank_id=bank_id,
            evidence=evidence,
            pair_universe=pair_universe,
        )
        return None
    try:
        payload = read_strict_json(result.payload_path)
        required = {
            "schema_version",
            "seed",
            "arm",
            "checkpoint_sha256",
            "scene",
            "bank_id",
            "tables",
            "pair_inventory",
            "small_rows",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise _SceneBundleSemanticError(
                "evaluation scene bundle fields are invalid"
            )
        if (
            payload["schema_version"] != SCENE_BUNDLE_SCHEMA
            or payload["seed"] != int(checkpoint["seed"])
            or payload["arm"] != str(checkpoint["arm"])
            or payload["checkpoint_sha256"] != str(checkpoint["sha256"])
            or payload["scene"] != int(scene)
            or payload["bank_id"] != str(bank_id)
        ):
            raise _SceneBundleSemanticError(
                "evaluation scene bundle identity is stale"
            )
        tables = payload["tables"]
        if not isinstance(tables, dict) or tuple(tables) != tuple(
            sorted(SCENE_TABLES)
        ):
            # JSON keys are sorted on disk, so validate as a set while retaining
            # canonical merge order.
            if not isinstance(tables, dict) or set(tables) != set(SCENE_TABLES):
                raise _SceneBundleSemanticError(
                    "evaluation scene bundle table inventory is invalid"
                )
        small_rows = payload["small_rows"]
        pair_inventory = payload["pair_inventory"]
        expected_pair_inventory: dict[str, dict[str, object]] = {}
        dataset_pair_inventory = (
            _dataset_pair_inventory(pair_universe)
            if pair_universe is not None
            else None
        )
        if not isinstance(pair_inventory, dict) or set(pair_inventory) != set(PAIR_SCENE_TABLES):
            raise _SceneBundleSemanticError("evaluation scene pair inventory is invalid")
        for table in PAIR_SCENE_TABLES:
            summary = pair_inventory[table]
            expected_fields = {
                "schema_version",
                "row_count",
                "sha256",
                *(('identity_schema',) if pair_universe is not None else ()),
            }
            if not isinstance(summary, dict) or set(summary) != expected_fields:
                raise _SceneBundleSemanticError(
                    "evaluation scene pair inventory summary is invalid"
                )
            if (
                summary["schema_version"] != PAIR_INVENTORY_SCHEMA
                or type(summary["row_count"]) is not int
                or summary["row_count"] <= 0
                or not _valid_sha256(summary["sha256"])
            ):
                raise _SceneBundleSemanticError(
                    "evaluation scene pair inventory summary identity is invalid"
                )
            if (
                dataset_pair_inventory is not None
                and summary != dataset_pair_inventory[table]
            ):
                raise _SceneBundleSemanticError(
                    "evaluation scene pair inventory differs from the dataset"
                )
            expected_pair_inventory[table] = summary
        try:
            _validate_scene_rows(
                small_rows,
                checkpoint,
                scene=scene,
                bank_id=bank_id,
                tables=SMALL_SCENE_TABLES,
                persisted=False,
            )
        except RuntimeError as error:
            raise _SceneBundleSemanticError(str(error)) from error
        for table in SCENE_TABLES:
            table_identity = _table_identity(
                run_identity, checkpoint, table, scene
            )
            shard = store.load_or_quarantine_shard(
                table_identity,
                suffix=".csv.gz",
            )
            if shard is None:
                raise _SceneBundleSemanticError(
                    "evaluation scene bundle references a missing table shard"
                )
            try:
                receipt, persisted_rows, observed_inventory = (
                    _visit_validated_scene_fragment(
                        shard.payload_path,
                        table,
                        checkpoint,
                        scene=scene,
                        bank_id=bank_id,
                        evidence=evidence,
                        pair_universe=pair_universe,
                    )
                )
                if table in PAIR_SCENE_TABLES:
                    if observed_inventory != expected_pair_inventory[table]:
                        raise _SceneBundleSemanticError(
                            "evaluation scene pair inventory differs from table",
                            table_identity=table_identity,
                        )
                elif persisted_rows is None:
                    raise _SceneBundleSemanticError(
                        f"evaluation scene small rows were not retained: {table}",
                        table_identity=table_identity,
                    )
            except RuntimeError as error:
                raise _SceneBundleSemanticError(
                    f"evaluation scene table failed semantic validation: {table}",
                    table_identity=table_identity,
                ) from error
            recorded = tables[table]
            if not isinstance(recorded, dict) or recorded != {
                "path": shard.payload_path.relative_to(store.root).as_posix(),
                "manifest_path": shard.manifest_path.relative_to(store.root).as_posix(),
                "bytes": receipt.bytes,
                "sha256": receipt.sha256,
                "row_count": receipt.row_count,
                "fieldnames": list(receipt.fieldnames),
            }:
                raise _SceneBundleSemanticError(
                    "evaluation scene bundle table receipt mismatch",
                    table_identity=table_identity,
                )
            if table in SMALL_SCENE_TABLES:
                if persisted_rows is None:
                    raise _SceneBundleSemanticError(
                        f"evaluation scene small rows were not retained: {table}",
                        table_identity=table_identity,
                    )
                expected_rows, expected_bytes, expected_sha256 = _fragment_fingerprint(
                    store,
                    small_rows[table],
                    fields=EVALUATION_TABLE_FIELDS[table],
                    evidence=evidence,
                )
                if (
                    expected_rows != receipt.row_count
                    or expected_bytes != receipt.bytes
                    or expected_sha256 != receipt.sha256
                ):
                    raise _SceneBundleSemanticError(
                        f"evaluation scene small rows differ from fragment: {table}",
                        table_identity=table_identity,
                    )
        return payload
    except StaleResumeError:
        raise
    except (RuntimeError, StrictJsonError, TypeError, ValueError) as error:
        table_identity = (
            error.table_identity
            if isinstance(error, _SceneBundleSemanticError)
            else None
        )
        if table_identity is not None:
            store.quarantine_completed_shard(
                table_identity,
                suffix=".csv.gz",
                reason=(
                    "persisted evaluation table failed semantic validation: "
                    f"{type(error).__name__}"
                ),
            )
        store.quarantine_completed_shard(
            bundle_identity,
            suffix=".json",
            reason=(
                "persisted evaluation scene bundle failed semantic validation: "
                f"{type(error).__name__}"
            ),
        )
        return None


def _validate_runner_identity(
    run_identity: EvaluationRunIdentity,
    evidence: Mapping[str, object],
    *,
    qualification_gate_path: Path,
    factorial_gate_path: Path,
    checkpoint_inventory_sha256: str,
) -> None:
    if not _valid_sha256(checkpoint_inventory_sha256):
        raise RuntimeError("authenticated checkpoint inventory SHA-256 is invalid")
    expected = {
        "source_tree_sha256": evidence["source_tree_sha256"],
        "config_sha256": evidence["config_sha256"],
        "dataset_sha256": evidence["dataset_sha256"],
        "runtime_provenance_sha256": evidence["runtime_provenance_sha256"],
        "qualification_gate_sha256": sha256_file(qualification_gate_path),
        "factorial_gate_sha256": sha256_file(factorial_gate_path),
        "legacy_checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "output_schema_sha256": evaluation_output_schema_sha256(),
    }
    actual = run_identity.as_dict()
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(f"streaming evaluation run identity {key} mismatch")
    if run_identity.output_schema_id != STREAMING_EVALUATION_SCHEMA:
        raise RuntimeError("streaming evaluation output schema ID mismatch")


def _frozen_evaluation_scenes(dataset) -> tuple[int, ...]:
    """Return the exact scene order used by the frozen evaluation matrix."""

    values = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    scenes = tuple(int(scene) for scene in values)
    if len(scenes) != len(set(scenes)):
        raise RuntimeError("streaming evaluation scenes must be unique")
    return scenes


def _ordered_evaluation_scenes(dataset) -> tuple[int, ...]:
    rows = [
        (str(dataset.bank_ids[scene]), scene)
        for scene in _frozen_evaluation_scenes(dataset)
    ]
    bank_ids = [bank for bank, _scene in rows]
    if len(bank_ids) != len(set(bank_ids)):
        raise RuntimeError("streaming evaluation requires one unique scene per bank ID")
    rows.sort(key=lambda row: row[0])
    return tuple(scene for _bank, scene in rows)


def _load_or_fit_probes(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    checkpoint: Mapping[str, object],
    checkpoint_index: int,
    *,
    model,
    dataset,
    teacher,
    config,
    normalization,
    route_normalization,
    batch_size: int,
) -> tuple[_ProbeBundle, bool]:
    identity = _probe_state_identity(run_identity, checkpoint, checkpoint_index)
    completed = store.load_or_quarantine_shard(identity, suffix=".pt")
    if completed is not None:
        try:
            restored = _restore_probe_bundle(
                completed.payload_path,
                config,
                seed=int(checkpoint["seed"]),
                arm=str(checkpoint["arm"]),
                expected_response_output_dim=_dataset_response_output_dim(dataset),
            )
        except Exception as error:
            # The resume manifest authenticates bytes, but not the serialized
            # probe payload's schema. Preserve a semantically invalid artifact
            # before fitting a replacement; stale shard identities still raise
            # from load_or_quarantine_shard above.
            store.quarantine_completed_shard(
                identity,
                suffix=".pt",
                reason=(
                    "persisted evaluation probe bundle failed semantic validation: "
                    f"{type(error).__name__}"
                ),
            )
        else:
            return restored, True
    probes = _fit_probe_bundle(
        model,
        dataset,
        teacher,
        config,
        normalization,
        route_normalization,
        seed=int(checkpoint["seed"]),
        batch_size=batch_size,
    )
    _commit_probe_bundle(store, identity, probes, config)
    return probes, False


PUBLIC_EVALUATION_STATUS_SCHEMA = "csi-pairs-evaluation-status-v1"


def _write_public_evaluation_status(store: EvaluationResumeStore) -> None:
    """Write operator progress to evaluation/status.json. Not a scientific gate."""

    progress = store.read_status()
    if progress is None:
        return
    output_dir = Path(store.root).resolve().parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = int(progress["total_units"])
    completed = int(progress["completed_units"])
    fraction = (completed / total) if total > 0 else 0.0
    write_atomic_json(
        output_dir / "status.json",
        {
            "schema_version": PUBLIC_EVALUATION_STATUS_SCHEMA,
            "status": progress.get("status"),
            "fraction_complete": fraction,
            "completed_units": completed,
            "total_units": total,
            "last_shard": progress.get("current_shard"),
            "current_seed": progress.get("current_seed"),
            "current_arm": progress.get("current_arm"),
            "current_substage": progress.get("current_substage"),
            "authoritative_gate": None,
            "note": "progress only; authoritative PASS/FAIL is evaluation/gate.json at 100% merge",
        },
    )


def _progress_update_if_reached(
    store: EvaluationResumeStore,
    completed_units: int,
    *,
    seed: int | None,
    arm: str | None,
    substage: str | None,
    shard: str | None,
) -> None:
    while True:
        progress = store.read_status()
        if progress is None:
            raise RuntimeError("streaming evaluation progress is not initialized")
        target = max(completed_units, int(progress["completed_units"]))
        try:
            store.update_progress(
                target,
                current_seed=seed,
                current_arm=arm,
                current_substage=substage,
                current_shard=shard,
            )
            _write_public_evaluation_status(store)
            return
        except ValueError:
            refreshed = store.read_status()
            if (
                refreshed is not None
                and int(refreshed["completed_units"]) > target
            ):
                continue
            raise


class _ProgressHeartbeat:
    """Refresh one in-flight unit without claiming that it completed."""

    def __init__(
        self,
        store: EvaluationResumeStore,
        completed_units: int,
        *,
        seed: int | None,
        arm: str | None,
        substage: str | None,
        shard: str | None,
        interval_seconds: float = DEFAULT_PROGRESS_HEARTBEAT_SECONDS,
        probe_progress=None,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or not 0.0 < float(interval_seconds) <= 60.0
        ):
            raise ValueError("progress heartbeat interval must be within (0, 60] seconds")
        if type(completed_units) is not int or completed_units < 0:
            raise ValueError("progress heartbeat completed_units is invalid")
        self.store = store
        self.completed_units = completed_units
        self.seed = seed
        self.arm = arm
        self.substage = substage
        self.shard = shard
        self.interval_seconds = float(interval_seconds)
        self.probe_progress = probe_progress
        self._stop = threading.Event()
        self._errors: list[BaseException] = []
        self._thread: threading.Thread | None = None

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _publish(self) -> None:
        _progress_update_if_reached(
            self.store,
            self.completed_units,
            seed=self.seed,
            arm=self.arm,
            substage=self.substage,
            shard=self.shard,
        )
        if self.probe_progress is not None:
            write_atomic_json(
                Path(self.store.root) / "probe_progress.json",
                self.probe_progress.snapshot(),
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._publish()
            except BaseException as error:
                self._errors.append(error)
                self._stop.set()
                return

    def __enter__(self) -> "_ProgressHeartbeat":
        if self._thread is not None:
            raise RuntimeError("progress heartbeat cannot be reused")
        self._publish()
        self._thread = threading.Thread(
            target=self._run,
            name="evaluation-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._errors:
            heartbeat_error = self._errors[0]
            if exception is None:
                raise heartbeat_error
            if hasattr(exception, "add_note"):
                exception.add_note(f"progress heartbeat also failed: {heartbeat_error!r}")
        return False


def _evaluation_total_units(checkpoint_count: int, scene_count: int) -> int:
    if (
        type(checkpoint_count) is not int
        or checkpoint_count < 0
        or type(scene_count) is not int
        or scene_count < 0
    ):
        raise ValueError("evaluation unit counts must be nonnegative integers")
    return checkpoint_count * (1 + scene_count) + 1


def _reconcile_progress_after_reauthentication(
    store: EvaluationResumeStore,
    authenticated_completed_units: int,
) -> None:
    progress = store.read_status()
    if progress is None:
        raise RuntimeError("streaming evaluation progress is not initialized")
    total_units = int(progress["total_units"])
    if (
        type(authenticated_completed_units) is not int
        or authenticated_completed_units < 0
        or authenticated_completed_units > total_units
    ):
        raise ValueError("authenticated evaluation progress count is invalid")
    persisted = int(progress["completed_units"])
    if authenticated_completed_units == persisted:
        return
    fields = {
        "current_seed": None,
        "current_arm": None,
        "current_substage": None,
        "current_shard": None,
    }
    if authenticated_completed_units < persisted:
        store.reconcile_progress(
            authenticated_completed_units,
            **fields,
            reason=(
                "streaming runner reauthenticated fewer completed units than the "
                "persisted progress receipt"
            ),
        )
    else:
        store.update_progress(authenticated_completed_units, **fields)


def _manifest_from_receipts(
    output_dir: Path,
    receipts: Mapping[str, object],
    evidence: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = []
    for name in sorted(receipts):
        receipt = receipts[name]
        rows.append(
            {
                "path": name,
                "bytes": int(receipt.bytes),
                "sha256": str(receipt.sha256),
                **evidence,
            }
        )
    gate_path = output_dir / "gate.json"
    rows.append(
        {
            "path": "gate.json",
            "bytes": gate_path.stat().st_size,
            "sha256": sha256_file(gate_path),
            **evidence,
        }
    )
    rows.sort(key=lambda row: str(row["path"]))
    return rows


class _FinalizationMismatch(RuntimeError):
    pass


class _UnsafeFinalOutput(RuntimeError):
    pass


def _final_output_paths(output_dir: Path) -> tuple[Path, ...]:
    names = tuple(EVALUATION_TABLE_FIELDS) + ("gate.json", "manifest.json")
    return tuple(output_dir / name for name in sorted(names))


def _final_output_record(path: Path, run_root: Path) -> dict[str, object]:
    if path.is_symlink() or (os.path.lexists(os.fspath(path)) and not path.is_file()):
        raise _UnsafeFinalOutput(
            f"final evaluation artifact must be a regular file: {path}"
        )
    if not path.is_file():
        raise _FinalizationMismatch(f"final evaluation artifact is missing: {path}")
    return {
        "path": path.relative_to(run_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finalization_payload(
    run_identity: EvaluationRunIdentity,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise _UnsafeFinalOutput(
            f"final evaluation output must be a regular directory: {output_dir}"
        )
    run_root = output_dir.parent
    return {
        "schema_version": FINALIZATION_SCHEMA,
        "run_identity": run_identity.as_dict(),
        "run_identity_sha256": run_identity.sha256,
        "output_schema_id": run_identity.output_schema_id,
        "output_schema_sha256": run_identity.output_schema_sha256,
        "artifacts": [
            _final_output_record(path, run_root)
            for path in _final_output_paths(output_dir)
        ],
    }


def _commit_finalization(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    output_dir: Path,
) -> Path:
    payload = _finalization_payload(run_identity, output_dir)
    encoded = _json_bytes(payload)
    result = store.commit_shard_bytes(
        _finalization_identity(run_identity), encoded, suffix=".json"
    )
    if result.reused and result.payload_path.read_bytes() != encoded:
        raise RuntimeError(
            "resumed evaluation finalization differs from deterministic output"
        )
    return result.payload_path


_FINALIZATION_CONTRACT_TABLES = (
    "compatibility_probe_contract.csv",
    "response_probe_contract.csv",
)


def _strict_json_match(expected: object, observed: object, *, path: str = "$") -> None:
    """Require recursive JSON equality, including scalar types."""

    if type(expected) is not type(observed):
        raise _FinalizationMismatch(
            f"replayed evaluation gate type differs at {path}"
        )
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        if set(expected) != set(observed):
            raise _FinalizationMismatch(
                f"replayed evaluation gate fields differ at {path}"
            )
        for key in sorted(expected):
            _strict_json_match(
                expected[key], observed[key], path=f"{path}.{key}"
            )
        return
    if isinstance(expected, list):
        assert isinstance(observed, list)
        if len(expected) != len(observed):
            raise _FinalizationMismatch(
                f"replayed evaluation gate list length differs at {path}"
            )
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _strict_json_match(left, right, path=f"{path}[{index}]")
        return
    if expected != observed:
        raise _FinalizationMismatch(
            f"replayed evaluation gate value differs at {path}"
        )


def _authenticated_final_fragment_paths(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    checkpoint_rows: Iterable[Mapping[str, object]],
    scenes: Iterable[int],
) -> dict[str, tuple[Path, ...]]:
    """Resolve and authenticate the canonical fragment stream for each table."""

    checkpoints = tuple(checkpoint_rows)
    ordered_scenes = tuple(int(scene) for scene in scenes)
    if len(ordered_scenes) != len(set(ordered_scenes)):
        raise _FinalizationMismatch("finalization scene inventory contains duplicates")
    paths_by_table: dict[str, tuple[Path, ...]] = {}
    for table in EVALUATION_TABLE_FIELDS:
        if table in _FINALIZATION_CONTRACT_TABLES:
            identities = [
                _contract_identity(run_identity, checkpoint, table, checkpoint_index)
                for checkpoint_index, checkpoint in enumerate(checkpoints)
            ]
        else:
            identities = [
                _table_identity(run_identity, checkpoint, table, scene)
                for checkpoint in checkpoints
                for scene in ordered_scenes
            ]
        paths: list[Path] = []
        for identity in identities:
            result = store.load_or_quarantine_shard(identity, suffix=".csv.gz")
            if result is None:
                raise _FinalizationMismatch(
                    "authenticated finalization is missing a table fragment: "
                    f"{table} {identity.shard_id}"
                )
            paths.append(result.payload_path)
        if len(paths) != len(set(paths)):
            raise _FinalizationMismatch(
                f"authenticated finalization has duplicate table fragments: {table}"
            )
        paths_by_table[table] = tuple(paths)
    return paths_by_table


def _final_csv_canonical_digest(
    fragment_paths: Iterable[Path],
    *,
    fields: tuple[str, ...],
    expected_evidence: Mapping[str, str],
) -> tuple[int, str]:
    """Render a canonical plain CSV stream from gzip fragments without retaining rows."""

    fragments = tuple(fragment_paths)

    class DigestSink:
        def __init__(self) -> None:
            self.digest = hashlib.sha256()
            self.byte_count = 0

        def write(self, value: str) -> int:
            encoded = value.encode("utf-8", errors="strict")
            self.digest.update(encoded)
            self.byte_count += len(encoded)
            return len(value)

    sink = DigestSink()
    try:
        writer = csv.writer(sink, dialect="excel")
        header_written = False

        def emit(row: dict[str, str]) -> None:
            nonlocal header_written
            if any(row[field] != value for field, value in expected_evidence.items()):
                raise _FinalizationMismatch(
                    "authenticated finalization fragment evidence differs from the run"
                )
            if not header_written:
                writer.writerow(fields)
                header_written = True
            writer.writerow([row[field] for field in fields])

        for fragment in fragments:
            visit_csv_fragment_rows(
                fragment,
                fieldnames=fields,
                visitor=emit,
            )
        return sink.byte_count, sink.digest.hexdigest()
    except _FinalizationMismatch:
        raise
    except (CsvFragmentError, OSError, UnicodeError, csv.Error) as error:
        raise _FinalizationMismatch(
            "authenticated finalization canonical CSV reconstruction failed"
        ) from error


def _visit_final_csv_rows(
    path: Path,
    *,
    fields: tuple[str, ...],
    visitor: Callable[[dict[str, str]], object],
) -> None:
    """Visit a final plain CSV output with the same strict width contract as fragments."""

    try:
        with path.open(
            mode="r", encoding="utf-8", errors="strict", newline=""
        ) as handle:
            reader = csv.DictReader(handle, dialect="excel", strict=True)
            if reader.fieldnames is None and path.stat().st_size == 0:
                return
            if tuple(reader.fieldnames or ()) != fields:
                raise _FinalizationMismatch(
                    f"final evaluation table header differs from canonical schema: {path.name}"
                )
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise _FinalizationMismatch(
                        f"final evaluation table row width is invalid: {path.name}"
                    )
                visitor(dict(row))
    except _FinalizationMismatch:
        raise
    except (csv.Error, OSError, UnicodeError) as error:
        raise _FinalizationMismatch(
            f"final evaluation table cannot be decoded: {path.name}"
        ) from error


def _decode_final_gate_row(table: str, row: Mapping[str, str]) -> dict[str, object]:
    """Convert canonical CSV spellings back to the in-memory gate row contract."""

    decoded: dict[str, object] = {}
    for field in _scene_table_core_fields(table):
        value = row[field]
        if field in SCENE_TEXT_FIELDS:
            decoded[field] = value
            continue
        if field in SCENE_INTEGER_FIELDS:
            if not value:
                raise _FinalizationMismatch(
                    f"final evaluation integer is empty: {table}.{field}"
                )
            try:
                parsed = int(value, 10)
            except ValueError as error:
                raise _FinalizationMismatch(
                    f"final evaluation integer is invalid: {table}.{field}"
                ) from error
            if str(parsed) != value:
                raise _FinalizationMismatch(
                    f"final evaluation integer is non-canonical: {table}.{field}"
                )
            decoded[field] = parsed
            continue
        if value == "":
            if field not in OPTIONAL_SCENE_NUMERIC_FIELDS:
                raise _FinalizationMismatch(
                    f"final evaluation required scalar is empty: {table}.{field}"
                )
            decoded[field] = None
            continue
        try:
            parsed_float = float(value)
        except ValueError as error:
            raise _FinalizationMismatch(
                f"final evaluation float is invalid: {table}.{field}"
            ) from error
        if not math.isfinite(parsed_float):
            raise _FinalizationMismatch(
                f"final evaluation float is non-finite: {table}.{field}"
            )
        decoded[field] = parsed_float
    return decoded


def _authenticate_final_csvs(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    output_dir: Path,
    *,
    checkpoint_rows: Iterable[Mapping[str, object]],
    scenes: Iterable[int],
    evidence: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Bind every final table to authenticated fragments and retain only gate rows."""

    paths_by_table = _authenticated_final_fragment_paths(
        store, run_identity, checkpoint_rows, scenes
    )
    expected_evidence = _csv_evidence_values(evidence)
    gate_rows: dict[str, list[dict[str, object]]] = {}
    for table, fields in EVALUATION_TABLE_FIELDS.items():
        final_path = output_dir / table
        expected_bytes, expected_sha256 = _final_csv_canonical_digest(
            paths_by_table[table],
            fields=fields,
            expected_evidence=expected_evidence,
        )
        actual_bytes = final_path.stat().st_size
        actual_sha256 = sha256_file(final_path)
        if (actual_bytes, actual_sha256) != (expected_bytes, expected_sha256):
            try:
                store.quarantine_final_output_artifacts(
                    (final_path,),
                    reason=(
                        "final evaluation table differs from the canonical merge of "
                        f"authenticated fragments: {table}"
                    ),
                )
            except (OSError, RuntimeError, ValueError):
                # The enclosing finalization failure still forces a deterministic
                # rebuild; preserving the damaged file is best-effort only.
                pass
            raise _FinalizationMismatch(
                f"final evaluation table is not the canonical fragment merge: {table}"
            )
        if table in SMALL_SCENE_TABLES:
            rows: list[dict[str, object]] = []
            _visit_final_csv_rows(
                final_path,
                fields=fields,
                visitor=lambda row: rows.append(_decode_final_gate_row(table, row)),
            )
            gate_rows[table] = rows
    return gate_rows


def _load_authenticated_finalization(
    store: EvaluationResumeStore,
    run_identity: EvaluationRunIdentity,
    output_dir: Path,
    *,
    config: dict | None = None,
    dataset=None,
    checkpoint_rows: Iterable[Mapping[str, object]] | None = None,
    scenes: Iterable[int] | None = None,
    frozen_scenes: Iterable[int] | None = None,
    factorial_gate: Mapping[str, object] | None = None,
    evidence: Mapping[str, object] | None = None,
    qualification_gate_sha256: str | None = None,
    factorial_gate_sha256: str | None = None,
) -> dict | None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise _UnsafeFinalOutput(
            f"final evaluation output must be a regular directory: {output_dir}"
        )
    identity = _finalization_identity(run_identity)
    completed = store.load_or_quarantine_shard(identity, suffix=".json")
    if completed is None:
        return None
    try:
        payload = read_strict_json(completed.payload_path)
        required = {
            "schema_version",
            "run_identity",
            "run_identity_sha256",
            "output_schema_id",
            "output_schema_sha256",
            "artifacts",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise _FinalizationMismatch(
                "evaluation finalization receipt fields are invalid"
            )
        if (
            payload["schema_version"] != FINALIZATION_SCHEMA
            or payload["run_identity"] != run_identity.as_dict()
            or payload["run_identity_sha256"] != run_identity.sha256
            or payload["output_schema_id"] != run_identity.output_schema_id
            or payload["output_schema_sha256"]
            != run_identity.output_schema_sha256
        ):
            raise _FinalizationMismatch(
                "evaluation finalization receipt identity is stale"
            )
        artifacts = payload["artifacts"]
        paths = _final_output_paths(output_dir)
        run_root = output_dir.parent
        expected_relative_paths = [
            path.relative_to(run_root).as_posix() for path in paths
        ]
        if (
            not isinstance(artifacts, list)
            or len(artifacts) != len(paths)
            or any(
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or record["path"] != expected_path
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                or not _valid_sha256(record["sha256"])
                for record, expected_path in zip(
                    artifacts, expected_relative_paths, strict=True
                )
            )
        ):
            raise _FinalizationMismatch(
                "evaluation finalization artifact inventory is invalid"
            )
        damaged_paths = []
        missing_artifact = False
        for path, recorded in zip(paths, artifacts, strict=True):
            try:
                actual = _final_output_record(path, run_root)
            except _FinalizationMismatch:
                missing_artifact = True
                continue
            if actual != recorded:
                damaged_paths.append(path)
        if damaged_paths:
            store.quarantine_final_output_artifacts(
                damaged_paths,
                reason="final artifact differs from authenticated finalization receipt",
            )
        if missing_artifact or damaged_paths:
            raise _FinalizationMismatch(
                "evaluation finalization artifact receipt mismatch"
            )

        json_outputs = {}
        invalid_json_paths = []
        for name in ("gate.json", "manifest.json"):
            path = output_dir / name
            try:
                value = read_strict_json(path)
            except StrictJsonError:
                invalid_json_paths.append(path)
                continue
            if not isinstance(value, dict):
                invalid_json_paths.append(path)
                continue
            json_outputs[name] = value
        if invalid_json_paths:
            store.quarantine_final_output_artifacts(
                invalid_json_paths,
                reason="authenticated final JSON artifact is invalid",
            )
            raise _FinalizationMismatch(
                "evaluation finalization JSON artifacts are invalid"
            )
        gate = json_outputs["gate.json"]
        manifest = json_outputs["manifest.json"]
        if (config is None) != (dataset is None):
            raise _FinalizationMismatch(
                "finalization gate validation requires config and dataset together"
            )
        if config is not None:
            if config is None or dataset is None:
                raise _FinalizationMismatch(
                    "finalization gate validation requires config and dataset together"
                )
            from .formal_evidence import require_stage_manifested_gate

            try:
                require_stage_manifested_gate(
                    output_dir / "gate.json",
                    gate,
                    config,
                    dataset,
                    schema_version="csi-pairs-v6-evaluation-gate-v3",
                )
            except (RuntimeError, ValueError, TypeError, KeyError, StrictJsonError) as error:
                store.quarantine_final_output_artifacts(
                    (output_dir / "gate.json", output_dir / "manifest.json"),
                    reason=(
                        "authenticated evaluation gate failed canonical stage "
                        f"validation: {type(error).__name__}"
                    ),
                )
                raise _FinalizationMismatch(
                    "evaluation gate failed canonical stage validation"
                ) from error
            required_gate_fields = {
                "schema_version",
                "status",
                "passed",
                "scientific_claim_status",
                "gate_vector",
                "g3_subgates",
                "g4_subgates",
            }
            if not required_gate_fields.issubset(gate):
                raise _FinalizationMismatch(
                    "evaluation gate semantic fields are incomplete"
                )
            if gate["status"] not in {"PASS", "FAIL"} or not isinstance(
                gate["passed"], bool
            ):
                raise _FinalizationMismatch(
                    "evaluation gate status contract is invalid"
                )
            if gate["passed"] != (gate["status"] == "PASS"):
                raise _FinalizationMismatch(
                    "evaluation gate passed flag disagrees with status"
                )

            replay_values = (
                checkpoint_rows,
                scenes,
                frozen_scenes,
                factorial_gate,
                evidence,
                qualification_gate_sha256,
                factorial_gate_sha256,
            )
            if any(value is not None for value in replay_values):
                if any(value is None for value in replay_values):
                    raise _FinalizationMismatch(
                        "finalization gate replay context is incomplete"
                    )
                assert scenes is not None
                assert frozen_scenes is not None
                ordered_scenes = tuple(int(scene) for scene in scenes)
                frozen_scene_values = tuple(int(scene) for scene in frozen_scenes)
                if (
                    len(ordered_scenes) != len(frozen_scene_values)
                    or set(ordered_scenes) != set(frozen_scene_values)
                    or len(ordered_scenes) != len(set(ordered_scenes))
                ):
                    raise _FinalizationMismatch(
                        "finalization scene inventory differs from the frozen inventory"
                    )
                assert checkpoint_rows is not None
                assert factorial_gate is not None
                assert evidence is not None
                assert qualification_gate_sha256 is not None
                assert factorial_gate_sha256 is not None
                try:
                    gate_rows = _authenticate_final_csvs(
                        store,
                        run_identity,
                        output_dir,
                        checkpoint_rows=checkpoint_rows,
                        scenes=ordered_scenes,
                        evidence=evidence,
                    )
                except _FinalizationMismatch:
                    raise
                except (CsvFragmentError, KeyError, IndexError, TypeError, ValueError) as error:
                    raise _FinalizationMismatch(
                        "authenticated finalization table replay failed"
                    ) from error
                try:
                    replayed_gate = legacy._evaluation_gate(
                        config,
                        dataset,
                        gate_rows["cgs_per_bank.csv"],
                        gate_rows["compatibility_route_distributions.csv"],
                        gate_rows["cgs_active_effect_bins.csv"],
                        gate_rows["response_per_bank.csv"],
                        gate_rows["alignment_shortcut_baselines.csv"],
                        factorial_gate,
                        evidence,
                        qualification_gate_sha256=qualification_gate_sha256,
                        factorial_gate_sha256=factorial_gate_sha256,
                    )
                except (RuntimeError, TypeError, ValueError, KeyError, IndexError) as error:
                    raise _FinalizationMismatch(
                        "authenticated finalization gate replay failed"
                    ) from error
                try:
                    _strict_json_match(replayed_gate, gate)
                except _FinalizationMismatch:
                    try:
                        store.quarantine_final_output_artifacts(
                            (output_dir / "gate.json", output_dir / "manifest.json"),
                            reason=(
                                "authenticated evaluation gate differs from strict "
                                "scientific replay"
                            ),
                        )
                    except (OSError, RuntimeError, ValueError):
                        pass
                    raise
        # The canonical stage validator above authenticates the complete manifest;
        # keep an explicit local binding as an additional guard for callers that
        # use this loader without a dataset object (unit-level recovery tests).
        manifest_gate = next(
            (
                row
                for row in manifest.get("files", [])
                if isinstance(row, dict) and row.get("path") == "gate.json"
            ),
            None,
        )
        if not isinstance(manifest_gate, dict) or manifest_gate.get(
            "sha256"
        ) != sha256_file(output_dir / "gate.json"):
            raise _FinalizationMismatch(
                "evaluation stage manifest does not bind gate.json"
            )
        return json_outputs["gate.json"]
    except _UnsafeFinalOutput:
        raise
    except (StrictJsonError, _FinalizationMismatch) as error:
        store.quarantine_completed_shard(
            identity,
            suffix=".json",
            reason=(
                "authenticated final output failed validation: "
                f"{type(error).__name__}"
            ),
        )
        return None


def run_streaming_formal_evaluation(
    config: dict,
    dataset,
    output_root: str | Path,
    *,
    upstream_root: str | Path,
    qualification_gate_path: str | Path,
    factorial_gate_path: str | Path,
    checkpoint_index_path: str | Path,
    checkpoint_inventory_sha256: str,
    run_identity: EvaluationRunIdentity,
    execution_devices: Iterable[str],
    batch_size: int = 1,
    probe_build_limit: int = 1,
    authenticated_origin=None,
    stop_after_units: int | None = None,
) -> dict:
    """Run one identity-bound evaluation with bank-level atomic resume."""

    from .formal_data_verification import require_verified_roles_from_root

    required_roles = (
        "source_encoder_train",
        "source_probe_train",
        "source_probe_selection",
        "source_final_unseen_bank",
        "target",
    )
    if authenticated_origin is None:
        require_verified_roles_from_root(output_root, config, dataset, required_roles)
    else:
        from .formal_evaluation_repair import AuthenticatedEvaluationOrigin

        if not isinstance(authenticated_origin, AuthenticatedEvaluationOrigin):
            raise TypeError("evaluation origin must be authenticated")
        authenticated_origin.require_compatible(config, dataset, upstream_root, required_roles)
    if stop_after_units is not None and (type(stop_after_units) is not int or stop_after_units < 1):
        raise ValueError("stop_after_units must be positive or None")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("streaming evaluation batch_size must be positive")
    if type(probe_build_limit) is not int or probe_build_limit not in (1, 2):
        raise ValueError("probe_build_limit must be one or two")
    worker_devices = _validated_worker_devices(execution_devices)
    actual_execution_profile = EvaluationExecutionProfile(
        execution_devices=worker_devices,
        batch_size=batch_size,
    )
    if run_identity.execution_profile != actual_execution_profile:
        raise StaleResumeError(
            "streaming evaluation execution profile differs from run identity"
        )
    root_candidate = Path(output_root)
    require_safe_directory_prefix(root_candidate)
    if root_candidate.is_symlink():
        raise _UnsafeFinalOutput("formal evaluation run root cannot be a symlink")
    root = root_candidate.resolve()
    upstream = Path(upstream_root).resolve()
    qualification_path = Path(qualification_gate_path).resolve()
    factorial_path = Path(factorial_gate_path).resolve()
    checkpoint_path = Path(checkpoint_index_path).resolve()
    output_dir = root / "evaluation"
    if output_dir.is_symlink() or (
        os.path.lexists(os.fspath(output_dir)) and not output_dir.is_dir()
    ):
        raise _UnsafeFinalOutput(
            f"final evaluation output must be a regular directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise _UnsafeFinalOutput(
            f"final evaluation output must be a regular directory: {output_dir}"
        )
    evidence = evidence_context(
        config,
        dataset,
        "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
    )
    _validate_runner_identity(
        run_identity,
        evidence,
        qualification_gate_path=qualification_path,
        factorial_gate_path=factorial_path,
        checkpoint_inventory_sha256=checkpoint_inventory_sha256,
    )
    qualification_gate = read_strict_json(qualification_path)
    factorial_gate = read_strict_json(factorial_path)
    checkpoint_index = read_strict_json(checkpoint_path)
    if not isinstance(qualification_gate, dict) or not isinstance(factorial_gate, dict):
        raise RuntimeError("streaming evaluation upstream gates must be JSON objects")
    checkpoint_rows = legacy._validate_checkpoint_index(
        checkpoint_index, config, dataset, qualification_gate
    )
    if not isinstance(checkpoint_rows, list):
        checkpoint_rows = list(checkpoint_rows)
    scenes = _ordered_evaluation_scenes(dataset)
    frozen_scenes = _frozen_evaluation_scenes(dataset)
    if len(frozen_scenes) != len(scenes) or set(frozen_scenes) != set(scenes):
        raise RuntimeError("streaming evaluation and shard scene inventories differ")
    pair_universes: dict[int, ScenePairUniverse] = {}
    # Lightweight unit-test doubles intentionally omit the full dataset identity
    # contract.  Real FormalDataset instances expose every field below, so their
    # pair domain is always rebuilt independently from the payload tables.
    required_pair_metadata = (
        "world_bits",
        "position_roles",
        "natural_world_index",
        "metadata",
        "directed_edges",
    )
    if all(hasattr(dataset, name) for name in required_pair_metadata):
        for scene in scenes:
            try:
                pair_universes[int(scene)] = build_scene_pair_universe(
                    dataset, int(scene)
                )
            except (PairInventoryError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"streaming evaluation dataset pair universe is invalid for scene {scene}"
                ) from error
    work_plan = build_evaluation_work_plan(
        checkpoint_rows,
        scenes,
        devices=worker_devices,
        batch_size=batch_size,
    )
    total_units = _evaluation_total_units(len(checkpoint_rows), len(scenes))
    if total_units != len(work_plan) + 1:
        raise RuntimeError("streaming evaluation work-plan size is inconsistent")
    store = EvaluationResumeStore(root / STATE_DIRECTORY, run_identity)
    fragments: dict[str, dict[int, Path]] = {
        name: {} for name in EVALUATION_TABLE_FIELDS
    }
    aggregate: dict[str, dict[int, list[dict]]] = {
        "cgs_per_bank.csv": {},
        "alignment_shortcut_baselines.csv": {},
        "compatibility_route_distributions.csv": {},
        "cgs_active_effect_bins.csv": {},
        "response_per_bank.csv": {},
    }
    with store.writer_lock():
        store.initialize_progress(total_units)
        _write_public_evaluation_status(store)
        completed_gate = _load_authenticated_finalization(
            store,
            run_identity,
            output_dir,
            config=config,
            dataset=dataset,
            checkpoint_rows=checkpoint_rows,
            scenes=scenes,
            frozen_scenes=frozen_scenes,
            factorial_gate=factorial_gate,
            evidence=evidence,
            qualification_gate_sha256=sha256_file(qualification_path),
            factorial_gate_sha256=sha256_file(factorial_path),
        )
        if completed_gate is not None:
            _progress_update_if_reached(
                store,
                total_units,
                seed=None,
                arm=None,
                substage=None,
                shard=None,
            )
            return completed_gate
        store.quarantine_final_output_temporaries(_final_output_paths(output_dir))
        authenticated_indices: set[int] = set()
        probe_paths: dict[int, Path] = {}
        scene_bundles: dict[int, dict[str, object]] = {}
        for unit in work_plan:
            checkpoint = checkpoint_rows[unit.checkpoint_index]
            with _ProgressHeartbeat(
                store,
                0,
                seed=unit.seed,
                arm=unit.arm,
                substage="resume_authentication",
                shard=unit.shard,
            ):
                if unit.substage == "probe_state":
                    probe_identity = _probe_state_identity(
                        run_identity, checkpoint, unit.checkpoint_index
                    )
                    completed = store.load_or_quarantine_shard(
                        probe_identity,
                        suffix=".pt",
                    )
                    if completed is not None:
                        try:
                            # Authenticate the serialized payload contract before
                            # counting the work unit as complete. A valid manifest
                            # alone only authenticates bytes, not probe semantics.
                            _restore_probe_bundle(
                                completed.payload_path,
                                config,
                                seed=unit.seed,
                                arm=unit.arm,
                                expected_response_output_dim=_dataset_response_output_dim(
                                    dataset
                                ),
                            )
                        except Exception as error:
                            store.quarantine_completed_shard(
                                probe_identity,
                                suffix=".pt",
                                reason=(
                                    "persisted evaluation probe bundle failed "
                                    "semantic validation: "
                                    f"{type(error).__name__}"
                                ),
                            )
                        else:
                            authenticated_indices.add(unit.canonical_index)
                            probe_paths[unit.canonical_index] = completed.payload_path
                    continue
                assert unit.scene is not None
                bank_id = str(dataset.bank_ids[unit.scene])
                bundle = _load_scene_bundle(
                    store,
                    run_identity,
                    checkpoint,
                    scene=unit.scene,
                    bank_id=bank_id,
                    evidence=evidence,
                    pair_universe=pair_universes.get(int(unit.scene)),
                )
                if bundle is not None:
                    authenticated_indices.add(unit.canonical_index)
                    scene_bundles[unit.canonical_index] = bundle

        completed_units = len(authenticated_indices)
        _reconcile_progress_after_reauthentication(store, completed_units)
        authenticated_scenes_by_checkpoint = {
            checkpoint_index: {
                int(unit.scene)
                for unit in work_plan
                if unit.checkpoint_index == checkpoint_index
                and unit.substage == "scene"
                and unit.canonical_index in authenticated_indices
                and unit.scene is not None
            }
            for checkpoint_index in range(len(checkpoint_rows))
        }
        worker_states: dict[str, dict[str, object]] = {
            device: {
                "teacher": None,
                "route_normalization": None,
                "normalization": None,
                "checkpoint_index": None,
                "model": None,
                "probes": None,
                "scene_evaluations": None,
            }
            for device in worker_devices
        }
        rng_lock = threading.Lock()
        available_probe_memory = _available_probe_memory()
        probe_capacity = min(probe_build_limit, _probe_build_capacity(len(worker_devices), available_probe_memory))
        probe_build_lock = threading.BoundedSemaphore(probe_capacity)
        probe_progress = _ProbeProgress(probe_capacity, available_probe_memory)

        def ensure_common_state(device: str, state: dict[str, object]) -> None:
            if state["teacher"] is not None:
                return
            torch.cuda.set_device(device)
            with rng_lock:
                teacher = load_teacher_bundle(
                    qualification_gate["teacher_checkpoint"],
                    config,
                    device=device,
                )
            route_normalization = fit_route_normalization(dataset, teacher)
            normalization = legacy._training_normalization(
                dataset,
                dataset.indices_for_role("source_encoder_train"),
                route_normalization,
                teacher.patch_spec,
            )
            state["teacher"] = teacher
            state["route_normalization"] = route_normalization
            state["normalization"] = normalization

        def ensure_checkpoint_model(
            unit: EvaluationWorkUnit,
            checkpoint: Mapping[str, object],
            state: dict[str, object],
        ) -> None:
            if state["model"] is not None:
                return
            ensure_common_state(unit.execution_device, state)
            with rng_lock:
                state["model"] = legacy._load_model(
                    upstream / "factorial",
                    checkpoint,
                    qualification_gate,
                    config,
                    dataset,
                    device=unit.execution_device,
                )

        def compute_work_unit(unit: EvaluationWorkUnit) -> object:
            checkpoint = checkpoint_rows[unit.checkpoint_index]
            state = worker_states[unit.execution_device]
            if unit.substage == "probe_state":
                report_probe = probe_progress.callback(unit)
                report_probe({"probe": "model", "phase": "loading"})
                previous_scenes = state["scene_evaluations"]
                if isinstance(previous_scenes, dict) and previous_scenes:
                    raise RuntimeError(
                        "evaluation worker retained scenes across checkpoints"
                    )
                state["checkpoint_index"] = unit.checkpoint_index
                state["model"] = None
                state["probes"] = None
                state["scene_evaluations"] = None
                probe_path = probe_paths.get(unit.canonical_index)
                if probe_path is not None:
                    with rng_lock:
                        probes = _restore_probe_bundle(
                            probe_path,
                            config,
                            seed=unit.seed,
                            arm=unit.arm,
                            expected_response_output_dim=_dataset_response_output_dim(
                                dataset
                            ),
                        )
                    requires_commit = False
                else:
                    ensure_checkpoint_model(unit, checkpoint, state)
                    probes = _fit_probe_bundle_exclusive(
                        probe_build_lock,
                        state["model"],
                        dataset,
                        state["teacher"],
                        config,
                        state["normalization"],
                        state["route_normalization"],
                        seed=unit.seed,
                        batch_size=unit.batch_size,
                        rng_lock=rng_lock,
                        device=unit.execution_device,
                        progress_callback=report_probe,
                    )
                    requires_commit = True
                state["probes"] = probes
                return _ProbeWorkResult(probes, requires_commit)
            if (
                state["checkpoint_index"] != unit.checkpoint_index
                or not isinstance(state["probes"], _ProbeBundle)
            ):
                raise RuntimeError("evaluation worker lost its checkpoint probe state")
            completed_bundle = scene_bundles.get(unit.canonical_index)
            if completed_bundle is not None:
                prepared_scenes = state["scene_evaluations"]
                if isinstance(prepared_scenes, dict) and unit.scene is not None:
                    prepared_scenes.pop(int(unit.scene), None)
                return _SceneWorkResult(completed_bundle, None)
            assert unit.scene is not None
            ensure_checkpoint_model(unit, checkpoint, state)
            prepared_scenes = state["scene_evaluations"]
            if prepared_scenes is None:
                if pair_universes:
                    # The probe fit is still checkpoint-global and remains a hard
                    # barrier.  Once it is authenticated, process one scene at a
                    # time so the large response feature matrices are never kept
                    # for all 123 evaluation banks simultaneously.
                    prepared = _prepare_scene_evaluation(
                        state["model"],
                        state["probes"],
                        dataset,
                        state["teacher"],
                        config,
                        state["normalization"],
                        state["route_normalization"],
                        scene=unit.scene,
                        batch_size=unit.batch_size,
                    )
                    rows_by_table = _evaluate_scene(
                        state["model"],
                        state["probes"],
                        dataset,
                        state["teacher"],
                        config,
                        state["normalization"],
                        state["route_normalization"],
                        scene=unit.scene,
                        seed=unit.seed,
                        arm=unit.arm,
                        batch_size=unit.batch_size,
                        scene_evaluation=prepared,
                    )
                    del prepared
                    gc.collect()
                    return _SceneWorkResult(None, rows_by_table)
                # Lightweight unit-test doubles do not implement the complete
                # FormalDataset contract. Keep their legacy hook usable while
                # real datasets always take the checkpoint-global path below.
                if not isinstance(getattr(dataset, "metadata", None), Mapping):
                    rows_by_table = _prepare_and_evaluate_scene(
                        state["model"],
                        state["probes"],
                        dataset,
                        state["teacher"],
                        config,
                        state["normalization"],
                        state["route_normalization"],
                        scene=unit.scene,
                        seed=unit.seed,
                        arm=unit.arm,
                        batch_size=unit.batch_size,
                    )
                    return _SceneWorkResult(None, rows_by_table)
                prepared_scenes = _prepare_checkpoint_scene_evaluations(
                    state["model"],
                    state["probes"],
                    dataset,
                    state["teacher"],
                    config,
                    state["normalization"],
                    state["route_normalization"],
                    scenes=frozen_scenes,
                    batch_size=unit.batch_size,
                )
                for completed_scene in authenticated_scenes_by_checkpoint[
                    unit.checkpoint_index
                ]:
                    prepared_scenes.pop(completed_scene, None)
                state["scene_evaluations"] = prepared_scenes
            if not isinstance(prepared_scenes, dict):
                raise RuntimeError("evaluation checkpoint prediction cache is invalid")
            prepared = prepared_scenes.pop(int(unit.scene), None)
            if not isinstance(prepared, _SceneEvaluation):
                raise RuntimeError("evaluation checkpoint prediction cache lost a scene")
            rows_by_table = _evaluate_scene(
                state["model"],
                state["probes"],
                dataset,
                state["teacher"],
                config,
                state["normalization"],
                state["route_normalization"],
                scene=unit.scene,
                seed=unit.seed,
                arm=unit.arm,
                batch_size=unit.batch_size,
                scene_evaluation=prepared,
            )
            del prepared
            gc.collect()
            return _SceneWorkResult(None, rows_by_table)

        def commit_work_unit(unit: EvaluationWorkUnit, result: object) -> int:
            nonlocal completed_units
            checkpoint = checkpoint_rows[unit.checkpoint_index]
            if unit.substage == "probe_state":
                if not isinstance(result, _ProbeWorkResult):
                    raise RuntimeError("evaluation probe worker result is invalid")
                if result.requires_commit:
                    if unit.canonical_index in authenticated_indices:
                        raise RuntimeError("authenticated probe was unexpectedly recomputed")
                    _commit_probe_bundle(
                        store,
                        _probe_state_identity(
                            run_identity, checkpoint, unit.checkpoint_index
                        ),
                        result.probes,
                        config,
                    )
                elif unit.canonical_index not in authenticated_indices:
                    raise RuntimeError("evaluation worker reused an unauthenticated probe")
                for table, rows in _probe_contract_rows(
                    result.probes, config, seed=unit.seed, arm=unit.arm
                ).items():
                    fragment, _receipt = _commit_table_fragment(
                        store,
                        _contract_identity(
                            run_identity,
                            checkpoint,
                            table,
                            unit.checkpoint_index,
                        ),
                        rows,
                        table=table,
                        evidence=evidence,
                        repair_reused_mismatch=True,
                    )
                    if unit.canonical_index in fragments[table]:
                        raise RuntimeError("evaluation fragment was published twice")
                    fragments[table][unit.canonical_index] = fragment.payload_path
                probe_progress.callback(unit)({"probe": "bundle", "phase": "committed"})
            else:
                if not isinstance(result, _SceneWorkResult):
                    raise RuntimeError("evaluation scene worker result is invalid")
                assert unit.scene is not None
                bank_id = str(dataset.bank_ids[unit.scene])
                if result.completed_bundle is not None and result.rows_by_table is None:
                    if unit.canonical_index not in authenticated_indices:
                        raise RuntimeError("evaluation worker reused an unauthenticated scene")
                    bundle = result.completed_bundle
                elif result.completed_bundle is None and result.rows_by_table is not None:
                    if unit.canonical_index in authenticated_indices:
                        raise RuntimeError("authenticated scene was unexpectedly recomputed")
                    bundle = _commit_scene_bundle(
                        store,
                        run_identity,
                        checkpoint,
                        scene=unit.scene,
                        bank_id=bank_id,
                        rows_by_table=result.rows_by_table,
                        evidence=evidence,
                        pair_universe=pair_universes.get(int(unit.scene)),
                    )
                    result.rows_by_table = None
                else:
                    raise RuntimeError("evaluation scene worker result is ambiguous")
                for table in SCENE_TABLES:
                    shard = store.load_or_quarantine_shard(
                        _table_identity(
                            run_identity, checkpoint, table, unit.scene
                        ),
                        suffix=".csv.gz",
                    )
                    if shard is None:
                        raise RuntimeError("completed scene lost a table shard")
                    if unit.canonical_index in fragments[table]:
                        raise RuntimeError("evaluation fragment was published twice")
                    fragments[table][unit.canonical_index] = shard.payload_path
                small_rows = bundle["small_rows"]
                if not isinstance(small_rows, dict):
                    raise RuntimeError("evaluation scene small-row bundle is invalid")
                for table in aggregate:
                    rows = small_rows.get(table)
                    if not isinstance(rows, list):
                        raise RuntimeError("evaluation scene aggregate rows are invalid")
                    if unit.canonical_index in aggregate[table]:
                        raise RuntimeError("evaluation aggregate rows were published twice")
                    aggregate[table][unit.canonical_index] = rows

            if unit.canonical_index not in authenticated_indices:
                completed_units += 1
            _progress_update_if_reached(
                store,
                completed_units,
                seed=unit.seed,
                arm=unit.arm,
                substage=unit.substage,
                shard=unit.shard,
            )
            return unit.canonical_index

        def wait_context(unit: EvaluationWorkUnit) -> ContextManager[object]:
            return _ProgressHeartbeat(
                store,
                completed_units,
                seed=unit.seed,
                arm=unit.arm,
                substage=unit.substage,
                shard=unit.shard,
                probe_progress=probe_progress,
            )

        run_evaluation_worker_coordinator(
            work_plan if stop_after_units is None else work_plan[:stop_after_units],
            compute_work_unit,
            commit_work_unit,
            wait_context_factory=wait_context,
        )
        worker_states.clear()
        gc.collect()
        if stop_after_units is not None and stop_after_units < len(work_plan):
            status = {
                "status": "PAUSED_AT_UNIT_BOUNDARY", "completed_units": completed_units,
                "total_units": total_units, "sota_ready": False,
                "note": "bounded execution validation; no final evaluation gate or ETA",
            }
            write_atomic_json(output_dir / "bounded_execution_status.json", status)
            return status
        if completed_units != len(work_plan):
            raise RuntimeError("streaming evaluation did not complete every work unit")

        with _ProgressHeartbeat(
            store,
            completed_units,
            seed=None,
            arm=None,
            substage="final_merge",
            shard="canonical-output",
        ):
            merged_receipts = {}
            for table, fields in EVALUATION_TABLE_FIELDS.items():
                merged_receipts[table] = merge_csv_fragments(
                    [
                        fragments[table][index]
                        for index in sorted(fragments[table])
                    ],
                    output_dir / table,
                    fieldnames=fields,
                )
            aggregate_rows = {
                table: [
                    row
                    for index in sorted(rows_by_unit)
                    for row in rows_by_unit[index]
                ]
                for table, rows_by_unit in aggregate.items()
            }
            gate = legacy._evaluation_gate(
                config,
                dataset,
                aggregate_rows["cgs_per_bank.csv"],
                aggregate_rows["compatibility_route_distributions.csv"],
                aggregate_rows["cgs_active_effect_bins.csv"],
                aggregate_rows["response_per_bank.csv"],
                aggregate_rows["alignment_shortcut_baselines.csv"],
                factorial_gate,
                evidence,
                qualification_gate_sha256=sha256_file(qualification_path),
                factorial_gate_sha256=sha256_file(factorial_path),
            )
            write_atomic_json(output_dir / "gate.json", gate)
            write_atomic_json(
                output_dir / "manifest.json",
                {
                    "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
                    **evidence,
                    "files": _manifest_from_receipts(
                        output_dir, merged_receipts, evidence
                    ),
                },
            )
            _commit_finalization(store, run_identity, output_dir)
        _progress_update_if_reached(
            store,
            total_units,
            seed=None,
            arm=None,
            substage=None,
            shard=None,
        )
    return gate
