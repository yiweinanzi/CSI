from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .formal_config import ARMS, public_formal_config
from .formal_dataset import FormalDataset, same_physical_position
from .formal_evidence import (
    FACTORIAL_SCHEMA,
    bind_rows,
    complete_gate_vector,
    evidence_context,
    require_manifested_formal_qualification,
)
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json
from .formal_localization import (
    adapt_position_head,
    fit_source_position_head,
    predict_position_distribution,
)
from .formal_model import (
    CSIPairsFormalModel,
    LossWeights,
    alignment_active_quartet_loss,
    alignment_null_quartet_loss,
    batch_for_module,
    endpoint_per_sample,
    module_device,
    portable_state_dict,
    require_torch,
    resolve_execution_devices,
    squared_rms_error,
    tensor_for_module,
    torch,
)
from .formal_protocol import (
    PatchSpec,
    headline_alignment_edge,
    patchify_csi,
    typed_signed_edit,
)
from .formal_routing import (
    CompactEdgeTensorMapping,
    RouteNormalization,
    RoutedEdges,
    fit_route_normalization,
    route_dataset,
)
from .formal_statistics import (
    bank_only_factorial_interval,
    exact_factorial_utilities,
    hierarchical_factorial_interval,
    holm_adjust,
    interval_decision,
    leave_one_factorial_sensitivity,
    paired_cluster_interval,
    paired_sign_flip_test,
)
from .formal_teacher import TeacherBundle, load_teacher_bundle
from .formal_training_resume import load_factorial_resume, save_factorial_resume


ARM_FACTORS = {
    "endpoint": (0.0, 0.0),
    "alignment": (1.0, 0.0),
    "response": (0.0, 1.0),
    "full": (1.0, 1.0),
}

# V6 lambda_sy is the physical term in the alignment compatibility score.  It
# is deliberately frozen separately from lambda_By (the endpoint loss weight).
ALIGNMENT_SCORE_PHYSICAL_WEIGHT = 1.0
FORMAL_CHECKPOINT_INTERVAL_STEPS = 100
NOOP_SCORE_BATCH_SIZE = 256


@dataclass(frozen=True)
class TrainingNormalization:
    patch_mean: np.ndarray
    patch_scale: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    map_scale: np.ndarray
    radio_mean: np.ndarray
    radio_scale: np.ndarray
    action_scale: np.ndarray
    position_mean: np.ndarray
    position_scale: np.ndarray


def _sequence_index(index, length: int):
    if isinstance(index, slice):
        return tuple(range(*index.indices(length)))
    value = int(index)
    if value < 0:
        value += length
    if not 0 <= value < length:
        raise IndexError(index)
    return value


class CartesianEndpointSequence(Sequence):
    def __init__(
        self, worlds: tuple[int, ...], position_count: int
    ) -> None:
        self.worlds = tuple(int(value) for value in worlds)
        self.position_count = int(position_count)

    def __len__(self) -> int:
        return len(self.worlds) * self.position_count

    def __getitem__(self, index):
        normalized = _sequence_index(index, len(self))
        if isinstance(normalized, tuple):
            return tuple(self[value] for value in normalized)
        world_index, position = divmod(normalized, self.position_count)
        return self.worlds[world_index], position


class EdgePositionSequence(Sequence):
    def __init__(
        self,
        edges: tuple[tuple[int, int], ...],
        position_count: int,
        flat_indices: np.ndarray,
    ) -> None:
        self.edges = edges
        self.position_count = int(position_count)
        self.flat_indices = np.asarray(flat_indices, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.flat_indices.size)

    def __getitem__(self, index):
        normalized = _sequence_index(index, len(self))
        if isinstance(normalized, tuple):
            return tuple(self[value] for value in normalized)
        edge_index, position = divmod(
            int(self.flat_indices[normalized]), self.position_count
        )
        source, target = self.edges[edge_index]
        return source, target, position


class ResponseUnitSequence(Sequence):
    def __init__(
        self,
        edges: tuple[tuple[int, int], ...],
        position_count: int,
        query_count: int,
        flat_indices: np.ndarray | None = None,
    ) -> None:
        self.edges = edges
        self.position_count = int(position_count)
        self.query_count = int(query_count)
        self.total_count = len(edges) * self.position_count * self.query_count
        self.flat_indices = (
            None
            if flat_indices is None
            else np.asarray(flat_indices, dtype=np.int32)
        )

    def __len__(self) -> int:
        return self.total_count if self.flat_indices is None else int(self.flat_indices.size)

    def __getitem__(self, index):
        normalized = _sequence_index(index, len(self))
        if isinstance(normalized, tuple):
            return tuple(self[value] for value in normalized)
        flat = normalized if self.flat_indices is None else int(self.flat_indices[normalized])
        edge_position, query = divmod(flat, self.query_count)
        edge_index, position = divmod(edge_position, self.position_count)
        source, target = self.edges[edge_index]
        return source, target, position, query


class ResponseBundleSequence(Sequence):
    def __init__(
        self,
        source_targets: tuple[tuple[int, tuple[int, ...]], ...],
        position_count: int,
        query_count: int,
        flat_indices: np.ndarray | None = None,
    ) -> None:
        self.source_targets = source_targets
        self.position_count = int(position_count)
        self.query_count = int(query_count)
        self.total_count = len(source_targets) * self.position_count * self.query_count
        self.flat_indices = (
            None
            if flat_indices is None
            else np.asarray(flat_indices, dtype=np.int32)
        )

    def __len__(self) -> int:
        return self.total_count if self.flat_indices is None else int(self.flat_indices.size)

    def __getitem__(self, index):
        normalized = _sequence_index(index, len(self))
        if isinstance(normalized, tuple):
            return tuple(self[value] for value in normalized)
        flat = normalized if self.flat_indices is None else int(self.flat_indices[normalized])
        source_position, query = divmod(flat, self.query_count)
        source_index, position = divmod(source_position, self.position_count)
        source, targets = self.source_targets[source_index]
        return source, targets, position, query


class ResponseTargetsBySourceQuery(Mapping):
    def __init__(
        self,
        targets: dict[tuple[int, int], tuple[int, ...]],
        position_count: int,
        query_count: int,
    ) -> None:
        self.targets = dict(targets)
        self.position_count = int(position_count)
        self.query_count = int(query_count)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 4:
            raise KeyError(key)
        scene, source, position, query = (int(value) for value in key)
        if not 0 <= position < self.position_count or not 0 <= query < self.query_count:
            raise KeyError(key)
        return self.targets[(scene, source)]

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        for scene, source in self.targets:
            for position in range(self.position_count):
                for query in range(self.query_count):
                    yield scene, source, position, query

    def __len__(self) -> int:
        return len(self.targets) * self.position_count * self.query_count


@dataclass
class TrainingCorpus:
    dataset: FormalDataset
    scenes: tuple[int, ...]
    routed: RoutedEdges
    teacher: TeacherBundle
    normalization: TrainingNormalization
    material_categories: int
    normalized_maps: dict[tuple[int, int], np.ndarray]
    normalized_actions: dict[tuple[int, int, int], np.ndarray]
    alignment_bank: tuple
    alignment_audit_bank: tuple
    endpoint: dict[int, Sequence[tuple[int, int]]]
    natural_endpoint: dict[int, Sequence[tuple[int, int]]]
    alignment_active: dict[int, Sequence[tuple[int, int, int]]]
    alignment_null: dict[int, Sequence[tuple[int, int, int]]]
    response_all: dict[int, Sequence[tuple[int, int, int, int]]]
    response_active: dict[int, Sequence[tuple[int, int, int, int]]]
    response_null: dict[int, Sequence[tuple[int, int, int, int]]]
    response_bundles: dict[int, Sequence[tuple[int, tuple[int, ...], int, int]]]
    response_active_bundles: dict[int, Sequence[tuple[int, tuple[int, ...], int, int]]]
    response_null_bundles: dict[int, Sequence[tuple[int, tuple[int, ...], int, int]]]
    response_targets_by_source_query: Mapping[
        tuple[int, int, int, int], tuple[int, ...]
    ]


@dataclass(frozen=True)
class StepPlan:
    endpoint: tuple
    natural_endpoint: tuple
    alignment_active: tuple
    alignment_null: tuple
    response_all: tuple
    response_active: tuple
    response_null: tuple
    endpoint_masks: tuple[int, ...]
    natural_masks: tuple[int, ...]
    response_all_masks: tuple[int, ...]
    response_active_masks: tuple[int, ...]
    response_null_masks: tuple[int, ...]
    response_all_weights: tuple[float, ...]
    response_active_weights: tuple[float, ...]
    response_null_weights: tuple[float, ...]
    response_bundle_count: int


def _train_device_queue(
    config,
    corpus,
    pilot,
    device,
    jobs,
    stop_event,
    *,
    checkpoint_root=None,
    checkpoint_context=None,
    checkpoint_interval_steps=FORMAL_CHECKPOINT_INTERVAL_STEPS,
):
    completed = []
    try:
        for job_index, seed, arm, model in jobs:
            if stop_event.is_set():
                break
            model.to(device)
            arguments = {
                "device": device,
                "prepared_model": model,
                "stop_event": stop_event,
            }
            if checkpoint_root is not None:
                arguments.update(
                    {
                        "checkpoint_path": (
                            Path(checkpoint_root) / f"seed_{seed}" / arm
                        ),
                        "checkpoint_context": checkpoint_context,
                        "checkpoint_interval_steps": checkpoint_interval_steps,
                    }
                )
            result = _train_arm(config, corpus, seed, arm, pilot, **arguments)
            result[0].to("cpu")
            completed.append((job_index, result))
    except Exception:
        stop_event.set()
        raise
    return completed


def _run_training_jobs(
    config,
    corpus,
    pilot,
    jobs_by_device,
    *,
    checkpoint_root=None,
    checkpoint_context=None,
    checkpoint_interval_steps=FORMAL_CHECKPOINT_INTERVAL_STEPS,
):
    stop_event = threading.Event()
    total_jobs = sum(len(jobs) for jobs in jobs_by_device.values())
    ordered = [None] * total_jobs
    if len(jobs_by_device) == 1:
        device, jobs = next(iter(jobs_by_device.items()))
        completed = _train_device_queue(
            config,
            corpus,
            pilot,
            device,
            jobs,
            stop_event,
            checkpoint_root=checkpoint_root,
            checkpoint_context=checkpoint_context,
            checkpoint_interval_steps=checkpoint_interval_steps,
        )
    else:
        with ThreadPoolExecutor(
            max_workers=len(jobs_by_device),
            thread_name_prefix="csi-pairs-gpu",
        ) as executor:
            futures = [
                executor.submit(
                    _train_device_queue,
                    config,
                    corpus,
                    pilot,
                    device,
                    jobs,
                    stop_event,
                    checkpoint_root=checkpoint_root,
                    checkpoint_context=checkpoint_context,
                    checkpoint_interval_steps=checkpoint_interval_steps,
                )
                for device, jobs in jobs_by_device.items()
            ]
            completed = []
            try:
                for future in as_completed(futures):
                    completed.extend(future.result())
            except Exception:
                stop_event.set()
                for future in futures:
                    future.cancel()
                raise
    for job_index, result in completed:
        ordered[job_index] = result
    if any(result is None for result in ordered):
        raise RuntimeError("formal factorial scheduler did not complete every arm and seed")
    return ordered


def run_formal_factorial(
    config: dict,
    dataset: FormalDataset,
    output_root: str | Path,
    qualification_gate: dict,
    allow_nonscientific_fixture: bool = False,
) -> dict:
    require_torch()
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root,
        config,
        dataset,
        ("source_encoder_train", "source_method_selection", "target"),
    )
    require_manifested_formal_qualification(
        qualification_gate,
        config,
        dataset,
        allow_nonscientific_fixture=allow_nonscientific_fixture,
    )
    teacher_path = Path(str(qualification_gate["teacher_checkpoint"]))
    if sha256_file(teacher_path) != qualification_gate["teacher_checkpoint_sha256"]:
        raise RuntimeError("teacher checkpoint hash does not match the qualification gate")
    execution_devices = resolve_execution_devices(dataset)
    execution_device = execution_devices[0]
    teacher = load_teacher_bundle(teacher_path, config, device=execution_device)
    output_dir = Path(output_root) / "factorial"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scenes = dataset.indices_for_role("source_encoder_train")
    selection_scenes = dataset.indices_for_role("source_method_selection")
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(dataset, train_scenes, route_normalization, teacher.patch_spec)
    normalization_path = output_dir / "normalization.json"
    write_json(normalization_path, _normalization_record(normalization))
    train_corpus = _build_corpus(dataset, train_scenes, teacher, config, route_normalization, normalization)
    selection_corpus = _build_corpus(
        dataset, selection_scenes, teacher, config, route_normalization, normalization
    )
    pilot = _pilot_scales(
        config,
        train_corpus,
        selection_corpus,
        int(config["seeds"][0]) + 6001,
        device=execution_device,
    )
    frozen_pilot_path = output_dir / "frozen_pilot.json"
    write_json(
        frozen_pilot_path,
        {
            "schema_version": "csi-pairs-v6-frozen-pilot-v1",
            "source_roles": ["source_encoder_train", "source_method_selection"],
            "role_permissions": {
                "source_encoder_train": "endpoint-only pilot optimization",
                "source_method_selection": "no-op kappa and loss-scale selection only",
            },
            **pilot,
        },
    )

    models: dict[tuple[int, str], object] = {}
    training_rows = []
    checkpoint_rows = []
    provisional_use = "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    evidence = evidence_context(config, dataset, provisional_use)
    checkpoint_context = {
        "schema_version": "csi-pairs-v6-factorial-resume-context-v1",
        **evidence,
        "teacher_checkpoint_sha256": qualification_gate["teacher_checkpoint_sha256"],
        "normalization_sha256": sha256_file(normalization_path),
        "frozen_pilot_sha256": sha256_file(frozen_pilot_path),
        "arms": list(ARMS),
        "seeds": [int(seed) for seed in config["seeds"]],
        "steps": int(config["factorial"]["steps"]),
        "batch_size": int(config["factorial"]["batch_size"]),
        "checkpoint_interval_steps": FORMAL_CHECKPOINT_INTERVAL_STEPS,
    }
    jobs = []
    jobs_by_device = {device: [] for device in execution_devices}
    for job_index, (seed, arm) in enumerate(
        (int(seed), arm) for seed in config["seeds"] for arm in ARMS
    ):
        device = execution_devices[job_index % len(execution_devices)]
        model = _new_model(config, train_corpus, seed, device=torch.device("cpu"))
        jobs.append((seed, arm, device, model))
        jobs_by_device[device].append(
            (job_index, seed, arm, model)
        )
    resume_root = output_dir / "resume"
    trained = _run_training_jobs(
        config,
        train_corpus,
        pilot,
        jobs_by_device,
        checkpoint_root=resume_root,
        checkpoint_context=checkpoint_context,
        checkpoint_interval_steps=FORMAL_CHECKPOINT_INTERVAL_STEPS,
    )

    for (seed, arm, _device, _prepared), (model, row) in zip(jobs, trained):
        checkpoint = output_dir / "checkpoints" / f"seed_{seed}" / f"{arm}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "csi-pairs-formal-checkpoint-v2.1-v6",
                "arm": arm,
                "seed": int(seed),
                "model_spec": _model_spec(config, train_corpus),
                "normalization": _normalization_record(normalization),
                "teacher_checkpoint_sha256": qualification_gate["teacher_checkpoint_sha256"],
                "checkpoint_rule": "fixed_final_step_no_target_selection",
                "state_dict": portable_state_dict(model),
                **evidence,
            },
            checkpoint,
        )
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            reloaded.get("schema_version") != "csi-pairs-formal-checkpoint-v2.1-v6"
            or reloaded.get("arm") != arm
            or int(reloaded.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError("factorial checkpoint identity changed during serialization")
        model.load_state_dict(reloaded["state_dict"], strict=True)
        row.update(
            {
                "checkpoint": str(checkpoint.relative_to(output_dir)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "alignment_scale": pilot["alignment_scale"],
                "response_scale": pilot["response_scale"],
                "alignment_null_tolerance": pilot["alignment_null_tolerance"],
                "execution_device": row["execution_device"],
                "parallel_worker_count": len(execution_devices),
            }
        )
        training_rows.append(row)
        checkpoint_rows.append(
            {
                "seed": int(seed),
                "arm": arm,
                "path": str(checkpoint.relative_to(output_dir)),
                "sha256": sha256_file(checkpoint),
                "parameters": row["parameters"],
                "measured_flops_per_step": row["measured_flops_per_step"],
                "execution_device": row["execution_device"],
                "teacher_checkpoint_sha256": qualification_gate["teacher_checkpoint_sha256"],
            }
        )
        models[(int(seed), arm)] = model
    write_csv(output_dir / "training_summary.csv", bind_rows(training_rows, evidence))
    write_json(
        output_dir / "checkpoint_index.json",
        {
            "schema_version": "csi-pairs-formal-checkpoint-index-v2.1-v6",
            **evidence,
            "checkpoints": bind_rows(checkpoint_rows, evidence),
        },
    )
    resume_rows = []
    for seed, arm, _device, _prepared in jobs:
        pointer = resume_root / f"seed_{seed}" / f"{arm}.latest.json"
        receipt = read_strict_json(pointer)
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "COMPLETE"
            or receipt.get("completed_steps") != int(config["factorial"]["steps"])
        ):
            raise RuntimeError("formal factorial resume pointer is not complete")
        resume_rows.append(
            {
                "seed": int(seed),
                "arm": arm,
                "pointer_path": str(pointer.relative_to(output_dir)),
                "pointer_sha256": sha256_file(pointer),
                "checkpoint_path": str(
                    (pointer.parent / receipt["checkpoint_path"]).relative_to(output_dir)
                ),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "completed_steps": receipt["completed_steps"],
            }
        )
    write_json(
        output_dir / "resume_index.json",
        {
            "schema_version": "csi-pairs-v6-factorial-resume-index-v1",
            **evidence,
            "checkpoint_interval_steps": FORMAL_CHECKPOINT_INTERVAL_STEPS,
            "jobs": resume_rows,
        },
    )

    bank_rows, sample_rows = _run_localization(
        config,
        dataset,
        models,
        normalization,
        teacher.patch_spec,
        execution_devices,
    )
    write_csv(output_dir / "localization_per_sample.csv", bind_rows(sample_rows, evidence))
    write_csv(output_dir / "localization_per_bank.csv", bind_rows(bank_rows, evidence))
    summary_rows = _localization_summary(bank_rows)
    write_csv(output_dir / "localization_summary.csv", bind_rows(summary_rows, evidence))
    statistics = _factorial_statistics(config, bank_rows)
    write_json(output_dir / "factorial_statistics.json", {**statistics, **evidence})
    gate = _preliminary_factorial_gate(
        config,
        dataset,
        bank_rows,
        statistics,
        evidence,
        qualification_gate_sha256=sha256_file(
            Path(output_root) / "qualification" / "gate.json"
        ),
    )
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "information_budget_checklist.json",
        _information_budget_checklist(config),
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def _training_normalization(
    dataset: FormalDataset,
    scenes: np.ndarray,
    route_normalization: RouteNormalization,
    spec: PatchSpec,
) -> TrainingNormalization:
    patch_mean = patchify_csi(route_normalization.channel_mean, spec)
    patch_scale = patchify_csi(route_normalization.channel_scale, spec)
    maps = dataset.maps[scenes]
    map_scale = np.max(np.abs(maps), axis=(0, 1, 3, 4))
    map_scale[map_scale < 1e-9] = 1.0
    radio = np.concatenate((dataset.radio_config[scenes], dataset.bs_pose[scenes]), axis=1)
    radio_mean = radio.mean(axis=0)
    radio_scale = radio.std(axis=0)
    radio_scale[radio_scale < 1e-9] = 1.0
    material_categories = int(dataset.metadata["assets"]["material_category_count"])
    occupancy_index = list(dataset.map_channel_names).index("occupancy")
    height_index = list(dataset.map_channel_names).index("height")
    action_scale = np.ones(4 + 2 * material_categories, dtype=np.float64)
    action_scale[:2] = map_scale[occupancy_index]
    action_scale[2:4] = map_scale[height_index]
    position_values = dataset.positions[scenes].reshape(-1, 2)
    position_mean = position_values.mean(axis=0)
    position_scale = position_values.std(axis=0)
    position_scale[position_scale < 1e-9] = 1.0
    return TrainingNormalization(
        patch_mean=patch_mean,
        patch_scale=patch_scale,
        latent_mean=route_normalization.latent_mean,
        latent_scale=route_normalization.latent_scale,
        map_scale=map_scale,
        radio_mean=radio_mean,
        radio_scale=radio_scale,
        action_scale=action_scale,
        position_mean=position_mean,
        position_scale=position_scale,
    )


def _normalization_record(normalization: TrainingNormalization) -> dict:
    return {
        "source_roles": ["source_encoder_train"],
        "patch_mean": normalization.patch_mean.tolist(),
        "patch_scale": normalization.patch_scale.tolist(),
        "latent_mean": normalization.latent_mean.tolist(),
        "latent_scale": normalization.latent_scale.tolist(),
        "map_scale": normalization.map_scale.tolist(),
        "radio_mean": normalization.radio_mean.tolist(),
        "radio_scale": normalization.radio_scale.tolist(),
        "action_scale": normalization.action_scale.tolist(),
        "position_mean": normalization.position_mean.tolist(),
        "position_scale": normalization.position_scale.tolist(),
    }


def _build_corpus(dataset, scenes, teacher, config, route_normalization, normalization):
    routed = route_dataset(dataset, teacher, config, scenes, normalization=route_normalization)
    scene_tuple = tuple(int(value) for value in scenes)
    normalized_maps = {}
    normalized_actions = {}
    for scene in scene_tuple:
        scene_maps = _cached_model_spatial_input(
            _normalized_map(normalization, dataset.maps[scene])
        )
        for world in range(dataset.world_count):
            normalized_maps[(scene, world)] = scene_maps[world]

        scene_edges = tuple(dataset.directed_edges(scene))
        sources = np.asarray(
            [int(edge.source_world) for edge in scene_edges], dtype=np.int64
        )
        targets = np.asarray(
            [int(edge.target_world) for edge in scene_edges], dtype=np.int64
        )
        actions = typed_signed_edit(
            dataset.maps[scene, sources],
            dataset.maps[scene, targets],
            dataset.map_channel_names,
            int(dataset.metadata["assets"]["material_category_count"]),
        )
        cached_actions = _cached_model_spatial_input(
            _normalized_action(normalization, actions)
        )
        for edge, cached_action in zip(scene_edges, cached_actions, strict=True):
            key = (scene, int(edge.source_world), int(edge.target_world))
            if key in normalized_actions:
                raise RuntimeError("duplicate directed edge in response action cache")
            normalized_actions[key] = cached_action
    endpoint = {}
    natural_endpoint = {}
    alignment_active = {}
    alignment_null = {}
    response_all = {}
    response_active = {}
    response_null = {}
    response_bundles = {}
    response_active_bundles = {}
    response_null_bundles = {}
    response_targets = {}
    query_count = int(teacher.patch_spec.patch_count)
    for scene in scene_tuple:
        endpoint[scene] = CartesianEndpointSequence(
            tuple(range(dataset.world_count)), dataset.position_count
        )
        natural = int(dataset.natural_world_index[scene])
        natural_endpoint[scene] = CartesianEndpointSequence(
            (natural,), dataset.position_count
        )
        scene_edges = tuple(dataset.directed_edges(scene))
        edge_pairs = tuple(
            (int(edge.source_world), int(edge.target_world)) for edge in scene_edges
        )
        if not isinstance(routed.response_route, CompactEdgeTensorMapping) or not isinstance(
            routed.alignment_route, CompactEdgeTensorMapping
        ):
            raise RuntimeError("training corpus requires compact routed edge tensors")
        scene_response_routes = routed.response_route.values_for_scene(scene)
        scene_alignment_routes = routed.alignment_route.values_for_scene(scene)
        if scene_response_routes.shape != (
            len(scene_edges),
            dataset.position_count,
            query_count,
        ):
            raise RuntimeError("compact response route tensor is not aligned to scene edges")

        response_all[scene] = ResponseUnitSequence(
            edge_pairs, dataset.position_count, query_count
        )
        response_active[scene] = ResponseUnitSequence(
            edge_pairs,
            dataset.position_count,
            query_count,
            np.flatnonzero(scene_response_routes.reshape(-1) == 2),
        )
        response_null[scene] = ResponseUnitSequence(
            edge_pairs,
            dataset.position_count,
            query_count,
            np.flatnonzero(scene_response_routes.reshape(-1) == 0),
        )

        alignment_edge_indices = tuple(
            index
            for index, edge in enumerate(scene_edges)
            if edge.source_world < edge.target_world
            and headline_alignment_edge(dataset, scene, edge)
        )
        alignment_edges = tuple(edge_pairs[index] for index in alignment_edge_indices)
        alignment_routes = scene_alignment_routes[
            np.asarray(alignment_edge_indices, dtype=np.int64)
        ]
        alignment_active[scene] = EdgePositionSequence(
            alignment_edges,
            dataset.position_count,
            np.flatnonzero(alignment_routes.reshape(-1) == 2),
        )
        alignment_null[scene] = EdgePositionSequence(
            alignment_edges,
            dataset.position_count,
            np.flatnonzero(alignment_routes.reshape(-1) == 0),
        )

        source_targets = tuple(
            (
                source,
                tuple(
                    sorted(
                        edge.target_world
                        for edge in scene_edges
                        if edge.source_world == source
                    )
                ),
            )
            for source in range(dataset.world_count)
        )
        pair_to_edge = {pair: index for index, pair in enumerate(edge_pairs)}
        active_bundle_indices = []
        null_bundle_indices = []
        bundle_stride = dataset.position_count * query_count
        for source_index, (source, targets) in enumerate(source_targets):
            if len(targets) < 2:
                raise RuntimeError("V6 branch bundle requires at least two actions per source state")
            actions = [normalized_actions[(scene, source, target)] for target in targets]
            if len({np.asarray(action).tobytes() for action in actions}) < 2:
                raise RuntimeError("branch bundle targets are not action-distinguishable")
            response_targets[(scene, source)] = tuple(int(target) for target in targets)
            target_edge_indices = np.asarray(
                [pair_to_edge[(source, int(target))] for target in targets],
                dtype=np.int64,
            )
            target_routes = scene_response_routes[target_edge_indices]
            active_local = np.flatnonzero(np.any(target_routes == 2, axis=0).reshape(-1))
            null_local = np.flatnonzero(np.any(target_routes == 0, axis=0).reshape(-1))
            active_bundle_indices.extend(
                (source_index * bundle_stride + active_local).tolist()
            )
            null_bundle_indices.extend(
                (source_index * bundle_stride + null_local).tolist()
            )
        response_bundles[scene] = ResponseBundleSequence(
            source_targets, dataset.position_count, query_count
        )
        response_active_bundles[scene] = ResponseBundleSequence(
            source_targets,
            dataset.position_count,
            query_count,
            np.asarray(active_bundle_indices, dtype=np.int32),
        )
        response_null_bundles[scene] = ResponseBundleSequence(
            source_targets,
            dataset.position_count,
            query_count,
            np.asarray(null_bundle_indices, dtype=np.int32),
        )
    for name, table in (
        ("endpoint", endpoint),
        ("natural endpoint", natural_endpoint),
        ("alignment active", alignment_active),
        ("alignment null", alignment_null),
        ("response all", response_all),
        ("response active", response_active),
        ("response null", response_null),
    ):
        if not any(table.values()):
            raise RuntimeError(f"no eligible banks for {name}")
    alignment_bank = tuple(entry for entry in teacher.mask_bank if entry.mode == "random_75")
    expected_queries = int(config["factorial"]["alignment_bank_queries"])
    if len(alignment_bank) != expected_queries:
        raise RuntimeError(
            f"frozen B_align must cover every patch exactly once; config={expected_queries}, actual={len(alignment_bank)}"
        )
    return TrainingCorpus(
        dataset=dataset,
        scenes=scene_tuple,
        routed=routed,
        teacher=teacher,
        normalization=normalization,
        material_categories=int(dataset.metadata["assets"]["material_category_count"]),
        normalized_maps=normalized_maps,
        normalized_actions=normalized_actions,
        alignment_bank=alignment_bank,
        alignment_audit_bank=tuple(
            entry for entry in teacher.audit_mask_bank if entry.mode == "random_75"
        ),
        endpoint=endpoint,
        natural_endpoint=natural_endpoint,
        alignment_active=alignment_active,
        alignment_null=alignment_null,
        response_all=response_all,
        response_active=response_active,
        response_null=response_null,
        response_bundles=response_bundles,
        response_active_bundles=response_active_bundles,
        response_null_bundles=response_null_bundles,
        response_targets_by_source_query=ResponseTargetsBySourceQuery(
            response_targets, dataset.position_count, query_count
        ),
    )


def _make_plan(corpus: TrainingCorpus, batch_size: int, seed: int, step: int) -> StepPlan:
    rng = np.random.default_rng(int(seed) * 1000003 + int(step))
    endpoint = _macro_sample(corpus.endpoint, rng, batch_size, corpus.dataset)
    natural = _macro_sample(corpus.natural_endpoint, rng, batch_size, corpus.dataset)
    alignment_active = _macro_sample(corpus.alignment_active, rng, batch_size, corpus.dataset)
    alignment_null = _macro_sample(corpus.alignment_null, rng, batch_size, corpus.dataset)
    # These are three different macro estimands.  In particular, the all-edge
    # target term must not inherit the former 50:50 active/null conditioning.
    response_all, response_all_weights = _macro_bundle_sample(
        corpus, corpus.response_all, rng, batch_size, route=None
    )
    response_active, response_active_weights = _macro_bundle_sample(
        corpus, corpus.response_active, rng, batch_size, route=2
    )
    response_null, response_null_weights = _macro_bundle_sample(
        corpus, corpus.response_null, rng, batch_size, route=0
    )

    # Every occurrence of the same source CSI/query uses byte-identical input
    # masking, including sibling actions and occurrences shared by target and
    # conditional estimators.  There is currently no stochastic augmentation
    # outside the frozen mask bank.
    response_mask_by_source_query = {}
    for unit in (*response_all, *response_active, *response_null):
        key = _response_source_query_key(unit)
        if key not in response_mask_by_source_query:
            response_mask_by_source_query[key] = _mask_index_for_query(
                corpus, unit[-1], rng
            )

    def response_masks(units):
        return tuple(
            response_mask_by_source_query[_response_source_query_key(unit)]
            for unit in units
        )

    return StepPlan(
        endpoint=tuple(endpoint),
        natural_endpoint=tuple(natural),
        alignment_active=tuple(alignment_active),
        alignment_null=tuple(alignment_null),
        response_all=tuple(response_all),
        response_active=tuple(response_active),
        response_null=tuple(response_null),
        endpoint_masks=tuple(int(rng.integers(0, len(corpus.teacher.mask_bank))) for _ in endpoint),
        natural_masks=tuple(int(rng.integers(0, len(corpus.teacher.mask_bank))) for _ in natural),
        response_all_masks=response_masks(response_all),
        response_active_masks=response_masks(response_active),
        response_null_masks=response_masks(response_null),
        response_all_weights=response_all_weights,
        response_active_weights=response_active_weights,
        response_null_weights=response_null_weights,
        response_bundle_count=3 * int(batch_size),
    )


def _response_source_query_key(unit):
    scene, source, _target, position, query = unit
    return int(scene), int(source), int(position), int(query)


def _macro_bundle_sample(corpus, anchor_table, rng, count, *, route):
    """Unbiased edge estimator that keeps every sampled source branch bundle."""
    anchors = _macro_sample(anchor_table, rng, count, corpus.dataset)
    units = []
    weights = []
    for scene, source, _anchor_target, position, query in anchors:
        key = int(scene), int(source), int(position), int(query)
        targets = corpus.response_targets_by_source_query.get(key)
        if targets is None:
            raise RuntimeError("response anchor has no registered sibling bundle")
        selected = [
            target
            for target in targets
            if route is None
            or corpus.routed.response_route[(scene, source, target, position, query)] == route
        ]
        if not selected:
            raise RuntimeError("response route anchor produced an empty conditional bundle")
        unit_weight = 1.0 / len(selected)
        units.extend((scene, source, target, position, query) for target in selected)
        weights.extend(unit_weight for _ in selected)
    return tuple(units), tuple(weights)


def _macro_sample(table, rng, count, dataset):
    eligible = [scene for scene, values in table.items() if values]
    if not eligible:
        raise RuntimeError("empty eligible-bank conditional sampler")
    by_cluster = {}
    for scene in eligible:
        by_cluster.setdefault(str(dataset.base_map_cluster_ids[scene]), []).append(scene)
    clusters = sorted(by_cluster)
    output = []
    for _ in range(int(count)):
        cluster = clusters[int(rng.integers(0, len(clusters)))]
        scenes = by_cluster[cluster]
        scene = scenes[int(rng.integers(0, len(scenes)))]
        values = table[scene]
        value = values[int(rng.integers(0, len(values)))]
        output.append((scene, *value))
    return output


def _mask_index_for_query(corpus, query, rng):
    candidates = [
        index for index, entry in enumerate(corpus.teacher.mask_bank) if int(entry.query) == int(query)
    ]
    return candidates[int(rng.integers(0, len(candidates)))]


def _new_model(
    config: dict,
    corpus: TrainingCorpus,
    seed: int,
    *,
    model_spec: dict | None = None,
    device=None,
):
    torch.manual_seed(int(seed))
    spec = _model_spec(config, corpus) if model_spec is None else dict(model_spec)
    execution_device = module_device(corpus.teacher.teacher) if device is None else torch.device(device)
    model = CSIPairsFormalModel(**spec).to(execution_device)
    teacher = corpus.teacher.teacher
    if int(spec["csi_encoder_layers"]) == len(teacher.encoder.layers):
        model.initialize_csi_from_teacher(teacher)
    else:
        if int(spec["csi_encoder_layers"]) > len(teacher.encoder.layers):
            raise ValueError("control encoder cannot add CSI layers beyond the frozen teacher")
        model.csi_patch_embedding.load_state_dict(teacher.patch_embedding.state_dict())
        with torch.no_grad():
            model.csi_mask_token.copy_(teacher.mask_token)
            model.csi_fixed_position.copy_(teacher.fixed_position)
        for index, layer in enumerate(model.csi_encoder.layers):
            layer.load_state_dict(teacher.encoder.layers[index].state_dict(), strict=True)
    return model


def _model_spec(config, corpus):
    dataset = corpus.dataset
    spec = corpus.teacher.patch_spec
    return {
        "patch_count": spec.patch_count,
        "patch_rows": spec.patch_rows,
        "patch_columns": spec.patch_columns,
        "patch_dim": spec.patch_dim,
        "map_channels": int(dataset.maps.shape[2]),
        "action_channels": 4 + 2 * corpus.material_categories,
        "radio_dim": int(dataset.radio_config.shape[1] + dataset.bs_pose.shape[1]),
        "latent_dim": int(config["teacher"]["latent_dim"]),
        "state_dim": int(config["model"]["state_dim"]),
        "map_dim": int(config["model"]["map_dim"]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "attention_heads": int(config["model"]["attention_heads"]),
        "csi_encoder_layers": int(config["teacher"]["encoder_layers"]),
    }


def _pilot_scales(config, train_corpus, selection_corpus, seed, *, device=None):
    model = _new_model(config, train_corpus, seed, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    for step in range(int(config["factorial"]["pilot_steps"])):
        plan = _make_plan(train_corpus, int(config["factorial"]["batch_size"]), seed, step)
        endpoint_batch = _identity_batch(
            train_corpus,
            plan.endpoint,
            [train_corpus.teacher.mask_bank[index] for index in plan.endpoint_masks],
        )
        natural_batch = _identity_batch(
            train_corpus,
            plan.natural_endpoint,
            [train_corpus.teacher.mask_bank[index] for index in plan.natural_masks],
        )
        physical_weight = float(config["factorial"]["endpoint_physical_weight"])
        loss = _endpoint_loss(model, endpoint_batch, physical_weight)
        loss = loss + float(config["factorial"]["natural_endpoint_weight"]) * _endpoint_loss(
            model, natural_batch, physical_weight
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite pilot loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    kappa = _estimate_noop_kappa(
        model,
        selection_corpus,
        float(config["qualification"]["alignment_noop_quantile"]),
    )
    alignment_values = []
    response_values = []
    weights = _loss_weights(config, 1.0, 1.0, 1.0, 1.0, kappa)
    model.eval()
    for step in range(int(config["factorial"]["pilot_steps"])):
        plan = _make_plan(
            selection_corpus,
            int(config["factorial"]["batch_size"]),
            seed + 1,
            step,
        )
        with torch.no_grad():
            components = _loss_components(model, selection_corpus, plan, weights)
        alignment_values.append(float(components["alignment"]))
        response_values.append(float(components["response"]))
    return {
        "pilot_seed": int(seed),
        "pilot_steps": int(config["factorial"]["pilot_steps"]),
        "pilot_contract": "endpoint-only source-encoder-train fit; source-method-selection no-op kappa and frozen scale sequence",
        "alignment_scale": max(float(np.mean(alignment_values)), 1e-6),
        "response_scale": max(float(np.mean(response_values)), 1e-6),
        "alignment_null_tolerance": float(kappa),
        "checkpoint_reused_for_final_training": False,
    }


def _estimate_noop_kappa(model, corpus, quantile):
    units = []
    for scene, values in corpus.endpoint.items():
        for world, position in values:
            units.append((scene, world, position))
    gaps = _noop_score_gaps(model, corpus, units)
    if not gaps:
        raise RuntimeError("canonical no-op pilot has no units")
    return float(np.quantile(np.asarray(gaps), float(quantile)))


def _noop_score_gaps(model, corpus, units, *, batch_size=NOOP_SCORE_BATCH_SIZE):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("no-op score batch size must be a positive integer")
    units = tuple(units)
    gaps = []
    for start in range(0, len(units), batch_size):
        chunk = units[start : start + batch_size]
        scores = []
        for use_noop in (False, True):
            supplied_maps = [
                (
                    corpus.dataset.noop_maps[scene, world]
                    if use_noop
                    else corpus.dataset.maps[scene, world]
                )
                for scene, world, _position in chunk
            ]
            errors = []
            for entry in corpus.alignment_bank:
                batch = _identity_batch(
                    corpus,
                    chunk,
                    [entry] * len(chunk),
                    supplied_maps=supplied_maps,
                )
                batch = batch_for_module(model, batch)
                state = model.state(
                    batch["visible"], batch["maps"], batch["radio"], batch["masks"]
                )
                prediction_z, prediction_y = model.predict_identity(
                    state, batch["query"]
                )
                errors.append(
                    endpoint_per_sample(
                        prediction_z,
                        prediction_y,
                        batch["target_z"],
                        batch["target_y"],
                        ALIGNMENT_SCORE_PHYSICAL_WEIGHT,
                    )
                )
            scores.append(-torch.mean(torch.stack(errors), dim=0))
        gaps.extend(torch.abs(scores[0] - scores[1]).detach().cpu().tolist())
    return [float(value) for value in gaps]


def _train_arm(
    config,
    corpus,
    seed,
    arm,
    pilot,
    *,
    pairing_break=None,
    step_count=None,
    model_spec=None,
    loss_trace=None,
    device=None,
    prepared_model=None,
    stop_event=None,
    checkpoint_path=None,
    checkpoint_context=None,
    checkpoint_interval_steps=FORMAL_CHECKPOINT_INTERVAL_STEPS,
    stop_after_step=None,
):
    if prepared_model is not None and model_spec is not None:
        raise ValueError("prepared model cannot be combined with a model specification")
    model = (
        prepared_model
        if prepared_model is not None
        else _new_model(
            config,
            corpus,
            seed,
            model_spec=model_spec,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    steps = int(config["factorial"]["steps"] if step_count is None else step_count)
    batch_size = int(config["factorial"]["batch_size"])
    if steps < 1:
        raise ValueError("control training step count must be positive")
    if (checkpoint_path is None) != (checkpoint_context is None):
        raise ValueError("factorial resume path and context must be supplied together")
    if checkpoint_path is not None and (
        type(checkpoint_interval_steps) is not int or checkpoint_interval_steps < 1
    ):
        raise ValueError("factorial checkpoint interval must be a positive integer")
    if stop_after_step is not None and (
        type(stop_after_step) is not int or not 1 <= stop_after_step <= steps
    ):
        raise ValueError("injected interruption step is outside the training schedule")
    alignment_factor, response_factor = ARM_FACTORS[arm]
    weights = _loss_weights(
        config,
        alignment_factor,
        response_factor,
        float(pilot["alignment_scale"]),
        float(pilot["response_scale"]),
        float(pilot["alignment_null_tolerance"]),
    )
    alignment_gradient_norms = []
    response_gradient_norms = []
    raw_alignment_gradient_norms = []
    raw_response_gradient_norms = []
    execution = None
    last_losses = None
    prior_elapsed = 0.0
    generation = -1
    start_step = 0
    if checkpoint_path is not None:
        payload = load_factorial_resume(
            checkpoint_path,
            context=checkpoint_context,
            seed=int(seed),
            arm=arm,
            total_steps=steps,
            batch_size=batch_size,
            map_location=module_device(model),
        )
        if payload is not None:
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            state = payload["training_state"]
            required_state = {
                "execution",
                "alignment_gradient_norms",
                "response_gradient_norms",
                "raw_alignment_gradient_norms",
                "raw_response_gradient_norms",
                "last_losses",
                "elapsed_seconds",
                "loss_trace",
            }
            if not isinstance(state, dict) or set(state) != required_state:
                raise RuntimeError("factorial resume training state is invalid")
            execution = state["execution"]
            alignment_gradient_norms = list(state["alignment_gradient_norms"])
            response_gradient_norms = list(state["response_gradient_norms"])
            raw_alignment_gradient_norms = list(state["raw_alignment_gradient_norms"])
            raw_response_gradient_norms = list(state["raw_response_gradient_norms"])
            last_losses = state["last_losses"]
            prior_elapsed = state["elapsed_seconds"]
            stored_trace = state["loss_trace"]
            if (
                not isinstance(execution, dict)
                or not isinstance(last_losses, dict)
                or not isinstance(prior_elapsed, (int, float))
                or isinstance(prior_elapsed, bool)
                or not math.isfinite(float(prior_elapsed))
                or prior_elapsed < 0.0
                or any(
                    not isinstance(values, list)
                    for values in (
                        alignment_gradient_norms,
                        response_gradient_norms,
                        raw_alignment_gradient_norms,
                        raw_response_gradient_norms,
                    )
                )
            ):
                raise RuntimeError("factorial resume numeric state is invalid")
            if (loss_trace is None) != (stored_trace is None):
                raise RuntimeError("factorial resume loss-trace contract changed")
            if loss_trace is not None:
                if not isinstance(stored_trace, list):
                    raise RuntimeError("factorial resume loss trace is invalid")
                loss_trace.clear()
                loss_trace.extend(stored_trace)
            start_step = int(payload["completed_steps"])
            generation = int(payload["generation"])

    started = time.perf_counter()

    def persist(completed_steps):
        nonlocal generation
        if checkpoint_path is None:
            return
        if execution is None or last_losses is None:
            raise RuntimeError("cannot checkpoint factorial training before one complete step")
        elapsed = prior_elapsed + (time.perf_counter() - started)
        generation = save_factorial_resume(
            checkpoint_path,
            previous_generation=generation,
            context=checkpoint_context,
            seed=int(seed),
            arm=arm,
            completed_steps=completed_steps,
            total_steps=steps,
            batch_size=batch_size,
            model=model,
            optimizer=optimizer,
            training_state={
                "execution": execution,
                "alignment_gradient_norms": alignment_gradient_norms,
                "response_gradient_norms": response_gradient_norms,
                "raw_alignment_gradient_norms": raw_alignment_gradient_norms,
                "raw_response_gradient_norms": raw_response_gradient_norms,
                "last_losses": last_losses,
                "elapsed_seconds": elapsed,
                "loss_trace": list(loss_trace) if loss_trace is not None else None,
            },
        )

    for step in range(start_step, steps):
        if stop_event is not None and stop_event.is_set():
            if step > 0 and execution is not None and last_losses is not None:
                persist(step)
            raise RuntimeError(
                "formal factorial training cancelled after a peer-device failure"
            )
        plan = _make_plan(corpus, batch_size, seed, step)
        if execution is None:
            execution = _measure_execution(
                model,
                corpus,
                plan,
                weights,
            )
        components = _loss_components(
            model, corpus, plan, weights, pairing_break=pairing_break
        )
        if step in {0, steps - 1}:
            retained = _retained_parameters(model)
            raw_alignment_gradient_norms.append(_gradient_norm(components["alignment"], retained))
            raw_response_gradient_norms.append(_gradient_norm(components["response"], retained))
            alignment_gradient_norms.append(
                _gradient_norm(
                    float(weights.alignment)
                    * components["alignment"]
                    / max(float(weights.alignment_scale), 1e-12),
                    retained,
                )
            )
            response_gradient_norms.append(
                _gradient_norm(
                    float(weights.response)
                    * components["response"]
                    / max(float(weights.response_scale), 1e-12),
                    retained,
                )
            )
        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        if loss_trace is not None:
            squared_gradient = sum(
                float(torch.sum(parameter.grad.detach() ** 2))
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            loss_trace.append(
                {
                    "step": step + 1,
                    "total_loss": float(components["total"].detach()),
                    "endpoint_loss": float(components["endpoint"].detach()),
                    "alignment_loss": float(
                        weights.alignment
                        * components["alignment"].detach()
                        / max(float(weights.alignment_scale), 1e-12)
                    ),
                    "response_loss": float(
                        weights.response
                        * components["response"].detach()
                        / max(float(weights.response_scale), 1e-12)
                    ),
                    "gradient_norm": math.sqrt(max(squared_gradient, 0.0)),
                }
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        last_losses = {
            name: float(value.detach()) for name, value in components.items()
        }
        completed_steps = step + 1
        if checkpoint_path is not None and (
            completed_steps % checkpoint_interval_steps == 0
            or completed_steps == steps
            or completed_steps == stop_after_step
        ):
            persist(completed_steps)
        if stop_after_step == completed_steps and completed_steps < steps:
            raise RuntimeError(
                "injected formal factorial interruption after a durable checkpoint"
            )
    elapsed = (
        prior_elapsed
        if start_step == steps
        else prior_elapsed + (time.perf_counter() - started)
    )
    if last_losses is None:
        raise RuntimeError("training schedule must contain at least one step")
    if execution is None:
        raise RuntimeError("training execution measurement was not performed")
    row = {
        "seed": int(seed),
        "arm": arm,
        "steps": steps,
        "batch_size": batch_size,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": elapsed,
        "state_calls_per_step": execution["state_calls"],
        "predict_calls_per_step": execution["predict_calls"],
        "forward_calls_per_step": execution["state_calls"] + execution["predict_calls"],
        "total_forward_calls": (execution["state_calls"] + execution["predict_calls"])
        * steps,
        "measured_flops_per_step": execution["flops"],
        "flop_measurement_status": execution["flop_measurement_status"],
        "compute_contract": "all arms reuse the identical deterministic batch plan; disabled alignment/response supervision skips its forward graph and has zero raw and weighted gradient",
        "alignment_score_physical_weight": ALIGNMENT_SCORE_PHYSICAL_WEIGHT,
        "alignment_gradient_norm_mean": float(np.mean(alignment_gradient_norms)),
        "response_gradient_norm_mean": float(np.mean(response_gradient_norms)),
        "raw_alignment_gradient_norm_mean": float(np.mean(raw_alignment_gradient_norms)),
        "raw_response_gradient_norm_mean": float(np.mean(raw_response_gradient_norms)),
        "checkpoint_rule": "fixed_final_step_no_target_selection",
        "execution_device": str(module_device(model)),
        "batch_plan_contract": "identical bank/edge/direction/mask/query sequence for all arms at a seed",
        "response_macro_contract": "independent cluster-bank anchor-edge macro samples for all-directed-edge target, active conditional, and null conditional terms; complete sibling bundles use inverse bundle-size weights",
        "response_all_anchor_bundles_per_step": int(config["factorial"]["batch_size"]),
        "response_active_anchor_bundles_per_step": int(config["factorial"]["batch_size"]),
        "response_null_anchor_bundles_per_step": int(config["factorial"]["batch_size"]),
        "response_mask_reuse_key": "scene/source/position/query across target and conditional estimators",
    }
    for name, value in last_losses.items():
        row[f"final_{name}_loss"] = float(value)
    return model, row


def _loss_weights(config, alignment_factor, response_factor, alignment_scale, response_scale, kappa):
    f = config["factorial"]
    q = config["qualification"]
    return LossWeights(
        endpoint_physical=float(f["endpoint_physical_weight"]),
        natural_endpoint=float(f["natural_endpoint_weight"]),
        alignment=alignment_factor * float(f["alignment_weight"]),
        alignment_null=float(f["alignment_null_weight"]),
        response=response_factor * float(f["response_weight"]),
        response_physical=float(f["response_physical_weight"]),
        response_delta=float(f["response_delta_weight"]),
        response_delta_physical=float(f["response_delta_physical_weight"]),
        response_null=float(f["response_null_weight"]),
        response_null_physical=float(f["response_null_physical_weight"]),
        active_margin=float(f["active_margin"]),
        effect_margin_scale=float(f["effect_margin_scale"]),
        effect_margin_cap=float(f["effect_margin_cap"]),
        alignment_primary_margin=str(f["alignment_primary_margin"]),
        alignment_null_tolerance=float(kappa),
        response_latent_null_tolerance=float(q["response_latent_null_rms_max"]),
        response_physical_null_tolerance=float(q["response_physical_null_rms_max"]),
        alignment_scale=float(alignment_scale),
        response_scale=float(response_scale),
    )


def _loss_components(model, corpus, plan, weights, *, pairing_break=None):
    endpoint_batch = _identity_batch(
        corpus, plan.endpoint, [corpus.teacher.mask_bank[index] for index in plan.endpoint_masks]
    )
    natural_batch = _identity_batch(
        corpus,
        plan.natural_endpoint,
        [corpus.teacher.mask_bank[index] for index in plan.natural_masks],
    )
    endpoint = _endpoint_loss(model, endpoint_batch, weights.endpoint_physical)
    natural = _endpoint_loss(model, natural_batch, weights.endpoint_physical)
    endpoint_total = endpoint + float(weights.natural_endpoint) * natural

    zero = endpoint_total.new_zeros(())
    alignment = zero
    alignment_active = zero
    alignment_active_fixed = zero
    alignment_active_effect_aware = zero
    alignment_null = zero
    if float(weights.alignment) != 0.0:
        active_scores = _alignment_scores(
            model,
            corpus,
            plan.alignment_active,
            ALIGNMENT_SCORE_PHYSICAL_WEIGHT,
            supplied_map_units=_pairing_overrides(
                plan.alignment_active, pairing_break, "alignment"
            ),
        )
        null_scores = _alignment_scores(
            model,
            corpus,
            plan.alignment_null,
            ALIGNMENT_SCORE_PHYSICAL_WEIGHT,
            supplied_map_units=_pairing_overrides(
                plan.alignment_null, pairing_break, "alignment"
            ),
        )
        active_effect = torch.as_tensor(
            [
                corpus.routed.alignment_distances[(scene, source, target, position)][0]
                for scene, source, target, position in plan.alignment_active
            ],
            dtype=torch.float32,
            device=module_device(model),
        )
        phi = torch.clamp(
            active_effect / max(float(weights.effect_margin_scale), 1e-12),
            min=0.0,
            max=float(weights.effect_margin_cap),
        )
        alignment_active_effect_aware = torch.mean(
            alignment_active_quartet_loss(
                *active_scores, margin=float(weights.active_margin) * phi
            )
        )
        alignment_active_fixed = torch.mean(
            alignment_active_quartet_loss(
                *active_scores, margin=float(weights.active_margin)
            )
        )
        alignment_active = (
            alignment_active_fixed
            if weights.alignment_primary_margin == "fixed"
            else alignment_active_effect_aware
        )
        alignment_null = torch.mean(
            alignment_null_quartet_loss(
                *null_scores, tolerance=float(weights.alignment_null_tolerance)
            )
        )
        alignment = alignment_active + float(weights.alignment_null) * alignment_null

    response = zero
    response_target = zero
    response_physical = zero
    response_active = zero
    response_null = zero
    if float(weights.response) != 0.0:
        target_batch = _response_batch(
            corpus,
            plan.response_all,
            [corpus.teacher.mask_bank[index] for index in plan.response_all_masks],
            target_units=_pairing_overrides(
                plan.response_all, pairing_break, "response"
            ),
        )
        active_batch = _response_batch(
            corpus,
            plan.response_active,
            [corpus.teacher.mask_bank[index] for index in plan.response_active_masks],
            target_units=_pairing_overrides(
                plan.response_active, pairing_break, "response"
            ),
        )
        null_batch = _response_batch(
            corpus,
            plan.response_null,
            [corpus.teacher.mask_bank[index] for index in plan.response_null_masks],
            target_units=_pairing_overrides(
                plan.response_null, pairing_break, "response"
            ),
        )
        response_target, response_physical = _response_target_loss(
            model, target_batch, plan.response_all_weights
        )
        response_active = _response_active_loss(
            model, active_batch, weights, plan.response_active_weights
        )
        response_null = _response_null_loss(
            model, null_batch, weights, plan.response_null_weights
        )
        response = (
            response_target
            + float(weights.response_physical) * response_physical
            + float(weights.response_delta) * response_active
            + float(weights.response_null) * response_null
        )
    total = (
        endpoint_total
        + float(weights.alignment) * alignment / max(float(weights.alignment_scale), 1e-12)
        + float(weights.response) * response / max(float(weights.response_scale), 1e-12)
    )
    return {
        "total": total,
        "endpoint": endpoint_total,
        "endpoint_world": endpoint,
        "endpoint_natural": natural,
        "alignment": alignment,
        "alignment_active": alignment_active,
        "alignment_active_fixed": alignment_active_fixed,
        "alignment_active_effect_aware": alignment_active_effect_aware,
        "alignment_null": alignment_null,
        "response": response,
        "response_target": response_target,
        "response_physical": response_physical,
        "response_active": response_active,
        "response_null": response_null,
    }


def _endpoint_loss(model, batch, physical_weight):
    batch = batch_for_module(model, batch)
    state = model.state(batch["visible"], batch["maps"], batch["radio"], batch["masks"])
    prediction_z, prediction_y = model.predict_identity(state, batch["query"])
    return torch.mean(
        endpoint_per_sample(
            prediction_z,
            prediction_y,
            batch["target_z"],
            batch["target_y"],
            physical_weight,
        )
    )


def _alignment_scores(
    model, corpus, units, physical_weight, *, supplied_map_units=None
):
    expanded = []
    entries = []
    for unit in units:
        for entry in corpus.alignment_bank:
            expanded.append(unit)
            entries.append(entry)
    left = [(scene, source, position) for scene, source, target, position in expanded]
    right = [(scene, target, position) for scene, source, target, position in expanded]
    map_units = units if supplied_map_units is None else supplied_map_units
    if len(map_units) != len(units):
        raise ValueError("alignment pairing override must match the sampled units")
    expanded_maps = [
        unit for unit in map_units for _ in corpus.alignment_bank
    ]
    map_u = [(scene, source) for scene, source, target, position in expanded_maps]
    map_v = [(scene, target) for scene, source, target, position in expanded_maps]
    batches = (
        _identity_batch(corpus, left, entries, supplied_map_units=map_u),
        _identity_batch(corpus, left, entries, supplied_map_units=map_v),
        _identity_batch(corpus, right, entries, supplied_map_units=map_v),
        _identity_batch(corpus, right, entries, supplied_map_units=map_u),
    )
    scores = []
    for batch in batches:
        batch = batch_for_module(model, batch)
        state = model.state(batch["visible"], batch["maps"], batch["radio"], batch["masks"])
        prediction_z, prediction_y = model.predict_identity(state, batch["query"])
        error = endpoint_per_sample(
            prediction_z,
            prediction_y,
            batch["target_z"],
            batch["target_y"],
            physical_weight,
        )
        scores.append(-error.reshape(len(units), len(corpus.alignment_bank)).mean(dim=1))
    return tuple(scores)


def _response_target_loss(model, batch, sample_weights):
    batch = batch_for_module(model, batch)
    state = model.state(batch["visible"], batch["maps"], batch["radio"], batch["masks"])
    prediction_z, prediction_y = model.predict(state, batch["action"], batch["query"])
    return (
        _weighted_mean(squared_rms_error(prediction_z, batch["target_z"]), sample_weights),
        _weighted_mean(squared_rms_error(prediction_y, batch["target_y"]), sample_weights),
    )


def _response_active_loss(model, batch, weights, sample_weights):
    batch = batch_for_module(model, batch)
    state = model.state(batch["visible"], batch["maps"], batch["radio"], batch["masks"])
    identity_z, identity_y = model.predict_identity(state, batch["query"])
    prediction_z, prediction_y = model.predict(state, batch["action"], batch["query"])
    return _weighted_mean(
        squared_rms_error(prediction_z - identity_z, batch["target_z"] - batch["source_z"])
        + float(weights.response_delta_physical)
        * squared_rms_error(prediction_y - identity_y, batch["target_y"] - batch["source_y"]),
        sample_weights,
    )


def _response_null_loss(model, batch, weights, sample_weights):
    batch = batch_for_module(model, batch)
    state = model.state(batch["visible"], batch["maps"], batch["radio"], batch["masks"])
    identity_z, identity_y = model.predict_identity(state, batch["query"])
    prediction_z, prediction_y = model.predict(state, batch["action"], batch["query"])
    latent_norm = torch.sqrt(torch.mean((prediction_z - identity_z) ** 2, dim=1) + 1e-12)
    physical_norm = torch.sqrt(torch.mean((prediction_y - identity_y) ** 2, dim=1) + 1e-12)
    return _weighted_mean(
        torch.relu(latent_norm - float(weights.response_latent_null_tolerance)) ** 2
        + float(weights.response_null_physical)
        * torch.relu(physical_norm - float(weights.response_physical_null_tolerance)) ** 2,
        sample_weights,
    )


def _centered_branch_mse(prediction, target, bundle_sizes):
    """Average within-source branch error without inventing a separation margin."""
    sizes = tuple(int(value) for value in bundle_sizes)
    if not sizes or any(value < 2 for value in sizes) or sum(sizes) != prediction.shape[0]:
        raise ValueError("response branch bundles must partition rows into sibling sets")
    if prediction.shape != target.shape:
        raise ValueError("response branch prediction and target shapes must match")
    losses = []
    offset = 0
    for size in sizes:
        predicted = prediction[offset : offset + size]
        expected = target[offset : offset + size]
        predicted = predicted - predicted.mean(dim=0, keepdim=True)
        expected = expected - expected.mean(dim=0, keepdim=True)
        losses.append(torch.mean((predicted - expected) ** 2))
        offset += size
    return torch.mean(torch.stack(losses))


def _response_branch_contrast_loss(model, corpus, units, entries, physical_weight):
    """Fit real sibling-action differences for each shared source/query bundle."""
    if len(units) != len(entries) or not units:
        raise ValueError("response branch contrast requires aligned non-empty units and masks")
    bundle_sizes = []
    offset = 0
    while offset < len(units):
        scene, source, _target, position, query = units[offset]
        key = int(scene), int(source), int(position), int(query)
        targets = corpus.response_targets_by_source_query.get(key)
        if targets is None or len(targets) < 2:
            raise RuntimeError("response branch contrast lacks a registered sibling bundle")
        size = len(targets)
        chunk = units[offset : offset + size]
        if (
            len(chunk) != size
            or tuple(int(unit[2]) for unit in chunk) != tuple(int(value) for value in targets)
            or any(
                (int(unit[0]), int(unit[1]), int(unit[3]), int(unit[4])) != key
                for unit in chunk
            )
        ):
            raise RuntimeError("response branch contrast units do not preserve complete bundles")
        bundle_sizes.append(size)
        offset += size

    batch = batch_for_module(model, _response_batch(corpus, units, entries))
    state = model.state(
        batch["visible"], batch["maps"], batch["radio"], batch["masks"]
    )
    prediction_z, prediction_y = model.predict_action_residual(
        state, model.encode_action(batch["action"]), batch["query"]
    )
    target_z = batch["target_z"] - batch["source_z"]
    target_y = batch["target_y"] - batch["source_y"]
    return _centered_branch_mse(
        prediction_z, target_z, bundle_sizes
    ) + float(physical_weight) * _centered_branch_mse(
        prediction_y, target_y, bundle_sizes
    )


def _weighted_mean(values, sample_weights):
    raw_weights = np.asarray(sample_weights, dtype=np.float64)
    if raw_weights.ndim != 1 or raw_weights.shape[0] != values.shape[0]:
        raise ValueError("response sample weights must match the batch")
    if not np.all(np.isfinite(raw_weights)) or not np.all(raw_weights > 0):
        raise ValueError("response sample weights must be finite and positive")
    weights = torch.as_tensor(
        raw_weights, dtype=values.dtype, device=values.device
    )
    return torch.sum(values * weights) / torch.sum(weights)


def _identity_batch(
    corpus, units, entries, supplied_maps=None, *, supplied_map_units=None
):
    dataset = corpus.dataset
    norm = corpus.normalization
    if not units or len(units) != len(entries):
        raise ValueError("identity batch requires aligned non-empty units and masks")
    if supplied_maps is not None and supplied_map_units is not None:
        raise ValueError("identity batch accepts only one supplied-map representation")
    if supplied_maps is not None and len(supplied_maps) != len(units):
        raise ValueError("supplied maps must match the identity batch")
    if supplied_map_units is not None and len(supplied_map_units) != len(units):
        raise ValueError("supplied map units must match the identity batch")
    visible = []
    maps = []
    radio = []
    masks = []
    query = []
    target_z = []
    target_y = []
    for index, ((scene, world, position), entry) in enumerate(zip(units, entries)):
        patches = _normalized_patches(corpus, scene, world, position)
        visible_patches = patches.copy()
        visible_patches[entry.mask] = 0.0
        visible.append(visible_patches)
        if supplied_map_units is not None:
            map_key = tuple(map(int, supplied_map_units[index]))
            if len(map_key) != 2 or map_key not in corpus.normalized_maps:
                raise RuntimeError("supplied map unit is outside the corpus map cache")
            maps.append(corpus.normalized_maps[map_key])
        elif supplied_maps is not None:
            maps.append(
                _cached_model_spatial_input(_normalized_map(norm, supplied_maps[index]))
            )
        else:
            maps.append(corpus.normalized_maps[(int(scene), int(world))])
        radio.append(_normalized_radio(norm, dataset.radio_config[scene], dataset.bs_pose[scene]))
        masks.append(entry.mask)
        query.append(entry.query)
        target_z.append(_normalized_latent(corpus, scene, world, position, entry.query))
        target_y.append(patches[entry.query])
    action_shape = next(iter(corpus.normalized_actions.values())).shape
    zero = np.zeros((len(units), *action_shape), dtype=np.float32)
    return {
        "visible": torch.as_tensor(np.asarray(visible), dtype=torch.float32),
        "maps": torch.as_tensor(np.asarray(maps), dtype=torch.float32),
        "radio": torch.as_tensor(np.asarray(radio), dtype=torch.float32),
        "masks": torch.as_tensor(np.asarray(masks), dtype=torch.bool),
        "query": torch.as_tensor(np.asarray(query), dtype=torch.long),
        "target_z": torch.as_tensor(np.asarray(target_z), dtype=torch.float32),
        "target_y": torch.as_tensor(np.asarray(target_y), dtype=torch.float32),
        "zero_action": torch.as_tensor(zero, dtype=torch.float32),
    }


def _response_batch(corpus, units, entries, *, target_units=None):
    base_units = [(scene, source, position) for scene, source, target, position, query in units]
    identity = _identity_batch(corpus, base_units, entries)
    actions = []
    source_z = []
    source_y = []
    target_z = []
    target_y = []
    supervision_units = units if target_units is None else target_units
    if len(supervision_units) != len(units):
        raise ValueError("response pairing override must match the sampled units")
    for unit, supervision in zip(units, supervision_units):
        scene, source, target, position, query = unit
        target_scene, _target_source, target_world, target_position, target_query = supervision
        if int(target_query) != int(query):
            raise ValueError("response pairing override must preserve the query patch")
        action_key = (int(scene), int(source), int(target))
        if action_key not in corpus.normalized_actions:
            raise RuntimeError("response unit is outside the corpus action cache")
        actions.append(corpus.normalized_actions[action_key])
        source_z.append(_normalized_latent(corpus, scene, source, position, query))
        target_z.append(
            _normalized_latent(
                corpus, target_scene, target_world, target_position, target_query
            )
        )
        source_y.append(_normalized_patches(corpus, scene, source, position)[query])
        target_y.append(
            _normalized_patches(
                corpus, target_scene, target_world, target_position
            )[target_query]
        )
    identity.update(
        {
            "action": torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            "source_z": torch.as_tensor(np.asarray(source_z), dtype=torch.float32),
            "source_y": torch.as_tensor(np.asarray(source_y), dtype=torch.float32),
            "target_z": torch.as_tensor(np.asarray(target_z), dtype=torch.float32),
            "target_y": torch.as_tensor(np.asarray(target_y), dtype=torch.float32),
        }
    )
    return identity


def _pairing_overrides(units, pairing_break, branch):
    if pairing_break is None:
        return None
    mapping = pairing_break.get(branch)
    if not isinstance(mapping, dict):
        raise ValueError(f"pairing break lacks the {branch} permutation")
    output = []
    for unit in units:
        key = tuple(map(int, unit))
        if key not in mapping:
            raise RuntimeError(f"pairing break omits sampled {branch} unit")
        output.append(tuple(map(int, mapping[key])))
    return tuple(output)


def _normalized_patches(corpus, scene, world, position):
    raw = corpus.routed.physical_patches[scene][world, position]
    return (raw - corpus.normalization.patch_mean) / corpus.normalization.patch_scale


def _normalized_latent(corpus, scene, world, position, query):
    raw = corpus.routed.teacher_latent[scene][world, position, query]
    return (raw - corpus.normalization.latent_mean) / corpus.normalization.latent_scale


def _normalized_map(norm, maps):
    array = np.asarray(maps, dtype=np.float64)
    return array / norm.map_scale.reshape(
        *((1,) * (array.ndim - 3)), -1, 1, 1
    )


def _normalized_radio(norm, radio, bs_pose=None):
    context = np.asarray(radio, dtype=np.float64)
    if bs_pose is not None:
        context = np.concatenate((context, np.asarray(bs_pose, dtype=np.float64)), axis=-1)
    if context.shape[-1] != norm.radio_mean.shape[-1]:
        raise ValueError("complete c must include radio_config and bs_pose")
    return (context - norm.radio_mean) / norm.radio_scale


def _normalized_action(norm, action):
    return np.asarray(action, dtype=np.float64) / norm.action_scale.reshape(
        *((1,) * (np.asarray(action).ndim - 3)), -1, 1, 1
    )


def _deterministic_adaptive_avg_pool2d(values, output_size=(16, 16)):
    """NumPy equivalent of the model's deterministic adaptive average pool."""
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("spatial values must have row and column dimensions")
    output_rows, output_columns = (int(value) for value in output_size)
    input_rows, input_columns = array.shape[-2:]
    if output_rows < 1 or output_columns < 1:
        raise ValueError("spatial pool output dimensions must be positive")
    if input_rows < output_rows or input_columns < output_columns:
        raise ValueError("spatial inputs must be at least as large as the pooled grid")
    if input_rows % output_rows == 0 and input_columns % output_columns == 0:
        row_block = input_rows // output_rows
        column_block = input_columns // output_columns
        reshaped = array.reshape(
            *array.shape[:-2],
            output_rows,
            row_block,
            output_columns,
            column_block,
        )
        return reshaped.mean(axis=(-3, -1))
    output = np.empty(
        (*array.shape[:-2], output_rows, output_columns), dtype=array.dtype
    )
    for row in range(output_rows):
        row_start = (row * input_rows) // output_rows
        row_end = ((row + 1) * input_rows + output_rows - 1) // output_rows
        for column in range(output_columns):
            column_start = (column * input_columns) // output_columns
            column_end = (
                ((column + 1) * input_columns + output_columns - 1)
                // output_columns
            )
            output[..., row, column] = array[
                ..., row_start:row_end, column_start:column_end
            ].mean(axis=(-2, -1))
    return output


def _cached_model_spatial_input(values):
    array = np.asarray(values)
    if array.shape[-2] >= 16 and array.shape[-1] >= 16:
        array = _deterministic_adaptive_avg_pool2d(array, (16, 16))
    output = np.ascontiguousarray(array, dtype=np.float32)
    output.setflags(write=False)
    return output


def _retained_parameters(model):
    prefixes = ("csi_", "map_encoder", "map_projection", "radio_encoder", "fusion", "state_norm")
    return [parameter for name, parameter in model.named_parameters() if name.startswith(prefixes)]


def _gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = [torch.sum(gradient.detach() ** 2) for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.sum(torch.stack(squared))).item()) if squared else 0.0


def _measure_execution(model, corpus, plan, weights):
    counts = {"state_calls": 0, "predict_calls": 0}
    state_hook = model.fusion.register_forward_hook(
        lambda _module, _inputs, _output: counts.__setitem__(
            "state_calls", counts["state_calls"] + 1
        )
    )
    predict_hooks = [
        module.register_forward_hook(
            lambda _module, _inputs, _output: counts.__setitem__(
                "predict_calls", counts["predict_calls"] + 1
            )
        )
        for module in (model.predictor, model.response_predictor)
    ]
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter:
            with torch.no_grad():
                _loss_components(model, corpus, plan, weights)
        value = int(counter.get_total_flops())
        counts["flops"] = value if value > 0 else None
        counts["flop_measurement_status"] = (
            "TORCH_DISPATCH_COUNTER_PER_ARM" if value > 0 else "NOT_ASSESSED"
        )
    except Exception:
        counts["state_calls"] = 0
        counts["predict_calls"] = 0
        with torch.no_grad():
            _loss_components(model, corpus, plan, weights)
        counts["flops"] = None
        counts["flop_measurement_status"] = "NOT_ASSESSED"
    finally:
        state_hook.remove()
        for hook in predict_hooks:
            hook.remove()
    return counts


INFORMATION_BUDGET_CHECKLIST_SCHEMA = "csi-pairs-v6-information-budget-checklist-v1"


def _information_budget_checklist(config) -> list[dict]:
    rows = []
    for budget in config["localization"]["label_budgets"]:
        label_budget = int(budget)
        labeled = label_budget > 0
        rows.append(
            {
                "label_budget": label_budget,
                "uses_map": True,
                "uses_bs_pose": True,
                "uses_unlabeled_target_csi": True,
                "uses_reference_library": False,
                "uses_position_labels": labeled,
                "uses_target_norm_stats": False,
                "allows_encoder_param_updates": False,
                "allows_position_head_param_updates": labeled,
                "schema_version": INFORMATION_BUDGET_CHECKLIST_SCHEMA,
            }
        )
    return rows


def _run_localization(config, dataset, models, normalization, patch_spec, execution_devices):
    bank_rows = []
    sample_rows = []
    source_scenes = [int(value) for value in dataset.indices_for_role("source_encoder_train")]
    target_scenes = [int(value) for value in dataset.indices_for_role("target")]
    target_cities = sorted(set(dataset.city_ids[target_scenes].tolist()))
    dataset_base_digests = getattr(dataset, "canonical_base_map_digests", None)
    canonical_base_digests = (
        {scene: str(dataset_base_digests[scene]) for scene in target_scenes}
        if dataset_base_digests is not None
        else {scene: _canonical_base_map_digest(dataset, scene) for scene in target_scenes}
    )
    canonical_bank_digests = {
        scene: _canonical_bank_digest(dataset, scene) for scene in target_scenes
    }
    localization_job_index = 0
    for seed in config["seeds"]:
        city_orders = {}
        for city in target_cities:
            candidates = city_support_candidates(dataset, city)
            for draw in range(int(config["localization"]["label_draws"])):
                rng = np.random.default_rng(int(seed) * 100003 + _stable_city_seed(city) + draw)
                city_orders[(city, draw)] = [candidates[index] for index in rng.permutation(len(candidates))]
        for arm in ARMS:
            device = execution_devices[localization_job_index % len(execution_devices)]
            localization_job_index += 1
            model = models[(int(seed), arm)].to(device)
            scene_representations = _natural_representations(
                model, dataset, source_scenes + target_scenes, normalization, patch_spec
            )
            source_x = np.vstack([scene_representations[scene] for scene in source_scenes])
            source_y = np.vstack([dataset.positions[scene] for scene in source_scenes])
            source_head = fit_source_position_head(
                source_x, source_y, config, seed=int(seed) + 17003
            )
            for city in target_cities:
                city_scenes = [scene for scene in target_scenes if str(dataset.city_ids[scene]) == city]
                for draw in range(int(config["localization"]["label_draws"])):
                    order = city_orders[(city, draw)]
                    for budget in config["localization"]["label_budgets"]:
                        if int(budget) == 0 and draw > 0:
                            continue
                        if int(budget) > len(order):
                            raise ValueError(
                                f"target city {city} has {len(order)} support positions, fewer than k={budget}"
                            )
                        selected = order[: int(budget)]
                        support_ids = [str(dataset.position_ids[scene, position]) for scene, position in selected]
                        if len(support_ids) != len(set(support_ids)):
                            raise RuntimeError("city-level support draw contains duplicate receiver positions")
                        support_x = (
                            np.vstack([scene_representations[scene][position] for scene, position in selected])
                            if selected
                            else np.empty((0, source_x.shape[1]))
                        )
                        support_y = (
                            np.vstack([dataset.positions[scene, position] for scene, position in selected])
                            if selected
                            else np.empty((0, 2))
                        )
                        head = adapt_position_head(source_head, support_x, support_y, config)
                        for scene in city_scenes:
                            query_indices = eligible_query_indices(dataset, scene, set(support_ids))
                            prediction, variance, uncertainty = predict_position_distribution(
                                head,
                                scene_representations[scene][query_indices],
                                sigma_min=float(config["localization"]["sigma_min"]),
                            )
                            errors = np.sqrt(
                                np.sum((prediction - dataset.positions[scene, query_indices]) ** 2, axis=1)
                            )
                            for local_index, position in enumerate(query_indices):
                                sample_rows.append(
                                    {
                                        "seed": int(seed),
                                        "arm": arm,
                                        "city_id": city,
                                        "bank_id": str(dataset.bank_ids[scene]),
                                        "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                                        "canonical_base_map_digest": canonical_base_digests[scene],
                                        "canonical_bank_digest": canonical_bank_digests[scene],
                                        "budget": int(budget),
                                        "draw": 0 if int(budget) == 0 else draw,
                                        "position_id": str(dataset.position_ids[scene, position]),
                                        "prediction_x": float(prediction[local_index, 0]),
                                        "prediction_y": float(prediction[local_index, 1]),
                                        "variance_x": float(variance[local_index, 0]),
                                        "variance_y": float(variance[local_index, 1]),
                                        "u_g": float(uncertainty[local_index]),
                                        "true_x": float(dataset.positions[scene, position, 0]),
                                        "true_y": float(dataset.positions[scene, position, 1]),
                                        "error_m": float(errors[local_index]),
                                        "failure": bool(
                                            errors[local_index]
                                            > float(config["localization"]["failure_threshold_m"])
                                        ),
                                    }
                                )
                            median = float(np.median(errors))
                            bank_rows.append(
                                {
                                    "seed": int(seed),
                                    "arm": arm,
                                    "city_id": city,
                                    "bank_id": str(dataset.bank_ids[scene]),
                                    "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                                    "canonical_base_map_digest": canonical_base_digests[scene],
                                    "canonical_bank_digest": canonical_bank_digests[scene],
                                    "budget": int(budget),
                                    "draw": 0 if int(budget) == 0 else draw,
                                    "city_support_unique_positions": len(selected),
                                    "support_position_ids": ";".join(support_ids),
                                    "query_unique_positions": int(query_indices.size),
                                    "median_error_m": median,
                                    "p90_error_m": float(np.percentile(errors, 90)),
                                    "utility_neg_log_median": float(-math.log(max(median, 1e-12))),
                                }
                            )
            model.to("cpu")
    return bank_rows, sample_rows


def _canonical_base_map_digest(dataset, scene):
    digests = getattr(dataset, "canonical_base_map_digests", None)
    if digests is not None:
        return str(digests[int(scene)])
    method = getattr(dataset, "canonical_base_map_digest", None)
    if callable(method):
        return str(method(int(scene)))
    # Backward compatibility for pre-digest fixture objects.  Formal datasets
    # expose the canonical digest; this fallback must not be used as provenance.
    world = int(dataset.natural_world_index[int(scene)])
    payload = np.ascontiguousarray(dataset.maps[int(scene), world]).tobytes()
    return "fallback-natural-map:" + hashlib.sha256(payload).hexdigest()


def _canonical_bank_digest(dataset, scene):
    """Content identity for de-duplicating copied evaluation banks."""
    scene = int(scene)
    digest = hashlib.sha256()
    digest.update(b"csi-pairs-v6-canonical-bank-v1\0")
    for value in (
        dataset.maps[scene],
        dataset.csi[scene],
        dataset.positions[scene],
        dataset.radio_config[scene],
        dataset.bs_pose[scene],
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update("\0".join(str(value) for value in dataset.map_channel_names).encode("utf-8"))
    return digest.hexdigest()


def _natural_representations(model, dataset, scenes, normalization, patch_spec):
    model.eval()
    outputs = {}
    with torch.no_grad():
        for scene in scenes:
            world = int(dataset.natural_world_index[scene])
            raw = patchify_csi(dataset.csi[scene, world], patch_spec)
            patches = (raw - normalization.patch_mean) / normalization.patch_scale
            maps = np.repeat(dataset.maps[scene, world][None, :, :, :], dataset.position_count, axis=0)
            maps = maps / normalization.map_scale[None, :, None, None]
            context = np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene]))
            radio = np.repeat(context[None, :], dataset.position_count, axis=0)
            radio = (radio - normalization.radio_mean) / normalization.radio_scale
            representation = model.retained_representation(
                tensor_for_module(model, patches, dtype=torch.float32),
                tensor_for_module(model, maps, dtype=torch.float32),
                tensor_for_module(model, radio, dtype=torch.float32),
            )
            outputs[scene] = representation.cpu().numpy().astype(np.float64)
    return outputs


def _localization_summary(rows):
    summary = []
    keys = sorted({(row["arm"], row["city_id"], row["budget"]) for row in rows})
    for arm, city, budget in keys:
        selected = [
            row for row in rows
            if (row["arm"], row["city_id"], row["budget"]) == (arm, city, budget)
        ]
        summary.append(
            {
                "arm": arm,
                "city_id": city,
                "budget": budget,
                "independent_base_map_clusters": len({row["base_map_cluster_id"] for row in selected}),
                "training_seeds": len({row["seed"] for row in selected}),
                "label_draws": len({row["draw"] for row in selected}),
                "mean_bank_median_error_m": float(np.mean([row["median_error_m"] for row in selected])),
                "mean_utility_neg_log_median": float(
                    np.mean([row["utility_neg_log_median"] for row in selected])
                ),
                "mean_bank_p90_error_m": float(np.mean([row["p90_error_m"] for row in selected])),
            }
        )
    return summary


def _factorial_statistics(config, rows):
    primary = list(config["localization"]["primary_budgets"])
    exact = exact_factorial_utilities(rows, primary)
    hierarchical = hierarchical_factorial_interval(
        rows,
        primary,
        int(config["factorial"]["bootstrap_resamples"]),
        int(config["seeds"][0]) + 700001,
    )
    bank_only = bank_only_factorial_interval(
        rows,
        primary,
        int(config["factorial"]["bootstrap_resamples"]),
        int(config["seeds"][0]) + 700002,
    )
    sensitivity = leave_one_factorial_sensitivity(rows, primary)
    return {
        "schema_version": "csi-pairs-v6-factorial-statistics-v1",
        "estimand": "equal city x k x base-map-cluster x seed x positive-budget-draw macro utility",
        "exact": exact,
        "hierarchical_bootstrap": hierarchical,
        "bank_only_bootstrap": bank_only,
        "sensitivity": sensitivity,
    }


def _target_cluster_coverage(config, rows):
    cities = sorted({str(row["city_id"]) for row in rows})
    required_city_count = int(config["data"]["minimum_target_cities"])
    required_clusters = int(
        config["data"][
            "minimum_independent_base_map_clusters_per_target_city"
        ]
    )
    observed = {
        city: len(
            {
                str(
                    row.get("canonical_base_map_digest")
                    or row["base_map_cluster_id"]
                )
                for row in rows
                if str(row["city_id"]) == city
            }
        )
        for city in cities
    }
    return {
        "passed": bool(
            len(cities) >= required_city_count
            and all(count >= required_clusters for count in observed.values())
        ),
        "required_target_city_count": required_city_count,
        "observed_target_city_count": len(cities),
        "required_independent_clusters_per_city": required_clusters,
        "observed_independent_clusters_by_city": observed,
        "identity": "canonical_base_map_digest",
    }


def _preliminary_factorial_gate(
    config,
    dataset,
    rows,
    statistics,
    evidence,
    *,
    qualification_gate_sha256,
):
    hierarchical = statistics["hierarchical_bootstrap"]
    minimum_interaction = float(config["factorial"]["minimum_interaction_effect"])
    primary = set(int(value) for value in config["localization"]["primary_budgets"])
    city_checks = []
    maximum_regression = 0.0
    target_cities_for_g4 = sorted({row["city_id"] for row in rows})
    city_family_size = max(1, len(target_cities_for_g4) * len(primary) * 2)
    familywise_alpha = float(config["evaluation"]["familywise_alpha"])
    for city in sorted({row["city_id"] for row in rows}):
        for budget in sorted(primary):
            for baseline in ("alignment", "response"):
                interval = _city_budget_arm_interval(
                    rows,
                    city,
                    budget,
                    baseline,
                    int(config["evaluation"]["bootstrap_resamples"]),
                    730000 + len(city_checks),
                    familywise_alpha=familywise_alpha,
                    family_size=city_family_size,
                )
                difference = float(interval["paired_mean_difference"])
                maximum_regression = max(maximum_regression, max(0.0, -difference))
                city_checks.append(
                    {
                        "city_id": city,
                        "budget": budget,
                        "baseline": baseline,
                        "difference": difference,
                        **interval,
                    }
                )
    subgates = {
        "1_full_beats_both_single_branches": "PASS"
        if hierarchical["full_vs_alignment"]["familywise_ci95_low"] > 0
        and hierarchical["full_vs_response"]["familywise_ci95_low"] > 0
        else "FAIL",
        "2_cgs_noninferior_to_alignment": "NOT_ASSESSED",
        "3_native_response_noninferior_to_response": "NOT_ASSESSED",
        "4_hierarchical_interaction_ci_exceeds_minimum": "PASS"
        if hierarchical["interaction"]["familywise_ci95_low"] > minimum_interaction
        else "FAIL",
        "5_no_city_k_reverse_regression": "PASS"
        if all(
            row["familywise_ci_low"]
            >= -float(config["localization"]["maximum_city_regression"])
            for row in city_checks
        )
        else "FAIL",
        "6_equal_flop_single_branch_superiority": "NOT_ASSESSED",
        "7_parameter_and_flop_matched_concat_superiority": "NOT_ASSESSED",
    }
    localization_checks = []
    for city in sorted({row["city_id"] for row in rows}):
        for budget in sorted(primary):
            interval = _city_budget_arm_interval(
                rows,
                city,
                budget,
                "endpoint",
                int(config["evaluation"]["bootstrap_resamples"]),
                735000 + len(localization_checks),
                null_threshold=float(
                    config["localization"]["minimum_city_improvement"]
                ),
            )
            localization_checks.append(
                {
                    "city_id": city,
                    "budget": budget,
                    "baseline": "endpoint",
                    "difference": float(interval["paired_mean_difference"]),
                    **interval,
                }
            )
    p_values = [row["p_value_two_sided"] for row in localization_checks]
    adjusted = holm_adjust(p_values)
    for row, value in zip(localization_checks, adjusted):
        row["holm_adjusted_p"] = value
    alpha = float(config["evaluation"]["familywise_alpha"])
    for row in localization_checks:
        row["passed"] = bool(
            interval_decision(
                row,
                threshold=float(config["localization"]["minimum_city_improvement"]),
                relation="superiority",
            )
            and float(row["holm_adjusted_p"]) < alpha
        )
    primary_complete = set(primary) == {0, 8}
    cluster_coverage = _target_cluster_coverage(config, rows)
    g5_subgates = {
        "1_two_target_cities_and_independent_clusters": "PASS"
        if cluster_coverage["passed"]
        else "FAIL",
        "2_strict_primary_k0_k8_complete": "PASS" if primary_complete else "FAIL",
        "3_every_city_k0_full_vs_endpoint_ci_holm": "PASS"
        if all(row["passed"] for row in localization_checks if row["budget"] == 0)
        else "FAIL",
        "4_every_city_k8_full_vs_endpoint_ci_holm": "PASS"
        if primary_complete
        and all(row["passed"] for row in localization_checks if row["budget"] == 8)
        else "FAIL",
    }
    g5_passed = all(value == "PASS" for value in g5_subgates.values())
    software_only = bool(dataset.is_fixture)
    gate_vector = complete_gate_vector(
        {
            "G1": "FAIL" if software_only else "PASS",
            "G2": "FAIL" if software_only else "PASS",
            "G4": "NOT_ASSESSED",
            "G5": "PASS" if g5_passed else "FAIL",
        }
    )
    return {
        "schema_version": FACTORIAL_SCHEMA,
        "status": "FAIL" if software_only else ("PASS" if g5_passed else "FAIL"),
        "passed": bool(g5_passed and not software_only),
        **evidence,
        "qualification_gate_sha256": qualification_gate_sha256,
        "gate_vector": gate_vector,
        "g4_subgates": subgates,
        "g5_subgates": g5_subgates,
        "city_budget_checks": city_checks,
        "g4_familywise_control": {
            "primary_hypotheses": hierarchical["familywise_method"],
            "city_budget_hypotheses": "bonferroni_bootstrap_intervals",
            "city_budget_family_size": city_family_size,
            "familywise_alpha": familywise_alpha,
        },
        "localization_city_budget_checks": localization_checks,
        "target_cluster_coverage": cluster_coverage,
        "claim_boundary": "G4 cannot PASS until all seven subgates are PASS; NOT_ASSESSED is never PASS.",
        "software_only_qualification_bypass": software_only,
        "config": public_formal_config(config),
    }


def _city_budget_arm_interval(
    rows,
    city,
    budget,
    baseline,
    resamples,
    seed,
    *,
    familywise_alpha=None,
    family_size=1,
    null_threshold=0.0,
):
    grouped = {}
    for row in rows:
        if str(row["city_id"]) != str(city) or int(row["budget"]) != int(budget):
            continue
        if row["arm"] not in {"full", baseline}:
            continue
        key = (
            str(row.get("canonical_base_map_digest") or row["base_map_cluster_id"]),
            int(row["seed"]),
            int(row["draw"]),
        )
        grouped.setdefault((key, str(row["arm"])), {}).setdefault(
            str(row.get("canonical_bank_digest") or row["bank_id"]), []
        ).append(float(row["utility_neg_log_median"]))
    cells = sorted(
        key
        for key in {item[0] for item in grouped}
        if (key, "full") in grouped and (key, baseline) in grouped
    )
    clusters = np.asarray([key[0] for key in cells])
    full = np.asarray(
        [
            np.mean([np.mean(values) for values in grouped[(key, "full")].values()])
            for key in cells
        ]
    )
    other = np.asarray(
        [
            np.mean([np.mean(values) for values in grouped[(key, baseline)].values()])
            for key in cells
        ]
    )
    interval = paired_cluster_interval(clusters, full, other, resamples, seed)
    if familywise_alpha is not None:
        adjusted_interval = paired_cluster_interval(
            clusters,
            full,
            other,
            resamples,
            seed,
            alpha=float(familywise_alpha) / max(1, int(family_size)),
        )
        interval.update(
            {
                "familywise_ci_low": adjusted_interval["ci95_low"],
                "familywise_ci_high": adjusted_interval["ci95_high"],
                "familywise_confidence_level": adjusted_interval["confidence_level"],
            }
        )
    test = paired_sign_flip_test(
        clusters,
        full,
        other,
        seed + 1,
        null_difference=float(null_threshold),
    )
    return {
        **interval,
        "null_difference": float(null_threshold),
        "p_value_two_sided": test["p_value_two_sided"],
    }


def _stable_city_seed(city):
    import hashlib

    return int.from_bytes(hashlib.sha256(str(city).encode("utf-8")).digest()[:4], "little")


def city_support_candidates(dataset: FormalDataset, city: str) -> list[tuple[int, int]]:
    candidates = dataset.unique_target_support_positions(city)
    return sorted(
        candidates,
        key=lambda row: str(dataset.position_ids[row[0], row[1]]),
    )


def eligible_query_indices(dataset: FormalDataset, scene: int, support_ids: set[str]) -> np.ndarray:
    candidates = np.flatnonzero(dataset.position_roles[scene] == "query")
    city = str(dataset.city_ids[scene])
    support_coordinates = [
        dataset.positions[target_scene, position]
        for target_scene_value in dataset.indices_for_role("target")
        for target_scene in (int(target_scene_value),)
        if str(dataset.city_ids[target_scene]) == city
        for position in range(dataset.position_count)
        if str(dataset.position_ids[target_scene, position]) in support_ids
    ]
    selected = np.asarray(
        [
            position
            for position in candidates
            if str(dataset.position_ids[scene, position]) not in support_ids
            and not any(
                same_physical_position(dataset.positions[scene, position], coordinate)
                for coordinate in support_coordinates
            )
        ],
        dtype=np.int64,
    )
    if selected.size == 0:
        raise RuntimeError("support sibling exclusion leaves no target query positions")
    return selected
