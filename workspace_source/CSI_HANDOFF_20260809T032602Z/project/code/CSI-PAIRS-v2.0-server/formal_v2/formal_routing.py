from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formal_dataset import FormalDataset
from .formal_protocol import PatchSpec, delay_angle_power, patchify_csi
from .formal_teacher import TeacherBundle, teacher_targets


ROUTE_NAMES = np.asarray(["null", "gray", "active"])
PRIMARY_ROUTE_CONTRACT = {
    "alignment_primary": "full_channel_physical_distance_only",
    "response_primary": "query_patch_physical_distance_only",
    "teacher_sensitivity": "independent_audit_stratum_and_auxiliary_qualification_only",
}


@dataclass(frozen=True)
class RouteNormalization:
    channel_mean: np.ndarray
    channel_scale: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    delay_angle_mean: np.ndarray
    delay_angle_scale: np.ndarray


@dataclass(frozen=True)
class RoutedEdges:
    teacher_latent: dict[int, np.ndarray]
    physical_patches: dict[int, np.ndarray]
    alignment_route: dict[tuple[int, int, int, int], int]
    response_route: dict[tuple[int, int, int, int, int], int]
    alignment_teacher_stratum: dict[tuple[int, int, int, int], int]
    response_teacher_stratum: dict[tuple[int, int, int, int, int], int]
    alignment_distances: dict[tuple[int, int, int, int], tuple[float, float]]
    response_distances: dict[tuple[int, int, int, int, int], tuple[float, float]]
    normalization: RouteNormalization


def fit_route_normalization(dataset: FormalDataset, bundle: TeacherBundle) -> RouteNormalization:
    scenes = dataset.indices_for_role("source_encoder_train")
    csi = dataset.csi[scenes]
    channel_mean = csi.reshape(-1, dataset.channel_count).mean(axis=0)
    channel_scale = csi.reshape(-1, dataset.channel_count).std(axis=0)
    channel_scale[channel_scale < 1e-9] = 1.0
    latent = teacher_targets(bundle, csi)
    latent_mean = latent.reshape(-1, latent.shape[-1]).mean(axis=0)
    latent_scale = latent.reshape(-1, latent.shape[-1]).std(axis=0)
    latent_scale[latent_scale < 1e-9] = 1.0
    spec = PatchSpec.from_metadata(dataset.metadata)
    delay_angle = delay_angle_power(csi, spec)
    delay_angle_mean = delay_angle.reshape(-1, delay_angle.shape[-1]).mean(axis=0)
    delay_angle_scale = delay_angle.reshape(-1, delay_angle.shape[-1]).std(axis=0)
    delay_angle_scale[delay_angle_scale < 1e-9] = 1.0
    return RouteNormalization(
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        latent_mean=latent_mean,
        latent_scale=latent_scale,
        delay_angle_mean=delay_angle_mean,
        delay_angle_scale=delay_angle_scale,
    )


def route_dataset(
    dataset: FormalDataset,
    bundle: TeacherBundle,
    config: dict,
    scenes: np.ndarray,
    normalization: RouteNormalization | None = None,
) -> RoutedEdges:
    norm = normalization or fit_route_normalization(dataset, bundle)
    spec = PatchSpec.from_metadata(dataset.metadata)
    channel_patch_scale = patchify_csi(norm.channel_scale, spec)
    scene_latent = {
        int(scene): teacher_targets(bundle, dataset.csi[int(scene)]) for scene in scenes
    }
    scene_patches = {
        int(scene): patchify_csi(dataset.csi[int(scene)], spec) for scene in scenes
    }
    alignment_route: dict[tuple[int, int, int, int], int] = {}
    response_route: dict[tuple[int, int, int, int, int], int] = {}
    alignment_teacher_stratum: dict[tuple[int, int, int, int], int] = {}
    response_teacher_stratum: dict[tuple[int, int, int, int, int], int] = {}
    alignment_distances: dict[tuple[int, int, int, int], tuple[float, float]] = {}
    response_distances: dict[tuple[int, int, int, int, int], tuple[float, float]] = {}
    q = config["qualification"]
    for scene_value in scenes:
        scene = int(scene_value)
        for edge in dataset.directed_edges(scene):
            for position in range(dataset.position_count):
                source_csi = dataset.csi[scene, edge.source_world, position]
                target_csi = dataset.csi[scene, edge.target_world, position]
                source_latent = scene_latent[scene][edge.source_world, position]
                target_latent = scene_latent[scene][edge.target_world, position]
                source_delay_angle = delay_angle_power(source_csi, spec)
                target_delay_angle = delay_angle_power(target_csi, spec)
                complex_difference = (target_csi - source_csi) / norm.channel_scale
                delay_angle_difference = (
                    target_delay_angle - source_delay_angle
                ) / norm.delay_angle_scale
                physical_a = float(
                    np.sqrt(
                        np.mean(
                            np.concatenate((complex_difference, delay_angle_difference)) ** 2
                        )
                    )
                )
                latent_a = float(np.sqrt(np.mean(((target_latent - source_latent) / norm.latent_scale) ** 2)))
                a_key = (scene, edge.source_world, edge.target_world, position)
                alignment_route[a_key] = route_code(
                    physical_a,
                    float(q["physical_null_rms_max"]),
                    float(q["physical_active_rms_min"]),
                )
                alignment_teacher_stratum[a_key] = teacher_sensitivity_code(
                    latent_a,
                    float(q["latent_null_rms_max"]),
                    float(q["latent_active_rms_min"]),
                )
                alignment_distances[a_key] = (physical_a, latent_a)
                source_patches = scene_patches[scene][edge.source_world, position]
                target_patches = scene_patches[scene][edge.target_world, position]
                for query in range(spec.patch_count):
                    physical_r = float(
                        np.sqrt(
                            np.mean(
                                ((target_patches[query] - source_patches[query]) / channel_patch_scale[query]) ** 2
                            )
                        )
                    )
                    latent_r = float(
                        np.sqrt(
                            np.mean(
                                ((target_latent[query] - source_latent[query]) / norm.latent_scale) ** 2
                            )
                        )
                    )
                    r_key = (*a_key, query)
                    response_route[r_key] = route_code(
                        physical_r,
                        float(q["response_physical_null_rms_max"]),
                        float(q["response_physical_active_rms_min"]),
                    )
                    response_teacher_stratum[r_key] = teacher_sensitivity_code(
                        latent_r,
                        float(q["response_latent_null_rms_max"]),
                        float(q["response_latent_active_rms_min"]),
                    )
                    response_distances[r_key] = (physical_r, latent_r)
    _assert_direction_invariance(alignment_route, response_route)
    return RoutedEdges(
        teacher_latent=scene_latent,
        physical_patches=scene_patches,
        alignment_route=alignment_route,
        response_route=response_route,
        alignment_teacher_stratum=alignment_teacher_stratum,
        response_teacher_stratum=response_teacher_stratum,
        alignment_distances=alignment_distances,
        response_distances=response_distances,
        normalization=norm,
    )


def route_code(
    physical: float,
    physical_null: float,
    physical_active: float,
) -> int:
    if physical <= physical_null:
        return 0
    if physical >= physical_active:
        return 2
    return 1


def teacher_sensitivity_code(
    latent: float,
    latent_null: float,
    latent_active: float,
) -> int:
    if latent <= latent_null:
        return 0
    if latent >= latent_active:
        return 2
    return 1


def route_coverage(dataset: FormalDataset, routed: RoutedEdges, scenes: np.ndarray) -> list[dict]:
    rows = []
    for scene_value in scenes:
        scene = int(scene_value)
        alignment = [
            value for key, value in routed.alignment_route.items() if key[0] == scene
        ]
        response = [value for key, value in routed.response_route.items() if key[0] == scene]
        row = {
            "scene_id": str(dataset.scene_ids[scene]),
            "bank_id": str(dataset.bank_ids[scene]),
            "role": str(dataset.scene_roles[scene]),
        }
        alignment_teacher = [
            value
            for key, value in routed.alignment_teacher_stratum.items()
            if key[0] == scene
        ]
        response_teacher = [
            value
            for key, value in routed.response_teacher_stratum.items()
            if key[0] == scene
        ]
        for prefix, values in (
            ("alignment", alignment),
            ("response_patch", response),
            ("alignment_teacher_stratum", alignment_teacher),
            ("response_patch_teacher_stratum", response_teacher),
        ):
            for code, name in enumerate(ROUTE_NAMES.tolist()):
                row[f"{prefix}_{name}_units"] = int(np.sum(np.asarray(values) == code))
        rows.append(row)
    return rows


def _assert_direction_invariance(
    alignment: dict[tuple[int, int, int, int], int],
    response: dict[tuple[int, int, int, int, int], int],
) -> None:
    for (scene, source, target, position), value in alignment.items():
        reverse = (scene, target, source, position)
        if alignment.get(reverse) != value:
            raise RuntimeError("alignment route is not direction invariant")
    for (scene, source, target, position, query), value in response.items():
        reverse = (scene, target, source, position, query)
        if response.get(reverse) != value:
            raise RuntimeError("response route is not direction invariant")
