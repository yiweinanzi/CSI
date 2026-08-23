from __future__ import annotations

from collections.abc import Iterator, Mapping
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
    "teacher_sensitivity_representation": (
        "source_train_frozen_readout_decoded_normalized_csi"
    ),
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
class CompactEdgeLayout:
    scenes: tuple[int, ...]
    scene_offsets: dict[int, tuple[int, int]]
    sources: np.ndarray
    targets: np.ndarray
    edge_lookup: dict[tuple[int, int, int], int]
    position_count: int
    query_count: int

    @property
    def edge_count(self) -> int:
        return int(self.sources.size)

    def equivalent(self, other: object) -> bool:
        return bool(
            isinstance(other, CompactEdgeLayout)
            and self.scenes == other.scenes
            and self.scene_offsets == other.scene_offsets
            and self.edge_lookup == other.edge_lookup
            and self.position_count == other.position_count
            and self.query_count == other.query_count
            and np.array_equal(self.sources, other.sources)
            and np.array_equal(self.targets, other.targets)
        )


class CompactEdgeTensorMapping(Mapping):
    """Tuple-key mapping backed by one compact edge/position/query tensor."""

    def __init__(
        self,
        layout: CompactEdgeLayout,
        values: np.ndarray,
        *,
        query_axis: bool,
        vector_value: bool = False,
    ) -> None:
        self.layout = layout
        self.array = np.asarray(values)
        self.query_axis = bool(query_axis)
        self.vector_value = bool(vector_value)
        expected = [layout.edge_count, layout.position_count]
        if self.query_axis:
            expected.append(layout.query_count)
        if self.vector_value:
            expected.append(2)
        if tuple(self.array.shape) != tuple(expected):
            raise ValueError(
                f"compact edge tensor shape mismatch: expected {tuple(expected)}, "
                f"received {self.array.shape}"
            )

    def __getitem__(self, key):
        expected_length = 5 if self.query_axis else 4
        if not isinstance(key, tuple) or len(key) != expected_length:
            raise KeyError(key)
        scene, source, target, position, *query = (int(value) for value in key)
        edge = self.layout.edge_lookup.get((scene, source, target))
        if edge is None or not 0 <= position < self.layout.position_count:
            raise KeyError(key)
        if self.query_axis:
            query_index = int(query[0])
            if not 0 <= query_index < self.layout.query_count:
                raise KeyError(key)
            value = self.array[edge, position, query_index]
        else:
            value = self.array[edge, position]
        if self.vector_value:
            return tuple(float(item) for item in np.asarray(value).tolist())
        return np.asarray(value).item()

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        for scene in self.layout.scenes:
            start, stop = self.layout.scene_offsets[scene]
            for edge in range(start, stop):
                source = int(self.layout.sources[edge])
                target = int(self.layout.targets[edge])
                for position in range(self.layout.position_count):
                    if self.query_axis:
                        for query in range(self.layout.query_count):
                            yield scene, source, target, position, query
                    else:
                        yield scene, source, target, position

    def __len__(self) -> int:
        multiplier = self.layout.query_count if self.query_axis else 1
        return self.layout.edge_count * self.layout.position_count * multiplier

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompactEdgeTensorMapping):
            return NotImplemented
        return bool(
            self.query_axis == other.query_axis
            and self.vector_value == other.vector_value
            and self.layout.equivalent(other.layout)
            and np.array_equal(self.array, other.array)
        )

    def values_for_scene(self, scene: int) -> np.ndarray:
        try:
            start, stop = self.layout.scene_offsets[int(scene)]
        except KeyError as error:
            raise KeyError(scene) from error
        return self.array[start:stop]


@dataclass(frozen=True)
class RoutedEdges:
    teacher_latent: dict[int, np.ndarray]
    physical_patches: dict[int, np.ndarray]
    alignment_route: Mapping[tuple[int, int, int, int], int]
    response_route: Mapping[tuple[int, int, int, int, int], int]
    alignment_teacher_stratum: Mapping[tuple[int, int, int, int], int]
    response_teacher_stratum: Mapping[tuple[int, int, int, int, int], int]
    alignment_distances: Mapping[tuple[int, int, int, int], tuple[float, float]]
    response_distances: Mapping[tuple[int, int, int, int, int], tuple[float, float]]
    normalization: RouteNormalization


def fit_route_normalization(dataset: FormalDataset, bundle: TeacherBundle) -> RouteNormalization:
    scenes = dataset.indices_for_role("source_encoder_train")
    csi = dataset.csi[scenes]
    observed_mean = csi.reshape(-1, dataset.channel_count).mean(axis=0)
    observed_scale = csi.reshape(-1, dataset.channel_count).std(axis=0)
    observed_scale[observed_scale < 1e-9] = 1.0
    if not np.array_equal(observed_mean, bundle.channel_mean) or not np.array_equal(
        observed_scale, bundle.channel_scale
    ):
        raise RuntimeError(
            "teacher checkpoint normalization does not match source_encoder_train CSI"
        )
    channel_mean = bundle.channel_mean.copy()
    channel_scale = bundle.channel_scale.copy()
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
    layout = _compact_edge_layout(dataset, scenes, spec.patch_count)
    alignment_route_values = np.empty(
        (layout.edge_count, dataset.position_count), dtype=np.uint8
    )
    response_route_values = np.empty(
        (layout.edge_count, dataset.position_count, spec.patch_count), dtype=np.uint8
    )
    alignment_teacher_values = np.empty_like(alignment_route_values)
    response_teacher_values = np.empty_like(response_route_values)
    alignment_distance_values = np.empty(
        (layout.edge_count, dataset.position_count, 2), dtype=np.float64
    )
    response_distance_values = np.empty(
        (layout.edge_count, dataset.position_count, spec.patch_count, 2),
        dtype=np.float64,
    )
    q = config["qualification"]
    for (scene, source, target), edge_index in layout.edge_lookup.items():
        source_csi = dataset.csi[scene, source]
        target_csi = dataset.csi[scene, target]
        source_latent = scene_latent[scene][source]
        target_latent = scene_latent[scene][target]
        source_delay_angle = delay_angle_power(source_csi, spec)
        target_delay_angle = delay_angle_power(target_csi, spec)
        complex_difference = (target_csi - source_csi) / norm.channel_scale
        delay_angle_difference = (
            target_delay_angle - source_delay_angle
        ) / norm.delay_angle_scale
        physical_a = np.sqrt(
            np.mean(
                np.concatenate((complex_difference, delay_angle_difference), axis=-1)
                ** 2,
                axis=-1,
            )
        )
        latent_a = np.sqrt(
            np.mean(
                ((target_latent - source_latent) / norm.latent_scale) ** 2,
                axis=(-2, -1),
            )
        )
        source_patches = scene_patches[scene][source]
        target_patches = scene_patches[scene][target]
        physical_r = np.sqrt(
            np.mean(
                ((target_patches - source_patches) / channel_patch_scale) ** 2,
                axis=-1,
            )
        )
        latent_r = np.sqrt(
            np.mean(
                ((target_latent - source_latent) / norm.latent_scale) ** 2,
                axis=-1,
            )
        )
        alignment_route_values[edge_index] = _route_codes(
            physical_a,
            float(q["physical_null_rms_max"]),
            float(q["physical_active_rms_min"]),
        )
        alignment_teacher_values[edge_index] = _route_codes(
            latent_a,
            float(q["latent_null_rms_max"]),
            float(q["latent_active_rms_min"]),
        )
        response_route_values[edge_index] = _route_codes(
            physical_r,
            float(q["response_physical_null_rms_max"]),
            float(q["response_physical_active_rms_min"]),
        )
        response_teacher_values[edge_index] = _route_codes(
            latent_r,
            float(q["response_latent_null_rms_max"]),
            float(q["response_latent_active_rms_min"]),
        )
        alignment_distance_values[edge_index, :, 0] = physical_a
        alignment_distance_values[edge_index, :, 1] = latent_a
        response_distance_values[edge_index, :, :, 0] = physical_r
        response_distance_values[edge_index, :, :, 1] = latent_r
    alignment_route = CompactEdgeTensorMapping(
        layout, alignment_route_values, query_axis=False
    )
    response_route = CompactEdgeTensorMapping(
        layout, response_route_values, query_axis=True
    )
    alignment_teacher_stratum = CompactEdgeTensorMapping(
        layout, alignment_teacher_values, query_axis=False
    )
    response_teacher_stratum = CompactEdgeTensorMapping(
        layout, response_teacher_values, query_axis=True
    )
    alignment_distances = CompactEdgeTensorMapping(
        layout, alignment_distance_values, query_axis=False, vector_value=True
    )
    response_distances = CompactEdgeTensorMapping(
        layout, response_distance_values, query_axis=True, vector_value=True
    )
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


def _route_codes(values: np.ndarray, null_max: float, active_min: float) -> np.ndarray:
    array = np.asarray(values)
    return np.where(array <= null_max, 0, np.where(array >= active_min, 2, 1)).astype(
        np.uint8, copy=False
    )


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
        alignment = _scene_values(routed.alignment_route, scene)
        response = _scene_values(routed.response_route, scene)
        row = {
            "scene_id": str(dataset.scene_ids[scene]),
            "bank_id": str(dataset.bank_ids[scene]),
            "role": str(dataset.scene_roles[scene]),
        }
        alignment_teacher = _scene_values(routed.alignment_teacher_stratum, scene)
        response_teacher = _scene_values(routed.response_teacher_stratum, scene)
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
    alignment: Mapping[tuple[int, int, int, int], int],
    response: Mapping[tuple[int, int, int, int, int], int],
) -> None:
    if isinstance(alignment, CompactEdgeTensorMapping) and isinstance(
        response, CompactEdgeTensorMapping
    ):
        layout = alignment.layout
        for (scene, source, target), edge in layout.edge_lookup.items():
            reverse = layout.edge_lookup.get((scene, target, source))
            if reverse is None or not np.array_equal(
                alignment.array[edge], alignment.array[reverse]
            ):
                raise RuntimeError("alignment route is not direction invariant")
            if not np.array_equal(response.array[edge], response.array[reverse]):
                raise RuntimeError("response route is not direction invariant")
        return
    for (scene, source, target, position), value in alignment.items():
        reverse = (scene, target, source, position)
        if alignment.get(reverse) != value:
            raise RuntimeError("alignment route is not direction invariant")
    for (scene, source, target, position, query), value in response.items():
        reverse = (scene, target, source, position, query)
        if response.get(reverse) != value:
            raise RuntimeError("response route is not direction invariant")


def _compact_edge_layout(
    dataset: FormalDataset, scenes: np.ndarray, query_count: int
) -> CompactEdgeLayout:
    scene_tuple = tuple(int(value) for value in scenes)
    if len(set(scene_tuple)) != len(scene_tuple):
        raise ValueError("route scenes must be unique")
    sources = []
    targets = []
    offsets = {}
    lookup = {}
    for scene in scene_tuple:
        start = len(sources)
        for edge in dataset.directed_edges(scene):
            key = (scene, int(edge.source_world), int(edge.target_world))
            if key in lookup:
                raise RuntimeError("duplicate directed edge in compact route layout")
            lookup[key] = len(sources)
            sources.append(key[1])
            targets.append(key[2])
        offsets[scene] = (start, len(sources))
    return CompactEdgeLayout(
        scenes=scene_tuple,
        scene_offsets=offsets,
        sources=np.asarray(sources, dtype=np.int16),
        targets=np.asarray(targets, dtype=np.int16),
        edge_lookup=lookup,
        position_count=int(dataset.position_count),
        query_count=int(query_count),
    )


def _scene_values(mapping: Mapping, scene: int) -> np.ndarray:
    if isinstance(mapping, CompactEdgeTensorMapping):
        return mapping.values_for_scene(scene).reshape(-1)
    return np.asarray([value for key, value in mapping.items() if int(key[0]) == int(scene)])
