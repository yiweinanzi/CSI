"""Restrict research computation to explicitly declared source scene rows."""

import numpy as np


SOURCE_RESEARCH_ROLES = ("source_encoder_train", "source_method_selection")
SCENE_ARRAYS = frozenset({
    "csi", "csi_clean", "csi_repeat", "maps", "positions", "position_ids",
    "free_space", "radio_config", "bs_pose", "repeat_seeds",
    "phase_reference_ids", "phase_reference_values", "phase_reference_source_sha256",
    "base_map_cluster_ids", "canonical_map_sha256", "noop_maps", "noop_map_sha256",
    "path_ids", "path_power", "path_surface_ids", "noop_path_ids", "noop_path_power",
    "noop_path_surface_ids", "primitive_surface_ids", "primitive_ids", "anchor_bits",
    "natural_world_index", "scene_ids", "city_ids", "bank_ids", "scene_roles",
    "position_roles",
})
GLOBAL_ARRAYS = frozenset({"world_bits", "map_channel_names"})


class SourceSceneArray:
    def __init__(self, values, allowed, observed):
        self._values = values
        self._allowed = allowed
        self._observed = observed
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("Source-only research forbids converting an unsliced scene array")

    def __getitem__(self, key):
        first = key[0] if isinstance(key, tuple) else key
        if first is None or first is Ellipsis or isinstance(first, (bool, np.bool_)):
            raise RuntimeError("Source-only research requires an explicit scene selector")
        scenes = np.atleast_1d(np.arange(self.shape[0])[first])
        selected = {int(value) for value in scenes.reshape(-1)}
        if not selected.issubset(self._allowed):
            raise RuntimeError("Source-only research attempted to access a forbidden scene")
        self._observed.update(selected)
        result = self._values[key]
        if isinstance(result, np.ndarray):
            result = result.view()
            result.setflags(write=False)
        return result


class SourceOnlyDataset:
    def __init__(self, dataset):
        self._dataset = dataset
        self._roles = {
            role: np.asarray(dataset.indices_for_role(role), dtype=np.int64)
            for role in SOURCE_RESEARCH_ROLES
        }
        if any(len(rows) == 0 for rows in self._roles.values()):
            raise RuntimeError("Source research requires both source partitions")
        if set(self._roles[SOURCE_RESEARCH_ROLES[0]]) & set(self._roles[SOURCE_RESEARCH_ROLES[1]]):
            raise RuntimeError("Source training and selection scenes overlap")
        self.allowed_scenes = frozenset(int(value) for rows in self._roles.values() for value in rows)
        self.observed_scenes = set()

    def indices_for_role(self, role):
        if role not in self._roles:
            raise RuntimeError("Forbidden research role: " + str(role))
        return self._roles[role].copy()

    def _scene(self, scene):
        value = int(scene)
        if value not in self.allowed_scenes:
            raise RuntimeError("Forbidden research scene")
        self.observed_scenes.add(value)
        return value

    def directed_edges(self, scene):
        return self._dataset.directed_edges(self._scene(scene))

    def independent_unit_id(self, scene):
        return str(self.base_map_cluster_ids[self._scene(scene)])

    def __getattr__(self, name):
        if name.startswith("_") or name in {"canonical_base_map_digests", "unique_target_support_positions"}:
            raise AttributeError(name)
        value = getattr(self._dataset, name)
        if callable(value):
            raise RuntimeError("Dataset method is not allowed in source research: " + name)
        if name in SCENE_ARRAYS:
            if not isinstance(value, np.ndarray) or not value.ndim or value.shape[0] != self._dataset.scene_count:
                raise RuntimeError("Source scene array has an invalid layout: " + name)
            return SourceSceneArray(value, self.allowed_scenes, self.observed_scenes)
        if isinstance(value, np.ndarray):
            if name not in GLOBAL_ARRAYS:
                raise RuntimeError("Unclassified research array: " + name)
            value = value.view()
            value.setflags(write=False)
        return value
