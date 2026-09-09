from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

import numpy as np

from .formal_io import parse_strict_json, sha256_file


DATASET_SCHEMA_VERSION = "csi-pairs-formal-dataset-v2.1-v6"
SOURCE_ROLES = (
    "source_encoder_train",
    "source_method_selection",
    "source_probe_train",
    "source_probe_selection",
    "source_calibration_fit",
    "source_calibration_selection",
    "source_final_unseen_bank",
)
SCENE_ROLES = SOURCE_ROLES + ("target", "external_validation")
POSITION_ROLES = ("standard", "support_pool", "query")
PHYSICAL_POSITION_ATOL_M = 1e-9
REQUIRED_ARRAYS = {
    "csi_repeat",
    "csi_clean",
    "maps",
    "map_channel_names",
    "positions",
    "position_ids",
    "free_space",
    "radio_config",
    "bs_pose",
    "repeat_seeds",
    "phase_reference_ids",
    "phase_reference_values",
    "phase_reference_source_sha256",
    "base_map_cluster_ids",
    "canonical_map_sha256",
    "noop_maps",
    "noop_map_sha256",
    "engine_config_json",
    "path_ids",
    "path_power",
    "path_surface_ids",
    "noop_path_ids",
    "noop_path_power",
    "noop_path_surface_ids",
    "primitive_surface_ids",
    "world_bits",
    "primitive_ids",
    "anchor_bits",
    "natural_world_index",
    "scene_ids",
    "city_ids",
    "bank_ids",
    "scene_roles",
    "position_roles",
    "metadata_json",
}


class FormalDatasetError(ValueError):
    pass


def same_physical_position(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare frozen BS-centered receiver coordinates at the contract tolerance."""
    return bool(
        np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=PHYSICAL_POSITION_ATOL_M,
        )
    )


@dataclass(frozen=True)
class FormalEdge:
    scene: int
    source_world: int
    target_world: int
    bit_index: int
    primitive_id: int
    direction: int


@dataclass
class FormalDataset:
    source_path: Path
    csi_repeat: np.ndarray
    csi_clean: np.ndarray
    maps: np.ndarray
    map_channel_names: np.ndarray
    positions: np.ndarray
    position_ids: np.ndarray
    free_space: np.ndarray
    radio_config: np.ndarray
    bs_pose: np.ndarray
    repeat_seeds: np.ndarray
    phase_reference_ids: np.ndarray
    phase_reference_values: np.ndarray
    phase_reference_source_sha256: np.ndarray
    base_map_cluster_ids: np.ndarray
    canonical_map_sha256: np.ndarray
    noop_maps: np.ndarray
    noop_map_sha256: np.ndarray
    engine_config: dict
    path_ids: np.ndarray
    path_power: np.ndarray
    path_surface_ids: np.ndarray
    noop_path_ids: np.ndarray
    noop_path_power: np.ndarray
    noop_path_surface_ids: np.ndarray
    primitive_surface_ids: np.ndarray
    world_bits: np.ndarray
    primitive_ids: np.ndarray
    anchor_bits: np.ndarray
    natural_world_index: np.ndarray
    scene_ids: np.ndarray
    city_ids: np.ndarray
    bank_ids: np.ndarray
    scene_roles: np.ndarray
    position_roles: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path: str | Path, require_clean_csi: bool = True) -> "FormalDataset":
        source = Path(path)
        if not source.is_file():
            raise FormalDatasetError(f"formal dataset not found: {source}")
        _validate_npz_members(source)
        with np.load(source, allow_pickle=False) as archive:
            try:
                metadata = parse_strict_json(str(np.asarray(archive["metadata_json"]).item()))
                engine_config = parse_strict_json(str(np.asarray(archive["engine_config_json"]).item()))
            except Exception as error:
                raise FormalDatasetError(f"metadata_json is not strict JSON: {error}") from error
            dataset = cls(
                source_path=source.resolve(),
                csi_repeat=np.asarray(archive["csi_repeat"], dtype=np.float64),
                csi_clean=np.asarray(archive["csi_clean"], dtype=np.float64),
                maps=np.asarray(archive["maps"], dtype=np.float64),
                map_channel_names=_string_array(archive["map_channel_names"]),
                positions=np.asarray(archive["positions"], dtype=np.float64),
                position_ids=_string_array(archive["position_ids"]),
                free_space=np.asarray(archive["free_space"]),
                radio_config=np.asarray(archive["radio_config"], dtype=np.float64),
                bs_pose=np.asarray(archive["bs_pose"], dtype=np.float64),
                repeat_seeds=np.asarray(archive["repeat_seeds"], dtype=np.int64),
                phase_reference_ids=_string_array(archive["phase_reference_ids"]),
                phase_reference_values=_complex_array(archive["phase_reference_values"]),
                phase_reference_source_sha256=_string_array(
                    archive["phase_reference_source_sha256"]
                ),
                base_map_cluster_ids=_string_array(archive["base_map_cluster_ids"]),
                canonical_map_sha256=_string_array(archive["canonical_map_sha256"]),
                noop_maps=np.asarray(archive["noop_maps"], dtype=np.float64),
                noop_map_sha256=_string_array(archive["noop_map_sha256"]),
                engine_config=engine_config,
                path_ids=np.asarray(archive["path_ids"], dtype=np.int64),
                path_power=np.asarray(archive["path_power"], dtype=np.float64),
                path_surface_ids=np.asarray(archive["path_surface_ids"], dtype=np.int64),
                noop_path_ids=np.asarray(archive["noop_path_ids"], dtype=np.int64),
                noop_path_power=np.asarray(archive["noop_path_power"], dtype=np.float64),
                noop_path_surface_ids=np.asarray(archive["noop_path_surface_ids"], dtype=np.int64),
                primitive_surface_ids=np.asarray(archive["primitive_surface_ids"], dtype=np.int64),
                world_bits=np.asarray(archive["world_bits"], dtype=np.int64),
                primitive_ids=np.asarray(archive["primitive_ids"], dtype=np.int64),
                anchor_bits=np.asarray(archive["anchor_bits"], dtype=np.int64),
                natural_world_index=np.asarray(archive["natural_world_index"], dtype=np.int64),
                scene_ids=_string_array(archive["scene_ids"]),
                city_ids=_string_array(archive["city_ids"]),
                bank_ids=_string_array(archive["bank_ids"]),
                scene_roles=_string_array(archive["scene_roles"]),
                position_roles=_string_array(archive["position_roles"]),
                metadata=metadata,
            )
        dataset.validate(require_clean_csi=require_clean_csi)
        return dataset

    @property
    def csi(self) -> np.ndarray:
        return self.csi_clean

    @property
    def scene_count(self) -> int:
        return int(self.csi_repeat.shape[0])

    @property
    def world_count(self) -> int:
        return int(self.csi_repeat.shape[1])

    @property
    def position_count(self) -> int:
        return int(self.csi_repeat.shape[2])

    @property
    def repeat_count(self) -> int:
        return int(self.csi_repeat.shape[3])

    @property
    def channel_count(self) -> int:
        return int(self.csi_repeat.shape[4])

    @property
    def bit_count(self) -> int:
        return int(self.world_bits.shape[1])

    @property
    def is_fixture(self) -> bool:
        return bool(self.metadata["fixture"])

    def indices_for_role(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.scene_roles == role)

    def independent_unit_id(self, scene: int) -> str:
        """Return the highest V6 split and inference unit for a scene row."""
        return str(self.base_map_cluster_ids[int(scene)])

    def independent_units_for_role(self, role: str) -> set[str]:
        return {
            self.independent_unit_id(int(scene))
            for scene in self.indices_for_role(role)
        }

    def unique_target_support_positions(self, city: str) -> list[tuple[int, int]]:
        """Return one row for each stable physical support position in a target city."""
        selected: list[tuple[int, int]] = []
        identifiers: set[str] = set()
        coordinates: list[np.ndarray] = []
        for scene_value in self.indices_for_role("target"):
            scene = int(scene_value)
            if str(self.city_ids[scene]) != str(city):
                continue
            for position_value in np.flatnonzero(
                self.position_roles[scene] == "support_pool"
            ):
                position = int(position_value)
                identifier = str(self.position_ids[scene, position])
                coordinate = self.positions[scene, position]
                if identifier in identifiers or any(
                    same_physical_position(coordinate, existing)
                    for existing in coordinates
                ):
                    continue
                identifiers.add(identifier)
                coordinates.append(coordinate)
                selected.append((scene, position))
        return selected

    def validate_target_support_capacity(self, minimum_unique_positions: int) -> None:
        if isinstance(minimum_unique_positions, bool) or not isinstance(
            minimum_unique_positions, int
        ):
            raise FormalDatasetError("minimum target support capacity must be an integer")
        if minimum_unique_positions < 0:
            raise FormalDatasetError("minimum target support capacity must be nonnegative")
        target_cities = sorted(
            set(str(value) for value in self.city_ids[self.scene_roles == "target"])
        )
        for city in target_cities:
            available = len(self.unique_target_support_positions(city))
            if available < minimum_unique_positions:
                raise FormalDatasetError(
                    f"target city {city!r} has {available} unique support_pool positions, "
                    f"fewer than configured max k={minimum_unique_positions}"
                )

    @cached_property
    def canonical_base_map_digests(self) -> np.ndarray:
        """Content identities for the unedited (all-zero bit state) foundations."""
        zero_worlds = np.flatnonzero(np.all(self.world_bits == 0, axis=1))
        if zero_worlds.size != 1:
            raise FormalDatasetError("world_bits must contain exactly one all-zero foundation world")
        world = int(zero_worlds[0])
        representation = self.metadata["representation"]
        return np.asarray(
            [
                _canonical_foundation_sha256(
                    self.maps[scene, world],
                    self.map_channel_names,
                    float(representation["map_resolution_m"]),
                    representation["map_origin_xy_m"],
                )
                for scene in range(self.scene_count)
            ],
            dtype="U64",
        )

    def canonical_base_map_digest(self, scene: int) -> str:
        return str(self.canonical_base_map_digests[int(scene)])

    def observation_noise_binding_digest(
        self,
        scene: int,
        csi_repeat: np.ndarray | None = None,
    ) -> str:
        """Bind every regenerated residual to its declared observation seed and index."""
        scene_index = int(scene)
        repeated = self.csi_repeat[scene_index] if csi_repeat is None else np.asarray(csi_repeat)
        if repeated.shape != self.csi_repeat[scene_index].shape:
            raise FormalDatasetError("regenerated csi_repeat scene has the wrong shape")
        residual = repeated - self.csi_clean[scene_index, :, :, None, :]
        digest = hashlib.sha256()
        for world in range(self.world_count):
            for position in range(self.position_count):
                for repeat in range(self.repeat_count):
                    digest.update(
                        np.asarray(
                            [world, position, repeat, self.repeat_seeds[scene_index, world, position, repeat]],
                            dtype="<i8",
                        ).tobytes()
                    )
                    digest.update(
                        np.ascontiguousarray(residual[world, position, repeat], dtype="<f8").tobytes()
                    )
        return digest.hexdigest()

    def directed_edges(self, scene: int) -> Iterator[FormalEdge]:
        lookup = {tuple(int(value) for value in row): index for index, row in enumerate(self.world_bits)}
        for source_world, bits in enumerate(self.world_bits):
            for bit_index in range(self.bit_count):
                target_bits = bits.copy()
                target_bits[bit_index] = 1 - target_bits[bit_index]
                target_world = lookup[tuple(int(value) for value in target_bits)]
                yield FormalEdge(
                    scene=scene,
                    source_world=source_world,
                    target_world=target_world,
                    bit_index=bit_index,
                    primitive_id=int(self.primitive_ids[scene, bit_index]),
                    direction=int(target_bits[bit_index] - bits[bit_index]),
                )

    def validate(
        self,
        require_clean_csi: bool = True,
        minimum_repeats: int = 2,
        minimum_target_cities: int = 2,
        minimum_source_cities: int = 2,
        minimum_banks_per_target_city: int = 2,
        minimum_independent_base_map_clusters_per_target_city: int = 2,
        minimum_banks_per_source_role: int = 1,
        minimum_unique_support_positions_per_target_city: int = 1,
    ) -> None:
        if require_clean_csi is not True:
            raise FormalDatasetError("V6 does not permit disabling clean CSI targets")
        if self.csi_repeat.ndim != 5:
            raise FormalDatasetError("csi_repeat must have shape [scene, world, position, repeat, channel]")
        scenes, worlds, positions, repeats, channels = self.csi_repeat.shape
        if scenes <= 0 or worlds <= 1 or positions <= 1 or repeats < minimum_repeats or channels <= 1:
            raise FormalDatasetError("dataset axes are too small for the requested formal protocol")
        if self.maps.ndim != 5 or self.maps.shape[:2] != (scenes, worlds):
            raise FormalDatasetError("maps must have shape [scene, world, map_channel, row, column]")
        if self.maps.shape[3] != self.maps.shape[4] or self.maps.shape[3] < 4:
            raise FormalDatasetError("formal maps must be square and at least 4x4")
        map_channels = self.maps.shape[2]
        if self.map_channel_names.shape != (map_channels,):
            raise FormalDatasetError("map_channel_names must have shape [map_channel]")
        required_map_channels = {"occupancy", "height", "material"}
        if not required_map_channels.issubset(set(self.map_channel_names.tolist())):
            raise FormalDatasetError("maps must contain occupancy, height, and material channels")
        if self.positions.shape != (scenes, positions, 2):
            raise FormalDatasetError("positions must have shape [scene, position, 2]")
        if self.position_ids.shape != (scenes, positions):
            raise FormalDatasetError("position_ids must have shape [scene, position]")
        if self.free_space.shape != (scenes, worlds, positions) or self.free_space.dtype.kind != "b":
            raise FormalDatasetError("free_space must be boolean [scene, world, position]")
        if not bool(np.all(self.free_space)):
            raise FormalDatasetError("every stored receiver position must be free in every sibling world")
        if self.radio_config.ndim != 2 or self.radio_config.shape[0] != scenes or self.radio_config.shape[1] < 1:
            raise FormalDatasetError("radio_config must have shape [scene, radio_feature]")
        if self.bs_pose.shape != (scenes, 7):
            raise FormalDatasetError(
                "bs_pose must have shape [scene, 7] as xyz plus scalar-first wxyz unit quaternion"
            )
        quaternion_norm = np.linalg.norm(self.bs_pose[:, 3:], axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1e-6, rtol=0.0):
            raise FormalDatasetError("bs_pose scalar-first wxyz quaternion must have unit norm")
        if self.repeat_seeds.shape != (scenes, worlds, positions, repeats):
            raise FormalDatasetError("repeat_seeds must have shape [scene, world, position, repeat]")
        if np.any(self.repeat_seeds < 0):
            raise FormalDatasetError("repeat_seeds must be nonnegative")
        for scene in range(scenes):
            for position in range(positions):
                seeds = self.repeat_seeds[scene, :, position, :].ravel()
                if len(set(int(value) for value in seeds)) != seeds.size:
                    raise FormalDatasetError("repeat seeds must be independent across sibling worlds")
        if self.phase_reference_ids.shape != (scenes, positions):
            raise FormalDatasetError("phase_reference_ids must have shape [scene, position]")
        if np.any(self.phase_reference_ids == ""):
            raise FormalDatasetError("phase_reference_ids must be nonempty and shared by sibling worlds")
        if self.phase_reference_values.shape != (scenes, positions):
            raise FormalDatasetError(
                "phase_reference_values must have world-independent shape [scene, position]"
            )
        if not np.all(np.isfinite(self.phase_reference_values)) or np.any(
            np.abs(self.phase_reference_values) == 0.0
        ):
            raise FormalDatasetError(
                "phase_reference_values must contain finite nonzero complex references"
            )
        if self.phase_reference_source_sha256.shape != (scenes, positions):
            raise FormalDatasetError(
                "phase_reference_source_sha256 must have world-independent shape [scene, position]"
            )
        if any(not _sha256(str(value)) for value in self.phase_reference_source_sha256.flat):
            raise FormalDatasetError(
                "phase_reference_source_sha256 must contain lowercase SHA-256 digests"
            )
        if self.base_map_cluster_ids.shape != (scenes,):
            raise FormalDatasetError("base_map_cluster_ids must have shape [scene]")
        if self.canonical_map_sha256.shape != (scenes, worlds):
            raise FormalDatasetError("canonical_map_sha256 must have shape [scene, world]")
        if self.noop_maps.shape != self.maps.shape or self.noop_map_sha256.shape != (scenes, worlds):
            raise FormalDatasetError("noop_maps and noop_map_sha256 must match canonical maps")
        for scene in range(scenes):
            for world in range(worlds):
                expected = _array_sha256(self.maps[scene, world])
                if self.canonical_map_sha256[scene, world] != expected:
                    raise FormalDatasetError("canonical map digest does not match the stored rendering")
                noop_expected = _array_sha256(self.noop_maps[scene, world])
                if self.noop_map_sha256[scene, world] != noop_expected:
                    raise FormalDatasetError("no-op canonical map digest does not match the stored rendering")
        if self.csi_clean.shape != (scenes, worlds, positions, channels):
            raise FormalDatasetError("csi_clean must match csi_repeat without the repeat axis")
        if self.path_ids.ndim != 4 or self.path_ids.shape[:3] != (scenes, worlds, positions):
            raise FormalDatasetError("path_ids must have shape [scene, world, position, path]")
        if self.path_power.shape != self.path_ids.shape:
            raise FormalDatasetError("path_power must match path_ids")
        if self.path_surface_ids.ndim != 5 or self.path_surface_ids.shape[:4] != self.path_ids.shape:
            raise FormalDatasetError("path_surface_ids must have shape [scene, world, position, path, interaction]")
        if self.noop_path_ids.shape != self.path_ids.shape:
            raise FormalDatasetError("noop_path_ids must match path_ids")
        if self.noop_path_power.shape != self.path_power.shape:
            raise FormalDatasetError("noop_path_power must match path_power")
        if self.noop_path_surface_ids.shape != self.path_surface_ids.shape:
            raise FormalDatasetError("noop_path_surface_ids must match path_surface_ids")
        if self.primitive_surface_ids.ndim != 3 or self.primitive_surface_ids.shape[:2] != (scenes, self.bit_count):
            raise FormalDatasetError("primitive_surface_ids must have shape [scene, bit, surface]")
        if np.any(self.path_power < 0) or np.any((self.path_ids < 0) & (self.path_power != 0)):
            raise FormalDatasetError("padded paths must have zero nonnegative power")

        bits = self.world_bits
        if bits.ndim != 2 or bits.shape[0] != worlds or worlds != 2 ** bits.shape[1]:
            raise FormalDatasetError("world_bits must enumerate one complete binary hypercube")
        if set(np.unique(bits)).difference({0, 1}):
            raise FormalDatasetError("world_bits must be binary")
        if len({tuple(row) for row in bits.tolist()}) != worlds:
            raise FormalDatasetError("world_bits rows must be unique")
        bit_count = bits.shape[1]
        if self.primitive_ids.shape != (scenes, bit_count):
            raise FormalDatasetError("primitive_ids must have shape [scene, bit]")
        for scene, mapping in enumerate(self.primitive_ids):
            if sorted(int(value) for value in mapping) != list(range(bit_count)):
                raise FormalDatasetError(f"primitive_ids[{scene}] must be a permutation of [0, bit)")
        if self.anchor_bits.shape != (scenes, bit_count) or set(np.unique(self.anchor_bits)).difference({0, 1}):
            raise FormalDatasetError("anchor_bits must have shape [scene, bit] and be binary")
        if self.natural_world_index.shape != (scenes,):
            raise FormalDatasetError("natural_world_index must have shape [scene]")
        for scene, world_index in enumerate(self.natural_world_index):
            if int(world_index) < 0 or int(world_index) >= worlds:
                raise FormalDatasetError("natural_world_index is out of range")
            if not np.array_equal(bits[int(world_index)], self.anchor_bits[scene]):
                raise FormalDatasetError("natural world must match the per-bank randomized anchor_bits")

        for name, array, shape in (
            ("scene_ids", self.scene_ids, (scenes,)),
            ("city_ids", self.city_ids, (scenes,)),
            ("bank_ids", self.bank_ids, (scenes,)),
            ("scene_roles", self.scene_roles, (scenes,)),
            ("position_roles", self.position_roles, (scenes, positions)),
            ("position_ids", self.position_ids, (scenes, positions)),
        ):
            if array.shape != shape:
                raise FormalDatasetError(f"{name} must have shape {shape}")
            if any(not str(value).strip() for value in array.flat):
                raise FormalDatasetError(f"{name} contains an empty identifier")
        for scene in range(scenes):
            if len(set(self.position_ids[scene].tolist())) != positions:
                raise FormalDatasetError("position_ids must be unique within each scene bank")
        city_position_coordinates = {}
        city_position_roles = {}
        city_coordinate_rows: dict[
            str, dict[tuple[int, int], list[tuple[np.ndarray, str, str]]]
        ] = {}
        for scene in range(scenes):
            city = str(self.city_ids[scene])
            for position in range(positions):
                position_id = str(self.position_ids[scene, position])
                key = (city, position_id)
                coordinate = self.positions[scene, position]
                if key in city_position_coordinates and not np.allclose(
                    city_position_coordinates[key], coordinate, rtol=0.0, atol=1e-9
                ):
                    raise FormalDatasetError(
                        "a city-level position_id maps to inconsistent BS-centered coordinates"
                    )
                city_position_coordinates[key] = coordinate
                role = str(self.position_roles[scene, position])
                if key in city_position_roles and city_position_roles[key] != role:
                    raise FormalDatasetError(
                        "a city-level position_id may not cross support_pool/query roles"
                    )
                city_position_roles[key] = role
                bucket = tuple(
                    int(value)
                    for value in np.floor(
                        np.asarray(coordinate, dtype=np.float64) / PHYSICAL_POSITION_ATOL_M
                    )
                )
                city_buckets = city_coordinate_rows.setdefault(city, {})
                matched = False
                for delta_x in (-1, 0, 1):
                    for delta_y in (-1, 0, 1):
                        candidate_bucket = (bucket[0] + delta_x, bucket[1] + delta_y)
                        for existing_coordinate, existing_id, existing_role in city_buckets.get(
                            candidate_bucket, ()
                        ):
                            if not same_physical_position(existing_coordinate, coordinate):
                                continue
                            if existing_id != position_id:
                                raise FormalDatasetError(
                                    "the same city-level BS-centered coordinate must map to one position_id"
                                )
                            if existing_role != role:
                                raise FormalDatasetError(
                                    "the same city-level physical position may not cross "
                                    "support_pool/query roles"
                                )
                            matched = True
                            break
                        if matched:
                            break
                    if matched:
                        break
                if not matched:
                    city_buckets.setdefault(bucket, []).append(
                        (coordinate, position_id, role)
                    )
        for name, identifiers in (
            ("scene_ids", self.scene_ids),
            ("bank_ids", self.bank_ids),
        ):
            if len(set(identifiers.tolist())) != scenes:
                raise FormalDatasetError(f"{name} must be globally unique")
        invalid_roles = set(self.scene_roles.tolist()).difference(SCENE_ROLES)
        if invalid_roles:
            raise FormalDatasetError(f"unknown scene_roles: {sorted(invalid_roles)}")
        invalid_position_roles = set(self.position_roles.ravel().tolist()).difference(POSITION_ROLES)
        if invalid_position_roles:
            raise FormalDatasetError(f"unknown position_roles: {sorted(invalid_position_roles)}")
        for scene, role in enumerate(self.scene_roles):
            observed = set(self.position_roles[scene].tolist())
            if role == "target":
                if not {"support_pool", "query"}.issubset(observed) or "standard" in observed:
                    raise FormalDatasetError("target banks require disjoint support_pool/query positions")
            elif observed != {"standard"}:
                raise FormalDatasetError("non-target banks must use only standard positions")
        for role in SOURCE_ROLES:
            if len(self.independent_units_for_role(role)) < minimum_banks_per_source_role:
                raise FormalDatasetError(f"role {role!r} has too few independent base-map clusters")
        target_cities = set(self.city_ids[self.scene_roles == "target"].tolist())
        if len(target_cities) < minimum_target_cities:
            raise FormalDatasetError("too few independent target cities")
        source_cities = set(self.city_ids[np.isin(self.scene_roles, SOURCE_ROLES)].tolist())
        if len(source_cities) < minimum_source_cities:
            raise FormalDatasetError("too few independent source cities")
        for role in SOURCE_ROLES:
            role_cities = set(self.city_ids[self.scene_roles == role].tolist())
            if len(role_cities) < minimum_source_cities:
                raise FormalDatasetError(
                    f"{role} covers too few independent source cities"
                )
        if source_cities.intersection(target_cities):
            raise FormalDatasetError("source and target city identifiers must be disjoint")
        self.validate_target_support_capacity(
            minimum_unique_support_positions_per_target_city
        )
        for city in target_cities:
            bank_count = len(
                {
                    str(self.bank_ids[scene])
                    for scene in range(scenes)
                    if self.scene_roles[scene] == "target" and self.city_ids[scene] == city
                }
            )
            if bank_count < minimum_banks_per_target_city:
                raise FormalDatasetError(
                    f"target city {city!r} has too few distinct banks"
                )
            canonical_cluster_count = len(
                {
                    self.canonical_base_map_digest(scene)
                    for scene in range(scenes)
                    if self.scene_roles[scene] == "target" and self.city_ids[scene] == city
                }
            )
            if (
                canonical_cluster_count
                < minimum_independent_base_map_clusters_per_target_city
            ):
                raise FormalDatasetError(
                    f"target city {city!r} has too few independent canonical base-map clusters"
                )
        for cluster in set(self.base_map_cluster_ids.tolist()):
            mask = self.base_map_cluster_ids == cluster
            if len(set(self.scene_roles[mask].tolist())) != 1:
                raise FormalDatasetError("a repeated base-map cluster may not cross scene roles")
            if len(set(self.city_ids[mask].tolist())) != 1:
                raise FormalDatasetError("a repeated base-map cluster may not cross cities")

        for name, array in (
            ("csi_repeat", self.csi_repeat),
            ("maps", self.maps),
            ("positions", self.positions),
            ("radio_config", self.radio_config),
            ("bs_pose", self.bs_pose),
            ("noop_maps", self.noop_maps),
            ("path_power", self.path_power),
            ("noop_path_power", self.noop_path_power),
        ):
            if not np.all(np.isfinite(array)):
                raise FormalDatasetError(f"{name} contains non-finite values")
        if not np.all(np.isfinite(self.csi_clean)):
            raise FormalDatasetError("csi_clean contains non-finite values")
        if not np.any(np.std(self.csi_repeat, axis=(0, 1, 2, 3)) > 1e-12):
            raise FormalDatasetError("CSI channels are constant")

        self._validate_metadata()
        representation = self.metadata["representation"]
        origin = np.asarray(representation["map_origin_xy_m"], dtype=np.float64)
        resolution = float(representation["map_resolution_m"])
        upper = origin + resolution * np.asarray(
            [self.maps.shape[-1], self.maps.shape[-2]], dtype=np.float64
        )
        if np.any(self.positions < origin[None, None]) or np.any(
            self.positions >= upper[None, None]
        ):
            raise FormalDatasetError(
                "receiver position lies outside the frozen map extent"
            )
        if np.any(self.bs_pose[:, :2] < origin[None]) or np.any(
            self.bs_pose[:, :2] >= upper[None]
        ):
            raise FormalDatasetError("BS pose lies outside the frozen map extent")
        engine_config_digest = hashlib.sha256(
            _canonical_json_bytes(self.engine_config)
        ).hexdigest()
        if engine_config_digest != self.metadata["engine"]["config_sha256"]:
            raise FormalDatasetError("engine_config_json does not match metadata.engine.config_sha256")
        self._validate_foundation_identity()
        self._validate_repeat_independence()
        self._validate_randomization()
        has_external_scenes = bool(np.any(self.scene_roles == "external_validation"))
        if has_external_scenes != bool(self.metadata["external_reference"]["available"]):
            raise FormalDatasetError(
                "external_validation scenes and metadata.external_reference.available must agree"
            )

    def contract_report(self) -> dict:
        roles = {role: int(np.sum(self.scene_roles == role)) for role in SCENE_ROLES}
        role_clusters = {
            role: len(self.independent_units_for_role(role)) for role in SCENE_ROLES
        }
        target_cities = sorted(set(self.city_ids[self.scene_roles == "target"].tolist()))
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "status": "PASS",
            "dataset": str(self.source_path),
            "dataset_sha256": sha256_file(self.source_path),
            "dataset_id": self.metadata["dataset_id"],
            "scientific_use": self.metadata["scientific_use"],
            "fixture": self.is_fixture,
            "shape": {
                "scenes": self.scene_count,
                "worlds": self.world_count,
                "positions": self.position_count,
                "repeats": self.repeat_count,
                "channels": self.channel_count,
                "map_channels": int(self.maps.shape[2]),
                "map_size": int(self.maps.shape[3]),
                "radio_features": int(self.radio_config.shape[1]),
                "bits": self.bit_count,
            },
            "scene_role_counts": roles,
            "independent_cluster_role_counts": role_clusters,
            "target_cities": target_cities,
            "engine": self.metadata["engine"],
            "external_reference": self.metadata["external_reference"],
            "randomization_digest": self.randomization_digest(),
            "base_map_clusters": len(set(self.base_map_cluster_ids.tolist())),
            "canonical_base_map_digest_count": len(set(self.canonical_base_map_digests.tolist())),
            "foundation_identity_enforced": not self.is_fixture,
            "canonical_base_map_digests_by_bank": {
                str(self.bank_ids[scene]): self.canonical_base_map_digest(scene)
                for scene in range(self.scene_count)
            },
        }

    def randomization_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.primitive_ids.astype("<i8").tobytes())
        digest.update(self.anchor_bits.astype("<i8").tobytes())
        return digest.hexdigest()

    def _validate_randomization(self) -> None:
        source_indices = np.flatnonzero(np.isin(self.scene_roles, SOURCE_ROLES[:2]))
        if source_indices.size >= 2:
            permutations = {tuple(row) for row in self.primitive_ids[source_indices].tolist()}
            anchors = {tuple(row) for row in self.anchor_bits[source_indices].tolist()}
            if len(permutations) < 2:
                raise FormalDatasetError("source banks do not randomize bit-to-primitive ordering")
            if len(anchors) < 2:
                raise FormalDatasetError("source banks do not randomize the natural anchor")

    def _validate_repeat_independence(self) -> None:
        residual = self.csi_repeat - self.csi_clean[:, :, :, None, :]
        for scene in range(self.scene_count):
            scale = max(float(np.max(np.abs(residual[scene]))), np.finfo(np.float64).eps)
            for position in range(self.position_count):
                observations = residual[scene, :, position].reshape(
                    self.world_count * self.repeat_count, self.channel_count
                )
                if self.channel_count >= 8:
                    norms = np.linalg.norm(observations, axis=1)
                    valid = norms > scale * 1e-12
                    normalized_directions = observations[valid] / norms[valid, None]
                    if normalized_directions.shape[0] > 1:
                        cosine = np.abs(normalized_directions @ normalized_directions.T)
                        cosine[np.diag_indices_from(cosine)] = 0.0
                        if bool(np.any(cosine >= 1.0 - 1e-10)):
                            raise FormalDatasetError(
                                "sibling observations reuse or near-perfectly correlate noise realizations"
                            )
                # Rounded normalized bytes make an exact copied residual detectable even
                # after adding it to, and subtracting it from, a different clean target.
                normalized = np.round(observations / scale, decimals=10)
                seen: dict[bytes, int] = {}
                for index, value in enumerate(normalized):
                    key = np.ascontiguousarray(value, dtype="<f8").tobytes()
                    if key in seen and np.allclose(
                        observations[index],
                        observations[seen[key]],
                        rtol=1e-10,
                        atol=scale * 1e-12,
                    ):
                        raise FormalDatasetError(
                            "sibling observations copy the same observation-noise realization"
                        )
                    seen[key] = index

    def _validate_foundation_identity(self) -> None:
        # The deterministic fixture deliberately reuses its toy boundary map. It is
        # permanently forbidden evidence; formal datasets must enforce content identity.
        if self.is_fixture:
            return
        digests = self.canonical_base_map_digests
        for digest in set(digests.tolist()):
            mask = digests == digest
            clusters = set(self.base_map_cluster_ids[mask].tolist())
            roles = set(self.scene_roles[mask].tolist())
            cities = set(self.city_ids[mask].tolist())
            if len(clusters) != 1 or len(roles) != 1 or len(cities) != 1:
                raise FormalDatasetError(
                    "identical canonical foundation content may not cross base-map clusters, roles, or cities"
                )
        for cluster in set(self.base_map_cluster_ids.tolist()):
            if len(set(digests[self.base_map_cluster_ids == cluster].tolist())) != 1:
                raise FormalDatasetError(
                    "one base-map cluster id may not alias different canonical foundation content"
                )

    def _validate_metadata(self) -> None:
        metadata = self.metadata
        required = {
            "schema_version",
            "dataset_id",
            "dataset_version",
            "scientific_use",
            "fixture",
            "engine",
            "representation",
            "assets",
            "generation",
            "external_reference",
        }
        if not isinstance(metadata, dict) or set(metadata) != required:
            actual = set(metadata) if isinstance(metadata, dict) else set()
            raise FormalDatasetError(
                f"metadata fields must be exact; missing={sorted(required - actual)}, "
                f"unexpected={sorted(actual - required)}"
            )
        if metadata["schema_version"] != DATASET_SCHEMA_VERSION:
            raise FormalDatasetError(f"metadata.schema_version must be {DATASET_SCHEMA_VERSION!r}")
        if metadata["scientific_use"] not in {"CANDIDATE", "QUALIFIED", "FORBIDDEN"}:
            raise FormalDatasetError("metadata.scientific_use is invalid")
        if type(metadata["fixture"]) is not bool:
            raise FormalDatasetError("metadata.fixture must be boolean")
        if metadata["fixture"] and metadata["scientific_use"] != "FORBIDDEN":
            raise FormalDatasetError("fixtures must set scientific_use=FORBIDDEN")
        _nonempty_strings(metadata, ("dataset_id", "dataset_version"), "metadata")

        engine = metadata["engine"]
        _exact_object(
            engine,
            {"name", "version", "source_revision", "license_id", "config_sha256", "deterministic"},
            "metadata.engine",
        )
        _nonempty_strings(engine, ("name", "version", "source_revision", "license_id"), "metadata.engine")
        if not _sha256(engine["config_sha256"]):
            raise FormalDatasetError("metadata.engine.config_sha256 must be lowercase SHA-256")
        if type(engine["deterministic"]) is not bool:
            raise FormalDatasetError("metadata.engine.deterministic must be boolean")

        representation = metadata["representation"]
        _exact_object(
            representation,
            {
                "csi_layout",
                "csi_units",
                "phase_gauge_rule",
                "coordinate_system",
                "position_units",
                "map_units",
                "clean_target_definition",
                "antenna_count",
                "subcarrier_count",
                "patch_complex_size",
                "patch_antenna_size",
                "patch_subcarrier_size",
                "map_resolution_m",
                "map_origin_xy_m",
                "alignment_physical_representation",
            },
            "metadata.representation",
        )
        _nonempty_strings(
            representation,
            (
                "csi_layout",
                "csi_units",
                "phase_gauge_rule",
                "coordinate_system",
                "position_units",
                "map_units",
                "clean_target_definition",
            ),
            "metadata.representation",
        )
        if representation["csi_layout"] != "real_then_imag":
            raise FormalDatasetError("V2 requires csi_layout=real_then_imag")
        if self.channel_count % 2:
            raise FormalDatasetError("real_then_imag CSI requires an even channel count")
        if representation["phase_gauge_rule"] != "shared_complex_reference":
            raise FormalDatasetError(
                "formal V6 raw-complex routing and response targets require "
                "phase_gauge_rule=shared_complex_reference; phase-invariant targets are not implemented"
            )
        if representation["alignment_physical_representation"] != "complex_csi_plus_delay_angle_power":
            raise FormalDatasetError(
                "V6 P0 requires alignment_physical_representation=complex_csi_plus_delay_angle_power"
            )
        if representation["coordinate_system"] != "bs_centered_right_handed_meters":
            raise FormalDatasetError("coordinate_system must be bs_centered_right_handed_meters")
        if representation["position_units"] != "m" or representation["map_units"] != "m":
            raise FormalDatasetError("position and map units must be meters")
        for key in (
            "antenna_count",
            "subcarrier_count",
            "patch_complex_size",
            "patch_antenna_size",
            "patch_subcarrier_size",
        ):
            if type(representation[key]) is not int or representation[key] < 1:
                raise FormalDatasetError(f"metadata.representation.{key} must be positive integer")
        complex_values = int(representation["antenna_count"]) * int(representation["subcarrier_count"])
        if self.channel_count != 2 * complex_values:
            raise FormalDatasetError("antenna/subcarrier shape does not match real_then_imag channels")
        patch_a = int(representation["patch_antenna_size"])
        patch_s = int(representation["patch_subcarrier_size"])
        if patch_a * patch_s != int(representation["patch_complex_size"]):
            raise FormalDatasetError("patch_complex_size must equal the 2D patch area")
        if int(representation["antenna_count"]) % patch_a or int(representation["subcarrier_count"]) % patch_s:
            raise FormalDatasetError("2D patches must tile the antenna x subcarrier grid")
        patch_count = (
            int(representation["antenna_count"]) // patch_a
        ) * (
            int(representation["subcarrier_count"]) // patch_s
        )
        if patch_count % 4:
            raise FormalDatasetError(
                "formal patch grid must support exact 75% masking"
            )
        if not isinstance(representation["map_resolution_m"], (int, float)) or float(
            representation["map_resolution_m"]
        ) <= 0:
            raise FormalDatasetError("metadata.representation.map_resolution_m must be positive")
        origin = representation["map_origin_xy_m"]
        if (
            not isinstance(origin, list)
            or len(origin) != 2
            or any(not isinstance(value, (int, float)) for value in origin)
        ):
            raise FormalDatasetError("metadata.representation.map_origin_xy_m must be [x,y]")

        assets = metadata["assets"]
        _exact_object(
            assets,
            {"license_ids", "provenance_uri", "material_library", "material_category_count", "redistribution_allowed"},
            "metadata.assets",
        )
        if not isinstance(assets["license_ids"], list) or not assets["license_ids"]:
            raise FormalDatasetError("metadata.assets.license_ids must be nonempty")
        if any(not isinstance(value, str) or not value.strip() for value in assets["license_ids"]):
            raise FormalDatasetError("metadata.assets.license_ids contains an empty value")
        _nonempty_strings(assets, ("provenance_uri", "material_library"), "metadata.assets")
        if type(assets["material_category_count"]) is not int or assets["material_category_count"] < 1:
            raise FormalDatasetError("metadata.assets.material_category_count must be positive integer")
        material_index = list(self.map_channel_names).index("material")
        materials = self.maps[:, :, material_index]
        if not np.allclose(materials, np.round(materials)):
            raise FormalDatasetError("material map channel must be categorical integers")
        if np.any(materials < 0) or np.any(materials >= int(assets["material_category_count"])):
            raise FormalDatasetError("material map value is outside the frozen category vocabulary")
        if type(assets["redistribution_allowed"]) is not bool:
            raise FormalDatasetError("metadata.assets.redistribution_allowed must be boolean")

        generation = metadata["generation"]
        _exact_object(generation, {"created_utc", "generator_command", "seed"}, "metadata.generation")
        _nonempty_strings(generation, ("created_utc", "generator_command"), "metadata.generation")
        if type(generation["seed"]) is not int or generation["seed"] < 0:
            raise FormalDatasetError("metadata.generation.seed must be nonnegative")

        external = metadata["external_reference"]
        _exact_object(external, {"available", "kind", "dataset_id", "pairing_rule"}, "metadata.external_reference")
        if type(external["available"]) is not bool:
            raise FormalDatasetError("metadata.external_reference.available must be boolean")
        _nonempty_strings(external, ("kind", "dataset_id", "pairing_rule"), "metadata.external_reference")


def _string_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"}:
        raise FormalDatasetError("identifier arrays must use fixed-width string dtype, not object")
    return array.astype(str)


def _complex_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "c":
        raise FormalDatasetError("phase_reference_values must use a complex numeric dtype")
    return np.asarray(array, dtype=np.complex128)


def _exact_object(value: object, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise FormalDatasetError(
            f"{name} fields must be exact; missing={sorted(keys - actual)}, unexpected={sorted(actual - keys)}"
        )


def _nonempty_strings(mapping: dict, keys: tuple[str, ...], name: str) -> None:
    for key in keys:
        if not isinstance(mapping[key], str) or not mapping[key].strip():
            raise FormalDatasetError(f"{name}.{key} must be a nonempty string")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _canonical_foundation_sha256(
    foundation_map: np.ndarray,
    channel_names: np.ndarray,
    map_resolution_m: float,
    map_origin_xy_m: object,
) -> str:
    canonical_map = np.array(foundation_map, dtype="<f8", order="C", copy=True)
    canonical_map = np.round(canonical_map, decimals=9)
    canonical_map[canonical_map == 0.0] = 0.0
    canonical_origin = [round(float(value), 9) for value in map_origin_xy_m]
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "channel_names": [str(value) for value in np.asarray(channel_names).tolist()],
                "map_resolution_m": round(float(map_resolution_m), 9),
                "map_origin_xy_m": canonical_origin,
                "shape": list(canonical_map.shape),
                "metric_quantization_decimals": 9,
            }
        )
    )
    digest.update(canonical_map.tobytes())
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _validate_npz_members(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
    except zipfile.BadZipFile as error:
        raise FormalDatasetError("dataset is not a valid NPZ/ZIP container") from error
    if any("/" in member or "\\" in member or not member.endswith(".npy") for member in members):
        raise FormalDatasetError("NPZ members must be flat .npy files")
    if len(members) != len(set(members)):
        raise FormalDatasetError("NPZ contains duplicate array members")
    actual = {member[:-4] for member in members}
    if actual != REQUIRED_ARRAYS:
        raise FormalDatasetError(
            f"dataset array fields must be exact; missing={sorted(REQUIRED_ARRAYS-actual)}, "
            f"unexpected={sorted(actual-REQUIRED_ARRAYS)}"
        )
