from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FROZEN_RANDOM_MASK_FRACTION = 0.75


@dataclass(frozen=True)
class PatchSpec:
    antennas: int
    subcarriers: int
    patch_complex_size: int
    patch_antenna_size: int | None = None
    patch_subcarrier_size: int | None = None

    def __post_init__(self) -> None:
        antenna_size = 1 if self.patch_antenna_size is None else int(self.patch_antenna_size)
        subcarrier_size = (
            int(self.patch_complex_size)
            if self.patch_subcarrier_size is None
            else int(self.patch_subcarrier_size)
        )
        if antenna_size < 1 or subcarrier_size < 1:
            raise ValueError("2D patch dimensions must be positive")
        if antenna_size * subcarrier_size != int(self.patch_complex_size):
            raise ValueError("patch_complex_size must equal the 2D patch area")
        if self.antennas % antenna_size or self.subcarriers % subcarrier_size:
            raise ValueError("2D patch dimensions must tile the antenna x subcarrier grid")
        object.__setattr__(self, "patch_antenna_size", antenna_size)
        object.__setattr__(self, "patch_subcarrier_size", subcarrier_size)

    @property
    def complex_values(self) -> int:
        return self.antennas * self.subcarriers

    @property
    def patch_count(self) -> int:
        return self.patch_rows * self.patch_columns

    @property
    def patch_rows(self) -> int:
        return self.antennas // int(self.patch_antenna_size)

    @property
    def patch_columns(self) -> int:
        return self.subcarriers // int(self.patch_subcarrier_size)

    @property
    def patch_dim(self) -> int:
        return 2 * self.patch_complex_size

    @classmethod
    def from_metadata(cls, metadata: dict) -> "PatchSpec":
        representation = metadata["representation"]
        return cls(
            antennas=int(representation["antenna_count"]),
            subcarriers=int(representation["subcarrier_count"]),
            patch_complex_size=int(representation["patch_complex_size"]),
            patch_antenna_size=int(representation["patch_antenna_size"]),
            patch_subcarrier_size=int(representation["patch_subcarrier_size"]),
        )


def patchify_csi(csi: np.ndarray, spec: PatchSpec) -> np.ndarray:
    array = np.asarray(csi)
    if array.shape[-1] != 2 * spec.complex_values:
        raise ValueError("CSI channel dimension does not match PatchSpec")
    real = array[..., : spec.complex_values]
    imag = array[..., spec.complex_values :]
    paired = np.stack((real, imag), axis=-1).reshape(
        *array.shape[:-1], spec.antennas, spec.subcarriers, 2
    )
    prefix = paired.shape[:-3]
    tiled = paired.reshape(
        *prefix,
        spec.patch_rows,
        int(spec.patch_antenna_size),
        spec.patch_columns,
        int(spec.patch_subcarrier_size),
        2,
    )
    axes = list(range(len(prefix))) + [len(prefix), len(prefix) + 2, len(prefix) + 1, len(prefix) + 3, len(prefix) + 4]
    return tiled.transpose(axes).reshape(*prefix, spec.patch_count, spec.patch_dim)


def unpatchify_csi(patches: np.ndarray, spec: PatchSpec) -> np.ndarray:
    array = np.asarray(patches)
    if array.shape[-2:] != (spec.patch_count, spec.patch_dim):
        raise ValueError("patch tensor does not match PatchSpec")
    prefix = array.shape[:-2]
    tiled = array.reshape(
        *prefix,
        spec.patch_rows,
        spec.patch_columns,
        int(spec.patch_antenna_size),
        int(spec.patch_subcarrier_size),
        2,
    )
    axes = list(range(len(prefix))) + [len(prefix), len(prefix) + 2, len(prefix) + 1, len(prefix) + 3, len(prefix) + 4]
    paired = tiled.transpose(axes).reshape(*prefix, spec.complex_values, 2)
    return np.concatenate((paired[..., 0], paired[..., 1]), axis=-1)


def delay_angle_power(csi: np.ndarray, spec: PatchSpec) -> np.ndarray:
    """Frozen phase-invariant delay-angle power summary of the complete CSI grid."""
    array = np.asarray(csi)
    if array.shape[-1] != 2 * spec.complex_values:
        raise ValueError("CSI channel dimension does not match PatchSpec")
    values = (
        array[..., : spec.complex_values]
        + 1j * array[..., spec.complex_values :]
    ).reshape(*array.shape[:-1], spec.antennas, spec.subcarriers)
    spectrum = np.fft.fft2(values, axes=(-2, -1), norm="ortho")
    return (np.abs(spectrum) ** 2).reshape(*array.shape[:-1], spec.complex_values)


@dataclass(frozen=True)
class MaskQuery:
    mode: str
    mask: np.ndarray
    query: int


def frozen_mask_query_bank(spec: PatchSpec, seed: int) -> tuple[MaskQuery, ...]:
    """Build the three V6 mask families with every patch covered as a query."""
    if spec.patch_count % 4:
        raise ValueError("V6 random masks require exact 75% mask cardinality")
    rng = np.random.default_rng(int(seed))
    output: list[MaskQuery] = []
    for query in range(spec.patch_count):
        random_count = int(FROZEN_RANDOM_MASK_FRACTION * spec.patch_count)
        candidates = np.asarray([index for index in range(spec.patch_count) if index != query])
        selected = rng.choice(candidates, size=max(0, random_count - 1), replace=False)
        random_mask = np.zeros(spec.patch_count, dtype=np.bool_)
        random_mask[query] = True
        random_mask[selected] = True
        output.append(MaskQuery("random_75", random_mask, query))

        patch_row, patch_column = divmod(query, spec.patch_columns)
        selected_rows = _centered_axis_block(spec.patch_rows, patch_row, 0.5)
        antenna_mask = np.zeros(spec.patch_count, dtype=np.bool_)
        for patch in range(spec.patch_count):
            row, _ = divmod(patch, spec.patch_columns)
            if row in selected_rows:
                antenna_mask[patch] = True
        output.append(MaskQuery("antenna_block_50", antenna_mask, query))

        selected_columns = _centered_axis_block(spec.patch_columns, patch_column, 0.5)
        subcarrier_mask = np.zeros(spec.patch_count, dtype=np.bool_)
        for patch in range(spec.patch_count):
            _, column = divmod(patch, spec.patch_columns)
            if column in selected_columns:
                subcarrier_mask[patch] = True
        output.append(MaskQuery("subcarrier_block_50", subcarrier_mask, query))
    for entry in output:
        if not bool(entry.mask[entry.query]):
            raise AssertionError("every query must be masked")
    return tuple(output)


def _centered_axis_block(size: int, query_index: int, fraction: float) -> frozenset[int]:
    count = max(1, min(int(size), int(round(float(fraction) * int(size)))))
    start = min(max(int(query_index) - count // 2, 0), int(size) - count)
    return frozenset(range(start, start + count))


def apply_patch_mask(patches: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values = np.asarray(patches).copy()
    mask_array = np.asarray(masks, dtype=np.bool_)
    if values.shape[-2] != mask_array.shape[-1]:
        raise ValueError("mask patch axis does not match CSI patches")
    values[mask_array] = 0.0
    return values


def typed_signed_edit(
    source_maps: np.ndarray,
    target_maps: np.ndarray,
    channel_names: np.ndarray,
    material_categories: int,
) -> np.ndarray:
    """Encode add/remove, height +/- and categorical material from/to planes."""
    source = np.asarray(source_maps, dtype=np.float64)
    target = np.asarray(target_maps, dtype=np.float64)
    if source.shape != target.shape or source.ndim < 3:
        raise ValueError("source and target maps must share [..., channel, row, column]")
    names = [str(value) for value in np.asarray(channel_names).tolist()]
    occupancy_index = names.index("occupancy")
    height_index = names.index("height")
    material_index = names.index("material")
    occupancy_delta = target[..., occupancy_index, :, :] - source[..., occupancy_index, :, :]
    height_delta = target[..., height_index, :, :] - source[..., height_index, :, :]
    source_material = source[..., material_index, :, :]
    target_material = target[..., material_index, :, :]
    if not np.allclose(source_material, np.round(source_material)) or not np.allclose(
        target_material, np.round(target_material)
    ):
        raise ValueError("material maps must contain categorical integer values")
    if np.any(source_material < 0) or np.any(target_material < 0):
        raise ValueError("material categories must be nonnegative")
    if np.any(source_material >= material_categories) or np.any(target_material >= material_categories):
        raise ValueError("material category exceeds the frozen vocabulary")
    changed = source_material != target_material
    planes = [
        np.maximum(occupancy_delta, 0.0),
        np.maximum(-occupancy_delta, 0.0),
        np.maximum(height_delta, 0.0),
        np.maximum(-height_delta, 0.0),
    ]
    for category in range(material_categories):
        planes.append(changed * (source_material == category))
    for category in range(material_categories):
        planes.append(changed * (target_material == category))
    return np.stack(planes, axis=-3).astype(np.float64)


def headline_alignment_edge(dataset, scene, edge):
    """Exclude edits incident to the privileged natural-map branch."""
    natural = int(dataset.natural_world_index[int(scene)])
    return bool(
        int(edge.source_world) != natural and int(edge.target_world) != natural
    )


def zero_typed_edit(shape_prefix: tuple[int, ...], map_size: int, material_categories: int) -> np.ndarray:
    channels = 4 + 2 * int(material_categories)
    return np.zeros((*shape_prefix, channels, map_size, map_size), dtype=np.float64)
