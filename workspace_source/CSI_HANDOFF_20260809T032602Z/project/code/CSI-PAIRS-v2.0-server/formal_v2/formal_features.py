from __future__ import annotations

import numpy as np


def spatial_pyramid(array: np.ndarray, levels: tuple[int, ...] = (1, 2, 4)) -> np.ndarray:
    """Fixed-size signed pooling; it never uses scene, world, or variant IDs."""
    image = np.asarray(array, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("spatial_pyramid expects one square map")
    features: list[float] = []
    for level in levels:
        if level <= 0 or level > image.shape[0]:
            raise ValueError("invalid pyramid level")
        row_groups = np.array_split(np.arange(image.shape[0]), level)
        col_groups = np.array_split(np.arange(image.shape[1]), level)
        for rows in row_groups:
            for columns in col_groups:
                patch = image[np.ix_(rows, columns)]
                features.extend((float(np.mean(patch)), float(np.sqrt(np.mean(patch**2)))))
    return np.asarray(features, dtype=np.float64)


def multichannel_spatial_features(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("multichannel map/action must have shape [channel, row, column]")
    return np.concatenate(
        [np.concatenate((spatial_pyramid(channel), _gradient_summary(channel))) for channel in values]
    )


def map_pair_features(source_map: np.ndarray, target_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_map, dtype=np.float64)
    target = np.asarray(target_map, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source and target maps must have equal shapes")
    if source.ndim == 3:
        scale = max(float(np.max(np.abs(np.concatenate((source.ravel(), target.ravel()))))), 1.0)
        source_normalized = source / scale
        signed_edit = (target - source) / scale
        return multichannel_spatial_features(source_normalized), multichannel_spatial_features(signed_edit)
    scale = max(float(np.max(np.abs(np.concatenate((source.ravel(), target.ravel()))))), 1.0)
    source_normalized = source / scale
    signed_edit = (target - source) / scale
    source_features = np.concatenate(
        (spatial_pyramid(source_normalized), _gradient_summary(source_normalized))
    )
    edit_features = np.concatenate(
        (
            spatial_pyramid(signed_edit),
            _gradient_summary(signed_edit),
            np.asarray(
                [
                    np.mean(signed_edit),
                    np.mean(np.abs(signed_edit)),
                    np.max(signed_edit),
                    np.min(signed_edit),
                    np.mean(signed_edit > 0),
                    np.mean(signed_edit < 0),
                ],
                dtype=np.float64,
            ),
        )
    )
    return source_features, edit_features


def protocol_response_features(
    visible_patches: np.ndarray,
    source_map: np.ndarray,
    typed_action: np.ndarray,
    radio_config: np.ndarray,
    mask: np.ndarray,
    query: int,
    *,
    position: np.ndarray | None = None,
    include_action: bool = True,
    include_csi: bool = True,
    include_map: bool = True,
) -> np.ndarray:
    """Fixed diagnostic features with the same V6 input allowlist as F/P."""
    patches = np.asarray(visible_patches, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=np.bool_)
    if patches.ndim != 2 or mask_array.shape != (patches.shape[0],):
        raise ValueError("visible patches/mask have incompatible shapes")
    blocks: list[np.ndarray] = []
    if include_csi:
        blocks.append(patches.ravel())
    if include_map:
        blocks.append(multichannel_spatial_features(np.asarray(source_map, dtype=np.float64)))
    if include_action:
        blocks.append(multichannel_spatial_features(np.asarray(typed_action, dtype=np.float64)))
    blocks.append(np.asarray(radio_config, dtype=np.float64).ravel())
    blocks.append(mask_array.astype(np.float64))
    query_one_hot = np.zeros(patches.shape[0], dtype=np.float64)
    query_one_hot[int(query)] = 1.0
    blocks.append(query_one_hot)
    if position is not None:
        blocks.append(np.asarray(position, dtype=np.float64).ravel())
    return np.concatenate(blocks)


def response_features(
    source_csi: np.ndarray,
    source_map: np.ndarray,
    target_map: np.ndarray,
    position: np.ndarray | None = None,
    include_action: bool = True,
    include_csi: bool = True,
) -> np.ndarray:
    source_map_features, edit_features = map_pair_features(source_map, target_map)
    csi = np.asarray(source_csi, dtype=np.float64).ravel()
    blocks: list[np.ndarray] = [source_map_features]
    if include_csi:
        blocks.insert(0, csi)
    if include_action:
        blocks.append(edit_features)
        if include_csi:
            # Low-rank multiplicative terms let the response depend on the source
            # channel without exposing receiver coordinates.
            csi_summary = _chunk_summary(csi, 8)
            edit_summary = _chunk_summary(edit_features, 8)
            blocks.append(np.outer(csi_summary, edit_summary).ravel())
    if position is not None:
        coordinate = np.asarray(position, dtype=np.float64).ravel()
        blocks.append(coordinate)
        if include_action:
            blocks.append(np.outer(coordinate, _chunk_summary(edit_features, 8)).ravel())
    return np.concatenate(blocks)


def variant_features(source_bits: np.ndarray, target_bits: np.ndarray) -> np.ndarray:
    source = np.asarray(source_bits, dtype=np.float64).ravel()
    target = np.asarray(target_bits, dtype=np.float64).ravel()
    return np.concatenate((source, target, target - source, source * target))


def _gradient_summary(image: np.ndarray) -> np.ndarray:
    row_gradient, column_gradient = np.gradient(image)
    return np.asarray(
        [
            np.mean(np.abs(row_gradient)),
            np.mean(np.abs(column_gradient)),
            np.sqrt(np.mean(row_gradient**2)),
            np.sqrt(np.mean(column_gradient**2)),
        ],
        dtype=np.float64,
    )


def _chunk_summary(vector: np.ndarray, chunks: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).ravel()
    groups = np.array_split(array, min(chunks, array.size))
    values = np.asarray([np.mean(group) for group in groups], dtype=np.float64)
    if values.size < chunks:
        values = np.pad(values, (0, chunks - values.size))
    return values
