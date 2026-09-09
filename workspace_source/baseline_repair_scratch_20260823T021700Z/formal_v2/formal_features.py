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
        [
            np.concatenate(
                (
                    spatial_pyramid(channel),
                    _gradient_summary(channel),
                    _coordinate_summary(channel),
                )
            )
            for channel in values
        ]
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
    map_features = (
        multichannel_spatial_features(np.asarray(source_map, dtype=np.float64))
        if include_map
        else None
    )
    action_features = (
        multichannel_spatial_features(np.asarray(typed_action, dtype=np.float64))
        if include_action
        else None
    )
    return assemble_protocol_response_features(
        visible_patches,
        map_features,
        action_features,
        radio_config,
        mask,
        query,
        position=position,
        include_csi=include_csi,
    )


def assemble_protocol_response_features(
    visible_patches: np.ndarray,
    map_features: np.ndarray | None,
    action_features: np.ndarray | None,
    radio_config: np.ndarray,
    mask: np.ndarray,
    query: int,
    *,
    position: np.ndarray | None = None,
    include_csi: bool = True,
) -> np.ndarray:
    """Assemble the diagnostic vector from already computed spatial blocks."""
    patches = np.asarray(visible_patches, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=np.bool_)
    if patches.ndim != 2 or mask_array.shape != (patches.shape[0],):
        raise ValueError("visible patches/mask have incompatible shapes")
    blocks: list[np.ndarray] = []
    csi_summary = None
    if include_csi:
        blocks.append(patches.ravel())
        csi_summary = _chunk_summary(patches.ravel(), 8)
    if map_features is not None:
        blocks.append(np.asarray(map_features, dtype=np.float64).ravel())
    if action_features is not None:
        action = np.asarray(action_features, dtype=np.float64).ravel()
        blocks.append(action)
    blocks.append(np.asarray(radio_config, dtype=np.float64).ravel())
    blocks.append(mask_array.astype(np.float64))
    query_one_hot = np.zeros(patches.shape[0], dtype=np.float64)
    query_one_hot[int(query)] = 1.0
    blocks.append(query_one_hot)
    map_summary = (
        _chunk_summary(np.asarray(map_features, dtype=np.float64), 8)
        if map_features is not None
        else None
    )
    action_summary = (
        _chunk_summary(np.asarray(action_features, dtype=np.float64), 8)
        if action_features is not None
        else None
    )
    if csi_summary is not None:
        blocks.append(np.outer(query_one_hot, csi_summary).ravel())
    if action_summary is not None:
        blocks.append(np.outer(query_one_hot, action_summary).ravel())
    if csi_summary is not None and map_summary is not None:
        blocks.append(np.outer(csi_summary, map_summary).ravel())
    if csi_summary is not None and action_summary is not None:
        blocks.append(np.outer(csi_summary, action_summary).ravel())
    if map_summary is not None and action_summary is not None:
        blocks.append(np.outer(map_summary, action_summary).ravel())
    if position is not None:
        coordinate = np.asarray(position, dtype=np.float64).ravel()
        blocks.append(coordinate)
        if action_features is not None:
            blocks.append(np.outer(coordinate, _chunk_summary(action, 8)).ravel())
    return np.concatenate(blocks)


def assemble_protocol_response_feature_matrix(
    visible_patch_rows: np.ndarray,
    map_features: np.ndarray | None,
    action_features: np.ndarray | None,
    radio_config: np.ndarray,
    masks: np.ndarray,
    queries: np.ndarray,
    *,
    position: np.ndarray | None = None,
    include_csi: bool = True,
) -> np.ndarray:
    """Vectorized equivalent of assemble_protocol_response_features."""
    patches = np.asarray(visible_patch_rows, dtype=np.float64)
    mask_rows = np.asarray(masks, dtype=np.bool_)
    query_rows = np.asarray(queries, dtype=np.int64).reshape(-1)
    if patches.ndim != 3:
        raise ValueError("visible patch rows must have shape [batch,patch,value]")
    if mask_rows.shape != patches.shape[:2] or query_rows.shape != (patches.shape[0],):
        raise ValueError("batched visible patches, masks, and queries are incompatible")
    if np.any(query_rows < 0) or np.any(query_rows >= patches.shape[1]):
        raise ValueError("batched response query is out of range")
    count = patches.shape[0]
    one_hot = np.zeros((count, patches.shape[1]), dtype=np.float64)
    one_hot[np.arange(count), query_rows] = 1.0
    blocks: list[np.ndarray] = []
    csi_summary = None
    if include_csi:
        flattened = patches.reshape(count, -1)
        blocks.append(flattened)
        csi_summary = np.vstack(
            [_chunk_summary(row, 8) for row in flattened]
        )
    map_vector = (
        np.asarray(map_features, dtype=np.float64).ravel()
        if map_features is not None
        else None
    )
    action_vector = (
        np.asarray(action_features, dtype=np.float64).ravel()
        if action_features is not None
        else None
    )
    if map_vector is not None:
        blocks.append(np.broadcast_to(map_vector, (count, map_vector.size)))
    if action_vector is not None:
        blocks.append(np.broadcast_to(action_vector, (count, action_vector.size)))
    radio = np.asarray(radio_config, dtype=np.float64).ravel()
    blocks.append(np.broadcast_to(radio, (count, radio.size)))
    blocks.append(mask_rows.astype(np.float64))
    blocks.append(one_hot)
    map_summary = _chunk_summary(map_vector, 8) if map_vector is not None else None
    action_summary = (
        _chunk_summary(action_vector, 8) if action_vector is not None else None
    )
    if csi_summary is not None:
        blocks.append((one_hot[:, :, None] * csi_summary[:, None, :]).reshape(count, -1))
    if action_summary is not None:
        blocks.append(
            (
                one_hot[:, :, None]
                * np.broadcast_to(action_summary, (count, 1, action_summary.size))
            ).reshape(count, -1)
        )
    if csi_summary is not None and map_summary is not None:
        blocks.append((csi_summary[:, :, None] * map_summary[None, None, :]).reshape(count, -1))
    if csi_summary is not None and action_summary is not None:
        blocks.append(
            (csi_summary[:, :, None] * action_summary[None, None, :]).reshape(count, -1)
        )
    if map_summary is not None and action_summary is not None:
        interaction = np.outer(map_summary, action_summary).ravel()
        blocks.append(np.broadcast_to(interaction, (count, interaction.size)))
    if position is not None:
        coordinate = np.asarray(position, dtype=np.float64).ravel()
        blocks.append(np.broadcast_to(coordinate, (count, coordinate.size)))
        if action_summary is not None:
            coordinate_action = np.outer(coordinate, action_summary).ravel()
            blocks.append(
                np.broadcast_to(coordinate_action, (count, coordinate_action.size))
            )
    return np.column_stack(blocks)


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


def _coordinate_summary(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    rows, columns = values.shape
    row_axis = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    column_axis = np.linspace(-1.0, 1.0, columns, dtype=np.float64)
    yy, xx = np.meshgrid(row_axis, column_axis, indexing="ij")
    weight = np.abs(values)
    support = weight > 1.0e-12
    mass = float(np.sum(weight))
    if mass <= 0.0:
        return np.zeros(13, dtype=np.float64)
    center_x = float(np.sum(weight * xx) / mass)
    center_y = float(np.sum(weight * yy) / mass)
    support_rows, support_columns = np.nonzero(support)
    return np.asarray(
        (
            float(np.mean(support)),
            float(np.log1p(mass)),
            float(np.sum(values) / mass),
            float(np.mean(weight[support])),
            float(np.max(weight[support])),
            center_x,
            center_y,
            float(np.sqrt(np.sum(weight * (xx - center_x) ** 2) / mass)),
            float(np.sqrt(np.sum(weight * (yy - center_y) ** 2) / mass)),
            float(column_axis[int(support_columns.min())]),
            float(column_axis[int(support_columns.max())]),
            float(row_axis[int(support_rows.min())]),
            float(row_axis[int(support_rows.max())]),
        ),
        dtype=np.float64,
    )


def _chunk_summary(vector: np.ndarray, chunks: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).ravel()
    groups = np.array_split(array, min(chunks, array.size))
    values = np.asarray([np.mean(group) for group in groups], dtype=np.float64)
    if values.size < chunks:
        values = np.pad(values, (0, chunks - values.size))
    return values
