from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formal_protocol import PatchSpec, patchify_csi


SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class ReflectionPath:
    """First-order path implied by one registered rectangular primitive."""

    basis: np.ndarray
    features: np.ndarray


def complex_csi(csi: np.ndarray) -> np.ndarray:
    values = np.asarray(csi, dtype=np.float64)
    if values.shape[-1] % 2:
        raise ValueError("real/imaginary CSI must have an even channel dimension")
    complex_values = values.shape[-1] // 2
    return values[..., :complex_values] + 1j * values[..., complex_values:]


def real_csi(csi: np.ndarray) -> np.ndarray:
    values = np.asarray(csi, dtype=np.complex128)
    return np.concatenate((values.real, values.imag), axis=-1)


def action_rectangle_xy(
    source_map: np.ndarray,
    target_map: np.ndarray,
    *,
    origin_xy_m: np.ndarray,
    resolution_m: float,
) -> tuple[float, float, float, float]:
    """Recover the exact axis-aligned registered primitive from its map edit."""
    source = np.asarray(source_map)
    target = np.asarray(target_map)
    if source.shape != target.shape or source.ndim != 3 or source.shape[1] != source.shape[2]:
        raise ValueError("source and target maps must have equal [channel,row,column] shape")
    changed = np.any(source != target, axis=0)
    rows, columns = np.nonzero(changed)
    if not rows.size:
        raise ValueError("typed action does not change any map cell")
    if np.any(~changed[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]):
        raise ValueError("registered physical response probe requires a rectangular action")
    origin = np.asarray(origin_xy_m, dtype=np.float64)
    if origin.shape != (2,) or float(resolution_m) <= 0.0:
        raise ValueError("map origin/resolution is malformed")
    resolution = float(resolution_m)
    return (
        float(origin[0] + columns.min() * resolution),
        float(origin[0] + (columns.max() + 1) * resolution),
        float(origin[1] + rows.min() * resolution),
        float(origin[1] + (rows.max() + 1) * resolution),
    )


def material_direction(source_map: np.ndarray, target_map: np.ndarray) -> tuple[int, int]:
    changed = np.any(np.asarray(source_map) != np.asarray(target_map), axis=0)
    if not np.any(changed):
        raise ValueError("material direction requires a nonzero action")
    source_values = np.unique(np.asarray(source_map)[2][changed])
    target_values = np.unique(np.asarray(target_map)[2][changed])
    if source_values.size != 1 or target_values.size != 1:
        raise ValueError("registered action must have one source and target material")
    return int(source_values[0]), int(target_values[0])


def reflection_path(
    rectangle_xy_m: tuple[float, float, float, float],
    receiver_xy_m: np.ndarray,
    *,
    primitive_height_m: float,
    transmitter_z_m: float,
    receiver_z_m: float,
    carrier_frequency_hz: float,
    subcarrier_spacing_hz: float,
    antennas: int,
    subcarriers: int,
    minimum_incidence_cosine: float,
) -> ReflectionPath | None:
    """Construct the path shape without using an RT response or route label."""
    x_min, x_max, y_min, y_max = map(float, rectangle_xy_m)
    receiver = np.asarray(receiver_xy_m, dtype=np.float64)
    if receiver.shape != (2,) or not (x_min < x_max and y_min < y_max):
        raise ValueError("reflection geometry is malformed")
    corners = (
        (np.asarray((x_min, y_min)), np.asarray((x_max, y_min))),
        (np.asarray((x_max, y_min)), np.asarray((x_max, y_max))),
        (np.asarray((x_max, y_max)), np.asarray((x_min, y_max))),
        (np.asarray((x_min, y_max)), np.asarray((x_min, y_min))),
    )
    candidates = []
    for first, second in corners:
        direction = second - first
        squared_length = float(direction @ direction)
        normal = np.asarray((-direction[1], direction[0])) / np.sqrt(squared_length)
        transmitter_side = float((-first) @ normal)
        receiver_side = float((receiver - first) @ normal)
        if transmitter_side >= -1e-6 or receiver_side >= -1e-6:
            continue
        image = -2.0 * transmitter_side * normal
        denominator = float((receiver - image) @ normal)
        if abs(denominator) <= 1e-9:
            continue
        fraction = -float((image - first) @ normal) / denominator
        reflection = image + fraction * (receiver - image)
        wall_fraction = float((reflection - first) @ direction / squared_length)
        reflection_z = float(transmitter_z_m) + fraction * (
            float(receiver_z_m) - float(transmitter_z_m)
        )
        if not (
            1e-6 < fraction < 1.0 - 1e-6
            and 1e-5 < wall_fraction < 1.0 - 1e-5
            and 1e-3 < reflection_z < float(primitive_height_m) - 1e-3
        ):
            continue
        transmitter_leg = float(
            np.sqrt(np.sum(reflection**2) + (reflection_z - float(transmitter_z_m)) ** 2)
        )
        receiver_leg = float(
            np.sqrt(
                np.sum((receiver - reflection) ** 2)
                + (float(receiver_z_m) - reflection_z) ** 2
            )
        )
        incidence = abs(float((reflection / np.linalg.norm(reflection)) @ normal))
        if incidence < float(minimum_incidence_cosine):
            continue
        candidates.append(
            (
                incidence / (np.linalg.norm(reflection) + np.linalg.norm(receiver - reflection)),
                incidence,
                transmitter_leg + receiver_leg,
                reflection,
                reflection_z,
                transmitter_leg,
                fraction,
                wall_fraction,
            )
        )
    if not candidates:
        return None
    _, incidence, path_length, point, point_z, transmitter_leg, fraction, wall_fraction = max(
        candidates, key=lambda row: row[0]
    )
    direct_length = float(
        np.sqrt(
            np.sum(receiver**2)
            + (float(receiver_z_m) - float(transmitter_z_m)) ** 2
        )
    )
    departure = np.asarray(
        (point[0], point[1], point_z - float(transmitter_z_m)), dtype=np.float64
    ) / transmitter_leg
    frequencies = (
        np.arange(-int(subcarriers) // 2, int(subcarriers) // 2, dtype=np.float64)
        * float(subcarrier_spacing_hz)
    )
    if frequencies.size != int(subcarriers):
        raise ValueError("physics response currently requires an even subcarrier count")
    antenna_phase = np.exp(1j * np.pi * np.arange(int(antennas)) * departure[1])
    frequency_phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequencies
        * (path_length - direct_length)
        / SPEED_OF_LIGHT_M_S
    )
    basis = antenna_phase[:, None] * frequency_phase[None, :]
    features = np.asarray(
        (
            incidence,
            fraction,
            wall_fraction,
            path_length / 256.0,
            direct_length / 256.0,
            departure[0],
            departure[1],
        ),
        dtype=np.float64,
    )
    return ReflectionPath(basis=basis, features=features)


def direct_path_basis(
    receiver_xy_m: np.ndarray,
    *,
    transmitter_z_m: float,
    receiver_z_m: float,
    antennas: int,
    subcarriers: int,
) -> np.ndarray:
    receiver = np.asarray(receiver_xy_m, dtype=np.float64)
    direction = np.asarray(
        (receiver[0], receiver[1], float(receiver_z_m) - float(transmitter_z_m))
    )
    direction /= np.linalg.norm(direction)
    antenna_phase = np.exp(1j * np.pi * np.arange(int(antennas)) * direction[1])
    return antenna_phase[:, None] * np.ones((1, int(subcarriers)), dtype=np.complex128)


def visible_complex_mask(mask: np.ndarray, spec: PatchSpec) -> np.ndarray:
    patch_mask = np.asarray(mask, dtype=np.bool_)
    if patch_mask.shape != (spec.patch_count,):
        raise ValueError("patch mask shape differs from PatchSpec")
    visible = np.zeros((spec.antennas, spec.subcarriers), dtype=np.bool_)
    for patch in np.flatnonzero(~patch_mask):
        row, column = divmod(int(patch), spec.patch_columns)
        antenna_start = row * int(spec.patch_antenna_size)
        carrier_start = column * int(spec.patch_subcarrier_size)
        visible[
            antenna_start : antenna_start + int(spec.patch_antenna_size),
            carrier_start : carrier_start + int(spec.patch_subcarrier_size),
        ] = True
    return visible


def decompose_visible_csi(
    csi: np.ndarray,
    bases: list[np.ndarray],
    visible: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    values = np.asarray(csi, dtype=np.complex128)
    keep = np.asarray(visible, dtype=np.bool_)
    if not bases or values.shape != keep.shape or any(np.asarray(b).shape != values.shape for b in bases):
        raise ValueError("CSI, bases, and visible mask shapes differ")
    design = np.stack([np.asarray(value).reshape(-1) for value in bases], axis=1)
    design = design[keep.reshape(-1)]
    target = values.reshape(-1)[keep.reshape(-1)]
    gram = design.conj().T @ design
    return np.linalg.solve(
        gram + float(ridge) * np.eye(gram.shape[0], dtype=np.complex128),
        design.conj().T @ target,
    )


def normalized_patch_delta(
    complex_delta: np.ndarray,
    patch_scale: np.ndarray,
    spec: PatchSpec,
) -> np.ndarray:
    patches = patchify_csi(real_csi(np.asarray(complex_delta).reshape(-1)), spec)
    scale = np.asarray(patch_scale, dtype=np.float64)
    if patches.shape != scale.shape:
        raise ValueError("patch scale differs from physical response shape")
    return patches / scale
