# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
This file provides some helper functions for transformations, e.g., for data augmentation,
and canonicalization.
"""
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def random_e2_transform(*tensors: Tensor, translation_scale: float = 1.0) -> Sequence[Tensor]:
    """Applies a random E2 transform to a 3D point cloud.

    The same E(2) transform is applied to the first two components of the last axis of all input
    arguments.

    The rotation is sampled uniformly, the translation from a isotropic Gaussian with std
    `translation_scale`.

    Parameters
    ----------
    tensors: torch.Tensor (each argument)
        A number of tensors with shape (..., 3). All should have the same device and dtype.
    translation_scale: float
        Scale for translation.

    Returns
    -------
    transformed_tensors: Sequence[torch.Tensor]
        input tensors with E(2) element applied to the first two outputs.
    """

    if not tensors:
        return tuple()

    device = tensors[0].device
    dtype = tensors[0].dtype

    phi = 2.0 * np.pi * np.random.rand()
    c, s = np.cos(phi), np.sin(phi)
    rot = torch.tensor([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    if translation_scale > 0.0:
        transl = torch.randn(3, device=device, dtype=dtype)
        transl[2] = 0.0  # Only transform with E(2), not E(3)
    else:
        transl = torch.zeros(3, device=device, dtype=dtype)

    return tuple((t @ rot + transl) for t in tensors)


def canonicalize_e2(reference: Tensor, *tensors: Tensor) -> Sequence[Tensor]:
    """Canonicalizes a 3D point cloud in the x-y plane.

    All tensors are shifted and rotated such that a reference tensor has center of mass at
    x,y=0, and rotated such that the largest principal direction is aligned with the x axis.

    Parameters
    ----------
    reference: torch.Tensor
        reference point cloud used to determine the E(2) canonicalization.
    tensors: torch.Tensor
        all tensors that should be transformed. (Note that if `reference` should be
        transformed as well, it should be included here again.) All should have the same device
        and dtype as `reference`.

    Returns
    -------
    transformed_tensors: Sequence[torch.Tensor]
        canonicalized `tensors`.
    """

    if not tensors:
        return tuple()

    device = reference.device
    dtype = reference.dtype

    # Shift center of mass to 0
    transl = -torch.mean(reference.view(-1, 3), dim=0)
    transl[2] = 0.0  # Only canonicalize in E(2), not E(3)

    # Rotate principal direction to x axis
    _, s, v = torch.pca_lowrank(reference.view(-1, 3) + transl, center=False)
    principal_dir = v[:, 0]  # (3,)
    if principal_dir[2] < 0:
        # Tie-break residual Z2 symmetry
        principal_dir = -principal_dir
    phi = torch.atan2(principal_dir[1], principal_dir[0])  # princiipal dir in polar coordinates
    c, s = torch.cos(phi), torch.sin(phi)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)

    return tuple((t + transl) @ rot for t in tensors)
