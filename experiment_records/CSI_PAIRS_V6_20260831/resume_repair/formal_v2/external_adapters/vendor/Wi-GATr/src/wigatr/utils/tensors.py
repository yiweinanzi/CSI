# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Several helper functions for tensor manipulation and checking.
"""
from typing import Tuple, Union

import torch

from wigatr.utils.logger import logger


def pad_to_shape(
    inputs: torch.Tensor, target_shape: tuple, dim: Union[None, int, Tuple[int]] = None
) -> torch.Tensor:
    """Zero-pads a tensor to a given target shape in selected dimensions"""

    # Inputs
    assert len(inputs.shape) == len(target_shape)

    if dim is None:
        dim = list(range(len(inputs.shape)))
    if isinstance(dim, int):
        dim = [dim]
    dim = set(dim)  # Avoid duplicates

    # Loop over axes
    for i in dim:
        current = inputs.shape[i]
        target_ = target_shape[i]

        assert target_ >= current
        if target_ == current:
            continue

        # Construct padding
        dummy_size = list(inputs.shape)
        dummy_size[i] = target_ - current
        dummy = torch.zeros(tuple(dummy_size), device=inputs.device, dtype=inputs.dtype)

        # Concatenate with padding
        inputs = torch.cat((inputs, dummy), dim=i)

    return inputs


@torch.no_grad()
def one_hot_vector(dim, i, batch_dims=tuple()):
    """
    Makes a one-hot vector of length dim, with a one at index i, and optionally batch dimensions
    """
    x = torch.zeros(*batch_dims, dim)
    x[..., i] = 1.0
    return x


def check_sane(tensor, desc, max_value=1e9):
    """
    Checks tensor sanity.
    """
    sane = torch.all(torch.isfinite(tensor)) and torch.all(torch.abs(tensor) < max_value)
    if not sane:
        error_msg = (
            f"Data not sane: {desc}. dtype {tensor.dtype}, device {tensor.device}, "
            f"shape {tensor.shape}, data:\n{tensor}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
