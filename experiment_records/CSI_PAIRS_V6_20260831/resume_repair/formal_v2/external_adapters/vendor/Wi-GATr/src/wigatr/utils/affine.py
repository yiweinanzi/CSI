# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Affine transformation module that scales the output of a wrapped network.
"""
import torch
from torch import nn


class AffineTransformation(nn.Module):
    """
    Affine transformation module that shifts and scales input given fixed parameters during
    `forward`.
    """

    def __init__(self, shift, scale):
        """
        Initialize affine transformation module. Sets shift and scale parameters.

        Parameters
        ----------
        shift : float
            Shift parameter.
        scale : float
            Scale parameter.
        """
        super().__init__()
        self.register_buffer("shift", torch.tensor(shift))
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        """
        Apply affine transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return self.scale * x + self.shift
