# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Utility functions for experiments.
"""
import torch


def mean_relative_absolute_error(pred, target):
    """Computes the mean relative absolute error between prediction and targets."""
    rae = torch.abs(pred - target) / target
    return torch.mean(rae)
