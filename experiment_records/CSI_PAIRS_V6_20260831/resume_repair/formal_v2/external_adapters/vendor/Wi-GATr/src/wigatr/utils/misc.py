# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Miscellaneous utility functions.
"""
from collections.abc import Mapping
from functools import lru_cache
from itertools import product

import torch
import xformers.ops.fmha
from torch_geometric.data import Batch, Data


def frequency_check(step, every_n_steps, skip_initial=False, include_fractional=None):
    """
    For an action that should be performed every `every_n_steps` steps and a step number, returns
    whether the action should be performed.

    If `include_fractional` is given, the check also returns True when
    `step == round(every_n_steps * fraction)` for each `fraction` in `include_fractional`.


    Parameters
    ----------
    step : int
        Step number (one-indexed)
    every_n_steps : None or int
        Desired action frequency. None or 0 correspond to never executing the action.
    skip_initial : bool
        If True, frequency_check returns False at step 0.
    include_fractional : None or tuple of float
        If not None, the check also returns True when `step == round(every_n_steps * fraction)`
        for each `fraction` in `include_fractional`.

    Returns
    -------
    decision : bool
        Whether the action should be executed.
    """

    if every_n_steps is None or every_n_steps == 0:
        return False

    if skip_initial and step == 0:
        return False

    if include_fractional is not None:
        for fraction in include_fractional:
            if step == int(round(fraction * every_n_steps)):
                return True

    return step % every_n_steps == 0


class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights"""


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@lru_cache()
@torch.no_grad()
def make_full_edge_index(num_nodes, batchsize=1, self_loops=False, device=torch.device("cpu")):
    """Creates a PyG-style edge index for a fully connected graph of `num_nodes` nodes"""

    # Construct fully connected edge index
    src, dst = [], []
    for i, j in product(range(num_nodes), repeat=2):
        if not self_loops and i == j:
            continue
        src.append(i)
        dst.append(j)

    edge_index_per_batch = torch.LongTensor([src, dst]).to(device)

    # Repeat for each batch element
    if batchsize > 1:
        edge_index = [edge_index_per_batch + k * num_nodes for k in range(batchsize)]
        edge_index = torch.cat(edge_index, dim=1)
    else:
        edge_index = edge_index_per_batch

    return edge_index


def build_pyg_attention_mask(inputs, causal=False, multiply_batch_sizes=1):
    """Construct attention mask from pytorch geometric batch."""
    block_sizes = multiply_batch_sizes * torch.bincount(inputs.batch).tolist()
    if causal:
        return xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(block_sizes)
    return xformers.ops.fmha.BlockDiagonalMask.from_seqlens(block_sizes)


def sum_over_sample_in_pyg_batch(graph_batch, nodewise_data):
    """Given a PyG batch, and a tensor with batch * node-wise data, performs sample-wise summing"""
    batchsize = max(graph_batch.batch) + 1
    feature_dims = nodewise_data.shape[1:]
    aggregated_outputs = torch.zeros(batchsize, *feature_dims, device=nodewise_data.device)
    aggregated_outputs.index_add_(0, graph_batch.batch, nodewise_data)  # (batchsize,)
    return aggregated_outputs


def get_batchsize(data):
    """Given either a tensor or a list of tensors or a dict of tensors, returns the batchsize"""

    if isinstance(data, Mapping):
        assert len(data) > 0
        tensor = next(iter(data.values()))
        return tensor.shape[0]
    if isinstance(data, (tuple, list)):
        assert len(data) > 0
        return get_batchsize(data[0])
    if isinstance(data, Batch):
        return (
            data.num_graphs
        )  # equivalent to batch_size property, which does not exist in older versions
    if isinstance(data, Data):
        return 1

    return data.shape[0]
