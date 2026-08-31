# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import pytest
import torch
from gatr import GATr, MLPConfig, SelfAttentionConfig
from torch.nn.functional import one_hot
from torch_geometric.data import Batch, Data

from wigatr.models.regression_gatr import RSRPRegressionGATr
from wigatr.utils.misc import make_full_edge_index
from wigatr.utils.tensors import one_hot_vector

_DEVICE = torch.device("cuda")
_NUM_MESH_FACES = 5
_NUM_TOKENS = 3 + _NUM_MESH_FACES
_NUM_MATERIALS = 6


@pytest.fixture
def inputs():
    x = torch.randn(_NUM_TOKENS, 3, 3, device=_DEVICE)
    types = torch.cat(
        [
            one_hot_vector(4, 0, batch_dims=(1,)).to(_DEVICE),
            one_hot_vector(4, 1, batch_dims=(1,)).to(_DEVICE),
            one_hot_vector(4, 2, batch_dims=(1,)).to(_DEVICE),
            one_hot_vector(4, 3, batch_dims=(_NUM_MESH_FACES,)).to(_DEVICE),
        ],
        dim=0,
    )
    materials = torch.cat(
        [
            torch.zeros(3, _NUM_MATERIALS, device=_DEVICE),
            one_hot(
                torch.randint(0, _NUM_MATERIALS, (_NUM_MESH_FACES,), device=_DEVICE), _NUM_MATERIALS
            ),
        ],
        dim=0,
    )
    edge_index = make_full_edge_index(num_nodes=len(x), self_loops=True, device=x.device)
    data = Data(x=x, types=types, materials=materials, edge_index=edge_index)
    batch = Batch.from_data_list([data])
    return batch


def make_gatr():
    gatr = GATr(
        in_mv_channels=15,
        out_mv_channels=1,
        hidden_mv_channels=8,
        in_s_channels=10,
        out_s_channels=1,
        hidden_s_channels=16,
        attention=SelfAttentionConfig(
            num_heads=2,
            increase_hidden_channels=2,
            multi_query=True,
        ),
        num_blocks=2,
        mlp=MLPConfig(),
    )
    wrapped_gatr = RSRPRegressionGATr(gatr).to(_DEVICE)
    return wrapped_gatr


def translation(batch):
    outputs = batch.clone()
    translation = torch.Tensor([10.0, 0.0, -100.0]).to(outputs.x.device)
    outputs.x = outputs.x + translation
    return outputs


def rotation(batch):
    outputs = batch.clone()
    outputs.x = torch.cat([-outputs.x[..., [1]], outputs.x[..., [0]], outputs.x[..., [2]]], dim=-1)
    return outputs


@pytest.mark.parametrize("transformation", [translation, rotation])
def test_invariance(transformation, inputs):
    model = make_gatr()
    outputs = model(inputs)
    trf_inputs = transformation(inputs)
    trf_outputs = model(trf_inputs)

    torch.testing.assert_close(outputs, trf_outputs)
