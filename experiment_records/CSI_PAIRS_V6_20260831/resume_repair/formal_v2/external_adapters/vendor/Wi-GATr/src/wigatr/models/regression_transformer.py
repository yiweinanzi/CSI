# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Wraps around a Transformer for predicting a scalar target variable in the wireless channel
modelling settings.
"""
from typing import Literal, Mapping, Optional

import torch
from torch.nn import Module

from wigatr.models.regression_gatr import fill_in_override_defaults
from wigatr.utils.affine import AffineTransformation
from wigatr.utils.misc import build_pyg_attention_mask, sum_over_sample_in_pyg_batch


# pylint: disable=duplicate-code
class RSRPRegressionTransformer(Module):
    """
    Wraps around a Transformer for predicting the total received power in wireless channel
    modelling settings.

    Requires Transformer model with:
    - 27 input channels
    - 1 output channel (which will be used to read off the power)

    Parameters
    ----------
    net : Transformer model, e.g., gatr.baselines.transformer.BaselineTransformer
    affine_shift : None or float
        If not None, sets shift applied to total received power outputs.
    affine_scale : None or float
        If not None, sets scale applied to total received power outputs.
    embedding : "classic" or "kitchen_sink"
        Which embedding scheme to use for the mesh faces. "kitchen_sink" includes more features.
    """

    data_mode = "geometry"

    def __init__(
        self,
        net: Module,
        affine_shift: Optional[float] = None,
        affine_scale: Optional[float] = None,
        embedding: Literal["classic", "kitchen_sink"] = "classic",
    ):
        super().__init__()
        assert embedding in ["classic", "kitchen_sink"]
        self.embedding = embedding
        self.net = net
        if affine_shift is not None and affine_scale is not None:
            self.affine = AffineTransformation(shift=affine_shift, scale=affine_scale)
        else:
            self.affine = torch.nn.Identity()

    def forward(self, inputs: torch.Tensor, overrides: Optional[Mapping[str, torch.Tensor]] = None):
        """
        Forward pass.

        Parses inputs into GA + scalar representation, calls the forward pass of the wrapped net,
        and extracts the outputs from the GA + scalar representation again.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.
        overrides : None or dict
            Optional overrides for the embedding, for inverse problems.

        Returns
        -------
        outputs : torch.Tensor
            Total received power in dB.
        """

        input_embedding = self.embed(inputs, overrides=overrides)
        mask = build_pyg_attention_mask(inputs)
        output_embedding = self.net(input_embedding, attention_mask=mask)
        outputs = self.extract(output_embedding, inputs)

        return outputs

    def embed(self, inputs, overrides=None):
        """Embeds data into transformer representation"""

        overrides = fill_in_override_defaults(overrides)

        # The dataset stores the types of tokens in `inputs.types`, and all other information (in
        # particular positions) in `inputs.x`.
        # Depending on the type, the information will be embedded differently.
        # See `self._embed_*()` for additional information.
        link_embedding = inputs.types[..., [0]] * self._embed_link(
            inputs.x, overrides["tx"], overrides["rx"]
        )  # Link object: 9 channels
        tx_embedding = inputs.types[..., [1]] * self._embed_tx_rx(
            inputs.x, overrides["tx"]
        )  # Tx object: 3 channels
        rx_embedding = inputs.types[..., [2]] * self._embed_tx_rx(
            inputs.x, overrides["rx"]
        )  # Rx object: 3 channels
        mesh_embedding = inputs.types[..., [3]] * self._embed_mesh_face(
            inputs.x
        )  # Mesh face object: 24 channels

        # Concatenate along channels
        embedding = torch.cat(
            [link_embedding, tx_embedding, rx_embedding, mesh_embedding], dim=-1
        )  # 39 channels

        # Add material info, if available
        if "materials" in inputs:
            embedding = torch.cat(
                [
                    embedding,  # (batchsize * tokens, 39)
                    inputs.materials,  # (batchsize * tokens, 6)
                ],
                dim=1,
            )  # (batchsize * tokens, 45)

        return embedding

    def extract(self, output_embedding, inputs):
        """
        Extracts power prediction from transformer representation.

        We extract the total received power in dB from the single scalar output channel associated
        with the Tx-Rx link object.
        """

        # Check channels of inputs. Batchsize and object numbers are free.
        assert output_embedding.shape[-1] == 1

        # To get the total power, we only use the tx-rx object...
        power = (inputs.types[..., [0]] * output_embedding[..., [0]]).reshape((-1, 1))

        # ... and sum over each structure
        power = sum_over_sample_in_pyg_batch(inputs, power)

        # Affine transformation
        power = self.affine(power)

        return power

    @staticmethod
    def _embed_link(data, tx_overwrite, rx_overwrite):
        """
        Embeds Tx-Rx link into multivector.

        This object is represented as follows:
        - channel 0-2: Tx position, represented as point
        - channel 3-5: Rx position, represented as point
        - channel 6-8: vector from Tx to Rx, represented as translation

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 3, 3)
            Data as provided by Wireless dataset. For links, it consists of the positions of Tx
            and Rx.
        tx_overwrite : Tensor or None
            Optional overrides for the Tx embedding, for inverse problems.
        rx_overwrite : Tensor or None
            Optional overrides for the rx embedding, for inverse problems.

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 9)
            Embedding: Tx and Rx positions and translation vector between them
        """

        tx_pos = data[..., 0, :]
        if tx_overwrite is not None:
            tx_pos = tx_overwrite + torch.zeros_like(tx_pos)
        rx_pos = data[..., 1, :]
        if rx_overwrite is not None:
            rx_pos = rx_overwrite + torch.zeros_like(rx_pos)

        return torch.cat([tx_pos, rx_pos, rx_pos - tx_pos], dim=-1)

    @staticmethod
    def _embed_tx_rx(data, overwrite):
        """
        Embeds Tx or Rx position into multivector.

        This object is represented as follows:
        - channel 0-2: position, represented as point

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 3, 3)
            Data as provided by Wireless dataset: Tx / Rx position

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 3)
            Embedding: Tx / Rx positions as trivectors
        """

        pos = data[..., 0, :]
        if overwrite is not None:
            pos = overwrite + torch.zeros_like(pos)

        return pos

    def _embed_mesh_face(self, data):
        """
        Embeds a mesh face (triangle) into multivector.

        In the "classic" embedding scheme, each mesh face is represented in five channels as
        follows:

        - channel 0-2: position of node 1
        - channel 3-5: position of node 2
        - channel 6-8: position of node 3
        - channel 9-11: normal vector on mesh face

        In the "kitchen_sink" embedding scheme, the following channels are added:

        - channel 12-14: mesh face center
        - channel 15-23: relative vectors from center to nodes

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 4, 3)
            Data as provided by Wireless dataset: mesh vertices

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 12)
            Embedding: vertex positions and face normal
        """

        # Compute normal
        vec1 = data[..., 1, :] - data[..., 0, :]
        vec2 = data[..., 2, :] - data[..., 0, :]
        normal = torch.nn.functional.normalize(vec1.cross(vec2), p=2, dim=-1)

        # "Classic" features
        features = [data[..., 0, :], data[..., 1, :], data[..., 2, :], normal]

        # Additional "kitchen-sink" features
        if self.embedding == "kitchen_sink":
            center_pos = torch.mean(data, dim=-2, keepdim=True)
            rel = data - center_pos
            features += [center_pos[..., 0, :], rel[..., 0, :], rel[..., 1, :], rel[..., 2, :]]

        return torch.cat(features, dim=-1)


# pylint: enable=duplicate-code
