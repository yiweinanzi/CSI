# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Wraps around a Geometric Algebra Transformer (GATr) for predicting a scalar target variable in the
wireless channel modelling settings.
"""
from typing import Literal, Mapping, Optional

import torch
from gatr import GATr
from gatr.interface import embed_oriented_plane, embed_point, embed_translation
from torch import Tensor
from torch.nn import Module

from wigatr.utils.affine import AffineTransformation
from wigatr.utils.misc import build_pyg_attention_mask, sum_over_sample_in_pyg_batch


# pylint: disable=duplicate-code
class RSRPRegressionGATr(Module):
    """
    Wraps around GATr for predicting the total received power in wireless channel modelling
    settings.

    Requires GATr model with:
    - 9 multivector input channel (for the "classic" representation, otherwise more)
    - 4 scalar input channels
    - 1 multivector output channel (which will be ignored)
    - 1 scalar output channels (which will be used to read off the power)

    Parameters
    ----------
    net : gatr.nets.GATr
        GATr model
    affine_shift : None or float
        If not None, sets shift applied to total received power outputs.
    affine_scale : None or float
        If not None, sets scale applied to total received power outputs.
    embedding : "classic", "classic_z", "kitchen_sink", or "kitchen_sink_z"
        Which embedding scheme to use for the mesh faces. "kitchen_sink" includes more features.
        The suffic "_z" means that the transmitter has a preferred direction (along the z axis).
    """

    data_mode = "geometry"

    def __init__(
        self,
        net: GATr,
        affine_shift: Optional[float] = None,
        affine_scale: Optional[float] = None,
        embedding: Literal[
            "classic", "classic_z", "kitchen_sink", "kitchen_sink_z"
        ] = "kitchen_sink_z",
    ):
        super().__init__()
        assert embedding in ["classic", "classic_z", "kitchen_sink", "kitchen_sink_z"]
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

        multivector, scalars = self.embed_into_ga(inputs, overrides=overrides)
        mask = build_pyg_attention_mask(inputs)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        outputs = self.extract_from_ga(multivector_outputs, scalar_outputs, inputs)

        return outputs

    def embed_into_ga(self, inputs, overrides=None):
        """Embeds data into GA"""

        # Override defaults
        overrides = fill_in_override_defaults(overrides)

        # Scalar channels
        if "materials" in inputs:
            s = torch.cat(
                [
                    inputs.types,  # (batchsize * tokens, 4)
                    inputs.materials,  # (batchsize * tokens, self.num_materials)
                ],
                dim=1,
            )  # (batchsize * tokens, 4 + self.num_materials)
        else:
            s = inputs.types  # (batchsize * tokens, 4)

        # The dataset stores the types of tokens in `inputs.types`, and all other information (in
        # particular positions) in `inputs.x`.
        # Depending on the type, the information will be embedded differently in the geometric
        # algebra.
        # See `self._embed_*()` for additional information.
        mv_link = inputs.types[..., [0], None] * self._embed_link(
            inputs.x, overrides["tx"], overrides["rx"]
        )  # Link object
        mv_tx = inputs.types[..., [1], None] * self._embed_tx_rx(
            inputs.x, overrides["tx"]
        )  # Tx object
        mv_rx = inputs.types[..., [2], None] * self._embed_tx_rx(
            inputs.x, overrides["rx"]
        )  # Rx object
        mv_mesh = inputs.types[..., [3], None] * self._embed_mesh_face(inputs.x)  # Mesh face object

        # Concatenate along channels
        mv = torch.cat([mv_link, mv_tx, mv_rx, mv_mesh], dim=-2)

        return mv, s

    def extract_from_ga(self, multivector, scalars, inputs):
        """
        Extracts power prediction from multivectors.

        We extract the total received power in dB from the single scalar output channel associated
        with the Tx-Rx link object.
        """

        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[-2:] == (1, 16)
        assert scalars.shape[-1:] == (1,)

        # Predicted power is stored in the scalars associated with tx-rx link object
        # First, we project to the scalar components:
        power = scalars[..., [0]]

        # Then we only use the tx-rx link object:
        power = (inputs.types[..., [0]] * power).reshape((-1, 1))

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
        - channel 0: Tx position, represented as point
        - channel 1: Rx position, represented as point
        - channel 2: vector from Tx to Rx, represented as translation

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 3, 3)
            Data as provided by Wireless dataset. For links, it consists of the positions of
            Tx and Rx.
        tx_overwrite : Tensor or None
            Optional overrides for the Tx embedding, for inverse problems.
        rx_overwrite : Tensor or None
            Optional overrides for the rx embedding, for inverse problems.

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 3, 16)
            Embedding: Tx and Rx positions and translation vector between them
        """

        tx_pos = data[..., [0], :]
        if tx_overwrite is not None:
            tx_pos = tx_overwrite + torch.zeros_like(tx_pos)
        rx_pos = data[..., [1], :]
        if rx_overwrite is not None:
            rx_pos = rx_overwrite + torch.zeros_like(rx_pos)

        return torch.cat(
            [embed_point(tx_pos), embed_point(rx_pos), embed_translation(rx_pos - tx_pos)], dim=-2
        )

    def _embed_tx_rx(self, data, overwrite):
        """
        Embeds Tx or Rx position into multivector.

        This object is represented as follows:

        - channel 0: position, represented as point
        - channel 1: in the "classic_z" or "kitchen_sink_z" schemes, we additionally
          represent the preferred z direction.

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 3, 3)
            Data as provided by Wireless dataset: Tx / Rx position
        overwrite : None or Tensor
            Optional overrides for the embedding, for inverse problems.

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 1 or 2, 16)
            Embedding: Tx / Rx positions as trivectors
        """

        pos = data[..., [0], :]
        if overwrite is not None:
            pos = overwrite + torch.zeros_like(pos)

        if self.embedding in ["classic_z", "kitchen_sink_z"]:
            antenna_orientation = torch.zeros(3, device=data.device, dtype=data.dtype)
            antenna_orientation[2] = 1.0
            embedding = torch.cat(
                [
                    embed_point(pos),
                    embed_oriented_plane(antenna_orientation, position=pos),
                ],
                dim=-2,
            )
            return embedding

        return embed_point(pos)

    def _embed_mesh_face(self, data):
        """
        Embeds a mesh face (triangle) into multivector.

        In the "classic" embedding scheme, each mesh face is represented in five channels as
        follows:

        - channel 0: position of node 1, represented as point
        - channel 1: position of node 2, represented as point
        - channel 2: position of node 3, represented as point
        - channel 3: mesh, represented as plane (i.e. vector)

        In the "kitchen_sink" embedding scheme, the following channels are added:

        - channel 4: mesh face center, represented as point
        - channel 5-7: relative vectors from center to nodes, representated as translations

        Parameters
        ----------
        data : torch.Tensor with shape (batchsize * num_tokens, 4, 3)
            Data as provided by Wireless dataset: mesh vertices

        Returns
        -------
        multivector : torch.Tensor with shape (batchsize * num_tokens, 3, 16)
            Embedding: vertex positions and face normal
        """

        # Compute center pos
        center_pos = torch.mean(data, dim=-2, keepdim=True)

        # Compute normal
        vec1 = data[..., [1], :] - data[..., [0], :]
        vec2 = data[..., [2], :] - data[..., [0], :]
        normal = torch.nn.functional.normalize(
            torch.linalg.cross(vec1, vec2), p=2, dim=-1  # pylint: disable=not-callable
        )

        # "Classic" features
        features = [
            embed_point(data),
            embed_oriented_plane(normal, position=center_pos),
        ]

        # Additional "kitchen-sink" features
        if self.embedding in ["kitchen_sink", "kitchen_sink_z"]:
            features.append(embed_point(center_pos))
            features.append(embed_translation(data - center_pos))

        return torch.cat(features, dim=-2)


def fill_in_override_defaults(overrides: Optional[Mapping[str, Tensor]]) -> Mapping[str, Tensor]:
    """Fills in default values into an overrides dictionary"""
    if overrides is None:
        overrides = {}
    if "rx" not in overrides:
        overrides["rx"] = None
    if "tx" not in overrides:
        overrides["tx"] = None
    for key in overrides.keys():
        if key not in ["rx", "tx"]:
            raise ValueError(f"Unknown overrides key {key}")
    return overrides


# pylint: enable=duplicate-code
