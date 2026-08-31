# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
A dataset wrapper for Geometric data (see class documentation for details).
"""
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from wigatr.data import utils
from wigatr.data.utils import TARGET_DATA_FCTS, random_e2_transform
from wigatr.utils.logger import logger
from wigatr.utils.misc import make_full_edge_index
from wigatr.utils.tensors import check_sane, one_hot_vector, pad_to_shape


class GeometricDataset(torch.utils.data.Dataset):
    """
    Mesh-based dataset in a geometric parameterization.

    Thin wrapper around MultiFloorDataset from wiinsim with following additions:
    - loads experiment partitions
    - sum over all paths to only get total power as target (wiinsim-only)
    - provide mesh as part of __get_item__
    - packages everything as a torch_geometric graph object
    """

    def __init__(self, data_cfg, partition):
        super().__init__()

        # Dataset partitioning
        self.partition = partition

        # Transfromations, augmentation, canonicalization
        self.reciprocity_augmentation = (
            data_cfg.get("reciprocity_augmentation", False) and partition == "train"
        )
        self.e2_augmentation = data_cfg.get("e2_augmentation", False) and partition == "train"
        self.transform = utils.load_transform(data_cfg.get("splits", {}), partition)
        self.canonicalize = utils.load_canonicalization(data_cfg)
        self.add_edge_index = data_cfg.get("add_edge_index", False)
        self.num_materials = data_cfg.get("num_materials", 0)
        max_floor_plans = data_cfg.get("max_floor_plans", None)

        if data_cfg.get("wiinsim", None) is not None:
            self._dataset, num_tx = utils.load_wiinsim_dataset(data_cfg, partition, max_floor_plans)
        else:
            raise ValueError("Could not find known dataset configuration in data_cfg.")

        # True Tx position, if that's uniquely defined (useful for inverse problems)
        self.true_tx_pos = None
        if len(self._dataset.floor_ids) == 1 and num_tx == 1:
            self.true_tx_pos = torch.Tensor(self._dataset[0]["tx_xyz"])
            logger.debug(
                "Found unique Tx position in dataset: %s",
                str(self.true_tx_pos.detach().cpu().numpy()),
            )
        else:
            logger.debug("Did not find unique Tx position")

        target_data_fct_name = data_cfg.get("target_data_fct", "compute_non_coherent_total_power")
        self.extract_target = TARGET_DATA_FCTS[target_data_fct_name]

        # Prepare mesh cache
        self._cached_meshes = {}
        self._cached_materials = {}

    @property
    def visualization_dataset(self):
        """
        Expose internal dataset for radio map like visualization.
        """
        return self._dataset.get_dataset(0)

    def _get_mesh(self, floor_idx):
        """Loads the mesh for a given floor index"""

        # Caching. Can't easily use lru_cache b/c this is an instance method, so we do it manually
        if floor_idx not in self._cached_meshes:
            mesh_dict = self._dataset.get_mesh_by_floor_idx(floor_idx=floor_idx)
            verts = mesh_dict["verts"]  # (num_nodes, 3) points
            idx = mesh_dict["faces"]  # (num_triangles, 3) indices
            self._cached_meshes[floor_idx] = verts[idx]  # (num_triangles, 3, 3)

            # Also store materials
            if "materials" in mesh_dict:
                self._cached_materials[floor_idx] = mesh_dict["materials"][idx][
                    :, 0
                ]  # (num_triangles,)
            else:
                self._cached_materials[floor_idx] = None

        return self._cached_meshes[floor_idx], self._cached_materials[floor_idx]

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self._dataset)

    def __getitem__(self, idx):
        """Returns the `idx`-th sample from the dataset"""

        # Get raw channel info
        data = self._dataset[idx]
        # This is a dict with keys rx_xyz, tx_xyz, mpc, tx_idx, rx_idx, floor_id, floor_idx

        # Compute total power received
        # This target is assume to be invariant under E(3).
        invariant_target = self.extract_target(data)

        # Get mesh from cache
        mesh, materials = self._get_mesh(data["floor_idx"])

        # Get Tx and Rx positions
        tx = data["tx_xyz"]
        rx = data["rx_xyz"]

        # Transforms
        tx, rx, mesh, invariant_target, _ = self.transform(tx, rx, mesh, invariant_target)

        # Reciprocity augmentation
        if self.reciprocity_augmentation:
            flip = np.random.choice(a=[False, True])
            if flip:
                tx, rx = rx, tx

        # E(2) augmentation
        if self.e2_augmentation:
            tx, rx, mesh, invariant_target, _ = random_e2_transform(tx, rx, mesh, invariant_target)

        # Canonicalize
        tx, rx, mesh, invariant_target, canonicalization_shift = self.canonicalize(
            tx, rx, mesh, invariant_target
        )

        # Check inputs
        check_sane(tx, f"loading sample {idx}: Tx")
        check_sane(rx, f"loading sample {idx}: Rx")
        check_sane(invariant_target, f"loading sample {idx}: invariant_target")
        check_sane(mesh, f"loading sample {idx}: mesh")
        if materials is not None:
            check_sane(materials, f"loading sample {idx}: materials")

        # Tokenize
        data = tokenize_scene(
            tx,
            rx,
            mesh,
            invariant_target,
            materials,
            add_edge_index=self.add_edge_index,
            num_materials=self.num_materials,
            canonicalization_shift=canonicalization_shift,
        )

        return data


def tokenize_scene(
    tx,
    rx,
    mesh,
    invariant_target,
    materials,
    add_edge_index,
    num_materials,
    canonicalization_shift=None,
):
    """
    Given Tx position, Rx position, mesh, and optionally the total power, returns the tokenized
    scene.
    """

    # Get into correct shapes
    target_shape = (1, *mesh.shape[1:])  # (1, 3, 3)
    link = pad_to_shape(torch.cat((tx.reshape(1, 1, 3), rx.reshape(1, 1, 3)), dim=1), target_shape)
    tx = pad_to_shape(tx.reshape(1, 1, 3), target_shape)
    rx = pad_to_shape(rx.reshape(1, 1, 3), target_shape)

    # Store all the information into one tensor for easier processing
    x = torch.cat([link, tx, rx, mesh], dim=0)
    # (3+num_faces, 3, 3) - the 3 tokens that are not faces are the link object, the Tx, and the Rx

    # One-hot encoding of types
    types = torch.cat(
        [
            one_hot_vector(4, 0, batch_dims=(1,)),
            one_hot_vector(4, 1, batch_dims=(1,)),
            one_hot_vector(4, 2, batch_dims=(1,)),
            one_hot_vector(4, 3, batch_dims=(len(mesh),)),
        ],
        dim=0,
    )

    # Package PyG-style
    if invariant_target is not None:
        data = Data(x=x, types=types, y=invariant_target.unsqueeze(0))
        # y: (1, 1)
    else:
        data = Data(x=x, types=types)

    # Optional one-hot encoding of materials (just zero for non-mesh tokens
    if materials is not None:
        data.materials = torch.cat(
            [
                torch.zeros(3, num_materials, device=materials.device, dtype=materials.dtype),
                one_hot(materials, num_materials),  # pylint: disable=not-callable
            ],
            dim=0,
        )

    # Optionally, add fully connected edges for graph-based models
    if add_edge_index:
        data.edge_index = make_full_edge_index(num_nodes=len(x), self_loops=True, device=x.device)

    # Optionally, add canonicalization shift
    if canonicalization_shift is not None:
        data.canonicalization_shift = canonicalization_shift

    return data
