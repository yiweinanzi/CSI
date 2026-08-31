# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Helper functions related to datasets.
"""

from copy import deepcopy

import numpy as np
import torch
from omegaconf import open_dict
from scipy.stats import special_ortho_group
from wiinsim import MultiFloorDataset

import wigatr.utils.augment_and_canonicalize as aac
from wigatr.utils.logger import logger

# Target data functions extract the correct target data from the values returned by __getitem__.
# Depending on the dataset the items may have different keys and require different target
# data functions.
TARGET_DATA_FCTS = {}


def target_data_function(fct):
    """Decorator to register a function as a target data function."""
    TARGET_DATA_FCTS[fct.__name__] = fct
    return fct


@target_data_function
def compute_non_coherent_total_power(data):
    """Computes non-coherent total power from data item"""
    return get_total_power_received_from_mpc(data["mpc"])


@target_data_function
def extract_power_db(data):
    """Extracts existing power in dB from data item"""
    return data["power_db"]


@target_data_function
def extract_delay_spread(data):
    """Extracts existing power in dB from data item"""
    return data["delay_spread"]


def get_total_power_received_from_mpc(mpc):
    """Given MPC data, computes the total power received in dB"""
    mask = mpc[-1].to(torch.bool)  # valid paths
    path_strength_db = mpc[0, mask]
    factor = np.log(10.0) / 10.0  # Take care of dB "units" correctly
    total_power = torch.logsumexp(2.0 * path_strength_db * factor, dim=0, keepdim=True) / factor
    # The 2.0 is because the mpc describes amplitudes, but we care about power
    return total_power


def translation_transform(tx, rx, mesh, invariant_target):
    """Translates a scene by a fixed amount along the x, y, and z coordinates"""
    translation = torch.Tensor([-10.0, 2.0, -5.0]).to(tx.device, tx.dtype)
    return (
        tx + translation,
        rx + translation,
        mesh + translation,
        invariant_target,
        translation,
    )


def rotation_transform(tx, rx, mesh, invariant_target):
    """
    Rotates a scene by 90 degrees

    (x, y, z) -> (-y, x, z)
    """

    def rot(data):
        """Rotates a set of vectors by 90 degrees around z axis"""
        assert data.shape[-1] == 3
        return torch.cat([-data[..., [1]], data[..., [0]], data[..., [2]]], dim=-1)

    return rot(tx), rot(rx), rot(mesh), invariant_target, torch.zeros_like(tx)


def random_e3_transform(tx, rx, mesh, invariant_target, translation_scale=None):
    """Applies a random E(3) transform. Not field-tested, use with caution."""
    if translation_scale is None:
        translation = torch.zeros_like(tx)
    else:
        translation = translation_scale * torch.randn(3, device=tx.device, dtype=tx.dtype)
    rotation = torch.tensor(special_ortho_group.rvs(3)).to(tx.device, tx.dtype)

    def e3(data):
        return translation + data @ rotation.T

    return e3(tx), e3(rx), e3(mesh), invariant_target, translation


def random_e2_transform(tx, rx, mesh, invariant_target, translation_scale=1.0):
    """Applies a random E(2) transform in the x-y plane"""
    tx_, rx_, mesh_ = aac.random_e2_transform(  # pylint: disable=unbalanced-tuple-unpacking
        tx, rx, mesh, translation_scale=translation_scale
    )
    return tx_, rx_, mesh_, invariant_target, tx_ - tx


def e2_canonicalization(tx, rx, mesh, invariant_target):
    """E(2) canonicalization as in ESL"""
    reference = torch.cat((tx.view(-1, 3), rx.view(-1, 3)), dim=0)
    tx_, rx_, mesh_ = aac.canonicalize_e2(  # pylint: disable=unbalanced-tuple-unpacking
        reference, tx, rx, mesh
    )
    translation = 0.5 * ((tx_ + rx_) - (tx + rx))
    return tx_, rx_, mesh_, invariant_target, translation


def reciprocity_transform(tx, rx, mesh, invariant_target):
    """Swaps Tx and Rx position"""
    return rx, tx, mesh, invariant_target, torch.zeros_like(tx)


def tx_to_origin_transform(tx, rx, mesh, invariant_target):
    """Translates all coordinates such that the Tx is at the origin"""
    return tx - tx, rx - tx, mesh - tx, invariant_target, -tx


def rx_to_origin_transform(tx, rx, mesh, invariant_target):
    """Translates all coordinates such that the Rx is at the origin"""
    return tx - rx, rx - rx, mesh - rx, invariant_target, -rx


def mesh_com_to_origin_transform(tx, rx, mesh, invariant_target):
    """Translates all coordinates such that the Tx is at the origin"""
    mean_dims = tuple(range(len(mesh.shape) - 1))
    com = torch.mean(mesh, dim=mean_dims)
    return tx - com, rx - com, mesh - com, invariant_target, -com


TRANSFORMS = {
    "translation": translation_transform,
    "rotation": rotation_transform,
    "reciprocity": reciprocity_transform,
}

CANONICALIZATIONS = {
    "tx": tx_to_origin_transform,
    "rx": rx_to_origin_transform,
    "com": mesh_com_to_origin_transform,
    "e2": e2_canonicalization,
}


def load_wiinsim_dataset(data_cfg, partition, max_floor_plans):
    """
    Applies default settings, overwrite directories, and splits to the data
    config before instantiating the MultiFloorDataset dataset.

    Parameters
    ----------
    data_cfg : OmegaConf
        cfg.data.splits
    partition : str
        Partition name, like "train"
    max_floor_plans : int or None
        Maximum number of floor plans to keep. Only used if partition is "train".

    Returns
    ----------
    dataset : MultiFloorDataset instance
    num_tx : int
        Number of tx that were loaded.
    """
    wiinsim_config = deepcopy(data_cfg.wiinsim)
    with open_dict(wiinsim_config):
        target_data_fct_name = data_cfg.get(
            "target_data_fct_name", "compute_non_coherent_total_power"
        )
        assert target_data_fct_name == "compute_non_coherent_total_power"
        # Note: This assumes that target_scaling refers to powers in db.
        # If you want to use different targets, first make sure that wiinsim can support that.
        wiinsim_config["gains_db_attrs"] = data_cfg.target_scaling
    overwrite_dir, floor_ids, wiinsim_config.tx_idx, wiinsim_config.rx_idx = load_partition_wiinsim(
        data_cfg.splits, partition, max_floor_plans
    )
    if overwrite_dir is not None:
        wiinsim_config.root = overwrite_dir

    # Load dataset
    logger.info(
        "Loading MultiFloorDataset for partition %s with %d floor plans", partition, len(floor_ids)
    )
    logger.debug("  Root dir: %s", str(wiinsim_config.root))
    logger.debug("  Floor IDs: %s", str(floor_ids))
    logger.debug("  Tx IDs %s", str(wiinsim_config.tx_idx))
    logger.debug("  Rx IDs %s", str(wiinsim_config.rx_idx))
    logger.info("Loading MultiFloorDataset. This may take a few minutes.")
    dataset = MultiFloorDataset(wiinsim_config, floor_ids=floor_ids)

    # Report floor plan properties
    logger.debug("Found the following floor plans:")
    for floor_id in dataset.floor_ids:
        dset = dataset.get_dataset_by_floor_id(floor_id)
        num_tx = dset.num_tx
        num_rx = dset.num_rx
        logger.debug("  Floor plan %d: %d Tx positions, %d Rx positions", floor_id, num_tx, num_rx)
    return dataset, num_tx


def load_partition_wiinsim(split_cfg, partition, max_floor_plans):
    """
    WiInSim specific function.
    Loads the indices for floor plan, Tx, and Rx for the given partition from the config

    Parameters
    ----------
    split_cfg : OmegaConf
        cfg.data.splits
    partition : str
        Partition name, like "train"
    max_floor_plans : int or None
        Maximum number of floor plans to keep. Only used if partition is "train".

    Returns
    -------
    overwrite_directory : None or str
        If not None, specifies a directory that should overwrite the default wiinsim root dir.
    floor_id : list of int
    tx_idx : list of int or -1
    rx_idx : list of int or -1
    """

    partition_cfg = split_cfg[partition]

    # DictConfig doesn't define get()
    overwrite_directory = partition_cfg["dir"] if "dir" in partition_cfg else None

    floor_id = list(range(partition_cfg.floor_id[0], partition_cfg.floor_id[1]))
    exclude_floor_id = partition_cfg.get("exclude_floor_id")
    if exclude_floor_id:
        for excluded_id in exclude_floor_id:
            try:
                floor_id.remove(excluded_id)
            except ValueError:
                logger.warning(
                    "Trying to remove floor ID that was not part of dataset split. "
                    "Please double-check config."
                )
    if partition == "train" and max_floor_plans is not None and max_floor_plans > 0:
        floor_id = floor_id[:max_floor_plans]

    tx_id = (
        -1
        if partition_cfg.tx_id is None
        else list(range(partition_cfg.tx_id[0], partition_cfg.tx_id[1]))
    )
    rx_id = (
        -1
        if partition_cfg.rx_id is None
        else list(range(partition_cfg.rx_id[0], partition_cfg.rx_id[1]))
    )

    return overwrite_directory, floor_id, tx_id, rx_id


def load_transform(split_cfg, partition):
    """
    Loads the indices for floor plan, Tx, and Rx for the given partition from the config

    Parameters
    ----------
    split_cfg : OmegaConf
        cfg.data.splits
    partition : str
        Partition name, like "train"

    Returns
    -------
    transform : callable
    """

    if partition in split_cfg:
        partition_cfg = split_cfg[partition]

        if "transform" in partition_cfg:
            transform = TRANSFORMS[partition_cfg.transform]
            return transform

    # No transform config found, return no-op.
    def transform(*args):  # pylint: disable=function-redefined
        return (*args, torch.zeros_like(args[0]))

    return transform


def load_canonicalization(data_cfg):
    """
    Loads the indices for floor plan, Tx, and Rx for the given partition from the config

    Parameters
    ----------
    data_cfg : OmegaConf
        cfg.data

    Returns
    -------
    transform : callable
    """

    canonicalize = data_cfg.get("canonicalize")
    if canonicalize is None:
        return lambda *args: (*args, torch.zeros_like(args[0]))

    return CANONICALIZATIONS[canonicalize]
