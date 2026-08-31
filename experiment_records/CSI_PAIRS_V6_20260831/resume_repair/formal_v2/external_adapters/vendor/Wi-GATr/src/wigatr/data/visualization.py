# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Helper functions to plot radio maps.
"""
import torch
import torch_geometric.data
import wiinsim.visualization as viz
from matplotlib import pyplot as plt


@torch.no_grad()
def _get_data_per_link(dataset, model, data_mode, device="cuda", batchsize=256, tx_pos=None):
    """
    Gets data to be plotted from dataset, for ground-truth predictions.

    Assumes dataset comprises a single floor.

    If tx_pos is not None, the Tx position is overwritten, but *only for the model predictions*,
    not the ground truth.

    Parameters
    ----------
    dataset: MultiFloorDataset object
    model: nn.Module object
    data_mode: Literal["geometry"]
    device: torch.device
    batchsize: int
        batch size to use for model predictions
    tx_pos: torch.tensor | None
        Override Tx position to use for model predictions, if not None

    Returns
    -------
    all_rx_pos: torch.tensor
        shape (N, 3)
    all_tx_pos: torch.tensor
        shape (N, 3)
    all_targets: torch.tensor
        shape (N, 1)
    all_predictions: torch.tensor
        shape (N, 1)
    """

    all_rx_pos, all_tx_pos, all_targets, all_predictions = [], [], [], []

    # Get ground-truth predictions
    for idx in range(len(dataset)):
        sample = dataset._dataset[idx]  # pylint:disable=protected-access

        target = dataset.extract_target(sample)
        all_rx_pos.append(sample["rx_xyz"].reshape(1, 3))
        if tx_pos is None:
            all_tx_pos.append(sample["tx_xyz"].reshape(1, 3))
        else:
            all_tx_pos.append(tx_pos.reshape(1, 3))
        all_targets.append(target.reshape(1, 1))

    # Compute model predictions
    idx = list(range(len(dataset)))
    model.eval()
    model.to(device)
    n_batches = (len(idx) - 1) // batchsize + 1
    for i_batch in range(n_batches):
        batch_idx = idx[i_batch * batchsize : (i_batch + 1) * batchsize]
        samples = [dataset[idx] for idx in batch_idx]
        if data_mode == "geometry":
            if tx_pos is not None:
                for sample in samples:
                    sample.x[0, 0] = tx_pos.reshape(3)
                    sample.x[1, 0] = tx_pos.reshape(3)
            batch = torch_geometric.data.Batch.from_data_list(samples)
            batch = batch.to(device)  # type: ignore
        else:
            raise ValueError(data_mode)
        prediction = model(batch).detach().cpu().flatten()
        all_predictions.append(prediction.reshape(-1, 1))

    # Package outputs
    all_rx_pos = torch.cat(all_rx_pos, dim=0)
    all_tx_pos = torch.cat(all_tx_pos, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    return all_rx_pos, all_tx_pos, all_targets, all_predictions


@torch.no_grad()
def visualize_radio_map(
    dataset,
    model,
    filename=None,
    cmin=None,
    cmax=None,
    data_mode="geometry",
    step=None,
    num_z_planes=5,
    batchsize=256,
):
    """Visualization function for three-room MultiFloorDataset"""
    cmin = -80.0 if cmin is None else cmin
    cmax = -35.0 if cmax is None else cmax

    # Get data
    if data_mode in ["geometry"]:
        room_dset = dataset.visualization_dataset
        rx_pos, tx_pos, targets, predictions = _get_data_per_link(
            dataset, model, data_mode, batchsize=batchsize
        )
    else:
        raise ValueError(f"Unknown data_mode {data_mode}.")

    # Loop over z planes
    num_samples_per_plane = len(room_dset) // num_z_planes
    plt.figure(figsize=(3 * 8, num_z_planes * 4))

    for i in range(num_z_planes):
        # Data at this z plane
        this_rx_pos = rx_pos[i * num_samples_per_plane : (i + 1) * num_samples_per_plane]
        this_target = targets[i * num_samples_per_plane : (i + 1) * num_samples_per_plane]
        this_prediction = predictions[i * num_samples_per_plane : (i + 1) * num_samples_per_plane]

        # Plot GT
        ax = plt.subplot(num_z_planes, 3, i * 3 + 1)
        viz.plot_room(ax, room_dset)
        viz.plot_rsrp(
            ax, tx_pos[0].numpy(), this_rx_pos.numpy(), this_target.numpy(), z_limits=(cmin, cmax)
        )
        plt.title("Ground truth")

        # Plot model predictions
        ax = plt.subplot(num_z_planes, 3, i * 3 + 2)
        viz.plot_room(ax, room_dset)
        viz.plot_rsrp(
            ax,
            tx_pos[0].numpy(),
            this_rx_pos.numpy(),
            this_prediction.numpy(),
            z_limits=(cmin, cmax),
        )
        if step is None:
            plt.title("Model predictions")
        else:
            plt.title(f"Model predictions (step {step})")

        # Plot residual predictions
        ax = plt.subplot(num_z_planes, 3, i * 3 + 3)
        error = torch.abs(this_prediction - this_target)
        viz.plot_room(ax, room_dset)
        viz.plot_rsrp(
            ax, tx_pos[0].numpy(), this_rx_pos.numpy(), error.numpy(), z_limits=(0.0, 10.0)
        )
        plt.title(f"Error: MAE = {error.mean().item():.2f} dB")

    plt.tight_layout()

    # Save or show plot
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
