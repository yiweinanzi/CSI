# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Base class for regression experiments, e.g., learning the received powers.
Together with the BaseExperiment it contains the main logic. The child
classes only contain dataset-specific split configurations.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from tqdm import trange

from wigatr.data.geometric import GeometricDataset
from wigatr.data.visualization import visualize_radio_map
from wigatr.experiments.base_experiment import BaseExperiment
from wigatr.experiments.utils import mean_relative_absolute_error
from wigatr.utils.logger import logger


class RegressionExperiment(BaseExperiment):
    """Base experiment for RSRP regression."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self._mse_criterion = torch.nn.MSELoss(reduction="mean")
        self._mae_criterion = torch.nn.L1Loss(reduction="mean")
        self._rae_criterion = mean_relative_absolute_error

    @property
    def mode(self):
        """
        Returns the experiment mode: "geometry" for GATr or Transformer experiments.

        Returns
        -------
        mode : None or {"geometry"}
            Returns the experiment mode, or None if it can't be determined.
        """

        if self.model is None:
            return None

        return self.model.data_mode

    def _load_dataset(self, tag):
        """
        Loads wireless dataset.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_dataset_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """

        logger.debug("Loading %s dataset", tag)

        if self.mode == "geometry":
            dataset = GeometricDataset(self.cfg.data, tag)
        else:
            raise ValueError(f"Experiment mode {self.mode} not supported")

        return dataset

    @lru_cache
    def _cached_dataset(self, tag):
        """
        Returns a cached dataset, usually for visualization."""

        return self._load_dataset(tag)

    def _forward(self, *data):
        """
        Model forward pass.

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """

        if self.mode == "geometry":
            outputs = self._geometry_forward(*data)
        else:
            raise ValueError(f"Experiment mode {self.mode} not supported")

        return outputs

    def _geometry_forward(self, *data):
        """
        Model forward pass for geometry-based methods

        Parameters
        ----------
        data : tuple of torch.Tensor
            Data batch.

        Returns
        -------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        """

        # Forward pass
        data = data[0]
        y_pred = self.model(data)

        # Compute loss
        loss = self._mse_criterion(y_pred, data.y)

        # Additional metrics
        mae = self._mae_criterion(y_pred, data.y)
        rae = self._rae_criterion(y_pred, data.y)
        metrics = {"rmse": loss.item() ** 0.5, "mae": mae.item(), "rae": rae.item()}

        return loss, metrics

    def _make_data_loader(self, dataset, batch_size, shuffle):
        """
        Creates a data loader.

        Parameters
        ----------
        dataset : torch.nn.utils.data.Dataset
            Dataset.
        batch_size : int
            Batch size.
        shuffle : bool
            Whether the dataset is shuffled.

        Returns
        -------
        dataloader
            Data loader.
        """

        logger.debug("Creating data loader")

        if self.mode == "geometry":
            loader = torch_geometric.loader.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True
            )
        else:
            raise ValueError(f"Experiment mode {self.mode} not supported")

        return loader

    def create_optimizer_and_scheduler(self):
        """Creates optimizer and scheduler.

        Overwritten here so that for RadioUNet, we can let the scheduler decay over *half*
        the training steps (not all training steps).

        Returns
        -------
        optim : torch.optim.Optimizer
            Adam optimizer for the parameters of self.model.
        sched : torch.optim.lr_scheduler.LRScheduler
            Exponential LR scheduler for optim.
        """

        if self.mode in ["geometry"]:
            return super().create_optimizer_and_scheduler()

        raise ValueError(f"Experiment mode {self.mode} not supported")

    def visualize(self, dataset, step):
        """Visualization function"""

        logger.debug("Plotting model predictions")

        # Forget about dataset, which points to the validation set. We want to use the dedicated
        # viz datasets
        for tag in ["viz_train", "viz_val"]:
            viz_dataset = self._cached_dataset(tag)
            filename = (
                None
                if self.cfg.debug
                else Path(self.cfg.exp_dir) / "figures" / f"{tag}_step_{step}.pdf"
            )
            visualize_radio_map(
                viz_dataset,
                self.model,
                filename,
                data_mode=self.mode,
                step=step,
                cmin=self.cfg.data.target_scaling.min,
                cmax=self.cfg.data.target_scaling.max,
                num_z_planes=self.cfg.data.get("viz_num_z_planes", 5),
                batchsize=self.cfg.training.eval_batchsize,
            )

    def eval_inverse(self, tag="eval_inverse_1"):
        """High-level function for gradient-based inverse modelling."""

        dataset = self._load_dataset(tag)
        results = {}
        for num_measurements in self.cfg.inverse.num_measurements:
            results[num_measurements] = self._tx_loop(dataset, num_measurements)

        return results

    def _tx_loop(self, dataset, num_measurements_per_exp):
        """Evaluates inverse Tx problem `num_samples` times, each time using `num_measurements`"""

        # Prepare inverse problems
        assert dataset.true_tx_pos is not None
        dataloader = self._make_data_loader(
            dataset, batch_size=num_measurements_per_exp, shuffle=False
        )
        results = defaultdict(list)

        logger.info(
            "Inverse solver. I: Initial position and loss. C: Current position and loss. "
            "B: Best position and loss. T: True position and loss"
        )
        # Run inverse problems
        for exp, data in enumerate(dataloader):
            if exp >= self.cfg.inverse.num_experiments:
                break
            inferred_tx_pos = self._solve_tx(data)

            # Compute error
            xy_error = (
                torch.sum((inferred_tx_pos[:2] - dataset.true_tx_pos[:2]) ** 2) ** 0.5
            ).item()
            xyz_error = (torch.sum((inferred_tx_pos - dataset.true_tx_pos) ** 2) ** 0.5).item()
            results["xy_error"].append(xy_error)
            results["xyz_error"].append(xyz_error)

            logger.info(
                "Finished inverse solver for %d measurements. Tx error: %.2f",
                num_measurements_per_exp,
                xy_error,
            )

        return results

    def _solve_tx(self, data):
        """
        Inverse problem: fit Tx position from one or multiple RSRP measurements, geometry.

        This method orchestrates `n` SGD fits and returns the one with the best loss.
        """

        best_loss = float("inf")
        best_tx_pos = None

        for _ in range(self.cfg.inverse.num_sgd_fits):
            tx_pos, loss = self._solve_tx_once(data)
            if loss < best_loss:
                best_tx_pos, best_loss = tx_pos, loss

        assert best_tx_pos is not None
        return best_tx_pos.detach().cpu()

    def _solve_tx_once(self, data):
        """
        Inverse problem: fit Tx position from one or multiple RSRP measurements, geometry.

        This method performs one SGD fit of the Tx position to the data.

        Parameters
        ----------
        data : Batch
            One or multiple samples from dataset (all sharing the same geometry).

        Returns
        -------
        inferred_rx : Tensor
            Inferred Tx position with shape (3,).
        loss : float
            Loss corresponding to inferred Tx position.
        """

        # Get floor plan size from dataset
        if self.cfg.inverse.ranges is not None:
            floor_plan_size = self.cfg.inverse.ranges
        else:
            # Need to infer floor plan by floor plan
            floor_plan_size = [
                (torch.min(data.x[..., i]).item(), torch.max(data.x[..., i]).item())
                for i in range(3)
            ]

        # Device
        device = self.cfg.inverse.device

        # Canonicalization shift for SEGNN
        if "canonicalization_shift" in data:
            cshift = data.canonicalization_shift.flatten()[:3].to(device)
        else:
            cshift = torch.zeros(3, device=device)

        # Disable original params
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Move to GPU
        data = data.to(device)
        self.model.to(device)
        self.model.eval()

        # Initialize Tx location (in canonicalized coordinated)
        tx = torch.zeros(3, device=device)
        tx.requires_grad_(True)
        with torch.no_grad():
            for i in range(3):
                torch.nn.init.uniform_(
                    tx[i], floor_plan_size[i][0] + cshift[i], floor_plan_size[i][1] + cshift[i]
                )

        # Optimizer
        optim = torch.optim.SGD(params=[tx], lr=self.cfg.inverse.lr)
        gamma = self.cfg.training.lr_decay ** (1.0 / self.cfg.inverse.steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
        criterion = torch.nn.MSELoss()

        # Book-keeping
        best_loss = float("inf")
        best_tx_pos = None
        true_tx = data.x[0, 0].detach().cpu().numpy()
        true_loss = criterion(self.model(data), data.y).item()
        tx_initial = tx.detach().cpu().numpy()
        loss_initial = None

        # Let's go
        for step in (pbar := trange(self.cfg.inverse.steps, desc="Inverse solver.")):
            # Model forward pass
            y_pred = self.model(data, overrides={"tx": tx})

            # MSE plus regularization terms for going out of bounds
            mse = criterion(y_pred, data.y)
            reg = 0.0
            zero = torch.zeros_like(mse)
            for i in range(3):
                low = floor_plan_size[i][0] + cshift[i]
                high = floor_plan_size[i][1] + cshift[i]
                reg += torch.where(tx[i] < low, (tx[i] - low) ** 2, zero)
                reg += torch.where(tx[i] > high, (tx[i] - high) ** 2, zero)
            loss = mse + self.cfg.inverse.reg_factor * reg

            # Gradient descent with added stochasticy
            optim.zero_grad()
            loss.backward()

            if step < self.cfg.inverse.noise_steps:
                noise_std = self.cfg.inverse.noise * np.exp(
                    step / self.cfg.inverse.noise_steps * np.log(self.cfg.inverse.noise_decay)
                )
                tx.grad = tx.grad + noise_std * torch.randn_like(tx.grad)
            torch.nn.utils.clip_grad_norm_(
                tx, self.cfg.inverse.clip_grad_norm, error_if_nonfinite=True
            )
            optim.step()
            scheduler.step()

            # Best loss so far?
            loss_float = loss.item()
            if loss_initial is None:
                loss_initial = loss_float
            if loss_float < best_loss:
                best_loss = loss_float
                best_tx_pos = tx.data + 0.0  # Force copy
            assert best_tx_pos is not None  # At this point we should have a best position

            # Keep the user informed
            tx_array = tx.data.cpu().numpy()
            best_tx_array = best_tx_pos.cpu().numpy()

            def fmt(x, y):
                return f"[{x[0]:.1f}, {x[1]:.1f}, {x[2]:.1f}] -> {y:.2f}"

            desc = (
                f"Inverse solver. I: {fmt(tx_initial, loss_initial)}. "
                f"C: {fmt(tx_array, loss_float)}. "
                f"B: {fmt(best_tx_array, best_loss)}. "
                f"T: {fmt(true_tx, true_loss)}"
            )
            pbar.set_description(desc=desc, refresh=True)

            # Stop when encountering NaNs
            if not (torch.all(torch.isfinite(loss)) and torch.all(torch.isfinite(tx))):
                break

        return best_tx_pos, best_loss


class Wi3RRegressionExperiment(RegressionExperiment):
    """
    Experiment manager for RSRP regression on the Wi3R dataset.
    """

    @property
    def _eval_dataset_tags(self):
        """
        Eval dataset tags, to be implemented by subclasses.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        eval_tags = {
            "eval_rx_gen",
            "eval_floor_gen",
            "eval_reciprocity",
            "eval_ood_shape",
            "eval_ood_layout",
            "eval_ood_scale",
        }
        # The transformations for the following tags were only implemented in
        # the geometry dataset so far.
        if self.mode == "geometry":
            eval_tags.update({"eval_rotation", "eval_translation"})

        return eval_tags


class WiPTRRegressionExperiment(RegressionExperiment):
    """
    Experiment manager for RSRP regression on the WiPTR dataset.
    """

    @property
    def _eval_dataset_tags(self):
        """
        Eval dataset tags, to be implemented by subclasses.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        return {
            "eval_rx_gen",
            "eval_floor_gen",
            "eval_rotation",
            "eval_translation",
            "eval_reciprocity",
            "eval_ood",
        }
