# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Base class for all experiments.
"""

import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import gatr.primitives.attention
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gatr import MLPConfig, SelfAttentionConfig
from gatr.utils.misc import NaNError, get_device
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import wigatr.utils.logger
from wigatr.utils.logger import logger
from wigatr.utils.misc import frequency_check, get_batchsize
from wigatr.utils.plotting import MATPLOTLIB_PARAMS

cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)


class BaseExperiment:
    """
    Experiment manager from GATr.

    Explicitly included here to make it easier for us to adapt it.
    """

    def __init__(self, cfg):
        # Store config
        self.cfg = cfg

        # Device, dtype, backend
        self.device, self.dtype = self._init_backend()

        # Initialize state
        self.model = None
        self.ema = None
        self.optim = None
        self.scheduler = None

        self.metrics = {}
        self._best_state = None
        self._training_start_time = None

        # Initialize folder and logger
        self._initialize_experiment_folder()
        self._initialize_logger()

        # Training hooks: list of (state, hook_function)
        self._hooks = []

    def __call__(self, train=True, evaluate=True):
        """
        Performs experiment:
        - initializes all the logistics
        - instantiates model (if necessary)
        - loads checkpoint (if applicable)
        - trains model (if `train`)
        - evaluates model (if `eval`)

        Parameters
        ----------
        train : bool
            Whether to train the model.
        evaluate : bool
            Whether to evaluate the model.

        Returns
        ---------
        metrics : dict
            Dictionary of metrics.
        """

        self._initialize_experiment()

        self._save_config()

        # Create / load model
        if self.model is None:
            self.create_model()
            self.load_model()

        # Train
        if train:
            self.train()
            # Save model
            self.save_model("model_final.pt")

        # Evaluate
        if evaluate:
            self.evaluate()

        logger.info("All done!")
        return self.metrics

    def create_model(self):
        """Creates self.model according to the config, sets up EMA, and reports parameter count"""

        # Create model
        self.model = self._create_model()
        self.optim, self.scheduler = self.create_optimizer_and_scheduler()

        # Report number of parameters
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model has %.1f M learnable parameters", num_parameters / 1e6)
        # Create exponential moving average object
        if self.cfg.training.ema:
            logger.info("Using EMA for validation and eval")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            )
        else:
            logger.debug("Not using EMA")
            self.ema = None

    def load_model(self, checkpoint=None):
        """
        Loads a model checkpoint from disk, either given explicitly or in `self.cfg.checkpoint`
        """

        # If model hasn't been created yet, do that first
        if self.model is None:
            self.create_model()

        if checkpoint is None:
            checkpoint = self.cfg.checkpoint

        if checkpoint is not None:
            logger.info("Loading model checkpoint from %s", checkpoint)
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state_dict)

            if self.cfg.training.ema:
                ema_checkpoint = checkpoint.replace(".pt", "_ema.pt")
                logger.info("Loading EMA checkpoint from %s", ema_checkpoint)
                state_dict = torch.load(ema_checkpoint, map_location="cpu")
                self.ema.load_state_dict(state_dict)

    def train(self):
        """High-level training function."""

        logger.info("Starting training")

        # Prepare data
        train_data = self._load_dataset("train")
        train_loader = self._make_data_loader(
            train_data, batch_size=self.cfg.training.batchsize, shuffle=True
        )
        val_data = self._load_dataset("val")
        eval_batchsize = self.cfg.training.get("eval_batchsize", self.cfg.training.batchsize)
        val_loader = self._make_data_loader(val_data, batch_size=eval_batchsize, shuffle=False)

        # Training
        num_epochs = (self.cfg.training.steps - 1) // (
            (len(train_data) - 1) // self.cfg.training.batchsize + 1
        ) + 1
        logger.info(
            "Training for %d steps, that is, %d epochs on a "
            "dataset of size %d with batchsize %d",
            self.cfg.training.steps,
            num_epochs,
            len(train_data),
            self.cfg.training.batchsize,
        )

        # Prepare book-keeping
        self._best_state = {"state_dict": None, "loss": None, "step": None}
        self._training_start_time = time.time()

        # GPU
        self.model = self.model.to(self.device)
        if self.ema:
            self.ema.to(self.device)

        # Loop over epochs
        step = 0
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            self.model.train()

            # Loop over steps
            for data in tqdm(
                train_loader,
                total=len(train_loader),
                disable=not self.cfg.training.progressbar,
                desc=f"Epoch {epoch}",
            ):
                self._step(data, step, val_data, val_loader)
                if step >= self.cfg.training.steps:
                    break
                step += 1

            logger.debug(
                "Finished epoch %d in %s seconds", epoch, time.perf_counter() - epoch_start
            )

        logger.debug("Finished training")

        # Final validation loop
        if (
            self.cfg.training.validate_every_n_steps is not None
            and self.cfg.training.validate_every_n_steps > 0
        ):
            logger.debug("Starting validation")
            self.validate(val_loader, step)

        # Wrap up early stopping
        if (
            self.cfg.training.early_stopping
            and self._best_state["step"] is not None
            and self._best_state["step"] < step
        ):
            logger.info(
                "Early stopping after step %s with validation loss %s",
                self._best_state["step"],
                self._best_state["loss"],
            )
            self.model.load_state_dict(self._best_state["state_dict"])
        else:
            logger.debug("Not using early stopping")

    def validate(self, dataloader, step):
        """
        Runs validation loop, logs the results, and stores the state dict for early stopping
        if appropriate.
        """

        # Compute validation metrics, using EMA if available
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._compute_metrics(dataloader)
        else:
            metrics = self._compute_metrics(dataloader)

        # Log
        self.metrics["val"] = metrics
        logger.info("Validation loop at step %d:", step)
        for key, value in metrics.items():
            logger.info("    %s = %s", key, value)

        # Early stopping: compare val loss to last val loss
        new_val_loss = metrics["loss"]
        if self._best_state["loss"] is None or new_val_loss < self._best_state["loss"]:
            self._best_state["loss"] = new_val_loss
            self._best_state["state_dict"] = self.model.state_dict().copy()
            self._best_state["step"] = step

    def evaluate(self):
        """Evaluates self.model on all eval datasets and logs the results"""

        # Should we evaluate with EMA in addition to without?
        ema_values = [False]
        if self.ema is not None:
            ema_values.append(True)

        # Loop over evaluation datasets
        dfs = {}
        for tag in sorted(self._eval_dataset_tags):
            dataset = self._load_dataset(tag)
            eval_batchsize = self.cfg.training.get("eval_batchsize", self.cfg.training.batchsize)
            dataloader = self._make_data_loader(dataset, batch_size=eval_batchsize, shuffle=False)

            # Loop over EMA on / off
            for ema in ema_values:
                # Effective tag name
                full_tag = (tag + "_ema") if ema else tag

                # Run evaluation
                if ema:
                    with self.ema.average_parameters():
                        metrics = self._compute_metrics(dataloader)
                else:
                    metrics = self._compute_metrics(dataloader)

                # Log results
                self.metrics[full_tag] = metrics
                logger.info("Ran evaluation on dataset %s:", full_tag)
                for key, val in metrics.items():
                    logger.info("    %s = %s", key, val)

                # Store results in csv file
                # Pandas does not like scalar values, have to be iterables
                test_metrics_ = {key: [val] for key, val in metrics.items()}
                df = pd.DataFrame.from_dict(test_metrics_)
                df.to_csv(Path(self.cfg.exp_dir) / "metrics" / f"eval_{full_tag}.csv")
                dfs[full_tag] = df
        return dfs

    def save_model(self, filename=None):
        """Save model in experiment folder"""

        if filename is None:
            filename = "model.pt"

        model_path = Path(self.cfg.exp_dir) / "models" / filename
        logger.info("Saving model at %s", model_path)
        torch.save(self.model.state_dict(), model_path)

        if self.ema is not None:
            ema_path = Path(self.cfg.exp_dir) / "models" / filename.replace(".pt", "_ema.pt")
            logger.debug("Saving EMA model at %s", ema_path)
            assert ema_path != model_path
            torch.save(self.ema.state_dict(), ema_path)

    def register_hook(self, step, function):
        """Registers a hook: a function called at a specific step during training"""
        self._hooks.append((step, function))

    def _init_backend(self):
        """Initializes device, dtype, and attention implementation"""

        # Device
        device = get_device()
        logger.debug("Training on %s", device)

        # Dtype
        if self.cfg.training.float16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Training on bfloat16")
        elif self.cfg.training.float16:
            dtype = torch.float16
            logger.info("Training on float16 (bfloat16 is not supported by environment)")
        else:
            dtype = torch.float32
            logger.info("Training on float32")

        # Attention implementation
        logger.debug("Forcing use of xformers' attention implementation")
        gatr.primitives.attention.FORCE_XFORMERS = True

        return device, dtype

    def create_optimizer_and_scheduler(self):
        """Creates optimizer and scheduler.

        Returns
        -------
        optim : torch.optim.Optimizer
            Adam optimizer for the parameters of self.model.
        sched : torch.optim.lr_scheduler.LRScheduler
            Exponential LR scheduler for optim.
        """

        optimizer_choice = self.cfg.training.get("optimizer", "adam")
        if optimizer_choice == "adam":
            logger.info("Initializing Adam optimizer.")
            optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif optimizer_choice == "rmsprop":
            logger.info("Initializing RMSProp optimizer.")
            optim = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif optimizer_choice == "sgd":
            logger.info("Initializing SGD optimizer.")
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                weight_decay=self.cfg.training.weight_decay,
            )
        elif optimizer_choice == "sgd+momentum":
            logger.info("Initializing SGD+Momentum optimizer.")
            optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                momentum=0.9,
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer choice {optimizer_choice} not found.")

        num_scheduler_steps = self.cfg.training.steps // self.cfg.training.update_lr_every_n_steps
        if num_scheduler_steps > 0:
            gamma = self.cfg.training.lr_decay ** (1 / num_scheduler_steps)
        else:
            gamma = 1.0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        return optim, scheduler

    def _step(self, data, step, val_data, val_loader):
        """Everything that that may happen per step"""

        # Move data to GPU, and other and other data prep stuff
        data = self._prep_data(data)

        # Forward pass
        with torch.autocast(
            device_type="cuda", dtype=self.dtype, enabled=self.cfg.training.float16
        ):
            ctx = torch.autograd.detect_anomaly if self.cfg.training.detect_anomaly else nullcontext
            with ctx():
                loss, metrics = self._forward(*data)

        # Optimizer step
        grad_norm = self._optimizer_step(loss)

        # Post-step hooks: logging, validating, checkpoint saving, etc
        self._post_step(loss, metrics, grad_norm, step, val_data, val_loader)

        return step

    def _post_step(self, loss, metrics, grad_norm, step, val_data, val_loader):
        # Log loss and metrics
        self._log(loss, metrics, grad_norm, step)

        # Debugging output
        if step == 0:
            logger.info("Finished first forward pass with loss %f", loss.item())

        # Validation loop
        if frequency_check(step, self.cfg.training.validate_every_n_steps, skip_initial=False):
            logger.info("Starting validation at step %d", step)
            self.validate(val_loader, step)

        # Plotting
        if frequency_check(
            step, self.cfg.training.plot_every_n_steps, include_fractional=(0.01, 0.1)
        ):
            self.visualize(val_data, step)

        # Save model checkpoint
        if frequency_check(step, self.cfg.training.save_model_every_n_steps, skip_initial=True):
            self.save_model(f"model_step_{step}.pt")

        # LR scheduler
        if frequency_check(step, self.cfg.training.update_lr_every_n_steps, skip_initial=True):
            self.scheduler.step()
            logger.debug("Decaying LR to %f", self.scheduler.get_last_lr()[0])

        # Custom hooks
        for hook_step, hook in self._hooks:
            if hook_step == step:
                hook(model=self.model, step=step, experiment=self)

    def _initialize_experiment(self):
        """Set random seed and initialize plotting"""

        # Print config to log
        logger.info("Running experiment at %s", self.cfg.exp_dir)
        logger.debug("Config: \n%s", str(OmegaConf.to_yaml(self.cfg)))

        # Set random seed
        torch.random.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Initialize plotting
        self._init_plt()

    def _initialize_experiment_folder(self):
        """Creates experiment folder"""

        exp_dir = Path(self.cfg.exp_dir).resolve()
        subfolders = [
            exp_dir / "models",
            exp_dir / "figures",
            exp_dir / "metrics",
        ]

        # Create experiment subfolders (main folder will be automatically created as well)
        for subdir in subfolders:
            if subdir.exists():
                logger.warning("Warning: directory %s already exists!", subdir.as_posix())
            subdir.mkdir(parents=True, exist_ok=True)

    def _initialize_logger(self):
        """Initializes logging"""

        # In sweeps (multiple experiments in one job) we don't want to set up the handlers again
        if wigatr.utils.logger.LOGGING_INITIALIZED:
            logger.info("Logger already initialized - hi again!")
            return

        logger.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)-19.19s %(levelname)-1.1s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(Path(self.cfg.exp_dir) / "output.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logger.level)
        logger.addHandler(stream_handler)

        # This is important to avoid duplicate log outputs
        logger.propagate = False

        wigatr.utils.logger.LOGGING_INITIALIZED = True
        logger.info("Logger initialized.")

    @staticmethod
    def _init_plt():
        """Initializes matplotlib's rcparams to look good"""

        sns.set_style("whitegrid")
        matplotlib.rcParams.update(MATPLOTLIB_PARAMS)

    def _save_config(self):
        """Stores the config in the experiment folder"""

        # Save config
        config_filename = Path(self.cfg.exp_dir) / "config.yaml"
        logger.info("Saving config at %s", config_filename.as_posix())
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

    def _prep_data(self, data, device=None):
        """Data preparation during training loop, e.g. to move data to correct device and dtype"""
        if isinstance(data, (tuple, list)):
            data = tuple(x.to(device or self.device) for x in data)
        elif isinstance(data, dict):
            data = ({k: x.to(device or self.device) for k, x in data.items()},)
        else:
            data = (data.to(device or self.device),)

        return data

    def _optimizer_step(self, loss):
        """Optimizer step and gradient norm clipping."""

        assert self.optim is not None
        if not torch.isfinite(loss):
            raise NaNError("NaN in loss!")

        self.optim.zero_grad()
        loss.backward()

        # Grad norm clipping
        try:
            clip_norm = self.cfg.training.clip_grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_norm, error_if_nonfinite=True
            )
            grad_norm = grad_norm.cpu().item()
        except RuntimeError as e:
            for n, p in self.model.named_parameters():
                if not torch.isfinite(p.grad.flatten().norm()):
                    print(f"Non-finite grad in {n}")
            raise e

        self.optim.step()
        if self.ema is not None:
            self.ema.update()

        return grad_norm

    def _log(self, loss, metrics, grad_norm, step, quiet=False):
        """
        Logging some metrics.

        Parameters
        ----------
        loss : torch.Tensor
            Loss
        metrics : dict with str keys and float values
            Additional metrics for logging
        grad_norm : float
            Gradient norm
        """

        if not frequency_check(step, self.cfg.training.log_every_n_steps):
            return {}

        metrics["loss"] = loss.item()
        metrics["grad_norm"] = grad_norm
        metrics["step"] = step
        metrics["time_total_s"] = time.time() - self._training_start_time
        metrics["time_per_step_s"] = (time.time() - self._training_start_time) / (step + 1)

        if not quiet:
            for key, values in metrics.items():
                logging.info("Step %i: train.%s = %s", step, key, str(values))

        return metrics

    def _compute_metrics(self, dataloader):
        """
        Given a dataloader, computes all relevant metrics. Can be adapted by subclasses.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader.

        Returns
        -------
        metrics : dict with str keys and float values
            Metrics computed from dataset.
        """

        # Move to eval mode and eval device
        self.model.eval()
        eval_device = torch.device(self.cfg.training.eval_device)
        self.model = self.model.to(eval_device)

        aggregate_metrics = defaultdict(float)

        # Loop over dataset and compute error
        for data in tqdm(dataloader, disable=not self.cfg.training.progressbar, desc="Evaluating"):
            data = self._prep_data(data, device=eval_device)

            # Forward pass
            loss, metrics = self._forward(*data)

            # Weight for this batch (last batches may be smaller)
            batchsize = get_batchsize(data[0])
            weight = batchsize / len(dataloader.dataset)

            # Book-keeping
            aggregate_metrics["loss"] += loss.item() * weight
            for key, val in metrics.items():
                aggregate_metrics[key] += val * weight

        # Move model back to training mode and training device
        self.model.train()
        self.model = self.model.to(self.device)

        # Return metrics
        return aggregate_metrics

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
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_model(self):
        """
        Creates the model from the config.

        Returns
        -------
        model : torch.nn.Module
            A randomly initialized model, following the specification in self.cfg.model
        """

        # Hydra magic!
        return instantiate(self.cfg.model)

    def _load_dataset(self, tag):
        """
        Loads dataset. To be implemented by subclasses.

        Parameters
        ----------
        tag : str
            Dataset tag, like "train", "val", or one of self._eval_tags.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            Dataset.
        """
        raise NotImplementedError

    def _forward(self, *data):
        """
        Model forward pass. To be implemented by subclasses.

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
        raise NotImplementedError

    def visualize(self, dataset, step):
        """Visualization function. To be implemented by subclasses."""
        raise NotImplementedError

    @property
    def _eval_dataset_tags(self):
        """
        Eval dataset tags, to be implemented by subclasses.

        Returns
        -------
        tags : iterable of str
            Eval dataset tags
        """
        return {"eval"}
