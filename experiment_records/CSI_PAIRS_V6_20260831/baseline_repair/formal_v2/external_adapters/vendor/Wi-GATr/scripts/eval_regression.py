# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Evaluate a trained regression model on evualtion splits.
"""

from pathlib import Path

import hydra
from omegaconf import open_dict


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg):
    """Entry point for the evaluation of checkpoints. Functionality lives in Experiment class."""

    # Determine checkpoint
    step = int(cfg.get("eval_at_step", -1))
    train_folder = str(cfg.exp_dir)
    if step <= 0:
        cfg.checkpoint = f"{train_folder}/models/model_final.pt"
    else:
        cfg.checkpoint = f"{train_folder}/models/model_step_{step}.pt"
    assert Path(cfg.checkpoint).exists(), f"{Path(cfg.checkpoint)}"

    with open_dict(cfg):
        cfg.exp_dir = cfg.exp_dir + "_eval"

    target_cfg = {"_target_": cfg.experiment_target}
    exp = hydra.utils.instantiate(target_cfg, cfg)
    exp(train=False, evaluate=True)
    exp.save_model(filename=Path(cfg.checkpoint).name)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
