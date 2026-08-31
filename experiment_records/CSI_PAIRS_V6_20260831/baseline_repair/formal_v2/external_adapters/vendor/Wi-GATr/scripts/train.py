# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Entrypoint to train a regression model. The model and dataset are
specified via the configuration file.
"""

import hydra


@hydra.main(config_path="../config", config_name="wigatr_wi3r", version_base=None)
def main(cfg):
    """Entry point for the training. Functionality lives in Experiment classes."""
    # Keeping the target config separate to use global config as argument
    target_cfg = {"_target_": cfg.experiment_target}
    exp = hydra.utils.instantiate(target_cfg, cfg)
    exp(train=True, evaluate=True)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
