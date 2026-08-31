# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#!/usr/bin/env python3.10
"""
Entrypoint script to run the transmitter inference experiments and load the approriate
configurations.
Note: In the paper, we report receiver localization results. Due to channel reciprocity
in our setup, the transmitter localization results are equivalent to the receiver
localization results.
"""

import datetime
import pickle
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from wigatr.utils.logger import logger


def create_inverse_split(exp_cfg):
    """
    Loads the appropriate dataset configuration depending on the base dataset.
    """
    match exp_cfg.experiment_target:
        case str(x) if x.endswith("WiPTRRegressionExperiment"):
            return {
                "dir": f"{exp_cfg.data_root_dir}/WiPTR/test",
                "floor_id": (exp_cfg.inverse.scene, exp_cfg.inverse.scene + 1),
                "tx_id": (exp_cfg.inverse.scene % 5, (exp_cfg.inverse.scene % 5) + 1),
                "rx_id": (0, 10),
            }
        case str(x) if x.endswith("Wi3RRegressionExperiment"):
            return {
                "floor_id": (4750 + exp_cfg.inverse.scene, 4751 + exp_cfg.inverse.scene),
                "tx_id": (exp_cfg.inverse.scene % 5, (exp_cfg.inverse.scene % 5) + 1),
                "rx_id": (0, 200),
            }
        case _:
            raise ValueError(f"Unknown experiment target: {exp_cfg.experiment_target}")


@hydra.main(config_path="../config", config_name="infer_tx_wiptr", version_base=None)
def main(cfg):
    """Entry point for inverse problems in the 3-rooms dataset."""

    # Load experiment config
    exp_dir = Path(cfg.exp_dir)
    exp_cfg_file = exp_dir / "config.yaml"
    exp_cfg = OmegaConf.load(exp_cfg_file)

    # Patch experiment config for eval
    exp_cfg.exp_dir = str(exp_dir.resolve())
    exp_cfg.inverse = cfg
    exp_cfg.data.splits["eval_inverse_1"] = create_inverse_split(exp_cfg)

    # Load experiment
    target_cfg = {"_target_": exp_cfg.experiment_target}
    exp = hydra.utils.instantiate(target_cfg, exp_cfg, _recursive_=False)
    logger.info("Beginning Tx inference experiment")

    # Load model checkpoint
    checkpoint_filename = "model_final.pt" if cfg.step is None else f"model_step_{cfg.step}.pt"
    checkpoint = exp_dir / "models" / checkpoint_filename
    exp.load_model(checkpoint)

    # Run inverse problem
    results = exp.eval_inverse("eval_inverse_1")

    # Store results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    if cfg.step is None:
        result_filename = exp_dir / "results" / f"tx_inference_scene_{cfg.scene}_{timestamp}.pickle"
    else:
        result_filename = (
            exp_dir
            / "results"
            / f"tx_inference_step_{cfg.step}_scene_{cfg.scene}_{timestamp}.pickle"
        )
    result_filename.parent.mkdir(exist_ok=True)
    with open(result_filename, "wb") as file:
        pickle.dump(results, file)

    logger.info("Tx inference experiment finished")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
