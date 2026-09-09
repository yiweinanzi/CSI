from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_factorial import (
    _build_corpus,
    _new_model,
    _pilot_scales,
    _train_arm,
    _training_normalization,
)
from formal_v2.formal_model import module_device
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_teacher import load_teacher_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = load_formal_config(args.config)
    dataset = FormalDataset.load(args.dataset)
    teacher = load_teacher_bundle(args.teacher, config, device="cuda:0")
    train_scenes = dataset.indices_for_role("source_encoder_train")
    selection_scenes = dataset.indices_for_role("source_method_selection")
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = _training_normalization(
        dataset, train_scenes, route_normalization, teacher.patch_spec
    )
    train_corpus = _build_corpus(
        dataset,
        train_scenes,
        teacher,
        config,
        route_normalization,
        normalization,
    )
    selection_corpus = _build_corpus(
        dataset,
        selection_scenes,
        teacher,
        config,
        route_normalization,
        normalization,
    )
    pilot = _pilot_scales(config, train_corpus, selection_corpus, 7018, device="cuda:0")
    jobs = [
        ("endpoint", "cuda:0", _new_model(config, train_corpus, 17, device="cuda:0")),
        ("alignment", "cuda:1", _new_model(config, train_corpus, 17, device="cuda:1")),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _train_arm,
                config,
                train_corpus,
                17,
                arm,
                pilot,
                step_count=2,
                device=device,
                prepared_model=model,
            )
            for arm, device, model in jobs
        ]
        trained = [future.result() for future in futures]
    rows = []
    for (arm, expected_device, _), (model, row) in zip(jobs, trained):
        observed_device = str(module_device(model))
        values = [value for key, value in row.items() if key.startswith("final_")]
        rows.append(
            {
                "arm": arm,
                "expected_device": expected_device,
                "observed_device": observed_device,
                "finite_losses": bool(np.all(np.isfinite(values))),
                "flop_measurement_status": row["flop_measurement_status"],
                "measured_flops_per_step": row["measured_flops_per_step"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    if any(
        row["observed_device"] != row["expected_device"] or not row["finite_losses"]
        for row in rows
    ):
        raise RuntimeError("factorial arms did not execute correctly on both CUDA devices")
    target = output / "result.json"
    target.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
