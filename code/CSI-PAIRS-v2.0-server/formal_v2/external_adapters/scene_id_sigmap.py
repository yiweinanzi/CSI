from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from formal_v2.external_adapters.controlled_map_adapter import (
    _build_model,
    load_controlled_map_config,
    validate_controlled_map_checkpoint,
)
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import evidence_context
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Source-only city-ID prompt audit for the frozen SigMap control"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_scene_id_sigmap(
        args.dataset, args.output, args.checkpoint, args.run_root
    )
    return 0


def run_scene_id_sigmap(dataset_path, output_root, checkpoint_path, run_root):
    root = Path(run_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = FormalDataset.load(dataset_path)
    qualification = read_strict_json(root / "qualification" / "gate.json")
    config = qualification["config"]
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    adapter_root = (
        root
        / "external_baselines"
        / "adapters"
        / "sigmap-controlled-csi-pairs-v1"
    )
    execution = read_strict_json(adapter_root / "execution_manifest.json")
    expected_checkpoint = (adapter_root / execution["checkpoint_path"]).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if (
        checkpoint_path != expected_checkpoint
    ):
        raise RuntimeError("scene-ID SigMap checkpoint differs from external-baseline evidence")
    config_path = adapter_root / execution["adapter_config_path"]
    if sha256_file(config_path) != execution["adapter_config_sha256"]:
        raise RuntimeError("scene-ID SigMap adapter config hash mismatch")
    model_config = load_controlled_map_config(config_path)
    if model_config["method"] != "sigmap":
        raise RuntimeError("built-in scene-ID audit only accepts the frozen SigMap control")
    payload = _load_authenticated_checkpoint(checkpoint_path, execution)
    _validate_checkpoint(payload, dataset, execution)
    model, _ = _build_model(model_config, dataset)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    normalizer = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["normalizer"].items()
    }
    prompts = _source_city_prompts(model, dataset, normalizer)
    rows = _evaluate(model, dataset, normalizer, prompts, evidence)
    path = output / "scene_id_results.csv"
    write_csv(path, rows)
    return path


def _load_authenticated_checkpoint(checkpoint_path, execution):
    path = Path(checkpoint_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != execution.get("checkpoint_sha256")
    ):
        raise RuntimeError("scene-ID SigMap checkpoint differs from external-baseline evidence")
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_checkpoint(payload, dataset, execution):
    validate_controlled_map_checkpoint(payload)
    if (
        payload["model_name"] != "SigMap"
        or payload["method"] != "sigmap"
        or payload["implementation_status"] != "style-controlled-implementation"
        or payload["source_revision"] != execution["source_revision"]
        or payload["dataset_sha256"] != execution["dataset_sha256"]
        or payload["dataset_sha256"] != sha256_file(dataset.source_path)
        or payload["train_role"] != "source_encoder_train"
        or payload["selection_role"] != "source_method_selection"
        or payload["target_roles_read"] != []
        or not isinstance(payload["state_dict"], dict)
        or not payload["state_dict"]
        or execution["checkpoint_sha256"] == ""
    ):
        raise RuntimeError("scene-ID SigMap checkpoint violates the source-only contract")


def _source_city_prompts(model, dataset, normalizer):
    by_city = {}
    with torch.no_grad():
        for scene_value in dataset.indices_for_role("source_encoder_train"):
            scene = int(scene_value)
            city = str(dataset.city_ids[scene])
            maps = (
                dataset.maps[scene]
                - normalizer["map_mean"][None, :, None, None]
            ) / normalizer["map_std"][None, :, None, None]
            encoded = model.map(
                torch.as_tensor(maps, dtype=torch.float32)
            ).numpy()
            by_city.setdefault(city, []).extend(encoded)
    if len(by_city) < 2:
        raise RuntimeError("scene-ID prompt audit requires at least two source cities")
    prompts = {
        city: np.mean(np.asarray(values, dtype=np.float64), axis=0)
        for city, values in by_city.items()
    }
    required_cities = {
        str(dataset.city_ids[int(scene)])
        for scene in dataset.indices_for_role("source_final_unseen_bank")
    }
    if not required_cities.issubset(prompts):
        raise RuntimeError(
            "held-out source banks contain a city with no source-trained scene-ID prompt"
        )
    return prompts


def _evaluate(model, dataset, normalizer, prompts, evidence):
    source_scene_by_city = {}
    for scene_value in dataset.indices_for_role("source_encoder_train"):
        scene = int(scene_value)
        source_scene_by_city.setdefault(str(dataset.city_ids[scene]), scene)
    cities = sorted(prompts)
    rows = []
    with torch.no_grad():
        for scene_value in dataset.indices_for_role("source_final_unseen_bank"):
            scene = int(scene_value)
            city = str(dataset.city_ids[scene])
            swap_city = next(value for value in cities if value != city)
            swap_scene = source_scene_by_city[swap_city]
            source_world = int(dataset.natural_world_index[scene])
            swap_world = int(dataset.natural_world_index[swap_scene])
            correct_map = dataset.maps[scene, source_world]
            swapped_map = dataset.maps[swap_scene, swap_world]
            for position in np.flatnonzero(
                dataset.position_roles[scene] == "standard"
            ):
                position = int(position)
                observed = dataset.csi_clean[scene, source_world, position]
                context = np.concatenate(
                    (dataset.radio_config[scene], dataset.bs_pose[scene])
                )
                predictions = {
                    "map": _map_prediction(
                        model, observed, correct_map, context, normalizer
                    ),
                    "scene_id": _prompt_prediction(
                        model, observed, prompts[city], context, normalizer
                    ),
                    "map_swap": _map_prediction(
                        model, observed, swapped_map, context, normalizer
                    ),
                    "id_swap": _prompt_prediction(
                        model, observed, prompts[swap_city], context, normalizer
                    ),
                }
                truth = dataset.positions[scene, position]
                unit_id = (
                    f"{dataset.bank_ids[scene]}:{dataset.position_ids[scene, position]}"
                )
                for condition, prediction in predictions.items():
                    rows.append(
                        {
                            "unit_id": unit_id,
                            "model_name": "SigMap",
                            "condition": condition,
                            "bank_id": str(dataset.bank_ids[scene]),
                            "position_id": str(dataset.position_ids[scene, position]),
                            "localization_error_m": float(
                                np.linalg.norm(prediction - truth)
                            ),
                            "prediction_x": float(prediction[0]),
                            "prediction_y": float(prediction[1]),
                            "source_role": "source_final_unseen_bank",
                            "dataset_sha256": evidence["dataset_sha256"],
                            "config_sha256": evidence["config_sha256"],
                        }
                    )
    return rows


def _normalized_inputs(observed, context, normalizer):
    csi = (observed - normalizer["csi_mean"]) / normalizer["csi_std"]
    context = (
        np.asarray(context, dtype=np.float64) - normalizer["context_mean"]
    ) / normalizer["context_std"]
    return (
        torch.as_tensor(csi[None], dtype=torch.float32),
        torch.as_tensor(context[None], dtype=torch.float32),
    )


def _map_prediction(model, observed, supplied_map, context, normalizer):
    csi, context_tensor = _normalized_inputs(observed, context, normalizer)
    maps = (
        supplied_map - normalizer["map_mean"][:, None, None]
    ) / normalizer["map_std"][:, None, None]
    prediction = model(
        csi,
        torch.as_tensor(maps[None], dtype=torch.float32),
        context_tensor,
    )
    return prediction[0].numpy().astype(np.float64)


def _prompt_prediction(model, observed, prompt, context, normalizer):
    csi, context_tensor = _normalized_inputs(observed, context, normalizer)
    features = torch.cat(
        (
            model.csi(csi),
            torch.as_tensor(prompt[None], dtype=torch.float32),
            model.context(context_tensor),
        ),
        dim=1,
    )
    return model.fusion(features)[0].numpy().astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
