from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from formal_v2.external_adapters.controlled_map_models import (
    RFIRForward,
    SigMapLocator,
    WiSERForward,
    complex_csi,
)
from formal_v2.external_adapters.wigatr_protocol import (
    SIX_CONDITIONS,
    build_six_condition_units,
    condition_action_sha256,
    condition_map,
    map_bounds,
    supplied_map_sha256,
)
from formal_v2.formal_data_verification import require_verified_roles_from_root
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import require_manifested_formal_qualification
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_resources import validate_resource_registry
from formal_v2.formal_routing import fit_route_normalization, route_dataset
from formal_v2.formal_teacher import load_teacher_bundle


CONFIG_SCHEMA = "csi-pairs-v6-controlled-map-adapter-v2"
EXECUTION_SCHEMA = "csi-pairs-v6-external-execution-v2"
METHODS = {"sigmap", "wiser", "rfir"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Controlled map-conditioned CSI-PAIRS baseline adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--command-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--model-name", required=True)
    try:
        run_adapter(parser.parse_args(argv))
    except Exception as error:
        print(f"controlled map adapter error: {error}", file=sys.stderr)
        return 2
    return 0


def run_adapter(args) -> dict:
    config_path = Path(args.config).resolve()
    config = load_controlled_map_config(config_path)
    for key, value in (
        ("source_revision", args.source_revision),
        ("adapter_id", args.adapter_id),
        ("model_name", args.model_name),
    ):
        if config[key] != value:
            raise RuntimeError(f"controlled adapter {key} differs from its manifest")
    if not _lower_sha256(args.command_sha256):
        raise RuntimeError("controlled adapter command hash must be lowercase SHA-256")
    dataset = FormalDataset.load(args.dataset)
    if not dataset.is_fixture and config["profile"] != "formal-paper-dose":
        raise RuntimeError("scientific controlled-map execution requires the formal-paper-dose profile")
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("six_condition_results.csv", "execution_manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite controlled adapter output: {name}")
    qualification = read_strict_json(run_root / "qualification" / "gate.json")
    formal_config = qualification["config"]
    qualification = require_manifested_formal_qualification(
        qualification, formal_config, dataset, allow_nonscientific_fixture=True
    )
    require_verified_roles_from_root(
        run_root,
        formal_config,
        dataset,
        ("source_encoder_train", "source_method_selection", "source_final_unseen_bank", "target"),
    )
    _verify_paper_resource(config)
    teacher = load_teacher_bundle(qualification["teacher_checkpoint"], formal_config)
    normalization = fit_route_normalization(dataset, teacher)
    evaluation_scenes = np.concatenate(
        (dataset.indices_for_role("source_final_unseen_bank"), dataset.indices_for_role("target"))
    )
    routed = route_dataset(dataset, teacher, formal_config, evaluation_scenes, normalization=normalization)
    model, model_metadata = _build_model(config, dataset)
    data_normalizer = _fit_data_normalizer(dataset)
    checkpoint, training_record = _train(
        model, config, dataset, data_normalizer, output, model_metadata
    )
    rows = _evaluate(model, config, dataset, routed, data_normalizer)
    result_path = output / "six_condition_results.csv"
    write_csv(result_path, rows)
    adapter_config_copy = output / "adapter_config.json"
    training_record_path = output / "training_record.json"
    shutil.copyfile(config_path, adapter_config_copy)
    write_json(training_record_path, training_record)
    manifest = {
        "schema_version": EXECUTION_SCHEMA,
        "adapter_id": config["adapter_id"],
        "model_name": config["model_name"],
        "implementation_status": config["implementation_status"],
        "source_revision": config["source_revision"],
        "dataset_sha256": sha256_file(dataset.source_path),
        "adapter_config_path": adapter_config_copy.name,
        "adapter_config_sha256": sha256_file(adapter_config_copy),
        "training_record_path": training_record_path.name,
        "training_record_sha256": sha256_file(training_record_path),
        "checkpoint_path": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "command_sha256": args.command_sha256,
        "results_sha256": sha256_file(result_path),
    }
    write_json(output / "execution_manifest.json", manifest)
    return manifest


def load_controlled_map_config(path: str | Path) -> dict:
    payload = read_strict_json(path)
    required = {
        "schema_version",
        "profile",
        "method",
        "model_name",
        "adapter_id",
        "implementation_status",
        "source_revision",
        "resource_id",
        "source_roles",
        "model",
        "training",
        "inverse",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("controlled map adapter config fields must be exact")
    if payload["schema_version"] != CONFIG_SCHEMA or payload["method"] not in METHODS:
        raise ValueError("controlled map adapter schema or method is invalid")
    if payload["profile"] not in {"formal-paper-dose", "software-smoke-only"}:
        raise ValueError("controlled map adapter profile is invalid")
    identities = {
        "sigmap": ("SigMap", "sigmap-controlled-csi-pairs-v1", "style-controlled-implementation"),
        "wiser": ("WiSER", "wiser-controlled-csi-pairs-v1", "style-controlled-implementation"),
        "rfir": ("RFIR", "rfir-controlled-csi-pairs-v1", "style-controlled-implementation"),
    }
    if (payload["model_name"], payload["adapter_id"], payload["implementation_status"]) != identities[payload["method"]]:
        raise ValueError("controlled map adapter identity is not frozen")
    if payload["source_roles"] != {
        "train": "source_encoder_train",
        "selection": "source_method_selection",
        "evaluation": ["source_final_unseen_bank", "target"],
    }:
        raise ValueError("controlled map adapter role ledger is not frozen V6")
    if set(payload["model"]) != {
        "hidden_dim", "scene_dim", "heads", "layers", "grid_size", "corridor_tokens", "tap_count", "maximum_primitives"
    }:
        raise ValueError("controlled map model fields must be exact")
    for key, value in payload["model"].items():
        _positive_integer(value, f"model.{key}")
    if payload["model"]["scene_dim"] % payload["model"]["heads"]:
        raise ValueError("controlled scene dimension must be divisible by attention heads")
    if set(payload["training"]) != {
        "seed", "steps", "batch_size", "microbatch_size", "precision",
        "learning_rate", "weight_decay", "selection_every_steps", "clip_grad_norm"
    }:
        raise ValueError("controlled map training fields must be exact")
    for key in ("seed", "steps", "batch_size", "microbatch_size", "selection_every_steps"):
        _positive_integer(payload["training"][key], f"training.{key}")
    for key in ("learning_rate", "clip_grad_norm"):
        _positive_number(payload["training"][key], f"training.{key}")
    if payload["training"]["weight_decay"] < 0:
        raise ValueError("controlled map weight decay must be nonnegative")
    if payload["training"]["batch_size"] % payload["training"]["microbatch_size"]:
        raise ValueError("controlled map microbatch must divide the effective batch")
    precision = payload["training"]["precision"]
    if precision not in {"float32", "bf16"}:
        raise ValueError("controlled map precision must be float32 or bf16")
    if payload["method"] in {"wiser", "rfir"} and precision != "float32":
        raise ValueError("WiSER and RFIR require float32 complex execution")
    if set(payload["inverse"]) != {
        "candidates_per_axis", "candidate_batch_size", "csi_weight", "power_weight", "receiver_z_m"
    }:
        raise ValueError("controlled map inverse fields must be exact")
    for key in ("candidates_per_axis", "candidate_batch_size"):
        _positive_integer(payload["inverse"][key], f"inverse.{key}")
    for key in ("csi_weight", "power_weight", "receiver_z_m"):
        _positive_number(payload["inverse"][key], f"inverse.{key}")
    return payload


def _verify_paper_resource(config):
    if config["method"] == "sigmap":
        if config["resource_id"] != "frozen-v6-section-11.2":
            raise RuntimeError("SigMap control has no authenticated paper resource")
        return
    project_root = Path(__file__).resolve().parents[2]
    registry = read_strict_json(project_root / "formal_v2" / "configs" / "waibu_resources_v1.json")
    rows = validate_resource_registry(registry, project_root / "waibu")
    matches = [row for row in rows if row["resource_id"] == config["resource_id"]]
    if len(matches) != 1 or matches[0]["status"] != "PASS":
        raise RuntimeError("controlled map paper resource failed authentication")


def _build_model(config, dataset):
    spec = PatchSpec.from_metadata(dataset.metadata)
    model_config = config["model"]
    context_dim = dataset.radio_config.shape[1] + dataset.bs_pose.shape[1]
    if config["method"] == "sigmap":
        model = SigMapLocator(
            dataset.channel_count,
            dataset.maps.shape[2],
            dataset.radio_config.shape[1] + dataset.bs_pose.shape[1],
            int(model_config["hidden_dim"]),
        )
    elif config["method"] == "wiser":
        model = WiSERForward(
            dataset.maps.shape[2],
            context_dim,
            int(model_config["scene_dim"]),
            int(model_config["heads"]),
            int(model_config["layers"]),
            int(model_config["grid_size"]),
            int(model_config["corridor_tokens"]),
            spec.antennas,
            spec.subcarriers,
            int(model_config["tap_count"]),
        )
    else:
        material_count = int(np.max(dataset.maps[:, :, 2])) + 1
        model = RFIRForward(
            material_count,
            spec.antennas,
            spec.subcarriers,
            context_dim,
            int(model_config["hidden_dim"]),
            int(model_config["maximum_primitives"]),
        )
    return model, {"antennas": spec.antennas, "subcarriers": spec.subcarriers}


def _train(model, config, dataset, normalizer, output, model_metadata):
    train_units = _units(dataset, "source_encoder_train")
    selection_units = _units(dataset, "source_method_selection")
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_precision = str(training["precision"])
    use_autocast = device.type == "cuda" and configured_precision == "bf16"
    if use_autocast and not torch.cuda.is_bf16_supported():
        raise RuntimeError("controlled-map BF16 execution is unsupported on this CUDA device")
    executed_precision = "bf16" if use_autocast else "float32"
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(training["steps"]), eta_min=float(training["learning_rate"]) * 0.1)
    rng = np.random.default_rng(int(training["seed"]))
    best_state = None
    best_selection = float("inf")
    best_step = 0
    last_loss = float("nan")
    schedule_counts = {"radiomap": 0, "cir": 0, "joint": 0}
    effective_batch_size = int(training["batch_size"])
    microbatch_size = int(training["microbatch_size"])
    for step in range(1, int(training["steps"]) + 1):
        chosen = rng.choice(
            len(train_units),
            size=effective_batch_size,
            replace=len(train_units) < effective_batch_size,
        )
        chosen_units = [train_units[int(index)] for index in chosen]
        task = _training_task(config["method"], step, int(training["steps"]), int(training["selection_every_steps"]))
        schedule_counts[task] += 1
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for start in range(0, effective_batch_size, microbatch_size):
            units = chosen_units[start : start + microbatch_size]
            batch = _batch(dataset, units, normalizer, device, config["method"])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                loss = _loss(model, config["method"], batch, task=task)
            if not torch.isfinite(loss):
                raise RuntimeError("controlled map model produced nonfinite loss")
            weight = len(units) / effective_batch_size
            (loss * weight).backward()
            accumulated_loss += float(loss.detach().cpu()) * weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["clip_grad_norm"]), error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        last_loss = accumulated_loss
        if step % int(training["selection_every_steps"]) == 0 or step == int(training["steps"]):
            selected = _selection_loss(
                model,
                config,
                dataset,
                selection_units,
                normalizer,
                device,
                use_autocast=use_autocast,
            )
            if selected < best_selection:
                best_selection = selected
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("controlled map source selection produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint = output / "checkpoints" / f"{config['method']}_source_selected.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "csi-pairs-v6-controlled-map-checkpoint-v2",
            "model_name": config["model_name"],
            "method": config["method"],
            "implementation_status": config["implementation_status"],
            "source_revision": config["source_revision"],
            "dataset_sha256": sha256_file(dataset.source_path),
            "train_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "target_roles_read": [],
            "selected_step": best_step,
            "selection_loss": best_selection,
            "normalizer": {key: value.tolist() for key, value in normalizer.items()},
            "model_metadata": model_metadata,
            "effective_batch_size": effective_batch_size,
            "microbatch_size": microbatch_size,
            "gradient_accumulation_steps": effective_batch_size // microbatch_size,
            "configured_precision": configured_precision,
            "executed_precision": executed_precision,
            "autocast_enabled": use_autocast,
            "state_dict": best_state,
        },
        checkpoint,
    )
    record = {
        "schema_version": "csi-pairs-v6-controlled-map-training-record-v2",
        "method": config["method"],
        "device": str(device),
        "steps": int(training["steps"]),
        "batch_size": int(training["batch_size"]),
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": effective_batch_size // microbatch_size,
        "configured_precision": configured_precision,
        "executed_precision": executed_precision,
        "autocast_enabled": use_autocast,
        "gradient_scaler_enabled": False,
        "optimizer": "AdamW",
        "scheduler": "cosine-to-0.1x",
        "last_training_loss": last_loss,
        "selected_step": best_step,
        "selection_loss": best_selection,
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "target_roles_read": [],
        "task_schedule": schedule_counts,
    }
    return checkpoint, record


def _evaluate(model, config, dataset, routed, normalizer):
    model.eval()
    units = build_six_condition_units(dataset, routed)
    rows = []
    for unit in units:
        observed = dataset.csi_clean[unit.scene, unit.source_world, unit.position]
        predictions = {}
        input_digests = {}
        for condition in SIX_CONDITIONS:
            supplied_map = condition_map(dataset, unit, condition)
            input_digests[condition] = supplied_map_sha256(supplied_map)
            if config["method"] == "sigmap":
                predictions[condition] = _sigmap_predict(
                    model, dataset, unit.scene, observed, supplied_map, normalizer
                )
            else:
                predictions[condition] = _inverse_predict(
                    model, config, dataset, unit.scene, observed, supplied_map, normalizer
                )
        truth = dataset.positions[unit.scene, unit.position]
        for condition in SIX_CONDITIONS:
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "model_name": config["model_name"],
                    "condition": condition,
                    "city_id": str(dataset.city_ids[unit.scene]),
                    "bank_id": str(dataset.bank_ids[unit.scene]),
                    "position_id": str(dataset.position_ids[unit.scene, unit.position]),
                    "localization_error_m": float(np.linalg.norm(predictions[condition] - truth)),
                    "csi_context_sha256": unit.csi_context_sha256,
                    "base_map_cluster_id": str(dataset.base_map_cluster_ids[unit.scene]),
                    "map_sha256": input_digests[condition],
                    "action_sha256": condition_action_sha256(dataset, unit, condition),
                    "query_count": 1,
                }
            )
    return rows


def _sigmap_predict(model, dataset, scene, observed, supplied_map, normalizer):
    device = next(model.parameters()).device
    csi = (observed - normalizer["csi_mean"]) / normalizer["csi_std"]
    maps = (supplied_map - normalizer["map_mean"][:, None, None]) / normalizer["map_std"][:, None, None]
    context = np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene]))
    context = (context - normalizer["context_mean"]) / normalizer["context_std"]
    with torch.no_grad():
        prediction = model(
            torch.as_tensor(csi[None], dtype=torch.float32, device=device),
            torch.as_tensor(maps[None], dtype=torch.float32, device=device),
            torch.as_tensor(context[None], dtype=torch.float32, device=device),
        )
    return prediction[0].cpu().numpy()


def _inverse_predict(model, config, dataset, scene, observed, supplied_map, normalizer):
    device = next(model.parameters()).device
    (x_min, x_max), (y_min, y_max) = map_bounds(dataset)
    count = int(config["inverse"]["candidates_per_axis"])
    xs = np.linspace(x_min, x_max, count, endpoint=False) + (x_max - x_min) / (2 * count)
    ys = np.linspace(y_min, y_max, count, endpoint=False) + (y_max - y_min) / (2 * count)
    candidates = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    occupancy = supplied_map[0]
    resolution = float(dataset.metadata["representation"]["map_resolution_m"])
    origin = np.asarray(dataset.metadata["representation"]["map_origin_xy_m"], dtype=np.float32)
    cell = np.floor((candidates - origin[None]) / resolution).astype(np.int64)
    cell[:, 0] = np.clip(cell[:, 0], 0, occupancy.shape[1] - 1)
    cell[:, 1] = np.clip(cell[:, 1], 0, occupancy.shape[0] - 1)
    free = occupancy[cell[:, 1], cell[:, 0]] < 0.5
    if np.any(free):
        candidates = candidates[free]
    target = torch.as_tensor(observed[None], dtype=torch.float32, device=device)
    target_complex = complex_csi(target)
    target_scale = torch.sqrt(torch.mean(target_complex.abs() ** 2)).clamp_min(1e-12)
    target_power = 10.0 * torch.log10(torch.mean(target_complex.abs() ** 2).clamp_min(1e-12))
    best_loss = float("inf")
    best = None
    batch_size = int(config["inverse"]["candidate_batch_size"])
    tx = torch.as_tensor(dataset.bs_pose[scene, :3], dtype=torch.float32, device=device)
    context = np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene]))
    context = (context - normalizer["context_mean"]) / normalizer["context_std"]
    context = torch.as_tensor(context, dtype=torch.float32, device=device)
    origin_tensor = torch.as_tensor(origin, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            candidate = torch.as_tensor(candidates[start : start + batch_size], dtype=torch.float32, device=device)
            maps = torch.as_tensor(supplied_map, dtype=torch.float32, device=device)[None].expand(candidate.shape[0], -1, -1, -1)
            tx_batch = tx[None].expand(candidate.shape[0], -1)
            context_batch = context[None].expand(candidate.shape[0], -1)
            origin_batch = origin_tensor[None].expand(candidate.shape[0], -1)
            if config["method"] == "wiser":
                predicted_power, predicted_complex = model(maps, tx_batch, context_batch, candidate, origin_batch, resolution)
            else:
                predicted_complex = model(maps, tx_batch, context_batch, candidate, origin_batch, resolution)
                predicted_power = 10.0 * torch.log10(torch.mean(predicted_complex.abs() ** 2, dim=1).clamp_min(1e-12))
            csi_loss = torch.mean(torch.abs(predicted_complex / target_scale - target_complex / target_scale) ** 2, dim=1)
            power_loss = (predicted_power - target_power) ** 2
            loss = float(config["inverse"]["csi_weight"]) * csi_loss + float(config["inverse"]["power_weight"]) * power_loss
            value, index = torch.min(loss, dim=0)
            if float(value) < best_loss:
                best_loss = float(value)
                best = candidate[int(index)].cpu().numpy()
    if best is None:
        raise RuntimeError("controlled inverse localization evaluated no candidate")
    return best


def _fit_data_normalizer(dataset):
    scenes = dataset.indices_for_role("source_encoder_train")
    csi = dataset.csi_clean[scenes]
    maps = dataset.maps[scenes]
    context = np.concatenate((dataset.radio_config[scenes], dataset.bs_pose[scenes]), axis=1)
    return {
        "csi_mean": np.mean(csi, axis=(0, 1, 2)),
        "csi_std": np.maximum(np.std(csi, axis=(0, 1, 2)), 1e-6),
        "map_mean": np.mean(maps, axis=(0, 1, 3, 4)),
        "map_std": np.maximum(np.std(maps, axis=(0, 1, 3, 4)), 1e-6),
        "context_mean": np.mean(context, axis=0),
        "context_std": np.maximum(np.std(context, axis=0), 1e-6),
    }


def _units(dataset, role):
    return [
        (int(scene), world, position)
        for scene in dataset.indices_for_role(role)
        for world in range(dataset.world_count)
        for position in range(dataset.position_count)
    ]


def _batch(dataset, units, normalizer, device, method):
    csi = np.asarray([dataset.csi_clean[scene, world, position] for scene, world, position in units])
    maps = np.asarray([dataset.maps[scene, world] for scene, world, _ in units])
    context = np.asarray([np.concatenate((dataset.radio_config[scene], dataset.bs_pose[scene])) for scene, _, _ in units])
    receiver = np.asarray([dataset.positions[scene, position] for scene, _, position in units])
    tx = np.asarray([dataset.bs_pose[scene, :3] for scene, _, _ in units])
    context = (context - normalizer["context_mean"]) / normalizer["context_std"]
    if method == "sigmap":
        csi = (csi - normalizer["csi_mean"]) / normalizer["csi_std"]
        maps = (maps - normalizer["map_mean"][None, :, None, None]) / normalizer["map_std"][None, :, None, None]
    origin = np.asarray(dataset.metadata["representation"]["map_origin_xy_m"], dtype=np.float32)
    return {
        "csi": torch.as_tensor(csi, dtype=torch.float32, device=device),
        "maps": torch.as_tensor(maps, dtype=torch.float32, device=device),
        "context": torch.as_tensor(context, dtype=torch.float32, device=device),
        "receiver": torch.as_tensor(receiver, dtype=torch.float32, device=device),
        "tx": torch.as_tensor(tx, dtype=torch.float32, device=device),
        "origin": torch.as_tensor(np.repeat(origin[None], len(units), axis=0), dtype=torch.float32, device=device),
        "resolution": float(dataset.metadata["representation"]["map_resolution_m"]),
    }


def _training_task(method, step, total_steps, phase_steps):
    if method != "wiser":
        return "joint"
    warmup = max(1, total_steps // 10)
    if step <= warmup:
        return "radiomap"
    if step <= 2 * warmup:
        return "cir"
    return "radiomap" if ((step - 2 * warmup - 1) // max(phase_steps, 1)) % 2 == 0 else "cir"


def _loss(model, method, batch, *, task="joint"):
    if method == "sigmap":
        return model.training_loss(batch["csi"], batch["maps"], batch["context"], batch["receiver"])
    if method == "wiser":
        return model.training_loss(
            batch["csi"], batch["maps"], batch["tx"], batch["context"], batch["receiver"],
            batch["origin"], batch["resolution"], task=task,
        )
    return model.training_loss(
        batch["csi"], batch["maps"], batch["tx"], batch["context"], batch["receiver"], batch["origin"], batch["resolution"]
    )


def _selection_loss(
    model,
    config,
    dataset,
    units,
    normalizer,
    device,
    *,
    use_autocast=False,
):
    model.eval()
    total = 0.0
    count = 0
    batch_size = int(config["training"]["microbatch_size"])
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            selected_units = units[start : start + batch_size]
            batch = _batch(
                dataset,
                selected_units,
                normalizer,
                device,
                config["method"],
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                loss = _loss(model, config["method"], batch)
            total += float(loss.cpu()) * len(selected_units)
            count += len(selected_units)
    model.train()
    if count == 0:
        raise RuntimeError("controlled map selection role is empty")
    return total / count


def _lower_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


if __name__ == "__main__":
    raise SystemExit(main())
