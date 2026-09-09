from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from formal_v2.formal_config import validate_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import (
    configure_reproducible_runtime,
    require_manifested_formal_qualification,
    runtime_provenance,
    validate_runtime_provenance,
)
from formal_v2.formal_external_runtime import _cuda_total_memory_bytes
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_routing import fit_route_normalization, route_dataset
from formal_v2.formal_teacher import load_teacher_bundle
from formal_v2.external_adapters.wigatr_protocol import (
    SIX_CONDITIONS,
    build_six_condition_units,
    condition_action_sha256,
    condition_map,
    relative_total_power_db,
    supplied_map_sha256,
    validate_fixed_radio_contract,
)


MODEL_NAME = "PMNet"
ADAPTER_ID = "pmnet-official-csi-pairs-v1"
PMNET_SOURCE_REVISION = "a0e0c5926de721074beeb23f630f2d313f6508dd"
PMNET_CONFIG_SCHEMA = "csi-pairs-pmnet-official-adapter-v3"
EXECUTION_SCHEMA = "csi-pairs-v6-external-execution-v4"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Official PMNet adaptation for the frozen CSI-PAIRS six-condition audit"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--command-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--adapter-id", default=ADAPTER_ID)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args(argv)
    try:
        run_adapter(args)
    except Exception as error:
        print(f"PMNet adapter error: {error}", file=sys.stderr)
        return 2
    return 0


def run_adapter(args) -> dict:
    if args.source_revision != PMNET_SOURCE_REVISION:
        raise RuntimeError("PMNet CLI source revision differs from the vendored snapshot")
    if args.adapter_id != ADAPTER_ID or args.model_name != MODEL_NAME:
        raise RuntimeError("PMNet adapter/model identity is frozen")
    if not _lower_sha256(args.command_sha256):
        raise RuntimeError("PMNet command hash must be lowercase SHA-256")

    config = load_pmnet_config(args.config)
    dataset = FormalDataset.load(args.dataset)
    if not dataset.is_fixture and config["profile"] != "formal-paper-dose":
        raise RuntimeError("scientific PMNet execution requires the formal paper-dose profile")
    validate_fixed_radio_contract(dataset)
    if not dataset.is_fixture:
        _require_formal_resources(config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for relative in (
        "six_condition_results.csv",
        "runtime_provenance.json",
        "execution_manifest.json",
    ):
        if (output / relative).exists():
            raise FileExistsError(f"refusing to overwrite PMNet output: {relative}")

    run_root = Path(args.run_root).resolve()
    qualification = read_strict_json(run_root / "qualification" / "gate.json")
    formal_config = qualification["config"]
    validate_formal_config(formal_config)
    qualification = require_manifested_formal_qualification(
        qualification,
        formal_config,
        dataset,
        allow_nonscientific_fixture=True,
    )

    configure_reproducible_runtime()
    runtime = validate_runtime_provenance(runtime_provenance())
    runtime_path = output / "runtime_provenance.json"
    write_json(runtime_path, runtime)
    _seed_runtime(int(config["training"]["seed"]))
    adapter_config_path = output / "adapter_config.json"
    shutil.copyfile(Path(args.config).resolve(), adapter_config_path)
    power_mean, power_scale = _fit_source_power_normalization(
        dataset, float(config["power"]["floor"])
    )
    material_count = int(dataset.metadata["assets"]["material_category_count"])
    model = _build_official_model(config, material_count)
    checkpoint, training_record = _fit_source_only_model(
        model,
        dataset,
        config,
        output,
        power_mean,
        power_scale,
    )
    write_json(output / "training_record.json", training_record)

    teacher = load_teacher_bundle(Path(qualification["teacher_checkpoint"]), formal_config)
    route_normalization = fit_route_normalization(dataset, teacher)
    evaluation_scenes = np.concatenate(
        (
            dataset.indices_for_role("source_final_unseen_bank"),
            dataset.indices_for_role("target"),
        )
    )
    routed = route_dataset(
        dataset,
        teacher,
        formal_config,
        evaluation_scenes,
        normalization=route_normalization,
    )
    result_rows = _evaluate_six_conditions(
        model,
        dataset,
        routed,
        config,
        power_mean,
        power_scale,
    )
    result_path = output / "six_condition_results.csv"
    write_csv(result_path, result_rows)
    execution = _build_execution_manifest(
        dataset,
        adapter_config_path,
        output / "training_record.json",
        checkpoint,
        result_path,
        args.command_sha256,
        runtime_path,
    )
    write_json(output / "execution_manifest.json", execution)
    return execution


def _build_execution_manifest(
    dataset,
    adapter_config_path,
    training_record_path,
    checkpoint_path,
    result_path,
    command_sha256,
    runtime_path,
):
    output = Path(result_path).resolve().parent
    paths = {
        "adapter_config": Path(adapter_config_path).resolve(),
        "training_record": Path(training_record_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
        "runtime_provenance": Path(runtime_path).resolve(),
    }
    if any(output not in path.parents for path in paths.values()):
        raise RuntimeError("PMNet execution artifact escapes its output directory")
    return {
        "schema_version": EXECUTION_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "model_name": MODEL_NAME,
        "implementation_status": "official-code-adaptation",
        "source_revision": PMNET_SOURCE_REVISION,
        "dataset_sha256": sha256_file(dataset.source_path),
        "adapter_config_path": str(paths["adapter_config"].relative_to(output)),
        "adapter_config_sha256": sha256_file(paths["adapter_config"]),
        "training_record_path": str(paths["training_record"].relative_to(output)),
        "training_record_sha256": sha256_file(paths["training_record"]),
        "checkpoint_path": str(paths["checkpoint"].relative_to(output)),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "command_sha256": command_sha256,
        "results_sha256": sha256_file(result_path),
        "runtime_provenance_path": str(paths["runtime_provenance"].relative_to(output)),
        "runtime_provenance_sha256": sha256_file(paths["runtime_provenance"]),
    }


def load_pmnet_config(path: str | Path) -> dict:
    config = read_strict_json(path)
    required = {
        "schema_version",
        "profile",
        "source_revision",
        "source_roles",
        "model",
        "training",
        "resources",
        "inverse",
        "power",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("PMNet adapter config fields must be exact")
    if config["schema_version"] != PMNET_CONFIG_SCHEMA:
        raise ValueError("PMNet adapter config schema mismatch")
    if config["profile"] not in {"formal-paper-dose", "software-smoke-only"}:
        raise ValueError("PMNet adapter profile is invalid")
    if config["source_revision"] != PMNET_SOURCE_REVISION:
        raise ValueError("PMNet source revision is not the frozen official snapshot")
    if config["source_roles"] != {
        "train": "source_encoder_train",
        "selection": "source_method_selection",
        "evaluation": ["source_final_unseen_bank", "target"],
    }:
        raise ValueError("PMNet role ledger differs from the frozen adapter contract")

    model = config["model"]
    if not isinstance(model, dict) or set(model) != {
        "n_blocks",
        "atrous_rates",
        "multi_grids",
        "output_stride",
        "input_encoding",
    }:
        raise ValueError("PMNet model config fields must be exact")
    if model != {
        "n_blocks": [3, 3, 27, 3],
        "atrous_rates": [6, 12, 18],
        "multi_grids": [1, 2, 4],
        "output_stride": 8,
        "input_encoding": "occupancy-height-material-one-hot-plus-tx-raster",
    }:
        raise ValueError("PMNet must retain the official v3 architecture and frozen input adaptation")

    training = config["training"]
    if not isinstance(training, dict) or set(training) != {
        "seed",
        "epochs",
        "batch_size",
        "microbatch_size",
        "precision",
        "learning_rate",
        "lr_decay",
        "lr_decay_every_epochs",
    }:
        raise ValueError("PMNet training config fields must be exact")
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "microbatch_size",
        "lr_decay_every_epochs",
    ):
        _positive_integer(training[key], f"training.{key}")
    for key in ("learning_rate", "lr_decay"):
        _positive_number(training[key], f"training.{key}")
    if config["profile"] == "formal-paper-dose" and (
        training["epochs"] != 30 or training["batch_size"] != 16
    ):
        raise ValueError("PMNet formal dose must retain the official 30-epoch batch-16 schedule")
    if training["batch_size"] % training["microbatch_size"]:
        raise ValueError("PMNet microbatch size must divide the effective batch size")
    if config["profile"] == "formal-paper-dose" and training["microbatch_size"] != 2:
        raise ValueError("formal PMNet requires the frozen microbatch size of two")
    if training["precision"] not in {"float32", "bf16"}:
        raise ValueError("PMNet precision must be float32 or bf16")
    if config["profile"] == "formal-paper-dose" and training["precision"] != "bf16":
        raise ValueError("formal PMNet training requires the frozen BF16 precision")
    if config["profile"] == "formal-paper-dose" and training["learning_rate"] != 1e-4:
        raise ValueError("PMNet formal dose must retain the official Adam learning rate")
    if config["profile"] == "formal-paper-dose" and (
        training["lr_decay"] != 0.5 or training["lr_decay_every_epochs"] != 10
    ):
        raise ValueError("PMNet formal dose must retain the official StepLR schedule")

    resources = config["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "minimum_cuda_memory_bytes"
    }:
        raise ValueError("PMNet resource fields must be exact")
    _positive_integer(
        resources["minimum_cuda_memory_bytes"],
        "resources.minimum_cuda_memory_bytes",
    )

    inverse = config["inverse"]
    if not isinstance(inverse, dict) or set(inverse) != {"occupancy_threshold"}:
        raise ValueError("PMNet inverse config fields must be exact")
    _positive_number(inverse["occupancy_threshold"], "inverse.occupancy_threshold")
    power = config["power"]
    if not isinstance(power, dict) or set(power) != {"floor", "definition"}:
        raise ValueError("PMNet power config fields must be exact")
    _positive_number(power["floor"], "power.floor")
    if power["definition"] != "10log10(mean(real_csi^2+imag_csi^2))":
        raise ValueError("PMNet power definition is not frozen")
    return config


def _build_official_model(config, material_category_count: int):
    vendor = Path(__file__).resolve().parent / "vendor" / "PMNet"
    _verify_vendor_tree(vendor)
    source = vendor / "models" / "pmnet_v3.py"
    spec = importlib.util.spec_from_file_location("csi_pairs_vendored_pmnet_v3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import the vendored PMNet source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_config = config["model"]
    model = module.PMNet(
        n_blocks=list(model_config["n_blocks"]),
        atrous_rates=list(model_config["atrous_rates"]),
        multi_grids=list(model_config["multi_grids"]),
        output_stride=int(model_config["output_stride"]),
    )
    input_channels = 2 + int(material_category_count) + 1
    model.layer1 = module._Stem(64, in_ch=input_channels)
    model.layer1.pool = torch.nn.MaxPool2d(2, 2, 1, ceil_mode=True)
    model.conv_up00[0] = torch.nn.Conv2d(
        128 + input_channels, 64, kernel_size=3, padding=1
    )
    return model


def _pmnet_input(dataset, scene: int, supplied_map: np.ndarray) -> np.ndarray:
    names = tuple(str(value) for value in dataset.map_channel_names.tolist())
    occupancy = np.asarray(supplied_map[names.index("occupancy")], dtype=np.float32)
    height = np.asarray(supplied_map[names.index("height")], dtype=np.float32)
    material = np.rint(supplied_map[names.index("material")]).astype(np.int64)
    material_count = int(dataset.metadata["assets"]["material_category_count"])
    one_hot = np.eye(material_count, dtype=np.float32)[material].transpose(2, 0, 1)
    one_hot *= (occupancy >= 0.5)[None]

    transmitter = np.zeros_like(occupancy, dtype=np.float32)
    origin = np.asarray(
        dataset.metadata["representation"]["map_origin_xy_m"], dtype=np.float64
    )
    resolution = float(dataset.metadata["representation"]["map_resolution_m"])
    column, row = np.floor(
        (np.asarray(dataset.bs_pose[scene, :2], dtype=np.float64) - origin) / resolution
    ).astype(np.int64)
    if not (0 <= row < transmitter.shape[0] and 0 <= column < transmitter.shape[1]):
        raise RuntimeError("PMNet transmitter lies outside the frozen map extent")
    transmitter[row, column] = 1.0
    return np.concatenate(
        (occupancy[None], height[None], one_hot, transmitter[None]), axis=0
    ).astype(np.float32, copy=False)


def _radiomap_samples(dataset, role: str) -> list[tuple[int, int]]:
    return [
        (int(scene), world)
        for scene in dataset.indices_for_role(role)
        for world in range(dataset.world_count)
    ]


def _fit_source_power_normalization(dataset, power_floor: float) -> tuple[float, float]:
    scenes = dataset.indices_for_role("source_encoder_train")
    power = relative_total_power_db(dataset.csi_clean[scenes], power_floor)
    mean = float(np.mean(power))
    scale = float(np.std(power))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("PMNet source power normalization is degenerate")
    return mean, scale


def _radiomap_target(
    dataset,
    scene: int,
    world: int,
    power_mean: float,
    power_scale: float,
    power_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = dataset.maps.shape[-2:]
    sums = np.zeros((rows, columns), dtype=np.float64)
    counts = np.zeros((rows, columns), dtype=np.int64)
    origin = np.asarray(
        dataset.metadata["representation"]["map_origin_xy_m"], dtype=np.float64
    )
    resolution = float(dataset.metadata["representation"]["map_resolution_m"])
    power = relative_total_power_db(dataset.csi_clean[scene, world], power_floor)
    cells = np.floor((dataset.positions[scene] - origin[None]) / resolution).astype(np.int64)
    for position, (column, row) in enumerate(cells):
        if not (0 <= row < rows and 0 <= column < columns):
            raise RuntimeError("PMNet receiver position lies outside the frozen map extent")
        sums[row, column] += (float(power[position]) - power_mean) / power_scale
        counts[row, column] += 1
    mask = counts > 0
    target = np.zeros_like(sums, dtype=np.float32)
    target[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return target, mask


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    return torch.mean((prediction[mask] - target[mask]) ** 2)


class _PMNetRadiomapDataset(Dataset):
    def __init__(self, dataset, role, power_mean, power_scale, power_floor):
        self.dataset = dataset
        self.samples = _radiomap_samples(dataset, role)
        self.power_mean = float(power_mean)
        self.power_scale = float(power_scale)
        self.power_floor = float(power_floor)
        if not self.samples:
            raise RuntimeError(f"PMNet role {role!r} has no radiomap samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene, world = self.samples[index]
        model_input = _pmnet_input(
            self.dataset, scene, self.dataset.maps[scene, world]
        )
        target, mask = _radiomap_target(
            self.dataset,
            scene,
            world,
            self.power_mean,
            self.power_scale,
            self.power_floor,
        )
        return (
            torch.from_numpy(model_input),
            torch.from_numpy(target[None]),
            torch.from_numpy(mask[None]),
        )


def _fit_source_only_model(
    model,
    dataset,
    config,
    output,
    power_mean,
    power_scale,
):
    training = config["training"]
    power_floor = float(config["power"]["floor"])
    train_dataset = _PMNetRadiomapDataset(
        dataset, "source_encoder_train", power_mean, power_scale, power_floor
    )
    selection_dataset = _PMNetRadiomapDataset(
        dataset, "source_method_selection", power_mean, power_scale, power_floor
    )
    generator = torch.Generator().manual_seed(int(training["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["microbatch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=int(training["microbatch_size"]),
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_precision = str(training["precision"])
    use_autocast = device.type == "cuda" and configured_precision == "bf16"
    if use_autocast and not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal PMNet BF16 execution is unsupported on this CUDA device")
    executed_precision = "bf16" if use_autocast else "float32"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["lr_decay_every_epochs"]),
        gamma=float(training["lr_decay"]),
    )
    best_state = None
    best_selection = float("inf")
    best_epoch = 0
    last_training = float("nan")
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        squared_error = 0.0
        observed = 0
        pending = []
        pending_samples = 0
        for index, batch in enumerate(train_loader):
            pending.append(batch)
            pending_samples += int(batch[0].shape[0])
            complete = pending_samples == int(training["batch_size"])
            final = index + 1 == len(train_loader)
            if not complete and not final:
                continue
            if pending_samples == 1 and len(train_dataset) > 1:
                break
            optimizer.zero_grad(set_to_none=True)
            total_observed = sum(int(batch_mask.sum()) for _, _, batch_mask in pending)
            for model_input, target, mask in pending:
                model_input = model_input.to(device)
                target = target.to(device)
                mask = mask.to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    prediction = model(model_input)
                    batch_squared_error = torch.sum(
                        (prediction[mask] - target[mask]) ** 2
                    )
                (batch_squared_error / total_observed).backward()
                squared_error += float(batch_squared_error.detach().cpu())
                observed += int(mask.sum())
            optimizer.step()
            pending = []
            pending_samples = 0
        scheduler.step()
        last_training = squared_error / observed
        selection_loss = _radiomap_mse(model, selection_loader, device)
        if selection_loss < best_selection:
            best_selection = selection_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("PMNet source-method-selection produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint = output / "checkpoints" / "pmnet_source_selected.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "csi-pairs-pmnet-checkpoint-v2",
            "source_revision": PMNET_SOURCE_REVISION,
            "adapter_config": config,
            "dataset_sha256": sha256_file(dataset.source_path),
            "fixture": dataset.is_fixture,
            "train_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "checkpoint_rule": "minimum_source_method_selection_masked_power_mse",
            "selected_epoch": best_epoch,
            "selection_power_mse": best_selection,
            "power_mean": power_mean,
            "power_scale": power_scale,
            "effective_batch_size": int(training["batch_size"]),
            "microbatch_size": int(training["microbatch_size"]),
            "configured_precision": configured_precision,
            "executed_precision": executed_precision,
            "autocast_enabled": use_autocast,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return checkpoint, {
        "schema_version": "csi-pairs-pmnet-training-record-v2",
        "source_revision": PMNET_SOURCE_REVISION,
        "device": str(device),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "microbatch_size": int(training["microbatch_size"]),
        "gradient_accumulation_steps": int(training["batch_size"])
        // int(training["microbatch_size"]),
        "configured_precision": configured_precision,
        "executed_precision": executed_precision,
        "autocast_enabled": use_autocast,
        "gradient_scaler_enabled": False,
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "last_training_power_mse": last_training,
        "selected_epoch": best_epoch,
        "selection_power_mse": best_selection,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "target_roles_read": [],
    }


def _radiomap_mse(model, loader, device):
    model.eval()
    squared_error = 0.0
    observed = 0
    with torch.no_grad():
        for model_input, target, mask in loader:
            model_input = model_input.to(device)
            target = target.to(device)
            mask = mask.to(device)
            prediction = model(model_input)
            squared_error += float(torch.sum((prediction[mask] - target[mask]) ** 2).cpu())
            observed += int(mask.sum())
    if observed == 0:
        raise RuntimeError("PMNet selection role has no observed radiomap cells")
    return squared_error / observed


def _evaluate_six_conditions(
    model,
    dataset,
    routed,
    config,
    power_mean,
    power_scale,
):
    model.eval()
    units = build_six_condition_units(dataset, routed)
    power_floor = float(config["power"]["floor"])
    occupancy_index = tuple(dataset.map_channel_names.tolist()).index("occupancy")
    origin = tuple(
        float(value)
        for value in dataset.metadata["representation"]["map_origin_xy_m"]
    )
    resolution = float(dataset.metadata["representation"]["map_resolution_m"])
    threshold = float(config["inverse"]["occupancy_threshold"])
    radiomap_cache = {}
    rows = []
    for unit in units:
        observed_power = float(
            relative_total_power_db(
                dataset.csi_clean[unit.scene, unit.source_world, unit.position],
                power_floor,
            )
        )
        predictions = {}
        input_digests = {}
        for condition in SIX_CONDITIONS:
            supplied_map = condition_map(dataset, unit, condition)
            map_digest = supplied_map_sha256(supplied_map)
            input_digests[condition] = map_digest
            key = (unit.scene, map_digest)
            if key not in radiomap_cache:
                radiomap_cache[key] = _predict_radiomap(
                    model,
                    dataset,
                    unit.scene,
                    supplied_map,
                    power_mean,
                    power_scale,
                )
            predictions[condition] = _inverse_localize_radiomap(
                radiomap_cache[key],
                supplied_map,
                observed_power,
                origin,
                resolution,
                threshold,
                occupancy_index=occupancy_index,
            )
        truth = dataset.positions[unit.scene, unit.position]
        for condition in SIX_CONDITIONS:
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "model_name": MODEL_NAME,
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


def _predict_radiomap(
    model,
    dataset,
    scene,
    supplied_map,
    power_mean,
    power_scale,
):
    device = next(model.parameters()).device
    model_input = torch.from_numpy(_pmnet_input(dataset, scene, supplied_map))[None].to(device)
    with torch.no_grad():
        normalized = model(model_input)[0, 0].cpu().numpy()
    return normalized.astype(np.float64) * float(power_scale) + float(power_mean)


def _inverse_localize_radiomap(
    radiomap: np.ndarray,
    supplied_map: np.ndarray,
    observed_power: float,
    origin_xy_m: tuple[float, float],
    resolution_m: float,
    occupancy_threshold: float,
    *,
    occupancy_index: int = 0,
) -> np.ndarray:
    free = supplied_map[occupancy_index] < float(occupancy_threshold)
    losses = np.where(free, np.abs(np.asarray(radiomap) - float(observed_power)), np.inf)
    if not np.any(np.isfinite(losses)):
        raise RuntimeError("PMNet supplied map has no free localization cell")
    row, column = np.unravel_index(int(np.argmin(losses)), losses.shape)
    return np.asarray(
        [
            float(origin_xy_m[0]) + (column + 0.5) * float(resolution_m),
            float(origin_xy_m[1]) + (row + 0.5) * float(resolution_m),
        ],
        dtype=np.float64,
    )


def _seed_runtime(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_formal_resources(config):
    if not torch.cuda.is_available():
        raise RuntimeError("formal PMNet execution requires an NVIDIA CUDA device")
    minimum = int(config["resources"]["minimum_cuda_memory_bytes"])
    device = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    available = _cuda_total_memory_bytes(torch, device, properties)
    if available < minimum:
        raise RuntimeError(
            "formal PMNet CUDA memory is below the frozen minimum: "
            f"available={available}, required={minimum}"
        )


def _verify_vendor_tree(vendor: Path):
    sums = vendor / "VENDOR_SHA256SUMS"
    if not sums.is_file():
        raise RuntimeError("PMNet vendor checksum manifest is missing")
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (vendor / relative).resolve()
        if vendor.resolve() not in path.parents or not path.is_file():
            raise RuntimeError("PMNet vendor checksum path is missing or escapes the snapshot")
        if sha256_file(path) != digest:
            raise RuntimeError(f"PMNet vendor file hash mismatch: {relative}")


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value, name):
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


if __name__ == "__main__":
    raise SystemExit(main())
