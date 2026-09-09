from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

from formal_v2.formal_config import validate_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import require_manifested_formal_qualification
from formal_v2.formal_external_runtime import (
    collect_external_runtime,
    validate_external_runtime,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_csv, write_json
from formal_v2.formal_routing import fit_route_normalization, route_dataset
from formal_v2.formal_teacher import load_teacher_bundle

from formal_v2.external_adapters.wigatr_protocol import (
    SIX_CONDITIONS,
    WIGATR_SOURCE_REVISION,
    build_six_condition_units,
    condition_action_sha256,
    condition_map,
    grid_to_triangular_mesh,
    load_wigatr_config,
    map_bounds,
    relative_total_power_db,
    require_compact_mesh_matches_map_surface,
    supplied_map_sha256,
    validate_fixed_radio_contract,
)


MODEL_NAME = "Wi-GATr"
ADAPTER_ID = "wigatr-official-csi-pairs-v1"
EXECUTION_SCHEMA = "csi-pairs-v6-external-execution-v4"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Official Wi-GATr adaptation for the frozen CSI-PAIRS six-condition audit"
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
        print(f"Wi-GATr adapter error: {error}", file=sys.stderr)
        return 2
    return 0


def run_adapter(args) -> dict:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("the frozen official Wi-GATr environment requires Python 3.10")
    if args.source_revision != WIGATR_SOURCE_REVISION:
        raise RuntimeError("Wi-GATr CLI source revision differs from the vendored snapshot")
    if args.adapter_id != ADAPTER_ID or args.model_name != MODEL_NAME:
        raise RuntimeError("Wi-GATr adapter/model identity is frozen")
    if not _lower_sha256(args.command_sha256):
        raise RuntimeError("Wi-GATr command hash must be lowercase SHA-256")

    config = load_wigatr_config(args.config)
    dataset = FormalDataset.load(args.dataset)
    validate_fixed_radio_contract(dataset)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for relative in (
        "six_condition_results.csv",
        "runtime_provenance.json",
        "execution_manifest.json",
    ):
        if (output / relative).exists():
            raise FileExistsError(f"refusing to overwrite Wi-GATr output: {relative}")

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

    runtime = _load_official_runtime()
    _require_official_cuda(runtime)
    runtime_provenance = collect_external_runtime(
        "wigatr", Path(__file__).resolve().parents[2]
    )
    validate_external_runtime(
        runtime_provenance,
        profile="wigatr",
        executable=sys.executable,
        require_execution_ready=True,
    )
    runtime_path = output / "runtime_provenance.json"
    write_json(runtime_path, runtime_provenance)
    _seed_runtime(runtime, int(config["training"]["seed"]))
    adapter_config_path = output / "adapter_config.json"
    shutil.copyfile(Path(args.config).resolve(), adapter_config_path)
    mesh_audit_path = output / "mesh_surface_audit.json"
    write_json(mesh_audit_path, _audit_mesh_preprocessing(dataset, config))
    mesh_audit_sha256 = sha256_file(mesh_audit_path)
    target_mean, target_std = _source_power_normalization(dataset, config)
    num_materials = int(dataset.metadata["assets"]["material_category_count"])
    model = _build_official_model(runtime, config, num_materials, target_mean, target_std)
    checkpoint, training_record = _fit_source_only_model(
        runtime,
        model,
        dataset,
        config,
        output,
        num_materials,
        target_mean,
        target_std,
        mesh_audit_sha256,
    )
    training_record.update(
        {
            "mesh_surface_audit_path": mesh_audit_path.name,
            "mesh_surface_audit_sha256": mesh_audit_sha256,
        }
    )
    write_json(output / "training_record.json", training_record)

    # Target-derived route objects are constructed only after source-only training is frozen.
    teacher_path = Path(qualification["teacher_checkpoint"])
    teacher = load_teacher_bundle(teacher_path, formal_config)
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
        runtime,
        model,
        dataset,
        routed,
        config,
        num_materials,
    )
    result_path = output / "six_condition_results.csv"
    write_csv(result_path, result_rows)
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "adapter_id": ADAPTER_ID,
        "model_name": MODEL_NAME,
        "implementation_status": "official-code-adaptation",
        "source_revision": WIGATR_SOURCE_REVISION,
        "dataset_sha256": sha256_file(dataset.source_path),
        "adapter_config_path": str(adapter_config_path.relative_to(output)),
        "adapter_config_sha256": sha256_file(adapter_config_path),
        "training_record_path": "training_record.json",
        "training_record_sha256": sha256_file(output / "training_record.json"),
        "checkpoint_path": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "mesh_surface_audit_path": mesh_audit_path.name,
        "mesh_surface_audit_sha256": mesh_audit_sha256,
        "command_sha256": args.command_sha256,
        "results_sha256": sha256_file(result_path),
        "runtime_provenance_path": runtime_path.name,
        "runtime_provenance_sha256": sha256_file(runtime_path),
        "runtime_environment_sha256": runtime_provenance["environment_sha256"],
    }
    write_json(output / "execution_manifest.json", execution)
    return execution


def _load_official_runtime() -> dict:
    vendor = Path(__file__).resolve().parent / "vendor" / "Wi-GATr"
    _verify_vendor_tree(vendor)
    source = vendor / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from torch_geometric.data import Batch
        from torch_geometric.loader import DataLoader
        from wigatr.data.geometric import tokenize_scene
    except ImportError as error:
        raise RuntimeError(
            "official Wi-GATr dependencies are unavailable; run setup_wigatr.sh"
        ) from error
    return {
        "torch": torch,
        "instantiate": instantiate,
        "OmegaConf": OmegaConf,
        "Batch": Batch,
        "DataLoader": DataLoader,
        "tokenize_scene": tokenize_scene,
        "vendor": vendor,
    }


def _build_official_model(runtime, config, num_materials, target_mean, target_std):
    model_config = config["model"]
    payload = {
        "_target_": "wigatr.models.regression_gatr.RSRPRegressionGATr",
        "embedding": model_config["embedding"],
        "affine_shift": float(target_mean),
        "affine_scale": float(target_std),
        "net": {
            "_target_": "gatr.nets.GATr",
            "in_mv_channels": 15,
            "out_mv_channels": 1,
            "hidden_mv_channels": int(model_config["hidden_mv_channels"]),
            "in_s_channels": 4 + int(num_materials),
            "out_s_channels": 1,
            "hidden_s_channels": int(model_config["hidden_s_channels"]),
            "num_blocks": int(model_config["num_blocks"]),
            "reinsert_mv_channels": None,
            "reinsert_s_channels": None,
            "dropout_prob": None,
            "checkpoint_blocks": True,
            "attention": {
                "multi_query": bool(model_config["multi_query"]),
                "in_mv_channels": None,
                "out_mv_channels": None,
                "in_s_channels": None,
                "out_s_channels": None,
                "num_heads": int(model_config["num_heads"]),
                "additional_qk_mv_channels": 0,
                "additional_qk_s_channels": 0,
                "normalizer_eps": 0.001,
                "pos_encoding": False,
                "pos_enc_base": 4096,
                "output_init": "default",
                "checkpoint": False,
                "increase_hidden_channels": 2,
                "dropout_prob": None,
            },
            "mlp": {
                "mv_channels": None,
                "s_channels": None,
                "activation": "gelu",
                "dropout_prob": None,
            },
        },
    }
    return runtime["instantiate"](runtime["OmegaConf"].create(payload))


def _seed_runtime(runtime, seed):
    torch = runtime["torch"]
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_official_cuda(runtime):
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal Wi-GATr requires an NVIDIA CUDA device because the frozen "
            "xFormers attention has no compatible CPU kernel"
        )


def _fit_source_only_model(
    runtime,
    model,
    dataset,
    config,
    output,
    num_materials,
    target_mean,
    target_std,
    mesh_audit_sha256,
):
    torch = runtime["torch"]
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = _PowerDataset(
        runtime,
        dataset,
        config,
        "source_encoder_train",
        num_materials,
    )
    selection_dataset = _PowerDataset(
        runtime,
        dataset,
        config,
        "source_method_selection",
        num_materials,
    )
    generator = torch.Generator().manual_seed(int(training["seed"]))
    loader = runtime["DataLoader"](
        train_dataset,
        batch_size=int(training["microbatch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    selection_loader = runtime["DataLoader"](
        selection_dataset,
        batch_size=int(training["microbatch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["steps"])
    )
    iterator = iter(loader)
    best_state = None
    best_selection = float("inf")
    best_step = None
    last_loss = None
    effective_batch_size = int(training["batch_size"])
    microbatch_size = int(training["microbatch_size"])
    accumulation_steps = effective_batch_size // microbatch_size
    for step in range(1, int(training["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        total_squared_error = 0.0
        total_targets = 0
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = batch.to(device)
            prediction = model(batch)
            squared_error = torch.sum((prediction - batch.y) ** 2)
            target_count = int(batch.y.numel())
            if target_count != microbatch_size:
                raise RuntimeError("Wi-GATr training yielded a partial microbatch")
            (squared_error / effective_batch_size).backward()
            total_squared_error += float(squared_error.detach().cpu())
            total_targets += target_count
        if total_targets != effective_batch_size:
            raise RuntimeError("Wi-GATr gradient accumulation changed the effective batch")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["clip_grad_norm"]), error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        last_loss = total_squared_error / total_targets
        if step % int(training["selection_every_steps"]) == 0 or step == int(training["steps"]):
            selection_loss = _power_mse(model, selection_loader, device, torch)
            if selection_loss < best_selection:
                best_selection = selection_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None or best_step is None:
        raise RuntimeError("Wi-GATr source-method-selection never produced a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = output / "checkpoints" / "wigatr_source_selected.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "csi-pairs-wigatr-checkpoint-v2",
            "source_revision": WIGATR_SOURCE_REVISION,
            "adapter_config": config,
            "dataset_sha256": sha256_file(dataset.source_path),
            "fixture": dataset.is_fixture,
            "train_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "checkpoint_rule": "minimum_source_method_selection_power_mse",
            "selected_step": int(best_step),
            "selection_power_mse": float(best_selection),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "num_materials": int(num_materials),
            "effective_batch_size": effective_batch_size,
            "microbatch_size": microbatch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "mesh_preprocessing": config["mesh"]["preprocessing"],
            "mesh_surface_audit_sha256": mesh_audit_sha256,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return checkpoint, {
        "schema_version": "csi-pairs-wigatr-training-record-v2",
        "source_revision": WIGATR_SOURCE_REVISION,
        "device": str(device),
        "steps": int(training["steps"]),
        "batch_size": int(training["batch_size"]),
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "mesh_preprocessing": config["mesh"]["preprocessing"],
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "last_training_power_mse": float(last_loss),
        "selected_step": int(best_step),
        "selection_power_mse": float(best_selection),
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "target_roles_read": [],
    }


def _evaluate_six_conditions(runtime, model, dataset, routed, config, num_materials):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    units = build_six_condition_units(dataset, routed)
    power_floor = float(config["power"]["floor"])
    bounds = map_bounds(dataset)
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
            input_digests[condition] = supplied_map_sha256(supplied_map)
            predictions[condition] = inverse_localize_power(
                runtime,
                model,
                dataset,
                supplied_map,
                dataset.bs_pose[unit.scene, :3],
                observed_power,
                bounds,
                config,
                num_materials,
                restart_salt=unit.unit_id,
            )
        true_position = dataset.positions[unit.scene, unit.position]
        for condition in SIX_CONDITIONS:
            error = float(np.linalg.norm(predictions[condition] - true_position))
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "model_name": MODEL_NAME,
                    "condition": condition,
                    "city_id": str(dataset.city_ids[unit.scene]),
                    "bank_id": str(dataset.bank_ids[unit.scene]),
                    "position_id": str(dataset.position_ids[unit.scene, unit.position]),
                    "localization_error_m": error,
                    "csi_context_sha256": unit.csi_context_sha256,
                    "base_map_cluster_id": str(dataset.base_map_cluster_ids[unit.scene]),
                    "map_sha256": input_digests[condition],
                    "action_sha256": condition_action_sha256(dataset, unit, condition),
                    "query_count": 1,
                }
            )
    return rows


def inverse_localize_power(
    runtime,
    model,
    dataset,
    supplied_map,
    transmitter_xyz,
    observed_power,
    bounds,
    config,
    num_materials,
    *,
    restart_salt,
) -> np.ndarray:
    """Infer x/y from map, Tx and observed power; true receiver coordinates are not an input."""
    torch = runtime["torch"]
    device = next(model.parameters()).device
    mesh, materials = _mesh_for_map(dataset, supplied_map, config)
    dummy_rx = np.asarray([0.0, 0.0, float(config["inverse"]["receiver_z_m"])])
    graph = runtime["tokenize_scene"](
        torch.as_tensor(transmitter_xyz, dtype=torch.float32),
        torch.as_tensor(dummy_rx, dtype=torch.float32),
        torch.as_tensor(mesh, dtype=torch.float32),
        None,
        torch.as_tensor(materials, dtype=torch.long),
        add_edge_index=False,
        num_materials=int(num_materials),
    )
    batch = runtime["Batch"].from_data_list([graph]).to(device)
    target = torch.as_tensor([[float(observed_power)]], device=device, dtype=torch.float32)
    inverse = config["inverse"]
    salt = hashlib.sha256(restart_salt.encode("utf-8")).digest()
    seed = int(inverse["seed"]) ^ int.from_bytes(salt[:8], "little")
    rng = np.random.default_rng(seed)
    starts = rng.uniform(
        low=np.asarray([bounds[0][0], bounds[1][0]]),
        high=np.asarray([bounds[0][1], bounds[1][1]]),
        size=(int(inverse["restarts"]), 2),
    )
    best_loss = float("inf")
    best_xy = None
    for start in starts:
        xy = torch.tensor(start, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([xy], lr=float(inverse["learning_rate"]))
        for _ in range(int(inverse["steps"])):
            receiver = torch.cat(
                (
                    xy,
                    torch.as_tensor(
                        [float(inverse["receiver_z_m"])], device=device, dtype=torch.float32
                    ),
                )
            )
            prediction = model(batch, overrides={"rx": receiver})
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [xy], float(inverse["clip_grad_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                xy[0].clamp_(float(bounds[0][0]), float(bounds[0][1]))
                xy[1].clamp_(float(bounds[1][0]), float(bounds[1][1]))
        with torch.no_grad():
            receiver = torch.cat(
                (
                    xy,
                    torch.as_tensor(
                        [float(inverse["receiver_z_m"])], device=device, dtype=torch.float32
                    ),
                )
            )
            final_loss = float(
                torch.mean((model(batch, overrides={"rx": receiver}) - target) ** 2).cpu()
            )
        if final_loss < best_loss:
            best_loss = final_loss
            best_xy = xy.detach().cpu().numpy().astype(np.float64)
    if best_xy is None:
        raise RuntimeError("Wi-GATr inverse localization produced no finite candidate")
    return best_xy


class _PowerDataset:
    def __init__(self, runtime, dataset, config, role, num_materials):
        self.runtime = runtime
        self.dataset = dataset
        self.config = config
        self.role = role
        self.num_materials = int(num_materials)
        self.samples = tuple(
            (int(scene), world, position)
            for scene in dataset.indices_for_role(role)
            for world in range(dataset.world_count)
            for position in range(dataset.position_count)
        )
        if not self.samples:
            raise RuntimeError(f"Wi-GATr role {role!r} has no training samples")
        self.mesh_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        torch = self.runtime["torch"]
        scene, world, position = self.samples[index]
        key = (scene, world)
        if key not in self.mesh_cache:
            self.mesh_cache[key] = _mesh_for_map(
                self.dataset, self.dataset.maps[scene, world], self.config
            )
        mesh, materials = self.mesh_cache[key]
        transmitter = self.dataset.bs_pose[scene, :3]
        receiver = np.asarray(
            [
                self.dataset.positions[scene, position, 0],
                self.dataset.positions[scene, position, 1],
                float(self.config["inverse"]["receiver_z_m"]),
            ],
            dtype=np.float32,
        )
        target = relative_total_power_db(
            self.dataset.csi_clean[scene, world, position],
            float(self.config["power"]["floor"]),
        )
        return self.runtime["tokenize_scene"](
            torch.as_tensor(transmitter, dtype=torch.float32),
            torch.as_tensor(receiver, dtype=torch.float32),
            torch.as_tensor(mesh, dtype=torch.float32),
            torch.as_tensor([float(target)], dtype=torch.float32),
            torch.as_tensor(materials, dtype=torch.long),
            add_edge_index=False,
            num_materials=self.num_materials,
        )


def _power_mse(model, loader, device, torch):
    model.eval()
    squared = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            squared += float(torch.sum((prediction - batch.y) ** 2).cpu())
            count += int(batch.y.numel())
    model.train()
    if count == 0:
        raise RuntimeError("Wi-GATr selection role is empty")
    return squared / count


def _source_power_normalization(dataset, config):
    scenes = dataset.indices_for_role("source_encoder_train")
    values = relative_total_power_db(
        dataset.csi_clean[scenes], float(config["power"]["floor"])
    )
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("Wi-GATr source power normalization is degenerate")
    return mean, scale


def _mesh_for_map(dataset, maps, config):
    representation = dataset.metadata["representation"]
    return grid_to_triangular_mesh(
        maps,
        tuple(str(value) for value in dataset.map_channel_names.tolist()),
        resolution_m=float(representation["map_resolution_m"]),
        origin_xy_m=tuple(float(value) for value in representation["map_origin_xy_m"]),
        occupancy_threshold=float(config["mesh"]["occupancy_threshold"]),
        minimum_height_m=float(config["mesh"]["minimum_height_m"]),
    )


def _audit_mesh_preprocessing(dataset, config):
    representation = dataset.metadata["representation"]
    channel_names = tuple(str(value) for value in dataset.map_channel_names.tolist())
    resolution = float(representation["map_resolution_m"])
    origin = tuple(float(value) for value in representation["map_origin_xy_m"])
    arguments = {
        "resolution_m": resolution,
        "origin_xy_m": origin,
        "occupancy_threshold": float(config["mesh"]["occupancy_threshold"]),
        "minimum_height_m": float(config["mesh"]["minimum_height_m"]),
    }
    digest = hashlib.sha256()
    map_count = 0
    reference_face_count = 0
    compact_face_count = 0
    role_counts: dict[str, int] = {}
    for scene in range(dataset.scene_count):
        role = str(dataset.scene_roles[scene])
        role_counts[role] = role_counts.get(role, 0) + dataset.world_count
        for world in range(dataset.world_count):
            maps = dataset.maps[scene, world]
            compact_mesh, compact_materials = grid_to_triangular_mesh(
                maps,
                channel_names,
                **arguments,
            )
            map_ledger_sha256, atomic_quad_count = (
                require_compact_mesh_matches_map_surface(
                    maps,
                    channel_names,
                    compact_mesh,
                    compact_materials,
                    **arguments,
                )
            )
            identity = f"{dataset.bank_ids[scene]}:{world}".encode("utf-8")
            digest.update(len(identity).to_bytes(8, "big"))
            digest.update(identity)
            digest.update(bytes.fromhex(map_ledger_sha256))
            map_count += 1
            reference_face_count += 2 * atomic_quad_count
            compact_face_count += int(compact_mesh.shape[0])
    expected_map_count = dataset.scene_count * dataset.world_count
    if map_count != expected_map_count:
        raise RuntimeError("Wi-GATr mesh audit did not cover the full formal map bank")
    return {
        "schema_version": "csi-pairs-v6-wigatr-mesh-surface-audit-v1",
        "status": "PASS",
        "passed": True,
        "dataset_sha256": sha256_file(dataset.source_path),
        "preprocessing": config["mesh"]["preprocessing"],
        "scene_count": int(dataset.scene_count),
        "world_count": int(dataset.world_count),
        "map_count": int(map_count),
        "role_map_counts": role_counts,
        "reference_face_count": reference_face_count,
        "compact_face_count": compact_face_count,
        "canonical_surface_ledger_sha256": digest.hexdigest(),
        "rule": (
            "Every scene/world map, including held-out roles, is checked without reading CSI "
            "or labels; each compact mesh must exactly match independent grid-derived "
            "top and directed-side geometry/material surface arrays."
        ),
    }


def _verify_vendor_tree(vendor):
    sums = vendor / "VENDOR_SHA256SUMS"
    if not sums.is_file():
        raise RuntimeError("Wi-GATr vendor checksum manifest is missing")
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (vendor / relative).resolve()
        if vendor.resolve() not in path.parents or not path.is_file():
            raise RuntimeError("Wi-GATr vendor checksum path is missing or escapes the snapshot")
        if sha256_file(path) != digest:
            raise RuntimeError(f"Wi-GATr vendor file hash mismatch: {relative}")


def _lower_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":
    raise SystemExit(main())
