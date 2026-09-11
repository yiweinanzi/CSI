"""Independent core evaluation, without publication approval/runtime closure gates.

This is a NEW probe protocol. It never overwrites historical formal artifacts.
Training data, frozen teacher, encoder checkpoints and source selection are still
checked. Optional risk/path/shortcut/control experiments do not run here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from . import formal_evaluation as evaluation
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import config_sha256
from .formal_factorial import TrainingNormalization, _normalization_record
from .formal_io import write_csv
from .formal_metrics import binary_auroc
from .formal_model import CSIPairsFormalModel
from .formal_probes import predict_binary_probe, predict_response_probe
from .formal_routing import fit_route_normalization, route_dataset, ROUTE_NAMES
from .formal_teacher import load_teacher_bundle
from .paper_arrays import concatenate_disk
from .paper_minibatch import select_probe
from .paper_suite import PROJECT, atomic_json, digest, suite_lock

TASK_FILES = {"alignment": ["cgs_per_bank.csv", "compatibility_route_distributions.csv"],
              "response": ["response_per_bank.csv"], "native": ["native_per_bank.csv"]}
SHORTCUTS = evaluation._alignment_shortcut_definitions()
VARIANTS = ("without_map", "edit_only", "csi_only", "oracle_x")
TASK_FILES.update({"alignment:" + key: ["alignment_shortcut_baselines.csv"] for key in SHORTCUTS})
TASK_FILES.update({"response:" + key: ["response_per_bank.csv"] for key in VARIANTS})


def is_binary(task):
    return task.split(":")[0] == "alignment"


def shortcut_scene(name, dataset, teacher, config, normalization, scene, route_normalization, *, train):
    """Raw input controls need no encoder forward passes or unused feature banks."""
    from .formal_protocol import headline_alignment_edge
    routed = route_dataset(dataset, teacher, config, [scene], normalization=route_normalization)
    features, labels, routes = [], [], []
    radio = evaluation._normalized_radio(normalization, dataset.radio_config[scene], dataset.bs_pose[scene]).reshape(-1)
    for edge in dataset.directed_edges(scene):
        if edge.source_world >= edge.target_world or not headline_alignment_edge(dataset, scene, edge):
            continue
        for position in evaluation._eligible_evaluation_positions(dataset, scene):
            route = routed.alignment_route[(scene, edge.source_world, edge.target_world, position)]
            if train and route != 2:
                continue
            for csi_world, matched, alternative in ((edge.source_world, edge.source_world, edge.target_world),
                                                   (edge.target_world, edge.target_world, edge.source_world)):
                for supplied_world, label in ((matched, 1), (alternative, 0)):
                    if name == "constant":
                        values = np.zeros(1)
                    elif name == "csi_only":
                        patches = evaluation._normalized_scene_patches(dataset, normalization, teacher.patch_spec, scene, csi_world, position)
                        values = np.concatenate((patches.reshape(-1), radio))
                    elif name == "map_only":
                        maps = evaluation._normalized_map(normalization, dataset.maps[scene, supplied_world])
                        values = np.concatenate((maps.reshape(-1), radio))
                    else:
                        metadata = evaluation._shortcut_metadata_features(dataset, scene, csi_world, supplied_world)
                        values = metadata[("scene_id_only", "edit_status_xor", "variant_id_matcher").index(name)]
                    features.append(values)
                    labels.append(label)
                    routes.append(str(ROUTE_NAMES[route]))
    return {"features": np.asarray(features), "labels": np.asarray(labels), "routes": np.asarray(routes)}


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != "csi-pairs-paper-core-v1" or profile.get("selection_role") != "source_probe_selection":
        raise ValueError("Unsupported core protocol or non-source selection role")
    if not profile.get("probe_updates") or any(type(v) is not int or v < 1 for v in profile["probe_updates"]):
        raise ValueError("Probe updates must be positive integers")
    if profile["probe_updates"] != sorted(set(profile["probe_updates"])):
        raise ValueError("Probe update budgets must be sorted and unique")
    for key in ("batch_rows", "encoding_batch_rows"):
        if type(profile[key]) is not int or profile[key] < 1:
            raise ValueError("Invalid batch size")
    if profile.get("feature_dtype") != "float32" or set(profile["tasks"]) - set(TASK_FILES):
        raise ValueError("Unsupported feature dtype or task")
    if not profile["tasks"] or len(set(profile["tasks"])) != len(profile["tasks"]):
        raise ValueError("Core tasks must be nonempty and unique")
    return profile


def runtime_record():
    # Versions and actual project source, never an installed-wheel closure scan.
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    return {"python": platform.python_version(), "executable": sys.executable,
            "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda": torch.version.cuda, "source_sha256": source.hexdigest()}


def configure_runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def frozen_model(path, expected_hash, teacher_hash, dataset_hash, config, device):
    if digest(path) != expected_hash:
        raise ValueError("Encoder checkpoint hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in (("schema_version", "csi-pairs-formal-checkpoint-v2.1-v6"),
                          ("dataset_sha256", dataset_hash), ("config_sha256", config_sha256(config)),
                          ("teacher_checkpoint_sha256", teacher_hash)):
        if payload.get(key) != expected:
            raise ValueError("Encoder checkpoint identity mismatch: " + key)
    model = CSIPairsFormalModel(**payload["model_spec"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if payload["normalization"].get("source_roles") != ["source_encoder_train"]:
        raise ValueError("Encoder normalization was not fitted on source training data")
    normalization = TrainingNormalization(**{key: np.asarray(value, dtype=np.float64) for key, value in payload["normalization"].items() if key != "source_roles"})
    return model, normalization, payload


def feature_scene(task, model, dataset, teacher, config, normalization, scene, route_normalization, batch_rows, *, train=False):
    options = dict(route_normalization=route_normalization, batch_size=batch_rows)
    kind, _, variant = task.partition(":")
    if kind == "alignment" and variant:
        return shortcut_scene(variant, dataset, teacher, config, normalization, scene, route_normalization, train=train)
    if kind == "alignment":
        data = evaluation._compatibility_dataset(model, dataset, teacher, config, normalization, [scene],
                                                active_only=train, include_shortcuts=False, **options)
    else:
        data = evaluation._response_probe_dataset(model, dataset, teacher, config, normalization, [scene],
                                                  active_only=False, include_variants=bool(variant), **options)
        if variant:
            data["features"] = data[variant + "_features"]
            data["no_action_features"] = data.get(variant + "_zero_action_features")
    return data


def source_arrays(task, role, root, model, dataset, teacher, config, normalization, route_normalization, profile):
    if role not in {"source_probe_train", "source_probe_selection"}:
        raise ValueError("Probe fitting/selection cannot consume target data")
    root = root / role
    root.mkdir(parents=True, exist_ok=True)
    parts = {key: [] for key in (("x", "y") if is_binary(task) or task == "response:csi_only" else ("x", "y", "zero"))}
    for scene in dataset.indices_for_role(role):
        paths = {key: root / f"{int(scene)}-{key}.npy" for key in parts}
        receipt = root / f"{int(scene)}.json"
        if receipt.exists():
            hashes = json.loads(receipt.read_text(encoding="utf-8"))
            if any(hashes.get(key) != digest(path) for key, path in paths.items()):
                raise ValueError("Source feature cache changed")
        else:
            print(json.dumps({"task": task, "stage": "source_features", "role": role, "scene": int(scene)}), flush=True)
            data = feature_scene(task, model, dataset, teacher, config, normalization, int(scene), route_normalization, profile["encoding_batch_rows"], train=True)
            arrays = {"x": data["features"], "y": data["labels"] if is_binary(task) else data["targets"] - data["source_targets"]}
            if "zero" in parts:
                arrays["zero"] = data["no_action_features"]
            for key, path in paths.items():
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.save(stream, np.asarray(arrays[key], dtype=np.float32), allow_pickle=False)
                os.replace(temp, path)
            atomic_json(receipt, {key: digest(path) for key, path in paths.items()})
            del data, arrays
            gc.collect()
        for key, path in paths.items():
            parts[key].append(path)
    combined = {key: concatenate_disk(paths, root / (key + ".npy")) for key, paths in parts.items()}
    return combined["x"], combined["y"], combined.get("zero")


def score_scene(task, probe, data, base, config, batch_rows):
    if is_binary(task):
        scores = predict_binary_probe(probe, data["features"], batch_rows=batch_rows)
        active = data["routes"] == "active"
        cgs = {**base, "cgs_auroc": binary_auroc(data["labels"][active], scores[active]) if np.any(active) else None, "n": int(active.sum())}
        if ":" in task:
            variant = task.split(":")[1]
            return {"alignment_shortcut_baselines.csv": [{**base, "baseline": variant,
                "unseen_bank_auroc": cgs["cgs_auroc"], "n": cgs["n"], "input_contract": SHORTCUTS[variant][1]}]}
        null = data["routes"] == "null"
        differences = evaluation._paired_score_differences(scores[null], data["labels"][null], data["pair_ids"][null]) if np.any(null) else np.array([])
        route = {**base, "route": "null", "overclassification_rate": float(np.mean(np.abs(differences) > config["evaluation"]["null_score_equivalence_margin"])) if len(differences) else None, "pair_count": len(differences)}
        return {"cgs_per_bank.csv": [cgs], "compatibility_route_distributions.csv": [route]}
    delta = predict_response_probe(probe, data["features"], data["no_action_features"], batch_rows=batch_rows)
    mask = data["routes"] == "active"
    denominator = max(float(np.sum(data["targets"][mask] ** 2)), 1e-12)
    nmse = float(np.sum((data["source_targets"][mask] + delta[mask] - data["targets"][mask]) ** 2) / denominator) if np.any(mask) else None
    field = "probe_" + task.split(":")[1] + "_active_patch_nmse" if ":" in task else "unified_response_probe_active_patch_nmse"
    row = {**base, field: nmse, "n": int(mask.sum())}
    if task == "response":
        row["probe_copy_active_patch_nmse"] = float(np.sum((data["source_targets"][mask] - data["targets"][mask]) ** 2) / denominator) if np.any(mask) else None
    return {"response_per_bank.csv": [row]}


def run_job(task, root, model, dataset, teacher, config, normalization, route_norm, profile, base, *, source_root=None):
    root.mkdir(parents=True, exist_ok=True)
    done = root / "complete.json"
    if done.exists():
        receipt = json.loads(done.read_text(encoding="utf-8"))
        if any(digest(root / name) != value for name, value in receipt["files"].items()):
            raise ValueError("Completed core task artifacts changed")
        return
    probe = None
    if task != "native":
        source_root = root if source_root is None else source_root
        train = source_arrays(task, "source_probe_train", source_root, model, dataset, teacher, config, normalization, route_norm, profile)
        selection = source_arrays(task, "source_probe_selection", source_root, model, dataset, teacher, config, normalization, route_norm, profile)
        def progress(event):
            if event["step"] == 1 or event["step"] % 25 == 0:
                event = {"task": task, "stage": "probe_fit", **event}
                atomic_json(root / "progress.json", event)
                print(json.dumps(event), flush=True)
        probe, record = select_probe(train, selection, binary=is_binary(task), profile=profile, config=config,
            seed=int(base["seed"]) + (31001 if is_binary(task) else 32001), device=next(model.parameters()).device, output=root / "probes", callback=progress)
        atomic_json(root / "selection.json", record)
        del train, selection
        gc.collect()
    rows = {name: [] for name in TASK_FILES[task]}
    for scene in dataset.indices_for_role("target"):
        scene = int(scene)
        scene_path = root / f"scene-{scene}.json"
        if scene_path.exists():
            result = json.loads(scene_path.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"task": task, "stage": "target_score", "scene": scene}), flush=True)
            metadata = {**base, "bank_id": str(dataset.bank_ids[scene]), "city_id": str(dataset.city_ids[scene]),
                        "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]), "canonical_base_map_digest": dataset.canonical_base_map_digest(scene)}
            if task == "native":
                metrics = evaluation._native_mask_cover_metrics(model, dataset, teacher, config, normalization, [scene], str(dataset.bank_ids[scene]))
                result = {"native_per_bank.csv": [{**metadata, **metrics}]}
            else:
                data = feature_scene(task, model, dataset, teacher, config, normalization, scene, route_norm, profile["encoding_batch_rows"])
                result = score_scene(task, probe, data, metadata, config, profile["batch_rows"])
                del data
            atomic_json(scene_path, result)
        for name in rows:
            rows[name].extend(result[name])
    for name, values in rows.items():
        write_csv(root / name, values)
    files = {name: digest(root / name) for name in rows}
    if task != "native":
        files["selection.json"] = digest(root / "selection.json")
    atomic_json(done, {"status": "COMPLETE", "files": files, "checkpoint_sha256": base["checkpoint_sha256"]})


def run(args):
    profile = load_profile(args.profile)
    tasks = args.tasks or profile["tasks"]
    config = load_formal_config(args.config)
    root = args.output.resolve()
    upstream = args.factorial_root.resolve()
    if root == upstream or upstream in root.parents or root in upstream.parents:
        raise ValueError("Core output must be separate from original factorial artifacts")
    configure_runtime()
    runtime = runtime_record()
    identity = {"schema_version": "csi-pairs-paper-core-run-v1", "profile": profile,
                "dataset_sha256": digest(args.dataset), "teacher_sha256": digest(args.teacher),
                "checkpoint_index_sha256": digest(upstream / "checkpoint_index.json"), "config_sha256": config_sha256(config), "device": args.device}
    with suite_lock(root):
        identity_path = root / "identity.json"
        if not identity_path.exists() and any(path.name != "suite.lock" for path in root.iterdir()):
            raise ValueError("New core output must be empty; existing files will not be overwritten")
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Core inputs/protocol/runtime changed; use a new output directory")
        atomic_json(identity_path, identity)
        atomic_json(root / "runtime.json", runtime)
        dataset = FormalDataset.load(args.dataset, array_cache=args.array_cache or root / "npz-cache")
        if dataset._storage_source_sha256 != identity["dataset_sha256"]:
            raise ValueError("Dataset changed during run initialization")
        if dataset.is_fixture and not args.allow_fixture:
            raise ValueError("Fixture execution requires --allow-fixture; its results remain FORBIDDEN")
        teacher = load_teacher_bundle(args.teacher, config, device=args.device)
        route_norm = fit_route_normalization(dataset, teacher)
        index = json.loads((upstream / "checkpoint_index.json").read_text(encoding="utf-8"))
        jobs = [row for row in index["checkpoints"] if not args.arms or row["arm"] in args.arms]
        if not jobs or len({(row["seed"], row["arm"]) for row in jobs}) != len(jobs):
            raise ValueError("Missing or duplicate encoder jobs")
        for row in jobs:
            checkpoint = (upstream / row["path"]).resolve()
            if upstream not in checkpoint.parents:
                raise ValueError("Checkpoint path escapes factorial root")
            model, normalization, payload = frozen_model(checkpoint, row["sha256"], identity["teacher_sha256"], identity["dataset_sha256"], config, args.device)
            if (payload["arm"], payload["seed"], payload["fixture"]) != (row["arm"], row["seed"], dataset.is_fixture):
                raise ValueError("Encoder arm/seed/fixture identity mismatch")
            base = {"seed": row["seed"], "arm": row["arm"], "dataset_sha256": identity["dataset_sha256"],
                    "config_sha256": identity["config_sha256"], "checkpoint_sha256": row["sha256"],
                    "evaluation_protocol_sha256": digest(args.profile), "source_tree_sha256": runtime["source_sha256"],
                    "fixture": dataset.is_fixture, "scientific_use": "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
                    "evaluation_protocol": "paper-core-minibatch-v1"}
            for task in tasks:
                destination = root / "jobs" / f"{row['seed']}-{row['arm']}" / task.replace(":", "-")
                if task.startswith("alignment:"):
                    # Identical raw input controls should be fitted once per
                    # algorithm seed, not four times for identical arm inputs.
                    norm_key = hashlib.sha256(json.dumps(_normalization_record(normalization), sort_keys=True).encode()).hexdigest()[:16]
                    shared = root / "input_controls" / f"{row['seed']}-{task.split(':')[1]}-{norm_key}"
                    common = {**base, "arm": "input_only", "checkpoint_sha256": "NOT_APPLICABLE_INPUT_CONTROL"}
                    run_job(task, shared, model, dataset, teacher, config, normalization, route_norm, profile, common,
                            source_root=root / "input_features" / f"{task.split(':')[1]}-{norm_key}")
                    import csv
                    destination.mkdir(parents=True, exist_ok=True)
                    for name in TASK_FILES[task]:
                        with (shared / name).open(encoding="utf-8", newline="") as stream:
                            values = [{**value, "arm": row["arm"], "shared_input_control": True} for value in csv.DictReader(stream)]
                        write_csv(destination / name, values)
                    atomic_json(destination / "complete.json", {"status": "COMPLETE", "files": {name: digest(destination / name) for name in TASK_FILES[task]}})
                else:
                    run_job(task, destination, model, dataset, teacher, config, normalization, route_norm, profile, base)
                # Publish completed task metrics immediately; appendices and
                # other checkpoints must not hold the core table hostage.
                merge_outputs(root, index["checkpoints"])
            del model
            gc.collect()
        merge_outputs(root, index["checkpoints"])
    return 0


def merge_outputs(root, jobs):
    import csv
    output = root / "evaluation"
    output.mkdir(exist_ok=True)
    merged = {}
    for task, names in TASK_FILES.items():
        for name in names:
            values = []
            for row in jobs:
                job_root = root / "jobs" / f"{row['seed']}-{row['arm']}" / task.replace(":", "-")
                path = job_root / name
                receipt_path = job_root / "complete.json"
                if receipt_path.is_file():
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if receipt.get("files", {}).get(name) != digest(path):
                        raise ValueError("Cannot merge a changed core artifact")
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        values.extend(csv.DictReader(stream))
            if values:
                merged.setdefault(name, []).extend(values)
    response = {}
    for row in merged.get("response_per_bank.csv", []):
        response.setdefault((row["seed"], row["arm"], row["bank_id"]), {}).update(row)
    if response:
        merged["response_per_bank.csv"] = list(response.values())
    native = {(r["seed"], r["arm"], r["bank_id"]): r for r in merged.get("native_per_bank.csv", [])}
    for row in merged.get("response_per_bank.csv", []):
        other = native.get((row["seed"], row["arm"], row["bank_id"]), {})
        row.update({k: v for k, v in other.items() if k.startswith("native_")})
    for name, values in merged.items():
        write_csv(output / name, values)
    atomic_json(root / "status.json", {"schema_version": "csi-pairs-paper-core-status-v1", "tasks": {
        task: {"complete": sum((root / "jobs" / f"{row['seed']}-{row['arm']}" / task.replace(":", "-") / "complete.json").exists() for row in jobs), "total": len(jobs)} for task in TASK_FILES}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/formal_v2.json")
    parser.add_argument("--profile", type=Path, default=PROJECT / "formal_v2/configs/paper_core.json")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_FILES))
    parser.add_argument("--arms", nargs="+", choices=("endpoint", "alignment", "response", "full"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--array-cache", type=Path)
    parser.add_argument("--allow-fixture", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
