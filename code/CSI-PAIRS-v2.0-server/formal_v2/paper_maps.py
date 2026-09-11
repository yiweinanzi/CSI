"""Source-trained map methods with shared few-shot localization and diagnostics.

Wi-GATr and PMNet are power-prediction methods. The main-table adaptation uses
their native inverse estimate as frozen features for the shared position head;
six-condition diagnostics report the native inverse directly. These are distinct
protocols, never interchangeable results or claims of native CSI localization.
"""
from __future__ import annotations
import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from . import formal_factorial as factorial
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import config_sha256
from .formal_io import write_csv
from .formal_routing import fit_route_normalization, route_dataset
from .formal_teacher import load_teacher_bundle
from .paper_suite import PROJECT, atomic_json, digest, suite_lock
from .external_adapters import pmnet_adapter as pm, wigatr_adapter as wg, controlled_map_adapter as cm
from .external_adapters.wigatr_protocol import (
    SIX_CONDITIONS, build_six_condition_units, condition_map, map_bounds,
    relative_total_power_db, supplied_map_sha256, SixConditionUnit,
)

METHODS = {"Wi-GATr": "wigatr_official_v1.json", "PMNet": "pmnet_official_v1.json",
           "SigMap": "sigmap_controlled_v1.json", "WiSER": "wiser_controlled_v1.json", "RFIR": "rfir_controlled_v1.json"}


def shared_units(root, dataset, config, device):
    """One registry for all runtimes, including the older official Wi-GATr Torch."""
    identity = {"dataset_sha256": digest(dataset.source_path), "config_sha256": config_sha256(config),
                "teacher_sha256": digest(root / "teacher.pt")}
    with suite_lock(root / "map_methods/unit_registry"):
        path = root / "map_methods/unit_registry/units.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity:
                raise ValueError("Map diagnostic inputs changed; use a new run")
            return [SixConditionUnit(**row) for row in record["units"]]
        teacher = load_teacher_bundle(root / "teacher.pt", config, device=device)
        scenes = np.concatenate((dataset.indices_for_role("source_final_unseen_bank"), dataset.indices_for_role("target")))
        routed = route_dataset(dataset, teacher, config, scenes, normalization=fit_route_normalization(dataset, teacher))
        units = build_six_condition_units(dataset, routed)
        atomic_json(path, {"identity": identity, "units": [asdict(unit) for unit in units]})
        return units


class MapMethod:
    def __init__(self, name, dataset, output, seed):
        self.name, self.dataset, self.output = name, dataset, output
        path = Path(__file__).parent / "configs" / METHODS[name]
        # Algorithm settings, with no resource certificates or approval stage.
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.config["training"]["seed"] = seed
        torch.manual_seed(seed)
        self.runtime = None
        materials = int(dataset.metadata["assets"]["material_category_count"])
        if name == "Wi-GATr":
            self.runtime = wg._load_official_runtime()
            self.power = wg._source_power_normalization(dataset, self.config)
            self.model = wg._build_official_model(self.runtime, self.config, materials, *self.power)
        elif name == "PMNet":
            self.power = pm._fit_source_power_normalization(dataset, self.config["power"]["floor"])
            self.model = pm._build_official_model(self.config, materials)
        else:
            self.normalizer = cm._fit_data_normalizer(dataset)
            self.model, self.metadata = cm._build_model(self.config, dataset)
        complete = output / "trained.json"
        if complete.exists():
            record = json.loads(complete.read_text(encoding="utf-8"))
            checkpoint = output / record["checkpoint"]
            if digest(checkpoint) != record["sha256"]:
                raise ValueError("Baseline checkpoint changed")
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        else:
            output.mkdir(parents=True, exist_ok=True)
            if name == "Wi-GATr":
                checkpoint, training = wg._fit_source_only_model(self.runtime, self.model, dataset, self.config,
                    output, materials, *self.power, mesh_audit_sha256="not_required_by_paper_protocol")
            elif name == "PMNet":
                checkpoint, training = pm._fit_source_only_model(self.model, dataset, self.config, output, *self.power)
            else:
                checkpoint, training = cm._train(self.model, self.config, dataset, self.normalizer, output, self.metadata)
            atomic_json(complete, {"checkpoint": checkpoint.relative_to(output).as_posix(),
                                   "sha256": digest(checkpoint), "training": training})
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cache = {}

    def predict(self, scene, observed, supplied_map, salt):
        dataset, config = self.dataset, self.config
        if self.name == "Wi-GATr":
            return wg.inverse_localize_power(self.runtime, self.model, dataset, supplied_map,
                dataset.bs_pose[scene, :3], float(relative_total_power_db(observed, config["power"]["floor"])),
                map_bounds(dataset), config, int(dataset.metadata["assets"]["material_category_count"]), restart_salt=salt)
        if self.name == "PMNet":
            key = (scene, supplied_map_sha256(supplied_map))
            if key not in self.cache:
                # Bounded cache: at most the maps for the current scene.
                if self.cache and next(iter(self.cache))[0] != scene:
                    self.cache.clear()
                self.cache[key] = pm._predict_radiomap(self.model, dataset, scene, supplied_map, *self.power)
            representation = dataset.metadata["representation"]
            return pm._inverse_localize_radiomap(self.cache[key], supplied_map,
                float(relative_total_power_db(observed, config["power"]["floor"])),
                representation["map_origin_xy_m"], representation["map_resolution_m"],
                config["inverse"]["occupancy_threshold"], occupancy_index=list(dataset.map_channel_names).index("occupancy"))
        if self.name == "SigMap":
            return cm._sigmap_predict(self.model, dataset, scene, observed, supplied_map, self.normalizer)
        return cm._inverse_predict(self.model, config, dataset, scene, observed, supplied_map, self.normalizer)

    def features(self, seed, name, scenes):
        result = {}
        root = self.output / "features"
        root.mkdir(exist_ok=True)
        for scene in scenes:
            path = root / f"{scene}.json"
            if path.exists():
                result[scene] = np.asarray(json.loads(path.read_text()), dtype=np.float32)
                continue
            world = int(self.dataset.natural_world_index[scene])
            values = []
            for position in range(self.dataset.position_count):
                observed = self.dataset.csi[scene, world, position]
                prediction = self.predict(scene, observed, self.dataset.maps[scene, world], f"natural:{scene}:{position}")
                values.append(np.asarray(prediction).tolist())
            atomic_json(path, values)
            result[scene] = np.asarray(values, dtype=np.float32)
            print(f"{name} seed={seed} natural features scene={scene}", flush=True)
        return result


def run(args):
    config = load_formal_config(args.config)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    root = args.run_root.resolve()
    dataset = FormalDataset.load(args.dataset, array_cache=root / "npz-cache")
    if dataset.is_fixture and not args.allow_fixture:
        raise ValueError("Fixture needs --allow-fixture")
    identity = {"dataset_sha256": digest(args.dataset), "config_sha256": config_sha256(config),
                "adapter_config_sha256": digest(Path(__file__).parent / "configs" / METHODS[args.method]),
                "fixture": dataset.is_fixture, "scientific_use": "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"}
    units = shared_units(root, dataset, config, args.device)
    name = args.method
    with suite_lock(root / "map_methods" / name):
        for seed in config["seeds"]:
            output = root / "map_methods" / name / str(seed)
            output.mkdir(parents=True, exist_ok=True)
            identity_path = output / "identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError("Baseline inputs changed; use a new run")
            atomic_json(identity_path, identity)
            if (output / "complete.json").exists():
                continue
            method = MapMethod(name, dataset, output, int(seed))
            if name in {"Wi-GATr", "PMNet"}:
                one_seed = {**config, "seeds": [seed]}
                banks, samples = factorial._run_localization(one_seed, dataset, {}, None, None, ["cpu"],
                    methods=[name], feature_provider=method.features)
                for filename, rows in (("localization_per_bank.csv", banks), ("localization_per_sample.csv", samples)):
                    write_csv(output / filename, [{**row, **identity, "adaptation": "native_inverse_features_shared_position_head"} for row in rows])
            diagnostics = []
            unit_root = output / "diagnostics"
            unit_root.mkdir(exist_ok=True)
            for index, unit in enumerate(units):
                path = unit_root / f"{index}.json"
                if path.exists():
                    rows = json.loads(path.read_text())
                else:
                    rows = []
                    observed = dataset.csi[unit.scene, unit.source_world, unit.position]
                    for condition in SIX_CONDITIONS:
                        supplied = condition_map(dataset, unit, condition)
                        prediction = method.predict(unit.scene, observed, supplied, unit.unit_id)
                        rows.append({**identity, "seed": seed, "unit_id": unit.unit_id, "model_name": name,
                            "condition": condition, "city_id": str(dataset.city_ids[unit.scene]),
                            "bank_id": str(dataset.bank_ids[unit.scene]), "position_id": str(dataset.position_ids[unit.scene, unit.position]),
                            "base_map_cluster_id": str(dataset.base_map_cluster_ids[unit.scene]),
                            "canonical_base_map_digest": dataset.canonical_base_map_digest(unit.scene),
                            "localization_error_m": float(np.linalg.norm(prediction - dataset.positions[unit.scene, unit.position])),
                            "observations": "measured_csi_same_for_all_methods"})
                    atomic_json(path, rows)
                diagnostics.extend(rows)
                if index % 25 == 0:
                    print(f"{name} seed={seed} diagnostics {index}/{len(units)}", flush=True)
            write_csv(output / "six_condition_results.csv", diagnostics)
            atomic_json(output / "complete.json", {"status": "COMPLETE", **identity})
            del method
            gc.collect()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/paper_formal.json")
    parser.add_argument("--allow-fixture", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
