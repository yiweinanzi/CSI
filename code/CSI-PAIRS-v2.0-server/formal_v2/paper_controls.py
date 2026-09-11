"""Matched-resource, shuffled-pair and original-recipe controls; no approval stage."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import formal_factorial as f
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import config_sha256
from .formal_io import write_csv
from .formal_model import portable_state_dict
from .formal_source_guard import SourceOnlyDataset
from .formal_teacher import load_teacher_bundle
from .formal_routing import fit_route_normalization
from .formal_training_resume import _atomic_torch_save
from .paper_core import run_job, load_profile
from .paper_suite import PROJECT, atomic_json, digest, suite_lock
from .external_adapters import resource_control as resource

CONTROLS = (*resource.CONTROL_ROLES, "shuffled_full", "original_recipe_full")


def shuffled_pairing(corpus, seed):
    """Permute source pairs globally, preserving marginals and query indices.

This is explicitly a global pairing control, not the legacy within-scene,
effect-bucket control (which cannot exist when a bucket has one map edge).
"""
    groups = {"alignment": {}, "response": {}}
    alignment = sorted({(int(scene), *map(int, unit)) for field in (corpus.alignment_active, corpus.alignment_null)
                        for scene, units in field.items() for unit in units})
    groups["alignment"][None] = alignment
    for scene, units in corpus.response_all.items():
        for unit in units:
            value = (int(scene), *map(int, unit))
            groups["response"].setdefault(value[-1], []).append(value)
    mappings, rows = {}, []
    rng = np.random.default_rng(int(seed) + 73001)
    for branch, partitions in groups.items():
        mappings[branch] = {}
        for query, values in partitions.items():
            values = sorted(set(values))
            if branch == "alignment":
                by_map = {}
                for unit in values:
                    by_map.setdefault(unit[:3], []).append(unit)
                keys = list(by_map)
                keys = [keys[i] for i in rng.permutation(len(keys))]
                ordered = [unit for key in keys for unit in by_map[key]]
                offset = max(map(len, by_map.values()))
                if 2 * offset > len(ordered):
                    raise ValueError("Global shuffled alignment needs at least two sufficiently represented source map edges")
            else:
                ordered = [values[i] for i in rng.permutation(len(values))]
                offset = 1
            if len(ordered) < 2:
                raise ValueError("Shuffled response needs at least two source pairs per query")
            partners = ordered[offset:] + ordered[:offset]
            for original, partner in zip(ordered, partners):
                if original == partner or (branch == "alignment" and original[:3] == partner[:3]):
                    raise RuntimeError("Pairing permutation did not break the input-target correspondence")
                mappings[branch][original] = partner
                rows.append({"seed": seed, "branch": branch, "query": query,
                             "original_pair_id": ":".join(map(str, original)),
                             "permuted_pair_id": ":".join(map(str, partner)), "protocol": "global_source_pair_permutation"})
    return mappings, rows


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run(args):
    config = load_formal_config(args.config)
    root = args.run_root.resolve()
    dataset = FormalDataset.load(args.dataset, array_cache=root / "npz-cache")
    if dataset.is_fixture and not args.allow_fixture:
        raise ValueError("Fixture needs --allow-fixture")
    evidence = {"dataset_sha256": digest(args.dataset), "config_sha256": config_sha256(config),
                "fixture": dataset.is_fixture, "scientific_use": "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"}
    teacher = load_teacher_bundle(root / "teacher.pt", config, device=args.device)
    source = SourceOnlyDataset(dataset)
    route_norm = fit_route_normalization(source, teacher)
    normalization = f._training_normalization(source, source.indices_for_role("source_encoder_train"), route_norm, teacher.patch_spec)
    corpus = f._build_corpus(source, source.indices_for_role("source_encoder_train"), teacher, config, route_norm, normalization)
    pilot = json.loads((root / "factorial/frozen_pilot.json").read_text())
    training = read_csv(root / "factorial/training_summary.csv")
    profile = load_profile(args.profile)
    with suite_lock(root / "paper_controls"):
        identity = {**evidence, "probe_profile": profile, "pilot_sha256": digest(root / "factorial/frozen_pilot.json")}
        identity_path = root / "paper_controls/identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Control inputs changed; use a new run")
        atomic_json(identity_path, identity)
        for control in args.controls or CONTROLS:
            if all((root / "paper_controls" / control / str(seed) / "complete.json").exists() for seed in config["seeds"]):
                continue
            control_config, control_pilot = config, pilot
            if control == "original_recipe_full":
                # Change only the encoder loss recipe. Keep the shared improved
                # localization protocol so this contrast isolates that recipe.
                control_config = json.loads(json.dumps(config))
                control_config["factorial"]["loss_normalization"] = "loss"
                control_config["factorial"]["full_joint_loss"] = "uncertainty_weighting"
                selection = f._build_corpus(source, source.indices_for_role("source_method_selection"), teacher, config, route_norm, normalization)
                control_pilot = f._pilot_scales(control_config, corpus, selection, int(config["seeds"][0]) + 6001, device=args.device)
                del selection
            if control in resource.CONTROL_ROLES:
                architecture = json.loads((Path(resource.__file__).parent / f"resource_{control}_v2.json").read_text())
                specs, steps = resource._select_control_design(control, architecture, config, corpus, pilot, training)
            else:
                specs, steps = {"full": f._model_spec(config, corpus)}, {"full": int(config["factorial"]["steps"])}
            for seed in config["seeds"]:
                output = root / "paper_controls" / control / str(seed)
                output.mkdir(parents=True, exist_ok=True)
                if (output / "complete.json").exists():
                    continue
                started = time.perf_counter()
                models, rows = {}, []
                context = {**evidence, "control": control, "teacher_sha256": digest(root / "teacher.pt"),
                           "recipe": control_config["factorial"], "specs": specs, "steps": steps}
                for role in specs:
                    path = output / (role + ".pt")
                    if path.exists():
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        if payload["context"] != context:
                            raise ValueError("Control configuration changed; use a new run")
                        model = f._new_model(control_config, corpus, seed, model_spec=specs[role], device=args.device)
                        model.load_state_dict(payload["state_dict"])
                        row = payload["training"]
                    else:
                        pairing = None
                        if control == "shuffled_full":
                            pairing, pairing_rows = shuffled_pairing(corpus, int(seed))
                            write_csv(output / "pairing.csv", pairing_rows)
                        model, row = f._train_arm(control_config, corpus, seed, role, control_pilot,
                            model_spec=specs[role], step_count=steps[role], pairing_break=pairing, device=args.device,
                            checkpoint_path=output / (role + "-resume"), checkpoint_context=context)
                        _atomic_torch_save(path, {"state_dict": portable_state_dict(model), "training": row, "context": context})
                    models[role] = model
                    rows.append(row)
                representations = resource._control_representations(control, models, dataset, normalization, teacher.patch_spec)
                one_seed = {**config, "seeds": [seed]}
                banks, samples = f._run_localization(one_seed, dataset, {}, None, None, ["cpu"], methods=[control],
                    feature_provider=lambda _seed, _arm, scenes: {scene: representations[scene] for scene in scenes})
                write_csv(output / "localization_per_bank.csv", [{**row, **evidence} for row in banks])
                write_csv(output / "localization_per_sample.csv", [{**row, **evidence} for row in samples])
                full = [row for row in training if row["arm"] == "full" and int(row["seed"]) == int(seed)][0]
                training_flops = sum(float(row["measured_flops_per_step"]) * int(row["steps"]) for row in rows)
                parameters = sum(sum(p.numel() for p in model.parameters()) for model in models.values())
                write_csv(output / "resources.csv", [{**evidence, "arm": control, "seed": seed, "parameters": parameters,
                    "training_flops": training_flops, "wall_seconds": time.perf_counter() - started,
                    "parameter_ratio_to_full": parameters / float(full["parameters"]),
                    "training_flop_ratio_to_full": training_flops / (float(full["measured_flops_per_step"]) * int(full["steps"])),
                    "matching_note": "Actual encoder ratios reported; concatenation uses the same head recipe without an extra bottleneck"}])
                if control == "shuffled_full":
                    base = {**evidence, "arm": control, "seed": seed, "checkpoint_sha256": digest(output / "full.pt")}
                    for task in ("alignment", "response", "native"):
                        run_job(task, output / task, models["full"], dataset, teacher, config, normalization, route_norm, profile, base)
                atomic_json(output / "complete.json", {"status": "COMPLETE", **context})
                del models, representations
                gc.collect()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/paper_formal.json")
    parser.add_argument("--profile", type=Path, default=PROJECT / "formal_v2/configs/paper_core.json")
    parser.add_argument("--controls", nargs="+", choices=CONTROLS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-fixture", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
