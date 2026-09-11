"""Source-calibrated risk using the same frozen encoder and minibatch probe."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import formal_evaluation as e, formal_factorial as f, formal_risk as r
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import config_sha256
from .formal_io import write_csv
from .formal_localization import fit_source_position_head, predict_position_distribution
from .formal_protocol import patchify_csi, zero_typed_edit
from .formal_probes import predict_binary_probe
from .formal_routing import fit_route_normalization, route_dataset
from .formal_teacher import load_teacher_bundle
from .paper_core import frozen_model, load_profile, source_arrays
from .paper_minibatch import select_probe
from .paper_suite import PROJECT, atomic_json, digest, suite_lock


def run(args):
    config, profile = load_formal_config(args.config), load_profile(args.profile)
    root = args.run_root.resolve()
    output = root / "paper_risk"
    dataset = FormalDataset.load(args.dataset, array_cache=root / "npz-cache")
    if dataset.is_fixture and not args.allow_fixture:
        raise ValueError("Fixture needs --allow-fixture")
    evidence = {"dataset_sha256": digest(args.dataset), "config_sha256": config_sha256(config),
                "fixture": dataset.is_fixture, "scientific_use": "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"}
    teacher_hash = digest(root / "teacher.pt")
    teacher = load_teacher_bundle(root / "teacher.pt", config, device=args.device)
    route_norm = fit_route_normalization(dataset, teacher)
    roles = ("source_calibration_fit", "source_calibration_selection", "target")
    routed, examples = {}, {}
    for role in roles:
        routed[role] = route_dataset(dataset, teacher, config, dataset.indices_for_role(role), normalization=route_norm)
        examples[role] = r._risk_audit_examples(dataset, routed[role], config, role, require_all_strata=False)
    index = json.loads((root / "factorial/checkpoint_index.json").read_text())
    records = []
    with suite_lock(output):
        identity = {**evidence, "teacher_sha256": teacher_hash, "probe_profile": profile,
                    "checkpoints": index["checkpoints"]}
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Risk inputs changed; use a new run")
        atomic_json(identity_path, identity)
        mixture = [{"role": role, "bank_id": str(dataset.bank_ids[int(scene)]), "condition": condition,
                    "n": sum(row[0] == int(scene) and row[4] == condition for row in examples[role])}
                   for role in roles for scene in dataset.indices_for_role(role)
                   for condition in ("correct", "active", "gray", "null")]
        write_csv(output / "mixture.csv", mixture)
        for checkpoint in index["checkpoints"]:
            seed, arm = int(checkpoint["seed"]), checkpoint["arm"]
            job = output / f"{seed}-{arm}"
            job.mkdir(exist_ok=True)
            model, normalization, _ = frozen_model(root / "factorial" / checkpoint["path"], checkpoint["sha256"],
                teacher_hash, evidence["dataset_sha256"], config, args.device)
            probe_root = root / "core/jobs" / f"{seed}-{arm}" / "alignment"
            train = source_arrays("alignment", "source_probe_train", probe_root, model, dataset, teacher, config, normalization, route_norm, profile)
            selection = source_arrays("alignment", "source_probe_selection", probe_root, model, dataset, teacher, config, normalization, route_norm, profile)
            probe, _ = select_probe(train, selection, binary=True, profile=profile, config=config,
                                   seed=seed + 31001, device=args.device, output=probe_root / "probes")
            scenes = [int(s) for s in dataset.indices_for_role("source_encoder_train")]
            features = f._natural_representations(model, dataset, scenes, normalization, teacher.patch_spec)
            head = fit_source_position_head(np.vstack([features[s] for s in scenes]), np.vstack([dataset.positions[s] for s in scenes]), config, seed=seed + 17003)
            values = {}
            for role in roles:
                parts = []
                for scene in dataset.indices_for_role(role):
                    path = job / f"{role}-{scene}.npz"
                    if not path.exists():
                        selected = [row for row in examples[role] if row[0] == int(scene)]
                        result = r._replay_risk_examples(model, head, dataset, teacher, config, normalization,
                            routed[role], selected, arm, seed, e._masked_alignment_state_and_score,
                            e._normalized_scene_patches, f._normalized_map, f._normalized_radio, f._normalized_action,
                            zero_typed_edit, patchify_csi, predict_position_distribution, probe, predict_binary_probe)
                        temp = path.with_suffix(".tmp")
                        with temp.open("wb") as stream:
                            np.savez(stream, **result)
                        temp.replace(path)
                        print(f"risk {seed} {arm} {role} scene={scene}", flush=True)
                    with np.load(path, allow_pickle=False) as archive:
                        parts.append({key: archive[key] for key in archive.files})
                values[role] = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            fit, selection, target = (values[role] for role in roles)
            weights = lambda data: r._hierarchical_sample_weights(np.full(len(data["y"]), seed), data["canonical_cluster"], data["canonical_bank"])
            joint = r.fit_constrained_risk_calibrator(fit["d"], fit["u"], fit["y"],
                (selection["d"], selection["u"], selection["y"]), config,
                fit_weights=weights(fit), selection_weights=weights(selection))
            probability, inside = joint.predict(target["d"], target["u"])
            predictions = {"joint": probability}
            for name, key, direction in (("d_only", "d", "nonpositive"), ("u_only", "u", "nonnegative")):
                calibrator = r.fit_scalar_risk_calibrator(fit[key], fit["y"], (selection[key], selection["y"]), config,
                    direction=direction, fit_weights=weights(fit), selection_weights=weights(selection))
                predictions[name] = calibrator.predict(target[key])
            for name, probabilities in predictions.items():
                for i in range(len(probabilities)):
                    records.append({**evidence, "arm": arm, "seed": seed, "model_name": name,
                        "city_id": str(target["city"][i]), "cluster": str(target["canonical_cluster"][i]),
                        "bank": str(target["canonical_bank"][i]), "unit": str(target["canonical_unit_id"][i]),
                        "error": float(target["error"][i]), "y": int(target["y"][i]), "p": float(probabilities[i]),
                        "inside": bool(inside[i]), "proposal_count": int(target["proposal_count"][i])})
            del model, probe, head, features
        write_csv(output / "predictions.csv", records)
        metrics = []
        for arm in config["arms"] if "arms" in config else ("endpoint", "alignment", "response", "full"):
            for name in ("joint", "d_only", "u_only"):
                for city in sorted({row["city_id"] for row in records}):
                    rows = [row for row in records if (row["arm"], row["model_name"], row["city_id"]) == (arm, name, city)]
                    if not rows:
                        continue
                    array = lambda field: np.asarray([row[field] for row in rows])
                    for scope in ("all_queries", "source_support"):
                        base = {**evidence, "arm": arm, "model_name": name, "city_id": city, "budget": 0, "scope": scope,
                                "training_seeds": len({row["seed"] for row in rows})}
                        try:
                            result = r.risk_metrics(array("y"), array("p"), array("error"),
                                np.ones(len(rows), dtype=bool) if scope == "all_queries" else array("inside"),
                                array("cluster"), array("bank"), array("unit"), array("seed"))
                            metrics.append({**base, **{key: value for key, value in result.items() if np.isscalar(value)},
                                            "status": "COMPLETE"})
                        except RuntimeError as error:
                            # Undefined support is an experimental result, not an
                            # approval failure. Preserve the reason and all-query results.
                            metrics.append({**base, "status": "UNDEFINED", "reason": str(error)})
        write_csv(output / "metrics.csv", metrics)
        atomic_json(output / "complete.json", {"status": "COMPLETE", **evidence})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/paper_formal.json")
    parser.add_argument("--profile", type=Path, default=PROJECT / "formal_v2/configs/paper_all_probes.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-fixture", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
