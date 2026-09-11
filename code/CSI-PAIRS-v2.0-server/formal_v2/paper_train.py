"""Source-only training and optional localization with simple run records.

No LLM approval, expired permits, wheel closure, migration, claims or risk stages.
Uses the same encoder, loss, paired batch plans and localization implementation.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import numpy as np

from . import formal_factorial as factorial
from .formal_config import ARMS, load_formal_config
from .formal_dataset import FormalDataset
from .formal_evidence import config_sha256
from .formal_io import write_csv
from .formal_model import portable_state_dict
from .formal_protocol import PatchSpec
from .formal_routing import fit_route_normalization
from .formal_source_guard import SourceOnlyDataset
from .formal_teacher import train_teacher_bundle, save_teacher_bundle, load_teacher_bundle
from .formal_training_resume import _atomic_torch_save
from .paper_core import configure_runtime, runtime_record
from .paper_suite import PROJECT, atomic_json, digest, suite_lock


class Trace(list):
    def __init__(self, root, seed, arm):
        super().__init__()
        self.root, self.seed, self.arm = root, seed, arm

    def append(self, row):
        super().append(row)
        if row["step"] == 1 or row["step"] % 100 == 0:
            event = {"seed": self.seed, "arm": self.arm, **row}
            atomic_json(self.root / "progress.json", event)
            print(json.dumps(event), flush=True)


def run(args):
    configure_runtime()
    config = load_formal_config(args.config)
    root = args.output.resolve()
    runtime = runtime_record()
    identity = {"schema_version": "csi-pairs-paper-training-v1", "config_sha256": config_sha256(config),
                "dataset_sha256": digest(args.dataset), "device": args.device}
    with suite_lock(root):
        identity_path = root / "training_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Training inputs/config/runtime changed; use a new output directory")
        if not identity_path.exists() and any(path.name != "suite.lock" for path in root.iterdir()):
            raise ValueError("Output contains existing files; choose a fresh paper run")
        atomic_json(identity_path, identity)
        atomic_json(root / "runtime.json", runtime)
        dataset = FormalDataset.load(args.dataset, array_cache=args.array_cache or root / "npz-cache")
        if dataset._storage_source_sha256 != identity["dataset_sha256"]:
            raise ValueError("Dataset changed during run initialization")
        if dataset.is_fixture and not args.allow_fixture:
            raise ValueError("Fixture requires --allow-fixture and remains FORBIDDEN")
        source = SourceOnlyDataset(dataset)
        source_scenes = source.indices_for_role("source_encoder_train")
        teacher_path = root / "teacher.pt"
        teacher_receipt = root / "teacher.json"
        if teacher_receipt.exists():
            record = json.loads(teacher_receipt.read_text(encoding="utf-8"))
            if record["sha256"] != digest(teacher_path):
                raise ValueError("Source teacher checkpoint changed")
            teacher = load_teacher_bundle(teacher_path, config, device=args.device)
        else:
            print(json.dumps({"stage": "source_teacher_training"}), flush=True)
            from .paper_arrays import concatenate_disk
            cache = root / "teacher_arrays"
            cache.mkdir(exist_ok=True)
            parts = []
            for scene in source_scenes:
                path = cache / f"source-{scene}.npy"
                values = source.csi[int(scene)]
                np.save(path, values.reshape(-1, values.shape[-1]), allow_pickle=False)
                parts.append(path)
            source_csi = concatenate_disk(parts, cache / "source.npy", dtype=np.float64)
            teacher = train_teacher_bundle(source_csi, PatchSpec.from_metadata(source.metadata), config,
                                           seed=int(config["seeds"][0]) + 43001, device=args.device, streaming_root=cache)
            save_teacher_bundle(teacher_path, teacher, config, seed=teacher.seed)
            atomic_json(teacher_receipt, {"sha256": digest(teacher_path), "source_roles": ["source_encoder_train"], **identity})
        route_norm = fit_route_normalization(source, teacher)
        normalization = factorial._training_normalization(source, source_scenes, route_norm, teacher.patch_spec)
        corpus = factorial._build_corpus(source, source_scenes, teacher, config, route_norm, normalization)
        output = root / "factorial"
        output.mkdir(exist_ok=True)
        pilot_path = output / "frozen_pilot.json"
        if pilot_path.exists():
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            if pilot.get("identity") != identity:
                raise ValueError("Pilot belongs to another source run")
        else:
            selection = factorial._build_corpus(source, source.indices_for_role("source_method_selection"), teacher, config, route_norm, normalization)
            pilot = factorial._pilot_scales(config, corpus, selection, int(config["seeds"][0]) + 6001, device=args.device)
            pilot["identity"] = identity
            atomic_json(pilot_path, pilot)
            del selection
            gc.collect()
        atomic_json(output / "normalization.json", factorial._normalization_record(normalization))
        evidence = {"dataset_sha256": identity["dataset_sha256"], "config_sha256": identity["config_sha256"],
                    "source_tree_sha256": runtime["source_sha256"], "fixture": dataset.is_fixture,
                    "scientific_use": "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM",
                    "training_protocol": "paper-training-v1"}
        context = {**identity, "teacher_sha256": digest(teacher_path), "pilot_sha256": digest(pilot_path)}
        models, rows, checkpoints = {}, [], []
        for seed in config["seeds"]:
            for arm in ARMS:
                job_root = output / "jobs" / f"{seed}-{arm}"
                job_root.mkdir(parents=True, exist_ok=True)
                checkpoint = job_root / "model.pt"
                done = job_root / "complete.json"
                if done.exists():
                    record = json.loads(done.read_text(encoding="utf-8"))
                    if record["checkpoint_sha256"] != digest(checkpoint) or record["context"] != context:
                        raise ValueError("Completed training job changed")
                    model = factorial._new_model(config, corpus, seed, device="cpu")
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
                    row = record["training"]
                else:
                    trace = Trace(job_root, seed, arm)
                    model, row = factorial._train_arm(config, corpus, seed, arm, pilot, device=args.device,
                        checkpoint_path=job_root / "resume", checkpoint_context=context, checkpoint_interval_steps=100, loss_trace=trace)
                    _atomic_torch_save(checkpoint, {"schema_version": "csi-pairs-formal-checkpoint-v2.1-v6",
                        "arm": arm, "seed": seed, "model_spec": factorial._model_spec(config, corpus),
                        "normalization": factorial._normalization_record(normalization),
                        "teacher_checkpoint_sha256": context["teacher_sha256"], "state_dict": portable_state_dict(model),
                        "checkpoint_rule": "fixed_final_step_no_target_selection", "joint_training_sidecar": row, **evidence})
                    atomic_json(job_root / "loss_trace.json", list(trace))
                    atomic_json(done, {"context": context, "checkpoint_sha256": digest(checkpoint), "training": row})
                models[(int(seed), arm)] = model.cpu()
                rows.append({**row, **evidence})
                checkpoints.append({"seed": seed, "arm": arm, "path": checkpoint.relative_to(output).as_posix(),
                                    "sha256": digest(checkpoint), "teacher_checkpoint_sha256": context["teacher_sha256"]})
                write_csv(output / "training_summary.csv", rows)
                atomic_json(output / "checkpoint_index.json", {"schema_version": "csi-pairs-paper-checkpoint-index-v1", "checkpoints": checkpoints, **evidence})
        if not source.observed_scenes <= source.allowed_scenes:
            raise RuntimeError("Source training accessed forbidden scenes")
        del corpus
        gc.collect()
        localization_receipt = output / "localization_complete.json"
        if (args.localize or args.evaluate) and not localization_receipt.exists():
            print(json.dumps({"stage": "localization", "encoder_updates": False}), flush=True)
            from .paper_controls import read_csv
            from .paper_run import merge_csv
            banks, sample_files = [], []
            for seed in config["seeds"]:
                for arm in ARMS:
                    job = output / "jobs" / f"{seed}-{arm}"
                    complete = job / "localization_complete.json"
                    if not complete.exists():
                        one_seed = {**config, "seeds": [seed]}
                        values, samples = factorial._run_localization(one_seed, dataset, models, normalization,
                            teacher.patch_spec, [args.device], methods=[arm])
                        write_csv(job / "localization_per_bank.csv", [{**row, **evidence} for row in values])
                        write_csv(job / "localization_per_sample.csv", [{**row, **evidence} for row in samples])
                        atomic_json(complete, {"status": "COMPLETE"})
                        del samples
                    banks.extend(read_csv(job / "localization_per_bank.csv"))
                    sample_files.append(job / "localization_per_sample.csv")
            write_csv(output / "localization_per_bank.csv", banks)
            merge_csv(sample_files, output / "localization_per_sample.csv")
            typed = [{**row, **{key: int(row[key]) for key in ("seed", "budget", "draw")},
                      **{key: float(row[key]) for key in ("median_error_m", "p90_error_m", "utility_neg_log_median")}} for row in banks]
            write_csv(output / "localization_summary.csv", [{**row, **evidence} for row in factorial._localization_summary(typed)])
            atomic_json(localization_receipt, {"status": "COMPLETE", "files": {name: digest(output / name) for name in ("localization_per_bank.csv", "localization_per_sample.csv", "localization_summary.csv")}, **evidence})
        elif args.localize or args.evaluate:
            for name, expected in json.loads(localization_receipt.read_text(encoding="utf-8"))["files"].items():
                if digest(output / name) != expected:
                    raise ValueError("Completed localization output changed")
        atomic_json(root / "training_status.json", {"status": "COMPLETE", "trained_models": len(rows), "source_roles": ["source_encoder_train", "source_method_selection"], "localization_complete": localization_receipt.exists(), **evidence})
    if args.evaluate:
        from .paper_run import main as run_all
        arguments = ["--dataset", str(args.dataset), "--output", str(root), "--config", str(args.config),
                     "--profile", str(args.profile), "--device", args.device,
                     "--stages", "probes", "controls", "maps", "risk", "tables"]
        if args.allow_fixture:
            arguments.append("--allow-fixture")
        return run_all(arguments)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/paper_formal.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--array-cache", type=Path)
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Train, localize, evaluate independent core tasks, then export tables and missing coverage")
    parser.add_argument("--profile", type=Path, default=PROJECT / "formal_v2/configs/paper_all_probes.json")
    parser.add_argument("--allow-fixture", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
