"""Run the two main and four appendix tables; rerun the same command to resume."""
from __future__ import annotations
import argparse
import csv
import json
import gc
import os
from pathlib import Path
import subprocess
import sys
import torch
from .formal_config import load_formal_config
from .paper_suite import PROJECT, atomic_json, digest

STAGES = ("training", "probes", "controls", "maps", "risk", "tables")


def merge_csv(paths, target):
    fields = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for field in next(csv.reader(stream), []):
                if field not in fields:
                    fields.append(field)
    target = Path(target)
    temporary = target.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            with Path(path).open(encoding="utf-8-sig", newline="") as source:
                writer.writerows(csv.DictReader(source))
    os.replace(temporary, target)


def commands(args):
    common = ["--dataset", str(args.dataset.resolve()), "--config", str(args.config.resolve())]
    fixture = ["--allow-fixture"] if args.allow_fixture else []
    run_root = ["--run-root", str(args.output.resolve())]
    python = [sys.executable, "-B", "-m"]
    result = {
        "training": [python + ["formal_v2.paper_train", *common, "--output", str(args.output.resolve()), "--device", args.device, "--localize", *fixture]],
        "probes": [python + ["formal_v2.paper_core", *common, "--factorial-root", str(args.output.resolve() / "factorial"),
            "--teacher", str(args.output.resolve() / "teacher.pt"), "--output", str(args.output.resolve() / "core"),
            "--profile", str(args.profile.resolve()), "--device", args.device, "--array-cache", str(args.output.resolve() / "npz-cache"), *fixture]],
        "controls": [python + ["formal_v2.paper_controls", *common, *run_root, "--profile", str(args.profile.resolve()), "--device", args.device, *fixture]],
        "risk": [python + ["formal_v2.paper_risk", *common, *run_root, "--profile", str(args.profile.resolve()), "--device", args.device, *fixture]],
        "maps": [],
    }
    from .paper_maps import METHODS
    default_wigatr = PROJECT / "formal_v2/external_adapters/.venv-wigatr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wigatr_python = args.wigatr_python if args.wigatr_python and args.wigatr_python.exists() else (default_wigatr if default_wigatr.exists() else None)
    for name in METHODS:
        executable = str(wigatr_python) if name == "Wi-GATr" and wigatr_python else sys.executable
        # External Wi-GATr is optional.  A missing environment must not block
        # source training, probes, controls, or the other map baselines;
        # exporter records its row as MISSING/N/A.
        if name == "Wi-GATr" and wigatr_python is None and not args.plan:
            continue
        result["maps"].append([executable, "-B", "-m", "formal_v2.paper_maps", *common, *run_root, "--method", name, "--device", args.device, *fixture])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "formal_v2/configs/paper_formal.json")
    parser.add_argument("--profile", type=Path, default=PROJECT / "formal_v2/configs/paper_all_probes.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wigatr-python", type=Path)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--plan", action="store_true", help="Print exact commands without needing the dataset or GPU")
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args(argv)
    config = load_formal_config(args.config)
    planned = commands(args)
    if args.plan:
        print(json.dumps({stage: planned.get(stage, ["Export six CSV/Markdown/LaTeX tables"])
                          for stage in STAGES if stage in args.stages}, ensure_ascii=False, indent=2))
        return 0
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    for stage in STAGES:
        if stage not in args.stages or stage == "tables":
            continue
        if stage == "maps":
            from .formal_dataset import FormalDataset
            from .paper_maps import shared_units
            try:
                dataset = FormalDataset.load(args.dataset, array_cache=args.output / "npz-cache")
                shared_units(args.output.resolve(), dataset, config, args.device)
            except (ValueError, RuntimeError, OSError) as error:
                errors.append({"stage": stage, "error": str(error)})
                print(json.dumps(errors[-1]), flush=True)
                continue
            finally:
                if "dataset" in locals():
                    del dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for command in planned[stage]:
            print(json.dumps({"stage": stage, "command": command}), flush=True)
            outcome = subprocess.run(command, cwd=PROJECT)
            if outcome.returncode:
                errors.append({"stage": stage, "command": command, "returncode": outcome.returncode})
                # A failed core training cannot feed downstream tasks. A failed
                # baseline must not prevent other baselines or table export.
                if stage == "training":
                    break
        if stage == "training" and errors:
            break
    from .formal_dataset import FormalDataset
    from .paper_export import export
    data = FormalDataset.load(args.dataset, array_cache=args.output / "npz-cache")
    if data.is_fixture and not args.allow_fixture:
        raise ValueError("Fixture needs --allow-fixture")
    coverage = export(args.output, config, digest(args.dataset), fixture=data.is_fixture)
    atomic_json(args.output / "paper_status.json", {"errors": errors, **coverage})
    print(json.dumps({"tables": coverage["tables"], "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors and all(row["status"] == "COMPLETE" for row in coverage["tables"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
