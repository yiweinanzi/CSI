"""Execution-only reevaluation with separately authenticated training provenance."""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import resource
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evaluation_identity import build_local_evaluation_execution, _canonical_sha256
from .formal_evaluation_resume import write_atomic_json
from .formal_evidence import evidence_context
from .formal_io import read_strict_json, sha256_file
from .formal_upstream import AuthenticatedUpstream


ORIGIN_COMMIT = "c5917018610b262f5263ccaeba2f087b2eb96a96"
DATASET_SHA256 = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
REPAIR_SCHEMA = "csi-pairs-evaluation-execution-repair-v1"
EXECUTION_FILES = {
    "formal_cli.py", "formal_evaluation.py", "formal_evaluation_streaming.py",
    "formal_metrics.py", "formal_probes.py", "formal_evaluation_repair.py",
    "tools/verify_evaluation_origin.py", "tools/validate_probe_execution.py",
    "tools/benchmark_probe_execution.py",
}


def _source_files(root):
    suffixes = {".py", ".json", ".sh", ".txt", ".toml", ".lock", ".yaml", ".yml"}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
        and not any(p.startswith((".venv-", ".runtime-")) or p == "__pycache__" for p in path.relative_to(root).parts)
    }


def _function_tree(path, omit):
    module = ast.parse(path.read_text())
    module.body = [item for item in module.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name not in omit]
    return ast.dump(module, include_attributes=False)


def require_execution_only_changes(origin_source, evaluation_source):
    before, after = _source_files(origin_source), _source_files(evaluation_source)
    changes = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    disallowed = [name for name in changes if name not in EXECUTION_FILES and not name.startswith("tests/")]
    if disallowed:
        raise RuntimeError(f"execution repair changed non-evaluation dependencies: {disallowed}")
    for name, allowed_functions in (
        ("formal_metrics.py", {"binary_auroc"}),
        ("formal_evaluation.py", {"_prepare_alignment_shortcut_probes"}),
    ):
        if _function_tree(origin_source / name, allowed_functions) != _function_tree(evaluation_source / name, allowed_functions):
            raise RuntimeError(f"execution repair modified unrelated functions in {name}")
    for class_name in ("CompatibilityProbe", "ActionResponseProbe"):
        classes = []
        for root in (origin_source, evaluation_source):
            module = ast.parse((root / "formal_probes.py").read_text())
            classes.append(ast.dump(next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name), include_attributes=False))
        if classes[0] != classes[1]:
            raise RuntimeError("execution repair changed the probe architecture")
    return [{"path": name, "upstream_sha256": before.get(name), "evaluation_sha256": after.get(name)} for name in changes]


@dataclass(frozen=True)
class AuthenticatedEvaluationOrigin:
    receipt_path: Path
    receipt_sha256: str
    record: dict

    def require_compatible(self, config, dataset, upstream_root, required_roles):
        if sha256_file(self.receipt_path) != self.receipt_sha256 or read_strict_json(self.receipt_path) != self.record:
            raise RuntimeError("evaluation origin receipt changed")
        upstream = self.record["upstream"]
        if Path(upstream_root).resolve() != Path(upstream["origin_run"]):
            raise RuntimeError("evaluation origin run mismatch")
        current = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
        for key in ("dataset_sha256", "config_sha256", "fixture", "requirements_lock_sha256"):
            if current[key] != upstream["evidence"][key]:
                raise RuntimeError(f"evaluation origin {key} mismatch")
        if current != self.record["evaluation_evidence"]:
            raise RuntimeError("evaluation source or runtime changed after compatibility authentication")
        if not set(required_roles).issubset(upstream["verified_roles"]):
            raise RuntimeError("evaluation origin lacks verified dataset roles")
        for path, digest in upstream["inventory"].items():
            if sha256_file(path) != digest:
                raise RuntimeError(f"authenticated upstream file changed: {path}")


def authenticate_evaluation_origin(config, dataset, *, origin_server, upstream_root, output_root, config_path):
    origin_server, upstream_root, output_root = (Path(p).resolve() for p in (origin_server, upstream_root, output_root))
    current_server = Path(__file__).resolve().parents[1]
    if origin_server == current_server or output_root == upstream_root or upstream_root in output_root.parents:
        raise RuntimeError("execution repair needs isolated source and output roots")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=origin_server, text=True).strip()
    if commit != ORIGIN_COMMIT:
        raise RuntimeError("execution repair requires the recorded c591701 training source")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=origin_server, text=True).strip():
        raise RuntimeError("upstream source has tracked changes")
    if sha256_file(dataset.source_path) != DATASET_SHA256 or dataset.is_fixture:
        raise RuntimeError("execution repair dataset SHA differs from the formal NPZ")
    changes = require_execution_only_changes(origin_server / "formal_v2", current_server / "formal_v2")
    command = [sys.executable, "-B", str(Path(__file__).parent / "tools/verify_evaluation_origin.py"),
               "--origin-server", str(origin_server), "--config", str(Path(config_path).resolve()),
               "--dataset", str(dataset.source_path), "--run", str(upstream_root)]
    completed = subprocess.run(command, cwd=origin_server, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("original-source upstream authentication failed: " + completed.stderr[-4000:])
    upstream = json.loads(completed.stdout)
    current = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
    old_runtime = {k: v for k, v in upstream["evidence"]["runtime_provenance"].items() if k != "source_tree_sha256"}
    new_runtime = {k: v for k, v in current["runtime_provenance"].items() if k != "source_tree_sha256"}
    if old_runtime != new_runtime:
        raise RuntimeError("execution repair runtime dependencies differ from authenticated upstream")
    if upstream["origin_git_commit"] != ORIGIN_COMMIT:
        raise RuntimeError("upstream source changed during verification")
    record = {
        "schema_version": REPAIR_SCHEMA, "upstream": upstream,
        "evaluation_evidence": current, "source_changes": changes,
        "evaluation_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=current_server, text=True).strip(),
        "evaluation_tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=current_server)).hexdigest(),
        "metric_implementation": "exact_binary_average_tie_ranks_sort_v1",
        "scope": "evaluation execution only; original checkpoint, NPZ and G5 identities remain unchanged",
        "sota_ready": False,
    }
    receipt_path = output_root / "evaluation_origin.json"
    if receipt_path.exists():
        if read_strict_json(receipt_path) != record:
            raise RuntimeError("existing evaluation origin receipt differs; use a new output root")
    else:
        write_atomic_json(receipt_path, record)
    return AuthenticatedEvaluationOrigin(receipt_path, sha256_file(receipt_path), record)


def run_representative_probe(config, dataset, origin, output, device):
    from . import formal_evaluation as evaluation
    from .formal_evaluation_streaming import PROBE_TRAIN_BATCH_ROWS, _compatibility_record
    from .formal_probes import fit_select_compatibility_probe, predict_binary_probe
    from .formal_metrics import binary_auroc, binary_nll
    from .formal_routing import fit_route_normalization, route_dataset
    from .formal_teacher import load_teacher_bundle

    root = Path(origin.record["upstream"]["origin_run"])
    origin.require_compatible(config, dataset, root, ("source_probe_train", "source_probe_selection"))
    qualification = read_strict_json(root / "qualification/gate.json")
    checkpoint = origin.record["upstream"]["checkpoints"][0]
    started = time.monotonic()
    phase_start = started
    last_written = 0.0
    phases = []
    current_phase = None

    def report(event):
        nonlocal phase_start, last_written, current_phase
        now = time.monotonic()
        phase = (event.get("phase"), event.get("family"))
        if phase != current_phase:
            if current_phase is not None:
                phases.append({"phase": current_phase, "seconds": now - phase_start})
            current_phase, phase_start = phase, now
        if now - last_written >= 5 or event.get("phase") in ("training_complete", "saved"):
            record = {"unit": "representative_compatibility_probe_not_whole_unit", "arm": checkpoint["arm"], "seed": checkpoint["seed"], "device": str(device), "elapsed_seconds": now - started, "stage_elapsed_seconds": now - phase_start, "monotonic": now, "host_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, **event}
            write_atomic_json(output / "representative_progress.json", record)
            print(json.dumps(record), flush=True)
            last_written = now

    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    report({"phase": "load_teacher_and_checkpoint"})
    teacher = load_teacher_bundle(qualification["teacher_checkpoint"], config, device=device)
    model = evaluation._load_model(root / "factorial", checkpoint, qualification, config, dataset, device=device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    report({"phase": "normalization"})
    route_normalization = fit_route_normalization(dataset, teacher)
    normalization = evaluation._training_normalization(dataset, dataset.indices_for_role("source_encoder_train"), route_normalization, teacher.patch_spec)
    data = []
    for role in ("source_probe_train", "source_probe_selection"):
        report({"phase": "prepare_" + role})
        scenes = dataset.indices_for_role(role)
        routed = route_dataset(dataset, teacher, config, scenes, normalization=route_normalization)
        rows = evaluation._compatibility_dataset(model, dataset, teacher, config, normalization, scenes, route_normalization=route_normalization, routed=routed, batch_size=int(config["evaluation"]["batch_size"]))
        data.append({"features": rows["features"], "labels": rows["labels"]})
        del rows, routed
    train, selection = data
    torch.cuda.synchronize(device)
    training_start = time.monotonic()
    cuda_start, cuda_stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    probe, selected = fit_select_compatibility_probe(
        train["features"], train["labels"], selection["features"], selection["labels"], config,
        seed=int(checkpoint["seed"]) + 31001, device=device,
        train_batch_rows=PROBE_TRAIN_BATCH_ROWS, prefer_full_batch=True,
        progress_callback=report,
    )
    cuda_stop.record()
    cuda_stop.synchronize()
    training_seconds = time.monotonic() - training_start
    report({"phase": "prediction"})
    predictions = predict_binary_probe(probe, selection["features"], batch_rows=PROBE_TRAIN_BATCH_ROWS)
    report({"phase": "statistics"})
    measured = {"selection_auroc": binary_auroc(selection["labels"], predictions), "selection_nll": binary_nll(selection["labels"], predictions)}
    assert measured == {k: selected[k] for k in measured}
    assert all(torch.equal(value, model.state_dict()[key].cpu()) for key, value in before.items())
    report({"phase": "validation_and_save"})
    checkpoint_path = output / "representative_compatibility_probe.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(_compatibility_record(probe), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(checkpoint_path)
    np.save(output / "representative_predictions.npy", predictions)
    result = {
        "status": "REPRESENTATIVE_PROBE_COMPLETE", "formal_work_unit_complete": False,
        "origin_receipt_sha256": origin.receipt_sha256,
        "upstream_checkpoint_sha256": checkpoint["sha256"],
        "dataset_sha256": origin.record["upstream"]["evidence"]["dataset_sha256"],
        "device": str(device), "training_steps_per_candidate": int(config["evaluation"]["probe_steps"]),
        "candidates": ["linear", "mlp2"], "train_shape": list(train["features"].shape),
        "selection_shape": list(selection["features"].shape),
        "selection": selected, "backbone_unchanged": True,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions_sha256": sha256_file(output / "representative_predictions.npy"),
        "phases": phases, "total_seconds": time.monotonic() - started,
        "training_selection_wall_seconds": training_seconds,
        "training_selection_cuda_interval_ms": cuda_start.elapsed_time(cuda_stop),
        "host_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "sota_ready": False,
    }
    write_atomic_json(output / "representative_probe_result.json", result)
    report({"phase": "saved"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "output", "upstream-root", "origin-server"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stop-after-units", type=int, default=1)
    parser.add_argument("--probe-build-limit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--representative-probe", action="store_true")
    args = parser.parse_args(argv)
    from .formal_cli import _acquire_output_lock, _acquire_legacy_read_lock
    from .formal_evaluation_streaming import run_streaming_formal_evaluation

    config = load_formal_config(args.config)
    dataset = FormalDataset.load(args.dataset, require_clean_csi=True)
    output = Path(args.output).resolve()
    lock = _acquire_output_lock(output)
    upstream_lock = None
    try:
        upstream_lock = _acquire_legacy_read_lock(args.upstream_root)
        origin = authenticate_evaluation_origin(
            config, dataset, origin_server=args.origin_server,
            upstream_root=args.upstream_root, output_root=output, config_path=args.config,
        )
        if args.representative_probe:
            from .formal_evaluation_identity import resolve_production_devices

            return_code = run_representative_probe(config, dataset, origin, output, resolve_production_devices(dataset)[0])
            print(json.dumps(return_code, sort_keys=True))
            return 0
        upstream_root = Path(args.upstream_root).resolve()
        upstream = AuthenticatedUpstream(upstream_root, upstream_root / "qualification", upstream_root / "factorial", None)
        execution = build_local_evaluation_execution(config, dataset, output, upstream)
        identity = replace(execution.identity, compute_plan_sha256=_canonical_sha256({
            "base_plan": execution.identity.compute_plan_sha256,
            "origin_receipt_sha256": origin.receipt_sha256,
            "probe_build_limit": args.probe_build_limit,
        }))
        write_atomic_json(output / "execution_identity.json", identity.as_dict())
        result = run_streaming_formal_evaluation(
            config, dataset, output, upstream_root=upstream_root,
            qualification_gate_path=upstream.qualification_gate,
            factorial_gate_path=upstream.factorial_gate,
            checkpoint_index_path=upstream.checkpoint_index,
            checkpoint_inventory_sha256=identity.legacy_checkpoint_inventory_sha256,
            run_identity=identity, execution_devices=execution.devices,
            batch_size=execution.batch_size, probe_build_limit=args.probe_build_limit,
            authenticated_origin=origin,
            stop_after_units=args.stop_after_units or None,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if upstream_lock is not None:
            upstream_lock.release()
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
