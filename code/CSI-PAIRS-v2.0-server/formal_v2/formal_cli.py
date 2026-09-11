"""Independent experiment stages. New training/core evaluation: paper_train/paper_core.

No full-chain approvals, migration certification, literature receipts or claim assembly.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from .formal_locks import LOCK_EX, LOCK_NB, LOCK_UN, flock
from .formal_config import load_formal_config, resolve_dataset_path
from .formal_dataset import FormalDataset
from .formal_io import read_strict_json, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_SHUFFLED_PAIR_MANIFEST = str(ROOT / "external_adapters/shuffled_pair_control_v3.json")
DEFAULT_RETENTION_MANIFEST = str(ROOT / "external_adapters/retention_control_v3.json")
DEFAULT_SCENE_ID_MANIFEST = "builtin:sigmap-scene-id-v1"
DEFAULT_RESOURCE_CONTROL_MANIFEST = str(ROOT / "external_adapters/resource_controls_v3.json")
COMMAND_OUTPUT_PATHS = {
    "inspect-data": "data_contract.json", "verify-data": "data_verification",
    "qualify": "qualification", "run-wrong-map": "wrong_map", "run-factorial": "factorial",
    "run-evaluation": "evaluation", "run-risk": "risk", "run-external-baselines": "external_baselines",
    "run-representation-baselines": "representation_baselines", "run-resource-controls": "controls",
    "run-scene-id-audit": "scene_id", "run-shuffled-pair-control": "controls/shuffled_pair",
    "run-retention-audit": "evaluation/retention",
}
FILE_OUTPUT_COMMANDS = frozenset({"inspect-data"})
SELF_RESERVING_DIRECTORY_COMMANDS = frozenset({"run-representation-baselines"})
RESUME_SAFE_DIRECTORY_COMMANDS = frozenset({"run-evaluation", "run-factorial"})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("make-fixture")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=20270805)
    fixture.add_argument("--positions", type=int, default=16)
    fixture.add_argument("--source-banks-per-role", type=int, default=2)
    status = commands.add_parser("evaluation-status")
    status.add_argument("--output", required=True)
    resources = commands.add_parser("verify-waibu-resources", help="inspect local baseline resources")
    resources.add_argument("--registry", required=True)
    resources.add_argument("--waibu-root", required=True)
    resources.add_argument("--output", required=True)
    from .sionna_scene_export import build_parser as scene_parser
    commands.add_parser("export-sionna-scenes", parents=[scene_parser()], add_help=False)
    for name in COMMAND_OUTPUT_PATHS:
        child = commands.add_parser(name)
        child.add_argument("--config", default=str(ROOT / "configs/formal_v2.json"))
        child.add_argument("--dataset")
        child.add_argument("--output", required=True)
        child.add_argument("--allow-nonscientific-fixture", action="store_true")
        if name in {"run-factorial", "run-wrong-map"}:
            child.add_argument("--qualification-gate")
        if name == "qualify":
            child.add_argument("--data-verification-gate")
            child.add_argument("--resume", action="store_true")
        if name == "verify-data":
            child.add_argument("--verifier-manifest", required=True)
        if name == "run-external-baselines":
            child.add_argument("--adapter-manifest", required=True)
        if name == "run-representation-baselines":
            child.add_argument("--representation-baseline-config", required=True)
        if name == "run-resource-controls":
            child.add_argument("--control-manifest", default=DEFAULT_RESOURCE_CONTROL_MANIFEST)
        if name == "run-scene-id-audit":
            child.add_argument("--scene-id-manifest", default=DEFAULT_SCENE_ID_MANIFEST)
        if name in {"run-shuffled-pair-control", "run-retention-audit"}:
            child.add_argument("--claim-control-manifest", default=(DEFAULT_SHUFFLED_PAIR_MANIFEST if name == "run-shuffled-pair-control" else DEFAULT_RETENTION_MANIFEST))
    return parser


def _dispatch(args, config, dataset, output):
    name = args.command
    if name == "inspect-data":
        result = dataset.contract_report()
        write_json(output / "data_contract.json", result)
        return result
    if name == "verify-data":
        from .formal_data_verification import run_data_verification
        return run_data_verification(config, dataset, args.verifier_manifest, output)
    if name == "qualify":
        from .formal_qualification import run_formal_qualification
        path = Path(args.data_verification_gate) if args.data_verification_gate else output / "data_verification/gate.json"
        return run_formal_qualification(config, dataset, output, read_strict_json(path) if path.is_file() else None,
                                       data_verification_gate_path=path if path.is_file() else None, resume=args.resume)
    if name in {"run-factorial", "run-wrong-map"}:
        path = Path(args.qualification_gate) if args.qualification_gate else output / "qualification/gate.json"
        gate = read_strict_json(path)
        if name == "run-wrong-map":
            from .formal_wrong_map import run_formal_wrong_map
            return run_formal_wrong_map(config, dataset, output, gate)
        from .formal_factorial import run_formal_factorial
        return run_formal_factorial(config, dataset, output, gate, allow_nonscientific_fixture=args.allow_nonscientific_fixture)
    if name == "run-evaluation":
        from .formal_upstream import resolve_authenticated_upstream
        return _run_local_streaming_evaluation(config, dataset, output, resolve_authenticated_upstream(config, dataset, output))
    if name == "run-risk":
        from .formal_risk import run_risk_contract
        return run_risk_contract(config, dataset, output)
    if name == "run-external-baselines":
        from .formal_external import run_external_baselines
        return run_external_baselines(config, dataset, args.adapter_manifest, output)
    if name == "run-representation-baselines":
        from .formal_representation_baselines import run_representation_baselines
        return run_representation_baselines(config, dataset, args.representation_baseline_config, output)
    if name == "run-resource-controls":
        from .formal_controls import run_resource_controls
        return run_resource_controls(config, dataset, args.control_manifest, output)
    if name == "run-scene-id-audit":
        from .formal_scene_id import run_scene_id_audit
        return run_scene_id_audit(config, dataset, args.scene_id_manifest, output)
    if name == "run-shuffled-pair-control":
        from .formal_claim_controls import run_shuffled_pair_control
        return run_shuffled_pair_control(config, dataset, args.claim_control_manifest, output)
    if name == "run-retention-audit":
        from .formal_claim_controls import run_retention_audit
        return run_retention_audit(config, dataset, args.claim_control_manifest, output)
    raise ValueError(f"Unknown experiment stage: {name}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    lock = None
    try:
        output = Path(args.output).resolve()
        if args.command == "make-fixture" and output.suffix != ".npz":
            output = Path(str(output) + ".npz")
        if args.command == "evaluation-status":
            from .formal_evaluation_resume import read_evaluation_status
            from .formal_evaluation_streaming import STATE_DIRECTORY
            state_root = output / STATE_DIRECTORY
            status = read_evaluation_status(state_root) or {"status": "NOT_STARTED", "state_root": str(state_root)}
            detail = state_root / "probe_progress.json"
            if detail.is_file():
                status["probe_progress"] = read_strict_json(detail)
            print(json.dumps(status, sort_keys=True))
            return 0
        if Path(args.output).is_symlink() or output.is_symlink() or output == ROOT or output in ROOT.parents or ROOT in output.parents:
            raise ValueError("Choose an experiment output directory outside the source tree")
        lock_root = output.parent if args.command == "export-sionna-scenes" and output.name == "inputs" else output
        lock = _acquire_output_lock(lock_root)
        if args.command == "verify-waibu-resources":
            from .formal_resources import verify_waibu_resources
            result = verify_waibu_resources(args.registry, args.waibu_root, output)
            print(json.dumps(result, sort_keys=True))
            return _result_exit_code(result)
        if args.command == "export-sionna-scenes":
            from .sionna_scene_export import export_sionna_scenes
            result = export_sionna_scenes(args.dataset, output, license_id=args.license_id,
                carrier_frequency_hz=args.carrier_frequency_hz, subcarrier_spacing_hz=args.subcarrier_spacing_hz,
                receiver_z_m=args.receiver_z_m, max_depth=args.max_depth, refraction=args.refraction)
            print(result)
            return 0
        if args.command == "make-fixture":
            from .formal_fixture import write_nonscientific_fixture
            if output.exists():
                raise FileExistsError(output)
            write_nonscientific_fixture(output, seed=args.seed, positions=args.positions, source_banks_per_role=args.source_banks_per_role)
            return 0
        from .formal_evidence import configure_reproducible_runtime
        configure_reproducible_runtime()
        config = load_formal_config(args.config)
        dataset_path = Path(args.dataset) if args.dataset else resolve_dataset_path(config)
        dataset = FormalDataset.load(dataset_path, array_cache=output / "npz-cache")
        if dataset.is_fixture and not args.allow_nonscientific_fixture:
            raise ValueError("Fixture requires --allow-nonscientific-fixture; its results remain FORBIDDEN")
        if args.command == "qualify" and args.resume:
            stage = output / "qualification"
            if not stage.is_dir() or (stage / "gate.json").exists() or (stage / "manifest.json").exists():
                raise ValueError("qualify --resume refuses a completed qualification; an incomplete run is required")
        else:
            _reserve_command_output(args.command, output)
        result = _dispatch(args, config, dataset, output)
        print(json.dumps({"status": result.get("status", "success"), "output": str(output)}, sort_keys=True))
        return _result_exit_code(result)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.release()


def _run_local_streaming_evaluation(config, dataset, output, upstream):
    from .formal_evaluation_identity import build_local_evaluation_execution
    from .formal_evaluation_streaming import run_streaming_formal_evaluation

    execution = build_local_evaluation_execution(config, dataset, output, upstream)
    return run_streaming_formal_evaluation(
        config,
        dataset,
        output,
        upstream_root=output,
        qualification_gate_path=upstream.qualification_gate,
        factorial_gate_path=upstream.factorial_gate,
        checkpoint_index_path=upstream.checkpoint_index,
        checkpoint_inventory_sha256=execution.identity.legacy_checkpoint_inventory_sha256,
        run_identity=execution.identity,
        execution_devices=execution.devices,
        batch_size=execution.batch_size,
    )

class _OutputLock:
    def __init__(self, path: Path, guard_path: Path, guard_handle) -> None:
        self.path = path
        self.guard_path = guard_path
        self.guard_handle = guard_handle
        self.released = False

    def is_file(self) -> bool:
        return self.path.is_file()

    def unlink(self, missing_ok: bool = False) -> None:
        self.release(missing_ok=missing_ok)

    def release(self, missing_ok: bool = True) -> None:
        if self.released:
            if not missing_ok:
                raise FileNotFoundError(self.path)
            return
        try:
            self.path.unlink(missing_ok=missing_ok)
        finally:
            flock(self.guard_handle.fileno(), LOCK_UN)
            self.guard_handle.close()
            self.released = True

def _acquire_output_lock(output_root: Path) -> _OutputLock:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.csi-pairs-operation.lock"
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    if guard_path.is_symlink():
        raise RuntimeError(f"operation lock guard cannot be a symlink: {guard_path}")
    descriptor = os.open(
        guard_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    guard_handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        flock(guard_handle.fileno(), LOCK_EX | LOCK_NB)
    except BlockingIOError as error:
        guard_handle.close()
        raise FileExistsError(
            f"refusing concurrent V2 output use; operation lock exists: {lock_path}"
        ) from error
    try:
        if lock_path.is_symlink():
            raise RuntimeError(f"operation lock cannot be a symlink: {lock_path}")
        owner = {
            "schema_version": "csi-pairs-v6-operation-lock-v2",
            "pid": os.getpid(),
            "output_root": str(output_root),
        }
        temporary = lock_path.with_name(
            f".{lock_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(owner, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, lock_path)
        return _OutputLock(lock_path, guard_path, guard_handle)
    except Exception:
        flock(guard_handle.fileno(), LOCK_UN)
        guard_handle.close()
        raise

def _reserve_command_output(command: str, output_root: Path) -> Path | None:
    relative_path = COMMAND_OUTPUT_PATHS.get(command)
    if relative_path is None:
        return None
    target = output_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if command in SELF_RESERVING_DIRECTORY_COMMANDS:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite V2 output: {target}")
        return target
    if command in RESUME_SAFE_DIRECTORY_COMMANDS:
        if target.is_symlink() or target.is_file():
            raise FileExistsError(f"refusing to overwrite V2 output: {target}")
        target.mkdir(parents=True, exist_ok=True)
        return target
    try:
        if command in FILE_OUTPUT_COMMANDS:
            target.touch(exist_ok=False)
        else:
            target.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite V2 output: {target}") from error
    return target

def _result_exit_code(result: object) -> int:
    if not isinstance(result, dict):
        return 0
    if result.get("passed") is False:
        return 1
    if result.get("status") in {
        "FAIL",
        "BLOCKED",
        "INVALID",
        "QUALIFICATION_NO_GO",
        "DRY_RUN_FAIL_NOT_EVIDENCE",
        "INCOMPLETE_FAIL_CLOSED",
    }:
        return 1
    for key in ("gate_vector", "upstream_gates"):
        vector = result.get(key)
        if isinstance(vector, dict) and any(
            value in {"FAIL", "INVALID"} for value in vector.values()
        ):
            return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
