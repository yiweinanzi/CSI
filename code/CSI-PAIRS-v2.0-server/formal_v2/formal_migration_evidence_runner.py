from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Mapping, Sequence

from . import formal_evaluation_subset_compare as subset
from .formal_config import load_formal_config
from .formal_dataset import FormalDataset
from .formal_evaluation_resume import (
    NO_MIGRATION_SHA256,
    CorruptShardError,
    EvaluationExecutionProfile,
    EvaluationResumeStore,
    EvaluationRunIdentity,
    EvaluationShardIdentity,
    StaleResumeError,
    WriterLockError,
    read_evaluation_status,
)
from .formal_evaluation_streaming import (
    STREAMING_EVALUATION_SCHEMA,
    evaluation_output_schema_sha256,
)
from .formal_evaluation_subset_compare import REPORT_SCHEMA as SUBSET_REPORT_SCHEMA
from .formal_evidence import config_sha256, evidence_context
from .formal_io import read_strict_json, sha256_file
from .formal_migration import _source_identity
from .formal_migration_evidence import (
    _validate_real_subset_report,
    bind_file,
    validate_evidence_report,
    write_corrupt_checkpoint_report,
    write_json_exclusive_atomic,
    write_lock_report,
    write_performance_report,
    write_progress_report,
    write_resume_report,
    write_stale_checkpoint_report,
)


PERFORMANCE_OBSERVATION_SCHEMA = (
    "csi-pairs-v6-migration-performance-observation-v1"
)
CONTROL_OBSERVATION_SCHEMA = "csi-pairs-v6-migration-control-observation-v1"
PERFORMANCE_RUNNER_SCHEMA = "csi-pairs-v6-migration-performance-runner-v1"
CONTROL_RUNNER_SCHEMA = "csi-pairs-v6-migration-control-runner-v1"
SUBSET_REFRESH_RECEIPT_SCHEMA = "csi-pairs-v6-subset-refresh-receipt-v1"
INTERRUPTED_EXIT_CODE = 75
CONTROL_CHILD_FAILURE_EXIT_CODE = 70
CONTROL_CHILD_READY_TIMEOUT_SECONDS = 120.0
CONTROL_CHILD_RELEASE_TIMEOUT_SECONDS = 60.0
CONTROL_CHILD_JOIN_TIMEOUT_SECONDS = 10.0
CONTROL_CHILD_COMPLETION_TIMEOUT_SECONDS = 120.0
DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS = 0.5
GPU_PROCESS_BINDING_METHOD = "EXCLUSIVE_TARGET_GPU_PROCESS_SET_DELTA"
SCALE_MULTIPLIERS = (1, 2, 4)
SCALE_NAMES = ("N", "2N", "4N")
CONTROL_NAMES = ("resume", "corrupt_checkpoint", "stale_checkpoint", "lock", "progress")
CONTROL_RESULT_FIELDS = {
    "resume": {
        "interrupted",
        "resumed",
        "completed",
        "reused_shard_count",
        "recomputed_shard_count",
        "output_equivalent",
    },
    "corrupt_checkpoint": {
        "corruption_detected",
        "corrupt_shard_count",
        "quarantined_shard_count",
        "recomputed_shard_count",
        "unaffected_shard_count",
    },
    "stale_checkpoint": {
        "stale_identity_detected",
        "rejected_shard_count",
        "reused_stale_shard_count",
    },
    "lock": {
        "second_writer_attempted",
        "second_writer_rejected",
        "first_writer_preserved",
        "write_count",
    },
    "progress": {
        "monotonic",
        "atomic",
        "final_complete",
        "update_count",
        "completed_units",
        "total_units",
    },
}
IDENTITY_FIELDS = {
    "legacy_run_root",
    "new_run_root",
    "legacy_commit",
    "legacy_source_tree_sha256",
    "new_commit",
    "new_source_tree_sha256",
    "requirements_lock_sha256",
    "dataset_sha256",
    "config_sha256",
    "protocol_sha256",
    "new_run_id",
    "new_run_nonce",
    "new_compute_plan_sha256",
}
_HEX40 = frozenset("0123456789abcdef")
_HEX64 = _HEX40
SUBSET_REPORT_FIELDS = {
    "schema_version",
    "status",
    "passed",
    "created_utc",
    "authoritative_layer",
    "contract",
    "workers",
    "shared_inputs",
    "selection",
    "layers",
    "read_only",
    "resources",
    "report_path",
}
GPU_PROCESS_BINDING_FIELDS = {
    "method",
    "evaluator_namespace_pid",
    "nvidia_host_pid",
    "gpu_index",
    "gpu_uuid",
    "baseline_compute_process_count",
    "target_gpu_empty_before_launch",
    "all_observed_samples_exclusive",
}
GPU_SAMPLE_FIELDS = {
    "observed_utc",
    "evaluator_namespace_pid",
    "nvidia_host_pid",
    "gpu_index",
    "gpu_uuid",
    "gpu_utilization_percent",
    "process_vram_bytes",
    "compute_process_count",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError(f"{label} must be a UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{label} must be a UTC timestamp")
    return parsed


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_hex(value: object, width: int) -> bool:
    return (
        type(value) is str
        and len(value) == width
        and all(character in _HEX40 for character in value)
    )


def _require_python_contract() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError(
            "migration evidence runner requires PYTHONDONTWRITEBYTECODE=1 and python -B"
        )


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"{label} must be a regular file: {candidate}")
    return candidate.resolve()


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"{label} must be a regular directory: {candidate}")
    return candidate.resolve()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_external_root(path: str | Path, identity: Mapping[str, object]) -> Path:
    requested = Path(path)
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(
            f"migration evidence output directory must not already exist: {requested}"
        )
    parent = _regular_directory(requested.parent, "migration evidence output parent")
    target = parent / requested.name
    legacy = Path(str(identity["legacy_run_root"])).resolve()
    new = Path(str(identity["new_run_root"])).resolve()
    if _inside(target, legacy) or _inside(target, new):
        raise RuntimeError("migration evidence output must be external to both run roots")
    target.mkdir(mode=0o700)
    return target


def load_expected_identity(path: str | Path) -> dict[str, object]:
    identity_path = _regular_file(path, "migration expected identity")
    payload = read_strict_json(identity_path)
    if not isinstance(payload, dict) or set(payload) != IDENTITY_FIELDS:
        raise RuntimeError("migration expected identity fields are invalid")
    for key in ("legacy_commit", "new_commit"):
        if not _is_hex(payload[key], 40):
            raise RuntimeError(f"migration expected identity {key} is invalid")
    for key in (
        "legacy_source_tree_sha256",
        "new_source_tree_sha256",
        "requirements_lock_sha256",
        "dataset_sha256",
        "config_sha256",
        "protocol_sha256",
        "new_run_nonce",
        "new_compute_plan_sha256",
    ):
        if not _is_hex(payload[key], 64):
            raise RuntimeError(f"migration expected identity {key} is invalid")
    if payload["new_run_id"] != f"csi-pairs-{str(payload['new_run_nonce'])[:16]}":
        raise RuntimeError("migration expected identity run ID is invalid")
    legacy = Path(str(payload["legacy_run_root"]))
    new = Path(str(payload["new_run_root"]))
    if (
        not legacy.is_absolute()
        or not new.is_absolute()
        or legacy.resolve() == new.resolve()
        or _inside(legacy.resolve(), new.resolve())
        or _inside(new.resolve(), legacy.resolve())
    ):
        raise RuntimeError("migration expected run roots are invalid")
    return payload


def _production_context(
    identity: Mapping[str, object],
    *,
    config_path: str | Path,
    dataset_path: str | Path,
) -> tuple[dict, FormalDataset, dict[str, object]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError(
            "migration technical evidence requires CUDA_VISIBLE_DEVICES=0,1"
        )
    config_file = _regular_file(config_path, "formal config")
    dataset_file = _regular_file(dataset_path, "formal dataset")
    source_root = Path(__file__).resolve().parent
    source = _source_identity(source_root)
    if (
        source["git_commit"] != identity["new_commit"]
        or source["source_tree_sha256"] != identity["new_source_tree_sha256"]
        or source["requirements_lock_sha256"]
        != identity["requirements_lock_sha256"]
    ):
        raise RuntimeError("running source identity does not match migration identity")
    if sha256_file(dataset_file) != identity["dataset_sha256"]:
        raise RuntimeError("formal dataset SHA-256 does not match migration identity")
    config = load_formal_config(config_file)
    if config_sha256(config) != identity["config_sha256"]:
        raise RuntimeError("formal config SHA-256 does not match migration identity")
    dataset = FormalDataset.load(dataset_file)
    if dataset.is_fixture:
        raise RuntimeError("migration technical evidence requires a nonfixture dataset")
    evidence = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
    for key in (
        "source_tree_sha256",
        "requirements_lock_sha256",
        "dataset_sha256",
        "config_sha256",
    ):
        expected_key = "new_source_tree_sha256" if key == "source_tree_sha256" else key
        if evidence[key] != identity[expected_key]:
            raise RuntimeError(f"runtime evidence {key} does not match migration identity")
    return config, dataset, evidence


def _binding(path: str | Path) -> dict[str, object]:
    return bind_file(path)


def _nvidia_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gpu = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if gpu.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU sampling failed: {gpu.stderr.strip()}")
    process = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi process sampling failed: {process.stderr.strip()}"
        )

    def rows(text: str, fields: Sequence[str]) -> list[dict[str, object]]:
        parsed = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            values = [value.strip() for value in raw.split(",")]
            if len(values) != len(fields):
                raise RuntimeError("nvidia-smi returned an unexpected CSV row")
            parsed.append(dict(zip(fields, values, strict=True)))
        return parsed

    return (
        rows(gpu.stdout, ("index", "uuid", "utilization_percent", "memory_mib")),
        rows(process.stdout, ("pid", "uuid", "used_memory_mib")),
    )


def _target_gpu_snapshot(physical_index: int) -> dict[str, object]:
    gpu_rows, process_rows = _nvidia_rows()
    matching_gpu = [row for row in gpu_rows if row["index"] == str(physical_index)]
    if len(matching_gpu) != 1:
        raise RuntimeError(f"nvidia-smi did not report GPU index {physical_index}")
    gpu = matching_gpu[0]
    try:
        utilization = float(str(gpu["utilization_percent"]))
        device_memory_mib = float(str(gpu["memory_mib"]))
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned a nonnumeric GPU measurement") from error
    if (
        not math.isfinite(utilization)
        or not 0.0 <= utilization <= 100.0
        or not math.isfinite(device_memory_mib)
        or device_memory_mib < 0.0
        or type(gpu["uuid"]) is not str
        or not gpu["uuid"]
    ):
        raise RuntimeError("nvidia-smi returned an invalid GPU measurement")
    processes: list[dict[str, object]] = []
    seen_pids: set[int] = set()
    for row in process_rows:
        if row["uuid"] != gpu["uuid"]:
            continue
        try:
            host_pid = int(str(row["pid"]))
            memory_mib = float(str(row["used_memory_mib"]))
        except ValueError as error:
            raise RuntimeError(
                "nvidia-smi returned a nonnumeric compute-process measurement"
            ) from error
        if (
            host_pid <= 0
            or host_pid in seen_pids
            or not math.isfinite(memory_mib)
            or memory_mib <= 0.0
        ):
            raise RuntimeError("nvidia-smi returned an invalid compute-process row")
        seen_pids.add(host_pid)
        processes.append(
            {
                "nvidia_host_pid": host_pid,
                "process_vram_bytes": int(memory_mib * 1024**2),
            }
        )
    return {
        "gpu_index": physical_index,
        "gpu_uuid": gpu["uuid"],
        "gpu_utilization_percent": utilization,
        "device_memory_bytes": int(device_memory_mib * 1024**2),
        "compute_processes": processes,
    }


def _sample_target_gpu(
    evaluator_namespace_pid: int,
    physical_index: int,
    *,
    expected_gpu_uuid: str,
    expected_nvidia_host_pid: int | None,
) -> dict[str, object] | None:
    snapshot = _target_gpu_snapshot(physical_index)
    if snapshot["gpu_uuid"] != expected_gpu_uuid:
        raise RuntimeError("target GPU UUID changed during performance sampling")
    processes = snapshot["compute_processes"]
    if not isinstance(processes, list):
        raise RuntimeError("target GPU compute-process inventory is invalid")
    if not processes:
        return None
    if len(processes) != 1:
        raise RuntimeError(
            "target GPU is not exclusive to the evaluator during sampling"
        )
    process = processes[0]
    host_pid = int(process["nvidia_host_pid"])
    if expected_nvidia_host_pid is not None and host_pid != expected_nvidia_host_pid:
        raise RuntimeError("target GPU compute-process identity changed during sampling")
    return {
        "observed_utc": _utc_now(),
        "evaluator_namespace_pid": evaluator_namespace_pid,
        "nvidia_host_pid": host_pid,
        "gpu_index": physical_index,
        "gpu_uuid": expected_gpu_uuid,
        "gpu_utilization_percent": snapshot["gpu_utilization_percent"],
        "process_vram_bytes": process["process_vram_bytes"],
        "compute_process_count": 1,
    }


def _terminate_exact_child(process: subprocess.Popen[bytes]) -> None:
    """Bounded cleanup for the exact worker created by this runner."""

    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=10.0)


def _device_index(device: str) -> int:
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("performance evidence requires a canonical CUDA device")
    return int(device[5:])


def _run_scale_worker(
    *,
    config_path: Path,
    dataset_path: Path,
    legacy_run_root: Path,
    output_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    device: str,
    batch_size: int,
    source_scenes: int,
    target_scenes_per_city: int,
    checkpoint_count: int,
    checkpoint_arm: str | None,
    gpu_sample_interval_seconds: float,
) -> dict[str, object]:
    source_root = Path(__file__).resolve().parent.parent
    expression = (
        "import sys;sys.path.insert(0," + repr(str(source_root)) + ");"
        "from formal_v2.formal_evaluation_subset_compare import "
        "run_same_source_decomposition_comparison as r;"
        "r(config_path=" + repr(str(config_path))
        + ",dataset_path=" + repr(str(dataset_path))
        + ",legacy_run_root=" + repr(str(legacy_run_root))
        + ",report_path=" + repr(str(output_path))
        + ",device=" + repr(device)
        + ",batch_size=" + repr(batch_size)
        + ",source_scenes=" + repr(source_scenes)
        + ",target_scenes_per_city=" + repr(target_scenes_per_city)
        + ",checkpoint_count=" + repr(checkpoint_count)
        + ",checkpoint_arm=" + repr(checkpoint_arm)
        + ")"
    )
    command = (sys.executable, "-I", "-B", "-c", expression)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    physical_index = _device_index(device)
    baseline = _target_gpu_snapshot(physical_index)
    baseline_processes = baseline["compute_processes"]
    if not isinstance(baseline_processes, list) or baseline_processes:
        raise RuntimeError(
            "performance benchmark requires an empty exclusive target GPU before launch"
        )
    started_utc = _utc_now()
    started = time.monotonic()
    gpu_samples: list[dict[str, object]] = []
    with (
        stdout_path.open("xb") as stdout_handle,
        stderr_path.open("xb") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        usage = None
        status = None
        nvidia_host_pid = None
        try:
            while usage is None:
                waited_pid, waited_status, waited_usage = os.wait4(
                    process.pid, os.WNOHANG
                )
                if waited_pid == process.pid:
                    status = waited_status
                    usage = waited_usage
                    process.returncode = os.waitstatus_to_exitcode(waited_status)
                    break
                sample = _sample_target_gpu(
                    process.pid,
                    physical_index,
                    expected_gpu_uuid=str(baseline["gpu_uuid"]),
                    expected_nvidia_host_pid=nvidia_host_pid,
                )
                if sample is not None:
                    observed_host_pid = int(sample["nvidia_host_pid"])
                    if nvidia_host_pid is None:
                        nvidia_host_pid = observed_host_pid
                    gpu_samples.append(sample)
                time.sleep(gpu_sample_interval_seconds)
        except BaseException:
            _terminate_exact_child(process)
            raise
        stdout_handle.flush()
        stderr_handle.flush()
        os.fsync(stdout_handle.fileno())
        os.fsync(stderr_handle.fileno())
    if usage is None or status is None:
        raise RuntimeError("performance child exited without wait4 resource evidence")
    wall_seconds = time.monotonic() - started
    if process.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"formal subset scale worker exited {process.returncode}: {tail}"
        )
    if not gpu_samples or nvidia_host_pid is None:
        raise RuntimeError(
            "formal subset scale worker produced no exclusively bound GPU samples"
        )
    output = read_strict_json(_regular_file(output_path, "performance scale output"))
    utilization = sum(
        float(row["gpu_utilization_percent"]) for row in gpu_samples
    ) / len(gpu_samples)
    return {
        "command": list(command),
        "pid": process.pid,
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "exit_code": process.returncode,
        "wall_seconds": wall_seconds,
        "cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_bytes": int(usage.ru_maxrss * 1024),
        "gpu_utilization_percent": utilization,
        "gpu_utilization_statistic": "MEAN",
        "gpu_sample_count": len(gpu_samples),
        "gpu_sample_interval_seconds": gpu_sample_interval_seconds,
        "peak_vram_bytes": max(
            int(row["process_vram_bytes"]) for row in gpu_samples
        ),
        "gpu_process_binding": {
            "method": GPU_PROCESS_BINDING_METHOD,
            "evaluator_namespace_pid": process.pid,
            "nvidia_host_pid": nvidia_host_pid,
            "gpu_index": physical_index,
            "gpu_uuid": baseline["gpu_uuid"],
            "baseline_compute_process_count": 0,
            "target_gpu_empty_before_launch": True,
            "all_observed_samples_exclusive": True,
        },
        "gpu_samples": gpu_samples,
        "output": output,
        "stdout": _binding(stdout_path),
        "stderr": _binding(stderr_path),
    }


def _validate_scale_output(
    output: object,
    *,
    identity: Mapping[str, object],
    output_path: Path,
    checkpoint_count: int,
    device: str,
) -> None:
    if not isinstance(output, dict) or set(output) != {
        "schema_version",
        "status",
        "passed",
        "created_utc",
        "contract",
        "inputs",
        "selection",
        "comparison",
        "gate_required_aggregates",
        "execution",
    }:
        raise RuntimeError("formal subset performance output fields are invalid")
    try:
        valid = (
            output["schema_version"] == SUBSET_REPORT_SCHEMA
            and output["status"] == "PASS"
            and output["passed"] is True
            and output["contract"]["dataset_kind"] == "formal_nonfixture_real_banks"
            and output["contract"]["legacy_source_equivalence_assessed"] is False
            and output["contract"]["scientific_float_requirement"]
            == "IEEE-754 binary64 bitwise equality"
            and output["comparison"]["exact"] is True
            and output["gate_required_aggregates"]["exact"] is True
            and output["inputs"]["dataset"]["fixture"] is False
            and output["inputs"]["dataset"]["identity_sha256"]
            == identity["dataset_sha256"]
            and output["inputs"]["config"]["identity_sha256"]
            == identity["config_sha256"]
            and output["inputs"]["implementation"][
                "comparator_source_tree_sha256"
            ]
            == identity["new_source_tree_sha256"]
            and output["selection"]["checkpoint_count"] == checkpoint_count
            and output["selection"]["requested_checkpoint_count"]
            == checkpoint_count
            and output["execution"]["device"] == device
            and Path(output["execution"]["report_path"]).resolve()
            == output_path.resolve()
            and output["execution"]["maximum_cuda_allocated_bytes"] > 0
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("formal subset performance output is incomplete") from error
    if not valid:
        raise RuntimeError("formal subset performance output failed oracle authentication")


def _performance_observation(
    *,
    scale: str,
    checkpoint_count: int,
    identity: Mapping[str, object],
    output_path: Path,
    device: str,
    batch_size: int,
    execution: Mapping[str, object],
) -> dict[str, object]:
    _validate_scale_output(
        execution["output"],
        identity=identity,
        output_path=output_path,
        checkpoint_count=checkpoint_count,
        device=device,
    )
    output_binding = _binding(output_path)
    payload = {
        "schema_version": PERFORMANCE_OBSERVATION_SCHEMA,
        "status": "PASS",
        "created_utc": _utc_now(),
        "identity_sha256": _canonical_sha256(identity),
        "source_commit": identity["new_commit"],
        "source_tree_sha256": identity["new_source_tree_sha256"],
        "scale": scale,
        "scaling_axis": "selected_formal_checkpoint_count",
        "sample_count": checkpoint_count,
        "execution_device": device,
        "batch_size": batch_size,
        "command": execution["command"],
        "pid": execution["pid"],
        "started_utc": execution["started_utc"],
        "completed_utc": execution["completed_utc"],
        "exit_code": execution["exit_code"],
        "wall_seconds": execution["wall_seconds"],
        "cpu_seconds": execution["cpu_seconds"],
        "peak_rss_bytes": execution["peak_rss_bytes"],
        "gpu_utilization_percent": execution["gpu_utilization_percent"],
        "gpu_utilization_statistic": execution["gpu_utilization_statistic"],
        "gpu_sample_count": execution["gpu_sample_count"],
        "gpu_sample_interval_seconds": execution["gpu_sample_interval_seconds"],
        "peak_vram_bytes": execution["peak_vram_bytes"],
        "gpu_process_binding": execution["gpu_process_binding"],
        "gpu_samples": execution["gpu_samples"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "output": output_binding,
        "oracle": {
            "kind": "candidate_merged_vs_scene_streaming_zero_tolerance",
            "report_schema": SUBSET_REPORT_SCHEMA,
            "equivalent": True,
            "comparison_exact": True,
            "gate_aggregates_exact": True,
            "formal_nonfixture": True,
        },
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
    }
    validate_performance_observation(payload, identity=identity)
    return payload


def validate_performance_observation(
    payload: object, *, identity: Mapping[str, object]
) -> None:
    required = {
        "schema_version",
        "status",
        "created_utc",
        "identity_sha256",
        "source_commit",
        "source_tree_sha256",
        "scale",
        "scaling_axis",
        "sample_count",
        "execution_device",
        "batch_size",
        "command",
        "pid",
        "started_utc",
        "completed_utc",
        "exit_code",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "gpu_utilization_percent",
        "gpu_utilization_statistic",
        "gpu_sample_count",
        "gpu_sample_interval_seconds",
        "peak_vram_bytes",
        "gpu_process_binding",
        "gpu_samples",
        "stdout",
        "stderr",
        "output",
        "oracle",
        "python_dont_write_bytecode",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("performance observation fields are invalid")
    if (
        payload["schema_version"] != PERFORMANCE_OBSERVATION_SCHEMA
        or payload["status"] != "PASS"
        or payload["identity_sha256"] != _canonical_sha256(identity)
        or payload["source_commit"] != identity["new_commit"]
        or payload["source_tree_sha256"] != identity["new_source_tree_sha256"]
        or payload["scale"] not in SCALE_NAMES
        or payload["scaling_axis"] != "selected_formal_checkpoint_count"
        or type(payload["sample_count"]) is not int
        or payload["sample_count"] <= 0
        or type(payload["batch_size"]) is not int
        or payload["batch_size"] <= 0
        or not isinstance(payload["execution_device"], str)
        or type(payload["pid"]) is not int
        or payload["pid"] <= 0
        or payload["exit_code"] != 0
        or payload["python_dont_write_bytecode"] is not True
        or not isinstance(payload["command"], list)
        or not payload["command"]
        or any(type(value) is not str or not value for value in payload["command"])
    ):
        raise RuntimeError("performance observation identity is invalid")
    physical_index = _device_index(payload["execution_device"])
    started = _parse_utc(payload["started_utc"], "performance started_utc")
    completed = _parse_utc(payload["completed_utc"], "performance completed_utc")
    if completed <= started:
        raise RuntimeError("performance observation chronology is invalid")
    for field in ("wall_seconds", "cpu_seconds"):
        value = payload[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"performance observation {field} is invalid")
    if type(payload["peak_rss_bytes"]) is not int or payload["peak_rss_bytes"] <= 0:
        raise RuntimeError("performance observation peak RSS is invalid")
    samples = payload["gpu_samples"]
    interval = payload["gpu_sample_interval_seconds"]
    if (
        not isinstance(samples, list)
        or not samples
        or type(payload["gpu_sample_count"]) is not int
        or payload["gpu_sample_count"] != len(samples)
        or payload["gpu_utilization_statistic"] != "MEAN"
        or type(interval) not in (int, float)
        or not math.isfinite(interval)
        or not 0.0 < interval <= 60.0
        or type(payload["peak_vram_bytes"]) is not int
        or payload["peak_vram_bytes"] <= 0
    ):
        raise RuntimeError("performance observation GPU trace is invalid")
    binding = payload["gpu_process_binding"]
    if (
        not isinstance(binding, dict)
        or set(binding) != GPU_PROCESS_BINDING_FIELDS
        or binding["method"] != GPU_PROCESS_BINDING_METHOD
        or binding["evaluator_namespace_pid"] != payload["pid"]
        or type(binding["nvidia_host_pid"]) is not int
        or binding["nvidia_host_pid"] <= 0
        or binding["gpu_index"] != physical_index
        or type(binding["gpu_uuid"]) is not str
        or not binding["gpu_uuid"]
        or binding["baseline_compute_process_count"] != 0
        or binding["target_gpu_empty_before_launch"] is not True
        or binding["all_observed_samples_exclusive"] is not True
    ):
        raise RuntimeError("performance observation GPU process binding is invalid")
    sample_times = []
    utilization_values = []
    vram_values = []
    for row in samples:
        if not isinstance(row, dict) or set(row) != GPU_SAMPLE_FIELDS:
            raise RuntimeError("performance observation GPU sample fields are invalid")
        observed = _parse_utc(row["observed_utc"], "GPU sample observed_utc")
        utilization = row["gpu_utilization_percent"]
        if (
            observed < started
            or observed > completed
            or row["evaluator_namespace_pid"] != payload["pid"]
            or row["nvidia_host_pid"] != binding["nvidia_host_pid"]
            or row["gpu_index"] != binding["gpu_index"]
            or row["gpu_uuid"] != binding["gpu_uuid"]
            or row["compute_process_count"] != 1
            or type(utilization) not in (int, float)
            or not math.isfinite(utilization)
            or not 0.0 <= utilization <= 100.0
            or type(row["process_vram_bytes"]) is not int
            or row["process_vram_bytes"] <= 0
        ):
            raise RuntimeError("performance observation GPU sample is invalid")
        sample_times.append(observed)
        utilization_values.append(float(utilization))
        vram_values.append(int(row["process_vram_bytes"]))
    if sample_times != sorted(sample_times):
        raise RuntimeError("performance observation GPU samples are not chronological")
    mean = sum(utilization_values) / len(utilization_values)
    peak = max(vram_values)
    reported_utilization = payload["gpu_utilization_percent"]
    if (
        type(reported_utilization) not in (int, float)
        or not math.isfinite(reported_utilization)
        or not 0.0 <= reported_utilization <= 100.0
        or not math.isclose(
            float(reported_utilization), mean, rel_tol=0.0, abs_tol=1e-12
        )
        or payload["peak_vram_bytes"] != peak
    ):
        raise RuntimeError("performance observation GPU summary is inconsistent")
    for field in ("stdout", "stderr", "output"):
        if _binding(str(payload[field]["path"])) != payload[field]:
            raise RuntimeError(f"performance observation {field} binding changed")
    oracle = payload["oracle"]
    if not isinstance(oracle, dict) or oracle != {
        "kind": "candidate_merged_vs_scene_streaming_zero_tolerance",
        "report_schema": SUBSET_REPORT_SCHEMA,
        "equivalent": True,
        "comparison_exact": True,
        "gate_aggregates_exact": True,
        "formal_nonfixture": True,
    }:
        raise RuntimeError("performance observation oracle result is invalid")


def run_performance_evidence(args: argparse.Namespace) -> dict[str, object]:
    _require_python_contract()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError(
            "performance evidence requires CUDA_VISIBLE_DEVICES=0,1 for exact "
            "logical-to-physical GPU binding"
        )
    identity = load_expected_identity(args.identity)
    _, _, runtime = _production_context(
        identity, config_path=args.config, dataset_path=args.dataset
    )
    legacy_root = _regular_directory(identity["legacy_run_root"], "legacy run root")
    output_root = _validate_external_root(args.output_dir, identity)
    measurements = []
    trace_paths = []
    log_paths = []
    command_text = " ".join(sys.argv)
    for name, multiplier in zip(SCALE_NAMES, SCALE_MULTIPLIERS, strict=True):
        checkpoint_count = args.base_checkpoint_count * multiplier
        stem = name.lower().replace("2", "two-").replace("4", "four-")
        output_path = output_root / f"{stem}formal-subset.json"
        stdout_path = output_root / f"{stem}worker.stdout.log"
        stderr_path = output_root / f"{stem}worker.stderr.log"
        trace_path = output_root / f"{stem}observations.json"
        execution = _run_scale_worker(
            config_path=_regular_file(args.config, "formal config"),
            dataset_path=_regular_file(args.dataset, "formal dataset"),
            legacy_run_root=legacy_root,
            output_path=output_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            device=args.device,
            batch_size=args.batch_size,
            source_scenes=args.source_scenes,
            target_scenes_per_city=args.target_scenes_per_city,
            checkpoint_count=checkpoint_count,
            checkpoint_arm=args.checkpoint_arm,
            gpu_sample_interval_seconds=args.gpu_sample_interval_seconds,
        )
        trace = _performance_observation(
            scale=name,
            checkpoint_count=checkpoint_count,
            identity=identity,
            output_path=output_path,
            device=args.device,
            batch_size=args.batch_size,
            execution=execution,
        )
        write_json_exclusive_atomic(trace_path, trace)
        measurements.append(
            {
                "scale": name,
                "sample_count": checkpoint_count,
                "wall_seconds": execution["wall_seconds"],
                "cpu_seconds": execution["cpu_seconds"],
                "peak_rss_bytes": execution["peak_rss_bytes"],
                "execution_device": args.device,
                "gpu_utilization_percent": execution[
                    "gpu_utilization_percent"
                ],
                "gpu_utilization_statistic": "MEAN",
                "gpu_sample_count": execution["gpu_sample_count"],
                "peak_vram_bytes": execution["peak_vram_bytes"],
                "output_path": output_path,
                "oracle_equivalent": True,
            }
        )
        trace_paths.append(trace_path)
        log_paths.extend((stdout_path, stderr_path))
    performance_path = output_root / "performance.json"
    gpu_trace_path = trace_paths[-1]
    artifact_paths = [*trace_paths[:-1], *log_paths]
    report = write_performance_report(
        performance_path,
        expected_identity=identity,
        command=command_text,
        measurements=measurements,
        measurement_artifact_paths=artifact_paths,
        gpu_benchmark={
            "execution_device": args.device,
            "batch_size": args.batch_size,
            "wall_seconds": measurements[-1]["wall_seconds"],
            "gpu_utilization_percent": measurements[-1][
                "gpu_utilization_percent"
            ],
            "gpu_utilization_statistic": "MEAN",
            "gpu_sample_count": measurements[-1]["gpu_sample_count"],
            "gpu_sampling_interval_seconds": args.gpu_sample_interval_seconds,
            "peak_vram_bytes": measurements[-1]["peak_vram_bytes"],
            "artifact_path": gpu_trace_path,
        },
    )
    receipt = {
        "schema_version": PERFORMANCE_RUNNER_SCHEMA,
        "status": "PASS",
        "created_utc": _utc_now(),
        "identity_sha256": _canonical_sha256(identity),
        "runtime_provenance_sha256": runtime["runtime_provenance_sha256"],
        "performance_report": _binding(performance_path),
        "scales": [
            {"name": name, "trace": _binding(trace), "output": _binding(output)}
            for name, trace, output in zip(
                SCALE_NAMES,
                trace_paths,
                [Path(row["output_path"]) for row in measurements],
                strict=True,
            )
        ],
    }
    write_json_exclusive_atomic(output_root / "runner_receipt.json", receipt)
    return report


def _control_run_identity(
    identity: Mapping[str, object], *, runtime_provenance_sha256: str
) -> EvaluationRunIdentity:
    legacy = _regular_directory(identity["legacy_run_root"], "legacy run root")
    return EvaluationRunIdentity(
        code_revision=str(identity["new_commit"]),
        source_tree_sha256=str(identity["new_source_tree_sha256"]),
        config_sha256=str(identity["config_sha256"]),
        dataset_sha256=str(identity["dataset_sha256"]),
        migration_accepted_sha256=NO_MIGRATION_SHA256,
        legacy_checkpoint_inventory_sha256=NO_MIGRATION_SHA256,
        qualification_gate_sha256=sha256_file(legacy / "qualification" / "gate.json"),
        factorial_gate_sha256=sha256_file(legacy / "factorial" / "gate.json"),
        runtime_provenance_sha256=runtime_provenance_sha256,
        run_nonce=str(identity["new_run_nonce"]),
        compute_plan_sha256=str(identity["new_compute_plan_sha256"]),
        execution_profile=EvaluationExecutionProfile(
            execution_devices=("cuda:0", "cuda:1"), batch_size=1
        ),
        output_schema_id=STREAMING_EVALUATION_SCHEMA,
        output_schema_sha256=evaluation_output_schema_sha256(),
    )


def _split_payload(payload: bytes, count: int = 4) -> tuple[tuple[int, int, bytes], ...]:
    if type(payload) is not bytes or len(payload) < count:
        raise RuntimeError("real evaluator output is too small for control shards")
    boundaries = [len(payload) * index // count for index in range(count + 1)]
    return tuple(
        (boundaries[index], boundaries[index + 1], payload[boundaries[index] : boundaries[index + 1]])
        for index in range(count)
    )


def _shard_identities(
    run: EvaluationRunIdentity,
    source_sha256: str,
    parts: Sequence[tuple[int, int, bytes]],
) -> tuple[EvaluationShardIdentity, ...]:
    return tuple(
        EvaluationShardIdentity(
            run=run,
            checkpoint_sha256=source_sha256,
            seed=0,
            arm="technical-evidence",
            substage="real-evaluator-output",
            shard_id=f"part-{index:03d}",
            range_start=start,
            range_stop=stop,
        )
        for index, (start, stop, _payload) in enumerate(parts)
    )


def _state_bindings(root: Path) -> list[dict[str, object]]:
    return [
        _binding(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _load_parts(
    store: EvaluationResumeStore,
    identities: Sequence[EvaluationShardIdentity],
) -> bytes:
    payload = bytearray()
    for identity in identities:
        completed = store.load_completed_shard(identity)
        if completed is None:
            raise RuntimeError("control shard disappeared after completion")
        payload.extend(completed.payload_path.read_bytes())
    return bytes(payload)


def _control_process_context() -> multiprocessing.context.BaseContext:
    """Return the only process start method permitted for control evidence."""

    return multiprocessing.get_context("spawn")


def _close_control_connections(
    connections: Sequence[Connection],
) -> list[BaseException]:
    errors: list[BaseException] = []
    for connection in connections:
        try:
            connection.close()
        except BaseException as error:
            errors.append(error)
    return errors


def _reap_exact_control_child(
    process: BaseProcess,
    *,
    initial_timeout: float,
) -> int:
    """Boundedly join, then terminate/kill only the supplied child."""

    errors: list[BaseException] = []
    try:
        process.join(timeout=initial_timeout)
    except BaseException as error:
        errors.append(error)
    try:
        alive = process.is_alive()
    except BaseException as error:
        errors.append(error)
        alive = True
    if alive:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append(error)
        try:
            process.join(timeout=CONTROL_CHILD_JOIN_TIMEOUT_SECONDS)
        except BaseException as error:
            errors.append(error)
    try:
        alive = process.is_alive()
    except BaseException as error:
        errors.append(error)
        alive = True
    if alive:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append(error)
        try:
            process.join(timeout=CONTROL_CHILD_JOIN_TIMEOUT_SECONDS)
        except BaseException as error:
            errors.append(error)
    try:
        alive = process.is_alive()
    except BaseException as error:
        errors.append(error)
        alive = True
    if alive:
        raise RuntimeError("exact control child remained alive after bounded kill")
    exit_code = process.exitcode
    if type(exit_code) is not int:
        raise RuntimeError("exact control child has no exit code after reap")
    if errors:
        summary = "; ".join(
            f"{type(error).__name__}: {error}" for error in errors
        )
        raise RuntimeError(f"exact control child cleanup was not clean: {summary}")
    return exit_code


def _finish_exact_control_child(
    process: BaseProcess,
    *,
    initial_timeout: float,
) -> int:
    error: BaseException | None = None
    traceback = None
    exit_code: int | None = None
    try:
        exit_code = _reap_exact_control_child(
            process, initial_timeout=initial_timeout
        )
    except BaseException as caught:
        error = caught
        traceback = caught.__traceback__
    try:
        process.close()
    except BaseException as close_error:
        if error is None:
            raise RuntimeError("failed to close exact control child") from close_error
        error.add_note(
            "process close also failed: "
            f"{type(close_error).__name__}: {close_error}"
        )
    if error is not None:
        raise error.with_traceback(traceback)
    if exit_code is None:
        raise RuntimeError("exact control child cleanup lost its exit code")
    return exit_code


def _interrupted_writer(
    root: Path,
    run: EvaluationRunIdentity,
    identities: Sequence[EvaluationShardIdentity],
    parts: Sequence[tuple[int, int, bytes]],
) -> None:
    try:
        store = EvaluationResumeStore(root, run)
        with store.writer_lock():
            for identity, (_start, _stop, payload) in zip(
                identities[:-1], parts[:-1], strict=True
            ):
                store.commit_shard_bytes(identity, payload)
        os._exit(INTERRUPTED_EXIT_CODE)
    except BaseException:
        os._exit(CONTROL_CHILD_FAILURE_EXIT_CODE)


def _run_resume_control(
    work_root: Path,
    run: EvaluationRunIdentity,
    identities: Sequence[EvaluationShardIdentity],
    parts: Sequence[tuple[int, int, bytes]],
    source_payload: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    context = _control_process_context()
    process = context.Process(
        name="csi-pairs-resume-control",
        target=_interrupted_writer,
        args=(work_root, run, identities, parts),
    )
    try:
        process.start()
    except BaseException as error:
        try:
            child_after_error = process.pid
        except BaseException as pid_error:
            error.add_note(
                "resume-control PID inspection also failed: "
                f"{type(pid_error).__name__}: {pid_error}"
            )
            child_after_error = None
        if type(child_after_error) is int and child_after_error > 0:
            try:
                _finish_exact_control_child(
                    process,
                    initial_timeout=CONTROL_CHILD_JOIN_TIMEOUT_SECONDS,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    "resume-control cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
    child = process.pid
    if type(child) is not int or child <= 0:
        _finish_exact_control_child(
            process, initial_timeout=CONTROL_CHILD_JOIN_TIMEOUT_SECONDS
        )
        raise RuntimeError("resume control child has no valid PID")
    exit_code = _finish_exact_control_child(
        process, initial_timeout=CONTROL_CHILD_COMPLETION_TIMEOUT_SECONDS
    )
    if exit_code != INTERRUPTED_EXIT_CODE:
        raise RuntimeError("resume control did not observe the planned interruption")
    store = EvaluationResumeStore(work_root, run)
    reused = 0
    recomputed = 0
    with store.writer_lock():
        for identity, (_start, _stop, payload) in zip(identities, parts, strict=True):
            completed = store.load_or_quarantine_shard(identity)
            if completed is None:
                completed = store.commit_shard_bytes(identity, payload)
                recomputed += 1
            else:
                reused += 1
    equivalent = _load_parts(store, identities) == source_payload
    results = {
        "interrupted": True,
        "resumed": True,
        "completed": True,
        "reused_shard_count": reused,
        "recomputed_shard_count": recomputed,
        "output_equivalent": equivalent,
    }
    if results != {
        "interrupted": True,
        "resumed": True,
        "completed": True,
        "reused_shard_count": 3,
        "recomputed_shard_count": 1,
        "output_equivalent": True,
    }:
        raise RuntimeError("resume control failed its exact expected result")
    return results, {"interrupted_pid": child, "interrupted_exit_code": exit_code}


def _run_corrupt_control(
    work_root: Path,
    run: EvaluationRunIdentity,
    identities: Sequence[EvaluationShardIdentity],
    parts: Sequence[tuple[int, int, bytes]],
    source_payload: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    store = EvaluationResumeStore(work_root, run)
    with store.writer_lock():
        for identity, (_start, _stop, payload) in zip(identities, parts, strict=True):
            store.commit_shard_bytes(identity, payload)
    payload_path, _manifest_path = store.shard_paths(identities[1])
    original = payload_path.read_bytes()
    payload_path.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
    corrupt_detected = False
    with store.writer_lock():
        try:
            store.load_completed_shard(identities[1])
        except CorruptShardError:
            corrupt_detected = True
        missing = store.load_or_quarantine_shard(identities[1])
        if missing is not None:
            raise RuntimeError("corrupt shard was silently reused")
        store.commit_shard_bytes(identities[1], parts[1][2])
    quarantine_receipts = sorted(work_root.glob("quarantine/*/receipt.json"))
    unaffected = sum(
        store.load_completed_shard(identity) is not None
        for index, identity in enumerate(identities)
        if index != 1
    )
    equivalent = _load_parts(store, identities) == source_payload
    results = {
        "corruption_detected": corrupt_detected,
        "corrupt_shard_count": 1,
        "quarantined_shard_count": len(quarantine_receipts),
        "recomputed_shard_count": 1,
        "unaffected_shard_count": unaffected,
    }
    if not equivalent or results != {
        "corruption_detected": True,
        "corrupt_shard_count": 1,
        "quarantined_shard_count": 1,
        "recomputed_shard_count": 1,
        "unaffected_shard_count": 3,
    }:
        raise RuntimeError("corrupt-shard control failed its exact expected result")
    return results, {"output_equivalent": equivalent}


def _run_stale_control(
    work_root: Path,
    run: EvaluationRunIdentity,
    source_sha256: str,
    part: tuple[int, int, bytes],
) -> tuple[dict[str, object], dict[str, object]]:
    stale_nonce = hashlib.sha256(
        (run.run_nonce + ":stale-control").encode("ascii")
    ).hexdigest()
    stale_run = replace(run, run_nonce=stale_nonce)
    stale_identity = _shard_identities(stale_run, source_sha256, (part,))[0]
    active_identity = _shard_identities(run, source_sha256, (part,))[0]
    stale_store = EvaluationResumeStore(work_root, stale_run)
    with stale_store.writer_lock():
        committed = stale_store.commit_shard_bytes(stale_identity, part[2])
    before = (_binding(committed.payload_path), _binding(committed.manifest_path))
    active_store = EvaluationResumeStore(work_root, run)
    detected = False
    with active_store.writer_lock():
        try:
            active_store.load_or_quarantine_shard(active_identity)
        except StaleResumeError:
            detected = True
    after = (_binding(committed.payload_path), _binding(committed.manifest_path))
    quarantines = list(work_root.glob("quarantine/*/receipt.json"))
    results = {
        "stale_identity_detected": detected,
        "rejected_shard_count": 1,
        "reused_stale_shard_count": 0,
    }
    if before != after or quarantines or results["stale_identity_detected"] is not True:
        raise RuntimeError("stale-shard control did not preserve and reject old evidence")
    return results, {
        "stale_run_identity_sha256": stale_run.sha256,
        "active_run_identity_sha256": run.sha256,
        "persisted_files_unchanged": True,
    }


def _lock_holder(
    work_root: Path,
    run: EvaluationRunIdentity,
    identity: EvaluationShardIdentity,
    payload: bytes,
    ready_sender: Connection,
    release_receiver: Connection,
) -> None:
    exit_code = CONTROL_CHILD_FAILURE_EXIT_CODE
    try:
        store = EvaluationResumeStore(work_root, run)
        with store.writer_lock():
            store.commit_shard_bytes(identity, payload)
            ready_sender.send_bytes(b"1")
            ready_sender.close()
            if not release_receiver.poll(CONTROL_CHILD_RELEASE_TIMEOUT_SECONDS):
                raise RuntimeError("lock control parent did not release the child")
            if release_receiver.recv_bytes() != b"1":
                raise RuntimeError("lock control child received an invalid release")
        exit_code = 0
    except BaseException:
        exit_code = CONTROL_CHILD_FAILURE_EXIT_CODE
    finally:
        cleanup_errors = _close_control_connections(
            (ready_sender, release_receiver)
        )
        if cleanup_errors:
            exit_code = CONTROL_CHILD_FAILURE_EXIT_CODE
    os._exit(exit_code)


def _probe_contender_lock(
    work_root: Path,
    run: EvaluationRunIdentity,
) -> bool:
    contender = EvaluationResumeStore(work_root, run)
    try:
        with contender.writer_lock():
            pass
    except WriterLockError:
        return True
    return False


def _send_lock_release(release_sender: Connection) -> None:
    release_sender.send_bytes(b"1")


def _run_lock_control(
    work_root: Path,
    run: EvaluationRunIdentity,
    identity: EvaluationShardIdentity,
    payload: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    context = _control_process_context()
    connections: list[Connection] = []
    ready_receiver: Connection | None = None
    ready_sender: Connection | None = None
    release_receiver: Connection | None = None
    release_sender: Connection | None = None
    process: BaseProcess | None = None
    started = False
    child: int | None = None
    exit_code: int | None = None
    rejected = False
    primary_error: BaseException | None = None
    primary_traceback = None
    cleanup_errors: list[BaseException] = []
    try:
        ready_receiver, ready_sender = context.Pipe(duplex=False)
        connections.extend((ready_receiver, ready_sender))
        release_receiver, release_sender = context.Pipe(duplex=False)
        connections.extend((release_receiver, release_sender))
        process = context.Process(
            name="csi-pairs-lock-control",
            target=_lock_holder,
            args=(
                work_root,
                run,
                identity,
                payload,
                ready_sender,
                release_receiver,
            ),
        )
        process.start()
        started = True
        child = process.pid
        if type(child) is not int or child <= 0:
            raise RuntimeError("lock control child has no valid PID")
        cleanup_errors.extend(
            _close_control_connections((ready_sender, release_receiver))
        )
        if not ready_receiver.poll(CONTROL_CHILD_READY_TIMEOUT_SECONDS):
            raise RuntimeError("first lock writer did not become ready before timeout")
        try:
            ready_message = ready_receiver.recv_bytes()
        except EOFError as error:
            raise RuntimeError("first lock writer exited before readiness") from error
        if ready_message != b"1":
            raise RuntimeError("first lock writer sent invalid readiness")
        rejected = _probe_contender_lock(work_root, run)
    except BaseException as error:
        primary_error = error
        primary_traceback = error.__traceback__
    finally:
        if not started and process is not None:
            try:
                child_after_error = process.pid
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                if type(child_after_error) is int and child_after_error > 0:
                    started = True
                    child = child_after_error
        if started and release_sender is not None:
            try:
                _send_lock_release(release_sender)
            except BaseException as error:
                cleanup_errors.append(error)
        cleanup_errors.extend(_close_control_connections(connections))
        if started and process is not None:
            try:
                exit_code = _finish_exact_control_child(
                    process,
                    initial_timeout=CONTROL_CHILD_JOIN_TIMEOUT_SECONDS,
                )
            except BaseException as error:
                cleanup_errors.append(error)
    if primary_error is not None:
        if cleanup_errors:
            primary_error.add_note(
                "lock-control cleanup errors: "
                + "; ".join(
                    f"{type(error).__name__}: {error}"
                    for error in cleanup_errors
                )
            )
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_errors:
        summary = "; ".join(
            f"{type(error).__name__}: {error}" for error in cleanup_errors
        )
        raise RuntimeError(f"lock-control cleanup failed: {summary}")
    if child is None or exit_code is None:
        raise RuntimeError("lock-control cleanup lost child identity")
    store = EvaluationResumeStore(work_root, run)
    completed = store.load_completed_shard(identity)
    preserved = (
        exit_code == 0
        and completed is not None
        and completed.payload_path.read_bytes() == payload
    )
    write_count = len(list(work_root.glob("shards/**/*.manifest.json")))
    results = {
        "second_writer_attempted": True,
        "second_writer_rejected": rejected,
        "first_writer_preserved": preserved,
        "write_count": write_count,
    }
    if results != {
        "second_writer_attempted": True,
        "second_writer_rejected": True,
        "first_writer_preserved": True,
        "write_count": 1,
    }:
        raise RuntimeError("writer-lock control failed its exact expected result")
    return results, {"first_writer_pid": child, "first_writer_exit_code": exit_code}


def _run_progress_control(
    work_root: Path, run: EvaluationRunIdentity
) -> tuple[dict[str, object], dict[str, object]]:
    store = EvaluationResumeStore(work_root, run)
    observations: list[dict[str, object]] = []
    reader_errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                value = read_evaluation_status(work_root, expected_identity=run)
                if value is not None:
                    observations.append(value)
            except Exception as error:
                reader_errors.append(f"{type(error).__name__}: {error}")
            time.sleep(0.001)

    with store.writer_lock():
        writer_rows = [store.initialize_progress(4)]
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        for completed in range(1, 5):
            writer_rows.append(
                store.update_progress(
                    completed,
                    current_seed=0 if completed < 4 else None,
                    current_arm="technical-evidence" if completed < 4 else None,
                    current_substage="progress" if completed < 4 else None,
                    current_shard=f"part-{completed - 1:03d}" if completed < 4 else None,
                )
            )
            time.sleep(0.005)
        stop.set()
        thread.join(timeout=2.0)
    if thread.is_alive():
        raise RuntimeError("progress atomicity reader did not stop")
    all_rows = [*writer_rows, *observations]
    monotonic = all(
        left["completed_units"] <= right["completed_units"]
        for left, right in zip(writer_rows, writer_rows[1:])
    )
    final = read_evaluation_status(work_root, expected_identity=run)
    atomic = not reader_errors and not list(work_root.glob(".evaluation_progress.json.*.tmp"))
    results = {
        "monotonic": monotonic,
        "atomic": atomic,
        "final_complete": bool(final and final["status"] == "COMPLETE"),
        "update_count": len(all_rows),
        "completed_units": int(final["completed_units"]) if final else -1,
        "total_units": int(final["total_units"]) if final else -1,
    }
    if (
        results["monotonic"] is not True
        or results["atomic"] is not True
        or results["final_complete"] is not True
        or results["completed_units"] != 4
        or results["total_units"] != 4
    ):
        raise RuntimeError("progress control failed its exact expected result")
    return results, {
        "writer_update_count": len(writer_rows),
        "reader_observation_count": len(observations),
        "reader_errors": reader_errors,
    }


def _control_observation(
    *,
    name: str,
    identity: Mapping[str, object],
    run: EvaluationRunIdentity,
    source_path: Path,
    state_root: Path,
    results: Mapping[str, object],
    details: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": CONTROL_OBSERVATION_SCHEMA,
        "status": "PASS",
        "created_utc": _utc_now(),
        "control": name,
        "identity_sha256": _canonical_sha256(identity),
        "source_commit": identity["new_commit"],
        "source_tree_sha256": identity["new_source_tree_sha256"],
        "run_identity": run.as_dict(),
        "run_identity_sha256": run.sha256,
        "source_evaluator_output": _binding(source_path),
        "results": dict(results),
        "details": dict(details),
        "state_files": _state_bindings(state_root),
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
    }
    validate_control_observation(payload, name=name, identity=identity)
    return payload


def validate_control_observation(
    payload: object, *, name: str, identity: Mapping[str, object]
) -> None:
    fields = {
        "schema_version",
        "status",
        "created_utc",
        "control",
        "identity_sha256",
        "source_commit",
        "source_tree_sha256",
        "run_identity",
        "run_identity_sha256",
        "source_evaluator_output",
        "results",
        "details",
        "state_files",
        "python_dont_write_bytecode",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise RuntimeError("control observation fields are invalid")
    if (
        name not in CONTROL_NAMES
        or payload["schema_version"] != CONTROL_OBSERVATION_SCHEMA
        or payload["status"] != "PASS"
        or payload["control"] != name
        or payload["identity_sha256"] != _canonical_sha256(identity)
        or payload["source_commit"] != identity["new_commit"]
        or payload["source_tree_sha256"] != identity["new_source_tree_sha256"]
        or payload["python_dont_write_bytecode"] is not True
    ):
        raise RuntimeError("control observation identity is invalid")
    run = EvaluationRunIdentity.from_dict(payload["run_identity"])
    if run.sha256 != payload["run_identity_sha256"]:
        raise RuntimeError("control observation run identity digest is invalid")
    if _binding(payload["source_evaluator_output"]["path"]) != payload[
        "source_evaluator_output"
    ]:
        raise RuntimeError("control observation source output changed")
    files = payload["state_files"]
    if not isinstance(files, list) or not files:
        raise RuntimeError("control observation has no state-file evidence")
    paths = []
    for record in files:
        if _binding(record["path"]) != record:
            raise RuntimeError("control observation state-file binding changed")
        paths.append(record["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("control observation state files are not canonical")
    results = payload["results"]
    if not isinstance(results, dict) or set(results) != CONTROL_RESULT_FIELDS[name]:
        raise RuntimeError("control observation result fields are invalid")
    valid = False
    if name == "resume":
        valid = bool(
            results["interrupted"] is True
            and results["resumed"] is True
            and results["completed"] is True
            and results["output_equivalent"] is True
            and type(results["reused_shard_count"]) is int
            and results["reused_shard_count"] > 0
            and type(results["recomputed_shard_count"]) is int
            and results["recomputed_shard_count"] >= 0
        )
    elif name == "corrupt_checkpoint":
        valid = bool(
            results["corruption_detected"] is True
            and type(results["corrupt_shard_count"]) is int
            and results["corrupt_shard_count"] > 0
            and results["quarantined_shard_count"]
            == results["corrupt_shard_count"]
            and results["recomputed_shard_count"]
            == results["corrupt_shard_count"]
            and type(results["unaffected_shard_count"]) is int
            and results["unaffected_shard_count"] >= 0
        )
    elif name == "stale_checkpoint":
        valid = bool(
            results["stale_identity_detected"] is True
            and type(results["rejected_shard_count"]) is int
            and results["rejected_shard_count"] > 0
            and results["reused_stale_shard_count"] == 0
        )
    elif name == "lock":
        valid = bool(
            results["second_writer_attempted"] is True
            and results["second_writer_rejected"] is True
            and results["first_writer_preserved"] is True
            and results["write_count"] == 1
        )
    else:
        valid = bool(
            results["monotonic"] is True
            and results["atomic"] is True
            and results["final_complete"] is True
            and type(results["update_count"]) is int
            and results["update_count"] > 0
            and type(results["completed_units"]) is int
            and results["completed_units"] > 0
            and results["completed_units"] == results["total_units"]
        )
    if not valid or not isinstance(payload["details"], dict):
        raise RuntimeError("control observation result is not a strict PASS")


def _performance_source(
    performance_path: Path,
    identity: Mapping[str, object],
) -> Path:
    report = read_strict_json(performance_path)
    if not isinstance(report, dict):
        raise RuntimeError("performance report is not a JSON object")
    validate_evidence_report(
        "performance",
        report,
        expected_identity=identity,
        legacy_run_root=identity["legacy_run_root"],
        new_run_root=identity["new_run_root"],
    )
    source = _regular_file(
        report["results"]["scales"][-1]["output"]["path"],
        "4N formal evaluator output",
    )
    _validate_scale_output(
        read_strict_json(source),
        identity=identity,
        output_path=source,
        checkpoint_count=int(report["results"]["scales"][-1]["sample_count"]),
        device=str(report["results"]["scales"][-1]["execution_device"]),
    )
    return source


def run_control_evidence(args: argparse.Namespace) -> dict[str, object]:
    _require_python_contract()
    identity = load_expected_identity(args.identity)
    _, _, runtime = _production_context(
        identity, config_path=args.config, dataset_path=args.dataset
    )
    performance_path = _regular_file(args.performance_report, "performance report")
    source_path = _performance_source(performance_path, identity)
    source_payload = source_path.read_bytes()
    source_sha256 = sha256_file(source_path)
    parts = _split_payload(source_payload)
    run = _control_run_identity(
        identity,
        runtime_provenance_sha256=str(runtime["runtime_provenance_sha256"]),
    )
    identities = _shard_identities(run, source_sha256, parts)
    output_root = _validate_external_root(args.output_dir, identity)
    command_text = " ".join(sys.argv)
    reports: dict[str, object] = {}

    operations = {
        "resume": lambda root: _run_resume_control(
            root, run, identities, parts, source_payload
        ),
        "corrupt_checkpoint": lambda root: _run_corrupt_control(
            root, run, identities, parts, source_payload
        ),
        "stale_checkpoint": lambda root: _run_stale_control(
            root, run, source_sha256, parts[0]
        ),
        "lock": lambda root: _run_lock_control(
            root, run, identities[0], parts[0][2]
        ),
        "progress": lambda root: _run_progress_control(root, run),
    }
    for name in CONTROL_NAMES:
        state_root = output_root / f"{name}-state"
        state_root.mkdir(mode=0o700)
        results, details = operations[name](state_root)
        observation = _control_observation(
            name=name,
            identity=identity,
            run=run,
            source_path=source_path,
            state_root=state_root,
            results=results,
            details=details,
        )
        observation_path = output_root / f"{name}-observations.json"
        write_json_exclusive_atomic(observation_path, observation)
        report_path = output_root / f"{name}.json"
        common = {
            "path": report_path,
            "expected_identity": identity,
            "command": command_text,
            "artifact_paths": [observation_path],
        }
        if name == "resume":
            report = write_resume_report(**common, **results)
        elif name == "corrupt_checkpoint":
            report = write_corrupt_checkpoint_report(**common, **results)
        elif name == "stale_checkpoint":
            report = write_stale_checkpoint_report(**common, **results)
        elif name == "lock":
            report = write_lock_report(**common, **results)
        else:
            report = write_progress_report(**common, **results)
        reports[name] = report

    receipt = {
        "schema_version": CONTROL_RUNNER_SCHEMA,
        "status": "PASS",
        "created_utc": _utc_now(),
        "identity_sha256": _canonical_sha256(identity),
        "run_identity_sha256": run.sha256,
        "source_evaluator_output": _binding(source_path),
        "reports": {
            name: _binding(output_root / f"{name}.json") for name in CONTROL_NAMES
        },
    }
    write_json_exclusive_atomic(output_root / "runner_receipt.json", receipt)
    return reports


def _authenticated_extended_binding(
    value: object, *, label: str
) -> tuple[dict[str, object], Path]:
    fields = {"path", "bytes", "mtime_ns", "sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError(f"{label} binding fields are invalid")
    path = _regular_file(value["path"], label)
    stat = path.stat()
    if (
        type(value["bytes"]) is not int
        or value["bytes"] <= 0
        or type(value["mtime_ns"]) is not int
        or stat.st_size != value["bytes"]
        or stat.st_mtime_ns != value["mtime_ns"]
        or not _is_hex(value["sha256"], 64)
        or sha256_file(path) != value["sha256"]
    ):
        raise RuntimeError(f"{label} binding changed")
    return dict(value), path


def _authenticate_live_input_record(value: object, *, label: str) -> Path:
    if not isinstance(value, dict) or set(value) not in (
        {"path", "bytes", "mtime_ns", "sha256"},
        {"path", "bytes", "mtime_ns", "sha256", "identity_sha256"},
    ):
        raise RuntimeError(f"{label} input fields are invalid")
    path = _regular_file(value["path"], label)
    stat = path.stat()
    if (
        type(value["bytes"]) is not int
        or value["bytes"] <= 0
        or type(value["mtime_ns"]) is not int
        or stat.st_size != value["bytes"]
        or stat.st_mtime_ns != value["mtime_ns"]
        or not _is_hex(value["sha256"], 64)
        or sha256_file(path) != value["sha256"]
        or (
            "identity_sha256" in value
            and not _is_hex(value["identity_sha256"], 64)
        )
    ):
        raise RuntimeError(f"{label} input changed")
    return path


def _expected_subset_layers(
    frozen_fragment: Mapping[str, object],
    candidate_fragment: Mapping[str, object],
) -> dict[str, object]:
    cross_comparison = subset.compare_evaluation_tables(
        frozen_fragment["tables"],
        candidate_fragment["tables"],
        implementation_fields=subset.IMPLEMENTATION_EVIDENCE_FIELDS,
    )
    cross_gate = subset.gate_aggregate_report(
        frozen_fragment["tables"],
        candidate_fragment["tables"],
        implementation_fields=subset.IMPLEMENTATION_EVIDENCE_FIELDS,
    )
    same_comparison = subset.compare_evaluation_tables(
        candidate_fragment["same_source_merged_tables"],
        candidate_fragment["tables"],
    )
    same_gate = subset.gate_aggregate_report(
        candidate_fragment["same_source_merged_tables"],
        candidate_fragment["tables"],
    )
    return {
        "cross_source_end_to_end": {
            "authoritative": True,
            "legacy_source_role": "frozen_legacy_oracle",
            "candidate_source_role": "streaming_candidate",
            "comparison": cross_comparison,
            "gate_required_aggregates": cross_gate,
        },
        "same_source_decomposition": {
            "authoritative": False,
            "description": (
                "candidate-source merged-scene decomposition versus candidate-source "
                "per-scene streaming; this layer is not legacy-source evidence"
            ),
            "comparison": same_comparison,
            "gate_required_aggregates": same_gate,
        },
    }


def _authenticate_prior_subset_report(
    prior_report_path: str | Path,
    *,
    identity: Mapping[str, object],
    frozen_source_root: str | Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    report_path = _regular_file(prior_report_path, "prior real-subset report")
    legacy_root = Path(str(identity["legacy_run_root"])).resolve()
    new_root = Path(str(identity["new_run_root"])).resolve()
    if _inside(report_path, legacy_root) or _inside(report_path, new_root):
        raise RuntimeError("prior real-subset report must be external to both run roots")
    report = read_strict_json(report_path)
    if (
        not isinstance(report, dict)
        or set(report) != SUBSET_REPORT_FIELDS
        or report["schema_version"] != SUBSET_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["passed"] is not True
        or report["authoritative_layer"] != "layers.cross_source_end_to_end"
        or Path(str(report["report_path"])).resolve() != report_path
    ):
        raise RuntimeError("prior real-subset report is not an authoritative PASS")
    workers = report["workers"]
    if not isinstance(workers, dict) or set(workers) != {"legacy", "new"}:
        raise RuntimeError("prior real-subset worker inventory is invalid")
    fragments: dict[str, dict[str, object]] = {}
    requests: dict[str, dict[str, object]] = {}
    for side in ("legacy", "new"):
        worker = workers[side]
        if not isinstance(worker, dict) or set(worker) != {
            "role",
            "source_identity",
            "runtime_provenance",
            "request",
            "fragment",
            "resources",
        }:
            raise RuntimeError(f"prior {side} worker fields are invalid")
        _request_binding, request_path = _authenticated_extended_binding(
            worker["request"], label=f"prior {side} worker request"
        )
        _fragment_binding, fragment_path = _authenticated_extended_binding(
            worker["fragment"], label=f"prior {side} worker fragment"
        )
        request = read_strict_json(request_path)
        fragment = read_strict_json(fragment_path)
        if not isinstance(request, dict) or not isinstance(fragment, dict):
            raise RuntimeError(f"prior {side} worker artifacts must be JSON objects")
        requests[side] = request
        fragments[side] = fragment
    frozen_fragment = fragments["legacy"]
    old_candidate = fragments["new"]
    old_frozen_source = frozen_fragment.get("source_identity")
    old_candidate_source = old_candidate.get("source_identity")
    if not isinstance(old_frozen_source, dict) or not isinstance(
        old_candidate_source, dict
    ):
        raise RuntimeError("prior worker source identities are missing")
    subset.validate_worker_pair(
        frozen_fragment,
        old_candidate,
        expected_frozen_root=frozen_source_root,
        expected_candidate_root=old_candidate_source["source_root"],
        expected_frozen_commit=old_frozen_source["git_commit"],
        expected_candidate_commit=old_candidate_source["git_commit"],
        expected_frozen_evaluation_sha256=old_frozen_source["files"][
            "formal_evaluation.py"
        ]["sha256"],
    )
    expected_layers = _expected_subset_layers(frozen_fragment, old_candidate)
    if (
        report["layers"] != expected_layers
        or expected_layers["cross_source_end_to_end"]["comparison"]["exact"]
        is not True
        or expected_layers["cross_source_end_to_end"][
            "gate_required_aggregates"
        ]["exact"]
        is not True
        or expected_layers["same_source_decomposition"]["comparison"]["exact"]
        is not True
        or expected_layers["same_source_decomposition"][
            "gate_required_aggregates"
        ]["exact"]
        is not True
    ):
        raise RuntimeError("prior real-subset report layers do not recompute exactly")
    if (
        report["shared_inputs"] != frozen_fragment["input_bindings"]
        or report["selection"] != frozen_fragment["selection"]
        or report["read_only"]
        != {
            "passed": True,
            "legacy": frozen_fragment["read_only"],
            "new": old_candidate["read_only"],
        }
        or report["resources"].get("legacy") != frozen_fragment["resources"]
        or report["resources"].get("new") != old_candidate["resources"]
    ):
        raise RuntimeError("prior real-subset report summaries are inconsistent")
    input_files = frozen_fragment.get("input_bindings", {}).get("files")
    if not isinstance(input_files, list) or not input_files:
        raise RuntimeError("frozen legacy fragment input inventory is empty")
    for index, record in enumerate(input_files):
        _authenticate_live_input_record(
            record, label=f"frozen legacy input {index}"
        )
    frozen_audit = frozen_fragment.get("read_only", {}).get("legacy_evaluation")
    expected_evaluation_root = legacy_root / "evaluation"
    if (
        not isinstance(frozen_audit, dict)
        or frozen_audit.get("root") != str(expected_evaluation_root.resolve())
        or frozen_audit.get("unchanged") is not True
        or frozen_audit.get("before") != frozen_audit.get("after")
        or frozen_audit.get("after")
        != subset._directory_tree_snapshot(expected_evaluation_root)
    ):
        raise RuntimeError("frozen legacy evaluation snapshot changed")
    live_frozen_source = subset.source_identity(
        frozen_source_root, require_streaming=False
    )
    for key in (
        "source_root",
        "git_root",
        "git_commit",
        "tracked_tree_clean",
        "tracked_status",
        "untracked_scientific_paths",
        "tracked_diff_sha256",
    ):
        if live_frozen_source[key] != old_frozen_source[key]:
            raise RuntimeError(f"frozen legacy source changed: {key}")
    if (
        set(live_frozen_source["files"])
        != {"formal_evaluation_subset_compare.py", "formal_evaluation.py"}
        or live_frozen_source["files"]["formal_evaluation.py"]
        != old_frozen_source["files"]["formal_evaluation.py"]
    ):
        raise RuntimeError("frozen legacy evaluation implementation changed")
    current_harness = _regular_file(
        Path(subset.__file__), "current subset comparator harness"
    )
    harness_record = old_frozen_source["files"].get(
        "formal_evaluation_subset_compare.py"
    )
    if (
        not isinstance(harness_record, dict)
        or harness_record.get("sha256") != sha256_file(current_harness)
        or live_frozen_source["files"]["formal_evaluation_subset_compare.py"][
            "sha256"
        ]
        != harness_record.get("sha256")
        or old_candidate_source["files"].get(
            "formal_evaluation_subset_compare.py"
        )
        != harness_record
    ):
        raise RuntimeError("subset comparator harness SHA changed; reuse is forbidden")
    frozen_request = requests["legacy"]
    if (
        frozen_request.get("schema_version") != subset.WORKER_REQUEST_SCHEMA
        or frozen_request.get("role") != "frozen_legacy"
        or frozen_request.get("expected_git_commit") != identity["legacy_commit"]
        or frozen_fragment["source_identity"].get("source_tree_sha256")
        != identity["legacy_source_tree_sha256"]
        or frozen_fragment.get("input_bindings", {}).get("dataset_sha256")
        != identity["dataset_sha256"]
        or frozen_fragment.get("input_bindings", {}).get("config_sha256")
        != identity["config_sha256"]
    ):
        raise RuntimeError("frozen legacy fragment does not match final migration identity")
    return report, frozen_request, frozen_fragment


def _candidate_request_from_frozen(
    frozen_request: Mapping[str, object],
    *,
    fragment_path: Path,
    candidate_identity: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "config_path",
        "dataset_path",
        "legacy_run_root",
        "device",
        "batch_size",
        "source_scenes",
        "target_scenes_per_city",
        "checkpoint_count",
        "checkpoint_arm",
        "role",
        "fragment_path",
        "expected_git_commit",
        "expected_evaluation_sha256",
    }
    if not isinstance(frozen_request, Mapping) or set(frozen_request) != expected_fields:
        raise RuntimeError("frozen worker request fields are invalid")
    request = dict(frozen_request)
    request.update(
        {
            "role": "streaming_candidate",
            "fragment_path": str(fragment_path.resolve()),
            "expected_git_commit": candidate_identity["git_commit"],
            "expected_evaluation_sha256": candidate_identity["files"][
                "formal_evaluation.py"
            ]["sha256"],
        }
    )
    for key in expected_fields - {
        "role",
        "fragment_path",
        "expected_git_commit",
        "expected_evaluation_sha256",
    }:
        if request[key] != frozen_request[key]:
            raise RuntimeError("candidate refresh request changed a frozen common field")
    return request


def _run_candidate_with_authenticated_harness(
    *,
    harness_path: Path,
    candidate_source_root: Path,
    request_path: Path,
) -> None:
    harness = _regular_file(harness_path, "authenticated subset comparator harness")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment[subset.BOUND_SOURCE_ROOT_ENV] = str(candidate_source_root)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-B",
            str(harness),
            "worker",
            "--request",
            str(request_path),
        ),
        cwd=candidate_source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "refreshed candidate subset worker failed with exit "
            f"{completed.returncode}: {completed.stderr[-4000:]}"
        )


def run_subset_refresh(args: argparse.Namespace) -> dict[str, object]:
    _require_python_contract()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("subset refresh requires CUDA_VISIBLE_DEVICES=0,1")
    identity = load_expected_identity(args.identity)
    target = Path(args.report)
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"refusing to overwrite subset refresh report: {target}")
    target_parent = _regular_directory(target.parent, "subset refresh report parent")
    target = (target_parent / target.name).resolve()
    legacy_root = Path(str(identity["legacy_run_root"])).resolve()
    new_root = Path(str(identity["new_run_root"])).resolve()
    if _inside(target, legacy_root) or _inside(target, new_root):
        raise RuntimeError("subset refresh report must be external to both run roots")
    prior, frozen_request, frozen_fragment = _authenticate_prior_subset_report(
        args.prior_report,
        identity=identity,
        frozen_source_root=args.frozen_source_root,
    )
    _production_context(
        identity,
        config_path=frozen_request["config_path"],
        dataset_path=frozen_request["dataset_path"],
    )
    candidate_root = Path(__file__).resolve().parent.parent
    candidate_identity = subset.source_identity(
        candidate_root, require_streaming=True
    )
    if (
        candidate_identity["git_commit"] != identity["new_commit"]
        or candidate_identity["tracked_tree_clean"] is not True
        or candidate_identity["untracked_scientific_paths"] != []
    ):
        raise RuntimeError("final candidate source identity is not clean and exact")
    current_harness = candidate_identity["files"][
        "formal_evaluation_subset_compare.py"
    ]
    frozen_harness = frozen_fragment["source_identity"]["files"][
        "formal_evaluation_subset_compare.py"
    ]
    if current_harness["sha256"] != frozen_harness["sha256"]:
        raise RuntimeError("final candidate comparator harness SHA changed")
    request_path = subset._worker_artifact_path(target, "streaming_candidate.request")
    fragment_path = subset._worker_artifact_path(target, "streaming_candidate.fragment")
    request = _candidate_request_from_frozen(
        frozen_request,
        fragment_path=fragment_path,
        candidate_identity=candidate_identity,
    )
    subset.write_report_atomic(request_path, request)
    started = time.monotonic()
    _run_candidate_with_authenticated_harness(
        harness_path=Path(str(frozen_harness["path"])),
        candidate_source_root=candidate_root,
        request_path=request_path,
    )
    candidate_fragment = read_strict_json(
        _regular_file(fragment_path, "refreshed candidate fragment")
    )
    if not isinstance(candidate_fragment, dict):
        raise RuntimeError("refreshed candidate fragment is not a JSON object")
    subset.validate_worker_pair(
        frozen_fragment,
        candidate_fragment,
        expected_frozen_root=args.frozen_source_root,
        expected_candidate_root=candidate_root,
        expected_frozen_commit=str(identity["legacy_commit"]),
        expected_candidate_commit=str(identity["new_commit"]),
        expected_frozen_evaluation_sha256=frozen_fragment["source_identity"][
            "files"
        ]["formal_evaluation.py"]["sha256"],
    )
    frozen_request_common = {
        key: value
        for key, value in frozen_request.items()
        if key
        not in {
            "role",
            "fragment_path",
            "expected_git_commit",
            "expected_evaluation_sha256",
        }
    }
    candidate_request_common = {
        key: value
        for key, value in request.items()
        if key
        not in {
            "role",
            "fragment_path",
            "expected_git_commit",
            "expected_evaluation_sha256",
        }
    }
    if frozen_request_common != candidate_request_common:
        raise RuntimeError("refreshed candidate request changed frozen execution inputs")
    layers = _expected_subset_layers(frozen_fragment, candidate_fragment)
    passed = bool(
        layers["cross_source_end_to_end"]["comparison"]["exact"]
        and layers["cross_source_end_to_end"]["gate_required_aggregates"]["exact"]
        and layers["same_source_decomposition"]["comparison"]["exact"]
        and layers["same_source_decomposition"]["gate_required_aggregates"]["exact"]
    )
    if not passed:
        raise RuntimeError("refreshed candidate is not bitwise equivalent to frozen legacy")
    legacy_worker = prior["workers"]["legacy"]
    candidate_worker = {
        "role": "streaming_candidate",
        "source_identity": candidate_fragment["source_identity"],
        "runtime_provenance": candidate_fragment["runtime_provenance"],
        "request": subset._input_record(request_path),
        "fragment": subset._input_record(fragment_path),
        "resources": candidate_fragment["resources"],
    }
    report = {
        "schema_version": SUBSET_REPORT_SCHEMA,
        "status": "PASS",
        "passed": True,
        "created_utc": _utc_now(),
        "authoritative_layer": "layers.cross_source_end_to_end",
        "contract": dict(prior["contract"]),
        "workers": {"legacy": legacy_worker, "new": candidate_worker},
        "shared_inputs": frozen_fragment["input_bindings"],
        "selection": frozen_fragment["selection"],
        "layers": layers,
        "read_only": {
            "passed": bool(
                frozen_fragment["read_only"]["passed"]
                and candidate_fragment["read_only"]["passed"]
            ),
            "legacy": frozen_fragment["read_only"],
            "new": candidate_fragment["read_only"],
        },
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "legacy": frozen_fragment["resources"],
            "new": candidate_fragment["resources"],
        },
        "report_path": str(target),
    }
    _validate_real_subset_report(
        report,
        report_path=target,
        expected_identity=identity,
        legacy_root=legacy_root,
        new_root=new_root,
    )
    subset.write_report_atomic(target, report)
    receipt_path = target.with_name(f"{target.name}.refresh-receipt.json")
    receipt = {
        "schema_version": SUBSET_REFRESH_RECEIPT_SCHEMA,
        "status": "PASS",
        "created_utc": _utc_now(),
        "identity_sha256": _canonical_sha256(identity),
        "prior_report": _binding(args.prior_report),
        "reused_frozen_request": _binding(
            frozen_fragment["request"]["path"]
        ),
        "reused_frozen_fragment": _binding(
            legacy_worker["fragment"]["path"]
        ),
        "comparator_harness_sha256": current_harness["sha256"],
        "candidate_request": _binding(request_path),
        "candidate_fragment": _binding(fragment_path),
        "refreshed_report": _binding(target),
        "legacy_worker_rerun": False,
        "candidate_worker_rerun": True,
    }
    write_json_exclusive_atomic(receipt_path, receipt)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real formal migration performance and resume-integrity evidence."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--identity", required=True)
    common.add_argument("--config", required=True)
    common.add_argument("--dataset", required=True)
    common.add_argument("--output-dir", required=True)
    performance = subparsers.add_parser("performance", parents=[common])
    performance.add_argument("--device", default="cuda:0")
    performance.add_argument("--batch-size", type=int, default=256)
    performance.add_argument("--base-checkpoint-count", type=int, default=1)
    performance.add_argument("--source-scenes", type=int, default=1)
    performance.add_argument("--target-scenes-per-city", type=int, default=1)
    performance.add_argument("--checkpoint-arm")
    performance.add_argument(
        "--gpu-sample-interval-seconds",
        type=float,
        default=DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS,
    )
    controls = subparsers.add_parser("controls", parents=[common])
    controls.add_argument("--performance-report", required=True)
    refresh = subparsers.add_parser(
        "refresh-subset",
        help=(
            "reuse an authenticated frozen-legacy worker fragment and rerun only "
            "the final committed candidate worker"
        ),
    )
    refresh.add_argument("--identity", required=True)
    refresh.add_argument("--prior-report", required=True)
    refresh.add_argument("--report", required=True)
    refresh.add_argument("--frozen-source-root", required=True)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.command == "performance":
        for field in (
            "batch_size",
            "base_checkpoint_count",
            "source_scenes",
            "target_scenes_per_city",
        ):
            if type(getattr(args, field)) is not int or getattr(args, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        interval = args.gpu_sample_interval_seconds
        if (
            type(interval) not in (int, float)
            or not math.isfinite(interval)
            or interval <= 0.0
            or interval > 60.0
        ):
            raise ValueError("GPU sample interval must be in (0, 60] seconds")
        _device_index(args.device)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    if args.command == "performance":
        result = run_performance_evidence(args)
        payload = {
            "status": result["status"],
            "report": str(Path(args.output_dir).resolve() / "performance.json"),
        }
    elif args.command == "controls":
        result = run_control_evidence(args)
        payload = {
            "status": "PASS",
            "reports": {
                name: str(Path(args.output_dir).resolve() / f"{name}.json")
                for name in result
            },
        }
    else:
        result = run_subset_refresh(args)
        payload = {
            "status": result["status"],
            "report": str(Path(args.report).resolve()),
            "legacy_worker_rerun": False,
            "candidate_worker_rerun": True,
        }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
