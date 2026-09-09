from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import bind_rows, evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json


ADAPTER_MANIFEST_SCHEMA = "csi-pairs-v6-external-validity-adapter-v3"
INDEPENDENT_ADAPTER_MANIFEST_SCHEMA = (
    "csi-pairs-v6-external-validity-independent-adapter-v1"
)
ARCHIVE_MANIFEST_SCHEMA = "csi-pairs-v6-external-validity-archive-v1"
INDEPENDENT_RT_SCENE_SCHEMA = "csi-pairs-v6-independent-rt-scene-manifest-v1"


def run_external_validity(config, dataset, manifest_path, output_root):
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root, config, dataset, ("external_validation",)
    )
    manifest_file = Path(manifest_path).resolve()
    manifest = read_strict_json(manifest_file)
    _validate_manifest(manifest)
    require_independent_primary_engine(dataset, manifest)
    output_dir = Path(output_root) / "external_validity"
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_mode = _execution_mode(manifest)
    adapter_source = None
    bound_manifest = output_dir / "adapter_manifest.json"
    evidence = evidence_context(
        config,
        dataset,
        (
            "DIAGNOSTIC_NOT_CLAIM"
            if execution_mode == "authenticated_precomputed_rt_archive"
            else "FORBIDDEN"
            if dataset.is_fixture
            else "CANDIDATE_NOT_CLAIM"
        ),
    )
    if execution_mode == "authenticated_sionna_adapter":
        adapter_source = _verify_adapter_source(manifest)
        write_json(
            bound_manifest,
            {**manifest, "adapter_source_path": str(adapter_source)},
        )
        command = [
            value.format(
                dataset=str(dataset.source_path),
                output=str(output_dir),
                run_root=str(Path(output_root).resolve()),
                project_root=str(Path(__file__).resolve().parents[1]),
                python=sys.executable,
                adapter_source=str(adapter_source),
            )
            for value in manifest["command"]
        ]
        independently_probed_runtime = _probe_sionna_runtime(command)
        rt_scene_manifest, rt_scene_manifest_path = _bind_rt_scene_manifest(
            command, dataset, output_dir
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=_adapter_environment(Path(__file__).resolve().parents[1]),
        )
        (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        result_path = output_dir / "external_csi.npz"
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError("external-validity adapter failed")
        runtime_path, runtime_record = _authenticate_sionna_runtime(
            command,
            output_dir,
            independently_probed_runtime,
        )
        external_engine_config_path = None
    elif execution_mode == "authenticated_independent_rt_adapter":
        adapter_source = _verify_adapter_source(manifest)
        (
            _source_context,
            source_config,
            rt_scene_manifest,
            source_assets,
        ) = _verify_independent_adapter_inputs(
            manifest, manifest_file.parent, dataset
        )
        external_engine_config_path = _copy_regular_file(
            source_config,
            output_dir / "external_engine_config.bin",
            "independent RT engine configuration",
        )
        rt_scene_manifest = _stage_independent_source_assets(
            rt_scene_manifest, source_assets, output_dir
        )
        rt_scene_manifest_path = output_dir / "rt_scene_manifest.json"
        write_json(rt_scene_manifest_path, rt_scene_manifest)
        bound_payload = {
            **manifest,
            "adapter_source_path": str(adapter_source),
            "adapter_source_sha256": sha256_file(adapter_source),
            "engine_config_path": str(external_engine_config_path.resolve()),
            "engine_config_sha256": sha256_file(external_engine_config_path),
            "rt_scene_manifest_path": str(rt_scene_manifest_path.resolve()),
            "rt_scene_manifest_sha256": sha256_file(rt_scene_manifest_path),
        }
        write_json(bound_manifest, bound_payload)
        command = _render_independent_adapter_command(
            manifest,
            dataset=dataset.source_path,
            output=output_dir,
            output_root=output_root,
            adapter_source=adapter_source,
            engine_config=external_engine_config_path,
            scene_manifest=rt_scene_manifest_path,
        )
        independently_probed_runtime = _probe_independent_runtime(
            command, external_engine_config_path
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=_independent_adapter_environment(
                Path(__file__).resolve().parents[1]
            ),
        )
        (output_dir / "stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output_dir / "stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        result_path = output_dir / "external_csi.npz"
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError("independent external-validity adapter failed")
        runtime_path, runtime_record = _authenticate_independent_runtime(
            command,
            output_dir,
            independently_probed_runtime,
            external_engine_config_path,
        )
    else:
        (
            source_csi,
            _source_context,
            source_config,
            rt_scene_manifest,
            source_assets,
        ) = _verify_archive_inputs(manifest, manifest_file.parent, dataset)
        result_path = _copy_regular_file(
            source_csi, output_dir / "external_csi.npz", "external CSI archive"
        )
        external_engine_config_path = _copy_regular_file(
            source_config,
            output_dir / "external_engine_config.bin",
            "independent RT engine configuration",
        )
        rt_scene_manifest = _stage_independent_source_assets(
            rt_scene_manifest, source_assets, output_dir
        )
        rt_scene_manifest_path = output_dir / "rt_scene_manifest.json"
        write_json(rt_scene_manifest_path, rt_scene_manifest)
        write_json(
            bound_manifest,
            {
                **manifest,
                "external_csi_path": str(result_path.resolve()),
                "external_csi_sha256": sha256_file(result_path),
                "rt_scene_manifest_path": str(rt_scene_manifest_path.resolve()),
                "rt_scene_manifest_sha256": sha256_file(rt_scene_manifest_path),
                "engine_config_path": str(external_engine_config_path.resolve()),
                "engine_config_sha256": sha256_file(external_engine_config_path),
            },
        )
        runtime_path = None
        runtime_record = None
    expected_registry = _expected_external_registry(
        config, dataset, Path(output_root), rt_scene_manifest
    )
    external_csi = _load_external_csi(result_path, dataset)
    rows = _rows_from_external_csi(external_csi, expected_registry)
    active = [row for row in rows if row["route"] == "active"]
    null = [row for row in rows if row["route"] == "null"]
    if (
        len({row["canonical_base_map_digest"] for row in active}) < 2
        or len({row["canonical_base_map_digest"] for row in null}) < 2
    ):
        raise RuntimeError("G8 requires at least two independent base-map clusters in active and null strata")
    agreement = _cluster_direction_interval(
        active,
        int(config["external_validity"]["bootstrap_resamples"]),
    )
    equivalence = _paired_bank_equivalence(
        null,
        float(config["external_validity"]["null_equivalence_margin"]),
        int(config["external_validity"]["bootstrap_resamples"]),
    )
    diagnostic_statistics_passed = bool(
        agreement["ci95_low"]
        >= float(config["external_validity"]["minimum_active_direction_agreement"])
        and equivalence["passed"]
    )
    passed = bool(
        execution_mode in {
            "authenticated_sionna_adapter",
            "authenticated_independent_rt_adapter",
        }
        and diagnostic_statistics_passed
    )
    write_csv(output_dir / "validated_paired_effects.csv", bind_rows(rows, evidence))
    gate = {
        "schema_version": "csi-pairs-v6-external-validity-gate-v4",
        "status": (
            "PASS"
            if passed
            else "DIAGNOSTIC_NOT_CLAIM"
            if execution_mode == "authenticated_precomputed_rt_archive"
            else "FAIL"
        ),
        "passed": passed,
        "diagnostic_statistics_passed": diagnostic_statistics_passed,
        **evidence,
        "gate": "G8",
        "evidence_type": manifest["evidence_type"],
        "execution_mode": execution_mode,
        "engine_family": _manifest_engine_family(manifest),
        "source_revision": manifest["source_revision"],
        "license_id": manifest["license_id"],
        "adapter_source_path": str(adapter_source) if adapter_source is not None else None,
        "adapter_source_sha256": manifest.get("adapter_source_sha256"),
        "input_manifest_path": bound_manifest.name,
        "input_manifest_sha256": sha256_file(bound_manifest),
        "rt_scene_manifest_path": rt_scene_manifest_path.name,
        "rt_scene_manifest_sha256": sha256_file(rt_scene_manifest_path),
        "external_csi_path": result_path.name,
        "external_csi_sha256": sha256_file(result_path),
        "external_engine_config_path": external_engine_config_path.name if external_engine_config_path is not None else None,
        "external_engine_config_sha256": sha256_file(external_engine_config_path) if external_engine_config_path is not None else None,
        "external_runtime_provenance_path": runtime_path.name if runtime_path is not None else None,
        "external_runtime_provenance_sha256": sha256_file(runtime_path) if runtime_path is not None else None,
        "external_runtime_environment_sha256": runtime_record["environment_sha256"] if runtime_record is not None else None,
        "external_runtime_provenance": runtime_record,
        "external_csi_contract": "outer-recomputed-direction-and-effect-from-raw-csi-v1",
        "external_scene_count": int(external_csi.shape[0]),
        "active_direction_agreement": agreement["estimate"],
        "active_direction_agreement_ci95_low": agreement["ci95_low"],
        "active_direction_agreement_ci95_high": agreement["ci95_high"],
        "active_direction_cluster_count": agreement["base_map_cluster_count"],
        "minimum_active_direction_agreement": float(
            config["external_validity"]["minimum_active_direction_agreement"]
        ),
        "null_equivalence": equivalence,
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("external-validity manifest must be an object")
    schema = manifest.get("schema_version")
    if schema == ADAPTER_MANIFEST_SCHEMA:
        required = {
            "schema_version", "evidence_type", "source_revision", "license_id", "command",
            "adapter_source_path", "adapter_source_sha256",
        }
    elif schema == INDEPENDENT_ADAPTER_MANIFEST_SCHEMA:
        required = {
            "schema_version", "evidence_type", "engine_family", "source_revision",
            "license_id", "command", "adapter_source_path", "adapter_source_sha256",
            "engine_config_path", "engine_config_sha256",
            "rt_scene_manifest_path", "rt_scene_manifest_sha256",
        }
    elif schema == ARCHIVE_MANIFEST_SCHEMA:
        required = {
            "schema_version", "evidence_type", "engine_family", "source_revision",
            "license_id", "external_csi_path", "external_csi_sha256",
            "engine_config_path", "engine_config_sha256",
            "rt_scene_manifest_path", "rt_scene_manifest_sha256",
        }
    else:
        raise ValueError("external-validity manifest schema mismatch")
    if set(manifest) != required:
        raise ValueError("external-validity manifest fields must be exact")
    if manifest["evidence_type"] != "independent_rt_engine":
        raise ValueError("external-validity evidence type is unsupported")
    for key in ("source_revision", "license_id"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"external-validity {key} must be nonempty")
    if schema in {ADAPTER_MANIFEST_SCHEMA, INDEPENDENT_ADAPTER_MANIFEST_SCHEMA}:
        if (
            not isinstance(manifest["command"], list)
            or not manifest["command"]
            or any(not isinstance(value, str) for value in manifest["command"])
        ):
            raise ValueError("external-validity command must be nonempty argv")
        if not _command_executes_adapter_source(
            manifest
        ):
            raise ValueError(
                "external-validity command must execute the authenticated adapter module"
            )
        if not _lower_sha256(manifest["adapter_source_sha256"]):
            raise ValueError("external-validity adapter source hash must be lowercase SHA-256")
        if schema == INDEPENDENT_ADAPTER_MANIFEST_SCHEMA:
            for flag, placeholder in (
                ("--dataset", "{dataset}"),
                ("--output", "{output}"),
                ("--engine-config", "{engine_config}"),
                ("--scene-manifest", "{scene_manifest}"),
            ):
                command = manifest["command"]
                if command.count(flag) != 1:
                    raise ValueError(
                        f"independent RT command must bind exactly one {flag}"
                    )
                index = command.index(flag)
                if index + 1 >= len(command) or command[index + 1] != placeholder:
                    raise ValueError(
                        f"independent RT command must bind {flag} to {placeholder}"
                    )
    if schema in {INDEPENDENT_ADAPTER_MANIFEST_SCHEMA, ARCHIVE_MANIFEST_SCHEMA}:
        if not isinstance(manifest["engine_family"], str) or not manifest["engine_family"].strip():
            raise ValueError("external-validity engine family must be nonempty")
        for key in (
            "engine_config_path", "rt_scene_manifest_path"
        ):
            if not isinstance(manifest[key], str) or not manifest[key].strip():
                raise ValueError(f"external-validity {key} must be nonempty")
        for key in (
            "engine_config_sha256",
            "rt_scene_manifest_sha256",
        ):
            if not _lower_sha256(manifest[key]):
                raise ValueError(f"external-validity {key} must be lowercase SHA-256")
    if schema == ARCHIVE_MANIFEST_SCHEMA:
        if not isinstance(manifest["external_csi_path"], str) or not manifest["external_csi_path"].strip():
            raise ValueError("external-validity external_csi_path must be nonempty")
        if not _lower_sha256(manifest["external_csi_sha256"]):
            raise ValueError("external-validity external_csi_sha256 must be lowercase SHA-256")


def _execution_mode(manifest) -> str:
    if manifest["schema_version"] == ADAPTER_MANIFEST_SCHEMA:
        return "authenticated_sionna_adapter"
    if manifest["schema_version"] == INDEPENDENT_ADAPTER_MANIFEST_SCHEMA:
        return "authenticated_independent_rt_adapter"
    if manifest["schema_version"] == ARCHIVE_MANIFEST_SCHEMA:
        return "authenticated_precomputed_rt_archive"
    raise ValueError("external-validity manifest schema mismatch")


def require_claim_eligible_manifest(manifest) -> None:
    """Reject archive-only diagnostics at every formal authorization boundary."""
    _validate_manifest(manifest)
    if _execution_mode(manifest) not in {
        "authenticated_sionna_adapter",
        "authenticated_independent_rt_adapter",
    }:
        raise RuntimeError(
            "precomputed RT archives are DIAGNOSTIC_NOT_CLAIM and cannot satisfy formal G8"
        )


def _manifest_engine_family(manifest) -> str:
    if manifest["schema_version"] == ADAPTER_MANIFEST_SCHEMA:
        return "sionna"
    return str(manifest["engine_family"]).strip().lower()


def require_independent_primary_engine(dataset, manifest) -> None:
    if bool(dataset.is_fixture):
        return
    primary = dataset.engine_config.get("engine", {})
    if not isinstance(primary, dict):
        return
    primary_name = str(primary.get("name", "")).lower()
    primary_revision = str(
        primary.get("source_revision", primary.get("sionna_revision", ""))
    )
    external_family = _manifest_engine_family(manifest)
    external_revision = str(manifest["source_revision"])
    same_family = bool(
        external_family
        and (
            external_family in primary_name.replace("_", "-")
            or ("sionna" in primary_name and "sionna" in external_family)
        )
    )
    if same_family or (
        primary_revision
        and (
            primary_revision == external_revision
            or primary_revision in external_revision
        )
    ):
        raise RuntimeError(
            "G8 engine is not independent of the primary dataset engine"
        )


def _command_executes_adapter_source(manifest):
    command = manifest["command"]
    adapter_source_path = manifest["adapter_source_path"]
    if Path(adapter_source_path).suffix != ".py":
        return False
    executable = (
        "{project_root}/formal_v2/external_adapters/.runtime-sionna/venv/bin/python"
        if manifest["schema_version"] == ADAPTER_MANIFEST_SCHEMA
        else "{project_root}/formal_v2/external_adapters/.runtime-differt/venv/bin/python"
    )
    return bool(
        len(command) >= 2
        and command[0] == executable
        and command[1] == "{adapter_source}"
    )


def _render_independent_adapter_command(
    manifest,
    *,
    dataset,
    output,
    output_root,
    adapter_source,
    engine_config,
    scene_manifest,
):
    if _execution_mode(manifest) != "authenticated_independent_rt_adapter":
        raise ValueError("external-validity manifest is not an independent adapter")
    replacements = {
        "dataset": str(Path(dataset).resolve()),
        "output": str(Path(output).resolve()),
        "run_root": str(Path(output_root).resolve()),
        "project_root": str(Path(__file__).resolve().parents[1]),
        "python": sys.executable,
        "adapter_source": str(Path(adapter_source).resolve()),
        "engine_config": str(Path(engine_config).resolve()),
        "scene_manifest": str(Path(scene_manifest).resolve()),
    }
    command = [value.format(**replacements) for value in manifest["command"]]
    if any("{" in value or "}" in value for value in command):
        raise RuntimeError("independent RT command contains an unresolved placeholder")
    for flag, expected in (
        ("--dataset", replacements["dataset"]),
        ("--output", replacements["output"]),
        ("--engine-config", replacements["engine_config"]),
        ("--scene-manifest", replacements["scene_manifest"]),
    ):
        if command.count(flag) != 1:
            raise RuntimeError(f"independent RT command must bind exactly one {flag}")
        index = command.index(flag)
        if index + 1 >= len(command) or command[index + 1] != expected:
            raise RuntimeError(f"independent RT command does not bind staged {flag}")
    return command


def _adapter_environment(project_root):
    root = str(Path(project_root).resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
    }


def _independent_adapter_environment(project_root):
    return {
        **_adapter_environment(project_root),
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORMS": "cpu",
    }


def _probe_sionna_runtime(command):
    from .formal_external_runtime import probe_external_runtime

    project_root = Path(__file__).resolve().parents[1]
    return probe_external_runtime(
        command[0],
        "sionna",
        project_root,
        require_execution_ready=True,
    )


def _authenticate_sionna_runtime(command, output_dir, independently_probed):
    from .formal_external_runtime import validate_external_runtime

    path = Path(output_dir) / "runtime_provenance.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Sionna adapter omitted regular runtime_provenance.json")
    record = read_strict_json(path)
    validate_external_runtime(
        record,
        profile="sionna",
        executable=command[0],
        require_execution_ready=True,
    )
    if record != independently_probed:
        raise RuntimeError(
            "Sionna runtime provenance differs from the independently probed interpreter"
        )
    return path, record


def _probe_independent_runtime(command, engine_config_path):
    probe = [
        command[0],
        command[1],
        "--probe-runtime",
        "--engine-config",
        str(Path(engine_config_path).resolve()),
    ]
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        probe,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=project_root,
        env=_independent_adapter_environment(project_root),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "independent RT runtime provenance probe failed: "
            f"{completed.stderr.strip()}"
        )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "independent RT runtime provenance probe emitted invalid JSON"
        ) from error
    return _validate_independent_runtime(
        record,
        executable=command[0],
        engine_config_sha256=sha256_file(engine_config_path),
    )


def _authenticate_independent_runtime(
    command, output_dir, independently_probed, engine_config_path
):
    path = Path(output_dir) / "runtime_provenance.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "independent RT adapter omitted regular runtime_provenance.json"
        )
    record = _validate_independent_runtime(
        read_strict_json(path),
        executable=command[0],
        engine_config_sha256=sha256_file(engine_config_path),
    )
    if record != independently_probed:
        raise RuntimeError(
            "independent RT runtime provenance differs from the probed interpreter"
        )
    return path, record


def _validate_independent_runtime(record, *, executable, engine_config_sha256):
    required = {
        "schema_version", "engine_family", "engine_name", "engine_revision",
        "license_id", "python_executable", "python_prefix", "python_version",
        "python_implementation", "platform_system", "platform_release",
        "platform_machine", "jax_enable_x64", "jax_platforms", "jax_devices",
        "package_records", "requirements_sha256", "engine_provenance_sha256",
        "engine_config_sha256", "environment_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RuntimeError("independent RT runtime provenance fields must be exact")
    expected_versions = {
        "differt": "0.10.0",
        "differt-core": "0.10.0",
        "jax": "0.11.0",
        "jaxlib": "0.11.0",
        "numpy": "2.5.2",
        "warp-lang": "1.16.0",
    }
    if (
        record["schema_version"]
        != "csi-pairs-v6-differt-runtime-provenance-v1"
        or record["engine_family"] != "differt"
        or record["engine_name"] != "DiffeRT"
        or record["engine_revision"]
        != "differt@673cc58ef61906b8ab0869dd206b3d032dbc01b2"
        or record["license_id"] != "MIT"
        or record["python_version"] != "3.12.13"
        or record["python_implementation"] != "CPython"
        or record["platform_system"] != "Linux"
        or record["platform_machine"] != "x86_64"
        or record["jax_enable_x64"] is not True
        or record["jax_platforms"] != "cpu"
        or record["requirements_sha256"]
        != "97afdca61c657fa23f1eeae3f64bfb37b54a048411c9b25edd230293bdecc573"
        or record["engine_provenance_sha256"]
        != "317ab9e495e6dee722a37bc471db77e2eedc9b3c6d699732f0d6c008bba0b2ed"
        or record["engine_config_sha256"] != engine_config_sha256
        or os.path.realpath(record["python_executable"])
        != os.path.realpath(executable)
    ):
        raise RuntimeError("independent RT runtime differs from the frozen profile")
    if (
        not isinstance(record["jax_devices"], list)
        or not record["jax_devices"]
        or any(
            not isinstance(value, str) or not value.startswith("cpu:")
            for value in record["jax_devices"]
        )
    ):
        raise RuntimeError("independent RT runtime is not CPU isolated")
    packages = record["package_records"]
    if not isinstance(packages, dict) or set(packages) != set(expected_versions):
        raise RuntimeError("independent RT package inventory is invalid")
    for name, version in expected_versions.items():
        value = packages[name]
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "record_sha256"}
            or value["version"] != version
            or not _lower_sha256(value["record_sha256"])
        ):
            raise RuntimeError(f"independent RT package provenance is invalid: {name}")
    for key in (
        "requirements_sha256",
        "engine_provenance_sha256",
        "engine_config_sha256",
        "environment_sha256",
    ):
        if not _lower_sha256(record[key]):
            raise RuntimeError(f"independent RT runtime {key} is invalid")
    without_hash = {
        key: value for key, value in record.items() if key != "environment_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(
            without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if record["environment_sha256"] != digest:
        raise RuntimeError("independent RT runtime environment digest mismatch")
    return record


def _verify_adapter_source(manifest):
    if _execution_mode(manifest) not in {
        "authenticated_sionna_adapter",
        "authenticated_independent_rt_adapter",
    }:
        raise ValueError("precomputed external-validity archives have no adapter source")
    path = Path(manifest["adapter_source_path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if path.is_symlink():
        raise RuntimeError("external-validity adapter source must be a regular file")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != manifest["adapter_source_sha256"]:
        raise RuntimeError("external-validity adapter source is missing or hash-mismatched")
    return path


def _resolve_manifest_file(value, digest, base_dir, label):
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    if path.is_symlink():
        raise RuntimeError(f"{label} must be a regular file")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"{label} is missing or hash-mismatched")
    return path


def _copy_regular_file(source, destination, label):
    source_path = Path(source)
    target = Path(destination)
    if source_path.is_symlink() or not source_path.is_file():
        raise RuntimeError(f"{label} must be a regular file")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {label}: {target}")
    shutil.copyfile(source_path, target)
    return target


def _stage_independent_source_assets(payload, source_assets, output_dir):
    entries = payload["worlds"]
    if len(entries) != len(source_assets):
        raise RuntimeError("independent RT source-asset count changed before staging")
    root = Path(output_dir)
    asset_root = root / "source_assets"
    asset_root.mkdir()
    for index, (entry, source) in enumerate(zip(entries, source_assets)):
        target = _copy_regular_file(
            source,
            asset_root / f"{index:04d}.bin",
            "independent RT source asset",
        )
        entry["source_asset_path"] = target.relative_to(root).as_posix()
    return payload


def _verify_archive_inputs(manifest, manifest_dir, dataset):
    if _execution_mode(manifest) != "authenticated_precomputed_rt_archive":
        raise ValueError("external-validity manifest is not a precomputed RT archive")
    csi_path = _resolve_manifest_file(
        manifest["external_csi_path"],
        manifest["external_csi_sha256"],
        manifest_dir,
        "external CSI archive",
    )
    context_path = _resolve_manifest_file(
        manifest["rt_scene_manifest_path"],
        manifest["rt_scene_manifest_sha256"],
        manifest_dir,
        "independent RT scene manifest",
    )
    config_path = _resolve_manifest_file(
        manifest["engine_config_path"],
        manifest["engine_config_sha256"],
        manifest_dir,
        "independent RT engine configuration",
    )
    source_assets = []
    context = _load_rt_scene_manifest(
        context_path,
        dataset,
        manifest,
        resolved_source_assets=source_assets,
    )
    if context["configuration_sha256"] != manifest["engine_config_sha256"]:
        raise ValueError(
            "independent RT scene manifest configuration differs from the archive"
        )
    _load_external_csi(csi_path, dataset)
    return csi_path, context_path, config_path, context, tuple(source_assets)


def _verify_independent_adapter_inputs(manifest, manifest_dir, dataset):
    if _execution_mode(manifest) != "authenticated_independent_rt_adapter":
        raise ValueError("external-validity manifest is not an independent adapter")
    context_path = _resolve_manifest_file(
        manifest["rt_scene_manifest_path"],
        manifest["rt_scene_manifest_sha256"],
        manifest_dir,
        "independent RT scene manifest",
    )
    config_path = _resolve_manifest_file(
        manifest["engine_config_path"],
        manifest["engine_config_sha256"],
        manifest_dir,
        "independent RT engine configuration",
    )
    from .external_adapters.differt_external_validity import load_engine_config

    config = load_engine_config(config_path)
    if (
        config["engine_family"] != _manifest_engine_family(manifest)
        or config["engine_revision"] != manifest["source_revision"]
        or config["license_id"] != manifest["license_id"]
    ):
        raise ValueError(
            "independent RT engine configuration differs from the adapter manifest"
        )
    source_assets = []
    context = _load_rt_scene_manifest(
        context_path,
        dataset,
        manifest,
        resolved_source_assets=source_assets,
    )
    return context_path, config_path, context, tuple(source_assets)


def _bind_rt_scene_manifest(command, dataset, output_dir):
    if command.count("--scene-manifest") != 1:
        raise RuntimeError("G8 command must bind exactly one RT scene manifest")
    index = command.index("--scene-manifest")
    if index + 1 >= len(command):
        raise RuntimeError("G8 command omits the RT scene manifest path")
    source_candidate = Path(command[index + 1])
    if source_candidate.is_symlink():
        raise RuntimeError("G8 RT scene manifest must be a regular file")
    source = source_candidate.resolve()
    if not source.is_file():
        raise RuntimeError("G8 RT scene manifest must be a regular file")
    from .external_adapters.sionna_external_validity import load_scene_manifest

    payload = load_scene_manifest(source, dataset)
    bound = Path(output_dir) / "rt_scene_manifest.json"
    write_json(bound, payload)
    return payload, bound


def _load_rt_scene_manifest(
    path, dataset, adapter_manifest=None, resolved_source_assets=None
):
    payload = read_strict_json(path)
    if adapter_manifest is not None:
        if _execution_mode(adapter_manifest) not in {
            "authenticated_independent_rt_adapter",
            "authenticated_precomputed_rt_archive",
        }:
            raise ValueError(
                "independent RT scene manifest requires an independent-engine manifest"
            )
        if payload.get("schema_version") != INDEPENDENT_RT_SCENE_SCHEMA:
            raise ValueError(
                "independent RT evidence requires the independent RT scene schema"
            )
        return _validate_independent_rt_scene_manifest(
            payload,
            dataset,
            adapter_manifest,
            Path(path).parent,
            resolved_source_assets,
        )
    if payload.get("schema_version") == INDEPENDENT_RT_SCENE_SCHEMA:
        raise ValueError(
            "independent RT scene manifest requires an independent-engine manifest"
        )
    from .external_adapters.sionna_external_validity import load_scene_manifest

    return load_scene_manifest(path, dataset)


def _resolve_independent_source_asset(value, digest, manifest_dir):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("independent RT source asset path must be nonempty")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("independent RT source asset path must be relative")
    root = Path(manifest_dir).resolve()
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError("independent RT source asset must be a regular file")
    path = candidate.resolve()
    if root != path.parent and root not in path.parents:
        raise RuntimeError("independent RT source asset escapes the scene manifest")
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(
            "independent RT source asset is missing or hash-mismatched"
        )
    return path


def _validate_independent_rt_scene_manifest(
    payload,
    dataset,
    adapter_manifest,
    manifest_dir,
    resolved_source_assets=None,
):
    required = {
        "schema_version",
        "dataset_sha256",
        "engine_family",
        "engine_name",
        "engine_revision",
        "configuration_sha256",
        "license_id",
        "worlds",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema_version") != INDEPENDENT_RT_SCENE_SCHEMA
    ):
        raise ValueError("independent RT scene manifest fields or schema are invalid")
    if payload["dataset_sha256"] != sha256_file(dataset.source_path):
        raise ValueError("independent RT scene manifest dataset hash mismatch")
    for key in ("engine_family", "engine_name", "engine_revision", "license_id"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"independent RT scene manifest {key} must be nonempty")
    if not _lower_sha256(payload["configuration_sha256"]):
        raise ValueError("independent RT configuration hash is invalid")
    if (
        payload["engine_family"].strip().lower()
        != _manifest_engine_family(adapter_manifest)
        or payload["engine_revision"] != adapter_manifest["source_revision"]
        or payload["license_id"] != adapter_manifest["license_id"]
    ):
        raise ValueError(
            "independent RT scene provenance differs from the adapter manifest"
        )
    if payload["configuration_sha256"] != adapter_manifest.get(
        "engine_config_sha256"
    ):
        raise ValueError(
            "independent RT scene configuration differs from the adapter manifest"
        )
    expected_worlds = {
        (str(dataset.scene_ids[int(scene)]), int(world))
        for scene in dataset.indices_for_role("external_validation")
        for world in range(dataset.world_count)
    }
    entry_fields = {
        "scene_id",
        "world",
        "canonical_map_sha256",
        "source_asset_id",
        "source_asset_path",
        "source_asset_sha256",
    }
    entries = payload["worlds"]
    if not isinstance(entries, list) or any(
        not isinstance(row, dict) or set(row) != entry_fields for row in entries
    ):
        raise ValueError("independent RT world entries have invalid fields")
    observed = {(str(row["scene_id"]), int(row["world"])) for row in entries}
    if len(entries) != len(expected_worlds) or observed != expected_worlds:
        raise ValueError(
            "independent RT scene manifest does not cover every external-validation sibling world"
        )
    source_ids = []
    source_paths = []
    source_digests = []
    scene_lookup = {
        str(dataset.scene_ids[int(scene)]): int(scene)
        for scene in dataset.indices_for_role("external_validation")
    }
    for row in entries:
        scene = scene_lookup[str(row["scene_id"])]
        world = int(row["world"])
        if row["canonical_map_sha256"] != str(
            dataset.canonical_map_sha256[scene, world]
        ):
            raise ValueError("independent RT world is not bound to its canonical map")
        if not isinstance(row["source_asset_id"], str) or not row["source_asset_id"].strip():
            raise ValueError("independent RT source asset identity must be nonempty")
        if not _lower_sha256(row["source_asset_sha256"]):
            raise ValueError("independent RT source asset hash is invalid")
        source_path = _resolve_independent_source_asset(
            row["source_asset_path"],
            row["source_asset_sha256"],
            manifest_dir,
        )
        source_ids.append(row["source_asset_id"])
        source_paths.append(source_path)
        source_digests.append(row["source_asset_sha256"])
    if (
        len(source_ids) != len(set(source_ids))
        or len(source_paths) != len(set(source_paths))
        or len(source_digests) != len(set(source_digests))
    ):
        raise ValueError("independent RT sibling worlds must use distinct source assets")
    if resolved_source_assets is not None:
        resolved_source_assets.extend(source_paths)
    return payload


def _expected_external_registry(config, dataset, output_root, rt_scene_manifest):
    from .formal_evidence import require_manifested_formal_qualification
    from .formal_factorial import _canonical_bank_digest
    from .formal_routing import fit_route_normalization, route_dataset
    from .formal_teacher import load_teacher_bundle

    qualification = require_manifested_formal_qualification(
        read_strict_json(Path(output_root) / "qualification" / "gate.json"),
        config,
        dataset,
        allow_nonscientific_fixture=True,
    )
    teacher = load_teacher_bundle(qualification["teacher_checkpoint"], config)
    scenes = dataset.indices_for_role("external_validation")
    normalization = fit_route_normalization(dataset, teacher)
    routed = route_dataset(
        dataset, teacher, config, scenes, normalization=normalization
    )
    scene_entries = {
        (str(row["scene_id"]), int(row["world"])): row
        for row in rt_scene_manifest["worlds"]
    }
    registry = {}
    for external_scene_index, scene_value in enumerate(scenes):
        scene = int(scene_value)
        canonical_bank = _canonical_bank_digest(dataset, scene)
        canonical_cluster = dataset.canonical_base_map_digest(scene)
        for edge in dataset.directed_edges(scene):
            for position in range(dataset.position_count):
                route_code = int(
                    routed.alignment_route[
                        (scene, edge.source_world, edge.target_world, position)
                    ]
                )
                if route_code not in {0, 2}:
                    continue
                unit_id = (
                    f"{dataset.bank_ids[scene]}:{edge.source_world}:"
                    f"{edge.target_world}:{dataset.position_ids[scene, position]}"
                )
                source = dataset.csi_clean[scene, edge.source_world, position]
                target = dataset.csi_clean[scene, edge.target_world, position]
                source_entry = scene_entries[
                    (str(dataset.scene_ids[scene]), edge.source_world)
                ]
                target_entry = scene_entries[
                    (str(dataset.scene_ids[scene]), edge.target_world)
                ]
                context_sha256 = hashlib.sha256(
                    json.dumps(
                        {
                            "scene_id": str(dataset.scene_ids[scene]),
                            "source_world": edge.source_world,
                            "target_world": edge.target_world,
                            "position_id": str(
                                dataset.position_ids[scene, position]
                            ),
                            "source_scene_sha256": _context_source_sha256(
                                source_entry
                            ),
                            "target_scene_sha256": _context_source_sha256(
                                target_entry
                            ),
                            "engine": _context_engine_identity(
                                rt_scene_manifest
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if unit_id in registry:
                    raise RuntimeError("external-validity outer registry duplicates a unit")
                registry[unit_id] = {
                    "unit_id": unit_id,
                    "scene_index": scene,
                    "external_scene_index": external_scene_index,
                    "source_world": edge.source_world,
                    "target_world": edge.target_world,
                    "position": position,
                    "bank_id": str(dataset.bank_ids[scene]),
                    "route": "active" if route_code == 2 else "null",
                    "primary_direction": _power_direction(source, target),
                    "primary_effect": _relative_effect(source, target),
                    "context_sha256": context_sha256,
                    "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
                    "canonical_base_map_digest": str(canonical_cluster),
                    "canonical_bank_digest": str(canonical_bank),
                    "canonical_unit_id": ":".join(
                        (
                            str(canonical_bank),
                            "".join(
                                str(int(value))
                                for value in dataset.world_bits[edge.source_world]
                            ),
                            "".join(
                                str(int(value))
                                for value in dataset.world_bits[edge.target_world]
                            ),
                            str(dataset.position_ids[scene, position]),
                        )
                    ),
                }
    if not registry:
        raise RuntimeError("external-validity outer registry is empty")
    return registry


def _context_source_sha256(entry):
    value = entry.get("scene_xml_sha256", entry.get("source_asset_sha256"))
    if not _lower_sha256(value):
        raise RuntimeError("external-validity context has no authenticated source asset")
    return value


def _context_engine_identity(manifest):
    if manifest.get("schema_version") == INDEPENDENT_RT_SCENE_SCHEMA:
        return f"{manifest['engine_family']}@{manifest['engine_revision']}"
    return str(manifest["sionna_revision"])


def _relative_effect(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(target - source) / max(np.linalg.norm(source), 1e-12))


def _power_direction(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    half = source.shape[-1] // 2
    source_power = np.mean(source[:half] ** 2 + source[half:] ** 2)
    target_power = np.mean(target[:half] ** 2 + target[half:] ** 2)
    return 1 if target_power >= source_power else -1


def _load_external_csi(path, dataset):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("external-validity raw CSI must be a regular NPZ file")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"scene_ids", "position_ids", "external_csi"}:
            raise RuntimeError("external-validity raw CSI arrays must be exact")
        scene_ids = np.asarray(archive["scene_ids"])
        position_ids = np.asarray(archive["position_ids"])
        values = np.asarray(archive["external_csi"])
    scenes = np.asarray(dataset.indices_for_role("external_validation"), dtype=np.int64)
    expected_shape = (
        len(scenes),
        dataset.world_count,
        dataset.position_count,
        dataset.channel_count,
    )
    if values.shape != expected_shape or values.dtype.kind not in {"f", "i", "u"}:
        raise RuntimeError("external-validity raw CSI shape or dtype is invalid")
    if scene_ids.dtype.kind not in {"U", "S"} or not np.array_equal(
        scene_ids.astype(str), np.asarray(dataset.scene_ids[scenes], dtype=str)
    ):
        raise RuntimeError("external-validity raw CSI scene order is invalid")
    if position_ids.dtype.kind not in {"U", "S"} or not np.array_equal(
        position_ids.astype(str), np.asarray(dataset.position_ids[scenes], dtype=str)
    ):
        raise RuntimeError("external-validity raw CSI position order is invalid")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("external-validity raw CSI contains nonfinite values")
    if np.array_equal(values, np.asarray(dataset.csi_clean[scenes], dtype=np.float64)):
        raise RuntimeError("external-validity raw CSI is an exact copy of the primary-engine CSI")
    return values


def _rows_from_external_csi(external_csi, expected_registry):
    rows = []
    for unit_id in sorted(expected_registry):
        expected = expected_registry[unit_id]
        source = external_csi[
            expected["external_scene_index"],
            expected["source_world"],
            expected["position"],
        ]
        target = external_csi[
            expected["external_scene_index"],
            expected["target_world"],
            expected["position"],
        ]
        rows.append(
            {
                **expected,
                "external_direction": _power_direction(source, target),
                "external_effect": _relative_effect(source, target),
            }
        )
    if not rows:
        raise RuntimeError("external-validity outer registry is empty")
    _deduplicate_canonical_rows(rows)
    return rows


def _cluster_direction_interval(rows, resamples):
    rows = _deduplicate_canonical_rows(rows)
    clusters, values = _canonical_cluster_values(
        rows,
        lambda row: float(
            int(row["primary_direction"]) == int(row["external_direction"])
        ),
    )
    if len(clusters) < 2:
        raise RuntimeError("external-validity direction inference requires two base-map clusters")
    rng = np.random.default_rng(180000)
    samples = np.asarray(
        [np.mean(values[rng.integers(0, len(values), size=len(values))]) for _ in range(int(resamples))]
    )
    return {
        "base_map_cluster_count": len(clusters),
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _paired_bank_equivalence(rows, margin, resamples):
    rows = _deduplicate_canonical_rows(rows)
    clusters, differences = _canonical_cluster_values(
        rows,
        lambda row: float(row["external_effect"]) - float(row["primary_effect"]),
    )
    if len(clusters) < 2:
        raise RuntimeError("external-validity equivalence requires two base-map clusters")
    rng = np.random.default_rng(180001)
    samples = np.asarray(
        [
            np.mean(
                differences[
                    rng.integers(0, len(clusters), size=len(clusters))
                ]
            )
            for _ in range(resamples)
        ]
    )
    low, high = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    return {
        "base_map_cluster_count": len(clusters),
        "mean_difference": float(np.mean(differences)),
        "ci95_low": low,
        "ci95_high": high,
        "margin": margin,
        "passed": bool(low >= -margin and high <= margin),
    }


def _deduplicate_canonical_rows(rows):
    unique = {}
    for row in rows:
        unit = str(row["canonical_unit_id"])
        signature = (
            str(row["route"]),
            int(row["primary_direction"]),
            int(row["external_direction"]),
            float(row["primary_effect"]),
            float(row["external_effect"]),
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
        )
        if unit in unique:
            prior = unique[unit]
            prior_signature = prior[1]
            exact = signature[:3] == prior_signature[:3] and signature[5:] == prior_signature[5:]
            numeric = np.allclose(signature[3:5], prior_signature[3:5], rtol=1e-10, atol=1e-12)
            if not (exact and numeric):
                raise RuntimeError(
                    "copied canonical external-validity unit has inconsistent values"
                )
            continue
        unique[unit] = (row, signature)
    return [unique[key][0] for key in sorted(unique)]


def _canonical_cluster_values(rows, value):
    bank_values = {}
    for row in rows:
        key = (
            str(row["canonical_base_map_digest"]),
            str(row["canonical_bank_digest"]),
        )
        bank_values.setdefault(key, []).append(float(value(row)))
    cluster_values = {}
    for (cluster, _), values in bank_values.items():
        cluster_values.setdefault(cluster, []).append(float(np.mean(values)))
    clusters = sorted(cluster_values)
    return clusters, np.asarray(
        [np.mean(cluster_values[cluster]) for cluster in clusters],
        dtype=np.float64,
    )


def _lower_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
