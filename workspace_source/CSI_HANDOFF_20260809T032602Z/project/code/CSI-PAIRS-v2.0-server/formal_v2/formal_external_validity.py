from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .formal_evidence import bind_rows, evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_csv, write_json


def run_external_validity(config, dataset, manifest_path, output_root):
    from .formal_data_verification import require_verified_roles_from_root

    require_verified_roles_from_root(
        output_root, config, dataset, ("external_validation",)
    )
    manifest_file = Path(manifest_path).resolve()
    manifest = read_strict_json(manifest_file)
    _validate_manifest(manifest)
    adapter_source = _verify_adapter_source(manifest)
    output_dir = Path(output_root) / "external_validity"
    output_dir.mkdir(parents=True, exist_ok=True)
    bound_manifest = output_dir / "adapter_manifest.json"
    write_json(bound_manifest, {**manifest, "adapter_source_path": str(adapter_source)})
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
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
    expected_registry = _expected_external_registry(
        config, dataset, Path(output_root), rt_scene_manifest
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
    passed = bool(
        agreement["ci95_low"]
        >= float(config["external_validity"]["minimum_active_direction_agreement"])
        and equivalence["passed"]
    )
    write_csv(output_dir / "validated_paired_effects.csv", bind_rows(rows, evidence))
    gate = {
        "schema_version": "csi-pairs-v6-external-validity-gate-v4",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "gate": "G8",
        "evidence_type": manifest["evidence_type"],
        "source_revision": manifest["source_revision"],
        "license_id": manifest["license_id"],
        "adapter_source_path": str(adapter_source),
        "adapter_source_sha256": manifest["adapter_source_sha256"],
        "input_manifest_path": bound_manifest.name,
        "input_manifest_sha256": sha256_file(bound_manifest),
        "rt_scene_manifest_path": rt_scene_manifest_path.name,
        "rt_scene_manifest_sha256": sha256_file(rt_scene_manifest_path),
        "external_csi_path": result_path.name,
        "external_csi_sha256": sha256_file(result_path),
        "external_runtime_provenance_path": runtime_path.name,
        "external_runtime_provenance_sha256": sha256_file(runtime_path),
        "external_runtime_environment_sha256": runtime_record[
            "environment_sha256"
        ],
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
    required = {
        "schema_version", "evidence_type", "source_revision", "license_id", "command",
        "adapter_source_path", "adapter_source_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("external-validity manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-external-validity-adapter-v3":
        raise ValueError("external-validity manifest schema mismatch")
    if manifest["evidence_type"] != "independent_rt_engine":
        raise ValueError("external-validity evidence type is unsupported")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValueError("external-validity command must be nonempty argv")
    if not _command_executes_adapter_source(
        manifest["command"], manifest["adapter_source_path"]
    ):
        raise ValueError(
            "external-validity command must execute the authenticated adapter module"
        )
    for key in ("source_revision", "license_id"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"external-validity {key} must be nonempty")
    digest = manifest["adapter_source_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("external-validity adapter source hash must be lowercase SHA-256")


def _command_executes_adapter_source(command, adapter_source_path):
    if Path(adapter_source_path).suffix != ".py":
        return False
    return bool(
        len(command) >= 2
        and command[0]
        == "{project_root}/formal_v2/external_adapters/.runtime-sionna/venv/bin/python"
        and command[1] == "{adapter_source}"
    )


def _adapter_environment(project_root):
    root = str(Path(project_root).resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
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


def _verify_adapter_source(manifest):
    path = Path(manifest["adapter_source_path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if path.is_symlink():
        raise RuntimeError("external-validity adapter source must be a regular file")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != manifest["adapter_source_sha256"]:
        raise RuntimeError("external-validity adapter source is missing or hash-mismatched")
    return path


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
                            "source_scene_sha256": source_entry[
                                "scene_xml_sha256"
                            ],
                            "target_scene_sha256": target_entry[
                                "scene_xml_sha256"
                            ],
                            "engine": rt_scene_manifest["sionna_revision"],
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
