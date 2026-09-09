from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from .formal_dataset import _canonical_foundation_sha256
from .formal_evidence import EVIDENCE_AUTH_KEYS, bind_rows, evidence_context
from .formal_io import (
    artifact_manifest,
    parse_strict_json,
    read_strict_json,
    sha256_file,
    write_csv,
    write_json,
)


SCHEMA = "csi-pairs-v6-data-verification-gate-v1"
BLOCKING_ROLES = ("source_encoder_train", "source_method_selection")
REGENERATED_FIELDS = {
    "maps",
    "csi_clean",
    "csi_repeat",
    "free_space",
    "phase_reference_ids",
    "phase_reference_values",
    "phase_reference_source_sha256",
    "engine_config_json",
    "noop_maps",
    "path_ids",
    "path_power",
    "path_surface_ids",
    "noop_path_ids",
    "noop_path_power",
    "noop_path_surface_ids",
}


def run_data_verification(config, dataset, manifest_path, output_root):
    manifest_file = _require_regular_file(manifest_path, "data verifier manifest")
    manifest = read_strict_json(manifest_file)
    _validate_manifest(manifest, dataset)
    verifier_source = _resolve_verifier_source(manifest, manifest_file.parent)
    output_dir = Path(output_root) / "data_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    bound_manifest = output_dir / "verifier_manifest.json"
    write_json(
        bound_manifest,
        {**manifest, "verifier_source_path": str(verifier_source)},
    )
    command = [
        value.format(
            dataset=str(Path(dataset.source_path).resolve()),
            output=str(output_dir.resolve()),
            python=sys.executable,
            verifier_source=str(verifier_source),
        )
        for value in manifest["command"]
    ]
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=project_root,
        env=_verifier_environment(project_root),
    )
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"independent data verifier failed with code {completed.returncode}")
    _require_file_sha256(
        verifier_source,
        manifest["verifier_source_sha256"],
        "data verifier source",
    )
    regenerated_path = output_dir / "regenerated.npz"
    if regenerated_path.is_symlink() or not regenerated_path.is_file():
        raise RuntimeError("independent data verifier did not emit regenerated.npz")
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    with np.load(regenerated_path, allow_pickle=False) as archive:
        if set(archive.files) != REGENERATED_FIELDS:
            raise RuntimeError("regenerated data fields must be exact")
        engine = parse_strict_json(str(np.asarray(archive["engine_config_json"]).item()))
        global_engine_match = engine == dataset.engine_config
        rows = [
            _scene_comparison(
                dataset,
                archive,
                scene,
                float(manifest["rtol"]),
                float(manifest["atol"]),
            )
            for scene in range(dataset.scene_count)
        ]
    blocking = [row for row in rows if row["role"] in BLOCKING_ROLES]
    blocking_passed = bool(global_engine_match and blocking and all(row["passed"] for row in blocking))
    nonblocking_failures = [row["scene_id"] for row in rows if row["role"] not in BLOCKING_ROLES and not row["passed"]]
    role_status = {}
    for role in sorted(set(str(row["role"]) for row in rows)):
        selected = [row for row in rows if row["role"] == role]
        role_status[role] = (
            "PASS" if global_engine_match and selected and all(row["passed"] for row in selected)
            else "FAIL"
        )
    write_csv(output_dir / "per_scene.csv", bind_rows(rows, evidence))
    gate = {
        "schema_version": SCHEMA,
        "status": "PASS" if blocking_passed else "FAIL",
        "passed": blocking_passed,
        "blocking_passed": blocking_passed,
        **evidence,
        "blocking_roles": list(BLOCKING_ROLES),
        "target_and_other_roles_are_nonblocking": True,
        "engine_config_match": global_engine_match,
        "engine_source_revision": manifest["engine_source_revision"],
        "engine_license_id": manifest["engine_license_id"],
        "asset_license_ids": manifest["asset_license_ids"],
        "verifier_manifest_path": str(bound_manifest.resolve()),
        "verifier_manifest_sha256": sha256_file(bound_manifest),
        "verifier_source_path": str(verifier_source),
        "verifier_source_sha256": manifest["verifier_source_sha256"],
        "rtol": float(manifest["rtol"]),
        "atol": float(manifest["atol"]),
        "nonblocking_scene_failures": nonblocking_failures,
        "role_status": role_status,
        "verified_properties": [
            "canonical_rendering",
            "canonical_foundation_content_identity",
            "complete_cross_generation",
            "common_free_positions",
            "independent_repeat_noise_regeneration",
            "observation_seed_to_residual_binding",
            "world_independent_complex_phase_reference",
            "phase_reference_source_sha256",
            "path_and_noop_retrace",
            "engine_config_and_license_binding",
        ],
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


def require_data_verification(gate, config, dataset, gate_path=None):
    if not isinstance(gate, dict) or gate.get("schema_version") != SCHEMA:
        raise RuntimeError("qualification requires a V6 independent data-verification gate")
    expected = evidence_context(config, dataset, str(gate.get("scientific_use", "")))
    for key in EVIDENCE_AUTH_KEYS:
        if gate.get(key) != expected[key]:
            raise RuntimeError(f"data-verification gate {key} mismatch")
    if gate.get("blocking_roles") != list(BLOCKING_ROLES):
        raise RuntimeError("data-verification gate uses the wrong blocking roles")
    if gate.get("target_and_other_roles_are_nonblocking") is not True:
        raise RuntimeError("data-verification gate lets target data control qualification")
    if gate.get("blocking_passed") is not True or gate.get("passed") is not True:
        raise RuntimeError("independent data-verification blocking partition failed")
    _require_verifier_binding(gate, config, dataset, gate_path=gate_path)
    require_verified_roles(
        gate,
        config,
        dataset,
        BLOCKING_ROLES,
        gate_path=gate_path,
        _binding_checked=True,
    )
    return gate


def require_verified_roles(
    gate: object,
    config: dict,
    dataset,
    roles: Iterable[str],
    *,
    gate_path: str | Path | None = None,
    _binding_checked: bool = False,
) -> dict:
    """Require regeneration PASS for every role a downstream stage will read."""
    if not isinstance(gate, dict) or gate.get("schema_version") != SCHEMA:
        raise RuntimeError("stage requires a V6 independent data-verification gate")
    expected = evidence_context(config, dataset, str(gate.get("scientific_use", "")))
    for key in EVIDENCE_AUTH_KEYS:
        if gate.get(key) != expected[key]:
            raise RuntimeError(f"data-verification gate {key} mismatch")
    if not _binding_checked:
        _require_verifier_binding(gate, config, dataset, gate_path=gate_path)
    statuses = gate.get("role_status")
    if not isinstance(statuses, dict):
        raise RuntimeError("data-verification gate is missing per-role status")
    missing_or_failed = [role for role in roles if statuses.get(str(role)) != "PASS"]
    if missing_or_failed:
        raise RuntimeError(
            "data regeneration failed or was not assessed for roles: "
            + ", ".join(sorted(missing_or_failed))
        )
    return gate


def require_verified_roles_from_root(
    output_root: str | Path,
    config: dict,
    dataset,
    roles: Iterable[str],
) -> dict:
    path = Path(output_root) / "data_verification" / "gate.json"
    if not path.is_file():
        raise RuntimeError(f"missing data-verification gate: {path}")
    return require_verified_roles(
        read_strict_json(path),
        config,
        dataset,
        roles,
        gate_path=path,
    )


def _validate_manifest(manifest, dataset):
    required = {
        "schema_version",
        "command",
        "verifier_source_path",
        "verifier_source_sha256",
        "engine_source_revision",
        "engine_license_id",
        "asset_license_ids",
        "rtol",
        "atol",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("data verifier manifest fields must be exact")
    if manifest["schema_version"] != "csi-pairs-v6-data-verifier-v1":
        raise ValueError("data verifier manifest schema mismatch")
    if not isinstance(manifest["command"], list) or not manifest["command"]:
        raise ValueError("data verifier command must be a nonempty argv list")
    if not _command_executes_verifier_source(manifest["command"]):
        raise ValueError(
            "data verifier command must directly execute the authenticated source "
            "and bind dataset/output exactly once"
        )
    source_path = manifest["verifier_source_path"]
    if (
        not isinstance(source_path, str)
        or not source_path.strip()
        or Path(source_path).suffix != ".py"
        or not _lower_sha256(manifest["verifier_source_sha256"])
    ):
        raise ValueError("data verifier source path/hash is invalid")
    if manifest["engine_source_revision"] != dataset.metadata["engine"]["source_revision"]:
        raise ValueError("data verifier engine revision does not match the dataset")
    if manifest["engine_license_id"] != dataset.metadata["engine"]["license_id"]:
        raise ValueError("data verifier engine license does not match the dataset")
    if sorted(manifest["asset_license_ids"]) != sorted(dataset.metadata["assets"]["license_ids"]):
        raise ValueError("data verifier asset licenses do not match the dataset")
    for name in ("rtol", "atol"):
        if not np.isfinite(manifest[name]) or float(manifest[name]) < 0:
            raise ValueError(f"data verifier {name} must be finite and nonnegative")


def _command_executes_verifier_source(command) -> bool:
    if not isinstance(command, list) or any(
        not isinstance(value, str) or not value for value in command
    ):
        return False
    placeholders = ("{python}", "{verifier_source}", "{dataset}", "{output}")
    if command[:2] != ["{python}", "{verifier_source}"]:
        return False
    if any(command.count(value) != 1 for value in placeholders):
        return False
    if any(
        ("{" in value or "}" in value) and value not in placeholders
        for value in command
    ):
        return False
    return bool(
        _command_binds_option(command, "--dataset", "{dataset}")
        and _command_binds_option(command, "--output", "{output}")
    )


def _command_binds_option(command, option, placeholder) -> bool:
    indices = [index for index, value in enumerate(command) if value == option]
    return bool(
        len(indices) == 1
        and indices[0] + 1 < len(command)
        and command[indices[0] + 1] == placeholder
    )


def _resolve_verifier_source(manifest, manifest_root: Path) -> Path:
    source = Path(manifest["verifier_source_path"])
    candidate = source if source.is_absolute() else Path(manifest_root) / source
    resolved = _require_regular_file(candidate, "data verifier source")
    _require_file_sha256(
        resolved,
        manifest["verifier_source_sha256"],
        "data verifier source",
    )
    return resolved


def _require_regular_file(path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} must be a regular file: {candidate}")
    return resolved


def _require_file_sha256(path, expected, label: str) -> None:
    if not _lower_sha256(expected) or sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch")


def _lower_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_verifier_binding(gate, config, dataset, *, gate_path=None) -> None:
    required = {
        "verifier_manifest_path",
        "verifier_manifest_sha256",
        "verifier_source_path",
        "verifier_source_sha256",
        "engine_source_revision",
        "engine_license_id",
        "asset_license_ids",
        "rtol",
        "atol",
    }
    missing = required.difference(gate)
    if missing:
        raise RuntimeError(
            "data-verification gate is missing authenticated verifier fields: "
            f"{sorted(missing)}"
        )
    manifest_path = _require_regular_file(
        gate["verifier_manifest_path"], "bound data verifier manifest"
    )
    _require_file_sha256(
        manifest_path,
        gate["verifier_manifest_sha256"],
        "bound data verifier manifest",
    )
    manifest = read_strict_json(manifest_path)
    _validate_manifest(manifest, dataset)
    source = _resolve_verifier_source(manifest, manifest_path.parent)
    gate_source = _require_regular_file(
        gate["verifier_source_path"], "data verifier source"
    )
    if source != gate_source:
        raise RuntimeError("data-verification gate verifier source path mismatch")
    _require_file_sha256(
        gate_source,
        gate["verifier_source_sha256"],
        "data verifier source",
    )
    if gate["verifier_source_sha256"] != manifest["verifier_source_sha256"]:
        raise RuntimeError("data-verification gate verifier source hash mismatch")
    for key in (
        "engine_source_revision",
        "engine_license_id",
        "asset_license_ids",
        "rtol",
        "atol",
    ):
        if gate[key] != manifest[key]:
            raise RuntimeError(f"data-verification gate {key} differs from verifier manifest")
    if gate_path is not None:
        from .formal_evidence import require_stage_manifested_gate

        resolved_gate = _require_regular_file(gate_path, "data-verification gate")
        require_stage_manifested_gate(
            resolved_gate,
            gate,
            config,
            dataset,
            schema_version=SCHEMA,
        )
        expected_manifest = (resolved_gate.parent / "verifier_manifest.json").resolve()
        if manifest_path != expected_manifest:
            raise RuntimeError(
                "data-verification gate does not bind its colocated verifier manifest"
            )
        _require_stage_file(resolved_gate.parent, manifest_path)


def _require_stage_file(stage_dir: Path, path: Path) -> None:
    manifest_path = stage_dir / "manifest.json"
    manifest = read_strict_json(manifest_path)
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    relative = str(path.relative_to(stage_dir))
    matches = [
        row
        for row in entries
        if isinstance(row, dict) and row.get("path") == relative
    ] if isinstance(entries, list) else []
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(path):
        raise RuntimeError(
            "bound data verifier manifest is absent from or mismatched with its stage manifest"
        )


def _verifier_environment(project_root: Path) -> dict[str, str]:
    root = str(project_root.resolve())
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": root if not existing else root + os.pathsep + existing,
    }


def _scene_comparison(dataset, archive, scene, rtol, atol):
    zero_world = int(np.flatnonzero(np.all(dataset.world_bits == 0, axis=1))[0])
    representation = dataset.metadata["representation"]
    regenerated_foundation_digest = _canonical_foundation_sha256(
        archive["maps"][scene, zero_world],
        dataset.map_channel_names,
        float(representation["map_resolution_m"]),
        representation["map_origin_xy_m"],
    )
    expected_foundation_digest = dataset.canonical_base_map_digest(scene)
    regenerated_noise_digest = dataset.observation_noise_binding_digest(
        scene, archive["csi_repeat"][scene]
    )
    expected_noise_digest = dataset.observation_noise_binding_digest(scene)
    checks = {
        "maps": _equal(dataset.maps[scene], archive["maps"][scene], rtol, atol),
        "canonical_foundation_identity": regenerated_foundation_digest
        == expected_foundation_digest,
        "csi_clean": _equal(dataset.csi_clean[scene], archive["csi_clean"][scene], rtol, atol),
        "csi_repeat": _equal(dataset.csi_repeat[scene], archive["csi_repeat"][scene], rtol, atol),
        "observation_noise_seed_binding": regenerated_noise_digest == expected_noise_digest,
        "free_space": np.array_equal(dataset.free_space[scene], archive["free_space"][scene]),
        "phase_reference_ids": np.array_equal(dataset.phase_reference_ids[scene], archive["phase_reference_ids"][scene]),
        "phase_reference_values": np.array_equal(
            dataset.phase_reference_values[scene],
            archive["phase_reference_values"][scene],
        ),
        "phase_reference_source_sha256": np.array_equal(
            dataset.phase_reference_source_sha256[scene],
            archive["phase_reference_source_sha256"][scene],
        ),
        "noop_maps": _equal(dataset.noop_maps[scene], archive["noop_maps"][scene], rtol, atol),
        "path_ids": np.array_equal(dataset.path_ids[scene], archive["path_ids"][scene]),
        "path_power": _equal(dataset.path_power[scene], archive["path_power"][scene], rtol, atol),
        "path_surface_ids": np.array_equal(dataset.path_surface_ids[scene], archive["path_surface_ids"][scene]),
        "noop_path_ids": np.array_equal(dataset.noop_path_ids[scene], archive["noop_path_ids"][scene]),
        "noop_path_power": _equal(dataset.noop_path_power[scene], archive["noop_path_power"][scene], rtol, atol),
        "noop_path_surface_ids": np.array_equal(dataset.noop_path_surface_ids[scene], archive["noop_path_surface_ids"][scene]),
    }
    return {
        "scene_id": str(dataset.scene_ids[scene]),
        "bank_id": str(dataset.bank_ids[scene]),
        "base_map_cluster_id": str(dataset.base_map_cluster_ids[scene]),
        "canonical_base_map_digest": expected_foundation_digest,
        "regenerated_canonical_base_map_digest": regenerated_foundation_digest,
        "observation_noise_binding_sha256": expected_noise_digest,
        "regenerated_observation_noise_binding_sha256": regenerated_noise_digest,
        "role": str(dataset.scene_roles[scene]),
        **{f"{name}_match": bool(value) for name, value in checks.items()},
        "passed": bool(all(checks.values())),
    }


def _equal(first, second, rtol, atol):
    left = np.asarray(first)
    right = np.asarray(second)
    return left.shape == right.shape and bool(np.allclose(left, right, rtol=rtol, atol=atol))
