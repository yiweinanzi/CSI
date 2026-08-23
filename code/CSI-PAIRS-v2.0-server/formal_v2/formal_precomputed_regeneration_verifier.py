from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from formal_v2 import formal_evidence
from formal_v2.formal_io import read_strict_json, sha256_file


RECEIPT_SCHEMA = "csi-pairs-v6-precomputed-regeneration-receipt-v1"
RENDERER_RUNTIME_SCHEMA = "csi-pairs-sionna-renderer-runtime-receipt-v1"
EXPECTED_DISTRIBUTIONS = {
    "drjit": "1.2.0",
    "h5py": "3.15.1",
    "mitsuba": "3.7.1",
    "sionna": "2.0.1",
    "sionna-rt": "1.2.1",
}
REGISTERED_CANDIDATE_EVIDENCE = (
    Path(formal_evidence.__file__).resolve().parents[1]
    / "artifacts/m4_llvm22_candidate_v1/candidate_evidence.json"
)


def validate_receipt(
    receipt_path: str | Path,
    dataset_path: str | Path,
    *,
    require_registration: bool = True,
) -> tuple[dict, Path]:
    receipt_file = _regular_file(receipt_path, "precomputed regeneration receipt")
    receipt = read_strict_json(receipt_file)
    required = {
        "schema_version",
        "status",
        "fixture",
        "scientific_use",
        "dataset_sha256",
        "dataset_bytes",
        "config_sha256",
        "source_tree_sha256",
        "origin_gate_sha256",
        "origin_manifest_sha256",
        "origin_per_scene_sha256",
        "origin_verifier_source_sha256",
        "regenerated_path",
        "regenerated_sha256",
        "regenerated_bytes",
        "scene_count",
        "role_status",
        "rtol",
        "atol",
        "engine_source_revision",
        "engine_license_id",
        "asset_license_ids",
        "renderer_runtime",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise RuntimeError("precomputed regeneration receipt fields must be exact")
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["status"] != "PASS":
        raise RuntimeError("precomputed regeneration receipt is not PASS")
    if type(receipt["fixture"]) is not bool:
        raise RuntimeError("precomputed regeneration receipt fixture flag is invalid")
    expected_use = "FORBIDDEN" if receipt["fixture"] else "CANDIDATE_NOT_CLAIM"
    if receipt["scientific_use"] != expected_use:
        raise RuntimeError("precomputed regeneration receipt scientific-use boundary is invalid")
    for key in (
        "dataset_sha256",
        "config_sha256",
        "source_tree_sha256",
        "origin_gate_sha256",
        "origin_manifest_sha256",
        "origin_per_scene_sha256",
        "origin_verifier_source_sha256",
        "regenerated_sha256",
    ):
        if not _lower_sha256(receipt[key]):
            raise RuntimeError(f"precomputed regeneration receipt {key} is invalid")
    if receipt["fixture"]:
        if receipt["source_tree_sha256"] != formal_evidence._source_tree_sha256():
            raise RuntimeError(
                "precomputed regeneration receipt source tree differs from this checkout"
            )
    dataset = _regular_file(dataset_path, "candidate dataset")
    if dataset.stat().st_size != receipt["dataset_bytes"] or sha256_file(dataset) != receipt["dataset_sha256"]:
        raise RuntimeError("precomputed regeneration receipt dataset mismatch")
    if type(receipt["scene_count"]) is not int or receipt["scene_count"] <= 0:
        raise RuntimeError("precomputed regeneration receipt scene count is invalid")
    dataset_scene_count, dataset_roles = _dataset_scene_inventory(dataset)
    if receipt["scene_count"] != dataset_scene_count:
        raise RuntimeError("precomputed regeneration receipt scene count differs from dataset")
    roles = receipt["role_status"]
    if not isinstance(roles, dict) or not roles or any(
        not isinstance(role, str) or not role or status != "PASS"
        for role, status in roles.items()
    ):
        raise RuntimeError("precomputed regeneration receipt does not pass every recorded role")
    if set(roles) != dataset_roles:
        raise RuntimeError("precomputed regeneration receipt role inventory differs from dataset")
    if not receipt["fixture"] and require_registration:
        _require_registered_nonfixture_receipt(receipt_file, receipt)
    if float(receipt["rtol"]) != 0.0 or float(receipt["atol"]) != 0.0:
        raise RuntimeError("precomputed regeneration receipt must use zero tolerance")
    if (
        not isinstance(receipt["engine_source_revision"], str)
        or not receipt["engine_source_revision"]
        or not isinstance(receipt["engine_license_id"], str)
        or not receipt["engine_license_id"]
        or not isinstance(receipt["asset_license_ids"], list)
        or not receipt["asset_license_ids"]
        or any(not isinstance(value, str) or not value for value in receipt["asset_license_ids"])
    ):
        raise RuntimeError("precomputed regeneration engine or asset provenance is invalid")
    regenerated_relative = receipt["regenerated_path"]
    if (
        not isinstance(regenerated_relative, str)
        or not regenerated_relative
        or Path(regenerated_relative).is_absolute()
        or ".." in Path(regenerated_relative).parts
        or type(receipt["regenerated_bytes"]) is not int
        or receipt["regenerated_bytes"] <= 0
    ):
        raise RuntimeError("precomputed regeneration receipt path or size is invalid")
    regenerated = _regular_file(
        receipt_file.parent / regenerated_relative,
        "precomputed regenerated archive",
    )
    if (
        regenerated.stat().st_size != receipt["regenerated_bytes"]
        or sha256_file(regenerated) != receipt["regenerated_sha256"]
    ):
        raise RuntimeError("precomputed regenerated hash mismatch")
    _validate_renderer_runtime(receipt["renderer_runtime"])
    return receipt, regenerated


def _dataset_scene_inventory(dataset: Path) -> tuple[int, set[str]]:
    try:
        with np.load(dataset, allow_pickle=False) as archive:
            if "scene_ids" not in archive.files or "scene_roles" not in archive.files:
                raise RuntimeError("candidate dataset lacks scene identity arrays")
            scene_ids = np.asarray(archive["scene_ids"])
            scene_roles = np.asarray(archive["scene_roles"])
    except (OSError, ValueError) as error:
        raise RuntimeError("candidate dataset is not a readable non-pickle NPZ") from error
    if (
        scene_ids.ndim != 1
        or scene_roles.shape != scene_ids.shape
        or scene_ids.size < 1
        or len(set(str(value) for value in scene_ids)) != scene_ids.size
        or any(not str(value).strip() for value in scene_roles)
    ):
        raise RuntimeError("candidate dataset scene inventory is invalid")
    return int(scene_ids.size), set(str(value) for value in scene_roles)


def _require_registered_nonfixture_receipt(receipt_file: Path, receipt: dict) -> None:
    evidence_path = _regular_file(
        REGISTERED_CANDIDATE_EVIDENCE,
        "registered LLVM 22 candidate evidence",
    )
    evidence = read_strict_json(evidence_path)
    candidate = evidence.get("candidate") if isinstance(evidence, dict) else None
    live = (
        evidence.get("live_independent_regeneration")
        if isinstance(evidence, dict)
        else None
    )
    portable = evidence.get("portable_replay") if isinstance(evidence, dict) else None
    registered = bool(
        evidence.get("schema_version")
        == "csi-pairs-m4-llvm22-candidate-evidence-v1"
        and evidence.get("status") == "STATIC_REGISTRY_PASS"
        and evidence.get("scientific_use") == "CANDIDATE_NOT_CLAIM"
        and isinstance(candidate, dict)
        and candidate.get("dataset_sha256") == receipt["dataset_sha256"]
        and candidate.get("dataset_bytes") == receipt["dataset_bytes"]
        and candidate.get("fixture") is False
        and candidate.get("scene_banks") == receipt["scene_count"]
        and isinstance(live, dict)
        and live.get("status")
        == "REPORTED_PASS_EXTERNAL_ARTIFACTS_NOT_VERIFIED"
        and live.get("verification_mode") == "live_independent_regeneration"
        and live.get("gate_sha256") == receipt["origin_gate_sha256"]
        and live.get("stage_manifest_sha256") == receipt["origin_manifest_sha256"]
        and live.get("origin_per_scene_sha256") == receipt["origin_per_scene_sha256"]
        and live.get("regenerated_sha256") == receipt["regenerated_sha256"]
        and live.get("source_tree_sha256") == receipt["source_tree_sha256"]
        and live.get("role_status") == receipt["role_status"]
        and float(live.get("rtol", -1.0)) == float(receipt["rtol"])
        and float(live.get("atol", -1.0)) == float(receipt["atol"])
        and isinstance(portable, dict)
        and portable.get("status") == "DIAGNOSTIC_REPLAY_PASS"
        and portable.get("verification_mode")
        == "precomputed_regeneration_replay_diagnostic"
        and portable.get("verification_receipt_sha256")
        == sha256_file(receipt_file)
    )
    if not registered:
        raise RuntimeError(
            "non-fixture precomputed regeneration receipt is not registered"
        )


def _validate_renderer_runtime(runtime: object) -> None:
    required = {
        "schema_version",
        "profile",
        "platform_system",
        "platform_machine",
        "python_version",
        "environment_sha256",
        "critical_distributions",
        "lock_files",
        "libllvm_sha256",
        "libllvm_registry_sha256",
        "libllvm_approval_provenance",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise RuntimeError("renderer runtime receipt fields must be exact")
    if (
        runtime["schema_version"] != RENDERER_RUNTIME_SCHEMA
        or runtime["profile"] != "sionna"
        or (runtime["platform_system"], runtime["platform_machine"])
        not in {("Darwin", "arm64"), ("Linux", "x86_64")}
        or runtime["python_version"] != "3.12.13"
        or not _lower_sha256(runtime["environment_sha256"])
    ):
        raise RuntimeError("renderer runtime receipt identity is invalid")
    distributions = runtime["critical_distributions"]
    expected = {
        **EXPECTED_DISTRIBUTIONS,
        "torch": "2.9.1" if runtime["platform_system"] == "Darwin" else "2.9.1+cpu",
    }
    if not isinstance(distributions, dict) or set(distributions) != set(expected):
        raise RuntimeError("renderer runtime critical distribution set is invalid")
    for name, version in expected.items():
        record = distributions[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"version", "record_sha256"}
            or record["version"] != version
            or not _lower_sha256(record["record_sha256"])
        ):
            raise RuntimeError(f"renderer runtime distribution {name} is invalid")
    locks = runtime["lock_files"]
    if not isinstance(locks, dict) or not locks or any(
        not isinstance(name, str) or not name or not _lower_sha256(digest)
        for name, digest in locks.items()
    ):
        raise RuntimeError("renderer runtime lock inventory is invalid")
    if (
        not _lower_sha256(runtime["libllvm_sha256"])
        or not _lower_sha256(runtime["libllvm_registry_sha256"])
        or not isinstance(runtime["libllvm_approval_provenance"], str)
        or not runtime["libllvm_approval_provenance"]
    ):
        raise RuntimeError("renderer runtime LLVM provenance is invalid")
    project = Path(formal_evidence.__file__).resolve().parents[1]
    registry_path = project / "formal_v2/configs/sionna_llvm_approved_v1.json"
    if sha256_file(registry_path) != runtime["libllvm_registry_sha256"]:
        raise RuntimeError("renderer runtime LLVM registry differs from this checkout")
    registry = read_strict_json(registry_path)
    matches = [
        row
        for row in registry.get("libraries", [])
        if isinstance(row, dict)
        and row.get("platform_system") == runtime["platform_system"]
        and row.get("platform_machine") == runtime["platform_machine"]
        and row.get("sha256") == runtime["libllvm_sha256"]
        and row.get("provenance") == runtime["libllvm_approval_provenance"]
    ]
    if len(matches) != 1:
        raise RuntimeError("renderer runtime LLVM bytes were not pre-approved")


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {candidate}")
    return candidate.resolve()


def _lower_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a content-bound regeneration diagnostic on another host"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    receipt, regenerated = validate_receipt(args.receipt, args.dataset)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "regenerated.npz"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite precomputed regeneration output: {target}")
    shutil.copyfile(regenerated, target)
    if sha256_file(target) != receipt["regenerated_sha256"]:
        raise RuntimeError("copied precomputed regeneration changed in transit")
    print(
        json.dumps(
            {
                "status": "DIAGNOSTIC_REPLAY_PASS",
                "mode": "precomputed_regeneration_replay_diagnostic",
                "scientific_use": "DIAGNOSTIC_NOT_CLAIM",
                "formal_gate_eligible": False,
                "output": str(target),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
