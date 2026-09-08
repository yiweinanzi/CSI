"""Authenticate existing source-only inputs using their original code version."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("origin-server", "config", "dataset", "upstream-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    server = Path(args.origin_server).resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip() != "c5917018610b262f5263ccaeba2f087b2eb96a96":
        raise RuntimeError("Source pilot requires its recorded original c591701 inputs")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=server, text=True).strip():
        raise RuntimeError("Original source has tracked changes")
    sys.path.insert(0, str(server))
    from formal_v2.formal_config import load_formal_config
    from formal_v2.formal_data_verification import require_verified_roles_from_root
    from formal_v2.formal_dataset import FormalDataset
    from formal_v2.formal_evidence import (
        FACTORIAL_SCHEMA, evidence_context, require_manifested_formal_qualification,
        require_stage_manifested_gate,
    )
    from formal_v2.formal_io import read_strict_json, sha256_file

    import formal_v2
    if Path(formal_v2.__file__).resolve().parent != server / "formal_v2":
        raise RuntimeError("Original input authentication imported a different source tree")

    root = Path(args.upstream_root).resolve()
    config = load_formal_config(args.config)
    dataset = FormalDataset.load(args.dataset, require_clean_csi=True)
    expected = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
    if dataset.is_fixture or sha256_file(dataset.source_path) != expected:
        raise RuntimeError("Not the recorded real formal NPZ")
    roles = ("source_encoder_train", "source_method_selection")
    verification = require_verified_roles_from_root(root, config, dataset, roles)
    gate = require_manifested_formal_qualification(read_strict_json(root / "qualification/gate.json"), config, dataset, allow_nonscientific_fixture=False)
    factorial = require_stage_manifested_gate(
        root / "factorial/gate.json", read_strict_json(root / "factorial/gate.json"),
        config, dataset, schema_version=FACTORIAL_SCHEMA,
    )
    normalization = read_strict_json(root / "factorial/normalization.json")
    if normalization["source_roles"] != ["source_encoder_train"]:
        raise RuntimeError("Original normalization was not source-training-only")
    teacher = Path(gate["teacher_checkpoint"]).resolve()
    if sha256_file(teacher) != gate["teacher_checkpoint_sha256"]:
        raise RuntimeError("Original teacher identity changed")
    inventory = {str(root / relative): sha256_file(root / relative) for relative in (
        "qualification/gate.json", "qualification/manifest.json", "data_verification/gate.json",
        "data_verification/manifest.json", "factorial/normalization.json",
        "factorial/gate.json", "factorial/manifest.json",
    )}
    inventory[str(teacher)] = gate["teacher_checkpoint_sha256"]
    print(json.dumps({
        "schema_version": "csi-pairs-source-method-input-authentication-v1",
        "origin_commit": "c5917018610b262f5263ccaeba2f087b2eb96a96",
        "dataset_sha256": expected, "dataset_path": str(dataset.source_path),
        "verified_roles": list(roles), "verification_status": verification["status"],
        "teacher_checkpoint": str(teacher), "inventory": inventory,
        "factorial_status_preserved": factorial["status"],
        "factorial_reuse_scope": "Manifest-authenticated source normalization only; no trained arm reused.",
        "evidence": evidence_context(config, dataset, "NON_CLAIM"),
        "source_scenes": {role: dataset.indices_for_role(role).tolist() for role in roles},
        "scope": "Source input authentication only; does not approve method modifications or target results.",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
