"""Read-only upstream authentication, executed with the original source imports."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("origin-server", "config", "dataset", "run"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    server = Path(args.origin_server).resolve()
    sys.path.insert(0, str(server))
    from formal_v2.formal_config import load_formal_config
    from formal_v2.formal_data_verification import require_verified_roles_from_root
    from formal_v2.formal_dataset import FormalDataset
    from formal_v2.formal_evaluation import _validate_checkpoint_index
    from formal_v2.formal_evidence import (
        FACTORIAL_SCHEMA, evidence_context, require_manifested_formal_qualification,
        require_stage_manifested_gate,
    )
    from formal_v2.formal_io import read_strict_json, sha256_file

    import formal_v2
    assert Path(formal_v2.__file__).resolve().parent == server / "formal_v2"
    root = Path(args.run).resolve()
    config = load_formal_config(args.config)
    dataset = FormalDataset.load(args.dataset, require_clean_csi=True)
    if dataset.is_fixture:
        raise RuntimeError("execution repair requires the actual non-fixture dataset")
    roles = (
        "source_encoder_train", "source_probe_train", "source_probe_selection",
        "source_final_unseen_bank", "target",
    )
    verification = require_verified_roles_from_root(root, config, dataset, roles)
    qualification = require_manifested_formal_qualification(
        read_strict_json(root / "qualification/gate.json"), config, dataset,
        allow_nonscientific_fixture=False,
    )
    factorial = require_stage_manifested_gate(
        root / "factorial/gate.json", read_strict_json(root / "factorial/gate.json"),
        config, dataset, schema_version=FACTORIAL_SCHEMA,
    )
    index = read_strict_json(root / "factorial/checkpoint_index.json")
    rows = _validate_checkpoint_index(index, config, dataset, qualification)
    inventory = {}
    for stage in ("qualification", "factorial", "data_verification"):
        manifest_path = root / stage / "manifest.json"
        if manifest_path.is_file():
            inventory[str(manifest_path)] = sha256_file(manifest_path)
    for relative in (
        "qualification/gate.json", "factorial/gate.json", "data_verification/gate.json",
        "factorial/checkpoint_index.json", "factorial/normalization.json", "factorial/frozen_pilot.json",
    ):
        path = root / relative
        inventory[str(path)] = sha256_file(path)
    for row in rows:
        path = (root / "factorial" / row["path"]).resolve()
        if root / "factorial" not in path.parents or sha256_file(path) != row["sha256"]:
            raise RuntimeError("checkpoint source identity mismatch")
        inventory[str(path)] = row["sha256"]
    teacher = Path(qualification["teacher_checkpoint"]).resolve()
    if sha256_file(teacher) != qualification["teacher_checkpoint_sha256"]:
        raise RuntimeError("teacher identity mismatch")
    inventory[str(teacher)] = qualification["teacher_checkpoint_sha256"]
    evidence = evidence_context(config, dataset, "CANDIDATE_NOT_CLAIM")
    print(json.dumps({
        "schema_version": "csi-pairs-evaluation-origin-v1", "origin_server": str(server),
        "origin_run": str(root), "dataset_path": str(dataset.source_path),
        "origin_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip(),
        "evidence": evidence, "verified_roles": list(roles),
        "verification_status": verification["status"], "factorial_status": factorial["status"],
        "inventory": inventory, "checkpoints": rows,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
