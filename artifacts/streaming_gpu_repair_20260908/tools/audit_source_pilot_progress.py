"""Read-only checkpoint/CSV audit of completed NON_CLAIM source pilot arms."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "run", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    server, root, output = (Path(value).resolve() for value in (args.server, args.run, args.output))
    if output.exists() or root == output or root in output.parents:
        raise RuntimeError("Use a new audit receipt outside the active research output")
    sys.path.insert(0, str(server))
    from formal_v2.formal_config import ARMS
    from formal_v2.formal_evaluation_resume import write_atomic_json
    from formal_v2.formal_io import read_strict_json, sha256_file
    from formal_v2.formal_source_method_pilot import json_hash, load_completed_arm, state_hash
    from formal_v2.formal_training_resume import load_factorial_resume
    import torch

    identity = read_strict_json(root / "identity.json")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip() != identity["code_commit"]:
        raise RuntimeError("Research source no longer matches the run identity")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=server, text=True).strip():
        raise RuntimeError("Research source has tracked changes")
    protocol = read_strict_json(root / "protocol.json")
    if sha256_file(root / "protocol.json") != identity["protocol_sha256"] or sha256_file(root / "source_origin.json") != identity["origin_sha256"]:
        raise RuntimeError("Research protocol or authenticated input receipt changed")
    initial_sha = state_hash(torch.load(root / "shared_arm_initialization.pt", weights_only=True))
    head_sha = state_hash(torch.load(root / "shared_head_initialization.pt", weights_only=True))
    plans = read_strict_json(root / "paired_plan_hashes.json")
    split = read_strict_json(root / "source_validation_split.json")
    calibration = read_strict_json(root / "calibration.json")
    if calibration["record_sha256"] != json_hash({key: value for key, value in calibration.items() if key != "record_sha256"}) or calibration["model_sha256"] != sha256_file(root / "shared_calibration_model.pt"):
        raise RuntimeError("Research calibration receipt or saved model changed")
    validated, pending = [], []
    for recipe in ("fixed_margin", "effect_aware_margin"):
        config_path = root / recipe / "config.json"
        if not config_path.exists():
            pending.extend(recipe + "/" + arm for arm in ARMS)
            continue
        config = read_strict_json(config_path)
        pilot = {"alignment_scale": calibration["alignment_scales"][recipe], "response_scale": calibration["response_scale"], "alignment_null_tolerance": calibration["alignment_null_tolerance"]}
        context = {"identity": identity, "recipe": recipe, "pilot": pilot, "initialization_sha256": initial_sha, "head_initialization_sha256": head_sha, "plan_hashes_sha256": json_hash(plans), "source_validation_split_sha256": json_hash(split), "steps": protocol["steps_per_arm"]}
        for arm in ARMS:
            record = load_completed_arm(root / recipe, arm, context)
            if record is None:
                pending.append(recipe + "/" + arm)
                continue
            payload = load_factorial_resume(root / recipe / "resume" / arm, context=context, seed=protocol["seed"], arm=arm, total_steps=protocol["steps_per_arm"], batch_size=config["factorial"]["batch_size"], map_location="cpu")
            if payload is None or payload["completed_steps"] != protocol["steps_per_arm"] or state_hash(payload["model_state_dict"]) != record["state_value_sha256"]:
                raise RuntimeError("Saved trained model does not match its completed optimizer checkpoint")
            optimizer_steps = {int(value["step"]) for value in payload["optimizer_state_dict"]["state"].values()}
            if optimizer_steps != {protocol["steps_per_arm"]}:
                raise RuntimeError("Saved optimizer did not perform the full declared schedule")
            if record["training"]["execution_device"] != protocol["device"] or payload["training_state"]["loss_trace"] != record["loss_trace"]:
                raise RuntimeError("Device or training trace differs from the completed arm receipt")
            expected_rows = sum(len(value["validation_rows"]) for value in split.values()) * (1 + (len(protocol["validation"]["label_budgets"]) - 1) * protocol["validation"]["draws"])
            if len(record["source_rows"]) != expected_rows:
                raise RuntimeError("Completed arm has a partial source validation grid")
            if state_hash(payload["model_state_dict"]) == initial_sha:
                raise RuntimeError("Completed arm did not change the actual shared initial model")
            validated.append({"recipe": recipe, "arm": arm, "completed_steps": payload["completed_steps"], "optimizer_steps": sorted(optimizer_steps), "device": record["training"]["execution_device"], "validation_rows": expected_rows, "result_sha256": sha256_file(root / recipe / (arm + "_result.json")), "state_value_sha256": record["state_value_sha256"], "validation_csv_sha256": record["validation_csv_sha256"]})
    result = {"schema_version": "csi-pairs-source-pilot-partial-audit-v1", "utc": datetime.now(timezone.utc).isoformat(), "status": "ALL_ARM_ARTIFACTS_VALIDATED" if not pending else "PARTIAL_ARTIFACTS_VALIDATED_RESEARCH_STILL_RUNNING", "run": str(root), "code_commit": identity["code_commit"], "dataset_sha256": protocol["dataset_sha256"], "validated_arm_jobs": len(validated), "total_arm_jobs": 8, "arms": validated, "pending_arm_jobs": pending, "scientific_use": "NON_CLAIM", "formal_units_added": 0, "sota_ready": False, "scope": "Artifact/optimizer consistency only; no formal factorial or SOTA acceptance", "auditor_source_sha256": sha256_file(__file__)}
    write_atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
