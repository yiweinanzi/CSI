"""Read authenticated existing summaries; do not run or modify experiments."""

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ORIGIN_COMMIT = "c5917018610b262f5263ccaeba2f087b2eb96a96"
DATASET_SHA256 = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
ARMS = ("endpoint", "alignment", "response", "full")
CITIES = ("target-boston", "target-seattle")
BUDGETS = (0, 8, 32, 128)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def meter_grid(rows):
    keyed = {}
    expected = {(city, budget, arm) for city in CITIES for budget in BUDGETS for arm in ARMS}
    for row in rows:
        key = row["city_id"], int(row["budget"]), row["arm"]
        value = float(row["mean_bank_median_error_m"])
        if key not in expected or key in keyed or not math.isfinite(value) or value <= 0:
            raise ValueError("Meter grid has duplicate, unexpected, or invalid cells")
        keyed[key] = value
    if set(keyed) != expected:
        raise ValueError("Meter grid is incomplete")
    return [
        {
            "city": city, "k": budget,
            "meters": {arm: keyed[city, budget, arm] for arm in ARMS},
            "full_minus_response_m": keyed[city, budget, "full"] - keyed[city, budget, "response"],
            "full_strictly_first_among_four_arms": all(keyed[city, budget, "full"] < keyed[city, budget, arm] for arm in ARMS if arm != "full"),
        }
        for city in CITIES for budget in BUDGETS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_bytes = args.origin_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    upstream = receipt["upstream"]
    if upstream["origin_git_commit"] != ORIGIN_COMMIT or upstream["evidence"]["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Unexpected original experiment identity")
    root = Path(upstream["origin_run"]) / "factorial"
    retained = {}

    def read_verified(name, expected):
        data = (root / name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Authenticated input changed: " + name)
        retained[name] = data
        return data

    manifest = json.loads(read_verified("manifest.json", upstream["inventory"][str(root / "manifest.json")]))
    gate = json.loads(read_verified("gate.json", upstream["inventory"][str(root / "gate.json")]))
    rows_by_file = {}
    for name in ("localization_summary.csv", "training_summary.csv"):
        matches = [row for row in manifest["files"] if row["path"] == name]
        if len(matches) != 1:
            raise ValueError("Original manifest must bind exactly one " + name)
        raw = read_verified(name, matches[0]["sha256"])
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
        for row in rows:
            if row["dataset_sha256"] != DATASET_SHA256 or row["config_sha256"] != manifest["config_sha256"] or row["fixture"] != "False" or row["source_tree_sha256"] != manifest["source_tree_sha256"]:
                raise ValueError("Summary row evidence identity mismatch")
        rows_by_file[name] = rows
    cells = meter_grid(rows_by_file["localization_summary.csv"])
    fields = (
        "seed", "steps", "checkpoint_rule", "full_joint_loss",
        "alignment_gradient_norm_mean", "response_gradient_norm_mean",
        "encoder_alignment_response_grad_cosine_mean",
        "encoder_alignment_response_grad_cosine_count", "encoder_grad_cosine_note",
        "task_log_variance_alignment", "task_log_variance_response",
        "final_alignment_active_fixed_loss", "final_response_loss",
    )
    diagnostics = [{key: row[key] for key in fields} for row in rows_by_file["training_summary.csv"] if row["arm"] == "full"]
    progress_path = args.origin_receipt.parent / "evaluation_state/probe_progress.json"
    progress = json.loads(progress_path.read_text())
    result = {
        "status": "AUDIT_COMPLETE_NON_CLAIM", "sota_ready": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "origin_commit": ORIGIN_COMMIT, "evaluation_commit": receipt["evaluation_git_commit"],
        "original_npz_sha256": DATASET_SHA256,
        "npz_identity_scope": "Original live-authentication receipt; this summary-only audit does not reread the full NPZ.",
        "origin_receipt_sha256": digest(receipt_bytes),
        "inputs_sha256": {name: digest(raw) for name, raw in retained.items()},
        "metric": "mean_bank_median_error_m; not pooled median, log utility or significance",
        "cells": cells,
        "full_strict_first_cell_count": sum(row["full_strictly_first_among_four_arms"] for row in cells),
        "factorial_gate_vector": gate["gate_vector"],
        "g4_assessed_subgates": gate["g4_assessed_subgates"],
        "g4_unassessed_subgates": gate["g4_unassessed_subgates"],
        "g5_subgates": gate["g5_subgates"],
        "full_training_diagnostics": diagnostics,
        "diagnostic_boundary": "Gradient norm magnitudes and mean cosine do not establish causality, conflict rates, or a successful remedy. This audit changes no model or gate.",
        "streaming_snapshot": {"updated_at": progress["updated_at"], "workers": progress["workers"]},
        "external_baseline_boundary": "No Wi-GATr/PMNet meters are inferred from papers or other runs. No assembled same-run comparison is certified by this audit.",
        "execution_boundary": "The existing evaluator is not interrupted. This command starts no training and buys no resources.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("gate.json", "localization_summary.csv", "training_summary.csv"):
        (args.output / name).write_bytes(retained[name])
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = [
        "# SOTA obstacle audit", "", "Existing evidence only. NON_CLAIM; no method change, new training, gate modification or SOTA claim.", "",
        "Meters below are the original mean of bank median errors, not pooled medians.", "",
        "| City | k | Endpoint | Alignment | Response | Full | Full minus Response |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        values = " | ".join(f"{cell['meters'][arm]:.6f}" for arm in ARMS)
        report.append(f"| {cell['city']} | {cell['k']} | {values} | {cell['full_minus_response_m']:.6f} |")
    report += [
        "", f"Full is strictly first among the four arms in {result['full_strict_first_cell_count']}/8 cells.",
        "This ranking is descriptive and does not assert significant degradation in every cell.", "",
        "The original assessed G4 conditions and G5 remain FAIL. Unassessed conditions remain unassessed.",
        "Execution repair cannot change this independent localization table.", "",
        "Training gradient summaries are retained in audit.json as hypotheses to investigate, not causal findings.",
        "Method changes or main-model retraining require a separately authorized experiment and locked selection protocol.",
        "Selection must not use target query results; repeated target exposure must be disclosed.", "",
        "Source summary bytes, input hashes, identities and the live progress snapshot are retained alongside this report.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "full_strict_first_cells": result["full_strict_first_cell_count"], "audit_sha256": digest((args.output / "audit.json").read_bytes())}))


if __name__ == "__main__":
    main()
