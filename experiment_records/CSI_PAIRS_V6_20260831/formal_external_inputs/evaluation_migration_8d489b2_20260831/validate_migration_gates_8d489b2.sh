#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=/root/xunlian/Futaoran/CSI_EVALUATION_RUNTIME_FINAL_20260831/code/CSI-PAIRS-v2.0-server
RUNTIME_PYTHON=/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/CSI-PAIRS-v2.0-server/.venv-core-formal-20260819T091049Z/bin/python
EVIDENCE_ROOT=/root/xunlian/Futaoran/formal_external_inputs/evaluation_migration_8d489b2_20260831
IDENTITY="$EVIDENCE_ROOT/expected_identity.json"
MODE=${1:-all}

case "$MODE" in
  core|all) ;;
  *)
    printf 'VALIDATION_REFUSAL=unsupported mode: %s\n' "$MODE" >&2
    exit 64
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
cd "$RUNTIME_ROOT"
exec "$RUNTIME_PYTHON" -B - "$MODE" "$IDENTITY" "$EVIDENCE_ROOT" <<'PY'
from pathlib import Path
import sys

from formal_v2.formal_io import read_strict_json
from formal_v2.formal_migration_evidence import (
    _validate_real_subset_report,
    validate_evidence_report,
)
from formal_v2.formal_migration_evidence_runner import (
    CONTROL_NAMES,
    _binding,
    _canonical_sha256,
    _validate_scale_output,
    load_expected_identity,
    validate_control_observation,
    validate_performance_observation,
)


mode = sys.argv[1]
identity_path = Path(sys.argv[2]).resolve()
root = Path(sys.argv[3]).resolve()
identity = load_expected_identity(identity_path)
legacy_root = Path(str(identity["legacy_run_root"])).resolve()
new_root = Path(str(identity["new_run_root"])).resolve()


def payload(path: Path, label: str):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    value = read_strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def validate_outer(name: str, path: Path):
    value = payload(path, f"{name} report")
    validate_evidence_report(
        name,
        value,
        expected_identity=identity,
        legacy_run_root=legacy_root,
        new_run_root=new_root,
    )
    return value


performance_path = root / "performance" / "performance.json"
performance = validate_outer("performance", performance_path)
scale_stems = ("n", "two-n", "four-n")
expected_scales = ("N", "2N", "4N")
performance_scales = performance["results"]["scales"]
if len(performance_scales) != len(expected_scales):
    raise RuntimeError("performance report scale count changed")
for report_scale, stem, expected_scale in zip(
    performance_scales, scale_stems, expected_scales, strict=True
):
    observation_path = root / "performance" / f"{stem}observations.json"
    observation = payload(observation_path, f"{expected_scale} observation")
    validate_performance_observation(observation, identity=identity)
    if observation["scale"] != expected_scale:
        raise RuntimeError(f"{expected_scale} observation scale changed")
    output_path = Path(str(observation["output"]["path"])).resolve()
    if _binding(output_path) != observation["output"]:
        raise RuntimeError(f"{expected_scale} observation output binding changed")
    for field in (
        "scale",
        "sample_count",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "execution_device",
        "gpu_utilization_percent",
        "gpu_utilization_statistic",
        "gpu_sample_count",
        "peak_vram_bytes",
        "output",
    ):
        if report_scale[field] != observation[field]:
            raise RuntimeError(
                f"{expected_scale} report and observation disagree on {field}"
            )
    if report_scale["oracle_equivalent"] is not True:
        raise RuntimeError(f"{expected_scale} oracle equivalence is not true")
    _validate_scale_output(
        payload(output_path, f"{expected_scale} scale output"),
        identity=identity,
        output_path=output_path,
        checkpoint_count=int(observation["sample_count"]),
        device=str(observation["execution_device"]),
    )

performance_receipt = payload(
    root / "performance" / "runner_receipt.json", "performance runner receipt"
)
if (
    set(performance_receipt)
    != {
        "schema_version",
        "status",
        "created_utc",
        "identity_sha256",
        "runtime_provenance_sha256",
        "performance_report",
        "scales",
    }
    or
    performance_receipt.get("schema_version")
    != "csi-pairs-v6-migration-performance-runner-v1"
    or performance_receipt.get("status") != "PASS"
    or performance_receipt.get("identity_sha256") != _canonical_sha256(identity)
    or performance_receipt.get("performance_report") != _binding(performance_path)
):
    raise RuntimeError("performance runner receipt is invalid")
expected_receipt_scales = [
    {
        "name": expected_scale,
        "trace": _binding(root / "performance" / f"{stem}observations.json"),
        "output": report_scale["output"],
    }
    for report_scale, stem, expected_scale in zip(
        performance_scales, scale_stems, expected_scales, strict=True
    )
]
if performance_receipt["scales"] != expected_receipt_scales:
    raise RuntimeError("performance runner receipt scale bindings changed")

subset_path = root / "real_subset_report_8d489b2.json"
subset = payload(subset_path, "real-subset report")
_validate_real_subset_report(
    subset,
    report_path=subset_path,
    expected_identity=identity,
    legacy_root=legacy_root,
    new_root=new_root,
)
print("MIGRATION_CORE_GATES=PASS")

if mode == "all":
    equivalence = validate_outer("equivalence", root / "equivalence.json")
    control_receipt = payload(
        root / "controls" / "runner_receipt.json", "control runner receipt"
    )
    expected_reports = {
        name: _binding(root / "controls" / f"{name}.json")
        for name in CONTROL_NAMES
    }
    if (
        set(control_receipt)
        != {
            "schema_version",
            "status",
            "created_utc",
            "identity_sha256",
            "run_identity_sha256",
            "source_evaluator_output",
            "reports",
        }
        or control_receipt.get("schema_version")
        != "csi-pairs-v6-migration-control-runner-v1"
        or control_receipt.get("status") != "PASS"
        or control_receipt.get("identity_sha256") != _canonical_sha256(identity)
        or control_receipt.get("reports") != expected_reports
    ):
        raise RuntimeError("control runner receipt is invalid")
    for name in CONTROL_NAMES:
        report = validate_outer(name, root / "controls" / f"{name}.json")
        observation_path = root / "controls" / f"{name}-observations.json"
        observation = payload(
            observation_path,
            f"{name} control observation",
        )
        validate_control_observation(observation, name=name, identity=identity)
        if report["inputs"]["artifacts"] != [_binding(observation_path)]:
            raise RuntimeError(f"{name} report does not bind its observation")
        if (
            observation["run_identity_sha256"]
            != control_receipt["run_identity_sha256"]
            or observation["source_evaluator_output"]
            != control_receipt["source_evaluator_output"]
        ):
            raise RuntimeError(f"{name} observation disagrees with runner receipt")
    validate_outer("base_to_new_diff", root / "base_to_new_diff.json")
    validate_outer(
        "legacy_evaluation_inventory",
        root / "post_exit_freeze" / "legacy_evaluation_inventory.json",
    )
    validate_outer(
        "legacy_evaluation_freeze",
        root / "post_exit_freeze" / "legacy_evaluation_post_exit_freeze.json",
    )
    if equivalence["results"]["formal_subset_report"]["path"] != str(subset_path):
        raise RuntimeError("equivalence report does not bind the final real subset")
    print("MIGRATION_ALL_GATES=PASS")
PY
