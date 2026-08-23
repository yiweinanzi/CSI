#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export CSI_PAIRS_FIXTURE_RUNTIME_CACHE=1
# Tiny fixture workloads regress badly when BLAS/OpenMP oversubscribe the host.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CSI_PAIRS_PYTHON:-python3}"
OUTPUT="${1:?usage: run_formal_v2_dry_run.sh UNUSED_OUTPUT_DIRECTORY}"
FIXTURE="${OUTPUT}.fixture.npz"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B - "${OUTPUT}" "${FIXTURE}" "${PROJECT_ROOT}" <<'PY'
import json
import sys
import time
from pathlib import Path

from formal_v2.formal_cli import main as formal_main
from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_evidence import (
    configure_reproducible_runtime,
    require_manifested_formal_qualification,
)
from formal_v2.formal_io import read_strict_json

output = Path(sys.argv[1])
fixture = Path(sys.argv[2])
project_root = Path(sys.argv[3])
config_path = project_root / "formal_v2/configs/formal_v2_smoke.json"
verifier_path = project_root / "formal_v2/configs/fixture_verifier.json"

def run_stage(name, arguments):
    started = time.perf_counter()
    status = formal_main(arguments)
    print(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "event": "dry_run_stage_complete",
                "stage": name,
                "status": status,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return status


make_status = run_stage(
    "make_fixture",
    ["make-fixture", "--output", str(fixture), "--positions", "8"],
)
if make_status != 0:
    raise SystemExit(f"dry-run fixture generation failed with exit {make_status}")
verify_status = run_stage(
    "verify_data",
    [
        "verify-data",
        "--config",
        str(config_path),
        "--dataset",
        str(fixture),
        "--output",
        str(output),
        "--verifier-manifest",
        str(verifier_path),
    ]
)
if verify_status != 0:
    raise SystemExit(f"dry-run verification failed with exit {verify_status}")
qualification_status = run_stage(
    "qualify",
    [
        "qualify",
        "--config",
        str(config_path),
        "--dataset",
        str(fixture),
        "--output",
        str(output),
    ]
)
if qualification_status != 1:
    raise SystemExit(
        "dry-run qualification must fail closed with exit 1; "
        f"got {qualification_status}"
    )

configure_reproducible_runtime()
config = load_formal_config(config_path)
dataset = FormalDataset.load(fixture)
gate = read_strict_json(output / "qualification" / "gate.json")
require_manifested_formal_qualification(
    gate,
    config,
    dataset,
    allow_nonscientific_fixture=True,
)
if (
    gate.get("status") != "DRY_RUN_FAIL_NOT_EVIDENCE"
    or gate.get("passed") is not False
    or gate.get("fixture") is not True
    or gate.get("scientific_use") != "FORBIDDEN"
):
    raise SystemExit("dry-run qualification did not preserve the expected fail-closed state")

root = read_strict_json(output / "manifest.json")
files = root.get("files") if isinstance(root, dict) else None
if (
    root.get("fixture") is not True
    or root.get("scientific_use") != "FORBIDDEN"
    or not isinstance(files, list)
    or not files
    or any(
        not isinstance(row, dict)
        or row.get("fixture") is not True
        or row.get("scientific_use") != "FORBIDDEN"
        for row in files
    )
):
    raise SystemExit("dry-run root manifest is not permanently non-scientific")

print(
    json.dumps(
        {
            "dry_run_status": "EXPECTED_FAIL_CLOSED",
            "scientific_use": "FORBIDDEN",
            "status": "PASS",
        },
        sort_keys=True,
    )
)
PY
