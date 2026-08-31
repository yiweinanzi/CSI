from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .formal_io import artifact_manifest, sha256_file, write_csv, write_json
from .formal_resources import validate_resource_registry
from .formal_io import read_strict_json


SIONNA_REVISION = "04ddb9312116b408093b9d3ad363a3df355093a6"
LRM_REVISION = "1ba19ae1df1d26302fcfbaab14efc2347313da5d"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Authenticated Sionna and large-radio-map facility")
    parser.add_argument("--runtime-root", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    tiling = commands.add_parser("generate-tiling")
    tiling.add_argument("--bbox", type=float, nargs=4, required=True)
    tiling.add_argument("--output-name", required=True)
    tiling.add_argument("--transmitters", required=True)
    scenes = commands.add_parser("build-scenes")
    scenes.add_argument("--bboxes", required=True)
    scenes.add_argument("--area-name", required=True)
    scenes.add_argument("--keep-local", action="store_true")
    maps = commands.add_parser("compute-radio-maps")
    maps.add_argument("--scenes", required=True)
    maps.add_argument("--bboxes", required=True)
    maps.add_argument("--transmitters", required=True)
    maps.add_argument("--output", required=True)
    maps.add_argument("--samples", type=int, default=500_000_000)
    audit = commands.add_parser("audit-radio-maps")
    audit.add_argument("--radio-map-root", required=True)
    audit.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(f"Sionna facility error: {error}", file=sys.stderr)
        return 2


def run(args) -> dict:
    runtime = Path(args.runtime_root).resolve()
    verified = verify_runtime(runtime)
    if args.command == "verify":
        return verified
    scripts = runtime / "src" / "sionna-large-radio-maps-main" / "scripts"
    python = runtime / "venv" / "bin" / "python"
    environment = _runtime_environment(runtime)
    if args.command == "generate-tiling":
        transmitter_target = runtime / "data" / "remote" / "transmitters" / "data.csv"
        transmitter_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(args.transmitters).resolve(), transmitter_target)
        command = [
            str(python), str(scripts / "generate_tiling.py"), "--bbox",
            *(str(value) for value in args.bbox), args.output_name,
        ]
        return _execute(command, runtime, environment, "generate-tiling")
    if args.command == "build-scenes":
        command = [
            str(python), str(scripts / "scene_builder.py"), "file",
            str(Path(args.bboxes).resolve()), "--subdir", args.area_name,
        ]
        if args.keep_local:
            command.extend(("--keep-local", "--no-compress"))
        return _execute(command, runtime, environment, "build-scenes")
    if args.command == "compute-radio-maps":
        command = [
            str(python), str(scripts / "compute_radio_maps.py"),
            "--scenes", str(Path(args.scenes).resolve()),
            "--bboxes", str(Path(args.bboxes).resolve()),
            "--transmitters", str(Path(args.transmitters).resolve()),
            "--output-dir", str(Path(args.output).resolve()),
            "--samples", str(int(args.samples)),
        ]
        return _execute(command, runtime, environment, "compute-radio-maps")
    return audit_radio_maps(Path(args.radio_map_root), Path(args.output), verified)


def verify_runtime(runtime: Path) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    registry_path = project_root / "formal_v2" / "configs" / "waibu_resources_v1.json"
    rows = validate_resource_registry(read_strict_json(registry_path), project_root / "waibu")
    required = {
        "sionna-main.zip": "fdbf89f307cc8933535af1587f00f1bcbd4b5edf7715275cd461bd4779f1fac7",
        "sionna-large-radio-maps-main.zip": "694ad17e7977e1c1adbdc8f93e6dcb1856e14cdf1b25f33da14c0e7aca80c33b",
    }
    authenticated = {row["file"]: row["actual_sha256"] for row in rows if row["file"] in required}
    if authenticated != required:
        raise RuntimeError("Sionna source archive authentication failed")
    python = runtime / "venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("Sionna runtime is not installed; run setup_sionna.sh first")
    probe = subprocess.run(
        [
            str(python), "-c",
            "import importlib.metadata as m; import h5py, sionna, sionna.rt, sionna_lrm, torch; "
            "print(m.version('sionna')); print(m.version('sionna-rt')); "
            "print('source@1ba19ae1df1d26302fcfbaab14efc2347313da5d'); "
            "print(torch.__version__); print(h5py.__version__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_runtime_environment(runtime),
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Sionna runtime import failed: {probe.stderr.strip()}")
    versions = probe.stdout.strip().splitlines()
    if (
        len(versions) != 5
        or versions[0] != "2.0.1"
        or versions[1] != "1.2.1"
        or versions[3] != "2.9.1+cpu"
        or versions[4] != "3.15.1"
    ):
        raise RuntimeError("Sionna runtime versions differ from the frozen facility")
    return {
        "schema_version": "csi-pairs-v6-sionna-runtime-v1",
        "status": "PASS",
        "passed": True,
        "runtime_root": str(runtime),
        "sionna_revision": SIONNA_REVISION,
        "large_radio_maps_revision": LRM_REVISION,
        "versions": {
            "sionna": versions[0],
            "sionna_rt": versions[1],
            "large_radio_maps": versions[2],
            "torch": versions[3],
            "h5py": versions[4],
        },
        "archive_sha256": authenticated,
        "license_id": "Apache-2.0",
    }


def _runtime_environment(runtime: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONPATH": str(runtime / "src" / "sionna-large-radio-maps-main"),
        "SLRM_DATA_DIR": str(runtime / "data"),
    }
    configured = environment.get("DRJIT_LIBLLVM_PATH")
    if configured and Path(configured).is_file():
        return environment
    candidates = []
    for root in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt")):
        if root.exists():
            candidates.extend(root.glob("**/libLLVM-*.so"))
    if not candidates:
        raise RuntimeError("Dr.Jit requires libLLVM; set DRJIT_LIBLLVM_PATH")
    environment["DRJIT_LIBLLVM_PATH"] = str(sorted(candidates)[-1])
    return environment


def audit_radio_maps(root: Path, output: Path, runtime_record: dict) -> dict:
    source = root.resolve()
    destination = output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(source.glob("rm_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"rm", "tx_positions", "measurement_z_offset"}:
                raise RuntimeError(f"unexpected official radio-map fields: {path}")
            radio_map = np.asarray(archive["rm"])
            transmitters = np.asarray(archive["tx_positions"])
            height = float(np.asarray(archive["measurement_z_offset"]).item())
            if radio_map.ndim != 1 or not np.all(np.isfinite(radio_map)) or np.any(radio_map < 0):
                raise RuntimeError(f"invalid path-gain radio map: {path}")
            if transmitters.ndim != 2 or transmitters.shape[0] != 3 or not np.all(np.isfinite(transmitters)):
                raise RuntimeError(f"invalid transmitter coordinates: {path}")
            if not np.isfinite(height):
                raise RuntimeError(f"invalid measurement height: {path}")
            rows.append(
                {
                    "tile_id": path.stem.removeprefix("rm_"),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "surface_entries": radio_map.size,
                    "transmitter_count": transmitters.shape[1],
                    "measurement_z_offset_m": height,
                    "maximum_path_gain": float(np.max(radio_map)) if radio_map.size else 0.0,
                }
            )
    if not rows:
        raise RuntimeError("no official rm_*.npz outputs were found")
    write_csv(destination / "radio_map_inventory.csv", rows)
    gate = {
        "schema_version": "csi-pairs-v6-sionna-radio-map-audit-v1",
        "status": "PASS",
        "passed": True,
        "tile_count": len(rows),
        "runtime": runtime_record,
        "scientific_claim_scope": "facility output integrity only",
    }
    write_json(destination / "gate.json", gate)
    write_json(destination / "manifest.json", {"schema_version": "csi-pairs-sionna-facility-manifest-v1", "files": artifact_manifest(destination)})
    return gate


def _execute(command, runtime, environment, operation):
    digest = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode("utf-8")).hexdigest()
    completed = subprocess.run(command, cwd=runtime, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"official Sionna facility operation failed: {operation}")
    return {
        "schema_version": "csi-pairs-v6-sionna-operation-v1",
        "status": "PASS",
        "passed": True,
        "operation": operation,
        "command_sha256": digest,
        "return_code": completed.returncode,
    }


if __name__ == "__main__":
    raise SystemExit(main())
