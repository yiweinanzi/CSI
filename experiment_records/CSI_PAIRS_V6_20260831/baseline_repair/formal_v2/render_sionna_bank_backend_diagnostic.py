#!/usr/bin/env python3
"""Render one Sionna bank under the approved LLVM diagnostic runtime."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

from formal_v2 import sionna_osm_candidate as candidate


def _runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "sionna": importlib.metadata.version("sionna"),
        "sionna_rt": importlib.metadata.version("sionna-rt"),
        "mitsuba": importlib.metadata.version("mitsuba"),
        "drjit": importlib.metadata.version("drjit"),
    }


def _expected_runtime_versions() -> dict[str, str]:
    return {
        "python": candidate.SIONNA_PYTHON_VERSION,
        "sionna": candidate.SIONNA_VERSION,
        "sionna_rt": candidate.SIONNA_RT_VERSION,
        "mitsuba": candidate.MITSUBA_VERSION,
        "drjit": candidate.DRJIT_VERSION,
    }


def _prepare_backend(args: argparse.Namespace, reexec_arguments: list[str]) -> Path | None:
    if args.backend != "llvm":
        raise ValueError("the audited bank diagnostic supports only the LLVM backend")
    if args.drjit_threads != 1:
        raise ValueError("the LLVM diagnostic requires --drjit-threads=1")
    if _runtime_versions() != _expected_runtime_versions():
        raise RuntimeError(
            "LLVM diagnostic runtime versions differ from the frozen Sionna runtime"
        )
    llvm_value = os.environ.get("DRJIT_LIBLLVM_PATH")
    if not llvm_value:
        raise RuntimeError("DRJIT_LIBLLVM_PATH is required for the LLVM backend")
    llvm_path = Path(llvm_value)
    if not llvm_path.is_absolute() or llvm_path.is_symlink() or not llvm_path.is_file():
        raise RuntimeError("DRJIT_LIBLLVM_PATH must name an absolute regular non-symlink file")
    if not Path(sys.executable).is_file():
        raise RuntimeError("the active fixed Python executable is not a regular file")
    from formal_v2.sionna_runtime_lock import approved_library_record

    approved = approved_library_record(Path(candidate.__file__).resolve().parents[1], llvm_path)
    return Path(approved["libllvm_path"])


def _canonical_path_record(
    surface_ids: np.ndarray,
    delay_s: float,
    interaction_vertices_m: np.ndarray,
    extra: tuple[object, ...],
) -> bytes:
    if hasattr(candidate, "_stable_path_record"):
        return candidate._stable_path_record(
            surface_ids, delay_s, interaction_vertices_m, *extra
        )
    if extra:
        raise RuntimeError("unexpected extra stable path identity fields")
    vertices = np.asarray(interaction_vertices_m, dtype=np.float64)
    quantized_vertices = np.rint(
        vertices / candidate.PATH_VERTEX_QUANTIZATION_M
    ).astype("<i8")
    record = np.concatenate(
        (
            np.asarray(surface_ids, dtype="<i8"),
            np.asarray((int(round(delay_s * 1e12)),), dtype="<i8"),
            quantized_vertices.reshape(-1),
        )
    )
    return record.tobytes()


class _StablePathRecorder:
    def __init__(self) -> None:
        self._original = candidate._stable_path_id
        self._records: Counter[tuple[int, str]] = Counter()

    def __enter__(self) -> "_StablePathRecorder":
        def recorded(
            surface_ids: np.ndarray,
            delay_s: float,
            interaction_vertices_m: np.ndarray,
            *extra: object,
        ) -> int:
            path_id = self._original(
                surface_ids, delay_s, interaction_vertices_m, *extra
            )
            record = _canonical_path_record(
                surface_ids, delay_s, interaction_vertices_m, extra
            )
            expected_id = int.from_bytes(hashlib.sha256(record).digest()[:8], "big") & (
                (1 << 63) - 1
            )
            if path_id != expected_id:
                raise RuntimeError("stable path ID differs from its frozen canonical record")
            self._records[(path_id, record.hex())] += 1
            return path_id

        candidate._stable_path_id = recorded
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        candidate._stable_path_id = self._original

    def payload(self) -> dict[str, object]:
        records = [
            {
                "path_id": path_id,
                "canonical_record_hex": record_hex,
                "canonical_record_sha256": hashlib.sha256(
                    bytes.fromhex(record_hex)
                ).hexdigest(),
                "occurrences": count,
            }
            for (path_id, record_hex), count in sorted(self._records.items())
        ]
        return {
            "schema_version": "csi-pairs-stable-path-signature-capture-v1",
            "simulation_not_measurement": True,
            "scientific_use": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
            "canonical_definition_source": str(Path(candidate.__file__).resolve()),
            "canonical_definition_sha256": candidate.sha256_file(
                Path(candidate.__file__).resolve()
            ),
            "unique_id_record_pairs": len(records),
            "total_calls": int(sum(self._records.values())),
            "records": records,
        }


def _llvm_runtime_record(llvm_path: Path, mi, dr) -> dict[str, object]:
    return {
        **_runtime_versions(),
        "backend": "llvm",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_executable": str(Path(sys.executable).resolve()),
        "mitsuba_variant": mi.variant(),
        "drjit_thread_count": int(dr.thread_count()),
        "drjit_libllvm_path": str(llvm_path),
        "drjit_libllvm_sha256": candidate.sha256_file(llvm_path),
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
    }


def _asset_manifest_record(root: Path) -> dict[str, str]:
    manifest = (root / "asset_manifest.json").resolve()
    return {
        "asset_manifest_path": str(manifest),
        "asset_manifest_sha256": candidate.sha256_file(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--backend", choices=("llvm",), required=True)
    parser.add_argument("--drjit-threads", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    signatures_path = args.output.with_suffix(".path_signatures.json")
    if args.output.exists() or manifest_path.exists() or signatures_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output {args.output}")

    llvm_path = _prepare_backend(args, [__file__, *sys.argv[1:]])
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    import drjit as dr

    if args.drjit_threads is not None:
        if args.drjit_threads < 1:
            raise ValueError("--drjit-threads requires a positive count")
        dr.set_thread_count(args.drjit_threads)
    import sionna.rt  # noqa: F401

    root, manifest, config = candidate.load_asset_manifest(args.asset_root)
    if not 0 <= args.scene_index < len(manifest["banks"]):
        raise ValueError("scene index is out of range")
    bank_row = manifest["banks"][args.scene_index]
    bank = candidate._read_json(root / str(bank_row["bank_record_path"]))
    started_utc = datetime.now(timezone.utc)
    started = time.monotonic()
    with _StablePathRecorder() as recorder:
        rendered = candidate.render_bank(bank, root, config, args.scene_index)
    duration = time.monotonic() - started
    ended_utc = datetime.now(timezone.utc)
    arrays = {
        "scene_indices": np.asarray((args.scene_index,), dtype=np.int64),
        **{name: np.asarray(value)[None, ...] for name, value in rendered.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate._write_npz_exclusive(args.output, arrays)
    candidate._write_json_exclusive(signatures_path, recorder.payload())
    tool_path = Path(__file__).resolve()
    runtime = _llvm_runtime_record(llvm_path, mi, dr)
    payload = {
        "schema_version": "csi-pairs-sionna-bank-backend-diagnostic-v2",
        "status": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "scientific_use": "DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "simulation_not_measurement": True,
        "fixture": False,
        "run_id": uuid.uuid4().hex,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "started_utc": started_utc.isoformat(),
        "ended_utc": ended_utc.isoformat(),
        "duration_seconds": duration,
        "backend": args.backend,
        "mitsuba_variant": mi.variant(),
        "drjit_thread_count": int(dr.thread_count()),
        "scene_index": args.scene_index,
        "scene_id": bank_row["scene_id"],
        "asset_root": str(root),
        **_asset_manifest_record(root),
        "output_path": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": candidate.sha256_file(args.output),
        "path_signatures_path": str(signatures_path.resolve()),
        "path_signatures_sha256": candidate.sha256_file(signatures_path),
        "runtime": runtime,
        "generator_path": str(Path(candidate.__file__).resolve()),
        "generator_sha256": candidate.sha256_file(Path(candidate.__file__).resolve()),
        "tool_path": str(tool_path),
        "tool_sha256": candidate.sha256_file(tool_path),
        "argv": [str(Path(sys.executable)), *sys.argv],
    }
    candidate._write_json_exclusive(manifest_path, payload)
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
