from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from formal_v2.sionna_osm_candidate import ensure_sionna_runtime


MAX_BANK_ATTEMPTS = 3
POLL_INTERVAL_SECONDS = 0.2


def _bank_fields() -> tuple[str, ...]:
    from formal_v2.sionna_osm_candidate import REGENERATED_FIELDS

    return tuple(field for field in REGENERATED_FIELDS if field != "engine_config_json")


def _load_bank_archive(path: str | Path) -> dict[str, np.ndarray]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"regenerated bank is not a regular file: {target}")
    fields = _bank_fields()
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise RuntimeError(f"regenerated bank fields are not exact: {target}")
        return {field: np.asarray(archive[field]) for field in fields}


def _write_bank_archive(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    from formal_v2.sionna_osm_candidate import _write_npz_exclusive

    if set(arrays) != set(_bank_fields()):
        raise RuntimeError("regenerated bank fields drifted from the V6 contract")
    return _write_npz_exclusive(path, arrays)


def _render_bank_process(asset_root: str | Path, scene_index: int, output: str | Path) -> Path:
    from formal_v2.sionna_osm_candidate import (
        _read_json,
        load_asset_manifest,
        render_bank,
    )

    root, manifest, config = load_asset_manifest(asset_root)
    index = int(scene_index)
    if index < 0 or index >= len(manifest["banks"]):
        raise ValueError(f"scene index is outside the frozen bank ledger: {index}")
    bank = _read_json(root / str(manifest["banks"][index]["bank_record_path"]))
    rendered = render_bank(bank, root, config, index)
    arrays = {field: np.asarray(rendered[field]) for field in _bank_fields()}
    return _write_bank_archive(output, arrays)


def _terminate_processes(active: dict[subprocess.Popen, dict]) -> None:
    for process in active:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30.0
    for process in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for job in active.values():
        job["stdout"].close()
        job["stderr"].close()


def _run_process_scheduler(
    scene_count: int,
    output_root: str | Path,
    workers: int,
    command_builder,
    validator,
    *,
    max_attempts: int = MAX_BANK_ATTEMPTS,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> list[Path]:
    """Run one native RT process per bank so a Dr.Jit abort is attributable and retryable."""
    count = int(scene_count)
    worker_count = int(workers)
    attempt_limit = int(max_attempts)
    if count < 1 or worker_count < 1 or attempt_limit < 1:
        raise ValueError("isolated regeneration counts must all be positive")

    root = Path(output_root)
    attempts_root = root / "bank_attempts"
    completed_root = root / "completed_banks"
    attempts_root.mkdir(parents=True, exist_ok=True)
    completed_root.mkdir(parents=True, exist_ok=True)
    attempts = [0] * count
    completed: dict[int, Path] = {}
    pending: deque[int] = deque()

    for scene_index in range(count):
        final_path = completed_root / f"bank-{scene_index:03d}.npz"
        existing_attempts = list(
            (attempts_root / f"bank-{scene_index:03d}").glob("attempt-*.stdout.txt")
        )
        attempts[scene_index] = len(existing_attempts)
        if final_path.exists() or final_path.is_symlink():
            validator(final_path)
            completed[scene_index] = final_path
        else:
            if attempts[scene_index] >= attempt_limit:
                raise RuntimeError(
                    f"bank {scene_index} exhausted {attempt_limit} preserved attempts"
                )
            pending.append(scene_index)

    active: dict[subprocess.Popen, dict] = {}
    try:
        while pending or active:
            while pending and len(active) < worker_count:
                scene_index = pending.popleft()
                attempts[scene_index] += 1
                attempt = attempts[scene_index]
                bank_root = attempts_root / f"bank-{scene_index:03d}"
                bank_root.mkdir(parents=True, exist_ok=True)
                prefix = bank_root / f"attempt-{attempt:02d}"
                attempt_output = prefix.with_suffix(".npz")
                stdout_handle = prefix.with_suffix(".stdout.txt").open("xb")
                stderr_handle = prefix.with_suffix(".stderr.txt").open("xb")
                command = [str(value) for value in command_builder(scene_index, attempt_output)]
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        close_fds=True,
                    )
                except BaseException:
                    stdout_handle.close()
                    stderr_handle.close()
                    raise
                active[process] = {
                    "scene_index": scene_index,
                    "attempt": attempt,
                    "output": attempt_output,
                    "stdout": stdout_handle,
                    "stderr": stderr_handle,
                }
                print(
                    f'{{"attempt":{attempt},"event":"independent_regeneration_bank_process_started",'
                    f'"pid":{process.pid},"scene_index":{scene_index}}}',
                    flush=True,
                )

            finished = [process for process in active if process.poll() is not None]
            if not finished:
                time.sleep(float(poll_interval))
                continue
            for process in finished:
                job = active.pop(process)
                job["stdout"].close()
                job["stderr"].close()
                scene_index = int(job["scene_index"])
                attempt = int(job["attempt"])
                error = None
                if process.returncode == 0:
                    try:
                        validator(job["output"])
                    except BaseException as caught:
                        error = f"invalid output: {caught}"
                else:
                    error = f"process exit code {process.returncode}"
                if error is None:
                    final_path = completed_root / f"bank-{scene_index:03d}.npz"
                    os.link(job["output"], final_path)
                    validator(final_path)
                    completed[scene_index] = final_path
                    print(
                        f'{{"attempt":{attempt},"event":"independent_regeneration_bank_complete",'
                        f'"scene_index":{scene_index}}}',
                        flush=True,
                    )
                    continue
                print(
                    json.dumps(
                        {
                            "attempt": attempt,
                            "error": error,
                            "event": "independent_regeneration_bank_retry",
                            "scene_index": scene_index,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                if attempt >= attempt_limit:
                    raise RuntimeError(
                        f"bank {scene_index} failed after {attempt_limit} isolated attempts: {error}"
                    )
                pending.append(scene_index)
    except BaseException:
        _terminate_processes(active)
        raise

    if set(completed) != set(range(count)):
        raise RuntimeError("isolated regeneration did not complete every formal bank")
    return [completed[index] for index in range(count)]


def _regenerate_dataset(dataset_path: str | Path, output_dir: str | Path) -> Path:
    from formal_v2.formal_dataset import FormalDataset
    from formal_v2.sionna_osm_candidate import (
        REGENERATED_FIELDS,
        _write_npz_exclusive,
        build_engine_config,
        canonical_json,
        load_asset_manifest,
    )

    dataset = FormalDataset.load(dataset_path)
    if dataset.is_fixture or dataset.metadata["scientific_use"] != "CANDIDATE":
        raise RuntimeError("Sionna OSM verifier only accepts the non-fixture CANDIDATE dataset")
    asset_root = Path(dataset.source_path).parent / "assets"
    root, manifest, config = load_asset_manifest(asset_root)
    expected_engine = build_engine_config(root, manifest, config)
    if expected_engine != dataset.engine_config:
        raise RuntimeError("dataset engine config does not reconstruct from immutable assets")
    if dataset.scene_count != len(manifest["banks"]):
        raise RuntimeError("dataset scene count differs from the frozen bank ledger")
    workers = min(int(config["renderer"]["verification_workers"]), dataset.scene_count)
    source = Path(__file__).resolve()

    def command_builder(scene_index: int, bank_output: Path) -> list[str]:
        return [
            sys.executable,
            str(source),
            "--render-bank",
            "--asset-root",
            str(root),
            "--scene-index",
            str(scene_index),
            "--output",
            str(bank_output),
        ]

    paths = _run_process_scheduler(
        dataset.scene_count,
        output_dir,
        workers,
        command_builder,
        _load_bank_archive,
    )
    rows = [_load_bank_archive(path) for path in paths]
    arrays = {
        field: np.stack([row[field] for row in rows])
        for field in _bank_fields()
    }
    arrays["engine_config_json"] = np.asarray(canonical_json(expected_engine))
    if set(arrays) != set(REGENERATED_FIELDS):
        raise RuntimeError("regeneration output fields drifted from the V6 contract")
    return _write_npz_exclusive(Path(output_dir) / "regenerated.npz", arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently regenerate the frozen Sionna/OSM formal candidate"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--render-bank", action="store_true")
    parser.add_argument("--asset-root")
    parser.add_argument("--scene-index", type=int)
    args = parser.parse_args(argv)
    if argv is not None:
        raise RuntimeError("the authenticated verifier requires process argv")
    ensure_sionna_runtime([__file__, *sys.argv[1:]])
    if args.render_bank:
        if args.dataset is not None or args.asset_root is None or args.scene_index is None:
            parser.error("--render-bank requires --asset-root and --scene-index, without --dataset")
        _render_bank_process(args.asset_root, args.scene_index, args.output)
    else:
        if args.dataset is None or args.asset_root is not None or args.scene_index is not None:
            parser.error("dataset regeneration requires --dataset only")
        _regenerate_dataset(args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
