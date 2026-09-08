"""Detached research-only process receipts and resource samples; never signals peers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--research-output", required=True)
    parser.add_argument("--yield-when")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if "run-source-method-pilot" not in command:
        parser.error("This supervisor only runs the independent source method pilot")
    logs = Path(args.logs).resolve()
    output = Path(args.research_output).resolve()
    if "--output" not in command or Path(command[command.index("--output") + 1]).resolve() != output:
        parser.error("Research output must match the supervised command")
    if args.detach:
        logs.mkdir(parents=True, exist_ok=False)
        child_args = [sys.executable, "-B", str(Path(__file__).resolve()), "--cwd", args.cwd, "--logs", str(logs), "--research-output", str(output)]
        if args.yield_when:
            child_args += ["--yield-when", args.yield_when]
        with (logs / "supervisor.log").open("xb") as handle:
            child = subprocess.Popen(child_args + ["--", *command], cwd=args.cwd, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        atomic_json(logs / "launch.json", {"launched_at": utc(), "supervisor_pid": child.pid, "status": "SUPERVISOR_LAUNCHED_NOT_EXPERIMENT_COMPLETE"})
        print(json.dumps({"supervisor_pid": child.pid, "logs": str(logs)}))
        return 0
    logs.mkdir(parents=True, exist_ok=True)
    if (logs / "start.json").exists():
        raise RuntimeError("Use a new log directory for every attempt")
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=args.cwd)
    base = {
        "command": command, "cwd": args.cwd, "supervisor_pid": os.getpid(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.cwd, text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CSI_PAIRS_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONDONTWRITEBYTECODE")},
        "scientific_use": "NON_CLAIM", "sota_ready": False,
    }
    with (logs / "stdout.log").open("xb") as stdout, (logs / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen(command, cwd=args.cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True)
        base.update(pid=child.pid, started_at=utc())
        try:
            base["process_start_ticks"] = int(Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            base["process_start_ticks"] = None
        atomic_json(logs / "start.json", {**base, "status": "STARTED_NOT_COMPLETED"})
        with (logs / "resources.jsonl").open("a") as handle:
            while child.poll() is None:
                row = {"utc": utc(), "pid": child.pid}
                try:
                    row["process"] = subprocess.check_output(["ps", "-p", str(child.pid), "-o", "pid,ppid,etime,time,pcpu,rss"], text=True, timeout=10)
                    row["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=15)
                    for name in ("memory.limit_in_bytes", "memory.usage_in_bytes"):
                        path = Path("/sys/fs/cgroup/memory") / name
                        if path.is_file():
                            row[name] = int(path.read_text())
                    progress = output / "progress.json"
                    if progress.is_file():
                        row["progress"] = json.loads(progress.read_text())
                    if args.yield_when and Path(args.yield_when).is_file() and output.is_dir():
                        pause = output / "PAUSE_REQUESTED"
                        if not pause.exists():
                            atomic_json(pause, {"reason": "Yield the research GPU at the next safe boundary for formal E validation", "observed_receipt": args.yield_when, "utc": utc()})
                        row["research_pause_requested"] = True
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    row["monitor_warning"] = str(error)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
    record = {**base, "completed_at": utc(), "returncode": child.returncode, "status": "PROCESS_EXITED"}
    atomic_json(logs / "exit.json", record)
    print(json.dumps(record), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
