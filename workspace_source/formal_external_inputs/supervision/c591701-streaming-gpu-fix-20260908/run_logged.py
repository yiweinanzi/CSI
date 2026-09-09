import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    root = Path(__file__).resolve().parent / "checks" / args.label
    root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=args.cwd)
    (root / "tracked.patch").write_bytes(patch)
    with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=args.cwd, stdout=stdout, stderr=stderr)
        with (root / "resources.jsonl").open("a") as samples:
            while child.poll() is None:
                proc = Path(f"/proc/{child.pid}")
                try:
                    record = {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "pid": child.pid, "monotonic": time.monotonic(),
                        "process": subprocess.check_output(["ps", "-p", str(child.pid), "-o", "pid,ppid,etime,time,pcpu,rss"], text=True),
                        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=15),
                    }
                    samples.write(json.dumps(record) + "\n")
                    samples.flush()
                except (OSError, subprocess.SubprocessError) as error:
                    samples.write(json.dumps({"monitor_warning": str(error)}) + "\n")
                time.sleep(5)
        result = child
    record = {
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "cwd": args.cwd, "returncode": result.returncode,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.cwd, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=args.cwd, text=True),
        "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CSI_PAIRS_DEVICES", "CUBLAS_WORKSPACE_CONFIG")},
    }
    (root / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"record": str(root / "record.json"), "returncode": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
