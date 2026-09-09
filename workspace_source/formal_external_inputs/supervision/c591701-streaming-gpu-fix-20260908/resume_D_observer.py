"""Resume read-only observations of the surviving D child, without adopting it."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sample_host_resources import process, snapshot


OPERATOR = Path(__file__).resolve().parent
OUTPUT = OPERATOR / "checks" / "D-observer-resumed-20260908T1927"
RUN = Path("/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/runs/complete-D-2ce2472")
EXPECTED_PID = 26999
EXPECTED_TICKS = "9109609"


def utc():
    return datetime.now(timezone.utc).isoformat()


def target_state():
    row = process(EXPECTED_PID)
    if row["start_ticks"] != EXPECTED_TICKS:
        raise RuntimeError("D PID has been reused")
    fields = Path(f"/proc/{EXPECTED_PID}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0]


def sample():
    row = snapshot(EXPECTED_PID)
    row["gpu_csv"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    for name in ("probe_progress", "evaluation_progress"):
        row[name] = json.loads((RUN / "evaluation_state" / (name + ".json")).read_text())
    return row


def observe():
    with (OUTPUT / "resources.jsonl").open("x") as handle:
        while True:
            try:
                state = target_state()
                if state == "Z":
                    reason = "Original D process observed as zombie; observer is not its parent."
                    break
            except (FileNotFoundError, ProcessLookupError, RuntimeError) as error:
                reason = str(error)
                break
            try:
                row = sample()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                row = {"utc": utc(), "monitor_warning": type(error).__name__ + ": " + str(error)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            time.sleep(10)
    (OUTPUT / "observer_finished.json").write_text(json.dumps({
        "utc": utc(), "reason": reason, "evaluation_returncode": None,
        "note": "Observer cannot recover the lost parent wait status; inspect and independently validate D artifacts.",
        "bounded_result_exists": (RUN / "bounded_validation_result.json").is_file(),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        observe()
        return
    target_state()
    command = Path(f"/proc/{EXPECTED_PID}/cmdline").read_bytes().split(b"\0")
    if b"run-evaluation-repair" not in command or os.fsencode(RUN) not in command:
        raise RuntimeError("Process is not the expected D evaluation")
    first_sample = sample()
    OUTPUT.mkdir(exist_ok=False)
    record = {
        "utc": utc(), "evaluation_pid": EXPECTED_PID, "evaluation_start_ticks": EXPECTED_TICKS,
        "evaluation_ppid": process(EXPECTED_PID)["ppid"],
        "old_wrapper_pid": 26991, "old_wrapper_present": Path("/proc/26991").exists(),
        "last_original_resource_log_mtime": datetime.fromtimestamp(
            (OPERATOR / "checks/D-complete-formal-unit/resources.jsonl").stat().st_mtime, timezone.utc,
        ).isoformat(),
        "warning": "Original resource observers disappeared. D survived. Missing samples and parent wait status cannot be reconstructed.",
        "action": "Read-only observer restarted. No evaluation signals, source changes or restart.",
        "scope": "New observations only; not retrospective peaks or function attribution.",
        "initial_observation": first_sample,
    }
    with (OUTPUT / "observer.stdout.log").open("xb") as stdout, (OUTPUT / "observer.stderr.log").open("xb") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True,
        )
    record["observer_pid"] = child.pid
    (OUTPUT / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"observer_pid": child.pid, "record": str(OUTPUT / "record.json")}))


if __name__ == "__main__":
    main()
