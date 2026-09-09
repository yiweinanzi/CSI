import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD_SUPERVISION = HERE.parent / "c591701-formal-resume-20260908T0120"
ROOT = Path("/root/xunlian/Futaoran/CSI_MAIN_C591701_20260901")
RUN = ROOT / "runs/c591701-formal-prepare-20260902T2040"
EXPECTED_PID = 6306


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    destination = HERE / "old_scene"
    destination.mkdir(exist_ok=False)
    launch = json.loads((OLD_SUPERVISION / "launch.json").read_text())
    assert launch["pid"] == EXPECTED_PID
    proc = Path(f"/proc/{EXPECTED_PID}")
    command = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    assert command == launch["command"]
    assert proc.joinpath("cwd").resolve() == ROOT / "code/CSI-PAIRS-v2.0-server"
    assert "formal_v2.formal_cli" in command and "all" in command
    config = proc.joinpath("cwd").resolve() / command[command.index("--config") + 1]
    dataset = Path(command[command.index("--dataset") + 1])
    expected_dataset = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
    assert sha(dataset) == expected_dataset
    index = json.loads((RUN / "factorial/checkpoint_index.json").read_text())
    checkpoints = []
    for row in index["checkpoints"]:
        path = RUN / "factorial" / row["path"]
        assert sha(path) == row["sha256"]
        checkpoints.append({"path": str(path), "sha256": row["sha256"], "seed": row["seed"], "arm": row["arm"]})
    snapshots = {}
    for relative in (
        "factorial/gate.json", "factorial/checkpoint_index.json", "factorial/resume_index.json",
        "factorial/localization_summary.csv", "factorial/normalization.json", "factorial/frozen_pilot.json",
        "qualification/gate.json", "qualification/manifest.json", "data_verification/gate.json",
        "evaluation/status.json", "evaluation_state/evaluation_progress.json", "approval/accepted.json",
    ):
        source = RUN / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            snapshots[relative] = sha(target)
    for name in ("launch.json", "all.log", "progress.json", "progress_5min.jsonl"):
        shutil.copy2(OLD_SUPERVISION / name, destination / name)
    process_table = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    parents = {int(pid): int(ppid) for pid, ppid in (line.split() for line in process_table.splitlines())}
    descendants = {EXPECTED_PID}
    while True:
        expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    samples = []
    for _ in range(3):
        samples.append({
            "utc": datetime.now(timezone.utc).isoformat(),
            "processes": subprocess.check_output(["ps", "-p", ",".join(map(str, sorted(descendants))), "-o", "pid,ppid,lstart,etime,time,stat,pcpu,rss"], text=True),
            "threads": subprocess.check_output(["ps", "-L", "-p", str(EXPECTED_PID), "-o", "tid,stat,pcpu,time,wchan:30"], text=True),
            "io": proc.joinpath("io").read_text(),
            "status": proc.joinpath("status").read_text(),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,utilization.gpu,memory.used,memory.total", "--format=csv"], text=True),
        })
        time.sleep(2)
    env = dict(entry.split("=", 1) for entry in proc.joinpath("environ").read_text().split("\0") if "=" in entry)
    record = {
        "utc": datetime.now(timezone.utc).isoformat(), "pid": EXPECTED_PID,
        "proc_start_ticks": proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19],
        "descendant_pids": sorted(descendants - {EXPECTED_PID}),
        "command": command, "cwd": str(proc.joinpath("cwd").resolve()),
        "python_executable": str(proc.joinpath("exe").resolve()),
        "environment": {k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CSI_PAIRS_DEVICES", "CUBLAS_WORKSPACE_CONFIG")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
        "config_path": str(config), "config_raw_sha256": sha(config),
        "dataset_path": str(dataset), "dataset_sha256": expected_dataset,
        "checkpoints": checkpoints, "snapshots": snapshots, "samples": samples,
        "sampling_scope": "Three procfs/process/resource observations over six seconds; no debugger attachment or signal. Does not reconstruct historical function timings.",
        "old_task_action": "PRESERVED_RUNNING",
        "reuse": "Completed upstream weights, qualification and factorial results retain original identity. No completed streaming probe bundle or optimizer state exists in captured evaluation_state; unfinished probe fitting cannot resume at its former step.",
    }
    path = destination / "scene.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"snapshot": str(path), "sha256": sha(path), "checkpoints_verified": len(checkpoints), "old_task_action": record["old_task_action"]}))


if __name__ == "__main__":
    main()
