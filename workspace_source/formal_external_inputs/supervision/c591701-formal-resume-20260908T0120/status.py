"""Read-only view of the current attempt; never launches or changes experiments."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {"read_warning": str(error)}


def main():
    launch = read(HERE / "launch.json")
    command = launch["command"]
    run = Path(command[command.index("--output") + 1])
    launched_at = datetime.fromisoformat(launch["utc"]).timestamp()
    progress = read(HERE / "progress.json")
    print("Observed UTC:", datetime.now(timezone.utc).isoformat())
    print("Run:", run)
    print("Latest supervisor observation:", progress.get("utc", "unavailable"))
    subprocess.run([
        "ps", "-p", str(launch["supervisor_pid"]) + "," + str(launch["pid"]),
        "-o", "pid,ppid,stat,etime,time,pcpu,rss",
    ], check=False)
    subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv",
    ], check=False)
    stages = {
        "LIVE verification": "data_verification/gate.json",
        "qualification": "qualification/gate.json",
        "approval": "approval/accepted.json",
        "wrong-map diagnostic": "wrong_map/status.json",
        "factorial training summary": "factorial/training_summary.csv",
        "factorial localization gate": "factorial/gate.json",
        "streaming evaluation": "evaluation/gate.json",
        "risk": "risk/gate.json",
        "path": "path/gate.json",
        "external baselines": "external_baselines/gate.json",
        "representation baselines": "representation_baselines/gate.json",
        "resource controls": "controls/gate.json",
        "scene-ID": "scene_id/gate.json",
        "shuffled pair": "controls/shuffled_pair/gate.json",
        "retention": "evaluation/retention/gate.json",
        "claims": "claims/claim_evidence.json",
    }
    for name, relative in stages.items():
        path = run / relative
        if not path.is_file():
            print(f"{name}: NOT_WRITTEN")
            continue
        modified = path.stat().st_mtime
        freshness = "THIS_ATTEMPT" if modified >= launched_at else "PREEXISTING"
        status = read(path).get("status", "PRESENT") if path.suffix == ".json" else "PRESENT"
        stamp = datetime.fromtimestamp(modified, timezone.utc).isoformat()
        print(f"{name}: {status} [{freshness}; modified {stamp}]")
    claims = read(run / "claims/claim_evidence.json")
    print("Reported sota_ready (not independently revalidated):", claims.get("sota_ready", "NO_EVIDENCE"))
    print("Operation lock:", read(run.parent / ("." + run.name + ".csi-pairs-operation.lock")))
    if (HERE / "exit.json").is_file():
        print("Attempt exit:", read(HERE / "exit.json").get("returncode"))
    if (run / "stage_failures.json").is_file():
        print("Stage failures:", read(run / "stage_failures.json"))
    print("File presence is not integrity verification or a scientific PASS.")


if __name__ == "__main__":
    main()
