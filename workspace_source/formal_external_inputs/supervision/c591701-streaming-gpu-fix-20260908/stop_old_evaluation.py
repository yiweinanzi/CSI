import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    record_path = here / "old_task_stop.json"
    if record_path.exists():
        raise RuntimeError("stop attempt already recorded")
    saved = json.loads((here / "old_scene/scene.json").read_text())
    pid = saved["pid"]
    proc = Path(f"/proc/{pid}")
    command = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    ticks = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
    assert pid == 6306 and command == saved["command"]
    assert ticks == saved["proc_start_ticks"]
    assert str(proc.joinpath("cwd").resolve()) == saved["cwd"]
    progress = Path(command[command.index("--output") + 1]) / "evaluation_state/evaluation_progress.json"
    observed = json.loads(progress.read_text())
    assert observed["status"] == "RUNNING" and observed["current_substage"] == "probe_state"
    record = {
        "utc": datetime.now(timezone.utc).isoformat(), "pid": pid,
        "proc_start_ticks": ticks, "verified_command": command,
        "signal": "SIGTERM", "reason": "User-authorized evaluation-only execution repair after scene preservation and A/B validation",
        "last_progress": observed,
        "recovery_boundary": "Completed upstream artifacts retained; unsaved probe fitting has no optimizer checkpoint and must restart.",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    os.kill(pid, signal.SIGTERM)
    print(json.dumps({"signal_sent_to_verified_evaluation_pid": pid, "record": str(record_path)}))


if __name__ == "__main__":
    main()
