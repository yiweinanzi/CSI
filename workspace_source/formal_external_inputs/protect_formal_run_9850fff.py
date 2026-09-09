#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import time
from pathlib import Path


LOG = Path("/tmp/csi-formal-9850fff-protection.log")
SMOKE_PRIORITY_GUARD = "/tmp/csi_smoke_priority_guard.py"
FORMAL_DATASET_GUARD = "/tmp/csi_formal_dataset_guard.py"
AUTHORIZED_SMOKE_OUTPUTS = {
    (
        "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/"
        "CSI-PAIRS-v2.0-server/runs/"
        "smoke-full-v6-2gpu-p16-20260825T155907Z"
    ),
    "runs/smoke-full-v6-2gpu-p16-20260825T155907Z",
    (
        "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/"
        "CSI-PAIRS-v2.0-server/runs/"
        "smoke-full-v6-2gpu-p16-20260825T202753Z"
    ),
    "runs/smoke-full-v6-2gpu-p16-20260825T202753Z",
    (
        "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/"
        "CSI-PAIRS-v2.0-server/runs/"
        "smoke-full-v6-p8-claimsfix-20260826T205334Z"
    ),
    "runs/smoke-full-v6-p8-claimsfix-20260826T205334Z",
    (
        "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/"
        "CSI-PAIRS-v2.0-server/runs/"
        "smoke-full-v6-p8-routefix-20260826T215954Z"
    ),
    "runs/smoke-full-v6-p8-routefix-20260826T215954Z",
    (
        "/root/xunlian/Futaoran/CSI_CLOUD_LATEST_3183664/code/"
        "CSI-PAIRS-v2.0-server/runs/"
        "smoke-full-v6-p8-clusterfix-20260826T230312Z"
    ),
    "runs/smoke-full-v6-p8-clusterfix-20260826T230312Z",
}


def argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def blocked_process(arguments: list[str]) -> bool:
    if SMOKE_PRIORITY_GUARD in arguments or FORMAL_DATASET_GUARD in arguments:
        return True
    runs_fixture = (
        "formal_v2.formal_cli" in arguments
        and "--allow-nonscientific-fixture" in arguments
    )
    return runs_fixture and not AUTHORIZED_SMOKE_OUTPUTS.intersection(arguments)


def log(message: str) -> None:
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def main() -> None:
    own_pid = os.getpid()
    own_group = os.getpgrp()
    log(f"started pid={own_pid} pgid={own_group}")
    while True:
        groups: dict[int, tuple[int, list[str]]] = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == own_pid:
                continue
            arguments = argv(pid)
            if not arguments or not blocked_process(arguments):
                continue
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if group != own_group:
                groups.setdefault(group, (pid, arguments))
        for group, (pid, arguments) in groups.items():
            try:
                os.killpg(group, signal.SIGTERM)
                log(
                    f"stopped pgid={group} sample_pid={pid} "
                    f"argv={arguments[:20]!r}"
                )
            except ProcessLookupError:
                pass
        time.sleep(0.25)


if __name__ == "__main__":
    main()
