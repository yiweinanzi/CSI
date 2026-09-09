#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_inventory() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                rows.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "uuid": fields[2],
                        "memory_total_mib": int(fields[3]),
                        "driver_version": fields[4],
                    }
                )
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": rows,
        "stderr": completed.stderr.strip(),
    }


def _torch_inventory() -> dict:
    try:
        import torch
    except Exception as error:
        return {"imported": False, "error": f"{type(error).__name__}: {error}"}

    record = {
        "imported": True,
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(torch.cuda.nccl.version()) if hasattr(torch.cuda, "nccl") else None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "devices": [],
        "initialization_error": None,
    }
    try:
        record["cuda_available"] = bool(torch.cuda.is_available())
        record["cuda_device_count"] = int(torch.cuda.device_count())
        if record["cuda_available"]:
            for index in range(record["cuda_device_count"]):
                properties = torch.cuda.get_device_properties(index)
                total_memory = int(properties.total_memory)
                memory_source = "device_properties"
                if total_memory <= 0:
                    _, total_memory = torch.cuda.mem_get_info(index)
                    total_memory = int(total_memory)
                    memory_source = "mem_get_info_fallback"
                record["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": total_memory,
                        "total_memory_source": memory_source,
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                )
            left = torch.ones(8, device="cuda:0", dtype=torch.float16)
            right = torch.ones(8, device="cuda:0", dtype=torch.float16)
            record["fp16_cuda_sum"] = float((left + right).sum().cpu())
    except Exception as error:
        record["initialization_error"] = f"{type(error).__name__}: {error}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime probe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "csi-pairs-pre-run-runtime-probe-v1",
        "label": args.label,
        "command": [sys.executable, *sys.argv],
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "prefix": sys.prefix,
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
        },
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "packages": {
            name: _distribution_version(name)
            for name in ("numpy", "pytest", "scipy", "torch")
        },
        "nvidia_smi": _nvidia_inventory(),
        "torch": _torch_inventory(),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
