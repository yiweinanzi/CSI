from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from formal_v2.formal_model import (
    CSIPairsFormalModel,
    batch_for_module,
    portable_state_dict,
    resolve_execution_device,
)
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_teacher import teacher_targets, train_teacher_bundle


def _worker(rank: int, world_size: int, init_file: str, output: str) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = resolve_execution_device(
        SimpleNamespace(is_fixture=True), f"cuda:{rank}"
    )
    torch.cuda.set_device(device)
    distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    reduced = torch.tensor(float(rank + 1), device=device)
    distributed.all_reduce(reduced)

    spec = PatchSpec(
        antennas=2,
        subcarriers=4,
        patch_complex_size=1,
        patch_antenna_size=1,
        patch_subcarrier_size=1,
    )
    config = {
        "teacher": {
            "latent_dim": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "learning_rate": 0.001,
            "batch_size": 8,
            "steps": 1,
            "mask_fraction": 0.75,
        },
        "model": {"attention_heads": 2, "mask_bank_seed": 101},
    }
    csi = np.random.default_rng(1000 + rank).normal(
        size=(8, 2 * spec.complex_values)
    ).astype(np.float32)
    teacher = train_teacher_bundle(csi, spec, config, seed=2000 + rank, device=device)
    latent = teacher_targets(teacher, csi)

    model = CSIPairsFormalModel(
        patch_count=spec.patch_count,
        patch_rows=spec.patch_rows,
        patch_columns=spec.patch_columns,
        patch_dim=spec.patch_dim,
        map_channels=3,
        action_channels=8,
        radio_dim=11,
        latent_dim=8,
        state_dim=8,
        map_dim=8,
        hidden_dim=16,
        attention_heads=2,
        csi_encoder_layers=1,
    ).to(device)
    model.initialize_csi_from_teacher(teacher.teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch_size = 64
    generator = torch.Generator(device="cpu").manual_seed(3000 + rank)
    cpu_batch = {
        "visible": torch.randn(batch_size, spec.patch_count, spec.patch_dim, generator=generator),
        "maps": torch.randn(batch_size, 3, 8, 8, generator=generator),
        "radio": torch.randn(batch_size, 11, generator=generator),
        "masks": torch.zeros(batch_size, spec.patch_count, dtype=torch.bool),
        "action": torch.randn(batch_size, 8, 8, 8, generator=generator),
        "query": torch.arange(batch_size) % spec.patch_count,
    }
    batch = batch_for_module(model, cpu_batch)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            state = model.state(
                batch["visible"], batch["maps"], batch["radio"], batch["masks"]
            )
            prediction_z, prediction_y = model.predict(
                state, batch["action"], batch["query"]
            )
            loss = torch.mean(prediction_z**2) + torch.mean(prediction_y**2)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(3):
        step()
    torch.cuda.synchronize(device)
    timed_steps = 10
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(timed_steps):
        final_loss = step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    state = portable_state_dict(model)
    properties = torch.cuda.get_device_properties(device)
    result = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "name": properties.name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "all_reduce_sum": float(reduced),
        "teacher_device": str(next(teacher.teacher.parameters()).device),
        "teacher_latent_shape": list(latent.shape),
        "teacher_reconstruction_nmse": teacher.reconstruction_nmse,
        "final_loss": final_loss,
        "finite_loss": bool(np.isfinite(final_loss)),
        "portable_state_all_cpu": all(value.device.type == "cpu" for value in state.values()),
        "mixed_precision": "fp16_autocast",
        "batch_size": batch_size,
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "diagnostic_samples_per_second": batch_size * timed_steps / elapsed,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    Path(output, f"rank_{rank}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    distributed.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    init_file = output / "nccl_init"
    multiprocessing.spawn(
        _worker,
        args=(2, str(init_file.resolve()), str(output.resolve())),
        nprocs=2,
        join=True,
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(output.glob("rank_*.json"))]
    if len(rows) != 2 or any(row["all_reduce_sum"] != 3.0 for row in rows):
        raise RuntimeError("dual-GPU probe did not complete the expected NCCL reduction")
    if any(not row["finite_loss"] or not row["portable_state_all_cpu"] for row in rows):
        raise RuntimeError("dual-GPU model execution or checkpoint portability failed")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
