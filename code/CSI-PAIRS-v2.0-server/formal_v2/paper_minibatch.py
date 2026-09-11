"""New probe protocol: one minibatch per optimizer update, source-only selection."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .formal_probes import CompatibilityProbe, ActionResponseProbe
from .paper_probe_resume import ProbeResume


def fit_candidate(x, y, zero, *, family, hidden, seed, steps, batch_rows, learning_rate, device, checkpoint=None, callback=None):
    if steps < 1 or batch_rows < 1 or len(x) != len(y) or not len(x):
        raise ValueError("Invalid minibatch probe dimensions or schedule")
    torch.manual_seed(seed)
    binary = family in {"linear", "mlp2"}
    model = (CompatibilityProbe(x.shape[1], family, hidden) if binary else ActionResponseProbe(x.shape[1], y.shape[1], hidden)).to(device)
    model.zero_preserving = zero is not None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    recovery = ProbeResume(checkpoint, model, optimizer, (x, y, zero), steps, batch_rows, interval=25,
                           extra_identity={"protocol": "minibatch-updates-v1", "sampler_seed": seed})
    batches = math.ceil(len(x) / batch_rows)
    previous_epoch, order = None, None
    model.train()
    for step in range(recovery.start_step, steps):
        epoch, batch = divmod(step, batches)
        if epoch != previous_epoch:
            order = np.random.default_rng(seed + epoch).permutation(len(x))
            previous_epoch = epoch
        indices = order[batch * batch_rows:(batch + 1) * batch_rows]
        inputs = torch.as_tensor(np.asarray(x[indices]), dtype=torch.float32, device=device)
        targets = torch.as_tensor(np.asarray(y[indices]), dtype=torch.float32, device=device)
        prediction = model(inputs)
        if zero is not None:
            prediction = prediction - model(torch.as_tensor(np.asarray(zero[indices]), dtype=torch.float32, device=device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets) if binary else torch.mean((prediction - targets) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Minibatch probe loss is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        recovery.save(step + 1)
        if callback:
            callback({"step": step + 1, "total_steps": steps, "batch_rows": len(indices), "loss": float(loss.detach())})
    model.eval()
    return model


def source_score(model, x, y, zero, *, binary, batch_rows=1024):
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(x), batch_rows):
            stop = min(start + batch_rows, len(x))
            values = torch.as_tensor(np.array(x[start:stop]), dtype=torch.float32, device=device)
            targets = torch.as_tensor(np.array(y[start:stop]), dtype=torch.float32, device=device)
            prediction = model(values)
            if zero is not None:
                prediction -= model(torch.as_tensor(np.array(zero[start:stop]), dtype=torch.float32, device=device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, targets, reduction="sum") if binary else torch.sum((prediction - targets) ** 2)
            total += float(loss)
            count += targets.numel()
    result = total / count
    if not math.isfinite(result):
        raise FloatingPointError("Source selection metric is nonfinite")
    return result


def select_probe(train, selection, *, binary, profile, config, seed, device, output, callback=None):
    # Both roles are chosen by the caller before accessing target scenes. Every
    # arm uses this same candidate grid, sampler and source validation objective.
    best, best_score, records = None, float("inf"), []
    for family_index, family in enumerate(("linear", "mlp2") if binary else ("response",)):
        for steps in profile["probe_updates"]:
            model = fit_candidate(*train, family=family, hidden=int(config["evaluation"]["probe_hidden_dim"]),
                seed=seed + family_index, steps=steps, batch_rows=profile["batch_rows"],
                learning_rate=float(config["evaluation"]["probe_learning_rate"]), device=device,
                checkpoint=Path(output) / f"{family}-{steps}.pt", callback=callback)
            score = source_score(model, *selection, binary=binary, batch_rows=profile["batch_rows"])
            records.append({"family": family, "updates": steps, "source_selection_loss": score})
            if score < best_score:
                best, best_score = model, score
            else:
                del model
    return best, {"selection_role": "source_probe_selection", "criterion": "binary_nll" if binary else "delta_patch_mse", "best_source_loss": best_score, "candidates": records}
