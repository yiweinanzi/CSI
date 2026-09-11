"""Small optimizer checkpoint shared by source-trained map baselines."""
from pathlib import Path
import torch
from .formal_training_resume import _atomic_torch_save


def restore(output, model, optimizer, scheduler):
    path = Path(output) / "training_resume.pt"
    if not path.exists():
        return {"step": 0, "best_state": None, "best_selection": float("inf"), "best_step": 0}
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.pop("model"))
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    torch.set_rng_state(state.pop("torch_rng"))
    cuda_rng = state.pop("cuda_rng")
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return state


def save(output, model, optimizer, scheduler, **state):
    _atomic_torch_save(Path(output) / "training_resume.pt", {
        **state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })
