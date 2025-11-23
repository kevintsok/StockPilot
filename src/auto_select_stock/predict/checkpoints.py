from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import nn
    from .train import TrainConfig


def load_training_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    return {
        "model_state": payload.get("model_state"),
        "optimizer_state": payload.get("optimizer_state"),
        "scaler": payload.get("scaler"),
        "config": payload.get("config"),
        "metrics": payload.get("metrics", {}),
        "epoch": payload.get("epoch", 0),
        "global_step": payload.get("global_step", 0),
        "samples_seen": payload.get("samples_seen", 0),
    }


def prune_old_checkpoints(directory: Path, experiment_name: str, keep: int = 3) -> None:
    """
    Keep only the latest `keep` checkpoints matching experiment_name-e*-s*.pt.
    """
    candidates = sorted(
        directory.glob(f"{experiment_name}-e*-s*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for obsolete in candidates[keep:]:
        try:
            obsolete.unlink()
            print(f"[Checkpoint] Removed old checkpoint {obsolete}")
        except FileNotFoundError:
            continue


def save_checkpoint(
    model: "nn.Module",
    cfg: "TrainConfig",
    metrics: Dict[str, float],
    scaler: Optional[Dict[str, np.ndarray]] = None,
    optimizer_state: Optional[Dict[str, object]] = None,
    path: Optional[Path] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    samples_seen: Optional[int] = None,
) -> Path:
    target = path or cfg.save_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": cfg,
        "scaler": scaler,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
    }
    torch.save(payload, target)
    print(f"Saved checkpoint to {target}")

    if path is not None and target.name.startswith(f"{cfg.experiment_name}-"):
        prune_old_checkpoints(target.parent, cfg.experiment_name, keep=3)
    return target
