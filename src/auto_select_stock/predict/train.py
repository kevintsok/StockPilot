import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from ..config import DATA_DIR, MODEL_DIR, PREPROCESSED_DIR
from ..torch_model import PriceTransformer, TrainConfig
from .checkpoints import load_training_checkpoint, save_checkpoint
from .data import (
    FINANCIAL_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    all_financial_columns,
    prepare_datasets,
    preprocess_symbol_features,
)


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_last_mae = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            total_last_mae += torch.mean(torch.abs(pred[:, -1] - y[:, -1])).item()
            steps += 1
    if steps == 0:
        return float("inf"), float("inf")
    return total_loss / steps, total_last_mae / steps


def _progress_iterator(loader: DataLoader, epoch: int):
    """
    Wrap a DataLoader with tqdm when available so we can see per-epoch progress.
    """
    total = len(loader)
    if tqdm is None or total == 0:
        return loader, total
    bar = tqdm(loader, total=total, desc=f"Epoch {epoch}", leave=False)
    return bar, total


def _maybe_init_wandb(cfg: TrainConfig, feature_columns: List[str]):
    """
    Initialize a wandb run when配置了项目名；缺少依赖时静默跳过。
    """
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"Skip wandb logging: {exc}")
        return None
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        mode=cfg.wandb_mode,
        config={
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "train_ratio": cfg.train_ratio,
            "num_features": len(feature_columns),
            "price_columns": cfg.price_columns,
            "financial_columns": cfg.financial_columns,
        },
        reinit=True,
    )
    return run


@contextmanager
def _time_block(name: str):
    start = time.perf_counter()
    print(f"[Timing] {name} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[Timing] {name} finished in {elapsed:.2f}s")


def train_transformer(
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: TrainConfig,
    scaler: Optional[Dict[str, np.ndarray]] = None,
    resume_state: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    device = _device(cfg)
    feature_columns = cfg.price_columns + cfg.financial_columns
    try:
        model = PriceTransformer(input_dim=len(feature_columns)).to(device)
    except RuntimeError as exc:
        msg = str(exc)
        if device.type == "cuda" and ("cudaGetDeviceCount" in msg or "out of memory" in msg.lower()):
            # Gracefully fall back to CPU when CUDA cannot be initialized.
            print(f"[Device] CUDA init failed ({msg}); falling back to CPU")
            device = torch.device("cpu")
            model = PriceTransformer(input_dim=len(feature_columns)).to(device)
        else:
            raise
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    start_epoch = 1
    global_step = 0
    samples_seen = 0

    resume = resume_state
    if resume is None and cfg.resume_checkpoint:
        resume = load_training_checkpoint(cfg.resume_checkpoint, device)

    if resume:
        model_state = resume.get("model_state")
        if model_state:
            model.load_state_dict(model_state)  # type: ignore[arg-type]
        opt_state = resume.get("optimizer_state")
        if opt_state:
            optimizer.load_state_dict(opt_state)  # type: ignore[arg-type]
        resume_metrics = resume.get("metrics") if isinstance(resume, dict) else {}
        if isinstance(resume_metrics, dict):
            best_val = resume_metrics.get("val_loss", best_val)
        start_epoch = int(resume.get("epoch", 0)) + 1
        global_step = int(resume.get("global_step", 0))
        samples_seen = int(resume.get("samples_seen", 0))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    stats = {"best_val_loss": float("inf"), "best_val_last_mae": float("inf")}
    if resume and isinstance(resume.get("metrics"), dict):
        stats["best_val_loss"] = resume["metrics"].get("val_loss", stats["best_val_loss"])
        stats["best_val_last_mae"] = resume["metrics"].get("val_last_mae", stats["best_val_last_mae"])
    wandb_run = _maybe_init_wandb(cfg, feature_columns)

    train_start = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        steps = 0
        progress, total_batches = _progress_iterator(train_loader, epoch)
        for batch_idx, (x, y) in enumerate(progress, 1):
            batch_start = time.time()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            global_step += 1
            samples_seen += x.size(0)
            if tqdm is not None and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    {"batch": f"{batch_idx}/{total_batches}", "loss": f"{loss.item():.6f}"},
                    refresh=False,
                )
            elif total_batches:
                if batch_idx == total_batches or batch_idx % max(1, total_batches // 5) == 0:
                    print(
                        f"\r[Epoch {epoch}] progress {batch_idx}/{total_batches} loss={loss.item():.4f}",
                        end="",
                        flush=True,
                    )
            if wandb_run:
                batch_time = max(time.time() - batch_start, 1e-6)
                samples_per_sec = x.size(0) / batch_time
                wall_elapsed = time.time() - train_start
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/samples": samples_seen,
                        "train/samples_per_sec": samples_per_sec,
                        "time/elapsed_sec": wall_elapsed,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            if cfg.checkpoint_steps > 0 and global_step % cfg.checkpoint_steps == 0:
                ckpt_name = f"{cfg.experiment_name}-e{epoch}-s{global_step}.pt"
                ckpt_path = cfg.save_path.parent / ckpt_name
                save_checkpoint(
                    model,
                    cfg,
                    {"train_loss": loss.item(), "epoch": epoch, "global_step": global_step},
                    scaler=scaler,
                    optimizer_state=optimizer.state_dict(),
                    path=ckpt_path,
                    epoch=epoch,
                    global_step=global_step,
                    samples_seen=samples_seen,
                )

        train_loss = running_loss / max(1, steps)
        epoch_time = time.perf_counter() - epoch_start
        if tqdm is None and total_batches:
            print()
        if epoch % cfg.eval_every == 0:
            val_loss, val_last_mae = evaluate(model, val_loader, loss_fn, device)
            if val_loss < best_val:
                best_val = val_loss
                stats["best_val_loss"] = val_loss
                stats["best_val_last_mae"] = val_last_mae
                save_checkpoint(
                    model,
                    cfg,
                    {"val_loss": val_loss, "val_last_mae": val_last_mae},
                    scaler=scaler,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    global_step=global_step,
                    samples_seen=samples_seen,
                )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        "val/last_mae": val_last_mae,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_last_mae={val_last_mae:.6f} device={device.type} epoch_time={epoch_time:.2f}s"
            )
        else:
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f} device={device.type} epoch_time={epoch_time:.2f}s")

    return stats


def train_from_symbols(
    symbols: Iterable[str],
    cfg: TrainConfig,
    base_dir: Path = DATA_DIR,
) -> Dict[str, float]:
    symbols = list(symbols)
    price_cols = cfg.price_columns or PRICE_FEATURE_COLUMNS
    fin_cols = cfg.financial_columns or all_financial_columns(base_dir)
    cfg.price_columns = price_cols
    cfg.financial_columns = fin_cols
    keep_in_memory = getattr(cfg, "keep_preprocessed_in_memory", False)
    # Compute model parameter count early for visibility
    tmp_model = PriceTransformer(input_dim=len(price_cols) + len(fin_cols))
    param_count = sum(p.numel() for p in tmp_model.parameters()) / 1e6
    print(
        f"[Train] symbols={len(symbols)} seq_len={cfg.seq_len} batch_size={cfg.batch_size} "
        f"stride={cfg.window_stride} epochs={cfg.epochs} device={cfg.device or 'auto'} "
        f"price_cols={len(price_cols)} fin_cols={len(fin_cols)} params={param_count:.2f}M"
    )
    resume_state: Optional[Dict[str, object]] = None
    resume_scaler: Optional[Dict[str, np.ndarray]] = None
    if cfg.resume_checkpoint:
        print(f"[Resume] loading checkpoint {cfg.resume_checkpoint}")
        resume_state = load_training_checkpoint(cfg.resume_checkpoint, _device(cfg))
        resume_cfg = resume_state.get("config")
        if isinstance(resume_cfg, TrainConfig):
            cfg.price_columns = resume_cfg.price_columns or price_cols
            cfg.financial_columns = resume_cfg.financial_columns or fin_cols
            cfg.seq_len = resume_cfg.seq_len
            cfg.window_stride = getattr(resume_cfg, "window_stride", cfg.window_stride)
        resume_scaler = resume_state.get("scaler")  # type: ignore[assignment]

    cache_dir = getattr(cfg, "preprocessed_dir", PREPROCESSED_DIR)
    with _time_block("Preprocess features"):
        preprocessed = preprocess_symbol_features(
            symbols=symbols,
            price_columns=cfg.price_columns,
            financial_columns=cfg.financial_columns,
            base_dir=base_dir,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
        )
    with _time_block("Prepare datasets"):
        train_ds, val_ds, scaler, feature_columns = prepare_datasets(
            symbols=symbols,
            seq_len=cfg.seq_len,
            price_columns=cfg.price_columns,
            financial_columns=cfg.financial_columns,
            train_ratio=cfg.train_ratio,
            stride=cfg.window_stride,
            base_dir=base_dir,
            cache_dir=cache_dir,
            preloaded_features=preprocessed,
            existing_scaler=resume_scaler,
        )
    with _time_block("Training loop"):
        stats = train_transformer(train_ds, val_ds, cfg, scaler=scaler, resume_state=resume_state)
    return stats
