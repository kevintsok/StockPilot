import time
import dataclasses
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

from ..config import DATA_DIR, MODEL_DIR, PREPROCESSED_DIR, REPORT_DIR
from ..torch_model import PriceTransformer, TrainConfig
from .checkpoints import load_training_checkpoint, save_checkpoint
from .data import (
    FINANCIAL_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    all_financial_columns,
    prepare_date_window_datasets,
    prepare_datasets,
    preprocess_symbol_features,
)


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: nn.Module, loader: DataLoader, loss_fn_reg, loss_fn_cls, cfg: TrainConfig, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    total_last_mae = 0.0
    steps = 0
    with torch.no_grad():
        for x, y_reg, y_cls in loader:
            x = x.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)
            pred_reg, pred_cls = model(x)
            reg_loss = loss_fn_reg(pred_reg, y_reg)
            cls_loss = loss_fn_cls(pred_cls, y_cls)
            loss = cfg.lambda_reg * reg_loss + cfg.lambda_cls * cls_loss
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            total_last_mae += torch.mean(torch.abs(pred_reg[:, -1] - y_reg[:, -1])).item()
            steps += 1
    if steps == 0:
        return float("inf"), float("inf"), float("inf")
    return total_loss / steps, total_last_mae / steps, total_reg_loss / steps


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
            "date_windows": getattr(cfg, "date_windows", []),
            "num_features": len(feature_columns),
            "price_columns": cfg.price_columns,
            "financial_columns": cfg.financial_columns,
        },
        reinit=True,
    )
    return run


def _parse_date_window_strings(raw_windows: Iterable[str]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw in raw_windows:
        if not raw:
            continue
        parts = raw.replace(",", ":").split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid date window '{raw}', expected format TRAIN_END:VAL_END")
        train_end = pd.to_datetime(parts[0]).normalize()
        val_end = pd.to_datetime(parts[1]).normalize()
        if pd.isna(train_end) or pd.isna(val_end):
            raise ValueError(f"Invalid date window '{raw}', could not parse into dates")
        if val_end <= train_end:
            raise ValueError(f"Invalid date window '{raw}', val_end must be after train_end")
        windows.append((train_end, val_end))
    return windows


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
    test_ds: Optional[Dataset] = None,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=getattr(cfg, "weight_decay", 0.0))
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
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
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) if test_ds else None

    stats = {"best_val_loss": float("inf"), "best_val_last_mae": float("inf"), "best_val_reg_loss": float("inf")}
    if resume and isinstance(resume.get("metrics"), dict):
        stats["best_val_loss"] = resume["metrics"].get("val_loss", stats["best_val_loss"])
        stats["best_val_last_mae"] = resume["metrics"].get("val_last_mae", stats["best_val_last_mae"])
    wandb_run = _maybe_init_wandb(cfg, feature_columns)

    profile_enabled = bool(getattr(cfg, "profile", False))
    profile_name = f"{cfg.experiment_name}-seq{cfg.seq_len}-bs{cfg.batch_size}-step2-profile.json"
    profile_trace_path = REPORT_DIR / profile_name

    train_start = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        steps = 0
        progress, total_batches = _progress_iterator(train_loader, epoch)
        for batch_idx, (x, y_reg, y_cls) in enumerate(progress, 1):
            batch_start = time.time()
            x = x.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            profile_this_step = profile_enabled and batch_idx == 2
            profile_ctx = nullcontext()
            prof = None
            if profile_this_step:
                try:
                    profile_trace_path.parent.mkdir(parents=True, exist_ok=True)
                    profile_ctx = torch.profiler.profile(
                        activities= [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(REPORT_DIR)),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_modules=True
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[Profile] Unable to start profiler: {exc}")
                    profile_this_step = False
                    profile_ctx = nullcontext()

            with profile_ctx as prof:
                optimizer.zero_grad()
                pred_reg, pred_cls = model(x)
                reg_loss = loss_fn_reg(pred_reg, y_reg)
                cls_loss = loss_fn_cls(pred_cls, y_cls)
                loss = cfg.lambda_reg * reg_loss + cfg.lambda_cls * cls_loss
                loss.backward()
                if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()

            running_loss += loss.item()
            steps += 1
            global_step += 1
            samples_seen += x.size(0)
            if tqdm is not None and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    {
                        "batch": f"{batch_idx}/{total_batches}",
                        "loss": f"{loss.item():.6f}",
                        "reg": f"{reg_loss.item():.6f}",
                        "cls": f"{cls_loss.item():.6f}",
                    },
                    refresh=False,
                )
            elif total_batches:
                if batch_idx == total_batches or batch_idx % max(1, total_batches // 5) == 0:
                    print(
                        f"\r[Epoch {epoch}] progress {batch_idx}/{total_batches} loss={loss.item():.4f} "
                        f"reg={reg_loss.item():.4f} cls={cls_loss.item():.4f}",
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
                        "train/reg_loss": reg_loss.item(),
                        "train/cls_loss": cls_loss.item(),
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
            val_loss, val_last_mae, val_reg_loss = evaluate(model, val_loader, loss_fn_reg, loss_fn_cls, cfg, device)
            if val_loss < best_val:
                best_val = val_loss
                stats["best_val_loss"] = val_loss
                stats["best_val_last_mae"] = val_last_mae
                stats["best_val_reg_loss"] = val_reg_loss
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
                            "train/epoch_reg_loss": running_loss / max(1, steps),  # same as train_loss components summed
                            "val/loss": val_loss,
                            "val/reg_loss": val_reg_loss,
                            "val/last_mae": val_last_mae,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_reg={val_reg_loss:.6f} val_last_mae={val_last_mae:.6f} "
                f"device={device.type} epoch_time={epoch_time:.2f}s"
            )
        else:
            print(f"[Epoch {epoch}] train_loss={train_loss:.6f} device={device.type} epoch_time={epoch_time:.2f}s")

    if test_loader:
        test_loss, test_last_mae, test_reg_loss = evaluate(model, test_loader, loss_fn_reg, loss_fn_cls, cfg, device)
        stats["test_loss"] = test_loss
        stats["test_last_mae"] = test_last_mae
        stats["test_reg_loss"] = test_reg_loss
        print(
            f"[Test] loss={test_loss:.6f} reg_loss={test_reg_loss:.6f} last_mae={test_last_mae:.6f} samples={len(test_ds)}"
        )

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
    parsed_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    # 默认窗口：只训练窗口1（最早→2022-01-01 训练，2022-2023 验证，之后测试）。
    default_windows = ["2022-01-01:2023-01-01"]
    raw_windows = getattr(cfg, "date_windows", []) or default_windows
    if raw_windows:
        parsed_windows = _parse_date_window_strings(raw_windows)
        window_text = ", ".join([f"{tr.date()}->{val.date()}" for tr, val in parsed_windows])
        print(f"[Train] using date windows: {window_text}")
    for attr, default in [("target_mode", "log_return"), ("weight_decay", 1e-4), ("grad_clip", 1.0)]:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)
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
            cfg.target_mode = getattr(resume_cfg, "target_mode", cfg.target_mode)
            cfg.weight_decay = getattr(resume_cfg, "weight_decay", cfg.weight_decay)
            cfg.grad_clip = getattr(resume_cfg, "grad_clip", cfg.grad_clip)
        resume_scaler = resume_state.get("scaler")  # type: ignore[assignment]
    scaler_for_windows = resume_scaler if len(parsed_windows) <= 1 else None
    if resume_scaler is not None and len(parsed_windows) > 1:
        print("[Train] Multiple date windows detected; ignoring resume scaler to recompute per window.")

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
        if parsed_windows:
            window_datasets = prepare_date_window_datasets(
                symbols=symbols,
                seq_len=cfg.seq_len,
                price_columns=cfg.price_columns,
                financial_columns=cfg.financial_columns,
                date_windows=parsed_windows,
                stride=cfg.window_stride,
                base_dir=base_dir,
                cache_dir=cache_dir,
                preloaded_features=preprocessed,
                existing_scaler=scaler_for_windows,
                target_mode=getattr(cfg, "target_mode", "log_return"),
            )
        else:
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
                target_mode=getattr(cfg, "target_mode", "log_return"),
            )
    if parsed_windows:
        results: Dict[str, Dict[str, float]] = {}
        base_save = cfg.save_path
        for idx, window in enumerate(window_datasets, 1):
            train_tag = window.train_end.strftime("%Y%m%d")
            val_tag = window.val_end.strftime("%Y%m%d")
            window_cfg = dataclasses.replace(cfg)
            window_cfg.experiment_name = f"{cfg.experiment_name}-train{train_tag}-val{val_tag}"
            window_cfg.save_path = base_save.with_name(f"{base_save.stem}-train{train_tag}-val{val_tag}{base_save.suffix}")
            window_resume = resume_state if len(parsed_windows) == 1 else None
            with _time_block(f"Training loop ({window_cfg.experiment_name})"):
                stats = train_transformer(
                    window.train_ds,
                    window.val_ds,
                    window_cfg,
                    scaler=window.scaler,
                    resume_state=window_resume,
                    test_ds=window.test_ds,
                )
            stats["checkpoint"] = window_cfg.save_path
            results[window.name] = stats
        return results

    with _time_block("Training loop"):
        stats = train_transformer(train_ds, val_ds, cfg, scaler=scaler, resume_state=resume_state)
    stats["checkpoint"] = cfg.save_path
    return stats
