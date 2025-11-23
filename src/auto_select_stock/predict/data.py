import time
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import DATA_DIR, PREPROCESSED_DIR
from ..storage import load_financial, load_stock_history

PRICE_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "volume_ratio",
    "pct_change",
    "amplitude",
    "change_amount",
]
# These are forward-filled onto交易日，用于把最新财报指标拼到价格特征后面。
FINANCIAL_FEATURE_COLUMNS = [
    "roe",
    "net_profit_margin",
    "gross_margin",
    "operating_cashflow_growth",
    "debt_to_asset",
    "eps",
    "operating_cashflow_per_share",
]
DEFAULT_FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS + FINANCIAL_FEATURE_COLUMNS
TARGET_COLUMN = "close"
_PREPROCESS_VERSION = 1
_DATASET_CACHE_VERSION = 2


def close_index(feature_columns: Sequence[str]) -> int:
    try:
        return feature_columns.index(TARGET_COLUMN)
    except ValueError as exc:
        raise RuntimeError("target column missing from feature columns") from exc


class AutoregressivePriceDataset(Dataset):
    """
    Sliding-window dataset that builds autoregressive targets (predict each next-day close).
    """

    def __init__(self, sequences: Sequence[np.ndarray], seq_len: int, close_index: int):
        self.seq_len = seq_len
        self.close_index = close_index
        self.sequences: List[np.ndarray] = []
        self.sample_offsets: List[int] = []
        total = 0
        for data in sequences:
            arr = np.asarray(data, dtype="float32")
            if len(arr) <= seq_len:
                continue
            self.sequences.append(arr)
            total += len(arr) - seq_len
            self.sample_offsets.append(total)
        self.total_samples = total

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("sample index out of range")
        seq_idx = bisect_left(self.sample_offsets, idx + 1)
        start_offset = self.sample_offsets[seq_idx - 1] if seq_idx > 0 else 0
        start = idx - start_offset
        seq = self.sequences[seq_idx]
        window = seq[start : start + self.seq_len + 1]
        x = window[:-1]
        y = window[1:, self.close_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@dataclass
class _SymbolSplit:
    symbol: str
    length: int
    split: int
    train_count: int
    val_count: int


class StreamingPriceDataset(Dataset):
    """
    Map-style dataset that keeps only lightweight symbol offsets in memory and loads
    feature windows on demand to avoid materializing all sequences at once.
    """

    def __init__(
        self,
        splits: Sequence[_SymbolSplit],
        seq_len: int,
        stride: int,
        close_index: int,
        price_columns: List[str],
        financial_columns: List[str],
        base_dir: Path,
        cache_dir: Optional[Path],
        scaler: Dict[str, np.ndarray],
        mode: str = "train",
        preloaded_features: Optional[Dict[str, np.ndarray]] = None,
        source_mtime: Optional[float] = None,
        cache_size: int = 8,
    ):
        self.seq_len = seq_len
        self.stride = max(1, int(stride))
        self.close_index = close_index
        self.price_columns = price_columns
        self.financial_columns = financial_columns
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.scaler = scaler
        self.mode = mode
        self.source_mtime = source_mtime
        self._feature_cache = OrderedDict()  # simple LRU
        self._cache_size = cache_size
        self._preloaded = preloaded_features or {}
        self.sample_offsets: List[int] = []
        self._entries: List[_SymbolSplit] = []
        total = 0
        for item in splits:
            count = item.train_count if mode == "train" else item.val_count
            if count <= 0:
                continue
            total += count
            self.sample_offsets.append(total)
            self._entries.append(item)
        self.total_samples = total

    def __len__(self) -> int:
        return self.total_samples

    def _load_raw_features(self, symbol: str) -> np.ndarray:
        feats = self._preloaded.get(symbol)
        if feats is None:
            feats = load_cached_features(symbol, self.price_columns, self.financial_columns, self.cache_dir, self.source_mtime)
        if feats is None:
            feats = load_feature_matrix(symbol, self.price_columns, self.financial_columns, base_dir=self.base_dir)
        return feats

    def _get_scaled_features(self, symbol: str) -> np.ndarray:
        cached = self._feature_cache.get(symbol)
        if cached is not None:
            # LRU update
            self._feature_cache.move_to_end(symbol)
            return cached
        feats = self._load_raw_features(symbol)
        scaled = apply_scaler(feats, self.scaler)
        self._feature_cache[symbol] = scaled
        if len(self._feature_cache) > self._cache_size:
            self._feature_cache.popitem(last=False)
        return scaled

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("sample index out of range")
        entry_idx = bisect_left(self.sample_offsets, idx + 1)
        prev_total = self.sample_offsets[entry_idx - 1] if entry_idx > 0 else 0
        local_idx = idx - prev_total
        entry = self._entries[entry_idx]
        start = (
            local_idx * self.stride
            if self.mode == "train"
            else entry.split - self.seq_len + local_idx * self.stride
        )
        feats = self._get_scaled_features(entry.symbol)
        window = feats[start : start + self.seq_len + 1]
        x = window[:-1]
        y = window[1:, self.close_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _load_financial_frame(symbol: str, columns: List[str], base_dir: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Load financial indicators from SQLite and normalize the requested columns.

    Missing columns are backfilled with zeros so downstream feature stacking stays aligned.
    """
    try:
        df = load_financial(symbol, base_dir=base_dir)
    except FileNotFoundError:
        return None
    df = df.copy()
    if "date" not in df.columns:
        raise RuntimeError(f"财报数据缺少 date 列: {symbol}")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
    df.sort_values("date", inplace=True)
    for c in columns:
        if c not in df.columns:
            df[c] = 0.0
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    df = df[["date"] + columns]
    df = df.ffill()
    df[columns] = df[columns].fillna(0.0)
    return df


def _merge_price_financial(price_df: pd.DataFrame, fin_df: Optional[pd.DataFrame], fin_columns: List[str]) -> np.ndarray:
    if not fin_columns:
        return np.empty((len(price_df), 0), dtype="float32")
    if fin_df is None:
        return np.zeros((len(price_df), len(fin_columns)), dtype="float32")
    price_dates = price_df.copy()
    price_dates["date"] = pd.to_datetime(price_dates["date"]).dt.floor("D").astype("datetime64[ns]")
    fin_df = fin_df.copy()
    fin_df["date"] = pd.to_datetime(fin_df["date"]).dt.floor("D").astype("datetime64[ns]")
    merged = pd.merge_asof(
        price_dates[["date"]],
        fin_df[["date"] + fin_columns],
        on="date",
        direction="backward",
    )
    merged = merged.ffill()
    merged[fin_columns] = merged[fin_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return merged[fin_columns].astype("float32").to_numpy()


def load_feature_matrix(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    base_dir: Path = DATA_DIR,
) -> np.ndarray:
    arr = load_stock_history(symbol, base_dir=base_dir)
    df = pd.DataFrame(arr)
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
    df.sort_values("date", inplace=True)
    price_features = df[price_columns].astype("float32").to_numpy()
    fin_df = _load_financial_frame(symbol, financial_columns, base_dir=base_dir)
    fin_features = _merge_price_financial(df, fin_df, financial_columns)
    return np.concatenate([price_features, fin_features], axis=1)


def compute_scaler(train_features: Iterable[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute feature-wise mean/std in a streaming manner to avoid materializing a giant
    stacked array that can blow up memory on large universes.
    """
    count = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    for feats in train_features:
        arr = np.asarray(feats, dtype="float64")  # use float64 for stable statistics
        if arr.size == 0:
            continue
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)  # population variance
        if mean is None:
            mean = batch_mean
            m2 = batch_var * batch_count
            count = batch_count
            continue
        assert m2 is not None  # for type checker
        delta = batch_mean - mean
        new_count = count + batch_count
        mean = mean + delta * (batch_count / new_count)
        # Update aggregated squared differences (West/Welford)
        m2 = m2 + batch_var * batch_count + delta * delta * (count * batch_count / new_count)
        count = new_count
    if mean is None or m2 is None or count == 0:
        raise RuntimeError("No data to compute scaler.")
    std = np.sqrt(m2 / count) + 1e-6
    return {"mean": mean.astype("float32"), "std": std.astype("float32")}


def apply_scaler(data: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return ((data - scaler["mean"]) / scaler["std"]).astype("float32")


def _stock_db_mtime(base_dir: Path) -> Optional[float]:
    db_path = base_dir / "stock.db"
    if db_path.exists():
        return db_path.stat().st_mtime
    return None


def all_financial_columns(base_dir: Path = DATA_DIR) -> List[str]:
    """
    Return all financial columns present in the DB (excluding symbol/date).
    """
    db_path = base_dir / "stock.db"
    if not db_path.exists():
        return FINANCIAL_FEATURE_COLUMNS.copy()
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("PRAGMA table_info(financial)")
        cols = [row[1] for row in cur.fetchall() if row[1] not in {"symbol", "date"}]
    return cols or FINANCIAL_FEATURE_COLUMNS.copy()


def _feature_cache_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{symbol}.npz"


def load_cached_features(
    symbol: str,
    price_columns: List[str],
    financial_columns: List[str],
    cache_dir: Optional[Path],
    source_mtime: Optional[float],
) -> Optional[np.ndarray]:
    if cache_dir is None:
        return None
    path = _feature_cache_path(symbol, cache_dir)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            files = set(data.files)
            if "features" not in files:
                return None
            version = int(data["version"][0]) if "version" in files else 0
            if version != _PREPROCESS_VERSION:
                return None
            cached_price = [str(x) for x in data["price_columns"].tolist()] if "price_columns" in files else []
            cached_fin = [str(x) for x in data["financial_columns"].tolist()] if "financial_columns" in files else []
            if cached_price != price_columns or cached_fin != financial_columns:
                return None
            cached_mtime = float(data["source_mtime"][0]) if "source_mtime" in files else None
            if source_mtime is not None and cached_mtime is not None and cached_mtime < source_mtime:
                return None
            features = data["features"]
            return features.astype("float32")
    except Exception:  # noqa: BLE001
        return None


def write_cached_features(
    symbol: str,
    features: np.ndarray,
    price_columns: List[str],
    financial_columns: List[str],
    cache_dir: Optional[Path],
    source_mtime: Optional[float],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _feature_cache_path(symbol, cache_dir)
    np.savez_compressed(
        path,
        features=features.astype("float32"),
        price_columns=np.array(price_columns),
        financial_columns=np.array(financial_columns),
        version=np.array([_PREPROCESS_VERSION]),
        source_mtime=np.array([source_mtime if source_mtime is not None else -1.0]),
    )
    return path


def preprocess_symbol_features(
    symbols: Iterable[str],
    price_columns: List[str],
    financial_columns: Optional[List[str]],
    base_dir: Path = DATA_DIR,
    cache_dir: Optional[Path] = PREPROCESSED_DIR,
    keep_in_memory: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Precompute merged日报/财报特征并落盘，加速后续训练数据组装。

    When keep_in_memory=False, features are only written to cache on disk to reduce RAM
    footprint; the returned dict will be empty in that case.
    """
    cache_dir = cache_dir or PREPROCESSED_DIR
    fin_cols = financial_columns or all_financial_columns(base_dir)
    source_mtime = _stock_db_mtime(base_dir)
    prepared: Dict[str, np.ndarray] = {}
    symbols_list = list(symbols)
    cache_hits = 0
    built = 0
    skipped = 0
    lengths: List[int] = []
    start = time.perf_counter()
    for sym in symbols_list:
        feats = load_cached_features(sym, price_columns, fin_cols, cache_dir, source_mtime)
        if feats is None:
            try:
                feats = load_feature_matrix(sym, price_columns, fin_cols, base_dir=base_dir)
            except FileNotFoundError:
                skipped += 1
                continue
            built += 1
            write_cached_features(sym, feats, price_columns, fin_cols, cache_dir, source_mtime)
        else:
            cache_hits += 1
        if keep_in_memory:
            prepared[sym] = feats
        lengths.append(len(feats))
    elapsed = time.perf_counter() - start
    if lengths:
        import numpy as _np

        lengths_arr = _np.array(lengths)
        print(
            f"[Preprocess] symbols={len(symbols_list)} cached={cache_hits} built={built} skipped={skipped} "
            f"dir={cache_dir} took={elapsed:.2f}s "
            f"len_avg={lengths_arr.mean():.2f} len_min={lengths_arr.min()} len_max={lengths_arr.max()} "
            f"len_median={_np.median(lengths_arr):.2f}"
        )
    else:
        print(
            f"[Preprocess] symbols={len(symbols_list)} cached={cache_hits} built={built} skipped={skipped} "
            f"dir={cache_dir} took={elapsed:.2f}s"
        )
    return prepared


def prepare_datasets(
    symbols: Iterable[str],
    seq_len: int,
    stride: int = 1,
    price_columns: Optional[List[str]] = None,
    financial_columns: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    base_dir: Path = DATA_DIR,
    cache_dir: Optional[Path] = PREPROCESSED_DIR,
    preloaded_features: Optional[Dict[str, np.ndarray]] = None,
    existing_scaler: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[StreamingPriceDataset, StreamingPriceDataset, Dict[str, np.ndarray], List[str]]:
    price_cols = price_columns or PRICE_FEATURE_COLUMNS
    fin_cols = financial_columns or all_financial_columns(base_dir)
    feature_columns = price_cols + fin_cols
    close_idx = close_index(feature_columns)
    symbols_list = list(symbols)
    cache_dir = cache_dir or PREPROCESSED_DIR
    source_mtime = _stock_db_mtime(base_dir)
    stride = max(1, int(stride))

    def _load_features(sym: str) -> Optional[np.ndarray]:
        feats = preloaded_features.get(sym) if preloaded_features else None
        if feats is None:
            feats = load_cached_features(sym, price_cols, fin_cols, cache_dir, source_mtime)
        if feats is None:
            try:
                feats = load_feature_matrix(sym, price_cols, fin_cols, base_dir=base_dir)
            except FileNotFoundError:
                return None
            write_cached_features(sym, feats, price_cols, fin_cols, cache_dir, source_mtime)
        return feats

    # Streaming scaler computation to avoid stacking all features.
    count = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None
    splits: List[_SymbolSplit] = []
    for sym in symbols_list:
        feats = _load_features(sym)
        if feats is None or len(feats) <= seq_len + 1:
            continue
        split = int(len(feats) * train_ratio)
        split = max(split, seq_len + 1)
        split = min(split, len(feats) - 1)
        train_available = split - seq_len
        val_available = len(feats) - split
        train_count = (train_available + stride - 1) // stride
        val_count = (val_available + stride - 1) // stride
        splits.append(_SymbolSplit(symbol=sym, length=len(feats), split=split, train_count=train_count, val_count=val_count))
        if existing_scaler:
            continue
        arr = np.asarray(feats[:split], dtype="float64")
        if arr.size == 0:
            continue
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        if mean is None:
            mean = batch_mean
            m2 = batch_var * batch_count
            count = batch_count
            continue
        assert m2 is not None
        delta = batch_mean - mean
        new_count = count + batch_count
        mean = mean + delta * (batch_count / new_count)
        m2 = m2 + batch_var * batch_count + delta * delta * (count * batch_count / new_count)
        count = new_count

    if not splits:
        raise RuntimeError("No sufficient data to build datasets.")

    if existing_scaler:
        scaler = existing_scaler
    else:
        if mean is None or m2 is None or count == 0:
            raise RuntimeError("No data to compute scaler.")
        std = np.sqrt(m2 / count) + 1e-6
        scaler = {"mean": mean.astype("float32"), "std": std.astype("float32")}

    train_ds = StreamingPriceDataset(
        splits,
        seq_len=seq_len,
        stride=stride,
        close_index=close_idx,
        price_columns=price_cols,
        financial_columns=fin_cols,
        base_dir=base_dir,
        cache_dir=cache_dir,
        scaler=scaler,
        mode="train",
        preloaded_features=preloaded_features,
        source_mtime=source_mtime,
    )
    val_ds = StreamingPriceDataset(
        splits,
        seq_len=seq_len,
        stride=stride,
        close_index=close_idx,
        price_columns=price_cols,
        financial_columns=fin_cols,
        base_dir=base_dir,
        cache_dir=cache_dir,
        scaler=scaler,
        mode="val",
        preloaded_features=preloaded_features,
        source_mtime=source_mtime,
    )
    print(
        f"[Dataset] symbols={len(splits)} train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"features={len(feature_columns)} stride={stride}"
    )
    return train_ds, val_ds, scaler, feature_columns


def _dataset_cache_dir(cache_root: Optional[Path]) -> Path:
    root = cache_root or PREPROCESSED_DIR
    return root / "datasets"


def _dataset_cache_key(
    symbols: Iterable[str],
    price_columns: List[str],
    financial_columns: List[str],
    seq_len: int,
    train_ratio: float,
    source_mtime: Optional[float],
) -> str:
    import hashlib

    sym_key = ",".join(sorted(symbols))
    price_key = ",".join(price_columns)
    fin_key = ",".join(financial_columns)
    raw = f"{sym_key}|{price_key}|{fin_key}|{seq_len}|{train_ratio}|{source_mtime or -1}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cached_datasets(
    symbols: Iterable[str],
    seq_len: int,
    price_columns: List[str],
    financial_columns: List[str],
    train_ratio: float,
    base_dir: Path = DATA_DIR,
    cache_root: Optional[Path] = PREPROCESSED_DIR,
) -> Optional[Tuple[AutoregressivePriceDataset, AutoregressivePriceDataset, Dict[str, np.ndarray], List[str]]]:
    # Dataset caching is disabled to avoid large in-memory tensors during save/load.
    return None


def write_cached_datasets(
    train_sequences: List[np.ndarray],
    val_sequences: List[np.ndarray],
    scaler: Dict[str, np.ndarray],
    feature_columns: List[str],
    symbols: List[str],
    price_columns: List[str],
    financial_columns: List[str],
    seq_len: int,
    train_ratio: float,
    base_dir: Path = DATA_DIR,
    cache_root: Optional[Path] = PREPROCESSED_DIR,
    source_mtime: Optional[float] = None,
) -> Path:
    cache_dir = _dataset_cache_dir(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "disabled.pt"
