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
from ..financial_dates import infer_publish_dates
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
_PREPROCESS_VERSION = 2
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

    def __init__(self, sequences: Sequence[np.ndarray], seq_len: int, close_index: int, target_mode: str = "log_return"):
        self.seq_len = seq_len
        self.close_index = close_index
        self.target_mode = target_mode
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
        close = window[:, self.close_index]
        if self.target_mode == "log_return":
            ret = np.log(close[1:] / np.clip(close[:-1], 1e-6, None))
            y_reg = ret
        else:
            y_reg = window[1:, self.close_index]
        y_cls = (close[1:] > close[:-1]).astype("float32")
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.float32),
        )


@dataclass
class _SymbolSplit:
    symbol: str
    length: int
    val_start: int
    train_count: int
    val_count: int
    test_start: Optional[int] = None
    test_count: int = 0


@dataclass
class _DatasetEntry:
    symbol: str
    start: int
    count: int


@dataclass
class DatasetWindow:
    name: str
    train_ds: Dataset
    val_ds: Dataset
    test_ds: Optional[Dataset]
    scaler: Dict[str, np.ndarray]
    feature_columns: List[str]
    train_end: pd.Timestamp
    val_end: pd.Timestamp


@dataclass
class _DateSplit:
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    test_start_idx: int
    train_count: int
    val_count: int
    test_count: int


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
        target_mode: str = "log_return",
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
        self._raw_cache = OrderedDict()
        self._cache_size = cache_size
        self._preloaded = preloaded_features or {}
        self.target_mode = target_mode
        self.sample_offsets: List[int] = []
        self._entries: List[_DatasetEntry] = []
        total = 0
        for item in splits:
            if mode == "train":
                count = item.train_count
                start_base = 0
            elif mode == "test":
                if item.test_start is None:
                    continue
                count = item.test_count
                start_base = item.test_start - self.seq_len
            else:
                count = item.val_count
                start_base = item.val_start - self.seq_len
            if start_base < 0:
                continue
            if count <= 0:
                continue
            total += count
            self.sample_offsets.append(total)
            self._entries.append(_DatasetEntry(symbol=item.symbol, start=start_base, count=count))
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

    def _get_features(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        raw = self._raw_cache.get(symbol)
        if raw is None:
            raw = self._load_raw_features(symbol)
            self._raw_cache[symbol] = raw
            if len(self._raw_cache) > self._cache_size:
                self._raw_cache.popitem(last=False)
        scaled = self._feature_cache.get(symbol)
        if scaled is None:
            scaled = apply_scaler(raw, self.scaler)
            self._feature_cache[symbol] = scaled
            if len(self._feature_cache) > self._cache_size:
                self._feature_cache.popitem(last=False)
        else:
            self._feature_cache.move_to_end(symbol)
        return raw, scaled

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("sample index out of range")
        entry_idx = bisect_left(self.sample_offsets, idx + 1)
        prev_total = self.sample_offsets[entry_idx - 1] if entry_idx > 0 else 0
        local_idx = idx - prev_total
        entry = self._entries[entry_idx]
        start = entry.start + local_idx * self.stride
        raw_feats, feats = self._get_features(entry.symbol)
        window = feats[start : start + self.seq_len + 1]
        raw_window = raw_feats[start : start + self.seq_len + 1]
        x = window[:-1]
        close = raw_window[:, self.close_index]
        if self.target_mode == "log_return":
            ret = np.log(close[1:] / np.clip(close[:-1], 1e-6, None))
            y_reg = ret
        else:
            y_reg = window[1:, self.close_index]
        y_cls = (close[1:] > close[:-1]).astype("float32")
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.float32),
        )


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
    publish_candidates = [
        df.get("publish_date"),
        df.get("NOTICE_DATE"),
        df.get("ANN_DATE"),
        df.get("REPORT_DATE"),
        df.get("report_date"),
    ]
    df["effective_date"] = infer_publish_dates(df["date"], *publish_candidates)
    df = df.dropna(subset=["effective_date"])
    df.sort_values("effective_date", inplace=True)
    for c in columns:
        if c not in df.columns:
            df[c] = 0.0
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    df = df[["date", "effective_date"] + columns]
    df[columns] = df[columns].ffill()
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
    fin_df["effective_date"] = pd.to_datetime(fin_df.get("effective_date", fin_df.get("publish_date", fin_df["date"])), errors="coerce")
    fin_df = fin_df.dropna(subset=["effective_date"])
    fin_df.sort_values("effective_date", inplace=True)
    merged = pd.merge_asof(
        price_dates[["date"]],
        fin_df[["effective_date"] + fin_columns],
        left_on="date",
        right_on="effective_date",
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


def _normalize_date(value: object) -> pd.Timestamp:
    dt = pd.to_datetime(value).normalize()
    if pd.isna(dt):
        raise ValueError(f"Invalid date: {value}")
    return dt


def _date_split_for_symbol(
    dates: np.ndarray,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    seq_len: int,
    stride: int,
) -> Optional[_DateSplit]:
    if len(dates) <= seq_len:
        return None
    # Normalize to int64 nanoseconds to avoid dtype-mismatch comparisons.
    dates_ns = pd.to_datetime(dates).astype("datetime64[ns]").view("int64")
    train_end_ns = np.int64(pd.to_datetime(train_end).to_datetime64())
    val_end_ns = np.int64(pd.to_datetime(val_end).to_datetime64())
    train_end_idx = int(np.searchsorted(dates_ns, train_end_ns, side="right") - 1)
    if train_end_idx < seq_len:
        return None
    val_start_idx = int(np.searchsorted(dates_ns, train_end_ns, side="right"))
    val_end_idx = int(np.searchsorted(dates_ns, val_end_ns, side="right") - 1)
    test_start_idx = int(np.searchsorted(dates_ns, val_end_ns, side="right"))

    train_max_start = train_end_idx - seq_len
    train_count = train_max_start // stride + 1 if train_max_start >= 0 else 0

    val_count = 0
    if val_start_idx <= val_end_idx and val_start_idx >= seq_len:
        val_start_base = val_start_idx - seq_len
        val_max_start = val_end_idx - seq_len
        if val_max_start >= val_start_base:
            val_count = (val_max_start - val_start_base) // stride + 1

    test_count = 0
    last_idx = len(dates) - 1
    if test_start_idx <= last_idx and test_start_idx >= seq_len:
        test_start_base = test_start_idx - seq_len
        test_max_start = last_idx - seq_len
        if test_max_start >= test_start_base:
            test_count = (test_max_start - test_start_base) // stride + 1

    return _DateSplit(
        train_end_idx=train_end_idx,
        val_start_idx=val_start_idx,
        val_end_idx=val_end_idx,
        test_start_idx=test_start_idx,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
    )


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


def prepare_date_window_datasets(
    symbols: Iterable[str],
    seq_len: int,
    stride: int = 1,
    price_columns: Optional[List[str]] = None,
    financial_columns: Optional[List[str]] = None,
    date_windows: Optional[Iterable[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    base_dir: Path = DATA_DIR,
    cache_dir: Optional[Path] = PREPROCESSED_DIR,
    preloaded_features: Optional[Dict[str, np.ndarray]] = None,
    existing_scaler: Optional[Dict[str, np.ndarray]] = None,
    target_mode: str = "log_return",
) -> List[DatasetWindow]:
    """
    Build datasets for multiple chronological windows defined by train_end/val_end.
    train <= train_end; val (train_end, val_end]; test > val_end.
    """
    if not date_windows:
        raise ValueError("date_windows must be provided for date-based splits")
    parsed_windows = [(_normalize_date(tr), _normalize_date(val)) for tr, val in date_windows]
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

    def _load_dates(sym: str) -> Optional[np.ndarray]:
        try:
            arr = load_stock_history(sym, base_dir=base_dir)
        except FileNotFoundError:
            return None
        df = pd.DataFrame(arr)
        df["date"] = pd.to_datetime(df["date"]).dt.floor("D").astype("datetime64[ns]")
        df.sort_values("date", inplace=True)
        return df["date"].to_numpy()

    window_states = [
        {"train_end": tr, "val_end": val, "splits": [], "mean": None, "m2": None, "count": 0}
        for tr, val in parsed_windows
    ]

    for sym in symbols_list:
        feats = _load_features(sym)
        dates = _load_dates(sym)
        if feats is None or dates is None or len(feats) <= seq_len:
            continue
        if len(dates) != len(feats):
            print(f"[Dataset] skip {sym}: feature/date length mismatch ({len(feats)} vs {len(dates)})")
            continue
        for state in window_states:
            split = _date_split_for_symbol(dates, state["train_end"], state["val_end"], seq_len, stride)
            if split is None or split.train_count <= 0:
                continue
            state["splits"].append(
                _SymbolSplit(
                    symbol=sym,
                    length=len(feats),
                    val_start=split.val_start_idx,
                    train_count=split.train_count,
                    val_count=split.val_count,
                    test_start=split.test_start_idx if split.test_count > 0 else None,
                    test_count=split.test_count,
                )
            )
            if existing_scaler:
                continue
            arr = np.asarray(feats[: split.train_end_idx + 1], dtype="float64")
            if arr.size == 0:
                continue
            batch_count = arr.shape[0]
            batch_mean = arr.mean(axis=0)
            batch_var = arr.var(axis=0)
            if state["mean"] is None:
                state["mean"] = batch_mean
                state["m2"] = batch_var * batch_count
                state["count"] = batch_count
                continue
            assert state["m2"] is not None
            delta = batch_mean - state["mean"]
            new_count = state["count"] + batch_count
            state["mean"] = state["mean"] + delta * (batch_count / new_count)
            state["m2"] = state["m2"] + batch_var * batch_count + delta * delta * (state["count"] * batch_count / new_count)
            state["count"] = new_count

    windows: List[DatasetWindow] = []
    for idx, state in enumerate(window_states, 1):
        splits = state["splits"]
        if not splits:
            continue
        if existing_scaler:
            scaler = existing_scaler
        else:
            if state["mean"] is None or state["m2"] is None or state["count"] == 0:
                raise RuntimeError(f"No training data to compute scaler for window {state['train_end']}->{state['val_end']}")
            std = np.sqrt(state["m2"] / state["count"]) + 1e-6
            scaler = {"mean": state["mean"].astype("float32"), "std": std.astype("float32")}

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
            target_mode=target_mode,
        )
        train_len = len(train_ds)
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
            target_mode=target_mode,
        )
        val_len = len(val_ds)
        test_ds = StreamingPriceDataset(
            splits,
            seq_len=seq_len,
            stride=stride,
            close_index=close_idx,
            price_columns=price_cols,
            financial_columns=fin_cols,
            base_dir=base_dir,
            cache_dir=cache_dir,
            scaler=scaler,
            mode="test",
            preloaded_features=preloaded_features,
            source_mtime=source_mtime,
            target_mode=target_mode,
        )
        test_ds = test_ds if len(test_ds) > 0 else None
        name = f"{state['train_end'].date()}_{state['val_end'].date()}"
        if train_len == 0 or val_len == 0:
            print(f"[Dataset] skip window={name} because train_samples={train_len}, val_samples={val_len}")
            continue
        print(
            f"[Dataset] window={name} symbols={len(splits)} train_samples={len(train_ds)} "
            f"val_samples={len(val_ds)} test_samples={len(test_ds) if test_ds else 0} "
            f"features={len(feature_columns)} stride={stride}"
        )
        windows.append(
            DatasetWindow(
                name=name,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                scaler=scaler,
                feature_columns=feature_columns,
                train_end=state["train_end"],
                val_end=state["val_end"],
            )
        )

    if not windows:
        raise RuntimeError("No sufficient data to build any date-based window.")
    return windows


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
    target_mode: str = "log_return",
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
        splits.append(
            _SymbolSplit(
                symbol=sym,
                length=len(feats),
                val_start=split,
                train_count=train_count,
                val_count=val_count,
                test_start=None,
                test_count=0,
            )
        )
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
        target_mode=target_mode,
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
        target_mode=target_mode,
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
