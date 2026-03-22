from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def _deadline_for_period_end(period_end: pd.Timestamp) -> pd.Timestamp:
    """
    根据财报期末日推导一个“最迟披露日”，避免在发布日期之前泄漏信息。
    采用交易所规定的最晚披露窗口，略微保守但不会早于实际公告日。
    """
    if pd.isna(period_end):
        return pd.NaT
    y = period_end.year
    month = period_end.month
    day = period_end.day
    if (month, day) == (3, 31):  # Q1 -> 4月末
        return pd.Timestamp(date(y, 4, 30))
    if (month, day) == (6, 30):  # 中报 -> 8月末
        return pd.Timestamp(date(y, 8, 31))
    if (month, day) == (9, 30):  # 三季报 -> 10月末
        return pd.Timestamp(date(y, 10, 31))
    if (month, day) == (12, 31):  # 年报 -> 次年4月末
        return pd.Timestamp(date(y + 1, 4, 30))
    return pd.Timestamp(period_end) + pd.Timedelta(days=90)


def infer_publish_dates(period_end: pd.Series, *candidates: Optional[pd.Series]) -> pd.Series:
    """
    归一化 publish_date：优先使用数据源自带的公告日列，缺失时使用监管披露截止日推导。

    返回同长度 Series，dtype=datetime64[ns]，确保不会早于 period_end。
    """
    end_dt = pd.to_datetime(period_end, errors="coerce").dt.normalize()
    publish = pd.Series(pd.NaT, index=end_dt.index, dtype="datetime64[ns]")
    for cand in candidates:
        if cand is None:
            continue
        cand_dt = pd.to_datetime(cand, errors="coerce").dt.normalize()
        if len(cand_dt) != len(end_dt):
            continue
        publish = publish.fillna(cand_dt)
    fallback = end_dt.apply(_deadline_for_period_end)
    publish = publish.fillna(fallback)
    publish = publish.where((publish >= end_dt) | end_dt.isna(), fallback)
    return publish.dt.normalize()
