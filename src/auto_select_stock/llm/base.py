from abc import ABC, abstractmethod
from typing import Any, Dict

from ..stock_types import StockSnapshot, StockScore


class LLMClient(ABC):
    @abstractmethod
    def score(self, snapshot: StockSnapshot) -> StockScore:
        """
        Return a StockScore with score in [0, 10].
        """
        raise NotImplementedError


def build_prompt(snapshot: StockSnapshot) -> str:
    """
    Construct a concise prompt for the model.
    """
    meta = snapshot.meta
    factors_text = "\n".join(f"- {k}: {v:.3f}" for k, v in snapshot.factors.items())
    recent = snapshot.recent_rows[-1]
    return (
        "你是价值投资助手，请根据财务与交易特征判断该 A 股是否被低估。\n"
        "输出严格为一个 0-10 的数字，10 代表最被低估。\n"
        f"股票代码: {meta.symbol}\n"
        f"最新收盘价: {recent.close:.3f}\n"
        f"最新换手率: {recent.turnover_rate:.3f}\n"
        f"近5日量比: {snapshot.factors.get('vol_ratio_5', 0):.3f}\n"
        f"因子摘要:\n{factors_text}\n"
        "仅输出数字，不要附加说明。"
    )
