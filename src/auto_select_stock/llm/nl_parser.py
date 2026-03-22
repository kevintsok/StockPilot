"""
Natural language to ScreenCriteria parser using LLM.

Uses MiniMax to translate Chinese natural language queries into structured
ScreenCriteria that can be applied against the SQLite database.
"""

import json
from typing import Optional

from ..web.screener import ScreenCriteria
from .base import LLMClient
from .openai_client import OpenAIClient


_SYSTEM_PROMPT = """你是一个A股选股条件解析器。你的任务是将用户的中文自然语言描述转换为结构化的选股条件JSON。

## 重要概念区分

- **涨幅/涨跌幅/价格变化** = pct_change = 价格变动百分比 → 使用 max_pct_change / min_pct_change 字段
- **ROE** = 净资产收益率（财务指标）→ 使用 min_roe 字段
- **EPS** = 每股收益（财务指标）→ 使用 min_eps 字段
- **换手率** = turnover_rate → 使用 min_turnover_rate / max_turnover_rate 字段

"涨幅不超过10%" ≠ ROE！"涨幅"是价格变化，不是财务指标！

## 可用的筛选字段

| 字段 | 中文描述 | 单位 | 说明 |
|------|---------|------|------|
| lookback_days | 涨跌幅统计周期 | 天 | 默认60天。例如"3个月内"=90天，"1年内"=365天 |
| min_pct_change | 涨幅下限 | % | 例如"涨幅超过10%" → min_pct_change=10.0 |
| max_pct_change | 涨幅上限 | % | 例如"涨幅不超过10%" → max_pct_change=10.0 |
| min_roe | ROE下限 | % | 例如"ROE高于10%" → min_roe=10.0 |
| min_eps | EPS下限 | 元 | 例如"EPS高于0.5元" → min_eps=0.5 |
| min_turnover_rate | 换手率下限 | % | 例如"换手率高于2%" → min_turnover_rate=2.0 |
| max_turnover_rate | 换手率上限 | % | 例如"换手率低于5%" → max_turnover_rate=5.0 |

## 规则

1. lookback_days: 从当前往回数多少天的数据进行统计
   - "N个月内" → N * 30 天
   - "N日内" / "N天内" → N 天
   - "N年内" → N * 365 天
   - 不指定则默认 60 天

2. 涨幅: (最新收盘价 - N天前收盘价) / N天前收盘价 * 100%
   - "涨幅不超过X%" = max_pct_change = X
   - "涨幅大于X%" / "涨幅超过X%" / "涨幅高于X%" = min_pct_change = X
   - 注意："涨幅"前面没有"财务"相关字眼！

3. 只返回符合条件的字段，不需要的字段省略不要（设为null）

## 输出格式

严格返回以下JSON格式，不要包含任何其他内容：
{
  "lookback_days": 60,
  "min_pct_change": null,
  "max_pct_change": 10.0,
  "min_roe": null,
  "min_eps": null,
  "min_turnover_rate": null,
  "max_turnover_rate": null
}

如果无法解析某个条件，对应字段设为null。"""


_USER_PROMPT_TEMPLATE = """将以下中文选股条件解析为JSON：

"{query}"

只输出JSON，不要其他内容："""


def parse_nl_query_with_llm(query: str, llm_client: Optional[LLMClient] = None) -> ScreenCriteria:
    """
    Parse a Chinese natural language query into a ScreenCriteria using LLM.

    Falls back to regex parsing if LLM is not available.

    Args:
        query: Chinese natural language query
        llm_client: Optional LLM client. If None, uses MiniMax from CC Switch.

    Returns:
        ScreenCriteria object
    """
    if llm_client is None:
        try:
            llm_client = OpenAIClient(provider="minimax")
        except Exception:
            # Fall back to regex parser
            from ..web.screener import parse_nl_query as regex_parse
            return regex_parse(query)

    # Try to use LLM
    try:
        # MiniMaxClient stores the OpenAI-compatible client as self.client
        client = llm_client.client
        model = getattr(llm_client, "model", "MiniMax-M2.7-highspeed")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(query=query)},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or "{}"
        # Extract JSON from response
        json_str = _extract_json(content)
        data = json.loads(json_str)
        return ScreenCriteria(
            lookback_days=data.get("lookback_days", 60),
            min_pct_change=data.get("min_pct_change"),
            max_pct_change=data.get("max_pct_change"),
            min_roe=data.get("min_roe"),
            min_eps=data.get("min_eps"),
            min_turnover_rate=data.get("min_turnover_rate"),
            max_turnover_rate=data.get("max_turnover_rate"),
        )
    except Exception:
        # Fall back to regex parser
        from ..web.screener import parse_nl_query as regex_parse
        return regex_parse(query)


def _extract_json(text: str) -> str:
    """Extract JSON object from LLM response text."""
    text = text.strip()
    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        return text[start:end]
    # Try array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        return text[start:end]
    return "{}"
