import os
import re
from typing import Optional

from openai import OpenAI

from ..stock_types import StockScore, StockSnapshot
from .base import LLMClient, build_prompt


class OpenAIClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)
        self.model = model or os.getenv("AUTO_SELECT_LLM_MODEL", "gpt-4o-mini")

    def score(self, snapshot: StockSnapshot) -> StockScore:
        prompt = build_prompt(snapshot)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only a number between 0 and 10."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "0"
        score = self._parse_score(content)
        return StockScore(
            symbol=snapshot.meta.symbol,
            score=score,
            rationale="LLM scored based on provided factors.",
            meta=snapshot.meta,
            factors=snapshot.factors,
        )

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0
        value = float(match.group(1))
        return max(0.0, min(10.0, value))
