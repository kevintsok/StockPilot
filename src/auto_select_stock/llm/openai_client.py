import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from openai import OpenAI

from ..core.types import StockScore, StockSnapshot
from .base import LLMClient, build_prompt


def _get_cc_switch_db_path() -> Optional[Path]:
    """Try to find CC Switch database on Windows or WSL."""
    candidates = [
        # Direct Windows path
        Path(os.path.expanduser("~/.cc-switch/cc-switch.db")),
        Path("/mnt/c/Users/kevin/.cc-switch/cc-switch.db"),
        Path("C:/Users/kevin/.cc-switch/cc-switch.db"),
        # WSL: Windows files mapped under /mnt/c
        Path("/mnt/c/Users/kevin/.cc-switch/cc-switch.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_minimax_config() -> Optional[dict]:
    """Load MiniMax provider config from CC Switch database."""
    db_path = _get_cc_switch_db_path()
    if not db_path:
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            "SELECT settings_config FROM providers WHERE LOWER(provider_type) = 'minimax' OR LOWER(name) = 'minimax' LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception:
        pass
    return None


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible LLM client supporting multiple providers.

    Provider is selected via the `provider` argument or OPENAI_API_KEY env var.
    For MiniMax, credentials are auto-loaded from CC Switch database.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,  # "openai" or "minimax"
    ):
        self._provider = provider or os.environ.get("AUTO_SELECT_LLM_PROVIDER", "openai")

        if self._provider == "minimax":
            config = _load_minimax_config()
            env = (config or {}).get("env", {}) if config else {}
            self._api_key = api_key or os.environ.get("MINIMAX_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN", "")
            self._base_url = base_url or os.environ.get("MINIMAX_BASE_URL") or env.get("ANTHROPIC_BASE_URL", "")
            self._model = model or os.environ.get("MINIMAX_MODEL") or env.get("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed")
            if not self._api_key or not self._base_url:
                raise RuntimeError(
                    "MiniMax credentials not found. Set MINIMAX_API_KEY / MINIMAX_BASE_URL "
                    "or ensure CC Switch has MiniMax provider configured."
                )
        else:
            self._api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self._model = model or os.getenv("AUTO_SELECT_LLM_MODEL", "gpt-4o-mini")

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=60.0)

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self):
        return self._client

    @property
    def provider(self) -> str:
        return self._provider

    def score(self, snapshot: StockSnapshot) -> StockScore:
        prompt = build_prompt(snapshot)
        resp = self._client.chat.completions.create(
            model=self._model,
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
        import re

        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0
        value = float(match.group(1))
        return max(0.0, min(10.0, value))
