"""
Push notification providers.

Currently supports PushPlus (pushplus.plus) for WeChat-compatible push.
"""

from abc import ABC, abstractmethod
from typing import Optional
import urllib.request
import urllib.parse


class BaseProvider(ABC):
    """Abstract base class for push notification providers."""

    @abstractmethod
    def send(self, title: str, content: str) -> None:
        """Send a notification. Raises on failure."""
        ...


class PushPlusProvider(BaseProvider):
    """PushPlus HTTP API provider.

    API docs: https://www.pushplus.plus/doc/
    """

    BASE_URL = "https://www.pushplus.plus/send"

    def __init__(self, token: str):
        self.token = token

    def send(self, title: str, content: str) -> None:
        params = {
            "token": self.token,
            "title": title,
            "content": content,
            "type": "html",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "StockPilot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            if '"code":' in body and '"code":200' not in body:
                raise RuntimeError(f"PushPlus API error: {body}")
