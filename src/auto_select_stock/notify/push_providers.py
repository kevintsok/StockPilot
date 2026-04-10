"""
Push notification providers.

Supports multiple providers via a plugin registry. To add a new provider:
1. Create a class that extends BaseProvider
2. Decorate it with @register_provider("provider_name")
3. The provider will be available via get_provider("provider_name", **config)

Currently registered providers:
- pushplus: PushPlus HTTP API (pushplus.plus) for WeChat-compatible push
"""

from abc import ABC, abstractmethod
from typing import Dict, Type
import urllib.request
import urllib.parse


# ---------------------------------------------------------------------------
# Provider registry (plugin pattern)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, Type["BaseProvider"]] = {}


def register_provider(name: str):
    """
    Decorator to register a provider class.

    Usage:
        @register_provider("myprovider")
        class MyProvider(BaseProvider):
            ...
    """
    def decorator(cls: Type["BaseProvider"]) -> Type["BaseProvider"]:
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


def get_provider(name: str, **config) -> "BaseProvider":
    """
    Factory function to get a provider instance by name.

    Usage:
        provider = get_provider("pushplus", token="xxx")
        provider = get_provider("serverchan", token="xxx", uid="yyy")
    """
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys()) or "(none)"
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return _PROVIDER_REGISTRY[name](**config)


# ---------------------------------------------------------------------------
# Base provider interface
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """Abstract base class for push notification providers."""

    @abstractmethod
    def send(self, title: str, content: str) -> None:
        """Send a notification. Raises on failure."""
        ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register if class name ends with "Provider"
        # (but only if not already registered via decorator)
        if cls.__name__.endswith("Provider") and cls.__name__ not in _PROVIDER_REGISTRY:
            provider_name = cls.__name__[:-8].lower()  # Strip "Provider" suffix
            _PROVIDER_REGISTRY[provider_name] = cls


# ---------------------------------------------------------------------------
# PushPlus provider
# ---------------------------------------------------------------------------

@register_provider("pushplus")
class PushPlusProvider(BaseProvider):
    """PushPlus HTTP API provider.

    API docs: https://www.pushplus.plus/doc/

    Usage:
        provider = get_provider("pushplus", token="YOUR_TOKEN")
        provider.send("title", "<h1>HTML content</h1>")
    """

    BASE_URL = "https://www.pushplus.plus/send"

    def __init__(self, token: str):
        self.token = token

    def send(self, title: str, content: str) -> None:
        import json
        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "type": "html",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "User-Agent": "StockPilot/1.0",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            if '"code":' in body and '"code":200' not in body:
                raise RuntimeError(f"PushPlus API error: {body}")


# ---------------------------------------------------------------------------
# ServerChan provider (example of adding new provider)
# ---------------------------------------------------------------------------
# Uncomment and fill in your serverchan details to enable:
#
# @register_provider("serverchan")
# class ServerChanProvider(BaseProvider):
#     """ServerChan (srewha.com) provider for WeChat/dingtalk push.
#
#     Usage:
#         provider = get_provider("serverchan", token="YOUR_TOKEN", uid="OPTIONAL_UID")
#         provider.send("title", "text content")
#     """
#
#     BASE_URL = "https://srewha.com/send"
#
#     def __init__(self, token: str, uid: Optional[str] = None):
#         self.token = token
#         self.uid = uid
#
#     def send(self, title: str, content: str) -> None:
#         params = {
#             "token": self.token,
#             "title": title,
#             "content": content,
#         }
#         if self.uid:
#             params["uid"] = self.uid
#         url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
#         req = urllib.request.Request(
#             url,
#             headers={"User-Agent": "StockPilot/1.0"},
#         )
#         with urllib.request.urlopen(req, timeout=30) as resp:
#             body = resp.read().decode("utf-8")
#             if '"code":' in body and '"code":200' not in body:
#                 raise RuntimeError(f"ServerChan API error: {body}")


# ---------------------------------------------------------------------------
# Email provider (example of adding new provider)
# ---------------------------------------------------------------------------
# Uncomment and configure SMTP settings to enable:
#
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
#
# @register_provider("email")
# class EmailProvider(BaseProvider):
#     """Email provider via SMTP.
#
#     Usage:
#         provider = get_provider("email",
#             smtp_host="smtp.gmail.com",
#             smtp_port=587,
#             smtp_user="user@gmail.com",
#             smtp_password="app_password",
#             from_addr="user@gmail.com",
#             to_addrs=["recipient@example.com"],
#         )
#         provider.send("title", "text content")
#     """
#
#     def __init__(
#         self,
#         smtp_host: str,
#         smtp_port: int,
#         smtp_user: str,
#         smtp_password: str,
#         from_addr: str,
#         to_addrs: list,
#     ):
#         self.smtp_host = smtp_host
#         self.smtp_port = smtp_port
#         self.smtp_user = smtp_user
#         self.smtp_password = smtp_password
#         self.from_addr = from_addr
#         self.to_addrs = to_addrs
#
#     def send(self, title: str, content: str) -> None:
#         msg = MIMEMultipart()
#         msg["From"] = self.from_addr
#         msg["To"] = ", ".join(self.to_addrs)
#         msg["Subject"] = title
#         msg.attach(MIMEText(content, "html"))
#
#         with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
#             server.starttls()
#             server.login(self.smtp_user, self.smtp_password)
#             server.send_message(msg)
