"""Chat channel adapters for rag_agent."""

from .base import BaseChannel
from .qq import QQChannel, QQConfig
from .weixin import WeixinChannel, WeixinConfig

__all__ = ["BaseChannel", "QQChannel", "QQConfig", "WeixinChannel", "WeixinConfig"]
