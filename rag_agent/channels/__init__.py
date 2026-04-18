"""Chat channel adapters for rag_agent."""

from .base import BaseChannel
from .feishu import FeishuChannel, FeishuConfig
from .qq import QQChannel, QQConfig
from .whatsapp import WhatsAppChannel, WhatsAppConfig
from .weixin import WeixinChannel, WeixinConfig

__all__ = [
	"BaseChannel",
	"QQChannel",
	"QQConfig",
	"WeixinChannel",
	"WeixinConfig",
	"FeishuChannel",
	"FeishuConfig",
	"WhatsAppChannel",
	"WhatsAppConfig",
]
