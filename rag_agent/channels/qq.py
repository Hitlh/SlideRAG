"""QQ channel implementation using botpy SDK."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.base import BaseChannel

try:
    import botpy
    from botpy.message import C2CMessage, GroupMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None
    GroupMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage, GroupMessage


@dataclass
class QQConfig:
    """QQ channel configuration for rag_agent minimal runtime."""

    enabled: bool = False
    app_id: str = ""
    secret: str = ""
    allow_from: list[str] = field(default_factory=list)
    msg_format: Literal["plain", "markdown"] = "plain"

    @classmethod
    def from_any(cls, data: Any) -> "QQConfig":
        """Build config from dict/object while accepting camelCase keys."""
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            msg_format = data.get("msg_format", data.get("msgFormat", "plain"))
            if msg_format not in ("plain", "markdown"):
                msg_format = "plain"
            return cls(
                enabled=bool(data.get("enabled", False)),
                app_id=str(data.get("app_id", data.get("appId", "")) or ""),
                secret=str(data.get("secret", "") or ""),
                allow_from=list(data.get("allow_from", data.get("allowFrom", [])) or []),
                msg_format=msg_format,
            )

        msg_format = getattr(data, "msg_format", getattr(data, "msgFormat", "plain"))
        if msg_format not in ("plain", "markdown"):
            msg_format = "plain"
        return cls(
            enabled=bool(getattr(data, "enabled", False)),
            app_id=str(getattr(data, "app_id", getattr(data, "appId", "")) or ""),
            secret=str(getattr(data, "secret", "") or ""),
            allow_from=list(getattr(data, "allow_from", getattr(data, "allowFrom", [])) or []),
            msg_format=msg_format,
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Default config payload in camelCase for user config files."""
        return {
            "enabled": self.enabled,
            "appId": self.app_id,
            "secret": self.secret,
            "allowFrom": list(self.allow_from),
            "msgFormat": self.msg_format,
        }


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log; rag_agent uses loguru.
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message, is_group=False)

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_message(message, is_group=True)

        async def on_direct_message_create(self, message):
            await channel._on_message(message, is_group=False)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"
    display_name = "QQ"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return QQConfig().to_public_dict()

    def __init__(self, config: Any, bus: MessageBus):
        parsed = QQConfig.from_any(config)
        super().__init__(parsed, bus)
        self.config: QQConfig = parsed
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque[str] = deque(maxlen=1000)
        self._msg_seq: int = 1
        self._chat_type_cache: dict[str, str] = {}

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        bot_cls = _make_bot_class(self)
        self._client = bot_cls()
        logger.info("QQ bot started")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning("QQ bot error: {}", e)

            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return

        try:
            msg_id = msg.metadata.get("message_id")
            self._msg_seq += 1
            use_markdown = self.config.msg_format == "markdown"
            payload: dict[str, Any] = {
                "msg_type": 2 if use_markdown else 0,
                "msg_id": msg_id,
                "msg_seq": self._msg_seq,
            }
            if use_markdown:
                payload["markdown"] = {"content": msg.content}
            else:
                payload["content"] = msg.content

            chat_type = self._chat_type_cache.get(msg.chat_id, "c2c")
            if chat_type == "group":
                await self._client.api.post_group_message(group_openid=msg.chat_id, **payload)
            else:
                await self._client.api.post_c2c_message(openid=msg.chat_id, **payload)
        except Exception as e:
            logger.error("Error sending QQ message: {}", e)

    async def _on_message(self, data: "C2CMessage | GroupMessage", is_group: bool = False) -> None:
        """Handle incoming message from QQ and push it to bus inbound."""
        try:
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            content = (data.content or "").strip()
            if not content:
                return

            if is_group:
                chat_id = data.group_openid
                user_id = data.author.member_openid
                self._chat_type_cache[chat_id] = "group"
            else:
                chat_id = str(
                    getattr(data.author, "id", None)
                    or getattr(data.author, "user_openid", "unknown")
                )
                user_id = chat_id
                self._chat_type_cache[chat_id] = "c2c"

            await self._handle_message(
                sender_id=user_id,
                chat_id=chat_id,
                content=content,
                metadata={"message_id": data.id},
            )
        except Exception:
            logger.exception("Error handling QQ message")
