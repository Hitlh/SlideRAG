"""Feishu/Lark channel implementation (single-chat text MVP)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.base import BaseChannel

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1.model import MentionEvent, P2ImMessageReceiveV1
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    MentionEvent = Any
    P2ImMessageReceiveV1 = Any
    FEISHU_DOMAIN = "https://open.feishu.cn"
    LARK_DOMAIN = "https://open.larksuite.com"

if TYPE_CHECKING:
    from lark_oapi.api.im.v1.model import P2ImMessageReceiveV1


@dataclass
class FeishuConfig:
    """Feishu channel configuration for SlideRAG runtime."""

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    allow_from: list[str] = field(default_factory=list)
    domain: Literal["feishu", "lark"] = "feishu"

    @classmethod
    def from_any(cls, data: Any) -> "FeishuConfig":
        """Build config from dict/object while accepting camelCase keys."""
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            domain = str(data.get("domain", "feishu") or "feishu").strip().lower()
            if domain not in ("feishu", "lark"):
                domain = "feishu"
            return cls(
                enabled=bool(data.get("enabled", False)),
                app_id=str(data.get("app_id", data.get("appId", "")) or ""),
                app_secret=str(data.get("app_secret", data.get("appSecret", "")) or ""),
                encrypt_key=str(data.get("encrypt_key", data.get("encryptKey", "")) or ""),
                verification_token=str(
                    data.get("verification_token", data.get("verificationToken", "")) or ""
                ),
                allow_from=list(data.get("allow_from", data.get("allowFrom", [])) or []),
                domain=domain,
            )

        domain = str(getattr(data, "domain", "feishu") or "feishu").strip().lower()
        if domain not in ("feishu", "lark"):
            domain = "feishu"
        return cls(
            enabled=bool(getattr(data, "enabled", False)),
            app_id=str(getattr(data, "app_id", getattr(data, "appId", "")) or ""),
            app_secret=str(getattr(data, "app_secret", getattr(data, "appSecret", "")) or ""),
            encrypt_key=str(getattr(data, "encrypt_key", getattr(data, "encryptKey", "")) or ""),
            verification_token=str(
                getattr(data, "verification_token", getattr(data, "verificationToken", "")) or ""
            ),
            allow_from=list(getattr(data, "allow_from", getattr(data, "allowFrom", [])) or []),
            domain=domain,
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Default config payload in camelCase for user config files."""
        return {
            "enabled": self.enabled,
            "appId": self.app_id,
            "appSecret": self.app_secret,
            "encryptKey": self.encrypt_key,
            "verificationToken": self.verification_token,
            "allowFrom": list(self.allow_from),
            "domain": self.domain,
        }


class FeishuChannel(BaseChannel):
    """Feishu channel using WebSocket long connection (text-only MVP)."""

    name = "feishu"
    display_name = "Feishu"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return FeishuConfig().to_public_dict()

    def __init__(self, config: Any, bus: MessageBus):
        parsed = FeishuConfig.from_any(config)
        super().__init__(parsed, bus)
        self.config: FeishuConfig = parsed
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Start the Feishu bot with WebSocket long connection."""
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        domain = LARK_DOMAIN if self.config.domain == "lark" else FEISHU_DOMAIN
        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        event_handler = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(self._on_message_sync).build()

        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            domain=domain,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        def run_ws() -> None:
            import lark_oapi.ws.client as _lark_ws_client

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            _lark_ws_client.loop = ws_loop
            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as exc:
                        logger.warning("Feishu WebSocket error: {}", exc)
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        logger.info("Feishu bot started with WebSocket long connection")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop Feishu channel loop."""
        self._running = False
        logger.info("Feishu bot stopped")

    def _on_message_sync(self, data: Any) -> None:
        """Sync callback from SDK thread; schedule async handler on main loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    @staticmethod
    def _resolve_mentions(text: str, mentions: list[Any] | None) -> str:
        """Replace @_user_n placeholders with mention names for readability."""
        if not mentions or not text:
            return text
        for mention in mentions:
            key = getattr(mention, "key", None)
            if not key or key not in text:
                continue
            name = getattr(mention, "name", None) or key
            text = text.replace(key, f"@{name}")
        return text

    @staticmethod
    def _extract_post_text(content_json: dict[str, Any]) -> str:
        """Extract plain text from Feishu post message content."""
        root: Any = content_json
        if isinstance(root, dict) and isinstance(root.get("post"), dict):
            root = root["post"]
        if not isinstance(root, dict):
            return ""

        candidate_blocks: list[dict[str, Any]] = []
        if isinstance(root.get("content"), list):
            candidate_blocks.append(root)
        for key in ("zh_cn", "en_us", "ja_jp"):
            if isinstance(root.get(key), dict):
                candidate_blocks.append(root[key])

        for block in candidate_blocks:
            content_rows = block.get("content")
            if not isinstance(content_rows, list):
                continue
            parts: list[str] = []
            for row in content_rows:
                if not isinstance(row, list):
                    continue
                for element in row:
                    if not isinstance(element, dict):
                        continue
                    tag = element.get("tag")
                    if tag in ("text", "a"):
                        parts.append(str(element.get("text", "") or ""))
                    elif tag == "at":
                        parts.append(f"@{element.get('user_name', 'user')}")
            text = " ".join(item for item in parts if item).strip()
            if text:
                return text
        return ""

    async def _on_message(self, data: Any) -> None:
        """Handle incoming Feishu event and route text messages to bus."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            if getattr(sender, "sender_type", "") == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_type = message.chat_type
            if chat_type != "p2p":
                return

            msg_type = message.message_type
            content = ""
            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}

            if msg_type == "text":
                text = str(content_json.get("text", "") or "").strip()
                content = self._resolve_mentions(text, getattr(message, "mentions", None))
            elif msg_type == "post":
                content = self._extract_post_text(content_json)
            else:
                # Text-only MVP: ignore non-text message types.
                return

            if not content:
                return

            # For p2p chats reply target equals sender open_id.
            await self._handle_message(
                sender_id=sender_id,
                chat_id=sender_id,
                content=content,
                metadata={
                    "message_id": message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                },
            )
        except Exception:
            logger.exception("Error processing Feishu message")

    async def send(self, msg: OutboundMessage) -> None:
        """Send plain text message to Feishu single chat by open_id."""
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        text = (msg.content or "").strip()
        if not text:
            return

        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = json.dumps({"text": text}, ensure_ascii=False)
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(msg.chat_id)
                .msg_type("text")
                .content(body)
                .build()
            )
            .build()
        )

        loop = asyncio.get_running_loop()

        def _send_sync() -> None:
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu text message: code={}, msg={}, log_id={}",
                    response.code,
                    response.msg,
                    response.get_log_id(),
                )

        await loop.run_in_executor(None, _send_sync)