"""WhatsApp channel adapter using local WebSocket bridge."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.base import BaseChannel

try:
    import websockets

    WHATSAPP_AVAILABLE = True
except Exception:
    websockets = None
    WHATSAPP_AVAILABLE = False

WHATSAPP_MAX_MESSAGE_LEN = 3500


@dataclass
class WhatsAppConfig:
    """WhatsApp channel configuration for runtime workers."""

    enabled: bool = False
    allow_from: list[str] = field(default_factory=list)
    bridge_url: str = "ws://127.0.0.1:3001"
    bridge_token: str = ""
    reconnect_delay_s: int = 5
    send_retry_attempts: int = 3
    send_retry_delay_ms: int = 400
    accept_group_messages: bool = True
    require_mention_in_group: bool = True

    @classmethod
    def from_any(cls, data: Any) -> "WhatsAppConfig":
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(
                enabled=bool(data.get("enabled", False)),
                allow_from=list(data.get("allow_from", data.get("allowFrom", [])) or []),
                bridge_url=str(data.get("bridge_url", data.get("bridgeUrl", cls.bridge_url)) or cls.bridge_url),
                bridge_token=str(data.get("bridge_token", data.get("bridgeToken", "")) or ""),
                reconnect_delay_s=int(
                    data.get("reconnect_delay_s", data.get("reconnectDelayS", 5)) or 5
                ),
                send_retry_attempts=int(
                    data.get("send_retry_attempts", data.get("sendRetryAttempts", 3)) or 3
                ),
                send_retry_delay_ms=int(
                    data.get("send_retry_delay_ms", data.get("sendRetryDelayMs", 400)) or 400
                ),
                accept_group_messages=bool(
                    data.get("accept_group_messages", data.get("acceptGroupMessages", True))
                ),
                require_mention_in_group=bool(
                    data.get("require_mention_in_group", data.get("requireMentionInGroup", True))
                ),
            )

        return cls(
            enabled=bool(getattr(data, "enabled", False)),
            allow_from=list(getattr(data, "allow_from", getattr(data, "allowFrom", [])) or []),
            bridge_url=str(getattr(data, "bridge_url", getattr(data, "bridgeUrl", cls.bridge_url)) or cls.bridge_url),
            bridge_token=str(getattr(data, "bridge_token", getattr(data, "bridgeToken", "")) or ""),
            reconnect_delay_s=int(
                getattr(data, "reconnect_delay_s", getattr(data, "reconnectDelayS", 5)) or 5
            ),
            send_retry_attempts=int(
                getattr(data, "send_retry_attempts", getattr(data, "sendRetryAttempts", 3)) or 3
            ),
            send_retry_delay_ms=int(
                getattr(data, "send_retry_delay_ms", getattr(data, "sendRetryDelayMs", 400)) or 400
            ),
            accept_group_messages=bool(
                getattr(data, "accept_group_messages", getattr(data, "acceptGroupMessages", True))
            ),
            require_mention_in_group=bool(
                getattr(data, "require_mention_in_group", getattr(data, "requireMentionInGroup", True))
            ),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowFrom": list(self.allow_from),
            "bridgeUrl": self.bridge_url,
            "bridgeToken": self.bridge_token,
            "reconnectDelayS": self.reconnect_delay_s,
            "sendRetryAttempts": self.send_retry_attempts,
            "sendRetryDelayMs": self.send_retry_delay_ms,
            "acceptGroupMessages": self.accept_group_messages,
            "requireMentionInGroup": self.require_mention_in_group,
        }


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel bridged through a localhost WebSocket service."""

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().to_public_dict()

    def __init__(self, config: Any, bus: MessageBus):
        parsed = WhatsAppConfig.from_any(config)
        super().__init__(parsed, bus)
        self.config: WhatsAppConfig = parsed
        self._ws: Any | None = None
        self._bridge_connected_event = asyncio.Event()
        self._wa_connected_event = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._processed_ids: OrderedDict[str, None] = OrderedDict()

    @property
    def is_connected(self) -> bool:
        return self._wa_connected_event.is_set()

    @property
    def is_bridge_connected(self) -> bool:
        return self._bridge_connected_event.is_set()

    async def start(self) -> None:
        if not WHATSAPP_AVAILABLE:
            logger.error("websockets package is not installed. Run: pip install websockets")
            return
        if not self.config.bridge_token:
            logger.error("WHATSAPP_BRIDGE_TOKEN is required")
            return

        self._running = True
        logger.info("WhatsApp channel started: bridge_url={}", self.config.bridge_url)
        while self._running:
            try:
                async with websockets.connect(self.config.bridge_url, max_size=4 * 1024 * 1024) as ws:
                    self._ws = ws
                    self._bridge_connected_event.clear()
                    self._wa_connected_event.clear()
                    await self._auth()
                    self._bridge_connected_event.set()
                    await self._receive_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WhatsApp bridge connection error: {}", exc)
            finally:
                self._bridge_connected_event.clear()
                self._wa_connected_event.clear()
                self._ws = None
            if self._running:
                await asyncio.sleep(max(self.config.reconnect_delay_s, 1))

    async def stop(self) -> None:
        self._running = False
        self._bridge_connected_event.clear()
        self._wa_connected_event.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        logger.info("WhatsApp channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        if not self._ws or not self.is_connected:
            logger.warning("WhatsApp bridge is not connected, skip outbound")
            return

        text = (msg.content or "").strip()
        media = list(msg.media or [])

        if text:
            for chunk in _split_message(text, WHATSAPP_MAX_MESSAGE_LEN):
                await self._send_with_retry(
                    {
                        "type": "send",
                        "to": msg.chat_id,
                        "text": chunk,
                    },
                    action="send_text",
                )

        for media_path in media:
            path = Path(media_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                logger.warning("Skip outbound media not found: {}", path)
                continue
            mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            await self._send_with_retry(
                {
                    "type": "send_media",
                    "to": msg.chat_id,
                    "filePath": str(path),
                    "mimetype": mimetype,
                    "fileName": path.name,
                },
                action="send_media",
            )

    async def wait_until_connected(self, timeout_s: float = 20.0) -> bool:
        if self.is_connected:
            return True
        try:
            await asyncio.wait_for(self._wa_connected_event.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    async def _auth(self) -> None:
        await self._send_json(
            {
                "type": "auth",
                "token": self.config.bridge_token,
            }
        )

    async def _receive_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except Exception:
                logger.warning("Ignore non-JSON message from WhatsApp bridge")
                continue
            await self._handle_bridge_event(data)

    async def _handle_bridge_event(self, data: dict[str, Any]) -> None:
        msg_type = str(data.get("type", "") or "")
        if msg_type == "message":
            await self._handle_bridge_message(data)
            return
        if msg_type == "status":
            status = str(data.get("status", "") or "").strip().lower()
            if status == "connected":
                self._wa_connected_event.set()
            elif status == "disconnected":
                self._wa_connected_event.clear()
            logger.info("WhatsApp bridge status: {}", status or data.get("status"))
            return
        if msg_type == "qr":
            logger.info("WhatsApp bridge QR updated. Please scan in terminal bridge process.")
            return
        if msg_type == "sent":
            return
        if msg_type == "error":
            logger.warning("WhatsApp bridge error: {}", data.get("error"))
            return

    async def _handle_bridge_message(self, data: dict[str, Any]) -> None:
        msg_id = str(data.get("id", "") or "").strip()
        if msg_id:
            if msg_id in self._processed_ids.keys():
                return
            self._processed_ids[msg_id] = None
            if len(self._processed_ids) > 2000:
                self._processed_ids.popitem(last=False)

        sender = str(data.get("sender", "") or "").strip()
        if not sender:
            return
        # Prefer PN jid for reply target when bridge provides both LID and PN.
        # Inbound sender_id keeps original sender for traceability.
        pn = str(data.get("pn", "") or "").strip()
        reply_chat_id = pn or sender

        content = str(data.get("content", "") or "").strip()
        media_list_raw = data.get("media") or []
        media = [str(item).strip() for item in media_list_raw if str(item).strip()]
        if not content and not media:
            return
        if not content and media:
            content = "[Media]"

        metadata = {"message_id": msg_id}
        is_group = bool(data.get("isGroup", False))
        was_mentioned = bool(data.get("wasMentioned", False))
        if is_group:
            metadata["is_group"] = True
            metadata["was_mentioned"] = was_mentioned
            if not self.config.accept_group_messages:
                return
            if self.config.require_mention_in_group and not was_mentioned:
                return

        await self._handle_message(
            sender_id=sender,
            chat_id=reply_chat_id,
            content=content,
            media=media,
            metadata=metadata,
        )

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._ws:
            raise RuntimeError("WhatsApp bridge websocket is not connected")
        text = json.dumps(payload, ensure_ascii=False)
        async with self._send_lock:
            await self._ws.send(text)

    async def _send_with_retry(self, payload: dict[str, Any], action: str) -> None:
        attempts = max(1, int(self.config.send_retry_attempts))
        delay_s = max(0.0, float(self.config.send_retry_delay_ms) / 1000.0)

        for idx in range(1, attempts + 1):
            try:
                await self._send_json(payload)
                return
            except Exception as exc:
                if idx >= attempts:
                    logger.error(
                        "WhatsApp {} failed after {} attempt(s): {}",
                        action,
                        attempts,
                        exc,
                    )
                    return
                logger.warning(
                    "WhatsApp {} attempt {}/{} failed: {}",
                    action,
                    idx,
                    attempts,
                    exc,
                )
                if delay_s > 0:
                    await asyncio.sleep(delay_s)


def _split_message(content: str, max_len: int) -> list[str]:
    if len(content) <= max_len:
        return [content]

    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        cut = remaining[:max_len]
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(remaining[:pos])
        remaining = remaining[pos:].lstrip()
    return chunks
