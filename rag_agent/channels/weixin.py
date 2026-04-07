"""Weixin (personal WeChat) channel with QR login and text-only messaging.

This MVP channel is intended for transport loopback testing before agent integration.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.base import BaseChannel

# MessageItemType
ITEM_TEXT = 1

# MessageType
MESSAGE_TYPE_BOT = 2

# MessageState
MESSAGE_STATE_FINISH = 2

WEIXIN_CHANNEL_VERSION = "1.0.3"
BASE_INFO: dict[str, str] = {"channel_version": WEIXIN_CHANNEL_VERSION}

ERRCODE_SESSION_EXPIRED = -14
SESSION_PAUSE_DURATION_S = 60 * 60

MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_DELAY_S = 30
RETRY_DELAY_S = 2
MAX_QR_REFRESH_COUNT = 3
DEFAULT_LONG_POLL_TIMEOUT_S = 35
WEIXIN_MAX_MESSAGE_LEN = 4000


@dataclass
class WeixinConfig:
    """Weixin channel configuration for rag_agent runtime."""

    enabled: bool = False
    allow_from: list[str] = field(default_factory=list)
    base_url: str = "https://ilinkai.weixin.qq.com"
    route_tag: str | int | None = None
    token: str = ""
    state_dir: str = ""
    poll_timeout: int = DEFAULT_LONG_POLL_TIMEOUT_S

    @classmethod
    def from_any(cls, data: Any) -> "WeixinConfig":
        """Build config from dict/object and accept camelCase keys."""
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(
                enabled=bool(data.get("enabled", False)),
                allow_from=list(data.get("allow_from", data.get("allowFrom", [])) or []),
                base_url=str(data.get("base_url", data.get("baseUrl", cls.base_url)) or cls.base_url),
                route_tag=data.get("route_tag", data.get("routeTag")),
                token=str(data.get("token", "") or ""),
                state_dir=str(data.get("state_dir", data.get("stateDir", "")) or ""),
                poll_timeout=int(data.get("poll_timeout", data.get("pollTimeout", DEFAULT_LONG_POLL_TIMEOUT_S)) or DEFAULT_LONG_POLL_TIMEOUT_S),
            )

        return cls(
            enabled=bool(getattr(data, "enabled", False)),
            allow_from=list(getattr(data, "allow_from", getattr(data, "allowFrom", [])) or []),
            base_url=str(getattr(data, "base_url", getattr(data, "baseUrl", cls.base_url)) or cls.base_url),
            route_tag=getattr(data, "route_tag", getattr(data, "routeTag", None)),
            token=str(getattr(data, "token", "") or ""),
            state_dir=str(getattr(data, "state_dir", getattr(data, "stateDir", "")) or ""),
            poll_timeout=int(getattr(data, "poll_timeout", getattr(data, "pollTimeout", DEFAULT_LONG_POLL_TIMEOUT_S)) or DEFAULT_LONG_POLL_TIMEOUT_S),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowFrom": list(self.allow_from),
            "baseUrl": self.base_url,
            "routeTag": self.route_tag,
            "token": self.token,
            "stateDir": self.state_dir,
            "pollTimeout": self.poll_timeout,
        }


class WeixinChannel(BaseChannel):
    """Weixin channel using HTTP long-poll API (text-only MVP)."""

    name = "weixin"
    display_name = "Weixin"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WeixinConfig().to_public_dict()

    def __init__(self, config: Any, bus: MessageBus):
        parsed = WeixinConfig.from_any(config)
        super().__init__(parsed, bus)
        self.config: WeixinConfig = parsed

        self._client: httpx.AsyncClient | None = None
        self._state_dir: Path | None = None
        self._token: str = ""
        self._get_updates_buf: str = ""
        self._context_tokens: dict[str, str] = {}
        self._processed_ids: OrderedDict[str, None] = OrderedDict()
        self._next_poll_timeout_s: int = DEFAULT_LONG_POLL_TIMEOUT_S
        self._session_pause_until: float = 0.0

    def _get_state_dir(self) -> Path:
        if self._state_dir:
            return self._state_dir
        if self.config.state_dir:
            d = Path(self.config.state_dir).expanduser()
        else:
            d = Path.home() / ".pptrag" / "weixin"
        d.mkdir(parents=True, exist_ok=True)
        self._state_dir = d
        return d

    def _load_state(self) -> bool:
        state_file = self._get_state_dir() / "account.json"
        if not state_file.exists():
            return False
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            self._token = str(data.get("token", "") or "")
            self._get_updates_buf = str(data.get("get_updates_buf", "") or "")
            ctx = data.get("context_tokens", {})
            if isinstance(ctx, dict):
                self._context_tokens = {
                    str(user_id): str(token)
                    for user_id, token in ctx.items()
                    if str(user_id).strip() and str(token).strip()
                }
            else:
                self._context_tokens = {}
            base_url = str(data.get("base_url", "") or "")
            if base_url:
                self.config.base_url = base_url
            return bool(self._token)
        except Exception as exc:
            logger.warning("Failed to load Weixin state: {}", exc)
            return False

    def _save_state(self) -> None:
        state_file = self._get_state_dir() / "account.json"
        try:
            payload = {
                "token": self._token,
                "get_updates_buf": self._get_updates_buf,
                "context_tokens": self._context_tokens,
                "base_url": self.config.base_url,
            }
            state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save Weixin state: {}", exc)

    @staticmethod
    def _random_wechat_uin() -> str:
        uint32 = int.from_bytes(os.urandom(4), "big")
        return base64.b64encode(str(uint32).encode()).decode()

    def _make_headers(self, *, auth: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-WECHAT-UIN": self._random_wechat_uin(),
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
        }
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if self.config.route_tag is not None and str(self.config.route_tag).strip():
            headers["SKRouteTag"] = str(self.config.route_tag).strip()
        return headers

    async def _api_get(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        assert self._client is not None
        url = f"{self.config.base_url}/{endpoint}"
        headers = self._make_headers(auth=auth)
        if extra_headers:
            headers.update(extra_headers)
        resp = await self._client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()

    async def _api_post(
        self,
        endpoint: str,
        body: dict | None = None,
        *,
        auth: bool = True,
    ) -> dict:
        assert self._client is not None
        url = f"{self.config.base_url}/{endpoint}"
        payload = body or {}
        if "base_info" not in payload:
            payload["base_info"] = BASE_INFO
        resp = await self._client.post(url, json=payload, headers=self._make_headers(auth=auth))
        resp.raise_for_status()
        return resp.json()

    async def _fetch_qr_code(self) -> tuple[str, str]:
        data = await self._api_get(
            "ilink/bot/get_bot_qrcode",
            params={"bot_type": "3"},
            auth=False,
        )
        qrcode_img_content = str(data.get("qrcode_img_content", "") or "")
        qrcode_id = str(data.get("qrcode", "") or "")
        if not qrcode_id:
            raise RuntimeError(f"Failed to get QR code from Weixin API: {data}")
        return qrcode_id, (qrcode_img_content or qrcode_id)

    async def _qr_login(self) -> bool:
        try:
            logger.info("Starting Weixin QR login...")
            refresh_count = 0
            qrcode_id, scan_url = await self._fetch_qr_code()
            self._print_qr_code(scan_url)

            logger.info("Waiting for QR scan confirmation...")
            while self._running:
                try:
                    status_data = await self._api_get(
                        "ilink/bot/get_qrcode_status",
                        params={"qrcode": qrcode_id},
                        auth=False,
                        extra_headers={"iLink-App-ClientVersion": "1"},
                    )
                except httpx.TimeoutException:
                    continue

                status = str(status_data.get("status", "") or "")
                if status == "confirmed":
                    token = str(status_data.get("bot_token", "") or "")
                    base_url = str(status_data.get("baseurl", "") or "")
                    if token:
                        self._token = token
                        if base_url:
                            self.config.base_url = base_url
                        self._save_state()
                        logger.info("Weixin QR login successful")
                        return True
                    logger.error("Login confirmed but response has no bot_token")
                    return False

                if status == "scaned":
                    logger.info("QR scanned, waiting for user confirmation...")
                elif status == "expired":
                    refresh_count += 1
                    if refresh_count > MAX_QR_REFRESH_COUNT:
                        logger.warning("QR expired too many times, aborting login")
                        return False
                    logger.warning("QR expired, refreshing ({}/{})", refresh_count, MAX_QR_REFRESH_COUNT)
                    qrcode_id, scan_url = await self._fetch_qr_code()
                    self._print_qr_code(scan_url)

                await asyncio.sleep(1)

        except Exception as exc:
            logger.error("Weixin QR login failed: {}", exc)
        return False

    @staticmethod
    def _print_qr_code(url: str) -> None:
        try:
            import qrcode as qr_lib

            qr = qr_lib.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except ImportError:
            logger.info("Install qrcode[pil] for terminal QR rendering")
            print(f"\nLogin URL: {url}\n")

    async def login(self, force: bool = False) -> bool:
        if force:
            self._token = ""
            self._get_updates_buf = ""
            state_file = self._get_state_dir() / "account.json"
            if state_file.exists():
                state_file.unlink()

        if self._token or self._load_state():
            return True

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=30),
            follow_redirects=True,
        )
        self._running = True
        try:
            return await self._qr_login()
        finally:
            self._running = False
            if self._client:
                await self._client.aclose()
                self._client = None

    async def start(self) -> None:
        self._running = True
        self._next_poll_timeout_s = self.config.poll_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._next_poll_timeout_s + 10, connect=30),
            follow_redirects=True,
        )

        if self.config.token:
            self._token = self.config.token
        elif not self._load_state():
            if not await self._qr_login():
                logger.error("Weixin login failed. Stop channel startup.")
                self._running = False
                return

        logger.info("Weixin channel started with long-poll")
        failures = 0
        while self._running:
            try:
                await self._poll_once()
                failures = 0
            except httpx.TimeoutException:
                continue
            except Exception as exc:
                if not self._running:
                    break
                failures += 1
                logger.error("Weixin poll error ({}/{}): {}", failures, MAX_CONSECUTIVE_FAILURES, exc)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    failures = 0
                    await asyncio.sleep(BACKOFF_DELAY_S)
                else:
                    await asyncio.sleep(RETRY_DELAY_S)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None
        self._save_state()
        logger.info("Weixin channel stopped")

    def _pause_session(self, duration_s: int = SESSION_PAUSE_DURATION_S) -> None:
        self._session_pause_until = time.time() + duration_s

    def _session_pause_remaining_s(self) -> int:
        remaining = int(self._session_pause_until - time.time())
        if remaining <= 0:
            self._session_pause_until = 0.0
            return 0
        return remaining

    def _assert_session_active(self) -> None:
        remaining = self._session_pause_remaining_s()
        if remaining > 0:
            remaining_min = max((remaining + 59) // 60, 1)
            raise RuntimeError(f"Weixin session paused, {remaining_min} min remaining")

    async def _poll_once(self) -> None:
        remaining = self._session_pause_remaining_s()
        if remaining > 0:
            await asyncio.sleep(remaining)
            return

        assert self._client is not None
        self._client.timeout = httpx.Timeout(self._next_poll_timeout_s + 10, connect=30)

        body: dict[str, Any] = {
            "get_updates_buf": self._get_updates_buf,
            "base_info": BASE_INFO,
        }
        data = await self._api_post("ilink/bot/getupdates", body)

        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        is_error = (ret is not None and ret != 0) or (errcode is not None and errcode != 0)
        if is_error:
            if errcode == ERRCODE_SESSION_EXPIRED or ret == ERRCODE_SESSION_EXPIRED:
                self._pause_session()
                return
            raise RuntimeError(
                f"getupdates failed: ret={ret} errcode={errcode} errmsg={data.get('errmsg', '')}"
            )

        server_timeout_ms = data.get("longpolling_timeout_ms")
        if server_timeout_ms and server_timeout_ms > 0:
            self._next_poll_timeout_s = max(int(server_timeout_ms) // 1000, 5)

        new_buf = str(data.get("get_updates_buf", "") or "")
        if new_buf:
            self._get_updates_buf = new_buf
            self._save_state()

        msgs = data.get("msgs", []) or []
        for msg in msgs:
            try:
                await self._process_message(msg)
            except Exception as exc:
                logger.error("Error processing Weixin message: {}", exc)

    async def _process_message(self, msg: dict) -> None:
        """Process one inbound message and push text payload to message bus."""
        if msg.get("message_type") == MESSAGE_TYPE_BOT:
            return

        msg_id = str(msg.get("message_id", "") or msg.get("seq", ""))
        if not msg_id:
            msg_id = f"{msg.get('from_user_id', '')}_{msg.get('create_time_ms', '')}"
        if msg_id in self._processed_ids:
            return
        self._processed_ids[msg_id] = None
        while len(self._processed_ids) > 1000:
            self._processed_ids.popitem(last=False)

        from_user_id = str(msg.get("from_user_id", "") or "")
        if not from_user_id:
            return

        ctx_token = str(msg.get("context_token", "") or "")
        if ctx_token:
            self._context_tokens[from_user_id] = ctx_token
            self._save_state()

        item_list = msg.get("item_list") or []
        content_parts: list[str] = []
        for item in item_list:
            if item.get("type", 0) != ITEM_TEXT:
                continue
            text = str((item.get("text_item") or {}).get("text", "") or "").strip()
            if text:
                content_parts.append(text)

        content = "\n".join(content_parts).strip()
        if not content:
            return

        logger.info("Weixin inbound: from={} textLen={}", from_user_id, len(content))
        await self._handle_message(
            sender_id=from_user_id,
            chat_id=from_user_id,
            content=content,
            metadata={"message_id": msg_id},
        )

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client or not self._token:
            logger.warning("Weixin client not initialized or not authenticated")
            return

        try:
            self._assert_session_active()
        except RuntimeError as exc:
            logger.warning("Weixin send blocked: {}", exc)
            return

        content = (msg.content or "").strip()
        if not content:
            return

        ctx_token = self._context_tokens.get(msg.chat_id, "")
        if not ctx_token:
            logger.warning("Weixin: no context_token for chat_id={}, cannot send", msg.chat_id)
            return

        for chunk in _split_message(content, WEIXIN_MAX_MESSAGE_LEN):
            await self._send_text(msg.chat_id, chunk, ctx_token)

    async def _send_text(self, to_user_id: str, text: str, context_token: str) -> None:
        client_id = f"pptrag-{uuid.uuid4().hex[:12]}"
        item_list = [{"type": ITEM_TEXT, "text_item": {"text": text}}] if text else []

        weixin_msg: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": to_user_id,
            "client_id": client_id,
            "message_type": MESSAGE_TYPE_BOT,
            "message_state": MESSAGE_STATE_FINISH,
            "item_list": item_list,
        }
        if context_token:
            weixin_msg["context_token"] = context_token

        body: dict[str, Any] = {
            "msg": weixin_msg,
            "base_info": BASE_INFO,
        }
        data = await self._api_post("ilink/bot/sendmessage", body)
        errcode = data.get("errcode", 0)
        if errcode and errcode != 0:
            logger.warning("Weixin send error code={} msg={}", errcode, data.get("errmsg", ""))


def _split_message(content: str, max_len: int) -> list[str]:
    if not content:
        return []
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
