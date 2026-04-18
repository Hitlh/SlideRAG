"""Minimal WhatsApp channel smoke test (receive + send echo)."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from loguru import logger

from client.utils import load_env_file
from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.whatsapp import WhatsAppChannel, WhatsAppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    load_env_file(PROJECT_ROOT / ".env")


def _parse_allow_from(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["*"]


def _build_config_from_env() -> WhatsAppConfig:
    return WhatsAppConfig(
        enabled=True,
        allow_from=_parse_allow_from(os.getenv("WHATSAPP_ALLOW_FROM", "*")),
        bridge_url=os.getenv("WHATSAPP_BRIDGE_URL", "ws://127.0.0.1:3001").strip(),
        bridge_token=os.getenv("WHATSAPP_BRIDGE_TOKEN", "").strip(),
        reconnect_delay_s=int(os.getenv("WHATSAPP_RECONNECT_DELAY_S", "5").strip() or "5"),
        send_retry_attempts=int(os.getenv("WHATSAPP_SEND_RETRY_ATTEMPTS", "3").strip() or "3"),
        send_retry_delay_ms=int(os.getenv("WHATSAPP_SEND_RETRY_DELAY_MS", "400").strip() or "400"),
        accept_group_messages=os.getenv("WHATSAPP_ACCEPT_GROUP_MESSAGES", "true").strip().lower() == "true",
        require_mention_in_group=os.getenv("WHATSAPP_REQUIRE_MENTION_IN_GROUP", "true").strip().lower() == "true",
    )


async def _consume_and_echo(bus: MessageBus, channel: WhatsAppChannel) -> None:
    while channel.is_running:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound message: channel={}, chat_id={}, sender_id={}",
            msg.channel,
            msg.chat_id,
            msg.sender_id,
        )
        await channel.send(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"[whatsapp_smoke_test] 收到：{msg.content}",
                metadata={"message_id": msg.metadata.get("message_id")},
            )
        )


async def _amain() -> None:
    _load_project_env()
    config = _build_config_from_env()
    if not config.bridge_token:
        raise SystemExit("Set WHATSAPP_BRIDGE_TOKEN in .env or shell before running whatsapp_smoke_test.")

    bus = MessageBus()
    channel = WhatsAppChannel(config=config, bus=bus)

    stop_event = asyncio.Event()

    def _on_stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    logger.info("Starting WhatsApp smoke test. Press Ctrl+C to stop.")
    logger.info(
        "allow_from={} bridge_url={} group_enabled={} require_mention={}",
        config.allow_from,
        config.bridge_url,
        config.accept_group_messages,
        config.require_mention_in_group,
    )

    channel_task = asyncio.create_task(channel.start())
    echo_task = asyncio.create_task(_consume_and_echo(bus, channel))

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping WhatsApp smoke test...")
        await channel.stop()
        channel_task.cancel()
        echo_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(channel_task, echo_task, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown timed out after 5s; force exiting remaining tasks")
        logger.info("WhatsApp smoke test stopped")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
