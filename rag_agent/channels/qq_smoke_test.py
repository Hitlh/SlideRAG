"""Minimal QQ channel smoke test (receive + send echo)."""

from __future__ import annotations

import asyncio
import os
import signal

from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.qq import QQChannel, QQConfig


def _build_config_from_env() -> QQConfig:
    """Build QQ test config from environment variables."""
    allow_from_raw = os.getenv("QQ_ALLOW_FROM", "*")
    allow_from = [item.strip() for item in allow_from_raw.split(",") if item.strip()]
    if not allow_from:
        allow_from = ["*"]

    msg_format = os.getenv("QQ_MSG_FORMAT", "plain").strip().lower()
    if msg_format not in ("plain", "markdown"):
        msg_format = "plain"

    return QQConfig(
        enabled=True,
        app_id=os.getenv("QQ_APP_ID", "").strip(),
        secret=os.getenv("QQ_SECRET", "").strip(),
        allow_from=allow_from,
        msg_format=msg_format,
    )


async def _consume_and_echo(bus: MessageBus, channel: QQChannel) -> None:
    """Consume inbound QQ messages and send an echo reply."""
    while channel.is_running:
        msg = await bus.consume_inbound()
        logger.info("Inbound message: channel={}, chat_id={}, sender_id={}", msg.channel, msg.chat_id, msg.sender_id)
        reply = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"[qq_smoke_test] 收到：{msg.content}",
            metadata={"message_id": msg.metadata.get("message_id")},
        )
        await channel.send(reply)


async def main() -> None:
    """Run QQ channel smoke test until Ctrl+C."""
    config = _build_config_from_env()
    if not config.app_id or not config.secret:
        raise SystemExit("Set QQ_APP_ID and QQ_SECRET before running qq_smoke_test.")

    bus = MessageBus()
    channel = QQChannel(config=config, bus=bus)

    stop_event = asyncio.Event()

    def _on_stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    logger.info("Starting QQ smoke test. Press Ctrl+C to stop.")
    logger.info("allow_from={} msg_format={}", config.allow_from, config.msg_format)

    channel_task = asyncio.create_task(channel.start())
    echo_task = asyncio.create_task(_consume_and_echo(bus, channel))

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping QQ smoke test...")
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
        logger.info("QQ smoke test stopped")


if __name__ == "__main__":
    asyncio.run(main())
