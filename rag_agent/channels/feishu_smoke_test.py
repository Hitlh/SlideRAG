"""Minimal Feishu channel smoke test (single-chat text echo)."""

from __future__ import annotations

import asyncio
import os
import signal
import warnings
from pathlib import Path

from loguru import logger

from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus

from client.utils import load_env_file

# Suppress a known warning from lark-oapi vendored protobuf namespace packages.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from rag_agent.channels.feishu import FeishuChannel, FeishuConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    """Load project-level .env so direct module runs can read local env vars."""
    load_env_file(PROJECT_ROOT / ".env")


def _first_nonempty_env(*names: str) -> str:
    """Return the first non-empty environment value from aliases."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _parse_allow_from(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["*"]


def _build_config_from_env() -> FeishuConfig:
    """Build Feishu test config from environment variables."""
    domain = _first_nonempty_env("FEISHU_DOMAIN", "LARK_DOMAIN") or "feishu"
    domain = domain.strip().lower()
    if domain not in ("feishu", "lark"):
        domain = "feishu"

    return FeishuConfig(
        enabled=True,
        app_id=_first_nonempty_env("FEISHU_APP_ID", "FEISHU_APPID"),
        app_secret=_first_nonempty_env("FEISHU_APP_SECRET", "FEISHU_APPSECRET"),
        encrypt_key=_first_nonempty_env("FEISHU_ENCRYPT_KEY", "FEISHU_ENCRYPTKEY"),
        verification_token=_first_nonempty_env(
            "FEISHU_VERIFICATION_TOKEN", "FEISHU_VERIFICATIONTOKEN"
        ),
        allow_from=_parse_allow_from(os.getenv("FEISHU_ALLOW_FROM", "*")),
        domain=domain,
    )


async def _consume_and_echo(bus: MessageBus, channel: FeishuChannel) -> None:
    """Consume inbound Feishu messages and send text echo replies."""
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
                content=f"[feishu_smoke_test] 收到：{msg.content}",
                metadata={"message_id": msg.metadata.get("message_id")},
            )
        )


async def _amain() -> None:
    """Run Feishu smoke test until Ctrl+C."""
    _load_project_env()
    config = _build_config_from_env()
    if not config.app_id or not config.app_secret:
        raise SystemExit(
            "Missing Feishu credentials. Set FEISHU_APP_ID and FEISHU_APP_SECRET in shell or .env. "
            "Also supports FEISHU_APPID / FEISHU_APPSECRET aliases."
        )

    bus = MessageBus()
    channel = FeishuChannel(config=config, bus=bus)

    stop_event = asyncio.Event()

    def _on_stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    logger.info("Starting Feishu smoke test. Press Ctrl+C to stop.")
    logger.info("allow_from={} domain={}", config.allow_from, config.domain)

    channel_task = asyncio.create_task(channel.start())
    echo_task = asyncio.create_task(_consume_and_echo(bus, channel))

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping Feishu smoke test...")
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
        logger.info("Feishu smoke test stopped")


def main() -> None:
    """Console entrypoint."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()