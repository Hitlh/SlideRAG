"""Standalone Weixin runtime worker.

Modes:
- echo: text loopback testing
- agent: route inbound messages into AgentLoop and send model responses
"""

from __future__ import annotations

import asyncio
import argparse
import signal
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from client.runtime_common_shared import build_provider, build_rag_instance
from client.weixin_runtime_common import load_settings
from client.weixin_runtime_common import (
    SWITCH_EXIT_CODE,
    WeixinRuntimeSettings,
    build_session_key,
    parse_switch_command,
    resolve_effective_rag_working_dir,
    resolve_uploaded_file,
    write_switch_request,
)
from rag_agent.agent.loop import AgentLoop
from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.weixin import WeixinChannel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RuntimeSwitchState:
    """Cross-task switch state used by inbound worker and runtime main."""

    event: object
    target_file_path: Path | None = None
    trigger_chat_id: str = ""


async def inbound_worker(bus: MessageBus, echo_prefix: str) -> None:
    """Consume inbound messages and push echo responses."""
    while True:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound Weixin message: chat_id={}, sender_id={}, len={}",
            msg.chat_id,
            msg.sender_id,
            len(msg.content or ""),
        )
        text = (msg.content or "").strip()
        if not text:
            continue

        await bus.publish_outbound(
            OutboundMessage(
                channel="weixin",
                chat_id=msg.chat_id,
                content=f"{echo_prefix}{text}",
                metadata={"message_id": msg.metadata.get("message_id")},
            )
        )


async def inbound_agent_worker(bus: MessageBus, agent_loop: AgentLoop, settings) -> None:
    """Consume inbound queue, run agent loop, and publish outbound responses."""
    while True:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound Weixin message: chat_id={}, sender_id={}, len={}",
            msg.chat_id,
            msg.sender_id,
            len(msg.content or ""),
        )
        try:
            session_key = build_session_key(
                channel=msg.channel,
                chat_id=msg.chat_id,
                file_path=settings.target_file_path,
                parse_method=settings.parse_method,
            )
            result = await agent_loop.process_message(
                user_message=msg.content,
                channel=msg.channel,
                chat_id=msg.chat_id,
                session_key=session_key,
                file_path=settings.target_file_path,
                parse_method=settings.parse_method,
            )
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=result.final_answer,
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )
        except Exception as exc:
            logger.exception("Failed to process inbound Weixin message: {}", exc)
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Error while processing message: {exc}",
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )


async def handle_switch_command(
    bus: MessageBus,
    settings: WeixinRuntimeSettings,
    switch_state: RuntimeSwitchState,
    channel: str,
    chat_id: str,
    message_id: str,
    argument: str,
) -> None:
    """Handle '/file <name>' command and schedule worker restart when valid."""
    if not argument:
        await bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="Usage: /file <filename> (e.g., /file example1.pdf)",
                metadata={"message_id": message_id},
            )
        )
        return

    resolved = resolve_uploaded_file(settings, argument)
    if resolved is None:
        await bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=(
                    f"File not found: {argument}. "
                    f"Please place the file in: {settings.uploaded_docs_dir}"
                ),
                metadata={"message_id": message_id},
            )
        )
        return

    if settings.target_file_path is not None and resolved == settings.target_file_path.resolve():
        await bus.publish_outbound(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=f"This file is already in use: {resolved.name}",
                metadata={"message_id": message_id},
            )
        )
        return

    await bus.publish_outbound(
        OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=f"Received. Switching to file: {resolved.name}. Please wait...",
            metadata={"message_id": message_id},
        )
    )
    switch_state.target_file_path = resolved
    switch_state.trigger_chat_id = chat_id
    switch_state.event.set()


async def inbound_agent_worker_with_switch(
    bus: MessageBus,
    agent_loop: AgentLoop,
    settings: WeixinRuntimeSettings,
    switch_state: RuntimeSwitchState,
) -> None:
    """Agent inbound worker with /file switching command support."""
    while True:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound Weixin message: chat_id={}, sender_id={}, len={}",
            msg.chat_id,
            msg.sender_id,
            len(msg.content or ""),
        )

        switch_arg = parse_switch_command(msg.content)
        if switch_arg is not None:
            await handle_switch_command(
                bus=bus,
                settings=settings,
                switch_state=switch_state,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_id=str(msg.metadata.get("message_id", "")),
                argument=switch_arg,
            )
            continue

        try:
            session_key = build_session_key(
                channel=msg.channel,
                chat_id=msg.chat_id,
                file_path=settings.target_file_path,
                parse_method=settings.parse_method,
            )
            result = await agent_loop.process_message(
                user_message=msg.content,
                channel=msg.channel,
                chat_id=msg.chat_id,
                session_key=session_key,
                file_path=settings.target_file_path,
                parse_method=settings.parse_method,
            )
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=result.final_answer,
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )
        except Exception as exc:
            logger.exception("Failed to process inbound Weixin message: {}", exc)
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Error while processing message: {exc}",
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )


async def outbound_worker(bus: MessageBus, channel: WeixinChannel) -> None:
    """Consume outbound queue and send through Weixin channel."""
    while True:
        msg = await bus.consume_outbound()
        await channel.send(msg)


async def maybe_ingest_target_file(rag: object, settings) -> None:
    """Ingest configured target file before runtime starts."""
    if settings.target_file_path is None:
        logger.warning("WEIXIN_TARGET_FILE is empty; skipping ingest")
        return

    target = settings.target_file_path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"WEIXIN_TARGET_FILE does not exist: {target}")

    settings.ingest_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting target document before runtime: {}", target)
    await rag.process_document_complete_with_page_topics(
        file_path=str(target),
        output_dir=str(settings.ingest_output_dir),
        parse_method=settings.parse_method,
    )
    logger.info("Document ingest completed: {}", target)


def _startup_notify_targets(settings: WeixinRuntimeSettings) -> list[str]:
    if settings.startup_notify_chat_id:
        return [settings.startup_notify_chat_id]
    return [item for item in settings.weixin_config.allow_from if item and item != "*"]


async def wait_weixin_channel_ready(channel: WeixinChannel, timeout_s: float = 20.0) -> bool:
    """Wait until Weixin channel has initialized client and auth token."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if getattr(channel, "_client", None) is not None and bool(getattr(channel, "_token", "")):
            return True
        await asyncio.sleep(0.2)
    return False


async def send_startup_ready_notification(channel: WeixinChannel, settings: WeixinRuntimeSettings) -> None:
    """Best-effort startup message to Weixin targets when context tokens are available."""
    if not settings.startup_notify_enabled:
        return

    if not await wait_weixin_channel_ready(channel):
        logger.warning("Startup notification skipped: Weixin channel is not ready yet")
        return

    targets = _startup_notify_targets(settings)
    if not targets:
        logger.warning(
            "Startup notification skipped: configure WEIXIN_STARTUP_NOTIFY_CHAT_ID or explicit WEIXIN_ALLOW_FROM"
        )
        return

    sent = 0
    for target in targets:
        if not getattr(channel, "_context_tokens", {}).get(target):
            logger.warning(
                "Startup notification skipped for {}: missing context_token (send one message first)",
                target,
            )
            continue
        try:
            await channel.send(
                OutboundMessage(
                    channel="weixin",
                    chat_id=target,
                    content=settings.startup_notify_message,
                    metadata={},
                )
            )
            sent += 1
        except Exception as exc:
            logger.warning("Startup notification to {} failed: {}", target, exc)

    if sent:
        logger.info("Startup notification sent to {} target(s)", sent)
    else:
        logger.warning(
            "Startup notification not delivered. Ensure target chat has context token (send one message first)."
        )


async def worker_main(
    target_file_override: str | None = None,
    force_relogin: bool = False,
) -> int:
    """Start Weixin runtime and return process exit code."""
    settings = load_settings(target_file_override=target_file_override)
    if not settings.weixin_config.enabled:
        raise SystemExit("WEIXIN_ENABLED is false; nothing to run")

    effective_working_dir = resolve_effective_rag_working_dir(settings)
    settings = replace(settings, rag_working_dir=effective_working_dir)

    provider = build_provider(settings)
    rag = build_rag_instance(settings)
    await maybe_ingest_target_file(rag, settings)

    agent_workspace = settings.rag_working_dir / "agent_loop_workspace"
    agent_workspace.mkdir(parents=True, exist_ok=True)
    agent_loop = AgentLoop(
        provider=provider,
        workspace=agent_workspace,
        rag=rag,
        model=settings.agent_model,
        retrieve_config={
            "mode": "hybrid",
            "top_k": settings.retrieve_top_k,
            "chunk_top_k": settings.retrieve_chunk_top_k,
        },
    )

    bus = MessageBus()
    weixin_channel = WeixinChannel(config=settings.weixin_config, bus=bus)

    if force_relogin:
        logger.info("Force relogin enabled, clearing cached session and logging in again")
        relogin_ok = await weixin_channel.login(force=True)
        if not relogin_ok:
            logger.error("Force relogin failed. Stop worker startup.")
            return 1

    stop_event = asyncio.Event()
    switch_state = RuntimeSwitchState(event=asyncio.Event())

    def _on_stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    channel_task = asyncio.create_task(weixin_channel.start())
    if settings.runtime_mode == "echo":
        in_task = asyncio.create_task(inbound_worker(bus, settings.echo_prefix))
    else:
        in_task = asyncio.create_task(
            inbound_agent_worker_with_switch(bus, agent_loop, settings, switch_state)
        )
    out_task = asyncio.create_task(outbound_worker(bus, weixin_channel))
    await send_startup_ready_notification(weixin_channel, settings)

    logger.info("Weixin runtime started (mode={})", settings.runtime_mode)
    logger.info("allow_from={}", settings.weixin_config.allow_from)
    logger.info("state_dir={}", settings.weixin_config.state_dir)
    logger.info("rag_working_dir={}", settings.rag_working_dir)
    logger.info("target_file_path={}", settings.target_file_path)
    logger.info("ingest_output_dir={}", settings.ingest_output_dir)
    logger.info("agent_provider={} agent_model={}", settings.agent_provider, settings.agent_model)

    exit_code = 0
    stop_wait = asyncio.create_task(stop_event.wait())
    switch_wait = asyncio.create_task(switch_state.event.wait())

    try:
        done, pending = await asyncio.wait(
            {stop_wait, switch_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if switch_wait in done and switch_state.target_file_path is not None:
            write_switch_request(
                settings=settings,
                target_file_path=switch_state.target_file_path,
                trigger_chat_id=switch_state.trigger_chat_id,
            )
            await asyncio.sleep(0.8)
            exit_code = SWITCH_EXIT_CODE
    finally:
        logger.info("Stopping Weixin runtime...")
        await weixin_channel.stop()
        for task in (channel_task, in_task, out_task):
            task.cancel()
        await asyncio.gather(channel_task, in_task, out_task, return_exceptions=True)
        logger.info("Weixin runtime stopped")

    return exit_code


def build_worker_cli_args() -> argparse.Namespace:
    """Parse CLI args for direct worker execution."""
    parser = argparse.ArgumentParser(description="Weixin runtime worker process")
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    parser.add_argument("-r", "--relogin", action="store_true", help="Force Weixin relogin before startup")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_worker_cli_args()
    raise SystemExit(
        asyncio.run(
            worker_main(
                target_file_override=args.target_file or None,
                force_relogin=bool(args.relogin),
            )
        )
    )
