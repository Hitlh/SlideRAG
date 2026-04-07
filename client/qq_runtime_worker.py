"""Worker process for QQ runtime.

This module hosts QQ channel loops and exits with SWITCH_EXIT_CODE when /file switch is requested.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.qq_runtime_common import (
    SWITCH_EXIT_CODE,
    RuntimeSettings,
    RuntimeSwitchState,
    build_provider,
    build_rag_instance,
    build_session_key,
    load_settings,
    parse_switch_command,
    resolve_effective_rag_working_dir,
    resolve_uploaded_file,
    write_switch_request,
)
from rag_agent.agent.loop import AgentLoop
from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.qq import QQChannel


async def handle_switch_command(
    bus: MessageBus,
    settings: RuntimeSettings,
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


async def inbound_worker(
    bus: MessageBus,
    agent_loop: AgentLoop,
    settings: RuntimeSettings,
    switch_state: RuntimeSwitchState,
) -> None:
    """Consume inbound queue, run agent loop, and publish outbound responses."""
    while True:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound message: channel={}, chat_id={}, sender_id={}",
            msg.channel,
            msg.chat_id,
            msg.sender_id,
        )
        try:
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
            logger.exception("Failed to process inbound QQ message: {}", exc)
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Error while processing message: {exc}",
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )


async def outbound_worker(bus: MessageBus, channel: QQChannel) -> None:
    """Consume outbound queue and deliver through QQ channel."""
    while True:
        msg = await bus.consume_outbound()
        await channel.send(msg)


def _startup_notify_targets(settings: RuntimeSettings) -> list[str]:
    if settings.startup_notify_chat_id:
        return [settings.startup_notify_chat_id]
    return [item for item in settings.qq_config.allow_from if item and item != "*"]


async def send_startup_ready_notification(channel: QQChannel, settings: RuntimeSettings) -> None:
    """Send one proactive runtime-ready message after QQ client reports ready."""
    if not settings.startup_notify_enabled:
        return

    targets = _startup_notify_targets(settings)
    if not targets:
        logger.warning(
            "Startup notification skipped: configure startup_notify_chat_id or explicit qq.allow_from"
        )
        return

    try:
        for _ in range(30):
            # botpy sets `robot` after successful on_ready; wait until then.
            if channel._client is not None and getattr(channel._client, "robot", None) is not None:
                break
            await asyncio.sleep(1)
        else:
            logger.warning("Startup notification skipped: QQ client not ready within timeout")
            return

        for target in targets:
            await channel.send(
                OutboundMessage(
                    channel="qq",
                    chat_id=target,
                    content=settings.startup_notify_message,
                    metadata={},
                )
            )
        logger.info("Startup notification sent to {} target(s)", len(targets))
    except Exception as exc:
        logger.exception("Failed to send startup notification: {}", exc)


async def maybe_ingest_target_file(rag: object, settings: RuntimeSettings) -> None:
    """Ingest configured target file before runtime starts."""
    if settings.target_file_path is None:
        logger.warning("QQ_TARGET_FILE is empty; skipping ingest")
        return

    target = settings.target_file_path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"QQ_TARGET_FILE does not exist: {target}")

    settings.ingest_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting target document before runtime: {}", target)

    await rag.process_document_complete_with_page_topics(
        file_path=str(target),
        output_dir=str(settings.ingest_output_dir),
        parse_method=settings.parse_method,
    )

    logger.info("Document ingest completed: {}", target)


async def worker_main(target_file_override: str | None = None) -> int:
    """Start QQ worker and return process exit code."""
    settings = load_settings(target_file_override=target_file_override)
    if not settings.qq_config.enabled:
        raise SystemExit("QQ_ENABLED is false; nothing to run")
    if not settings.qq_config.app_id or not settings.qq_config.secret:
        raise SystemExit("QQ_APP_ID and QQ_SECRET are required")

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
    qq_channel = QQChannel(config=settings.qq_config, bus=bus)

    logger.info("QQ runtime(worker) starting")
    logger.info("rag_working_dir={}", settings.rag_working_dir)
    logger.info("target_file_path={}", settings.target_file_path)
    logger.info("ingest_output_dir={}", settings.ingest_output_dir)
    logger.info("agent_provider={} agent_model={}", settings.agent_provider, settings.agent_model)
    logger.info("uploaded_docs_dir={}", settings.uploaded_docs_dir)
    logger.info("runtime_state_dir={}", settings.runtime_state_dir)

    stop_event = asyncio.Event()
    switch_state = RuntimeSwitchState(event=asyncio.Event())

    def _on_stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    qq_task = asyncio.create_task(qq_channel.start())
    in_task = asyncio.create_task(inbound_worker(bus, agent_loop, settings, switch_state))
    out_task = asyncio.create_task(outbound_worker(bus, qq_channel))
    await send_startup_ready_notification(qq_channel, settings)

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
            # Give outbound worker a brief window to flush the switch acknowledgement.
            await asyncio.sleep(0.8)
            exit_code = SWITCH_EXIT_CODE
    finally:
        logger.info("Stopping QQ runtime(worker)...")
        await qq_channel.stop()
        for task in (qq_task, in_task, out_task):
            task.cancel()
        await asyncio.gather(qq_task, in_task, out_task, return_exceptions=True)
        logger.info("QQ runtime(worker) stopped")
    return exit_code


def build_worker_cli_args() -> argparse.Namespace:
    """Parse CLI args for direct worker execution."""
    parser = argparse.ArgumentParser(description="QQ runtime worker process")
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_worker_cli_args()
    raise SystemExit(asyncio.run(worker_main(target_file_override=args.target_file or None)))
