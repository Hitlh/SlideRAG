"""Shared runtime core for channel worker/supervisor/launcher flows."""

from __future__ import annotations

import argparse
import asyncio
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from rag_agent.agent.loop import AgentLoop
from rag_agent.bus.events import OutboundMessage
from rag_agent.bus.queue import MessageBus


@dataclass(frozen=True)
class WorkerHooks:
    """Channel-specific hooks used by shared worker core."""

    channel_name: str
    channel_display_name: str
    disabled_flag_env: str
    switch_exit_code: int
    missing_credentials_message: str
    target_file_env_var: str
    startup_notify_missing_targets_message: str

    load_settings: Callable[..., Any]
    is_enabled: Callable[[Any], bool]
    has_required_credentials: Callable[[Any], bool]

    build_provider: Callable[[Any], Any]
    build_rag_instance: Callable[[Any], Any]
    resolve_effective_rag_working_dir: Callable[[Any], Path]
    build_session_key: Callable[[str, str, Path | None, str], str]
    parse_switch_command: Callable[[str], str | None]
    resolve_uploaded_file: Callable[[Any, str], Path | None]
    write_switch_request: Callable[..., None]

    create_switch_state: Callable[[], Any]
    build_channel: Callable[[Any, MessageBus], Any]
    startup_notify_targets: Callable[[Any], list[str]]
    wait_until_ready: Callable[[Any], Awaitable[bool]]
    before_channel_start: Callable[[Any, Any], Awaitable[None]] | None = None
    custom_startup_notify: Callable[[Any, Any], Awaitable[None]] | None = None


@dataclass(frozen=True)
class SupervisorHooks:
    """Channel-specific hooks used by shared supervisor core."""

    channel_display_name: str
    worker_module: str
    project_root: Path

    load_settings: Callable[..., Any]
    switch_exit_code: int
    switch_request_path: Callable[[Any], Path]
    read_switch_request: Callable[[Any], tuple[Path, str] | None]
    build_extra_worker_args: Callable[[int], list[str]] | None = None


def build_runtime_cli_args(
    description: str,
    extra_arg_builder: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Build standard runtime CLI args used by QQ/Feishu launchers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--worker", action="store_true", help="Run worker mode directly")
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    if extra_arg_builder is not None:
        extra_arg_builder(parser)
    return parser.parse_args()


def build_worker_cli_args(
    description: str,
    extra_arg_builder: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    """Build standard direct-worker CLI args used by QQ/Feishu workers."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    if extra_arg_builder is not None:
        extra_arg_builder(parser)
    return parser.parse_args()


def runtime_main_core(
    *,
    description: str,
    worker_main: Callable[..., Awaitable[int]],
    supervisor_main: Callable[..., int],
    extra_arg_builder: Callable[[argparse.ArgumentParser], None] | None = None,
    worker_kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    supervisor_kwargs_builder: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> int:
    """Dispatch runtime process to supervisor or worker mode."""
    args = build_runtime_cli_args(description, extra_arg_builder=extra_arg_builder)
    worker_kwargs = worker_kwargs_builder(args) if worker_kwargs_builder is not None else {}
    supervisor_kwargs = (
        supervisor_kwargs_builder(args) if supervisor_kwargs_builder is not None else {}
    )
    if args.worker:
        return int(
            asyncio.run(
                worker_main(
                    target_file_override=args.target_file or None,
                    **worker_kwargs,
                )
            )
        )
    return int(supervisor_main(**supervisor_kwargs))


async def _handle_switch_command(
    *,
    bus: MessageBus,
    settings: Any,
    switch_state: Any,
    hooks: WorkerHooks,
    channel: str,
    chat_id: str,
    message_id: str,
    argument: str,
) -> None:
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

    resolved = hooks.resolve_uploaded_file(settings, argument)
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


async def _inbound_worker(
    *,
    bus: MessageBus,
    agent_loop: AgentLoop,
    settings: Any,
    switch_state: Any,
    hooks: WorkerHooks,
) -> None:
    while True:
        msg = await bus.consume_inbound()
        logger.info(
            "Inbound message: channel={}, chat_id={}, sender_id={}",
            msg.channel,
            msg.chat_id,
            msg.sender_id,
        )
        try:
            switch_arg = hooks.parse_switch_command(msg.content)
            if switch_arg is not None:
                await _handle_switch_command(
                    bus=bus,
                    settings=settings,
                    switch_state=switch_state,
                    hooks=hooks,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    message_id=str(msg.metadata.get("message_id", "")),
                    argument=switch_arg,
                )
                continue

            session_key = hooks.build_session_key(
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
            logger.exception("Failed to process inbound {} message: {}", hooks.channel_display_name, exc)
            await bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"Error while processing message: {exc}",
                    metadata={"message_id": msg.metadata.get("message_id")},
                )
            )


async def _outbound_worker(*, bus: MessageBus, channel: Any) -> None:
    while True:
        msg = await bus.consume_outbound()
        await channel.send(msg)


async def _send_startup_ready_notification(*, channel: Any, settings: Any, hooks: WorkerHooks) -> None:
    if hooks.custom_startup_notify is not None:
        await hooks.custom_startup_notify(channel, settings)
        return

    if not settings.startup_notify_enabled:
        return

    targets = hooks.startup_notify_targets(settings)
    if not targets:
        logger.warning(hooks.startup_notify_missing_targets_message)
        return

    try:
        if not await hooks.wait_until_ready(channel):
            logger.warning(
                "Startup notification skipped: {} client not ready within timeout",
                hooks.channel_display_name,
            )
            return

        for target in targets:
            await channel.send(
                OutboundMessage(
                    channel=hooks.channel_name,
                    chat_id=target,
                    content=settings.startup_notify_message,
                    metadata={},
                )
            )
        logger.info("Startup notification sent to {} target(s)", len(targets))
    except Exception as exc:
        logger.exception("Failed to send startup notification: {}", exc)


async def _maybe_ingest_target_file(*, rag: Any, settings: Any, hooks: WorkerHooks) -> None:
    if settings.target_file_path is None:
        logger.warning("{} is empty; skipping ingest", hooks.target_file_env_var)
        return

    target = settings.target_file_path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"{hooks.target_file_env_var} does not exist: {target}")

    settings.ingest_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting target document before runtime: {}", target)

    await rag.process_document_complete_with_page_topics(
        file_path=str(target),
        output_dir=str(settings.ingest_output_dir),
        parse_method=settings.parse_method,
    )

    logger.info("Document ingest completed: {}", target)


async def worker_main_core(*, target_file_override: str | None = None, hooks: WorkerHooks) -> int:
    """Shared worker loop for channel runtimes."""
    settings = hooks.load_settings(target_file_override=target_file_override)
    if not hooks.is_enabled(settings):
        raise SystemExit(f"{hooks.disabled_flag_env} is false; nothing to run")
    if not hooks.has_required_credentials(settings):
        raise SystemExit(hooks.missing_credentials_message)

    effective_working_dir = hooks.resolve_effective_rag_working_dir(settings)
    settings = replace(settings, rag_working_dir=effective_working_dir)

    provider = hooks.build_provider(settings)
    rag = hooks.build_rag_instance(settings)
    await _maybe_ingest_target_file(rag=rag, settings=settings, hooks=hooks)

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
    channel = hooks.build_channel(settings, bus)

    if hooks.before_channel_start is not None:
        await hooks.before_channel_start(channel, settings)

    logger.info("{} runtime(worker) starting", hooks.channel_display_name)
    logger.info("rag_working_dir={}", settings.rag_working_dir)
    logger.info("target_file_path={}", settings.target_file_path)
    logger.info("ingest_output_dir={}", settings.ingest_output_dir)
    logger.info("agent_provider={} agent_model={}", settings.agent_provider, settings.agent_model)
    logger.info("uploaded_docs_dir={}", settings.uploaded_docs_dir)
    logger.info("runtime_state_dir={}", settings.runtime_state_dir)

    stop_event = asyncio.Event()
    switch_state = hooks.create_switch_state()

    def _on_stop(*_args: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    channel_task = asyncio.create_task(channel.start())
    in_task = asyncio.create_task(
        _inbound_worker(
            bus=bus,
            agent_loop=agent_loop,
            settings=settings,
            switch_state=switch_state,
            hooks=hooks,
        )
    )
    out_task = asyncio.create_task(_outbound_worker(bus=bus, channel=channel))
    await _send_startup_ready_notification(channel=channel, settings=settings, hooks=hooks)

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
            hooks.write_switch_request(
                settings=settings,
                target_file_path=switch_state.target_file_path,
                trigger_chat_id=switch_state.trigger_chat_id,
            )
            await asyncio.sleep(0.8)
            exit_code = hooks.switch_exit_code
    finally:
        logger.info("Stopping {} runtime(worker)...", hooks.channel_display_name)
        await channel.stop()
        for task in (channel_task, in_task, out_task):
            task.cancel()
        await asyncio.gather(channel_task, in_task, out_task, return_exceptions=True)
        logger.info("{} runtime(worker) stopped", hooks.channel_display_name)
    return exit_code


def _run_worker_process(
    worker_module: str,
    target_file_path: Path | None,
    project_root: Path,
    extra_args: list[str] | None = None,
) -> int:
    cmd = [sys.executable, "-m", worker_module]
    if target_file_path is not None:
        cmd.extend(["--target-file", str(target_file_path)])
    if extra_args:
        cmd.extend(extra_args)

    child = subprocess.Popen(cmd, cwd=str(project_root))
    try:
        return child.wait()
    except KeyboardInterrupt:
        logger.info("Supervisor interrupted, stopping worker...")
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
        return 130


def supervisor_main_core(*, hooks: SupervisorHooks) -> int:
    """Shared supervisor loop for channel runtimes."""
    settings = hooks.load_settings()
    current_target = settings.target_file_path
    worker_start_index = 0

    while True:
        req_path = hooks.switch_request_path(settings)
        req_path.unlink(missing_ok=True)

        logger.info(
            "Supervisor starting {} worker with target_file_path={}",
            hooks.channel_display_name,
            current_target,
        )
        extra_args = (
            hooks.build_extra_worker_args(worker_start_index)
            if hooks.build_extra_worker_args is not None
            else []
        )
        exit_code = _run_worker_process(
            hooks.worker_module,
            current_target,
            hooks.project_root,
            extra_args=extra_args,
        )
        worker_start_index += 1

        if exit_code != hooks.switch_exit_code:
            logger.info("Worker exited with code {}", exit_code)
            return exit_code

        try:
            request = hooks.read_switch_request(settings)
        except Exception as exc:
            logger.error("Worker requested switch but request file parse failed: {}", exc)
            return 1

        if request is None:
            logger.error("Worker requested switch but request file is missing/invalid")
            return 1

        next_target, trigger_chat_id = request
        logger.info(
            "Supervisor accepted switch request: target={} trigger_chat_id={}",
            next_target,
            trigger_chat_id,
        )
        current_target = next_target
