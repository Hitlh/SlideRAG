"""Worker process for QQ runtime."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.channel_runtime_core import WorkerHooks, build_worker_cli_args, worker_main_core
from client.qq.common import (
    SWITCH_EXIT_CODE,
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
from rag_agent.bus.queue import MessageBus
from rag_agent.channels.qq import QQChannel


def _startup_notify_targets(settings) -> list[str]:
    if settings.startup_notify_chat_id:
        return [settings.startup_notify_chat_id]
    return [item for item in settings.qq_config.allow_from if item and item != "*"]


async def _wait_until_qq_ready(channel: QQChannel) -> bool:
    for _ in range(30):
        if channel._client is not None and getattr(channel._client, "robot", None) is not None:
            return True
        await asyncio.sleep(1)
    return False


def _create_switch_state() -> RuntimeSwitchState:
    return RuntimeSwitchState(event=asyncio.Event())


def _build_channel(settings, bus: MessageBus) -> QQChannel:
    return QQChannel(config=settings.qq_config, bus=bus)


QQ_WORKER_HOOKS = WorkerHooks(
    channel_name="qq",
    channel_display_name="QQ",
    disabled_flag_env="QQ_ENABLED",
    switch_exit_code=SWITCH_EXIT_CODE,
    missing_credentials_message="QQ_APP_ID and QQ_SECRET are required",
    target_file_env_var="QQ_TARGET_FILE",
    startup_notify_missing_targets_message=(
        "Startup notification skipped: configure startup_notify_chat_id or explicit qq.allow_from"
    ),
    load_settings=load_settings,
    is_enabled=lambda settings: bool(settings.qq_config.enabled),
    has_required_credentials=lambda settings: bool(settings.qq_config.app_id and settings.qq_config.secret),
    build_provider=build_provider,
    build_rag_instance=build_rag_instance,
    resolve_effective_rag_working_dir=resolve_effective_rag_working_dir,
    build_session_key=build_session_key,
    parse_switch_command=parse_switch_command,
    resolve_uploaded_file=resolve_uploaded_file,
    write_switch_request=write_switch_request,
    create_switch_state=_create_switch_state,
    build_channel=_build_channel,
    startup_notify_targets=_startup_notify_targets,
    wait_until_ready=_wait_until_qq_ready,
)


async def worker_main(target_file_override: str | None = None) -> int:
    """Start QQ worker and return process exit code."""
    return await worker_main_core(target_file_override=target_file_override, hooks=QQ_WORKER_HOOKS)


if __name__ == "__main__":
    args = build_worker_cli_args("QQ runtime worker process")
    raise SystemExit(asyncio.run(worker_main(target_file_override=args.target_file or None)))
