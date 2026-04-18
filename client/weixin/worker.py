"""Standalone Weixin runtime worker."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from client.channel_runtime_core import WorkerHooks, build_worker_cli_args, worker_main_core
from client.runtime_common_shared import build_provider, build_rag_instance
from client.weixin.common import (
    SWITCH_EXIT_CODE,
    WeixinRuntimeSettings,
    build_session_key,
    load_settings,
    parse_switch_command,
    resolve_effective_rag_working_dir,
    resolve_uploaded_file,
    write_switch_request,
)
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


def _startup_notify_targets(settings: WeixinRuntimeSettings) -> list[str]:
    if settings.startup_notify_chat_id:
        return [settings.startup_notify_chat_id]
    return [item for item in settings.weixin_config.allow_from if item and item != "*"]


async def _wait_weixin_channel_ready(channel: WeixinChannel, timeout_s: float = 20.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if getattr(channel, "_client", None) is not None and bool(getattr(channel, "_token", "")):
            return True
        await asyncio.sleep(0.2)
    return False


async def _send_weixin_startup_ready_notification(
    channel: WeixinChannel,
    settings: WeixinRuntimeSettings,
) -> None:
    if not settings.startup_notify_enabled:
        return

    if not await _wait_weixin_channel_ready(channel):
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


def _create_switch_state() -> RuntimeSwitchState:
    return RuntimeSwitchState(event=asyncio.Event())


def _build_channel(settings: WeixinRuntimeSettings, bus: MessageBus) -> WeixinChannel:
    return WeixinChannel(config=settings.weixin_config, bus=bus)


async def _force_relogin_before_start(channel: WeixinChannel, _settings: WeixinRuntimeSettings) -> None:
    logger.info("Force relogin enabled, clearing cached session and logging in again")
    relogin_ok = await channel.login(force=True)
    if not relogin_ok:
        raise SystemExit("Force relogin failed. Stop worker startup.")


WEIXIN_WORKER_HOOKS = WorkerHooks(
    channel_name="weixin",
    channel_display_name="Weixin",
    disabled_flag_env="WEIXIN_ENABLED",
    switch_exit_code=SWITCH_EXIT_CODE,
    missing_credentials_message="",
    target_file_env_var="WEIXIN_TARGET_FILE",
    startup_notify_missing_targets_message=(
        "Startup notification skipped: configure WEIXIN_STARTUP_NOTIFY_CHAT_ID or explicit WEIXIN_ALLOW_FROM"
    ),
    load_settings=load_settings,
    is_enabled=lambda settings: bool(settings.weixin_config.enabled),
    has_required_credentials=lambda _settings: True,
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
    wait_until_ready=_wait_weixin_channel_ready,
    custom_startup_notify=_send_weixin_startup_ready_notification,
)


async def worker_main(
    target_file_override: str | None = None,
    force_relogin: bool = False,
) -> int:
    """Start Weixin runtime and return process exit code."""
    hooks = replace(
        WEIXIN_WORKER_HOOKS,
        before_channel_start=_force_relogin_before_start if force_relogin else None,
    )
    return await worker_main_core(target_file_override=target_file_override, hooks=hooks)


def _add_weixin_worker_args(parser) -> None:
    parser.add_argument("-r", "--relogin", action="store_true", help="Force Weixin relogin before startup")


if __name__ == "__main__":
    args = build_worker_cli_args(
        "Weixin runtime worker process",
        extra_arg_builder=_add_weixin_worker_args,
    )
    raise SystemExit(
        asyncio.run(
            worker_main(
                target_file_override=args.target_file or None,
                force_relogin=bool(args.relogin),
            )
        )
    )
