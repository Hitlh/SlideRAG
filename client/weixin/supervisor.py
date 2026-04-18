"""Supervisor process for Weixin runtime."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.channel_runtime_core import SupervisorHooks, supervisor_main_core
from client.weixin.common import (
    SWITCH_EXIT_CODE,
    load_settings,
    read_switch_request,
    switch_request_path,
)

WEIXIN_SUPERVISOR_HOOKS = SupervisorHooks(
    channel_display_name="Weixin",
    worker_module="client.weixin.worker",
    project_root=PROJECT_ROOT,
    load_settings=load_settings,
    switch_exit_code=SWITCH_EXIT_CODE,
    switch_request_path=switch_request_path,
    read_switch_request=read_switch_request,
)


def supervisor_main(*, force_relogin: bool = False) -> int:
    """Run supervisor loop until worker exits normally."""
    hooks = WEIXIN_SUPERVISOR_HOOKS
    if force_relogin:
        hooks = replace(
            WEIXIN_SUPERVISOR_HOOKS,
            build_extra_worker_args=lambda start_index: ["--relogin"] if start_index == 0 else [],
        )
    return supervisor_main_core(hooks=hooks)


if __name__ == "__main__":
    raise SystemExit(supervisor_main())
