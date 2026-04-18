"""Supervisor process for QQ runtime."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.channel_runtime_core import SupervisorHooks, supervisor_main_core
from client.qq.common import (
    SWITCH_EXIT_CODE,
    load_settings,
    read_switch_request,
    switch_request_path,
)

QQ_SUPERVISOR_HOOKS = SupervisorHooks(
    channel_display_name="QQ",
    worker_module="client.qq.worker",
    project_root=PROJECT_ROOT,
    load_settings=load_settings,
    switch_exit_code=SWITCH_EXIT_CODE,
    switch_request_path=switch_request_path,
    read_switch_request=read_switch_request,
)


def supervisor_main() -> int:
    """Run supervisor loop until worker exits normally."""
    return supervisor_main_core(hooks=QQ_SUPERVISOR_HOOKS)


if __name__ == "__main__":
    raise SystemExit(supervisor_main())
