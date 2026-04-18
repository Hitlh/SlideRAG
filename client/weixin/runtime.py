"""Weixin runtime entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.channel_runtime_core import runtime_main_core
from client.weixin.supervisor import supervisor_main
from client.weixin.worker import worker_main


def _add_weixin_runtime_args(parser) -> None:
    parser.add_argument("-r", "--relogin", action="store_true", help="Force Weixin relogin before startup")


def _build_worker_kwargs(args) -> dict[str, bool]:
    return {"force_relogin": bool(args.relogin)}


def _build_supervisor_kwargs(args) -> dict[str, bool]:
    return {"force_relogin": bool(args.relogin)}


def main() -> int:
    """Dispatch to supervisor or worker according to CLI flags."""
    return runtime_main_core(
        description="Weixin runtime launcher",
        worker_main=worker_main,
        supervisor_main=supervisor_main,
        extra_arg_builder=_add_weixin_runtime_args,
        worker_kwargs_builder=_build_worker_kwargs,
        supervisor_kwargs_builder=_build_supervisor_kwargs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
