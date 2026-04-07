"""Supervisor process for Weixin runtime.

This module starts worker process, monitors switch exit code, and restarts worker
with a new target file when requested.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.weixin_runtime_common import (
    SWITCH_EXIT_CODE,
    load_settings,
    read_switch_request,
    switch_request_path,
)


def run_worker_process(target_file_path: Path | None, *, force_relogin: bool = False) -> int:
    """Start worker process and return its exit code."""
    cmd = [sys.executable, "-m", "client.weixin_runtime_worker"]
    if target_file_path is not None:
        cmd.extend(["--target-file", str(target_file_path)])
    if force_relogin:
        cmd.append("--relogin")

    child = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
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


def supervisor_main(*, force_relogin: bool = False) -> int:
    """Run supervisor loop until worker exits normally."""
    settings = load_settings()
    current_target = settings.target_file_path
    relogin_on_next_start = force_relogin

    while True:
        req_path = switch_request_path(settings)
        req_path.unlink(missing_ok=True)

        logger.info(
            "Supervisor starting worker with target_file_path={} force_relogin={}",
            current_target,
            relogin_on_next_start,
        )
        exit_code = run_worker_process(current_target, force_relogin=relogin_on_next_start)
        relogin_on_next_start = False

        if exit_code != SWITCH_EXIT_CODE:
            logger.info("Worker exited with code {}", exit_code)
            return exit_code

        try:
            request = read_switch_request(settings)
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


if __name__ == "__main__":
    raise SystemExit(supervisor_main())
