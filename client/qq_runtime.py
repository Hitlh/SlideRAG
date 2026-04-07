"""QQ runtime entrypoint.

Backward-compatible launcher:
- default: supervisor mode
- --worker: run worker directly
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.qq_runtime_supervisor import supervisor_main
from client.qq_runtime_worker import worker_main


def build_cli_args() -> argparse.Namespace:
    """Parse CLI args for backward-compatible runtime entrypoint."""
    parser = argparse.ArgumentParser(description="QQ runtime launcher")
    parser.add_argument("--worker", action="store_true", help="Run worker mode directly")
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    return parser.parse_args()


def main() -> int:
    """Dispatch to supervisor or worker according to CLI flags."""
    args = build_cli_args()
    if args.worker:
        return int(asyncio.run(worker_main(target_file_override=args.target_file or None)))
    return int(supervisor_main())


if __name__ == "__main__":
    raise SystemExit(main())
