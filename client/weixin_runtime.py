"""Weixin runtime entrypoint.

Usage:
  python3 client/weixin_runtime.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.weixin_runtime_supervisor import supervisor_main
from client.weixin_runtime_worker import worker_main


def build_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weixin runtime launcher")
    parser.add_argument("--worker", action="store_true", help="Run worker mode directly")
    parser.add_argument("--target-file", default="", help="Worker target file path override")
    parser.add_argument("-r", "--relogin", action="store_true", help="Force Weixin relogin before startup")
    return parser.parse_args()


def main() -> int:
    args = build_cli_args()
    if args.worker:
        return int(
            asyncio.run(
                worker_main(
                    target_file_override=args.target_file or None,
                    force_relogin=bool(args.relogin),
                )
            )
        )
    return int(supervisor_main(force_relogin=bool(args.relogin)))


if __name__ == "__main__":
    raise SystemExit(main())
