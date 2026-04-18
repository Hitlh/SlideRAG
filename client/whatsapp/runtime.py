"""WhatsApp runtime entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.channel_runtime_core import runtime_main_core
from client.whatsapp.supervisor import supervisor_main
from client.whatsapp.worker import worker_main


def main() -> int:
    """Dispatch to supervisor or worker according to CLI flags."""
    return runtime_main_core(
        description="WhatsApp runtime launcher",
        worker_main=worker_main,
        supervisor_main=supervisor_main,
    )


if __name__ == "__main__":
    raise SystemExit(main())
