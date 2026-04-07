import atexit
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    import streamlit as st
except Exception:
    class _NoopStreamlit:
        @staticmethod
        def cache_resource(*_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

    st = _NoopStreamlit()


def load_env_file(env_path: Path):
    """Load .env key-value pairs into process environment if not already set."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid integer for {name}: {value}. Using default={default}")
        return default


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    print(f"Invalid boolean for {name}: {value}. Using default={default}")
    return default


def _child_pid_registry_path() -> Path:
    return Path(tempfile.gettempdir()) / f"sliderag_streamlit_children_{os.getpid()}.json"


def _load_child_pids() -> list[int]:
    registry_path = _child_pid_registry_path()
    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [int(pid) for pid in data if isinstance(pid, int) or str(pid).isdigit()]
    except Exception:
        return []


def _save_child_pids(pids: list[int]):
    registry_path = _child_pid_registry_path()
    unique_sorted_pids = sorted(set(pids))
    registry_path.write_text(json.dumps(unique_sorted_pids, ensure_ascii=False), encoding="utf-8")


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _register_child_pid(pid: int):
    current = _load_child_pids()
    current.append(pid)
    _save_child_pids(current)


def _cleanup_spawned_children():
    pids = _load_child_pids()
    if not pids:
        return

    # subprocess uses start_new_session=True, pid is also process group id.
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any(_is_process_alive(pid) for pid in pids):
            break
        time.sleep(0.1)

    for pid in pids:
        if not _is_process_alive(pid):
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    try:
        _child_pid_registry_path().unlink(missing_ok=True)
    except Exception:
        pass


def register_cleanup_handlers_once():
    marker_key = "SLIDERAG_CHILD_CLEANUP_REGISTERED_PID"
    current_pid = str(os.getpid())
    if os.environ.get(marker_key) == current_pid:
        return
    os.environ[marker_key] = current_pid

    atexit.register(_cleanup_spawned_children)

    if threading.current_thread() is not threading.main_thread():
        return

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, frame):
        _cleanup_spawned_children()

        previous_handler = (
            previous_sigint_handler if signum == signal.SIGINT else previous_sigterm_handler
        )
        if callable(previous_handler):
            previous_handler(signum, frame)
            return
        if previous_handler == signal.SIG_IGN:
            return
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
        pass


@st.cache_resource(show_spinner=False)
def get_async_runtime():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True, name="rag-streamlit-async-loop")
    thread.start()
    ready.wait()
    return loop, thread


def run_async(coro):
    loop, _ = get_async_runtime()
    if loop.is_closed():
        get_async_runtime.clear()
        loop, _ = get_async_runtime()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def find_available_port(start_port: int = 8502, max_tries: int = 100) -> int:
    for port in range(start_port, start_port + max_tries):
        if is_port_available(port):
            return port
    raise RuntimeError("未找到可用端口，请手动释放端口后重试。")


def wait_for_port_listening(
    port: int,
    host: str = "127.0.0.1",
    timeout_seconds: float = 20.0,
    poll_interval_seconds: float = 0.2,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not is_port_available(port, host=host):
            return True
        time.sleep(poll_interval_seconds)
    return False


def launch_streamlit_process(port: int, app_path: Path):
    app_path = app_path.resolve()
    project_root = app_path.parent.parent
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _register_child_pid(proc.pid)
