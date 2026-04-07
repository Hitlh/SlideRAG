"""Shared settings helpers for Weixin runtime.

Supports both:
- echo mode (loopback test)
- agent mode (RAG agent inference)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from client.runtime_common_shared import (
    RuntimeCoreSettings,
    SWITCH_COMMAND,
    SWITCH_EXIT_CODE,
    build_session_key as _build_session_key,
    compute_file_sha256 as _compute_file_sha256,
    parse_allow_from,
    parse_switch_command as _parse_switch_command,
    read_switch_request as _read_switch_request,
    resolve_effective_rag_working_dir as _resolve_effective_rag_working_dir,
    resolve_uploaded_file as _resolve_uploaded_file,
    switch_request_path as _switch_request_path,
    write_switch_request as _write_switch_request,
)
from client.utils import get_env_bool, get_env_int, get_env_str, load_env_file
from rag_agent.channels.weixin import WeixinConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIXIN_SWITCH_REQUEST_FILE = "weixin_runtime_switch_request.json"


@dataclass
class WeixinRuntimeSettings(RuntimeCoreSettings):
    """Runtime settings for standalone Weixin runtime."""

    weixin_config: WeixinConfig
    runtime_mode: str
    echo_prefix: str


def load_settings(target_file_override: str | None = None) -> WeixinRuntimeSettings:
    """Load runtime settings from .env and process environment."""
    load_env_file(PROJECT_ROOT / ".env")

    state_dir = get_env_str("WEIXIN_STATE_DIR", str((PROJECT_ROOT / "rag_storage_weixin_runtime").resolve()))
    target_file_raw = (
        str(target_file_override or "").strip()
        if target_file_override is not None
        else get_env_str("WEIXIN_TARGET_FILE", "")
    )
    target_file = Path(target_file_raw).expanduser().resolve() if target_file_raw else None

    cfg = WeixinConfig(
        enabled=get_env_bool("WEIXIN_ENABLED", True),
        allow_from=parse_allow_from(get_env_str("WEIXIN_ALLOW_FROM", "*")),
        base_url=get_env_str("WEIXIN_BASE_URL", "https://ilinkai.weixin.qq.com"),
        route_tag=get_env_str("WEIXIN_ROUTE_TAG", "") or None,
        token=get_env_str("WEIXIN_TOKEN", ""),
        state_dir=state_dir,
        poll_timeout=get_env_int("WEIXIN_POLL_TIMEOUT", 35),
    )
    runtime_mode = get_env_str("WEIXIN_RUNTIME_MODE", "agent").lower()
    if runtime_mode not in {"agent", "echo"}:
        runtime_mode = "agent"

    return WeixinRuntimeSettings(
        weixin_config=cfg,
        runtime_mode=runtime_mode,
        echo_prefix=get_env_str("WEIXIN_ECHO_PREFIX", "echo: "),

        openai_api_key=get_env_str("OPENAI_API_KEY", ""),
        openai_base_url=get_env_str("OPENAI_BASE_URL", "https://api.yunwu.ai/v1"),
        text_llm_model=get_env_str("TEXT_LLM_MODEL", "gpt-4o-mini"),
        vision_llm_model=get_env_str("VLM_MODEL", get_env_str("VISION_LLM_MODEL", "gpt-4o")),
        agent_provider=get_env_str("AGENT_PROVIDER", "openai").lower(),
        agent_model=get_env_str("AGENT_MODEL", "gpt-4o"),
        anthropic_api_key=get_env_str("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=get_env_str("ANTHROPIC_BASE_URL", ""),
        embedding_model=get_env_str("EMBEDDING_MODEL", "text-embedding-3-large"),
        embedding_dim=get_env_int("EMBEDDING_DIM", 3072),
        embedding_max_token_size=get_env_int("EMBEDDING_MAX_TOKEN_SIZE", 8192),

        parser=get_env_str("PARSER", "mineru"),
        parse_method=get_env_str("PARSE_METHOD", "auto"),
        enable_image_processing=get_env_bool("ENABLE_IMAGE_PROCESSING", True),
        enable_table_processing=get_env_bool("ENABLE_TABLE_PROCESSING", True),
        enable_equation_processing=get_env_bool("ENABLE_EQUATION_PROCESSING", True),
        retrieve_top_k=get_env_int("RETRIEVE_TOP_K", 20),
        retrieve_chunk_top_k=get_env_int("RETRIEVE_CHUNK_TOP_K", 20),

        rag_working_dir=Path(
            get_env_str("WEIXIN_RAG_WORKING_DIR", "./rag_storage_by_file_weixin")
        ).expanduser().resolve(),
        runtime_state_dir=Path(
            get_env_str("WEIXIN_RUNTIME_STATE_DIR", "./rag_storage_weixin_runtime")
        ).expanduser().resolve(),
        uploaded_docs_dir=Path(
            get_env_str("WEIXIN_UPLOADED_DOCS_DIR", "./uploaded_docs")
        ).expanduser().resolve(),
        target_file_path=target_file,
        ingest_output_dir=Path(get_env_str("WEIXIN_INGEST_OUTPUT_DIR", "./output")).expanduser().resolve(),
        startup_notify_enabled=get_env_bool("WEIXIN_STARTUP_NOTIFY_ENABLED", True),
        startup_notify_message=get_env_str("WEIXIN_STARTUP_NOTIFY_MESSAGE", "agent is ready."),
        startup_notify_chat_id=get_env_str("WEIXIN_STARTUP_NOTIFY_CHAT_ID", ""),
    )


def compute_file_sha256(file_path: Path) -> str:
    """Compute file hash used for per-document working directory."""
    return _compute_file_sha256(file_path)


def resolve_effective_rag_working_dir(settings: WeixinRuntimeSettings) -> Path:
    """Match app.py behavior: each target file gets its own working_dir by file hash."""
    return _resolve_effective_rag_working_dir(
        rag_working_dir=settings.rag_working_dir,
        target_file_path=settings.target_file_path,
    )


def build_session_key(channel: str, chat_id: str, file_path: Path | None, parse_method: str) -> str:
    """Build channel+chat+file scoped key for stable QA session continuity."""
    return _build_session_key(channel, chat_id, file_path, parse_method)


def parse_switch_command(content: str) -> str | None:
    """Parse '/file <name>' command. Return None when not a switch command."""
    return _parse_switch_command(content)


def resolve_uploaded_file(settings: WeixinRuntimeSettings, file_name: str) -> Path | None:
    """Resolve a file by basename under uploaded_docs_dir."""
    return _resolve_uploaded_file(settings.uploaded_docs_dir, file_name)


def switch_request_path(settings: WeixinRuntimeSettings) -> Path:
    """Return stable request path used for supervisor-worker handoff."""
    return _switch_request_path(settings.runtime_state_dir, WEIXIN_SWITCH_REQUEST_FILE)


def write_switch_request(settings: WeixinRuntimeSettings, target_file_path: Path, trigger_chat_id: str) -> None:
    """Persist a file-switch request for supervisor to read after worker exits."""
    _write_switch_request(
        runtime_state_dir=settings.runtime_state_dir,
        request_file_name=WEIXIN_SWITCH_REQUEST_FILE,
        target_file_path=target_file_path,
        trigger_chat_id=trigger_chat_id,
    )


def read_switch_request(settings: WeixinRuntimeSettings) -> tuple[Path, str] | None:
    """Read supervisor switch request and return target path + trigger chat id."""
    return _read_switch_request(settings.runtime_state_dir, WEIXIN_SWITCH_REQUEST_FILE)
