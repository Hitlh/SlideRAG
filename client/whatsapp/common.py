"""Shared configuration and helpers for WhatsApp runtime supervisor/worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from client.runtime_common_shared import (
    RuntimeCoreSettings,
    SWITCH_COMMAND,
    SWITCH_EXIT_CODE,
    build_provider as _build_provider,
    build_rag_instance as _build_rag_instance,
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
from rag_agent.channels.whatsapp import WhatsAppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHATSAPP_SWITCH_REQUEST_FILE = "whatsapp_runtime_switch_request.json"


@dataclass
class RuntimeSettings(RuntimeCoreSettings):
    """WhatsApp runtime settings loaded from environment."""

    whatsapp_config: WhatsAppConfig


@dataclass
class RuntimeSwitchState:
    """Cross-task switch state used by inbound worker and runtime main."""

    event: object
    target_file_path: Path | None = None
    trigger_chat_id: str = ""


def load_settings(target_file_override: str | None = None) -> RuntimeSettings:
    """Load runtime settings from project `.env` and shell environment."""
    load_env_file(PROJECT_ROOT / ".env")

    whatsapp_config = WhatsAppConfig(
        enabled=get_env_bool("WHATSAPP_ENABLED", False),
        allow_from=parse_allow_from(get_env_str("WHATSAPP_ALLOW_FROM", "*")),
        bridge_url=get_env_str("WHATSAPP_BRIDGE_URL", "ws://127.0.0.1:3001"),
        bridge_token=get_env_str("WHATSAPP_BRIDGE_TOKEN", ""),
        reconnect_delay_s=get_env_int("WHATSAPP_RECONNECT_DELAY_S", 5),
        send_retry_attempts=get_env_int("WHATSAPP_SEND_RETRY_ATTEMPTS", 3),
        send_retry_delay_ms=get_env_int("WHATSAPP_SEND_RETRY_DELAY_MS", 400),
        accept_group_messages=get_env_bool("WHATSAPP_ACCEPT_GROUP_MESSAGES", True),
        require_mention_in_group=get_env_bool("WHATSAPP_REQUIRE_MENTION_IN_GROUP", True),
    )

    target_file_raw = (
        str(target_file_override or "").strip()
        if target_file_override is not None
        else get_env_str("WHATSAPP_TARGET_FILE", "")
    )
    target_file = Path(target_file_raw).expanduser().resolve() if target_file_raw else None

    return RuntimeSettings(
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
        rag_working_dir=Path(get_env_str("WHATSAPP_RAG_WORKING_DIR", "./rag_storage_by_whatsapp_file")).expanduser().resolve(),
        runtime_state_dir=Path(get_env_str("WHATSAPP_RUNTIME_STATE_DIR", "./rag_storage_whatsapp_runtime")).expanduser().resolve(),
        uploaded_docs_dir=Path(get_env_str("WHATSAPP_UPLOADED_DOCS_DIR", "./uploaded_docs")).expanduser().resolve(),
        target_file_path=target_file,
        ingest_output_dir=Path(get_env_str("WHATSAPP_INGEST_OUTPUT_DIR", "./output")).expanduser().resolve(),
        startup_notify_enabled=get_env_bool("WHATSAPP_STARTUP_NOTIFY_ENABLED", True),
        startup_notify_message=get_env_str("WHATSAPP_STARTUP_NOTIFY_MESSAGE", "agent is ready."),
        startup_notify_chat_id=get_env_str("WHATSAPP_STARTUP_NOTIFY_CHAT_ID", ""),
        whatsapp_config=whatsapp_config,
    )


def build_provider(settings: RuntimeSettings):
    """Build agent provider using shared channel-agnostic helper."""
    return _build_provider(settings)


def build_rag_instance(settings: RuntimeSettings):
    """Build RAGAnything engine using shared channel-agnostic helper."""
    return _build_rag_instance(settings)


def compute_file_sha256(file_path: Path) -> str:
    """Compute file hash used for per-document working directory."""
    return _compute_file_sha256(file_path)


def resolve_effective_rag_working_dir(settings: RuntimeSettings) -> Path:
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


def resolve_uploaded_file(settings: RuntimeSettings, file_name: str) -> Path | None:
    """Resolve a file by basename under uploaded_docs_dir."""
    return _resolve_uploaded_file(settings.uploaded_docs_dir, file_name)


def switch_request_path(settings: RuntimeSettings) -> Path:
    """Return a stable request file path used for supervisor-worker handoff."""
    return _switch_request_path(settings.runtime_state_dir, WHATSAPP_SWITCH_REQUEST_FILE)


def write_switch_request(settings: RuntimeSettings, target_file_path: Path, trigger_chat_id: str) -> None:
    """Persist a file-switch request for supervisor to read after worker exits."""
    _write_switch_request(
        runtime_state_dir=settings.runtime_state_dir,
        request_file_name=WHATSAPP_SWITCH_REQUEST_FILE,
        target_file_path=target_file_path,
        trigger_chat_id=trigger_chat_id,
    )


def read_switch_request(settings: RuntimeSettings) -> tuple[Path, str] | None:
    """Read supervisor switch request file and return target path + trigger chat id."""
    return _read_switch_request(settings.runtime_state_dir, WHATSAPP_SWITCH_REQUEST_FILE)
