"""Shared runtime helpers for QQ/Weixin worker and supervisor flows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path

from rag_agent.llm import AnthropicProvider, OpenAIProvider

SWITCH_EXIT_CODE = 42
SWITCH_COMMAND = "/file"


@dataclass
class RuntimeCoreSettings:
    """Channel-agnostic runtime settings shared by QQ and Weixin."""

    openai_api_key: str
    openai_base_url: str
    text_llm_model: str
    vision_llm_model: str
    agent_provider: str
    agent_model: str
    anthropic_api_key: str
    anthropic_base_url: str
    embedding_model: str
    embedding_dim: int
    embedding_max_token_size: int
    parser: str
    parse_method: str
    enable_image_processing: bool
    enable_table_processing: bool
    enable_equation_processing: bool
    retrieve_top_k: int
    retrieve_chunk_top_k: int
    rag_working_dir: Path
    runtime_state_dir: Path
    uploaded_docs_dir: Path
    target_file_path: Path | None
    ingest_output_dir: Path
    startup_notify_enabled: bool
    startup_notify_message: str
    startup_notify_chat_id: str


def parse_allow_from(raw: str) -> list[str]:
    """Parse comma-separated allow list and provide sane default."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


def build_provider(settings: RuntimeCoreSettings):
    """Build agent provider using the same strategy as app.py."""
    if settings.agent_provider == "openai":
        if OpenAIProvider is None:
            raise RuntimeError("OpenAIProvider unavailable. Install openai dependency first.")
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when AGENT_PROVIDER=openai")
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            api_base=settings.openai_base_url,
            default_model=settings.agent_model,
        )

    if settings.agent_provider == "anthropic":
        if AnthropicProvider is None:
            raise RuntimeError("AnthropicProvider unavailable. Install anthropic dependency first.")
        anthropic_key = settings.anthropic_api_key or settings.openai_api_key
        anthropic_base = settings.anthropic_base_url or settings.openai_base_url
        return AnthropicProvider(
            api_key=anthropic_key,
            api_base=anthropic_base,
            default_model=settings.agent_model,
        )

    raise RuntimeError(f"Unsupported AGENT_PROVIDER: {settings.agent_provider}")


def build_rag_instance(settings: RuntimeCoreSettings):
    """Build RAGAnything engine compatible with app.py setup."""
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    settings.rag_working_dir.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(settings.rag_working_dir),
        parser=settings.parser,
        parse_method=settings.parse_method,
        enable_image_processing=settings.enable_image_processing,
        enable_table_processing=settings.enable_table_processing,
        enable_equation_processing=settings.enable_equation_processing,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            settings.text_llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return openai_complete_if_cache(
                settings.vision_llm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                **kwargs,
            )
        if image_data:
            payload_messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                },
            ]
            payload_messages = [m for m in payload_messages if m is not None]
            return openai_complete_if_cache(
                settings.vision_llm_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=payload_messages,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages or [], **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        max_token_size=settings.embedding_max_token_size,
        send_dimensions=True,
        func=lambda texts, embedding_dim=None: openai_embed.func(
            texts,
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            embedding_dim=embedding_dim,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


def compute_file_sha256(file_path: Path) -> str:
    """Compute file hash used for per-document working directory."""
    hasher = sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_effective_rag_working_dir(rag_working_dir: Path, target_file_path: Path | None) -> Path:
    """Match app.py behavior: each target file gets its own working_dir by file hash."""
    if target_file_path is None:
        return rag_working_dir / "chat_default"

    target = target_file_path.expanduser().resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"target_file_path does not exist: {target}")

    file_hash = compute_file_sha256(target)
    return rag_working_dir / file_hash


def build_session_key(channel: str, chat_id: str, file_path: Path | None, parse_method: str) -> str:
    """Build channel+chat+file scoped key for stable QA session continuity."""
    file_part = "chat:default"
    if file_path is not None:
        resolved = file_path.expanduser().resolve()
        mtime = resolved.stat().st_mtime if resolved.exists() else None
        payload = json.dumps(
            {
                "file_path": str(resolved),
                "mtime": mtime,
                "parse_method": parse_method,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        file_part = f"file:{md5(payload.encode()).hexdigest()}"
    return f"{channel}:{chat_id}:{file_part}"


def parse_switch_command(content: str) -> str | None:
    """Parse '/file <name>' command. Return None when not a switch command."""
    text = (content or "").strip()
    if not text:
        return None

    parts = text.split(maxsplit=1)
    if not parts:
        return None
    if parts[0].lower() != SWITCH_COMMAND:
        return None
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def resolve_uploaded_file(uploaded_docs_dir: Path, file_name: str) -> Path | None:
    """Resolve a file by basename under uploaded_docs_dir."""
    candidate_name = Path(file_name).name
    if candidate_name != file_name:
        return None

    candidate = uploaded_docs_dir / candidate_name
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate.resolve()


def switch_request_path(runtime_state_dir: Path, request_file_name: str) -> Path:
    """Return stable request path used for supervisor-worker handoff."""
    return runtime_state_dir / request_file_name


def write_switch_request(
    runtime_state_dir: Path,
    request_file_name: str,
    target_file_path: Path,
    trigger_chat_id: str,
) -> None:
    """Persist a file-switch request for supervisor to read after worker exits."""
    runtime_state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_file_path": str(target_file_path.resolve()),
        "trigger_chat_id": trigger_chat_id,
    }
    switch_request_path(runtime_state_dir, request_file_name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_switch_request(runtime_state_dir: Path, request_file_name: str) -> tuple[Path, str] | None:
    """Read supervisor switch request and return target path + trigger chat id."""
    req_path = switch_request_path(runtime_state_dir, request_file_name)
    if not req_path.exists():
        return None

    data = json.loads(req_path.read_text(encoding="utf-8"))
    target = str(data.get("target_file_path", "") or "").strip()
    if not target:
        return None

    trigger_chat_id = str(data.get("trigger_chat_id", "") or "").strip()
    return Path(target).expanduser().resolve(), trigger_chat_id
