"""Manual validation script for AgentLoop.

Usage:
    # OpenAI path
    OPENAI_API_KEY=... LLM_PROVIDER=openai python -m tests.validate_loop

    # Anthropic path (Claude)
    ANTHROPIC_API_KEY=... LLM_PROVIDER=anthropic LLM_MODEL=claude-3-5-sonnet-latest python -m tests.validate_loop

Optional env:
    LLM_API_BASE=http://localhost:8000/v1
    WORKING_DIR=./rag_storage_validate_loop
    PARSE_METHOD=auto
    TEST_FILE_PATH=./uploaded_docs/example.pdf
    INGEST_BEFORE_TEST=0
    TEST_OUTPUT_DIR=./output
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_DIM=3072
    EMBEDDING_MAX_TOKEN_SIZE=8192
    RETRIEVE_TOP_K=20
    RETRIEVE_CHUNK_TOP_K=20
    TEST_QUESTION=什么是RAG
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    # Allow direct execution via: python tests/validate_loop.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.agent.loop import AgentLoop
from rag_agent.llm.anthropic_provider import AnthropicProvider
from rag_agent.llm.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# Inline config: edit these values directly for local validation
# ---------------------------------------------------------------------------
PROVIDER_KIND = "anthropic"  # "openai" or "anthropic"
MODEL = "claude-sonnet-4-6"
LLM_API_BASE: str | None = "https://yunwu.ai/v1"

OPENAI_API_KEY = "sk-REgh3TOxQQJxf4bfSMQP6g3YnMG3iivCTT6ndZgFJ3g2Y1qp"
ANTHROPIC_API_KEY = "sk-REgh3TOxQQJxf4bfSMQP6g3YnMG3iivCTT6ndZgFJ3g2Y1qp"

WORKING_DIR = "./rag_storage_validate_loop"
PARSE_METHOD = "auto"
TEST_FILE_PATH: str | None = "example1.pdf"  # e.g. "./uploaded_docs/example.pdf"
INGEST_BEFORE_TEST = True
TEST_OUTPUT_DIR = "./output"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
EMBEDDING_MAX_TOKEN_SIZE = 8192

RETRIEVE_TOP_K = 20
RETRIEVE_CHUNK_TOP_K = 20
TEST_QUESTION = "帮我总结文档信息。"

RAG_API_KEY: str | None = "sk-REgh3TOxQQJxf4bfSMQP6g3YnMG3iivCTT6ndZgFJ3g2Y1qp"
RAG_API_BASE: str | None = "https://yunwu.ai/v1"


def _int_env(name: str, default: int) -> int:
    return default


def _bool_env(name: str, default: bool = False) -> bool:
    return default


def _build_rag(
    api_key: str,
    base_url: str | None,
    model: str,
    embedding_model: str,
    working_dir: str,
) -> Any:
    """Build a RAGAnything instance following app.py wiring style."""
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
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
                model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        if image_data:
            assembled_messages: list[dict[str, Any]] = []
            if system_prompt:
                assembled_messages.append({"role": "system", "content": system_prompt})
            assembled_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }
            )
            return openai_complete_if_cache(
                model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=assembled_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=_int_env("EMBEDDING_DIM", 3072),
        max_token_size=_int_env("EMBEDDING_MAX_TOKEN_SIZE", 8192),
        send_dimensions=True,
        func=lambda texts, embedding_dim=None: openai_embed.func(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            embedding_dim=embedding_dim,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def _maybe_ingest_file(rag: Any, file_path: str | None, parse_method: str) -> None:
    if not file_path:
        return

    path_obj = Path(file_path).expanduser().resolve()
    if not path_obj.exists():
        raise SystemExit(f"TEST_FILE_PATH does not exist: {path_obj}")

    await rag.process_document_complete_with_page_topics(
        file_path=str(path_obj),
        output_dir=TEST_OUTPUT_DIR,
        parse_method=parse_method,
    )


async def main() -> None:
    provider_kind = PROVIDER_KIND.strip().lower()
    api_base = LLM_API_BASE
    model = MODEL.strip()
    working_dir = WORKING_DIR
    parse_method = PARSE_METHOD
    file_path = TEST_FILE_PATH
    ingest_before_test = INGEST_BEFORE_TEST
    embedding_model = EMBEDDING_MODEL
    retrieve_top_k = RETRIEVE_TOP_K
    retrieve_chunk_top_k = RETRIEVE_CHUNK_TOP_K
    question = TEST_QUESTION

    if provider_kind == "anthropic":
        api_key = ANTHROPIC_API_KEY.strip()
        if not api_key:
            raise SystemExit("Missing ANTHROPIC_API_KEY")
        provider = AnthropicProvider(api_key=api_key, api_base=api_base, default_model=model)
    else:
        api_key = OPENAI_API_KEY.strip()
        #if not api_key:
         #   raise SystemExit("Missing OPENAI_API_KEY")
        #provider = OpenAIProvider(api_key=api_key, api_base=api_base, default_model=model)

    rag_api_key = (RAG_API_KEY or "").strip() or api_key
    rag_api_base = (RAG_API_BASE or "").strip() or api_base
    rag = _build_rag(
        api_key=rag_api_key,
        base_url=rag_api_base,
        model="gpt-4o",
        embedding_model=embedding_model,
        working_dir=working_dir,
    )
    if ingest_before_test:
        await _maybe_ingest_file(rag, file_path=file_path, parse_method=parse_method)

    loop = AgentLoop(
        provider=provider,
        workspace="./rag_agent_loop",
        rag=rag,
        model=model,
        retrieve_config={
            "mode": "hybrid",
            "top_k": retrieve_top_k,
            "chunk_top_k": retrieve_chunk_top_k,
        },
    )

    result = await loop.process_message(
        question,
        file_path=file_path,
        parse_method=parse_method,
    )

    print("final_answer:", result.final_answer)
    print("iterations:", result.iterations)
    print("tools_used:", result.tools_used)


if __name__ == "__main__":
    asyncio.run(main())
