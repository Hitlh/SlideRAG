"""Validate minimal memory control in AgentLoop.

Usage:
    python -m tests.validate_memory_control
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from rag_agent.agent.context import ContextBuilder
from rag_agent.agent.loop import AgentLoop
from rag_agent.llm.base import LLMProvider, LLMResponse


class FakeTokenizer:
    """Simple tokenizer used by the memory controller test."""

    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeProvider(LLMProvider):
    """Deterministic provider that reports what history it received."""

    def __init__(self) -> None:
        super().__init__(api_key="fake", default_model="fake-model")
        self.last_messages: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        combined = "\n".join(str(m.get("content", "")) for m in messages)
        has_old_1 = "OLD_1" in combined
        has_old_4 = "OLD_4" in combined
        return LLMResponse(
            content=f"has_old_1={has_old_1};has_old_4={has_old_4};msg_count={len(messages)}",
            finish_reason="stop",
        )


class TinyContext(ContextBuilder):
    """Tiny prompt to keep token-budget test deterministic."""

    def build_system_prompt(self) -> str:
        return "tiny-system"


def _mk_message(role: str, marker: str, words: int) -> dict[str, str]:
    body = " ".join([marker] + [f"w{i}" for i in range(words)])
    return {"role": role, "content": body}


async def main() -> None:
    provider = FakeProvider()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        loop = AgentLoop(
            provider=provider,
            workspace=workspace,
            context=TinyContext(),
            tokenizer=FakeTokenizer(),
            memory_max_prompt_tokens=120,
            memory_reserved_response_tokens=40,
            max_history_messages=0,
        )

        session_key = "chat:mem-test"
        session = loop.sessions.get_or_create(session_key)

        # Build a long history: old turns should be trimmed, latest turns retained.
        session.messages = [
            _mk_message("user", "OLD_1", 30),
            _mk_message("assistant", "OLD_1_A", 30),
            _mk_message("user", "OLD_2", 30),
            _mk_message("assistant", "OLD_2_A", 30),
            _mk_message("user", "OLD_3", 30),
            _mk_message("assistant", "OLD_3_A", 30),
            _mk_message("user", "OLD_4", 30),
            _mk_message("assistant", "OLD_4_A", 30),
        ]
        loop.sessions.save(session)

        result = await loop.process_message(
            "请基于当前文档回答问题",
            session_key=session_key,
            file_path="example1.pdf",
        )

        print("final_answer:", result.final_answer)

        saw_old_1 = "has_old_1=True" in result.final_answer
        saw_old_4 = "has_old_4=True" in result.final_answer

        # Expect early history trimmed, recent history retained.
        if (not saw_old_1) and saw_old_4:
            print("MEMORY_CONTROL_TEST: PASS")
            return

        print("MEMORY_CONTROL_TEST: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
