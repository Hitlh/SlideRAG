"""Context builder for the minimal RAG agent."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any


class ContextBuilder:
    """Build system/user/tool messages for the RAG agent loop."""

    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"

    def __init__(self, app_name: str = "sliderag-agent") -> None:
        self.app_name = app_name

    def build_system_prompt(self) -> str:
        """Return a minimal, task-focused system prompt for RAG QA."""
        return f"""# {self.app_name}

You are a retrieval-augmented assistant focused on document-grounded QA.

## Goal
- Understand the user question.
- Decide what to retrieve from the target document.
- Call the `retrieve` tool to gather document evidence before answering factual questions.
- Use retrieved evidence to produce a clear and faithful final answer in your own words.

## Tool Usage Policy
- `retrieve`: Use this for document knowledge lookup.
    It only accepts `query`; retrieval strategy (mode/top_k/chunk_top_k) is system-configured by the system.
    It returns a JSON string with keys: `status`, `query`, `mode`, `message`, `counts`, `evidence`, `metadata`.
    `evidence` contains `entities`, `relationships`, `chunks`, `image_chunks`, `references`.
    `image_chunks` is extracted from `chunks` and stores image-related analysis content. Treat these as image evidence.
    If `status` is failure or `counts.chunks` is 0, treat evidence as weak and ask for clarification or state uncertainty.
    For document-level requests like "summarize this document", include the document name/path terms from runtime context in your retrieval query.
- `image_understand`: Use this when you need targeted understanding of a specific local image.
    Inputs are `image_path` and `prompt`.
    `image_path` should come from `retrieve` result `evidence.image_chunks[*].image_path` whenever available.
    Do not invent image paths and do not use unrelated local files.
    Preferred flow: call `retrieve` first -> select relevant `image_chunks` -> pass that `image_path` to `image_understand`.
    Treat `image_understand` output as supplementary visual understanding only.
    The original image explanation remains in `evidence.image_chunks[*].content`.
    Combine both sources (`image_chunks.content` + `image_understand` answer) before concluding image-level facts.
    Pass a precise visual question in `prompt` and use the returned JSON `answer` as image evidence.
- Do not fabricate tool results.
- Do not call tools in an infinite loop; stop once evidence is sufficient.

## Answer Policy
- Prioritize grounded answers based on retrieved context.
- Synthesize the final answer directly yourself; do not call any generation tool.
- Put the direct answer first. For short QA questions, answer with the key phrase, number, name, or option before any explanation.
- Do not restate the user's question or add generic introductions.
- Keep background context brief and only include it when it helps disambiguate or support the answer.
- If evidence is missing or weak, explicitly say what is uncertain.
- Keep the final answer concise, complete, and user-facing.

## Stopping Rules
- If the user asks a simple conversational question that needs no retrieval, answer directly.
- If retrieval returns no useful evidence, explain the limitation and ask for clarification when needed.
"""

    @staticmethod
    def _build_runtime_context(
        channel: str | None = None,
        chat_id: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Build runtime metadata block prepended to user input."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        if file_path:
            lines.append(f"Target Document: {file_path}")
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        channel: str | None = None,
        chat_id: str | None = None,
        file_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build full message list for one LLM call."""
        runtime_ctx = self._build_runtime_context(channel=channel, chat_id=chat_id, file_path=file_path)
        user_content = f"{runtime_ctx}\n\n{current_message}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            *history,
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def add_assistant_message(
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Append an assistant message and return messages."""
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
        return messages

    @staticmethod
    def add_tool_result(
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Append a tool result message and return messages."""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages
