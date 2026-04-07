"""Minimal token-budget memory control for rag_agent."""

from __future__ import annotations

import json
from typing import Any


class MinimalMemoryController:
    """Trim history messages to fit a prompt token budget."""

    def __init__(
        self,
        tokenizer: Any,
        max_prompt_tokens: int,
        reserved_response_tokens: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max(0, int(max_prompt_tokens))
        self.reserved_response_tokens = max(0, int(reserved_response_tokens))

    def trim_history(
        self,
        history: list[dict[str, Any]],
        *,
        system_prompt: str,
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Return a suffix of history that fits prompt budget."""
        if self.max_prompt_tokens <= 0 or not history:
            return history

        fixed_tokens = (
            self._count_text_tokens(system_prompt)
            + self._count_text_tokens(current_message)
            + self.reserved_response_tokens
        )
        history_budget = self.max_prompt_tokens - fixed_tokens
        if history_budget <= 0:
            return []

        total_history_tokens = sum(self._count_message_tokens(msg) for msg in history)
        if total_history_tokens <= history_budget:
            return history

        kept_reversed: list[dict[str, Any]] = []
        used = 0
        for message in reversed(history):
            cost = self._count_message_tokens(message)
            if kept_reversed and used + cost > history_budget:
                break
            if not kept_reversed and cost > history_budget:
                # Always keep at least one latest message if any history exists.
                kept_reversed.append(message)
                break
            kept_reversed.append(message)
            used += cost

        kept = list(reversed(kept_reversed))

        # Align to user-turn boundary to avoid leading orphan assistant/tool messages.
        for idx, msg in enumerate(kept):
            if msg.get("role") == "user":
                return kept[idx:]
        return kept

    def _count_message_tokens(self, message: dict[str, Any]) -> int:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        data: dict[str, Any] = {
            "role": role,
            "content": content,
        }
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                data[key] = message[key]

        text = json.dumps(data, ensure_ascii=False)
        return self._count_text_tokens(text)

    def _count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            encoded = self.tokenizer.encode(text)
            return len(encoded) if encoded is not None else 0
        except Exception:
            # Fallback approximation if tokenizer implementation differs.
            return max(1, len(text) // 4)
