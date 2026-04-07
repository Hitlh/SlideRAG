"""Minimal Anthropic provider for rag_agent."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import AsyncAnthropic

from .base import LLMProvider, LLMResponse, ToolCallRequest


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with normalized response output."""

    @staticmethod
    def _normalize_api_base(api_base: str | None) -> str | None:
        if not api_base:
            return None
        normalized = api_base.rstrip("/")
        # Anthropic SDK appends /v1/messages internally. Keep base host-level.
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        return normalized

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
        default_model: str = "claude-3-5-sonnet-latest",
        supports_prompt_caching: bool | None = None,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base, default_model=default_model)

        # Enabled by default. Can be disabled with ANTHROPIC_PROMPT_CACHING=0/false/no.
        if supports_prompt_caching is None:
            env_flag = os.getenv("ANTHROPIC_PROMPT_CACHING", "1").strip().lower()
            supports_prompt_caching = env_flag not in {"0", "false", "no", "off"}
        self.supports_prompt_caching = supports_prompt_caching

        api_base = self._normalize_api_base(api_base)
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self._client = AsyncAnthropic(**client_kwargs)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model[len("anthropic/") :] if model.startswith("anthropic/") else model

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        result: list[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", tool)
            entry: dict[str, Any] = {
                "name": func.get("name", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            }
            description = func.get("description")
            if description:
                entry["description"] = description
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        system: str | list[dict[str, Any]] = ""
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                system = content if isinstance(content, (str, list)) else str(content or "")
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, (str, list)) else str(content or ""),
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    prev_content = anthropic_messages[-1]["content"]
                    if isinstance(prev_content, list):
                        prev_content.append(block)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {"type": "text", "text": str(prev_content or "")},
                            block,
                        ]
                else:
                    anthropic_messages.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                assistant_blocks: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    assistant_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for item in content:
                        assistant_blocks.append(item if isinstance(item, dict) else {"type": "text", "text": str(item)})

                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    function = tc.get("function", {})
                    raw_args = function.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            parsed_args = {}
                    elif isinstance(raw_args, dict):
                        parsed_args = raw_args
                    else:
                        parsed_args = {}

                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": function.get("name", ""),
                            "input": parsed_args,
                        }
                    )

                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_blocks or [{"type": "text", "text": ""}],
                    }
                )
                continue

            if role == "user":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": self._convert_user_content(content),
                    }
                )

        return system, self._merge_consecutive_roles(anthropic_messages)

    def _convert_user_content(self, content: Any) -> Any:
        if isinstance(content, str) or content is None:
            return content or "(empty)"
        if not isinstance(content, list):
            return str(content)

        result: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                result.append({"type": "text", "text": str(item)})
                continue

            if item.get("type") == "image_url":
                converted = self._convert_image_block(item)
                if converted:
                    result.append(converted)
                continue

            result.append(item)

        return result or "(empty)"

    @staticmethod
    def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
        url = (block.get("image_url") or {}).get("url", "")
        if not url:
            return None

        matched = re.match(r"data:(image/\w+);base64,(.+)", url, re.DOTALL)
        if matched:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": matched.group(1),
                    "data": matched.group(2),
                },
            }

        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    @staticmethod
    def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for msg in messages:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                cur_content = msg["content"]
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(cur_content, str):
                    cur_content = [{"type": "text", "text": cur_content}]
                if isinstance(cur_content, list):
                    prev_content.extend(cur_content)
                merged[-1]["content"] = prev_content
            else:
                merged.append(msg)
        return merged

    @staticmethod
    def _apply_cache_control(
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
        marker = {"type": "ephemeral"}

        if isinstance(system, str) and system:
            system = [{"type": "text", "text": system, "cache_control": marker}]
        elif isinstance(system, list) and system:
            system = list(system)
            system[-1] = {**system[-1], "cache_control": marker}

        new_messages = list(messages)
        # Anthropic 官方建议缓存靠近末尾的稳定内容块，常见为倒数第二条。
        if len(new_messages) >= 3:
            msg = new_messages[-2]
            content = msg.get("content")
            if isinstance(content, str):
                new_messages[-2] = {
                    **msg,
                    "content": [{"type": "text", "text": content, "cache_control": marker}],
                }
            elif isinstance(content, list) and content:
                patched = list(content)
                patched[-1] = {**patched[-1], "cache_control": marker}
                new_messages[-2] = {**msg, "content": patched}

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": marker}

        return system, new_messages, new_tools

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        selected_model = self._strip_prefix(model or self.get_default_model())
        system, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        if self.supports_prompt_caching:
            system, anthropic_messages, anthropic_tools = self._apply_cache_control(
                system,
                anthropic_messages,
                anthropic_tools,
            )

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": anthropic_messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            converted_choice = self._convert_tool_choice(tool_choice)
            if converted_choice:
                kwargs["tool_choice"] = converted_choice

        try:
            response = await self._client.messages.create(**kwargs)
            return self._parse_response(response)
        except Exception as exc:
            return LLMResponse(content=f"Error calling Anthropic: {exc}", finish_reason="error")

    def _parse_response(self, response: Any) -> LLMResponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(getattr(block, "text", ""))
                continue

            if block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    ToolCallRequest(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=arguments,
                    )
                )

        stop_map = {
            "tool_use": "tool_calls",
            "end_turn": "stop",
            "max_tokens": "length",
        }
        finish_reason = stop_map.get(getattr(response, "stop_reason", "") or "", "stop")

        usage_data: dict[str, int] = {}
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            usage_data = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            if cache_creation:
                usage_data["cache_creation_input_tokens"] = cache_creation
            if cache_read:
                usage_data["cache_read_input_tokens"] = cache_read

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage_data,
        )
