"""Manual validation script for AnthropicProvider.

Usage:
    ANTHROPIC_API_KEY=... python -m tests.validate_anthropic_provider

Optional env:
    ANTHROPIC_BASE_URL=https://api.anthropic.com
    ANTHROPIC_MODEL=claude-3-5-sonnet-latest

Note:
    If your proxy URL ends with /v1 (for example https://yunwu.ai/v1),
    provider will normalize it automatically for Anthropic SDK.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    # Allow running via: python tests/validate_anthropic_provider.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.llm.anthropic_provider import AnthropicProvider


async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    api_base = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest").strip()

    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY")

    provider = AnthropicProvider(api_key=api_key, api_base=api_base, default_model=model)

    print("[1/2] validating plain chat...")
    plain = await provider.chat(
        messages=[{"role": "user", "content": "请回复: claude provider ok"}],
    )
    print("finish_reason:", plain.finish_reason)
    print("content:", plain.content)
    print("usage:", plain.usage)

    print("\n[2/2] validating tool call output...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Retrieve passages for a query.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    tool_resp = await provider.chat(
        messages=[
            {
                "role": "system",
                "content": "This is a tool-calling compliance test. Use the retrieve tool exactly once.",
            },
            {
                "role": "user",
                "content": "请帮我检索这个主题: agent loop 最小实现",
            },
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "retrieve"}},
    )

    print("finish_reason:", tool_resp.finish_reason)
    print("content:", tool_resp.content)
    print("tool_calls_count:", len(tool_resp.tool_calls))
    if tool_resp.tool_calls:
        first = tool_resp.tool_calls[0]
        print("first_tool:", first.name)
        print("first_args:", first.arguments)
    else:
        print("no tool call returned; check model/tool-call support")


if __name__ == "__main__":
    asyncio.run(main())
