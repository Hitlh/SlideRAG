"""Validate ImageUnderstandTool with a fake rag vision model function.

Usage:
    python -m tests.validate_image_understand_tool
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from rag_agent.agent.tools.image_understand import ImageUnderstandTool


class _FakeRAG:
    async def vision_model_func(self, prompt, system_prompt=None, image_data=None, messages=None, **kwargs):
        _ = (prompt, system_prompt, image_data, kwargs)
        if messages:
            user = messages[1]["content"]
            text = user[0]["text"]
            return f"FAKE_VLM_OK: {text}"
        return "FAKE_VLM_OK"


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "demo.png"
        # Minimal PNG header + bytes, enough for base64 path test.
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        tool = ImageUnderstandTool(rag=_FakeRAG())
        result = await tool.execute(image_path=str(img_path), prompt="图里有什么核心元素？")
        parsed = json.loads(result)

        assert parsed["status"] == "success"
        assert parsed["metadata"]["tool"] == "image_understand"
        assert parsed["metadata"]["call_mode"] in {"messages", "image_data"}
        assert "FAKE_VLM_OK" in parsed["answer"]

        print("IMAGE_UNDERSTAND_TOOL_TEST: PASS")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
