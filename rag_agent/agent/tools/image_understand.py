"""Image understanding tool powered by rag vision_model_func (VLM)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .base import Tool


class ImageUnderstandTool(Tool):
    """Use VLM to answer a targeted question about one local image."""

    def __init__(self, rag: Any | None = None) -> None:
        self.rag = rag

    @property
    def name(self) -> str:
        return "image_understand"

    @property
    def description(self) -> str:
        return (
            "Understand one local image with VLM. "
            "Inputs: image_path and prompt(question). "
            "image_path should come from retrieve result evidence.image_chunks[*].image_path when available; do not invent paths. "
            "Returns a JSON string with status, image_path, prompt, answer, and metadata."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Local path to the image file.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Question or task the agent wants to ask about this image.",
                },
            },
            "required": ["image_path", "prompt"],
        }

    async def execute(self, **kwargs: Any) -> str:
        image_path_raw = str(kwargs.get("image_path", "")).strip()
        prompt = str(kwargs.get("prompt", "")).strip()

        if not image_path_raw:
            return self._failure("image_path is empty", image_path=image_path_raw, prompt=prompt)
        if not prompt:
            return self._failure("prompt is empty", image_path=image_path_raw, prompt=prompt)

        path = Path(image_path_raw).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return self._failure(
                "image_path does not exist or is not a file",
                image_path=str(path),
                prompt=prompt,
            )

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
            return self._failure(
                "image_path extension is not a supported image format",
                image_path=str(path),
                prompt=prompt,
            )

        vision_model_func = self._resolve_vision_model_func()
        if vision_model_func is None:
            return self._failure(
                "vision_model_func is unavailable on rag instance",
                image_path=str(path),
                prompt=prompt,
            )

        image_base64 = self._encode_image_to_base64(path)
        if not image_base64:
            return self._failure("failed to encode image as base64", image_path=str(path), prompt=prompt)

        system_prompt = (
            "You are a visual assistant. Answer only based on the provided image and user question. "
            "If uncertain, say what is unclear."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]

        try:
            answer = await vision_model_func("", messages=messages)
            call_mode = "messages"
        except Exception:
            try:
                answer = await vision_model_func(
                    prompt,
                    system_prompt=system_prompt,
                    image_data=image_base64,
                )
                call_mode = "image_data"
            except Exception as exc:
                return self._failure(
                    f"vlm call failed: {exc}",
                    image_path=str(path),
                    prompt=prompt,
                )

        payload = {
            "status": "success",
            "image_path": str(path),
            "prompt": prompt,
            "answer": str(answer) if answer is not None else "",
            "metadata": {
                "tool": self.name,
                "call_mode": call_mode,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _resolve_vision_model_func(self) -> Any | None:
        if self.rag is None:
            return None

        func = getattr(self.rag, "vision_model_func", None)
        if callable(func):
            return func

        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            return None

        func = getattr(lightrag, "vision_model_func", None)
        return func if callable(func) else None

    @staticmethod
    def _encode_image_to_base64(path: Path) -> str:
        try:
            with path.open("rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception:
            return ""

    def _failure(self, message: str, image_path: str, prompt: str) -> str:
        payload = {
            "status": "failure",
            "image_path": image_path,
            "prompt": prompt,
            "answer": "",
            "error": message,
            "metadata": {"tool": self.name},
        }
        return json.dumps(payload, ensure_ascii=False)
