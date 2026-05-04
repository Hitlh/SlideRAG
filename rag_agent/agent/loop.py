"""Minimal agent loop for RAG tool-calling workflow."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_agent.llm.base import LLMProvider

from .context import ContextBuilder
from .memory import MinimalMemoryController
from .session import Session, SessionManager
from .tools import ImageUnderstandTool, RetrieveTool, ToolRegistry


@dataclass
class AgentLoopResult:
    """Result object returned by one agent loop run."""

    final_answer: str
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    """Minimal loop that lets the LLM decide when to call RAG tools."""

    _VISUAL_EVIDENCE_PREFIX = "[Retrieved Visual Evidence]"

    def __init__(
        self,
        provider: LLMProvider,
        workspace: str | Path,
        model: str | None = None,
        max_iterations: int = 8,
        max_tool_calls: int = 8,
        max_history_messages: int = 200,
        memory_max_prompt_tokens: int = 65536,
        memory_reserved_response_tokens: int = 2048,
        tokenizer: Any | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        rag: Any | None = None,
        retrieve_config: dict[str, Any] | None = None,
        context: ContextBuilder | None = None,
        tools: ToolRegistry | None = None,
        sessions: SessionManager | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = Path(workspace).expanduser().resolve()
        self.rag = rag
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_history_messages = max_history_messages
        self.memory_max_prompt_tokens = memory_max_prompt_tokens
        self.memory_reserved_response_tokens = memory_reserved_response_tokens
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieve_config = dict(retrieve_config or {})
        self.attach_retrieve_images = bool(self.retrieve_config.get("attach_images", True))
        self.max_retrieve_images = int(self.retrieve_config.get("max_images", 10) or 10)
        self.max_retrieve_image_bytes = int(
            self.retrieve_config.get("max_image_bytes", 5 * 1024 * 1024) or 5 * 1024 * 1024
        )

        self.context = context or ContextBuilder()
        self.tools = tools or ToolRegistry()
        self.sessions = sessions or SessionManager(self.workspace)
        self.memory = self._build_memory_controller(tokenizer=tokenizer)
        if tools is None:
            self._register_default_tools()

    def _build_memory_controller(self, tokenizer: Any | None) -> MinimalMemoryController | None:
        """Build minimal token-based memory controller when tokenizer and budget are available."""
        if not isinstance(self.memory_max_prompt_tokens, int) or self.memory_max_prompt_tokens <= 0:
            return None

        resolved_tokenizer = tokenizer or self._resolve_tokenizer_from_rag()
        if resolved_tokenizer is None:
            return None

        return MinimalMemoryController(
            tokenizer=resolved_tokenizer,
            max_prompt_tokens=self.memory_max_prompt_tokens,
            reserved_response_tokens=self.memory_reserved_response_tokens,
        )

    def _resolve_tokenizer_from_rag(self) -> Any | None:
        """Resolve tokenizer from rag.lightrag.tokenizer safely."""
        if self.rag is None:
            return None
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            return None
        return getattr(lightrag, "tokenizer", None)

    def _register_default_tools(self) -> None:
        """Register default RAG tools for MVP."""
        self.tools.register(
            RetrieveTool(
                rag=self.rag,
                mode=str(self.retrieve_config.get("mode", "hybrid")),
                top_k=self.retrieve_config.get("top_k"),
                chunk_top_k=self.retrieve_config.get("chunk_top_k"),
            )
        )
        self.tools.register(ImageUnderstandTool(rag=self.rag))

    async def process_message(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        session_key: str | None = None,
        file_path: str | Path | None = None,
        parse_method: str | None = None,
        session_options: dict[str, Any] | None = None,
    ) -> AgentLoopResult:
        """Build session/context and then run one agent loop turn."""
        resolved_session_key = session_key or self._build_session_key(
            file_path=file_path,
            parse_method=parse_method,
            **(session_options or {}),
        )
        session = self.sessions.get_or_create(resolved_session_key)

        history_messages = history
        if history_messages is None:
            history_messages = session.get_history(max_messages=self.max_history_messages)

        if self.memory is not None:
            history_messages = self.memory.trim_history(
                history_messages,
                system_prompt=self.context.build_system_prompt(),
                current_message=user_message,
            )
        
        messages = self.context.build_messages(
            history=history_messages,
            current_message=user_message,
            channel=channel,
            chat_id=chat_id,
            file_path=file_path,
        )
        result = await self.run_once(messages)
        self._save_turn(session=session, messages=result.messages, skip=1 + len(history_messages))
        self.sessions.save(session)
        return result

    async def run_once(self, initial_messages: list[dict[str, Any]]) -> AgentLoopResult:
        """Run the pure agent loop over prepared messages."""
        messages = initial_messages

        tools_used: list[str] = []
        tool_calls_count = 0

        for iteration in range(1, self.max_iterations + 1):
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if response.finish_reason == "error":
                return AgentLoopResult(
                    final_answer=response.content or "LLM returned an error.",
                    tools_used=tools_used,
                    iterations=iteration,
                    messages=messages,
                )

            if response.has_tool_calls:
                tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                messages = self.context.add_assistant_message(messages, response.content, tool_call_dicts)
                visual_items: list[dict[str, Any]] = []

                for tool_call in response.tool_calls:
                    tool_calls_count += 1
                    tools_used.append(tool_call.name)

                    if tool_calls_count > self.max_tool_calls:
                        return AgentLoopResult(
                            final_answer=(
                                "Reached max tool call budget without a final answer. "
                                "Please refine the question or increase the budget."
                            ),
                            tools_used=tools_used,
                            iterations=iteration,
                            messages=messages,
                        )

                    tool_result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages,
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        result=tool_result,
                    )

                    if self.attach_retrieve_images:
                        visual_items.extend(self._extract_retrieve_image_items(tool_call.name, tool_result))

                visual_message = self._build_visual_evidence_message(
                    visual_items,
                    max_images=self.max_retrieve_images,
                    max_image_bytes=self.max_retrieve_image_bytes,
                )
                if visual_message:
                    messages.append(visual_message)
                continue

            final_answer = response.content or ""
            messages = self.context.add_assistant_message(messages, final_answer)
            return AgentLoopResult(
                final_answer=final_answer,
                tools_used=tools_used,
                iterations=iteration,
                messages=messages,
            )

        return AgentLoopResult(
            final_answer=(
                f"Reached max iterations ({self.max_iterations}) without producing a final answer."
            ),
            tools_used=tools_used,
            iterations=self.max_iterations,
            messages=messages,
        )

    @staticmethod
    def _extract_retrieve_image_items(tool_name: str, tool_result: str) -> list[dict[str, Any]]:
        """Extract image metadata from retrieve JSON while preserving chunk bindings."""
        if tool_name != "retrieve":
            return []

        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            return []

        evidence = payload.get("evidence") if isinstance(payload, dict) else None
        image_chunks = evidence.get("image_chunks") if isinstance(evidence, dict) else None
        if not isinstance(image_chunks, list):
            return []

        items: list[dict[str, Any]] = []
        for idx, item in enumerate(image_chunks):
            if not isinstance(item, dict):
                continue

            image_path = item.get("image_path")
            if not isinstance(image_path, str) or not image_path.strip():
                continue

            items.append(
                {
                    "visual_id": item.get("visual_id") or f"IMG-{idx + 1}",
                    "stable_visual_id": item.get("stable_visual_id"),
                    "image_path": image_path.strip(),
                    "chunk_id": item.get("chunk_id"),
                    "reference_id": item.get("reference_id"),
                    "image_chunk_index": idx,
                }
            )

        return items

    @classmethod
    def _build_visual_evidence_message(
        cls,
        image_items: list[dict[str, Any]],
        max_images: int,
        max_image_bytes: int,
    ) -> dict[str, Any] | None:
        """Build a synthetic multimodal message that binds each image to its image_chunk."""
        if max_images <= 0 or not image_items:
            return None

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{cls._VISUAL_EVIDENCE_PREFIX}\n"
                    "The retrieve tool returned the following images as visual evidence. "
                    "Each image is preceded by metadata. Match each image to "
                    "`retrieve.evidence.image_chunks[*]` by `visual_id`, `chunk_id`, "
                    "`reference_id`, or `image_path`. Do not mix facts across different visual_ids."
                ),
            }
        ]

        seen_paths: set[str] = set()
        attached_count = 0
        skipped_visual_ids: list[str] = []

        for item_index, item in enumerate(image_items):
            image_path = item.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                continue

            visual_id = str(item.get("visual_id") or f"IMG-{attached_count + 1}")
            if image_path in seen_paths:
                skipped_visual_ids.append(visual_id)
                continue
            seen_paths.add(image_path)

            data_url = cls._image_path_to_data_url(image_path, max_bytes=max_image_bytes)
            if not data_url:
                skipped_visual_ids.append(visual_id)
                continue

            content.append(
                {
                    "type": "text",
                    "text": (
                        f"visual_id: {visual_id}\n"
                        f"stable_visual_id: {item.get('stable_visual_id')}\n"
                        f"chunk_id: {item.get('chunk_id')}\n"
                        f"reference_id: {item.get('reference_id')}\n"
                        f"image_path: {image_path}\n"
                        f"corresponds_to: retrieve.evidence.image_chunks[{item.get('image_chunk_index')}]"
                    ),
                }
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})

            attached_count += 1
            if attached_count >= max_images:
                remaining = image_items[item_index + 1 :]
                skipped_visual_ids.extend(
                    str(remaining_item.get("visual_id") or "")
                    for remaining_item in remaining
                    if isinstance(remaining_item, dict) and remaining_item.get("visual_id")
                )
                break

        if attached_count == 0:
            return None

        if skipped_visual_ids:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Not all retrieved images were attached. These visual_ids have only text "
                        f"evidence in the retrieve JSON or were skipped: {', '.join(skipped_visual_ids)}"
                    ),
                }
            )

        return {"role": "user", "content": content}

    @staticmethod
    def _image_path_to_data_url(image_path: str, max_bytes: int) -> str | None:
        path = Path(image_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return None

        if max_bytes > 0 and path.stat().st_size > max_bytes:
            return None

        mime_type, _ = mimetypes.guess_type(str(path))
        allowed_mime_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if mime_type not in allowed_mime_types:
            return None

        try:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        except OSError:
            return None

        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_session_key(
        file_path: str | Path | None,
        parse_method: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Build file-scoped session key, inspired by processor cache-key logic."""
        if not file_path:
            return "chat:default"

        path = Path(file_path).expanduser().resolve()
        mtime = path.stat().st_mtime if path.exists() else None

        config_dict: dict[str, Any] = {
            "file_path": str(path),
            "mtime": mtime,
            "parse_method": parse_method,
        }

        relevant_keys = {
            "lang",
            "device",
            "start_page",
            "end_page",
            "formula",
            "table",
            "backend",
            "source",
        }
        for key, value in kwargs.items():
            if key in relevant_keys:
                config_dict[key] = value

        key_payload = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        key_hash = hashlib.md5(key_payload.encode()).hexdigest()
        return f"file:{key_hash}"

    @staticmethod
    def _save_turn(session: Session, messages: list[dict[str, Any]], skip: int) -> None:
        """Save only new turn messages into session history."""
        for msg in messages[skip:]:
            entry = dict(msg)
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue

            if role == "user" and AgentLoop._is_visual_evidence_message(content):
                continue

            if role == "user" and isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                parts = content.split("\n\n", 1)
                if len(parts) > 1 and parts[1].strip():
                    entry["content"] = parts[1]
                else:
                    continue

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)

        session.updated_at = datetime.now()

    @classmethod
    def _is_visual_evidence_message(cls, content: Any) -> bool:
        if not isinstance(content, list) or not content:
            return False
        first = content[0]
        if not isinstance(first, dict):
            return False
        text = first.get("text")
        return isinstance(text, str) and text.startswith(cls._VISUAL_EVIDENCE_PREFIX)
