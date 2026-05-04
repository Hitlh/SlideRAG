"""Minimal agent loop for RAG tool-calling workflow."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
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
        page_image_root: str | Path | None = None,
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
        self.page_image_root = Path(page_image_root).expanduser().resolve() if page_image_root else None
        self.max_forced_visual_images = int(
            self.retrieve_config.get("max_forced_visual_images", 1) or 1
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
        question = self._extract_question_from_messages(initial_messages)
        force_visual_verification = self._should_force_visual_verification(question)
        forced_visual_done = False

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

                    if (
                        tool_call.name == "retrieve"
                        and force_visual_verification
                        and not forced_visual_done
                    ):
                        auto_visual_results = await self._maybe_run_forced_image_understand(
                            retrieve_result=tool_result,
                            question=question,
                            messages=messages,
                            iteration=iteration,
                            tool_calls_count=tool_calls_count,
                        )
                        if auto_visual_results is not None:
                            messages, tool_calls_count, auto_tools_used = auto_visual_results
                            tools_used.extend(auto_tools_used)
                            forced_visual_done = True
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

    async def _maybe_run_forced_image_understand(
        self,
        *,
        retrieve_result: str,
        question: str,
        messages: list[dict[str, Any]],
        iteration: int,
        tool_calls_count: int,
    ) -> tuple[list[dict[str, Any]], int, list[str]] | None:
        payload = self._parse_tool_json(retrieve_result)
        if payload.get("status") != "success":
            return None

        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return None

        image_chunks = evidence.get("image_chunks")
        if not isinstance(image_chunks, list):
            return None

        if (
            not self._is_strong_visual_question(question)
            and self._retrieval_has_direct_answer_signal(payload=payload, question=question)
        ):
            return None

        page_image_chunks = self._build_page_image_chunks_from_retrieval(payload)
        if self._is_strong_visual_question(question) and page_image_chunks:
            # Strong layout/chart questions need the complete slide more than OCR crops.
            image_chunks = [*page_image_chunks, *image_chunks]
        elif not image_chunks:
            image_chunks = page_image_chunks
            if not image_chunks:
                return None

        selected_chunks = self._select_relevant_image_chunks(
            image_chunks=image_chunks,
            question=question,
            max_chunks=max(1, self.max_forced_visual_images),
        )
        if not selected_chunks:
            return None

        auto_tools_used: list[str] = []
        image_tool = self.tools.get("image_understand")
        if image_tool is None:
            return None

        for image_index, image_chunk in enumerate(selected_chunks, start=1):
            if tool_calls_count >= self.max_tool_calls:
                break

            image_path = str(image_chunk.get("image_path", "")).strip()
            if not image_path:
                continue

            page_number = image_chunk.get("page_number") or image_chunk.get("source_page") or image_chunk.get("page_idx")
            prompt = self._build_forced_visual_prompt(question=question, page_number=page_number)
            tool_call_id = f"auto-image-understand-{iteration}-{image_index}"

            messages = self.context.add_assistant_message(
                messages,
                "Running targeted visual follow-up because retrieved evidence did not directly answer the question.",
                [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "image_understand",
                            "arguments": json.dumps(
                                {"image_path": image_path, "prompt": prompt},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            )
            tool_result = await image_tool.execute(image_path=image_path, prompt=prompt)
            messages = self.context.add_tool_result(
                messages,
                tool_call_id=tool_call_id,
                tool_name="image_understand",
                result=tool_result,
            )
            tool_calls_count += 1
            auto_tools_used.append("image_understand")

        if not auto_tools_used:
            return None

        return messages, tool_calls_count, auto_tools_used

    def _build_page_image_chunks_from_retrieval(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.page_image_root is None:
            return []

        page_contents: dict[int, list[str]] = {}
        evidence = payload.get("evidence")
        if isinstance(evidence, dict):
            chunks = evidence.get("chunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    page_number = self._coerce_positive_int(chunk.get("page_number"))
                    if page_number is None:
                        continue
                    content = chunk.get("content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    page_contents.setdefault(page_number, []).append(content.strip())

        metadata = payload.get("metadata")
        retrieved_pages = metadata.get("retrieved_pages") if isinstance(metadata, dict) else []
        if isinstance(retrieved_pages, list):
            for raw_page in retrieved_pages:
                page_number = self._coerce_positive_int(raw_page)
                if page_number is not None:
                    page_contents.setdefault(page_number, [])

        fallback_chunks: list[dict[str, Any]] = []
        for page_number, contents in page_contents.items():
            image_path = self._resolve_page_image_path(page_number)
            if image_path is None:
                continue

            retrieved_text = "\n".join(contents)
            fallback_chunks.append(
                {
                    "chunk_type": "page_image",
                    "is_image": True,
                    "page_number": page_number,
                    "image_path": str(image_path),
                    "content": (
                        f"Full slide page image for page {page_number}."
                        f"\nRetrieved text from this page:\n{retrieved_text}"
                    ),
                }
            )

        return fallback_chunks

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _resolve_page_image_path(self, page_number: int) -> Path | None:
        if self.page_image_root is None:
            return None

        direct_candidates = (
            self.page_image_root / f"page_{page_number:02d}.jpg",
            self.page_image_root / f"page_{page_number:02d}.jpeg",
            self.page_image_root / f"page_{page_number:02d}.png",
        )
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        for candidate in sorted(self.page_image_root.glob(f"page_{page_number:02d}_*")):
            if candidate.is_file() and candidate.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".gif",
                ".tif",
                ".tiff",
            }:
                return candidate
        return None

    @staticmethod
    def _extract_question_from_messages(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            match = re.search(r"Question:\s*(.+)", content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return content.strip()
        return ""

    @staticmethod
    def _should_force_visual_verification(question: str) -> bool:
        lowered = question.lower()
        # Force VLM only for questions whose answer depends on visual layout or
        # reading a named figure. Plain numeric/category comparisons are often
        # answered more reliably from retrieved OCR/chart text.
        explicit_visual_markers = (
            "pictured",
            "shown",
            "figure",
            "image",
            "photo",
            "diagram",
            "chart",
            "graph",
            "table",
            "flow chart",
            "flowchart",
            "screenshot",
            "visual",
            "illustration",
            "map",
        )
        spatial_or_label_markers = (
            "above",
            "below",
            "left",
            "right",
            "under",
            "over",
            "next to",
            "beside",
            "column",
            "row",
            "closest",
            "farthest",
            "label",
            "labeled",
            "dial",
            "icon",
            "color",
            "colour",
            "legend",
            "axis",
            "arrow",
            "node",
            "box",
        )
        layout_phrases = (
            "comes under",
            "fall under",
            "falls under",
            "listed under",
            "to the left of",
            "to the right of",
            "in the column",
            "in the row",
            "which treats conditions",
            "where is",
            "where are",
        )
        if AgentLoop._is_strong_visual_question(question):
            return True

        return any(
            re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", lowered)
            for marker in explicit_visual_markers + spatial_or_label_markers + layout_phrases
        )

    @classmethod
    def _is_strong_visual_question(cls, question: str) -> bool:
        lowered = question.lower()
        strong_markers = (
            "flow chart",
            "flowchart",
            "directly before",
            "directly after",
            "comes before",
            "comes after",
            "what follows",
            "comes between",
            "comes under",
            "listed under",
            "fall under",
            "falls under",
            "column to the left",
            "column to the right",
            "to the left of",
            "to the right of",
            "in the column",
            "in the row",
            "which label",
            "closest to",
            "arrow",
            "pictured",
            "percentage points",
            "will drop",
            "drop how many",
            "dropped how many",
        )
        if any(marker in lowered for marker in strong_markers):
            return True

        # General figure/image questions still need VLM when they ask for a
        # visual count or identification, but not for plain numeric comparisons.
        if re.search(r"\b(how many|what|which)\b", lowered) and re.search(
            r"\b(image|figure|diagram|chart|graph|screenshot)\b", lowered
        ):
            return True

        return False

    @classmethod
    def _retrieval_has_direct_answer_signal(cls, *, payload: dict[str, Any], question: str) -> bool:
        """Return True when retrieved text already looks sufficient for a concise answer.

        This is intentionally conservative: forced VLM is a rescue path for missing
        evidence, not a second judge that should override clear OCR/retrieval text.
        """
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            return False

        chunks: list[dict[str, Any]] = []
        for key in ("chunks", "image_chunks"):
            raw_items = evidence.get(key)
            if isinstance(raw_items, list):
                chunks.extend(item for item in raw_items if isinstance(item, dict))

        question_tokens = cls._significant_question_tokens(question)
        if not question_tokens:
            return False

        relevant_texts: list[str] = []
        quantitative_question = cls._is_quantitative_question(question)
        min_overlap = 1 if quantitative_question else min(2, len(question_tokens))
        for chunk in chunks:
            content = chunk.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            lowered = content.lower()
            overlap = sum(1 for token in question_tokens if token in lowered)
            if overlap >= min_overlap:
                relevant_texts.append(content)

        if not relevant_texts:
            return False

        combined = "\n".join(relevant_texts).lower()
        if quantitative_question:
            return cls._contains_numeric_answer_signal(combined)

        # Only skip VLM when retrieval text contains an explicit answer cue, not
        # merely overlapping topic words. Layout questions are handled before this.
        return cls._contains_factoid_answer_signal(combined, question_tokens)

    @staticmethod
    def _significant_question_tokens(question: str) -> set[str]:
        stopwords = {
            "about",
            "according",
            "after",
            "answer",
            "before",
            "between",
            "does",
            "from",
            "have",
            "image",
            "figure",
            "chart",
            "graph",
            "shown",
            "table",
            "that",
            "the",
            "this",
            "what",
            "when",
            "where",
            "which",
            "while",
            "with",
            "who",
            "whose",
            "will",
            "would",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) >= 3 and token not in stopwords
        }

    @staticmethod
    def _contains_factoid_answer_signal(text: str, question_tokens: set[str]) -> bool:
        answer_cue_patterns = (
            r"\b(?:is|are|was|were|include|includes|included|called|named|belong(?:s)? to|target(?:s)?)\b",
            r"\b(?:example of|type of|source of|position of|based in|located in)\b",
            r"\b(?:consists of|composed of|represented by|associated with)\b",
        )
        if not any(re.search(pattern, text) for pattern in answer_cue_patterns):
            return False

        # A cue without enough question overlap is usually just nearby context.
        return sum(1 for token in question_tokens if token in text) >= min(2, len(question_tokens))

    @staticmethod
    def _is_quantitative_question(question: str) -> bool:
        lowered = question.lower()
        quantitative_markers = (
            "how many",
            "how much",
            "percentage",
            "percent",
            "difference",
            "more",
            "less",
            "greater",
            "lower",
            "higher",
            "highest",
            "lowest",
            "most",
            "least",
            "average",
            "total",
            "increase",
            "decrease",
        )
        return any(marker in lowered for marker in quantitative_markers)

    @staticmethod
    def _contains_numeric_answer_signal(text: str) -> bool:
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage|mm|cm|m|kv|kwh|hours?|days?|years?)?\b", text):
            return True
        number_words = (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        )
        return any(re.search(rf"\b{word}\b", text) for word in number_words)

    @staticmethod
    def _parse_tool_json(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _select_relevant_image_chunks(
        image_chunks: list[dict[str, Any]],
        question: str,
        max_chunks: int = 1,
    ) -> list[dict[str, Any]]:
        question_tokens = AgentLoop._significant_question_tokens(question)
        scored_chunks: list[tuple[int, dict[str, Any]]] = []
        for chunk in image_chunks:
            if not isinstance(chunk, dict):
                continue
            image_path = str(chunk.get("image_path", "")).strip()
            if not image_path:
                continue
            haystacks = [
                str(chunk.get("content", "")),
                str(chunk.get("file_path", "")),
                str(chunk.get("page_number", "")),
                str(chunk.get("source_page", "")),
                str(chunk.get("page_idx", "")),
            ]
            lowered = " ".join(haystacks).lower()
            score = sum(1 for token in question_tokens if token in lowered)
            if chunk.get("chunk_type") == "page_image":
                score += 2
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        if not scored_chunks:
            return []

        best_score = scored_chunks[0][0]
        if best_score <= 0:
            return []

        return [chunk for _, chunk in scored_chunks[:max_chunks]]

    @staticmethod
    def _build_forced_visual_prompt(question: str, page_number: Any) -> str:
        page_hint = ""
        if page_number is not None:
            page_hint = f"Focus on slide page {page_number}. "
        return (
            f"{page_hint}"
            "Answer this question by inspecting the image directly. "
            "Pay close attention to labels, legends, relative positions, flow arrows, and small text. "
            "If the image does not clearly answer the question, say what is unclear instead of guessing. "
            f"Question: {question}"
        )

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

            if role == "user" and isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                parts = content.split("\n\n", 1)
                if len(parts) > 1 and parts[1].strip():
                    entry["content"] = parts[1]
                else:
                    continue

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)

        session.updated_at = datetime.now()
