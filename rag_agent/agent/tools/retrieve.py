"""Retrieve tool backed by an initialized RAGAnything instance."""

from __future__ import annotations

import json
from typing import Any

from .base import Tool


class RetrieveTool(Tool):
    """Retrieve external knowledge chunks for a user query."""

    def __init__(
        self,
        rag: Any | None = None,
        mode: str = "hybrid",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
    ) -> None:
        self.rag = rag
        self.mode = mode
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k

    @property
    def name(self) -> str:
        return "retrieve"

    @property
    def description(self) -> str:
        return (
            "Retrieve relevant evidence from the prebuilt RAG knowledge base by query. "
            "Returns a JSON string with keys: status, query, mode, message, counts, evidence, metadata. "
            "evidence includes entities, relationships, chunks, references. "
            "Use this tool for retrieval only; the final user-facing answer is produced by the agent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User question or search query."},
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if self.rag is None:
            return (
                "retrieve tool is not ready: missing RAGAnything instance. "
                "Please inject an initialized rag instance before calling agent loop."
            )

        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "retrieve tool error: query is empty"

        query_param_kwargs: dict[str, Any] = {"mode": self.mode}
        if isinstance(self.top_k, int) and self.top_k > 0:
            query_param_kwargs["top_k"] = self.top_k
        if isinstance(self.chunk_top_k, int) and self.chunk_top_k > 0:
            query_param_kwargs["chunk_top_k"] = self.chunk_top_k

        query_param_cls = self._get_query_param_cls()
        if query_param_cls is None:
            return "retrieve tool failed: QueryParam import error; check lightrag installation"

        query_param = query_param_cls(**query_param_kwargs)

        try:
            raw = await self.rag.lightrag.aquery_data(query, query_param)
        except Exception as exc:
            return f"retrieve tool failed: {exc}"

        simplified = self._simplify_result(raw, query=query, mode=self.mode)
        return json.dumps(simplified, ensure_ascii=False)

    @staticmethod
    def _get_query_param_cls() -> Any | None:
        """Import QueryParam lazily to avoid hard import failure at module load time."""
        try:
            from lightrag import QueryParam as _QueryParam  # type: ignore
            return _QueryParam
        except Exception:
            pass

        try:
            from lightrag.base import QueryParam as _QueryParam  # type: ignore
            return _QueryParam
        except Exception:
            return None

    @staticmethod
    def _simplify_result(raw: dict[str, Any], query: str, mode: str) -> dict[str, Any]:
        """Convert full retrieval payload into concise tool output."""
        if not isinstance(raw, dict):
            return {
                "status": "failure",
                "query": query,
                "mode": mode,
                "message": "invalid retrieval result format",
                "evidence": {"entities": [], "relationships": [], "chunks": [], "references": []},
            }

        status = raw.get("status", "failure")
        message = raw.get("message", "")
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}

        entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        relationships = data.get("relationships") if isinstance(data.get("relationships"), list) else []
        chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
        references = data.get("references") if isinstance(data.get("references"), list) else []

        image_chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content")
            if not isinstance(content, str):
                continue
            if not RetrieveTool._is_image_analysis_chunk(content):
                continue

            image_item = {
                "chunk_type": "image_analysis",
                "is_image": True,
                "content": content,
            }
            if "chunk_id" in chunk:
                image_item["chunk_id"] = chunk.get("chunk_id")
            if "reference_id" in chunk:
                image_item["reference_id"] = chunk.get("reference_id")

            image_path = RetrieveTool._extract_image_path(content)
            if image_path:
                image_item["image_path"] = image_path

            image_chunks.append(image_item)

        def _pick(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {k: item.get(k) for k in keys if k in item}

        return {
            "status": status,
            "query": query,
            "mode": mode,
            "message": message,
            "counts": {
                "entities": len(entities),
                "relationships": len(relationships),
                "chunks": len(chunks),
                "image_chunks": len(image_chunks),
                "references": len(references),
            },
            "evidence": {
                "entities": [
                    _pick(
                        e,
                        (
                            "entity_name",
                            "entity_type",
                            "description",
                            "reference_id",
                        ),
                    )
                    for e in entities
                    if isinstance(e, dict)
                ],
                "relationships": [
                    _pick(
                        r,
                        (
                            "src_id",
                            "tgt_id",
                            "description",
                            "keywords",
                            "weight",
                            "reference_id",
                        ),
                    )
                    for r in relationships
                    if isinstance(r, dict)
                ],
                "chunks": [
                    _pick(c, ("content", "chunk_id", "reference_id"))
                    for c in chunks
                    if isinstance(c, dict)
                ],
                "image_chunks": image_chunks,
                "references": [
                    _pick(ref, ("reference_id", "file_path"))
                    for ref in references
                    if isinstance(ref, dict)
                ],
            },
            "metadata": raw.get("metadata", {}),
        }

    @staticmethod
    def _is_image_analysis_chunk(content: str) -> bool:
        lowered = content.lower()
        required_signals = ("image content analysis", "image path")
        if all(signal in lowered for signal in required_signals):
            return True
        return "visual analysis:" in lowered and "image path:" in lowered

    @staticmethod
    def _extract_image_path(content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("image path:"):
                value = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                return value or None
        return None
