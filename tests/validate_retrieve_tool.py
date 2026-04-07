"""Validate RetrieveTool with a fake RAGAnything-like object.

Usage:
    python -m tests.validate_retrieve_tool
"""

from __future__ import annotations

import asyncio
import json

from rag_agent.agent.tools.retrieve import RetrieveTool


class _FakeQueryParam:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeLightRAG:
    async def aquery_data(self, query: str, param) -> dict:
        _ = param  # only to mirror signature
        return {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "entities": [
                    {
                        "entity_name": "RAG",
                        "entity_type": "concept",
                        "description": "Retrieval-Augmented Generation",
                        "source_id": "chunk-1",
                        "file_path": "demo.pdf",
                        "reference_id": "[1]",
                    }
                ],
                "relationships": [],
                "chunks": [
                    {
                        "content": "RAG combines retrieval with generation.",
                        "file_path": "demo.pdf",
                        "chunk_id": "chunk-1",
                        "reference_id": "[1]",
                    },
                    {
                        "content": "\nImage Content Analysis:\nImage Path: /tmp/demo-image.jpg\nCaptions: None\nFootnotes: None\n\nVisual Analysis: A chart image about retrieval pipeline.",
                        "file_path": "demo.pdf",
                        "chunk_id": "chunk-2",
                        "reference_id": "[1]",
                    }
                ],
                "references": [{"reference_id": "[1]", "file_path": "demo.pdf"}],
            },
            "metadata": {"query_mode": "hybrid"},
        }


class _FakeRAG:
    def __init__(self) -> None:
        self.lightrag = _FakeLightRAG()


async def main() -> None:
    RetrieveTool._get_query_param_cls = staticmethod(lambda: _FakeQueryParam)
    tool = RetrieveTool(rag=_FakeRAG(), mode="hybrid", top_k=3, chunk_top_k=3)
    result = await tool.execute(query="what is rag")
    parsed = json.loads(result)

    assert parsed["status"] == "success"
    assert parsed["counts"]["chunks"] == 2
    assert parsed["counts"]["image_chunks"] == 1
    assert parsed["evidence"]["entities"][0]["entity_name"] == "RAG"
    assert parsed["evidence"]["image_chunks"][0]["is_image"] is True
    assert parsed["evidence"]["image_chunks"][0]["chunk_type"] == "image_analysis"
    assert parsed["evidence"]["image_chunks"][0]["image_path"] == "/tmp/demo-image.jpg"

    print("RETRIEVE_TOOL_TEST: PASS")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
