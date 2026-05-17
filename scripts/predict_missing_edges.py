from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


load_env_file(PROJECT_ROOT / ".env")
if "TIKTOKEN_CACHE_DIR" in os.environ:
    os.environ["TIKTOKEN_CACHE_DIR"] = str(
        (PROJECT_ROOT / os.environ["TIKTOKEN_CACHE_DIR"]).resolve()
        if not Path(os.environ["TIKTOKEN_CACHE_DIR"]).is_absolute()
        else Path(os.environ["TIKTOKEN_CACHE_DIR"])
    )

from lightrag.llm.openai import openai_complete_if_cache, openai_embed


ENTITY_ORDER = [
    "ScientificPrinciple",
    "ApplicationTechnology",
    "SystemEffect",
]

ENTITY_LABELS = {
    "ScientificPrinciple": "科学原理层",
    "ApplicationTechnology": "应用技术层",
    "SystemEffect": "系统效果层",
    "Other": "其他实体",
}

ENTITY_TYPE_ALIASES = {
    "scientificprinciple": "ScientificPrinciple",
    "scientific_principle": "ScientificPrinciple",
    "applicationtechnology": "ApplicationTechnology",
    "application_technology": "ApplicationTechnology",
    "systemeffect": "SystemEffect",
    "system_effect": "SystemEffect",
}


def get_env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def workspace_dir(working_dir: Path, workspace: str | None) -> Path:
    return working_dir / workspace if workspace else working_dir


def normalize_entity_type(entity_type: str) -> str:
    cleaned = (entity_type or "").strip()
    if cleaned in ENTITY_ORDER:
        return cleaned
    return ENTITY_TYPE_ALIASES.get(cleaned.lower(), cleaned or "UNKNOWN")


def extract_tag(description: str, key: str) -> str:
    marker = f"{key}="
    if marker not in description:
        return ""
    tail = description.split(marker, 1)[1]
    for sep in [";", "；", "\n"]:
        if sep in tail:
            tail = tail.split(sep, 1)[0]
            break
    return tail.strip()


def entity_tags(entity_type: str, description: str) -> str:
    if entity_type == "ScientificPrinciple":
        return extract_tag(description, "discipline")
    if entity_type == "ApplicationTechnology":
        domain = extract_tag(description, "domain")
        based_on = extract_tag(description, "based_on")
        return "; ".join(
            part
            for part in [
                f"domain={domain}" if domain else "",
                f"based_on={based_on}" if based_on else "",
            ]
            if part
        )
    if entity_type == "SystemEffect":
        scenario = extract_tag(description, "scenario")
        performance = extract_tag(description, "performance")
        return "; ".join(
            part
            for part in [
                f"scenario={scenario}" if scenario else "",
                f"performance={performance}" if performance else "",
            ]
            if part
        )
    return ""


def load_graph(storage_dir: Path) -> nx.Graph:
    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    return nx.read_graphml(graph_path)


def load_entities(graph: nx.Graph) -> list[dict[str, Any]]:
    entities = []
    for node_id, data in graph.nodes(data=True):
        description = str(data.get("description", "") or "").strip()
        entity_type = normalize_entity_type(str(data.get("entity_type", "UNKNOWN")))
        if not description:
            continue
        tags = entity_tags(entity_type, description)
        entities.append(
            {
                "name": str(node_id),
                "entity_type": entity_type,
                "tags": tags,
                "description": description,
                "source_id": str(data.get("source_id", "")),
                "file_path": str(data.get("file_path", "")),
            }
        )
    return entities


def embedding_text(entity: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"实体名称：{entity['name']}",
            f"实体类型：{entity['entity_type']}",
            f"标签：{entity.get('tags', '')}",
            f"描述：{entity['description']}",
        ]
    )


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_embedding_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


async def ensure_embeddings(
    storage_dir: Path,
    entities: list[dict[str, Any]],
    batch_size: int,
    embedding_model: str,
    embedding_dim: int | None,
    api_key: str,
    base_url: str,
) -> dict[str, np.ndarray]:
    cache_path = storage_dir / "edge_prediction_entity_embeddings.json"
    cache = load_embedding_cache(cache_path)
    vectors: dict[str, np.ndarray] = {}
    missing: list[tuple[dict[str, Any], str, str]] = []

    for entity in entities:
        text = embedding_text(entity)
        digest = text_hash(text)
        cached = cache.get(entity["name"])
        if (
            isinstance(cached, dict)
            and cached.get("description_hash") == digest
            and isinstance(cached.get("embedding"), list)
        ):
            vectors[entity["name"]] = np.array(cached["embedding"], dtype=np.float32)
        else:
            missing.append((entity, text, digest))

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        texts = [item[1] for item in batch]
        # Match the rest of this project: call the underlying embedding function
        # directly so custom EMBEDDING_DIM values are honored. The decorated
        # openai_embed wrapper validates against its built-in 1536-dim default.
        embeddings = await openai_embed.func(
            texts,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
            embedding_dim=embedding_dim,
        )
        for (entity, text, digest), vector in zip(batch, embeddings):
            vector_list = [float(v) for v in vector]
            cache[entity["name"]] = {
                "entity_type": entity["entity_type"],
                "description_hash": digest,
                "embedding_text": text,
                "description": entity["description"],
                "embedding": vector_list,
            }
            vectors[entity["name"]] = np.array(vector_list, dtype=np.float32)
        save_embedding_cache(cache_path, cache)

    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def existing_edge_pairs(graph: nx.Graph) -> set[tuple[str, str]]:
    pairs = set()
    for source, target in graph.edges():
        pairs.add(tuple(sorted((str(source), str(target)))))
    return pairs


def candidate_relation(source_type: str, target_type: str) -> tuple[str, str]:
    pair = {source_type, target_type}
    if pair == {"ScientificPrinciple", "ApplicationTechnology"}:
        return "VerticalEnable", "principle_to_technology"
    if pair == {"ApplicationTechnology", "SystemEffect"}:
        return "VerticalEnable", "technology_to_effect"
    if source_type == target_type and source_type in ENTITY_ORDER:
        return "HorizontalRelation", "candidate_similarity"
    return "CandidateRelation", "semantic_similarity"


def type_filter_ok(
    source_type: str,
    target_type: str,
    same_type_only: bool,
    cross_type_only: bool,
    allowed_type_pairs: set[tuple[str, str]] | None,
) -> bool:
    if same_type_only and source_type != target_type:
        return False
    if cross_type_only and source_type == target_type:
        return False
    if allowed_type_pairs:
        pair = tuple(sorted((source_type, target_type)))
        if pair not in allowed_type_pairs:
            return False
    return True


def parse_type_pairs(raw_pairs: str | None) -> set[tuple[str, str]] | None:
    if not raw_pairs:
        return None
    pairs = set()
    for raw_pair in raw_pairs.split(","):
        if ":" not in raw_pair:
            continue
        left, right = raw_pair.split(":", 1)
        pairs.add(tuple(sorted((normalize_entity_type(left), normalize_entity_type(right)))))
    return pairs or None


def rank_candidates(
    graph: nx.Graph,
    entities: list[dict[str, Any]],
    vectors: dict[str, np.ndarray],
    top_k: int,
    min_score: float,
    same_type_only: bool,
    cross_type_only: bool,
    allowed_type_pairs: set[tuple[str, str]] | None,
) -> list[dict[str, Any]]:
    by_name = {entity["name"]: entity for entity in entities}
    existing = existing_edge_pairs(graph)
    results = []
    for i, source in enumerate(entities):
        for target in entities[i + 1 :]:
            pair = tuple(sorted((source["name"], target["name"])))
            if pair in existing:
                continue
            if not type_filter_ok(
                source["entity_type"],
                target["entity_type"],
                same_type_only,
                cross_type_only,
                allowed_type_pairs,
            ):
                continue
            score = cosine_similarity(vectors[source["name"]], vectors[target["name"]])
            if score < min_score:
                continue
            rel_type, subtype = candidate_relation(
                source["entity_type"], target["entity_type"]
            )
            results.append(
                {
                    "source": source["name"],
                    "target": target["name"],
                    "source_type": source["entity_type"],
                    "target_type": target["entity_type"],
                    "source_tags": source.get("tags", ""),
                    "target_tags": target.get("tags", ""),
                    "source_description": source["description"],
                    "target_description": target["description"],
                    "similarity": round(score, 6),
                    "existing_edge": False,
                    "candidate_relation_type": rel_type,
                    "candidate_subtype": subtype,
                }
            )
    results.sort(key=lambda row: row["similarity"], reverse=True)
    return results[:top_k]


def explanation_prompt(candidate: dict[str, Any]) -> str:
    return f"""你是科学创意知识图谱的边预测审查助手。请只基于两个实体的描述，解释它们为什么可能存在潜在概念关联。

安全约束：
1. 不生成新的武器设计方案。
2. 不提供制造步骤、参数配方、操作流程、战术建议或可执行使用建议。
3. 只解释概念层面的相似性、互补性、类比、协同或纵向使能可能性。
4. 如果证据不足，请降低置信度并说明风险。

请输出严格 JSON，不要输出 Markdown：
{{
  "relation_type": "VerticalEnable 或 HorizontalRelation 或 CandidateRelation",
  "subtype": "principle_to_technology / technology_to_effect / analogy / complement / synergy / substitute / conflict / tradeoff / semantic_similarity",
  "reason": "为什么两个实体可能有关联",
  "evidence": "仅基于实体描述可见的信息",
  "risk": "可能误判的原因",
  "confidence": 0.0
}}

候选关系：
相似度：{candidate['similarity']}
候选类型：{candidate['candidate_relation_type']} / {candidate['candidate_subtype']}

实体A：
名称：{candidate['source']}
类型：{candidate['source_type']}
标签：{candidate.get('source_tags', '')}
描述：{candidate['source_description']}

实体B：
名称：{candidate['target']}
类型：{candidate['target_type']}
标签：{candidate.get('target_tags', '')}
描述：{candidate['target_description']}
"""


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"raw_response": text}


async def explain_candidates(
    candidates: list[dict[str, Any]],
    explain_top_n: int,
    model: str,
    api_key: str,
    base_url: str,
) -> None:
    system_prompt = (
        "你负责审查知识图谱候选边，只能基于给定实体描述解释概念关联，"
        "不得提供操作性、制造性或战术性建议。"
    )
    for candidate in candidates[:explain_top_n]:
        response = await openai_complete_if_cache(
            model,
            explanation_prompt(candidate),
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
        )
        parsed = parse_llm_json(response)
        candidate["llm_relation_type"] = parsed.get("relation_type", "")
        candidate["llm_subtype"] = parsed.get("subtype", "")
        candidate["llm_reason"] = parsed.get("reason", "")
        candidate["llm_evidence"] = parsed.get("evidence", "")
        candidate["llm_risk"] = parsed.get("risk", "")
        candidate["llm_confidence"] = parsed.get("confidence", "")
        if parsed.get("raw_response"):
            candidate["llm_raw_response"] = parsed["raw_response"]


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields = [
        "source",
        "target",
        "source_type",
        "target_type",
        "similarity",
        "candidate_relation_type",
        "candidate_subtype",
        "llm_relation_type",
        "llm_subtype",
        "llm_confidence",
        "llm_reason",
        "llm_evidence",
        "llm_risk",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def write_html(path: Path, data: list[dict[str, Any]]) -> None:
    payload = html.escape(json.dumps(data, ensure_ascii=False), quote=False)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Edge Predictions</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    header {{ padding: 14px 18px; background: white; border-bottom: 1px solid #e5e7eb; position: sticky; top: 0; }}
    main {{ padding: 16px 18px; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    input, select {{ height: 32px; border: 1px solid #cbd5e1; padding: 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ text-align: left; background: #f1f5f9; position: sticky; top: 61px; }}
    tr:hover {{ background: #f8fafc; }}
    .score {{ font-weight: 700; color: #b91c1c; }}
    .muted {{ color: #64748b; font-size: 12px; }}
    .reason {{ max-width: 440px; }}
  </style>
</head>
<body>
  <header>
    <div class="toolbar">
      <strong>Edge Predictions</strong>
      <input id="query" type="search" placeholder="搜索实体或解释">
      <select id="relation-filter">
        <option value="">全部候选类型</option>
        <option value="VerticalEnable">VerticalEnable</option>
        <option value="HorizontalRelation">HorizontalRelation</option>
        <option value="CandidateRelation">CandidateRelation</option>
      </select>
      <span class="muted" id="count"></span>
    </div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>实体对</th>
          <th>类型</th>
          <th>相似度</th>
          <th>候选关系</th>
          <th>LLM解释</th>
          <th>风险</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script id="data" type="application/json">{payload}</script>
  <script>
    const data = JSON.parse(document.getElementById("data").textContent);
    const rows = document.getElementById("rows");
    const query = document.getElementById("query");
    const relationFilter = document.getElementById("relation-filter");
    const count = document.getElementById("count");
    function esc(s) {{
      return String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}
    function render() {{
      const q = query.value.trim().toLowerCase();
      const rel = relationFilter.value;
      const filtered = data.filter(row => {{
        const text = JSON.stringify(row).toLowerCase();
        return (!q || text.includes(q)) && (!rel || row.candidate_relation_type === rel || row.llm_relation_type === rel);
      }});
      count.textContent = `${{filtered.length}} / ${{data.length}}`;
      rows.innerHTML = filtered.map((row, idx) => `
        <tr>
          <td>${{idx + 1}}</td>
          <td><strong>${{esc(row.source)}}</strong><br><span class="muted">-> ${{esc(row.target)}}</span></td>
          <td>${{esc(row.source_type)}}<br><span class="muted">${{esc(row.target_type)}}</span></td>
          <td class="score">${{Number(row.similarity).toFixed(4)}}</td>
          <td>${{esc(row.candidate_relation_type)}}<br><span class="muted">${{esc(row.candidate_subtype)}}</span></td>
          <td class="reason"><strong>${{esc(row.llm_relation_type || "")}} ${{esc(row.llm_subtype || "")}}</strong><br>${{esc(row.llm_reason || "")}}<br><span class="muted">${{esc(row.llm_evidence || "")}}</span></td>
          <td>${{esc(row.llm_risk || "")}}<br><span class="muted">confidence=${{esc(row.llm_confidence || "")}}</span></td>
        </tr>`).join("");
    }}
    query.addEventListener("input", render);
    relationFilter.addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    load_env_file(PROJECT_ROOT / ".env")
    storage_dir = workspace_dir(args.working_dir, args.workspace)
    graph = load_graph(storage_dir)
    entities = load_entities(graph)
    if len(entities) < 2:
        raise ValueError("Need at least two entities with descriptions for edge prediction")

    api_key = args.api_key or get_env_str("OPENAI_API_KEY")
    base_url = args.base_url or get_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
    embedding_model = args.embedding_model or get_env_str(
        "EMBEDDING_MODEL", "text-embedding-3-large"
    )
    embedding_dim = args.embedding_dim or get_env_int("EMBEDDING_DIM", 0) or None
    llm_model = args.llm_model or get_env_str("TEXT_LLM_MODEL", "gpt-4o-mini")

    vectors = await ensure_embeddings(
        storage_dir=storage_dir,
        entities=entities,
        batch_size=args.embedding_batch_size,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        api_key=api_key,
        base_url=base_url,
    )
    candidates = rank_candidates(
        graph=graph,
        entities=entities,
        vectors=vectors,
        top_k=args.top_k,
        min_score=args.min_score,
        same_type_only=args.same_type_only,
        cross_type_only=args.cross_type_only,
        allowed_type_pairs=parse_type_pairs(args.type_pairs),
    )

    if args.llm_explain and candidates:
        explain_top_n = args.explain_top_n or len(candidates)
        await explain_candidates(
            candidates,
            explain_top_n=min(explain_top_n, len(candidates)),
            model=llm_model,
            api_key=api_key,
            base_url=base_url,
        )

    prefix = args.output_prefix or "edge_predictions"
    json_path = storage_dir / f"{prefix}.json"
    csv_path = storage_dir / f"{prefix}.csv"
    html_path = storage_dir / f"{prefix}.html"
    write_json(json_path, candidates)
    write_csv(csv_path, candidates)
    write_html(html_path, candidates)

    print(f"entities: {len(entities)}")
    print(f"candidates: {len(candidates)}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"html: {html_path}")
    for idx, row in enumerate(candidates[: min(10, len(candidates))], start=1):
        print(
            f"{idx}. {row['source']} -> {row['target']} "
            f"score={row['similarity']:.4f} "
            f"{row['candidate_relation_type']}/{row['candidate_subtype']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict high-similarity missing edges in a LightRAG GraphML knowledge graph."
    )
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--same-type-only", action="store_true")
    parser.add_argument("--cross-type-only", action="store_true")
    parser.add_argument(
        "--type-pairs",
        default=None,
        help="Comma-separated allowed type pairs, e.g. ScientificPrinciple:ApplicationTechnology,ApplicationTechnology:SystemEffect",
    )
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--explain-top-n", type=int, default=0)
    parser.add_argument("--output-prefix", default="edge_predictions")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    if args.same_type_only and args.cross_type_only:
        raise ValueError("--same-type-only and --cross-type-only cannot both be set")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
