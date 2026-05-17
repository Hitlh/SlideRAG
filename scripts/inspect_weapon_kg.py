from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import networkx as nx


ENTITY_COLORS = {
    "ScientificPrinciple": "#2563eb",
    "ApplicationTechnology": "#f97316",
    "SystemEffect": "#16a34a",
}

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

RELATION_ORDER = [
    "VerticalEnable",
    "HorizontalRelation",
]

RELATION_LABELS = {
    "VerticalEnable": "纵向使能关系",
    "HorizontalRelation": "层内横向关系",
    "Other": "其他关系",
}


def _workspace_dir(working_dir: Path, workspace: str | None) -> Path:
    return working_dir / workspace if workspace else working_dir


def _load_graph(storage_dir: Path) -> nx.Graph:
    graph_path = storage_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    return nx.read_graphml(graph_path)


def _extract_tag(description: str, key: str) -> str:
    marker = f"{key}="
    if marker not in description:
        return ""
    tail = description.split(marker, 1)[1]
    for sep in [";", "；", "\n"]:
        if sep in tail:
            tail = tail.split(sep, 1)[0]
            break
    return tail.strip()


def _entity_tags(entity_type: str, description: str) -> str:
    if entity_type == "ScientificPrinciple":
        return _extract_tag(description, "discipline")
    if entity_type == "ApplicationTechnology":
        domain = _extract_tag(description, "domain")
        based_on = _extract_tag(description, "based_on")
        return "; ".join(part for part in [f"domain={domain}" if domain else "", f"based_on={based_on}" if based_on else ""] if part)
    if entity_type == "SystemEffect":
        scenario = _extract_tag(description, "scenario")
        performance = _extract_tag(description, "performance")
        return "; ".join(part for part in [f"scenario={scenario}" if scenario else "", f"performance={performance}" if performance else ""] if part)
    return ""


def _normalize_entity_type(entity_type: str) -> str:
    cleaned = (entity_type or "").strip()
    if cleaned in ENTITY_ORDER:
        return cleaned
    return ENTITY_TYPE_ALIASES.get(cleaned.lower(), cleaned or "UNKNOWN")


def _relation_type(keywords: str) -> tuple[str, str]:
    parts = [part.strip() for part in (keywords or "").split(",") if part.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _truncate(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _print_table(headers: list[str], rows: list[list[Any]], limit: int) -> None:
    shown = rows[:limit]
    widths = [len(header) for header in headers]
    for row in shown:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(_truncate(cell, 80)))

    def fmt(row: list[Any]) -> str:
        cells = [_truncate(cell, 80) for cell in row]
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in shown:
        print(fmt(row))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows not shown")


def _group_key(value: str, allowed: list[str]) -> str:
    return value if value in allowed else "Other"


def _entity_groups(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {key: [] for key in ENTITY_ORDER + ["Other"]}
    for node in nodes:
        groups[_group_key(node["type"], ENTITY_ORDER)].append(node)
    return groups


def _relation_groups(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {key: [] for key in RELATION_ORDER + ["Other"]}
    for edge in edges:
        groups[_group_key(edge["type"], RELATION_ORDER)].append(edge)
    return groups


def _build_payload(graph: nx.Graph) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    for node_id, data in graph.nodes(data=True):
        entity_type = _normalize_entity_type(str(data.get("entity_type", "UNKNOWN")))
        description = str(data.get("description", ""))
        nodes.append(
            {
                "id": str(node_id),
                "type": entity_type,
                "tags": _entity_tags(entity_type, description),
                "description": description,
                "source_id": str(data.get("source_id", "")),
                "file_path": str(data.get("file_path", "")),
                "color": ENTITY_COLORS.get(entity_type, "#64748b"),
            }
        )

    edges = []
    for source, target, data in graph.edges(data=True):
        rel_type, rel_subtype = _relation_type(str(data.get("keywords", "")))
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "type": rel_type,
                "subtype": rel_subtype,
                "keywords": str(data.get("keywords", "")),
                "description": str(data.get("description", "")),
                "source_id": str(data.get("source_id", "")),
                "file_path": str(data.get("file_path", "")),
                "weight": str(data.get("weight", "")),
            }
        )
    return nodes, edges


def _write_html(storage_dir: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Path:
    output_path = storage_dir / "weapon_kg_view.html"
    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    safe_payload = html.escape(payload, quote=False)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Weapon Equipment Knowledge Graph</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 18px; border-bottom: 1px solid #e5e7eb; background: white; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 390px; height: calc(100vh - 58px); }}
    .content {{ min-width: 0; min-height: 0; overflow: hidden; background: #ffffff; }}
    aside {{ border-left: 1px solid #e5e7eb; padding: 14px; overflow: auto; background: #f9fafb; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }}
    .search {{ position: relative; margin-left: auto; min-width: 280px; }}
    .search input {{ width: 100%; height: 32px; border: 1px solid #cbd5e1; padding: 0 10px; font-size: 13px; outline: none; }}
    .search input:focus {{ border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12); }}
    .search-results {{ position: absolute; top: 36px; left: 0; right: 0; max-height: 260px; overflow: auto; border: 1px solid #cbd5e1; background: white; z-index: 5; display: none; }}
    .search-result {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; cursor: pointer; }}
    .search-result:hover {{ background: #f1f5f9; }}
    .search-result strong {{ display: block; font-size: 13px; }}
    .reset-button {{ height: 32px; border: 1px solid #cbd5e1; background: #ffffff; padding: 0 10px; font-size: 13px; cursor: pointer; }}
    .reset-button:hover {{ background: #f1f5f9; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
    svg {{ display: block; width: 100%; height: 100%; background: #ffffff; }}
    .cluster {{ fill: #f8fafc; stroke: #dbe4ef; stroke-width: 1.5; }}
    .cluster-title {{ font-size: 15px; font-weight: 700; fill: #111827; }}
    .cluster-count {{ font-size: 12px; fill: #64748b; }}
    .edge {{ cursor: pointer; }}
    .edge {{ fill: none; opacity: 0.22; }}
    .edge.vertical {{ stroke: #475569; stroke-width: 1.3; }}
    .edge.horizontal {{ stroke: #94a3b8; stroke-width: 1.1; stroke-dasharray: 7 5; opacity: 0.18; }}
    .edge.other {{ stroke: #cbd5e1; stroke-width: 1; stroke-dasharray: 3 4; opacity: 0.14; }}
    .edge:hover {{ stroke: #111827; stroke-width: 3; }}
    .edge.dim {{ opacity: 0.025; }}
    .edge.focus {{ stroke: #dc2626; stroke-width: 4; opacity: 1; }}
    .edges-muted .edge {{ opacity: 0.08; }}
    .edges-hidden .edge:not(.focus) {{ opacity: 0; pointer-events: none; }}
    .node {{ cursor: pointer; }}
    .node circle {{ stroke: white; stroke-width: 2.4; }}
    .node:hover circle {{ stroke: #111827; stroke-width: 3; }}
    .node text {{ font-size: 12px; font-weight: 700; fill: #ffffff; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
    .node.dim {{ opacity: 0.18; }}
    .node.focus circle {{ stroke: #dc2626; stroke-width: 4; }}
    .node.neighbor circle {{ stroke: #111827; stroke-width: 3; }}
    .hint {{ fill: #64748b; font-size: 12px; }}
    .muted {{ color: #6b7280; font-size: 12px; }}
    .item {{ margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb; }}
    .title {{ font-weight: 700; margin-bottom: 6px; }}
    .tag {{ margin-top: 4px; color: #475569; font-size: 12px; line-height: 1.35; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid #e5e7eb; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="legend">
      <strong>Weapon Equipment Knowledge Graph</strong>
      <span><i class="dot" style="background:#2563eb"></i>ScientificPrinciple</span>
      <span><i class="dot" style="background:#f97316"></i>ApplicationTechnology</span>
      <span><i class="dot" style="background:#16a34a"></i>SystemEffect</span>
      <span class="muted">solid=VerticalEnable, dashed=HorizontalRelation</span>
      <label class="muted"><input id="edge-mode" type="checkbox"> 仅显示选中实体相关边</label>
      <button id="reset-view" class="reset-button" type="button">重置视图</button>
      <div class="search">
        <input id="entity-search" type="search" placeholder="检索实体名称、标签或描述">
        <div id="search-results" class="search-results"></div>
      </div>
    </div>
  </header>
  <main>
    <section class="content">
      <svg id="graph" viewBox="0 0 1280 820" role="img" aria-label="Knowledge graph"></svg>
    </section>
    <aside id="details"></aside>
  </main>
  <script id="kg-data" type="application/json">{safe_payload}</script>
  <script>
    const data = JSON.parse(document.getElementById("kg-data").textContent);
    const svg = document.getElementById("graph");
    const details = document.getElementById("details");
    const searchInput = document.getElementById("entity-search");
    const searchResults = document.getElementById("search-results");
    const resetButton = document.getElementById("reset-view");
    const edgeMode = document.getElementById("edge-mode");
    const width = 1280;
    const height = 820;
    const entityOrder = ["ScientificPrinciple", "ApplicationTechnology", "SystemEffect"];
    const entityLabels = {{
      ScientificPrinciple: "科学原理层",
      ApplicationTechnology: "应用技术层",
      SystemEffect: "系统效果层",
      Other: "其他实体"
    }};
    const relationOrder = ["VerticalEnable", "HorizontalRelation", "Other"];
    const relationLabels = {{
      VerticalEnable: "纵向使能关系",
      HorizontalRelation: "层内横向关系",
      Other: "其他关系"
    }};
    const clusters = {{
      ScientificPrinciple: {{ x: 30, y: 70, w: 380, h: 610, cx: 220, cy: 375 }},
      ApplicationTechnology: {{ x: 450, y: 70, w: 380, h: 610, cx: 640, cy: 375 }},
      SystemEffect: {{ x: 870, y: 70, w: 380, h: 610, cx: 1060, cy: 375 }},
      Other: {{ x: 450, y: 700, w: 380, h: 90, cx: 640, cy: 745 }}
    }};
    let nodeElements = new Map();
    let edgeElements = [];

    function escapeHtml(s) {{
      return String(s || "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    function groupBy(items, keyFn) {{
      const groups = new Map();
      for (const item of items) {{
        const key = keyFn(item);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
      }}
      return groups;
    }}

    function shortLabel(name) {{
      const cleaned = String(name || "").replace(/[\\s_\\-()（）\\[\\]【】]/g, "");
      const chinese = cleaned.match(/[\\u4e00-\\u9fff]/g);
      if (chinese && chinese.length) return chinese.slice(0, Math.min(2, chinese.length)).join("");
      const acronym = String(name || "").match(/\\b[A-Za-z]/g);
      if (acronym && acronym.length >= 2) return acronym.slice(0, 3).join("").toUpperCase();
      return cleaned.slice(0, 2) || "?";
    }}

    function layoutNodes() {{
      const groups = groupBy(data.nodes, n => entityOrder.includes(n.type) ? n.type : "Other");
      for (const type of [...entityOrder, "Other"]) {{
        const items = groups.get(type) || [];
        const box = clusters[type];
        if (!box || !items.length) continue;
        const cols = Math.max(1, Math.ceil(Math.sqrt(items.length)));
        const rows = Math.max(1, Math.ceil(items.length / cols));
        const cellW = box.w / (cols + 1);
        const cellH = box.h / (rows + 1);
        items.forEach((node, idx) => {{
          const col = idx % cols;
          const row = Math.floor(idx / cols);
          node.x = box.x + cellW * (col + 1);
          node.y = box.y + cellH * (row + 1);
          node.r = 15;
        }});
      }}
    }}

    function svgEl(name, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      return el;
    }}

    function edgePath(source, target, idx) {{
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const length = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const normalX = -dy / length;
      const normalY = dx / length;
      const spread = ((idx % 9) - 4) * 10;
      const curve = source.type === target.type ? 44 + Math.abs(spread) : 18 + Math.abs(spread) * 0.55;
      const cx = (source.x + target.x) / 2 + normalX * (curve + spread);
      const cy = (source.y + target.y) / 2 + normalY * (curve + spread);
      return `M ${{source.x}} ${{source.y}} Q ${{cx}} ${{cy}} ${{target.x}} ${{target.y}}`;
    }}

    function drawGraph() {{
      svg.innerHTML = "";
      nodeElements = new Map();
      edgeElements = [];
      layoutNodes();
      const nodeById = new Map(data.nodes.map(n => [n.id, n]));
      const defs = svgEl("defs");
      const marker = svgEl("marker", {{ id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" }});
      marker.appendChild(svgEl("path", {{ d: "M 0 0 L 10 5 L 0 10 z", fill: "#475569" }}));
      defs.appendChild(marker);
      svg.appendChild(defs);

      for (const type of entityOrder) {{
        const box = clusters[type];
        const count = data.nodes.filter(n => n.type === type).length;
        svg.appendChild(svgEl("rect", {{ class: "cluster", x: box.x, y: box.y, width: box.w, height: box.h, rx: "8" }}));
        const title = svgEl("text", {{ class: "cluster-title", x: box.x + 14, y: box.y + 26 }});
        title.textContent = entityLabels[type];
        svg.appendChild(title);
        const countText = svgEl("text", {{ class: "cluster-count", x: box.x + box.w - 54, y: box.y + 26 }});
        countText.textContent = `${{count}} 个`;
        svg.appendChild(countText);
      }}

      const edgeLayer = svgEl("g");
      const nodeLayer = svgEl("g");
      svg.appendChild(edgeLayer);
      svg.appendChild(nodeLayer);

      data.edges.forEach((edge, idx) => {{
        const source = nodeById.get(edge.source);
        const target = nodeById.get(edge.target);
        if (!source || !target) return;
        const lineClass = edge.type === "VerticalEnable" ? "edge vertical" : edge.type === "HorizontalRelation" ? "edge horizontal" : "edge other";
        const line = svgEl("path", {{
          class: lineClass,
          d: edgePath(source, target, idx),
          "marker-end": edge.type === "VerticalEnable" ? "url(#arrow)" : ""
        }});
        line.addEventListener("click", () => showEdge(edge));
        edgeLayer.appendChild(line);
        edgeElements.push({{ edge, el: line }});
      }});

      for (const node of data.nodes) {{
        if (node.x === undefined || node.y === undefined) continue;
        const group = svgEl("g", {{ class: "node" }});
        group.addEventListener("click", () => showNode(node));
        group.appendChild(svgEl("circle", {{ cx: node.x, cy: node.y, r: node.r, fill: node.color || "#64748b" }}));
        const label = svgEl("text", {{ x: node.x, y: node.y + 0.5 }});
        label.textContent = shortLabel(node.id);
        group.appendChild(label);
        const tooltip = svgEl("title");
        tooltip.textContent = node.id;
        group.appendChild(tooltip);
        nodeLayer.appendChild(group);
        nodeElements.set(node.id, group);
      }}

      const hint = svgEl("text", {{ class: "hint", x: 30, y: 805 }});
      hint.textContent = "节点按实体类型聚集；节点内为短标签，悬停或点击查看完整实体名；实线箭头为纵向使能，虚线为层内横向关系。";
      svg.appendChild(hint);
    }}

    function clearFocus() {{
      for (const {{ el }} of edgeElements) el.classList.remove("dim", "focus");
      for (const el of nodeElements.values()) el.classList.remove("dim", "focus", "neighbor");
      svg.classList.remove("edges-hidden");
    }}

    function resetView() {{
      clearFocus();
      searchInput.value = "";
      searchResults.style.display = "none";
      details.innerHTML = `<div class="item"><div class="title">${{data.nodes.length}} entities, ${{data.edges.length}} relations</div>
        <div class="muted">Click an entity or relation to inspect extracted content.</div></div>`;
    }}

    function focusNode(nodeId) {{
      clearFocus();
      const connected = new Set([nodeId]);
      const connectedEdges = new Set();
      for (const item of edgeElements) {{
        if (item.edge.source === nodeId || item.edge.target === nodeId) {{
          connectedEdges.add(item);
          connected.add(item.edge.source);
          connected.add(item.edge.target);
        }}
      }}
      for (const item of edgeElements) {{
        if (connectedEdges.has(item)) item.el.classList.add("focus");
        else if (!edgeMode.checked) item.el.classList.add("dim");
      }}
      if (edgeMode.checked) svg.classList.add("edges-hidden");
      for (const [id, el] of nodeElements.entries()) {{
        if (id === nodeId) el.classList.add("focus");
        else if (connected.has(id)) el.classList.add("neighbor");
        else el.classList.add("dim");
      }}
    }}

    function focusEdge(edge) {{
      clearFocus();
      for (const item of edgeElements) {{
        if (item.edge === edge) item.el.classList.add("focus");
        else if (!edgeMode.checked) item.el.classList.add("dim");
      }}
      if (edgeMode.checked) svg.classList.add("edges-hidden");
      for (const [id, el] of nodeElements.entries()) {{
        if (id === edge.source || id === edge.target) el.classList.add("focus");
        else el.classList.add("dim");
      }}
    }}

    function setupSearch() {{
      searchInput.addEventListener("input", () => {{
        const query = searchInput.value.trim().toLowerCase();
        searchResults.innerHTML = "";
        if (!query) {{
          searchResults.style.display = "none";
          clearFocus();
          return;
        }}
        const matches = data.nodes.filter(node => {{
          const haystack = `${{node.id}} ${{node.type}} ${{node.tags}} ${{node.description}}`.toLowerCase();
          return haystack.includes(query);
        }}).slice(0, 20);
        if (!matches.length) {{
          searchResults.innerHTML = `<div class="search-result"><span class="muted">没有匹配实体</span></div>`;
          searchResults.style.display = "block";
          return;
        }}
        for (const node of matches) {{
          const item = document.createElement("div");
          item.className = "search-result";
          item.innerHTML = `<strong>${{escapeHtml(node.id)}}</strong><span class="muted">${{escapeHtml(entityLabels[node.type] || node.type)}} ${{escapeHtml(node.tags || "")}}</span>`;
          item.addEventListener("click", () => {{
            searchInput.value = node.id;
            searchResults.style.display = "none";
            showNode(node);
          }});
          searchResults.appendChild(item);
        }}
        searchResults.style.display = "block";
      }});
      searchInput.addEventListener("keydown", event => {{
        if (event.key === "Escape") {{
          searchInput.value = "";
          searchResults.style.display = "none";
          clearFocus();
        }}
      }});
      document.addEventListener("click", event => {{
        if (!event.target.closest(".search")) searchResults.style.display = "none";
      }});
      resetButton.addEventListener("click", resetView);
      edgeMode.addEventListener("change", () => {{
        const focused = [...nodeElements.entries()].find(([, el]) => el.classList.contains("focus"));
        if (focused) focusNode(focused[0]);
      }});
      svg.addEventListener("click", event => {{
        if (event.target === svg || event.target.classList.contains("cluster")) resetView();
      }});
    }}

    function showNode(node) {{
      focusNode(node.id);
      details.innerHTML = `<div class="item"><div class="title">${{escapeHtml(node.id)}}</div>
        <div class="muted">${{escapeHtml(node.type)}} ${{escapeHtml(node.tags)}}</div>
        <p>${{escapeHtml(node.description)}}</p>
        <div class="muted">source_id: ${{escapeHtml(node.source_id)}}</div>
        <div class="muted">file_path: ${{escapeHtml(node.file_path)}}</div></div>`;
    }}

    function showEdge(edge) {{
      focusEdge(edge);
      details.innerHTML = `<div class="item"><div class="title">${{escapeHtml(edge.source)}} -> ${{escapeHtml(edge.target)}}</div>
        <div class="muted">${{escapeHtml(edge.type)}} / ${{escapeHtml(edge.subtype)}} / ${{escapeHtml(edge.keywords)}}</div>
        <p>${{escapeHtml(edge.description)}}</p>
        <div class="muted">source_id: ${{escapeHtml(edge.source_id)}}</div>
        <div class="muted">file_path: ${{escapeHtml(edge.file_path)}}</div></div>`;
    }}

    drawGraph();
    setupSearch();
    resetView();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def inspect(working_dir: Path, workspace: str | None, limit: int) -> Path:
    storage_dir = _workspace_dir(working_dir, workspace)
    graph = _load_graph(storage_dir)
    nodes, edges = _build_payload(graph)

    print(f"storage_dir: {storage_dir}")
    print(f"entities: {len(nodes)}")
    print(f"relations: {len(edges)}")

    entity_groups = _entity_groups(nodes)
    for group_key in ENTITY_ORDER + ["Other"]:
        group_nodes = entity_groups[group_key]
        if not group_nodes:
            continue
        print(f"\nEntities / {ENTITY_LABELS[group_key]} ({len(group_nodes)})")
        _print_table(
            ["name", "type", "tags", "source_id", "description"],
            [
                [n["id"], n["type"], n["tags"], n["source_id"], n["description"]]
                for n in group_nodes
            ],
            limit,
        )

    relation_groups = _relation_groups(edges)
    for group_key in RELATION_ORDER + ["Other"]:
        group_edges = relation_groups[group_key]
        if not group_edges:
            continue
        print(f"\nRelations / {RELATION_LABELS[group_key]} ({len(group_edges)})")
        _print_table(
            ["source", "target", "type", "subtype", "keywords", "description"],
            [
                [
                    e["source"],
                    e["target"],
                    e["type"],
                    e["subtype"],
                    e["keywords"],
                    e["description"],
                ]
                for e in group_edges
            ],
            limit,
        )

    output_path = _write_html(storage_dir, nodes, edges)
    print(f"\nhtml: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize weapon-equipment concept graph extraction results."
    )
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    inspect(args.working_dir, args.workspace, args.limit)


if __name__ == "__main__":
    main()
