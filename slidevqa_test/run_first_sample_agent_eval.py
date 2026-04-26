from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "cache" / "decks"

load_dotenv(PROJECT_ROOT / ".env")

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from rag_agent.agent.loop import AgentLoop
from rag_agent.llm import AnthropicProvider, OpenAIProvider
from raganything import RAGAnything, RAGAnythingConfig


def get_env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    return cleaned.strip("._-") or "sample"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deck_cache_dir(deck_name: str) -> Path:
    return CACHE_ROOT / sanitize_component(deck_name)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def record_timing(timings: dict[str, float], name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    timings[name] = elapsed
    print(f"[TIMING] {name}: {format_duration(elapsed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agent-based evaluation on the first SlideVQA sample."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=SCRIPT_DIR / "test-slidevqa-1.parquet",
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Persistent output root for this evaluation run.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Override the sample question. Defaults to the question in the first row.",
    )
    parser.add_argument(
        "--use-llm-for-topics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM-based page topic extraction.",
    )
    parser.add_argument(
        "--qa-groups",
        type=Path,
        default=SCRIPT_DIR / "cache" / "qa_groups" / "by_deck_name.json",
        help="Preprocessed grouped QA payload produced by preprocess_slidevqa_groups.py.",
    )
    return parser.parse_args()


def extract_image_bytes(page_value: Any) -> tuple[bytes, str]:
    if not isinstance(page_value, dict):
        raise TypeError(f"Unsupported page value type: {type(page_value).__name__}")

    image_bytes = page_value.get("bytes")
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError(
            "Expected page entry to contain bytes-like data under the 'bytes' key"
        )

    original_name = page_value.get("path") or "page_image.jpg"
    return bytes(image_bytes), str(original_name)


def save_first_sample_images(row: pd.Series, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_pages: list[dict[str, Any]] = []
    for page_number in range(1, 21):
        column_name = f"page_{page_number}"
        if column_name not in row.index:
            continue

        page_value = row[column_name]
        if page_value is None:
            continue

        image_bytes, original_name = extract_image_bytes(page_value)
        original_suffix = Path(original_name).suffix or ".jpg"
        image_name = f"page_{page_number:02d}_{Path(original_name).stem}{original_suffix}"
        image_path = output_dir / image_name
        image_path.write_bytes(image_bytes)

        saved_pages.append(
            {
                "page_number": page_number,
                "source_column": column_name,
                "image_path": str(image_path),
                "original_name": original_name,
            }
        )

    if not saved_pages:
        raise ValueError("No page images were found in the first row.")

    return saved_pages


def remap_page_indices(content_list: list[dict[str, Any]], page_number: int) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        cloned = dict(item)
        if cloned.get("page_idx") is None:
            cloned["page_idx"] = page_number
        else:
            cloned["page_idx"] = page_number
        cloned["source_page"] = page_number
        remapped.append(cloned)
    return remapped


def find_deck_group(grouped_payload: dict[str, Any], deck_name: str) -> dict[str, Any]:
    decks = grouped_payload.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Invalid QA groups payload: missing decks list.")

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        if str(deck.get("deck_name", "")).strip() == deck_name:
            return deck

    raise ValueError(f"Cannot find QA group for deck_name: {deck_name}")


async def answer_deck_questions(
    rag: RAGAnything,
    deck_qa_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qa_results: list[dict[str, Any]] = []

    for qa_item in deck_qa_items:
        question = str(qa_item.get("question", "")).strip()
        if not question:
            continue

        answer = await rag.aquery(question, mode="hybrid")

        qa_results.append(
            {
                "row_index": qa_item.get("row_index"),
                "qa_id": qa_item.get("qa_id"),
                "question": question,
                "ground_truth_answer": qa_item.get("answer"),
                "metadata": qa_item.get("metadata"),
                "query_result": {
                    "answer": answer,
                    "mode": "hybrid",
                },
            }
        )

    return qa_results


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def coerce_judge_payload(
    payload: dict[str, Any],
    *,
    judge_model: str,
    finish_reason: str,
    raw_content: str,
) -> dict[str, Any]:
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0

    confidence_value = max(0.0, min(1.0, confidence_value))
    is_correct = payload.get("is_correct", False)
    if isinstance(is_correct, str):
        is_correct = is_correct.strip().lower() in {"1", "true", "yes", "correct"}

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided by judge model."

    result = {
        "is_correct": bool(is_correct),
        "confidence": confidence_value,
        "reason": reason,
        "judge_model": judge_model,
        "judge_method": "llm",
        "finish_reason": finish_reason,
    }
    if raw_content and payload.get("reason") is None:
        result["raw_judge_response"] = raw_content
    return result


def build_answer_judge_messages(
    *,
    question: str,
    ground_truth_answer: Any,
    predicted_answer: Any,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict but fair evaluator for document QA results. "
        "Judge whether the predicted answer correctly answers the question "
        "when compared with the ground truth answer. Do not use outside knowledge. "
        "Return only valid JSON."
    )
    user_prompt = f"""Evaluate the predicted answer against the ground truth.

Return only this JSON object:
{{
  "is_correct": true or false,
  "confidence": a number from 0.0 to 1.0,
  "reason": "brief explanation"
}}

Rules:
- Semantic equivalence counts as correct; exact wording is not required.
- Units, abbreviations, punctuation, and formatting may differ if the meaning is equivalent.
- If the question asks for a percentage, an answer like "3.6" may match "3.6%" when context makes the unit clear.
- If the prediction restates the question, includes references, or adds explanation, judge only whether its answer content matches.
- If the prediction contains the ground truth but also gives a contradictory final answer, mark it incorrect.
- If the prediction refuses to answer or says information is unavailable while the ground truth has a concrete answer, mark it incorrect.

Question:
{question}

Ground truth answer:
{ground_truth_answer}

Predicted answer:
{predicted_answer}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def judge_single_qa_result(
    provider: OpenAIProvider,
    qa_result: dict[str, Any],
    *,
    judge_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    query_result = qa_result.get("query_result")
    predicted_answer = ""
    if isinstance(query_result, dict):
        predicted_answer = query_result.get("answer", "")

    messages = build_answer_judge_messages(
        question=str(qa_result.get("question", "")),
        ground_truth_answer=qa_result.get("ground_truth_answer", ""),
        predicted_answer=predicted_answer,
    )
    response = await provider.chat_with_retry(
        messages=messages,
        model=judge_model,
        max_tokens=max_tokens,
        temperature=0,
    )

    if response.finish_reason == "error":
        return {
            "is_correct": False,
            "confidence": 0.0,
            "reason": response.content or "Judge model returned an error.",
            "judge_model": judge_model,
            "judge_method": "error",
            "finish_reason": response.finish_reason,
            "needs_review": True,
        }

    raw_content = response.content or ""
    parsed = extract_json_object(raw_content)
    if parsed is None:
        return {
            "is_correct": False,
            "confidence": 0.0,
            "reason": "Judge model did not return valid JSON.",
            "judge_model": judge_model,
            "judge_method": "parse_error",
            "finish_reason": response.finish_reason,
            "needs_review": True,
            "raw_judge_response": raw_content,
        }

    return coerce_judge_payload(
        parsed,
        judge_model=judge_model,
        finish_reason=response.finish_reason,
        raw_content=raw_content,
    )


async def judge_qa_results(
    qa_results: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    judge_model = get_env_str("EVAL_JUDGE_MODEL", "gpt-4o")
    judge_api_key = get_env_str("EVAL_OPENAI_API_KEY", api_key).strip()
    judge_base_url = get_env_str("EVAL_OPENAI_BASE_URL", base_url).strip()
    judge_max_tokens = get_env_int("EVAL_JUDGE_MAX_TOKENS", 512)

    if OpenAIProvider is None:
        raise RuntimeError("OpenAIProvider is unavailable.")
    if not judge_api_key:
        raise RuntimeError("EVAL_OPENAI_API_KEY or OPENAI_API_KEY is required for judge evaluation.")

    provider = OpenAIProvider(
        api_key=judge_api_key,
        api_base=judge_base_url or None,
        default_model=judge_model,
    )

    correct = 0
    needs_review = 0
    for index, qa_result in enumerate(qa_results, start=1):
        judgement = await judge_single_qa_result(
            provider,
            qa_result,
            judge_model=judge_model,
            max_tokens=judge_max_tokens,
        )
        qa_result["llm_judgement"] = judgement
        if judgement.get("is_correct"):
            correct += 1
        if judgement.get("needs_review"):
            needs_review += 1
        print(
            f"[JUDGE] {index}/{len(qa_results)} "
            f"qa_id={qa_result.get('qa_id')} "
            f"is_correct={judgement.get('is_correct')} "
            f"confidence={judgement.get('confidence')}"
        )

    total = len(qa_results)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "needs_review": needs_review,
        "accuracy": correct / total if total else 0.0,
        "judge_model": judge_model,
        "judge_method": "llm",
    }


class EvaluationRAGService:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.engine_pool: dict[str, RAGAnything] = {}
        self.agent_loop_pool: dict[str, AgentLoop] = {}

    def get_engine(self, working_dir: str) -> RAGAnything:
        if working_dir in self.engine_pool:
            return self.engine_pool[working_dir]

        os.makedirs(working_dir, exist_ok=True)

        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser=get_env_str("PARSER", "mineru"),
            parse_method=get_env_str("PARSE_METHOD", "auto"),
            enable_image_processing=get_env_bool("ENABLE_IMAGE_PROCESSING", True),
            enable_table_processing=get_env_bool("ENABLE_TABLE_PROCESSING", True),
            enable_equation_processing=get_env_bool("ENABLE_EQUATION_PROCESSING", True),
        )

        text_model = get_env_str("TEXT_LLM_MODEL", "gpt-4o-mini")
        vlm_model = get_env_str("VLM_MODEL", get_env_str("VISION_LLM_MODEL", "gpt-4o"))
        embedding_model = get_env_str("EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_dim = get_env_int("EMBEDDING_DIM", 3072)
        embedding_max_token_size = get_env_int("EMBEDDING_MAX_TOKEN_SIZE", 8192)

        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return openai_complete_if_cache(
                text_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    vlm_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

            if image_data:
                payload_messages = []
                if system_prompt:
                    payload_messages.append({"role": "system", "content": system_prompt})
                payload_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                )
                return openai_complete_if_cache(
                    vlm_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=payload_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

            return llm_model_func(prompt, system_prompt, history_messages or [], **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_token_size,
            send_dimensions=True,
            func=lambda texts, embedding_dim=None: openai_embed.func(
                texts,
                model=embedding_model,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding_dim=embedding_dim,
            ),
        )

        rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        self.engine_pool[working_dir] = rag_instance
        return rag_instance

    def _build_agent_provider(self):
        agent_provider = get_env_str("AGENT_PROVIDER", "openai").strip().lower()
        agent_model = get_env_str("AGENT_MODEL", "gpt-4o")
        anthropic_api_key = get_env_str("ANTHROPIC_API_KEY", "")
        anthropic_base_url = get_env_str("ANTHROPIC_BASE_URL", "")

        if agent_provider == "openai":
            if OpenAIProvider is None:
                raise RuntimeError("OpenAIProvider is unavailable.")
            return OpenAIProvider(
                api_key=self.api_key,
                api_base=self.base_url,
                default_model=agent_model,
            )

        if agent_provider == "anthropic":
            if AnthropicProvider is None:
                raise RuntimeError("AnthropicProvider is unavailable.")
            return AnthropicProvider(
                api_key=anthropic_api_key or self.api_key,
                api_base=anthropic_base_url or self.base_url,
                default_model=agent_model,
            )

        raise RuntimeError(f"Unsupported AGENT_PROVIDER: {agent_provider}")

async def parse_pages_into_combined_content(
    engine: RAGAnything,
    saved_pages: list[dict[str, Any]],
    page_parse_root: Path,
    parse_method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined_content_list: list[dict[str, Any]] = []
    page_reports: list[dict[str, Any]] = []

    for page_info in saved_pages:
        page_number = int(page_info["page_number"])
        image_path = Path(page_info["image_path"])
        per_page_output_dir = page_parse_root / f"page_{page_number:02d}"
        per_page_output_dir.mkdir(parents=True, exist_ok=True)

        content_list, parsed_doc_id = await engine.parse_document(
            file_path=str(image_path),
            output_dir=str(per_page_output_dir),
            parse_method=parse_method,
            display_stats=False,
        )

        normalized_items = remap_page_indices(content_list, page_number)
        combined_content_list.extend(normalized_items)

        page_reports.append(
            {
                "page_number": page_number,
                "image_path": str(image_path),
                "parsed_doc_id": parsed_doc_id,
                "parsed_blocks": len(normalized_items),
            }
        )

    return combined_content_list, page_reports


async def main() -> None:
    total_started_at = time.perf_counter()
    timings: dict[str, float] = {}

    stage_started_at = time.perf_counter()
    args = parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")

    api_key = get_env_str("OPENAI_API_KEY", "").strip()
    base_url = get_env_str("OPENAI_BASE_URL", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for this evaluation script.")

    df = pd.read_parquet(args.parquet)
    if df.empty:
        raise ValueError("The parquet file is empty.")

    row = df.iloc[1]
    deck_name = str(row.get("deck_name", "sample")).strip() or "sample"
    qa_id = int(row.get("qa_id", 0))

    deck_slug = sanitize_component(deck_name)
    deck_dir = deck_cache_dir(deck_name)
    grouped_payload = load_json(args.qa_groups)
    if not isinstance(grouped_payload, dict):
        raise ValueError(f"Invalid grouped QA payload: {args.qa_groups}")
    deck_group = find_deck_group(grouped_payload, deck_name)
    deck_qa_items = deck_group.get("qa_items")
    if not isinstance(deck_qa_items, list) or not deck_qa_items:
        raise ValueError(f"No QA items found for deck_name: {deck_name}")
    record_timing(timings, "setup_and_load_inputs", stage_started_at)

    stage_started_at = time.perf_counter()
    run_name = f"{deck_slug}_qa_{qa_id:04d}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root = args.output_root / run_name
    source_image_dir = deck_dir / "source_pages"
    page_parse_root = deck_dir / "page_parse"
    results_root = run_root / "results"
    rag_storage_root = deck_dir / "rag_storage"
    doc_ref_path = deck_dir / "document_ref.txt"
    combined_content_path = deck_dir / "combined_content_list.json"
    page_reports_path = deck_dir / "page_reports.json"
    combined_doc_id_path = deck_dir / "combined_doc_id.json"
    combined_cache_key_path = deck_dir / "combined_cache_key.json"

    for path in [
        source_image_dir,
        page_parse_root,
        rag_storage_root,
        results_root,
        deck_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if not doc_ref_path.exists():
        doc_ref_path.write_text(
            "\n".join(
                [
                    f"deck_name: {deck_name}",
                    f"parquet: {args.parquet.resolve()}",
                ]
            ),
            encoding="utf-8",
        )

    saved_pages = save_first_sample_images(row, source_image_dir)
    save_json(deck_dir / "saved_pages.json", saved_pages)
    record_timing(timings, "prepare_run_dirs_and_images", stage_started_at)

    stage_started_at = time.perf_counter()
    service = EvaluationRAGService(api_key=api_key, base_url=base_url)
    engine = service.get_engine(str(rag_storage_root))
    record_timing(timings, "initialize_rag_engine", stage_started_at)

    stage_started_at = time.perf_counter()
    parse_cache_hit = False
    combined_content_list: list[dict[str, Any]]
    page_reports: list[dict[str, Any]]
    if combined_content_path.exists() and page_reports_path.exists():
        cached_content = load_json(combined_content_path)
        cached_reports = load_json(page_reports_path)
        if isinstance(cached_content, list) and isinstance(cached_reports, list):
            combined_content_list = cached_content
            page_reports = cached_reports
            parse_cache_hit = True
        else:
            combined_content_list, page_reports = await parse_pages_into_combined_content(
                engine=engine,
                saved_pages=saved_pages,
                page_parse_root=page_parse_root,
                parse_method=get_env_str("PARSE_METHOD", "auto"),
            )
            save_json(combined_content_path, combined_content_list)
            save_json(page_reports_path, page_reports)
    else:
        combined_content_list, page_reports = await parse_pages_into_combined_content(
            engine=engine,
            saved_pages=saved_pages,
            page_parse_root=page_parse_root,
            parse_method=get_env_str("PARSE_METHOD", "auto"),
        )
        save_json(combined_content_path, combined_content_list)
        save_json(page_reports_path, page_reports)
    record_timing(timings, "load_or_parse_pages", stage_started_at)

    stage_started_at = time.perf_counter()
    if combined_doc_id_path.exists():
        loaded_doc_id = load_json(combined_doc_id_path)
        combined_doc_id = str(loaded_doc_id) if loaded_doc_id else f"doc-{deck_slug}"
    else:
        combined_doc_id = f"doc-{deck_slug}"
        save_json(combined_doc_id_path, combined_doc_id)

    if combined_cache_key_path.exists():
        loaded_cache_key = load_json(combined_cache_key_path)
        combined_cache_key = (
            str(loaded_cache_key) if loaded_cache_key else f"deck-{deck_slug}-preparsed"
        )
    else:
        combined_cache_key = f"deck-{deck_slug}-preparsed"
        save_json(combined_cache_key_path, combined_cache_key)

    await engine.process_document_complete_with_page_topics(
        file_path=str(doc_ref_path),
        output_dir=str(deck_dir / "document_output"),
        parse_method=get_env_str("PARSE_METHOD", "auto"),
        pre_parsed_content_list=combined_content_list,
        pre_parsed_doc_id=combined_doc_id,
        pre_parsed_cache_key=combined_cache_key,
        file_name=f"{deck_slug}.pptx",
    )
    record_timing(timings, "build_or_update_rag_index", stage_started_at)

    stage_started_at = time.perf_counter()
    qa_results = await answer_deck_questions(
        rag=engine,
        deck_qa_items=deck_qa_items,
    )
    record_timing(timings, "answer_deck_questions", stage_started_at)

    llm_judgement_summary = None
    if get_env_bool("EVAL_JUDGE_ENABLED", True):
        stage_started_at = time.perf_counter()
        llm_judgement_summary = await judge_qa_results(
            qa_results,
            api_key=api_key,
            base_url=base_url,
        )
        record_timing(timings, "judge_answers", stage_started_at)
    else:
        timings["judge_answers"] = 0.0

    total_elapsed = time.perf_counter() - total_started_at
    timings["total"] = total_elapsed
    timing_summary = {
        "total_seconds": total_elapsed,
        "total_human": format_duration(total_elapsed),
        "parse_cache_hit": parse_cache_hit,
        "qa_count": len(qa_results),
        "seconds_per_qa": total_elapsed / len(qa_results) if qa_results else 0.0,
        "stages": {
            name: {
                "seconds": elapsed,
                "human": format_duration(elapsed),
            }
            for name, elapsed in timings.items()
        },
    }

    final_result = {
        "deck_name": deck_name,
        "qa_id": qa_id,
        "run_root": str(run_root),
        "deck_cache_dir": str(deck_dir),
        "doc_ref_path": str(doc_ref_path),
        "combined_doc_id": combined_doc_id,
        "combined_cache_key": combined_cache_key,
        "pages": page_reports,
        "combined_blocks": len(combined_content_list),
        "qa_group": {
            "deck_name": deck_group.get("deck_name", deck_name),
            "qa_count": deck_group.get("qa_count", len(deck_qa_items)),
            "qa_items": deck_qa_items,
        },
        "qa_results": qa_results,
        "timing_summary": timing_summary,
    }
    if llm_judgement_summary is not None:
        final_result["llm_judgement_summary"] = llm_judgement_summary

    save_json(results_root / "result.json", final_result)
    print(json.dumps(final_result, ensure_ascii=False, indent=2))
    print("[TIMING] program_finished:", format_duration(time.perf_counter() - total_started_at))


if __name__ == "__main__":
    asyncio.run(main())
