from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR := Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from rag_agent.llm import OpenAIProvider
from run_first_sample_agent_eval import get_env_int, get_env_str, judge_single_qa_result


DEFAULT_QA_SAMPLE: dict[str, Any] = {
    "row_index": 63,
    "qa_id": 63,
    "question": "What is the content of Chapter IV?",
    "ground_truth_answer": "Notification and Acquisition",
    "metadata": {
        "deck_url": "https://www.slideshare.net/pkachare/1-new-land-acquisition-act-process-flow",
        "arithmetic_expression": "None",
        "evidence_pages": 3,
    },
    "query_result": {
        "answer": """### Content of Chapter IV - Notification and Acquisition

Chapter IV focuses on the processes related to notification and acquisition of land. It spans **Sections 11 to 30** of the legislative act and provides detailed guidelines for initiating and executing land acquisition. Key aspects of this chapter include:

1. **Preliminary Notification**: A legal process transitioning from Section 4 in the Old Act to Section 11 in the New Act. This step marks the formal announcement of the intention to acquire land.

2. **Declaration of Resettlement Area and Public Purpose**: Managed by a secretary or authorized officer as per Section 19 in the New Act, this process ensures the proper declaration of areas meant for resettlement and purposes for land acquisition.

3. **Award Process**: Under the new act, awarding compensation and project-related provisions now extend across Sections 23, 30, and 31, showcasing broader and more detailed coverage compared to the old act, where it was confined to Section 11.

4. **Procedures to Correct Clerical Errors and Possession**: Sections 33 and 38 of the New Act address these matters, reinforcing administrative clarity during acquisition proceedings.

5. **Ref. to Court/Authority**: Section 64 in the New Act enables affected parties to seek judicial review, ensuring legal recourse and transparency during disputes.

Chapter IV aligns with broader themes like stakeholder consultation, fair processes, and compensation practices, emphasizing the importance of transparency and inclusiveness during land acquisition.

### References

- [1] 1-150711120533-lva1-app6892_95.pptx
- [2] document_ref.txt""",
        "mode": "hybrid",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score SlideVQA QA result(s) with the 0-100 LLM judge."
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="A result.json containing qa_results. If omitted, uses the built-in sample.",
    )
    parser.add_argument(
        "--qa-id",
        type=str,
        default="",
        help="QA id to score from --result-json. Defaults to the first qa_result.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all qa_results in --result-json and write a full scored result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the scored QA JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def select_qa_result(result_json: Path | None, qa_id: str) -> dict[str, Any]:
    if result_json is None:
        return dict(DEFAULT_QA_SAMPLE)

    payload = load_json(result_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid result JSON: {result_json}")

    qa_results = payload.get("qa_results")
    if not isinstance(qa_results, list) or not qa_results:
        raise ValueError(f"result JSON has no qa_results: {result_json}")

    if qa_id:
        for qa_result in qa_results:
            if str(qa_result.get("qa_id", "")) == qa_id:
                return dict(qa_result)
        raise ValueError(f"Cannot find qa_id={qa_id} in {result_json}")

    return dict(qa_results[0])


def load_result_payload(result_json: Path) -> dict[str, Any]:
    payload = load_json(result_json)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid result JSON: {result_json}")
    qa_results = payload.get("qa_results")
    if not isinstance(qa_results, list) or not qa_results:
        raise ValueError(f"result JSON has no qa_results: {result_json}")
    return payload


async def score_one(
    provider: OpenAIProvider,
    qa_result: dict[str, Any],
    *,
    judge_model: str,
    judge_max_tokens: int,
) -> dict[str, Any]:
    scored = dict(qa_result)
    judgement = await judge_single_qa_result(
        provider,
        scored,
        judge_model=judge_model,
        max_tokens=judge_max_tokens,
    )
    scored["llm_judgement"] = judgement
    return scored


async def score_all_result(
    provider: OpenAIProvider,
    payload: dict[str, Any],
    *,
    judge_model: str,
    judge_max_tokens: int,
) -> dict[str, Any]:
    qa_results = payload.get("qa_results")
    if not isinstance(qa_results, list):
        raise ValueError("payload has no qa_results list.")

    scored_results: list[dict[str, Any]] = []
    total_score = 0.0
    needs_review = 0
    for index, qa_result in enumerate(qa_results, start=1):
        if not isinstance(qa_result, dict):
            continue
        scored = await score_one(
            provider,
            qa_result,
            judge_model=judge_model,
            judge_max_tokens=judge_max_tokens,
        )
        scored_results.append(scored)
        judgement = scored.get("llm_judgement", {})
        if isinstance(judgement, dict):
            try:
                total_score += float(judgement.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            if judgement.get("needs_review"):
                needs_review += 1
        print(
            f"[SCORE] {index}/{len(qa_results)} "
            f"qa_id={scored.get('qa_id')} "
            f"score={scored.get('llm_judgement', {}).get('score')}"
        )

    result = dict(payload)
    result["qa_results"] = scored_results
    result["llm_judgement_summary"] = {
        "total": len(scored_results),
        "needs_review": needs_review,
        "total_score": total_score,
        "average_score": total_score / len(scored_results) if scored_results else 0.0,
        "score_scale": "0-100",
        "judge_model": judge_model,
        "judge_method": "llm",
    }
    return result


async def main() -> None:
    args = parse_args()
    api_key = get_env_str("EVAL_OPENAI_API_KEY", get_env_str("OPENAI_API_KEY", "")).strip()
    base_url = get_env_str("EVAL_OPENAI_BASE_URL", get_env_str("OPENAI_BASE_URL", "")).strip()
    judge_model = get_env_str("EVAL_JUDGE_MODEL", "gpt-4o")
    judge_max_tokens = get_env_int("EVAL_JUDGE_MAX_TOKENS", 512)

    if not api_key:
        raise RuntimeError("EVAL_OPENAI_API_KEY or OPENAI_API_KEY is required.")

    provider = OpenAIProvider(
        api_key=api_key,
        api_base=base_url or None,
        default_model=judge_model,
    )

    if args.all:
        if args.result_json is None:
            raise ValueError("--all requires --result-json")
        payload = load_result_payload(args.result_json)
        scored_payload = await score_all_result(
            provider,
            payload,
            judge_model=judge_model,
            judge_max_tokens=judge_max_tokens,
        )
        print(json.dumps(scored_payload["llm_judgement_summary"], ensure_ascii=False, indent=2))
        if args.output is not None:
            save_json(args.output, scored_payload)
        return

    qa_result = select_qa_result(args.result_json, args.qa_id)
    qa_result = await score_one(
        provider,
        qa_result,
        judge_model=judge_model,
        judge_max_tokens=judge_max_tokens,
    )

    print(json.dumps(qa_result, ensure_ascii=False, indent=2, default=str))
    if args.output is not None:
        save_json(args.output, qa_result)


if __name__ == "__main__":
    asyncio.run(main())
