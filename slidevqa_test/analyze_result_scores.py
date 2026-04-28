from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze existing SlideVQA result.json scores without re-running judge."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=SCRIPT_DIR / "runs_dataset",
        help="Root containing run-id directories, e.g. slidevqa_test/runs_dataset.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="default",
        help="Run id under --runs-root. Use --all-runs to scan every run id.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Scan all run directories under --runs-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the analysis report.",
    )
    parser.add_argument(
        "--list-lowest",
        type=int,
        default=10,
        help="Print the N lowest-scoring QA items.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def iter_result_paths(args: argparse.Namespace) -> list[Path]:
    if args.all_runs:
        return sorted(args.runs_root.glob("*/decks/*/result.json"))
    return sorted((args.runs_root / args.run_id / "decks").glob("*/result.json"))


def coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, score))


def score_bucket(score: float) -> str:
    if score >= 90:
        return "90-100"
    if score >= 80:
        return "80-89"
    if score >= 60:
        return "60-79"
    if score >= 40:
        return "40-59"
    if score > 0:
        return "1-39"
    return "0"


def analyze_results(result_paths: list[Path]) -> dict[str, Any]:
    qa_items: list[dict[str, Any]] = []
    deck_reports: list[dict[str, Any]] = []
    malformed_results: list[str] = []
    unscored_qa_count = 0

    for result_path in result_paths:
        payload = load_json(result_path)
        if not isinstance(payload, dict):
            malformed_results.append(str(result_path))
            continue

        deck_name = str(payload.get("deck_name", result_path.parent.name))
        qa_results = payload.get("qa_results")
        if not isinstance(qa_results, list):
            malformed_results.append(str(result_path))
            continue

        deck_scores: list[float] = []
        for qa in qa_results:
            if not isinstance(qa, dict):
                unscored_qa_count += 1
                continue
            judgement = qa.get("llm_judgement")
            if not isinstance(judgement, dict):
                unscored_qa_count += 1
                continue
            score = coerce_score(judgement.get("score"))
            if score is None:
                unscored_qa_count += 1
                continue

            deck_scores.append(score)
            qa_items.append(
                {
                    "score": score,
                    "deck_name": deck_name,
                    "qa_id": qa.get("qa_id"),
                    "row_index": qa.get("row_index"),
                    "question": qa.get("question"),
                    "ground_truth_answer": qa.get("ground_truth_answer"),
                    "reason": judgement.get("reason"),
                    "result_path": str(result_path),
                }
            )

        deck_reports.append(
            {
                "deck_name": deck_name,
                "result_path": str(result_path),
                "qa_count": len(qa_results),
                "scored_qa_count": len(deck_scores),
                "average_score": sum(deck_scores) / len(deck_scores) if deck_scores else None,
                "min_score": min(deck_scores) if deck_scores else None,
                "max_score": max(deck_scores) if deck_scores else None,
            }
        )

    scores = [item["score"] for item in qa_items]
    buckets = {"90-100": 0, "80-89": 0, "60-79": 0, "40-59": 0, "1-39": 0, "0": 0}
    for score in scores:
        buckets[score_bucket(score)] += 1

    return {
        "result_file_count": len(result_paths),
        "valid_result_file_count": len(deck_reports),
        "malformed_result_file_count": len(malformed_results),
        "malformed_results": malformed_results,
        "scored_qa_count": len(scores),
        "unscored_qa_count": unscored_qa_count,
        "average_score": sum(scores) / len(scores) if scores else None,
        "median_score": median(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "score_buckets": buckets,
        "lowest_qa": sorted(qa_items, key=lambda item: item["score"]),
        "deck_reports": sorted(
            deck_reports,
            key=lambda item: (
                item["average_score"] is None,
                item["average_score"] if item["average_score"] is not None else 101,
            ),
        ),
    }


def main() -> None:
    args = parse_args()
    result_paths = iter_result_paths(args)
    report = analyze_results(result_paths)

    printable = dict(report)
    lowest_limit = max(0, args.list_lowest)
    printable["lowest_qa"] = report["lowest_qa"][:lowest_limit]

    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    if args.output is not None:
        save_json(args.output, report)


if __name__ == "__main__":
    main()
