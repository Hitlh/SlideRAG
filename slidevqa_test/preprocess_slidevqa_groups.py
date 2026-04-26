from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess SlideVQA parquet into deck-level QA groups "
            "without image payload columns."
        )
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(__file__).resolve().parent / "test-slidevqa-2.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "cache"
        / "qa_groups"
        / "by_deck_name2.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--sort-by-qa-id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort each deck group's QA list by qa_id when available.",
    )
    return parser.parse_args()


def find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def is_page_image_column(column_name: str) -> bool:
    return re.fullmatch(r"page_\d+", column_name.lower()) is not None


def is_missing(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, bytes, bytearray)):
        return False

    try:
        na_result = pd.isna(value)
    except Exception:
        return False

    if isinstance(na_result, bool):
        return na_result

    try:
        return bool(na_result.all())
    except Exception:
        return False


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return None
    if is_missing(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [normalize_scalar(v) for v in value]

    if isinstance(value, dict):
        # Keep only lightweight dict values and drop binary payload-like keys.
        lightweight: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key.lower() in {"bytes", "binary", "image_bytes", "content"}:
                continue
            if isinstance(v, (bytes, bytearray)):
                continue
            lightweight[key] = normalize_scalar(v)
        return lightweight

    return str(value)


def extract_qa_item(
    row: pd.Series,
    row_index: int,
    deck_col: str,
    qa_id_col: str | None,
    question_col: str | None,
    answer_col: str | None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "row_index": row_index,
        "deck_name": str(row.get(deck_col, "")),
    }

    if qa_id_col is not None:
        item["qa_id"] = normalize_scalar(row.get(qa_id_col))
    if question_col is not None:
        item["question"] = normalize_scalar(row.get(question_col))
    if answer_col is not None:
        item["answer"] = normalize_scalar(row.get(answer_col))

    metadata: dict[str, Any] = {}
    for column_name in row.index:
        if column_name in {deck_col, qa_id_col, question_col, answer_col}:
            continue
        if is_page_image_column(column_name):
            continue

        value = row.get(column_name)
        if isinstance(value, (bytes, bytearray)):
            continue
        if isinstance(value, dict) and "bytes" in {str(k).lower() for k in value.keys()}:
            continue

        normalized = normalize_scalar(value)
        if normalized is None:
            continue
        metadata[column_name] = normalized

    if metadata:
        item["metadata"] = metadata

    return item


def build_grouped_payload(df: pd.DataFrame, sort_by_qa_id: bool) -> dict[str, Any]:
    columns = list(df.columns)
    deck_col = find_first_column(columns, ["deck_name", "deck", "deckid", "deck_id"])
    if deck_col is None:
        raise ValueError("Cannot find deck_name column in parquet.")

    qa_id_col = find_first_column(columns, ["qa_id", "qid", "question_id", "id"])
    question_col = find_first_column(columns, ["question", "query", "prompt"])
    answer_col = find_first_column(columns, ["answer", "ground_truth", "label", "target"])

    grouped: dict[str, dict[str, Any]] = {}
    for row_index, row in df.iterrows():
        deck_name = str(row.get(deck_col, "")).strip() or "unknown_deck"
        qa_item = extract_qa_item(
            row=row,
            row_index=int(row_index),
            deck_col=deck_col,
            qa_id_col=qa_id_col,
            question_col=question_col,
            answer_col=answer_col,
        )

        if deck_name not in grouped:
            grouped[deck_name] = {
                "deck_name": deck_name,
                "qa_items": [],
            }
        grouped[deck_name]["qa_items"].append(qa_item)

    decks = list(grouped.values())
    for deck in decks:
        if sort_by_qa_id:
            deck["qa_items"].sort(
                key=lambda x: (
                    x.get("qa_id") is None,
                    str(x.get("qa_id")),
                )
            )
        deck["qa_count"] = len(deck["qa_items"])

    decks.sort(key=lambda x: x["deck_name"])

    return {
        "generated_at": int(time.time()),
        "total_rows": int(len(df)),
        "deck_count": int(len(decks)),
        "schema": {
            "deck_column": deck_col,
            "qa_id_column": qa_id_col,
            "question_column": question_col,
            "answer_column": answer_col,
            "excluded_columns_rule": "page_* and binary payload columns",
        },
        "decks": decks,
    }


def main() -> None:
    args = parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    payload = build_grouped_payload(df=df, sort_by_qa_id=args.sort_by_qa_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_rows": payload["total_rows"],
                "deck_count": payload["deck_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
