from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export page images from the first row of a SlideVQA parquet file."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=SCRIPT_DIR / "test-slidevqa-1.parquet",
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "slidevqa_first_row_images",
        help="Directory where the first row images will be saved.",
    )
    return parser.parse_args()


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", value.strip())
    return cleaned or "first_row"


def extract_image_bytes(page_value: Any) -> tuple[bytes, str]:
    if isinstance(page_value, dict):
        if "bytes" in page_value:
            image_bytes = page_value["bytes"]
            file_name = page_value.get("path") or "page_image.jpg"
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise TypeError(
                    "Expected page['bytes'] to be bytes-like, "
                    f"got {type(image_bytes).__name__}"
                )
            return bytes(image_bytes), str(file_name)

        if "path" in page_value:
            raise ValueError("Page entry contains a path but no bytes payload.")

    raise TypeError(f"Unsupported page value type: {type(page_value).__name__}")


def main() -> None:
    args = parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    if df.empty:
        raise ValueError("The parquet file is empty.")

    first_row = df.iloc[0]
    deck_name = sanitize_component(str(first_row.get("deck_name", "first_row")))
    row_output_dir = args.output_dir / deck_name
    row_output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for page_index in range(1, 21):
        column_name = f"page_{page_index}"
        if column_name not in first_row.index:
            continue

        page_value = first_row[column_name]
        if page_value is None:
            continue

        image_bytes, original_name = extract_image_bytes(page_value)
        original_suffix = Path(original_name).suffix or ".jpg"
        output_name = f"{column_name}_{Path(original_name).stem}{original_suffix}"
        output_path = row_output_dir / output_name
        output_path.write_bytes(image_bytes)
        saved_files.append(output_path)

    if not saved_files:
        raise ValueError("No page images were found in the first row.")

    print(f"Saved {len(saved_files)} images to {row_output_dir}")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()