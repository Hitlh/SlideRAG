from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from preprocess_slidevqa_groups import build_grouped_payload


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ID = "default"


def sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    return cleaned.strip("._-") or "sample"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SlideVQA deck-level agent evaluation sequentially. "
            "Each deck is evaluated in a fresh worker process."
        )
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=SCRIPT_DIR / "test-slidevqa-1.parquet",
        help="Input SlideVQA parquet file.",
    )
    parser.add_argument(
        "--qa-groups",
        type=Path,
        default=SCRIPT_DIR / "cache" / "qa_groups" / "by_deck_name.json",
        help="Grouped QA JSON from preprocess_slidevqa_groups.py. Created if missing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "runs_dataset",
        help="Root directory for dataset-level manifests and deck results.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help="Stable run id. Reusing it enables resume with the same command.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=SCRIPT_DIR / "cache" / "decks",
        help="Shared deck cache root reused across runs.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N decks.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this zero-based index in the grouped deck list.",
    )
    parser.add_argument(
        "--only-deck",
        type=str,
        default="",
        help="Evaluate only this exact deck_name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run decks even when a valid result.json already exists.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM answer judging in worker processes.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the dataset run after the first failed deck.",
    )
    parser.add_argument(
        "--quiet-worker-logs",
        action="store_true",
        help="Do not mirror worker stdout/stderr to the current terminal.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp_path.replace(path)


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_qa_groups(parquet_path: Path, qa_groups_path: Path) -> dict[str, Any]:
    payload = load_json(qa_groups_path)
    if isinstance(payload, dict):
        return payload

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    payload = build_grouped_payload(df=df, sort_by_qa_id=True)
    save_json(qa_groups_path, payload)
    print(
        f"[SETUP] Created QA groups: {qa_groups_path} "
        f"(decks={payload.get('deck_count')}, rows={payload.get('total_rows')})"
    )
    return payload


def get_decks(grouped_payload: dict[str, Any]) -> list[dict[str, Any]]:
    decks = grouped_payload.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Invalid QA groups payload: missing decks list.")
    return [deck for deck in decks if isinstance(deck, dict)]


def deck_result_path(run_dir: Path, deck_name: str) -> Path:
    return run_dir / "decks" / sanitize_component(deck_name) / "result.json"


def result_is_complete(result_path: Path, deck_name: str, expected_qa_count: int, judge_enabled: bool) -> bool:
    payload = load_json(result_path)
    if not isinstance(payload, dict):
        return False
    if str(payload.get("deck_name", "")).strip() != deck_name:
        return False
    qa_results = payload.get("qa_results")
    if not isinstance(qa_results, list) or len(qa_results) != expected_qa_count:
        return False
    if judge_enabled and "llm_judgement_summary" not in payload:
        return False
    return True


def build_initial_manifest(args: argparse.Namespace, decks: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "parquet": str(args.parquet),
        "qa_groups": str(args.qa_groups),
        "output_root": str(args.output_root),
        "cache_root": str(args.cache_root),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "deck_count": len(decks),
        "decks": {
            str(deck.get("deck_name", "")).strip(): {
                "deck_name": str(deck.get("deck_name", "")).strip(),
                "qa_count": int(deck.get("qa_count", len(deck.get("qa_items", [])))),
                "status": "pending",
                "result_path": str(deck_result_path(run_dir, str(deck.get("deck_name", "")).strip())),
                "attempt_count": 0,
            }
            for deck in decks
            if str(deck.get("deck_name", "")).strip()
        },
    }


def load_or_create_manifest(args: argparse.Namespace, decks: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    existing = load_json(manifest_path)
    if isinstance(existing, dict) and isinstance(existing.get("decks"), dict):
        manifest = existing
    else:
        manifest = build_initial_manifest(args=args, decks=decks, run_dir=run_dir)

    manifest_decks = manifest.setdefault("decks", {})
    for deck in decks:
        deck_name = str(deck.get("deck_name", "")).strip()
        if not deck_name:
            continue
        manifest_decks.setdefault(
            deck_name,
            {
                "deck_name": deck_name,
                "status": "pending",
                "attempt_count": 0,
            },
        )
        entry = manifest_decks[deck_name]
        entry["qa_count"] = int(deck.get("qa_count", len(deck.get("qa_items", []))))
        entry["result_path"] = str(deck_result_path(run_dir, deck_name))

    manifest["updated_at"] = int(time.time())
    manifest["deck_count"] = len(decks)
    save_json(manifest_path, manifest)
    return manifest


def select_decks(args: argparse.Namespace, decks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = decks
    if args.only_deck:
        selected = [
            deck
            for deck in selected
            if str(deck.get("deck_name", "")).strip() == args.only_deck.strip()
        ]
    if args.start_index:
        selected = selected[args.start_index :]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def write_summary(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    entries = list(manifest.get("decks", {}).values())
    success = sum(1 for entry in entries if entry.get("status") == "success")
    failed = sum(1 for entry in entries if entry.get("status") == "failed")
    skipped = sum(1 for entry in entries if entry.get("status") == "skipped")
    pending = sum(1 for entry in entries if entry.get("status") in {"pending", "running"})

    total_qa = 0
    judged = 0
    total_score = 0.0
    for entry in entries:
        result_path = Path(str(entry.get("result_path", "")))
        payload = load_json(result_path)
        if not isinstance(payload, dict):
            continue
        qa_results = payload.get("qa_results")
        if isinstance(qa_results, list):
            total_qa += len(qa_results)
        judgement = payload.get("llm_judgement_summary")
        if isinstance(judgement, dict):
            judged += int(judgement.get("total", 0) or 0)
            total_score += float(judgement.get("total_score", 0.0) or 0.0)

    summary = {
        "run_id": manifest.get("run_id"),
        "updated_at": int(time.time()),
        "deck_count": len(entries),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "pending": pending,
        "total_qa_results": total_qa,
        "judged_qa": judged,
        "total_score": total_score,
        "average_score": total_score / judged if judged else None,
        "score_scale": "0-100",
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def mirror_pipe_to_file_and_terminal(
    pipe: Any,
    log_file: TextIO,
    terminal: TextIO,
    *,
    mirror_to_terminal: bool,
) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_file.write(line)
            log_file.flush()
            if mirror_to_terminal:
                terminal.write(line)
                terminal.flush()
    finally:
        pipe.close()


def run_deck_worker(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    deck_name: str,
    deck_index: int,
    result_path: Path,
) -> int:
    log_slug = f"{deck_index + 1:06d}_{sanitize_component(deck_name)}"
    stdout_path = run_dir / "logs" / f"{log_slug}.stdout.log"
    stderr_path = run_dir / "logs" / f"{log_slug}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    worker_script = SCRIPT_DIR / "run_first_sample_agent_eval.py"
    cmd = [
        sys.executable,
        str(worker_script),
        "--parquet",
        str(args.parquet),
        "--qa-groups",
        str(args.qa_groups),
        "--deck-name",
        deck_name,
        "--output-root",
        str(run_dir / "worker_runs"),
        "--cache-root",
        str(args.cache_root),
        "--result-json",
        str(result_path),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.no_judge:
        env["EVAL_JUDGE_ENABLED"] = "false"

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=mirror_pipe_to_file_and_terminal,
            kwargs={
                "pipe": process.stdout,
                "log_file": stdout_f,
                "terminal": sys.stdout,
                "mirror_to_terminal": not args.quiet_worker_logs,
            },
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=mirror_pipe_to_file_and_terminal,
            kwargs={
                "pipe": process.stderr,
                "log_file": stderr_f,
                "terminal": sys.stderr,
                "mirror_to_terminal": not args.quiet_worker_logs,
            },
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            return_code = process.wait()
            stdout_thread.join()
            stderr_thread.join()
            return return_code
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            raise


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")

    grouped_payload = ensure_qa_groups(args.parquet, args.qa_groups)
    decks = select_decks(args, get_decks(grouped_payload))
    if not decks:
        raise ValueError("No decks selected for evaluation.")

    run_dir = args.output_root / sanitize_component(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = load_or_create_manifest(args=args, decks=get_decks(grouped_payload), run_dir=run_dir)
    judge_enabled = (not args.no_judge) and get_env_bool("EVAL_JUDGE_ENABLED", True)

    print(
        f"[START] run_dir={run_dir} selected_decks={len(decks)} "
        f"resume_manifest={manifest_path}"
    )

    try:
        for deck_index, deck in enumerate(decks):
            deck_name = str(deck.get("deck_name", "")).strip()
            if not deck_name:
                continue

            expected_qa_count = int(deck.get("qa_count", len(deck.get("qa_items", []))))
            result_path = deck_result_path(run_dir, deck_name)
            entry = manifest["decks"][deck_name]

            if not args.force and result_is_complete(
                result_path=result_path,
                deck_name=deck_name,
                expected_qa_count=expected_qa_count,
                judge_enabled=judge_enabled,
            ):
                entry["status"] = "success"
                entry["result_path"] = str(result_path)
                entry["skipped_at"] = int(time.time())
                print(f"[SKIP] {deck_index + 1}/{len(decks)} {deck_name}")
                save_json(manifest_path, manifest)
                continue

            entry["status"] = "running"
            entry["started_at"] = int(time.time())
            entry["attempt_count"] = int(entry.get("attempt_count", 0) or 0) + 1
            entry["result_path"] = str(result_path)
            save_json(manifest_path, manifest)

            deck_started_at = time.perf_counter()
            print(
                f"[RUN] {deck_index + 1}/{len(decks)} {deck_name} "
                f"qa_count={expected_qa_count} attempt={entry['attempt_count']}"
            )
            return_code = run_deck_worker(
                args=args,
                run_dir=run_dir,
                deck_name=deck_name,
                deck_index=deck_index,
                result_path=result_path,
            )
            elapsed = time.perf_counter() - deck_started_at

            entry["return_code"] = return_code
            entry["finished_at"] = int(time.time())
            entry["elapsed_seconds"] = elapsed
            entry["elapsed_human"] = format_duration(elapsed)

            if return_code == 0 and result_is_complete(
                result_path=result_path,
                deck_name=deck_name,
                expected_qa_count=expected_qa_count,
                judge_enabled=judge_enabled,
            ):
                entry["status"] = "success"
                print(f"[DONE] {deck_name} in {format_duration(elapsed)}")
            else:
                entry["status"] = "failed"
                print(f"[FAIL] {deck_name} return_code={return_code} in {format_duration(elapsed)}")
                if args.stop_on_error:
                    save_json(manifest_path, manifest)
                    write_summary(run_dir, manifest)
                    return return_code or 1

            manifest["updated_at"] = int(time.time())
            save_json(manifest_path, manifest)
            write_summary(run_dir, manifest)
    except KeyboardInterrupt:
        manifest["updated_at"] = int(time.time())
        save_json(manifest_path, manifest)
        write_summary(run_dir, manifest)
        print("[STOP] Interrupted; manifest saved for resume.")
        return 130

    summary = write_summary(run_dir, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[FINISH] total:", format_duration(time.perf_counter() - started_at))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
