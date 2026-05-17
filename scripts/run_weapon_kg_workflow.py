from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime
import json
import logging
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    cache_dir = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    os.environ["TIKTOKEN_CACHE_DIR"] = str(
        cache_dir if cache_dir.is_absolute() else (PROJECT_ROOT / cache_dir).resolve()
    )

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig
from scripts.inspect_weapon_kg import inspect as inspect_weapon_kg
from scripts.predict_missing_edges import run as run_edge_prediction


class WorkflowUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_rich = False
        self.console = None
        if not args.plain:
            try:
                from rich.console import Console

                # Internal stages redirect sys.stdout/sys.stderr to log files.
                # Bind Rich to the real terminal so spinner/timer refreshes keep
                # working while those global streams are redirected in a worker.
                self.console = Console(file=sys.__stdout__)
                self.use_rich = True
            except Exception:
                self.console = None

    def print(self, message: str = "") -> None:
        if self.args.quiet:
            return
        if self.use_rich:
            self.console.print(message)
        else:
            print(message)

    def logo(self) -> None:
        if self.args.quiet:
            return
        title = "WorkFlowDemo"
        subtitle = "Parse -> Knowledge Graph -> Edge Predict"
        if self.use_rich:
            from rich.panel import Panel
            from rich.text import Text

            text = Text()
            text.append(title + "\n", style="bold cyan")
            text.append(subtitle, style="green")
            self.console.print(Panel(text, border_style="cyan"))
        else:
            print("╔════════════════════════════════════════════╗")
            print("║               WorkFlowDemo                 ║")
            print("║   Parse -> Knowledge Graph -> Edge Predict ║")
            print("╚════════════════════════════════════════════╝")

    def config_summary(self) -> None:
        if self.args.quiet:
            return
        rows = [
            ("Input file", str(self.args.input_file) if self.args.input_file else "(skipped)"),
            ("Working dir", str(self.args.working_dir)),
            ("Parser", f"{self.args.parser} / {self.args.parse_method}"),
            ("Extraction", "weapon_equipment"),
            (
                "Inner logs",
                "console"
                if self.args.debug or self.args.show_inner_logs
                else str(self.args.inner_log_path),
            ),
            ("KG HTML", "disabled" if self.args.skip_kg_html else "enabled"),
            (
                "Edge prediction",
                "disabled"
                if self.args.skip_edge_prediction
                else f"enabled, top {self.args.predict_top_k}, min_score {self.args.min_score}",
            ),
            (
                "LLM explain",
                "disabled"
                if not self.args.llm_explain
                else f"enabled, top {self.args.explain_top_n or self.args.predict_top_k}",
            ),
        ]
        if self.use_rich:
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Key", style="bold")
            table.add_column("Value")
            for key, value in rows:
                table.add_row(key, value)
            self.console.print(table)
            self.console.print()
        else:
            for key, value in rows:
                print(f"{key:16}: {value}")
            print()

    def skip(self, index: int, total: int, label: str) -> None:
        self.print(f"- [{index}/{total}] {label} skipped")

    @staticmethod
    def _run_coroutine_in_thread(func):
        return asyncio.run(func())

    async def run_async_stage(self, index: int, total: int, label: str, func):
        async def run_stage_work():
            # RAGAnything/MinerU have synchronous sections inside async entrypoints.
            # Running the stage in a worker thread keeps Rich's spinner/timer alive.
            return await asyncio.to_thread(self._run_coroutine_in_thread, func)

        if self.use_rich and not self.args.quiet:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                progress.add_task(f"[{index}/{total}] {label}", total=None)
                result = await run_stage_work()
            self.console.print(f"[green]✓[/green] [{index}/{total}] {label} done")
            return result

        if not self.args.quiet:
            print(f"→ [{index}/{total}] {label} running")
        result = await run_stage_work()
        if not self.args.quiet:
            print(f"✓ [{index}/{total}] {label} done")
        return result

    async def run_sync_stage(self, index: int, total: int, label: str, func):
        async def wrapper():
            return func()

        return await self.run_async_stage(index, total, label, wrapper)

    def final_summary(self, storage_dir: Path, kg_html_path: Path | None) -> None:
        if self.args.quiet:
            return
        edge_json = storage_dir / f"{self.args.edge_output_prefix}.json"
        edge_html = storage_dir / f"{self.args.edge_output_prefix}.html"
        candidate_count = None
        if edge_json.exists():
            try:
                candidate_count = len(json.loads(edge_json.read_text(encoding="utf-8")))
            except Exception:
                candidate_count = None

        if self.use_rich:
            from rich.panel import Panel

            lines = ["[bold green]Workflow completed[/bold green]"]
            if kg_html_path:
                lines.append(f"Knowledge graph HTML: {kg_html_path}")
            if not self.args.skip_edge_prediction:
                lines.append(f"Edge prediction JSON: {edge_json}")
                lines.append(f"Edge prediction HTML: {edge_html}")
                if candidate_count is not None:
                    lines.append(f"Predicted candidates: {candidate_count}")
            if not self.args.debug and not self.args.show_inner_logs:
                lines.append(f"Inner logs: {self.args.inner_log_path}")
            self.console.print(Panel("\n".join(lines), border_style="green"))
        else:
            print("\nWorkflow completed")
            if kg_html_path:
                print(f"Knowledge graph HTML: {kg_html_path}")
            if not self.args.skip_edge_prediction:
                print(f"Edge prediction JSON: {edge_json}")
                print(f"Edge prediction HTML: {edge_html}")
                if candidate_count is not None:
                    print(f"Predicted candidates: {candidate_count}")
            if not self.args.debug and not self.args.show_inner_logs:
                print(f"Inner logs: {self.args.inner_log_path}")


def iter_loggers():
    yield logging.getLogger()
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            yield logger


@contextlib.contextmanager
def capture_inner_output(args: argparse.Namespace, stage_name: str):
    if args.debug or args.show_inner_logs:
        yield
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.inner_log_path
    saved_handler_levels = []
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n===== {stage_name} started {datetime.now().isoformat(timespec='seconds')} =====\n"
        )
        log_file.flush()
        try:
            for logger in iter_loggers():
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(
                        handler, logging.FileHandler
                    ):
                        saved_handler_levels.append((handler, handler.level))
                        handler.setLevel(logging.CRITICAL + 1)
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                yield
        finally:
            for handler, level in saved_handler_levels:
                try:
                    handler.setLevel(level)
                except Exception:
                    pass
            root_logger.removeHandler(file_handler)
            file_handler.close()
            log_file.write(
                f"===== {stage_name} ended {datetime.now().isoformat(timespec='seconds')} =====\n"
            )
            log_file.flush()


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


def get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def build_rag(args: argparse.Namespace) -> RAGAnything:
    api_key = args.api_key or get_env_str("OPENAI_API_KEY")
    base_url = args.base_url or get_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
    text_model = args.text_model or get_env_str("TEXT_LLM_MODEL", "gpt-4o-mini")
    vision_model = args.vision_model or get_env_str(
        "VLM_MODEL", get_env_str("VISION_LLM_MODEL", "gpt-4o")
    )
    embedding_model = args.embedding_model or get_env_str(
        "EMBEDDING_MODEL", "text-embedding-3-large"
    )
    embedding_dim = args.embedding_dim or get_env_int("EMBEDDING_DIM", 3072)
    embedding_max_token_size = args.embedding_max_token_size or get_env_int(
        "EMBEDDING_MAX_TOKEN_SIZE", 8192
    )

    config = RAGAnythingConfig(
        working_dir=str(args.working_dir),
        parser=args.parser,
        parse_method=args.parse_method,
        extraction_profile="weapon_equipment",
        enable_image_processing=not args.disable_image_processing,
        enable_table_processing=not args.disable_table_processing,
        enable_equation_processing=not args.disable_equation_processing,
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            text_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
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
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            }
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    user_message,
                ],
                api_key=api_key,
                base_url=base_url,
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
            api_key=api_key,
            base_url=base_url,
            embedding_dim=embedding_dim,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def run_workflow(args: argparse.Namespace) -> None:
    ui = WorkflowUI(args)
    ui.logo()
    ui.config_summary()
    args.working_dir.mkdir(parents=True, exist_ok=True)
    if not args.debug and not args.show_inner_logs:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        args.inner_log_path.write_text(
            f"SlideRAG workflow log started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    total_stages = 3

    if args.skip_insert:
        ui.skip(1, total_stages, "Parse document and extract KG")
    else:
        if not args.input_file:
            raise ValueError("--input-file is required unless --skip-insert is set")

        async def parse_document():
            with capture_inner_output(args, "Parse document and extract KG"):
                rag = build_rag(args)
                await rag.process_document_complete(
                    file_path=str(args.input_file),
                    output_dir=str(args.output_dir),
                    parse_method=args.parse_method,
                )
                await rag.finalize_storages()

        await ui.run_async_stage(1, total_stages, "Parse document and extract KG", parse_document)

    kg_html_path = None
    if args.skip_kg_html:
        ui.skip(2, total_stages, "Render knowledge graph HTML")
    else:
        def render_kg_html():
            with capture_inner_output(args, "Render knowledge graph HTML"):
                return inspect_weapon_kg(
                    working_dir=args.working_dir,
                    workspace=args.workspace,
                    limit=args.inspect_limit,
                )

        kg_html_path = await ui.run_sync_stage(
            2, total_stages, "Render knowledge graph HTML", render_kg_html
        )

    if args.skip_edge_prediction:
        ui.skip(3, total_stages, "Predict missing edges")
    else:
        edge_args = argparse.Namespace(
            working_dir=args.working_dir,
            workspace=args.workspace,
            top_k=args.predict_top_k,
            min_score=args.min_score,
            same_type_only=args.same_type_only,
            cross_type_only=args.cross_type_only,
            type_pairs=args.type_pairs,
            llm_explain=args.llm_explain,
            explain_top_n=args.explain_top_n,
            output_prefix=args.edge_output_prefix,
            embedding_batch_size=args.embedding_batch_size,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            llm_model=args.text_model,
            api_key=args.api_key,
            base_url=args.base_url,
        )

        async def predict_edges():
            with capture_inner_output(args, "Predict missing edges"):
                await run_edge_prediction(edge_args)

        await ui.run_async_stage(3, total_stages, "Predict missing edges", predict_edges)

    storage_dir = args.working_dir / args.workspace if args.workspace else args.working_dir
    ui.final_summary(storage_dir, kg_html_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full weapon-equipment KG workflow: parse document, visualize KG, predict missing edges."
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--working-dir", required=True, type=Path)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--parser", default=get_env_str("PARSER", "mineru"))
    parser.add_argument("--parse-method", default=get_env_str("PARSE_METHOD", "auto"))
    parser.add_argument("--skip-insert", action="store_true")
    parser.add_argument("--skip-kg-html", action="store_true")
    parser.add_argument("--skip-edge-prediction", action="store_true")
    parser.add_argument("--inspect-limit", type=int, default=30)

    parser.add_argument("--predict-top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--same-type-only", action="store_true")
    parser.add_argument("--cross-type-only", action="store_true")
    parser.add_argument(
        "--type-pairs",
        default=None,
        help="Comma-separated type pairs, e.g. ScientificPrinciple:ApplicationTechnology,ApplicationTechnology:SystemEffect",
    )
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--explain-top-n", type=int, default=0)
    parser.add_argument("--edge-output-prefix", default="edge_predictions")
    parser.add_argument("--embedding-batch-size", type=int, default=16)

    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--embedding-max-token-size", type=int, default=0)
    parser.add_argument("--disable-image-processing", action="store_true")
    parser.add_argument("--disable-table-processing", action="store_true")
    parser.add_argument("--disable-equation-processing", action="store_true")
    parser.add_argument("--plain", action="store_true", help="Disable rich UI output.")
    parser.add_argument("--quiet", action="store_true", help="Only print errors.")
    parser.add_argument("--debug", action="store_true", help="Show detailed subprocess output and tracebacks.")
    parser.add_argument("--show-inner-logs", action="store_true", help="Print RAGAnything/MinerU/LightRAG logs directly to the terminal.")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"), help="Directory for captured workflow logs.")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.inner_log_path = args.log_dir / f"workflow_{timestamp}.log"

    try:
        if args.same_type_only and args.cross_type_only:
            raise ValueError("--same-type-only and --cross-type-only cannot both be set")
        asyncio.run(run_workflow(args))
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Workflow failed: {exc}", file=sys.stderr)
            if not args.show_inner_logs:
                print(f"Inner logs: {args.inner_log_path}", file=sys.stderr)
            print("Run again with --debug for a full traceback.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
