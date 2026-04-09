"""Label intent blocks for well-known-task generations.

For each generation JSON under results/well-known-tasks/generations/<model>/,
identifies intent blocks with ``identify_intent_blocks`` and labels every block
using the specified oracle.  Results are saved as labeled-tree JSON files under
results/well-known-tasks/labels/<model>/<oracle>/<task>.json.

Oracle options
──────────────
claude-*   — calls ``ask_claude`` (disk-cached; no GPU required)
<HF key>   — loads a local ModelandTokenizer and uses batched LM generation

Typical usage
─────────────
# Claude oracle (default, no GPU needed)
conda run -n goals python scripts/label_well_known_task_blocks.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

# Cheaper Claude model, subset of tasks
conda run -n goals python scripts/label_well_known_task_blocks.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --oracle claude-haiku-4-5-20251001 \\
    --tasks bfs dfs dijkstra

# Local LM oracle (loads the model onto GPU)
conda run -n goals python scripts/label_well_known_task_blocks.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --oracle Qwen/Qwen2.5-Coder-7B-Instruct \\
    --batch-size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

sys.path.append(".")

from src.goals.scope_id import extract_code_from_tags, identify_intent_blocks
from src.goals.scope_labeler import label_all_blocks, save_labeled_tree
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)

WELL_KNOWN_DIR = Path(DEFAULT_RESULTS_DIR) / "well-known-tasks"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label intent blocks for well-known-task LM generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key whose generations to label.",
    )
    parser.add_argument(
        "--oracle",
        default="claude-sonnet-4-6",
        metavar="MODEL",
        help=(
            "Oracle for labeling. "
            "claude-* → ask_claude (disk-cached, no GPU). "
            "Any other value is treated as a HuggingFace model key and loaded locally."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5],
        help="Intent block granularity level (1=function … 5=sub-expression).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Subset of task names to label (e.g. bfs dfs bubble_sort). "
            "Defaults to all tasks found in the generations directory."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens for the oracle to generate per label.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Prompts per batch (LM oracle only; ignored for Claude).",
    )
    parser.add_argument(
        "--generations-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override input directory for generation JSONs.  "
            "Defaults to results/well-known-tasks/generations/<model-short-name>/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override output directory for labeled-tree JSONs.  "
            "Defaults to results/well-known-tasks/labels/<model-short-name>/<oracle-short-name>/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label tasks that already have an output file.",
    )
    add_logging_args(parser)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _is_claude_oracle(oracle: str) -> bool:
    return oracle.startswith("claude-")


def _load_generation_paths(
    gen_dir: Path,
    task_names: list[str] | None,
) -> list[Path]:
    """Return sorted generation JSON paths, optionally filtered by task name."""
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generations directory not found: {gen_dir}")

    paths = sorted(gen_dir.glob("*.json"))
    if task_names:
        task_set = set(task_names)
        paths = [p for p in paths if p.stem in task_set]
        missing = task_set - {p.stem for p in paths}
        if missing:
            logger.warning(
                "Tasks not found in %s: %s", gen_dir, ", ".join(sorted(missing))
            )

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure(args)

    model_short = args.model_key.split("/")[-1]
    oracle_short = args.oracle.split("/")[-1]   # "claude-sonnet-4-6" or "Qwen2.5-Coder-7B-Instruct"

    # ── Directories ───────────────────────────────────────────────────────────
    gen_dir = (
        Path(args.generations_dir)
        if args.generations_dir
        else WELL_KNOWN_DIR / "generations" / model_short
    )
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else WELL_KNOWN_DIR / "labels" / model_short / oracle_short
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generations dir : %s", gen_dir)
    logger.info("Output dir      : %s", out_dir)
    logger.info("Oracle          : %s", args.oracle)
    logger.info("Level           : L%d", args.level)

    # ── Load oracle (LM path only) ────────────────────────────────────────────
    mt = None
    if not _is_claude_oracle(args.oracle):
        import torch
        from src.models import ModelandTokenizer

        logger.info("Loading oracle LM: %s", args.oracle)
        mt = ModelandTokenizer(
            model_key=args.oracle,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        logger.info(
            "Oracle LM loaded: %s  (%d layers, d_model=%d)",
            oracle_short, mt.n_layer, mt.n_embd,
        )

    # ── Load generation paths ─────────────────────────────────────────────────
    gen_paths = _load_generation_paths(gen_dir, args.tasks)
    logger.info("Found %d generation files", len(gen_paths))

    if not args.overwrite:
        before = len(gen_paths)
        gen_paths = [p for p in gen_paths if not (out_dir / p.name).exists()]
        skipped = before - len(gen_paths)
        if skipped:
            logger.info(
                "Skipping %d already-labeled tasks (use --overwrite to redo)", skipped
            )

    if not gen_paths:
        console.print("[green]All tasks already labeled. Done.[/green]")
        return

    # ── Progress bar ──────────────────────────────────────────────────────────
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    label_task = progress.add_task("labeling", total=len(gen_paths))

    saved = 0
    errors: list[str] = []

    with progress:
        for gen_path in gen_paths:
            task_name = gen_path.stem
            logger.info("Processing: %s", task_name)

            # ── Load generation ───────────────────────────────────────────────
            with open(gen_path) as f:
                gen_data = json.load(f)

            raw_gen = gen_data.get("generation", "")
            code = extract_code_from_tags(raw_gen) or raw_gen

            # ── Identify intent blocks ────────────────────────────────────────
            try:
                root = identify_intent_blocks(code, level=args.level)
            except SyntaxError as exc:
                logger.warning("SyntaxError in %s — skipping: %s", task_name, exc)
                errors.append(task_name)
                progress.advance(label_task)
                continue

            # ── Label ─────────────────────────────────────────────────────────
            # mt=None  → ask_claude (disk-cached)
            # mt=<lm>  → batched local LM generation
            label_all_blocks(
                root,
                code,
                mt=mt,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens,
                claude_model=args.oracle,
                show_progress=False,
            )

            # ── Save ─────────────────────────────────────────────────────────
            out_path = out_dir / gen_path.name
            save_labeled_tree(
                root,
                out_path,
                full_code=code,
                metadata={
                    "function_name": gen_data.get("function_name", task_name),
                    "model": model_short,
                    "oracle": args.oracle,
                    "level": args.level,
                },
            )
            logger.info("Saved %s", out_path.name)
            saved += 1
            progress.advance(label_task)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(
        f"[green]Done.[/green] Labeled {saved} tasks → [bold]{out_dir}[/bold]"
    )
    if errors:
        console.print(
            f"[red]Skipped {len(errors)} tasks (SyntaxError):[/red] "
            + ", ".join(errors)
        )


if __name__ == "__main__":
    main()
