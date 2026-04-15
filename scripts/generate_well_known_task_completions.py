"""Generate code completions for all well-known-tasks using a code-LM.

For each IncompleteCodeSample in data/well-known-tasks/, builds a chat-template
prompt and runs batched greedy generation.  Results are saved as JSON files
under results/well-known-tasks/generations/<model-short-name>/<task-name>.json.

Each output file is a JSON dict with keys:
    function_name  : str
    prompt         : str    (full prompt sent to the model)
    generation     : str    (raw LM completion, decoded)

Typical usage
─────────────
conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct --batch-size 8 --tasks bfs dfs dijkstra
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
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

from src.goals.clock_probe import batch_generate
from src.goals.data import IncompleteCodeSample
from src.goals.functional import apply_chat_template, supports_chat_template
from src.models import ModelandTokenizer
from src.utils.env_utils import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)

TASKS_DIR = Path(DEFAULT_DATA_DIR) / "well-known-tasks"

# Known source subdirectories (order determines processing order)
_ALL_SOURCES = ["custom", "humaneval", "mbpp"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate well-known-task code completions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Subset of task names to run (e.g. bfs dfs bubble_sort). "
            "Defaults to all tasks in data/well-known-tasks/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of prompts per batched generate call.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per generation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Root output directory. "
            "Defaults to results/well-known-tasks/generations/<model-short-name>/."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="SRC",
        help=(
            "Task source subdirectory to process (e.g. custom, humaneval, mbpp). "
            "Defaults to all sources found under data/well-known-tasks/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate tasks that already have an output file.",
    )
    add_logging_args(parser)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_samples(
    source: str | None,
    task_names: list[str] | None,
) -> list[tuple[str, IncompleteCodeSample]]:
    """Load (source_name, IncompleteCodeSample) pairs from data/well-known-tasks/<source>/."""
    if not TASKS_DIR.exists():
        raise FileNotFoundError(f"Tasks directory not found: {TASKS_DIR}")

    if source:
        source_dirs = [TASKS_DIR / source]
    else:
        source_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())

    results = []
    for src_dir in source_dirs:
        src_name = src_dir.name
        paths = sorted(src_dir.glob("*.json"))
        if task_names:
            task_set = set(task_names)
            paths = [p for p in paths if p.stem in task_set]
            missing = task_set - {p.stem for p in paths}
            if missing:
                logger.warning(
                    "Tasks not found in %s: %s", src_dir, ", ".join(sorted(missing))
                )
        for path in paths:
            with open(path) as f:
                d = json.load(f)
            sample = IncompleteCodeSample.from_dict(d)
            sample.without_comments = True
            results.append((src_name, sample))

    return results


def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure(args)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_key)
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model_short = args.model_key.split("/")[-1]
    logger.info(
        "Model loaded: %s  (%d layers, d_model=%d)", model_short, mt.n_layer, mt.n_embd
    )

    # ── Base output directory (override or default) ───────────────────────────
    base_out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(DEFAULT_RESULTS_DIR) / "well-known-tasks" / "generations" / model_short
    )

    # ── Load samples (list of (source_name, IncompleteCodeSample)) ───────────
    source_samples = _load_samples(args.source, args.tasks)
    logger.info("Loaded %d tasks from %s", len(source_samples), TASKS_DIR)

    if not args.overwrite:
        before = len(source_samples)
        source_samples = [
            (src, s)
            for src, s in source_samples
            if not (base_out_dir / src / f"{s.function_name}.json").exists()
        ]
        skipped = before - len(source_samples)
        if skipped:
            logger.info(
                "Skipping %d already-generated tasks (use --overwrite to redo)", skipped
            )

    if not source_samples:
        console.print("[green]All tasks already generated. Done.[/green]")
        return

    # ── Build prompts ─────────────────────────────────────────────────────────
    logger.info("Building prompts...")
    prompts: list[str] = []
    for _, sample in source_samples:
        if supports_chat_template(mt):
            prompt = apply_chat_template(
                mt,
                user_query=sample.full_instruction.strip(),
                incomplete_code=sample.code,
            )
        else:
            prompt = sample.code
        prompts.append(prompt)

    # ── Batched generation ────────────────────────────────────────────────────
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
    gen_task = progress.add_task("generating", total=len(source_samples))

    results: list[tuple[list[int], list[int]]] = []
    with progress:
        for chunk in _chunk(list(zip(source_samples, prompts)), args.batch_size):
            chunk_src_samples, chunk_prompts = zip(*chunk)
            chunk_samples = [s for _, s in chunk_src_samples]
            logger.info(f"current batch: {[s.function_name for s in chunk_samples]}")
            chunk_results = batch_generate(mt, list(chunk_prompts), args.max_new_tokens)
            results.extend(chunk_results)
            progress.advance(gen_task, len(chunk))

    # ── Decode and save ───────────────────────────────────────────────────────
    saved = 0
    for (src_name, sample), prompt, (_, gen_ids) in zip(source_samples, prompts, results):
        out_dir = base_out_dir / src_name
        out_dir.mkdir(parents=True, exist_ok=True)
        generation = mt.tokenizer.decode(gen_ids, skip_special_tokens=True)
        out = {
            "function_name": sample.function_name,
            "prompt": prompt,
            "generation": generation,
            "source": src_name,
        }
        out_path = out_dir / f"{sample.function_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Saved %s/%s  (gen_len=%d tokens)", src_name, out_path.name, len(gen_ids))
        saved += 1

    console.print(
        f"[green]Done.[/green] Saved {saved} generations under [bold]{base_out_dir}[/bold]"
    )


if __name__ == "__main__":
    main()
