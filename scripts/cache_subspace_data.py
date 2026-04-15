"""Cache source residuals for SubspaceProbe training.

For every labeled task, finds the pivot token positions (first token of each
labeled intent block + the immediately preceding token) and caches the layer
residuals at those positions.

Outputs a single file:
    results/subspace_probe/<model_short>/cache.pt

which is a list[SubspaceCache] (torch.save / torch.load compatible).

Usage
-----
conda run -n goals python scripts/cache_subspace_data.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --oracle     claude-sonnet-4-6 \\
    --layer-range 35 46
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(".")

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.goals.scope_labeler import load_labeled_tree
from src.goals.subspace_probe import (
    SubspaceCache,
    cache_source_residuals,
    find_block_pivot_positions,
)
from src.tokens import prepare_input
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--oracle", default="claude-sonnet-4-6")
    p.add_argument(
        "--layer-range",
        type=int,
        nargs=2,
        default=[0, -1],
        metavar=("START", "STOP"),
        help="Layer range [start, stop) to cache (default: 0 -1, all layers).",
    )
    p.add_argument(
        "--source",
        default=None,
        metavar="SRC",
        help=(
            "Task source subdirectory to process (e.g. custom, humaneval, mbpp). "
            "Defaults to all source subdirs found under the labels directory."
        ),
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to include (default: all labeled tasks).",
    )
    p.add_argument("--overwrite", action="store_true")
    add_logging_args(p)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    configure(args)

    model_short = args.model_key.split("/")[-1]
    oracle_short = args.oracle.split("/")[-1]

    results_root = Path(DEFAULT_RESULTS_DIR)
    base_gen_dir = results_root / "well-known-tasks" / "generations" / model_short
    base_label_dir = results_root / "well-known-tasks" / "labels" / model_short / oracle_short
    out_dir = results_root / "subspace_probe" / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cache.pt"

    if out_path.exists() and not args.overwrite:
        logger.info(
            "Cache already exists at %s — pass --overwrite to regenerate.", out_path
        )
        return

    # Resolve source subdirs
    if args.source:
        source_names = [args.source]
    else:
        source_names = [
            d.name for d in sorted(base_label_dir.iterdir()) if d.is_dir()
        ] if base_label_dir.exists() else []

    # Enumerate label files across sources
    label_files = []
    for src_name in source_names:
        label_dir = base_label_dir / src_name
        gen_dir = base_gen_dir / src_name
        if not label_dir.exists():
            logger.warning("Label dir not found, skipping: %s", label_dir)
            continue
        files = sorted(label_dir.glob("*.json"))
        if args.tasks:
            files = [f for f in files if f.stem in args.tasks]
        logger.info("Source %-12s: %d label files", src_name, len(files))
        label_files.append((src_name, gen_dir, label_dir, files))

    total = sum(len(f) for _, _, _, f in label_files)
    logger.info("Processing %d task(s) total across %d source(s)", total, len(label_files))

    # Lazy model load (only after arg validation)

    from src.models import ModelandTokenizer

    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(
        "Model loaded: %s  (%d layers, d_model=%d)", model_short, mt.n_layer, mt.n_embd
    )
    start_layer, stop_layer = args.layer_range
    if stop_layer == -1:
        stop_layer = mt.n_layer
    layers = list(range(start_layer, stop_layer))
    logger.info("Caching layers: %s", layers)

    all_caches: list[SubspaceCache] = []
    skipped = []

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task_bar = progress.add_task("caching…", total=total)

        for src_name, gen_dir, label_dir, src_label_files in label_files:
            for label_path in src_label_files:
                task_name = label_path.stem
                gen_path = gen_dir / f"{task_name}.json"
                progress.update(task_bar, description=f"[bold]{src_name}/{task_name}[/bold]")

                if not gen_path.exists():
                    logger.warning("No generation for %s — skipping", task_name)
                    skipped.append(task_name)
                    progress.advance(task_bar)
                    continue

                try:
                    with open(gen_path) as f:
                        gen_data = json.load(f)

                    raw_gen = gen_data["generation"]
                    code_prefill = gen_data["prompt"].split("<code>")[-1]
                    code_gen = code_prefill + raw_gen
                    if "</code>" in code_gen:
                        code_gen = code_gen.split("</code>")[0]
                    source_text = gen_data["prompt"] + gen_data["generation"]

                    root_block, full_code, _ = load_labeled_tree(label_path)

                    tokenized = prepare_input(
                        source_text, return_offsets_mapping=True, tokenizer=mt
                    )
                    offset_map = tokenized["offset_mapping"][0]

                    pivots = find_block_pivot_positions(
                        block_tree=root_block,
                        source_prompt=source_text,
                        code_str=full_code,
                        offset_mapping=offset_map,
                    )

                    if not pivots:
                        logger.warning("%s: no labeled pivots found — skipping", task_name)
                        skipped.append(task_name)
                        progress.advance(task_bar)
                        continue

                    positions = sorted({p for _, ft, pt in pivots for p in [ft, pt]})
                    residuals = cache_source_residuals(mt, source_text, positions, layers)

                    for block, first_tok, prev_tok in pivots:
                        for tok_pos, is_prec in [(first_tok, False), (prev_tok, True)]:
                            all_caches.append(
                                SubspaceCache(
                                    task=task_name,
                                    block_kind=block.kind,
                                    label=block.label,
                                    src_residuals=residuals[tok_pos],
                                    token_pos=tok_pos,
                                    is_preceding=is_prec,
                                )
                            )

                    logger.info(
                        "%s/%s: %d blocks → %d cache entries",
                        src_name,
                        task_name,
                        len(pivots),
                        len(pivots) * 2,
                    )

                except Exception as exc:
                    logger.exception("Error processing %s: %s", task_name, exc)
                    skipped.append(task_name)

                progress.advance(task_bar)

    torch.save(all_caches, out_path)
    logger.info(
        "Saved %d cache entries → %s  (skipped: %s)",
        len(all_caches),
        out_path,
        skipped or "none",
    )


if __name__ == "__main__":
    main()
