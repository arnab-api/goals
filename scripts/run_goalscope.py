"""Generate GoalScope HTML visualizations for well-known-task generations.

For each generation JSON under results/well-known-tasks/generations/<model>/,
runs the Head-Output Steering patchscope: at every token position inside the
generated code, patches the summed o_proj contribution of the trained
selected heads into a destination prompt and records the model's verbalization.
Results are saved as interactive HTML files under
results/well-known-tasks/goalscope/<model>/<task>.html.

The steering adds the per-position head-output vector to the residual stream
at position -1 (the last token) of the destination prompt across a range of
layers, using additive patching via baukit.

Typical usage
─────────────
conda run -n goals python scripts/run_goalscope.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

conda run -n goals python scripts/run_goalscope.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --tasks bfs dijkstra merge_sort
"""

import argparse
import json
import logging
import math
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

from src.functional import PatchSpec, generate_with_patch
from src.goals.goalscope import render_patchscope_html
from src.goals.head_loc_attn import HeadMask, collect_head_steering_vecs
from src.models import ModelandTokenizer
from src.tokens import TokenizerOutput, find_token_range, prepare_input
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)

WELL_KNOWN_DIR = Path(DEFAULT_RESULTS_DIR) / "well-known-tasks"
LOCALIZATION_DIR = Path(DEFAULT_RESULTS_DIR) / "localization"

# ─────────────────────────────────────────────────────────────────────────────
# Destination prompt
# ─────────────────────────────────────────────────────────────────────────────

_DEST_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant proficient in programming.",
    },
    {
        "role": "user",
        "content": "What are you doing at this line of the code. Keep your answer as brief as possible.",
    },
    {
        "role": "assistant",
        "content": "",
    },
]


def _build_dest_prompt(mt: ModelandTokenizer) -> str:
    """Build the destination prompt using the model's chat template."""
    try:
        return mt.tokenizer.apply_chat_template(
            _DEST_MESSAGES,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(
            _DEST_MESSAGES,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GoalScope (Head-Output Steering patchscope) for well-known-task generations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Mask value threshold for selecting active heads.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Steering strength multiplier applied to head-output vectors.",
    )
    parser.add_argument(
        "--steer-layer-range",
        type=int,
        nargs=2,
        default=[35, 45],
        metavar=("START", "END"),
        help="Range of layers [START, END) to inject the steering vector into.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Source token positions per patched generation batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per patchscope position.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subset of task names to run. Defaults to all tasks.",
    )
    parser.add_argument(
        "--generations-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override input directory for generation JSONs. "
            "Defaults to results/well-known-tasks/generations/<model-short-name>/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override output directory for HTML files. "
            "Defaults to results/well-known-tasks/goalscope/<model-short-name>/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run tasks that already have an output HTML file.",
    )
    add_logging_args(parser)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_mask(model_short: str, n_heads: int) -> tuple[HeadMask, list[int]]:
    """Load the trained HeadMask and layer list from disk.

    Returns:
        (mask, layers) where layers is the list of tested layer indices.
    """
    mask_path = LOCALIZATION_DIR / model_short / "head_mask.pt"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Head mask not found: {mask_path}\nRun scripts/train_head_mask.py first."
        )
    ckpt = torch.load(mask_path, map_location="cpu")
    layers: list[int] = ckpt["layers"]
    n_layers_tested = len(layers)
    mask = HeadMask(n_layers_tested, n_heads)
    with torch.no_grad():
        mask.values.copy_(ckpt["mask_values"])
    return mask, layers


def _build_source_prompt(gen_data: dict) -> str:
    """Reconstruct the full source text (prompt + generation), trimmed at </code>."""
    full = gen_data["prompt"] + gen_data["generation"]
    return full


def _find_code_segment(
    source_prompt: str,
    src_ids: list[int],
    mt: ModelandTokenizer,
    src_offsets,
) -> tuple[int, int]:
    """Return (code_start, src_len) — token index range covering the code body.

    code_start is the token immediately after the last ``<code>`` tag.
    """
    _, code_start = find_token_range(
        string=source_prompt,
        substring="<code>",
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=src_offsets,
    )
    return code_start, len(src_ids)


def _load_generation_paths(
    gen_dir: Path,
    task_names: list[str] | None,
) -> list[Path]:
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

    # ── Directories ───────────────────────────────────────────────────────────
    gen_dir = (
        Path(args.generations_dir)
        if args.generations_dir
        else WELL_KNOWN_DIR / "generations" / model_short
    )
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else WELL_KNOWN_DIR / "goalscope" / model_short
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generations dir : %s", gen_dir)
    logger.info("Output dir      : %s", out_dir)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_key)
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(
        "Model loaded: %s  (%d layers, d_model=%d, n_heads=%d)",
        model_short,
        mt.n_layer,
        mt.n_embd,
        mt._model.config.num_attention_heads,
    )

    # ── Head mask ─────────────────────────────────────────────────────────────
    n_heads = mt._model.config.num_attention_heads
    mask, layers = _load_mask(model_short, n_heads)
    selected_heads = mask.selected_heads(layers, threshold=args.threshold)
    logger.info(
        "Mask loaded: %d layers tested, %d heads selected (threshold=%.2f)",
        len(layers),
        len(selected_heads),
        args.threshold,
    )

    # ── Destination prompt ────────────────────────────────────────────────────
    dest_prompt = _build_dest_prompt(mt)
    dest_tokenized = prepare_input(prompts=dest_prompt, tokenizer=mt)
    logger.info("Destination prompt: %d tokens", dest_tokenized["input_ids"].shape[1])

    steer_layers = list(range(args.steer_layer_range[0], args.steer_layer_range[1]))
    logger.info(
        "Steer layers: %d–%d  |  alpha=%.2f  |  batch=%d  |  max_new_tokens=%d",
        steer_layers[0],
        steer_layers[-1],
        args.alpha,
        args.batch_size,
        args.max_new_tokens,
    )

    # ── Generation paths ──────────────────────────────────────────────────────
    gen_paths = _load_generation_paths(gen_dir, args.tasks)
    logger.info("Found %d generation files", len(gen_paths))

    if not args.overwrite:
        before = len(gen_paths)
        gen_paths = [p for p in gen_paths if not (out_dir / f"{p.stem}.html").exists()]
        skipped = before - len(gen_paths)
        if skipped:
            logger.info(
                "Skipping %d already-processed tasks (use --overwrite to redo)", skipped
            )

    if not gen_paths:
        console.print("[green]All tasks already processed. Done.[/green]")
        return

    # ── Per-task progress ─────────────────────────────────────────────────────
    task_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    outer_task = task_progress.add_task("tasks", total=len(gen_paths))

    saved = 0
    errors: list[str] = []

    with task_progress:
        for gen_path in gen_paths:
            task_name = gen_path.stem
            logger.info("Processing: %s", task_name)

            # ── Load + reconstruct source prompt ──────────────────────────────
            with open(gen_path) as f:
                gen_data = json.load(f)

            source_prompt = _build_source_prompt(gen_data)

            # ── Tokenize source ───────────────────────────────────────────────
            src_tokenized = prepare_input(prompts=source_prompt, tokenizer=mt)
            src_ids = src_tokenized["input_ids"][0].tolist()
            src_offsets = src_tokenized.get("offset_mapping")
            if src_offsets is None:
                src_tokenized_with_offsets = mt.tokenizer(
                    source_prompt,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                src_offsets = src_tokenized_with_offsets["offset_mapping"][0].tolist()

            # ── Find code segment ─────────────────────────────────────────────
            try:
                code_start, src_len = _find_code_segment(
                    source_prompt, src_ids, mt, src_offsets
                )
            except Exception as exc:
                logger.warning(
                    "Could not find <code> in %s — using full source: %s",
                    task_name,
                    exc,
                )
                code_start, src_len = 0, len(src_ids)

            seg_positions = list(range(code_start, src_len))
            src_text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]

            logger.info(
                "%s: %d source tokens, code segment [%d:%d] (%d positions)",
                task_name,
                src_len,
                code_start,
                src_len,
                len(seg_positions),
            )

            # ── Collect steering vectors ──────────────────────────────────────
            steering_vecs = collect_head_steering_vecs(
                mt=mt,
                src_tokenized=src_tokenized,
                selected_heads=selected_heads,
                token_positions=seg_positions,
            )  # (n_seg, n_embd)

            # ── Batched patched generation ────────────────────────────────────
            dest_ids = dest_tokenized["input_ids"]
            dest_mask = dest_tokenized["attention_mask"]
            n_seg = len(seg_positions)
            results: list[dict] = []

            for chunk_start in range(0, n_seg, args.batch_size):
                chunk_end = min(chunk_start + args.batch_size, n_seg)
                B = chunk_end - chunk_start

                sv_batch = (
                    args.alpha * steering_vecs[chunk_start:chunk_end]
                )  # (B, n_embd)

                tiled_inputs = TokenizerOutput(
                    data=dict(
                        input_ids=dest_ids.expand(B, -1).clone(),
                        attention_mask=dest_mask.expand(B, -1).clone(),
                    )
                )

                patches = [
                    PatchSpec(
                        location=(mt.layer_name_format.format(li), -1),
                        patch=sv_batch,
                        strategy="add",
                        on_pass="all",
                    )
                    for li in steer_layers
                ]

                gens = generate_with_patch(
                    mt,
                    tiled_inputs,
                    max_new_tokens=args.max_new_tokens,
                    patches=patches,
                    do_sample=False,
                    patch_interface="baukit",
                    remove_prefix=True,
                    skip_special_tokens=True,
                )  # list[str] of length B

                for i in range(B):
                    src_pos = seg_positions[chunk_start + i]
                    results.append(
                        {
                            "pos": src_pos,
                            "token": src_text_tokens[src_pos],
                            "patched_gen": gens[i],
                        }
                    )

            # ── Render and save HTML ──────────────────────────────────────────
            html_str = render_patchscope_html(
                results,
                title=f"GoalScope — {task_name} ({model_short})",
            )
            out_path = out_dir / f"{task_name}.html"
            out_path.write_text(html_str, encoding="utf-8")
            logger.info(
                "Saved %s  (%d positions, %d batches)",
                out_path.name,
                n_seg,
                math.ceil(n_seg / args.batch_size),
            )
            saved += 1
            task_progress.advance(outer_task)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(
        f"[green]Done.[/green] Saved {saved} HTML files → [bold]{out_dir}[/bold]"
    )
    if errors:
        console.print(f"[red]Skipped {len(errors)} tasks:[/red] " + ", ".join(errors))


if __name__ == "__main__":
    main()
