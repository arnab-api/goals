"""Generate code completions for all well-known-tasks using a code-LM.

For each IncompleteCodeSample in data/well-known-tasks/, builds a chat-template
prompt and runs batched greedy generation.  Results are saved as JSON files
under results/well-known-tasks/generations/<model-short-name>/<task-name>.json.

In addition to the greedy completion, the script labels each generation
with a proactive-strategy classifier (recursive vs iterative) and records
the token positions where recursive calls happen — used downstream to mine
*proactively recursive* generations (see ``src/goals/proactive_strategy.py``).

Optional ``--n-samples K`` runs K additional sampled completions per task
(temperature/top_p hardcoded) so post-hoc analysis can measure the
empirical rate at which the model chooses each strategy under sampling.

Each output file is a JSON dict with keys:
    function_name  : str
    source         : str    (custom / humaneval / mbpp)
    prompt         : str    (full prompt sent to the model)
    generation     : str    (greedy LM completion, decoded — backward compat)
    prompt_bias    : str    (neutral / recursive / iterative, derived from instruction)
    prompt_token_ids        : list[int]
    generation_token_ids    : list[int]   (greedy gen tokens, used for token-position tagging)
    greedy : {
        extracted_code            : str
        used_code_tag_fallback    : bool
        strategy                  : recursive / iterative / Unsure
        has_recursion             : bool
        has_loop                  : bool
        used_ast_fallback         : bool   (regex fallback was used because AST failed)
        recursive_calls : [
            {name: str, name_char_span: [s, e], name_token_positions: [t, ...]}
        ]
    }
    samples : [   (only when --n-samples > 0)
        {text: str, has_recursion: bool, has_loop: bool, strategy: str}
    ]

Typical usage
─────────────
conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct --batch-size 8 --tasks bfs dfs dijkstra

conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct --n-samples 8
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
from src.goals.proactive_strategy import (
    attach_token_positions,
    code_offset_in_generation,
    detect_prompt_bias,
    extract_code_with_fallback,
    label_python_strategy,
)
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.env_utils import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

# Sampling hyperparameters for --n-samples (hardcoded; tuned for code-LM defaults).
_SAMPLE_TEMPERATURE = 0.7
_SAMPLE_TOP_P = 0.9

# Schema version keys — presence of these in an existing JSON indicates the
# new richer schema.  Missing → forced regen even without --overwrite.
_NEW_SCHEMA_KEYS = ("prompt_bias", "generation_token_ids", "greedy")

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
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        metavar="K",
        help=(
            "If >0, run K additional sampled completions per task in addition "
            f"to the greedy one (temperature={_SAMPLE_TEMPERATURE}, top_p={_SAMPLE_TOP_P}). "
            "Sampled completions are stored as text only (+ has_recursion / "
            "has_loop flags) for post-hoc strategy-rate analysis."
        ),
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


def _existing_has_new_schema(path: Path) -> bool:
    """True iff *path* already exists and contains all _NEW_SCHEMA_KEYS.

    Used to skip tasks whose on-disk JSON is already up to date.  Stale
    files written by an older version of this script (missing
    ``prompt_bias`` / token IDs / ``greedy`` block) are *not* considered
    up-to-date and will be re-generated even without ``--overwrite``.
    """
    if not path.exists():
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return all(k in d for k in _NEW_SCHEMA_KEYS)


def _batch_generate_sampled(
    mt: ModelandTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    n_samples: int,
) -> list[list[list[int]]]:
    """Run a batched sampling pass returning ``[n_prompts][n_samples]`` gen_ids.

    Uses ``num_return_sequences=n_samples`` so the prompt encoder runs once
    per batch (the K samples share the prompt-side compute).
    """
    if n_samples <= 0 or not prompts:
        return [[] for _ in prompts]

    batch_inputs = prepare_input(prompts, tokenizer=mt)
    padded_input_len: int = batch_inputs["input_ids"].shape[1]
    input_ids = batch_inputs["input_ids"].to(mt.device)
    attention_mask = batch_inputs["attention_mask"].to(mt.device)
    pad_token_id = (
        mt.tokenizer.pad_token_id
        if mt.tokenizer.pad_token_id is not None
        else mt.tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = mt._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=_SAMPLE_TEMPERATURE,
            top_p=_SAMPLE_TOP_P,
            num_return_sequences=n_samples,
            pad_token_id=pad_token_id,
        )

    # HF returns (batch * n_samples, seq); samples for input i are at i*K..(i+1)*K.
    per_prompt: list[list[list[int]]] = []
    for i in range(len(prompts)):
        per_prompt.append(
            [outputs[i * n_samples + k][padded_input_len:].tolist() for k in range(n_samples)]
        )
    return per_prompt


def _build_output_record(
    mt: ModelandTokenizer,
    sample: IncompleteCodeSample,
    src_name: str,
    prompt: str,
    prompt_ids: list[int],
    gen_ids: list[int],
    sampled_gen_ids: list[list[int]],
) -> dict:
    """Assemble the per-task JSON record: greedy decode + strategy label +
    token-position tagging + per-sample strategy flags.
    """
    generation = mt.tokenizer.decode(gen_ids, skip_special_tokens=True)
    prompt_bias = detect_prompt_bias(sample.instruction)

    extracted_code, used_tag_fallback = extract_code_with_fallback(generation)
    label = label_python_strategy(extracted_code, sample.function_name)
    offset = code_offset_in_generation(generation, extracted_code)
    attach_token_positions(label.recursive_calls, mt.tokenizer, gen_ids, offset)

    greedy_block = {
        "extracted_code": extracted_code,
        "used_code_tag_fallback": used_tag_fallback,
        "strategy": label.strategy,
        "has_recursion": label.has_recursion,
        "has_loop": label.has_loop,
        "used_ast_fallback": label.used_fallback,
        "recursive_calls": [
            {
                "name": c.name,
                "name_char_span": list(c.name_char_span),
                "name_token_positions": c.name_token_positions or [],
            }
            for c in label.recursive_calls
        ],
    }

    out: dict = {
        "function_name": sample.function_name,
        "source": src_name,
        "prompt": prompt,
        "generation": generation,
        "prompt_bias": prompt_bias,
        "prompt_token_ids": prompt_ids,
        "generation_token_ids": gen_ids,
        "greedy": greedy_block,
    }

    if sampled_gen_ids:
        samples_block = []
        for s_ids in sampled_gen_ids:
            s_text = mt.tokenizer.decode(s_ids, skip_special_tokens=True)
            s_code, _ = extract_code_with_fallback(s_text)
            s_label = label_python_strategy(s_code, sample.function_name)
            samples_block.append(
                {
                    "text": s_text,
                    "strategy": s_label.strategy,
                    "has_recursion": s_label.has_recursion,
                    "has_loop": s_label.has_loop,
                }
            )
        out["samples"] = samples_block

    return out


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
            if not _existing_has_new_schema(base_out_dir / src / f"{s.function_name}.json")
        ]
        skipped = before - len(source_samples)
        if skipped:
            logger.info(
                "Skipping %d up-to-date tasks (use --overwrite to redo). "
                "Stale-schema files are auto-regenerated.",
                skipped,
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

    saved = 0
    with progress:
        for chunk in _chunk(list(zip(source_samples, prompts)), args.batch_size):
            chunk_src_samples, chunk_prompts = zip(*chunk)
            chunk_samples = [s for _, s in chunk_src_samples]
            logger.info(f"current batch: {[s.function_name for s in chunk_samples]}")

            # Greedy pass: prompt_ids + greedy gen_ids per task.
            greedy_results = batch_generate(
                mt, list(chunk_prompts), args.max_new_tokens
            )

            # Optional sampling pass: K extra completions per task.
            sampled_results = _batch_generate_sampled(
                mt, list(chunk_prompts), args.max_new_tokens, args.n_samples
            )

            # Assemble + write per-task JSONs.
            for (src_name, sample), prompt, (prompt_ids, gen_ids), sampled_gen_ids in zip(
                chunk_src_samples, chunk_prompts, greedy_results, sampled_results
            ):
                out = _build_output_record(
                    mt, sample, src_name, prompt, prompt_ids, gen_ids, sampled_gen_ids
                )
                out_dir = base_out_dir / src_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{sample.function_name}.json"
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                logger.info(
                    "Saved %s/%s  (gen_len=%d tokens, strategy=%s, prompt_bias=%s, n_samples=%d)",
                    src_name,
                    out_path.name,
                    len(gen_ids),
                    out["greedy"]["strategy"],
                    out["prompt_bias"],
                    len(sampled_gen_ids),
                )
                saved += 1

            progress.advance(gen_task, len(chunk))

    console.print(
        f"[green]Done.[/green] Saved {saved} generations under [bold]{base_out_dir}[/bold]"
    )


if __name__ == "__main__":
    main()
