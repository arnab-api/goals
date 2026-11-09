"""Generate code completions for all well-known-tasks using a code-LM.

For each IncompleteCodeSample in data/well-known-tasks/, builds the v2 chat prompt
(no-comment instruction in the system turn, assistant prefill ending at ``def <name>(``)
and runs batched greedy generation.  Results are saved as JSON files under
results/well-known-tasks/generations_v2/<model-short-name>/<source>/<task-id>.json.

In addition to the greedy completion, the script labels each generation
with a proactive-strategy classifier (recursive vs iterative) and records
the token positions where recursive calls happen (see ``src/goals/proactive_strategy.py``).

Each output file is a JSON dict with keys:
    function_name  : str
    task_id        : str
    source         : str    (ds_and_algo / humaneval / mbpp)
    prompt         : str    (full prompt sent to the model)
    generation     : str    (greedy LM completion, decoded)
    prompt_bias    : str    (neutral / recursive / iterative, derived from instruction)
    prompt_token_ids        : list[int]
    generation_token_ids    : list[int]   (greedy gen tokens, used for token-position tagging)
    greedy : {
        extracted_code            : str
        n_hash_comments, n_docstrings, has_comments
        used_code_tag_fallback    : bool
        strategy                  : recursive / iterative / Unsure
        has_recursion             : bool
        has_loop                  : bool
        used_ast_fallback         : bool   (regex fallback was used because AST failed)
        recursive_calls : [
            {name: str, name_char_span: [s, e], name_token_positions: [t, ...]}
        ]
    }

Usage
─────
conda run -n goals python scripts/generate_well_known_task_completions.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct
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

from src.goals.data import IncompleteCodeSample
from src.goals.functional import apply_chat_template, batch_generate
from src.goals.proactive_strategy import (
    attach_token_positions,
    code_offset_in_generation,
    detect_prompt_bias,
    extract_code_with_fallback,
    label_python_strategy,
)
from src.models import ModelandTokenizer
from src.utils.env_utils import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

# Schema version keys — presence of these in an existing JSON indicates the
# new richer schema.  Missing → forced regen even without --overwrite.
_NEW_SCHEMA_KEYS = ("prompt_bias", "generation_token_ids", "greedy")

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)

TASKS_DIR = Path(DEFAULT_DATA_DIR) / "well-known-tasks"

# v2 prompt (the GoalScope-demo prompt): the "no comments" instruction lives in the SYSTEM turn, and the
# assistant prefill stops right after ``def <name>(`` so the model chooses the parameters and writes no
# docstring.
V2_SYSTEM_PROMPT = (
    "You are a helpful assistant proficient in programming. "
    "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain."
)


def v2_prefill(sample) -> str:
    """``def <name>(`` (or ``class <Name>`` for the class tasks) — the bank's header cut before parameters.
    A task with ``"prefill_full_signature": true`` keeps its whole (typed) ``def`` line instead, for tasks whose
    interface the model otherwise gets wrong (e.g. stable_marriage)."""
    head = sample.code.split("\n", 1)[0]
    if getattr(sample, "prefill_full_signature", False) and head.startswith("def "):
        return head
    if head.startswith("def ") and "(" in head:
        return head[: head.index("(") + 1]
    if head.startswith("class "):
        return head.split(":", 1)[0].split("(", 1)[0] + ":"   # `class Trie:` (bare `class Trie` -> `TrieNode`)
    raise ValueError(f"{sample.function_name}: unexpected header {head!r}")


def _out_name(sample) -> str:
    """Outputs are named by bank task id (unique, unlike function_name)."""
    return sample.task_id


def comment_stats(code: str) -> dict:
    """How much commenting the model wrote: ``#`` comments and docstrings (best effort on partial code)."""
    import ast
    import io
    import tokenize
    n_hash = 0
    try:
        n_hash = sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        n_hash = sum(1 for ln in code.split("\n") if ln.lstrip().startswith("#"))
    try:
        tree = ast.parse(code)
        n_doc = sum(1 for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
                    and ast.get_docstring(n) is not None)
    except SyntaxError:
        n_doc = code.count('"""') // 2 + code.count("'''") // 2
    return {"n_hash_comments": n_hash, "n_docstrings": n_doc, "has_comments": bool(n_hash or n_doc)}


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
        "--overwrite",
        action="store_true",
        help="Re-generate tasks that already have an output file.",
    )
    add_logging_args(parser)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_samples() -> list[tuple[str, IncompleteCodeSample]]:
    """Load (source_name, IncompleteCodeSample) pairs from data/well-known-tasks/<source>/."""
    if not TASKS_DIR.exists():
        raise FileNotFoundError(f"Tasks directory not found: {TASKS_DIR}")

    source_dirs = sorted(d for d in TASKS_DIR.iterdir() if d.is_dir())

    results = []
    for src_dir in source_dirs:
        src_name = src_dir.name
        paths = sorted(src_dir.glob("*.json"))
        for path in paths:
            with open(path) as f:
                d = json.load(f)
            sample = IncompleteCodeSample.from_dict(d)
            sample.task_id = path.stem   # bank file name; unique, unlike function_name (e.g. 2x is_palindrome)
            sample.prefill_full_signature = bool(d.get("prefill_full_signature", False))
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


def _build_output_record(
    mt: ModelandTokenizer,
    sample: IncompleteCodeSample,
    src_name: str,
    prompt: str,
    prompt_ids: list[int],
    gen_ids: list[int],
) -> dict:
    """Assemble the per-task JSON record: greedy decode + strategy label +
    token-position tagging.
    """
    generation = mt.tokenizer.decode(gen_ids, skip_special_tokens=True)
    prompt_bias = detect_prompt_bias(sample.instruction)

    extracted_code, used_tag_fallback = extract_code_with_fallback(generation)
    if "</code>" in extracted_code and "<code>" in prompt:
        # the prompt opened the <code> block, so the generation holds only the closing tag: keep what precedes it
        extracted_code = extracted_code.split("</code>", 1)[0].rstrip()
    label = label_python_strategy(extracted_code, sample.function_name)
    offset = code_offset_in_generation(generation, extracted_code)
    attach_token_positions(label.recursive_calls, mt.tokenizer, gen_ids, offset)

    greedy_block = {
        "extracted_code": extracted_code,
        # v2 prompts end at `def <name>(` — score commenting on the full function (prefill header + code)
        **comment_stats((v2_prefill(sample) if prompt.rstrip().endswith(v2_prefill(sample)) else "") + extracted_code),
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
        "task_id": getattr(sample, "task_id", sample.function_name),
        "source": src_name,
        "prompt": prompt,
        "generation": generation,
        "prompt_bias": prompt_bias,
        "prompt_token_ids": prompt_ids,
        "generation_token_ids": gen_ids,
        "greedy": greedy_block,
    }

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

    # ── Base output directory ─────────────────────────────────────────────────
    base_out_dir = Path(DEFAULT_RESULTS_DIR) / "well-known-tasks" / "generations_v2" / model_short

    # ── Load samples (list of (source_name, IncompleteCodeSample)) ───────────
    source_samples = _load_samples()
    logger.info("Loaded %d tasks from %s", len(source_samples), TASKS_DIR)

    if not args.overwrite:
        before = len(source_samples)
        source_samples = [
            (src, s)
            for src, s in source_samples
            if not _existing_has_new_schema(base_out_dir / src / f"{_out_name(s)}.json")
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
        prompt = apply_chat_template(
            mt,
            user_query=sample.instruction.strip(),
            incomplete_code=v2_prefill(sample),
            system_prompt=V2_SYSTEM_PROMPT,
        )
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

            # Assemble + write per-task JSONs.
            for (src_name, sample), prompt, (prompt_ids, gen_ids) in zip(
                chunk_src_samples, chunk_prompts, greedy_results
            ):
                out = _build_output_record(mt, sample, src_name, prompt, prompt_ids, gen_ids)
                out_dir = base_out_dir / src_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{_out_name(sample)}.json"
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                logger.info(
                    "Saved %s/%s  (gen_len=%d tokens, strategy=%s, prompt_bias=%s)",
                    src_name,
                    out_path.name,
                    len(gen_ids),
                    out["greedy"]["strategy"],
                    out["prompt_bias"],
                )
                saved += 1

            progress.advance(gen_task, len(chunk))

    console.print(
        f"[green]Done.[/green] Saved {saved} generations under [bold]{base_out_dir}[/bold]"
    )


if __name__ == "__main__":
    main()
