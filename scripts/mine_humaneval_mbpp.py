"""Mine well-known tasks from HumanEval and MBPP datasets.

Downloads HumanEval and MBPP via the HuggingFace `datasets` library, filters
for tasks with non-trivial control flow (at least one loop AND one conditional,
minimum 8 lines of solution body), and writes them as IncompleteCodeSample
JSON files to:

    data/well-known-tasks/humaneval/<task_id>.json
    data/well-known-tasks/mbpp/<task_id>.json

The ``code`` field of each JSON is the function signature + empty docstring,
matching the format of the existing custom tasks.  The full solution is NOT
stored — the LM will generate it from the instruction.

Usage
-----
conda run -n goals python scripts/mine_humaneval_mbpp.py
conda run -n goals python scripts/mine_humaneval_mbpp.py --sources humaneval
conda run -n goals python scripts/mine_humaneval_mbpp.py --min-lines 6 --overwrite
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from pathlib import Path

sys.path.append(".")

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)

TASKS_ROOT = Path(DEFAULT_DATA_DIR) / "well-known-tasks"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        nargs="+",
        default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp"],
        help="Which datasets to mine (default: both).",
    )
    p.add_argument(
        "--min-lines",
        type=int,
        default=8,
        help="Minimum lines in the solution body to keep a task (default: 8).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-write task files that already exist.",
    )
    add_logging_args(p)
    return p


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def _has_loop(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            return True
    return False


def _has_conditional(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            return True
    return False


def _solution_lines(code: str) -> int:
    """Count non-blank, non-docstring lines in the solution body."""
    lines = [l.strip() for l in code.splitlines()]
    return sum(1 for l in lines if l and not l.startswith(("#", '"""', "'''", '"', "'")))


def _passes_filter(code: str, min_lines: int) -> bool:
    return (
        _has_loop(code)
        and _has_conditional(code)
        and _solution_lines(code) >= min_lines
    )


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------


def _extract_signature(code: str) -> tuple[str, str] | None:
    """Return (function_name, 'def func(args):\\n    \"\"\"') or None on failure."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Reconstruct bare signature from AST source positions
            lines = code.splitlines()
            sig_lines = []
            for li in range(node.lineno - 1, node.body[0].lineno - 1):
                sig_lines.append(lines[li])
            sig = "\n".join(sig_lines).rstrip().rstrip(":")
            incomplete = sig + ":\n    \"\"\""
            return node.name, incomplete
    return None


# ---------------------------------------------------------------------------
# Instruction extraction
# ---------------------------------------------------------------------------

def _extract_instruction(docstring: str | None, task_desc: str | None) -> str:
    """Build a clean instruction string from docstring or task description."""
    raw = (docstring or task_desc or "").strip()
    # Take only the first sentence/paragraph
    first = re.split(r"\n\n|\.\s", raw)[0].strip().rstrip(".")
    if not first:
        return "Implement the function in Python"
    # Capitalise and add trailing period
    first = first[0].upper() + first[1:]
    if not first.endswith((".", "?", "!")):
        first += " in Python"
    return first


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------


def mine_humaneval(min_lines: int, overwrite: bool) -> int:
    from datasets import load_dataset  # type: ignore

    out_dir = TASKS_ROOT / "humaneval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    saved = skipped = filtered = 0

    for item in ds:
        task_id = item["task_id"].replace("/", "_").lower()  # e.g. humaneval_0
        solution = item.get("canonical_solution", "")
        prompt = item.get("prompt", "")
        full_code = prompt + solution

        if not _passes_filter(full_code, min_lines):
            filtered += 1
            continue

        sig = _extract_signature(prompt.strip())
        if sig is None:
            logger.debug("Could not extract signature for %s — skipping", task_id)
            filtered += 1
            continue

        fn_name, code_field = sig
        docstring = item.get("prompt", "")
        # Extract docstring text from prompt
        doc_match = re.search(r'"""(.*?)"""', docstring, re.DOTALL)
        doc_text = doc_match.group(1).strip() if doc_match else ""
        instruction = _extract_instruction(doc_text, None)

        out_path = out_dir / f"{task_id}.json"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        record = {
            "instruction": instruction,
            "code": code_field,
            "unit_tests": [],
            "prefills": {},
            "function_name": fn_name,
            "augmentation": None,
            "without_comments": True,
        }
        out_path.write_text(json.dumps(record, indent=2))
        saved += 1

    logger.info(
        "HumanEval: saved=%d  skipped(exists)=%d  filtered(too simple)=%d",
        saved, skipped, filtered,
    )
    return saved


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------


def mine_mbpp(min_lines: int, overwrite: bool) -> int:
    from datasets import load_dataset  # type: ignore

    out_dir = TASKS_ROOT / "mbpp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the sanitized split which has cleaner code
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
    saved = skipped = filtered = 0

    for item in ds:
        task_id = f"mbpp_{item['task_id']}"
        code = item.get("code", "")

        if not _passes_filter(code, min_lines):
            filtered += 1
            continue

        sig = _extract_signature(code)
        if sig is None:
            logger.debug("Could not extract signature for %s — skipping", task_id)
            filtered += 1
            continue

        fn_name, code_field = sig
        instruction = _extract_instruction(None, item.get("text", ""))

        out_path = out_dir / f"{task_id}.json"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        record = {
            "instruction": instruction,
            "code": code_field,
            "unit_tests": [],
            "prefills": {},
            "function_name": fn_name,
            "augmentation": None,
            "without_comments": True,
        }
        out_path.write_text(json.dumps(record, indent=2))
        saved += 1

    logger.info(
        "MBPP: saved=%d  skipped(exists)=%d  filtered(too simple)=%d",
        saved, skipped, filtered,
    )
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    configure(args)

    total = 0
    if "humaneval" in args.sources:
        total += mine_humaneval(args.min_lines, args.overwrite)
    if "mbpp" in args.sources:
        total += mine_mbpp(args.min_lines, args.overwrite)

    logger.info("Total tasks written: %d", total)
    print(f"Done. {total} task JSONs written to {TASKS_ROOT}/{{humaneval,mbpp}}/")


if __name__ == "__main__":
    main()
