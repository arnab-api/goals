"""Diagnostic script: iterate the oracle pair prompt and understand what's going wrong.

Runs the CURRENT prompt N times for a fixed pair, then runs 2-3 VARIANT prompts,
prints raw responses side-by-side with annotations, and produces a verdict on what
to fix.

Usage:
    conda run -n goals python scripts/diagnose_oracle_prompt.py
    conda run -n goals python scripts/diagnose_oracle_prompt.py --pair is_upper is_palindrome
    conda run -n goals python scripts/diagnose_oracle_prompt.py --runs 3 --show-raw
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap

sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

from src.goals.head_loc_data import HeadLocDataset, SinglePredicateSample
from src.goals.head_loc_oracle import (
    _ORACLE_PAIR_PROMPT_TEMPLATE as PROMPT_V3_LIVE,
    _code_with_docstring,
)
from src.utils.oracle_llms import ask_claude

# ---------------------------------------------------------------------------
# Prompt variants to compare
# ---------------------------------------------------------------------------

# --- CURRENT (v0) ---
PROMPT_V0 = '''\
You are given two Python predicate functions that both end at `return`.

Your task: find at least {n} shared prefixes P (what follows `return `) such that:

  **The very first new token each function generates after P is different.**
  **And, the diverging token reflects the semantic difference between the two predicates.**

  That is: if you write `return P` and let function A continue, the first new token
  it produces is X. If you let function B continue from the same `return P`, the
  first new token it produces is Y. Then:
  * X ≠ Y
  * The difference between X and Y is semantically meaningful — it encodes what
    makes the two predicates distinct (see the examples below).

  Additional constraints:
  - Both functions must be completable to a valid, correct implementation from P.
  - Make P as long as possible — a longer shared prefix means a more specific pivot.

For each prefix output this block exactly, no extra text:

PREFIX: <prefix>
A: <full return expression for A>
B: <full return expression for B>
DIVERGES: A=<first new token A generates after P> B=<first new token B generates after P>

=== Example 1 (short prefix, index-level divergence) ===

Function A:
def starts_with_vowel(s: str) -> bool:
    """Return True if the first character of s is a vowel."""
    return

Function B:
def ends_with_vowel(s: str) -> bool:
    """Return True if the last character of s is a vowel."""
    return

PREFIX: "s["
A: s[0].lower() in 'aeiou'
B: s[-1].lower() in 'aeiou'
DIVERGES: A='0' B='-'

PREFIX: "not s["
A: not s[0].lower() in 'aeiou'
B: not s[-1].lower() in 'aeiou'
DIVERGES: A='0' B='-'

=== Example 2 (long prefix, operator-level divergence) ===

Function A:
def is_strictly_increasing(lst: list[int]) -> bool:
    """Return True if each element is strictly greater than the previous."""
    return

Function B:
def is_strictly_decreasing(lst: list[int]) -> bool:
    """Return True if each element is strictly less than the previous."""
    return

PREFIX: "all(lst[i]"
A: all(lst[i] < lst[i+1] for i in range(len(lst)-1))
B: all(lst[i] > lst[i+1] for i in range(len(lst)-1))
DIVERGES: A='<' B='>'

PREFIX: "not all(lst[i]"
A: not all(lst[i] < lst[i+1] for i in range(len(lst)-1))
B: not all(lst[i] > lst[i+1] for i in range(len(lst)-1))
DIVERGES: A='<' B='>'

=== Tips ===

- Start from the longest natural shared expression, then work backwards to find P.
- Simple augmentations like `not`, `bool(`, `int(` can flip or wrap the expression
  and give you additional valid prefixes.

=== Your turn ===

Function A (destination):
{dest_code}

Function B (target):
{targ_code}
'''

# --- VARIANT v1: concrete "think-aloud" framing, explicit bad-example warning ---
PROMPT_V1 = '''\
You are given two Python predicate functions (A and B), both ending at `return`.

Step 1 — write down 3-5 valid, complete return expressions for A.
Step 2 — write down 3-5 valid, complete return expressions for B.
Step 3 — find the longest common prefix P shared by at least one expression from A
          and one from B, such that after P the two expressions diverge at a
          **single semantically meaningful token** (the token that encodes the
          functional difference between A and B).

Repeat Step 3 for different expression pairs to collect {n} distinct prefixes.

Rules for a good PREFIX:
  - P must be a strict prefix of a valid return expression for both A and B.
  - The very next token after P must differ between the two completions.
  - That diverging token must reflect the semantic difference between A and B
    (e.g. an index `0` vs `-1`, an operator `<` vs `>`, a method `.upper` vs `.lower`).
  - P should be as long as possible while still leaving room for divergence.

Bad PREFIX examples (DO NOT output these):
  - Single-character prefixes like "s", "n", "l" (too short, carries no structure).
  - Prefixes where both A and B would naturally continue identically.
  - Prefixes that don't correspond to a real sub-expression boundary.

For each good prefix output exactly:

PREFIX: <prefix>
A: <complete return expression for A>
B: <complete return expression for B>
DIVERGES: A=<first new token for A after P>  B=<first new token for B after P>

=== Example ===

Function A:
def starts_with_vowel(s: str) -> bool:
    """Return True if the first character of s is a vowel."""
    return

Function B:
def ends_with_vowel(s: str) -> bool:
    """Return True if the last character of s is a vowel."""
    return

IMPLEMENTATIONS:
  A: s[0].lower() in 'aeiou'  |  s[0] in 'aeiouAEIOU'  |  s[:1].lower() in 'aeiou'
  B: s[-1].lower() in 'aeiou' |  s[-1] in 'aeiouAEIOU'  |  s[-1:].lower() in 'aeiou'

PREFIX: "s["
A: s[0].lower() in 'aeiou'
B: s[-1].lower() in 'aeiou'
DIVERGES: A='0' B='-'

PREFIX: "s[0].lower() in" vs "s[-1" — both share "s[" only, so P = "s["
(already covered above — showing the reasoning)

PREFIX: "not s["
A: not s[0] in 'aeiouAEIOU'
B: not s[-1] in 'aeiouAEIOU'
DIVERGES: A='0' B='-'

=== Your turn ===

Function A (destination):
{dest_code}

Function B (target):
{targ_code}
'''

# --- VARIANT v2: anchor on concrete known implementations, then find P ---
PROMPT_V2 = '''\
Two Python predicate functions are given below (A and B), ending at `return`.

Your task is to find {n} pivot prefixes — strings P such that:
  (a) `return P` can be legitimately continued to a correct implementation of A
  (b) `return P` can be legitimately continued to a correct implementation of B
  (c) The first continuation token for A differs from the first continuation token for B
  (d) That token difference is semantically meaningful — it captures the functional
      difference between A and B (not just a stylistic choice)

PROCESS:
1. List concrete implementations of A (at least 3).
2. List concrete implementations of B (at least 3).
3. For each pair (impl_A, impl_B), find their longest common prefix P.
   Check: does the very next character differ, AND is that difference semantic?
   If yes → this is a good pivot.
4. Also consider wrapping prefixes: prepend `not `, `bool(`, `all(`, etc.

OUTPUT FORMAT — one block per pivot, nothing else between blocks:

PREFIX: <the prefix P, without quotes>
A: <full implementation of A from this pivot>
B: <full implementation of B from this pivot>
DIVERGES: A=<next token after P for A>  B=<next token after P for B>

=== Example (is_strictly_increasing vs is_strictly_decreasing) ===

Implementations of A (is_strictly_increasing):
  all(lst[i] < lst[i+1] for i in range(len(lst)-1))
  all(a < b for a, b in zip(lst, lst[1:]))
  lst == sorted(lst) and len(lst) == len(set(lst))

Implementations of B (is_strictly_decreasing):
  all(lst[i] > lst[i+1] for i in range(len(lst)-1))
  all(a > b for a, b in zip(lst, lst[1:]))
  lst == sorted(lst, reverse=True) and len(lst) == len(set(lst))

LCP of "all(lst[i] < ..." and "all(lst[i] > ...":
  → P = "all(lst[i]", next token A='<', B='>'  ✓ semantic

PREFIX: all(lst[i]
A: all(lst[i] < lst[i+1] for i in range(len(lst)-1))
B: all(lst[i] > lst[i+1] for i in range(len(lst)-1))
DIVERGES: A='<'  B='>'

LCP of "all(a < ..." and "all(a > ...":
  → P = "all(a", next token A=' ' — but space is not meaningful. Go to "all(a "
  → still both have same space. Find first real divergence: '<' vs '>'
  → P = "all(a ", next A='<', B='>'  ✓

PREFIX: all(a
A: all(a < b for a, b in zip(lst, lst[1:]))
B: all(a > b for a, b in zip(lst, lst[1:]))
DIVERGES: A='<'  B='>'

=== Your turn ===

Function A (destination):
{dest_code}

Function B (target):
{targ_code}
'''

VARIANTS = {
    "v0_current": PROMPT_V0,
    "v1_think_aloud": PROMPT_V1,
    "v2_anchor_impls": PROMPT_V2,
    "v3_lcp_process": PROMPT_V3_LIVE,
}

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _is_trivial(prefix: str) -> bool:
    """Heuristic: single-char or whitespace-only prefixes are probably bad."""
    return len(prefix.strip()) <= 2


def _annotate_prefixes(prefixes: list[str]) -> list[str]:
    lines = []
    for p in prefixes:
        tag = " ⚠ TRIVIAL" if _is_trivial(p) else " ✓"
        lines.append(f"  {p!r}{tag}")
    return lines


def _parse_blocks(response: str) -> list[dict]:
    """Parse PREFIX/A/B/DIVERGES blocks from <pivot> tags in oracle response."""
    def _strip_delimiters(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] in ('"', "'", "`") and s[0] == s[-1]:
            return s[1:-1]
        return s

    blocks = []
    for block_text in re.findall(r"<pivot>(.*?)</pivot>", response, re.DOTALL):
        current: dict = {}
        for line in block_text.splitlines():
            line_s = line.strip()
            if line_s.startswith("PREFIX:"):
                current["PREFIX"] = _strip_delimiters(line_s[len("PREFIX:"):].strip())
            elif line_s.startswith("A:"):
                current["A"] = _strip_delimiters(line_s[2:].strip())
            elif line_s.startswith("B:"):
                current["B"] = _strip_delimiters(line_s[2:].strip())
            elif line_s.startswith("DIVERGES:"):
                current["DIVERGES"] = line_s[len("DIVERGES:"):].strip()
        if "PREFIX" in current:
            blocks.append(current)
    return blocks


def _check_diverges_claim(block: dict) -> str:
    """Verify that the DIVERGES claim is consistent with A and B."""
    p = block.get("PREFIX", "")
    a_expr = block.get("A", "")
    b_expr = block.get("B", "")

    # Check A and B actually start with PREFIX
    issues = []
    if not a_expr.startswith(p):
        issues.append("A expr doesn't start with PREFIX")
    if not b_expr.startswith(p):
        issues.append("B expr doesn't start with PREFIX")

    # Extract next token after prefix (rough: first non-space char or identifier start)
    a_after = a_expr[len(p):].lstrip() if a_expr.startswith(p) else ""
    b_after = b_expr[len(p):].lstrip() if b_expr.startswith(p) else ""

    if a_after and b_after and a_after[0] == b_after[0]:
        issues.append(f"A and B share same next char ({a_after[0]!r}) — no real divergence")

    if issues:
        return "  ISSUES: " + "; ".join(issues)
    if a_after and b_after:
        return f"  next-char check: A={a_after[0]!r} B={b_after[0]!r} {'✓' if a_after[0]!=b_after[0] else '✗'}"
    return "  (could not verify)"


def run_variant(
    variant_name: str,
    prompt_template: str,
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n_prefixes: int,
    n_runs: int,
    temperature: float,
) -> dict:
    dest_code = _code_with_docstring(dest_sample)
    targ_code = _code_with_docstring(targ_sample)

    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant_name}")
    print(f"  Pair: ({dest_sample.predicate_name}, {targ_sample.predicate_name})")
    print(f"{'='*70}")

    all_blocks: list[dict] = []
    all_prefixes: list[str] = []

    for run_i in range(n_runs):
        t = temperature + run_i * 0.01  # nudge to avoid identical cache hits
        prompt = prompt_template.format(
            dest_code=dest_code,
            targ_code=targ_code,
            n=n_prefixes,
        )
        response = ask_claude(prompt, temperature=t, max_tokens=4096)
        blocks = _parse_blocks(response)
        prefixes = [b["PREFIX"] for b in blocks]
        all_blocks.extend(blocks)
        all_prefixes.extend(prefixes)

        print(f"\n  --- Run {run_i+1}/{n_runs} ---")
        print("\n  [Raw oracle response]")
        print(textwrap.indent(response, "    "))
        print(f"\n  Parsed {len(blocks)} blocks, {len(prefixes)} prefixes:")
        for line in _annotate_prefixes(prefixes):
            print(line)

        # Check each block for consistency
        print("\n  Block consistency:")
        for b in blocks:
            p = b.get("PREFIX", "")
            print(f"    PREFIX={p!r}")
            print(f"      A: {b.get('A', '(missing)')}")
            print(f"      B: {b.get('B', '(missing)')}")
            print(f"      DIVERGES: {b.get('DIVERGES', '(missing)')}")
            print(_check_diverges_claim(b))

    # Summary
    unique = list(dict.fromkeys(all_prefixes))
    trivial = [p for p in unique if _is_trivial(p)]
    good = [p for p in unique if not _is_trivial(p)]

    print(f"\n  === Summary for {variant_name} ===")
    print(f"  Total prefixes returned : {len(all_prefixes)}")
    print(f"  Unique                  : {len(unique)}")
    print(f"  Trivial (len<=2)        : {len(trivial)}  {trivial}")
    print(f"  Good candidates         : {len(good)}  {good}")

    return {"variant": variant_name, "total": len(all_prefixes), "unique": len(unique),
            "trivial": trivial, "good": good}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_PAIRS = [
    ("is_upper", "is_palindrome"),
    ("is_odd", "is_power_of_two"),
    ("has_duplicates", "is_sorted"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, metavar=("DEST", "TARG"),
                        help="Run a single pair instead of all pairs")
    parser.add_argument("--runs", type=int, default=3,
                        help="Oracle calls per variant per pair")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of prefixes to request per call")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()),
                        help="Run only this variant (default: all)")
    parser.add_argument("--model", default="unknown",
                        help="Model short name for pairs file (mt.name.split('/')[-1])")
    args = parser.parse_args()

    ds = HeadLocDataset.load(model_name=args.model)
    samples = {s.predicate_name: s for s in ds.samples}

    pairs_to_run = [tuple(args.pair)] if args.pair else ALL_PAIRS
    variants_to_run = {args.variant: VARIANTS[args.variant]} if args.variant else VARIANTS

    # all_summaries: list of (pair_str, variant_name, result_dict)
    all_summaries = []

    for dest_name, targ_name in pairs_to_run:
        dest_sample = samples[dest_name]
        targ_sample = samples[targ_name]
        pair_str = f"({dest_name}, {targ_name})"

        print(f"\n{'#'*70}")
        print(f"#  PAIR: {pair_str}")
        print(f"#  Runs: {args.runs}  |  N: {args.n}  |  Temperature: {args.temperature}")
        print(f"{'#'*70}")

        for vname, vtemplate in variants_to_run.items():
            result = run_variant(
                variant_name=vname,
                prompt_template=vtemplate,
                dest_sample=dest_sample,
                targ_sample=targ_sample,
                n_prefixes=args.n,
                n_runs=args.runs,
                temperature=args.temperature,
            )
            all_summaries.append((pair_str, vname, result))

    # Final comparison table across all pairs × variants
    print(f"\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Pair':<36} {'Variant':<22} {'Total':>6} {'Unique':>7} {'Good':>6}")
    print("  " + "-" * 80)
    for pair_str, vname, s in all_summaries:
        print(f"  {pair_str:<36} {vname:<22} {s['total']:>6} {s['unique']:>7} {len(s['good']):>6}")
    print()

    for pair_str, vname, s in all_summaries:
        if s["good"]:
            print(f"  {pair_str} | {vname} — good: {s['good']}")


if __name__ == "__main__":
    main()
