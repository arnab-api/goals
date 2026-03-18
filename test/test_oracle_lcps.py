"""Iterate over _ORACLE_PAIR_PROMPT_TEMPLATE multiple times for several predicate pairs
and analyze the quality / pass rates of returned prefixes.

Usage:
    conda run -n goals python scripts/test_oracle_lcps.py

What it does:
    1. Calls get_oracle_lcps N_RUNS times per pair (with temperature > 0, so each call
       gets a fresh sample from the oracle).
    2. After collecting all candidate prefixes, checks each through the divergence gate
       (dest_top_token != target_top_token) using a small placeholder model check — the
       actual gate needs a loaded MT model, so we skip that part unless --with-model.
    3. Prints per-pair statistics: total prefixes returned, unique prefixes, examples.
    4. Also prints the raw oracle response for the first run so we can inspect formatting.

Note: completion gate requires a live MT model (slow). This script focuses on the oracle
LLM quality. Pass --with-model to also run the completion gate with the 14B model.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter, defaultdict

sys.path.insert(0, ".")

import logging

logging.basicConfig(level=logging.WARNING)  # suppress debug noise

from src.goals.head_loc_data import HeadLocDataset, SinglePredicateSample
from src.goals.head_loc_oracle import (
    ask_claude,
    build_oracle_pair_prompt,
    get_oracle_lcps,
    parse_oracle_pair_response,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_RUNS = 5          # oracle calls per pair
N_PREFIXES = 8      # prefixes to request per call
TEMPERATURE = 0.7   # higher temperature → more diversity across runs

# Pairs to evaluate: (dest_predicate, target_predicate)
PAIRS = [
    ("is_upper", "is_palindrome"),
    ("is_upper", "is_sorted"),
    ("is_palindrome", "is_sorted"),
    ("is_odd", "is_power_of_two"),
    ("has_duplicates", "is_sorted"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _subheader(text: str) -> None:
    print(f"\n--- {text} ---")


def analyze_pair(
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n_runs: int,
    n_prefixes: int,
    temperature: float,
    show_raw_response: bool = True,
    show_prompt: bool = False,
) -> dict:
    """Call get_oracle_lcps n_runs times and aggregate results."""
    dest_name = dest_sample.predicate_name
    targ_name = targ_sample.predicate_name
    _header(f"Pair: ({dest_name}, {targ_name})")

    if show_prompt:
        prompt = build_oracle_pair_prompt(dest_sample, targ_sample, n=n_prefixes, include_docstring=True)
        _subheader("Prompt")
        print(prompt)

    all_prefixes: list[str] = []
    raw_responses: list[str] = []

    for run_i in range(n_runs):
        prompt = build_oracle_pair_prompt(dest_sample, targ_sample, n=n_prefixes, include_docstring=True)
        # Call ask_claude directly so we bypass caching (temperature != 0 means different
        # responses, but ask_claude caches by prompt+temperature hash — so different
        # temperature values give different cache keys).  Use slightly varying temperatures
        # to break cache while staying in a similar range.
        t = temperature + run_i * 0.01  # tiny nudge so cache keys differ across runs
        response = ask_claude(prompt, temperature=t, max_tokens=4096)
        parsed = parse_oracle_pair_response(response)
        raw_responses.append(response)
        all_prefixes.extend(parsed)
        print(f"  Run {run_i+1}/{n_runs}: {len(parsed)} prefixes → {parsed}")

    # Show raw response from first run for format inspection
    if show_raw_response and raw_responses:
        _subheader("Raw oracle response (run 1)")
        print(textwrap.indent(raw_responses[0], "  "))

    # Aggregate stats
    unique = list(dict.fromkeys(all_prefixes))  # deduplicated, order preserved
    counts = Counter(all_prefixes)

    _subheader("Aggregated prefixes (sorted by frequency)")
    for prefix, cnt in counts.most_common():
        bar = "█" * cnt
        print(f"  {cnt}x {bar}  {prefix!r}")

    _subheader("Summary")
    print(f"  Total returned : {len(all_prefixes)}")
    print(f"  Unique         : {len(unique)}")
    print(f"  Avg per run    : {len(all_prefixes)/n_runs:.1f}")

    return {
        "dest": dest_name,
        "targ": targ_name,
        "total": len(all_prefixes),
        "unique": len(unique),
        "prefixes": unique,
        "counts": dict(counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-prompt", action="store_true", help="Print the full prompt for each pair")
    parser.add_argument("--no-raw", action="store_true", help="Suppress printing raw oracle responses")
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--n", type=int, default=N_PREFIXES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--pair", nargs=2, metavar=("DEST", "TARG"),
                        help="Run only this pair instead of all PAIRS")
    args = parser.parse_args()

    ds = HeadLocDataset.load()
    samples_by_name = {s.predicate_name: s for s in ds.samples}

    pairs = [tuple(args.pair)] if args.pair else PAIRS

    results = []
    for dest_name, targ_name in pairs:
        if dest_name not in samples_by_name:
            print(f"[WARN] Unknown predicate: {dest_name!r}")
            continue
        if targ_name not in samples_by_name:
            print(f"[WARN] Unknown predicate: {targ_name!r}")
            continue
        r = analyze_pair(
            samples_by_name[dest_name],
            samples_by_name[targ_name],
            n_runs=args.runs,
            n_prefixes=args.n,
            temperature=args.temperature,
            show_raw_response=not args.no_raw,
            show_prompt=args.show_prompt,
        )
        results.append(r)

    # Final summary table
    _header("Overall Summary")
    print(f"  {'Pair':<40} {'Total':>7} {'Unique':>7} {'Avg/run':>8}")
    print("  " + "-" * 65)
    for r in results:
        pair_str = f"({r['dest']}, {r['targ']})"
        avg = r["total"] / args.runs
        print(f"  {pair_str:<40} {r['total']:>7} {r['unique']:>7} {avg:>8.1f}")


if __name__ == "__main__":
    main()
