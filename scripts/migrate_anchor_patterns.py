"""One-shot migration script: convert literal anchor_patterns to variable-agnostic regex.

Reads all TargetPredicate JSON files in data/multi_predicate/, converts every
anchor pattern string using convert_anchor_tokens_to_patterns(), prints a diff,
and writes back via tp.save().

Usage:
    conda run -n goals python scripts/migrate_anchor_patterns.py
    conda run -n goals python scripts/migrate_anchor_patterns.py --dry-run
"""

import argparse
import re
import sys

sys.path.insert(0, ".")

from src.goals.anchor import convert_anchor_tokens_to_patterns
from src.goals.predicate_data import MultiPredicateDataset


def extract_var_names(code: str) -> list[str]:
    """Extract parameter names from a Python function signature.

    Parses the first 'def ...(...)' line and returns the positional parameter
    names (skipping 'self', *args, **kwargs style params).

    Example:
        "def is_valid_palindrome(s: str) -> bool:" -> ["s"]
        "def validate(host: str, port: int) -> bool:" -> ["host", "port"]
    """
    m = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not m:
        return []
    params_str = m.group(1)
    var_names: list[str] = []
    for param in params_str.split(","):
        param = param.strip()
        if not param or param.startswith("*"):
            continue
        # Take the name before ':' or '='
        name = re.split(r"[:\s=]", param)[0].strip()
        if name and name.isidentifier():
            var_names.append(name)
    return var_names


def migrate_tp(tp, dry_run: bool) -> int:
    """Migrate one TargetPredicate in-place.  Returns number of patterns changed."""
    changes = 0

    # Collect all var names across samples (for p_plus which is shared)
    all_var_names: list[str] = []
    for sample in tp.samples:
        all_var_names.extend(extract_var_names(sample.code))
    all_var_names = sorted(set(all_var_names))

    # ── P+ patterns (shared across all samples) ──────────────────────────────
    old_p_plus = list(tp.p_plus.anchor_patterns)
    new_p_plus = convert_anchor_tokens_to_patterns(old_p_plus, all_var_names)
    if old_p_plus != new_p_plus:
        print(f"\n  [p_plus] anchor_patterns diff:")
        for o, n in zip(old_p_plus, new_p_plus):
            marker = "  " if o == n else "* "
            print(f"    {marker}{o!r:40s}  →  {n!r}")
        changes += sum(o != n for o, n in zip(old_p_plus, new_p_plus))
        if not dry_run:
            tp.p_plus.anchor_patterns = new_p_plus

    # ── P- patterns (per sample) ──────────────────────────────────────────────
    for i, sample in enumerate(tp.samples):
        if sample.p_minus is None:
            continue
        var_names = extract_var_names(sample.code)
        old_p_minus = list(sample.p_minus.anchor_patterns)
        new_p_minus = convert_anchor_tokens_to_patterns(old_p_minus, var_names)
        if old_p_minus != new_p_minus:
            fn = sample.function_name or f"sample_{i}"
            print(f"\n  [p_minus / {fn}] anchor_patterns diff:")
            for o, n in zip(old_p_minus, new_p_minus):
                marker = "  " if o == n else "* "
                print(f"    {marker}{o!r:40s}  →  {n!r}")
            changes += sum(o != n for o, n in zip(old_p_minus, new_p_minus))
            if not dry_run:
                sample.p_minus.anchor_patterns = new_p_minus

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs but do not write JSON files.",
    )
    args = parser.parse_args()

    ds = MultiPredicateDataset.load_dataset()
    total_changes = 0

    for tp in ds.groups:
        print(f"\n{'='*60}")
        print(f"TargetPredicate: {tp.name}  ({len(tp.samples)} samples)")
        print(f"  All var names: {sorted(set(v for s in tp.samples for v in extract_var_names(s.code)))}")
        n = migrate_tp(tp, dry_run=args.dry_run)
        total_changes += n
        if n == 0:
            print("  (no changes)")
        elif not args.dry_run:
            tp.save()
            print(f"  Saved → {tp._source_path}")

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY RUN: {total_changes} pattern(s) would be changed across {len(ds.groups)} files.")
    else:
        print(f"Done. {total_changes} pattern(s) changed across {len(ds.groups)} files.")


if __name__ == "__main__":
    main()
