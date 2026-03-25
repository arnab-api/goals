"""Mine pivot instances for head localization experiments.

Pipeline:
  Step 1 — Global pairs (model-agnostic, one-time):
    Check if ``data/head_loc/global_pairs.json`` exists.  If not, enumerate all
    same-type predicate combinations and ask the oracle to verify each is a
    semantically meaningful contrastive pair.  Results saved to global_pairs.json.

  Step 2 — Model-specific pairs (per model, one-time):
    Check if ``data/head_loc/pairs/{model_name}.json`` exists.  If not, bootstrap
    it from global_pairs.json (oracle-verified pairs only) with empty model fields
    (kl=0.0, top_token_diverges=False, pivot_instances=[]).

  Step 3 — Pivot mining (per model, re-runnable):
    Run ``get_oracle_pivot_instances`` (Case 3 + Case 3b + augmentation) across
    all pairs and save back to ``pairs/{model_name}.json``.

Usage:
    # Full pipeline
    conda run -n goals python scripts/mine_pivot_instances.py \\
        --model-key Qwen/Qwen2.5-Coder-32B-Instruct

    # Force regenerate global_pairs.json (e.g. after adding new predicates)
    conda run -n goals python scripts/mine_pivot_instances.py \\
        --model-key Qwen/Qwen2.5-Coder-32B-Instruct --regen-global-pairs

    # Pivot mining only on a subset of pairs (smoke-test)
    conda run -n goals python scripts/mine_pivot_instances.py \\
        --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
        --pairs is_upper,is_palindrome is_palindrome,is_upper
"""

import argparse
import logging
import os
import sys

import torch
from rich.console import Console
from rich.table import Table

sys.path.append(".")

from src.goals.head_loc_data import (
    GLOBAL_PAIRS_FILENAME,
    HEAD_LOC_DATA_DIR,
    HEAD_LOC_PAIRS_SUBDIR,
    HeadLocDataset,
)
from src.goals.head_loc_oracle import generate_global_pairs, get_oracle_pivot_instances
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)


# ---------------------------------------------------------------------------
# Step 1 — Global pairs
# ---------------------------------------------------------------------------


def ensure_global_pairs(ds: HeadLocDataset, force: bool = False) -> None:
    """Generate global_pairs.json if absent (or if --regen-global-pairs is set)."""
    path = os.path.join(HEAD_LOC_DATA_DIR, GLOBAL_PAIRS_FILENAME)
    if os.path.isfile(path) and not force:
        logger.info(
            "global_pairs.json already exists — skipping (use --regen-global-pairs to force)"
        )
        return

    action = "Regenerating" if os.path.isfile(path) else "Generating"
    logger.info(f"=== Step 1: {action} global_pairs.json ===")
    pairs = generate_global_pairs(ds, save=True)
    n_unordered = len(pairs) // 2
    logger.info(
        f"Done: {n_unordered} verified unordered pairs → {len(pairs)} directed pairs"
    )


# ---------------------------------------------------------------------------
# Step 2 — Model-specific pairs
# ---------------------------------------------------------------------------


def ensure_model_pairs(ds: HeadLocDataset, model_name: str) -> HeadLocDataset:
    """Bootstrap pairs/{model_name}.json from global_pairs.json if absent.

    Returns a freshly loaded HeadLocDataset with the model-specific pairs.
    """
    path = os.path.join(HEAD_LOC_DATA_DIR, HEAD_LOC_PAIRS_SUBDIR, f"{model_name}.json")
    if os.path.isfile(path):
        logger.info(f"pairs/{model_name}.json already exists — loading")
    else:
        logger.info(f"=== Step 2: Initializing pairs/{model_name}.json ===")
        ds.initialize_model_pairs(model_name)
        logger.info(f"Initialized pairs/{model_name}.json")

    ds = HeadLocDataset.load(model_name=model_name)
    logger.info(f"Loaded {len(ds.pairs)} pairs for {model_name}")
    return ds


# ---------------------------------------------------------------------------
# Step 3 — Pivot mining
# ---------------------------------------------------------------------------


def run_pivot_mining(mt, ds: HeadLocDataset, pairs_subset=None) -> None:
    """Run Case 3 + Case 3b + augmentation and save to pairs/{model_name}.json."""
    logger.info("=== Step 3: Mining pivot instances ===")
    instances = get_oracle_pivot_instances(
        mt,
        ds,
        pairs=pairs_subset,
        include_case3=True,
        include_case3b=True,
        include_augmentation=True,
        oracle_include_docstring=True,
        include_docstring=True,
        save=True,
    )

    # Summary table
    table = Table(title="Pivot Instances", highlight=False)
    table.add_column("Dest", style="cyan")
    table.add_column("Target", style="magenta")
    table.add_column("Prefix", style="yellow")
    table.add_column("dest→", style="green")
    table.add_column("targ→", style="red")
    for pair, pi in instances:
        table.add_row(
            pair.destination_predicate,
            pair.source_predicate,
            repr(pi.pivot_prefix),
            repr(pi.dest_top_token),
            repr(pi.target_top_token),
        )
    console.print(table)
    console.print(f"[bold]Total:[/bold] {len(instances)} pivot instances saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine pivot instances for head localization."
    )
    parser.add_argument(
        "--model-key",
        required=True,
        help="HuggingFace model key, e.g. Qwen/Qwen2.5-Coder-32B-Instruct",
    )
    parser.add_argument(
        "--regen-global-pairs",
        action="store_true",
        help="Force regeneration of global_pairs.json even if it already exists.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        metavar="SRC,DEST",
        help=(
            "Restrict pivot mining to specific pairs, e.g. "
            "'is_upper,is_palindrome is_palindrome,is_upper'. "
            "Steps 1 and 2 still run across all pairs."
        ),
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure(args)

    model_name = args.model_key.split("/")[-1]

    # Parse optional pairs subset
    pairs_subset = None
    if args.pairs:
        pairs_subset = []
        for spec in args.pairs:
            parts = spec.split(",")
            if len(parts) != 2:
                parser.error(f"--pairs entries must be 'SRC,DEST', got {spec!r}")
            pairs_subset.append((parts[0].strip(), parts[1].strip()))

    # Load model (GPU required for Step 3)
    logger.info(f"Loading model {args.model_key} ...")
    from src.models import ModelandTokenizer

    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Load samples (pairs don't matter yet for Step 1)
    ds = HeadLocDataset.load(model_name=model_name)
    if len(ds.samples) == 0:
        logger.error("No predicate samples found in data/head_loc/predicates/")
        sys.exit(1)
    logger.info(f"Loaded {len(ds.samples)} predicate samples")

    # Step 1
    ensure_global_pairs(ds, force=args.regen_global_pairs)

    # Step 2
    ds = ensure_model_pairs(ds, model_name)

    # Resolve pairs subset to PredicatePair objects
    resolved_pairs = None
    if pairs_subset is not None:
        resolved_pairs = []
        for src, dest in pairs_subset:
            pair = ds.get_pair(src, dest)
            if pair is None:
                logger.error(
                    f"Pair ({src!r}, {dest!r}) not found in pairs/{model_name}.json. "
                    "Check that it passed oracle verification."
                )
                sys.exit(1)
            resolved_pairs.append(pair)
        logger.info(f"Restricting to {len(resolved_pairs)} specified pairs")

    # Step 3
    run_pivot_mining(mt, ds, pairs_subset=resolved_pairs)


if __name__ == "__main__":
    main()
