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
    conda run -n goals python scripts/mine_pivot_instances.py \\
        --model-key Qwen/Qwen2.5-Coder-32B-Instruct
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


def ensure_global_pairs(ds: HeadLocDataset) -> None:
    """Generate global_pairs.json if absent."""
    path = os.path.join(HEAD_LOC_DATA_DIR, GLOBAL_PAIRS_FILENAME)
    if os.path.isfile(path):
        logger.info("global_pairs.json already exists — skipping")
        return

    logger.info("=== Step 1: Generating global_pairs.json ===")
    pairs = generate_global_pairs(ds)
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


def run_pivot_mining(mt, ds: HeadLocDataset) -> None:
    """Run Case 3 + Case 3b + augmentation and save to pairs/{model_name}.json."""
    logger.info("=== Step 3: Mining pivot instances ===")
    instances = get_oracle_pivot_instances(
        mt,
        ds,
        oracle_include_docstring=True,
        include_docstring=True,
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
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key (default: Qwen/Qwen2.5-Coder-32B-Instruct)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure(args)

    model_name = args.model_key.split("/")[-1]

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
    ensure_global_pairs(ds)

    # Step 2
    ds = ensure_model_pairs(ds, model_name)

    # Step 3
    run_pivot_mining(mt, ds)


if __name__ == "__main__":
    main()
