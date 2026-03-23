"""[OBSOLETE] Train a sparse head mask using the additive delta patching approach.

This is the original delta-based training script, kept for reference and comparison.
The current approach uses o_proj pre-hook replacement (scripts/train_head_mask.py),
which gives true per-head replacement semantics instead of additive delta injection.

Difference from the new script:
- Uses ``cache_pivot_data(..., precompute=True)`` to cache per-head deltas (W_o_h @ diff).
- Training calls ``forward_with_masked_patch_obsolete`` (baukit post-hook, additive).
- ``--lazy`` flag skips delta precomputation (saves ~42 GB RAM for N=1480 Qwen-32B).
- ``--augment`` requires ``--lazy``.

Usage:
    conda run -n goals python scripts/train_head_mask_delta_obsolete.py \\
        --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
        --layers-start 6 --layers-end 40 \\
        --n-epochs 100 --lambda-sparse 20 --batch-size 16

Outputs:
    results/localization/<model-short-name>/head_mask_delta.pt
"""

import argparse
import logging
import os
import sys

import torch
from rich.console import Console
from rich.table import Table

sys.path.append(".")

from src.goals.head_loc_attn import cache_pivot_data, train_head_mask
from src.goals.head_loc_data import HeadLocDataset
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="[OBSOLETE] Train a sparse head mask using the delta patching approach."
    )
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key (default: Qwen/Qwen2.5-Coder-32B-Instruct)",
    )
    parser.add_argument(
        "--layers-start",
        type=int,
        default=6,
        help="First layer index, inclusive (default: 6)",
    )
    parser.add_argument(
        "--layers-end",
        type=int,
        default=40,
        help="Last layer index, exclusive (default: 40)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="AdamW learning rate (default: 1e-2)",
    )
    parser.add_argument(
        "--lambda-sparse",
        type=float,
        default=20.0,
        help="L1 sparsity coefficient (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Instances per optimizer step (default: 16)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap the number of pivot instances (useful for smoke-tests).",
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help=(
            "Skip delta precomputation (saves ~42 GB RAM for N=1480 Qwen-32B at the "
            "cost of 2 extra forward passes per optimizer step per epoch)"
        ),
    )
    parser.add_argument(
        "--cache-batch-size",
        type=int,
        default=64,
        help=(
            "Max prompts per forward pass during caching (default: 64). "
            "Lower this if cache_pivot_data OOMs (e.g. --cache-batch-size 32)."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save a mask checkpoint every K epochs (default: 5). 0 = disabled.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "Re-augment all instances at the start of every epoch "
            "(source_lang 55%%, variable_rename 35%%, toggle_type_hints 10%%). "
            "Requires --lazy."
        ),
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=123456,
        help="Random seed for per-epoch augmentation (default: 123456).",
    )
    parser.add_argument(
        "--no-docstring",
        action="store_true",
        help="Exclude docstring from prompts (default: include).",
    )

    add_logging_args(parser)
    args = parser.parse_args()
    configure(args)

    model_short_name = args.model_key.split("/")[-1]
    layers = list(range(args.layers_start, args.layers_end))

    logger.info(f"model:            {args.model_key}")
    logger.info(
        f"layers:           {args.layers_start}..{args.layers_end - 1} ({len(layers)} total)"
    )
    logger.info(f"n_epochs:         {args.n_epochs}")
    logger.info(f"lr:               {args.lr}")
    logger.info(f"lambda_sparse:    {args.lambda_sparse}")
    logger.info(f"batch_size:       {args.batch_size}")
    logger.info(f"lazy:             {args.lazy}")
    logger.info(f"augment:          {args.augment}")
    logger.info(f"cache_batch_size: {args.cache_batch_size}")
    logger.info(f"checkpoint_every: {args.checkpoint_every}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset for {model_short_name} ...")
    ds = HeadLocDataset.load(model_name=model_short_name)
    pivot_instances = ds.pivot_instances

    if not pivot_instances:
        logger.error(
            f"No pivot instances found for {model_short_name}. "
            "Run scripts/mine_pivot_instances.py first."
        )
        sys.exit(1)

    if args.max_instances is not None:
        pivot_instances = pivot_instances[: args.max_instances]
        logger.info(f"--max-instances: using {len(pivot_instances)} pivot instances")
    else:
        logger.info(f"Loaded {len(pivot_instances)} pivot instances")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model {args.model_key} ...")
    from src.models import ModelandTokenizer

    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # ------------------------------------------------------------------
    # Cache pivot data (delta-based)
    # ------------------------------------------------------------------
    if args.augment and not args.lazy:
        logger.warning("--augment requires lazy caching; forcing --lazy on.")
        args.lazy = True

    logger.info("=== Caching pivot data (delta path) ===")
    pivot_caches = cache_pivot_data(
        mt,
        pivot_instances,
        ds,
        layers=layers,
        precompute=not args.lazy,
        cache_batch_size=args.cache_batch_size,
        include_docstring=not args.no_docstring,
    )

    # ------------------------------------------------------------------
    # Train head mask
    # ------------------------------------------------------------------
    logger.info("=== Training head mask (delta path) ===")
    checkpoint_dir = os.path.join(
        DEFAULT_RESULTS_DIR, "localization", model_short_name, "checkpoints_delta"
    )
    mask, history = train_head_mask(
        mt,
        pivot_caches,
        layers=layers,
        n_epochs=args.n_epochs,
        lr=args.lr,
        lambda_sparse=args.lambda_sparse,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir if args.checkpoint_every > 0 else None,
        checkpoint_every=args.checkpoint_every,
        augment=args.augment,
        augment_cache_batch_size=args.cache_batch_size,
        augment_seed=args.augment_seed,
        augment_include_docstring=not args.no_docstring,
    )

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    selected_heads = mask.selected_heads(layers)
    save_dir = os.path.join(DEFAULT_RESULTS_DIR, "localization", model_short_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "head_mask_delta.pt")

    checkpoint = {
        "mask_values": mask.values.detach().cpu(),
        "layers": layers,
        "selected_heads": selected_heads,
        "history": {
            "loss_causal": history.loss_causal,
            "loss_sparse": history.loss_sparse,
            "loss_total": history.loss_total,
            "n_active": history.n_active,
            "val_recall": history.val_recall,
            "val_epochs": history.val_epochs,
        },
        "config": {
            "model": args.model_key,
            "n_instances": len(pivot_caches),
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "lambda_sparse": args.lambda_sparse,
            "batch_size": args.batch_size,
            "precompute": not args.lazy,
            "augment": args.augment,
            "augment_seed": args.augment_seed,
            "approach": "delta_obsolete",
        },
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved head_mask_delta.pt → {save_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    final_kl = history.loss_causal[-1] if history.loss_causal else float("nan")

    table = Table(title="Head Mask Training Summary (Delta Path)", highlight=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Model", args.model_key)
    table.add_row("N instances", str(len(pivot_caches)))
    table.add_row("Layers", f"{args.layers_start}–{args.layers_end - 1}")
    table.add_row("Epochs", str(args.n_epochs))
    table.add_row("Final KL", f"{final_kl:.4f}")
    table.add_row("N selected heads", str(len(selected_heads)))
    table.add_row("Selected heads", str(selected_heads))
    if history.val_recall:
        last_rec = history.val_recall[-1]
        table.add_row(
            "Val recall (last)",
            "  ".join(f"@{k + 1}={v:.3f}" for k, v in enumerate(last_rec)),
        )
    table.add_row("Saved to", save_path)
    console.print(table)


if __name__ == "__main__":
    main()
