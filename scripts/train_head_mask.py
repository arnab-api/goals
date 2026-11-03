"""Train a sparse attention head mask for the head localization experiment.

Uses o_proj input replacement: for each head h at layer l, the head's slice of the
o_proj input (nnterp's ``attention_premix``) at the last token is interpolated as:
    mask[l,h]*src_h + (1-mask[l,h])*cascade_h
(mask=1 → pure source, mask=0 → pure cascade dest), inside one nnterp trace of the
destination with gradients to the mask. All layers are trained.

Two caching modes:
  * default (augment): src head values are recomputed per batch, and every
    instance is re-augmented at the start of each epoch (source_lang 55%,
    variable_rename 35%, toggle_type_hints 10%).
  * --no-augment: src head values are precomputed once for all instances.

Usage (the paper's Qwen2.5-Coder-32B-Instruct lam20 mask):
    conda run -n goals python scripts/train_head_mask.py

    # precomputed, no augmentation (e.g. the Coder-14B lam20 mask)
    conda run -n goals python scripts/train_head_mask.py \\
        --model-key Qwen/Qwen2.5-Coder-14B-Instruct --no-augment --batch-size 32

Outputs:
    results/localization/<model-short-name>/<tag>/head_mask.pt
    results/localization/<model-short-name>/<tag>/checkpoints/
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

LR = 1e-2  # AdamW learning rate
CACHE_BATCH_SIZE = 64  # prompts per forward pass while caching / augmenting
CHECKPOINT_EVERY = 5  # epochs between mask checkpoints (and val-recall evals)
AUGMENT_SEED = 123456


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a sparse attention head mask for head localization."
    )
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key (default: Qwen/Qwen2.5-Coder-32B-Instruct)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--tag",
        default="lam20",
        help="subdir under results/localization/<model>/ for this run (default: lam20)",
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
        default=64,
        help="Instances per optimizer step (default: 64)",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-augment all instances at the start of every epoch, recomputing src head "
            "values per batch (default). --no-augment precomputes src head values once."
        ),
    )

    add_logging_args(parser)
    args = parser.parse_args()
    configure(args)

    model_short_name = args.model_key.split("/")[-1]

    logger.info(f"model:            {args.model_key}")
    logger.info(f"n_epochs:         {args.n_epochs}")
    logger.info(f"lambda_sparse:    {args.lambda_sparse}")
    logger.info(f"batch_size:       {args.batch_size}")
    logger.info(f"augment:          {args.augment}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    _FALLBACK_MODEL = "Qwen2.5-Coder-32B-Instruct"

    logger.info(f"Loading dataset for {model_short_name} ...")
    ds = HeadLocDataset.load(model_name=model_short_name)
    pivot_instances = ds.pivot_instances

    if not pivot_instances:
        if model_short_name == _FALLBACK_MODEL:
            logger.error(
                f"No pivot instances found for {_FALLBACK_MODEL}. "
                "Run scripts/mine_pivot_instances.py first."
            )
            sys.exit(1)
        logger.warning(
            f"No pivot instances found for {model_short_name}. "
            f"Falling back to {_FALLBACK_MODEL} pivot instances — "
            "target_top_token will be recalculated for the current model inside cache_pivot_data."
        )
        ds = HeadLocDataset.load(model_name=_FALLBACK_MODEL)
        pivot_instances = ds.pivot_instances
        if not pivot_instances:
            logger.error(
                f"Fallback dataset {_FALLBACK_MODEL} also has no pivot instances. "
                "Run scripts/mine_pivot_instances.py first."
            )
            sys.exit(1)

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
    # Cache pivot data
    # ------------------------------------------------------------------
    layers = list(range(mt.num_layers))
    logger.info(f"layers:           0..{mt.num_layers - 1} ({len(layers)} total)")

    logger.info("=== Caching pivot data ===")
    pivot_caches = cache_pivot_data(
        mt,
        pivot_instances,
        ds,
        layers=layers,
        precompute=not args.augment,
        cache_batch_size=CACHE_BATCH_SIZE,
        include_docstring=True,
    )

    # ------------------------------------------------------------------
    # Train head mask
    # ------------------------------------------------------------------
    logger.info("=== Training head mask ===")
    # --tag isolates a run so a lambda sweep doesn't race on one head_mask.pt.
    out_root = os.path.join(DEFAULT_RESULTS_DIR, "localization", model_short_name, args.tag)
    checkpoint_dir = os.path.join(out_root, "checkpoints")
    logger.info(f"output root: {out_root}")
    mask, history = train_head_mask(
        mt,
        pivot_caches,
        layers=layers,
        n_epochs=args.n_epochs,
        lr=LR,
        lambda_sparse=args.lambda_sparse,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=CHECKPOINT_EVERY,
        augment=args.augment,
        augment_cache_batch_size=CACHE_BATCH_SIZE,
        augment_seed=AUGMENT_SEED,
        augment_include_docstring=True,
    )

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    selected_heads = mask.selected_heads(layers)
    save_dir = out_root
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "head_mask.pt")

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
            "tag": args.tag,
            "n_instances": len(pivot_caches),
            "n_epochs": args.n_epochs,
            "lr": LR,
            "lambda_sparse": args.lambda_sparse,
            "batch_size": args.batch_size,
            "precompute": not args.augment,
            "augment": args.augment,
            "augment_seed": AUGMENT_SEED,
            "approach": "src_replacement",
        },
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved head_mask.pt → {save_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    final_kl = history.loss_causal[-1] if history.loss_causal else float("nan")

    table = Table(title="Head Mask Training Summary", highlight=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Model", args.model_key)
    table.add_row("N instances", str(len(pivot_caches)))
    table.add_row("Layers", f"0–{layers[-1]}")
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
