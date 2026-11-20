"""Subspace localization with the head-localization objective (see src/goals/subspace_loc.py).

Same pivot pairs, target and next-token KL as scripts/train_head_mask.py; the intervention is a gated residual
subspace interchange at the destination's pivot, per layer of a band (default 40-49, where most goal heads sit).

Usage (defaults = Coder-32B, layers 40-49, r 256 ceiling, λ 20, 50 epochs, batch 64, augmentation on):
    python -m scripts.train_subspace_loc
    python -m scripts.train_subspace_loc --limit 64 --n-epochs 2 --batch-size 8 --tag smoke   # smoke

Outputs: results/localization_subspace/<model>/<tag>/subspace_mask.pt (+ checkpoints/)
"""

import argparse
import logging
import os
import sys

import torch

sys.path.append(".")

from src.goals.head_loc_attn import cache_pivot_data  # noqa: E402
from src.goals.head_loc_data import HeadLocDataset  # noqa: E402
from src.goals.subspace_loc import PRICE_UNIT, train_subspace_mask  # noqa: E402
from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402
from src.utils.logging_utils import add_logging_args, configure  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_BATCH_SIZE = 64
CHECKPOINT_EVERY = 5
AUGMENT_SEED = 123456   # = train_head_mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--layer-lo", type=int, default=40)
    ap.add_argument("--layer-hi", type=int, default=50, help="exclusive (40-50 = layers 40..49)")
    ap.add_argument("--r", type=int, default=256, help="directions per layer: a ceiling, the gates pick the rank")
    ap.add_argument("--lambda-sparse", type=float, default=20.0)
    ap.add_argument("--lr-W", type=float, default=1e-3)
    ap.add_argument("--lr-g", type=float, default=1e-2, help="= the head mask's lr")
    ap.add_argument("--n-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: use only the first N pivot instances")
    ap.add_argument("--seed", type=int, default=0, help="W initialisation")
    ap.add_argument("--tag", default=None, help="default: r<r>_lam<λ>_L<lo>-<hi>")
    add_logging_args(ap)
    args = ap.parse_args()
    configure(args)

    slug = args.model_key.split("/")[-1]
    ds = HeadLocDataset.load(model_name=slug)
    pivots = ds.pivot_instances
    if not pivots:
        raise SystemExit(f"no pivot instances for {slug} (run scripts/mine_pivot_instances.py)")
    if args.limit:
        pivots = pivots[: args.limit]
        logger.warning(f"--limit {args.limit}: SMOKE on {len(pivots)} pivot instances")

    from src.models import ModelandTokenizer
    mt = ModelandTokenizer(model_key=args.model_key, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    layers = list(range(args.layer_lo, args.layer_hi))
    caches = cache_pivot_data(mt, pivots, ds, layers=layers, precompute=False, cache_batch_size=CACHE_BATCH_SIZE,
                              include_docstring=True)

    tag = args.tag or f"r{args.r}_lam{args.lambda_sparse:g}_L{args.layer_lo}-{args.layer_hi}"
    out_root = os.path.join(DEFAULT_RESULTS_DIR, "localization_subspace", slug, tag)
    sm, history = train_subspace_mask(
        mt, caches, layers, r=args.r, n_epochs=args.n_epochs, lr_W=args.lr_W, lr_g=args.lr_g,
        lambda_sparse=args.lambda_sparse, batch_size=args.batch_size, checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_dir=os.path.join(out_root, "checkpoints"), augment=args.augment, augment_seed=AUGMENT_SEED,
        augment_cache_batch_size=CACHE_BATCH_SIZE, include_docstring=True, seed=args.seed)

    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "subspace_mask.pt")
    torch.save(dict(W=sm.W.detach().cpu(), g=sm.g.detach().cpu(), layers=layers, history=history.__dict__,
                    config=dict(model=args.model_key, r=args.r, lambda_sparse=args.lambda_sparse,
                                price_unit=PRICE_UNIT, lr_W=args.lr_W, lr_g=args.lr_g, n_epochs=args.n_epochs,
                                batch_size=args.batch_size, augment=args.augment, augment_seed=AUGMENT_SEED,
                                n_instances=len(caches), seed=args.seed)), path)
    logger.info(f"saved -> {path} | open directions per layer {sm.n_active()} (total {sum(sm.n_active())})")


if __name__ == "__main__":
    main()
