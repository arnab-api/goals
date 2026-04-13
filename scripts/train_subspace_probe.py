"""Train the SubspaceProbe on cached source residuals.

Loads the cache produced by cache_subspace_data.py, holds out ~20 tasks for
testing, and trains W_l matrices + soft-prompt token embeddings.

Outputs:
    results/subspace_probe/<model_short>/probe.pt   — checkpoint
    results/subspace_probe/<model_short>/history.pt — training history

Usage
-----
conda run -n goals python scripts/train_subspace_probe.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --layer-range 35 46 \\
    --r 256 \\
    --n-soft-tokens 4 \\
    --lambda-ortho 0.1 \\
    --lr 3e-4 \\
    --epochs 50 \\
    --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(".")

from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--layer-range", type=int, nargs=2, default=[35, 46], metavar=("START", "STOP"))
    p.add_argument("--r", type=int, default=256, help="Subspace rank.")
    p.add_argument("--n-soft-tokens", type=int, default=4)
    p.add_argument("--lambda-ortho", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Accumulate this many instances before an optimizer step.")
    p.add_argument("--n-test-tasks", type=int, default=20,
                   help="Number of tasks to hold out for testing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    add_logging_args(p)
    return p


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


@dataclass
class ProbeTrainingHistory:
    train_ce:    list[float] = field(default_factory=list)
    train_ortho: list[float] = field(default_factory=list)
    train_total: list[float] = field(default_factory=list)
    val_ce:      list[float] = field(default_factory=list)
    val_epochs:  list[int]   = field(default_factory=list)
    train_tasks: list[str]   = field(default_factory=list)
    test_tasks:  list[str]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    configure(args)
    random.seed(args.seed)

    import torch
    from torch.optim import AdamW

    from src.goals.subspace_probe import (
        SubspaceCache,
        SubspaceProbe,
        SubspaceProbeConfig,
        build_dest_template_parts,
        compute_loss,
        patched_lm_forward,
    )
    from src.models import ModelandTokenizer
    from src.utils.env_utils import DEFAULT_RESULTS_DIR

    model_short = args.model_key.split("/")[-1]
    layers      = list(range(args.layer_range[0], args.layer_range[1]))

    probe_dir = Path(DEFAULT_RESULTS_DIR) / "subspace_probe" / model_short
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path   = probe_dir / "probe.pt"
    history_path = probe_dir / "history.pt"
    cache_path   = probe_dir / "cache.pt"

    if probe_path.exists() and not args.overwrite:
        logger.info("Probe checkpoint exists at %s — pass --overwrite to retrain.", probe_path)
        return

    # ── Load cache ────────────────────────────────────────────────────────────
    if not cache_path.exists():
        logger.error(
            "Cache not found at %s — run scripts/cache_subspace_data.py first.", cache_path
        )
        sys.exit(1)

    all_caches: list[SubspaceCache] = torch.load(cache_path, map_location="cpu")
    logger.info("Loaded %d cache entries from %s", len(all_caches), cache_path)

    # ── Train / test split by task ────────────────────────────────────────────
    all_tasks = sorted({sc.task for sc in all_caches})
    random.shuffle(all_tasks)
    test_tasks  = set(all_tasks[: args.n_test_tasks])
    train_tasks = set(all_tasks[args.n_test_tasks :])

    train_caches = [sc for sc in all_caches if sc.task in train_tasks]
    test_caches  = [sc for sc in all_caches if sc.task in test_tasks]
    logger.info(
        "Train: %d tasks (%d entries)  |  Test: %d tasks (%d entries)",
        len(train_tasks), len(train_caches), len(test_tasks), len(test_caches),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info("Model loaded: %s  (%d layers, d_model=%d)", model_short, mt.n_layer, mt.n_embd)

    device = next(mt._model.parameters()).device

    # ── Build destination template ────────────────────────────────────────────
    before_soft_ids, after_soft_ids = build_dest_template_parts(mt)
    logger.info(
        "Dest template: before=%d tokens, after=%d tokens, soft=%d tokens",
        before_soft_ids.shape[0], after_soft_ids.shape[0], args.n_soft_tokens,
    )

    # ── Initialise probe ──────────────────────────────────────────────────────
    cfg = SubspaceProbeConfig(
        d_model=mt.n_embd,
        r=args.r,
        layers=layers,
        n_soft_tokens=args.n_soft_tokens,
        lambda_ortho=args.lambda_ortho,
    )
    probe = SubspaceProbe(cfg).to(device)
    probe.reorthogonalize()
    logger.info(
        "Probe: %d layers × (%d × %d) + %d soft tokens  |  total params: %d",
        len(layers), args.r, mt.n_embd, args.n_soft_tokens,
        sum(p.numel() for p in probe.parameters()),
    )

    optimizer = AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history   = ProbeTrainingHistory(
        train_tasks=sorted(train_tasks),
        test_tasks=sorted(test_tasks),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        random.shuffle(train_caches)
        epoch_ce = epoch_ortho = epoch_total = 0.0
        n_steps = 0

        optimizer.zero_grad()
        for i, sc in enumerate(train_caches):
            label_ids = torch.tensor(
                mt.tokenizer.encode(sc.label, add_special_tokens=False), dtype=torch.long
            )
            if label_ids.numel() == 0:
                continue

            logits = patched_lm_forward(
                mt, probe,
                before_soft_ids, after_soft_ids, label_ids,
                src_residuals=sc.src_residuals,
            )
            total, ce, ortho = compute_loss(logits, label_ids, probe)
            # Gradient accumulation
            (total / args.batch_size).backward()
            epoch_ce    += ce.item()
            epoch_ortho += ortho.item()
            epoch_total += total.item()

            if (i + 1) % args.batch_size == 0 or (i + 1) == len(train_caches):
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
                probe.reorthogonalize()
                optimizer.zero_grad()
                n_steps += 1

        n = max(len(train_caches), 1)
        history.train_ce.append(epoch_ce / n)
        history.train_ortho.append(epoch_ortho / n)
        history.train_total.append(epoch_total / n)

        # ── Validation every 5 epochs ─────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_ce = 0.0
            with torch.no_grad():
                for sc in test_caches:
                    label_ids = torch.tensor(
                        mt.tokenizer.encode(sc.label, add_special_tokens=False), dtype=torch.long
                    )
                    if label_ids.numel() == 0:
                        continue
                    logits = patched_lm_forward(
                        mt, probe,
                        before_soft_ids, after_soft_ids, label_ids,
                        src_residuals=sc.src_residuals,
                    )
                    val_ce += torch.nn.functional.cross_entropy(
                        logits, label_ids.to(logits.device)
                    ).item()
            val_ce /= max(len(test_caches), 1)
            history.val_ce.append(val_ce)
            history.val_epochs.append(epoch)
            logger.info(
                "Epoch %3d  train_total=%.4f  train_ce=%.4f  val_ce=%.4f",
                epoch, epoch_total / n, epoch_ce / n, val_ce,
            )
        else:
            logger.info(
                "Epoch %3d  train_total=%.4f  train_ce=%.4f",
                epoch, epoch_total / n, epoch_ce / n,
            )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save(
        {
            "config": cfg,
            "W": probe.W.detach().cpu(),
            "soft_tokens": probe.soft_tokens.detach().cpu(),
            "layers": layers,
        },
        probe_path,
    )
    torch.save(history, history_path)
    logger.info("Saved probe → %s", probe_path)
    logger.info("Saved history → %s", history_path)


if __name__ == "__main__":
    main()
