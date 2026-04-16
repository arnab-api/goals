"""Train the SubspaceProbe on cached source residuals.

Loads the per-task .pt files produced by cache_subspace_data.py (from
results/subspace_probe/<model_short>/cache/<source>/<task>.pt), holds out
~20 tasks for testing, and trains W_l matrices + soft-prompt token embeddings.

Cache files are loaded lazily in chunks during training: at each epoch the file
list is shuffled, then loaded in windows of --files-per-chunk files.  Entries
within each window are also shuffled before batching.  Only the layers
requested by --layer-range are kept in memory per window.

Outputs:
    results/subspace_probe/<model_short>/<run_tag>/probe.pt   — final checkpoint
    results/subspace_probe/<model_short>/<run_tag>/history.pt — training history

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
    --batch-size 16 \\
    --files-per-chunk 32
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
    p.add_argument(
        "--layer-range", type=int, nargs=2, default=[35, 46], metavar=("START", "STOP")
    )
    p.add_argument("--r", type=int, default=256, help="Subspace rank.")
    p.add_argument("--n-soft-tokens", type=int, default=3)
    p.add_argument("--lambda-ortho", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Accumulate this many instances before an optimizer step.",
    )
    p.add_argument(
        "--n-test-tasks",
        type=int,
        default=32,
        help="Number of tasks to hold out for testing.",
    )
    p.add_argument(
        "--files-per-chunk",
        type=int,
        default=32,
        metavar="N",
        help="Number of per-task cache files to load into RAM at once per epoch window.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs (default: 1). Set to 0 to disable mid-run checkpoints.",
    )
    add_logging_args(p)
    return p


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


@dataclass
class ProbeTrainingHistory:
    train_ce: list[float] = field(default_factory=list)
    train_ortho: list[float] = field(default_factory=list)
    train_total: list[float] = field(default_factory=list)
    val_ce: list[float] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)
    train_tasks: list[str] = field(default_factory=list)
    test_tasks: list[str] = field(default_factory=list)


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
    layers = list(range(args.layer_range[0], args.layer_range[1]))
    layer_set = set(layers)

    run_tag = f"r={args.r}__layer={args.layer_range[0]}_{args.layer_range[1]}"
    probe_dir = Path(DEFAULT_RESULTS_DIR) / "subspace_probe" / model_short / run_tag
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / "probe.pt"
    history_path = probe_dir / "history.pt"
    cache_dir = probe_dir.parent / "cache"

    if probe_path.exists() and not args.overwrite:
        logger.info(
            "Probe checkpoint exists at %s — pass --overwrite to retrain.", probe_path
        )
        return

    # ── Discover per-task cache files ─────────────────────────────────────────
    if not cache_dir.exists():
        logger.error(
            "Cache directory not found at %s — run scripts/cache_subspace_data.py first.",
            cache_dir,
        )
        sys.exit(1)

    all_cache_files: list[Path] = sorted(cache_dir.rglob("*.pt"))
    if not all_cache_files:
        logger.error("No .pt cache files found under %s", cache_dir)
        sys.exit(1)

    # Task name = file stem (unique across sources because source dirs don't overlap)
    all_tasks = sorted({p.stem for p in all_cache_files})
    logger.info("Found %d cache files (%d tasks) in %s", len(all_cache_files), len(all_tasks), cache_dir)

    # ── Train / test split by task ────────────────────────────────────────────
    random.shuffle(all_tasks)
    test_task_names = set(all_tasks[: args.n_test_tasks])
    train_task_names = set(all_tasks[args.n_test_tasks :])

    train_files = [p for p in all_cache_files if p.stem in train_task_names]
    test_files  = [p for p in all_cache_files if p.stem in test_task_names]
    logger.info(
        "Train: %d tasks (%d files)  |  Test: %d tasks (%d files)",
        len(train_task_names), len(train_files),
        len(test_task_names),  len(test_files),
    )

    # ── Cache loading helper ──────────────────────────────────────────────────

    def _load_chunk(file_paths: list[Path]) -> list[SubspaceCache]:
        """Load and layer-slice entries from a list of per-task .pt files."""
        entries: list[SubspaceCache] = []
        for p in file_paths:
            task_caches: list[SubspaceCache] = torch.load(
                p, map_location="cpu", weights_only=False
            )
            for sc in task_caches:
                # Keep only the layers needed for this training run
                sc.src_residuals = {l: t for l, t in sc.src_residuals.items() if l in layer_set}
            entries.extend(task_caches)
        return entries

    # Pre-load test set (small enough to hold in RAM; loaded once)
    test_caches = _load_chunk(test_files)
    logger.info("Loaded %d test entries", len(test_caches))

    # ── Load model ────────────────────────────────────────────────────────────
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(
        "Model loaded: %s  (%d layers, d_model=%d)", model_short, mt.n_layer, mt.n_embd
    )

    device = next(mt._model.parameters()).device

    # ── Build destination template ────────────────────────────────────────────
    before_soft_ids, after_soft_ids = build_dest_template_parts(mt)
    logger.info(
        "Dest template: before=%d tokens, after=%d tokens, soft=%d tokens",
        before_soft_ids.shape[0],
        after_soft_ids.shape[0],
        args.n_soft_tokens,
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
        len(layers),
        args.r,
        mt.n_embd,
        args.n_soft_tokens,
        sum(p.numel() for p in probe.parameters()),
    )

    optimizer = AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = ProbeTrainingHistory(
        train_tasks=sorted(train_task_names),
        test_tasks=sorted(test_task_names),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(
        "Starting training: %d epochs  |  %d train files  |  %d files/chunk  |  batch_size=%d",
        args.epochs,
        len(train_files),
        args.files_per_chunk,
        args.batch_size,
    )

    for epoch in range(args.epochs):
        logger.info(
            "── Epoch %d / %d ─────────────────────────────", epoch, args.epochs - 1
        )

        # Shuffle file order for this epoch
        epoch_files = list(train_files)
        random.shuffle(epoch_files)

        epoch_ce = epoch_ortho = epoch_total = 0.0
        n_steps = 0
        n_seen = 0
        n_since_step = 0
        batch_ce = batch_ortho = batch_total = 0.0

        optimizer.zero_grad()

        n_chunks = (len(epoch_files) + args.files_per_chunk - 1) // args.files_per_chunk
        for chunk_idx in range(n_chunks):
            chunk_paths = epoch_files[
                chunk_idx * args.files_per_chunk : (chunk_idx + 1) * args.files_per_chunk
            ]
            chunk_entries = _load_chunk(chunk_paths)
            random.shuffle(chunk_entries)

            is_last_chunk = chunk_idx == n_chunks - 1

            for j, sc in enumerate(chunk_entries):
                label_ids = torch.tensor(
                    mt.tokenizer.encode(sc.label, add_special_tokens=False)
                    + [mt.tokenizer.eos_token_id],
                    dtype=torch.long,
                )
                if label_ids.numel() == 0:
                    logger.debug("  skipped (empty label): task=%s", sc.task)
                    continue

                logits = patched_lm_forward(
                    mt,
                    probe,
                    before_soft_ids,
                    after_soft_ids,
                    label_ids,
                    src_residuals=sc.src_residuals,
                )
                total, ce, ortho = compute_loss(logits, label_ids, probe)
                # Gradient accumulation — normalize by batch_size so LR is scale-invariant
                (total / args.batch_size).backward()

                epoch_ce += ce.item()
                epoch_ortho += ortho.item()
                epoch_total += total.item()
                batch_ce += ce.item()
                batch_ortho += ortho.item()
                batch_total += total.item()
                n_seen += 1
                n_since_step += 1

                logger.debug(
                    "  chunk %d/%d  [%d]  task=%-20s  ce=%.4f  ortho=%.4f  total=%.4f",
                    chunk_idx + 1, n_chunks, n_seen,
                    sc.task, ce.item(), ortho.item(), total.item(),
                )

                is_last_entry = is_last_chunk and (j + 1 == len(chunk_entries))
                if n_since_step >= args.batch_size or is_last_entry:
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    optimizer.step()
                    probe.reorthogonalize()
                    optimizer.zero_grad()
                    n_steps += 1
                    b = max(n_since_step, 1)
                    logger.info(
                        "  step %4d  chunk %d/%d  [seen %d]"
                        "  batch_ce=%.4f  batch_ortho=%.4f  batch_total=%.4f",
                        n_steps, chunk_idx + 1, n_chunks, n_seen,
                        batch_ce / b, batch_ortho / b, batch_total / b,
                    )
                    batch_ce = batch_ortho = batch_total = 0.0
                    n_since_step = 0

        n = max(n_seen, 1)
        history.train_ce.append(epoch_ce / n)
        history.train_ortho.append(epoch_ortho / n)
        history.train_total.append(epoch_total / n)

        # ── Validation every 5 epochs ─────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("  Running validation on %d test instances…", len(test_caches))
            val_ce = 0.0
            n_val = 0
            with torch.no_grad():
                for sc in test_caches:
                    label_ids = torch.tensor(
                        mt.tokenizer.encode(sc.label, add_special_tokens=False)
                        + [mt.tokenizer.eos_token_id],
                        dtype=torch.long,
                    )
                    if label_ids.numel() == 0:
                        continue
                    logits = patched_lm_forward(
                        mt,
                        probe,
                        before_soft_ids,
                        after_soft_ids,
                        label_ids,
                        src_residuals=sc.src_residuals,
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits, label_ids.to(logits.device)
                    ).item()
                    logger.debug("  val  task=%-20s  ce=%.4f", sc.task, loss)
                    val_ce += loss
                    n_val += 1
            val_ce /= max(n_val, 1)
            history.val_ce.append(val_ce)
            history.val_epochs.append(epoch)
            logger.info(
                "Epoch %3d/%d  train_total=%.4f  train_ce=%.4f  train_ortho=%.4f  val_ce=%.4f"
                "  (%d steps)",
                epoch,
                args.epochs - 1,
                epoch_total / n,
                epoch_ce / n,
                epoch_ortho / n,
                val_ce,
                n_steps,
            )
        else:
            logger.info(
                "Epoch %3d/%d  train_total=%.4f  train_ce=%.4f  train_ortho=%.4f"
                "  (%d steps)",
                epoch,
                args.epochs - 1,
                epoch_total / n,
                epoch_ce / n,
                epoch_ortho / n,
                n_steps,
            )

        # ── Per-epoch checkpoint ──────────────────────────────────────────────
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            ckpt_path = probe_dir / f"probe_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "config": cfg,
                    "W": probe.W.detach().cpu(),
                    "soft_tokens": probe.soft_tokens.detach().cpu(),
                    "layers": layers,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            logger.info("  Checkpoint saved → %s", ckpt_path)

    # ── Save final checkpoint ─────────────────────────────────────────────────
    torch.save(
        {
            "config": cfg,
            "W": probe.W.detach().cpu(),
            "soft_tokens": probe.soft_tokens.detach().cpu(),
            "layers": layers,
            "epoch": args.epochs - 1,
        },
        probe_path,
    )
    torch.save(history, history_path)
    logger.info("Saved probe → %s", probe_path)
    logger.info("Saved history → %s", history_path)


if __name__ == "__main__":
    main()
