#!/usr/bin/env python
"""Score attention heads on the "step lookback" criterion.

Hypothesis: while generating code for step k, certain attention heads
preferentially look back at the *instruction tokens* that describe step k.

For each cached sample (output of build_multistep_clock_data.py) we compute,
per (layer, head):

    For every target step k with both a valid prompt_token_range and gen_token_range:
      For every generated token q in gen_token_range[k]:
        agg[k'] = sum over keys s ∈ prompt_token_range[k'] of attn[L, H, q, s]
        ratio  = agg[k] / mean_{k'} agg[k']
      We accumulate `ratio` into a (L, H) sum and a (L, H) count.

Final score per (L, H) is `sum / count` (pooled mean across samples, steps, queries).

For modularity == "monolithic" the *last* step is excluded from being a target
(its code span typically subsumes earlier steps' work). It is still kept as a
distractor in the denominator's mean.

Reads:
    results/multistep_clock/<model>/<modularity>/<sample>/labels.json
    results/multistep_clock/<model>/attn/<modularity>/<sample>/attn.npz

Writes:
    results/multistep_clock/<model>/head_scores/
      metadata.json
      per_sample/<sample>__<modularity>.npz   {sum_ratios, count}
      aggregated.npz                           {modular_sum, modular_count, modular_score,
                                                monolithic_sum, monolithic_count, monolithic_score,
                                                combined_sum, combined_count, combined_score}

Resumable: per-sample files are skipped if present (unless --overwrite). The
aggregation step always re-runs from per-sample files.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Make sure the project root is importable when called directly via
# `python scripts/clock/score_multistep_heads.py`. When invoked as
# `python -m scripts.clock.score_multistep_heads`, this is a no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample scoring.
# ──────────────────────────────────────────────────────────────────────────────


def score_one_sample(
    attn: np.ndarray,
    labels: list[dict],
    modularity: str,
    skip_last_step_for_monolithic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sum_ratios, count) of shape (n_layers, n_heads).

    sum_ratios is the running sum of per-query ratios; count is the number of
    (k, q) pairs that contributed (denominator > 0).
    """
    n_layers, n_heads, seq, _ = attn.shape
    sum_ratios = np.zeros((n_layers, n_heads), dtype=np.float64)
    count = np.zeros((n_layers, n_heads), dtype=np.int64)

    # Every step whose prompt span is locatable contributes to the denominator.
    keyed_steps = [
        (k, lbl["prompt_token_range"])
        for k, lbl in enumerate(labels)
        if lbl.get("prompt_token_range") is not None
    ]
    if len(keyed_steps) < 2:
        return sum_ratios, count  # need ≥ 2 steps for a meaningful ratio

    # Steps that can serve as a target query (have a gen span).
    target_steps = [
        (k, lbl)
        for k, lbl in enumerate(labels)
        if lbl.get("gen_token_range") is not None
        and lbl.get("prompt_token_range") is not None
    ]
    if skip_last_step_for_monolithic and modularity == "monolithic":
        last_k = len(labels) - 1
        target_steps = [(k, lbl) for k, lbl in target_steps if k != last_k]

    if not target_steps:
        return sum_ratios, count

    # Pre-aggregate attention from every query to each step's prompt span.
    # per_step_agg[i] has shape (n_layers, n_heads, seq) for keyed_steps[i].
    per_step_agg = []
    for _k, (a, b) in keyed_steps:
        if b <= a:
            per_step_agg.append(np.zeros((n_layers, n_heads, seq), dtype=np.float32))
            continue
        # cast a thin slice to float32 to avoid summing in float16
        sl = attn[:, :, :, a:b].astype(np.float32)
        per_step_agg.append(sl.sum(axis=-1))
        del sl
    stacked = np.stack(per_step_agg, axis=0)  # (n_keyed, L, H, seq)
    mean_step_agg = stacked.mean(axis=0)        # (L, H, seq)

    # Index map: original step index → row of `stacked`.
    keyed_idx_by_k = {k: i for i, (k, _) in enumerate(keyed_steps)}

    for k_tgt, lbl in target_steps:
        if k_tgt not in keyed_idx_by_k:
            continue
        q_start, q_end = lbl["gen_token_range"]
        if q_end <= q_start:
            continue
        i_tgt = keyed_idx_by_k[k_tgt]
        target_agg = stacked[i_tgt, :, :, q_start:q_end]   # (L, H, Q)
        denom = mean_step_agg[:, :, q_start:q_end]          # (L, H, Q)
        mask = denom > 0
        ratio = np.zeros_like(target_agg, dtype=np.float32)
        np.divide(target_agg, denom, out=ratio, where=mask)
        sum_ratios += (ratio * mask).sum(axis=-1).astype(np.float64)
        count += mask.sum(axis=-1).astype(np.int64)

    return sum_ratios, count


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers.
# ──────────────────────────────────────────────────────────────────────────────


def cache_root(model: str) -> Path:
    return (
        Path(env_utils.DEFAULT_RESULTS_DIR)
        / "multistep_clock"
        / model.split("/")[-1]
    )


def load_cached(labeled_sample_dir: Path, attn_sample_dir: Path) -> tuple[dict, np.ndarray]:
    """Return (labels_dict, attn_array). Caller is responsible for `del attn` after."""
    with open(labeled_sample_dir / "labels.json") as f:
        labels = json.load(f)
    npz = np.load(attn_sample_dir / "attn.npz", allow_pickle=True)
    attn = npz["attn"]  # float16
    return labels, attn


# ──────────────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument(
        "--modularities",
        nargs="+",
        choices=["modular", "monolithic"],
        default=["modular", "monolithic"],
    )
    p.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="optional subset of sample names; default = all under the cache root",
    )
    p.add_argument(
        "--skip_last_step_for_monolithic",
        type=lambda x: x.lower() in {"1", "true", "yes"},
        default=True,
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    cache = cache_root(args.model)
    if not cache.exists():
        raise FileNotFoundError(f"cache root not found: {cache}")

    score_root = cache / "head_scores"
    per_sample_root = score_root / "per_sample"
    per_sample_root.mkdir(parents=True, exist_ok=True)

    # Discover (sample, modularity) pairs from the on-disk cache. We require
    # *both* a labels.json (under <model>/<mod>/<sample>/) AND an attn.npz
    # (under <model>/attn/<mod>/<sample>/).
    tasks: list[tuple[str, str, Path, Path]] = []
    for modularity in args.modularities:
        mod_root = cache / modularity
        attn_mod_root = cache / "attn" / modularity
        if not mod_root.exists():
            logger.warning(f"missing labels root {mod_root} — skipping")
            continue
        if not attn_mod_root.exists():
            logger.warning(f"missing attn root {attn_mod_root} — skipping")
            continue
        for labeled_sample_dir in sorted(mod_root.iterdir()):
            if not labeled_sample_dir.is_dir():
                continue
            name = labeled_sample_dir.name
            if args.samples is not None and name not in args.samples:
                continue
            attn_sample_dir = attn_mod_root / name
            if not (labeled_sample_dir / "labels.json").exists():
                continue
            if not (attn_sample_dir / "attn.npz").exists():
                continue
            tasks.append((name, modularity, labeled_sample_dir, attn_sample_dir))

    logger.info(f"discovered {len(tasks)} (sample, modularity) tasks")
    if not tasks:
        logger.warning("nothing to score — exiting")
        return

    # Process tasks.
    n_layers = None
    n_heads = None
    failed: list[tuple[str, str, str]] = []
    for i, (sample_name, modularity, labeled_sample_dir, attn_sample_dir) in enumerate(tasks, start=1):
        out_path = per_sample_root / f"{sample_name}__{modularity}.npz"
        if out_path.exists() and not args.overwrite:
            logger.info(f"[{i}/{len(tasks)}] {sample_name}/{modularity} [skip cached]")
            continue
        logger.info(f"[{i}/{len(tasks)}] {sample_name}/{modularity}")
        try:
            labels_dict, attn = load_cached(labeled_sample_dir, attn_sample_dir)
            sum_ratios, count = score_one_sample(
                attn=attn,
                labels=labels_dict["labels"],
                modularity=modularity,
                skip_last_step_for_monolithic=args.skip_last_step_for_monolithic,
            )
            if n_layers is None:
                n_layers, n_heads = attn.shape[:2]
            np.savez_compressed(out_path, sum_ratios=sum_ratios, count=count)
            del attn, labels_dict
            gc.collect()
        except Exception as e:
            logger.exception(f"  FAILED: {e}")
            failed.append((sample_name, modularity, str(e)))

    # Aggregate per-sample files into modular / monolithic / combined.
    logger.info("aggregating per-sample scores")

    def _aggregate(modality_filter: str | None) -> tuple[np.ndarray, np.ndarray]:
        nonlocal n_layers, n_heads
        agg_sum: np.ndarray | None = None
        agg_count: np.ndarray | None = None
        for p in sorted(per_sample_root.glob("*.npz")):
            if modality_filter is not None and not p.stem.endswith(f"__{modality_filter}"):
                continue
            with np.load(p) as d:
                s = d["sum_ratios"]
                c = d["count"]
            if agg_sum is None:
                agg_sum = s.astype(np.float64).copy()
                agg_count = c.astype(np.int64).copy()
            else:
                agg_sum += s
                agg_count += c
        if agg_sum is None:
            shape = (n_layers, n_heads) if n_layers is not None else (1, 1)
            agg_sum = np.zeros(shape, dtype=np.float64)
            agg_count = np.zeros(shape, dtype=np.int64)
        return agg_sum, agg_count

    mod_sum, mod_count = _aggregate("modular")
    mon_sum, mon_count = _aggregate("monolithic")
    comb_sum, comb_count = _aggregate(None)

    def _safe_div(s: np.ndarray, c: np.ndarray) -> np.ndarray:
        out = np.zeros_like(s, dtype=np.float32)
        np.divide(s, c, out=out, where=c > 0)
        return out

    aggregated_path = score_root / "aggregated.npz"
    np.savez_compressed(
        aggregated_path,
        modular_sum=mod_sum,
        modular_count=mod_count,
        modular_score=_safe_div(mod_sum, mod_count),
        monolithic_sum=mon_sum,
        monolithic_count=mon_count,
        monolithic_score=_safe_div(mon_sum, mon_count),
        combined_sum=comb_sum,
        combined_count=comb_count,
        combined_score=_safe_div(comb_sum, comb_count),
    )
    logger.info(f"saved aggregated scores → {aggregated_path}")

    # Metadata.
    metadata = {
        "model": args.model,
        "n_layers": int(n_layers) if n_layers is not None else None,
        "n_heads": int(n_heads) if n_heads is not None else None,
        "modularities": list(args.modularities),
        "n_tasks": len(tasks),
        "n_per_sample_files": len(list(per_sample_root.glob("*.npz"))),
        "skip_last_step_for_monolithic": args.skip_last_step_for_monolithic,
        "failed": failed,
    }
    with open(score_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"wrote metadata → {score_root / 'metadata.json'}")

    if failed:
        logger.warning(f"{len(failed)} task(s) failed; see metadata.json")


if __name__ == "__main__":
    main()
