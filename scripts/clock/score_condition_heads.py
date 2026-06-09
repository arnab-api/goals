#!/usr/bin/env python
"""Score attention heads on the "step lookback" criterion — condition dataset.

Same ratio as score_multistep_heads, but the *key* spec tokens are no longer the
whole prompt span (with a `--key_agg` top-k heuristic to drop stopwords). Instead
they are the precomputed token ranges of each condition's `id_substrs` — the
informative spec sub-spans (e.g. "prime", "divisible 18") written by
build_condition_data.py into key_ranges.json. This removes the stopword problem
by construction.

For each cached (sample, modularity) and per (layer, head):

    For every target condition k (with a gen span) the query window is the first
    `q_span` generated tokens of k's code span (q_span=None → whole code span).
    For every condition k' with key tokens K_{k'}:

        score[k'] = mean_{s ∈ K_{k'}} Σ_{q ∈ window} attn[L,H,q,s]

    head_score = score[k] / AGG_{k'}( score[k'] )

    AGG = mean (compare_with="mean", target included) or max (compare_with="max",
    target excluded). Accumulated into a (L,H) sum/count, pooled across samples.

Reads:
    results/multistep_clock/<version>/<model>/<modularity>/<spec>/labels.json
    results/multistep_clock/<version>/<model>/<modularity>/<spec>/key_ranges.json
    results/multistep_clock/<version>/<model>/attn/<modularity>/<spec>/attn.npz

Writes (under <version>/<model>/head_scores/):
    metadata.json, per_sample/<spec>__<modularity>.npz, aggregated.npz
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample scoring.
# ──────────────────────────────────────────────────────────────────────────────


def _key_indices(step_key_ranges: list[list[int]], seq: int) -> np.ndarray:
    """Flatten a step's list of [start, end) token ranges into a sorted unique
    array of in-bounds key token indices."""
    idx: set[int] = set()
    for rng in step_key_ranges:
        if not rng:
            continue
        a, b = int(rng[0]), int(rng[1])
        idx.update(t for t in range(a, b) if 0 <= t < seq)
    return np.array(sorted(idx), dtype=np.int64)


def score_one_sample(
    attn: np.ndarray,
    labels: list[dict],
    step_key_ranges: list[list[list[int]]],
    modularity: str,
    skip_last_step_for_monolithic: bool = True,
    q_span: int | None = None,
    compare_with: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sum_ratios, count), each (n_layers, n_heads).

    `step_key_ranges[i]` = list of [start, end) token ranges (into the attn
    sequence) for condition i's id_substrs.
    """
    if compare_with not in {"mean", "max"}:
        raise ValueError(f"unknown compare_with={compare_with!r}")

    n_layers, n_heads, seq, _ = attn.shape
    sum_ratios = np.zeros((n_layers, n_heads), dtype=np.float64)
    count = np.zeros((n_layers, n_heads), dtype=np.int64)

    n = min(len(labels), len(step_key_ranges))
    key_idx = {i: _key_indices(step_key_ranges[i], seq) for i in range(n)}
    keyed = [i for i in range(n) if key_idx[i].size > 0]   # denominator candidates
    if len(keyed) < 2:
        return sum_ratios, count

    target_steps = [
        i for i in range(n)
        if labels[i].get("gen_token_range") is not None and key_idx[i].size > 0
    ]
    if skip_last_step_for_monolithic and modularity == "monolithic":
        target_steps = [i for i in target_steps if i != len(labels) - 1]
    if not target_steps:
        return sum_ratios, count

    def _spec_score(window: slice, Ki: np.ndarray) -> np.ndarray:
        # attn[:, :, window, Ki] -> sum over query window -> mean over key tokens.
        block = attn[:, :, window][:, :, :, Ki].astype(np.float32)  # (L,H,Q,|Ki|)
        return block.sum(axis=2).mean(axis=-1)                       # (L,H)

    for k in target_steps:
        q0, q1 = labels[k]["gen_token_range"]
        if q_span is not None:
            q1 = min(q1, q0 + q_span)
        if q1 <= q0:
            continue
        window = slice(q0, q1)
        scores = {i: _spec_score(window, key_idx[i]) for i in keyed}
        target_score = scores[k]
        if compare_with == "mean":
            denom = np.stack([scores[i] for i in keyed], axis=0).mean(axis=0)
        else:  # max over the *other* keyed steps (else the ratio is trivially ≤ 1)
            others = [scores[i] for i in keyed if i != k]
            denom = np.stack(others, axis=0).max(axis=0)
        mask = denom > 0
        head_score = np.zeros((n_layers, n_heads), dtype=np.float32)
        np.divide(target_score, denom, out=head_score, where=mask)
        sum_ratios += (head_score * mask).astype(np.float64)
        count += mask.astype(np.int64)

    return sum_ratios, count


# ──────────────────────────────────────────────────────────────────────────────
# IO.
# ──────────────────────────────────────────────────────────────────────────────


def cache_root(model: str, version: str) -> Path:
    return Path(env_utils.DEFAULT_RESULTS_DIR) / "multistep_clock" / version / model.split("/")[-1]


def config_tag(q_span: int | None, compare_with: str) -> str:
    """A filesystem-safe tag identifying a scoring config. The per-sample sums and
    aggregated scores both depend on (q_span, compare_with), so outputs for distinct
    configs must live under distinct subdirs (otherwise the skip-cache reuses stale
    sums computed under a different config)."""
    qs = "full" if q_span is None else str(q_span)
    return f"qspan-{qs}_cmp-{compare_with}"


def head_scores_dir(model: str, version: str, q_span: int | None, compare_with: str) -> Path:
    return cache_root(model, version) / "head_scores" / config_tag(q_span, compare_with)


def load_head_scores(
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    version: str = "conditions",
    q_span: int | None = None,
    compare_with: str = "mean",
) -> dict:
    """Load aggregated head scores for one scoring config.

    Returns a dict with `modular`/`monolithic`/`combined` score arrays ((n_layers,
    n_heads)), the raw npz, parsed `metadata`, and the resolved `dir`.
    """
    d = head_scores_dir(model, version, q_span, compare_with)
    agg_path = d / "aggregated.npz"
    if not agg_path.exists():
        raise FileNotFoundError(
            f"no aggregated scores at {agg_path} — run score_condition_heads.py with "
            f"--q_span {q_span} --compare_with {compare_with}"
        )
    agg = np.load(agg_path)
    meta = {}
    meta_path = d / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return {
        "modular": agg["modular_score"],
        "monolithic": agg["monolithic_score"],
        "combined": agg["combined_score"],
        "raw": agg,
        "metadata": meta,
        "dir": d,
    }


def load_cached(labeled_dir: Path, attn_dir: Path):
    with open(labeled_dir / "labels.json") as f:
        labels = json.load(f)["labels"]
    with open(labeled_dir / "key_ranges.json") as f:
        key_ranges = json.load(f)["step_key_token_ranges"]
    attn = np.load(attn_dir / "attn.npz", allow_pickle=True)["attn"]
    return labels, key_ranges, attn


# ──────────────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--version", default="conditions")
    p.add_argument("--modularities", nargs="+", choices=["modular", "monolithic"],
                   default=["modular", "monolithic"])
    p.add_argument("--samples", nargs="*", default=None, help="optional subset of spec ids")
    p.add_argument("--skip_last_step_for_monolithic",
                   type=lambda x: x.lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--q_span", type=int, default=None,
                   help="use only the first q_span generated tokens of each step as queries")
    p.add_argument("--compare_with", choices=["mean", "max"], default="mean")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    cache = cache_root(args.model, args.version)
    if not cache.exists():
        raise FileNotFoundError(f"cache root not found: {cache}")
    # Outputs are namespaced by scoring config so distinct (q_span, compare_with)
    # runs don't collide — the per-sample skip-cache is only valid within a config.
    score_root = head_scores_dir(args.model, args.version, args.q_span, args.compare_with)
    per_sample_root = score_root / "per_sample"
    per_sample_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"scoring config: {config_tag(args.q_span, args.compare_with)} → {score_root}")

    # Discover (spec, modularity) with labels.json + key_ranges.json + attn.npz.
    tasks = []
    for modularity in args.modularities:
        mod_root = cache / modularity
        attn_mod_root = cache / "attn" / modularity
        if not mod_root.exists() or not attn_mod_root.exists():
            logger.warning(f"missing root for {modularity} — skipping")
            continue
        for d in sorted(mod_root.iterdir()):
            if not d.is_dir():
                continue
            if args.samples is not None and d.name not in args.samples:
                continue
            attn_d = attn_mod_root / d.name
            if (d / "labels.json").exists() and (d / "key_ranges.json").exists() \
                    and (attn_d / "attn.npz").exists():
                tasks.append((d.name, modularity, d, attn_d))
    logger.info(f"discovered {len(tasks)} (spec, modularity) tasks")
    if not tasks:
        logger.warning("nothing to score — exiting")
        return

    n_layers = n_heads = None
    failed = []
    for i, (spec_id, modularity, lab_dir, attn_dir) in enumerate(tasks, start=1):
        out_path = per_sample_root / f"{spec_id}__{modularity}.npz"
        if out_path.exists() and not args.overwrite:
            logger.info(f"[{i}/{len(tasks)}] {spec_id}/{modularity} [skip cached]")
            continue
        logger.info(f"[{i}/{len(tasks)}] {spec_id}/{modularity}")
        try:
            labels, key_ranges, attn = load_cached(lab_dir, attn_dir)
            sum_ratios, count = score_one_sample(
                attn=attn, labels=labels, step_key_ranges=key_ranges, modularity=modularity,
                skip_last_step_for_monolithic=args.skip_last_step_for_monolithic,
                q_span=args.q_span, compare_with=args.compare_with,
            )
            if n_layers is None:
                n_layers, n_heads = attn.shape[:2]
            np.savez_compressed(out_path, sum_ratios=sum_ratios, count=count)
            del attn
            gc.collect()
        except Exception as e:
            logger.exception(f"  FAILED: {e}")
            failed.append((spec_id, modularity, str(e)))

    # Aggregate.
    def _aggregate(modality_filter):
        nonlocal n_layers, n_heads
        agg_sum = agg_count = None
        for p in sorted(per_sample_root.glob("*.npz")):
            if modality_filter is not None and not p.stem.endswith(f"__{modality_filter}"):
                continue
            with np.load(p) as d:
                s, c = d["sum_ratios"], d["count"]
            if agg_sum is None:
                agg_sum, agg_count = s.astype(np.float64).copy(), c.astype(np.int64).copy()
            else:
                agg_sum += s
                agg_count += c
        if agg_sum is None:
            shape = (n_layers, n_heads) if n_layers is not None else (1, 1)
            agg_sum, agg_count = np.zeros(shape), np.zeros(shape, dtype=np.int64)
        return agg_sum, agg_count

    def _safe_div(s, c):
        out = np.zeros_like(s, dtype=np.float32)
        np.divide(s, c, out=out, where=c > 0)
        return out

    mod_sum, mod_count = _aggregate("modular")
    mon_sum, mon_count = _aggregate("monolithic")
    comb_sum, comb_count = _aggregate(None)
    np.savez_compressed(
        score_root / "aggregated.npz",
        modular_sum=mod_sum, modular_count=mod_count, modular_score=_safe_div(mod_sum, mod_count),
        monolithic_sum=mon_sum, monolithic_count=mon_count, monolithic_score=_safe_div(mon_sum, mon_count),
        combined_sum=comb_sum, combined_count=comb_count, combined_score=_safe_div(comb_sum, comb_count),
    )
    with open(score_root / "metadata.json", "w") as f:
        json.dump({"model": args.model, "version": args.version,
                   "n_layers": int(n_layers) if n_layers else None,
                   "n_heads": int(n_heads) if n_heads else None,
                   "modularities": list(args.modularities), "n_tasks": len(tasks),
                   "q_span": args.q_span, "compare_with": args.compare_with,
                   "key_source": "id_substrs", "failed": failed}, f, indent=2)
    logger.info(f"saved aggregated scores → {score_root / 'aggregated.npz'}")
    if failed:
        logger.warning(f"{len(failed)} task(s) failed; see metadata.json")


if __name__ == "__main__":
    main()
