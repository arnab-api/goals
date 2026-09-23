#!/usr/bin/env python
"""Temporal persistence of the goal encoding — across datatypes AND languages (Python + C++).

For each (datatype, language) it rolls out full multi-predicate functions, segments each sub-goal's
implementation with an oracle LLM (robust to multi-statement C++ blocks; rule-based Python fast path via
``--no-oracle``), and records the goal-head encoding at the `=` pivot + along the implementation tokens.
Retention/stability curves are cos(pivot, encoding at x% of the impl) over relation masks — including the
cross-language "same sub-goal, different context" comparison.

Per-condition records are cached under ``<results>/temporal_persistence/<model_slug>/`` (shared with the
notebook via ``src.goals.temporal_persistence``), so contexts accumulate per sub-task and never recompute.
One model per process (``--cuda`` pins the GPU) → several models can run in parallel.

Usage:
    conda run -n goals python scripts/evaluation/temporal_persistence_eval.py --cuda 0
    conda run -n goals python scripts/evaluation/temporal_persistence_eval.py --cuda 0 \\
        --datatypes int str --langs python cpp --n-trials 48
    # wiring check (no model, no oracle):
    conda run -n goals python scripts/evaluation/temporal_persistence_eval.py --dry-run
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(".")


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda  # must precede torch import

import src.diff.data as D  # noqa: E402
from src.goals import temporal_persistence as TP  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger("temporal_persistence")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--datatypes", nargs="+", default=["int", "str", "list[int]", "list[str]"])
    ap.add_argument("--langs", nargs="+", default=TP.LANGS, choices=TP.LANGS)
    ap.add_argument("--head-budget", type=int, default=122, choices=[122, 250])
    ap.add_argument("--n-trials", type=int, default=64, help="contexts sampled per (datatype, lang)")
    ap.add_argument("--n-cond", type=int, nargs="+", default=[3, 4], help="task sizes to mix over")
    ap.add_argument("--seed", type=int, default=1123)
    ap.add_argument("--min-impl", type=int, default=4, help="drop conditions with < this many impl tokens")
    ap.add_argument("--max-new", type=int, default=96, help="generation budget (tokens) per probe")
    ap.add_argument("--prog-grid", type=float, nargs="+", default=TP.DEFAULT_PROG_GRID)
    ap.add_argument("--oracle-model", default="claude-sonnet-4-6")
    ap.add_argument("--no-oracle", action="store_true", help="rule-based python segmentation (no oracle)")
    ap.add_argument("--control", action="store_true",
                    help="use a matched-count CONTROL head set (later-half of the LM, non-goal) instead of "
                         "the goal heads; records go to a 'control/' sub-dir (head_tag='control')")
    ap.add_argument("--control-seed", type=int, default=7,
                    help="RNG seed for sampling the control head set")
    ap.add_argument("--force", action="store_true", help="ignore cache; recompute every condition")
    ap.add_argument("--dry-run", action="store_true", help="build corpus + prompts only; NO model/oracle")
    logging_utils.add_logging_args(ap)
    return ap


def _fmt_curves(curves, prog_grid):
    lines = []
    header = "comparison".ljust(38) + "n".rjust(7) + "".join(f"{int(p*100):>8d}%" for p in prog_grid)
    lines.append(header)
    for name, cur in curves.items():
        n0 = cur[prog_grid[0]]["n"]
        cells = "".join(f"{cur[p]['mean']:>+9.3f}" for p in prog_grid)
        lines.append(name.ljust(38) + str(n0).rjust(7) + cells)
    return "\n".join(lines)


def main():
    args = parse_args().parse_args()
    logging_utils.configure(args) if hasattr(logging_utils, "configure") else logging.basicConfig(level=logging.INFO)
    experiment_utils.set_seed(args.seed)

    slug = args.model.split("/")[-1]
    out_dir = TP.cache_dir(slug)
    os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        logger.info(f"[dry-run] datatypes={args.datatypes} langs={args.langs} "
                    f"budget={args.head_budget} n_trials={args.n_trials} n_cond={args.n_cond}")
        for dt in args.datatypes:
            for lang in args.langs:
                setups = D.multi_predicate_setups(
                    dt, n_conditions=list(args.n_cond), n_samples=args.n_trials, seed=args.seed,
                    naming="letters", src_lang=lang, dst_lang=lang,
                )
                ex = setups[0]
                preds = ex._preds_for(ex.src_frontier)
                logger.info(f"  {dt:10s}/{lang:6s}: {len(setups)} contexts · "
                            f"example sub-goals={[p.id for p in preds[1:]]}")
        logger.info(f"[dry-run] would write to {out_dir}")
        return

    import torch
    from src.models import ModelandTokenizer
    # When --cuda pins a single visible GPU, force the whole model onto it ({"":0}) instead of
    # device_map="auto". "auto" leaves headroom and offloads the LAST layers to CPU/meta on a tight
    # GPU; reading an offloaded head's o_proj.weight OUTSIDE a forward pass then raises
    # "Cannot copy out of meta tensor". The control heads live in the later half (layers 32-63), so
    # they hit exactly those offloaded layers. Weight placement doesn't affect the numeric result.
    single_gpu = args.cuda is not None and "," not in str(args.cuda)
    device_map = {"": 0} if single_gpu else "auto"
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map=device_map, attn_implementation="eager")
    goal_heads = TP.load_goal_heads(mt, slug, args.head_budget)
    head_tag = "goal"
    heads = goal_heads
    if args.control:
        import random as _random
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from ablation_common import random_heads_like
        rng = _random.Random(args.control_seed)
        heads = random_heads_like(
            ablated=goal_heads, from_layer=mt.n_layer // 2,
            n_layer=mt.n_layer, n_heads=mt.n_heads, exclude=goal_heads, rng=rng,
        )
        head_tag = "control"
        logger.info(f"{slug}: CONTROL head set — {len(heads)} heads sampled from layers "
                    f"[{mt.n_layer // 2}, {mt.n_layer}) (seed={args.control_seed}), disjoint from goal heads")
    logger.info(f"{slug}: {len(heads)} {head_tag} heads (budget={args.head_budget}) -> {out_dir}")

    all_records = []
    per_cell = {}
    for dt in args.datatypes:
        for lang in args.langs:
            recs = TP.build_records(
                mt, slug, heads, dt, lang,
                budget=args.head_budget, n_trials=args.n_trials, n_cond=tuple(args.n_cond),
                seed=args.seed, min_impl=args.min_impl, max_new=args.max_new,
                use_oracle=not args.no_oracle, oracle_model=args.oracle_model, force=args.force,
                head_tag=head_tag,
            )
            per_cell[f"{dt}/{lang}"] = len(recs)
            all_records.extend(recs)
            logger.info(f"  {dt:10s}/{lang:6s}: {len(recs)} conditions")

    if not all_records:
        logger.warning("no conditions built — aborting summary")
        return

    pooled = TP.compute_curves(all_records, args.prog_grid)
    by_cell = {
        cell: TP.compute_curves([r for r in all_records if f"{r['datatype']}/{r['lang']}" == cell],
                                args.prog_grid)
        for cell in per_cell
    }
    summ = dict(
        model=args.model, budget=args.head_budget, head_tag=head_tag,
        n_conditions_total=len(all_records),
        per_cell_counts=per_cell, prog_grid=args.prog_grid, pooled=pooled, by_cell=by_cell,
    )
    summ_name = "summary.json" if head_tag == "goal" else f"summary_{head_tag}.json"
    with open(os.path.join(out_dir, summ_name), "w") as f:
        json.dump(summ, f, indent=2)

    logger.info(f"=== {slug} · pooled retention curves (n_conditions={len(all_records)}) ===")
    for line in _fmt_curves(pooled, args.prog_grid).split("\n"):
        logger.info(line)
    logger.info(f"per-cell counts: {per_cell}")
    logger.info(f"summary -> {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
