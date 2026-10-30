"""GoalScope read-outs for well-known-task generations -> data for the demo site (goals-web repo).

Reads generation JSONs written by ``scripts/generate_well_known_task_completions.py`` (default: the ``v2``
prompts in results/well-known-tasks/generations_v2/<model>/<source>/), runs ``apply_goalscope`` over the
code segment of prompt + generation (the prompt is kept as dimmed, non-scoped context), and stores the rows
with ``src.goals.site_data.write_data`` (into goals-web/site/data). Then ``make demo`` in goals-web renders the pages.

Default configuration = the paper's main GoalScope config: the 122 goal heads
(head_sets/<model>/goal__lam20_thr0.9_band25-56.json), ``additive_norm``, alpha 0.1, steer layers 35-46,
every decode step steered, 64 new tokens per read-out.

    python scripts/run_goalscope_web.py --model-key Qwen/Qwen2.5-Coder-32B-Instruct --source-suffix=-v2

Parallel: run N processes with ``--num-shards N --shard-id i`` (i = 0..N-1). Shards are
disjoint; finished tasks are skipped on restart (``--overwrite`` to redo); then build the site once.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.append(".")

import torch  # noqa: E402

from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger("run_goalscope_web")

MAX_NEW_TOKENS = 64  # new tokens per read-out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--num-shards", type=int, default=1, help="split the task list into this many shards")
    ap.add_argument("--shard-id", type=int, default=0, help="which shard this process runs (0-based)")
    ap.add_argument("--heads", default=None, help="head-set json (default: the model's goal__lam20_thr0.9_band*.json)")
    ap.add_argument("--strategy", default="additive_norm")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steer-layers", type=int, nargs=2, default=[35, 46])
    ap.add_argument("--source-suffix", default="", help="appended to the source name in the site data dir (e.g. -v2-smoke)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging_utils.configure(level=logging.INFO, force=True)

    from src.goals.head_sets import load_json_heads
    from src.goals.goalscope import apply_goalscope
    from src.models import ModelandTokenizer
    from src.goals.site_data import CODE_DATA as DATA, write_data

    model = args.model_key.split("/")[-1]
    gen_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "well-known-tasks" / "generations_v2" / model
    files = sorted(gen_dir.glob("*/*.json"))
    assert 0 <= args.shard_id < args.num_shards, "need 0 <= --shard-id < --num-shards"
    files = files[args.shard_id::args.num_shards]   # round-robin over the sorted list: disjoint, balanced shards
    heads_fp = args.heads or str(sorted(Path("head_sets", model).glob("goal__lam20_thr0.9_band*.json"))[0])
    logger.info(f"{len(files)} generations from {gen_dir} · heads {heads_fp}")

    mt = ModelandTokenizer(model_key=args.model_key, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    heads = load_json_heads(heads_fp, mt.n_layer, mt.model.config.num_attention_heads)
    for i, fp in enumerate(files, 1):
        g = json.loads(fp.read_text())
        source, task = fp.parent.name + args.source_suffix, fp.stem
        if (DATA / model / source / f"{task}.json").exists() and not args.overwrite:
            continue
        gen = g["generation"]
        if "</code>" in gen:                       # keep the closed code block, drop anything after it
            gen = gen.split("</code>", 1)[0] + "</code>"
        t0 = time.time()
        rows = apply_goalscope(g["prompt"] + gen, mt=mt, selected_heads=heads, strategy=args.strategy,
                               alpha=args.alpha, steer_layer_range=tuple(args.steer_layers), n_steer_tokens="all",
                               max_new_tokens=MAX_NEW_TOKENS, include_noncode=True, show=False)
        meta = dict(prompt=g["prompt"], generation=g["generation"], heads=heads_fp, n_heads=len(heads),
                    strategy=args.strategy, alpha=args.alpha, steer_layers=args.steer_layers,
                    max_new_tokens=MAX_NEW_TOKENS,
                    comments={k: g.get("greedy", {}).get(k) for k in ("n_hash_comments", "n_docstrings", "has_comments")})
        write_data(model, source, task, rows, meta=meta)
        logger.info(f"[{i}/{len(files)}] {source}/{task}: {sum(r['scoped'] for r in rows)} scoped tokens "
                    f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
