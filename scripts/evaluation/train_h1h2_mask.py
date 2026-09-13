#!/usr/bin/env python
"""Train the H1-vs-H2 query mask (see agent_reports/claude/h1h2_mask_design.md).

For a datatype, sample N contrastive pairs (src.diff.h1h2), cache them at L_READ under the chosen
intervention, and train a sparse query mask (layers < L_READ) that pushes the collapsed goal-head gaze
onto ONE hypothesis's target span. Run per (hypothesis x intervention) and compare: which hypothesis's
target is achievable, how sparse the mask, which heads/layers it selects.

    hypothesis: copy (->D) | h1 (->B) | h2 (->E) | noop (->C)
    intervention: free_form | mlps_frozen | block_head_communication

Usage:
    conda run -n goals python scripts/evaluation/train_h1h2_mask.py \
        --cuda 2,3,6,7 --datatype list[str] --n_pairs 32 --l_read 40 \
        --hypotheses h1 h2 copy --intervention block_head_communication

Results -> <results>/evaluation/h1h2_mask/<model>/L<l_read>/<datatype>/<intervention>/<hypothesis>/
           mask.pt (mask_values + layers + selected_heads + history + config) and epochs/mask_epochNN.pt
"""
import argparse
import logging
import os
import sys
import time

sys.path.append(".")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_c = _pre.parse_known_args()[0].cuda
if _c is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_c)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# NB: CUDA_VISIBLE_DEVICES is set above BEFORE any torch import; the transitive torch import inside
# ``src.diff.functional`` below is the first, so the device pin holds.
import torch  # noqa: E402
import src.diff.data as DATA  # noqa: E402
import src.diff.functional as FUNC  # noqa: E402
import src.diff.h1h2 as H  # noqa: E402
import src.diff.h1h2_mask as M  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--datatype", default="list[str]", choices=list(DATA.DATATYPES))
    ap.add_argument("--lang", default="python", choices=["python", "cpp"])
    ap.add_argument("--n_pairs", type=int, default=32)
    ap.add_argument("--l_read", type=int, default=40,
                    help="single knob: maskable heads at layers <L_READ; collapse reads h^L_READ")
    ap.add_argument("--stub", default="var_eq", choices=["var_eq", "indent"])
    ap.add_argument("--patch_last_k", type=int, default=1,
                    help="how many trailing query positions to patch (1=pivot ' =' only; 3=[indent,var,' ='])")
    ap.add_argument("--src_targ_in_dst", action="store_true", default=True)
    ap.add_argument("--no_src_targ_in_dst", dest="src_targ_in_dst", action="store_false")
    ap.add_argument("--hypotheses", nargs="+", default=["h1", "h2", "copy"],
                    choices=list(M.HYPOTHESIS_ROLE),
                    help="hypotheses to train (all share one cache, so cheap to run together)")
    ap.add_argument("--intervention", default="block_head_communication",
                    choices=list(M.INTERVENTIONS),
                    help="one intervention per run (each has its own cache; shard across jobs)")
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda_sparse", type=float, default=1e-2)
    ap.add_argument("--lambda_sweep", type=float, nargs="+", default=None,
                    help="λ sweep: build the cache once, train a mask per λ (writes .../lam<λ>/<hyp>/)")
    ap.add_argument("--comp_weight", type=float, default=1.0,
                    help="explicit competing-span penalty weight in the gaze loss")
    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=5,
                    help="write a mask checkpoint every K epochs (0 = disabled)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cache_dir", default=None,
                    help="dir for the per-pair caches (default <results>/notebook_caches/h1h2_mask)")
    ap.add_argument("--overwrite_cache", action="store_true")
    ap.add_argument("--out", default=None)
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    # free_form / mlps_frozen do a grad-enabled TRUNCATED forward; under device_map="auto" the whole
    # backward graph piles on GPU0 (it holds layers 0..~N/2, all computed) and OOMs while GPU1 idles.
    # A balanced map splits the computed layers 0..l_read-1 evenly across GPUs. #3 does no grad forward.
    # NB: count GPUs from CUDA_VISIBLE_DEVICES, not torch.cuda.device_count() — under the dispatch
    # launch (screen + conda activate) device_count() came back <2 and silently skipped the balance.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    n_gpu = len([x for x in cvd.split(",") if x.strip()]) if cvd else torch.cuda.device_count()
    dmap = "auto"
    if args.intervention in ("free_form", "mlps_frozen") and n_gpu >= 2:
        dmap = M.build_balanced_device_map(args.model, args.l_read, n_gpu=n_gpu)
        print(f"[device_map] balanced over {n_gpu} GPUs (computed layers 0-{args.l_read-1} split evenly)",
              flush=True)
    FUNC.setup(args.model, device_map=dmap)      # loads model + goal-head mask; binds FUNC globals
    mt = FUNC.mt
    slug = args.model.split("/")[-1]
    out_root = args.out or os.path.join(env_utils.DEFAULT_RESULTS_DIR, "evaluation", "h1h2_mask",
                                        slug, f"L{args.l_read}", args.datatype.replace("[", "").replace("]", ""))

    specs = H.sample_pairs(args.datatype, args.n_pairs, lang=args.lang, seed=args.seed)
    cache_dir = args.cache_dir or os.path.join(env_utils.DEFAULT_RESULTS_DIR, "notebook_caches", "h1h2_mask")
    interv = args.intervention
    for k in ("model", "datatype", "lang", "n_pairs", "l_read", "intervention", "hypotheses",
              "n_epochs", "lr", "lambda_sparse", "comp_weight", "checkpoint_every", "batch_size", "seed", "patch_last_k"):
        logger.info(f"{k:16s}: {getattr(args, k)}")
    logger.info(f"goal heads      : {len(FUNC.GOAL_HEADS)}")

    # cache once (deltas / MLP-freeze are intervention-specific), reuse across hypotheses
    t0 = time.time()
    caches = M.load_or_build_caches(
        mt, specs, args.l_read, interv, H.build_prompts, cache_dir,
        datatype=args.datatype, lang=args.lang, seed=args.seed,
        stub=args.stub, src_targ_in_dst=args.src_targ_in_dst, overwrite=args.overwrite_cache,
        patch_last_k=args.patch_last_k)
    logger.info(f"[{interv}] {len(caches)} caches ready in {time.time()-t0:.0f}s")

    lambdas = args.lambda_sweep or [args.lambda_sparse]
    sweeping = args.lambda_sweep is not None
    for lam in lambdas:
        for hyp in args.hypotheses:
            out_dir = os.path.join(out_root, interv, f"lam{lam}", hyp) if sweeping \
                else os.path.join(out_root, interv, hyp)
            os.makedirs(out_dir, exist_ok=True)
            config = {"model": slug, "datatype": args.datatype, "lang": args.lang, "l_read": args.l_read,
                      "intervention": interv, "hypothesis": hyp, "n_pairs": len(specs),
                      "n_epochs": args.n_epochs, "lr": args.lr, "lambda_sparse": lam,
                      "comp_weight": args.comp_weight, "seed": args.seed, "patch_last_k": args.patch_last_k}
            # train_query_mask owns checkpointing: writes epochs/mask_epochNN.pt + mask.pt under out_dir
            mask, hist = M.train_query_mask(
                mt, caches, args.l_read, hypothesis=hyp, intervention=interv,
                n_epochs=args.n_epochs, lr=args.lr, lambda_sparse=lam,
                comp_weight=args.comp_weight, batch_size=args.batch_size, seed=args.seed,
                checkpoint_dir=out_dir, checkpoint_every=args.checkpoint_every, config=config)
            n_sel = len(sorted(mask.to_full_depth(list(range(args.l_read)), mt.n_layer)
                               .selected_heads(list(range(mt.n_layer)), threshold=0.5)))
            sp, tot = hist["sparse"][-1], hist["loss"][-1]
            logger.info(f"[{interv}/lam{lam}/{hyp}] loss={tot:.4f} task={hist['task'][-1]:.4f} "
                        f"sparse={sp:.4f} sparse/total={sp/max(tot,1e-9):.2f} "
                        f"mass={hist['mass'][-1]:.3f} selected={n_sel} -> {out_dir}/mask.pt")

    print(f"\ndone -> {out_root}/{interv}")


if __name__ == "__main__":
    main()
