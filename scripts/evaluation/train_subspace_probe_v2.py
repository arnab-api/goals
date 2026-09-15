#!/usr/bin/env python
"""Train the Subspace Probe v2 by KL self-distillation (see design doc §5).

Per example (source instance × pivot): read the frontier residual ``h`` at ``--read_layer`` of the
structured SOURCE prompt, form the read-out ``o = Wᵀ W h``, and steer it (additive_norm, α) into an
augmented C++ causal stub while teacher-forcing the cached teacher trajectory ``y``. The loss matches
the steered STUDENT's per-token distribution to the goal-informed TEACHER's (recomputed on the fly,
full vocab): ``L = mean_t KL(p_t ‖ q_t) + λ·‖WWᵀ − I‖²``. Only ``W`` is trained.

Usage (14B smoke, cuda 0,1):
    conda run -n goals python scripts/evaluation/train_subspace_probe_v2.py \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --cuda 0,1 \
        --targets data/subspace_v2/teacher_targets/Qwen2.5-Coder-14B-Instruct/train.json \
        --epochs 3 --r 256
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(".")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_c = _pre.parse_known_args()[0].cuda
if _c is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_c)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from src.goals.subspace_probe import cache_source_residuals  # noqa: E402  (generic residual cache)
from src.goals.subspace_probe_v2 import (  # noqa: E402
    SubspaceProbeV2,
    SubspaceV2Config,
    build_causal_chat,
    distill_kl,
    steered_student_logits,
    teacher_logits,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def default_band(n_layer):
    """Read layer + steer band scaled from the 32B reference (read L50, band 35-46 on 64 layers)."""
    read = round(0.78 * n_layer)
    lo, hi = round(0.547 * n_layer), round(0.719 * n_layer)
    return read, list(range(lo, hi + 1))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--targets", required=True, help="teacher-target json from gen_teacher_targets_v2.py")
    ap.add_argument("--r", type=int, default=256)
    ap.add_argument("--read_layer", type=int, default=None, help="default: scaled from n_layer (~0.78·L)")
    ap.add_argument("--steer_lo", type=int, default=None)
    ap.add_argument("--steer_hi", type=int, default=None, help="inclusive")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lambda_ortho", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default=None)
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    slug = args.model.split("/")[-1]

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    mt._model.eval()
    for p in mt._model.parameters():  # freeze the LM — only W is trained
        p.requires_grad_(False)

    read_layer, band = default_band(mt.n_layer)
    if args.read_layer is not None:
        read_layer = args.read_layer
    if args.steer_lo is not None and args.steer_hi is not None:
        band = list(range(args.steer_lo, args.steer_hi + 1))
    logger.info(f"n_layer={mt.n_layer} | read_layer=L{read_layer} | steer band={band[0]}-{band[-1]}")

    data = json.loads(Path(args.targets).read_text())
    examples = data["examples"]
    logger.info(f"targets: {args.targets} | {len(examples)} source instances | pivots={data['pivots']}")

    # ── pre-cache source residuals @ read_layer (once, no grad) → one training row per (example,pivot)
    rows = []  # (h_src(d,), stub, goal_desc, y_ids)
    with torch.no_grad():
        for ex in examples:
            full_text = ex["source_prompt"] + ex["source_completion"]
            positions = sorted(set(ex["pivots"].values()))
            res = cache_source_residuals(mt, full_text, positions, [read_layer])  # {pos:{layer:(d,)}}
            for pivot, pos in ex["pivots"].items():
                rows.append(dict(
                    h=res[pos][read_layer].clone(),  # (d,) float32 CPU
                    stub=ex["stub"], goal_desc=ex["goal_desc"],
                    y_ids=torch.tensor(ex["y_ids"], dtype=torch.long),
                    tag=f"{ex['datatype']}/{ex['frontier_id']}/{pivot}",
                ))
    logger.info(f"cached {len(rows)} training rows (example × pivot) @ L{read_layer}")

    cfg = SubspaceV2Config(d_model=mt.n_embd, r=args.r, read_layer=read_layer,
                           steer_layers=band, alpha=args.alpha, lambda_ortho=args.lambda_ortho)
    probe = SubspaceProbeV2(cfg).to(mt.device)
    opt = torch.optim.AdamW([probe.W], lr=args.lr)

    out = Path(args.out) if args.out else (
        Path(env_utils.DEFAULT_RESULTS_DIR) / "subspace_probe_v2" / slug
        / f"r{args.r}__L{read_layer}__a{args.alpha}" / "probe.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    import random as _rnd
    rng = _rnd.Random(0)
    for epoch in range(args.epochs):
        rng.shuffle(rows)
        tot_kl = tot_ortho = 0.0
        t0 = time.time()
        for i, row in enumerate(rows):
            # teacher + student prefixes (share the SAME stub + y; teacher sees the goal, student doesn't)
            t_prefix = prepare_input(prompts=build_causal_chat(mt, row["stub"], goal_desc=row["goal_desc"]),
                                     tokenizer=mt)["input_ids"][0]
            s_prefix = prepare_input(prompts=build_causal_chat(mt, row["stub"], goal_desc=None),
                                     tokenizer=mt)["input_ids"][0]
            y = row["y_ids"]
            tl = teacher_logits(mt, t_prefix, y)                       # (T,V) no grad
            sl = steered_student_logits(mt, probe, s_prefix, y, row["h"])  # (T,V) grad via W
            kl = distill_kl(tl, sl)
            ortho = probe.orthogonality_loss()
            loss = kl.to(ortho.device) + cfg.lambda_ortho * ortho  # kl may be on a different GPU
            opt.zero_grad()
            loss.backward()
            opt.step()
            probe.reorthogonalize()
            tot_kl += float(kl); tot_ortho += float(ortho)
        logger.info(f"epoch {epoch}: KL={tot_kl/len(rows):.4f} ortho={tot_ortho/len(rows):.2e} "
                    f"({time.time()-t0:.1f}s)")
        torch.save(dict(W=probe.W.detach().cpu(), config=cfg, epoch=epoch,
                        read_layer=read_layer, steer_layers=band, model=args.model), out)
    logger.info(f"saved -> {out}")


if __name__ == "__main__":
    main()
