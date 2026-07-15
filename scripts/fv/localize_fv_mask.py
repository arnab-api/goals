#!/usr/bin/env python
"""Script 2 — FV head localization by learning a joint sparse head mask (goalscope-style).

Same machinery as the goalscope / completion head mask (`src.goals.head_loc_attn.train_head_mask`),
applied to the FV Relations/abstractive data instead of the pivot dataset. Instead of scoring each
head independently (Script 1 / Todd et al.), we learn ONE differentiable mask that selects the
minimal set of heads whose mean-activation transplant recovers task behavior on corrupted prompts.

Per instance (one corrupted ICL prompt):
    dest_prompt   = corrupted prompt p̃ (shuffle_labels=True)
    src_head_vals = the task mean ā^t (mean o_proj input at last token over clean prompts) — the
                    SAME general task signal Script 1 transplants, shared across a task's instances
    log_p_target  = the CLEAN prompt's last-token distribution (same demos un-shuffled + same query)
                    — the task-informed behavior the patched run should recover
    pivot_instance.target_top_token = clean argmax token (for the val recall@k metric)

train_head_mask then minimizes  KL(P_patched ‖ P_clean) + λ·mean(mask)  by projected GD, patching
the masked heads' o_proj-input slices at the last token with ā^t. The learned mask (values in
[0,1]) is saved in the same schema as the completion mask (`mask_values`, `layers`, `patch_tokens`)
so the 09.1 loader / overlap analysis can read it directly.

Usage:
    conda run -n goals python scripts/fv/localize_fv_mask.py \
        --model Qwen/Qwen2.5-Coder-32B-Instruct --tasks all --n-per-task 50 \
        --epochs 100 --lambda-sparse 20 --tag lam20
"""
import argparse
import logging
import os
import random
import sys
import time

sys.path.append(".")

import numpy as np
import torch

from src.models import ModelandTokenizer
from src.goals.head_loc_attn import PivotCache, train_head_mask
from src.goals.head_loc_data import PivotInstance
from src.tokens import prepare_input
from src.utils import env_utils

from scripts.fv.fv_common import (
    REPRESENTATIVE_TASKS,
    build_icl_prompt,
    compute_mean_head_activations,
    list_abstractive_tasks,
    load_fv_task,
)

logger = logging.getLogger(__name__)


def _resolve_tasks(spec: str) -> list[str]:
    if spec == "all":
        return list_abstractive_tasks()
    if spec == "representative":
        return REPRESENTATIVE_TASKS
    return [t.strip() for t in spec.split(",") if t.strip()]


@torch.inference_mode()
def _last_token_logits(mt, prompts: list[str], batch_size: int) -> torch.Tensor:
    """Batched plain forward → (N, vocab) float32 CPU last-token logits."""
    out = []
    for i in range(0, len(prompts), batch_size):
        inputs = prepare_input(prompts=prompts[i:i + batch_size], tokenizer=mt,
                               padding_side="left")
        logits = mt._model(**dict(inputs)).logits[:, -1, :].float().cpu()
        out.append(logits)
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--tasks", default="all",
                    help="'all' | 'representative' | comma-separated names")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--n-per-task", type=int, default=50,
                    help="corrupted/clean instances per task")
    ap.add_argument("--mean-trials", type=int, default=100, help="clean prompts per task for ā")
    ap.add_argument("--fwd-batch", type=int, default=16, help="forward batch for target dists")
    ap.add_argument("--mean-batch", type=int, default=16)
    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lambda-sparse", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # NB: do NOT globally disable grad here. train_head_mask needs ambient grad ON to build
    # the autograd graph for the mask (it freezes only the model params, not the whole graph).
    # Our own forward helpers (_last_token_logits) are already @torch.inference_mode().

    tasks = _resolve_tasks(args.tasks)
    logger.info(f"tasks ({len(tasks)}): {tasks}")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    n_layer, n_head = mt.n_layer, mt.config.num_attention_heads
    layers = list(range(n_layer))
    logger.info(f"model {args.model}: n_layer={n_layer} n_head={n_head}")

    rng = np.random.RandomState(args.seed)

    # ── build instances: (corrupted prompt, clean prompt, task) ───────────────
    mean_by_task: dict[str, torch.Tensor] = {}
    clean_prompts, corrupt_prompts, inst_tasks = [], [], []
    for t in tasks:
        task = load_fv_task(t, seed=args.seed)
        mean_by_task[t] = compute_mean_head_activations(
            mt, task, layers, n_shots=args.n_shots, n_trials=args.mean_trials,
            batch_size=args.mean_batch, seed=args.seed,
        )
        for _ in range(args.n_per_task):
            demo_idx = rng.choice(len(task.train), args.n_shots, replace=False)
            q = task.valid[int(rng.choice(len(task.valid)))]
            demos = [task.train[i] for i in demo_idx]
            r = random.Random(int(rng.randint(1 << 30)))
            clean_prompts.append(build_icl_prompt(demos, q["input"], shuffle_labels=False))
            corrupt_prompts.append(build_icl_prompt(demos, q["input"], shuffle_labels=True, rng=r))
            inst_tasks.append(t)
        logger.info(f"[{t}] +{args.n_per_task} instances (ā computed)")

    N = len(corrupt_prompts)
    logger.info(f"total instances: {N} — computing clean target distributions")

    # ── target distributions from the CLEAN prompts (task-informed behavior) ──
    clean_logits = _last_token_logits(mt, clean_prompts, args.fwd_batch)   # (N, vocab)
    log_p_target = torch.log_softmax(clean_logits, dim=-1)                 # (N, vocab)
    clean_argmax = clean_logits.argmax(-1)
    corrupt_logits = _last_token_logits(mt, corrupt_prompts, args.fwd_batch)
    corrupt_argmax = corrupt_logits.argmax(-1)

    # ── assemble FV PivotCache objects (reuse train_head_mask verbatim) ───────
    caches: list[PivotCache] = []
    for i in range(N):
        pi = PivotInstance(
            pivot_prefix="",
            dest_top_token=mt.tokenizer.decode([int(corrupt_argmax[i])]),
            target_top_token=mt.tokenizer.decode([int(clean_argmax[i])]),
        )
        caches.append(PivotCache(
            pivot_instance=pi,
            dest_prompt=corrupt_prompts[i],
            log_p_target=log_p_target[i].contiguous(),      # (vocab,)
            per_head_deltas=None,
            src_sample=None,
            dest_sample=None,
            src_prompt=None,
            src_head_vals=mean_by_task[inst_tasks[i]].contiguous(),  # (L, n_embd)
        ))

    # ── train the joint mask ──────────────────────────────────────────────────
    tag = args.tag or f"{args.tasks}_lam{args.lambda_sparse:g}"
    out_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, "fv",
                           args.model.split("/")[-1], "head_mask", tag)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"training mask: {N} caches, {args.epochs} epochs, λ={args.lambda_sparse}, "
                f"batch={args.batch_size} -> {out_dir}")
    t0 = time.time()
    mask, history = train_head_mask(
        mt, caches, layers, n_epochs=args.epochs, lr=args.lr,
        lambda_sparse=args.lambda_sparse, batch_size=args.batch_size,
        checkpoint_dir=out_dir, checkpoint_every=10,
    )
    logger.info(f"training done in {time.time()-t0:.0f}s")

    # ── save in the completion-mask schema so the 09.1 loader can read it ─────
    mask_path = os.path.join(out_dir, "mask.pt")
    torch.save({
        "mask_values": mask.values.detach().cpu(),   # (n_layer, n_head)
        "layers": layers,
        "patch_tokens": [-1],                         # FV patches the last (query-predictive) token
        "tasks": tasks,
        "model": args.model,
        "config": vars(args),
        "history": {
            "loss_total": history.loss_total, "loss_causal": history.loss_causal,
            "loss_sparse": history.loss_sparse, "n_active": history.n_active,
        },
    }, mask_path)
    logger.info(f"saved mask -> {mask_path}")

    for thr in (0.5, 0.8, 0.9):
        n_sel = int((mask.values.detach().cpu() > thr).sum())
        logger.info(f"  heads > {thr}: {n_sel}")


if __name__ == "__main__":
    main()
