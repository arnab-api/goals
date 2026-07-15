#!/usr/bin/env python
"""Script 1 — FV head localization via per-head causal mediation analysis (Todd et al. 2024).

Replicates the AIE computation of `function_vectors/src/compute_indirect_effect.py` in this
codebase, on Qwen2.5-Coder-32B-Instruct, so the resulting FV heads can be compared head-for-head
against the goal / goalscope / gaze heads.

Method (one attention head at a time):
  1. For each task t, compute the mean head activation ā_ℓj^t = mean o_proj input at the last
     token over `--mean-trials` CLEAN N-shot ICL prompts (fv_common.compute_mean_head_activations).
  2. Build `--n-trials` CORRUPTED prompts p̃ (shuffle_labels=True) per task, each with a real query
     whose true first answer-token id we track.
  3. Baseline: p̃'s probability of the target first token, unpatched.
  4. For every (layer ℓ, head j): replace ONLY that head's o_proj-input slice at the last token
     with ā_ℓj^t (alpha=1 full replace, mask=1 on the single head) via forward_with_src_replacement,
     and measure CIE = f(p̃ | a_ℓj:=ā)[y_q] − f(p̃)[y_q] on the target's FIRST token.
  5. AIE(ℓ,j) = mean CIE over trials within a task, then equal-weighted mean over tasks (Eq 4).

Only layer ℓ is hooked per head, so downstream layers propagate the perturbation naturally — exact
CMA, 64× fewer hooks than masking all layers. Corrupted prompts are batched across tasks (each row
carries its own task mean and target id), so one forward per (ℓ,j) covers a whole chunk.

The AIE tensor is saved to results/fv/<model>/aie_<tag>.pt for threshold/top-K selection in a
notebook. Nothing is thresholded here.

Usage:
    conda run -n goals python scripts/fv/localize_fv_cma.py \
        --model Qwen/Qwen2.5-Coder-32B-Instruct --tasks all --n-trials 25 --mean-trials 100
"""
import argparse
import os
import random
import sys
import time

sys.path.append(".")

import numpy as np
import torch
from tqdm import tqdm

from src.models import ModelandTokenizer
from src.goals.head_loc_attn import HeadMask, forward_with_src_replacement
from src.tokens import prepare_input
from src.utils import env_utils
import logging

from scripts.fv.fv_common import (
    REPRESENTATIVE_TASKS,
    build_icl_prompt,
    compute_mean_head_activations,
    first_answer_token_id,
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
def _last_token_probs(mt, inputs, token_ids: torch.Tensor) -> torch.Tensor:
    """Plain (unpatched) forward → softmax at last token → gather `token_ids` (batch,)."""
    out = mt._model(**dict(inputs))
    probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)  # (b, vocab)
    return probs.gather(1, token_ids.view(-1, 1).to(probs.device)).squeeze(1).cpu()


@torch.inference_mode()
def _patched_probs(mt, inputs, src_layer, mask, layer, token_ids) -> torch.Tensor:
    """Single-head patched forward at last token → p(target first token) per row (batch,)."""
    logits = forward_with_src_replacement(
        mt, inputs, src_layer, mask, layers=[layer], alpha=1.0, patch_positions=[-1],
    )  # (b, vocab)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.gather(1, token_ids.view(-1, 1).to(probs.device)).squeeze(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--tasks", default="all",
                    help="'all' | 'representative' | comma-separated task names")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=25,
                    help="corrupted prompts per task (CMA trials)")
    ap.add_argument("--mean-trials", type=int, default=100,
                    help="clean prompts per task for ā")
    ap.add_argument("--batch", type=int, default=32, help="corrupted rows per forward chunk")
    ap.add_argument("--mean-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="output filename tag (default: derived)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_grad_enabled(False)

    tasks = _resolve_tasks(args.tasks)
    logger.info(f"tasks ({len(tasks)}): {tasks}")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    n_layer, n_head = mt.n_layer, mt.config.num_attention_heads
    layers = list(range(n_layer))
    logger.info(f"model {args.model}: n_layer={n_layer} n_head={n_head}")

    # ── per-task: mean ā + corrupted rows (prompt, target-first-token id) ──────
    rng = np.random.RandomState(args.seed)
    mean_by_task: dict[str, torch.Tensor] = {}
    rows: list[dict] = []  # {task, prompt, target_id}
    for t in tasks:
        task = load_fv_task(t, seed=args.seed)
        logger.info(f"[{t}] train={len(task.train)} valid={len(task.valid)} — computing ā")
        mean_by_task[t] = compute_mean_head_activations(
            mt, task, layers, n_shots=args.n_shots, n_trials=args.mean_trials,
            batch_size=args.mean_batch, seed=args.seed,
        )  # (L, n_embd)
        n_added = 0
        for _ in range(args.n_trials):
            demo_idx = rng.choice(len(task.train), args.n_shots, replace=False)
            q = task.valid[int(rng.choice(len(task.valid)))]
            demos = [task.train[i] for i in demo_idx]
            prompt = build_icl_prompt(demos, q["input"], shuffle_labels=True,
                                      rng=random.Random(int(rng.randint(1 << 30))))
            try:
                tid = first_answer_token_id(mt, prompt, q["output"])
            except AssertionError:
                continue
            rows.append({"task": t, "prompt": prompt, "target_id": tid})
            n_added += 1
        logger.info(f"[{t}] corrupted rows: {n_added}")

    logger.info(f"total corrupted rows: {len(rows)}")

    # ── CMA sweep — accumulate per-task CIE sums ──────────────────────────────
    cie_sum = {t: torch.zeros(n_layer, n_head) for t in tasks}
    cie_cnt = {t: 0 for t in tasks}
    total_forwards = n_layer * n_head * ((len(rows) + args.batch - 1) // args.batch)
    logger.info(f"CMA: {n_layer}x{n_head} heads x {len(rows)} rows "
                f"(~{total_forwards} patched forwards, batch={args.batch})")

    t0 = time.time()
    for c0 in range(0, len(rows), args.batch):
        chunk = rows[c0:c0 + args.batch]
        prompts = [r["prompt"] for r in chunk]
        target_ids = torch.tensor([r["target_id"] for r in chunk])
        chunk_tasks = [r["task"] for r in chunk]
        inputs = prepare_input(prompts=prompts, tokenizer=mt, padding_side="left")
        # per-row source means stacked across layers -> (b, L, n_embd)
        src = torch.stack([mean_by_task[r["task"]] for r in chunk])  # (b, L, n_embd)

        p_base = _last_token_probs(mt, inputs, target_ids)  # (b,)

        for ell in tqdm(layers, desc=f"chunk {c0//args.batch+1}", leave=False):
            src_ell = src[:, ell:ell + 1, :]  # (b, 1, n_embd)
            for j in range(n_head):
                mask = HeadMask(1, n_head)
                mask.values.data.zero_()
                mask.values.data[0, j] = 1.0
                p_pat = _patched_probs(mt, inputs, src_ell, mask, ell, target_ids)  # (b,)
                cie = (p_pat - p_base)  # (b,)
                for bi, tk in enumerate(chunk_tasks):
                    cie_sum[tk][ell, j] += float(cie[bi])
        for tk in chunk_tasks:
            cie_cnt[tk] += 1
        done = c0 + len(chunk)
        logger.info(f"rows {done}/{len(rows)} | elapsed {time.time()-t0:.0f}s")

    # per-task AIE, then equal-weighted mean over tasks (Eq 4)
    aie_per_task = {t: (cie_sum[t] / max(cie_cnt[t], 1)) for t in tasks}
    aie = torch.stack([aie_per_task[t] for t in tasks]).mean(0)  # (n_layer, n_head)

    tag = args.tag or f"{args.tasks}_ns{args.n_shots}_nt{args.n_trials}"
    out_dir = os.path.join(env_utils.DEFAULT_RESULTS_DIR, "fv", args.model.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aie_{tag}.pt")
    torch.save({
        "aie": aie,                                    # (n_layer, n_head)
        "aie_per_task": aie_per_task,                  # {task: (n_layer, n_head)}
        "tasks": tasks,
        "n_rows_per_task": cie_cnt,
        "model": args.model, "n_layer": n_layer, "n_head": n_head,
        "config": vars(args),
    }, out_path)
    logger.info(f"saved AIE -> {out_path}")

    # quick peek: top-10 heads by AIE
    flat = torch.argsort(aie.flatten(), descending=True)[:10]
    logger.info("top-10 heads by AIE:")
    for f in flat.tolist():
        l, h = divmod(f, n_head)
        logger.info(f"  L{l:>2} H{h:>2}  AIE={aie[l, h]:+.4f}")


if __name__ == "__main__":
    main()
