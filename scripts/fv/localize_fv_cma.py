#!/usr/bin/env python
"""Independent port of the Function-Vectors head-level CMA (causal indirect effect, CIE).

Reimplements the *method* — per-head mean-activation extraction and the systematic
(layer, head) causal sweep — in a single clean driver, while REUSING the FV repo's
deterministic data/prompt plumbing (dataset split, ICL prompt construction, token
labelling, answer-token id). Because the plumbing + seed order are identical, at a
fixed seed this reproduces the reference
``existing_works/function_vectors/src/compute_indirect_effect.py`` bit-for-bit — which
is the point: if our port matches, we own a clean entrypoint we can build on.

The reimplemented pieces (what could actually diverge, hence what the exact-match test
guards): the mean-activation trial loop + per-head averaging, and the head-sweep
intervention (split o_proj input by head, overwrite the last query-token's head slice
with its task-mean, recompute the bias-free o_proj output).

Runs in the FV uv venv (needs FV src on sys.path + matched torch/transformers):

    existing_works/function_vectors/.venv/bin/python scripts/fv/localize_fv_cma.py \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct --dataset_name antonym \
        --device cuda --n_shots 4 --n_trials 2 --seed 42 --save_path_root /tmp/fv_port

Validation (see --compare_*): reproduces the reference on Qwen2.5-Coder-7B antonym.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from baukit import TraceDict

# ── bootstrap the FV repo src onto the path (deterministic plumbing we reuse) ──
FV_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "existing_works", "function_vectors", "src")
)
if FV_SRC not in sys.path:
    sys.path.insert(0, FV_SRC)

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed          # noqa: E402
from utils.prompt_utils import (                                              # noqa: E402
    load_dataset,
    word_pairs_to_prompt_data,
    get_dummy_token_labels,
    get_token_meta_labels,
    compute_duplicated_labels,
    update_idx_map,
)
from utils.eval_utils import get_answer_id                                    # noqa: E402
from utils.intervention_utils import get_module                              # noqa: E402

_T0 = time.time()


def log(msg):
    el = int(time.time() - _T0)
    print(f"[+{el // 60:02d}:{el % 60:02d}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Reimplemented method
# ══════════════════════════════════════════════════════════════════════════════
def _split_by_head(activations, mc):
    """(..., resid_dim) -> (..., n_heads, head_dim) over the query heads (GQA-safe)."""
    head_dim = mc["resid_dim"] // mc["n_heads"]
    return activations.view(*activations.size()[:-1], mc["n_heads"], head_dim)


def compute_mean_head_activations(dataset, model, mc, tokenizer, n_shots, n_trials, prefixes, separators):
    """Task-mean of every head's o_proj-input activation, condensing multi-token words
    into one slot. Mirrors FV get_mean_head_activations (shuffle_labels=False)."""
    dummy = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes,
                                   separators=separators, model_config=mc)
    head_dim = mc["resid_dim"] // mc["n_heads"]
    store = torch.zeros(n_trials, mc["n_layers"], mc["n_heads"], len(dummy), head_dim)
    filter_set = np.arange(len(dataset["valid"]))
    prepend_bos = not mc["prepend_bos"]

    for n in range(n_trials):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, 1, replace=False)]
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos,
            shuffle_labels=False, prefixes=prefixes, separators=separators,
        )
        query = prompt_data["query_target"]["input"]
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query, prepend_bos=mc["prepend_bos"]
        )
        inputs = tokenizer([prompt_string], return_tensors="pt").to(model.device)
        idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy)

        with TraceDict(model, layers=mc["attn_hook_names"], retain_input=True, retain_output=False) as td:
            model(**inputs)

        # (layers, tokens, heads, head_dim) -> (layers, heads, tokens, head_dim)
        stack = torch.vstack(
            [_split_by_head(td[layer].input, mc) for layer in mc["attn_hook_names"]]
        ).permute(0, 2, 1, 3)
        filt = stack[:, :, list(idx_map.keys())]
        for (i, j) in idx_avg.values():  # average activations of multi-token words
            filt[:, :, idx_map[i]] = stack[:, :, i:j + 1].mean(axis=2)
        store[n] = filt

    return store.mean(dim=0)


def _make_replace_hook(layer, head, mean_activations, qp_token_inds, idx_map, mc, model):
    """baukit edit_output fn: overwrite the last query-token's (layer, head) slice of the
    o_proj INPUT with its task-mean, then recompute the (bias-free) o_proj OUTPUT."""
    head_dim = mc["resid_dim"] // mc["n_heads"]

    def rep(output, layer_name, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        original_shape = inputs.shape
        inputs = inputs.view(*inputs.size()[:-1], mc["n_heads"], head_dim)
        for token_n in qp_token_inds:  # last_token_only: write into the final position
            inputs[-1, -1, head] = mean_activations[layer, head, idx_map[token_n]]
        inputs = inputs.view(*original_shape)
        out_proj = get_module(model, layer_name).weight  # Qwen/Llama o_proj: bias-free Linear
        return torch.matmul(inputs, out_proj.T)

    return rep


def head_sweep_trial(prompt_data, mean_activations, dummy, model, mc, tokenizer):
    """Indirect effect (n_layers, n_heads) for one ICL prompt: for each head, replace its
    last-token activation with the task-mean and measure Δ prob of the answer token."""
    device = model.device
    query = prompt_data["query_target"]["input"]
    token_labels, prompt_string = get_token_meta_labels(
        prompt_data, tokenizer, query=query, prepend_bos=mc["prepend_bos"]
    )
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy)
    idx_map = update_idx_map(idx_map, idx_avg)

    # query_predictive token position(s) — the slot whose mean we splice in (last_token_only)
    qp_token_inds = [x[0] for x in token_labels if re.match(r"^query_predictive_token$", x[2])]

    target = prompt_data["query_target"]["output"]
    token_id_of_interest = get_answer_id(prompt_string, target, tokenizer)
    if isinstance(token_id_of_interest, list):
        token_id_of_interest = token_id_of_interest[:1]

    inputs = tokenizer([prompt_string], return_tensors="pt").to(device)
    clean_probs = torch.softmax(model(**inputs).logits[:, -1, :][0], dim=-1)

    out = torch.zeros(mc["n_layers"], mc["n_heads"])
    tid = torch.LongTensor(token_id_of_interest).to(device).squeeze()
    for layer in range(mc["n_layers"]):
        hook_name = mc["attn_hook_names"][layer]
        for head in range(mc["n_heads"]):
            fn = _make_replace_hook(layer, head, mean_activations, qp_token_inds, idx_map, mc, model)
            with TraceDict(model, layers=[hook_name], edit_output=fn):
                logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            out[layer, head] = (probs - clean_probs).index_select(1, tid).squeeze()
    return out


def compute_indirect_effect(dataset, mean_activations, model, mc, tokenizer, n_shots, n_trials, prefixes, separators):
    """CIE tensor (n_trials, n_layers, n_heads). Mirrors FV compute_indirect_effect
    (shuffle_labels=True per trial, last_token_only)."""
    dummy = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes,
                                   separators=separators, model_config=mc)
    filter_set = np.arange(len(dataset["valid"]))
    prepend_bos = not mc["prepend_bos"]
    indirect_effect = torch.zeros(n_trials, mc["n_layers"], mc["n_heads"])

    for n in range(n_trials):
        word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset["valid"][np.random.choice(filter_set, 1, replace=False)]
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True,
            prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators,
        )
        indirect_effect[n] = head_sweep_trial(prompt_data, mean_activations, dummy, model, mc, tokenizer)
        log(f"  trial {n + 1}/{n_trials} done")
    return indirect_effect


# ══════════════════════════════════════════════════════════════════════════════
# Comparison helpers
# ══════════════════════════════════════════════════════════════════════════════
def report_match(name, ours, golden_path):
    golden = torch.load(golden_path, map_location="cpu").float()
    ours = ours.float()
    if ours.shape != golden.shape:
        log(f"[{name}] SHAPE MISMATCH ours={tuple(ours.shape)} golden={tuple(golden.shape)}")
        return
    exact = bool(torch.equal(ours, golden))
    max_abs = float((ours - golden).abs().max())
    log(f"[{name}] torch.equal={exact} | max_abs_diff={max_abs:.3e} | "
        f"allclose(1e-5)={bool(torch.allclose(ours, golden, atol=1e-5, rtol=0))}")
    if ours.dim() == 3:  # CIE (trials, L, H) -> top-head overlap of the trial-mean
        mo, mg = ours.mean(0).flatten(), golden.mean(0).flatten()
        k = min(10, mo.numel())
        to, tg = set(torch.topk(mo, k).indices.tolist()), set(torch.topk(mg, k).indices.tolist())
        log(f"[{name}] top-{k} head overlap = {len(to & tg)}/{k}")


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--root_data_dir", type=str, default=os.path.join(FV_SRC, "..", "dataset_files"))
    ap.add_argument("--save_path_root", type=str, required=True, help="results go to <root>/<dataset>/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_shots", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--test_split", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--mean_activations_path", type=str, default=None,
                    help="load a precomputed mean-activation .pt instead of computing it")
    ap.add_argument("--stage", choices=["mean", "cie", "both"], default="both")
    ap.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    ap.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--compare_mean_activations", type=str, default=None, help="golden mean-act .pt to diff against")
    ap.add_argument("--compare_indirect_effect", type=str, default=None, help="golden CIE .pt to diff against")
    return ap.parse_args()


def main():
    args = parse_args()
    save_path_root = os.path.join(args.save_path_root, args.dataset_name)
    os.makedirs(save_path_root, exist_ok=True)

    torch.set_grad_enabled(False)
    log(f"loading model {args.model_name} (device={args.device}) ...")
    model, tokenizer, mc = load_gpt_model_and_tokenizer(args.model_name, device=args.device, revision=args.revision)
    log(f"model loaded: {mc['n_layers']} layers x {mc['n_heads']} heads | "
        f"resid_dim={mc['resid_dim']} | prepend_bos={mc['prepend_bos']}")
    dm = getattr(model, "hf_device_map", None)
    if dm and any(str(v) in ("cpu", "disk") for v in dm.values()):
        log("*** WARNING: part of the model is offloaded to CPU/disk — give this job MORE GPU memory ***")

    set_seed(args.seed)
    log(f"loading dataset {args.dataset_name} (test_split={args.test_split}) ...")
    dataset = load_dataset(args.dataset_name, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
    log(f"dataset: train={len(dataset['train'])} valid={len(dataset['valid'])} test={len(dataset['test'])}")

    # ── mean head activations (load or compute) ──
    default_mean_path = os.path.join(save_path_root, f"{args.dataset_name}_mean_head_activations.pt")
    mean_path = args.mean_activations_path
    if mean_path is None and os.path.exists(default_mean_path):
        mean_path = default_mean_path
    if mean_path is not None and os.path.exists(mean_path):
        log(f"loading mean activations <- {mean_path}")
        mean_activations = torch.load(mean_path, map_location=model.device)
    else:
        log(f"computing mean head activations (n_shots={args.n_shots}, n_trials={args.n_trials}) ...")
        mean_activations = compute_mean_head_activations(
            dataset, model, mc, tokenizer, args.n_shots, args.n_trials, args.prefixes, args.separators
        )
        torch.save(mean_activations, default_mean_path)
        log(f"mean activations done, shape={tuple(mean_activations.shape)} -> {default_mean_path}")

    if args.compare_mean_activations:
        report_match("mean_act", mean_activations, args.compare_mean_activations)

    if args.stage == "mean":
        log("stage=mean: skipping indirect effect. done.")
        return

    # ── causal indirect effect ──
    log(f"computing indirect effect: {args.n_trials} trials x {mc['n_layers']}L x {mc['n_heads']}H "
        f"= {args.n_trials * mc['n_layers'] * mc['n_heads']} patched forwards ...")
    indirect_effect = compute_indirect_effect(
        dataset, mean_activations, model, mc, tokenizer, args.n_shots, args.n_trials, args.prefixes, args.separators
    )
    ie_path = os.path.join(save_path_root, f"{args.dataset_name}_indirect_effect.pt")
    torch.save(indirect_effect, ie_path)
    with open(os.path.join(save_path_root, "indirect_effect_args.txt"), "w") as f:
        json.dump({**vars(args), "save_path_root": save_path_root, "n_layers": mc["n_layers"],
                   "n_heads": mc["n_heads"]}, f, indent=2)
    log(f"DONE: indirect_effect {tuple(indirect_effect.shape)} -> {ie_path}")

    if args.compare_indirect_effect:
        report_match("indirect_effect", indirect_effect, args.compare_indirect_effect)


if __name__ == "__main__":
    main()
