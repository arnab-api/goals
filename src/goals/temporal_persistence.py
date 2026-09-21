"""Temporal persistence of the goal encoding — shared core for the notebook and the eval script.

Question: once the model commits to a sub-goal at its `=` pivot, does the goal-head encoding stay on
that sub-goal across the tokens that implement it (retention), and is it stable across *different
contexts* (same sub-goal in a different function / a different language)?

Unit of analysis = one **condition**: a single sub-goal `g` implemented at a specific position in one
**context** (one multi-predicate task) in one **language**. We probe it with a clean **frontier prefill**:
the task lists all predicates; the conditions *before* the target are prefilled as full, newline-terminated
lines; the target is prefilled only up to its `=` (Python ``    b =`` / C++ ``    auto b =``), so the `=`
pivot sits exactly at the prompt boundary and the model writes only that sub-goal's value. Probing every
non-seed condition of a context (each in its own forward pass) keeps the *same-context / different-sub-goal*
comparison (``iii``) intact.

The value the model writes is frequently **multi-line in C++** (loops, reassignments), so the target's
implementation span is found by an **oracle LLM** (`--no-oracle` falls back to a next-letter boundary rule).
For each condition we store the goal-encoding at the `=` pivot (`pivot`) and along the implementation
tokens (`traj`); retention/stability is cosine similarity between a condition's pivot and encodings sampled
at x% through an implementation.

Per-condition records are cached under ``<results>/temporal_persistence/<model_slug>/`` so contexts (in
either language) accumulate per sub-task and never recompute. The notebook (`notebooks/persistance/
01_temporal_persistance.ipynb`) and `scripts/evaluation/temporal_persistence_eval.py` both import this
module, so the cache and key logic never drift.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re

import numpy as np
import torch

import src.diff.data as D
from scripts.evaluation.eval_goalscope import _chat_prompt, greedy_continue, load_json_heads
from src.goals.head_loc_attn import collect_head_steering_vecs
from src.tokens import prepare_input
from src.utils import env_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

PRIMITIVE_VERSION = "goal_heads_resid_write_v1"  # bump if goal_encoding changes → invalidates cache
DESIGN_VERSION = "frontier_prefill_v1"           # bump if the probe construction changes
DEFAULT_PROG_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LANGS = ["python", "cpp"]


def _letter(i: int) -> str:
    return chr(ord("a") + i)


# ── head sets (mirror the notebook) ─────────────────────────────────────────────
def head_set_path(slug: str, budget: int) -> str:
    return {
        250: f"head_sets/{slug}/goal_topk/goal__lam20_top250.json",
        122: f"head_sets/{slug}/goal__lam20_thr0.9_band25-56.json",
    }[budget]


def load_goal_heads(mt, slug: str, budget: int) -> list[tuple[int, int]]:
    return load_json_heads(head_set_path(slug, budget), mt.n_layer, mt.n_heads)


# ── goal-encoding primitive (same as clock/14 & the notebook) ────────────────────
@torch.no_grad()
def goal_encoding(mt, text, positions, goal_heads) -> torch.Tensor:
    """(n_positions, n_embd) float32 CPU — the goal heads' summed residual write at each position.
    `text` may be a str or a pre-tokenized TokenizerOutput (positions must match its tokenization)."""
    inp = prepare_input(prompts=text, tokenizer=mt) if isinstance(text, str) else text
    vecs = collect_head_steering_vecs(mt, inp, goal_heads, token_positions=list(positions))
    return vecs[0].float().cpu()  # batch of 1 → (n_pos, n_embd)


# ── context + frontier-prefill probe construction ────────────────────────────────
def context_key(inst) -> str:
    """Stable identity of a context = ordered predicate ids of the task list + datatype + language.
    Shared across every target position of the same instance, so same-ctx pairs are well-defined."""
    preds = inst._preds_for(inst.src_frontier)
    return f"{inst.datatype}|{inst.src_lang}|" + ",".join(p.id for p in preds)


def build_probe_prompt(mt, inst, preds, t):
    """Frontier-prefill probe for target position ``t`` (1 ≤ t < len(preds)). The task lists ALL preds;
    positions 0..t-1 are prefilled as full newline-terminated lines; position t is prefilled up to `=`
    (python ``    b =`` / c++ ``    auto b =``). Returns (chat_prompt, target_var)."""
    lang = inst.src_lang
    descs = [p.description for p in preds]
    q = D._question(descs, inst.datatype, inst.var_name, lang, "listed")
    q = q.replace("computes:", "computes in sequential order:")
    lines = []
    for j in range(t):
        rn = _letter(j)
        ln = D._assign_line(preds[j], inst.var_name, lang, res_name=rn)
        if lang == "cpp":
            ln = D._cpp_autoify(ln, rn)  # match the `auto x =` stub style
        lines.append(ln)
    base = "<code>\n" + D._signature(inst.datatype, inst.var_name, lang)
    if lines:
        base += "\n" + "\n".join(lines)
    tv = _letter(t)
    stub = f"auto {tv} =" if lang == "cpp" else f"{tv} ="
    return _chat_prompt(mt, q, base + "\n    " + stub), tv


@torch.no_grad()
def rollout_probe(mt, prompt, max_new=96):
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    gen = greedy_continue(mt, **inp, max_new_tokens=max_new)
    return prompt + gen


def _offsets(mt, full_text):
    enc = prepare_input(prompts=full_text, tokenizer=mt, return_offsets_mapping=True)
    return [tuple(p) for p in enc.pop("offset_mapping")[0].tolist()]


def _toks_in(offsets, a_char, b_char):
    return [t for t, (a, b) in enumerate(offsets) if b > a and a < b_char and b > a_char]


def prompt_boundary(mt, prompt, full_text):
    """Token index P where the generation begins (== the `first_impl` position; `=` pivot is P-1).
    Falls back to the common prefix length if a boundary token merges on re-tokenisation."""
    pid = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    fid = prepare_input(prompts=full_text, tokenizer=mt)["input_ids"][0].tolist()
    P = len(pid)
    if fid[:P] != pid:
        P = next((i for i in range(min(len(fid), len(pid))) if fid[i] != pid[i]),
                 min(len(fid), len(pid)))
    return P


# ── value segmentation: oracle (multi-line aware) + next-letter fallback ──────────
_SEG_SYS = (
    "You are a precise code-analysis tool. Return only the exact verbatim substring requested — copied "
    "character-for-character from the given text, with no quoting, no code fences, and no commentary."
)


def _strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def oracle_value_segment(code_prefix, target_var, target_desc, generation,
                         model="claude-sonnet-4-6", use_cache=True):
    """Verbatim substring of ``generation`` that is ``target_var``'s value (its RHS, possibly multi-line),
    excluding any later variable's code or the final return. Returns str | None."""
    prompt = (
        f"A function is being written incrementally; variables are assigned in order. The variable "
        f"`{target_var}` implements this sub-goal:\n  {target_desc}\n\n"
        f"CODE SO FAR (ends exactly where `{target_var}` is assigned, right after the `=`):\n"
        f"```\n{code_prefix}\n```\n\n"
        f"MODEL CONTINUATION (generated after the `=`):\n```\n{generation}\n```\n\n"
        f"Return ONLY the exact verbatim substring of MODEL CONTINUATION that is the value assigned to "
        f"`{target_var}` — its complete right-hand side, which may span multiple lines (e.g. a loop that "
        f"computes it). Do NOT include any later variable's assignment or the final return. "
        f"Output just that substring."
    )
    resp = ask_claude(prompt, system_prompt=_SEG_SYS, model_name=model, temperature=0.0,
                      max_tokens=512, use_cache=use_cache)
    seg = _strip_fence(resp)
    return seg or None


def rule_based_value(generation, target_var, next_var):
    """No-oracle fallback: ``target_var``'s value = generation up to the next condition's assignment
    (``next_var =`` / ``auto next_var =``) or the terminal return/close. Handles multi-line targets
    (reassignments to target_var stay inside the span)."""
    if next_var:
        m = re.search(rf"(?m)^[ \t]*(?:auto\s+)?{re.escape(next_var)}\s*=(?!=)", generation)
        end = m.start() if m else None
    else:
        end = None
    if end is None:
        mterm = re.search(r"(?m)^[ \t]*(?:return\b|\}|</code>)", generation)
        end = mterm.start() if mterm else len(generation)
    return generation[:end].strip() or None


def locate_value(offsets, full_text, prompt_len, P, substring):
    """Map the target's value substring → (eq_tok, impl_toks). eq_tok = P-1 (the prefilled `=`);
    impl_toks = generation tokens (≥ P) spanned by the value. Returns None if unlocatable."""
    sub = _strip_fence(substring)
    if not sub or P <= 0:
        return None
    start = full_text.find(sub, prompt_len)
    if start >= 0:
        end = start + len(sub)
    else:  # oracle reformatted whitespace → anchor on first & last non-empty lines
        lines = [ln.strip() for ln in sub.splitlines() if ln.strip()]
        if not lines:
            return None
        start = full_text.find(lines[0], prompt_len)
        if start < 0:
            return None
        le = full_text.find(lines[-1], start + len(lines[0])) if len(lines) > 1 else start + len(lines[0])
        end = (le + len(lines[-1])) if le >= 0 else full_text.find("\n", start)
        if end < 0:
            end = len(full_text)
    impl = [t for t in _toks_in(offsets, start, end) if t >= P]
    if not impl:
        return None
    return P - 1, impl


# ── per-condition records + caching ──────────────────────────────────────────────
def cache_dir(slug, results_dir=None):
    rd = results_dir or env_utils.DEFAULT_RESULTS_DIR
    return os.path.join(rd, "temporal_persistence", slug)


def _cond_key(slug, budget, datatype, pred_id, lang, ctx_key, max_new):
    payload = (f"{slug}|{budget}|{datatype}|{pred_id}|{lang}|{ctx_key}|{max_new}"
               f"|{PRIMITIVE_VERSION}|{DESIGN_VERSION}")
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def build_records(
    mt,
    slug,
    goal_heads,
    datatype,
    lang,
    *,
    budget,
    n_trials=64,
    n_cond=(3, 4),
    seed=1123,
    min_impl=4,
    max_new=96,
    use_oracle=True,
    oracle_model="claude-sonnet-4-6",
    force=False,
    results_dir=None,
    progress=True,
):
    """Return list[record] for one (datatype, lang). Each non-seed condition of each context is probed
    with a frontier prefill and cached per (sub-task, lang, context). record = dict(datatype, lang,
    pred_id, pred, ctx_key, var, pos, pivot(T), traj(T), n_impl, model, budget)."""
    cdir = cache_dir(slug, results_dir)
    os.makedirs(cdir, exist_ok=True)
    setups = D.multi_predicate_setups(
        datatype, n_conditions=list(n_cond), n_samples=n_trials, seed=seed,
        naming="letters", src_lang=lang, dst_lang=lang,
    )
    it = setups
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(setups, desc=f"contexts [{datatype}/{lang}]")
        except Exception:
            pass

    records = []
    for inst in it:
        ck = context_key(inst)
        preds = inst._preds_for(inst.src_frontier)
        for t in range(1, len(preds)):  # probe every non-seed condition (a=pos0 is the prefilled seed)
            pid = preds[t].id
            cpath = os.path.join(cdir, f"cond_{_cond_key(slug, budget, datatype, pid, lang, ck, max_new)}.pt")
            if os.path.exists(cpath) and not force:
                rec = torch.load(cpath)["record"]
                if rec["n_impl"] >= min_impl:  # apply the filter on reload too (cache ignores min_impl)
                    records.append(rec)
                continue
            prompt, tv = build_probe_prompt(mt, inst, preds, t)
            full_text = rollout_probe(mt, prompt, max_new=max_new)
            gen = full_text[len(prompt):]
            next_var = _letter(t + 1) if t + 1 < len(preds) else None
            if use_oracle:
                code_prefix = prompt.split("<code>\n", 1)[-1] if "<code>\n" in prompt else prompt
                seg = oracle_value_segment(code_prefix, tv, preds[t].description, gen, model=oracle_model)
            else:
                seg = rule_based_value(gen, tv, next_var)
            if not seg:
                continue
            offs = _offsets(mt, full_text)
            P = prompt_boundary(mt, prompt, full_text)
            loc = locate_value(offs, full_text, len(prompt), P, seg)
            if loc is None:
                continue
            eq_tok, impl = loc
            if len(impl) < min_impl:
                continue
            enc = goal_encoding(mt, full_text, [eq_tok] + impl, goal_heads)
            rec = dict(
                datatype=datatype, lang=lang, pred_id=pid, pred=preds[t].description,
                ctx_key=ck, var=tv, pos=t, pivot=enc[0].clone(), traj=enc[1:].clone(),
                n_impl=len(impl), model=slug, budget=budget,
            )
            tmp = cpath + ".tmp"
            torch.save({"record": rec}, tmp)
            os.replace(tmp, cpath)
            records.append(rec)
    return records


# ── curves ───────────────────────────────────────────────────────────────────────
def _prog_vec(rec, p):
    if p <= 0:
        return rec["pivot"]
    n = rec["n_impl"]
    return rec["traj"][min(n - 1, max(0, math.ceil(p * n) - 1))]


def _unit(M):
    M = M.float()
    return (M / M.norm(dim=-1, keepdim=True)).numpy()


def compute_curves(records, prog_grid=DEFAULT_PROG_GRID):
    """cos(pivot_i, encoding_at_p_j) aggregated over relation masks, incl. cross-language same-sub-goal."""
    if not records:
        return {}
    P = _unit(torch.stack([r["pivot"] for r in records]))
    G = {p: _unit(torch.stack([_prog_vec(r, p) for r in records])) for p in prog_grid}
    pred = np.array([r["pred_id"] for r in records])
    ctx = np.array([r["ctx_key"] for r in records])
    lang = np.array([r["lang"] for r in records])
    n = len(records)
    eye = np.eye(n, dtype=bool)
    same_pred = pred[:, None] == pred[None, :]
    same_ctx = ctx[:, None] == ctx[None, :]
    same_lang = lang[:, None] == lang[None, :]
    # note: same_ctx ⟹ same_lang (ctx_key includes the language), so there is no cross-lang `iii`.
    masks = {
        "i   same condition": eye,
        "ii  same pred, diff ctx, same lang": same_pred & ~same_ctx & same_lang,
        "ii* same pred, diff ctx, cross lang": same_pred & ~same_ctx & ~same_lang,
        "iii diff pred, same ctx": ~same_pred & same_ctx,
        "iv  diff pred, diff ctx, same lang": ~same_pred & ~same_ctx & same_lang,
        "iv* diff pred, diff ctx, cross lang": ~same_pred & ~same_ctx & ~same_lang,
    }
    curves = {}
    for name, m in masks.items():
        curves[name] = {}
        for p in prog_grid:
            v = (P @ G[p].T)[m]
            if len(v):
                mu = float(v.mean())
                sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                curves[name][p] = dict(mean=mu, std=sd, sem=sd / np.sqrt(len(v)), n=int(len(v)))
            else:
                curves[name][p] = dict(mean=float("nan"), std=float("nan"), sem=float("nan"), n=0)
    return curves
