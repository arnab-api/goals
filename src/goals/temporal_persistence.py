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
implementation span is found by an **oracle LLM**.
For each condition we store the goal-encoding at the `=` pivot (`pivot`) and along the implementation
tokens (`traj`); retention/stability is cosine similarity between a condition's pivot and encodings sampled
at x% through an implementation.

Per-condition records are cached under ``<results>/temporal_persistence/<model_slug>/`` so contexts (in
either language) accumulate per sub-task and never recompute. `scripts/evaluation/temporal_persistence_eval.py`
writes the cache; `notebooks/fig_n_table_generators/temporal_persistence.ipynb` reads it.
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

import src.nrgs.data as D
from scripts.evaluation.eval_goalscope import _chat_prompt, greedy_continue, load_json_heads
from src.goals.head_loc_attn import collect_head_steering_vecs
from src.tokens import prepare_input
from src.utils import env_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

PRIMITIVE_VERSION = "goal_heads_resid_write_v1"  # bump if goal_encoding changes → invalidates cache
DESIGN_VERSION = "frontier_prefill_v1"           # bump if the probe construction changes
LANGS = ["python", "cpp"]
DATATYPES = ["int", "str", "list[int]", "list[str]"]
HEAD_BUDGET = 122     # the head-set part of the cache key of the paper's runs (see heads_key)
N_COND = (3, 4)       # task sizes mixed over
MIN_IMPL = 4          # drop conditions with < this many impl tokens
MAX_NEW = 96          # generation budget per probe (part of every cache key)
ORACLE_MODEL = "claude-sonnet-4-6"


def _letter(i: int) -> str:
    return chr(ord("a") + i)


# ── head set ─────────────────────────────────────────────────────────────────────
# The set the paper's runs read (Coder-32B: the 122 goal heads); the default when no other set is given.
PAPER_HEADS = "head_sets/{slug}/goal__lam20_thr0.9_band25-56.json"


def load_goal_heads(mt, slug: str, path: str | None = None) -> list[tuple[int, int]]:
    """The goal heads: the json at ``path``, by default :data:`PAPER_HEADS` of ``slug``."""
    path = path or PAPER_HEADS.format(slug=slug)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: pass the model's goal head set (--heads <json>)")
    return load_json_heads(path, mt.n_layer, mt.n_heads)


def heads_key(slug: str, goal_heads, heads=None) -> str:
    """The head set's part of every cache key. ``str(HEAD_BUDGET)`` (``"122"``, what every key of the paper's runs
    holds) when ``goal_heads`` is the set those runs read (:data:`PAPER_HEADS` of ``slug``), so their records stay
    hits; otherwise ``heads-<md5[:12]>`` of the sorted ``heads`` the encoding is formed from (``goal_heads`` itself,
    or the control set drawn from it), so a different set never reuses another set's records."""
    heads = goal_heads if heads is None else heads
    paper = PAPER_HEADS.format(slug=slug)
    if os.path.exists(paper) and sorted(map(tuple, json.load(open(paper)))) == sorted(map(tuple, goal_heads)):
        return str(HEAD_BUDGET)
    payload = json.dumps(sorted([int(l), int(h)] for l, h in heads))
    return "heads-" + hashlib.md5(payload.encode()).hexdigest()[:12]


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
def rollout_probe(mt, prompt, max_new=MAX_NEW):
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


# ── value segmentation: oracle (multi-line aware) ──────────────────────────────────
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


def oracle_value_segment(code_prefix, target_var, target_desc, generation, model=ORACLE_MODEL):
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
                      max_tokens=512, use_cache=True)
    seg = _strip_fence(resp)
    return seg or None


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
def cache_dir(slug):
    return os.path.join(env_utils.DEFAULT_RESULTS_DIR, "temporal_persistence", slug)


def _cond_key(slug, hkey, datatype, pred_id, lang, ctx_key, max_new, head_tag="goal"):
    payload = (f"{slug}|{hkey}|{datatype}|{pred_id}|{lang}|{ctx_key}|{max_new}"
               f"|{PRIMITIVE_VERSION}|{DESIGN_VERSION}")
    if head_tag != "goal":  # keep goal keys unchanged (reuse existing cache); control gets its own
        payload += f"|{head_tag}"
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def build_records(
    mt,
    slug,
    goal_heads,
    datatype,
    lang,
    *,
    n_trials=256,
    seed=1123,
    force=False,
    head_tag="goal",
    hkey=None,
):
    """Return list[record] for one (datatype, lang). Each non-seed condition of each context is probed
    with a frontier prefill and cached per (sub-task, lang, context). record = dict(datatype, lang,
    pred_id, pred, ctx_key, var, pos, pivot(T), traj(T), n_impl, model, budget, head_tag, heads_key).

    ``goal_heads`` is the head set whose summed residual write forms the encoding; ``head_tag`` names it
    (``goal`` = the goal heads, cached in the root dir; anything else, e.g. ``control``, is cached in a
    ``<head_tag>/`` subdir so it never collides with the goal cache). ``hkey`` (:func:`heads_key`; default the
    key of ``goal_heads`` itself) is the head set's part of every cache key."""
    hkey = heads_key(slug, goal_heads) if hkey is None else hkey
    budget, max_new, min_impl = len(goal_heads), MAX_NEW, MIN_IMPL
    cdir = cache_dir(slug)
    if head_tag != "goal":
        cdir = os.path.join(cdir, head_tag)
    os.makedirs(cdir, exist_ok=True)
    setups = D.multi_predicate_setups(
        datatype, n_conditions=list(N_COND), n_samples=n_trials, seed=seed,
        naming="letters", src_lang=lang, dst_lang=lang,
    )
    from tqdm.auto import tqdm
    it = tqdm(setups, desc=f"contexts [{datatype}/{lang}]")

    records = []
    for inst in it:
        ck = context_key(inst)
        preds = inst._preds_for(inst.src_frontier)
        for t in range(1, len(preds)):  # probe every non-seed condition (a=pos0 is the prefilled seed)
            pid = preds[t].id
            cpath = os.path.join(cdir, f"cond_{_cond_key(slug, hkey, datatype, pid, lang, ck, max_new, head_tag)}.pt")
            if os.path.exists(cpath) and not force:
                rec = torch.load(cpath)["record"]
                if rec["n_impl"] >= min_impl:  # apply the filter on reload too (cache ignores min_impl)
                    records.append(rec)
                continue
            prompt, tv = build_probe_prompt(mt, inst, preds, t)
            full_text = rollout_probe(mt, prompt, max_new=max_new)
            gen = full_text[len(prompt):]
            code_prefix = prompt.split("<code>\n", 1)[-1] if "<code>\n" in prompt else prompt
            seg = oracle_value_segment(code_prefix, tv, preds[t].description, gen)
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
                n_impl=len(impl), model=slug, budget=budget, head_tag=head_tag, heads_key=hkey,
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


def compute_curves(records, prog_grid):
    """cos(pivot_i, encoding_at_p_j) aggregated over the four relation masks (the same-lang and
    cross-lang variants of the different-context comparisons are pooled)."""
    if not records:
        return {}
    P = _unit(torch.stack([r["pivot"] for r in records]))
    G = {p: _unit(torch.stack([_prog_vec(r, p) for r in records])) for p in prog_grid}
    pred = np.array([r["pred_id"] for r in records])
    ctx = np.array([r["ctx_key"] for r in records])
    n = len(records)
    eye = np.eye(n, dtype=bool)
    same_pred = pred[:, None] == pred[None, :]
    same_ctx = ctx[:, None] == ctx[None, :]
    # note: same_ctx ⟹ same_lang (ctx_key includes the language), so there is no cross-lang `iii`.
    masks = {
        "i   same condition": eye,
        "ii  same pred, diff ctx": same_pred & ~same_ctx,
        "iii diff pred, same ctx": ~same_pred & same_ctx,
        "iv  diff pred, diff ctx": ~same_pred & ~same_ctx,
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
