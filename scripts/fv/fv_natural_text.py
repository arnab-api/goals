#!/usr/bin/env python
"""Figure-1 replication: does a function vector transfer to *natural text*?

Todd et al. 2024, Figure 1: an FV extracted from an antonym ICL prompt
(``arrive:depart, small:big, common:``) is inserted into an unrelated natural-text
prompt (``The word "fast" means``) and makes the model emit ``slow``.  §3.3 of the
paper generalizes this over 6 tasks and several templates.

Two deliberate deviations from the paper, both to stay consistent with the rest of
this codebase (see ``notebooks/clock/10_goal_vs_FV_heads.ipynb`` §2/§4):

  * **Replace, not add.**  The paper adds the summed FV to the residual stream at a
    single layer ℓ ≈ |L|/3 (layer 9 for GPT-J; ≈21 for a 64-layer model) — see
    ``EDIT_LAYER_FRACTION``.  Here we instead *replace* the selected heads' o_proj
    inputs with the mean activation ā via ``forward_with_src_replacement``, which is
    the same primitive the CMA localization and the head-mask training use.
  * **No src/dest position alignment.**  ā is a mean over ICL prompts of a different
    length than the natural-text destination.  Position-invariance is part of the FV
    claim, so this is intended, not an oversight.

Arms.  Every (task, template) is evaluated under a clean baseline plus one patched arm
per head set (FV heads / goal heads / a layer-matched random control), crossed with two
sources for ā: ``"clean"`` (correct labels) and ``"shuffled"`` (permuted demo labels —
the source-side null; a shuffled ā should carry no task).

Scoring — both protocols the specs asked for:

  1. ``first_token``  — scaffolded prompt (``… means "``) so the answer is the very next
     token, then top-1 must be a non-trivial prefix of the target
     (``src.functional.is_nontrivial_prefix``).  Also reports mean Δp on the
     contextualized first answer token.  The scaffold is ours, not the paper's; it
     exists because the paper's own +FV completions start with a quote
     (``The word "limitless", means "finite"``), which a bare first-token check would
     score as a miss.  The clean arm sees the identical scaffold, so it is controlled.
  2. ``generate``    — paper protocol (§3.3): the unscaffolded template, patch held at
     the final prompt token *and every subsequent decode step*, generate
     ``max_new_tokens`` (paper: 5), then substring-match the target.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from scripts.fv.fv_common import (FVTask, _drop_pairs_touching, first_answer_token_id,
                                  compute_mean_head_activations, load_fv_task)
from src.functional import is_nontrivial_prefix
from src.goals.goalscope import headpatched_generate
from src.goals.head_loc_attn import HeadMask, forward_with_src_replacement
from src.tokens import prepare_input

# Paper §B.3: "we add the FV to the hidden state at layer ℓ ≈ |L|/3, which we found works
# well in practice" (layer 9 / GPT-J, 15 / GPT-NeoX, 11 / Llama-2-7B, 26 / Llama-2-70B).
# Only relevant if the replace intervention comes out flat and we fall back to additive.
EDIT_LAYER_FRACTION = 1 / 3


# ── templates ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NaturalTemplate:
    """One natural-text destination context.

    `free` is the paper-verbatim template used by the `generate` protocol; `scaffold` is
    appended to it for the `first_token` protocol so the answer lands on the next token.
    `prepend_space` says whether the target carries a leading space *after the scaffold*
    (False right after an opening quote, True after a bare word)."""

    name: str
    free: str  # contains "{x}"
    scaffold: str
    prepend_space: bool

    def render(self, x: str, scaffolded: bool) -> str:
        return self.free.format(x=x) + (self.scaffold if scaffolded else "")


# Paper §3.3 / Appendix F word templates, verbatim.
WORD_TEMPLATES = [
    NaturalTemplate("word-means", 'The word "{x}", means', ' "', prepend_space=False),
    NaturalTemplate("word-think", 'When I think of the word "{x}", it usually means',
                    ' "', prepend_space=False),
]

# Paper location templates.  loc-visit needs no scaffold (the target can be the next
# token already); loc-ago's scaffold is ours and does hint at "a city" — the clean arm
# gets the same hint, so read that pair against each other, not across templates.
LOC_TEMPLATES = [
    NaturalTemplate("loc-visit", "If you ever travel to {x}, you have to visit",
                    "", prepend_space=True),
    NaturalTemplate("loc-ago", "A couple years ago I visited {x}, and",
                    " the first city I went to was", prepend_space=True),
]

# Tasks whose query is a place, so the "The word X means" frame does not apply.
LOC_TASKS = {"country-capital", "landmark-country", "national_parks", "park-country"}


def templates_for(task_name: str) -> list[NaturalTemplate]:
    return LOC_TEMPLATES if task_name in LOC_TASKS else WORD_TEMPLATES


# ── head sets ─────────────────────────────────────────────────────────────────
def binary_mask(heads, n_layer: int, n_heads: int) -> HeadMask:
    """HeadMask that is 1.0 exactly at `heads` — full replace there, no-op elsewhere."""
    m = HeadMask(n_layer, n_heads)
    with torch.no_grad():
        m.values.zero_()
        for (l, h) in heads:
            m.values[int(l), int(h)] = 1.0
    return m


def layer_matched_random_heads(reference, n_heads: int,
                               rng: random.Random | None = None) -> list[tuple[int, int]]:
    """Random control with the *same per-layer head count* as `reference`.

    Matching the layer profile matters: FV heads cluster in early-middle layers, so a
    uniformly random set would differ from them in depth as well as identity, and a null
    result could then be attributed to depth rather than to head identity."""
    rng = rng or random.Random(0)
    by_layer: dict[int, int] = {}
    for (l, _) in reference:
        by_layer[int(l)] = by_layer.get(int(l), 0) + 1
    out = []
    for l, k in by_layer.items():
        taken = {h for (ll, h) in reference if int(ll) == l}
        pool = [h for h in range(n_heads) if h not in taken]
        out += [(l, h) for h in rng.sample(pool, min(k, len(pool)))]
    return sorted(out)


# ── queries ───────────────────────────────────────────────────────────────────
def pick_queries(task: FVTask, n_query: int, disjoint: bool = True) -> list[dict]:
    """Eval pairs from the valid split, optionally guaranteed unseen by ā.

    `disjoint=True` drops any valid pair whose input or output also occurs anywhere in
    the train split — the pool ā's demos are drawn from.  That is the "{w} does not
    appear in the source demonstrations" requirement, enforced on both sides of each
    pair (an eval word must not show up as a demo *output* either)."""
    pool = task.valid
    if disjoint:
        train_words = {p["input"] for p in task.train} | {p["output"] for p in task.train}
        pool = _drop_pairs_touching(pool, train_words)
    return pool[:n_query]


# ── evaluation ────────────────────────────────────────────────────────────────
@dataclass
class NaturalTextEval:
    """Per-(task, template, source, arm) rows plus a few verbatim completions."""

    rows: pd.DataFrame
    examples: list[dict] = field(default_factory=list)


@torch.inference_mode()
def eval_natural_text(
    mt,
    head_sets: dict[str, list],
    tasks: list[str],
    layers: list[int] | None = None,
    n_query: int = 20,
    n_shots: int = 10,
    mean_trials: int = 25,
    sources: tuple[str, ...] = ("clean", "shuffled"),
    max_new_tokens: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
    disjoint: bool = True,
    n_examples: int = 3,
    logger=None,
) -> NaturalTextEval:
    """Run the Fig-1 replication over `tasks` × templates × `sources` × `head_sets`.

    `head_sets` maps an arm name to a list of (layer, head).  A clean (unpatched) arm is
    always added.  Returns tidy rows with `acc_first` / `acc_gen` / `dP` per cell."""
    n_layer = mt.n_layer
    n_heads = mt.model.config.num_attention_heads
    layers = list(range(n_layer)) if layers is None else list(layers)
    masks = {name: binary_mask(hs, n_layer, n_heads) for name, hs in head_sets.items()}
    # mask≡0 → pure cascade dest output, i.e. an exactly unpatched forward.  Reusing the
    # patched code path for the baseline keeps the decoding loop bit-identical across arms.
    zero_mask = binary_mask([], n_layer, n_heads)

    rows, examples = [], []
    for task_name in tasks:
        task = load_fv_task(task_name, seed=seed)
        queries = pick_queries(task, n_query, disjoint=disjoint)
        if not queries:
            raise ValueError(f"[{task_name}] no eval queries survived the disjointness filter")
        held_out = {q["input"] for q in queries} | {q["output"] for q in queries}
        if logger:
            logger.info(f"[{task_name}] {len(queries)} queries "
                        f"(valid={len(task.valid)}, train={len(task.train)})")

        a_bars = {
            src: compute_mean_head_activations(
                mt, task, layers, n_shots=n_shots, n_trials=mean_trials,
                batch_size=16, seed=seed, shuffle_labels=(src == "shuffled"),
                exclude_words=held_out if disjoint else None)
            for src in sources
        }

        for tpl in templates_for(task_name):
            # ── clean arm: independent of which ā we would have patched, so run once ──
            clean_first, clean_gen, clean_p, tids = [], [], [], []
            for q in queries:
                sc = tpl.render(q["input"], scaffolded=True)
                tid = first_answer_token_id(mt, sc, q["output"],
                                            prepend_space=tpl.prepend_space)
                inp = prepare_input(prompts=sc, tokenizer=mt)
                p = torch.softmax(mt._model(**dict(inp)).logits[:, -1, :].float(), -1)[0]
                clean_first.append(mt.tokenizer.decode([int(p.argmax())]))
                clean_p.append(float(p[tid]))
                tids.append(tid)
                free_inp = prepare_input(prompts=tpl.render(q["input"], scaffolded=False),
                                         tokenizer=mt)
                clean_gen.append(headpatched_generate(
                    mt, dict(free_inp), a_bars[sources[0]], zero_mask, layers,
                    max_new_tokens=max_new_tokens, patch_input_positions=[-1]))
            rows.append(_row(task_name, tpl.name, "-", "clean", queries, clean_first,
                             clean_gen, clean_p, clean_p))

            # ── patched arms: same query loop, only ā and the mask change ──
            for src in sources:
                for arm, mask in masks.items():
                    pf, pg, pp = [], [], []
                    for i, q in enumerate(queries):
                        sc = tpl.render(q["input"], scaffolded=True)
                        inp = prepare_input(prompts=sc, tokenizer=mt)
                        lg = forward_with_src_replacement(
                            mt, inp, a_bars[src], mask, layers=layers, alpha=alpha,
                            patch_positions=[-1])
                        if lg.ndim > 1:
                            lg = lg[0]
                        p = torch.softmax(lg.float(), -1)
                        pf.append(mt.tokenizer.decode([int(p.argmax())]))
                        pp.append(float(p[tids[i]]))
                        free_inp = prepare_input(
                            prompts=tpl.render(q["input"], scaffolded=False), tokenizer=mt)
                        # n_steer_tokens=None → patch the prefill AND every decode step,
                        # which is what §3.3 does for natural text.
                        pg.append(headpatched_generate(
                            mt, dict(free_inp), a_bars[src], mask, layers,
                            max_new_tokens=max_new_tokens, n_steer_tokens=None,
                            patch_input_positions=[-1], alpha=alpha))
                    rows.append(_row(task_name, tpl.name, src, arm, queries, pf, pg,
                                     pp, clean_p))
                    for i in range(min(n_examples, len(queries))):
                        examples.append(dict(
                            task=task_name, template=tpl.name, source=src, arm=arm,
                            x=queries[i]["input"], y=queries[i]["output"],
                            clean_first=clean_first[i], patch_first=pf[i],
                            clean_gen=clean_gen[i], patch_gen=pg[i]))
            if logger:
                last = [r for r in rows if r["task"] == task_name and r["template"] == tpl.name]
                logger.info(f"  [{tpl.name}] " + "  ".join(
                    f"{r['source']}/{r['arm']} {r['acc_first']:.2f}/{r['acc_gen']:.2f}"
                    for r in last))

    return NaturalTextEval(rows=pd.DataFrame(rows), examples=examples)


def _row(task, template, source, arm, queries, firsts, gens, probs, clean_probs) -> dict:
    """One tidy row.  `acc_first` uses the prefix rule; `acc_gen` the paper's substring rule."""
    hits = [is_nontrivial_prefix(f, q["output"]) for f, q in zip(firsts, queries)]
    gen_hits = [q["output"].lower().strip() in g.lower() for g, q in zip(gens, queries)]
    return dict(
        task=task, template=template, source=source, arm=arm, n=len(queries),
        acc_first=float(np.mean(hits)),
        acc_gen=float(np.mean(gen_hits)),
        dP=float(np.mean(np.array(probs) - np.array(clean_probs))),
    )
