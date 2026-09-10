#!/usr/bin/env python
"""GoalScope goal-head eval on the multi-predicate ``check_conditions`` setup — two ``--mode``s:

  faithfulness  — read out an NL description at each anchor ("what are you doing at this line?")
                  and grade whether it faithfully describes the implementation span.
  causality     — steer the goal encoding into an information-less C++ stub
                  (``bool check(auto value){``) and grade whether the C++ continuation implements
                  the same *idea* as the Python implementation span (cross-language transfer).

Both share the whole pipeline; they differ only in the GoalScope destination prompt, the read-out
post-processing (causality prepends the C++ prefix), and the oracle rubric/prompt. Results save to
``<results>/evaluation/goalscope/<mode>/<strategy>/<alpha>/<leaf>/``.

Pipeline (consolidated from ``notebooks/clock/12_goal_eval_scaleup.ipynb`` (faithfulness) and
``12.1_goal_eval_causality.ipynb`` (causality)):

  1. Corpus — sample single-line INT predicates into ``check_conditions(n: int)`` tasks
     (letters naming ``a/b/c``); greedy-decode the body.
  2. Oracle spans — ``label_generation`` segments the body into per-condition spans (impl_span)
     + condition descriptions (the ground truth: *what* each line implements).
  3. Anchors — per assignment line, locate the token positions {prev_end, indent, var, eq,
     first_impl} in the GoalScope tokenization.
  4. Read-outs — ``apply_goalscope`` restricted to just those anchor positions (fast).
  5. Scoring — hard eval (oracle: is the read-out a faithful description of the impl_span?) and
     k-way (oracle: which of the sample's K conditions does the read-out name?). Soft eval is
     intentionally dropped (didn't discriminate).

Everything is saved under
``<results>/evaluation/goalscope/faithfulness/<strategy>/<alpha>/<leaf>/`` — config (incl. the
selected heads), the full per-anchor records with the raw oracle prompts + responses, a per-anchor
summary, and per-sample generations + labels. Parse with ``scripts.evaluation.faithfulness_io``.

All ``ask_claude`` calls use ``use_cache=True``.

Usage:
    conda run -n goals python scripts/evaluation/eval_goalscope.py \
        --mode faithfulness --heads goal --strategy additive --alpha 0.2 \
        --steer_layer_range 35 46 --n_steer_tokens all --max_new_tokens 64

    conda run -n goals python scripts/evaluation/eval_goalscope.py \
        --mode causality --heads goal --strategy additive --alpha 0.2   # C++ transfer

    # a custom head set — the way to eval any filtered/ranked selection (fv, concept, top-k, ...):
    ... --heads /path/to/heads.json          # json array of [layer_idx, head_idx] pairs
    # It supersedes every head-selection flag: combining it with --goal_threshold/--goal_band is an
    # error, not a silent no-op. The run dir is keyed on the file's STEM, so give distinct files
    # distinct basenames — two `heads.json` in different dirs would otherwise resume each other.
    # optional k-way oracle (faithfulness only for now):
    ... --eval-k-way
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch

from scripts.clock.score_condition_heads import load_head_scores
from scripts.evaluation import faithfulness_io as fio
from src.goals.condition_data import (INT_PREDICATES, materialize, predicates_for,
                                      sample_specs)
from src.goals.functional import apply_chat_template, supports_chat_template
from src.goals.goalscope import GOALSCOPE_SYS, apply_goalscope, greedy_continue
from src.goals.multistep_label import label_generation
from src.goals.oid_gaze import cache_root
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)  # configured in main() via logging_utils

ANCHOR_ORDER = ["prev_end", "indent", "var", "eq", "first_impl"]
ORACLE_MODEL = "claude-sonnet-4-6"


# ══════════════════════════════════════════════════════════════════════════════
# Head sets
# ══════════════════════════════════════════════════════════════════════════════
# Layer band the goal heads are read from, per model (mirrors clock/12 cell 4). The trained mask
# covers every layer; the band restricts selection to the region where that model's goal heads
# actually live. A model missing here falls back to the full depth, which is deliberately
# conservative — set its band explicitly once you have looked at the mask heatmap.
GOAL_BANDS = {
    "Qwen2.5-Coder-32B-Instruct": (25, 56),
    "Qwen2.5-32B-Instruct": (25, 56),  # code axis: matched to Coder (both 64 layers)
    "Llama-3.3-70B-Instruct": (18, 45),
    "Olmo-3.1-32B-Instruct": (10, 36),
    "Olmo-3.1-32B-Instruct-SFT": (10, 36),  # RL axis: matched to Olmo Instruct
}


def default_goal_band(model_name, n_layer=None):
    """`GOAL_BANDS[model]`, else the full depth (needs `n_layer`) — never a foreign model's band."""
    slug = model_name.split("/")[-1]
    if slug in GOAL_BANDS:
        return GOAL_BANDS[slug]
    if n_layer is None:
        raise ValueError(
            f"no GOAL_BANDS entry for {slug!r} — pass --goal_band explicitly (or add it to "
            f"GOAL_BANDS). Falling back to another model's band would select the wrong heads."
        )
    logger.warning(f"no GOAL_BANDS entry for {slug!r} — using the full depth (0, {n_layer})")
    return (0, n_layer)


def load_goal_heads(model_name, results_dir, threshold=0.9, band=None, tag="lam20",
                    n_layer=None):
    ck = torch.load(
        os.path.join(
            results_dir, f"localization/{model_name.split('/')[-1]}/{tag}/head_mask.pt"
        ),
        map_location="cpu",
    )
    layers = [int(x) for x in ck["layers"]]
    m = ck["mask_values"].numpy().copy()
    if band is None:
        band = default_goal_band(model_name, n_layer=n_layer)
    m[: band[0]] = 0
    m[band[1] :] = 0
    return [(layers[li], int(h)) for li, h in np.argwhere(m > threshold)]


def load_decision_heads(model_name, tag="lam10.0", threshold=0.8, version="conditions"):
    root = os.path.join(cache_root(model_name, version), "completion_mask")
    cands = [
        t for t in os.listdir(root) if os.path.isdir(os.path.join(root, t)) and tag in t
    ]
    assert cands, f"no completion_mask with tag {tag!r} under {root}"
    latest = max(cands, key=lambda t: os.path.getmtime(os.path.join(root, t)))
    ck = torch.load(
        os.path.join(root, latest, "mask.pt"), map_location="cpu", weights_only=False
    )
    layers = [int(x) for x in ck["layers"]]
    m = ck["mask_values"].numpy()
    return [(layers[li], int(h)) for li, h in np.argwhere(m > threshold)]


def load_gaze_heads(
    model_name, n_heads, topk, q_span=15, compare="mean", version="conditions"
):
    scores = load_head_scores(
        model=model_name, version=version, q_span=q_span, compare_with=compare
    )["combined"]
    ranked = [
        (int(i // n_heads), int(i % n_heads))
        for i in np.argsort(scores, axis=None)[::-1]
    ]
    return ranked[:topk]


def resolve_heads_name(spec: str) -> str:
    return f"json:{Path(spec).stem}" if spec.endswith(".json") else spec


def load_json_heads(spec, n_layer, n_heads):
    """Read + validate a ``[[layer, head], ...]`` json head set.

    A hand-supplied head set is the one code path with no schema behind it, so everything is
    checked here rather than failing deep inside the patching hooks (where an out-of-range head
    silently indexes the wrong slice of the o_proj input instead of raising)."""
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"--heads {spec!r}: file not found")
    try:
        arr = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"--heads {spec!r}: not valid json ({e})") from None
    if not isinstance(arr, list) or not arr:
        raise SystemExit(
            f"--heads {spec!r}: expected a non-empty json array of [layer, head] pairs, "
            f"got {type(arr).__name__} of length {len(arr) if hasattr(arr, '__len__') else '?'}"
        )
    heads = []
    for i, item in enumerate(arr):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise SystemExit(f"--heads {spec!r}: entry {i} is {item!r}, want a [layer, head] pair")
        try:
            l, h = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            raise SystemExit(f"--heads {spec!r}: entry {i} = {item!r} is not a pair of ints") from None
        if not (0 <= l < n_layer):
            raise SystemExit(
                f"--heads {spec!r}: entry {i} has layer {l}, outside [0, {n_layer}) for this model"
            )
        if not (0 <= h < n_heads):
            raise SystemExit(
                f"--heads {spec!r}: entry {i} has head {h}, outside [0, {n_heads}) for this model"
            )
        heads.append((l, h))
    dupes = {x for x in heads if heads.count(x) > 1}
    if dupes:
        raise SystemExit(
            f"--heads {spec!r}: duplicate entries {sorted(dupes)[:5]}"
            f"{' …' if len(dupes) > 5 else ''} — a repeated head would be steered twice"
        )
    return heads


def load_head_set(spec, model_name, results_dir, n_heads, goal_threshold=0.9,
                  goal_band=None, n_layer=None):
    """Return list[(layer, head)] for spec in {'goal','gaze','decision', <path>.json}."""
    goal_kw = dict(threshold=goal_threshold, band=goal_band, n_layer=n_layer)
    if spec.endswith(".json"):
        heads = load_json_heads(spec, n_layer, n_heads)
    elif spec == "goal":
        heads = load_goal_heads(model_name, results_dir, **goal_kw)
    elif spec == "decision":
        heads = load_decision_heads(model_name)
    elif spec == "gaze":  # equal-N to goal (matches 09.1)
        heads = load_gaze_heads(
            model_name, n_heads,
            topk=len(load_goal_heads(model_name, results_dir, **goal_kw)),
        )
    else:
        raise SystemExit(
            f"unknown --heads spec: {spec!r} (want goal|gaze|decision|<path>.json). "
            f"A path must end in .json."
        )
    # An empty set is not a no-op experiment: apply_goalscope would run with nothing steered and
    # the summary would look like a real result. Fail instead.
    if not heads:
        raise SystemExit(
            f"--heads {spec!r} selected 0 heads"
            + (f" (threshold {goal_threshold} too high for this mask?)"
               if spec in ("goal", "gaze") else "")
        )
    return heads


# ══════════════════════════════════════════════════════════════════════════════
# Corpus + generation
# ══════════════════════════════════════════════════════════════════════════════
def build_corpus(n_specs, num_conditions, seed, datatype="int"):
    # `predicates_for`, not INT_PREDICATES: the pool must follow `datatype`, otherwise a
    # --datatype=str run silently gets int predicates written against the wrong argument name
    # and only the spec_id reflects the request.
    preds = [p for p in predicates_for(datatype) if p.has_singleline_for("python")]
    specs = sample_specs(
        n_specs, num_conditions, seed=seed, predicates=preds, allow_repeats=False
    )
    for j, sp in enumerate(specs):
        sp.spec_id = f"eval_{datatype}_{j:02d}"
    return materialize(specs, langs=["python"], modularities=["monolithic"])


def build_prompt(mt, sample):
    instr = sample.render_instruction(
        name_rule="letters", multistep=False
    )  # mandates a/b/c
    stub = sample.render_code_stub()  # def check_conditions(n: int):\n
    if supports_chat_template(mt):
        return apply_chat_template(
            mt=mt, user_query=instr.strip(), incomplete_code=stub
        )
    return instr + "\nAnswer:\n" + stub


def generate_completion(mt, prompt, max_new_tokens):
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    return greedy_continue(mt, **inp, max_new_tokens=max_new_tokens)


def rule_based_spans(full_text, gen_char_start, n_conditions):
    """Per-condition impl_span for naming="letters", purely rule-based (no oracle).
    span[k] = code between `<letter_k> =` and the next present `<letter> =` line; for the last
    present letter, up to the terminal print/return. Returns list[str|None] (None if not found)."""
    gen = full_text[gen_char_start:]
    starts = []
    for k in range(n_conditions):
        letter = chr(97 + k)
        m = re.search(
            rf"(?m)^[ \t]*{re.escape(letter)}\s*=(?!=)", gen
        )  # assignment, not `==`
        starts.append(m.start() if m else None)
    present = [s for s in starts if s is not None]
    if not present:
        return [None] * n_conditions
    last_start = max(present)
    mterm = re.search(
        r"(?m)^[ \t]*(?:print|return)\b", gen[last_start:]
    )  # terminal statement
    term = last_start + mterm.start() if mterm else len(gen)
    spans = []
    for k in range(n_conditions):
        s = starts[k]
        if s is None:
            spans.append(None)
            continue
        later = [t for t in starts[k + 1 :] if t is not None and t > s]
        e = min(later) if later else term
        spans.append(gen[s:e].strip())
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# Anchors
# ══════════════════════════════════════════════════════════════════════════════
def _tok_at_char(offsets, cpos):
    for ti, (a, b) in enumerate(offsets):
        if a <= cpos < b:
            return ti
    return None


def line_anchors(full_text, offsets, var, gen_char_start):
    """Token positions of {var, eq, first_impl, indent, prev_end} for `    {var} = ...`."""
    m = re.search(rf"(?m)^([ \t]*){re.escape(var)}\s*=", full_text[gen_char_start:])
    if not m:
        return {}
    base = gen_char_start + m.start()
    indent_c = base
    var_c = base + len(m.group(1))
    eq_c = gen_char_start + m.start() + m.group(0).rfind("=")
    prev_nl = full_text.rfind("\n", 0, base)
    out = {}
    out["var"] = _tok_at_char(offsets, var_c)
    out["eq"] = _tok_at_char(offsets, eq_c)
    out["first_impl"] = (
        out["eq"] + 1
        if out["eq"] is not None and out["eq"] + 1 < len(offsets)
        else None
    )
    out["indent"] = (
        _tok_at_char(offsets, indent_c) if full_text[indent_c] in " \t" else out["var"]
    )
    out["prev_end"] = _tok_at_char(offsets, prev_nl) if prev_nl != -1 else None
    return {k: v for k, v in out.items() if v is not None}


def anchors_for_sample(mt, full_text, prompt, conditions, impl_spans):
    """Per-condition rows {cond, var, condition, impl_span, anchors}. `conditions` and `impl_spans`
    are aligned by index (letter k -> chr(97+k)); source-agnostic (rule- or oracle-derived)."""
    enc = prepare_input(prompts=full_text, tokenizer=mt, return_offsets_mapping=True)
    offsets = [tuple(p) for p in enc.pop("offset_mapping")[0].tolist()]
    gen_char_start = len(prompt)
    rows = []
    for k, (cond_text, impl) in enumerate(zip(conditions, impl_spans)):
        var = chr(97 + k)
        rows.append(
            dict(
                cond=k,
                var=var,
                condition=cond_text,
                impl_span=(impl or "").strip(),
                anchors=line_anchors(full_text, offsets, var, gen_char_start),
            )
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Oracle scoring  (always use_cache=True)
# ══════════════════════════════════════════════════════════════════════════════
# ── faithfulness rubric (NL read-out vs impl) — with the <verdict> tag for robust parsing ──
FAITH_HARD_SYS = """You are grading a lightweight interpretability probe (a patchscope). The DESCRIPTION below was read out of a model's hidden state at a SINGLE token while the model was writing one line of Python CODE that checks some property of its input. Because it is squeezed out of one token under a steering/patching setup, the description is usually NOISY: it may be wrapped in chat scaffolding (like 'Sure, but first let us ...'), contain code fences or half-written code, be cut off mid-sentence, or repeat and loop. Look PAST all of that surface noise and grade only the SUBSTANCE.

The question: does the description surface ANYTHING genuinely relevant to what the CODE is doing — any relevant operation, quantity, sub-step, or property of the input? Pulling even a small correct fragment out of a single token is a SUCCESS, so grade as leniently as possible — EXCEPT that a read-out which has collapsed into a degenerate repeating loop is NOT a success, no matter what tokens the loop contains (see HEAVY_LOOP).

Decide in this order — check MALFORMED and HEAVY_LOOP FIRST, and only then YES / NO / NO_INFO:

  MALFORMED  - the description is contentless: empty, pure punctuation or garbage, or repetition that carries NO relevant meaning.
  HEAVY_LOOP - the description is DOMINATED by a short repeating cycle: a unit of roughly five words/tokens or fewer that repeats over and over (e.g. 'value >= value >= value >= ...', 'len len len len', 'sum sum ( sum % sum == 0' on repeat, 'x % x == x % 0' on repeat). Once you collapse the repetition, little or no coherent description is left. Assign HEAVY_LOOP EVEN IF the looped fragment contains a relevant operator, name, or token — a stuck loop is not a faithful read-out. This takes PRECEDENCE over YES.
  YES        - the description contains SOME intelligible content genuinely relevant to what the code does, however partial, imprecise, or garbled — and it is NOT merely a short repeating loop.
  NO         - the description is coherent and about a specific thing, but that thing is COMPLETELY UNRELATED to what the code does. No conceptual sub-parts have been identified correctly.
  NO_INFO    - the description is coherent but too generic to indicate any specific aspect of the code (e.g. just 'doing a computation on the number').

The following do NOT count against the description (still YES, provided it is not a HEAVY_LOOP):
  - Wrong specifics: a wrong constant, threshold, bound, modulus, operator, or comparison. If the code computes some quantity and compares it to a value, naming the quantity alone is enough and the exact value can be wrong or missing. Example: the code implements `average temperature > 30`; a read-out about `computing the average temperature` is YES even though the `> 30` part is wrong or absent.
  - Wrong implementation nuance: naming a different mechanism for the same idea. Example: the read-out says `sorting with bubble sort` while the code calls .sort() or uses quicksort -> still YES.
  - Partial coverage: naming just one relevant ingredient of a multi-part check counts.

Grade on meaning, not on shared keywords. Output only YES, NO, NO_INFO, MALFORMED, or HEAVY_LOOP.
Put your final answer inside <verdict>{ans}</verdict> tags, with no other text or explanation, to facilitate automated parsing."""

# ── causality rubric (C++ read-out vs Python impl) — cross-language transfer of the *idea* ──
CAUSAL_HARD_SYS = """You are grading a lightweight interpretability probe (a patchscope). The DESCRIPTION below was read out of a model's hidden state at a SINGLE token while the model was writing one line of Python CODE that checks some property of its input. The read-out method is a steering setup on a informationless C++ function stub. Because it is squeezed out of one token, the description is usually NOISY: it may be half-written code, be cut off mid-generation, be syntactically incorrect, or repeat and loop. Look PAST all of that surface noise and grade only the SUBSTANCE.

The question: does the C++ completion surface **ANYTHING GENUINELY RELEVANT** to what the CODE is doing — any relevant operation, quantity, sub-step, or property of the input? Pulling even a small correct fragment should be considered a SUCCESS, so grade leniently — EXCEPT that a completion which has collapsed into a degenerate repeating loop is NOT a success, no matter what tokens the loop contains (see HEAVY_LOOP).

Decide in this order — check MALFORMED and HEAVY_LOOP FIRST, and only then YES / NO / NO_INFO:

  MALFORMED  - the completion is contentless: empty, pure punctuation or garbage, or repetition that carries NO relevant meaning.
  HEAVY_LOOP - the completion is DOMINATED by a short repeating cycle: a unit of roughly five words/tokens or fewer that repeats over and over (e.g. 'value >= value >= value >= ...', '(value(value(value...', 'sum sum ( sum % sum == 0' on repeat, 'value == sum sum sum value ==' on repeat). Once you collapse the repetition, little or no coherent content is left. Assign HEAVY_LOOP EVEN IF the looped fragment contains a relevant operator, name, or token — a stuck loop is not a successful transfer. This takes PRECEDENCE over YES.
  YES        - the completion contains SOME intelligible content genuinely relevant to what the code does, however partial, imprecise, or garbled — and it is NOT merely a short repeating loop.
  NO         - the completion is coherent and about a specific thing, but that thing is COMPLETELY UNRELATED to what the code does. No conceptual sub-parts have been identified correctly.
  NO_INFO    - the completion is coherent but too generic to indicate any specific aspect of the code (e.g. just 'doing a computation on the number').

The following do NOT count against the completion (still YES, provided it is not a HEAVY_LOOP):
  - Wrong specifics: a wrong constant, threshold, bound, modulus, operator, or comparison. If the code computes some quantity and compares it to a value, naming the quantity alone is enough and the exact value can be wrong or missing. Example: the code implements `count(word, "b") > 5`; a c++ completion that counts the number of a `substr` in the `word` should be judged YES, even if the substr != b and/or the threshold is wrong.
  - Partial coverage: naming just one relevant ingredient of a multi-part check counts. For `count(word, "b") > 5`, a c++ completion that only counts the number of `substr` in the `word` is still YES, even if it doesn't check the threshold.
  - Different implementation nuance: different implementation strategies for the same idea. Example: the python code implements nth fibonacci recursively; a c++ completion implements it iteratively. Still YES, because the underlying property is the same.

Finally, grade on meaning, not on shared keywords. Output only YES, NO, NO_INFO, MALFORMED, or HEAVY_LOOP.
Put your final answer inside <verdict>{ans}</verdict> tags, with no other text or explanation, to facilitate automated parsing."""

# GoalScope destination prompt for causality: an information-less C++ function stub. The goal
# encoding is steered in on top of this; the model then continues the C++ body.
CAUSAL_PREFIX = "bool check(auto value){"
# exposed so other read-out mechanisms (eval_act_oracles.py) can pose the *same* instruction
CAUSAL_INSTRUCTION = (
    "You are implementing a simple C++ function that checks for a condition. "
    "Just write the function body, without any comments or explanations. Assume all the necessary "
    "headers are already included."
)

_KWAY_SYS = "You match a description to the closest condition. Output only an integer."


def build_causality_prompt(mt) -> str:
    """C++ function-body destination prompt (assistant turn == CAUSAL_PREFIX)."""
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": CAUSAL_INSTRUCTION},
        {"role": "assistant", "content": CAUSAL_PREFIX},
    ]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        return mt.tokenizer.apply_chat_template(
            messages, chat_template_kwargs={"enable_thinking": False}, **kw
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(messages, **kw)


def _parse_verdict(resp: str) -> str:
    """Prefer the <verdict>TOKEN</verdict> tag; fall back to the last bare verdict token."""
    m = re.search(
        r"<verdict>\s*(HEAVY_LOOP|MALFORMED|NO_INFO|YES|NO)\s*</verdict>", resp, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    hits = re.findall(r"\b(HEAVY_LOOP|MALFORMED|NO_INFO|YES|NO)\b", resp.upper())
    return hits[-1] if hits else "UNPARSED"


# Scaffolding tokens of the causal C++ stub + boolean/return keywords. Stripping these, then the
# comparison/logic/ternary operators, then the ONLY literal the no-goal default compares against
# (``0``), leaves NOTHING iff the read-out is a bare sign/zero comparison (``value > 0``,
# ``value == 0``, ``value < 0``, ``value < 0 ? value : 0``, ``if(value>0) return true; else …``).
# It must NOT flag genuine predicates: a non-zero literal (``value >= 10`` — is_two_digit) or a real
# arithmetic/bitwise op (``value & (value - 1)`` — is_power_of_two, ``value % 2`` — is_even) survives
# and keeps the read-out non-trivial. Hence: only ``&&``/``||`` (logical) are removed — a lone ``&``
# or ``|`` (bitwise), ``%``, ``+``, ``-``, ``*``, ``~``, ``^``, ``.`` or any non-zero digit remains.
_TRIVIAL_READOUT_TOKENS = ("bool", "check", "auto", "value", "val", "return",
                           "if", "else", "true", "false", "nullptr")


def is_trivial_causal_readout(readout: str) -> bool:
    """True iff a causality C++ read-out compares ``value`` to ZERO and does nothing else — the
    sign/zero test the no-goal default emits (see :data:`src.diff.data.TRIVIAL_FRONTIERS`). A
    non-zero constant or any real arithmetic/bitwise op (``%``, ``&``, ``-``, ``.size()``,
    ``max_element``, ``abs`` …) keeps it non-trivial."""
    if not readout or not readout.strip():
        return False
    t = readout
    for tok in _TRIVIAL_READOUT_TOKENS:
        t = re.sub(rf"\b{tok}\b", "", t)
    t = t.replace("&&", "").replace("||", "")            # logical connectives (not bitwise &/|)
    for op in ("==", "!=", ">=", "<=", ">", "<"):        # comparison operators
        t = t.replace(op, "")
    t = re.sub(r"(?<![0-9])0(?![0-9])", "", t)           # the standalone literal 0 (NOT 10, 99, …)
    t = re.sub(r"[?:;(){}\s!]", "", t)                   # ternary / scaffolding punctuation
    return t == ""


def hard_eval(readout, impl_span, mode):
    """dict(verdict, prompt, system, response). verdict in {YES, NO, NO_INFO, MALFORMED, UNPARSED}.
    faithfulness: grade NL read-out vs impl; causality: grade C++ read-out vs the Python impl."""
    hard_sys = CAUSAL_HARD_SYS if mode == "causality" else FAITH_HARD_SYS
    if not readout or not readout.strip():
        return dict(verdict="MALFORMED", prompt=None, system=hard_sys, response=None)
    # Sanitize: a trivial sign/zero read-out is what the model emits with NO goal signal, so it
    # can never count as a causal transfer, no matter what the LLM oracle says. Short-circuit NO.
    if mode == "causality" and is_trivial_causal_readout(readout):
        return dict(verdict="NO", prompt=None, system=hard_sys,
                    response="[sanitizer: trivial sign/zero read-out]")
    if mode == "causality":
        prompt = (
            f"Python (gold): {impl_span}\nC++ (read-out): {readout.strip()}\nAnswer:"
        )
    else:
        prompt = f"Code: {impl_span}\nDescription: {readout.strip()}\nAnswer:"
    # no rubric tag in the user prompt: ask_claude caches on (system_prompt, prompt, max_tokens),
    # and the rubric lives in system_prompt, so a rubric change already invalidates the cache.
    # 512 (not 256): on borderline cases the oracle reasons before the <verdict> tag and can
    # otherwise run out of budget mid-sentence -> UNPARSED.
    resp = ask_claude(
        prompt=prompt,
        system_prompt=hard_sys,
        max_tokens=512,
        use_cache=True,
        model_name=ORACLE_MODEL,
    )
    return dict(
        verdict=_parse_verdict(resp), prompt=prompt, system=hard_sys, response=resp
    )


def kway_assign(readout, conditions):
    """dict(pred, prompt, system, response). pred = matched condition index or None."""
    if not readout or not readout.strip():
        return dict(pred=None, prompt=None, system=_KWAY_SYS, response=None)
    menu = "\n".join(f"{i}. {c}" for i, c in enumerate(conditions))
    prompt = (
        f"Conditions:\n{menu}\n\nDescription: {readout.strip()}\n"
        f"Which condition number does the description best match? Output only the integer, or -1 if none."
    )
    resp = ask_claude(
        prompt=prompt,
        system_prompt=_KWAY_SYS,
        max_tokens=256,
        use_cache=True,
        model_name=ORACLE_MODEL,
    )
    m = re.search(r"-?\d+", resp)
    pred = int(m.group()) if m else None
    if pred is not None and not (0 <= pred < len(conditions)):
        pred = None
    return dict(pred=pred, prompt=prompt, system=_KWAY_SYS, response=resp)


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(records: list[dict], n_samples: int) -> dict:
    """Per-anchor + overall hard_yes (and kway_ok iff k-way records are present)."""
    has_kway = bool(records) and all("kway_pred" in r for r in records)

    def _stats(grp):
        d = dict(n=len(grp), hard_yes=float(np.mean([r["hard"] == "YES" for r in grp])),
                 heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in grp])))
        if has_kway:
            d["kway_ok"] = float(
                np.mean([r["kway_pred"] == r["kway_true"] for r in grp])
            )
        return d

    by_anchor = {
        a: _stats(grp)
        for a in ANCHOR_ORDER
        if (grp := [r for r in records if r["anchor"] == a])
    }
    if records:
        overall = dict(hard_yes=float(np.mean([r["hard"] == "YES" for r in records])),
                       heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in records])))
        if has_kway:
            overall["kway_ok"] = float(
                np.mean([r["kway_pred"] == r["kway_true"] for r in records])
            )
    else:
        overall = dict(hard_yes=float("nan"))
    return dict(
        n_records=len(records),
        n_samples=n_samples,
        by_anchor=by_anchor,
        overall=overall,
    )


def log_summary(summary: dict, mode: str = "faithfulness"):
    metric = "causality" if mode == "causality" else "faithfulness"
    has_kway = any("kway_ok" in m for m in summary["by_anchor"].values())
    logger.info(
        f"per-anchor {metric} (hard_yes" + (" | kway_ok):" if has_kway else "):")
    )
    for aname in ANCHOR_ORDER:
        if aname in summary["by_anchor"]:
            m = summary["by_anchor"][aname]
            k = f"  kway_ok={m['kway_ok']:.3f}" if "kway_ok" in m else ""
            hl = f"  hvy_loop={m['heavy_loop']:.3f}" if "heavy_loop" in m else ""
            logger.info(f"  {aname:10s} n={m['n']:<4d} hard_yes={m['hard_yes']:.3f}{hl}{k}")
    o = summary["overall"]
    k = f"  kway_ok={o['kway_ok']:.3f}" if "kway_ok" in o else ""
    hl = f"  hvy_loop={o['heavy_loop']:.3f}" if "heavy_loop" in o else ""
    logger.info(
        f"  {'OVERALL':10s} n={summary['n_records']:<4d} hard_yes={o['hard_yes']:.3f}{hl}{k}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        default="faithfulness",
        choices=["faithfulness", "causality"],
        help="faithfulness: NL read-out vs impl | causality: C++ transfer of the idea",
    )
    ap.add_argument(
        "--eval-k-way",
        dest="eval_k_way",
        action="store_true",
        help="also run the k-way oracle (which condition does the read-out name?). "
        "Default off. Not implemented for --mode causality yet.",
    )
    # ── GOALSCOPE_CONFIG ──
    ap.add_argument(
        "--heads", default="goal", help="goal | gaze | decision | <path>.json"
    )
    ap.add_argument(
        "--goal_threshold", type=float, default=None,
        help="mask-value cutoff for --heads goal (default 0.9). Default is None rather than 0.9 so "
        "that passing it explicitly is detectable: it is meaningless for a .json head set and "
        "combining the two is rejected rather than silently ignored.",
    )
    ap.add_argument(
        "--goal_band", type=int, nargs=2, default=None, metavar=("START", "END"),
        help="layer band the goal heads are read from; default is per-model (GOAL_BANDS). "
        "Passing this explicitly also adds band=START-END to the run dir, so band sweeps "
        "do not overwrite each other.",
    )
    ap.add_argument(
        "--strategy", default="additive",
        choices=["additive", "additive_norm", "replace"],
        help="additive_norm makes the injection norm-relative "
        "(h + alpha*||h||*g/||g||) so alpha does not carry over from additive; "
        "strategy is a run-dir path component, so the two never collide.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="default 0.2 (additive) / 1.0 (replace)",
    )
    ap.add_argument(
        "--steer_layer_range",
        type=int,
        nargs=2,
        default=[35, 46],
        help="additive only: layers range(a, b) receive the injection",
    )
    ap.add_argument("--n_steer_tokens", default="all", help="'all' | int")
    ap.add_argument(
        "--max_new_tokens", type=int, default=64, help="read-out length per anchor"
    )
    # ── corpus / model ──
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--datatype", default="int")
    ap.add_argument("--n_specs", type=int, default=256)
    ap.add_argument("--num_conditions", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--gen_max_new_tokens", type=int, default=256, help="body generation budget"
    )
    ap.add_argument(
        "--span_source",
        default="rule",
        choices=["rule", "oracle"],
        help="impl-span labelling: 'rule' (letters-based, no oracle) | 'oracle' (label_generation)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe an existing run dir and redo it; default is to RESUME from the last "
        "completed sample (same args => same run dir)",
    )
    logging_utils.add_logging_args(ap)  # -v (DEBUG) / -q (WARNING); default INFO
    args = ap.parse_args()

    # A .json head set is the complete specification of which heads to steer. Anything that would
    # also select heads is therefore rejected — even when it happens to agree — so a run dir can
    # never claim a selection knob that had no effect on the heads actually used.
    if args.heads.endswith(".json"):
        conflicting = [
            name for name, val in (("--goal_threshold", args.goal_threshold),
                                   ("--goal_band", args.goal_band)) if val is not None
        ]
        if conflicting:
            ap.error(
                f"{', '.join(conflicting)} cannot be combined with --heads {args.heads!r}: a json "
                f"head set supersedes every head-selection flag. Drop it, or bake the filtering "
                f"into the json."
            )
    if args.goal_threshold is None:
        args.goal_threshold = 0.9  # materialise the documented default for the mask-based specs
    return args


def main():
    args = parse_args()
    logging_utils.configure(args)  # -v => DEBUG: every oracle prompt/verdict logged
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    alpha = (
        args.alpha
        if args.alpha is not None
        # additive_norm has different units, so it gets no inherited default
        else (0.2 if args.strategy == "additive" else 1.0)
    )
    n_steer = (
        "all" if str(args.n_steer_tokens).lower() == "all" else int(args.n_steer_tokens)
    )
    heads_name = resolve_heads_name(args.heads)

    if args.mode == "causality" and args.eval_k_way:
        raise SystemExit("--eval-k-way is not implemented for --mode causality yet")
    hard_sys = CAUSAL_HARD_SYS if args.mode == "causality" else FAITH_HARD_SYS

    # config (sans selected_heads) — enough to resolve the run dir + skip check before loading the model
    config = dict(
        mode=args.mode,
        eval_k_way=args.eval_k_way,
        heads_spec=args.heads,
        heads_name=heads_name,
        goal_threshold=args.goal_threshold,
        # resolved below once mt.n_layer is known if it is neither given nor registered
        goal_band=list(args.goal_band) if args.goal_band else None,
        goal_band_explicit=args.goal_band is not None,  # -> band=A-B in the run dir
        strategy=args.strategy,
        alpha=alpha,
        steer_layer_range=list(args.steer_layer_range),
        n_steer_tokens=n_steer,
        max_new_tokens=args.max_new_tokens,
        model=args.model,
        datatype=args.datatype,
        n_specs=args.n_specs,
        num_conditions=list(args.num_conditions),
        seed=args.seed,
        gen_max_new_tokens=args.gen_max_new_tokens,
        span_source=args.span_source,
        anchors=ANCHOR_ORDER,
        oracle_model=ORACLE_MODEL,
        readout_prefix=(CAUSAL_PREFIX if args.mode == "causality" else ""),
        hard_rubric_tag=hashlib.md5(hard_sys.encode()).hexdigest()[:8],
    )
    run_dir = fio.run_dir_for(config, results_dir)

    # ── overwrite vs resume ──
    if args.overwrite and run_dir.exists():
        for f in ("records.jsonl", "samples.jsonl", "samples.json", "summary.json"):
            (run_dir / f).unlink(missing_ok=True)
        logger.info(f"--overwrite: cleared existing run at {run_dir}")
    prev = None if args.overwrite else fio.read_config(run_dir)
    # The run dir is keyed on `json:<stem>` for a json head set, so two different files with the
    # same basename (results/a/heads.json, results/b/heads.json) resolve to the SAME dir and the
    # second invocation would silently resume the first one's samples under a different head set.
    # Refuse rather than mix. The selected_heads check below catches an edited file at the same path.
    if prev is not None and prev.get("heads_spec") not in (None, args.heads):
        raise SystemExit(
            f"run dir {run_dir} already holds a run with --heads {prev['heads_spec']!r}, but this "
            f"invocation passed {args.heads!r}. Two head sets sharing a run name resume each "
            f"other's samples — rename the json, or pass --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run dir -> {run_dir}")

    # corpus is deterministic from (n_specs, num_conditions, seed, datatype) and needs no model,
    # so we can compute the full sample set up front and figure out what's already done.
    samples = build_corpus(args.n_specs, args.num_conditions, args.seed, args.datatype)
    completed = fio.read_completed_spec_ids(run_dir)
    kept = fio.prune_orphan_records(
        run_dir, completed
    )  # drop rows from a half-finished sample
    remaining = [s for s in samples if s.spec.spec_id not in set(completed)]
    logger.info(
        f"corpus: {len(samples)} samples | already done: {len(completed)} "
        f"({kept} records kept) | remaining: {len(remaining)}"
    )

    if not remaining:
        # nothing to compute — just (re)build the summary from what's on disk and exit.
        recs = (
            [
                json.loads(l)
                for l in fio.records_path(run_dir).read_text().splitlines()
                if l.strip()
            ]
            if fio.records_path(run_dir).exists()
            else []
        )
        summary = compute_summary(recs, n_samples=len(completed))
        fio.write_summary(run_dir, summary)
        logger.info(f"all samples already complete; summary refreshed -> {run_dir}")
        log_summary(summary, args.mode)
        return

    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    n_heads = mt.model.config.num_attention_heads

    heads = load_head_set(
        args.heads, args.model, results_dir, n_heads,
        goal_threshold=args.goal_threshold, goal_band=args.goal_band, n_layer=mt.n_layer,
    )
    if args.heads == "goal":  # record the band actually used, resolved or not
        band = tuple(args.goal_band) if args.goal_band else default_goal_band(
            args.model, n_layer=mt.n_layer
        )
        config["goal_band"] = list(band)
        logger.info(
            f"goal heads: band {band[0]}-{band[1] - 1} @ >{args.goal_threshold} -> {len(heads)} heads"
        )
    config["n_heads"] = len(heads)
    config["selected_heads"] = [[int(l), int(h)] for l, h in heads]
    # Same-name/different-content case: the json at this path (or the mask behind `goal`) changed
    # since the samples already on disk were produced. Resuming would splice two head sets into one
    # summary, and nothing downstream could tell.
    if prev is not None and prev.get("selected_heads") is not None:
        old = [[int(l), int(h)] for l, h in prev["selected_heads"]]
        if old != config["selected_heads"] and fio.read_completed_spec_ids(run_dir):
            raise SystemExit(
                f"run dir {run_dir} holds {len(fio.read_completed_spec_ids(run_dir))} samples "
                f"produced with a DIFFERENT head set ({len(old)} heads, now {len(heads)}; "
                f"{len(set(map(tuple, old)) ^ set(map(tuple, config['selected_heads'])))} differ). "
                f"Resuming would mix them — pass --overwrite to redo the run."
            )
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(
        run_dir, config
    )  # persist config now so an in-progress run is self-describing
    logger.info(f"heads[{heads_name}] = {len(heads)} (layer,head) pairs")

    gs_kwargs = dict(
        selected_heads=heads,
        strategy=args.strategy,
        alpha=alpha,
        steer_layer_range=tuple(args.steer_layer_range),
        n_steer_tokens=n_steer,
        max_new_tokens=args.max_new_tokens,
    )
    # causality: steer into the C++ stub prompt; faithfulness: apply_goalscope's default NL probe
    if args.mode == "causality":
        gs_kwargs["goalscope_prompt"] = build_causality_prompt(mt)
    readout_prefix = CAUSAL_PREFIX if args.mode == "causality" else ""

    # append-mode handles both a fresh run and a resume (existing kept rows/samples are preserved)
    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    n_done = len(completed)
    for si, sample in enumerate(remaining):
        sample_records = []
        prompt = build_prompt(mt, sample)
        completion = generate_completion(mt, prompt, args.gen_max_new_tokens)
        full_text = prompt + completion

        # impl-span + condition text, either rule-based (letters, no oracle) or oracle-labelled
        if args.span_source == "rule":
            conditions = list(sample.spec.spec_texts)  # ground-truth descriptions
            impl_spans = rule_based_spans(
                full_text, len(prompt), len(sample.spec.predicate_ids)
            )
            labels_meta = [
                dict(step_text=c, gen_code=(s or ""), gen_token_range=None)
                for c, s in zip(conditions, impl_spans)
            ]
        else:  # oracle
            labeled = label_generation(mt, sample, prompt, completion, use_cache=True)
            conditions = [sl.step_text for sl in labeled.labels]
            impl_spans = [sl.gen_code.strip() for sl in labeled.labels]
            labels_meta = [
                dict(
                    step_text=sl.step_text,
                    gen_code=sl.gen_code,
                    gen_token_range=list(sl.gen_token_range)
                    if sl.gen_token_range
                    else None,
                )
                for sl in labeled.labels
            ]

        rows = anchors_for_sample(mt, full_text, prompt, conditions, impl_spans)
        conds = [r["condition"] for r in rows]
        apos = sorted({p for r in rows for p in r["anchors"].values()})
        if not apos:
            logger.warning(f"[{sample.spec.spec_id}] no anchors found — skipping")
            continue
        res = apply_goalscope(
            full_text,
            mt=mt,
            show=False,
            include_noncode=False,
            token_positions=apos,
            **gs_kwargs,
        )
        readouts = {r["pos"]: r for r in res}

        for row in rows:
            true_id = sample.spec.predicate_ids[row["cond"]]
            for aname, pos in row["anchors"].items():
                # causality prepends the C++ stub so the oracle sees a complete-ish snippet
                ro = readout_prefix + readouts.get(pos, {}).get("patched_gen", "")
                hv = hard_eval(ro, row["impl_span"], args.mode)
                logger.debug(
                    "[%s c%d %s] impl=%r cond=%r",
                    sample.spec.spec_id,
                    row["cond"],
                    aname,
                    row["impl_span"],
                    row["condition"],
                )
                logger.debug("    read-out : %r", ro)
                logger.debug(
                    "    HARD  verdict=%s  oracle_response=%r",
                    hv["verdict"],
                    hv["response"],
                )
                rec = dict(
                    spec=sample.spec.spec_id,
                    cond=row["cond"],
                    var=row["var"],
                    anchor=aname,
                    pos=pos,
                    readout=ro,
                    impl_span=row["impl_span"],
                    condition=row["condition"],
                    true_pred_id=true_id,
                    hard=hv["verdict"],
                    hard_prompt=hv["prompt"],
                    hard_response=hv["response"],
                )
                if args.eval_k_way:
                    kv = kway_assign(ro, conds)
                    logger.debug(
                        "    KWAY  pred=%s (true %d)  oracle_response=%r",
                        kv["pred"],
                        row["cond"],
                        kv["response"],
                    )
                    rec.update(
                        kway_pred=kv["pred"],
                        kway_true=row["cond"],
                        kway_prompt=kv["prompt"],
                        kway_response=kv["response"],
                    )
                sample_records.append(rec)
        sample_meta = dict(
            spec_id=sample.spec.spec_id,
            predicate_ids=list(sample.spec.predicate_ids),
            prompt=prompt,
            completion=completion,
            labels=labels_meta,
            anchors=[
                dict(cond=r["cond"], var=r["var"], anchors=r["anchors"]) for r in rows
            ],
        )
        # write this sample's records FIRST, then its meta line: a sample counts as "done"
        # only once its meta line lands, so a crash mid-sample leaves recoverable orphan
        # records (pruned on the next resume) rather than a phantom-complete sample.
        for r in sample_records:
            fio.append_jsonl(rec_fh, r)
        fio.append_jsonl(samp_fh, sample_meta)
        n_done += 1
        logger.info(
            f"[{n_done}/{len(samples)}] {sample.spec.spec_id}: "
            f"{sum(len(r['anchors']) for r in rows)} anchor read-outs scored"
        )

    rec_fh.close()
    samp_fh.close()

    # ── aggregate from everything on disk (resumed rows included) ──
    all_records = [
        json.loads(l)
        for l in fio.records_path(run_dir).read_text().splitlines()
        if l.strip()
    ]
    n_samples = len(fio.read_completed_spec_ids(run_dir))
    summary = compute_summary(all_records, n_samples)
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {run_dir}")
    log_summary(summary, args.mode)


if __name__ == "__main__":
    main()
