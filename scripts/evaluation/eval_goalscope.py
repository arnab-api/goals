#!/usr/bin/env python
"""GoalScope goal-head eval on the LOCKED prefill-to-frontier dataset (QTASKS bank) — two ``--mode``s:

  faithfulness  — read out an NL description at each pivot ("what are you doing at this line?")
                  and grade whether it faithfully describes the frontier implementation span.
  causality     — steer the goal encoding into an information-less C++ stub
                  (``bool check(auto value){``) and grade whether the C++ continuation implements
                  the same *idea* as the Python frontier span (cross-language transfer).

Corpus: ONE frozen dataset (``data/goalscope_eval.json`` — QTASKS bank, prefill-to-frontier, ``var_eq``
stub, letters naming, trivial frontiers excluded, balanced frontier coverage), all four datatypes
aggregated into one 253-sample run. No per-run random sampling.

Pipeline (mirrors the OVReader skyline in ``scripts/evaluation/eval_ov_reader.py``):

  1. Corpus — load the recipes from the locked JSON; rebuild each ``QTransferInstance`` from
     ``src.nrgs.data.QTASKS`` by predicate id.
  2. Prompt — prefilled source prompt ending at the frontier's ``{var} =`` (var_eq stub); greedy-decode.
  3. impl_span — the frontier's ``{var} =`` line + the first generated line (single-line predicate,
     so exact — no oracle span-labelling).
  4. Pivots — locate {prev_end, indent, var, eq, first_impl} in the GoalScope tokenization.
  5. Read-outs — ``apply_goalscope`` restricted to just those pivot positions.
  6. Scoring — ``hard_eval`` (oracle YES/NO; causality is sanitized for trivial read-outs).

Saved under ``<results>/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/``.

All ``ask_claude`` calls use ``use_cache=True``.

Usage (no flags = the paper's main configuration: Qwen2.5-Coder-32B-Instruct, the 122 goal heads,
additive_norm, alpha 0.1, steer layers 35-46, faithfulness):
    conda run -n goals python scripts/evaluation/eval_goalscope.py
    conda run -n goals python scripts/evaluation/eval_goalscope.py --mode causality

    # model axis (replace strategy):  --strategy replace --alpha 1.0 [--model ...]
    # custom head set (supersedes --goal_band):  --heads /path/to/heads.json
    # smoke:  --limit 8   (separate run dir: nspec=8)
"""

import argparse
import functools
import dataclasses
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.append(".")

import numpy as np
import torch

import src.nrgs.data as DATA
from scripts.evaluation import faithfulness_io as fio
from src.goals.goalscope import (  # noqa: F401  (CAUSAL_PREFIX re-exported)
    CAUSAL_PREFIX,
    _build_causality_prompt,
    apply_goalscope,
    greedy_continue,
)
from src.goals.head_sets import (  # noqa: F401  (re-exported)
    default_goal_band,
    load_goal_heads,
    load_json_heads,
    resolve_heads_name,
)
from src.models import ModelandTokenizer
from src.tokens import chat_prompt, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.oracle_llms import JudgeQueue, ask_claude

DEFAULT_DATASET = "data/goalscope_eval.json"
ALL_DATATYPES = ["int", "list[int]", "str", "list[str]"]
GOAL_THRESHOLD = 0.9  # mask-value cutoff for --heads goal (goal_heads_path builds ..._thr0.9_... from it)
MAX_NEW_TOKENS = 64  # read-out length per pivot
GEN_MAX_NEW_TOKENS = 64  # frontier-body greedy budget

logger = logging.getLogger(__name__)  # configured in main() via logging_utils

ANCHOR_ORDER = ["prev_end", "indent", "var", "eq", "first_impl"]
ORACLE_MODEL = "claude-sonnet-4-6"


def load_head_set(spec, model_name, results_dir, n_heads, goal_band=None, n_layer=None):
    """Return list[(layer, head)] for spec in {'goal', <path>.json}."""
    if spec.endswith(".json"):
        heads = load_json_heads(spec, n_layer, n_heads)
    elif spec == "goal":
        heads = load_goal_heads(
            model_name, results_dir, threshold=GOAL_THRESHOLD, band=goal_band, n_layer=n_layer
        )
    else:
        raise SystemExit(f"unknown --heads spec: {spec!r} (want goal|<path>.json). A path must end in .json.")
    # An empty set is not a no-op experiment: apply_goalscope would run with nothing steered and
    # the summary would look like a real result. Fail instead.
    if not heads:
        raise SystemExit(
            f"--heads {spec!r} selected 0 heads"
            + (
                f" (threshold {GOAL_THRESHOLD} too high for this mask?)"
                if spec == "goal"
                else ""
            )
        )
    return heads


# ══════════════════════════════════════════════════════════════════════════════
# Corpus — LOCKED prefill-to-frontier dataset (QTASKS); the corpus / prompt / pivot helpers are
# shared with eval_ov_reader, eval_readout_sources and ablation_common.
# ══════════════════════════════════════════════════════════════════════════════
def load_locked_datapoints(dataset_path, datatypes):
    """Rebuild ``QTransferInstance``s from the frozen JSON recipes (QTASKS by id), for one or more
    datatypes AGGREGATED into a single corpus. ``datatypes`` may be a str or a list. Returns
    list[(spec_id, instance)] in dataset order (spec_ids are datatype-prefixed, so unique across
    datatypes). No RNG — the dataset IS the corpus."""
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    data = json.loads(Path(dataset_path).read_text())
    out = []
    for dt in datatypes:
        recs = data["datapoints"].get(dt)
        if not recs:
            raise SystemExit(
                f"dataset {dataset_path!r} has no datapoints for datatype {dt!r} "
                f"(have {sorted(data['datapoints'])})"
            )
        for r in recs:
            bank = DATA.QTASKS[r["datatype"]]
            by_id = {p.id: p for p in bank.predicates}
            subset = tuple(by_id[pid] for pid in r["predicate_ids"])
            task = dataclasses.replace(
                bank, id=f"{bank.id}_T{r['T']}", predicates=subset
            )
            inst = DATA.QTransferInstance(
                task=task,
                var_name=bank.var_names[0],
                presentation=r["presentation"],
                src_lang=r["lang"],
                dst_lang=r["lang"],
                src_progress=list(r["progress_ids"]),
                dst_progress=list(r["progress_ids"]),
                expected_next=r["frontier_id"],
                naming=r["naming"],
                meta={"datatype": r["datatype"], "T": r["T"], "j": r["j"]},
            )
            out.append((r["spec_id"], inst))
    return out, data.get("meta", {})


def limit_round_robin(datapoints, limit):
    """SMOKE: the first ``limit`` datapoints taken round-robin across datatypes, so a small N hits
    every prompt builder. Changes n_specs -> a separate run dir."""
    groups = {}
    for sid, inst in datapoints:
        groups.setdefault(inst.meta["datatype"], []).append((sid, inst))
    picked, i = [], 0
    while len(picked) < limit and any(i < len(v) for v in groups.values()):
        for v in groups.values():
            if i < len(v) and len(picked) < limit:
                picked.append(v[i])
        i += 1
    logger.warning(
        f"--limit {limit}: SMOKE run on {len(picked)} datapoints "
        f"({sorted({inst.meta['datatype'] for _, inst in picked})})"
    )
    return picked


def _chat_prompt(mt, question, prefill):
    """Replicate src.nrgs.functional.get_prompt without depending on its module globals."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant proficient in programming. "
            "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    return chat_prompt(mt, messages, add_generation_prompt=False, continue_final_message=True)


def build_prompt(mt, inst):
    """Prefilled (Python) source prompt ending at the frontier's ``{var} =`` (var_eq stub, letters)."""
    q, pf = inst.src_parts()
    q = q.replace(
        "computes:", "computes in sequential order:"
    )  # nudge the model to follow the listing
    var = inst.var_of(inst.expected_next)
    body = pf.rstrip("\n ")  # drop the trailing indent line
    pf2 = body + f"\n    {var} ="
    return _chat_prompt(mt, q, DATA._finalize_prefill(pf2)), var


def generate_completion(mt, prompt, max_new_tokens):
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    return greedy_continue(mt, **inp, max_new_tokens=max_new_tokens)


def frontier_impl_span(prompt, completion, var):
    """The frontier condition's code = its ``{var} =`` line + the first generated line (single-line
    predicate, so this is exact — no oracle span-labelling needed)."""
    last_prompt_line = prompt.rsplit("\n", 1)[-1]  # "    c =" | "    auto c ="
    return (last_prompt_line + completion.split("\n", 1)[0]).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Pivots (== the old "anchors", same five names)
# ══════════════════════════════════════════════════════════════════════════════
def pivot_positions(mt, prompt, full_text, pivots):
    """{pivot: absolute token pos}. prev_end/indent/var/eq = the last four tokens of the prompt;
    first_impl = the first generated token. Positions index the tokenization of ``full_text``."""
    prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    full = prepare_input(prompts=full_text, tokenizer=mt)["input_ids"][0].tolist()
    P = len(prompt_ids)
    if (
        full[:P] != prompt_ids
    ):  # boundary token merged on re-tokenisation — fall back to common prefix
        P = next(
            (
                i
                for i in range(min(len(full), len(prompt_ids)))
                if full[i] != prompt_ids[i]
            ),
            min(len(full), len(prompt_ids)),
        )
    idx = {
        "prev_end": P - 4,
        "indent": P - 3,
        "var": P - 2,
        "eq": P - 1,
        "first_impl": P,
    }
    return {p: idx[p] for p in pivots if 0 <= idx[p] < len(full)}


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

def _parse_verdict(resp: str) -> str:
    """Prefer the <verdict>TOKEN</verdict> tag; fall back to the last bare verdict token."""
    m = re.search(
        r"<verdict>\s*(HEAVY_LOOP|MALFORMED|NO_INFO|YES|NO)\s*</verdict>",
        resp,
        re.IGNORECASE,
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
_TRIVIAL_READOUT_TOKENS = (
    "bool",
    "check",
    "auto",
    "value",
    "val",
    "return",
    "if",
    "else",
    "true",
    "false",
    "nullptr",
)


def is_trivial_causal_readout(readout: str) -> bool:
    """True iff a causality C++ read-out compares ``value`` to ZERO and does nothing else — the
    sign/zero test the no-goal default emits (see :data:`src.nrgs.data.TRIVIAL_FRONTIERS`). A
    non-zero constant or any real arithmetic/bitwise op (``%``, ``&``, ``-``, ``.size()``,
    ``max_element``, ``abs`` …) keeps it non-trivial."""
    if not readout or not readout.strip():
        return False
    t = readout
    for tok in _TRIVIAL_READOUT_TOKENS:
        t = re.sub(rf"\b{tok}\b", "", t)
    t = t.replace("&&", "").replace("||", "")  # logical connectives (not bitwise &/|)
    for op in ("==", "!=", ">=", "<=", ">", "<"):  # comparison operators
        t = t.replace(op, "")
    t = re.sub(
        r"(?<![0-9])0(?![0-9])", "", t
    )  # the standalone literal 0 (NOT 10, 99, …)
    t = re.sub(r"[?:;(){}\s!]", "", t)  # ternary / scaffolding punctuation
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
        return dict(
            verdict="NO",
            prompt=None,
            system=hard_sys,
            response="[sanitizer: trivial sign/zero read-out]",
        )
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


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(records: list[dict], n_samples: int) -> dict:
    """Per-anchor + overall hard_yes / heavy_loop."""

    def _stats(grp):
        return dict(
            n=len(grp),
            hard_yes=float(np.mean([r["hard"] == "YES" for r in grp])),
            heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in grp])),
        )

    by_anchor = {
        a: _stats(grp)
        for a in ANCHOR_ORDER
        if (grp := [r for r in records if r["anchor"] == a])
    }
    if records:
        overall = dict(
            hard_yes=float(np.mean([r["hard"] == "YES" for r in records])),
            heavy_loop=float(np.mean([r["hard"] == "HEAVY_LOOP" for r in records])),
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
    logger.info(f"per-anchor {mode} (hard_yes):")
    for aname in ANCHOR_ORDER:
        if aname in summary["by_anchor"]:
            m = summary["by_anchor"][aname]
            logger.info(
                f"  {aname:10s} n={m['n']:<4d} hard_yes={m['hard_yes']:.3f}  hvy_loop={m['heavy_loop']:.3f}"
            )
    o = summary["overall"]
    hl = f"  hvy_loop={o['heavy_loop']:.3f}" if "heavy_loop" in o else ""
    logger.info(
        f"  {'OVERALL':10s} n={summary['n_records']:<4d} hard_yes={o['hard_yes']:.3f}{hl}"
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument(
        "--heads", default="goal", help="goal (the thresholded goal-head mask) | <path>.json"
    )
    ap.add_argument(
        "--goal_band",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="layer band the goal heads are read from; default is per-model (GOAL_BANDS). "
        "Passing this explicitly also adds band=START-END to the run dir.",
    )
    ap.add_argument(
        "--strategy",
        default="additive_norm",
        choices=["additive_norm", "replace"],
        help="additive_norm: h + alpha*||h||*g/||g|| at --steer_layer_range | replace: swap the "
        "goal heads' o_proj inputs (model axis). strategy is a run-dir path component.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="additive_norm: added fraction of ||h|| | replace: source-side weight (1.0)",
    )
    ap.add_argument(
        "--steer_layer_range",
        type=int,
        nargs=2,
        default=[35, 46],
        help="layers range(a, b) receive the additive_norm injection. Also a run-dir component "
        "(layers=a-b) for --strategy replace, where it has no effect.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="torch/global seed for generation determinism (the corpus is fixed by the dataset)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="SMOKE ONLY: cap the corpus to N datapoints (round-robin across datatypes). Changes "
        "n_specs -> a separate run dir; not for real runs.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe an existing run dir and redo it; default is to RESUME from the last "
        "completed sample (same args => same run dir)",
    )
    logging_utils.add_logging_args(ap)  # -v (DEBUG) / -q (WARNING); default INFO
    args = ap.parse_args()

    # A .json head set is the complete specification of which heads to steer, so a run dir can
    # never claim a selection knob that had no effect on the heads actually used.
    if args.heads.endswith(".json") and args.goal_band is not None:
        ap.error(
            f"--goal_band cannot be combined with --heads {args.heads!r}: a json head set "
            f"supersedes it. Drop it, or bake the filtering into the json."
        )
    return args


def main():
    args = parse_args()
    logging_utils.configure(args)  # -v => DEBUG: every oracle prompt/verdict logged
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    heads_name = resolve_heads_name(args.heads)
    hard_sys = CAUSAL_HARD_SYS if args.mode == "causality" else FAITH_HARD_SYS

    # ── LOCKED corpus: load up front (no model needed) so the run dir + skip check resolve first ──
    datapoints, ds_meta = load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)
    if args.limit:
        datapoints = limit_round_robin(datapoints, args.limit)

    # config (sans selected_heads) — enough to resolve the run dir + skip check before loading the model
    config = dict(
        mode=args.mode,
        heads_spec=args.heads,
        heads_name=heads_name,
        goal_threshold=GOAL_THRESHOLD,
        # resolved below once mt.n_layer is known if it is neither given nor registered
        goal_band=list(args.goal_band) if args.goal_band else None,
        goal_band_explicit=args.goal_band is not None,  # -> band=A-B in the run dir
        strategy=args.strategy,
        alpha=args.alpha,
        steer_layer_range=list(args.steer_layer_range),
        n_steer_tokens="all",
        max_new_tokens=MAX_NEW_TOKENS,
        model=args.model,
        datatype="all",  # one run over the full aggregated corpus
        datatypes=list(ALL_DATATYPES),
        n_specs=len(datapoints),
        num_conditions=list(ds_meta["n_conditions"]),
        seed=ds_meta["seed"],
        gen_max_new_tokens=GEN_MAX_NEW_TOKENS,
        span_source="frontier",  # prefill-to-frontier: the leaf's span=frontier
        dataset=DEFAULT_DATASET,
        bank="QTASKS",
        anchors=ANCHOR_ORDER,
        oracle_model=ORACLE_MODEL,
        readout_prefix=(CAUSAL_PREFIX if args.mode == "causality" else ""),
        hard_rubric_tag=hashlib.md5(hard_sys.encode()).hexdigest()[:8],
    )
    run_dir = fio.run_dir_for(config, results_dir)

    # ── overwrite vs resume ──
    if args.overwrite and run_dir.exists():
        for f in ("records.jsonl", "samples.jsonl", "summary.json"):
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

    # corpus is the LOCKED dataset (no RNG) — already loaded; figure out what's done.
    completed = set(fio.read_completed_spec_ids(run_dir))
    kept = fio.prune_orphan_records(
        run_dir, completed
    )  # drop rows from a half-finished sample
    remaining = [(sid, inst) for sid, inst in datapoints if sid not in completed]
    logger.info(
        f"corpus: {len(datapoints)} datapoints (locked {DEFAULT_DATASET}) | "
        f"already done: {len(completed)} ({kept} records kept) | remaining: {len(remaining)}"
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
        args.heads,
        args.model,
        results_dir,
        n_heads,
        goal_band=args.goal_band,
        n_layer=mt.n_layer,
    )
    if args.heads == "goal":  # record the band actually used, resolved or not
        band = (
            tuple(args.goal_band)
            if args.goal_band
            else default_goal_band(args.model, n_layer=mt.n_layer)
        )
        config["goal_band"] = list(band)
        logger.info(
            f"goal heads: band {band[0]}-{band[1] - 1} @ >{GOAL_THRESHOLD} -> {len(heads)} heads"
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
        alpha=args.alpha,
        steer_layer_range=tuple(args.steer_layer_range),
        n_steer_tokens="all",
        max_new_tokens=MAX_NEW_TOKENS,
    )
    # causality: steer into the C++ stub prompt; faithfulness: apply_goalscope's default NL probe
    if args.mode == "causality":
        gs_kwargs["goalscope_prompt"] = _build_causality_prompt(mt)
    readout_prefix = CAUSAL_PREFIX if args.mode == "causality" else ""

    # append-mode handles both a fresh run and a resume (existing kept rows/samples are preserved)
    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    n_done = len(completed)
    n_total = len(datapoints)
    jq = JudgeQueue()  # scores a sample's read-outs while the next sample generates

    def finish_sample(sid, inst, pivs, ros, impl_span, frontier_desc, sample_meta, verdicts):
        nonlocal n_done
        sample_records = []
        for (aname, pos), hv in zip(pivs.items(), verdicts):
            ro = ros[aname]
            logger.debug(
                "[%s %s] impl=%r frontier=%r", sid, aname, impl_span, frontier_desc
            )
            logger.debug("    read-out : %r", ro)
            logger.debug(
                "    HARD  verdict=%s  oracle_response=%r",
                hv["verdict"],
                hv["response"],
            )
            sample_records.append(
                dict(
                    spec=sid,
                    datatype=inst.meta[
                        "datatype"
                    ],  # per-record tag -> per-datatype breakdown of an aggregated run
                    anchor=aname,
                    pos=pos,
                    readout=ro,
                    impl_span=impl_span,
                    frontier_id=inst.expected_next,
                    condition=frontier_desc,
                    hard=hv["verdict"],
                    hard_prompt=hv["prompt"],
                    hard_response=hv["response"],
                )
            )
        # write this sample's records FIRST, then its meta line: a sample counts as "done"
        # only once its meta line lands, so a crash mid-sample leaves recoverable orphan
        # records (pruned on the next resume) rather than a phantom-complete sample.
        for r in sample_records:
            fio.append_jsonl(rec_fh, r)
        fio.append_jsonl(samp_fh, sample_meta)
        n_done += 1
        logger.info(f"[{n_done}/{n_total}] {sid}: {len(pivs)} pivot read-outs scored")

    for sid, inst in remaining:
        prompt, var = build_prompt(mt, inst)
        completion = generate_completion(mt, prompt, GEN_MAX_NEW_TOKENS)
        full_text = prompt + completion

        # frontier impl_span is exact (single-line predicate) — no oracle span-labelling
        impl_span = frontier_impl_span(prompt, completion, var)
        frontier_desc = inst.expected_next_desc

        pivs = pivot_positions(mt, prompt, full_text, ANCHOR_ORDER)
        if not pivs:
            logger.warning(f"[{sid}] no valid pivots — skipping")
            continue
        apos = sorted(set(pivs.values()))
        res = apply_goalscope(
            full_text,
            mt=mt,
            include_noncode=False,
            token_positions=apos,
            **gs_kwargs,
        )
        readouts = {r["pos"]: r for r in res}

        # causality prepends the C++ stub so the oracle sees the read-out as the model's full answer
        ros = {aname: readout_prefix + readouts.get(pos, {}).get("patched_gen", "") for aname, pos in pivs.items()}
        verdicts = [jq.submit(hard_eval, ros[aname], impl_span, args.mode) for aname in pivs]
        sample_meta = dict(
            spec_id=sid,
            datatype=inst.meta["datatype"],
            lang=inst.src_lang,
            T=inst.meta["T"],
            j=inst.meta["j"],
            predicate_ids=[p.id for p in inst.task.predicates],
            progress_ids=list(inst.src_progress),
            frontier_id=inst.expected_next,
            var=var,
            prompt=prompt,
            completion=completion,
            impl_span=impl_span,
            pivots=pivs,
        )
        jq.then(verdicts, functools.partial(
            finish_sample, sid, inst, pivs, ros, impl_span, frontier_desc, sample_meta))

    jq.close()
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
