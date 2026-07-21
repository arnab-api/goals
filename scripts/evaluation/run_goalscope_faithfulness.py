#!/usr/bin/env python
"""GoalScope goal-head *faithfulness* eval on the multi-predicate ``check_conditions`` setup.

Pipeline (consolidated from ``notebooks/clock/12_goal_eval_scaleup.ipynb``):

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
    conda run -n goals python scripts/evaluation/run_goalscope_faithfulness.py \
        --heads goal --strategy additive --alpha 0.2 --steer_layer_range 35 46 \
        --n_steer_tokens all --max_new_tokens 64

    # a custom head set:
    ... --heads /path/to/heads.json          # json array of [layer_idx, head_idx] pairs
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

from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.goals.condition_data import INT_PREDICATES, materialize, sample_specs
from src.goals.functional import apply_chat_template, supports_chat_template
from src.goals.goalscope import apply_goalscope, greedy_continue
from src.goals.multistep_label import label_generation
from src.goals.oid_gaze import cache_root
from src.utils.oracle_llms import ask_claude
from scripts.clock.score_condition_heads import load_head_scores
from scripts.evaluation import faithfulness_io as fio

logger = logging.getLogger(__name__)  # configured in main() via logging_utils

ANCHOR_ORDER = ["prev_end", "indent", "var", "eq", "first_impl"]
ORACLE_MODEL = "claude-sonnet-4-6"


# ══════════════════════════════════════════════════════════════════════════════
# Head sets
# ══════════════════════════════════════════════════════════════════════════════
def load_goal_heads(model_name, results_dir, threshold=0.9, band=(25, 56)):
    ck = torch.load(
        os.path.join(
            results_dir, f"localization/{model_name.split('/')[-1]}/head_mask.pt"
        ),
        map_location="cpu",
    )
    layers = [int(x) for x in ck["layers"]]
    m = ck["mask_values"].numpy().copy()
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


def load_head_set(spec, model_name, results_dir, n_heads):
    """Return list[(layer, head)] for spec in {'goal','gaze','decision', <path>.json}."""
    if spec.endswith(".json"):
        arr = json.loads(Path(spec).read_text())
        return [(int(l), int(h)) for l, h in arr]
    if spec == "goal":
        return load_goal_heads(model_name, results_dir)
    if spec == "decision":
        return load_decision_heads(model_name)
    if spec == "gaze":  # equal-N to goal (matches 09.1)
        return load_gaze_heads(
            model_name, n_heads, topk=len(load_goal_heads(model_name, results_dir))
        )
    raise ValueError(
        f"unknown --heads spec: {spec!r} (want goal|gaze|decision|<path>.json)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Corpus + generation
# ══════════════════════════════════════════════════════════════════════════════
def build_corpus(n_specs, num_conditions, seed, datatype="int"):
    preds = [p for p in INT_PREDICATES if p.has_singleline_for("python")]
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


def anchors_for_sample(mt, full_text, prompt, labeled):
    enc = prepare_input(prompts=full_text, tokenizer=mt, return_offsets_mapping=True)
    offsets = [tuple(p) for p in enc.pop("offset_mapping")[0].tolist()]
    gen_char_start = len(prompt)
    rows = []
    for k, sl in enumerate(labeled.labels):
        var = chr(97 + k)
        rows.append(
            dict(
                cond=k,
                var=var,
                condition=sl.step_text,
                impl_span=sl.gen_code.strip(),
                anchors=line_anchors(full_text, offsets, var, gen_char_start),
            )
        )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Oracle scoring  (always use_cache=True)
# ══════════════════════════════════════════════════════════════════════════════
HARD_SYS = """You are grading a lightweight interpretability probe (a patchscope). The DESCRIPTION below was read out of a model's hidden state at a SINGLE token while the model was writing one line of Python CODE that checks some property of its input. Because it is squeezed out of one token under a steering/patching setup, the description is usually NOISY: it may be wrapped in chat scaffolding (like 'Sure, but first let us ...'), contain code fences or half-written code, be cut off mid-sentence, or repeat and loop. Look PAST all of that surface noise and grade only the SUBSTANCE.

The question: does the description surface ANYTHING genuinely relevant to what the CODE is doing — any relevant operation, quantity, sub-step, or property of the input? Pulling even a small correct fragment out of a single token is a SUCCESS, so grade as leniently as possible.

The following do NOT count against the description (still YES):
  - Wrong specifics: a wrong constant, threshold, bound, modulus, operator, or comparison. If the code computes some quantity and compares it to a value, naming the quantity alone is enough and the exact value can be wrong or missing. Example: the code implements `average temperature > 30`; a read-out about `computing the average temperature` is YES even though the `> 30` part is wrong or absent.
  - Wrong implementation nuance: naming a different mechanism for the same idea. Example: the read-out says `sorting with bubble sort` while the code calls .sort() or uses quicksort -> still YES.
  - Broken, repetitive, or looping text, AS LONG AS the repeated content is relevant. Example: a read-out that keeps repeating the correct operation is YES, not MALFORMED.
  - Partial coverage: naming just one relevant ingredient of a multi-part check counts.

Answer with exactly one token:
  YES        - the description contains SOME content genuinely relevant to what the code does, however partial, imprecise, or garbled.
  NO         - the description is coherent and about a specific thing, but that thing is UNRELATED to what the code does.
  NO_INFO    - the description is coherent but too generic to indicate any specific aspect of the code (e.g. just 'doing a computation on the number').
  MALFORMED  - the description is contentless: empty, pure punctuation or garbage, or repetition that carries NO relevant meaning.

Grade on meaning, not on shared keywords. Output only YES, NO, NO_INFO, or MALFORMED."""
_HARD_TAG = hashlib.md5(HARD_SYS.encode()).hexdigest()[:8]

_KWAY_SYS = "You match a description to the closest condition. Output only an integer."


def hard_faithfulness(readout, impl_span):
    """dict(verdict, prompt, system, response). verdict in {YES, NO, NO_INFO, MALFORMED, UNPARSED}."""
    if not readout or not readout.strip():
        return dict(verdict="MALFORMED", prompt=None, system=HARD_SYS, response=None)
    prompt = f"[rubric {_HARD_TAG}]\nCode: {impl_span}\nDescription: {readout.strip()}\nAnswer:"
    resp = ask_claude(
        prompt=prompt,
        system_prompt=HARD_SYS,
        max_tokens=8,
        use_cache=True,
        model_name=ORACLE_MODEL,
    )
    r = resp.strip().upper()
    verdict = next(
        (t for t in ("MALFORMED", "NO_INFO", "YES", "NO") if r.startswith(t)),
        "UNPARSED",
    )
    return dict(verdict=verdict, prompt=prompt, system=HARD_SYS, response=resp)


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
        max_tokens=32,
        use_cache=True,
        model_name=ORACLE_MODEL,
    )
    m = re.search(r"-?\d+", resp)
    pred = int(m.group()) if m else None
    if pred is not None and not (0 <= pred < len(conditions)):
        pred = None
    return dict(pred=pred, prompt=prompt, system=_KWAY_SYS, response=resp)


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # ── GOALSCOPE_CONFIG ──
    ap.add_argument(
        "--heads", default="goal", help="goal | gaze | decision | <path>.json"
    )
    ap.add_argument("--strategy", default="additive", choices=["additive", "replace"])
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
    ap.add_argument("--overwrite", action="store_true")
    logging_utils.add_logging_args(ap)  # -v (DEBUG) / -q (WARNING); default INFO
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)  # -v => DEBUG: every oracle prompt/verdict logged
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    alpha = (
        args.alpha
        if args.alpha is not None
        else (0.2 if args.strategy == "additive" else 1.0)
    )
    n_steer = (
        "all" if str(args.n_steer_tokens).lower() == "all" else int(args.n_steer_tokens)
    )
    heads_name = resolve_heads_name(args.heads)

    # config (sans selected_heads) — enough to resolve the run dir + skip check before loading the model
    config = dict(
        heads_spec=args.heads,
        heads_name=heads_name,
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
        anchors=ANCHOR_ORDER,
        oracle_model=ORACLE_MODEL,
        hard_rubric_tag=_HARD_TAG,
    )
    run_dir = fio.run_dir_for(config, results_dir)
    if run_dir.exists() and not args.overwrite:
        logger.info(f"run dir exists ({run_dir}); pass --overwrite to redo. exiting.")
        return
    logger.info(f"run dir -> {run_dir}")

    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    n_heads = mt.model.config.num_attention_heads

    heads = load_head_set(args.heads, args.model, results_dir, n_heads)
    config["n_heads"] = len(heads)
    config["selected_heads"] = [[int(l), int(h)] for l, h in heads]
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"heads[{heads_name}] = {len(heads)} (layer,head) pairs")

    gs_kwargs = dict(
        selected_heads=heads,
        strategy=args.strategy,
        alpha=alpha,
        steer_layer_range=tuple(args.steer_layer_range),
        n_steer_tokens=n_steer,
        max_new_tokens=args.max_new_tokens,
    )

    samples = build_corpus(args.n_specs, args.num_conditions, args.seed, args.datatype)
    logger.info(f"corpus: {len(samples)} samples")

    records, samples_meta = [], []
    for si, sample in enumerate(samples):
        prompt = build_prompt(mt, sample)
        completion = generate_completion(mt, prompt, args.gen_max_new_tokens)
        full_text = prompt + completion
        labeled = label_generation(mt, sample, prompt, completion, use_cache=True)
        rows = anchors_for_sample(mt, full_text, prompt, labeled)
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
                ro = readouts.get(pos, {}).get("patched_gen", "")
                hv = hard_faithfulness(ro, row["impl_span"])
                kv = kway_assign(ro, conds)
                logger.debug(
                    "[%s c%d %s] impl=%r cond=%r",
                    sample.spec.spec_id, row["cond"], aname, row["impl_span"], row["condition"],
                )
                logger.debug("    read-out : %r", ro)
                logger.debug("    HARD  verdict=%s  oracle_response=%r", hv["verdict"], hv["response"])
                logger.debug("    KWAY  pred=%s (true %d)  oracle_response=%r",
                             kv["pred"], row["cond"], kv["response"])
                records.append(
                    dict(
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
                        kway_pred=kv["pred"],
                        kway_true=row["cond"],
                        kway_prompt=kv["prompt"],
                        kway_response=kv["response"],
                    )
                )
        samples_meta.append(
            dict(
                spec_id=sample.spec.spec_id,
                predicate_ids=list(sample.spec.predicate_ids),
                prompt=prompt,
                completion=completion,
                labels=[
                    dict(
                        step_text=sl.step_text,
                        gen_code=sl.gen_code,
                        gen_token_range=list(sl.gen_token_range)
                        if sl.gen_token_range
                        else None,
                    )
                    for sl in labeled.labels
                ],
                anchors=[
                    dict(cond=r["cond"], var=r["var"], anchors=r["anchors"])
                    for r in rows
                ],
            )
        )
        logger.info(
            f"[{si + 1}/{len(samples)}] {sample.spec.spec_id}: "
            f"{sum(len(r['anchors']) for r in rows)} anchor read-outs scored"
        )

    # ── aggregate ──
    by_anchor = {}
    for aname in ANCHOR_ORDER:
        grp = [r for r in records if r["anchor"] == aname]
        if not grp:
            continue
        by_anchor[aname] = dict(
            n=len(grp),
            hard_yes=float(np.mean([r["hard"] == "YES" for r in grp])),
            kway_ok=float(np.mean([r["kway_pred"] == r["kway_true"] for r in grp])),
        )
    overall = dict(
        hard_yes=float(np.mean([r["hard"] == "YES" for r in records]))
        if records
        else float("nan"),
        kway_ok=float(np.mean([r["kway_pred"] == r["kway_true"] for r in records]))
        if records
        else float("nan"),
    )
    summary = dict(
        n_records=len(records),
        n_samples=len(samples_meta),
        by_anchor=by_anchor,
        overall=overall,
    )

    fio.save_run(run_dir, config, records, summary, samples_meta)
    logger.info(f"saved -> {run_dir}")
    logger.info("per-anchor faithfulness (hard_yes | kway_ok):")
    for aname in ANCHOR_ORDER:
        if aname in by_anchor:
            m = by_anchor[aname]
            logger.info(
                f"  {aname:10s} n={m['n']:<4d} hard_yes={m['hard_yes']:.3f}  kway_ok={m['kway_ok']:.3f}"
            )
    logger.info(
        f"  {'OVERALL':10s} n={len(records):<4d} hard_yes={overall['hard_yes']:.3f}  kway_ok={overall['kway_ok']:.3f}"
    )


if __name__ == "__main__":
    main()
