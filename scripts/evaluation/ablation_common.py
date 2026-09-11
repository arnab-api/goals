"""Shared machinery for the goal-head ablation evals (sufficiency + necessity), scaled up from
``notebooks/diff/07.2_goalhead_sink_sufficiency.ipynb`` and ``notebooks/clock/13_middle.ipynb``.

Question: are the goal heads SUFFICIENT (ablate everything else, in layers >= FROM_LAYER, and the LM
still implements the next(DIFF) frontier) but NOT NECESSARY (ablate just the goal heads and it barely
hurts)? Both drivers reuse everything here; they differ only in which heads get ablated.

Mechanism (attention-sink ablation): an ablated head may attend ONLY to a sink position (default 0)
for every query row >= the pivot, while it greedy-decodes the frontier value after ``{var} =``. The
generated RHS is graded by an oracle: does it implement the frontier's NL description (ORACLE_SYS_2)?

Corpus = the locked DIFF dataset (``data/goalscope_eval.json``, prefill-to-frontier, QTASKS), all
datatypes aggregated. Each datapoint is run under three conditions: baseline (no ablation), the
ablated goal-related set, and a matched-count RANDOM control (same size, same layer range).

Results -> ``<results>/evaluation/ablation/<model>/<axis>/<leaf>/`` (records.jsonl / summary.json /
config.json). Resume-safe by spec_id.
"""
import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from scripts.evaluation.eval_goalscope import (
    ALL_DATATYPES, ORACLE_MODEL, build_prompt, load_json_heads, load_locked_datapoints,
)
from src.functional import get_module_nnsight
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "data/goalscope_eval.json"
DEFAULT_HEADS_JSON = "head_sets/{slug}/goal__lam20_thr0.9_band25-56.json"

# ── oracle judge (verbatim from 13_middle): does the CODE implement the CONDITION? ──
ORACLE_SYS_2 = """You are grading a short snippet of Python CODE against ONE natural-language CONDITION. The code is the right-hand side of an assignment that a model produced while (under a mechanistic intervention) trying to implement that condition on its input. Because of the intervention the code is often NOISY: it may have minor bugs, use a wrong constant / threshold / operator, be incomplete or cut off, contain half-written or repeated fragments, or wander. Look PAST that surface noise and grade only the SUBSTANCE.

The question: does the CODE do something genuinely relevant to the CONDITION — a genuine attempt at, or a non-trivial step toward, implementing that condition? Even a small correct fragment of the right idea is a SUCCESS, so grade as leniently as possible.

The following do NOT count against the code (still YES):
  - Wrong specifics: a wrong constant, threshold, bound, modulus, operator, or comparison direction. If the condition is "average temperature > 30" and the code computes the average / compares the relevant quantity, that is YES even if the number or operator is off or missing.
  - Wrong implementation nuance: a different but plausible mechanism for the same idea (e.g. the condition wants a palindrome check and the code reverses a string and compares it, even with a buggy method call) -> YES.
  - Incomplete or cut-off code, or broken / repetitive fragments, AS LONG AS what IS there is relevant to the condition.
  - Partial coverage: implementing just one relevant ingredient of a multi-part condition counts.

Answer with exactly one token:
  YES        - the code contains SOME operation, quantity, or sub-step genuinely relevant to the condition, however partial, buggy, or incomplete.
  NO         - the code is coherent and about a specific thing, but that thing is UNRELATED to the condition (it implements a different check).
  NO_INFO    - the code is coherent but too generic or trivial to indicate the condition (e.g. just `n`, `True`, or a bare `n > 0` with no tie to the condition's concept).
  MALFORMED  - the code is contentless: empty, pure punctuation / garbage, or repetition carrying no relevant meaning.

Grade on meaning, not on shared keywords. Output only YES, NO, NO_INFO, or MALFORMED.
Put your one word answer inside <verdict></verdict> tags to make it easy to parse."""


def judge_implements(code: str, condition: str, use_cache: bool = True) -> dict:
    """Oracle verdict in {YES, NO, NO_INFO, MALFORMED, UNPARSED}: does ``code`` implement ``condition``?"""
    if not code or not code.strip():
        return dict(verdict="MALFORMED", response=None)
    prompt = f"Condition: {condition}\nCode stub: {code.strip()}\nAnswer:"
    resp = ask_claude(prompt=prompt, system_prompt=ORACLE_SYS_2, max_tokens=256,
                      use_cache=use_cache, model_name=ORACLE_MODEL)
    m = re.search(r"<verdict>\s*(YES|NO_INFO|NO|MALFORMED)\s*</verdict>", resp, re.IGNORECASE)
    if m:
        return dict(verdict=m.group(1).upper(), response=resp)
    hits = re.findall(r"\b(YES|NO_INFO|NO|MALFORMED)\b", resp.upper())
    return dict(verdict=hits[-1] if hits else "UNPARSED", response=resp)


# ── head-set helpers ──────────────────────────────────────────────────────────
def complement_heads(preserved, from_layer, n_layer, n_heads):
    """All heads in layers ``[from_layer, n_layer)`` that are NOT in ``preserved`` (the SUFFICIENCY
    ablation set: silence everyone but the goal heads, keeping layers < from_layer fully intact)."""
    sel = set(map(tuple, preserved))
    return [(l, h) for l in range(from_layer, n_layer)
            for h in range(n_heads) if (l, h) not in sel]


def random_heads_like(ablated, from_layer, n_layer, n_heads, exclude, rng):
    """A random head set of the SAME SIZE as ``ablated``, drawn from layers ``[from_layer, n_layer)``
    and disjoint from ``exclude`` (the goal heads) — the matched-count control."""
    excl = set(map(tuple, exclude))
    pool = [(l, h) for l in range(from_layer, n_layer)
            for h in range(n_heads) if (l, h) not in excl]
    k = min(len(ablated), len(pool))
    return rng.sample(pool, k)


# ── blind-from-span helpers (13_middle §5 / 07.2 isolation): heads RUN but can't see the goal span ──
def frontier_block_positions(mt, prompt, frontier_desc):
    """Absolute key positions to BLIND: from a token before the frontier's description span through
    the end of the prompt (the goal spec + the trailing pivot region), mirroring 13_middle §5's
    ``range(span_start-1, len(prompt_ids))``. Returns None if the description isn't located."""
    cpos = prompt.find(frontier_desc)
    if cpos < 0:
        return None
    enc = prepare_input(prompts=prompt, tokenizer=mt, return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    P = len(offsets)
    end = cpos + len(frontier_desc)
    toks = [ti for ti, (a, b) in enumerate(offsets) if b > a and a < end and b > cpos]
    if not toks:
        return None
    return list(range(max(min(toks) - 1, 0), P))


@torch.no_grad()
def generate_blocking_span_cached(mt, prompt_ids, block_heads, block_positions, q_start,
                                  max_new=64, stop_suffix=None):
    """Greedy decode while every head in ``block_heads`` is BLIND to ``block_positions`` (adds -inf
    there; it attends normally to every OTHER key) for query rows whose absolute position >= q_start.
    Non-goal heads keep running — they just can't read the goal span. KV-cache aware, O(seq)/step.
    ``block_heads=[]`` -> clean baseline. Deterministic (argmax)."""
    if stop_suffix is None:
        stop_suffix = [mt.tokenizer.eos_token, "\n", "\n\n"]
    stop_suffix = [s for s in stop_suffix if s]
    device = next(mt._model.parameters()).device
    n_heads = mt.n_heads
    l2h = defaultdict(list)
    for l, h in sorted(block_heads):
        l2h[l].append(h)
    block = sorted(set(int(p) for p in block_positions))

    ids_full = torch.tensor([list(prompt_ids)], device=device)
    attn = torch.ones_like(ids_full)
    past = None
    gen = []

    def _kvec(kv_len):
        v = torch.zeros(kv_len, device=device)
        bp = [p for p in block if p < kv_len]
        if bp:
            v[torch.tensor(bp, device=device)] = float("-inf")   # -inf ONLY on the blinded span
        return v

    for step in range(max_new):
        kv_len = ids_full.shape[1]
        if step == 0:
            cur_ids, q_len, q_offset = ids_full, kv_len, 0        # prefill: q rows are absolute 0..P-1
        else:
            cur_ids, q_len, q_offset = ids_full[:, -1:], 1, kv_len - 1  # decode: single new query
        inp = {"input_ids": cur_ids, "attention_mask": attn,
               "past_key_values": past, "use_cache": True}
        kvec = _kvec(kv_len) if block_heads else None
        with mt.trace(inp, scan=False, validate=False):
            if block_heads:
                for L in sorted(l2h):
                    sm = get_module_nnsight(mt, mt.attn_module_name_format.format(L)) \
                            .source.attention_interface_0.source.nn_functional_softmax_0
                    km = torch.zeros(1, n_heads, q_len, kv_len, device=device)
                    for qi in range(q_len):
                        if q_offset + qi >= q_start:
                            for h in l2h[L]:
                                km[0, h, qi, :] += kvec
                    sm.input = sm.input + km.to(sm.input.device, sm.input.dtype)
            logits = mt.output.logits[:, -1, :].save()
            past = mt.output.past_key_values.save()
        nxt = int(logits[0].argmax(-1))
        if nxt == mt.tokenizer.eos_token_id:
            break
        tok_str = mt.tokenizer.decode([nxt])
        gen.append(nxt)
        if any(tok_str.endswith(s) for s in stop_suffix):
            break
        ids_full = torch.cat([ids_full, torch.tensor([[nxt]], device=device)], dim=1)
        attn = torch.ones_like(ids_full)
    return mt.tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def sink_generate_cached(mt, prompt_ids, ablated_heads, q_start, sink_pos=0,
                         max_new=64, stop_suffix=None):
    """Greedy decode while every head in ``ablated_heads`` may attend ONLY to ``sink_pos`` for query
    rows whose absolute position >= q_start (13_middle §4/§7 sink-ablation = restrict-to-sink; ≈ zero
    the head's useful attention output). KV-cache aware. ``ablated_heads=[]`` -> clean baseline."""
    if stop_suffix is None:
        stop_suffix = [mt.tokenizer.eos_token, "\n", "\n\n"]
    stop_suffix = [s for s in stop_suffix if s]
    device = next(mt._model.parameters()).device
    n_heads = mt.n_heads
    l2h = defaultdict(list)
    for l, h in sorted(ablated_heads):
        l2h[l].append(h)

    ids_full = torch.tensor([list(prompt_ids)], device=device)
    attn = torch.ones_like(ids_full)
    past = None
    gen = []

    def _kvec(kv_len):
        v = torch.full((kv_len,), float("-inf"), device=device)   # -inf everywhere except the sink
        v[sink_pos] = 0.0
        return v

    for step in range(max_new):
        kv_len = ids_full.shape[1]
        if step == 0:
            cur_ids, q_len, q_offset = ids_full, kv_len, 0
        else:
            cur_ids, q_len, q_offset = ids_full[:, -1:], 1, kv_len - 1
        inp = {"input_ids": cur_ids, "attention_mask": attn,
               "past_key_values": past, "use_cache": True}
        kvec = _kvec(kv_len) if ablated_heads else None
        with mt.trace(inp, scan=False, validate=False):
            if ablated_heads:
                for L in sorted(l2h):
                    sm = get_module_nnsight(mt, mt.attn_module_name_format.format(L)) \
                            .source.attention_interface_0.source.nn_functional_softmax_0
                    km = torch.zeros(1, n_heads, q_len, kv_len, device=device)
                    for qi in range(q_len):
                        if q_offset + qi >= q_start:
                            for h in l2h[L]:
                                km[0, h, qi, :] += kvec
                    sm.input = sm.input + km.to(sm.input.device, sm.input.dtype)
            logits = mt.output.logits[:, -1, :].save()
            past = mt.output.past_key_values.save()
        nxt = int(logits[0].argmax(-1))
        if nxt == mt.tokenizer.eos_token_id:
            break
        tok_str = mt.tokenizer.decode([nxt])
        gen.append(nxt)
        if any(tok_str.endswith(s) for s in stop_suffix):
            break
        ids_full = torch.cat([ids_full, torch.tensor([[nxt]], device=device)], dim=1)
        attn = torch.ones_like(ids_full)
    return mt.tokenizer.decode(gen, skip_special_tokens=True)


# ── driver ────────────────────────────────────────────────────────────────────
def _dt_tag(datatypes):
    return ("all" if set(datatypes) == set(ALL_DATATYPES)
            else "+".join(d.replace("[", "").replace("]", "") for d in datatypes))


def run_dir_for(results_dir, model, axis, config):
    slug = model.split("/")[-1]
    leaf = (f"heads={config['ablated_name']}__fromL={config['from_layer']}__strat={config['strategy']}"
            f"__dtype={config['datatype']}__nspec={config['n_specs']}__seed={config['seed']}")
    return Path(results_dir) / "evaluation" / "ablation" / slug / axis / leaf


def common_argparser(description):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--datatypes", nargs="+", default=ALL_DATATYPES,
                    help="datatypes aggregated into one run (default: all four)")
    ap.add_argument("--heads", default=None,
                    help="goal head-set json (default: the model's 122-head goal set)")
    ap.add_argument("--from_layer", type=int, default=20,
                    help="sufficiency: blind non-goal heads in layers >= this (layers < it untouched). "
                         "necessity: lower bound of the matched random-control pool.")
    ap.add_argument("--sink_pos", type=int, default=0,
                    help="sink strategy only: the single key an ablated head may attend to (default 0)")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="frontier-body greedy budget")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE ONLY: cap datapoints (round-robin across datatypes)")
    ap.add_argument("--overwrite", action="store_true")
    logging_utils.add_logging_args(ap)
    return ap


def load_setup(args):
    """Configure logging/seed, load the locked corpus (optionally --limit round-robin), the model, and
    the goal head set. Returns (mt, goal_heads, datapoints, config, results_dir)."""
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    datapoints, ds_meta = load_locked_datapoints(args.dataset, args.datatypes)
    if args.limit:  # round-robin across datatypes so a small N hits every prompt builder
        groups = defaultdict(list)
        for sid, inst in datapoints:
            groups[inst.meta["datatype"]].append((sid, inst))
        picked, i = [], 0
        while len(picked) < args.limit and any(i < len(v) for v in groups.values()):
            for v in groups.values():
                if i < len(v) and len(picked) < args.limit:
                    picked.append(v[i])
            i += 1
        datapoints = picked
        logger.warning(f"--limit {args.limit}: SMOKE run on {len(datapoints)} datapoints")

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")
    slug = args.model.split("/")[-1]
    heads_path = args.heads or DEFAULT_HEADS_JSON.format(slug=slug)
    goal_heads = load_json_heads(heads_path, mt.n_layer, mt.n_heads)
    config = dict(model=args.model, dataset=str(args.dataset), datatype=_dt_tag(args.datatypes),
                  datatypes=list(args.datatypes), n_specs=len(datapoints), seed=args.seed,
                  sink_pos=args.sink_pos, heads_spec=heads_path)
    return mt, goal_heads, datapoints, config, results_dir


def run_ablation_eval(mt, datapoints, ablated_heads, control_heads, goal_heads, *, axis, ablated_name,
                      strategy, config, results_dir, from_layer, sink_pos, gen_max_new_tokens, seed,
                      overwrite=False):
    """Run three conditions (baseline / ablated / control) over ``datapoints`` and score each
    generation with the oracle judge. ``strategy`` picks the intervention on the condition's head set:

      "sink"       — restrict those heads to attend ONLY to ``sink_pos`` (13_middle §4/§7; necessity).
      "blind_span" — blind those heads from the frontier spec span (13_middle §5 / 07.2; sufficiency).

    ``ablated_heads`` = the axis set; ``control_heads`` = the matched contrast — both get the SAME
    intervention, differing only in WHICH heads it hits."""
    assert strategy in ("sink", "blind_span"), strategy
    config = dict(config, ablated_name=ablated_name, from_layer=from_layer, strategy=strategy,
                  sink_pos=sink_pos, n_specs=len(datapoints))
    run_dir = run_dir_for(results_dir, config["model"], axis, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    if overwrite:
        rec_path.unlink(missing_ok=True)
    done = set()
    if rec_path.exists():
        done = {json.loads(l)["spec"] for l in rec_path.read_text().splitlines() if l.strip()}
    (run_dir / "config.json").write_text(json.dumps(
        dict(config, axis=axis, from_layer=from_layer, strategy=strategy, sink_pos=sink_pos,
             n_ablated=len(ablated_heads), n_control=len(control_heads), n_goal=len(goal_heads),
             gen_max_new_tokens=gen_max_new_tokens, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    logger.info(f"run dir -> {run_dir} | {strategy} on {len(ablated_heads)} (ablated) / "
                f"{len(control_heads)} (control) heads | {len(datapoints)} datapoints ({len(done)} done)")

    conditions = {"baseline": [], "ablated": ablated_heads, "control": control_heads}
    n_done = len(done)
    with open(rec_path, "a") as fh:
        for sid, inst in datapoints:
            if sid in done:
                continue
            prompt, var = build_prompt(mt, inst, inst.src_lang)
            prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
            q_start = len(prompt_ids) - 1                       # intervene from the frontier pivot on
            cond_desc = inst.expected_next_desc
            block_pos = None
            if strategy == "blind_span":
                block_pos = frontier_block_positions(mt, prompt, cond_desc)
                if not block_pos:
                    logger.warning(f"[{sid}] frontier span not located — skipping")
                    continue
            row = dict(spec=sid, datatype=inst.meta["datatype"], frontier_id=inst.expected_next,
                       condition=cond_desc, var=var, n_blocked=(len(block_pos) if block_pos else 0))
            for cname, heads in conditions.items():
                if strategy == "sink":
                    gen = sink_generate_cached(mt, prompt_ids, heads, q_start, sink_pos=sink_pos,
                                               max_new=gen_max_new_tokens)
                else:
                    gen = generate_blocking_span_cached(mt, prompt_ids, heads, block_pos, q_start,
                                                        max_new=gen_max_new_tokens)
                code = gen.split("\n", 1)[0].strip()
                jv = judge_implements(code, cond_desc)
                row[cname] = dict(gen=code, verdict=jv["verdict"])
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_done += 1
            logger.info(f"[{n_done}/{len(datapoints)}] {sid}: "
                        f"base={row['baseline']['verdict']} ablated={row['ablated']['verdict']} "
                        f"control={row['control']['verdict']}")

    _summarize(run_dir, conditions)
    return run_dir


def _summarize(run_dir, conditions):
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines() if l.strip()]
    def rate(cond, recs):
        return (sum(r[cond]["verdict"] == "YES" for r in recs) / len(recs)) if recs else float("nan")
    summary = dict(n=len(recs), overall={c: rate(c, recs) for c in conditions})
    by_dt = {}
    for dt in sorted({r["datatype"] for r in recs}):
        sub = [r for r in recs if r["datatype"] == dt]
        by_dt[dt] = dict(n=len(sub), **{c: rate(c, sub) for c in conditions})
    summary["by_datatype"] = by_dt
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    o = summary["overall"]
    logger.info("implement-rate (YES): " + "  ".join(f"{c}={o[c]:.3f}" for c in conditions))
    return summary
