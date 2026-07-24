#!/usr/bin/env python
"""clock/13 — scaled "what drives the *middle*" eval (Sections 7 + 8 of 13_middle.ipynb).

At a mid-implementation pivot (the model is about to write the RHS for the CURRENT condition of a
K-condition function), we test whether a head set carries the "implement CURRENT" intent, via two
complementary interventions:

  ablation  (§7, "blind the set")        — restrict the selected head set to the attention sink
                                           (they see ONLY position `sink_pos`); does CURRENT survive?
  isolation (§8, "blind everyone else")  — blind the COMPLEMENT (every head in layers >= from_layer
                                           NOT in the set) to CURRENT's condition span; only the set
                                           may see it. Does CURRENT survive when only they can?

Head sets: goal | gaze | decision | control  (control = per-layer random set, equal-N to goal).

For each sample we do ONE clean read-out (baseline generation + oracle verdict + soft metrics), then
apply both modes to the SAME prompt for each requested head set. Two signals per (mode, headset):
  soft — closed-set p(CURRENT's canonical RHS) among the K conditions (+ delta vs clean)
  hard — lenient single-condition oracle: does the generated stub genuinely attempt CURRENT?

Everything streams to <out>/records.jsonl (resumable on (example_id, mode, headset) triples) with a
sibling config.json. The output dir is keyed by every result-changing arg so --overwrite can't bleed
across configs.

Run (needs eager attn for the softmax-input hooks; a 32B needs the whole model on GPU):
    conda run -n goals python scripts/evaluation/middle_ground.py --n-samples 256 --mode both
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict

# ── repo root on path (…/goals/scripts/evaluation/middle_ground.py -> …/goals) ──
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # core
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--mode", choices=["ablation", "isolation", "both"], default="both")
    ap.add_argument("--head-sets", nargs="+", default=["goal", "gaze", "decision", "control"],
                    choices=["goal", "gaze", "decision", "control"])
    # experiment shape (all result-changing -> encoded in the output dir tag)
    ap.add_argument("--k-range", nargs=2, type=int, default=[4, 6], metavar=("LO", "HI"),
                    help="inclusive #conditions range")
    ap.add_argument("--max-new", type=int, default=128, help="tokens to generate before the rule-cut")
    ap.add_argument("--from-layer", type=int, default=20, help="isolation: blind complement heads in layers >= this")
    ap.add_argument("--block-which", choices=["full", "id"], default="full",
                    help="isolation: block the full spec span or just the discriminative id substrings")
    ap.add_argument("--sink-pos", type=int, default=0, help="ablation: the attention-sink position heads are pinned to")
    ap.add_argument("--seed", type=int, default=2000, help="base seed for prompt sampling")
    ap.add_argument("--stop-at-newline", action=argparse.BooleanOptionalAction, default=True,
                    help="cut generation at the first newline (default ON). Under ablation the LM output "
                         "is low-quality, so the first RHS line is the useful signal; a long free-run is "
                         "usually junk the rule-based span cut can't rescue. Pass --no-stop-at-newline to "
                         "generate the full --max-new and rule-cut instead.")
    # head-set construction knobs (result-changing -> folded into the dir tag via a short hash)
    ap.add_argument("--goal-threshold", type=float, default=0.9)
    ap.add_argument("--goal-band", nargs=2, type=int, default=[25, 56], metavar=("LO", "HI"))
    ap.add_argument("--gaze-qspan", type=int, default=15)
    ap.add_argument("--dec-tag", type=str, default="lam10.0")
    ap.add_argument("--dec-threshold", type=float, default=0.8)
    ap.add_argument("--head-version", type=str, default="conditions")
    # oracle
    ap.add_argument("--oracle-model", type=str, default="claude-sonnet-4-6")
    ap.add_argument("--oracle-no-cache", action="store_true", help="bypass the ask_claude disk cache")
    # infra
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--gpus", type=str, default=None, help="sets CUDA_VISIBLE_DEVICES before torch import")
    ap.add_argument("--out-dir", type=str, default=None, help="override the results root (default: <RESULTS>/middle)")
    ap.add_argument("--run-tag", type=str, default=None, help="override the auto config tag (dir name)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing records for this exact config and restart")
    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Oracle rubric (single-condition, lenient) — ported from 13_middle §6, with the
# <verdict></verdict> tag + tag-first parser.
# ══════════════════════════════════════════════════════════════════════════════
HARD_SYS = """You are grading a short snippet of Python CODE against ONE natural-language CONDITION. The code is the right-hand side of an assignment that a model produced while (under a mechanistic intervention) trying to implement that condition on its input. Because of the intervention the code is often NOISY: it may have minor bugs, use a wrong constant / threshold / operator, be incomplete or cut off, contain half-written or repeated fragments, or wander. Look PAST that surface noise and grade only the SUBSTANCE.

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
_HARD_TAG = hashlib.md5(HARD_SYS.encode()).hexdigest()[:8]

PIVOT_TRAILING_SPACE = False   # cut at `= ` so the RHS starts clean

logger = logging.getLogger("middle_ground")


# ══════════════════════════════════════════════════════════════════════════════
# The bulk of the module is defined inside main() after torch/model imports so we
# can set CUDA_VISIBLE_DEVICES first. Everything below `main()` is straight ports.
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch

    from src.models import ModelandTokenizer
    from src.functional import get_module_nnsight
    from src.tokens import find_token_range
    from src.utils import env_utils
    from src.utils.oracle_llms import ask_claude
    from src.goals.condition_data import INT_PREDICATES
    from src.goals.multiline_prompts import build_example
    from src.goals.oid_gaze import cache_root
    from scripts.clock.score_condition_heads import load_head_scores

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _PRED_BY_ID = {p.id: p for p in INT_PREDICATES}

    # ── output dir keyed by every result-changing arg (so --overwrite can't bleed) ──
    model_short = args.model.split("/")[-1]
    lo, hi = args.k_range
    knob_str = (f"gt{args.goal_threshold}_gb{args.goal_band[0]}-{args.goal_band[1]}"
                f"_gq{args.gaze_qspan}_dt{args.dec_tag}_dh{args.dec_threshold}_v{args.head_version}")
    knob_hash = hashlib.md5(knob_str.encode()).hexdigest()[:6]
    mnew_seg = f"mnew{args.max_new}" + ("nl" if args.stop_at_newline else "full")
    tag = args.run_tag or (
        f"seed{args.seed}__k{lo}-{hi}__{mnew_seg}__sink{args.sink_pos}"
        f"__isoL{args.from_layer}-{args.block_which}__hs{knob_hash}"
    )
    base = args.out_dir or os.path.join(env_utils.DEFAULT_RESULTS_DIR, "middle")
    out_dir = os.path.join(base, model_short, tag)
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "records.jsonl")
    config_path = os.path.join(out_dir, "config.json")
    if args.overwrite:
        for p in (records_path, config_path):
            if os.path.exists(p):
                os.remove(p)
        logger.info(f"--overwrite: cleared {out_dir}")
    logger.info(f"out_dir -> {out_dir}")

    # ── model ──
    logger.info(f"loading {args.model} (device_map={args.device_map}, eager attn) ...")
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map=args.device_map, attn_implementation="eager")
    N_LAYER = mt.n_layer
    N_HEADS = mt.model.config.num_attention_heads
    logger.info(f"n_layer={N_LAYER} n_heads={N_HEADS} n_embd={mt.n_embd}")

    # ─────────────────────────────────────────────────────────────────────────
    # Head sets (ported from §1)
    # ─────────────────────────────────────────────────────────────────────────
    def load_head_sets():
        # goal — trained mask @ threshold, restricted to a layer band
        gck = torch.load(os.path.join(env_utils.DEFAULT_RESULTS_DIR,
                         f"localization/{model_short}/head_mask.pt"), map_location="cpu")
        goal_layers = [int(x) for x in gck["layers"]]
        gmask = gck["mask_values"].numpy().copy()
        gmask[: args.goal_band[0]] = 0; gmask[args.goal_band[1]:] = 0
        goal = [(goal_layers[li], int(h)) for li, h in np.argwhere(gmask > args.goal_threshold)]
        # gaze — top-K step-lookback heads, equal-N to goal
        gaze_scores = load_head_scores(model=args.model, version=args.head_version,
                                       q_span=args.gaze_qspan, compare_with="mean")["combined"]
        flat = np.argsort(gaze_scores, axis=None)[::-1][: len(goal)]
        gaze = [(int(i // N_HEADS), int(i % N_HEADS)) for i in flat]
        # decision — completion mask @ threshold, latest run for the tag
        root = os.path.join(cache_root(args.model, args.head_version), "completion_mask")
        cand = [t for t in os.listdir(root) if os.path.isdir(os.path.join(root, t)) and args.dec_tag in t]
        assert cand, f"no completion_mask with tag {args.dec_tag!r} under {root}"
        latest = max(cand, key=lambda t: os.path.getmtime(os.path.join(root, t)))
        dck = torch.load(os.path.join(root, latest, "mask.pt"), map_location="cpu", weights_only=False)
        dec_layers = [int(x) for x in dck["layers"]]
        dmv = dck["mask_values"].numpy()
        decision = [(dec_layers[li], int(h)) for li, h in np.argwhere(dmv > args.dec_threshold)]
        # control — per-layer random set, equal-N-per-layer to goal (fixed seed for reproducibility)
        rng0 = random.Random(0)
        per_layer = Counter(l for l, _ in goal)
        control = []
        for l, cnt in per_layer.items():
            control += [(l, h) for h in rng0.sample(range(N_HEADS), min(cnt, N_HEADS))]
        return {k: sorted(v) for k, v in
                {"goal": goal, "gaze": gaze, "decision": decision, "control": control}.items()}

    head_sets = load_head_sets()
    for name, hs in head_sets.items():
        ls = sorted({l for l, _ in hs}) or [-1]
        logger.info(f"  {name:9s}: {len(hs):4d} heads · layers {ls[0]}-{ls[-1]}")

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt (§2)
    # ─────────────────────────────────────────────────────────────────────────
    def build_dissociation_prompt(rng, current_slot, n_conditions, naming="letters", lang="python"):
        ts = PIVOT_TRAILING_SPACE
        ex = build_example(mt=mt, rng=rng, target_slot=current_slot, n_conditions=n_conditions,
                           mode="chat", style="singleline", naming=naming, lang=lang,
                           prior_style="singleline")
        tvar = ex["target_var"]; fp = ex["full_prompt"]
        line_start = fp.rfind(f"\n    {tvar} =") + 1
        assign_eq = fp.index("=", line_start)
        cut = assign_eq + 2 if ts else assign_eq + 1
        ex["dissoc_prompt"] = fp[:cut]; ex["current_slot"] = current_slot; ex["trailing_space"] = ts
        ids = mt.tokenizer(ex["dissoc_prompt"], add_special_tokens=False)["input_ids"]
        ex["prompt_ids"] = ids; ex["pivot_pos"] = len(ids) - 1
        return ex

    # ─────────────────────────────────────────────────────────────────────────
    # Interventions (§3 sink-restrict + §5 span-block), forward + generate
    # ─────────────────────────────────────────────────────────────────────────
    def _softmax_mod(L):
        return get_module_nnsight(mt, mt.attn_module_name_format.format(L)) \
            .source.attention_interface_0.source.nn_functional_softmax_0

    @torch.no_grad()
    def forward_restrict_to_span(inputs, selected_heads, keep_positions, q_positions):
        seq_len = inputs["input_ids"].shape[1]
        device = next(mt._model.parameters()).device
        keep = torch.tensor(sorted(set(keep_positions)), device=device, dtype=torch.long)
        qpos = torch.tensor(sorted(set(q_positions)), device=device, dtype=torch.long)
        by_layer = {}
        for (l, h) in selected_heads:
            m = by_layer.get(l)
            if m is None:
                m = torch.zeros(1, N_HEADS, seq_len, seq_len, device=device); by_layer[l] = m
            m[0, h, qpos.unsqueeze(1), :] = -torch.inf
            m[0, h, qpos.unsqueeze(1), keep.unsqueeze(0)] = 0.0
        with mt.trace(inputs, scan=False, validate=False):
            for l in by_layer:
                sm = _softmax_mod(l)
                sm.input = sm.input + by_layer[l].to(sm.input.device)
            logits = mt.output.logits.save()
        return logits.float()

    @torch.no_grad()
    def forward_blocking_span(inputs, block_heads, block_positions, q_positions):
        seq_len = inputs["input_ids"].shape[1]
        device = next(mt._model.parameters()).device
        block = torch.tensor(sorted(set(int(p) for p in block_positions)), device=device, dtype=torch.long)
        qpos = torch.tensor(sorted(set(int(p) for p in q_positions)), device=device, dtype=torch.long)
        by_layer = {}
        for (l, h) in block_heads:
            m = by_layer.get(l)
            if m is None:
                m = torch.zeros(1, N_HEADS, seq_len, seq_len, device=device); by_layer[l] = m
            if len(block):
                m[0, h, qpos.unsqueeze(1), block.unsqueeze(0)] = -torch.inf
        with mt.trace(inputs, scan=False, validate=False):
            for l in by_layer:
                sm = _softmax_mod(l)
                sm.input = sm.input + by_layer[l].to(sm.input.device)
            logits = mt.output.logits.save()
        return logits.float()

    def _stop_suffix():
        # default ON: cut at newline. --no-stop-at-newline -> [] -> only the EOS token id halts.
        if args.stop_at_newline:
            return [s for s in [mt.tokenizer.eos_token, "\n", "\n\n"] if s]
        return []

    @torch.no_grad()
    def generate_restricted_to_span(prompt_ids, selected_heads, keep_positions, q_start, max_new):
        stop_suffix = _stop_suffix()
        device = next(mt._model.parameters()).device
        ids = torch.tensor([list(prompt_ids)], device=device)
        l2h = defaultdict(list)
        for l, h in sorted(selected_heads): l2h[l].append(h)
        keep_set = set(int(k) for k in keep_positions)
        gen = []
        for _ in range(max_new):
            seq = ids.shape[1]
            kvec = torch.zeros(seq, device=device)
            if selected_heads:
                out_keys = [k for k in range(seq) if k not in keep_set]
                if out_keys: kvec[torch.tensor(out_keys, device=device)] = float("-inf")
            with mt.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                          scan=False, validate=False):
                for L in sorted(l2h):
                    sm = _softmax_mod(L)
                    km = torch.zeros(1, N_HEADS, seq, seq, device=device)
                    for h in l2h[L]: km[0, h, q_start:, :] += kvec
                    sm.input = sm.input + km.to(sm.input.device, sm.input.dtype)
                logits = mt.output.logits[:, -1, :].save()
            nxt = int(logits[0].argmax(-1))
            if nxt == mt.tokenizer.eos_token_id: break
            tok_str = mt.tokenizer.decode([nxt]); gen.append(nxt)
            if any(tok_str.endswith(s) for s in stop_suffix): break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return mt.tokenizer.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def generate_blocking_span(prompt_ids, block_heads, block_positions, q_start, max_new):
        stop_suffix = _stop_suffix()
        device = next(mt._model.parameters()).device
        ids = torch.tensor([list(prompt_ids)], device=device)
        l2h = defaultdict(list)
        for l, h in sorted(block_heads): l2h[l].append(h)
        block = torch.tensor(sorted(set(int(p) for p in block_positions)), device=device, dtype=torch.long)
        gen = []
        for _ in range(max_new):
            seq = ids.shape[1]
            with mt.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                          scan=False, validate=False):
                if block_heads and len(block):
                    for L in sorted(l2h):
                        sm = _softmax_mod(L)
                        km = torch.zeros(1, N_HEADS, seq, seq, device=device)
                        for h in l2h[L]: km[0, h, q_start:, block] = float("-inf")
                        sm.input = sm.input + km.to(sm.input.device, sm.input.dtype)
                logits = mt.output.logits[:, -1, :].save()
            nxt = int(logits[0].argmax(-1))
            if nxt == mt.tokenizer.eos_token_id: break
            tok_str = mt.tokenizer.decode([nxt]); gen.append(nxt)
            if any(tok_str.endswith(s) for s in stop_suffix): break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        return mt.tokenizer.decode(gen, skip_special_tokens=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Condition spans + complement (§5)
    # ─────────────────────────────────────────────────────────────────────────
    def locate_condition_spans(ex):
        prompt = ex["dissoc_prompt"]
        offsets = [tuple(p) for p in mt.tokenizer(prompt, return_offsets_mapping=True,
                                                  add_special_tokens=False)["offset_mapping"]]
        spec = ex["spec"]; out = []
        for k, (text, subs) in enumerate(zip(spec.spec_texts, spec.id_substrs)):
            base_i = prompt.find(text); full = None; sub_ranges = []
            if base_i != -1:
                try:
                    full = list(find_token_range(prompt, text, offset_mapping=offsets, occurrence=0))
                except (ValueError, AssertionError):
                    full = None
                for ss in subs:
                    rel = text.find(ss)
                    if rel == -1: continue
                    occ = prompt.count(ss, 0, base_i + rel)
                    try:
                        ts, te = find_token_range(prompt, ss, offset_mapping=offsets, occurrence=occ)
                        sub_ranges.append([ts, te])
                    except (ValueError, AssertionError):
                        continue
            out.append({"cond": k, "var": ex["var_names"][k], "text": text,
                        "full_span": full, "id_substr_spans": sub_ranges})
        return out

    def condition_block_positions(spans, slot, which):
        s = spans[slot]
        if which == "id":
            return sorted({p for r in s["id_substr_spans"] for p in range(r[0], r[1])})
        return list(range(*s["full_span"])) if s["full_span"] else []

    def complement_heads(selected, from_layer):
        sel = set(map(tuple, selected))
        return [(l, h) for l in range(from_layer, N_LAYER) for h in range(N_HEADS) if (l, h) not in sel]

    # ─────────────────────────────────────────────────────────────────────────
    # Soft (canonical-RHS log-likelihood) machinery (§6)
    # ─────────────────────────────────────────────────────────────────────────
    def get_condition_references(ex):
        ts = ex.get("trailing_space", PIVOT_TRAILING_SPACE)
        spec = ex["spec"]; out = []
        for k in range(ex["n_conditions"]):
            pid, prm, var = spec.predicate_ids[k], spec.predicate_params[k], ex["var_names"][k]
            pred = _PRED_BY_ID[pid]; has_sl = pred.has_singleline_for(ex["lang"])
            full_line = (pred.render_singleline(prm, var, lang=ex["lang"]) if has_sl
                         else " / ".join(pred.render_multiline(prm, var, lang=ex["lang"])))
            m = re.match(rf"^\s*{re.escape(var)}\s*=", full_line)
            rhs = full_line[m.end():] if m else full_line
            if ts and rhs.startswith(" "): rhs = rhs[1:]
            rhs_ids = mt.tokenizer(rhs, add_special_tokens=False)["input_ids"]
            out.append({"cond": k, "var": var, "pred": pid, "rhs": rhs, "rhs_ids": rhs_ids})
        return out

    @torch.no_grad()
    def score_condition_lls(ex, refs, forward_fn=None, length_norm=True):
        device = next(mt._model.parameters()).device
        pivot = ex["pivot_pos"]; base_ids = list(ex["prompt_ids"]); lls = []
        for r in refs:
            rhs_ids = r["rhs_ids"]; ids = base_ids + rhs_ids
            inp = {"input_ids": torch.tensor([ids], device=device),
                   "attention_mask": torch.ones(1, len(ids), device=device)}
            qpos = list(range(pivot, pivot + len(rhs_ids)))
            if forward_fn is not None:
                logits = forward_fn(inp, qpos)
            else:
                with mt.trace(inp, scan=False, validate=False):
                    logits = mt.output.logits.save()
                logits = logits.float()
            logp = torch.log_softmax(logits[0], dim=-1)
            tgt = torch.tensor(rhs_ids, device=logp.device)
            pos = torch.arange(pivot, pivot + len(rhs_ids), device=logp.device)
            s = logp[pos, tgt].sum()
            lls.append(float((s / len(rhs_ids) if length_norm else s).cpu()))
        return np.asarray(lls)

    def soft_metrics_current(cond_lls, current, tau=1.0):
        lls = np.asarray(cond_lls, dtype=np.float64)
        p = np.exp((lls - lls.max()) / tau); p = p / p.sum()
        order = np.argsort(-lls)
        rank_current = int(np.where(order == current)[0][0]) + 1
        return {"p": p, "p_current": float(p[current]), "argmax": int(order[0]),
                "rank_current": rank_current, "current_is_argmax": bool(order[0] == current)}

    # ─────────────────────────────────────────────────────────────────────────
    # Hard oracle (§6) — <verdict> tag + tag-first parser
    # ─────────────────────────────────────────────────────────────────────────
    def judge_current(code_stub, condition_text):
        if not code_stub or not code_stub.strip():
            return dict(verdict="MALFORMED", condition=condition_text, code_stub=code_stub,
                        prompt=None, response=None)
        prompt = f"[rubric {_HARD_TAG}]\nCondition: {condition_text}\nCode: {code_stub}\nAnswer:"
        resp = ask_claude(prompt=prompt, system_prompt=HARD_SYS, temperature=0.0, max_tokens=256,
                          use_cache=not args.oracle_no_cache, model_name=args.oracle_model)
        r = resp.strip().upper()
        m = re.search(r"<VERDICT>\s*(MALFORMED|NO_INFO|YES|NO)\s*</VERDICT>", r)
        if m:
            verdict = m.group(1)
        else:
            hits = re.findall(r"\b(MALFORMED|NO_INFO|YES|NO)\b", r)
            verdict = hits[-1] if hits else "UNPARSED"
        return dict(verdict=verdict, condition=condition_text, code_stub=code_stub,
                    prompt=prompt, response=resp)

    def isolate_current_span(text, ex, current):
        K = ex["n_conditions"]
        if current + 1 < K:
            nxt = ex["var_names"][current + 1]
            m = re.search(rf"\n\s*{re.escape(nxt)}\s*=(?!=)", text)
        else:
            m = re.search(r"\n\s*(return|print)", text)
        return (text[:m.start()] if m else text).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-sample runner (unified: one example, one clean read-out, both modes)
    # ─────────────────────────────────────────────────────────────────────────
    def run_one(i, need):
        """`need` = set of (mode, headset) triples to (re)compute for this sample."""
        rng = random.Random(args.seed + i)
        K = rng.randint(lo, hi)
        current = rng.randint(1, K - 2)                       # a strict-middle slot
        ex = build_dissociation_prompt(random.Random(7 * args.seed + i), current_slot=current, n_conditions=K)
        refs = get_condition_references(ex)
        q_start = ex["pivot_pos"]
        ll_clean = score_condition_lls(ex, refs)
        m_clean = soft_metrics_current(ll_clean, current)

        common = dict(example_id=i, K=K, current=current,
                      current_pred=ex["spec"].predicate_ids[current],
                      current_text=ex["spec"].spec_texts[current],
                      spec_texts=list(ex["spec"].spec_texts), var_names=list(ex["var_names"]),
                      dissoc_prompt=ex["dissoc_prompt"], pivot_pos=ex["pivot_pos"],
                      ref_rhs=[r["rhs"] for r in refs], rubric_tag=_HARD_TAG,
                      p_current_clean=m_clean["p_current"])

        # isolation needs the CURRENT condition's span; locate once
        spans = locate_condition_spans(ex)
        bp = condition_block_positions(spans, current, args.block_which)
        block_pos = list(range(bp[0] - 1, len(ex["prompt_ids"]))) if bp else []   # span_start-1 .. end
        blocked_txt = mt.tokenizer.decode([ex["prompt_ids"][p] for p in block_pos]) if block_pos else "(not found)"

        def finalize(mode, headset, m, raw, extra):
            stub = isolate_current_span(raw, ex, current)
            oracle = judge_current(stub, ex["spec"].spec_texts[current])
            return dict(**common, mode=mode, headset=headset,
                        p_current=m["p_current"], dp_current=m["p_current"] - m_clean["p_current"],
                        current_is_argmax=m["current_is_argmax"], rank_current=m["rank_current"],
                        soft_p=[float(x) for x in m["p"]],
                        gen_stub=stub, gen_full=raw,
                        hard_verdict=oracle["verdict"], oracle=oracle,
                        survives=oracle["verdict"] == "YES", **extra)

        recs = []
        # clean read-out (mode="clean")
        if ("clean", "baseline") in need:
            raw = generate_restricted_to_span(ex["prompt_ids"], [], [args.sink_pos], q_start, args.max_new)
            recs.append(finalize("clean", "baseline", m_clean, raw, extra=dict(n_heads=0)))
        # ablation: restrict the SET to the sink
        for hs in args.head_sets:
            if ("ablation", hs) not in need: continue
            sel = head_sets[hs]
            ll = score_condition_lls(ex, refs, forward_fn=lambda inp, qp, s=sel:
                                     forward_restrict_to_span(inp, s, [args.sink_pos], qp))
            raw = generate_restricted_to_span(ex["prompt_ids"], sel, [args.sink_pos], q_start, args.max_new)
            recs.append(finalize("ablation", hs, soft_metrics_current(ll, current), raw,
                                 extra=dict(n_heads=len(sel), heads=sel, sink_pos=args.sink_pos)))
        # isolation: blind the COMPLEMENT to the CURRENT span
        if any(m == "isolation" for m, _ in need):
            if not block_pos:
                # span not located -> emit skip markers so we don't retry forever
                for hs in args.head_sets:
                    if ("isolation", hs) in need:
                        recs.append(dict(**common, mode="isolation", headset=hs, skipped="span_not_located",
                                         from_layer=args.from_layer, block_which=args.block_which))
            else:
                for hs in args.head_sets:
                    if ("isolation", hs) not in need: continue
                    comp = complement_heads(head_sets[hs], args.from_layer)
                    ll = score_condition_lls(ex, refs, forward_fn=lambda inp, qp, c=comp:
                                             forward_blocking_span(inp, c, block_pos, qp))
                    raw = generate_blocking_span(ex["prompt_ids"], comp, block_pos, q_start, args.max_new)
                    recs.append(finalize("isolation", hs, soft_metrics_current(ll, current), raw,
                                         extra=dict(n_blinded=len(comp), from_layer=args.from_layer,
                                                    block_which=args.block_which,
                                                    block_pos=[int(p) for p in block_pos],
                                                    blocked_txt=blocked_txt)))
        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # what each sample needs, given --mode / --head-sets
    # ─────────────────────────────────────────────────────────────────────────
    wanted = {("clean", "baseline")}
    if args.mode in ("ablation", "both"):
        wanted |= {("ablation", hs) for hs in args.head_sets}
    if args.mode in ("isolation", "both"):
        wanted |= {("isolation", hs) for hs in args.head_sets}

    # resume on (example_id, mode, headset) triples
    done = defaultdict(set)                                   # example_id -> {(mode, headset)}
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["example_id"]].add((r["mode"], r["headset"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    n_have = sum(len(v & wanted) for v in done.values())
    logger.info(f"resume: {n_have} of {args.n_samples * len(wanted)} target records present")

    # config.json (provenance) — full args + resolved head lists + rubric
    with open(config_path, "w") as f:
        json.dump({"args": vars(args), "model": args.model, "n_layer": N_LAYER, "n_heads": N_HEADS,
                   "tag": tag, "knob_str": knob_str, "wanted": sorted(wanted),
                   "head_set_sizes": {k: len(v) for k, v in head_sets.items()},
                   "head_sets": {k: [list(t) for t in v] for k, v in head_sets.items()},
                   "oracle_model": args.oracle_model, "rubric_tag": _HARD_TAG, "HARD_SYS": HARD_SYS},
                  f, indent=2)

    # ── main loop ──
    for i in range(args.n_samples):
        need = wanted - done.get(i, set())
        if not need:
            continue
        recs = run_one(i, need)
        with open(records_path, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush(); os.fsync(f.fileno())
        surv = {f"{r['mode'][:3]}:{r['headset']}": ("Y" if r.get("survives") else
                ("-" if r.get("skipped") else ".")) for r in recs}
        logger.info(f"[{i+1:>4}/{args.n_samples}] K={recs[0].get('K')} cur={recs[0].get('current')} :: {surv}")
    logger.info(f"done. records -> {records_path}")


if __name__ == "__main__":
    main()
