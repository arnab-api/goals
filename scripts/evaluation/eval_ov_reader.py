#!/usr/bin/env python
"""OV-reader eval on the DIFF prefill-to-frontier setup — does the residual already carry the goal?

For each datapoint we build a multi-predicate ``func`` task (letters naming ``a/b/c/…``), prefill the
first ``j`` single-line predicates and stop at the frontier's ``{var} =`` (DIFF ``var_eq`` pivot), then
greedy-decode ``--gen_max_new_tokens`` tokens. The frontier predicate is the KNOWN goal; its code is
exactly the first generated line (no oracle span-labelling needed). At five pivots — the last four
prompt tokens ``(prev_end, indent, var, eq)`` plus the first generated token ``(first_impl)`` — we run:

  * SKYLINE — ``apply_goalscope`` (additive_norm), i.e. the goal heads' actual OUTPUT read-out.
  * OVReader — ``OVReader.apply_probe`` across ``--layers``: the OV-write direction of the residual
    after each layer, steered into the same probe.

Read-outs are graded with the SAME oracle rubric/pipeline as ``eval_goalscope.py`` (``--mode``
faithfulness | causality; causality steers into a C++ stub and prepends it before grading).

Results:  <results>/evaluation/ov-reader/<model>/<layer_idx>/<mode>/<leaf>/  (layer_idx=int | "goalscope")
Parse with ``scripts.evaluation.ov_reader_io`` (``across_layers_df`` / ``grid_df``). Resume-safe:
``--datatypes`` and ``--lang`` are baked into the leaf, so disjoint invocations run in parallel.

Usage:
    conda run -n goals python scripts/evaluation/eval_ov_reader.py \
        --datatypes int str --lang python --mode faithfulness --cuda 0

    # parallel shards (disjoint dirs):
    ... --datatypes int      --lang python --cuda 0
    ... --datatypes str      --lang python --cuda 1
    ... --datatypes list[int] list[str] --lang cpp --cuda 2
"""
import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time

sys.path.append(".")

# --cuda must be applied before torch initialises the CUDA context (mirrors scripts/diff/*).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_cuda = _pre.parse_known_args()[0].cuda
if _cuda is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda)

import torch  # noqa: E402

import src.diff.data as D  # noqa: E402
from scripts.evaluation import ov_reader_io as ovio  # noqa: E402
from scripts.evaluation.eval_goalscope import (CAUSAL_PREFIX, ORACLE_MODEL,  # noqa: E402
                                               hard_eval, load_json_heads)
from src.goals.goalscope import (OVReader, _build_causality_prompt,  # noqa: E402
                                 apply_goalscope, greedy_continue)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)

STEER_LAYER_RANGE = (35, 46)  # GoalScope steer band (shared by skyline + OVReader)
DEFAULT_HEADS_JSON = "head_sets/{slug}/goal__lam20_thr0.9_band25-56.json"  # the 122 goal heads


# ══════════════════════════════════════════════════════════════════════════════
# Corpus — prefill-to-frontier datapoints (letters naming), built from QTASKS predicates
# ══════════════════════════════════════════════════════════════════════════════
def sample_datapoints(datatype, n_conditions, n_samples, lang, seed):
    """``n_samples`` QTransferInstances: each draws ``T`` predicates (T in ``n_conditions``), prefills
    the first ``j`` (1 <= j < T), and sets the frontier = predicate ``j`` (letters naming). Deterministic
    in ``seed`` -> resume-stable. Deduped by (frontier id, prefilled ids)."""
    bank = D.QTASKS[datatype]
    preds = list(bank.predicates)
    # The frontier (measured target) must NOT be a trivial sign/zero comparison — the no-goal
    # causality default emits exactly those, so a trivial frontier scores as a false positive
    # (see D.TRIVIAL_FRONTIERS). Trivial predicates may still appear as prefilled context.
    trivial = D.TRIVIAL_FRONTIERS.get(datatype, frozenset())
    ks = [k for k in n_conditions if 2 <= k <= len(preds)]
    if not ks:
        raise SystemExit(
            f"--n_conditions has no value in [2, {len(preds)}] for datatype {datatype!r}"
        )
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 400):
        if len(out) >= n_samples:
            break
        T = rng.choice(ks)
        subset = rng.sample(preds, T)
        # frontier sits at index j (1..T-1, so >=1 prefilled); constrain j so it is non-trivial
        elig_j = [jj for jj in range(1, T) if subset[jj].id not in trivial]
        if not elig_j:
            continue
        j = rng.choice(elig_j)
        ids = [p.id for p in subset]
        key = (ids[j], tuple(ids[:j]))
        if key in seen:
            continue
        seen.add(key)
        task = dataclasses.replace(bank, id=f"{bank.id}_T{T}", predicates=tuple(subset))
        out.append(D.QTransferInstance(
            task=task, var_name=bank.var_names[0], presentation="listed",
            src_lang=lang, dst_lang=lang, src_progress=ids[:j], dst_progress=ids[:j],
            expected_next=ids[j], naming="letters",
            meta={"datatype": datatype, "T": T, "j": j},
        ))
    if len(out) < n_samples:
        logger.warning(
            f"[{datatype}/{lang}] only {len(out)}/{n_samples} distinct datapoints available"
        )
    return out


def _chat_prompt(mt, question, prefill):
    """Replicate src.diff.functional.get_prompt without depending on its module globals."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant proficient in programming. "
         "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    try:
        return mt.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True,
            tokenize=False, chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True, tokenize=False,
        )


def build_prompt(mt, inst, lang):
    """Prefilled source prompt ending at the frontier's ``{var} =`` (var_eq stub, letters)."""
    q, pf = inst.src_parts()
    q = q.replace("computes:", "computes in sequential order:")  # nudge the model to follow the listing
    var = inst.var_of(inst.expected_next)
    body = pf.rstrip("\n ")  # drop the trailing indent line
    if lang == "cpp":
        for pid in inst.src_progress:
            body = D._cpp_autoify(body, inst.var_of(pid))
        pf2 = body + f"\n    auto {var} ="
    else:
        pf2 = body + f"\n    {var} ="
    return _chat_prompt(mt, q, D._finalize_prefill(pf2)), var


# ══════════════════════════════════════════════════════════════════════════════
# Per-datapoint probing
# ══════════════════════════════════════════════════════════════════════════════
def pivot_positions(mt, prompt, full_text, pivots):
    """{pivot: absolute token pos} for the requested pivots. prev_end/indent/var/eq = the last four
    tokens of the prompt; first_impl = the first generated token. Positions index the tokenization of
    ``full_text`` (what both apply_goalscope and OVReader see)."""
    prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    full = prepare_input(prompts=full_text, tokenizer=mt)["input_ids"][0].tolist()
    P = len(prompt_ids)
    if full[:P] != prompt_ids:  # boundary token merged on re-tokenisation — fall back to common prefix
        P = next((i for i in range(min(len(full), len(prompt_ids)))
                  if full[i] != prompt_ids[i]), min(len(full), len(prompt_ids)))
    idx = {"prev_end": P - 4, "indent": P - 3, "var": P - 2, "eq": P - 1, "first_impl": P}
    return {p: idx[p] for p in pivots if 0 <= idx[p] < len(full)}


def frontier_impl_span(prompt, completion, var):
    """The frontier condition's code = its ``{var} =`` line + the first generated line (single-line
    predicate, so this is exact — no oracle span-labelling needed)."""
    last_prompt_line = prompt.rsplit("\n", 1)[-1]  # "    c =" | "    auto c ="
    return (last_prompt_line + completion.split("\n", 1)[0]).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--mode", default="faithfulness", choices=["faithfulness", "causality"])
    ap.add_argument("--datatypes", nargs="+", default=list(D.DATATYPES),
                    help="datatype(s); each is an independent (parallel-safe) shard")
    ap.add_argument("--lang", nargs="+", default=["python"], choices=["python", "cpp"],
                    help="source-prompt language(s); each is an independent shard")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="probe layers (default: 0..36 step6, 38..54 step2, 56..n_layer step4, + last layer)")
    ap.add_argument("--heads", default=None,
                    help="json head set (default: the model's 122-head goal__lam20_thr0.9_band25-56.json)")
    ap.add_argument("--n_conditions", type=int, nargs="+", default=[2, 3, 4],
                    help="total conditions per task; prefill count j is sampled in 1..T-1")
    ap.add_argument("--pivot", nargs="+", default=ovio.PIVOT_ORDER, choices=ovio.PIVOT_ORDER)
    ap.add_argument("--alpha", type=float, default=0.15, help="OVReader steering alpha (additive_norm)")
    ap.add_argument("--skyline_alpha", type=float, default=0.1, help="GoalScope skyline alpha")
    ap.add_argument("--n_samples", type=int, default=64, help="datapoints per datatype")
    ap.add_argument("--max_new_tokens", type=int, default=64, help="read-out length per pivot")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64,
                    help="frontier-body greedy budget (needs >=1 for first_impl)")
    ap.add_argument("--span_source", default="rule", choices=["rule", "oracle"],
                    help="impl-span for grading: 'rule' (first generated line, exact here) — 'oracle' reserved")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--overwrite", action="store_true", help="wipe matching run dirs and redo")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def _target_dirs(config, layers, results_dir):
    """{layer_idx: dir} — the goalscope skyline dir plus one dir per probe layer, all sharing leaf."""
    dirs = {ovio.SKYLINE_IDX: ovio.run_dir_for(config, ovio.SKYLINE_IDX, results_dir)}
    for L in layers:
        dirs[L] = ovio.run_dir_for(config, L, results_dir)
    return dirs


def run_shard(mt, ov, heads, datatype, lang, args, layers, results_dir, base_config):
    config = dict(base_config, datatype=datatype, lang=lang)
    dirs = _target_dirs(config, layers, results_dir)

    if args.overwrite:
        for d in dirs.values():
            for f in ("records.jsonl", "samples.jsonl", "summary.json"):
                (d / f).unlink(missing_ok=True)

    # resolve resume state: a datapoint is done only if present in EVERY target dir.
    completed = {}
    for idx, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        done = set(ovio.read_completed_spec_ids(d)) if args.resume else set()
        ovio.prune_orphan_records(d, done)
        cfg = dict(config, layer_idx=idx, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        ovio.write_config(d, cfg)
        completed[idx] = done

    datapoints = sample_datapoints(datatype, args.n_conditions, args.n_samples, lang, args.seed)
    spec_ids = [f"{datatype}_{lang}_{i:03d}" for i in range(len(datapoints))]
    remaining = [(sid, inst) for sid, inst in zip(spec_ids, datapoints)
                 if any(sid not in completed[idx] for idx in dirs)]
    logger.info(f"[{datatype}/{lang}] {len(datapoints)} datapoints | remaining {len(remaining)} "
                f"| dirs {len(dirs)} (goalscope + {len(layers)} layers)")
    if not remaining:
        _finalize(dirs)
        return

    goalscope_prompt = _build_causality_prompt(mt) if args.mode == "causality" else None
    readout_prefix = CAUSAL_PREFIX if args.mode == "causality" else ""
    fhs = {idx: (open(ovio.records_path(d), "a"), open(ovio.samples_path(d), "a"))
           for idx, d in dirs.items()}
    try:
        for n, (sid, inst) in enumerate(remaining, 1):
            prompt, var = build_prompt(mt, inst, lang)
            completion = greedy_continue(
                mt, **prepare_input(prompts=prompt, tokenizer=mt),
                max_new_tokens=args.gen_max_new_tokens,
            )
            full_text = prompt + completion
            impl_span = frontier_impl_span(prompt, completion, var)
            frontier_desc = inst.expected_next_desc
            pivs = pivot_positions(mt, prompt, full_text, args.pivot)
            if not pivs:
                logger.warning(f"[{sid}] no valid pivots — skipping")
                continue
            apos = sorted(set(pivs.values()))
            meta = dict(
                spec_id=sid, datatype=datatype, lang=lang, T=inst.meta["T"], j=inst.meta["j"],
                var=var, frontier_id=inst.expected_next, frontier_desc=frontier_desc,
                prompt=prompt, completion=completion, impl_span=impl_span, pivots=pivs,
            )

            def _record(pivot, pos, readout, layer_idx):
                ro = readout_prefix + (readout or "")
                hv = hard_eval(ro, impl_span, args.mode)
                return dict(
                    spec=sid, datatype=datatype, lang=lang, pivot=pivot, pos=pos,
                    layer_idx=layer_idx, readout=ro, impl_span=impl_span,
                    frontier_id=inst.expected_next, condition=frontier_desc,
                    hard=hv["verdict"], hard_prompt=hv["prompt"], hard_response=hv["response"],
                )

            # ── skyline: goal-head OUTPUT read-out at each pivot (one apply_goalscope call) ──
            sky = {r["pos"]: r for r in apply_goalscope(
                full_text, mt=mt, selected_heads=heads, strategy="additive_norm",
                steer_layer_range=STEER_LAYER_RANGE, alpha=args.skyline_alpha,
                n_steer_tokens="all", max_new_tokens=args.max_new_tokens, show=False,
                include_noncode=False, token_positions=apos, goalscope_prompt=goalscope_prompt,
            )}
            recs = {ovio.SKYLINE_IDX: [_record(p, pos, sky.get(pos, {}).get("patched_gen", ""),
                                               ovio.SKYLINE_IDX) for p, pos in pivs.items()]}
            # ── OVReader: one batched call per pivot -> {layer: readout} ──
            for L in layers:
                recs[L] = []
            for pivot, pos in pivs.items():
                by_layer = ov.apply_probe_on_prompt(
                    full_text, pos, layers, mode=args.mode, steer_layers=list(range(*STEER_LAYER_RANGE)),
                    max_new_tokens=args.max_new_tokens, alpha=args.alpha,
                )
                for L in layers:
                    recs[L].append(_record(pivot, pos, by_layer.get(L, ""), L))

            for idx, (rec_fh, samp_fh) in fhs.items():
                if sid in completed[idx]:
                    continue
                for r in recs[idx]:
                    ovio.append_jsonl(rec_fh, r)
                ovio.append_jsonl(samp_fh, meta)
            logger.info(f"[{datatype}/{lang}] [{n}/{len(remaining)}] {sid}: "
                        f"{len(pivs)} pivots x {len(layers)} layers + skyline scored")
    finally:
        for rec_fh, samp_fh in fhs.values():
            rec_fh.close()
            samp_fh.close()
    _finalize(dirs)


def _finalize(dirs):
    for d in dirs.values():
        recs = [json.loads(ln) for ln in ovio.records_path(d).read_text().splitlines()
                if ln.strip()] if ovio.records_path(d).exists() else []
        n_samp = len(ovio.read_completed_spec_ids(d))
        summary = ovio.compute_summary(recs, n_samp)
        ovio.write_summary(d, summary)
        o = summary["overall"]
        logger.info(f"  {d.parent.parent.name:>10}  n={summary['n_records']:<5d} "
                    f"hard_yes={o['hard_yes']:.3f}  hvy_loop={o['heavy_loop']:.3f}")


def main():
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR
    slug = args.model.split("/")[-1]

    heads_path = args.heads or DEFAULT_HEADS_JSON.format(slug=slug)

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    n_heads = mt.model.config.num_attention_heads
    heads = load_json_heads(heads_path, mt.n_layer, n_heads)
    heads_name = os.path.splitext(os.path.basename(heads_path))[0]
    layers = args.layers or sorted(set(
        list(range(0, 37, 6)) + list(range(38, 56, 2))
        + list(range(56, mt.n_layer, 4)) + [mt.n_layer - 1]  # always include the last layer
    ))
    logger.info(f"heads[{heads_name}] = {len(heads)} | probe layers = {layers}")

    ov = OVReader(mt, heads)  # readers cached once, reused across every shard/pivot/datapoint

    base_config = dict(
        model=args.model, mode=args.mode, heads_spec=heads_path, heads_name=heads_name,
        n_heads=len(heads), selected_heads=[[int(lyr), int(h)] for lyr, h in heads],
        alpha=args.alpha, skyline_alpha=args.skyline_alpha, steer_layer_range=list(STEER_LAYER_RANGE),
        n_conditions=list(args.n_conditions), pivots=list(args.pivot), n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens, gen_max_new_tokens=args.gen_max_new_tokens,
        span_source=args.span_source, seed=args.seed, layers=layers, oracle_model=ORACLE_MODEL,
    )

    for datatype in args.datatypes:
        for lang in args.lang:
            run_shard(mt, ov, heads, datatype, lang, args, layers, results_dir, base_config)
    logger.info("done.")


if __name__ == "__main__":
    main()
