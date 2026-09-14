#!/usr/bin/env python
"""Evaluate the trained SUBSPACE PROBE (notebooks/11_goalscope_subspace.ipynb) with the SAME
faithfulness / causality check GoalScope uses.

The grader `eval_goalscope.hard_eval(readout, impl_span, mode)` is agnostic to how the readout was
produced, so we reuse the entire GoalScope harness (locked 253-spec corpus, build_prompt, the exact
frontier impl_span, the 5 pivots, the oracle hard-eval, the summary) and swap ONLY the readout
producer: for each pivot we cache the source residual and verbalize it through the trained subspace
probe (per-layer W_lᵀW_l projection + soft prompt) instead of the goal-head OV readout.

  faithfulness  — NL "helpful assistant" dest template the probe was trained on (faithful reproduction).
  causality     — C++ stub prefill (CAUSAL_PREFIX) as the assistant turn; the soft prompt stays the
                  user turn. NOTE: OFF-DISTRIBUTION — the probe was only trained on the NL template, so
                  low causality numbers reflect that train/eval mismatch, not just probe quality.

Results -> <results>/evaluation/subspace_probe/<model>/<mode>/<probe_tag>/nspec<N>/
           records.jsonl, samples.jsonl, summary.json, config.json   (separate from goal-head goalscope)

Usage:
    conda run -n goals python scripts/evaluation/eval_subspace_probe.py --mode faithfulness
    conda run -n goals python scripts/evaluation/eval_subspace_probe.py --mode causality
    # smoke: ... --limit 8
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.append(".")

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--cuda", default=None)
_c = _pre.parse_known_args()[0].cuda
if _c is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_c)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from scripts.evaluation import faithfulness_io as fio  # noqa: E402
from scripts.evaluation.eval_goalscope import (  # noqa: E402
    ALL_DATATYPES, ANCHOR_ORDER, CAUSAL_PREFIX, DEFAULT_DATASET, build_prompt,
    compute_summary, frontier_impl_span, generate_completion, hard_eval,
    load_locked_datapoints, log_summary, pivot_positions,
)
from src.goals.subspace_probe import (  # noqa: E402
    SubspaceProbe, _DEST_SYSTEM, _SOFT_SENTINEL, build_dest_template_parts,
    cache_source_residuals, verbalize_batch,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def build_causal_dest_parts(mt, system=_DEST_SYSTEM, assistant_prefill=CAUSAL_PREFIX):
    """Dest template for the causality read-out: soft tokens stay the user turn (as trained), the
    assistant turn is prefilled with the C++ stub CAUSAL_PREFIX so the probe continues C++ code.
    Mirrors subspace_probe.build_dest_template_parts (split at the soft-token sentinel)."""
    tok = mt.tokenizer
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": _SOFT_SENTINEL},
            {"role": "assistant", "content": assistant_prefill}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                               continue_final_message=True)
    before_str, after_str = s.split(_SOFT_SENTINEL)
    before = torch.tensor(tok.encode(before_str, add_special_tokens=False), dtype=torch.long)
    after = torch.tensor(tok.encode(after_str, add_special_tokens=False), dtype=torch.long)
    return before, after


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (applied before torch import)")
    ap.add_argument("--probe", default=None,
                    help="path to probe.pt (default <results>/subspace_probe/<slug>/r=256/probe.pt)")
    ap.add_argument("--mode", default="faithfulness", choices=["faithfulness", "causality"])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--datatypes", nargs="+", default=ALL_DATATYPES)
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="frontier-body greedy budget")
    ap.add_argument("--max_new_tokens", type=int, default=32, help="read-out (verbalization) length")
    ap.add_argument("--chunk_size", type=int, default=8, help="pivots verbalized per generate() call")
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: cap datapoints (round-robin)")
    ap.add_argument("--overwrite", action="store_true")
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR
    slug = args.model.split("/")[-1]

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")

    probe_path = args.probe or os.path.join(results_dir, "subspace_probe", slug, "r=256", "probe.pt")
    ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe = SubspaceProbe(ckpt["config"])
    probe.W.data.copy_(ckpt["W"]); probe.soft_tokens.data.copy_(ckpt["soft_tokens"])
    probe = probe.to(mt.device).eval()
    layers = list(probe.config.layers)
    epoch = ckpt.get("epoch", "?")
    logger.info(f"probe: {probe_path} | r={probe.config.r} layers={min(layers)}-{max(layers)} "
                f"soft={probe.config.n_soft_tokens} epoch={epoch}")

    # dest template (readout producer): NL for faithfulness (as trained), C++ stub for causality
    if args.mode == "causality":
        before_ids, after_ids = build_causal_dest_parts(mt)
        readout_prefix = CAUSAL_PREFIX
        logger.warning("causality mode: OFF-DISTRIBUTION for this probe (trained on the NL template only)")
    else:
        before_ids, after_ids = build_dest_template_parts(mt)
        readout_prefix = ""

    datapoints, _ = load_locked_datapoints(args.dataset, args.datatypes)
    if args.limit:
        from collections import defaultdict
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

    tag = f"r{probe.config.r}__L{min(layers)}_{max(layers)}__ep{epoch}"
    run_dir = Path(results_dir) / "evaluation" / "subspace_probe" / slug / args.mode / tag / f"nspec{len(datapoints)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = fio.records_path(run_dir); samp_path = fio.samples_path(run_dir)
    if args.overwrite:
        rec_path.unlink(missing_ok=True); samp_path.unlink(missing_ok=True)
    done = set()
    if rec_path.exists():
        done = {json.loads(l)["spec"] for l in rec_path.read_text().splitlines() if l.strip()}
    fio.write_config(run_dir, dict(model=args.model, mode=args.mode, probe=probe_path, r=probe.config.r,
                                   layers=layers, epoch=epoch, n_specs=len(datapoints),
                                   dataset=str(args.dataset), datatypes=list(args.datatypes),
                                   max_new_tokens=args.max_new_tokens, span_source="frontier"))
    logger.info(f"run dir -> {run_dir} | {args.mode} | {len(datapoints)} datapoints ({len(done)} done)")

    rec_fh = open(rec_path, "a"); samp_fh = open(samp_path, "a")
    n_done = len(done)
    for sid, inst in datapoints:
        if sid in done:
            continue
        prompt, var = build_prompt(mt, inst, inst.src_lang)
        completion = generate_completion(mt, prompt, args.gen_max_new_tokens)
        full_text = prompt + completion
        impl_span = frontier_impl_span(prompt, completion, var)
        frontier_desc = inst.expected_next_desc

        pivs = pivot_positions(mt, prompt, full_text, ANCHOR_ORDER)
        if not pivs:
            logger.warning(f"[{sid}] no valid pivots — skipping")
            continue
        apos = sorted(set(pivs.values()))
        # ── readout producer: subspace-probe verbalization at each pivot ──
        res = cache_source_residuals(mt, full_text, apos, layers)          # {pos: {layer: (d,)}}
        verbs = verbalize_batch(mt, probe, before_ids, after_ids, [res[p] for p in apos],
                                max_new_tokens=args.max_new_tokens, chunk_size=args.chunk_size)
        readouts = {p: v for p, v in zip(apos, verbs)}

        sample_records = []
        for aname, pos in pivs.items():
            ro = readout_prefix + readouts.get(pos, "")
            hv = hard_eval(ro, impl_span, args.mode)
            sample_records.append(dict(
                spec=sid, datatype=inst.meta["datatype"], anchor=aname, pos=pos,
                readout=ro, impl_span=impl_span, frontier_id=inst.expected_next,
                condition=frontier_desc, hard=hv["verdict"], hard_prompt=hv["prompt"],
                hard_response=hv["response"]))
        for r in sample_records:
            fio.append_jsonl(rec_fh, r)
        fio.append_jsonl(samp_fh, dict(spec_id=sid, datatype=inst.meta["datatype"], lang=inst.src_lang,
                                       frontier_id=inst.expected_next, var=var, prompt=prompt,
                                       completion=completion, impl_span=impl_span, pivots=pivs))
        n_done += 1
        logger.info(f"[{n_done}/{len(datapoints)}] {sid}: {len(pivs)} pivot read-outs scored")
    rec_fh.close(); samp_fh.close()

    all_records = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    n_samples = len({json.loads(l)["spec_id"] for l in samp_path.read_text().splitlines() if l.strip()})
    summary = compute_summary(all_records, n_samples)
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {run_dir}")
    log_summary(summary, args.mode)


if __name__ == "__main__":
    main()
