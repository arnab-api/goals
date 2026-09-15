#!/usr/bin/env python
"""Evaluate the Subspace Probe v2 with the GoalScope CAUSALITY metric on the locked 253-spec corpus.

Read-out producer (swapped into the eval_goalscope harness): at each pivot, read the source residual
``h`` at the probe's ``read_layer``, form ``o = Wᵀ W h``, and steer it (additive_norm, α) into the C++
causal stub via ``head_additive_generate`` — byte-for-byte the training student path. Everything else
(locked corpus, impl_span, 5 pivots, oracle hard_eval, summary) is the shared GoalScope harness.

By default only HELD-OUT frontier predicates are scored (the honest generalization test); pass
``--all_predicates`` to score the full corpus.

Results -> <results>/evaluation/subspace_probe_v2/<model>/causality/<tag>/nspec<N>/

Usage (14B smoke):
    conda run -n goals python scripts/evaluation/eval_subspace_probe_v2.py \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --cuda 0,1 \
        --probe <results>/subspace_probe_v2/Qwen2.5-Coder-14B-Instruct/r256__L37__a0.1/probe.pt \
        --split_file data/subspace_v2/teacher_targets/Qwen2.5-Coder-14B-Instruct/predicate_split__seed7__ho0.3.json \
        --limit 8
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
from src.goals.goalscope import _build_causality_prompt, head_additive_generate  # noqa: E402
from src.goals.subspace_probe import cache_source_residuals  # noqa: E402
from src.goals.subspace_probe_v2 import SubspaceProbeV2  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--datatypes", nargs="+", default=ALL_DATATYPES)
    ap.add_argument("--split_file", default=None, help="predicate-split json; restricts to held-out frontiers")
    ap.add_argument("--all_predicates", action="store_true", help="score the full corpus (ignore split)")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=32, help="read-out length")
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

    ckpt = torch.load(args.probe, map_location="cpu", weights_only=False)
    probe = SubspaceProbeV2(ckpt["config"])
    probe.W.data.copy_(ckpt["W"])
    probe = probe.to(mt.device).eval()
    read_layer = ckpt["read_layer"]
    band = list(ckpt["steer_layers"])
    epoch = ckpt.get("epoch", "?")
    logger.info(f"probe: {args.probe} | r={probe.config.r} read=L{read_layer} band={band[0]}-{band[-1]} "
                f"alpha={probe.config.alpha} ep{epoch}")

    # held-out predicate restriction
    heldout = None
    if args.split_file and not args.all_predicates:
        sp = json.loads(Path(args.split_file).read_text())
        heldout = {dt: set(sp[dt]["eval"]) for dt in sp}
        logger.info(f"held-out frontiers only: { {k: sorted(v) for k, v in heldout.items()} }")

    datapoints, _ = load_locked_datapoints(args.dataset, args.datatypes)
    if heldout is not None:
        datapoints = [(sid, inst) for sid, inst in datapoints
                      if inst.expected_next in heldout.get(inst.meta["datatype"], set())]
        logger.info(f"{len(datapoints)} datapoints after held-out filter")
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
        logger.warning(f"--limit {args.limit}: SMOKE on {len(datapoints)} datapoints")

    causal_prompt = _build_causality_prompt(mt)
    dest_inputs = prepare_input(prompts=causal_prompt, tokenizer=mt)
    model_dtype = next(mt._model.parameters()).dtype

    tag = f"r{probe.config.r}__L{read_layer}__a{probe.config.alpha}__ep{epoch}"
    scope = "all" if (heldout is None) else "heldout"
    run_dir = Path(results_dir) / "evaluation" / "subspace_probe_v2" / slug / "causality" / tag / f"{scope}_nspec{len(datapoints)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = fio.records_path(run_dir); samp_path = fio.samples_path(run_dir)
    if args.overwrite:
        rec_path.unlink(missing_ok=True); samp_path.unlink(missing_ok=True)
    done = set()
    if rec_path.exists():
        done = {json.loads(l)["spec"] for l in rec_path.read_text().splitlines() if l.strip()}
    fio.write_config(run_dir, dict(model=args.model, mode="causality", probe=args.probe,
                                   r=probe.config.r, read_layer=read_layer, steer_layers=band,
                                   alpha=probe.config.alpha, epoch=epoch, scope=scope,
                                   n_specs=len(datapoints), dataset=str(args.dataset),
                                   datatypes=list(args.datatypes), span_source="frontier"))
    logger.info(f"run dir -> {run_dir} | causality | {len(datapoints)} datapoints ({len(done)} done)")

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
        # ── read-out producer: o = WᵀW h steered into the C++ stub, batched over pivots ──
        res = cache_source_residuals(mt, full_text, apos, [read_layer])
        o = torch.stack([probe.readout_vec(res[p][read_layer].to(mt.device).float()) for p in apos])  # (B,d)
        gens = head_additive_generate(mt, dest_inputs, o.to(model_dtype), layers=band,
                                      max_new_tokens=args.max_new_tokens, strategy="additive_norm",
                                      alpha=probe.config.alpha)
        readouts = {p: g for p, g in zip(apos, gens)}

        sample_records = []
        for aname, pos in pivs.items():
            ro = CAUSAL_PREFIX + readouts.get(pos, "")
            hv = hard_eval(ro, impl_span, "causality")
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
    log_summary(summary, "causality")


if __name__ == "__main__":
    main()
