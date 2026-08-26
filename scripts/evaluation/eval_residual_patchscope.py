#!/usr/bin/env python
"""Residual-stream PatchScope baseline for the GoalScope evals.

The counterpart to ``eval_goalscope.py``: identical corpus, anchors, oracle rubric and scoring —
the *only* difference is **what** gets injected into the read-out probe.

    eval_goalscope.py          g_enc = summed o_proj contribution of the selected goal heads
    eval_residual_patchscope   h_src = the residual stream itself, averaged over --source-layers

That is the comparison experiment 1a asks for: how much of GoalScope's read-out quality comes from
the *head selection*, versus what you would get from the raw residual at the same positions?

Injection is norm-relative, matching ``--strategy additive_norm``:

    h_dest <- h_dest + alpha * ||h_dest|| * (v / ||v||)

Holding the injection mechanism fixed and varying only the source is the whole point. A plain
additive injection would not be comparable: ``||h_src||`` (a full residual) is far larger than
``||g_enc||`` (a sum over ~100 heads), so at fixed alpha the residual arm would be steered much
harder and the comparison would read magnitude rather than content. See ``PatchSpec`` strategy
``add_norm`` in ``src/functional.py``.

Self-contained under ``scripts/evaluation``: the corpus/anchor/oracle/scoring machinery is imported
from ``eval_goalscope`` (same package) and nothing is added to ``src/``.

Results land in the same tree as the GoalScope runs, so ``faithfulness_io`` parses them unchanged:

    <results>/evaluation/goalscope/<model>/<mode>/resid_norm/<alpha>/<leaf>/

with ``heads=resid:L<lo>-<hi>:<agg>`` as the leaf's head-set component, so a residual run can never
collide with a head-set run.

Usage:
    conda run -n goals python scripts/evaluation/eval_residual_patchscope.py \
        --mode faithfulness --alpha 0.1 --steer_layer_range 35 46

    conda run -n goals python scripts/evaluation/eval_residual_patchscope.py \
        --mode causality --alpha 0.1 --source-layers 25 56
"""

import argparse
import json
import logging
import sys
import time

sys.path.append(".")

import baukit
import torch
from tqdm.auto import tqdm

from scripts.evaluation import faithfulness_io as fio

# Every stage except the injection is shared with the head-set eval, so the two runs differ only
# in the thing under test.
from scripts.evaluation.eval_goalscope import (
    ANCHOR_ORDER,
    CAUSAL_PREFIX,
    ORACLE_MODEL,
    anchors_for_sample,
    build_causality_prompt,
    build_corpus,
    build_prompt,
    compute_summary,
    default_goal_band,
    generate_completion,
    hard_eval,
    kway_assign,
    log_summary,
    rule_based_spans,
)
from src.functional import PatchSpec, generate_with_patch
from src.goals.goalscope import _build_goalscope_prompt
from src.goals.multistep_label import label_generation
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, experiment_utils, logging_utils
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Residual extraction + injection
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_residual_latents(mt, prompt, positions, layers):
    """``(n_layers, n_pos, n_embd)`` residual-stream latents, one forward pass.

    Reads the *output* of each decoder layer (the residual stream after that block), which is the
    same tensor ``apply_goalscope`` injects into — so source and destination live in the same
    space, as the PatchScope construction requires.
    """
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    names = [mt.layer_name_format.format(l) for l in layers]
    idx = torch.tensor(list(positions))
    cache: dict[str, torch.Tensor] = {}

    def hook(output, layer):
        h = output[0] if isinstance(output, tuple) else output
        cache[layer] = h[0].index_select(0, idx.to(h.device)).float().cpu()
        return output

    with baukit.TraceDict(mt._model, names, edit_output=hook):
        mt._model(**dict(inp))
    missing = [n for n in names if n not in cache]
    assert not missing, f"no activations captured for {missing[:3]}"
    return torch.stack([cache[n] for n in names])  # (n_layers, n_pos, n_embd)


def aggregate_latents(stack, how="mean"):
    """Collapse the layer axis of ``(n_layers, n_pos, n_embd)`` -> ``(n_pos, n_embd)``.

    ``mean``   — average across the band (default). The band is where the goal heads live, so this
                 is the residual analogue of "the goal signal over that depth range".
    ``last``   — the deepest source layer only; closest to a textbook single-layer PatchScope.
    ``sum``    — magnitude is discarded by the norm-relative injection anyway, so this differs from
                 ``mean`` only if you later switch to a plain additive strategy.
    """
    if how == "mean":
        return stack.mean(0)
    if how == "last":
        return stack[-1]
    if how == "sum":
        return stack.sum(0)
    raise ValueError(f"unknown --source-agg {how!r}")


@torch.no_grad()
def residual_patchscope(
    mt,
    prompt,
    token_positions,
    source_layers,
    steer_layer_range,
    alpha=0.1,
    source_agg="mean",
    max_new_tokens=64,
    batch_size=16,
    n_steer_tokens="all",
    probe_prompt=None,
):
    """Verbalize each source position by injecting its residual latent into the read-out probe.

    Mirrors ``apply_goalscope``'s additive path — same probe, same greedy decode, same
    ``n_steer_tokens`` semantics — with the residual latent in place of the head aggregate.

    Returns ``[{"pos", "token", "patched_gen", "scoped"}]``, the shape ``faithfulness_io`` and
    ``render_patchscope_html`` both expect.
    """
    positions = list(token_positions)
    stack = extract_residual_latents(mt, prompt, positions, source_layers)
    vecs = aggregate_latents(stack, source_agg)  # (n_pos, n_embd)
    # Unit direction: `add_norm` supplies the magnitude from ||h_dest|| at each patch site.
    vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dest = prepare_input(
        prompts=probe_prompt or _build_goalscope_prompt(mt), tokenizer=mt
    )
    dest_ids, dest_attn = dest.input_ids, dest.attention_mask
    steer_layers = list(range(*steer_layer_range))
    n_steer = None if str(n_steer_tokens).lower() == "all" else int(n_steer_tokens)

    src_ids = prepare_input(prompts=prompt, tokenizer=mt).input_ids[0].tolist()
    out = []
    for s in tqdm(
        range(0, len(positions), batch_size), desc=f"resid[a={alpha}]", leave=False
    ):
        chunk = positions[s : s + batch_size]
        B = len(chunk)
        tiled = TokenizerOutput(
            data=dict(
                input_ids=dest_ids.expand(B, -1).clone(),
                attention_mask=dest_attn.expand(B, -1).clone(),
            )
        )
        v = (alpha * vecs[s : s + B]).to(mt._model.dtype)
        patches = [
            PatchSpec(
                location=(mt.layer_name_format.format(li), -1),
                patch=v,
                strategy="add_norm",
                on_pass="all",
            )
            for li in steer_layers
        ]
        gens = generate_with_patch(
            mt,
            tiled,
            max_new_tokens=max_new_tokens,
            patches=patches,
            n_steer_tokens=n_steer,
            do_sample=False,
            patch_interface="baukit",
            remove_prefix=True,
            skip_special_tokens=True,
        )
        for i, p in enumerate(chunk):
            out.append(
                dict(
                    pos=int(p),
                    token=mt.tokenizer.decode([src_ids[p]]),
                    patched_gen=gens[i],
                    scoped=True,
                )
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", default="faithfulness", choices=["faithfulness", "causality"]
    )
    ap.add_argument(
        "--eval-k-way",
        action="store_true",
        dest="eval_k_way",
        help="also run the k-way oracle (faithfulness only)",
    )
    ap.add_argument(
        "--source-layers",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="residual layers to read the source latent from, range(START, END). "
        "Default: the model's GOAL_BANDS entry, so the residual baseline covers "
        "the same depth as the goal heads it is being compared against.",
    )
    ap.add_argument(
        "--source-agg",
        default="mean",
        choices=["mean", "last", "sum"],
        help="how to collapse the source-layer axis (default mean)",
    )
    ap.add_argument(
        "--steer_layer_range",
        type=int,
        nargs=2,
        default=(35, 46),
        metavar=("START", "END"),
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="injected fraction of ||h_dest|| (norm-relative; default 0.1)",
    )
    ap.add_argument("--n_steer_tokens", default="all", help="'all' | int")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--datatype", default="int")
    ap.add_argument("--n_specs", type=int, default=512)
    ap.add_argument("--num_conditions", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gen_max_new_tokens", type=int, default=256)
    ap.add_argument("--span_source", default="rule", choices=["rule", "oracle"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="wipe an existing run dir; default is to RESUME",
    )
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    if args.mode == "causality" and args.eval_k_way:
        raise SystemExit("--eval-k-way is not implemented for --mode causality yet")

    n_steer = (
        "all" if str(args.n_steer_tokens).lower() == "all" else int(args.n_steer_tokens)
    )
    # Source band needs n_layer for the fallback, so it is resolved after the model loads; the
    # explicit case is known now and is what the run dir keys on.
    src_band = tuple(args.source_layers) if args.source_layers else None

    def leaf_heads_name(band):
        return f"resid:L{band[0]}-{band[1]}:{args.source_agg}"

    config = dict(
        mode=args.mode,
        eval_k_way=args.eval_k_way,
        # `strategy` is a path component, so residual runs never land in a head-set run's dir
        strategy="resid_norm",
        alpha=args.alpha,
        source_layers=list(src_band) if src_band else None,
        source_layers_explicit=args.source_layers is not None,
        source_agg=args.source_agg,
        heads_name=leaf_heads_name(src_band) if src_band else None,
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
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # The run dir needs heads_name, which needs the resolved band. Resolve it without the model
    # when possible (registered model) so the resume check can run before the 32B load.
    if src_band is None:
        try:
            src_band = default_goal_band(args.model)
            config["source_layers"] = list(src_band)
            config["heads_name"] = leaf_heads_name(src_band)
        except ValueError:
            pass  # unregistered model: needs n_layer, resolved after load

    mt = None
    if config["heads_name"] is None:
        mt = ModelandTokenizer(
            model_key=args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        src_band = default_goal_band(args.model, n_layer=mt.n_layer)
        config["source_layers"] = list(src_band)
        config["heads_name"] = leaf_heads_name(src_band)

    run_dir = fio.run_dir_for(config, results_dir)
    if args.overwrite and run_dir.exists():
        for f in ("records.jsonl", "samples.jsonl", "samples.json", "summary.json"):
            (run_dir / f).unlink(missing_ok=True)
        logger.info(f"--overwrite: cleared existing run at {run_dir}")
    prev = None if args.overwrite else fio.read_config(run_dir)
    # Same guard as the head-set eval: the leaf encodes the source band and aggregation, but not
    # every knob, so refuse to resume a run whose source differs from this invocation's.
    if prev is not None:
        for k in ("source_layers", "source_agg", "strategy"):
            if prev.get(k) is not None and prev.get(k) != config[k]:
                raise SystemExit(
                    f"run dir {run_dir} holds a run with {k}={prev[k]!r}, but this invocation "
                    f"passed {config[k]!r}. Resuming would mix them — pass --overwrite."
                )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"run dir -> {run_dir}")

    samples = build_corpus(args.n_specs, args.num_conditions, args.seed, args.datatype)
    completed = fio.read_completed_spec_ids(run_dir)
    kept = fio.prune_orphan_records(run_dir, completed)
    remaining = [s for s in samples if s.spec.spec_id not in set(completed)]
    logger.info(
        f"corpus: {len(samples)} samples | already done: {len(completed)} "
        f"({kept} records kept) | remaining: {len(remaining)}"
    )

    if not remaining:
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

    if mt is None:
        mt = ModelandTokenizer(
            model_key=args.model,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    source_layers = list(range(*src_band))
    config["n_source_layers"] = len(source_layers)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fio.write_config(run_dir, config)
    logger.info(
        f"residual source: layers {src_band[0]}-{src_band[1] - 1} ({len(source_layers)}) "
        f"agg={args.source_agg} -> steer {tuple(args.steer_layer_range)} @ alpha={args.alpha}"
    )

    probe_prompt = build_causality_prompt(mt) if args.mode == "causality" else None
    readout_prefix = CAUSAL_PREFIX if args.mode == "causality" else ""

    rec_fh = open(fio.records_path(run_dir), "a")
    samp_fh = open(fio.samples_path(run_dir), "a")
    n_done = len(completed)
    for sample in remaining:
        sample_records = []
        prompt = build_prompt(mt, sample)
        completion = generate_completion(mt, prompt, args.gen_max_new_tokens)
        full_text = prompt + completion

        if args.span_source == "rule":
            conditions = list(sample.spec.spec_texts)
            impl_spans = rule_based_spans(
                full_text, len(prompt), len(sample.spec.predicate_ids)
            )
            labels_meta = [
                dict(step_text=c, gen_code=(s or ""), gen_token_range=None)
                for c, s in zip(conditions, impl_spans)
            ]
        else:
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

        res = residual_patchscope(
            mt,
            full_text,
            token_positions=apos,
            source_layers=source_layers,
            steer_layer_range=tuple(args.steer_layer_range),
            alpha=args.alpha,
            source_agg=args.source_agg,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            n_steer_tokens=args.n_steer_tokens,
            probe_prompt=probe_prompt,
        )
        readouts = {r["pos"]: r for r in res}

        for row in rows:
            true_id = sample.spec.predicate_ids[row["cond"]]
            for aname, pos in row["anchors"].items():
                ro = readout_prefix + readouts.get(pos, {}).get("patched_gen", "")
                hv = hard_eval(ro, row["impl_span"], args.mode)
                logger.debug(
                    "[%s c%d %s] read-out: %r -> %s",
                    sample.spec.spec_id,
                    row["cond"],
                    aname,
                    ro,
                    hv["verdict"],
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
        # records first, then the meta line — a crash mid-sample leaves prunable orphans rather
        # than a phantom-complete sample (same contract as eval_goalscope)
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

    all_records = [
        json.loads(l)
        for l in fio.records_path(run_dir).read_text().splitlines()
        if l.strip()
    ]
    summary = compute_summary(
        all_records, n_samples=len(fio.read_completed_spec_ids(run_dir))
    )
    fio.write_summary(run_dir, summary)
    logger.info(f"saved -> {run_dir}")
    log_summary(summary, args.mode)


if __name__ == "__main__":
    main()
