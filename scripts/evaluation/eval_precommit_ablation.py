#!/usr/bin/env python
"""PRE-COMMITMENT ablation: is the upcoming sub-goal's encoding at the pre-implementation pivots needed?

On the locked 253-spec Python corpus (``data/goalscope_eval.json``; the frontier is b / c / d = sub-task
index j in {1, 2, 3}), the prompt ends at the frontier's ``    {var} =``. Its last four tokens are the
pivots ``prev_end`` (the token holding the previous line's newline, e.g. ``\\n`` or ``)\\n``),
``indent``, ``var`` (`` b``) and ``eq`` (`` =``).

Ablation (filter paper, Fig. 9 / App. B): at each ablated pivot, replace the residual stream at EVERY
layer with the latents of the same token read in a context-free prompt: ``--n_pad`` pad tokens followed
by the bare token (the pads keep the token off position 0, which would make it an attention sink).
Default ablated pivots: prev_end, indent, var. ``eq`` (the token whose prediction starts the frontier
implementation) is left intact. Because every layer is replaced, the keys/values that later tokens read
at those positions are context-free too.

Per spec, two conditions — ``clean`` and ``ablated``:
  1. greedy-decode the frontier RHS (first line, <= 64 tokens) and grade it with the Section-4 ablation
     judge (``judge_implements``: does the code implement the frontier description?);
  2. GoalScope read-out at ``eq`` (122 goal heads, additive_norm alpha 0.1, steer layers 35-46 = the
     paper's faithfulness config) from the clean / ablated source forward, graded with the faithfulness
     judge (``hard_eval``) against the CLEAN frontier implementation.

Results -> ``<results>/evaluation/precommit_ablation/<model>/<leaf>/`` (records.jsonl / summary.json /
config.json), resume-safe by spec id.

Usage:
    conda run -n goals python scripts/evaluation/eval_precommit_ablation.py --limit 8   # smoke
    conda run -n goals python scripts/evaluation/eval_precommit_ablation.py
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

sys.path.append(".")

import torch  # noqa: E402

from scripts.evaluation.ablation_common import judge_implements  # noqa: E402
from scripts.evaluation.eval_goalscope import (  # noqa: E402
    ALL_DATATYPES, build_prompt, default_goal_band, frontier_impl_span, hard_eval, load_head_set,
    load_locked_datapoints,
)
from src.goals.goalscope import apply_goalscope  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)

PIVOTS = ["prev_end", "indent", "var", "eq"]          # the prompt's last four tokens, in order


def _layers(mt):
    return mt._model.model.layers


def pivot_positions(mt, prompt_ids, var):
    """{pivot: absolute position} = the last four prompt tokens, with sanity checks on their text."""
    P = len(prompt_ids)
    pos = dict(zip(PIVOTS, range(P - 4, P)))
    txt = {k: mt.tokenizer.decode([prompt_ids[p]]) for k, p in pos.items()}
    ok = ("\n" in txt["prev_end"] and txt["indent"].strip() == "" and txt["var"].strip() == var
          and txt["eq"].strip() == "=")
    return pos, txt, ok


class BareLatents:
    """Context-free latents of a single token: ``n_pad`` pad tokens + the token; the output of every
    decoder layer at the token's position. Cached per token id."""

    def __init__(self, mt, n_pad):
        self.mt, self.n_pad, self.cache = mt, n_pad, {}
        self.pad_id = mt.tokenizer.pad_token_id if mt.tokenizer.pad_token_id is not None else mt.tokenizer.eos_token_id

    @torch.no_grad()
    def __call__(self, tid):
        if tid not in self.cache:
            dev = next(self.mt._model.parameters()).device
            ids = torch.tensor([[self.pad_id] * self.n_pad + [int(tid)]], device=dev)
            self.cache[tid] = layer_outputs(self.mt, ids)[:, 0, -1].clone()     # (n_layer, d)
        return self.cache[tid]


@torch.no_grad()
def layer_outputs(mt, ids):
    """Output of every decoder layer (NOT HF's hidden_states, whose last entry is post final-norm):
    (n_layer, batch, seq, d)."""
    outs = []

    def hook(module, args, output):
        outs.append((output[0] if isinstance(output, tuple) else output).detach())

    hs = [layer.register_forward_hook(hook) for layer in _layers(mt)]
    try:
        mt._model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        for h in hs:
            h.remove()
    return torch.stack([o.to(outs[0].device) for o in outs])


@contextmanager
def replace_residual(mt, pos_to_latents):
    """For each absolute position p in ``pos_to_latents`` ({p: [latent per layer]}), overwrite the output
    of every decoder layer at p with the cached latent — only on forwards whose input covers p (the
    prefill; cached decode steps only see later positions)."""
    handles = []

    def make(L):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            S = hs.shape[1]
            if S == 1:                                                   # cached decode step
                return output
            for p, lat in pos_to_latents.items():
                if p < S:
                    hs[:, p] = lat[L].to(hs.device, hs.dtype)
            return output
        return hook

    try:
        for L, layer in enumerate(_layers(mt)):
            handles.append(layer.register_forward_hook(make(L)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def greedy_line(mt, prompt_ids, ctx_factory, max_new=64):
    """Greedy decode (KV cache) until a newline / EOS; ``ctx_factory()`` wraps every forward."""
    dev = next(mt._model.parameters()).device
    ids = torch.tensor([list(prompt_ids)], device=dev)
    am = torch.ones_like(ids)
    with ctx_factory():
        out = mt._model(input_ids=ids, attention_mask=am, use_cache=True)
    pkv, gen = out.past_key_values, []
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        if nxt == mt.tokenizer.eos_token_id:
            break
        gen.append(nxt)
        if "\n" in mt.tokenizer.decode([nxt]):
            break
        am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
        with ctx_factory():
            out = mt._model(input_ids=torch.tensor([[nxt]], device=dev), attention_mask=am,
                            past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
    return mt.tokenizer.decode(gen, skip_special_tokens=True)


def make_iface(ctx_factory):
    def iface(mt, inputs, patches=None, model_kwargs=None):
        with ctx_factory():
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))
    return iface


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--dataset", default="data/goalscope_eval.json")
    ap.add_argument("--ablate", nargs="+", default=["prev_end", "indent", "var"], choices=PIVOTS)
    ap.add_argument("--n_pad", type=int, default=16, help="pad tokens before the bare token")
    ap.add_argument("--heads", default="goal", help="read-out heads (goal = the paper's 122)")
    ap.add_argument("--strategy", default="additive_norm")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steer_layer_range", type=int, nargs=2, default=[35, 46])
    ap.add_argument("--max_new_tokens", type=int, default=64, help="read-out length")
    ap.add_argument("--gen_max_new_tokens", type=int, default=64, help="frontier-body greedy budget")
    ap.add_argument("--no_readout", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=None, help="SMOKE ONLY: first N specs (round-robin dtypes)")
    ap.add_argument("--overwrite", action="store_true")
    logging_utils.add_logging_args(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)
    torch.set_grad_enabled(False)
    results_dir = env_utils.DEFAULT_RESULTS_DIR

    datapoints, _ = load_locked_datapoints(args.dataset, ALL_DATATYPES)
    if args.limit:
        groups = defaultdict(list)
        for sid, inst in datapoints:
            groups[inst.meta["datatype"]].append((sid, inst))
        picked, i = [], 0
        while len(picked) < args.limit:
            for v in groups.values():
                if i < len(v) and len(picked) < args.limit:
                    picked.append(v[i])
            i += 1
        datapoints = picked

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    heads = load_head_set(args.heads, args.model, results_dir, mt.n_heads, goal_threshold=0.9,
                          goal_band=default_goal_band(args.model, n_layer=mt.n_layer), n_layer=mt.n_layer)
    bare = BareLatents(mt, args.n_pad)

    leaf = (f"ablate={'+'.join(args.ablate)}__npad={args.n_pad}__ro={args.heads}{len(heads)}"
            f"-{args.strategy}{args.alpha:g}-L{args.steer_layer_range[0]}-{args.steer_layer_range[1]}"
            f"__nspec={len(datapoints)}__seed={args.seed}")
    run_dir = Path(results_dir) / "evaluation" / "precommit_ablation" / args.model.split("/")[-1] / leaf
    run_dir.mkdir(parents=True, exist_ok=True)
    rec_path = run_dir / "records.jsonl"
    if args.overwrite:
        rec_path.unlink(missing_ok=True)
    done = {json.loads(l)["spec"] for l in rec_path.read_text().splitlines() if l.strip()} if rec_path.exists() else set()
    (run_dir / "config.json").write_text(json.dumps(dict(
        vars(args), n_specs=len(datapoints), n_readout_heads=len(heads),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    logger.info(f"run dir -> {run_dir} | {len(datapoints)} specs ({len(done)} done)")

    gs_kwargs = dict(selected_heads=heads, strategy=args.strategy, alpha=args.alpha,
                     steer_layer_range=tuple(args.steer_layer_range), n_steer_tokens="all",
                     max_new_tokens=args.max_new_tokens)
    with open(rec_path, "a") as fh:
        for sid, inst in datapoints:
            if sid in done:
                continue
            prompt, var = build_prompt(mt, inst, inst.src_lang)
            prompt_ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
            pos, txt, ok = pivot_positions(mt, prompt_ids, var)
            if not ok:
                logger.warning(f"[{sid}] unexpected pivot tokens {txt} — skipping")
                continue
            lat = {pos[k]: bare(prompt_ids[pos[k]]) for k in args.ablate}

            # sanity: how far the context-free latents are from the in-context ones (mid layer)
            dev = next(mt._model.parameters()).device
            hs = layer_outputs(mt, torch.tensor([prompt_ids], device=dev))[:, 0]   # (n_layer, P, d)
            mid = mt.n_layer // 2
            norm_ratio = {k: round(float(lat[pos[k]][mid].float().norm() / hs[mid, pos[k]].float().norm()), 3)
                          for k in args.ablate}
            cos = {k: round(float(torch.nn.functional.cosine_similarity(
                lat[pos[k]][mid].float(), hs[mid, pos[k]].float().to(lat[pos[k]].device), dim=0)), 3)
                for k in args.ablate}
            del hs

            conds = {"clean": lambda: replace_residual(mt, {}),
                     "ablated": lambda: replace_residual(mt, lat)}
            row = dict(spec=sid, datatype=inst.meta["datatype"], j=inst.meta["j"], T=inst.meta["T"],
                       frontier_id=inst.expected_next, condition=inst.expected_next_desc, var=var,
                       pivot_tokens=txt, bare_norm_ratio_mid=norm_ratio, bare_cos_mid=cos)
            clean_impl = None
            for cname, cf in conds.items():
                gen = greedy_line(mt, prompt_ids, cf, max_new=args.gen_max_new_tokens)
                code = gen.split("\n", 1)[0].strip()
                jv = judge_implements(code, inst.expected_next_desc)
                impl = frontier_impl_span(prompt, gen, var)
                if cname == "clean":
                    clean_impl = impl
                rec = dict(gen=code, impl_span=impl, verdict=jv["verdict"])
                if not args.no_readout:
                    res = apply_goalscope(prompt, mt=mt, show=False, include_noncode=False,
                                          token_positions=[pos["eq"]], patch_interface=make_iface(cf),
                                          **gs_kwargs)
                    ro = res[0]["patched_gen"] if res else ""
                    hv = hard_eval(ro, clean_impl, "faithfulness")
                    rec.update(readout_eq=ro, readout_verdict=hv["verdict"])
                row[cname] = rec
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.add(sid)
            logger.info(f"[{len(done)}/{len(datapoints)}] {sid} j={row['j']}: impl clean={row['clean']['verdict']} "
                        f"ablated={row['ablated']['verdict']}"
                        + ("" if args.no_readout else f" · readout@eq clean={row['clean']['readout_verdict']} "
                           f"ablated={row['ablated']['readout_verdict']}")
                        + f" · bare/ctx norm {norm_ratio} cos {cos}")

    summarize(run_dir)


def summarize(run_dir):
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines() if l.strip()]

    def rate(rs, c, key):
        rs = [r for r in rs if key in r[c]]
        return round(sum(r[c][key] == "YES" for r in rs) / len(rs), 4) if rs else None

    def block(rs):
        return dict(n=len(rs), **{f"{c}_{k}": rate(rs, c, key) for c in ("clean", "ablated")
                                  for k, key in (("impl", "verdict"), ("readout", "readout_verdict"))})

    summ = dict(overall=block(recs),
                by_datatype={dt: block([r for r in recs if r["datatype"] == dt]) for dt in sorted({r["datatype"] for r in recs})},
                by_frontier_j={j: block([r for r in recs if r["j"] == j]) for j in sorted({r["j"] for r in recs})})
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    logger.info(f"SUMMARY {json.dumps(summ['overall'])}")
    return summ


if __name__ == "__main__":
    main()
