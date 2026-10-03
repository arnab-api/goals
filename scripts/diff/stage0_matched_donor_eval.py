#!/usr/bin/env python
"""Stage 0 — score FROZEN query masks on matched source donors (no retraining, no pipeline changes).

Question: are the existing head / sub-head masks *hypothesis-specific*? Each primary contrast is paired with a
MATCHED donor built from the SAME role -> predicate assignment that changes only the variable of interest; the
destination prompt and its informative expression prefix stay fixed (agent_reports/silico/
compact_query_localization_design.md §3, as revised in review).

    design  config          primary donor        control donor(s)                       varied
    S1      order_diff_2    a' b c' d' -> b      a' b c' e' -> b  (+ legacy SRC')       completed identity
    C1      h1c_1           a' b c' -> b         a' b c -> b  (+ comment / assignment   completed count
                                                 padding to the primary's token length)
    P1      h2_1            a' b' c d' -> c      a' c b' d' -> c   (identical code)     frontier spec position
    I1      h3_1            a' b' c -> c         b' a' c -> c                           predecessor
    OD1     order_diff_1    (H1)                 legacy SRC'                            completed count
    OM1     order_matched_1 (H1)                 — (primary only)

Every matched donor's prediction under the design's hypothesis is the destination's NATIVE task (checked with
``nrgs_prefix.predict``). Instances are the VALIDATION split of the dataset each design was mined into
(results/nrgs_prefix/<M>/<tag>__python-python/ — the pre-fix datasets, untouched by the re-mining), so for the
mask trained on that dataset they are held out (pair-disjoint); ``heldout`` records it per (mask, instance).

Stub-letter vs count (C1, OD1). Prompts use the pipeline's sequential letter names, so the source's frontier stub
letter equals its written count (2 written -> ``c =``) and can coincide with DST*'s target stub. Two letter-swap
donors put the letter and the count in opposition, staying one token away from the training distribution:
``count_low_stub_high`` = the control's written lines with the PRIMARY's stub letter, and ``count_high_stub_low``
= the primary's written lines with the CONTROL's stub letter. A count-carrying mask flips on the latter only; a
stub-letter-copying mask on the former only. ``expect`` (count reading) and ``expect_letter`` (letter reading) are
stored per donor.

Per (mask variant, design, rendering, donor, instance): top-1 == target (flip), top-1 == native, and the
target-vs-native log-odds (+ its gain over clean DST). Rival check: for every other task present in DST that a
competing hypothesis predicts, the top-1 token of a reference DST context where that task is next is recorded;
an instance is ``distinguishing`` when no rival shares the target token.

Outputs (``results/stage0_matched/<M>/<job_tag>/``): rows.jsonl (per instance), prompts.jsonl (rendered donor /
DST prompts + token ids), summary.json (per mask variant x design x rendering: primary flip, control native acc,
joint paired correctness, specificity S = flip(primary) - flip(control), paired log-odds difference; all + the
``distinguishing`` and ``heldout`` subsets).

    conda run -n goals python scripts/diff/stage0_matched_donor_eval.py --masks heads --job-tag heads
    conda run -n goals python scripts/diff/stage0_matched_donor_eval.py --masks svd:h1c --job-tag svd_h1c
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.append(".")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

import src.diff.data as D  # noqa: E402
import src.diff.nrgs_mask as NM  # noqa: E402
import src.diff.nrgs_prefix as N  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

logger = logging.getLogger("stage0")
MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
N_get_prompt = None   # bound in main() (needs the model's tokenizer chat template)

# design -> (dataset tag, config, {donor_name: spec}); spec = (spec order, completed set) | "legacy" (the config's
# SRC') | ("pad", kind) (matched donor padded to the primary's token length) | ("swap", lines_from, stub_from)
# (written lines of one donor with the frontier stub letter of another; "primary" | "matched" | "legacy")
DESIGNS = {
    "S1": ("mixed", "order_diff_2", {"matched": ("abce", "ace"), "legacy_srcp": "legacy"}),
    "C1": ("h1c", "h1c_1", {"matched": ("abc", "a"), "pad_comment": ("pad", "comment"), "pad_assign": ("pad", "assign"),
                            "count_low_stub_high": ("swap", "matched", "primary"),
                            "count_high_stub_low": ("swap", "primary", "matched")}),
    "P1": ("h2", "h2_1", {"matched": ("acbd", "abd")}),
    "I1": ("h3", "h3_1", {"matched": ("bac", "ab")}),
    "OD1": ("mixed", "order_diff_1", {"legacy_srcp": "legacy",
                                      "count_low_stub_high": ("swap", "legacy", "primary"),
                                      "count_high_stub_low": ("swap", "primary", "legacy")}),
    "OM1": ("mixed", "order_matched_1", {}),
}
RENDERING = "historical"   # the pipeline's sequential letter names (what every mask was trained on)
HYP_OF = {"S1": "H1", "C1": "H1c", "P1": "H2", "I1": "H3", "OD1": "H1", "OM1": "H1"}

M_SHORT = MODEL.split("/")[-1]
NM_ROOT = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_mask" / M_SHORT
SVD_ROOT = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_svd_mask" / M_SHORT
HEAD_MASKS = [  # (label, run dir) — the §9a head masks
    ("head_mix_all", "mixed__python-python__L0-64__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_mix_L20-56", "mixed__python-python__L20-56__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_h1c_all", "h1c__python-python__L0-64__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_h1c_L20-56", "h1c__python-python__L20-56__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_h2_L20-56", "h2__python-python__L20-56__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_h3_all", "h3__python-python__L0-64__lam20__kl1_dir1_ctrl0__logistic__s0"),
    ("head_h3_L20-56", "h3__python-python__L20-56__lam20__kl1_dir1_ctrl0__logistic__s0"),
]
SVD_MASKS = {  # hypothesis -> [(label, run dir)] — the §9c sub-head masks, L20-56
    h: [(f"svd_{h}_all", f"{tag}__python-python__L20-56__lam3__kl1_dir1_ctrl0__logistic__s0"),
        (f"svd_{h}_goal", f"{tag}__python-python__L20-56__only-goal__lam3__kl1_dir1_ctrl0__logistic__s0")]
    for h, tag in (("mix", "mixed"), ("h1c", "h1c"), ("h2", "h2"), ("h3", "h3"))
}
FIXED_HEADSETS = [("goal250", "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"),
                  ("ctrl250", "head_sets/Qwen2.5-Coder-32B-Instruct/control/control250__laterhalf.json")]


# ══════════════════════════════════════════════════════════════════════════════
# rendering (same renderer as the pipeline: DiffTriggerInstance._parts over the instance's own roles)
# ══════════════════════════════════════════════════════════════════════════════
def _code_for(inst, order_letters, done_letters):
    """(question, code) for a spec order / completed set given as letters, using the instance's role map."""
    bank, r, base = N.BANK[inst.datatype], inst.roles, inst.base
    order = tuple(bank[r[x]] for x in order_letters)
    return base._parts(base.src_lang, order, [r[x] for x in order_letters if x in set(done_letters)])


def _prompt(q, code):
    return N_get_prompt(q, D._finalize_prefill(code))


def _with_stub(code, letter):
    """Replace the frontier stub's variable (the last line ``    x =``) with ``letter``. If a written line already
    binds ``letter``, that line takes the old stub letter instead (names are swapped, never reassigned), e.g.
    ``a = ..; b = ..; c =`` with letter b -> ``a = ..; c = ..; b =``."""
    head, last = code.rsplit("\n", 1)
    assert re.fullmatch(r"    [a-z] =", last), last
    old = last.strip()[0]
    head, n = re.subn(rf"(?m)^    {letter} = ", f"    {old} = ", head)
    assert n <= 1, code
    return head + f"\n    {letter} ="


def _stub_letter(code):
    return code.rsplit("\n", 1)[1].strip()[0]


def dst_prompt(inst, prefilled_letters=None):
    """DST (prefilled = the config's DST prefill) or a rival reference context (prefilled = the DST tasks before
    the rival), with the instance's informative expression prefix appended after the stub."""
    cfg = inst.cfg
    pre = cfg.dst_prefill if prefilled_letters is None else set(prefilled_letters)
    q, code = _code_for(inst, cfg.dst, pre)
    return _prompt(q, code + " " + inst.pivot.prefix)


def pad_to_length(tok, q, code, target_len, kind):
    """Insert one padding line before the frontier stub and grow it 1 token at a time until the full prompt is
    ``target_len`` tokens (the primary donor's length = same capture position)."""
    head, last = code.rsplit("\n", 1)
    # minimal starting lines so even a short primary line can be matched; every grown form stays valid Python
    # (`_ = 1`, `_ = 10`, `_ = 100`, ... — Qwen splits digits, so one token per appended '0')
    unit = " x" if kind == "comment" else "0"
    body = "#" if kind == "comment" else "_ = 1"
    for _ in range(400):
        p = _prompt(q, head + "\n    " + body + "\n" + last)
        n = len(tok(p)["input_ids"])
        if n == target_len:
            return p
        if n > target_len:
            return None                                  # overshoot (rare): drop this donor for the instance
        body += unit
    return None


# ══════════════════════════════════════════════════════════════════════════════
# instances + donors
# ══════════════════════════════════════════════════════════════════════════════
def load_design(design, limit=None):
    tag, cfg_name, _ = DESIGNS[design]
    fp = Path(env_utils.DEFAULT_RESULTS_DIR) / "nrgs_prefix" / M_SHORT / f"{tag}__python-python" / "dataset.json"
    recs = [r for r in json.loads(fp.read_text())["instances"] if r["config"] == cfg_name and r["split"] == "val"]
    return recs[:limit] if limit else recs


def build_views(tok, design, rec):
    """{RENDERING: {"dst", "donors": {name: prompt}, "expect": {name: letter}, "expect_letter": {name: letter},
    "rivals": {letter: prompt}}}. ``expect`` = the design hypothesis' prediction from the donor's written
    (completed) tasks; ``expect_letter`` = the prediction if only the stub letter were read (swap donors)."""
    _, cfg_name, donors = DESIGNS[design]
    inst = N.from_record(rec)
    cfg, hyp = inst.cfg, HYP_OF[design]
    pred = lambda order, done: N.predict(tuple(order), frozenset(done), cfg.dst, cfg.dst_prefill)[hyp]
    base_specs = {"primary": (cfg.src, cfg.src_prefill)}
    if "matched" in donors:
        base_specs["matched"] = donors["matched"]
    if cfg.src_ctrl_prefill is not None:
        base_specs["legacy"] = (cfg.src, cfg.src_ctrl_prefill)
    codes = {k: _code_for(inst, *sp) for k, sp in base_specs.items()}

    v = {"dst": dst_prompt(inst), "donors": {}, "expect": {}, "expect_letter": {}, "rivals": {}}
    v["donors"]["primary"] = _prompt(*codes["primary"])
    v["expect"]["primary"] = v["expect_letter"]["primary"] = cfg.predictions.get(hyp)
    for name, spec in donors.items():
        if spec == "legacy":
            if "legacy" not in codes:
                continue
            v["donors"][name] = _prompt(*codes["legacy"])
            v["expect"][name] = v["expect_letter"][name] = pred(*base_specs["legacy"])
        elif spec[0] == "pad":
            q, code = codes["matched"]
            p = pad_to_length(tok, q, code, len(tok(v["donors"]["primary"])["input_ids"]), spec[1])
            if p is None:
                continue
            v["donors"][name] = p
            v["expect"][name] = v["expect_letter"][name] = pred(*base_specs["matched"])
        elif spec[0] == "swap":
            lines_from, stub_from = spec[1], spec[2]
            if lines_from not in codes or stub_from not in codes:
                continue
            q, code = codes[lines_from]
            letter = _stub_letter(codes[stub_from][1])
            if letter == _stub_letter(code):
                continue                                  # no letter/count conflict to create
            v["donors"][name] = _prompt(q, _with_stub(code, letter))
            v["expect"][name] = pred(*base_specs[lines_from])           # count (written lines) reading
            v["expect_letter"][name] = pred(*base_specs[stub_from])     # stub-letter reading
        else:
            order, done = spec
            v["donors"][name] = _prompt(*_code_for(inst, order, done))
            v["expect"][name] = v["expect_letter"][name] = pred(order, done)
    # rivals: other hypotheses' predicted tasks that exist in DST and are not the native / target
    rivals = {x for h, x in cfg.predictions.items()
              if x and h != hyp and x not in (cfg.native, cfg.target) and x in cfg.dst}
    for x in sorted(rivals):
        v["rivals"][x] = dst_prompt(inst, prefilled_letters=cfg.dst[: cfg.dst.index(x)])
    return inst, {RENDERING: v}


# ══════════════════════════════════════════════════════════════════════════════
# masks
# ══════════════════════════════════════════════════════════════════════════════
def mask_variants(mt, spec):
    """[(label, kind, layers, tensor, train_pairs)] — kind 'head' (len(layers), H) or 'svd' (len(layers), H, d)."""
    out = []

    def train_pairs(config):
        ds = Path(config.get("dataset") or "")
        if not ds.exists():
            return None
        return {(r["datatype"], frozenset((r["b"], r["d"]))) for r in json.loads(ds.read_text())["instances"]
                if r["split"] == "train"}

    if spec == "heads":
        for label, run in HEAD_MASKS:
            ck = torch.load(NM_ROOT / run / "mask.pt", map_location="cpu", weights_only=False)
            layers, tp = list(ck["layers"]), train_pairs(ck["config"])
            out.append((f"{label}|soft", "head", layers, ck["mask_values"][layers].float(), tp))
            out.append((f"{label}|bin", "head", layers, NM.mask_from_heads(ck["selected_heads"], layers, mt.n_heads), tp))
        allL = list(range(mt.n_layer))
        for label, fp in FIXED_HEADSETS:
            hs = [tuple(h) for h in json.loads(Path(fp).read_text())]
            out.append((label, "head", allL, NM.mask_from_heads(hs, allL, mt.n_heads), None))
    elif spec.startswith("svd:"):
        for label, run in SVD_MASKS[spec.split(":", 1)[1]]:
            ck = torch.load(SVD_ROOT / run / "mask.pt", map_location="cpu", weights_only=False)
            layers, tp = list(ck["layers"]), train_pairs(ck["config"])
            vals = ck["mask_values"][layers].float()
            out.append((f"{label}|soft", "svd", layers, vals, tp))
            out.append((f"{label}|comp", "svd", layers, (vals > ck.get("threshold", 0.9)).float(), tp))
    else:
        raise ValueError(spec)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def summarize(rows, prompts=None):
    """Per (mask, design, control donor): paired metrics over instances whose clean DST predicts the native token.

    Each control donor is scored against ITS OWN expected outcome under the design's hypothesis (``expect``:
    target or native) — e.g. ``count_high_stub_low`` should flip under the count reading. ``specificity`` =
    flip(primary) - flip(control) is reported only for native-expected controls; ``joint`` = primary flips AND
    the control does what it should. Subsets: ``distinguishing`` (no rival task shares the target token) and
    ``heldout`` (the instance's predicate pair is NOT in the mask's training split — ``heldout is True`` only;
    ``heldout_unknown`` counts fixed head sets with no training provenance). Also returns a primary-only table
    (covers OM1, which has no control donor) and per-design donor coverage (accepted / instances)."""
    import numpy as np
    from collections import defaultdict
    by = defaultdict(dict)
    for r in rows:
        by[(r["mask"], r["design"], r["key"])][r["donor"]] = r
    paired, prim = defaultdict(list), defaultdict(list)
    for (mask, design, key), dn in by.items():
        p = dn.get("primary")
        if p is None or not p["clean_native"]:
            continue
        prim[(mask, design)].append(p)
        for cname, c in dn.items():
            if cname != "primary":
                paired[(mask, design, cname)].append((p, c))
    mean = lambda xs: float(np.mean(xs)) if xs else None
    controls = []
    for (mask, design, cname), pairs in sorted(paired.items()):
        p0, c0 = pairs[0]
        reading = lambda letter: "target" if letter == p0["expect"] else "native"
        exp_count, exp_letter = reading(c0["expect"]), reading(c0.get("expect_letter", c0["expect"]))

        def stats(ps):
            if not ps:
                return None
            ok = (lambda c: c["flip"]) if exp_count == "target" else (lambda c: c["native"])
            pf, cf = mean([x[0]["flip"] for x in ps]), mean([x[1]["flip"] for x in ps])
            return dict(n=len(ps), primary_flip=pf, primary_native=mean([x[0]["native"] for x in ps]),
                        control_flip=cf, control_native=mean([x[1]["native"] for x in ps]),
                        control_correct=mean([ok(x[1]) for x in ps]),
                        joint=mean([x[0]["flip"] and ok(x[1]) for x in ps]),
                        specificity=(pf - cf) if exp_count == "native" else None,
                        logodds_diff=mean([x[0]["logodds"] - x[1]["logodds"] for x in ps]),
                        primary_gain=mean([x[0]["gain"] for x in ps]), control_gain=mean([x[1]["gain"] for x in ps]))
        controls.append(dict(mask=mask, design=design, control=cname, control_expect_count=exp_count,
                             control_expect_letter=exp_letter, all=stats(pairs),
                             distinguishing=stats([x for x in pairs if x[0]["distinguishing"]]),
                             heldout=stats([x for x in pairs if x[0]["heldout"] is True]),
                             heldout_unknown=sum(x[0]["heldout"] is None for x in pairs)))
    primary = [dict(mask=mask, design=design, n=len(ps), flip=mean([r["flip"] for r in ps]),
                    native=mean([r["native"] for r in ps]), gain=mean([r["gain"] for r in ps]),
                    heldout_n=sum(r["heldout"] is True for r in ps))
               for (mask, design), ps in sorted(prim.items())]
    coverage = {}
    if prompts:
        for pr in prompts:
            d = coverage.setdefault(pr["design"], defaultdict(int))
            d["instances"] += 1
            for k in pr["donors"]:
                d[k] += 1
        coverage = {k: dict(v) for k, v in coverage.items()}
    return dict(controls=controls, primary=primary, donor_coverage=coverage)


def _log_summary(summ):
    for s in summ["controls"]:
        a = s["all"]
        sp = f"S={a['specificity']:+.2f}" if a["specificity"] is not None else "S=n/a"
        logger.info(f"{s['mask']:<26} {s['design']:<3} vs {s['control']:<20} (count->{s['control_expect_count']:<6} "
                    f"letter->{s['control_expect_letter']:<6}) n={a['n']:>3} P.flip={a['primary_flip']:.2f} "
                    f"C.flip={a['control_flip']:.2f} C.correct={a['control_correct']:.2f} joint={a['joint']:.2f} {sp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True, help="'heads' (head masks + goal250/ctrl250) | 'svd:<mix|h1c|h2|h3>'")
    ap.add_argument("--designs", nargs="+", default=list(DESIGNS), choices=list(DESIGNS))
    ap.add_argument("--only", default=None, help="regex on mask-variant labels (split one mask family across jobs)")
    ap.add_argument("--job-tag", required=True)
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: cap instances per design")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--resummarize", action="store_true",
                    help="rebuild summary.json from an existing job's rows.jsonl / prompts.jsonl (no model)")
    logging_utils.add_logging_args(ap)
    args = ap.parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    logging_utils.configure(level=logging.INFO, force=True)
    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "stage0_matched" / M_SHORT / args.job_tag
    if args.resummarize:
        rows = [json.loads(l) for l in open(out_dir / "rows.jsonl")]
        prompts = [json.loads(l) for l in open(out_dir / "prompts.jsonl")]
        summ = summarize(rows, prompts)
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
        logger.info(f"re-summarized {len(rows)} rows -> {out_dir / 'summary.json'}")
        return

    ctx = N.load_ctx(MODEL, device_map={"": 0} if torch.cuda.device_count() == 1 else "auto")
    mt, tok = ctx.mt, ctx.mt.tokenizer
    global N_get_prompt
    from src.diff.functional import get_prompt
    N_get_prompt = lambda q, code: get_prompt(q, code, ctx=ctx)

    out_dir = Path(env_utils.DEFAULT_RESULTS_DIR) / "stage0_matched" / M_SHORT / args.job_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── render every view; compute donor queries, clean DST and rival top-1 once ──
    items = []   # (design, rec, inst, rend, view)
    for design in args.designs:
        recs = load_design(design, args.limit)
        for rec in recs:
            inst, views = build_views(tok, design, rec)
            for rend, v in views.items():
                items.append((design, rec, inst, rend, v))
        logger.info(f"{design}: {len(recs)} val instances")
    donor_prompts = sorted({p for *_, v in items for p in v["donors"].values()})
    dst_prompts = sorted({v["dst"] for *_, v in items} | {p for *_, v in items for p in v["rivals"].values()})
    logger.info(f"{len(items)} instance-views · {len(donor_prompts)} donor prompts · {len(dst_prompts)} DST/rival prompts")
    q = {}
    for s in range(0, len(donor_prompts), args.batch_size):
        ch = donor_prompts[s: s + args.batch_size]
        for p, t in zip(ch, NM.q_last_batch(mt, ch)):
            q[p] = t
    lp_clean = {}
    for s in range(0, len(dst_prompts), args.batch_size):
        ch = dst_prompts[s: s + args.batch_size]
        for p, t in zip(ch, N.last_token_logprobs(mt, ch)):
            lp_clean[p] = t
    with open(out_dir / "prompts.jsonl", "w") as fh:
        for design, rec, inst, rend, v in items:
            fh.write(json.dumps(dict(design=design, key=rec["key"], rendering=rend, dst=v["dst"],
                                     donors=v["donors"], expect=v["expect"], expect_letter=v["expect_letter"],
                                     rivals=v["rivals"],
                                     b_id=rec["b_id"], d_id=rec["d_id"],
                                     donor_len={k: len(tok(p)["input_ids"]) for k, p in v["donors"].items()})) + "\n")

    # ── score every mask variant ──
    rows = []
    variants = mask_variants(mt, args.masks)
    if args.only:
        variants = [v for v in variants if re.search(args.only, v[0])]
    logger.info(f"{len(variants)} mask variants: {[v[0] for v in variants]}")
    svd_bases = {}
    for label, kind, layers, mv, train_pairs in variants:
        if kind == "svd":
            import src.diff.nrgs_svd_mask as SM
            key = tuple(layers)
            if key not in svd_bases:
                svd_bases[key] = SM.svd_bases(mt, layers)
            bases = svd_bases[key]
        jobs = []   # NRGSCache per (item, donor)
        meta = []
        for design, rec, inst, rend, v in items:
            b, d = rec["b_id"], rec["d_id"]
            lc = lp_clean[v["dst"]]
            clean_native = int(lc.argmax()) == b
            rival_tops = {x: int(lp_clean[p].argmax()) for x, p in v["rivals"].items()}
            distinguishing = all(t != d for t in rival_tops.values())
            pair = (rec["datatype"], frozenset((rec["b"], rec["d"])))
            heldout = None if train_pairs is None else pair not in train_pairs
            for name, sp in v["donors"].items():
                jobs.append(NM.NRGSCache(key=rec["key"], datatype=rec["datatype"], pair=(rec["datatype"], rec["b"], rec["d"]),
                                         split="val", dst_prompt=v["dst"], b_id=b, d_id=d, q_src=q[sp], q_ctrl=None,
                                         log_p_target=torch.zeros(1), clean_margin=float(lc[d] - lc[b]),
                                         config=rec["config"]))
                meta.append(dict(mask=label, design=design, rendering=rend, donor=name, key=rec["key"],
                                 datatype=rec["datatype"], expect=v["expect"].get(name),
                                 expect_letter=v["expect_letter"].get(name), clean_native=clean_native,
                                 distinguishing=distinguishing, rival_tops=rival_tops, heldout=heldout))
        mvd = mv.to(mt._model.device)
        for s in range(0, len(jobs), args.batch_size):
            cs = jobs[s: s + args.batch_size]
            with torch.no_grad():
                lp = (NM.patched_logprobs(mt, mvd, layers, cs) if kind == "head"
                      else SM.patched_logprobs(mt, mvd, layers, bases, cs))
            for i, c in enumerate(cs):
                top = int(lp[i].argmax())
                lo = float(lp[i][c.d_id] - lp[i][c.b_id])
                rows.append(dict(meta[s + i], flip=top == c.d_id, native=top == c.b_id, top1=top,
                                 logodds=lo, gain=lo - c.clean_margin))
        logger.info(f"[{label}] scored {len(jobs)} (instance, donor) views")
        with open(out_dir / "rows.jsonl", "w") as fh:          # rewrite after every mask (resumable inspection)
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    summ = summarize(rows, [json.loads(l) for l in open(out_dir / "prompts.jsonl")])
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    _log_summary(summ)
    logger.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
