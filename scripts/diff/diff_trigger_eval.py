#!/usr/bin/env python
"""Diff-trigger q-transfer eval — does the goal-head query carry an ABSTRACT progress-relative DIFF?

SRC and DST share two sub-tasks {A, B}; SRC has a unique third C, DST a unique third D:

    order-matched:  SRC=[A,B,C] prefill{A,B} (frontier C)  |  DST=[A,B,D] prefill{A} (native B)
    order-diff:     SRC=[A,C,B] prefill{A,B} (frontier C)  |  DST=[A,B,D] prefill{A} (native B)

We transplant SRC's pivot goal-head *query* into DST and read what DST writes:
    D -> the query carried "{A,B} done, do the remaining", resolved against DST's list (DIFF hypothesis)
    B -> no transfer (native)          C -> literal content copy (C is not even in DST's spec)
order-matched confounds D with a positional "do the 3rd-listed" copy (C and D are both 3rd); order-diff
breaks it (C is 2nd in SRC, D is 3rd in DST, so a positional copy predicts B).

Reuses the q-transfer machinery: ``src.diff.functional.run_q_transfer_intervention`` (single-pivot query
patch) + ``src.diff.eval`` (``judge_causal`` = wrote D; ``eval_q_transfer`` gaze = pivot gazes at D's span
> B's span). Adds a literal-C check (``classify_concept`` against C). Results:
    results/diff/<model>/diff_trigger/<variant>/<datatype>__<sl>-<dl>/results.json  (+ summary.json)

Usage:
    conda run -n goals python scripts/diff/diff_trigger_eval.py --cuda 0,1 \\
        --variants order_matched order_diff --datatypes int str --n-samples 12
    # cheap dry pass (rule-based gaze only):
    conda run -n goals python scripts/diff/diff_trigger_eval.py --no-oracle --max-instances 2 --cuda 0
    # closed-loop intervention (notebook 04.4) instead of re-clamping the same SRC query at every decode step:
    conda run -n goals python scripts/diff/diff_trigger_eval.py --closed-loop --variants order_diff --cuda 0

``--closed-loop`` keeps the prefill patch (SRC pivot query -> DST pivot, -> t1) and then, at every step, replaces
the goal heads' query at t_k with the query of an information-less probe (DST's signature + the bare indent, or
``return`` with ``--pivot eq``) that is steered with the patched pivot's goal encoding and fed the same tokens
(``src.diff.functional.closed_loop_generate``). ``--n-steer`` is unused there. Output dirs get ``__cl``
(+ ``-pspivot`` / ``-m<N>`` when those differ from the defaults), so they never mix with re-clamp runs.

``--kv-transfer`` (notebooks 04.3 / 04.4) moves the "which task is pending" computation into the goal heads. Per
instance, the current DST ``A'BD`` becomes p_kv, and a new DST ``A'XY`` (X, Y sampled per instance: X from a family
distinct from B / C / D, all pairwise compatible) gets p_kv's task-list key+value on the patch heads:
    p_src A'CB' -> C    p_dst A'XY -> X    p_kv A'BD -> B
    kv-only -> B  ·  kv+q: set progress -> D, count -> Y   (C = SRC's own target, in neither prompt)
Judged with one exclusive verdict over {X, Y, B, C, D} (clean DST, kv-only, kv+q). Gaze / the D-vs-B judges are
skipped. Works with ``--closed-loop``. Output dirs get ``__kv``.
"""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _preparse_cuda(argv):
    for i, a in enumerate(argv):
        if a == "--cuda" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--cuda="):
            return a.split("=", 1)[1]
    return None


_cuda = _preparse_cuda(sys.argv[1:])
if _cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import src.diff.data as DATA  # noqa: E402
import src.diff.eval as E  # noqa: E402
import src.diff.functional as FUNC  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger("diff_trigger_eval")

LANG_TRANSFERS = {"py-py": ("python", "python"), "py-cpp": ("python", "cpp"),
                  "cpp-py": ("cpp", "python"), "cpp-cpp": ("cpp", "cpp")}


def _resolve_heads(ctx, spec):
    """'goal'|'union' built-ins, or a path to a JSON [[layer, head], ...] head set."""
    if spec == "goal":
        return ctx.goal_heads
    if spec == "union":
        return ctx.union_heads
    arr = json.loads(Path(spec).read_text())
    return [(int(l), int(h)) for l, h in arr]


def _heads_tag(spec, n):
    if spec in ("goal", "union"):
        return spec
    return f"{Path(spec).stem.split('__')[0]}_{n}"


def _inst_key(inst):
    ids = lambda seq: "-".join(p.id for p in seq)
    return (f"{inst.meta['variant']}|src={ids(inst.src_order)}|dst={ids(inst.dst_order)}"
            f"|prog_src={'-'.join(inst.src_progress)}|prog_dst={'-'.join(inst.dst_progress)}")


KV_LETTERS = ("X", "Y", "B", "C", "D")


def _kv_roles(dt, inst, seed, distinct):
    """(X, Y) predicates for the kv design, or None: not in {A,B,C,D}, all pairwise compatible, non-trivial X/Y,
    X's family distinct from B / C / D (unless ``distinct`` is off). Deterministic per instance."""
    import random
    bank = DATA.QTASKS[dt].predicates
    A, B = inst.dst_order[0], next(p for p in inst.dst_order if p.id == inst.dst_native)
    C = next(p for p in inst.src_order if p.id == inst.src_target)
    D = next(p for p in inst.dst_order if p.id == inst.diff_target)
    used = {A.id, B.id, C.id, D.id}
    trivial = DATA.TRIVIAL_FRONTIERS.get(dt, frozenset())
    fams = {DATA._diff_family(dt, q.id) for q in (B, C, D)}
    cands = [q for q in bank if q.id not in used and q.id not in trivial]
    rng = random.Random(f"{seed}|{_inst_key(inst)}")
    pairs = [(x, y) for x in cands for y in cands if x.id != y.id]
    rng.shuffle(pairs)
    for x, y in pairs:
        if distinct and DATA._diff_family(dt, x.id) in fams:
            continue
        ids = [q.id for q in (A, B, C, D, x, y)]
        if all(DATA.qpred_compatible(dt, a, b) for k, a in enumerate(ids) for b in ids[k + 1:]):
            return dict(A=A, B=B, C=C, D=D, X=x, Y=y)
    return None


def _judge_kv(roles, text, var, seed_key, model_name):
    """Exclusive verdict over {X, Y, B, C, D} on the FIRST computation (options shuffled per instance)."""
    import random
    import re
    from src.diff.eval import CONCEPT_SYSTEM
    from src.diff.hyp_eval import _letters_prompt
    from src.utils.oracle_llms import ask_claude
    letters = list(KV_LETTERS)
    random.Random(seed_key).shuffle(letters)
    if not text or not text.strip():
        return {"label": "neither", "why": "empty", "order": letters}
    raw = ask_claude(_letters_prompt(text, [(k, roles[k].description) for k in letters],
                                     f"computing several properties of `{var}`", roles["A"].description,
                                     "the code the model generated to continue the function after the pivot. "
                                     "Judge ONLY the FIRST computation/statement it performs (ignore anything after)"),
                     system_prompt=CONCEPT_SYSTEM, max_tokens=768, model_name=model_name)
    v = re.search(r"<verdict>\s*(\d+|NEITHER)\s*</verdict>", raw, re.I)
    w = re.search(r"<why>\s*(.*?)\s*</why>", raw, re.I | re.S)
    if not v:
        label = "parse_error"
    elif v.group(1).upper() == "NEITHER":
        label = "neither"
    else:
        k = int(v.group(1)) - 1
        label = letters[k] if 0 <= k < len(letters) else "parse_error"
    return {"label": label, "why": w.group(1).strip() if w else "", "order": letters}


def _summarize_kv(records):
    out = dict(n=len(records))
    for run in ("clean", "kv", "patched"):
        rs = [r["kv_judge"][run]["label"] for r in records if r.get("kv_judge")]
        for lab in (*KV_LETTERS, "neither"):
            out[f"{run}_{lab}_rate"] = (sum(x == lab for x in rs) / len(rs)) if rs else None
    out.update(clean_native_X=out["clean_X_rate"], kv_only_B=out["kv_B_rate"],
               set_progress_D=out["patched_D_rate"], count_Y=out["patched_Y_rate"])
    return out


def _summarize(records):
    def rate(pred):
        vals = [pred(r) for r in records]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None
    return dict(
        n=len(records),
        causal_trigger_rate=rate(lambda r: r["causal"]["triggered"] if r.get("causal") else None),
        readout_shift_rate=rate(lambda r: r["readout"]["shifted"] if r.get("readout") else None),
        literal_c_rate=rate(lambda r: r["literal_c"]["wrote_c"] if r.get("literal_c") else None),
        native_b_rate=rate(lambda r: (r["causal"]["label"] == "dst") if r.get("causal") else None),
        gaze_shift_rate=rate(lambda r: r["gaze"]["shifted"] if r.get("gaze") else None),
        gaze_clean_baseline_rate=rate(lambda r: r["gaze"].get("clean_prefers_src") if r.get("gaze") else None),
        # joint B/C/D verdict — MUTUALLY EXCLUSIVE shares (sum to 1 minus parse errors), unlike the
        # binary causal(D)/literalC(C) marginals above which can double-count when C and D compete.
        joint_native_rate=rate(lambda r: (r["joint"]["label"] == "native") if r.get("joint") else None),
        joint_c_rate=rate(lambda r: (r["joint"]["label"] == "src_target") if r.get("joint") else None),
        joint_d_rate=rate(lambda r: (r["joint"]["label"] == "diff_target") if r.get("joint") else None),
        joint_neither_rate=rate(lambda r: (r["joint"]["label"] == "neither") if r.get("joint") else None),
    )


def run_config(mt, args, variant, dt, sl, dl, patch_heads, readout_heads, out_root, alpha):
    tag = f"{DATA._safe(dt) if hasattr(DATA, '_safe') else dt.replace('[', '(').replace(']', ')')}__{sl}-{dl}"
    outdir = out_root / variant / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    done = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{variant}/{tag}] resume: {len(done)} done")

    cfg_seed = (args.seed + int(hashlib.md5(f"{variant}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)
    set_seed(cfg_seed)
    setups = DATA.diff_trigger_setups(
        dt, variant=variant, n_samples=args.n_samples, seed=cfg_seed,
        naming=args.naming, letter_stub=("var_eq" if args.pivot == "eq" else "indent"),
        src_lang=sl, dst_lang=dl,
        distinct_families=not args.allow_family_overlap,
        nontrivial_frontier=not args.allow_family_overlap,
    )
    if args.max_instances:
        setups = setups[: args.max_instances]

    records = dict(done)
    _done_plain = {k.split("|kv:")[0] for k in records}                       # kv keys extend the plain key
    todo = [inst for inst in setups if _inst_key(inst) not in _done_plain or args.force]
    logger.info(f"[{variant}/{tag}] {len(setups)} instances (seed={cfg_seed}, alpha={alpha}) — "
                f"{len(setups) - len(todo)} resumed, {len(todo)} to run")

    for j, inst in enumerate(todo, 1):
        key = _inst_key(inst)
        if args.kv_transfer:
            roles = _kv_roles(dt, inst, cfg_seed, distinct=not args.allow_family_overlap)
            if roles is None:
                logger.warning(f"[{variant}/{tag}] no valid (X, Y) for {key} — skipped")
                continue
            kv_prompt = DATA.materialize(inst)["dst_prompt"]                     # p_kv = the current DST A'BD
            run_inst = dataclasses.replace(inst, dst_order=(roles["A"], roles["X"], roles["Y"]),
                                           dst_native=roles["X"].id)            # p_dst = A'XY
            key = f"{key}|kv:X={roles['X'].id}|Y={roles['Y'].id}"
            kv_arg = dict(prompt=kv_prompt, predicates=inst.dst_order, dst_predicates=run_inst.dst_order)
            logger.info(f"[{variant}/{tag}] ({j}/{len(todo)}) kv: p_dst A'XY X={roles['X'].id} Y={roles['Y'].id} · "
                        f"p_kv A'BD B={roles['B'].id} D={roles['D'].id} · SRC C={roles['C'].id}")
            raw = FUNC.run_q_transfer_intervention(
                mt, run_inst, patch_heads, readout_heads=readout_heads,
                n_steer=args.n_steer, max_new=args.max_new, alpha=alpha, n_pivot=1,
                compute_readout=not args.no_readout,
                component_proj=getattr(args, "_component_proj", None),
                closed_loop=(dict(probe_steer=args.cl_probe_steer, max_new=args.cl_max_new, alpha=args.cl_alpha,
                                  steer_layers=tuple(args.cl_steer_layers)) if args.closed_loop else None),
                kv=kv_arg,
            )
            judged = None
            if not args.no_oracle:
                judged = {run: _judge_kv(roles, raw[g], inst.var_name, f"{key}|{run}", args.oracle_model)
                          for run, g in (("clean", "gen_dst"), ("kv", "gen_kv"), ("patched", "gen_patched"))}
                logger.info(f"[{variant}/{tag}]   => clean={judged['clean']['label']} (X) · kv={judged['kv']['label']} "
                            f"(B) · kv+q={judged['patched']['label']} (set D / count Y)")
            records[key] = dict(
                key=key, variant=variant, datatype=dt, src_lang=sl, dst_lang=dl, design="kv",
                roles={k: v.id for k, v in roles.items()}, role_desc={k: v.description for k, v in roles.items()},
                intervention="closed_loop" if args.closed_loop else "reclamp",
                gen_src=raw["gen_src"], gen_dst=raw["gen_dst"], gen_kv=raw["gen_kv"], gen_patched=raw["gen_patched"],
                ro_src=raw["ro_src"], ro_dst=raw["ro_dst"], ro_kv=raw.get("ro_kv"), ro_patched=raw["ro_patched"],
                cl_steps=raw.get("cl_steps"), cl_probe_prefill=raw.get("cl_probe_prefill"),
                kv_n_tokens=raw["kv_n_tokens"], kv_judge=judged,
            )
            results_fp.write_text(json.dumps(list(records.values()), indent=2))
            continue
        A, B = inst.src_progress
        C, D = inst.src_target, inst.diff_target
        logger.info(f"[{variant}/{tag}] ({j}/{len(todo)}) shared[{A},{B}] SRC+{C} DST+{D}  "
                    f"| SRC→C={inst.src_target_desc!r} · DIFF→D={inst.expected_next_desc!r} "
                    f"· nativeB={inst.dst_next_desc!r}")
        raw = FUNC.run_q_transfer_intervention(
            mt, inst, patch_heads, readout_heads=readout_heads,
            n_steer=args.n_steer, max_new=args.max_new, alpha=alpha, n_pivot=1,
            restrict_to_task=args.restrict_goal_heads_to_task,
            compute_readout=not args.no_readout,
            component_proj=getattr(args, "_component_proj", None),
            closed_loop=(dict(probe_steer=args.cl_probe_steer, max_new=args.cl_max_new, alpha=args.cl_alpha,
                              steer_layers=tuple(args.cl_steer_layers)) if args.closed_loop else None),
        )
        causal = literal_c = readout = joint = None
        if not args.no_oracle:
            causal = E.judge_causal(inst, raw["gen_patched"], model_name=args.oracle_model)
            # joint B/C/D — single mutually-exclusive verdict (B=no-op, C=src-target, D=progress)
            joint = E.judge_joint(inst, raw["gen_patched"], model_name=args.oracle_model)
            # readout-shift: does the goal-head READOUT of the patched state express D (vs native B)?
            if not args.no_readout:
                readout = E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"],
                                          model_name=args.oracle_model)
            # literal-C check: does the patched continuation express SRC's literal target C (vs native B)?
            lit = E.classify_concept(raw["gen_patched"], inst.src_target_desc, inst.dst_next_desc,
                                     inst.task_desc, model_name=args.oracle_model,
                                     kind="the code the model generated after the pivot; judge ONLY the "
                                          "first computation")
            literal_c = {"label": lit["label"], "wrote_c": lit["label"] in ("src", "both"), "why": lit["why"]}
        gaze = E.eval_q_transfer(inst, gaze=dict(
            prompt=raw["dst_prompt"], task_positions=raw["task_positions"],
            attn_patched=raw["attn_patched"], attn_clean=raw["attn_clean"],
        )).gaze

        _c = f"{'HIT-D' if causal['triggered'] else 'miss'}({causal['label']})" if causal else "off"
        _ro = (f"readout={readout['source']}>{readout['dest']}>{readout['patched']}"
               f"[{'shift→D' if readout['shifted'] else 'no'}]") if readout else ""
        _l = f"literalC={'YES' if literal_c['wrote_c'] else 'no'}" if literal_c else ""
        _j = f"joint={ {'native':'B','src_target':'C','diff_target':'D','neither':'-'}.get(joint['label'],'?') }" if joint else ""
        _g = (f"gaze={'shift→D' if gaze['shifted'] else 'no'}(m={gaze['margin']:+.3f},"
              f"clean_prefers_src={gaze['clean_prefers_src']})" if gaze else "—")
        logger.info(f"[{variant}/{tag}]   => {_c} · {_ro} · {_l} · {_j} · {_g}")

        records[key] = dict(
            key=key, variant=variant, datatype=dt, src_lang=sl, dst_lang=dl,
            shared=[A, B], src_target_C=C, diff_target_D=D,
            src_target_desc=inst.src_target_desc, diff_target_desc=inst.expected_next_desc,
            native_desc=inst.dst_next_desc,
            gen_src=raw["gen_src"], gen_dst=raw["gen_dst"], gen_patched=raw["gen_patched"],
            intervention="closed_loop" if args.closed_loop else "reclamp",
            cl_steps=raw.get("cl_steps"), cl_probe_prefill=raw.get("cl_probe_prefill"),
            ro_src=raw["ro_src"], ro_dst=raw["ro_dst"], ro_patched=raw["ro_patched"],
            causal=causal, readout=readout, literal_c=literal_c, joint=joint, gaze=gaze,
        )
        results_fp.write_text(json.dumps(list(records.values()), indent=2))

    if args.kv_transfer:
        summ = _summarize_kv(list(records.values()))
        (outdir / "summary.json").write_text(json.dumps(summ, indent=2))
        logger.info(f"[{variant}/{tag}] KV SUMMARY: clean X={summ['clean_native_X']} · kv-only B={summ['kv_only_B']} · "
                    f"kv+q: D(set)={summ['set_progress_D']} Y(count)={summ['count_Y']} "
                    f"X={summ['patched_X_rate']} B={summ['patched_B_rate']} C={summ['patched_C_rate']} "
                    f"neither={summ['patched_neither_rate']} n={summ['n']}")
        return dict(variant=variant, datatype=dt, src_lang=sl, dst_lang=dl, **summ)
    summ = _summarize(list(records.values()))
    (outdir / "summary.json").write_text(json.dumps(summ, indent=2))
    logger.info(f"[{variant}/{tag}] SUMMARY: causal(D)={summ['causal_trigger_rate']} "
                f"readout(D)={summ['readout_shift_rate']} native(B)={summ['native_b_rate']} "
                f"literalC={summ['literal_c_rate']} gaze={summ['gaze_shift_rate']} "
                f"(clean_baseline={summ['gaze_clean_baseline_rate']}) n={summ['n']}")
    logger.info(f"[{variant}/{tag}] JOINT (exclusive): B(no-op)={summ['joint_native_rate']} "
                f"C(src-target)={summ['joint_c_rate']} D(progress)={summ['joint_d_rate']} "
                f"neither={summ['joint_neither_rate']}")
    return dict(variant=variant, datatype=dt, src_lang=sl, dst_lang=dl, **summ)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--variants", nargs="+", default=["order_matched", "order_diff"],
                    choices=["order_matched", "order_diff", "c_in_dst"])
    ap.add_argument("--datatypes", nargs="+", default=["int", "str", "list[int]", "list[str]"])
    ap.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(LANG_TRANSFERS))
    ap.add_argument("--naming", default="letters", choices=["natural", "letters"])
    ap.add_argument("--pivot", default="indent", choices=["indent", "eq"],
                    help="query pivot: 'indent' (bare trailing indent) or 'eq' (the frontier's {var}= stub)")
    ap.add_argument("--restrict_goal_heads_to_task", "--attn_restrict", dest="restrict_goal_heads_to_task",
                    action="store_true",
                    help="blind the goal heads (carrying the transplanted query) to everything except the task span")
    ap.add_argument("--allow_family_overlap", action="store_true",
                    help="DISABLE the diff-trigger overlap constraint (default: B/C/D from distinct "
                         "predicate families + non-trivial C/D). Only for ablating the constraint.")
    ap.add_argument("--patch-heads", default="goal", help="'goal'|'union'|path to JSON [[layer,head],...]")
    ap.add_argument("--readout-heads", default="goal")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--n-steer", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0],
                    help="OV-boost value(s) for the patch heads on the patched gen/readout; "
                         "each runs in its own __a<alpha> dir (e.g. --alphas 1.0 2.0)")
    ap.add_argument("--oracle-model", default="claude-sonnet-4-6")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--component-mask", default=None,
                    help="SVD-component mask.pt (train_nrgs_svd_mask.py): patch only those W_QK components of the "
                         "heads it covers (overrides --patch-heads)")
    ap.add_argument("--component-threshold", type=float, default=None,
                    help="binarize the component mask at this value (default: soft trained values)")
    ap.add_argument("--no-readout", action="store_true",
                    help="skip the goalscope readouts + readout-shift oracle (readout_shift_rate -> None)")
    ap.add_argument("--closed-loop", action="store_true",
                    help="closed-loop intervention: regenerate the goal heads' query at every step from the patched "
                         "pivot's goal encoding via a steered probe (instead of re-clamping the SRC query)")
    ap.add_argument("--cl-probe-steer", default="all", choices=["all", "pivot"],
                    help="closed loop: steer the probe at every appended token ('all') or only its pivot")
    ap.add_argument("--cl-max-new", type=int, default=64,
                    help="closed loop: generated tokens (no statement cut-off; the oracle judges the first computation)")
    ap.add_argument("--cl-alpha", type=float, default=0.1, help="closed loop: additive_norm alpha of the probe steer")
    ap.add_argument("--cl-steer-layers", type=int, nargs=2, default=[35, 46],
                    help="closed loop: probe steer into the residual after layers range(a, b)")
    ap.add_argument("--kv-transfer", action="store_true",
                    help="04.3/04.4 kv design: p_dst A'XY gets p_kv A'BD's task-list k+v on the patch heads; "
                         "judge over {X, Y, B, C, D}")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the run tag (e.g. '__seq'): a fresh output dir, so resume never picks "
                         "up records from a run whose prompts differ (the resume key is predicate ids only)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    if args.cuda:
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    ctx = FUNC.setup(args.model)
    mt = ctx.mt
    patch_heads = _resolve_heads(ctx, args.patch_heads)
    readout_heads = _resolve_heads(ctx, args.readout_heads)
    cm_tag = None
    if args.component_mask:
        import src.diff.nrgs_svd_mask as SM
        args._component_proj, patch_heads = SM.component_projectors(mt, args.component_mask, args.component_threshold)
        run = Path(args.component_mask).parent.name.split("__")
        cm_tag = (f"cm-{run[0]}-{run[2]}" + ("-goal" if "only-goal" in run else "") + f"-{run[3] if 'only-goal' not in run else run[4]}"
                  + ("-soft" if args.component_threshold is None else f"-thr{args.component_threshold:g}"))
        logger.info(f"component mask {args.component_mask}: {len(patch_heads)} heads "
                    f"({'soft' if args.component_threshold is None else f'> {args.component_threshold}'})")
    logger.info(f"patch_heads={_heads_tag(args.patch_heads, len(patch_heads))} (n={len(patch_heads)}) · "
                f"readout_heads={_heads_tag(args.readout_heads, len(readout_heads))} (n={len(readout_heads)})")

    model_short = args.model.split("/")[-1]
    for alpha in args.alphas:
        run_tag = f"ph-{cm_tag or _heads_tag(args.patch_heads, len(patch_heads))}__a{alpha:g}__piv-{args.pivot}"
        if args.restrict_goal_heads_to_task:
            run_tag += "__restrict"
        if args.kv_transfer:
            assert not args.restrict_goal_heads_to_task, "--kv-transfer does not support --attn_restrict"
            run_tag += "__kv"
        if args.closed_loop:
            run_tag += "__cl"
            if args.cl_probe_steer != "all":
                run_tag += f"-ps{args.cl_probe_steer}"
            if args.cl_max_new != 64:
                run_tag += f"-m{args.cl_max_new}"
            if args.cl_alpha != 0.1 or list(args.cl_steer_layers) != [35, 46]:
                run_tag += f"-a{args.cl_alpha:g}-L{args.cl_steer_layers[0]}-{args.cl_steer_layers[1]}"
        run_tag += args.tag_suffix
        out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / "diff_trigger" / run_tag
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"########## OV boost alpha={alpha} -> {run_tag} ##########")
        metas = []
        for variant in args.variants:
            for dt in args.datatypes:
                for lt in args.lang_transfers:
                    sl, dl = LANG_TRANSFERS[lt]
                    logger.info(f"═══ a{alpha} · {variant} · {dt} · {sl}->{dl} ═══")
                    metas.append(run_config(mt, args, variant, dt, sl, dl, patch_heads,
                                            readout_heads, out_root, alpha))
        (out_root / "summary.json").write_text(json.dumps(metas, indent=2))
        logger.info(f"done alpha={alpha}. summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
