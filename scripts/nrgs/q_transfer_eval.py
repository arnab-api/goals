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

Reuses the q-transfer machinery: ``src.nrgs.functional.run_q_transfer_intervention`` (single-pivot query
patch at the frontier's ``{var} =`` stub; OV boost alpha on the patched readout) with the gaze-select-restrict
intervention: the patched pivot's gaze picks one listed task, then a separate run without the query transfer in which
the goal heads, from the pivot on, read only that task's span + the pivot + the generated tokens. Judged with
``src.nrgs.eval`` (``judge_causal`` = wrote D; ``judge_joint``; ``eval_readouts``; gaze = pivot gazes at D's span
> B's span) + a literal-C check (``classify_concept`` against C). Defaults = the paper's configuration (alpha 2, the
250 goal heads, 32 instances per cell). Results:
    results/diff/<model>/diff_trigger/<run_tag>/<variant>/<datatype>__<sl>-<dl>/results.json  (+ summary.json)

Usage:
    conda run -n goals python scripts/nrgs/q_transfer_eval.py --cuda 0,1 \\
        --variants order_matched order_diff --datatypes int str --tag-suffix __seq
"""
import argparse
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

import src.nrgs.data as DATA  # noqa: E402
import src.nrgs.eval as E  # noqa: E402
import src.nrgs.functional as FUNC  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger("q_transfer_eval")

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
GOAL250 = "head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json"
MAX_NEW = 96            # clean SRC / DST generations
GSR_MAX_NEW = 64        # the patched (gaze-select-restrict) generation
ORACLE_MODEL = "claude-sonnet-4-6"

LANG_TRANSFERS = {"py-py": ("python", "python"), "py-cpp": ("python", "cpp"),
                  "cpp-py": ("cpp", "python"), "cpp-cpp": ("cpp", "cpp")}


def _resolve_heads(spec):
    """A JSON [[layer, head], ...] head set."""
    arr = json.loads(Path(spec).read_text())
    return [(int(l), int(h)) for l, h in arr]


def _heads_tag(spec, n):
    return f"{Path(spec).stem.split('__')[0]}_{n}"


def _inst_key(inst):
    ids = lambda seq: "-".join(p.id for p in seq)
    return (f"{inst.meta['variant']}|src={ids(inst.src_order)}|dst={ids(inst.dst_order)}"
            f"|prog_src={'-'.join(inst.src_progress)}|prog_dst={'-'.join(inst.dst_progress)}")


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
        select_d_rate=rate(lambda r: (r["cl_selected"] == r["diff_target_D"]) if r.get("cl_selected") else None),
        select_b_rate=rate(lambda r: (r["cl_selected"] == r["key"].split("dst=")[1].split("|")[0].split("-")[1])
                           if r.get("cl_selected") else None),
        gaze_clean_baseline_rate=rate(lambda r: r["gaze"].get("clean_prefers_src") if r.get("gaze") else None),
        # joint B/C/D verdict — MUTUALLY EXCLUSIVE shares (sum to 1 minus parse errors), unlike the
        # binary causal(D)/literalC(C) marginals above which can double-count when C and D compete.
        joint_native_rate=rate(lambda r: (r["joint"]["label"] == "native") if r.get("joint") else None),
        joint_c_rate=rate(lambda r: (r["joint"]["label"] == "src_target") if r.get("joint") else None),
        joint_d_rate=rate(lambda r: (r["joint"]["label"] == "diff_target") if r.get("joint") else None),
        joint_neither_rate=rate(lambda r: (r["joint"]["label"] == "neither") if r.get("joint") else None),
    )


def run_config(mt, args, variant, dt, sl, dl, patch_heads, readout_heads, out_root, alpha):
    # cell tag: datatype with "[" / "]" -> "(" / ")" (also seeds the cell's sample below; keep verbatim)
    tag = f"{dt.replace('[', '(').replace(']', ')')}__{sl}-{dl}"
    outdir = out_root / variant / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    done = {}
    if results_fp.exists() and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{variant}/{tag}] resume: {len(done)} done")

    cfg_seed = (args.seed + int(hashlib.md5(f"{variant}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)
    set_seed(cfg_seed)
    setups = DATA.diff_trigger_setups(dt, variant=variant, n_samples=args.n_samples, seed=cfg_seed,
                                      src_lang=sl, dst_lang=dl)
    if args.max_instances:
        setups = setups[: args.max_instances]

    records = dict(done)
    todo = [inst for inst in setups if _inst_key(inst) not in records or args.force]
    logger.info(f"[{variant}/{tag}] {len(setups)} instances (seed={cfg_seed}, alpha={alpha}) — "
                f"{len(setups) - len(todo)} resumed, {len(todo)} to run")

    for j, inst in enumerate(todo, 1):
        key = _inst_key(inst)
        A, B = inst.src_progress
        C, D = inst.src_target, inst.diff_target
        logger.info(f"[{variant}/{tag}] ({j}/{len(todo)}) shared[{A},{B}] SRC+{C} DST+{D}  "
                    f"| SRC→C={inst.src_target_desc!r} · DIFF→D={inst.expected_next_desc!r} "
                    f"· nativeB={inst.dst_next_desc!r}")
        raw = FUNC.run_q_transfer_intervention(
            mt, inst, patch_heads, readout_heads=readout_heads, max_new=MAX_NEW, alpha=alpha,
            compute_readout=not args.no_readout, gsr_max_new=GSR_MAX_NEW,
        )
        readout = None
        causal = E.judge_causal(inst, raw["gen_patched"], model_name=ORACLE_MODEL)
        # joint B/C/D — single mutually-exclusive verdict (B=no-op, C=src-target, D=progress)
        joint = E.judge_joint(inst, raw["gen_patched"], model_name=ORACLE_MODEL)
        # readout-shift: does the goal-head READOUT of the patched state express D (vs native B)?
        if not args.no_readout:
            readout = E.eval_readouts(inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"], model_name=ORACLE_MODEL)
        # literal-C check: does the patched continuation express SRC's literal target C (vs native B)?
        lit = E.classify_concept(raw["gen_patched"], inst.src_target_desc, inst.dst_next_desc,
                                 inst.task_desc, model_name=ORACLE_MODEL,
                                 kind="the code the model generated after the pivot; judge ONLY the "
                                      "first computation")
        literal_c = {"label": lit["label"], "wrote_c": lit["label"] == "src", "why": lit["why"]}
        gaze = E.eval_q_transfer(inst, gaze=dict(
            prompt=raw["dst_prompt"], task_positions=raw["task_positions"],
            attn_patched=raw["attn_patched"], attn_clean=raw["attn_clean"],
        )).gaze

        _c = f"{'HIT-D' if causal['triggered'] else 'miss'}({causal['label']})"
        _ro = (f"readout={readout['source']}>{readout['dest']}>{readout['patched']}"
               f"[{'shift→D' if readout['shifted'] else 'no'}]") if readout else ""
        _l = f"literalC={'YES' if literal_c['wrote_c'] else 'no'}"
        _j = f"joint={ {'native':'B','src_target':'C','diff_target':'D','neither':'-'}.get(joint['label'],'?') }"
        _g = (f"gaze={'shift→D' if gaze['shifted'] else 'no'}(m={gaze['margin']:+.3f},"
              f"clean_prefers_src={gaze['clean_prefers_src']})" if gaze else "—")
        logger.info(f"[{variant}/{tag}]   => {_c} · {_ro} · {_l} · {_j} · {_g}")

        records[key] = dict(
            key=key, variant=variant, datatype=dt, src_lang=sl, dst_lang=dl,
            shared=[A, B], src_target_C=C, diff_target_D=D,
            src_target_desc=inst.src_target_desc, diff_target_desc=inst.expected_next_desc,
            native_desc=inst.dst_next_desc,
            gen_src=raw["gen_src"], gen_dst=raw["gen_dst"], gen_patched=raw["gen_patched"],
            intervention="closed_loop",
            # cl_steps / cl_probe_prefill / cl_select_wins / cl_select_unsteered / cl_select_order: always None here,
            # kept so the records match the cached results' schema
            cl_steps=None, cl_probe_prefill=None,
            cl_probe_mode=raw["cl_probe_mode"], cl_selected=raw["cl_selected"],
            cl_select_scores=raw["cl_select_scores"], cl_select_wins=None,
            cl_select_unsteered=None, cl_select_order=None, cl_restrict_keep=raw["cl_restrict_keep"],
            ro_src=raw["ro_src"], ro_dst=raw["ro_dst"], ro_patched=raw["ro_patched"],
            causal=causal, readout=readout, literal_c=literal_c, joint=joint, gaze=gaze,
        )
        results_fp.write_text(json.dumps(list(records.values()), indent=2))

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
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--variants", nargs="+", default=["order_matched", "order_diff"],
                    choices=["order_matched", "order_diff", "c_in_dst"])
    ap.add_argument("--datatypes", nargs="+", default=["int", "str", "list[int]", "list[str]"])
    ap.add_argument("--lang-transfers", nargs="+", default=["py-py"], choices=list(LANG_TRANSFERS))
    ap.add_argument("--patch-heads", default=GOAL250, help="JSON [[layer, head], ...]: the heads whose query is transplanted")
    ap.add_argument("--readout-heads", default=GOAL250, help="JSON [[layer, head], ...]: the GoalScope readout heads")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0],
                    help="OV-boost value(s) for the patch heads on the patched readout; "
                         "each runs in its own __a<alpha> dir (e.g. --alphas 1.0 2.0)")
    ap.add_argument("--no-readout", action="store_true",
                    help="skip the goalscope readouts + readout-shift oracle (readout_shift_rate -> None)")
    ap.add_argument("--max-instances", type=int, default=None, help="cap instances per cell (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default=MODEL, help="HF id (default = the paper's Coder-32B)")
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
    patch_heads = _resolve_heads(args.patch_heads)
    readout_heads = _resolve_heads(args.readout_heads)
    logger.info(f"patch_heads={_heads_tag(args.patch_heads, len(patch_heads))} (n={len(patch_heads)}) · "
                f"readout_heads={_heads_tag(args.readout_heads, len(readout_heads))} (n={len(readout_heads)})")

    model_short = args.model.split("/")[-1]
    for alpha in args.alphas:
        # `__piv-eq` (pivot = the frontier's `{var} =` stub), `__cl` (closed loop), `-ppgazerestrict` (gaze-select-
        # restrict) and `-anceq` (anchor = the pivot) name the configuration of the original runs; kept verbatim
        run_tag = (f"ph-{_heads_tag(args.patch_heads, len(patch_heads))}__a{alpha:g}__piv-eq__cl-ppgazerestrict-anceq"
                   + args.tag_suffix)
        out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "diff" / model_short / "diff_trigger" / run_tag   # legacy results dir names, kept for cache compatibility
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
