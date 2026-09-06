#!/usr/bin/env python
"""Batch kv-transfer (Spec-2) intervention + eval over the DIFF data configs.

For every contrastive instance — kv-cluster (``src.diff.data.kv_transfer_setups``: SRC.task != DST.task,
same prefilled progress) or multi-predicate (``src.diff.data.multi_predicate_setups``: shared prefix
T1,T2, divergent frontier T3 vs T4) — this runs the k/v span transplant
(``src.diff.functional.run_kv_transfer_intervention``, which handles both) and the two oracle evals
(``src.diff.eval.eval_kv_transfer``): causal (does the patched DST's first computation jump to the SRC
target?) and readout-shift. **No gaze** — the SRC target never appears in the DST prompt.

Sibling of ``q_transfer_eval.py`` (same interface, resume/force semantics, per-config folders). The
two setups share the whole mechanism + eval; they differ only in how instances are SAMPLED/keyed.
``--setups dual_comp multi_pred`` selects which to run (default ``dual_comp``). Config axes (the save folder is decided
by all of them):
  dual_comp :  datatype × lang-transfer × patch_heads × ablate-range × ablation-strat × alpha × align
  multi_pred:  datatype × naming × letter_stub × lang-transfer × patch_heads × ablate-range × ablation-strat × alpha × align
(presentation is fixed: dual_comp=description, multi_pred=listed.) Results live under
``results/diff/<model_short>/kv_transfer/<setup>/<run_tag>/<config_tag>/`` with ``results.json``
(per-instance, written after each instance so a crash is resumable) and ``meta.json`` (config +
summary rates). ``--resume`` (default) skips instances already present; ``--force`` re-runs them.

READOUT_HEADS (the goalscope readout heads) is ALWAYS ``GOAL_HEADS`` — kept separate from the patched
heads and intentionally NOT a config axis (the readout injects the goal-head write into a neutral
probe, so it carries no cross-context inconsistency and needs no per-config variation).

Typical usage
─────────────
    conda run -n goals python scripts/diff/kv_transfer_eval.py \\
        --setups dual_comp multi_pred --datatypes list[str] list[int] \\
        --lang-transfers py-py py-cpp --patch-heads goal --align pad \\
        --ablate-layer-range 35,50 --ablation-strat restrict --alpha 1.0 --cuda 0,7

    # cheap dry pass: no oracle, cap 2 instances/config
    conda run -n goals python scripts/diff/kv_transfer_eval.py --no-oracle --max-instances 2 --cuda 0,7
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _preparse_cuda(argv):
    """Read --cuda from argv BEFORE importing torch (CUDA_VISIBLE_DEVICES is honored only if set
    before the CUDA context initializes, which the import chain triggers)."""
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

import src.diff.data as D  # noqa: E402
import src.diff.eval as E  # noqa: E402
import src.diff.functional as P  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger("kv_transfer_eval")

MECHANISM = "kv_transfer"  # results tree: diff/<model>/<MECHANISM>/<setup>/<run_tag>/<config_tag>/

LANG_TRANSFERS = {
    "py-py": ("python", "python"),
    "py-cpp": ("python", "cpp"),
    "cpp-py": ("cpp", "python"),
    "cpp-cpp": ("cpp", "cpp"),
}


# ══════════════════════════════════════════════════════════════════════════════
# saved data structures
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class InstanceRecord:
    """Everything about one contrastive instance: identity + RAW artifacts + eval verdicts."""

    key: str
    setup: str  # "dual_comp" | "multi_pred"
    datatype: str
    group: str  # kv: cluster id ; multi: the shared prefix (T1,T2) ids
    naming: str  # multi: variable-naming scheme ; kv: "desc" (n/a)
    presentation: str
    src_lang: str
    dst_lang: str
    var_name: str
    expected_next: str  # SRC target (transplant target)
    expected_next_desc: str
    dst_next: str  # clean DST target
    dst_next_desc: str
    # multi: frontier-pivot shape ("indent" | "var_eq"); kv: "na". Needed for faithful 05.2 rebuild.
    # Defaulted + placed last so pre-existing results.json (no such field) still reconstruct.
    letter_stub: str = "na"
    # raw intervention artifacts (no attention — gaze is N/A for Spec-2)
    raw: dict = field(default_factory=dict)
    # eval verdicts (no gaze)
    causal: Optional[dict] = None
    readout: Optional[dict] = None


@dataclass
class ConfigMeta:
    model: str
    setup: str
    datatype: str
    naming: str
    letter_stub: str  # multi: "indent" | "var_eq" ; kv: "na"
    presentation: str
    src_lang: str
    dst_lang: str
    patch_heads: str  # "goal" | "union"
    readout_heads: str  # always "goal" (kept out of the config axes)
    align: str  # "right" | "pad"
    ablate_layer_range: str  # e.g. "35-50" or "off"
    ablation_strat: (
        str  # "zero" (kill OV write) | "restrict" (attn-knockout the task span)
    )
    alpha: float
    oracle_model: str
    n_instances: int
    # summary rates (over instances with a verdict)
    causal_trigger_rate: Optional[float] = None
    readout_valid_rate: Optional[float] = None
    readout_shift_rate: Optional[float] = None
    readout_shift_rate_valid: Optional[float] = None  # shift among VALID cases only
    n_parse_error: int = 0  # instances with >=1 oracle PARSE_ERROR — re-run these


def _safe(dt: str) -> str:
    return dt.replace("[", "_").replace("]", "").replace(" ", "")


def _config_tag(setup, dt, variant, sl, dl, letter_stub="na") -> str:
    """Per-config DATA axes only (setup is a parent dir now; the intervention knobs live in the run tag).
    multi_pred carries the frontier-pivot shape (``letter_stub``) as its own segment; kv omits it."""
    v = variant or ("randnm" if setup == "multi_pred" else "desc")
    if setup == "multi_pred":
        return f"{_safe(dt)}__{v}__{letter_stub}__{sl}-{dl}"
    return f"{_safe(dt)}__{v}__{sl}-{dl}"


def _run_tag(patch_heads, abl, ablation_strat, alpha, align) -> str:
    """Run-level intervention knobs (single-valued per invocation). Each run gets its own subdir —
    ``.../{model}/{run_tag}/{config_tag}/`` — with its OWN summary.json, so different runs never
    overwrite each other. The ablation strategy is folded into the ablate segment (``abl-35-50-restrict``)
    only when ablation is ON — ``abl-off`` stays strat-free so those runs keep their existing tag."""
    abl_seg = abl if abl == "off" else f"{abl}-{ablation_strat}"
    return f"ph-{patch_heads}__abl-{abl_seg}__a{alpha:g}__{align}"


def _inst_key(inst) -> str:
    if isinstance(inst, D.KVTransferInstance):
        return f"kv|{inst.meta['cluster']}|src={inst.src_task}|dst={inst.dst_task}"
    # multi-predicate: identity = shared prefix + the two frontiers
    return (
        f"multi|{inst.datatype}|shared={'-'.join(p.id for p in inst.shared)}"
        f"|src={inst.src_frontier.id}|dst={inst.dst_frontier.id}"
    )


def _group_of(inst) -> str:
    if isinstance(inst, D.KVTransferInstance):
        return inst.meta["cluster"]
    return "shared:" + "-".join(p.id for p in inst.shared)


def _naming_of(inst) -> str:
    return getattr(inst, "naming", None) or "desc"


def _resolve_datatypes(setup: str, requested) -> list[str]:
    avail = list(D.KVCLUSTERS) if setup == "dual_comp" else list(D.QTASKS)
    if not requested:
        return avail
    keep = [d for d in requested if d in avail]
    skipped = [d for d in requested if d not in avail]
    if skipped:
        logger.warning(
            f"[{setup}] datatypes not available for this setup, skipping: {skipped}"
        )
    return keep


def _variant_axis(setup: str, args) -> list:
    """multi: the naming axis (default ['natural'] — MultiPredicateInstance is frozen, so no per-pair
    random). kv: no naming axis (['desc'])."""
    if setup == "multi_pred":
        return args.namings if args.namings else ["natural"]
    return ["desc"]


def _letter_stub_axis(setup: str, args) -> list:
    """multi: frontier-pivot shape axis ('indent' | 'var_eq'). kv: n/a (['na'])."""
    if setup == "multi_pred":
        return args.letter_stubs if args.letter_stubs else ["indent"]
    return ["na"]


def _parse_ablate_range(s: str) -> tuple[Optional[int], Optional[int], str]:
    """'35,50' / '35-50' -> (35, 50, '35-50'); 'off'/'none'/'' -> (None, None, 'off')."""
    if s is None or s.strip().lower() in ("", "off", "none"):
        return None, None, "off"
    lo, hi = (int(x) for x in s.replace("-", ",").split(","))
    return lo, hi, f"{lo}-{hi}"


def _summarize(records: list[InstanceRecord]) -> dict:
    def rate(pred, over=None):
        pool = [r for r in records if (over is None or over(r))]
        vals = [pred(r) for r in pool if pred(r) is not None]
        return (sum(vals) / len(vals)) if vals else None

    causal = rate(lambda r: r.causal["triggered"] if r.causal else None)
    r_valid = rate(lambda r: r.readout["valid"] if r.readout else None)
    r_shift = rate(lambda r: r.readout["shifted"] if r.readout else None)
    r_shift_valid = rate(
        lambda r: (r.readout["shifted"] if r.readout else None),
        over=lambda r: r.readout and r.readout["valid"],
    )

    def _has_parse_error(r) -> bool:
        if r.causal and r.causal.get("label") == "parse_error":
            return True
        if r.readout:
            return any(
                r.readout.get(k) == "parse_error" for k in ("source", "dest", "patched")
            )
        return False

    n_parse_error = sum(1 for r in records if _has_parse_error(r))
    return dict(
        causal_trigger_rate=causal,
        readout_valid_rate=r_valid,
        readout_shift_rate=r_shift,
        readout_shift_rate_valid=r_shift_valid,
        n_parse_error=n_parse_error,
    )


# ══════════════════════════════════════════════════════════════════════════════
# per-config run
# ══════════════════════════════════════════════════════════════════════════════
def run_config(
    mt,
    args,
    setup,
    dt,
    variant,
    letter_stub,
    sl,
    dl,
    patch_heads,
    readout_heads,
    ablate_heads,
    out_root,
):
    _, _, abl_tag = _parse_ablate_range(args.ablate_layer_range)
    tag = _config_tag(setup, dt, variant, sl, dl, letter_stub)
    outdir = out_root / tag  # out_root already includes the run tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    done: dict[str, dict] = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{tag}] resume: {len(done)} instances already done")

    # REPRODUCIBLE per-config sample (seed = base seed + config tag) -> resume-stable.
    cfg_seed = (args.seed + int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)) % (
        2**32
    )
    set_seed(cfg_seed)
    presentation = "description" if setup == "dual_comp" else "listed"
    if setup == "dual_comp":
        setups = D.kv_transfer_setups(
            dt,
            presentation=presentation,
            src_lang=sl,
            dst_lang=dl,
            n_samples=args.n_samples,
            seed=cfg_seed,
        )
        size_note = "kv-cluster"
    else:  # multi-predicate
        setups = D.multi_predicate_setups(
            dt,
            n_conditions=args.n_conditions,
            n_samples=args.n_samples,
            seed=cfg_seed,
            presentation=presentation,
            naming=variant or "natural",
            letter_stub=letter_stub,
            src_lang=sl,
            dst_lang=dl,
        )
        size_note = (
            f"n_conditions={args.n_conditions},naming={variant or 'natural'},"
            f"letter_stub={letter_stub}"
        )

    if args.max_instances:
        setups = setups[: args.max_instances]

    records: dict[str, dict] = dict(done)
    todo = [inst for inst in setups if _inst_key(inst) not in records or args.force]
    logger.info(
        f"[{tag}] sampled {len(setups)} instances ({size_note}, seed={cfg_seed}, "
        f"align={args.align}, alpha={args.alpha}, ablate={abl_tag}, n_ablate={len(ablate_heads)}, "
        f"strat={args.ablation_strat}) — {len(setups) - len(todo)} resumed, {len(todo)} to run"
    )

    for j, inst in enumerate(todo, 1):
        key = _inst_key(inst)
        logger.info(
            f"[{tag}] ({j}/{len(todo)}) [{_group_of(inst)},{_naming_of(inst)}] "
            f"SRC {inst.expected_next!r} -> DST {inst.dst_next!r}"
            f"  |  {inst.expected_next_desc!r}  vs  {inst.dst_next_desc!r}"
        )
        raw = P.run_kv_transfer_intervention(
            mt,
            inst,
            patch_heads,
            readout_heads=readout_heads,
            max_new=args.max_new,
            alpha=args.alpha,
            align=args.align,
            ablate_heads=ablate_heads,
            ablation_strat=args.ablation_strat,
        )

        ev_kw = dict(model_name=args.oracle_model)
        causal = readout = None
        if not args.no_oracle:
            causal = E.judge_causal(inst, raw["gen_patched"], **ev_kw)
            readout = E.eval_readouts(
                inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"], **ev_kw
            )

        _c = (
            f"{'HIT' if causal['triggered'] else 'miss'}({causal['label']})"
            if causal
            else "off"
        )
        _r = (
            f"{readout['source']}>{readout['dest']}>{readout['patched']}"
            f"[{'valid' if readout['valid'] else 'inv'},{'shift' if readout['shifted'] else 'no'}]"
            if readout
            else "off"
        )
        logger.info(f"[{tag}]   => causal={_c} · readout={_r}")

        rec = InstanceRecord(
            key=key,
            setup=setup,
            datatype=dt,
            group=_group_of(inst),
            naming=_naming_of(inst),
            letter_stub=letter_stub,
            presentation=presentation,
            src_lang=sl,
            dst_lang=dl,
            var_name=inst.var_name,
            expected_next=inst.expected_next,
            expected_next_desc=inst.expected_next_desc,
            dst_next=inst.dst_next,
            dst_next_desc=inst.dst_next_desc,
            raw=raw,
            causal=causal,
            readout=readout,
        )
        records[key] = dataclasses.asdict(rec)
        tmp = outdir / "results.json.tmp"
        tmp.write_text(json.dumps(list(records.values()), indent=2))
        tmp.replace(results_fp)

    recs = [InstanceRecord(**{k: v for k, v in r.items()}) for r in records.values()]
    meta = ConfigMeta(
        model=args.model,
        setup=setup,
        datatype=dt,
        naming=(variant or ("random" if setup == "multi_pred" else "desc")),
        letter_stub=letter_stub,
        presentation=presentation,
        src_lang=sl,
        dst_lang=dl,
        patch_heads=args.patch_heads,
        readout_heads="goal",
        align=args.align,
        ablate_layer_range=abl_tag,
        ablation_strat=args.ablation_strat,
        alpha=args.alpha,
        oracle_model=args.oracle_model,
        n_instances=len(recs),
        **_summarize(recs),
    )
    (outdir / "meta.json").write_text(json.dumps(dataclasses.asdict(meta), indent=2))
    logger.info(
        f"[{tag}] done — {len(recs)} instances | causal={meta.causal_trigger_rate} "
        f"readout(valid={meta.readout_valid_rate}, shift|valid={meta.readout_shift_rate_valid})"
    )
    return meta


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g. '0,7')")
    p.add_argument(
        "--setups",
        nargs="+",
        default=["dual_comp"],
        choices=["dual_comp", "multi_pred"],
        help="which Spec-2 setup(s): dual_comp (kv-cluster, default) and/or multi_pred "
        "(multi-predicate).",
    )
    p.add_argument(
        "--datatypes",
        nargs="+",
        default=None,
        choices=sorted(set(D.KVCLUSTERS) | set(D.QTASKS)),
        help="which datatypes to run (default: all available for each setup).",
    )
    p.add_argument(
        "--namings",
        nargs="+",
        default=["letters"],
        choices=["natural", "letters"],
        help="multi-predicate variable-naming axis (ignored by kv). Default: ['letters'].",
    )
    p.add_argument(
        "--letter-stubs",
        nargs="+",
        default=["indent"],
        choices=["indent", "var_eq"],
        help="multi-predicate frontier-pivot shape axis (ignored by kv): 'indent' (default) = trailing "
        "indent, model writes the whole frontier line; 'var_eq' = also prefill the frontier's letter "
        "var up to '=' (python '    d =' / c++ '    auto d ='), model writes only the frontier VALUE. "
        "Intended with --namings letters.",
    )
    p.add_argument(
        "--lang-transfers",
        nargs="+",
        default=["py-py", "py-cpp"],
        choices=list(LANG_TRANSFERS),
    )
    p.add_argument(
        "--patch-heads",
        default="goal",
        choices=["goal", "union"],
        help="heads whose k/v is transplanted (config axis). Readout is ALWAYS goal, regardless.",
    )
    p.add_argument(
        "--align",
        default="pad",
        choices=["right", "pad"],
        help="transplant-window alignment (config axis): 'right' = right-edge (K=min, drop the longer "
        "span's front); 'pad' = token-splice newlines before the shorter target so both spans are "
        "K=max and the dest span is FULLY overwritten.",
    )
    p.add_argument(
        "--ablate-layer-range",
        default="33,43",
        help="layers whose non-patch heads are ablated on the patched GENERATION (via --ablation-strat; "
        "kills the k/v-inconsistency 'backup' goal calc). 'lo,hi' or 'lo-hi' (inclusive); "
        "'off' to disable. Config axis. ABLATE_HEADS = these layers' heads minus PATCH_HEADS.",
    )
    p.add_argument(
        "--ablation-strat",
        default="restrict",
        choices=["zero", "restrict"],
        help="HOW ABLATE_HEADS are ablated on the patched gen (config axis; folded into the run tag "
        "when ablation is on): 'zero' = ZERO their OV write at all positions; 'restrict' = keep their "
        "write but attention-knockout the DST task span (they attend everywhere except the task text).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="OV boost of the patch heads at the pivot on the patched gen + readout (1.0 = off). "
        "Config axis.",
    )
    p.add_argument(
        "--n-conditions",
        type=int,
        nargs="+",
        default=[2, 3],
        help="multi-predicate task sizes to sample from (per instance, one is chosen at random from "
        "this list); each = #independent commutative predicates in the task (shared prefix + the two "
        "frontiers). Ignored by the kv setup.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="contrastive instances to RANDOM-sample per config (reproducible, seed-stable).",
    )
    p.add_argument(
        "--seed", type=int, default=1123, help="base RNG seed for per-config sampling"
    )
    p.add_argument(
        "--max-instances", type=int, default=None, help="further cap instances/config"
    )
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument("--no-oracle", action="store_true", help="skip causal+readout eval")
    p.add_argument(
        "--force", action="store_true", help="re-run instances even if present"
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false", help="ignore prior results"
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging_utils.configure(level=logging.INFO, force=True)
    if args.cuda:
        logger.info(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (requested {args.cuda})"
        )

    ctx = P.setup(args.model)  # loads model + head masks; binds module globals
    mt = ctx.mt
    patch_heads = ctx.goal_heads if args.patch_heads == "goal" else ctx.union_heads
    # READOUT_HEADS — always the goal heads, kept SEPARATE from the patched heads and NOT a config axis.
    readout_heads = ctx.goal_heads

    # ABLATE_HEADS (I2): every head in the ablate layer-range, minus PATCH_HEADS. Zeroed on the
    # patched generation only (the runner also defensively excludes patch_heads).
    lo, hi, abl_tag = _parse_ablate_range(args.ablate_layer_range)
    ablate_all = (
        [(l, h) for l in range(lo, hi + 1) for h in range(mt.n_heads)]
        if lo is not None
        else []
    )
    ablate_heads = [h for h in ablate_all if h not in set(patch_heads)]
    logger.info(
        f"patch_heads={args.patch_heads} (n={len(patch_heads)}) · readout_heads=goal "
        f"(n={len(readout_heads)}) · ablate={abl_tag} (n={len(ablate_heads)}, strat={args.ablation_strat}) "
        f"· align={args.align} · alpha={args.alpha}"
    )

    model_short = args.model.split("/")[-1]
    # groups this run's outputs (ablation strat folded into the ablate segment when ablation is on)
    run_tag = _run_tag(
        args.patch_heads, abl_tag, args.ablation_strat, args.alpha, args.align
    )
    logger.info(f"run_tag={run_tag}")

    configs = []  # (setup, dt, variant, letter_stub, lt)
    for setup in args.setups:
        for dt in _resolve_datatypes(setup, args.datatypes):
            for variant in _variant_axis(setup, args):
                for ls in _letter_stub_axis(setup, args):
                    for lt in args.lang_transfers:
                        configs.append((setup, dt, variant, ls, lt))
    logger.info(
        f"{len(configs)} configs to process across setups={args.setups} "
        f"lang_transfers={args.lang_transfers} letter_stubs={args.letter_stubs}"
    )

    metas_by_setup: dict[str, list] = {}
    for ci, (setup, dt, variant, letter_stub, lt) in enumerate(configs, 1):
        sl, dl = LANG_TRANSFERS[lt]
        out_root = (
            Path(env_utils.DEFAULT_RESULTS_DIR)
            / "diff"
            / model_short
            / MECHANISM
            / setup
            / run_tag
        )
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"═══ config {ci}/{len(configs)}: "
            f"{setup}/{run_tag}/{_config_tag(setup, dt, variant, sl, dl, letter_stub)} ═══"
        )
        meta = run_config(
            mt,
            args,
            setup,
            dt,
            variant,
            letter_stub,
            sl,
            dl,
            patch_heads,
            readout_heads,
            ablate_heads,
            out_root,
        )
        metas_by_setup.setdefault(setup, []).append(meta)

    for setup, metas in metas_by_setup.items():
        summ_fp = (
            Path(env_utils.DEFAULT_RESULTS_DIR)
            / "diff"
            / model_short
            / MECHANISM
            / setup
            / run_tag
            / "summary.json"
        )
        summ_fp.write_text(json.dumps([dataclasses.asdict(m) for m in metas], indent=2))
        logger.info(f"[{setup}] summary ({len(metas)} configs) at {summ_fp}")
    logger.info(
        f"all done — {len(configs)} configs across {len(metas_by_setup)} setup(s)"
    )


if __name__ == "__main__":
    main()
