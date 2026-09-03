#!/usr/bin/env python
"""Batch q-transfer / filter-transfer intervention + eval over the DIFF data configs.

For every contrastive instance — q-transfer (``src.diff.data.q_transfer_setups``) or filter-transfer
(``src.diff.data.filter_transfer_setups``) — this runs the query transplant
(``src.diff.functional.run_q_transfer_intervention``, which handles both) and the three evals
(``src.diff.eval.eval_q_transfer``): causal (oracle), readout-shift (oracle), gaze (rule-based).
Both the RAW intervention artifacts (prompts, generations, readouts, attention) and the eval
verdicts are saved.

The two setups share the whole mechanism + eval pipeline; they differ only in how instances are
SAMPLED and keyed. ``--setups q filter`` selects which to run (default ``q``). Config axes:
  q      :  datatype × naming × lang-transfer   (naming collapses to per-pair random if unset)
  filter :  datatype × postfix × lang-transfer  (postfix fixed to one value; 2 sub-tasks each)
(presentation is fixed to ``listed``.) Each config's results live in its own folder under
``results/diff/{q_transfer,filter_transfer}/<model_short>/<config_tag>/`` with ``results.json``
(per-instance records, written after each instance so a crash is resumable) and ``meta.json``
(config + summary rates). ``--resume`` (default) skips instances already present; ``--force``
re-runs them.

Typical usage
─────────────
    conda run -n goals python scripts/diff/q_transfer_eval.py \\
        --setups q filter --datatypes list[str] list[int] \\
        --lang-transfers py-py py-cpp --cuda 0,7

    # cheap dry pass: rule-based gaze only, cap 2 instances/config
    conda run -n goals python scripts/diff/q_transfer_eval.py --no-oracle --max-instances 2 --cuda 0,7
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
    """Read --cuda from argv BEFORE importing torch. CUDA_VISIBLE_DEVICES is honored only if set
    before the CUDA context initializes, which happens on the first torch.cuda call — and something
    in the import chain (transformers/nnsight/src.models) touches it at import time. Setting it in
    main() is too late (the model then spreads over ALL gpus)."""
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

logger = logging.getLogger("q_transfer_eval")

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
    # identity / config
    datatype: str
    naming: str  # q: variable-naming scheme; filter: the postfix variant
    presentation: str
    src_lang: str
    dst_lang: str
    var_name: str
    expected_next: str  # SRC frontier (transplant target)
    expected_next_desc: str
    dst_next: str  # clean DST frontier
    dst_next_desc: str
    src_progress: list
    dst_progress: list
    # raw intervention artifacts
    raw: dict = field(default_factory=dict)
    # eval verdicts
    causal: Optional[dict] = None
    readout: Optional[dict] = None
    gaze: Optional[dict] = None
    setup: str = "q"  # "q" | "filter" (default keeps pre-filter records loadable)


@dataclass
class ConfigMeta:
    model: str
    setup: str
    datatype: str
    naming: str
    presentation: str
    src_lang: str
    dst_lang: str
    patch_heads: str
    oracle_model: str
    n_instances: int
    # summary rates (over instances with a verdict)
    causal_trigger_rate: Optional[float] = None
    readout_valid_rate: Optional[float] = None
    readout_shift_rate: Optional[float] = None
    readout_shift_rate_valid: Optional[float] = None  # shift among VALID cases only
    gaze_shift_rate: Optional[float] = None            # patched gaze(SRC) > gaze(DST)
    gaze_clean_baseline_rate: Optional[float] = None   # clean gaze(SRC) > gaze(DST) — should be LOW
    n_parse_error: int = 0  # instances with >=1 oracle PARSE_ERROR (causal or any readout) — re-run these


def _safe(dt: str) -> str:
    return dt.replace("[", "_").replace("]", "").replace(" ", "")


def _config_tag(setup, dt, variant, sl, dl) -> str:
    # variant = naming (q; None => per-pair random) or postfix (filter)
    v = variant or ("randnm" if setup == "q" else "targeted")
    return f"{setup}__{_safe(dt)}__{v}__{sl}-{dl}"


def _setup_dir(setup: str) -> str:
    return "q_transfer" if setup == "q" else "filter_transfer"


def _pick_naming(tag: str, key: str) -> str:
    """Deterministic per-pair naming when the naming axis is collapsed (no --namings). Stable across
    runs (md5, not the salted built-in hash) so resume assigns the same naming to the same pair."""
    h = int(hashlib.md5(f"{tag}|{key}".encode()).hexdigest()[:8], 16)
    return "letters" if h % 2 else "natural"


def _inst_key(inst) -> str:
    if hasattr(inst, "src_progress"):  # q-transfer
        return (
            f"{inst.expected_next}|src={'-'.join(inst.src_progress)}"
            f"|dst={'-'.join(inst.dst_progress)}"
        )
    # filter-transfer: identity = the two sub-tasks + which side prefilled which
    return (
        f"{inst.expected_next}|pair={'-'.join(inst.pair)}"
        f"|srcfill={inst.src_fill}|dstfill={inst.dst_fill}"
    )


def _naming_of(inst) -> str:
    """q: the variable-naming scheme; filter: the postfix variant (there is no naming axis)."""
    return getattr(inst, "naming", None) or getattr(inst, "postfix", "n/a")


def _task_size(inst) -> int:
    # q: #predicates in the sampled task (k). filter: the 2 sub-tasks in play (inst.pair),
    # NOT the full condition bank on inst.task.
    if hasattr(inst, "pair"):
        return len(inst.pair)
    return len(inst.task.predicates)


def _progress_of(inst) -> tuple[list, list]:
    """Uniform (src_progress, dst_progress) for the record schema. Filter has no progress list —
    represent it by the single prefilled sub-task on each side."""
    if hasattr(inst, "src_progress"):
        return list(inst.src_progress), list(inst.dst_progress)
    return [inst.src_fill], [inst.dst_fill]


def _resolve_datatypes(setup: str, requested) -> list[str]:
    avail = list(D.QTASKS) if setup == "q" else list(D.FILTER_TASKS)
    if not requested:
        return avail
    keep = [d for d in requested if d in avail]
    skipped = [d for d in requested if d not in avail]
    if skipped:
        logger.warning(f"[{setup}] datatypes not available for this setup, skipping: {skipped}")
    return keep


def _variant_axis(setup: str, args) -> list:
    """q: the naming axis ([None] => collapse to per-pair random). filter: a single postfix."""
    if setup == "q":
        return args.namings if args.namings else [None]
    return [args.postfix]


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
    gaze = rate(lambda r: r.gaze["shifted"] if r.gaze else None)
    gaze_clean = rate(lambda r: r.gaze.get("clean_prefers_src") if r.gaze else None)

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
        gaze_shift_rate=gaze,
        gaze_clean_baseline_rate=gaze_clean,
        n_parse_error=n_parse_error,
    )


# ══════════════════════════════════════════════════════════════════════════════
# per-config run
# ══════════════════════════════════════════════════════════════════════════════
def run_config(mt, args, setup, dt, variant, sl, dl, patch_heads, readout_heads, out_root):
    tag = _config_tag(setup, dt, variant, sl, dl)
    outdir = out_root / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    # load prior results for resume
    done: dict[str, dict] = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{tag}] resume: {len(done)} instances already done")

    # REPRODUCIBLE RANDOM SAMPLE of --n-samples instances (the setup's *_setups fn samples internally).
    # Per-config seed = base seed + config tag, so the sampled set is identical across runs -> resume-stable.
    cfg_seed = (args.seed + int(hashlib.md5(tag.encode()).hexdigest()[:8], 16)) % (
        2**32
    )
    set_seed(cfg_seed)
    if setup == "q":
        # each instance = a random n_conditions-predicate task + a fixed contrastive split
        setups = D.q_transfer_setups(
            dt,
            n_conditions=args.n_conditions,
            n_samples=args.n_samples,
            seed=cfg_seed,
            src_n_prefill=args.src_n_prefill,
            dst_n_prefill=args.dst_n_prefill,
            presentation="listed",
            naming=variant or "natural",
            src_lang=sl,
            dst_lang=dl,
        )
        # naming=None => collapse the axis: deterministic per-pair naming (independent of the sample).
        if variant is None:
            for inst in setups:
                inst.naming = _pick_naming(tag, _inst_key(inst))
        size_note = f"k={args.n_conditions}"
    else:  # filter — 2 commutative sub-tasks, SRC/DST prefill different single sub-tasks
        setups = D.filter_transfer_setups(
            dt,
            n_samples=args.n_samples,
            seed=cfg_seed,
            postfix=variant or "targeted",
            src_lang=sl,
            dst_lang=dl,
        )
        size_note = f"postfix={variant or 'targeted'}"

    if args.max_instances:
        setups = setups[: args.max_instances]

    records: dict[str, dict] = dict(done)
    todo = [inst for inst in setups if _inst_key(inst) not in records or args.force]
    logger.info(
        f"[{tag}] sampled {len(setups)} instances ({size_note}, seed={cfg_seed}) — "
        f"{len(setups) - len(todo)} resumed, {len(todo)} to run"
    )

    for j, inst in enumerate(todo, 1):
        key = _inst_key(inst)
        # contrastive concept pair: SRC frontier (transplant target) -> DST frontier (clean)
        logger.info(
            f"[{tag}] ({j}/{len(todo)}) [n={_task_size(inst)},{_naming_of(inst)}] "
            f"SRC {inst.expected_next!r} -> DST {inst.meta['dst_next']!r}"
            f"  |  {inst.expected_next_desc!r}  vs  {inst.dst_next_desc!r}"
        )
        raw = P.run_q_transfer_intervention(
            mt,
            inst,
            patch_heads,
            readout_heads=readout_heads,
            n_steer=args.n_steer,
            max_new=args.max_new,
        )

        ev_kw = dict(model_name=args.oracle_model)
        causal = readout = None
        if not args.no_oracle:
            causal = E.judge_causal(inst, raw["gen_patched"], **ev_kw)
            readout = E.eval_readouts(
                inst, raw["ro_src"], raw["ro_dst"], raw["ro_patched"], **ev_kw
            )
        gaze_res = E.eval_q_transfer(
            inst,
            gaze=dict(
                prompt=raw["dst_prompt"],
                task_positions=raw["task_positions"],
                attn_patched=raw["attn_patched"],
                attn_clean=raw["attn_clean"],
            ),
        ).gaze

        # short per-pair eval result
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
        _g = (
            f"{'shift' if gaze_res['shifted'] else 'no'}(src>dst m={gaze_res['margin']:+.3f}, "
            f"clean_prefers_src={gaze_res['clean_prefers_src']})"
            if gaze_res
            else "—"
        )
        logger.info(f"[{tag}]   => causal={_c} · readout={_r} · gaze={_g}")

        src_prog, dst_prog = _progress_of(inst)
        rec = InstanceRecord(
            key=key,
            setup=setup,
            datatype=dt,
            naming=_naming_of(inst),
            presentation="listed",
            src_lang=sl,
            dst_lang=dl,
            var_name=inst.var_name,
            expected_next=inst.expected_next,
            expected_next_desc=inst.expected_next_desc,
            dst_next=inst.meta["dst_next"],
            dst_next_desc=inst.dst_next_desc,
            src_progress=src_prog,
            dst_progress=dst_prog,
            raw=raw,
            causal=causal,
            readout=readout,
            gaze=gaze_res,
        )
        records[key] = dataclasses.asdict(rec)
        # write after each instance (atomic) so partial configs are resumable
        tmp = outdir / "results.json.tmp"
        tmp.write_text(json.dumps(list(records.values()), indent=2))
        tmp.replace(results_fp)

    recs = [InstanceRecord(**{k: v for k, v in r.items()}) for r in records.values()]
    meta = ConfigMeta(
        model=args.model,
        setup=setup,
        datatype=dt,
        naming=(variant or ("random" if setup == "q" else "targeted")),
        presentation="listed",
        src_lang=sl,
        dst_lang=dl,
        patch_heads=args.patch_heads,
        oracle_model=args.oracle_model,
        n_instances=len(recs),
        **_summarize(recs),
    )
    (outdir / "meta.json").write_text(json.dumps(dataclasses.asdict(meta), indent=2))
    logger.info(
        f"[{tag}] done — {len(recs)} instances | causal={meta.causal_trigger_rate} "
        f"readout(valid={meta.readout_valid_rate}, shift|valid={meta.readout_shift_rate_valid}) "
        f"gaze(patched={meta.gaze_shift_rate}, clean_baseline={meta.gaze_clean_baseline_rate})"
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
        default=["q"],
        choices=["q", "filter"],
        help="which contrastive setup(s) to run: q (q-transfer, default) and/or filter "
        "(composed-filter). Each has its own config axes + results subdirectory.",
    )
    p.add_argument(
        "--datatypes",
        nargs="+",
        default=None,
        choices=sorted(set(D.QTASKS) | set(D.FILTER_TASKS)),
        help="which datatype tasks to run. Default: all datatypes available for each setup "
        "(q has 4, filter has 3). Requested datatypes not available for a setup are skipped.",
    )
    p.add_argument(
        "--postfix",
        default="targeted",
        choices=["return", "targeted"],
        help="filter-transfer pivot postfix (ignored by q-transfer). "
        "targeted = a filter opener forcing the next token to be a condition.",
    )
    p.add_argument(
        "--namings",
        nargs="+",
        default=None,
        choices=["natural", "letters"],
        help="naming axis. If omitted (default): naming is NOT a config axis — each "
        "contrastive pair gets a deterministic RANDOM naming. If set: split configs "
        "by the given naming(s), as before.",
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
        help="heads carrying the query transplant (also the readout/gaze heads)",
    )
    p.add_argument(
        "--n-conditions",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="task sizes to sample from (per instance, one is chosen at random from this list); "
        "each = #independent commutative predicates in the task",
    )
    p.add_argument(
        "--src-n-prefill",
        type=int,
        default=None,
        help="#conditions prefilled in SRC (default k-1 => SRC target is the LAST condition)",
    )
    p.add_argument(
        "--dst-n-prefill",
        type=int,
        default=1,
        help="#conditions prefilled in DST (default 1 => only the first prefilled). "
        "Requires dst_n_prefill < src_n_prefill < n_conditions.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="how many contrastive instances to RANDOM-sample per config (reproducible, seed-stable): "
        "each = a random n_conditions-predicate task + a random contrastive split",
    )
    p.add_argument(
        "--seed", type=int, default=1123, help="base RNG seed for per-config sampling"
    )
    p.add_argument(
        "--max-instances", type=int, default=None, help="further cap instances/config"
    )
    p.add_argument("--n-steer", type=int, default=32)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--oracle-model", default="claude-sonnet-4-6")
    p.add_argument(
        "--no-oracle", action="store_true", help="skip causal+readout (gaze only)"
    )
    p.add_argument(
        "--force", action="store_true", help="re-run instances even if present"
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false", help="ignore prior results"
    )
    return p.parse_args()


def main():
    args = parse_args()
    # NOTE: --cuda is already applied at import time by _preparse_cuda (must precede the torch
    # import). args.cuda here is only for logging / sanity.
    # force=True: importing src.diff.* (nnsight/baukit/transformers) installs a root StreamHandler
    # at import time, which makes a plain basicConfig() a no-op (root stays at WARNING) and silently
    # swallows our INFO logs. force=True drops that handler and reconfigures at INFO.
    logging_utils.configure(level=logging.INFO, force=True)
    if args.cuda:
        logger.info(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"(requested {args.cuda})"
        )

    ctx = P.setup(args.model)  # loads model + head masks; binds module globals
    mt = ctx.mt
    patch_heads = ctx.goal_heads if args.patch_heads == "goal" else ctx.union_heads
    readout_heads = ctx.goal_heads

    model_short = args.model.split("/")[-1]

    # enumerate configs per setup (each setup owns its axes + results subdir)
    configs = []  # (setup, dt, variant, lt)
    for setup in args.setups:
        for dt in _resolve_datatypes(setup, args.datatypes):
            for variant in _variant_axis(setup, args):
                for lt in args.lang_transfers:
                    configs.append((setup, dt, variant, lt))
    logger.info(
        f"{len(configs)} configs to process across setups={args.setups} "
        f"lang_transfers={args.lang_transfers}"
    )

    metas_by_setup: dict[str, list] = {}
    for ci, (setup, dt, variant, lt) in enumerate(configs, 1):
        sl, dl = LANG_TRANSFERS[lt]
        out_root = (
            Path(env_utils.DEFAULT_RESULTS_DIR)
            / "diff"
            / _setup_dir(setup)
            / model_short
        )
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"═══ config {ci}/{len(configs)}: {_config_tag(setup, dt, variant, sl, dl)} ═══"
        )
        meta = run_config(
            mt, args, setup, dt, variant, sl, dl, patch_heads, readout_heads, out_root
        )
        metas_by_setup.setdefault(setup, []).append(meta)

    # one summary.json per setup subdir
    for setup, metas in metas_by_setup.items():
        summ_fp = (
            Path(env_utils.DEFAULT_RESULTS_DIR)
            / "diff"
            / _setup_dir(setup)
            / model_short
            / "summary.json"
        )
        summ_fp.write_text(json.dumps([dataclasses.asdict(m) for m in metas], indent=2))
        logger.info(f"[{setup}] summary ({len(metas)} configs) at {summ_fp}")
    logger.info(f"all done — {len(configs)} configs across {len(metas_by_setup)} setup(s)")


if __name__ == "__main__":
    main()
