#!/usr/bin/env python
"""Batch q-transfer intervention + eval over the DIFF data configs.

For every contrastive q-transfer instance (``src.diff.data.q_transfer_setups``) this runs the
query transplant (``src.diff.functional.run_q_transfer_intervention``) and the three evals
(``src.diff.eval.eval_q_transfer``): causal (oracle), readout-shift (oracle), gaze (rule-based).
Both the RAW intervention artifacts (prompts, generations, readouts, attention) and the eval
verdicts are saved.

Configs are the cross-product of the requested axes:  datatype × naming × lang-transfer
(presentation is fixed to ``listed``). Each config's results live in its own folder under
``results/diff/q_transfer/<model_short>/<config_tag>/`` with ``results.json`` (per-instance
records, written after each instance so a crash is resumable) and ``meta.json`` (config + summary
rates). ``--resume`` (default) skips instances already present; ``--force`` re-runs them.

Typical usage
─────────────
    conda run -n goals python scripts/diff/q_transfer_eval.py \\
        --datatypes list[str] list[int] --namings natural letters \\
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

import src.diff.data as D  # noqa: E402
import src.diff.eval as E  # noqa: E402
import src.diff.functional as P  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402

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
    naming: str
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


@dataclass
class ConfigMeta:
    model: str
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
    gaze_shift_rate: Optional[float] = None


def _safe(dt: str) -> str:
    return dt.replace("[", "_").replace("]", "").replace(" ", "")


def _config_tag(dt, naming, sl, dl) -> str:
    return f"{_safe(dt)}__{naming or 'randnm'}__{sl}-{dl}"


def _pick_naming(tag: str, key: str) -> str:
    """Deterministic per-pair naming when the naming axis is collapsed (no --namings). Stable across
    runs (md5, not the salted built-in hash) so resume assigns the same naming to the same pair."""
    h = int(hashlib.md5(f"{tag}|{key}".encode()).hexdigest()[:8], 16)
    return "letters" if h % 2 else "natural"


def _inst_key(inst) -> str:
    return (
        f"{inst.expected_next}|src={'-'.join(inst.src_progress)}"
        f"|dst={'-'.join(inst.dst_progress)}"
    )


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
    return dict(
        causal_trigger_rate=causal,
        readout_valid_rate=r_valid,
        readout_shift_rate=r_shift,
        readout_shift_rate_valid=r_shift_valid,
        gaze_shift_rate=gaze,
    )


# ══════════════════════════════════════════════════════════════════════════════
# per-config run
# ══════════════════════════════════════════════════════════════════════════════
def run_config(mt, args, dt, naming, sl, dl, patch_heads, readout_heads, out_root):
    tag = _config_tag(dt, naming, sl, dl)
    outdir = out_root / tag
    outdir.mkdir(parents=True, exist_ok=True)
    results_fp = outdir / "results.json"

    # load prior results for resume
    done: dict[str, dict] = {}
    if results_fp.exists() and args.resume and not args.force:
        done = {r["key"]: r for r in json.loads(results_fp.read_text())}
        logger.info(f"[{tag}] resume: {len(done)} instances already done")

    # naming=None => collapse the axis: pick a naming per pair (deterministic). Generate with a
    # placeholder naming, then override each instance's naming field below.
    random_naming = naming is None
    setups = D.q_transfer_setups(
        dt,
        presentation="listed",
        naming=naming or "natural",
        src_lang=sl,
        dst_lang=dl,
        max_setups=args.max_setups,
    )
    if random_naming:
        for inst in setups:
            inst.naming = _pick_naming(tag, _inst_key(inst))
    if args.max_instances:
        setups = setups[: args.max_instances]

    records: dict[str, dict] = dict(done)
    todo = [inst for inst in setups if _inst_key(inst) not in records or args.force]
    n_total = len(setups)
    logger.info(
        f"[{tag}] {n_total} contrastive pairs — {n_total - len(todo)} resumed, "
        f"{len(todo)} to run"
    )

    for j, inst in enumerate(todo, 1):
        key = _inst_key(inst)
        # contrastive concept pair: SRC frontier (transplant target) -> DST frontier (clean)
        logger.info(
            f"[{tag}] ({j}/{len(todo)}) [{inst.naming}] "
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
                attn_delta=raw["attn_delta"],
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
            f"{'shift' if gaze_res['shifted'] else 'no'}(m={gaze_res['margin']:+.3f})"
            if gaze_res
            else "—"
        )
        logger.info(f"[{tag}]   => causal={_c} · readout={_r} · gaze={_g}")

        rec = InstanceRecord(
            key=key,
            datatype=dt,
            naming=inst.naming,
            presentation="listed",
            src_lang=sl,
            dst_lang=dl,
            var_name=inst.var_name,
            expected_next=inst.expected_next,
            expected_next_desc=inst.expected_next_desc,
            dst_next=inst.meta["dst_next"],
            dst_next_desc=inst.dst_next_desc,
            src_progress=inst.src_progress,
            dst_progress=inst.dst_progress,
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
        datatype=dt,
        naming=naming or "random",
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
        f"gaze={meta.gaze_shift_rate}"
    )
    return meta


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES (e.g. '0,7')")
    p.add_argument(
        "--datatypes",
        nargs="+",
        default=list(D.QTASKS),
        choices=list(D.QTASKS),
        help="which datatype tasks to run",
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
        "--max-setups", type=int, default=4, help="cap contrastive pairs per config"
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
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging_utils.configure(level=logging.INFO)

    ctx = P.setup(args.model)  # loads model + head masks; binds module globals
    mt = ctx.mt
    patch_heads = ctx.goal_heads if args.patch_heads == "goal" else ctx.union_heads
    readout_heads = ctx.goal_heads

    out_root = (
        Path(env_utils.DEFAULT_RESULTS_DIR)
        / "diff"
        / "q_transfer"
        / args.model.split("/")[-1]
    )
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"writing to {out_root}")

    namings_axis = (
        args.namings if args.namings else [None]
    )  # [None] => random naming per pair
    configs = [
        (dt, naming, lt)
        for dt in args.datatypes
        for naming in namings_axis
        for lt in args.lang_transfers
    ]
    logger.info(
        f"{len(configs)} configs to process: datatypes={args.datatypes} "
        f"namings={args.namings or 'random-per-pair'} lang_transfers={args.lang_transfers}"
    )

    metas = []
    for ci, (dt, naming, lt) in enumerate(configs, 1):
        sl, dl = LANG_TRANSFERS[lt]
        logger.info(
            f"═══ config {ci}/{len(configs)}: {_config_tag(dt, naming, sl, dl)} ═══"
        )
        metas.append(
            run_config(
                mt, args, dt, naming, sl, dl, patch_heads, readout_heads, out_root
            )
        )

    (out_root / "summary.json").write_text(
        json.dumps([dataclasses.asdict(m) for m in metas], indent=2)
    )
    logger.info(
        f"all done — {len(metas)} configs; summary at {out_root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
