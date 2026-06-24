#!/usr/bin/env python
"""Cross-style activation-patching sweep for the multiline/singleline forced-prefill setup.

Ports the `## Causal Experiment` of `notebooks/clock/04.2_multiline.ipynb` into a batch job.

Pairs come from the SHARED sampler `src.goals.completion_pairs.build_completion_pairs` — the
SAME one `scripts/clock/train_completion_mask.py` and the 04.2 / 04.3 notebooks use — so a fix to
the sampling logic (distinct predicate classes, the `--same`/`--match-*` schemes, the gates)
propagates to every consumer. Each pair is a (SOURCE, DESTINATION) of `StyledContext`s (one spec
rendered in BOTH styles). For each patch leg we transplant hidden states from the SOURCE into the
DESTINATION at the final whitespace token(s) and read the next-token distribution:

    "ml->sl":  source donates its MULTILINE  (stay) signal -> dest run clean as SINGLELINE
    "sl->ml":  source donates its SINGLELINE (moved_on) signal -> dest run clean as MULTILINE

The readout token lives entirely in the DEST's own token space (NEVER the source's): for a leg
with source-donating style `ss`,

    target_ans = dest's argmax in style `ss`   (the dest's OWN flipped-style counterfactual)
    dst_ans    = dest's argmax in its clean run style   (what it emits unpatched)

so `source_ans`, `dst_ans`, `target_ans` can be three mutually-distinct tokens. (An earlier
version drove the IE toward the *source's* argmax, which is only correct under `--same`.)

Varied per pair: direction x patch-tokens ([-1]/[-2]/[-1,-2]) x module/window (residual/mlp/attn)
x layer (`layer_idx`). `--layer-mode` selects which layers are patched per `layer_idx`: `window`
(default), `asc` (idx..n_layer-1) or `desc` (0..idx). Metrics `logit` and `prob` read off the SAME
patched run. The recorded indirect effect, in the dest's token space, is

    IE = (patched - base) / (gold - base)

where `gold` = dest's FLIPPED-style score for `target_ans` (the counterfactual ceiling), `base` =
dest's CLEAN-style score for `target_ans` (the unpatched floor), `patched` = the patched score for
`target_ans`. We also record patched/base scores for `dst_ans` so the notebook can plot the
target_ans-vs-dst_ans crossover.

Pairing follows the shared sampler's gates: distinct predicate classes per example, per-leg token
alignment at every patch position, `target_ans != dst_ans` by construction, and (when not `--same`)
`source_ans != target_ans` so the transplant is genuinely cross-context.

Writes (under results/multiline_patch/<model>/<tag>/):
    checkpoints/pair_XXXX.pt  per-pair checkpoint: that pair's records + BOTH full samples
                              (full_prompt, prefill_code, all build fields, argmax) — written
                              immediately after each pair so a crashed run is resumable.
    records.parquet           one row per (pair, direction, patch_tokens, module, layer, metric)
    pairs.json                per-pair full sample metadata (full_prompt etc.; no tensors)
    metadata.json             run config + n_layer + counts

Resumes by default: existing pair_XXXX.pt are reused (logged `[skip cached]`); pass --overwrite
to recompute. Read back with `load_sweep_results(run_dir)` (defined below) — which also works on
a partial run (no records.parquet yet) by concatenating the checkpoints.

Example
-------
    # cheap smoke test (small model so it fits next to a running 14B notebook)
    conda run -n goals python scripts/clock/sweep_multiline_patch.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --tag smoke \
        --n-pairs 2 --layer-step 8 --modules residual attn

    # full run, minimal pairs (source IS dest)
    conda run -n goals python scripts/clock/sweep_multiline_patch.py \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --tag full \
        --n-pairs 10 --layer-step 1 --modules residual mlp attn --same

    # cross-context pairs, only the sl->ml leg
    conda run -n goals python scripts/clock/sweep_multiline_patch.py \
        --n-pairs 50 --direction sl->ml --match-target-slot
"""

import argparse
import gc
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from transformers import BitsAndBytesConfig


import torch
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.functional import PatchSpec, get_hs  # noqa: E402
from src.goals.completion_pairs import (  # noqa: E402  — SHARED sampler (also used by train_completion_mask.py + 04.2/04.3)
    LEG_STYLES,
    build_completion_pairs,
    legs_for,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.trace import get_window  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)

# Module-name formats keyed by the friendly module name used on the CLI / in records.
MODULE_NAME_FORMATS = {
    "residual": "layer_name_format",
    "mlp": "mlp_module_name_format",
    "attn": "attn_module_name_format",
}
DEFAULT_WINDOWS = {"residual": 3, "mlp": 7, "attn": 7}
METRICS = ("logit", "prob")
_KEY_TS_RE = re.compile(r"_ts(\d+)_")  # CompletionPair.key == f"p{i}_ts{ts}_nc{nc}"


# ──────────────────────────────────────────────────────────────────────────────
# Patching primitives.
# ──────────────────────────────────────────────────────────────────────────────


def score_token(logits: torch.Tensor, token_id: int, metric: str) -> float:
    if metric == "logit":
        return float(logits[token_id])
    return float(torch.softmax(logits.float(), dim=-1)[token_id])


def capture_source_hs(
    mt: ModelandTokenizer,
    src_tok,
    module_formats: list[str],
    layers: list[int],
    positions: list[int],
) -> dict:
    """Cache source latents at every (module, layer, patch-pos).

    One get_hs call PER MODULE: get_hs reads module outputs in the insertion order of
    `locations`, and nnsight (scan=False) requires accesses in ascending execution order.
    Mixing modules in a single call reads layers.N before layers.0.mlp -> OutOfOrderError.
    """
    src_hs: dict = {}
    for fmt in module_formats:
        locs = [(fmt.format(L), t) for L in layers for t in positions]
        src_hs.update(get_hs(mt=mt, input=src_tok, locations=locs, return_dict=True))
    return src_hs


def patched_layer_names(
    layer_name_format: str,
    layer_idx: int,
    layer_mode: str,
    window_size: int,
    n_layer: int,
) -> list[str]:
    """Layer-name list to patch, given the layer-selection mode:

        "window": a window of `window_size` layers centered on `layer_idx`.
        "asc":    every layer from `layer_idx` up to `n_layer - 1` (inclusive).
        "desc":   every layer from 0 up to `layer_idx` (inclusive).
    """
    if layer_mode == "window":
        return list(get_window(layer_name_format, layer_idx, window_size, n_layer))
    if layer_mode == "asc":
        return [layer_name_format.format(L) for L in range(layer_idx, n_layer)]
    if layer_mode == "desc":
        return [layer_name_format.format(L) for L in range(0, layer_idx + 1)]
    raise ValueError(f"unknown layer_mode {layer_mode!r}")


def run_patched(
    mt: ModelandTokenizer,
    clean_input,
    patch_hs: dict,
    layer_idx: int,
    layer_name_format: str,
    logit_location: tuple,
    window_size: int,
    patch_token_indices: list[int],
    layer_mode: str = "window",
) -> tuple[torch.Tensor, int]:
    """Patch `patch_hs` into `clean_input` over the layers selected by `layer_mode` (see
    `patched_layer_names`), return `(next-token logits (vocab,), n_patched_layers)`."""
    layer_names = patched_layer_names(
        layer_name_format, layer_idx, layer_mode, window_size, mt.n_layer
    )
    patches = [
        PatchSpec(
            location=(layer_name, tok_idx),
            patch=patch_hs[(layer_name, tok_idx)],
            strategy="replace",
            on_pass="first",
        )
        for layer_name in layer_names
        for tok_idx in patch_token_indices
    ]
    logits = get_hs(
        mt=mt,
        input=clean_input,
        locations=logit_location,
        patches=patches,
        return_dict=False,
    ).squeeze()
    return logits, len(layer_names)


# ──────────────────────────────────────────────────────────────────────────────
# Pair building.
# ──────────────────────────────────────────────────────────────────────────────


def cache_context(mt, ctx, logit_location: tuple, memo: dict) -> dict:
    """Tokenize + cache BOTH styles of one `StyledContext` -> {style: {prompt, tok, logits,
    argmax}}. Memoized on `id(ctx)` so under `--same` (source IS dest) the shared context is only
    run once. `argmax` is asserted to match the sampler's cached verdict token."""
    if id(ctx) in memo:
        return memo[id(ctx)]
    d: dict = {}
    for style in ("ml", "sl"):
        prompt = ctx.prompt(style)
        tok = prepare_input(prompt, tokenizer=mt)
        logits = get_hs(mt=mt, input=tok, locations=logit_location).float().cpu()
        amax = int(logits.argmax())
        assert amax == ctx.ans(style), "sweep vs sampler argmax mismatch (stale build?)"
        d[style] = {"prompt": prompt, "tok": tok, "logits": logits, "argmax": amax}
    memo[id(ctx)] = d
    return d


def build_pairs(
    mt: ModelandTokenizer,
    *,
    n_pairs: int,
    target_slots: list[int],
    direction: str,
    same: bool,
    match_target_slot: bool,
    match_n_cond: bool,
    max_conditions: int,
    needed_pos: list[int],
    logit_location: tuple,
    prompt_mode: str,
    seed: int,
    max_attempts_factor: int,
) -> list[dict]:
    """Sample `n_pairs` (source, destination) pairs via the SHARED `build_completion_pairs`, then
    cache per-(context, style) tok + unpatched logits for the sweep. Each returned dict is
    `{"key", "source", "dest"}` where source/dest are `cache_context` outputs."""
    cps = build_completion_pairs(
        mt,
        n_pairs=n_pairs,
        target_slots=target_slots,
        direction=direction,
        same=same,
        match_target_slot=match_target_slot,
        match_n_cond=match_n_cond,
        max_conditions=max_conditions,
        mode=prompt_mode,
        seed=seed,
        patch_tokens=tuple(needed_pos),
        max_attempts_factor=max_attempts_factor,
        log=logger.info,
    )
    if len(cps) < n_pairs:
        raise RuntimeError(
            f"only collected {len(cps)}/{n_pairs} pairs (max_attempts_factor={max_attempts_factor}); "
            f"raise --max-attempts-factor / --n-pairs or loosen the pairing flags"
        )

    memo: dict = {}
    pairs: list[dict] = []
    for cp in tqdm(cps, desc="caching pairs"):
        pairs.append(
            {
                "key": cp.key,
                "source": cache_context(mt, cp.source, logit_location, memo),
                "dest": cache_context(mt, cp.dest, logit_location, memo),
            }
        )
    ts_dist = dict(Counter(int(m.group(1)) for p in pairs if (m := _KEY_TS_RE.search(p["key"]))))
    logger.info(
        f"cached {len(pairs)} pairs | unique contexts {len(memo)} "
        f"(same={same} -> source IS dest) | source target_slot dist: {ts_dist}"
    )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Sweep.
# ──────────────────────────────────────────────────────────────────────────────


def sweep_one_pair(
    mt: ModelandTokenizer,
    pi: int,
    pair: dict,
    module_specs: dict[str, tuple[str, int]],
    patch_tok_options: list[tuple[int, ...]],
    sweep_layers: list[int],
    needed_layers: list[int],
    needed_pos: list[int],
    directions: list[str],
    logit_location: tuple,
    ie_denom_eps: dict[str, float],
    layer_mode: str = "window",
) -> list[dict]:
    """Run the full (direction x patch-tokens x module x layer x metric) grid for ONE pair.

    Returns the list of record dicts for this pair (each tagged with `pair=pi`). Kept
    self-contained so the caller can checkpoint per pair.
    """
    module_formats = [fmt for fmt, _ in module_specs.values()]
    records: list[dict] = []
    for direction in directions:
        ss = LEG_STYLES[direction]["src_style"]  # source-donating style == dest's target style
        ds = LEG_STYLES[direction]["dst_style"]  # dest's clean (run) style
        src = pair["source"][ss]                 # donor: provides head/module activations only
        dst_clean = pair["dest"][ds]             # dest, clean style: patched + read out
        dst_flip = pair["dest"][ss]              # dest, flipped style: counterfactual ceiling
        target_ans = dst_flip["argmax"]          # token we push the dest TOWARD (dest's OWN)
        dst_ans = dst_clean["argmax"]            # dest's clean top token
        source_ans = src["argmax"]               # donor's own argmax (informational)
        gold = {m: score_token(dst_flip["logits"], target_ans, m) for m in METRICS}   # ceiling
        base = {m: score_token(dst_clean["logits"], target_ans, m) for m in METRICS}  # floor (unpatched)
        base_dst = {m: score_token(dst_clean["logits"], dst_ans, m) for m in METRICS}
        src_hs = capture_source_hs(
            mt, src["tok"], module_formats, needed_layers, needed_pos
        )
        for patch_toks in patch_tok_options:
            for mod, (fmt, win) in module_specs.items():
                for L in sweep_layers:
                    patched_logits, n_patched = run_patched(
                        mt=mt,
                        clean_input=dst_clean["tok"],
                        patch_hs=src_hs,
                        layer_idx=L,
                        layer_name_format=fmt,
                        logit_location=logit_location,
                        window_size=win,
                        patch_token_indices=list(patch_toks),
                        layer_mode=layer_mode,
                    )
                    patched_logits = patched_logits.float().cpu()
                    for m in METRICS:
                        patched = score_token(patched_logits, target_ans, m)
                        patched_dst = score_token(patched_logits, dst_ans, m)
                        denom = gold[m] - base[m]
                        ie = (
                            (patched - base[m]) / denom
                            if abs(denom) >= ie_denom_eps[m]
                            else float("nan")
                        )
                        records.append(
                            dict(
                                pair=pi,
                                key=pair["key"],
                                direction=direction,
                                patch_tokens=str(list(patch_toks)),
                                module=mod,
                                window=win,
                                layer_mode=layer_mode,
                                n_patched_layers=n_patched,
                                layer=L,
                                metric=m,
                                source_ans=source_ans,  # donor token (informational)
                                dst_ans=dst_ans,         # dest clean argmax
                                target_ans=target_ans,   # dest's OWN flipped-style counterfactual
                                gold=gold[m],            # ceiling: dest-flipped -> target_ans
                                base=base[m],            # floor: dest-clean -> target_ans
                                base_dst=base_dst[m],    # dest-clean -> dst_ans
                                patched=patched,         # patched -> target_ans
                                patched_dst=patched_dst, # patched -> dst_ans
                                ie=ie,
                            )
                        )
        del src_hs
        gc.collect()
        torch.cuda.empty_cache()
    return records


# ──────────────────────────────────────────────────────────────────────────────
# IO + read utilities.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SweepResults:
    """In-memory view of a saved sweep run."""

    records: (
        pd.DataFrame
    )  # one row per (pair, direction, patch_tokens, module, layer, metric)
    pairs: list[dict]  # per-pair full sample metadata (full_prompt etc.; no tensors)
    metadata: dict  # run config + n_layer + counts
    path: Path = field(default=None)


def run_dir(model: str, tag: str) -> Path:
    return (
        Path(env_utils.DEFAULT_RESULTS_DIR)
        / "multiline_patch"
        / model.split("/")[-1]
        / tag
    )


def checkpoint_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "checkpoints"


def checkpoint_path(out_dir: Path, pair_index: int) -> Path:
    return checkpoint_dir(out_dir) / f"pair_{pair_index:04d}.pt"


def serialize_context(ctx_cache: dict, tokenizer) -> dict:
    """Tensor-free view of one cached `StyledContext` (both styles): prompt + decoded argmax."""
    out = {}
    for style in ("ml", "sl"):
        e = ctx_cache[style]
        out[style] = {
            "prompt": e["prompt"],
            "argmax": int(e["argmax"]),
            "argmax_str": tokenizer.decode(int(e["argmax"])),
        }
    return out


def serialize_pair(pi: int, pair: dict, tokenizer) -> dict:
    """Full, tensor-free view of a (source, dest) pair, suitable for JSON / checkpoint."""
    return {
        "pair_index": pi,
        "key": pair["key"],
        "source": serialize_context(pair["source"], tokenizer),
        "dest": serialize_context(pair["dest"], tokenizer),
    }


def _load_checkpoints(ckpt_dir: Path) -> tuple[list[dict], list[dict]]:
    """Read every per-pair checkpoint -> (all records, per-pair sample metadata)."""
    records: list[dict] = []
    pairs_meta: list[dict] = []
    for f in sorted(ckpt_dir.glob("pair_*.pt")):
        ck = torch.load(f, weights_only=False)
        records.extend(ck["records"])
        pairs_meta.append({k: ck[k] for k in ("pair_index", "key", "source", "dest")})
    return records, pairs_meta


def load_sweep_results(path: str | Path) -> SweepResults:
    """Load a saved run. `path` is the run directory (…/multiline_patch/<model>/<tag>/).

    Prefers the finalized `records.parquet` / `pairs.json`; if those are absent (a partial /
    in-progress run) it falls back to concatenating the per-pair `checkpoints/pair_*.pt`, so an
    interrupted sweep is still inspectable.
    """
    path = Path(path)
    ckpt_dir = checkpoint_dir(path)
    has_parquet = (path / "records.parquet").exists()
    has_pairs = (path / "pairs.json").exists()

    ck_records = ck_pairs = None
    if (not has_parquet or not has_pairs) and ckpt_dir.exists():
        ck_records, ck_pairs = _load_checkpoints(ckpt_dir)

    if has_parquet:
        records = pd.read_parquet(path / "records.parquet")
    elif ck_records:
        records = pd.DataFrame(ck_records)
    else:
        raise FileNotFoundError(f"no records.parquet or checkpoints under {path}")

    if has_pairs:
        with open(path / "pairs.json") as f:
            pairs = json.load(f)
    else:
        pairs = ck_pairs or []

    metadata = {}
    if (path / "metadata.json").exists():
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
    return SweepResults(records=records, pairs=pairs, metadata=metadata, path=path)


def aggregate_sweep(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across pairs -> per (metric, module, direction, patch_tokens, layer) means/sems.

    Mirrors the notebook's `agg` (IE + src/dst patched scores + unpatched baselines), so the
    notebook viz cells can consume it directly.
    """
    import numpy as np

    def _sem(x):
        x = x.dropna()
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

    return (
        records.groupby(["metric", "module", "direction", "patch_tokens", "layer"])
        .agg(
            ie_mean=("ie", "mean"),
            ie_sem=("ie", _sem),
            n=("ie", "count"),
            src_mean=("patched", "mean"),
            src_sem=("patched", _sem),
            dst_mean=("patched_dst", "mean"),
            dst_sem=("patched_dst", _sem),
            base_src=("base", "mean"),
            base_dst=("base_dst", "mean"),
        )
        .reset_index()
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────────────


def _parse_patch_tokens(specs: list[str]) -> list[tuple[int, ...]]:
    out = []
    for s in specs:
        out.append(tuple(int(x) for x in s.split(",")))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    p.add_argument(
        "--tag", default=None, help="run subdir name (default: auto from config)"
    )
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--target-slots", nargs="+", type=int, default=[1, 2, 3, 4],
                   help="target slot choices (each in {1,2,3,4}: need a prior + a next var)")
    p.add_argument("--pair-seed", type=int, default=12345)
    p.add_argument("--max-attempts-factor", type=int, default=50,
                   help="pair-sampling attempt cap = factor * n_pairs (shared sampler default 50)")
    # ── pairing scheme (mirrors scripts/clock/train_completion_mask.py / completion_pairs.py) ──
    p.add_argument("--direction", default="both", choices=["ml->sl", "sl->ml", "both"],
                   help="which patch leg(s) to sweep")
    p.add_argument("--same", action="store_true",
                   help="source IS dest (minimal pair; source_ans == target_ans); ignores --match-* flags")
    p.add_argument("--match-target-slot", action="store_true",
                   help="force matched target_slot on both sides (ignored if --same)")
    p.add_argument("--match-n-cond", action="store_true",
                   help="force matched n_conditions on both sides (ignored if --same)")
    p.add_argument("--max-conditions", type=int, default=6, help="upper bound when sampling n_conditions")
    p.add_argument("--prompt-mode", choices=["chat", "fim"], default="chat")
    p.add_argument(
        "--layer-step", type=int, default=1, help="sweep range(0, n_layer, step)"
    )
    p.add_argument(
        "--layer-mode",
        choices=["window", "asc", "desc"],
        default="window",
        help="layers patched per layer_idx: window (around idx, uses --*-window) | "
        "asc (idx..n_layer-1) | desc (0..idx)",
    )
    p.add_argument(
        "--modules",
        nargs="+",
        choices=list(MODULE_NAME_FORMATS),
        default=["residual", "mlp", "attn"],
    )
    p.add_argument("--residual-window", type=int, default=DEFAULT_WINDOWS["residual"])
    p.add_argument("--mlp-window", type=int, default=DEFAULT_WINDOWS["mlp"])
    p.add_argument("--attn-window", type=int, default=DEFAULT_WINDOWS["attn"])
    p.add_argument(
        "--patch-tokens",
        nargs="+",
        default=["-1", "-2", "-2,-1"],
        help="patch position sets, comma-joined (e.g. -1 -2 -2,-1)",
    )
    p.add_argument("--ie-eps-logit", type=float, default=0.5)
    p.add_argument("--ie-eps-prob", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    pair_tag = ("same" if args.same else "diff") + (
        "" if args.same else
        f"{'_mts' if args.match_target_slot else ''}{'_mnc' if args.match_n_cond else ''}"
    )
    tag = args.tag or (
        f"{args.prompt_mode}_{args.layer_mode}_{pair_tag}_{args.direction.replace('->', '2')}"
        f"_n{args.n_pairs}_step{args.layer_step}_seed{args.pair_seed}"
    )
    out_dir = run_dir(args.model, tag)
    ckpt_dir = checkpoint_dir(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for f in ckpt_dir.glob("pair_*.pt"):
            f.unlink()
        logger.info("--overwrite: cleared existing per-pair checkpoints")
    logger.info(f"output dir: {out_dir}  (per-pair checkpoints under {ckpt_dir.name}/)")
    quant_args = {}
    if args.quantize:
        quant_args = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        logger.info("quantization enabled: using 8-bit weights (no quantization for activations)")

    logger.info(f"loading model {args.model} ...")
    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        **quant_args
    )
    logger.info(f"n_layer={mt.n_layer}, n_heads={mt.model.config.num_attention_heads}")

    logit_location = (mt.lm_head_name, -1)
    windows = {
        "residual": args.residual_window,
        "mlp": args.mlp_window,
        "attn": args.attn_window,
    }
    module_specs = {
        m: (getattr(mt, MODULE_NAME_FORMATS[m]), windows[m]) for m in args.modules
    }
    patch_tok_options = _parse_patch_tokens(args.patch_tokens)
    needed_pos = sorted({t for pt in patch_tok_options for t in pt})
    needed_layers = list(range(mt.n_layer))
    sweep_layers = list(range(0, mt.n_layer, args.layer_step))
    ie_denom_eps = {"logit": args.ie_eps_logit, "prob": args.ie_eps_prob}
    directions = legs_for(args.direction)

    n_runs = (
        args.n_pairs
        * len(directions)
        * len(patch_tok_options)
        * len(module_specs)
        * len(sweep_layers)
    )
    logger.info(
        f"pairing: same={args.same} match_ts={args.match_target_slot} match_nc={args.match_n_cond} "
        f"max_cond={args.max_conditions} | direction={args.direction} -> legs={directions}"
    )
    logger.info(
        f"layer_mode={args.layer_mode} | "
        f"modules={ {m: w for m, (_, w) in module_specs.items()} } | "
        f"patch_tokens={patch_tok_options} | sweep_layers({len(sweep_layers)})={sweep_layers}"
    )
    logger.info(f"=> {n_runs} patched forward passes")

    pairs = build_pairs(
        mt=mt,
        n_pairs=args.n_pairs,
        target_slots=args.target_slots,
        direction=args.direction,
        same=args.same,
        match_target_slot=args.match_target_slot,
        match_n_cond=args.match_n_cond,
        max_conditions=args.max_conditions,
        needed_pos=needed_pos,
        logit_location=logit_location,
        prompt_mode=args.prompt_mode,
        seed=args.pair_seed,
        max_attempts_factor=args.max_attempts_factor,
    )

    # Per-pair sweep with checkpointing. Each pair writes a self-contained
    # checkpoints/pair_XXXX.pt (its records + both full samples); existing checkpoints are
    # reused (resume) unless --overwrite. Pair building is deterministic from --pair-seed, so
    # the i-th pair is identical across runs and resuming is safe.
    all_records: list[dict] = []
    n_cached = n_swept = 0
    for pi, pair in enumerate(tqdm(pairs, desc="pairs")):
        ckpt = checkpoint_path(out_dir, pi)
        if ckpt.exists() and not args.overwrite:
            ck = torch.load(ckpt, weights_only=False)
            all_records.extend(ck["records"])
            n_cached += 1
            logger.info(
                f"[{pi + 1}/{len(pairs)}] [skip cached] {ckpt.name} ({len(ck['records'])} rows)"
            )
            continue
        records = sweep_one_pair(
            mt=mt,
            pi=pi,
            pair=pair,
            module_specs=module_specs,
            patch_tok_options=patch_tok_options,
            sweep_layers=sweep_layers,
            needed_layers=needed_layers,
            needed_pos=needed_pos,
            directions=directions,
            logit_location=logit_location,
            ie_denom_eps=ie_denom_eps,
            layer_mode=args.layer_mode,
        )
        payload = serialize_pair(pi, pair, mt.tokenizer)
        payload["records"] = records
        torch.save(payload, ckpt)  # checkpoint immediately after the pair completes
        all_records.extend(records)
        n_swept += 1
        logger.info(
            f"[{pi + 1}/{len(pairs)}] swept + saved {ckpt.name} ({len(records)} rows)"
        )

    df = pd.DataFrame(all_records)
    df.to_parquet(out_dir / "records.parquet")
    with open(out_dir / "pairs.json", "w") as f:
        json.dump(
            [serialize_pair(pi, p, mt.tokenizer) for pi, p in enumerate(pairs)],
            f,
            indent=2,
        )
    metadata = {
        "model": args.model,
        "tag": tag,
        "n_pairs": args.n_pairs,
        "target_slots": args.target_slots,
        "pair_seed": args.pair_seed,
        "prompt_mode": args.prompt_mode,
        "same": args.same,
        "match_target_slot": args.match_target_slot,
        "match_n_cond": args.match_n_cond,
        "max_conditions": args.max_conditions,
        "n_layer": int(mt.n_layer),
        "layer_step": args.layer_step,
        "layer_mode": args.layer_mode,
        "sweep_layers": sweep_layers,
        "modules": {m: w for m, (_, w) in module_specs.items()},
        "patch_tokens": [list(pt) for pt in patch_tok_options],
        "metrics": list(METRICS),
        "directions": directions,
        "ie_denom_eps": ie_denom_eps,
        "n_records": int(len(df)),
        "n_valid_ie": int(df["ie"].notna().sum()),
        "frac_ie_dropped": float(df["ie"].isna().mean()),
        "n_pairs_swept": n_swept,
        "n_pairs_cached": n_cached,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"saved {len(df)} rows -> {out_dir / 'records.parquet'} "
        f"({metadata['n_valid_ie']} non-NaN IE, {metadata['frac_ie_dropped']:.1%} dropped) "
        f"| {n_swept} pairs swept, {n_cached} resumed from checkpoint"
    )


if __name__ == "__main__":
    main()
