#!/usr/bin/env python
"""Cross-style activation-patching sweep for the multiline/singleline forced-prefill setup.

Ports the `## Causal Experiment` of `notebooks/clock/04.2_multiline.ipynb` into a batch job.

For each of N contrastive pairs — one *multiline* (incomplete target line -> hypothesis
"continue") and one *singleline* (complete target line -> hypothesis "moved_on") example built
with the SAME `target_slot` — we patch hidden states from a SOURCE prompt into a DESTINATION
prompt at the final whitespace token(s) and read the next-token distribution. Both patch
DIRECTIONS reuse the same two prompts (swap source<->dest):

    "ml->sl":  source = multiline,  destination = singleline
    "sl->ml":  source = singleline, destination = multiline

Varied per pair: direction (2) x patch-tokens ([-1]/[-2]/[-1,-2]) x module/window
(residual/mlp/attn) x layer (window center). Metrics `logit` and `prob` are read off the SAME
patched run. The recorded indirect effect is

    IE = (patched - base) / (gold - base)

where `gold` = source's own score for `src_ans` (its argmax), `base` = destination's UNPATCHED
score for `src_ans`. We also record patched/base scores for `dst_ans` (destination's own argmax)
so the notebook can plot the src_ans-vs-dst_ans crossover.

Pairs are kept only if (a) src & dst share the token at every patch position (token-aligned
replace-patch) and (b) their next-token argmax differs (a genuine contrast for the IE).

Writes (under results/multiline_patch/<model>/<tag>/):
    metadata.json   run config + n_layer + counts
    records.parquet one row per (pair, direction, patch_tokens, module, layer, metric)
    pairs.json      per-pair prompts + target tokens (no tensors), for inspection

Read back with `load_sweep_results(run_dir)` (defined below).

Example
-------
    # cheap smoke test (small model so it fits next to a running 14B notebook)
    conda run -n goals python scripts/clock/sweep_multiline_patch.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --tag smoke \
        --n-pairs 2 --layer-step 8 --modules residual attn

    # full run
    conda run -n goals python scripts/clock/sweep_multiline_patch.py \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --tag full \
        --n-pairs 10 --layer-step 1 --modules residual mlp attn
"""

import argparse
import gc
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.functional import PatchSpec, get_hs  # noqa: E402
from src.goals.condition_data import (  # noqa: E402
    INT_PREDICATES,
    ConditionSample,
    ConditionSpec,
)
from src.goals.functional import apply_chat_template  # noqa: E402
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

# Prompt-building constants (ported from the notebook).
MULTILINE_PREDS = [p for p in INT_PREDICATES if p.is_multiline]
VAR_LETTERS = "abcdefghij"
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction (ported verbatim from notebooks/clock/04.2_multiline.ipynb).
# ──────────────────────────────────────────────────────────────────────────────


def result_print(letters: list[str]) -> str:
    """FIM suffix: an f-string print referencing EVERY result var."""
    conds = ", ".join("Condition %d: {%s}" % (i + 1, v) for i, v in enumerate(letters))
    all_expr = " and ".join(letters)
    return '    print(f"' + conds + ", ALL: {" + all_expr + '}")\n'


def replace_in_code_newline_with_tabnewline(prompt: str) -> str:
    """Turn every code-region `\\n` into `    \\n` (4 spaces + newline)."""
    code_split = prompt.split("<code>")
    code_segment = code_split[-1]
    prefix = "<code>".join(code_split[:-1])
    code_segment = code_segment.replace("\n", "    \n")
    return prefix + "<code>" + code_segment


def build_example(
    mt: ModelandTokenizer,
    rng: random.Random,
    target_slot: int,
    n_conditions: int,
    mode: str = "chat",
    style: str = "multiline",
    aug: Optional[Callable] = None,
) -> dict:
    """Forced-prefill example for the 'stay vs move-on' test (see notebook docstring)."""
    assert 1 <= target_slot < n_conditions
    preds = [rng.choice(MULTILINE_PREDS) for _ in range(n_conditions)]
    params = [p.sample_params(rng) for p in preds]
    spec = ConditionSpec(
        spec_id=f"ml_s{target_slot}_{style[:3]}_{rng.randrange(10 ** 6):06d}",
        predicate_ids=[p.id for p in preds],
        predicate_params=params,
        spec_texts=[p.render(pr) for p, pr in zip(preds, params)],
        id_substrs=[p.render_substrs(pr) for p, pr in zip(preds, params)],
        test_inputs=[rng.randint(-5, 1000) for _ in range(6)],
    )
    sample = ConditionSample(
        spec=spec, lang="python", modularity="monolithic", without_comment=True
    )
    stub = sample.render_code_stub()

    # priors: force-filled COMPLETE, random single/multi mix, but >=1 multiline.
    prior_styles = [rng.choice(["multiline", "singleline"]) for _ in range(target_slot)]
    if "multiline" not in prior_styles:
        prior_styles[rng.randrange(target_slot)] = "multiline"
    body: list[str] = []
    for j in range(target_slot):
        if prior_styles[j] == "multiline":
            body += preds[j].render_multiline(params[j], VAR_LETTERS[j])
        else:
            body.append(preds[j].render_singleline(params[j], VAR_LETTERS[j]))

    # target: the test variable.
    if style == "multiline":
        target_impl = preds[target_slot].render_multiline(params[target_slot], VAR_LETTERS[target_slot])
        target_first = target_impl[0]
        expected_rest = "\n".join(target_impl[1:])
        expected_verdict = "continue"
    else:
        target_first = preds[target_slot].render_singleline(params[target_slot], VAR_LETTERS[target_slot])
        expected_rest = ""
        expected_verdict = "moved_on"
    body.append(target_first)

    letters = [VAR_LETTERS[i] for i in range(n_conditions)]
    fim_suffix = result_print(letters)
    if mode == "fim":
        doc = (
            '    """\n'
            + "".join(f"    {i + 1}. {t}\n" for i, t in enumerate(spec.spec_texts))
            + '    """\n'
        )
        prefix_code = stub + doc + "\n".join(body) + "\n"
        full_prompt = FIM_PREFIX + prefix_code + FIM_SUFFIX + fim_suffix + FIM_MIDDLE
    else:
        prefix_code = stub + "\n".join(body) + "\n"
        full_prompt = apply_chat_template(
            mt=mt, user_query=sample.render_instruction().strip(), incomplete_code=prefix_code
        )

    full_prompt = aug(full_prompt) if aug else full_prompt
    next_slot = target_slot + 1
    return {
        "spec_id": spec.spec_id,
        "mode": mode,
        "prefill_style": style,
        "prior_styles": prior_styles,
        "full_prompt": full_prompt,
        "prefill_code": prefix_code,
        "fim_suffix": fim_suffix.rstrip("\n") if mode == "fim" else None,
        "target_slot": target_slot,
        "target_var": VAR_LETTERS[target_slot],
        "next_var": VAR_LETTERS[next_slot] if next_slot < n_conditions else None,
        "target_pred": preds[target_slot].id,
        "target_first_line": target_first.strip(),
        "expected_rest": expected_rest,
        "expected_verdict": expected_verdict,
        "n_conditions": n_conditions,
    }


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


def run_patched(
    mt: ModelandTokenizer,
    clean_input,
    patch_hs: dict,
    layer_idx: int,
    layer_name_format: str,
    logit_location: tuple,
    window_size: int,
    patch_token_indices: list[int],
) -> torch.Tensor:
    """Patch `patch_hs` into `clean_input` over a window of `layer_name_format` centered at
    `layer_idx`, return next-token logits (vocab,)."""
    window = get_window(layer_name_format, layer_idx, window_size, mt.n_layer)
    patches = [
        PatchSpec(
            location=(layer_name, tok_idx),
            patch=patch_hs[(layer_name, tok_idx)],
            strategy="replace",
            on_pass="first",
        )
        for layer_name in window
        for tok_idx in patch_token_indices
    ]
    return get_hs(
        mt=mt, input=clean_input, locations=logit_location, patches=patches, return_dict=False
    ).squeeze()


# ──────────────────────────────────────────────────────────────────────────────
# Pair building.
# ──────────────────────────────────────────────────────────────────────────────


def build_pairs(
    mt: ModelandTokenizer,
    n_pairs: int,
    target_slot_options: list[int],
    needed_pos: list[int],
    logit_location: tuple,
    prompt_mode: str,
    seed: int,
    max_attempts: int,
) -> list[dict]:
    """Sample & validate `n_pairs` contrastive (multiline, singleline) pairs."""

    def prep(ex: dict) -> dict:
        ex["prompt"] = replace_in_code_newline_with_tabnewline(ex["full_prompt"]) + "   "
        ex["tok"] = prepare_input(ex["prompt"], tokenizer=mt)
        ex["logits"] = get_hs(mt=mt, input=ex["tok"], locations=logit_location).float().cpu()
        ex["argmax"] = int(ex["logits"].argmax())
        return ex

    def tokens_match(a: dict, b: dict) -> bool:
        for t in needed_pos:
            ta = mt.tokenizer.decode(a["tok"]["input_ids"][0, t].item())
            tb = mt.tokenizer.decode(b["tok"]["input_ids"][0, t].item())
            if ta != tb:
                return False
        return True

    prng = random.Random(seed)
    pairs: list[dict] = []
    attempts = 0
    pbar = tqdm(total=n_pairs, desc="building pairs")
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        target_slot = prng.choice(target_slot_options)
        n_conditions = target_slot + 2  # >= target_slot, gives a 'next var' for moved_on
        ml = prep(build_example(mt, prng, target_slot, n_conditions, prompt_mode, "multiline"))
        sl = prep(build_example(mt, prng, target_slot, n_conditions, prompt_mode, "singleline"))
        if not tokens_match(ml, sl):
            continue
        if ml["argmax"] == sl["argmax"]:  # not contrastive at the next token
            continue
        pairs.append({"target_slot": target_slot, "n_conditions": n_conditions, "ml": ml, "sl": sl})
        pbar.update(1)
    pbar.close()
    logger.info(
        f"collected {len(pairs)}/{n_pairs} valid pairs in {attempts} attempts | "
        f"target_slot dist: {dict(Counter(p['target_slot'] for p in pairs))}"
    )
    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"only collected {len(pairs)}/{n_pairs} pairs in {max_attempts} attempts; "
            f"raise --max-attempts or loosen the filter"
        )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Sweep.
# ──────────────────────────────────────────────────────────────────────────────


def run_sweep(
    mt: ModelandTokenizer,
    pairs: list[dict],
    module_specs: dict[str, tuple[str, int]],
    patch_tok_options: list[tuple[int, ...]],
    sweep_layers: list[int],
    needed_layers: list[int],
    needed_pos: list[int],
    directions: list[str],
    logit_location: tuple,
    ie_denom_eps: dict[str, float],
) -> list[dict]:
    module_formats = [fmt for fmt, _ in module_specs.values()]
    records: list[dict] = []
    for pi, pair in enumerate(tqdm(pairs, desc="pairs")):
        for direction in directions:
            src, dst = (
                (pair["ml"], pair["sl"]) if direction == "ml->sl" else (pair["sl"], pair["ml"])
            )
            src_ans = src["argmax"]  # token we patch TOWARD
            dst_ans = dst["argmax"]  # dest's own top token
            gold = {m: score_token(src["logits"], src_ans, m) for m in METRICS}
            base = {m: score_token(dst["logits"], src_ans, m) for m in METRICS}
            base_dst = {m: score_token(dst["logits"], dst_ans, m) for m in METRICS}
            src_hs = capture_source_hs(mt, src["tok"], module_formats, needed_layers, needed_pos)
            for patch_toks in patch_tok_options:
                for mod, (fmt, win) in module_specs.items():
                    for L in sweep_layers:
                        patched_logits = run_patched(
                            mt=mt,
                            clean_input=dst["tok"],
                            patch_hs=src_hs,
                            layer_idx=L,
                            layer_name_format=fmt,
                            logit_location=logit_location,
                            window_size=win,
                            patch_token_indices=list(patch_toks),
                        ).float().cpu()
                        for m in METRICS:
                            patched = score_token(patched_logits, src_ans, m)
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
                                    target_slot=pair["target_slot"],
                                    direction=direction,
                                    patch_tokens=str(list(patch_toks)),
                                    module=mod,
                                    window=win,
                                    layer=L,
                                    metric=m,
                                    src_ans=src_ans,
                                    dst_ans=dst_ans,
                                    gold=gold[m],
                                    base=base[m],
                                    base_dst=base_dst[m],
                                    patched=patched,
                                    patched_dst=patched_dst,
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

    records: pd.DataFrame          # one row per (pair, direction, patch_tokens, module, layer, metric)
    pairs: list[dict]              # per-pair prompts + target tokens (no tensors)
    metadata: dict                 # run config + n_layer + counts
    path: Path = field(default=None)


def run_dir(model: str, tag: str) -> Path:
    return Path(env_utils.DEFAULT_RESULTS_DIR) / "multiline_patch" / model.split("/")[-1] / tag


def load_sweep_results(path: str | Path) -> SweepResults:
    """Load a saved run. `path` is the run directory (…/multiline_patch/<model>/<tag>/)."""
    path = Path(path)
    records = pd.read_parquet(path / "records.parquet")
    with open(path / "pairs.json") as f:
        pairs = json.load(f)
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
            ie_mean=("ie", "mean"), ie_sem=("ie", _sem), n=("ie", "count"),
            src_mean=("patched", "mean"), src_sem=("patched", _sem),
            dst_mean=("patched_dst", "mean"), dst_sem=("patched_dst", _sem),
            base_src=("base", "mean"), base_dst=("base_dst", "mean"),
        )
        .reset_index()
    )


def _serialize_pairs(pairs: list[dict], tokenizer) -> list[dict]:
    """Strip tensors from the in-memory pairs so they're JSON-serializable / inspectable."""
    keep = (
        "spec_id", "prefill_style", "prior_styles", "prompt", "target_slot", "target_var",
        "next_var", "target_pred", "target_first_line", "expected_verdict", "n_conditions",
    )
    out = []
    for p in pairs:
        rec = {"target_slot": p["target_slot"], "n_conditions": p["n_conditions"]}
        for role in ("ml", "sl"):
            ex = p[role]
            rec[role] = {k: ex[k] for k in keep if k in ex}
            rec[role]["argmax"] = ex["argmax"]
            rec[role]["argmax_str"] = tokenizer.decode(int(ex["argmax"]))
        out.append(rec)
    return out


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
    p.add_argument("--tag", default=None, help="run subdir name (default: auto from config)")
    p.add_argument("--n-pairs", type=int, default=10)
    p.add_argument("--target-slots", nargs="+", type=int, default=[1, 2])
    p.add_argument("--pair-seed", type=int, default=12345)
    p.add_argument("--max-attempts", type=int, default=None,
                   help="pair-sampling attempt cap (default: 40 * n_pairs)")
    p.add_argument("--prompt-mode", choices=["chat", "fim"], default="chat")
    p.add_argument("--layer-step", type=int, default=1, help="sweep range(0, n_layer, step)")
    p.add_argument("--modules", nargs="+", choices=list(MODULE_NAME_FORMATS),
                   default=["residual", "mlp", "attn"])
    p.add_argument("--residual-window", type=int, default=DEFAULT_WINDOWS["residual"])
    p.add_argument("--mlp-window", type=int, default=DEFAULT_WINDOWS["mlp"])
    p.add_argument("--attn-window", type=int, default=DEFAULT_WINDOWS["attn"])
    p.add_argument("--patch-tokens", nargs="+", default=["-1", "-2", "-2,-1"],
                   help="patch position sets, comma-joined (e.g. -1 -2 -2,-1)")
    p.add_argument("--ie-eps-logit", type=float, default=0.5)
    p.add_argument("--ie-eps-prob", type=float, default=1e-3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    tag = args.tag or (
        f"n{args.n_pairs}_step{args.layer_step}_{args.prompt_mode}_seed{args.pair_seed}"
    )
    out_dir = run_dir(args.model, tag)
    if out_dir.exists() and not args.overwrite:
        if (out_dir / "records.parquet").exists():
            raise FileExistsError(f"{out_dir} already has results — pass --overwrite or a new --tag")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output dir: {out_dir}")

    logger.info(f"loading model {args.model} ...")
    mt = ModelandTokenizer(
        model_key=args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    logger.info(f"n_layer={mt.n_layer}, n_heads={mt.model.config.num_attention_heads}")

    logit_location = (mt.lm_head_name, -1)
    windows = {
        "residual": args.residual_window, "mlp": args.mlp_window, "attn": args.attn_window,
    }
    module_specs = {
        m: (getattr(mt, MODULE_NAME_FORMATS[m]), windows[m]) for m in args.modules
    }
    patch_tok_options = _parse_patch_tokens(args.patch_tokens)
    needed_pos = sorted({t for pt in patch_tok_options for t in pt})
    needed_layers = list(range(mt.n_layer))
    sweep_layers = list(range(0, mt.n_layer, args.layer_step))
    ie_denom_eps = {"logit": args.ie_eps_logit, "prob": args.ie_eps_prob}
    max_attempts = args.max_attempts or (40 * args.n_pairs)

    n_runs = (
        args.n_pairs * 2 * len(patch_tok_options) * len(module_specs) * len(sweep_layers)
    )
    logger.info(
        f"modules={ {m: w for m, (_, w) in module_specs.items()} } | "
        f"patch_tokens={patch_tok_options} | sweep_layers({len(sweep_layers)})={sweep_layers}"
    )
    logger.info(f"=> {n_runs} patched forward passes")

    pairs = build_pairs(
        mt=mt, n_pairs=args.n_pairs, target_slot_options=args.target_slots,
        needed_pos=needed_pos, logit_location=logit_location, prompt_mode=args.prompt_mode,
        seed=args.pair_seed, max_attempts=max_attempts,
    )

    records = run_sweep(
        mt=mt, pairs=pairs, module_specs=module_specs, patch_tok_options=patch_tok_options,
        sweep_layers=sweep_layers, needed_layers=needed_layers, needed_pos=needed_pos,
        directions=["ml->sl", "sl->ml"], logit_location=logit_location, ie_denom_eps=ie_denom_eps,
    )

    df = pd.DataFrame(records)
    df.to_parquet(out_dir / "records.parquet")
    with open(out_dir / "pairs.json", "w") as f:
        json.dump(_serialize_pairs(pairs, mt.tokenizer), f, indent=2)
    metadata = {
        "model": args.model,
        "tag": tag,
        "n_pairs": args.n_pairs,
        "target_slots": args.target_slots,
        "pair_seed": args.pair_seed,
        "prompt_mode": args.prompt_mode,
        "n_layer": int(mt.n_layer),
        "layer_step": args.layer_step,
        "sweep_layers": sweep_layers,
        "modules": {m: w for m, (_, w) in module_specs.items()},
        "patch_tokens": [list(pt) for pt in patch_tok_options],
        "metrics": list(METRICS),
        "directions": ["ml->sl", "sl->ml"],
        "ie_denom_eps": ie_denom_eps,
        "n_records": int(len(df)),
        "n_valid_ie": int(df["ie"].notna().sum()),
        "frac_ie_dropped": float(df["ie"].isna().mean()),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"saved {len(df)} rows -> {out_dir / 'records.parquet'} "
        f"({metadata['n_valid_ie']} non-NaN IE, {metadata['frac_ie_dropped']:.1%} dropped)"
    )


if __name__ == "__main__":
    main()
