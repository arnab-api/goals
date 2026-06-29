"""Shared (source, destination) pair construction for the multiline "completion" (stop-vs-continue)
transplant experiments — used by BOTH ``scripts/clock/train_completion_mask.py`` and the
``notebooks/clock/04.2`` / ``04.3`` notebooks, so the notebook samples *exactly* what the script
trains on.

Why this module exists — the counterfactual readout
───────────────────────────────────────────────────
We transplant a SOURCE context's head activations into a DESTINATION context and ask whether the
dest flips its verdict. The verdict is the model's *actual* next token — a **variable name** (for
"moved_on") or an **expression continuation** (for "stay"). Crucially the dest's correct flipped
answer (``target_ans``) lives entirely in the DEST's own token space and has **no relationship**
to the source's answer:

    source = "stay"/multiline   →  target_ans = the dest's CURRENT var       (dest keeps going)
    source = "moved_on"/single  →  target_ans = the dest's NEXT var          (dest moves on)

So for a sl→ml leg ``source_ans``, ``dst_ans`` and ``target_ans`` can be three mutually-distinct
tokens. The readout must therefore be ``logp[target_ans] - logp[dst_ans]`` with ``target_ans``
computed from the dest's flipped-style sibling — NEVER from the source. (An earlier version used
``src_ans`` as the readout target, which is only correct under ``--same``; this module fixes it.)

API
───
- ``StyledContext`` — one spec rendered in both styles, with each style's argmax verdict token.
- ``CompletionPair`` — a (source, dest) pair of styled contexts (source IS dest under ``same``).
- ``LEG_STYLES`` / ``legs_for`` — per-leg source/dest style mapping (target_style == src_style).
- ``build_prompt`` / ``argmax_next`` / ``build_context`` — low-level builders.
- ``build_completion_pairs(mt, n_pairs=..., ...)`` — sample + gate validated pairs.
- ``iter_leg_batches`` — yield ``(src_prompts, dst_prompts, target_ans, dst_ans)`` per leg.
"""

import random
from dataclasses import dataclass

import torch

from src.goals.multiline_prompts import (
    MULTILINE_PREDS,
    build_example,
    replace_in_code_newline_with_tabnewline,
)
from src.models import ModelandTokenizer
from src.tokens import prepare_input

N_MULTILINE_PREDS = len(
    MULTILINE_PREDS
)  # n_conditions cannot exceed this (distinct classes)


# ──────────────────────────────────────────────────────────────────────────────
# Data types.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StyledContext:
    """One predicate spec rendered in BOTH styles (same seed → identical spec/var-names, only the
    target line differs). ``ml`` = multiline target (incomplete → "continue"/stay); ``sl`` =
    singleline target (complete → "moved_on"). The two argmaxes are this context's OWN verdict
    tokens — staying vs moving on for THIS context's variables."""

    ml_prompt: str
    sl_prompt: str
    ml_ans: int  # multiline argmax  → this context's "stay/continue" token
    sl_ans: int  # singleline argmax → this context's "moved_on" (next-var) token

    def prompt(self, style: str) -> str:
        return self.ml_prompt if style == "ml" else self.sl_prompt

    def ans(self, style: str) -> int:
        return self.ml_ans if style == "ml" else self.sl_ans


@dataclass
class CompletionPair:
    """A (source, destination) pair of styled contexts. The SOURCE only donates head activations
    to transplant; the DESTINATION is the one we patch and read out. ``target_ans`` (the desired
    patched answer) is always computed from the DEST's own flipped-style sibling — never from the
    source. Under ``same`` source IS dest (a minimal pair)."""

    key: str
    source: StyledContext
    dest: StyledContext


# Per leg: which style the SOURCE donates and which style the DEST runs (clean). The dest's
# counterfactual ("target") is the OPPOSITE of its clean style — which equals the source's
# donating style. So target_style == src_style for both legs.
#   ml->sl : transplant a STAY signal (src=ml) into a moved_on dest (clean=sl) → push dest to STAY
#            → target = dest's ml answer (dest's current var).
#   sl->ml : transplant a MOVE_ON signal (src=sl) into a staying dest (clean=ml) → push dest to
#            MOVE ON → target = dest's sl answer (dest's NEXT var).
LEG_STYLES = {
    "ml->sl": {"src_style": "ml", "dst_style": "sl"},
    "sl->ml": {"src_style": "sl", "dst_style": "ml"},
}


def legs_for(direction: str) -> list[str]:
    if direction == "both":
        return ["ml->sl", "sl->ml"]
    return [direction]


# ──────────────────────────────────────────────────────────────────────────────
# Low-level builders.
# ──────────────────────────────────────────────────────────────────────────────


def build_prompt(
    mt: ModelandTokenizer, rng, target_slot, n_conditions, mode, style
) -> str:
    """One forced-prefill prompt (naming='letters' → single-token verdicts), tokenization-stable
    (newlines → ``"    \\n"``) with a trailing ``"   "`` so the pivot whitespace tokenizes the
    same across styles."""
    ex = build_example(
        mt=mt,
        rng=rng,
        target_slot=target_slot,
        n_conditions=n_conditions,
        mode=mode,
        style=style,
        naming="letters",
    )
    return replace_in_code_newline_with_tabnewline(ex["full_prompt"]) + "   "


@torch.no_grad()
def argmax_next(mt: ModelandTokenizer, inputs) -> int:
    """Greedy next-token id for a B=1 tokenized input (argmax is padding-invariant)."""
    out = mt._model(**dict(inputs), use_cache=False)
    return int(out.logits[0, -1, :].argmax())


def sample_ncond(rng, ts: int, max_conditions: int) -> int:
    """Sample ``n_conditions`` with a NEXT var available for moved_on (``>= ts + 2``) and capped
    by both ``max_conditions`` and the number of distinct multiline predicates."""
    lo = ts + 2
    hi = min(max(lo, max_conditions), N_MULTILINE_PREDS)
    return rng.randint(lo, hi)


def build_context(
    mt: ModelandTokenizer, seed: int, target_slot: int, n_conditions: int, mode: str
) -> "StyledContext | None":
    """Render one spec in BOTH styles (shared seed → identical spec, only the target line differs)
    and take each style's next-token argmax. Returns ``StyledContext`` or ``None`` if the two
    styles don't yield a genuine stay↔moved_on contrast (``ml_ans == sl_ans``)."""
    ml_prompt = build_prompt(
        mt, random.Random(seed), target_slot, n_conditions, mode, "multiline"
    )
    sl_prompt = build_prompt(
        mt, random.Random(seed), target_slot, n_conditions, mode, "singleline"
    )
    ml = prepare_input(ml_prompt, tokenizer=mt, padding_side="left")
    sl = prepare_input(sl_prompt, tokenizer=mt, padding_side="left")
    ml_ans = argmax_next(mt, ml)
    sl_ans = argmax_next(mt, sl)
    if ml_ans == sl_ans:  # no genuine stay↔moved_on contrast for this context
        return None
    return StyledContext(
        ml_prompt=ml_prompt, sl_prompt=sl_prompt, ml_ans=ml_ans, sl_ans=sl_ans
    )


def validate_pair(
    mt: ModelandTokenizer,
    source: StyledContext,
    dest: StyledContext,
    legs: list[str],
    patch_tokens,
    same: bool,
) -> bool:
    """Per active leg, gate that (a) the leg's src/dst prompts are token-aligned at every patch
    position (so the per-head transplant lands on the same token), and (b) — when ``same`` is
    False — the source's own answer differs from the dest's target token
    (``source_ans != target_ans``), proving the transplant is genuinely cross-context rather than
    copying a token. Skipped under ``same`` (there source IS dest, so they're equal by
    construction). ``target_ans != dst_ans`` is already guaranteed by ``build_context``."""
    for leg in legs:
        ss = LEG_STYLES[leg]["src_style"]
        ds = LEG_STYLES[leg]["dst_style"]
        src_prompt, dst_prompt = source.prompt(ss), dest.prompt(ds)
        si = prepare_input(src_prompt, tokenizer=mt, padding_side="left")
        di = prepare_input(dst_prompt, tokenizer=mt, padding_side="left")
        for t in patch_tokens:
            a = mt.tokenizer.decode(si["input_ids"][0, t].item())
            b = mt.tokenizer.decode(di["input_ids"][0, t].item())
            if a != b:
                return False
        if not same and source.ans(ss) == dest.ans(ss):  # target_style == src_style
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Top-level sampling + batching.
# ──────────────────────────────────────────────────────────────────────────────


def build_completion_pairs(
    mt: ModelandTokenizer,
    *,
    n_pairs: int,
    target_slots=(1, 2, 3, 4),
    direction: str = "both",
    same: bool = False,
    match_target_slot: bool = False,
    match_n_cond: bool = False,
    max_conditions: int = 6,
    mode: str = "chat",
    seed: int = 0,
    patch_tokens=(-2,),
    max_attempts_factor: int = 50,
    log=print,
) -> list[CompletionPair]:
    """Sample up to ``n_pairs`` validated (source, destination) pairs.

    Each side is a ``StyledContext`` (both styles of one spec). The DEST's two argmaxes give the
    clean answer ``dst_ans`` and the counterfactual ``target_ans`` (its flipped-style sibling), so
    ``target_ans != dst_ans`` holds by construction. The SOURCE only donates head activations;
    ``validate_pair`` applies the per-leg alignment + (non-``same``) ``source_ans != target_ans``
    gates. ``target_slots`` entries must be in ``{1,2,3,4}``. ``log`` is a progress callback
    (default ``print``; pass e.g. ``logger.info``)."""
    target_slots = [int(x) for x in target_slots]
    assert all(1 <= ts <= 4 for ts in target_slots), "target_slot must be in {1,2,3,4}"
    legs = legs_for(direction)

    prng = random.Random(seed)
    pairs: list[CompletionPair] = []
    attempts = 0
    max_attempts = max_attempts_factor * n_pairs
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        ts = prng.choice(target_slots)
        nc = sample_ncond(prng, ts, max_conditions)

        if same:
            source = build_context(mt, prng.randrange(10**9), ts, nc, mode)
            dest = source  # minimal pair: source IS dest
        else:
            ts2 = ts if match_target_slot else prng.choice(target_slots)
            nc2 = nc if match_n_cond else sample_ncond(prng, ts2, max_conditions)
            source = build_context(mt, prng.randrange(10**9), ts, nc, mode)
            dest = build_context(mt, prng.randrange(10**9), ts2, nc2, mode)
        if source is None or dest is None:
            continue
        if not validate_pair(mt, source, dest, legs, patch_tokens, same):
            continue

        pairs.append(
            CompletionPair(key=f"p{len(pairs)}_ts{ts}_nc{nc}", source=source, dest=dest)
        )
        if log and len(pairs) % 25 == 0:
            log(f"  built {len(pairs)}/{n_pairs} pairs ({attempts} attempts)")
    if log:
        log(f"built {len(pairs)} pairs from {attempts} attempts")
    return pairs


def iter_leg_batches(pairs: list[CompletionPair], batch_size: int, direction: str):
    """Yield ``(src_prompts, dst_prompts, target_ans, dst_ans)`` chunks of ≤ ``batch_size`` — one
    stream per leg implied by ``direction`` (ml→sl and/or sl→ml). ``src_prompts`` come from the
    SOURCE (donate activations only); ``dst_prompts``/``dst_ans``/``target_ans`` all come from the
    DEST — ``target_ans`` is the dest's flipped-style (counterfactual) answer, in the dest's token
    space (target_style == src_style; see ``LEG_STYLES``)."""
    for leg in legs_for(direction):
        ss = LEG_STYLES[leg][
            "src_style"
        ]  # source-donating style == dest's target style
        ds = LEG_STYLES[leg]["dst_style"]  # dest's clean (run) style
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            yield (
                [
                    p.source.prompt(ss) for p in chunk
                ],  # source: donates head activations
                [p.dest.prompt(ds) for p in chunk],  # dest: patched + read out
                [
                    p.dest.ans(ss) for p in chunk
                ],  # target_ans: dest's counterfactual token
                [p.dest.ans(ds) for p in chunk],  # dst_ans: dest's clean token
            )
