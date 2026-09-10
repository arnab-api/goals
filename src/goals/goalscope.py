"""PatchScope generation helpers and HTML visualization utilities.

This module provides:
- Greedy generation helpers (plain and patched, single and batched)
- HTML visualization for patchscope results: hover-over tooltip per token
"""

from __future__ import annotations

import html as _html
import keyword as _kw
import warnings
from typing import Callable, Literal, Optional, Union

import baukit
import torch
from tqdm.auto import tqdm

from src.functional import PatchSpec, generate_with_patch, get_hs
from src.goals.head_loc_attn import (
    HeadMask,
    collect_head_steering_vecs,
    forward_with_src_replacement,
    get_o_proj_inputs_batch,
)
from src.tokens import find_token_range, prepare_input
from src.utils.typing import TokenizerOutput

# ---------------------------------------------------------------------------
# Greedy generation helpers
# ---------------------------------------------------------------------------


def greedy_continue(
    mt,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    stop_ids: Optional[set[int]] = None,
) -> str:
    """Plain greedy generation — no patching.

    Args:
        mt: ModelandTokenizer.
        input_ids: (1, seq) prompt token ids.
        attention_mask: (1, seq) attention mask.
        max_new_tokens: Maximum tokens to generate.
        stop_ids: Token ids that halt generation (in addition to EOS).

    Returns:
        Decoded generated string (prompt stripped).
    """
    _stop = stop_ids or [mt.tokenizer.eos_token_id]
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = mt._model(input_ids=input_ids, attention_mask=attention_mask)
            next_tok = int(out.logits[0, -1].argmax())
            generated.append(next_tok)
            if next_tok in _stop:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=input_ids.device)], dim=1
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1), dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                ],
                dim=1,
            )
    return mt.tokenizer.decode(generated, skip_special_tokens=True).rstrip()


def _normalize_input_positions(
    positions: Optional[list[int]],
    L_prompt: int,
    default_last: bool = True,
) -> list[int]:
    """Normalize prompt-position list: handle negatives, de-dup, sort, validate.

    Returns positions in absolute (non-negative) form, sorted ascending.
    ``None`` → ``[L_prompt - 1]`` if ``default_last`` else ``[]``.
    """
    if positions is None:
        return [L_prompt - 1] if default_last else []
    norm = sorted({p if p >= 0 else L_prompt + p for p in positions})
    for p in norm:
        if p < 0 or p >= L_prompt:
            raise IndexError(
                f"patch_input_positions out of range for L_prompt={L_prompt}: "
                f"{positions} -> {norm}"
            )
    return norm


def headpatched_generate(
    mt,
    dest_inputs: dict,
    src_o_proj_inps: torch.Tensor,
    mask,
    mask_layers: Optional[list[int]] = None,
    max_new_tokens: int = 64,
    stop_ids: Optional[set[int]] = None,
    n_steer_tokens: Optional[int] = None,
    patch_input_positions: Optional[list[int]] = None,
    alpha: float = 1.0,
    return_raw_logits: bool = False,
    return_raw_toks: bool = False,
    *,
    layers: Optional[list[int]] = None,
) -> Union[str, list[str]]:
    """Greedy generation with head-replacement patching — KV-cache aware, single or batched.

    ``src_o_proj_inps`` decides both the batch size and the return type:

    * ``(n_layers, n_embd)``      -> one generation, returns ``str``
    * ``(B, n_layers, n_embd)``   -> B generations sharing ONE destination prompt, returns
      ``list[str]`` of length B — even when ``B == 1``.

    Keying the return on the input's *rank* rather than on ``B`` matters: a caller that slices a
    batch down to one row (``src[i:i+1]``) still gets a list, so batch code never has to special-case
    its own length.

    The batch axis is over SOURCES, not destinations — ``dest_inputs`` is a single tokenized prompt,
    tiled internally. Running B different destination prompts is not supported here.

    Decoding pattern:
        Step 0 (prefill): full forward over the prompt, patches every position in
            ``patch_input_positions`` (default: last prompt token only). Returns
            ``past_key_values`` so the patched K/V (at layers above the patched o_proj) are
            written into the cache for those positions.
        Step >= 1 (decode): 1-token forward reusing the cache. If still inside the steer window,
            the new token's o_proj input is patched too and its K/V get cached. Past patched K/V
            remain in the cache.

    ``n_steer_tokens`` counts *forward passes*, with the prefill as the first; ``n_steer_tokens=1``
    patches only the prefill. ``None`` patches every step (legacy behavior).

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict (input_ids, attention_mask), batch 1.
        src_o_proj_inps: source o_proj inputs — see the shape contract above.
        mask: HeadMask with per-head interpolation weights. To limit WHERE the intervention
            happens, zero the rows outside the band — ``mask.restrict_to_layers(lo, hi)`` — do NOT
            slice ``mask_layers``, which renumbers the rows and shifts the whole mask.
        mask_layers: the ordered key decoding the mask's ROWS (row ``li`` <-> ``mask_layers[li]``),
            also matching ``src_o_proj_inps``' layer axis. ``None`` (default) = identity, which is
            what a full-depth mask from ``load_head_mask`` wants.
        max_new_tokens: Maximum tokens to generate per item.
        stop_ids: Token ids that halt generation for an item.
        n_steer_tokens: Apply a fresh patch only on the first ``n_steer_tokens`` forward passes.
        patch_input_positions: Positions within the prompt to patch during the prefill. Negative
            indices supported (``-1`` = last token). Duplicates removed. ``None`` (default) ->
            ``[L_prompt - 1]``. Validated to lie in ``[0, L_prompt)`` after normalization.

            Note: one source snapshot is broadcast across every position in this list; per-position
            source variation is not supported. See ``forward_with_src_replacement``.
        return_raw_logits: If True, also return the raw first-forward-pass logits.
        return_raw_toks: If True, also return the EXACT generated token ids (no re-tokenization),
            with any trailing stop token stripped so they match the ``skip_special_tokens`` decode.
            Concatenate these onto the prompt's own token ids to build the full sequence for a
            downstream forward — this avoids the boundary-merge drift that re-tokenizing
            ``prompt + decode(gen)`` introduces at the cut token. Shape mirrors the return: a
            ``list[int]`` for a 2D source, a ``list[list[int]]`` for a 3D source.
    Returns:
        ``str`` (2D source) or ``list[str]`` (3D source); with ``return_raw_logits`` / and or
        ``return_raw_toks`` set, a tuple ``(outs, [raw_logits], [raw_toks])`` in that order.
    """
    if layers is not None:
        if mask_layers is not None:
            raise TypeError(
                "pass `mask_layers`, not both it and the deprecated `layers`"
            )
        warnings.warn(
            "`layers=` is now `mask_layers=` — it decodes the mask's rows rather than "
            "selecting layers to intervene at. To limit WHERE the intervention happens, "
            "use mask.restrict_to_layers(lo, hi).",
            DeprecationWarning,
            stacklevel=2,
        )
        mask_layers = layers

    _stop = stop_ids or set()
    if len(_stop) == 0:
        _stop = {mt.tokenizer.eos_token_id}

    single = src_o_proj_inps.dim() == 2
    src = src_o_proj_inps.unsqueeze(0) if single else src_o_proj_inps
    if src.dim() != 3:
        raise ValueError(
            f"src_o_proj_inps must be (n_layers, n_embd) or (B, n_layers, n_embd); "
            f"got {tuple(src_o_proj_inps.shape)}"
        )
    B = src.shape[0]
    # src is always 3D from here, so forward_with_src_replacement always yields (B, vocab) and the
    # decode loop needs no rank special-casing.
    input_ids = dest_inputs["input_ids"].expand(B, -1).clone()
    attention_mask = dest_inputs["attention_mask"].expand(B, -1).clone()
    L_prompt = input_ids.shape[1]
    prefill_positions = _normalize_input_positions(patch_input_positions, L_prompt)

    generated = [[] for _ in range(B)]
    finished = [False] * B
    past_key_values = None

    if return_raw_logits:
        raw_logits = None
    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(finished):
                break
            cur_input_ids = input_ids if step == 0 else input_ids[:, -1:]
            inputs = {
                "input_ids": cur_input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if n_steer_tokens is None or step < n_steer_tokens:
                positions_this_step = prefill_positions if step == 0 else [-1]
                logits, past_key_values = forward_with_src_replacement(
                    mt,
                    inputs,
                    src,
                    mask,
                    mask_layers,
                    alpha=alpha,
                    patch_positions=positions_this_step,
                    return_past_key_values=True,
                )  # (B, vocab)
            else:
                out = mt._model(**inputs)
                logits = out.logits[:, -1, :].float()  # (B, vocab)
                past_key_values = out.past_key_values
            next_toks = logits.argmax(dim=-1)  # (B,)
            if return_raw_logits and step == 0:
                raw_logits = logits.clone()

            for i in range(B):
                if not finished[i]:
                    tok = int(next_toks[i])
                    generated[i].append(tok)
                    if tok in _stop:
                        finished[i] = True

            input_ids = torch.cat([input_ids, next_toks.unsqueeze(1)], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (B, 1), dtype=attention_mask.dtype, device=attention_mask.device
                    ),
                ],
                dim=1,
            )

    outs = [
        mt.tokenizer.decode(g, skip_special_tokens=True).rstrip() for g in generated
    ]
    outs = outs[0] if single else outs

    extras = []
    if return_raw_logits:
        extras.append(raw_logits)
    if return_raw_toks:
        # the exact ids the model emitted, minus a trailing stop token, so this lines up with the
        # skip_special_tokens decode above and can be appended to the prompt's own token ids.
        stripped = []
        for g in generated:
            g = list(g)
            while g and g[-1] in _stop:
                g.pop()
            stripped.append(g)
        extras.append(stripped[0] if single else stripped)
    return (outs, *extras) if extras else outs


def headpatched_generate_batch(*args, **kwargs) -> list[str]:
    """Deprecated alias — :func:`headpatched_generate` now handles both ranks.

    Kept so existing call sites keep working AND keep returning a list: it forwards unchanged, and
    a 3D source already yields ``list[str]``.
    """
    warnings.warn(
        "headpatched_generate_batch is deprecated; call headpatched_generate directly — a 3D "
        "src_o_proj_inps returns list[str].",
        DeprecationWarning,
        stacklevel=2,
    )
    out = headpatched_generate(*args, **kwargs)
    return [out] if isinstance(out, str) else out


def head_additive_generate(
    mt,
    dest_inputs: dict,
    steering_vecs: torch.Tensor,
    layers: list[int],
    max_new_tokens: int,
    strategy: Literal["additive", "additive_norm"] = "additive",
    stop_ids: Optional[set[int]] = None,
    n_steer_tokens: Optional[int] = None,
    patch_input_positions: Optional[list[int]] = None,
    alpha: float = 1.0,
) -> Union[str, list[str]]:
    """Greedy generation with an ADDITIVE residual-stream injection — the counterpart of
    :func:`headpatched_generate`, which replaces per-head o_proj inputs instead.

    Same interface and the same rank-keyed return contract:

    * ``steering_vecs`` ``(n_embd,)``      -> one generation, returns ``str``
    * ``steering_vecs`` ``(B, n_embd)``    -> B generations sharing ONE destination prompt,
      returns ``list[str]`` — even when ``B == 1``.
    * ``steering_vecs`` ``{layer: tensor}`` -> a DIFFERENT vector per layer (same rank rule). Use
      this to mirror :func:`headpatched_generate`, which writes each layer's own source value at
      that layer; a single tensor broadcasts one vector to every layer in ``layers`` and collapses
      the source's per-layer structure.

    The two strategies differ only in what magnitude is injected:

    * ``"additive"`` — ``h + alpha * g``. The added magnitude is ``alpha * ||g||``, an ABSOLUTE
      quantity. Since ``||g||`` grows with the number of aggregated heads (it is a sum over them)
      and ``||h||`` grows with depth, comparing head sets / bands / models at a fixed ``alpha``
      can measure the injection's magnitude rather than the encoding's quality.
    * ``"additive_norm"`` — ``h + alpha * ||h|| * g/||g||``. ``alpha`` is then "add this fraction
      of the local residual norm along the goal direction", which is scale-invariant. Its value
      does NOT carry over from ``"additive"`` and needs re-tuning.

    Normalisation for ``"additive_norm"`` happens HERE, so callers always pass the raw aggregated
    vector and the strategy alone decides the semantics.

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict, batch 1 (tiled internally).
        steering_vecs: aggregated head output(s) — see the shape contract above. Raw, unnormalised.
        layers: layer indices to inject at (the steer band), e.g. ``range(35, 46)``.
        max_new_tokens: Maximum tokens to generate per item.
        strategy: ``"additive"`` | ``"additive_norm"``.
        stop_ids: token ids that halt generation, forwarded as HF ``generate``'s ``eos_token_id``
            (which takes a list). A non-empty set REPLACES the model's default EOS rather than
            adding to it — same convention as :func:`headpatched_generate`. ``None``/empty keeps
            the default.
        n_steer_tokens: how many decode steps apply the patch. ``None`` = every step.
        patch_input_positions: position(s) to inject at. ``[-1]`` (default) and ``[0]`` are
            patched at every decode step. Any OTHER position is patched during the prefill only
            (with a ``RuntimeWarning``): with KV-cache enabled a fixed position like ``-2`` is out
            of bounds once the key dim is 1. The prefill edit still reaches every later token
            via the cached K/V above the patched layer, exactly as in :func:`headpatched_generate` —
            "prefill only" restricts WHERE a fresh patch is written, not how far its effect
            travels. The two therefore agree at ``n_steer_tokens=1`` (what the move-on setups use)
            and differ above it, where ``headpatched_generate`` additionally patches each newly
            generated token at ``-1``.
        alpha: injection coefficient (see the strategy notes above).

    Returns:
        ``str`` for a 1D source, ``list[str]`` for a 2D source.
    """
    if strategy not in ("additive", "additive_norm"):
        raise ValueError(
            f"strategy must be 'additive' or 'additive_norm'; got {strategy!r}. "
            f"For head replacement use headpatched_generate()."
        )
    positions = (
        [-1]
        if patch_input_positions is None
        else list(dict.fromkeys(patch_input_positions))
    )
    # With KV-cache on, a fixed position other than 0/-1 is out of bounds on decode steps (key dim
    # is 1 there), so those positions are patched during the PREFILL only, via on_pass="first".
    # That is NOT the same as "the intervention stops": the prefill patch edits the residual stream
    # at that position, the layers above it compute K/V from the edited residual, and generate()
    # caches them — so every later token attending back to that position sees the patched keys and
    # values. headpatched_generate behaves identically in this respect (its docstring: "the patched
    # K/V ... are written into the cache for those positions").
    #
    # What DOES differ above n_steer_tokens=1 is whether each NEWLY GENERATED token also gets a
    # fresh patch: headpatched_generate drives its own loop and switches to [-1] after step 0
    # (`positions_this_step = prefill_positions if step == 0 else [-1]`), whereas generate_with_patch
    # cannot vary the position per step and so applies nothing further.
    prefill_only = positions not in ([-1], [0])
    if prefill_only:
        msg = (
            f"patch_input_positions={positions}: with KV-cache on, only 0/-1 are valid at every "
            f"decode step, so these are patched during the PREFILL only (on_pass='first'). The "
            f"edit still propagates to every later token through the cached K/V above the "
            f"patched layer — it is not confined to one token."
        )
        if n_steer_tokens != 1:
            msg += (
                f" But n_steer_tokens={n_steer_tokens} has no additional effect here, whereas "
                f"headpatched_generate would ALSO apply a fresh patch at -1 to each newly "
                f"generated token for {n_steer_tokens} passes. The two agree at "
                f"n_steer_tokens=1."
            )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Validate against the prompt length here, the way headpatched_generate does via
    # _normalize_input_positions. Without this an out-of-range position surfaces as a raw
    # IndexError from inside the baukit hook ("index -24 is out of bounds for dimension 1"), with
    # no hint that it came from patch_input_positions. Signs are preserved: -1 must stay negative
    # so that on_pass="all" lands on the NEW token at each decode step, not a fixed absolute slot.
    _L = dest_inputs["input_ids"].shape[1]
    _bad = [p for p in positions if not (-_L <= p < _L)]
    if _bad:
        raise ValueError(
            f"patch_input_positions={_bad} out of range for a {_L}-token prompt "
            f"(valid: -{_L} .. {_L - 1})"
        )
    if _L < 2:
        raise ValueError(
            f"prompt is {_L} token(s): the baukit hook detects the prefill as `seq_len > 1`, so a "
            f"1-token prompt has no distinguishable prefill pass and patches would misfire."
        )

    # A dict {layer: vec} injects a DIFFERENT vector per layer — the faithful additive analogue of
    # headpatched_generate, which writes each layer's own source value at that layer. Passing one
    # tensor broadcasts the SAME vector to every layer in `layers`, which collapses whatever
    # per-layer structure the source had: if the heads that produced it span layers 11-62 and you
    # inject at 40-49, roughly half the aggregate lands at the wrong depth.
    per_layer = isinstance(steering_vecs, dict)
    if per_layer:
        missing = [l for l in layers if l not in steering_vecs]
        if missing:
            raise ValueError(
                f"steering_vecs is missing layers {missing[:8]}"
                f"{' …' if len(missing) > 8 else ''}"
            )
        ranks = {v.dim() for v in steering_vecs.values()}
        if len(ranks) != 1:
            raise ValueError(f"steering_vecs entries have mixed ranks {sorted(ranks)}")
        single = ranks.pop() == 1
        by_layer = {
            l: (v.unsqueeze(0) if single else v) for l, v in steering_vecs.items()
        }
    else:
        single = steering_vecs.dim() == 1
        if steering_vecs.dim() not in (1, 2):
            raise ValueError(
                f"steering_vecs must be (n_embd,), (B, n_embd) or {{layer: tensor}}; "
                f"got {tuple(steering_vecs.shape)}"
            )
        v = steering_vecs.unsqueeze(0) if single else steering_vecs
        by_layer = {l: v for l in layers}

    def _prep(v):
        if strategy == "additive_norm":
            # Hand the patch site a UNIT direction; the magnitude comes from ||h|| there.
            v = v / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(v.dtype)
        return alpha * v

    by_layer = {l: _prep(v) for l, v in by_layer.items()}
    B = next(iter(by_layer.values())).shape[0]
    if any(v.shape[0] != B for v in by_layer.values()):
        raise ValueError("steering_vecs entries disagree on batch size")

    tiled = TokenizerOutput(
        dict(
            input_ids=dest_inputs["input_ids"].expand(B, -1).clone(),
            attention_mask=dest_inputs["attention_mask"].expand(B, -1).clone(),
        )
    )
    _strat = "add" if strategy == "additive" else "add_norm"
    patches = [
        PatchSpec(
            location=(mt.layer_name_format.format(li), p),
            patch=by_layer[li],
            strategy=_strat,
            on_pass="first" if prefill_only else "all",
        )
        for li in layers
        for p in positions
    ]
    # `stop_ids` maps onto HF generate's `eos_token_id`, which accepts a LIST and stops each
    # sequence on any of them — matching headpatched_generate, where a non-empty `stop_ids`
    # REPLACES the default EOS rather than adding to it.
    extra = {"eos_token_id": sorted(stop_ids)} if stop_ids else {}
    gens = generate_with_patch(
        mt,
        tiled,
        max_new_tokens=max_new_tokens,
        patches=patches,
        n_steer_tokens=n_steer_tokens,
        do_sample=False,
        patch_interface="baukit",
        remove_prefix=True,
        skip_special_tokens=True,
        **extra,
    )
    return gens[0] if single else list(gens)


# ---------------------------------------------------------------------------
# PatchScope HTML visualization
# ---------------------------------------------------------------------------

_PRISM_CSS_CDN = (
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism'
    '/1.29.0/themes/prism-tomorrow.min.css">'
)

_CSS = """
.ps-wrap {
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px; line-height: 2.2em;
    max-width: 960px; margin: 16px auto;
    background: #2d2d2d; color: #ccc;
    padding: 16px 20px; border-radius: 8px;
    white-space: pre-wrap;
}
.ps-tok {
    cursor: default;
    border-radius: 3px; padding: 1px 0;
    transition: background 0.12s;
}
.ps-tok:hover { background: rgba(255,210,80,0.22); outline: 1px solid rgba(255,210,80,0.4); }
#ps-tooltip {
    display: none;
    position: fixed;
    background: #1a1a2e; border: 1px solid #555;
    color: #e8e8e8; font-size: 12px;
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    padding: 6px 10px; border-radius: 5px;
    white-space: pre-wrap; min-width: 240px; max-width: 440px;
    z-index: 9999; box-shadow: 0 4px 14px rgba(0,0,0,0.6);
    pointer-events: none;
}
.ps-legend { font-size: 11px; color: #777; margin-bottom: 8px; }
"""

_JS = """
<div id="ps-tooltip"></div>
<script>
(function() {
    var tip = document.getElementById('ps-tooltip');
    document.querySelectorAll('.ps-tok').forEach(function(tok) {
        tok.addEventListener('mouseenter', function(e) {
            tip.textContent = tok.dataset.tip;
            tip.style.display = 'block';
            _moveTip(e);
        });
        tok.addEventListener('mousemove', _moveTip);
        tok.addEventListener('mouseleave', function() {
            tip.style.display = 'none';
        });
    });
    function _moveTip(e) {
        var x = e.clientX + 14, y = e.clientY - 36;
        var tw = tip.offsetWidth, th = tip.offsetHeight;
        if (x + tw > window.innerWidth  - 8) x = e.clientX - tw - 14;
        if (y - th < 8)                      y = e.clientY + 20;
        tip.style.left = x + 'px';
        tip.style.top  = y + 'px';
    }
})();
</script>
"""

_PY_KEYWORDS = set(_kw.kwlist) | {"True", "False", "None"}
_PY_BUILTINS = {
    "print",
    "len",
    "range",
    "int",
    "str",
    "float",
    "list",
    "dict",
    "set",
    "tuple",
    "bool",
    "type",
    "isinstance",
}
_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "//",
    "**",
    "%",
    "=",
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",
    "+=",
    "-=",
    "&",
    "|",
    "^",
    "~",
    "<<",
    ">>",
    "->",
    "...",
}
_PUNCTUATION = {"(", ")", "[", "]", "{", "}", ",", ":", ";", "."}
_STR_PREFIXES = (
    '"',
    "'",
    'f"',
    "f'",
    'b"',
    "b'",
    'r"',
    "r'",
    '"""',
    "'''",
    'f"""',
    "f'''",
)


def _prism_class(token_str: str) -> str:
    """Map a raw token string to a Prism.js CSS class string."""
    s = token_str.strip()
    if not s:
        return ""
    if s in _PY_KEYWORDS:
        return "token keyword"
    if s.startswith("#"):
        return "token comment"
    if any(s.startswith(p) for p in _STR_PREFIXES):
        return "token string"
    if s.replace(".", "", 1).lstrip("-").isdigit():
        return "token number"
    if s in _OPERATORS:
        return "token operator"
    if s in _PUNCTUATION:
        return "token punctuation"
    if s in _PY_BUILTINS:
        return "token builtin"
    return ""


def render_patchscope_html(
    results: list[dict],
    title: str = "PatchScope",
) -> str:
    """Render patchscope results as a self-contained HTML string.

    Each result dict must have keys: ``pos`` (int), ``token`` (str),
    ``patched_gen`` (str).  Hovering a token shows its patched verbalization.

    Returns:
        Full HTML document string.
    """
    spans = []
    for r in results:
        raw = r["token"]
        # Newline tokens render as invisible — replace \n with a visible glyph
        # (↵) so the span is hoverable, then emit the real \n for indentation.
        tok_html = _html.escape(raw).replace("\n", "↵\n")
        # Non-scoped context tokens (e.g. the prompt prefix when include_noncode=True)
        # are shown dimmed for readability, with no goal-scope tooltip.
        if not r.get("scoped", True):
            spans.append(
                f'<span class="ps-noncode" style="opacity:0.4">{tok_html}</span>'
            )
            continue
        tip_text = (
            f"pos {r['pos']} {r['token']!r}: {r['patched_gen'].strip() or '(empty)'}"
        )
        tip_attr = _html.escape(tip_text, quote=True)
        cls = ("ps-tok " + _prism_class(r["token"])).strip()
        spans.append(f'<span class="{cls}" data-tip="{tip_attr}">{tok_html}</span>')
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_html.escape(title)}</title>\n"
        f"{_PRISM_CSS_CDN}\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="ps-wrap">\n'
        f'<div class="ps-legend">{_html.escape(title)}'
        " &mdash; hover a token to see the patched verbalization</div>"
        + "".join(spans)
        + f"\n</div>\n{_JS}\n</body>\n</html>"
    )


# Content of the GoalScope destination prompt, exposed so other read-out mechanisms (e.g. the
# activation oracle in scripts/evaluation/eval_act_oracles.py) can pose the *same* question.
GOALSCOPE_SYS = "You are a helpful assistant proficient in programming."
GOALSCOPE_QUESTION = "What are you doing at this line of the code. Keep your answer as brief as possible."


def _build_goalscope_prompt(mt) -> str:
    """Build the GoalScope destination prompt ("What are you doing at this line?")."""
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": GOALSCOPE_QUESTION},
        {"role": "assistant", "content": ""},
    ]
    try:
        return mt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=False,
        )


# GoalScope destination prompt for the causality read-out: an information-less C++ function stub. The
# goal encoding is steered in on top of it; the model then continues the C++ body cross-lingually.
# Kept in sync with scripts/evaluation/eval_goalscope.py (CAUSAL_PREFIX / CAUSAL_INSTRUCTION).
CAUSAL_PREFIX = "bool check(auto value){"
CAUSAL_INSTRUCTION = (
    "You are implementing a simple C++ function that checks for a condition. "
    "Just write the function body, without any comments or explanations. Assume all the necessary "
    "headers are already included."
)


def _build_causality_prompt(mt) -> str:
    """Build the GoalScope *causality* destination prompt — an information-less C++ function stub
    (assistant turn == ``CAUSAL_PREFIX``). Cross-language counterpart of :func:`_build_goalscope_prompt`."""
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": CAUSAL_INSTRUCTION},
        {"role": "assistant", "content": CAUSAL_PREFIX},
    ]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        return mt.tokenizer.apply_chat_template(
            messages, chat_template_kwargs={"enable_thinking": False}, **kw
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(messages, **kw)


class OVReader:
    """OV-write residual probe: is the goal already in the residual, in the goal heads' OV-write direction?

    For a residual ``h`` (the stream after some layer ``L``, at a pivot token), a goal head ``(l, j)`` would
    write ``W_O_j W_V_kv (g ⊙ h)`` into the residual if ``h`` were its value source — the full OV transform,
    NOT a projection: the head reads its value on the RMSNorm'd residual (per-dim gain ``g`` folded; the
    direction-preserving ``1/rms`` scalar is dropped), maps it through ``W_V`` (GQA kv-head ``kv = j // group``),
    then writes it through ``W_O``. Aggregating the unit OV-writes over the readout heads gives a residual
    DIRECTION ``h_ov``; steering the GoalScope probe with it (``head_additive_generate`` / ``additive_norm``)
    verbalizes the goal iff ``h`` already carries it in that write direction.

    The per-head factors ``(W_O_j, W_V_kv, g)`` are computed once in ``__init__`` and cached in ``readers``.
    """

    def __init__(self, mt, readout_heads: list[tuple[int, int]]):
        self.mt = mt
        self.readout_heads = list(readout_heads)
        self.n_kv = mt.n_kv_heads
        self.group = mt.n_heads // self.n_kv  # GQA group size
        self.readers = self._build_readers()  # per-head (W_O_j, W_V_kv, g), CPU float32
        self._probe_inputs = {}  # {mode: tokenized GoalScope probe}, built lazily

    def _proj_weight(self, layer: int, which: str) -> torch.Tensor:
        name = self.mt.attn_module_name_format.format(layer) + f".{which}_proj"
        return baukit.get_module(self.mt._model, name).weight

    def _build_readers(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per goal head ``(l, j)``: ``(W_O_j, W_V_kv, g)`` — small factors of the OV map
        ``W_O_j W_V_kv diag(g)``, kept as CPU float32 and applied lazily in :meth:`resid_ov_vec`."""
        hd = self.mt.head_dim
        readers = []
        for (layer, j) in tqdm(self.readout_heads, desc="Building OV readers"):
            kv = j // self.group
            Wo = self._proj_weight(layer, "o")[:, j * hd : (j + 1) * hd].detach().float().cpu()
            Wv = self._proj_weight(layer, "v")[kv * hd : (kv + 1) * hd, :].detach().float().cpu()
            g = (
                baukit.get_module(
                    self.mt._model, self.mt.layer_name_format.format(layer) + ".input_layernorm"
                )
                .weight.detach()
                .float()
                .cpu()
            )
            readers.append((Wo, Wv, g))
        return readers

    def resid_ov_vec(self, h: torch.Tensor) -> torch.Tensor:
        """``h_ov = mean_j unit(W_O_j W_V_kv (g ⊙ h))`` — the aggregated OV-write direction for value ``h``.
        ``h`` is any ``(n_embd,)`` residual (moved to CPU float32); returns ``(n_embd,)`` CPU float32."""
        h = h.detach().float().cpu()
        vecs = []
        for Wo, Wv, g in self.readers:
            h_ov = Wo @ (Wv @ (g * h))
            vecs.append(h_ov / h_ov.norm().clamp_min(1e-8))
        return torch.stack(vecs).mean(0)

    def probe_inputs(self, mode: Literal["faithfulness", "causality"] = "faithfulness"):
        """Tokenized GoalScope destination prompt for ``mode`` (cached per mode). ``"faithfulness"`` =
        the NL "what are you doing at this line?" probe; ``"causality"`` = the C++ function stub."""
        if mode not in self._probe_inputs:
            builder = _build_goalscope_prompt if mode == "faithfulness" else _build_causality_prompt
            self._probe_inputs[mode] = prepare_input(prompts=builder(self.mt), tokenizer=self.mt)
        return self._probe_inputs[mode]

    def apply_probe(
        self,
        latents,
        *,
        mode: Literal["faithfulness", "causality"] = "faithfulness",
        steer_layers: Optional[list[int]] = None,
        max_new_tokens: int = 128,
        strategy: Literal["additive", "additive_norm"] = "additive_norm",
        alpha: float = 0.15,
        n_steer_tokens: Optional[int] = None,
        probe_inputs: Optional[dict] = None,
    ):
        """Probe already-extracted residual latents — all in ONE batched ``head_additive_generate``.

        ``latents`` is either a ``{key: (n_embd,) residual}`` dict (returns ``{key: readout}``) or a
        list/sequence of residual vectors (returns ``list[str]``). Each residual is turned into its
        OV-write steering vector (:meth:`resid_ov_vec`) and steered into the GoalScope probe for ``mode``.
        Batches every latent into a single generation, so N latents cost ONE generate call.
        """
        steer_layers = list(range(35, 46)) if steer_layers is None else list(steer_layers)
        is_dict = isinstance(latents, dict)
        keys = list(latents) if is_dict else None
        vecs = [latents[k] for k in keys] if is_dict else list(latents)
        steer = torch.stack([self.resid_ov_vec(h) for h in vecs])          # (n, n_embd) CPU f32
        outs = head_additive_generate(
            self.mt,
            probe_inputs if probe_inputs is not None else self.probe_inputs(mode),
            steer,
            layers=steer_layers,
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            alpha=alpha,
            n_steer_tokens=n_steer_tokens,
        )
        return dict(zip(keys, outs)) if is_dict else outs

    def apply_probe_on_prompt(
        self,
        prompt: str | TokenizerOutput,
        pivot: int,
        probe_layers: list[int],
        **kwargs,
    ) -> dict[int, str]:
        """Extract the residual after each ``probe_layer`` at ``pivot`` in ONE :func:`get_hs` forward,
        then :meth:`apply_probe` on those latents. Returns ``{L: readout}``. ``kwargs`` (mode, steer_layers,
        max_new_tokens, strategy, alpha, n_steer_tokens, probe_inputs) are forwarded to :meth:`apply_probe`."""
        inp = prepare_input(prompts=prompt, tokenizer=self.mt) if isinstance(prompt, str) else prompt
        probe_layers = list(probe_layers)
        locations = [(self.mt.layer_name_format.format(L), pivot) for L in probe_layers]
        hs = get_hs(self.mt, inp, locations=locations, return_dict=True)
        latents = {L: hs[(self.mt.layer_name_format.format(L), pivot)] for L in probe_layers}
        return self.apply_probe(latents, **kwargs)


def apply_goalscope(
    prompt: str | TokenizerOutput,
    *,
    mt,
    selected_heads: list[tuple[int, int]],
    strategy: Literal["additive", "additive_norm", "replace"] = "additive",
    steer_layer_range: tuple[int, int] = (35, 46),
    batch_size: int = 16,
    alpha: float = 0.2,
    max_new_tokens: int = 64,
    n_steer_tokens: Union[int, Literal["all"]] = "all",
    title: str = "GoalScope",
    show: bool = True,
    goalscope_prompt: Optional[str] = None,
    include_noncode: bool = False,
    save_path: Optional[str] = None,
    token_positions: Optional[list[int]] = None,
    patch_interface: Optional[Callable] = None,
) -> list[dict]:
    """Run GoalScope on a single prompt.

    Extracts goal signals from the ``<code>`` segment of ``prompt`` and injects
    them into the GoalScope probe ("What are you doing at this line?") to produce
    a verbalization per token position.

    Two intervention strategies:

    - ``"additive"`` (default): aggregate the selected heads' o_proj contributions
      into a single residual-stream vector per source position, scale by
      ``alpha``, and add it to the residual stream at every layer in
      ``steer_layer_range`` (last token).  Uses ``generate_with_patch``.
    - ``"additive_norm"``: the same injection made **norm-relative** —
      ``h + alpha * ||h|| * g_enc/||g_enc||``.  Under ``"additive"`` the added
      magnitude is ``alpha * ||g_enc||``, an absolute quantity, while ``||h||``
      grows with depth and ``||g_enc||`` grows with the number of selected heads
      (it is a *sum* over heads).  A comparison across head sets, layer bands or
      models at fixed ``alpha`` can therefore read the injection's *magnitude*
      rather than the encoding's quality.  Here ``alpha`` means "add this
      fraction of the local residual norm along the goal direction", so its
      value does NOT carry over from ``"additive"`` and needs re-tuning.
    - ``"replace"``: surgically replace the selected heads' o_proj inputs at the
      destination's last token with the corresponding source o_proj inputs.
      Layers come from ``selected_heads``; ``steer_layer_range`` is ignored.
      ``alpha`` controls the source-side weight inside the per-head
      interpolation ``alpha * mask * src + (1 - mask) * cascade``.  Uses
      ``headpatched_generate_batch``.

    Args:
        prompt: Full text (prompt + generation) to analyse, OR a pre-tokenized
            ``TokenizerOutput``.  A string is tokenized here and the ``<code>``
            segment is auto-located.  A ``TokenizerOutput`` is used as-is — token
            positions then align exactly with the caller's own tokenization and
            the prompt is not re-tokenized, but the whole sequence is scoped (no
            string to find ``<code>`` in), so pass ``token_positions`` to restrict.
        mt: ModelandTokenizer.
        selected_heads: ``(layer, head_idx)`` pairs from the trained head mask.
        strategy: ``"additive"`` (default), ``"additive_norm"`` or
            ``"replace"``.  See above.
        steer_layer_range: ``(start, end)`` — layers ``range(start, end)``
            receive the additive injection.  **Additive strategy only.**
        batch_size: Token positions per generation call.
        alpha: Steering strength.  Additive: scale of the added vector.
            Additive-norm: added fraction of ``||h||`` (different units — do not
            reuse an ``"additive"`` alpha).  Replace: source-side interpolation
            weight.
        max_new_tokens: Max tokens generated per position.
        n_steer_tokens: How many forward passes apply a fresh patch.
            ``"all"`` (default) → patch every step.  An integer ``k`` → patch
            only the first ``k`` forward passes (step 0 = prefill counts as 1).
            Applies to both strategies.
        title: HTML visualization title.
        show: Display the HTML visualization inline when ``True``.
        goalscope_prompt: Override the default GoalScope dest prompt.
        include_noncode: When ``True``, also include the non-code tokens (the
            prompt text outside the ``<code>…</code>`` segment — e.g. the prefix
            / instruction) in the returned results and the visualization.  These
            positions are shown dimmed and are **not** goal-scoped: no steering
            source is extracted and no generation is run for them.  Extraction /
            generation still run only over the code segment.
        save_path: When set (e.g. ``"…/scope.html"``), write the rendered HTML to
            this path.  Saving happens regardless of ``show``.  ``None`` (default)
            saves nothing.
        token_positions: Absolute token indices (in the tokenization of ``prompt``) to
            extract + generate at.  Must fall inside the ``<code>`` segment.  ``None``
            (default) uses every code-segment token.  Restricting to a handful of
            positions (e.g. the anchors) skips generation for all others and is much
            faster.
        patch_interface: optional ``(mt, inputs, patches=None, model_kwargs=None) -> output`` — the
            same interface ``src.attention.get_attention_matrices`` accepts, typically one from
            :func:`make_head_replacement_patch_interface`.  When given, the SOURCE forward that reads
            ``selected_heads`` (the goal heads) runs under it, so an intervention on a DIFFERENT head
            set (e.g. the boundary heads, transplanted at the pivot) is applied FIRST and its effect
            is carried into the goal-head signal that is verbalized.  Mirrors the ``patch_interface``
            of ``get_attention_matrices``.  ``None`` (default) → no intervention; source extraction
            is unchanged.  Note the intervention's own positions are absolute indices into
            ``prompt``'s tokenization, exactly like ``token_positions``.

    Returns:
        List of dicts with keys ``"pos"``, ``"token"``, ``"patched_gen"`` and
        ``"scoped"`` (``True`` for code-segment tokens; ``False`` for non-code
        tokens when ``include_noncode=True``).  Code-segment entries come first
        in generation order; with ``include_noncode`` the full list is sorted by
        ``pos`` so the prompt reads left-to-right.
    """
    if strategy not in ("additive", "additive_norm", "replace"):
        raise ValueError(
            f"strategy must be 'additive', 'additive_norm' or 'replace', got {strategy!r}"
        )

    # Translate n_steer_tokens="all" → None for the underlying APIs
    n_steer = None if n_steer_tokens == "all" else int(n_steer_tokens)

    if goalscope_prompt is None:
        goalscope_prompt = _build_goalscope_prompt(mt)

    # Source tokenization — accept a raw ``str`` (tokenized here) or a pre-tokenized
    # ``TokenizerOutput`` (used as-is, so token positions align EXACTLY with the caller's own
    # tokenization and the prompt is not re-tokenized).  With a ``TokenizerOutput`` there is no
    # string to locate ``<code>`` in, so the whole sequence is scoped — pass ``token_positions`` to
    # restrict it.
    if isinstance(prompt, TokenizerOutput):
        src_tokenized = prompt
        if "offset_mapping" in src_tokenized:
            src_tokenized.pop("offset_mapping")
        src_ids = src_tokenized["input_ids"][0].tolist()
        text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]
        code_start, code_end = 0, len(src_ids)
    else:
        src_tokenized = prepare_input(
            prompts=prompt, tokenizer=mt, return_offsets_mapping=True
        )
        src_offsets = src_tokenized.pop("offset_mapping")[0]
        src_ids = src_tokenized["input_ids"][0].tolist()
        text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]

        # Find <code> … </code> segment; fall back to full sequence
        try:
            code_start, _ = find_token_range(
                string=prompt,
                substring="<code>",
                tokenizer=mt.tokenizer,
                occurrence=1,
                offset_mapping=src_offsets,
            )
            if prompt.count("</code>") > 1:
                _, code_end = find_token_range(
                    string=prompt,
                    substring="</code>",
                    tokenizer=mt.tokenizer,
                    occurrence=1,
                    offset_mapping=src_offsets,
                )
            else:
                code_end = len(src_ids)
        except Exception:
            code_start, code_end = 0, len(src_ids)

    # print(f"Scoping tokens from {code_start} to {code_end}")

    seg_positions = list(range(code_start, code_end))
    if token_positions is not None:
        # Restrict extraction + generation to the requested positions (must lie inside the scoped
        # segment). Steering-source extraction and generation both key off seg_positions, so this is
        # all that's needed to skip the other tokens — much faster.
        allowed = set(seg_positions)
        seg_positions = [int(p) for p in token_positions if int(p) in allowed]
        if not seg_positions:
            raise ValueError(
                "token_positions is empty after intersecting with the scoped segment "
                f"[{code_start}, {code_end})"
            )
    n_seg = len(seg_positions)

    # Tokenize GoalScope dest once
    dest_tok = prepare_input(prompts=goalscope_prompt, tokenizer=mt)
    dest_ids = dest_tok["input_ids"]
    dest_attn = dest_tok["attention_mask"]

    results: list[dict] = []

    # -----------------------------------------------------------------
    # Strategy-specific source extraction
    # -----------------------------------------------------------------
    if strategy in ("additive", "additive_norm"):
        # Aggregate selected heads into one residual-stream vector per position.
        # collect_head_steering_vecs returns (batch_size, n_seg, n_embd);
        # src_tokenized here has batch_size=1, so squeeze the leading dim.
        steering_vecs = collect_head_steering_vecs(
            mt=mt,
            src_tokenized=src_tokenized,
            selected_heads=selected_heads,
            token_positions=seg_positions,
            patch_interface=patch_interface,
        )[0]  # (n_seg, n_embd)

        # additive_norm's unit-direction normalisation happens inside
        # head_additive_generate, so `steering_vecs` stays raw here.

        layer_start, layer_end = steer_layer_range
        steer_layers = list(range(layer_start, layer_end))

    else:  # strategy == "replace"
        # Layers come from selected_heads; build a HeadMask over them.
        layers_replace = sorted({layer for (layer, _) in selected_heads})
        n_heads_model = mt.n_heads
        mask_module = HeadMask(
            n_layers_tested=len(layers_replace), n_heads=n_heads_model
        )
        with torch.no_grad():
            mask_module.values.zero_()
            for layer, h in selected_heads:
                li = layers_replace.index(layer)
                mask_module.values[li, h] = 1.0

        # Extract per-layer o_proj inputs at every segment position.
        # Shape per layer: (1, n_seg, n_embd); stack → (n_seg, n_layers, n_embd).
        src_o_inputs = get_o_proj_inputs_batch(
            mt,
            src_tokenized,
            layers_replace,
            token_positions=seg_positions,
            patch_interface=patch_interface,
        )
        src_head_vals_all = torch.stack(
            [src_o_inputs[layer][0] for layer in layers_replace], dim=1
        )  # (n_seg, n_layers, n_embd)

    # -----------------------------------------------------------------
    # Batched generation
    # -----------------------------------------------------------------
    for chunk_start in tqdm(range(0, n_seg, batch_size), desc=f"GoalScope[{strategy}]"):
        chunk_end = min(chunk_start + batch_size, n_seg)
        B = chunk_end - chunk_start

        tiled = TokenizerOutput(
            data=dict(
                input_ids=dest_ids.expand(B, -1).clone(),
                attention_mask=dest_attn.expand(B, -1).clone(),
            )
        )

        if strategy in ("additive", "additive_norm"):
            gens = head_additive_generate(
                mt=mt,
                dest_inputs=tiled,
                steering_vecs=steering_vecs[chunk_start:chunk_end],  # (B, n_embd), raw
                layers=steer_layers,
                max_new_tokens=max_new_tokens,
                strategy=strategy,
                n_steer_tokens=n_steer,
                alpha=alpha,
            )
        else:  # replace
            src_chunk = src_head_vals_all[
                chunk_start:chunk_end
            ]  # (B, n_layers, n_embd)
            gens = headpatched_generate(
                mt=mt,
                dest_inputs=tiled,
                src_o_proj_inps=src_chunk,
                mask=mask_module,
                layers=layers_replace,
                max_new_tokens=max_new_tokens,
                n_steer_tokens=n_steer,
                alpha=alpha,
            )

        for i in range(B):
            results.append(
                {
                    "pos": seg_positions[chunk_start + i],
                    "token": text_tokens[seg_positions[chunk_start + i]],
                    "patched_gen": gens[i],
                    "scoped": True,
                }
            )

    # Optionally fold in the non-code tokens (everything outside [code_start, code_end))
    # as display-only, non-scoped context — no source extraction / generation for them.
    if include_noncode:
        seg_set = set(seg_positions)
        noncode = [
            {"pos": p, "token": text_tokens[p], "patched_gen": "", "scoped": False}
            for p in range(len(src_ids))
            if p not in seg_set
        ]
        results = sorted(results + noncode, key=lambda r: r["pos"])

    if show:
        show_patchscope(results, title=title, save_path=save_path)
    elif save_path is not None:
        html_str = render_patchscope_html(results, title=title)
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(html_str)
        print(f"Saved -> {save_path}")

    return results


def show_patchscope(
    results: list[dict],
    title: str = "PatchScope",
    save_path: Optional[str] = None,
) -> None:
    """Display patchscope results inline in a Jupyter/VSCode notebook.

    Optionally saves the HTML to ``save_path``.
    Uses ``display(HTML(...))`` which works in both Jupyter and VSCode notebook
    WebView (``<iframe srcdoc>`` does not render in VSCode).
    """
    from IPython.display import HTML, display

    html_str = render_patchscope_html(results, title=title)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(html_str)
        print(f"Saved -> {save_path}")
    display(HTML(html_str))
