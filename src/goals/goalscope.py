"""PatchScope generation helpers and HTML visualization utilities.

This module provides:
- Greedy generation helpers (plain and patched, single and batched)
- HTML visualization for patchscope results: hover-over tooltip per token
"""

from __future__ import annotations

import html as _html
import keyword as _kw
from typing import Literal, Optional, Union

import torch

from src.goals.head_loc_attn import forward_with_src_replacement

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
    positions: Optional[list[int]], L_prompt: int, default_last: bool = True,
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
    layers: list[int],
    max_new_tokens: int,
    stop_ids: Optional[set[int]] = None,
    n_steer_tokens: Optional[int] = None,
    patch_input_positions: Optional[list[int]] = None,
    alpha: float = 1.0,
) -> str:
    """Greedy generation with head-replacement patching — KV-cache aware.

    Decoding pattern:
        Step 0 (prefill): full forward over the prompt, patches every position
            in ``patch_input_positions`` (default: last prompt token only).
            Returns ``past_key_values`` so the patched K/V (at layers above the
            patched o_proj) are written into the cache for those positions.
        Step >= 1 (decode): 1-token forward reusing the cache.  If still inside
            the steer window, the new token's o_proj input is patched too and
            its K/V get cached.  Past patched K/V remain in the cache.

    ``n_steer_tokens`` controls how many *forward passes* apply a fresh patch.
    Step 0 counts as the first such pass; ``n_steer_tokens=1`` patches only the
    prefill.  ``None`` patches every step (legacy behavior).

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict (input_ids, attention_mask).
        src_o_proj_inps: (n_layers, n_embd) source o_proj inputs stacked across layers.
        mask: HeadMask controlling per-head interpolation weights.
        layers: Layer indices matching src_o_proj_inps axis 0.
        max_new_tokens: Maximum tokens to generate.
        stop_ids: Token ids that halt generation.
        n_steer_tokens: Apply a fresh patch only on the first ``n_steer_tokens``
            forward passes (counting the prefill).  ``None`` (default) patches
            every step.
        patch_input_positions: Positions within the input prompt to patch during
            the prefill.  Negative indices supported (``-1`` = last token).
            Duplicates removed.  ``None`` (default) → ``[L_prompt - 1]`` (legacy
            behavior: patch only the last prompt token).  All positions are
            validated to lie in ``[0, L_prompt)`` after normalization.

            Note: the same ``src_o_proj_inps`` snapshot is broadcast across every
            position in this list; per-position source variation is not
            supported.  See ``forward_with_src_replacement`` for details.

    Returns:
        Decoded generated string.
    """
    _stop = stop_ids or set()
    if len(_stop) == 0:
        _stop = {mt.tokenizer.eos_token_id}
    input_ids = dest_inputs["input_ids"].clone()
    attention_mask = dest_inputs["attention_mask"].clone()
    L_prompt = input_ids.shape[1]
    prefill_positions = _normalize_input_positions(patch_input_positions, L_prompt)

    generated = []
    past_key_values = None
    with torch.no_grad():
        for step in range(max_new_tokens):
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
                    mt, inputs, src_o_proj_inps, mask, layers,
                    alpha=alpha,
                    patch_positions=positions_this_step,
                    return_past_key_values=True,
                )
            else:
                out = mt._model(**inputs)
                logits = out.logits[0, -1, :].float()  # (vocab,)
                past_key_values = out.past_key_values

            next_tok = int(logits.argmax())
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


def headpatched_generate_batch(
    mt,
    dest_inputs: dict,
    src_o_proj_inps_batch: torch.Tensor,
    mask,
    layers: list[int],
    max_new_tokens: int,
    stop_ids: Optional[set[int]] = None,
    n_steer_tokens: Optional[int] = None,
    patch_input_positions: Optional[list[int]] = None,
    alpha: float = 1.0,
) -> list[str]:
    """Batched greedy generation with head-replacement patching — KV-cache aware.

    Runs B independent patched generations in parallel — one per source token
    position.  Each item i in the batch uses src_o_proj_inps_batch[i] as its patch
    while sharing the same destination prompt.

    Decoding pattern:
        Step 0 (prefill): full forward over the prompt, patches every position
            in ``patch_input_positions`` (default: last prompt token only).
            Returns ``past_key_values`` so the patched K/V (at layers above the
            patched o_proj) are written into the cache for those positions.
        Step >= 1 (decode): 1-token forward reusing the cache.  If still inside
            the steer window, the new token's o_proj input is patched too and
            its K/V get cached.  Past patched K/V remain in the cache.

    Args:
        dest_inputs: Tokenized destination prompt (batch=1); tiled internally to B.
        src_o_proj_inps_batch: (B, n_layers, n_embd) — one row per source position.
        mask: HeadMask with per-head interpolation weights.
        layers: Layer indices matching src_o_proj_inps_batch axis 1.
        max_new_tokens: Maximum tokens to generate per item.
        stop_ids: Token ids that halt generation for an item.
        n_steer_tokens: Apply a fresh patch only on the first ``n_steer_tokens``
            forward passes (counting the prefill).  After that, plain forward
            passes continue, but earlier patched K/V stay in the cache and
            keep influencing subsequent attention.  ``None`` (default) patches
            every step.
        patch_input_positions: Positions within the input prompt to patch during
            the prefill.  Negative indices supported (``-1`` = last token).
            Duplicates removed.  ``None`` (default) → ``[L_prompt - 1]`` (legacy
            behavior: patch only the last prompt token).  All positions are
            validated to lie in ``[0, L_prompt)`` after normalization.

            Note: the same ``src_o_proj_inps_batch[i]`` snapshot is broadcast
            across every position in this list for item ``i``; per-position
            source variation is not supported.

    Returns:
        list[str] of length B — decoded greedy continuations.
    """
    _stop = stop_ids or set()
    if len(_stop) == 0:
        _stop = {mt.tokenizer.eos_token_id}
    B = src_o_proj_inps_batch.shape[0]
    input_ids = dest_inputs["input_ids"].expand(B, -1).clone()
    attention_mask = dest_inputs["attention_mask"].expand(B, -1).clone()
    L_prompt = input_ids.shape[1]
    prefill_positions = _normalize_input_positions(patch_input_positions, L_prompt)

    generated = [[] for _ in range(B)]
    finished = [False] * B
    past_key_values = None

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
                    mt, inputs, src_o_proj_inps_batch, mask, layers,
                    alpha=alpha,
                    patch_positions=positions_this_step,
                    return_past_key_values=True,
                )  # (B, vocab)
            else:
                out = mt._model(**inputs)
                logits = out.logits[:, -1, :].float()  # (B, vocab)
                past_key_values = out.past_key_values
            next_toks = logits.argmax(dim=-1)  # (B,)

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

    return [
        mt.tokenizer.decode(g, skip_special_tokens=True).rstrip() for g in generated
    ]


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
            spans.append(f'<span class="ps-noncode" style="opacity:0.4">{tok_html}</span>')
            continue
        tip_text = f'pos {r["pos"]} {r["token"]!r}: {r["patched_gen"].strip() or "(empty)"}'
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


def _build_goalscope_prompt(mt) -> str:
    """Build the GoalScope destination prompt ("What are you doing at this line?")."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant proficient in programming."},
        {"role": "user", "content": "What are you doing at this line of the code. Keep your answer as brief as possible."},
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


def apply_goalscope(
    prompt: str,
    *,
    mt,
    selected_heads: list[tuple[int, int]],
    strategy: Literal["additive", "replace"] = "additive",
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
    - ``"replace"``: surgically replace the selected heads' o_proj inputs at the
      destination's last token with the corresponding source o_proj inputs.
      Layers come from ``selected_heads``; ``steer_layer_range`` is ignored.
      ``alpha`` controls the source-side weight inside the per-head
      interpolation ``alpha * mask * src + (1 - mask) * cascade``.  Uses
      ``headpatched_generate_batch``.

    Args:
        prompt: Full text (prompt + generation) to analyse.
        mt: ModelandTokenizer.
        selected_heads: ``(layer, head_idx)`` pairs from the trained head mask.
        strategy: ``"additive"`` (default) or ``"replace"``.  See above.
        steer_layer_range: ``(start, end)`` — layers ``range(start, end)``
            receive the additive injection.  **Additive strategy only.**
        batch_size: Token positions per generation call.
        alpha: Steering strength.  Additive: scale of the added vector.
            Replace: source-side interpolation weight.
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

    Returns:
        List of dicts with keys ``"pos"``, ``"token"``, ``"patched_gen"`` and
        ``"scoped"`` (``True`` for code-segment tokens; ``False`` for non-code
        tokens when ``include_noncode=True``).  Code-segment entries come first
        in generation order; with ``include_noncode`` the full list is sorted by
        ``pos`` so the prompt reads left-to-right.
    """
    from tqdm.auto import tqdm

    from src.functional import PatchSpec, generate_with_patch
    from src.goals.head_loc_attn import (
        HeadMask,
        collect_head_steering_vecs,
        get_o_proj_inputs_batch,
    )
    from src.tokens import TokenizerOutput, find_token_range, prepare_input

    if strategy not in ("additive", "replace"):
        raise ValueError(f"strategy must be 'additive' or 'replace', got {strategy!r}")

    # Translate n_steer_tokens="all" → None for the underlying APIs
    n_steer = None if n_steer_tokens == "all" else int(n_steer_tokens)

    if goalscope_prompt is None:
        goalscope_prompt = _build_goalscope_prompt(mt)

    # Tokenize with offsets
    src_tokenized = prepare_input(prompts=prompt, tokenizer=mt, return_offsets_mapping=True)
    src_offsets = src_tokenized.pop("offset_mapping")[0]
    src_ids = src_tokenized["input_ids"][0].tolist()
    text_tokens = [mt.tokenizer.decode([t]) for t in src_ids]

    # Find <code> … </code> segment; fall back to full sequence
    try:
        code_start, _ = find_token_range(
            string=prompt, substring="<code>", tokenizer=mt.tokenizer,
            occurrence=1, offset_mapping=src_offsets,
        )
        if prompt.count("</code>") > 1:
            _, code_end = find_token_range(
                string=prompt, substring="</code>", tokenizer=mt.tokenizer,
                occurrence=1, offset_mapping=src_offsets,
            )
        else:
            code_end = len(src_ids)
    except Exception:
        code_start, code_end = 0, len(src_ids)

    seg_positions = list(range(code_start, code_end))
    if token_positions is not None:
        # Restrict extraction + generation to the requested positions (must lie inside the
        # <code> segment). Steering-source extraction and generation both key off seg_positions,
        # so this is all that's needed to skip the other code tokens — much faster.
        allowed = set(seg_positions)
        seg_positions = [int(p) for p in token_positions if int(p) in allowed]
        if not seg_positions:
            raise ValueError(
                "token_positions is empty after intersecting with the <code> segment "
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
    if strategy == "additive":
        # Aggregate selected heads into one residual-stream vector per position.
        # collect_head_steering_vecs returns (batch_size, n_seg, n_embd);
        # src_tokenized here has batch_size=1, so squeeze the leading dim.
        steering_vecs = collect_head_steering_vecs(
            mt=mt, src_tokenized=src_tokenized,
            selected_heads=selected_heads, token_positions=seg_positions,
        )[0]  # (n_seg, n_embd)

        layer_start, layer_end = steer_layer_range
        steer_layers = list(range(layer_start, layer_end))

    else:  # strategy == "replace"
        # Layers come from selected_heads; build a HeadMask over them.
        layers_replace = sorted({layer for (layer, _) in selected_heads})
        n_heads_model = mt.model.config.num_attention_heads
        mask_module = HeadMask(n_layers_tested=len(layers_replace), n_heads=n_heads_model)
        with torch.no_grad():
            mask_module.values.zero_()
            for (layer, h) in selected_heads:
                li = layers_replace.index(layer)
                mask_module.values[li, h] = 1.0

        # Extract per-layer o_proj inputs at every segment position.
        # Shape per layer: (1, n_seg, n_embd); stack → (n_seg, n_layers, n_embd).
        src_o_inputs = get_o_proj_inputs_batch(
            mt, src_tokenized, layers_replace, token_positions=seg_positions,
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

        tiled = TokenizerOutput(data=dict(
            input_ids=dest_ids.expand(B, -1).clone(),
            attention_mask=dest_attn.expand(B, -1).clone(),
        ))

        if strategy == "additive":
            sv_batch = alpha * steering_vecs[chunk_start:chunk_end]  # (B, n_embd)
            patches = [
                PatchSpec(
                    location=(mt.layer_name_format.format(li), -1),
                    patch=sv_batch, strategy="add", on_pass="all",
                )
                for li in steer_layers
            ]
            gens = generate_with_patch(
                mt, tiled, max_new_tokens=max_new_tokens, patches=patches,
                n_steer_tokens=n_steer,
                do_sample=False, patch_interface="baukit",
                remove_prefix=True, skip_special_tokens=True,
            )
        else:  # replace
            src_chunk = src_head_vals_all[chunk_start:chunk_end]  # (B, n_layers, n_embd)
            gens = headpatched_generate_batch(
                mt=mt, dest_inputs=tiled,
                src_o_proj_inps_batch=src_chunk,
                mask=mask_module, layers=layers_replace,
                max_new_tokens=max_new_tokens,
                n_steer_tokens=n_steer,
                alpha=alpha,
            )

        for i in range(B):
            results.append({
                "pos": seg_positions[chunk_start + i],
                "token": text_tokens[seg_positions[chunk_start + i]],
                "patched_gen": gens[i],
                "scoped": True,
            })

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
