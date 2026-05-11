"""PatchScope generation helpers and HTML visualization utilities.

This module provides:
- Greedy generation helpers (plain and patched, single and batched)
- HTML visualization for patchscope results: hover-over tooltip per token
"""

from __future__ import annotations

import html as _html
import keyword as _kw
from typing import Optional

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


def patched_generate(
    mt,
    dest_inputs: dict,
    src_head_vals: torch.Tensor,
    mask,
    layers: list[int],
    max_new_tokens: int,
    stop_ids: Optional[set[int]] = None,
) -> str:
    """Greedy generation with head-replacement patching at every token position.

    Args:
        mt: ModelandTokenizer.
        dest_inputs: Tokenized destination prompt dict (input_ids, attention_mask).
        src_head_vals: (n_layers, n_embd) source o_proj inputs stacked across layers.
        mask: HeadMask controlling per-head interpolation weights.
        layers: Layer indices matching src_head_vals axis 0.
        max_new_tokens: Maximum tokens to generate.
        stop_ids: Token ids that halt generation.

    Returns:
        Decoded generated string.
    """
    _stop = stop_ids or set()
    input_ids = dest_inputs["input_ids"].clone()
    attention_mask = dest_inputs["attention_mask"].clone()
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits = forward_with_src_replacement(
                mt, inputs, src_head_vals, mask, layers
            )
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


def patched_generate_batch(
    mt,
    dest_inputs: dict,
    src_head_vals_batch: torch.Tensor,
    mask,
    layers: list[int],
    max_new_tokens: int,
    stop_ids: Optional[set[int]] = None,
) -> list[str]:
    """Batched greedy generation with head-replacement patching.

    Runs B independent patched generations in parallel — one per source token
    position.  Each item i in the batch uses src_head_vals_batch[i] as its patch
    while sharing the same destination prompt.

    Args:
        dest_inputs: Tokenized destination prompt (batch=1); tiled internally to B.
        src_head_vals_batch: (B, n_layers, n_embd) — one row per source position.
        mask: HeadMask with per-head interpolation weights.
        layers: Layer indices matching src_head_vals_batch axis 1.
        max_new_tokens: Maximum tokens to generate per item.
        stop_ids: Token ids that halt generation for an item.

    Returns:
        list[str] of length B — decoded greedy continuations.
    """
    _stop = stop_ids or set()
    B = src_head_vals_batch.shape[0]
    input_ids = dest_inputs["input_ids"].expand(B, -1).clone()
    attention_mask = dest_inputs["attention_mask"].expand(B, -1).clone()

    generated = [[] for _ in range(B)]
    finished = [False] * B

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if all(finished):
                break
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits = forward_with_src_replacement(
                mt, inputs, src_head_vals_batch, mask, layers
            )  # (B, vocab)
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
    steer_layer_range: tuple[int, int] = (35, 46),
    batch_size: int = 16,
    alpha: float = 0.2,
    max_new_tokens: int = 64,
    title: str = "GoalScope",
    show: bool = True,
) -> list[dict]:
    """Run GoalScope on a single prompt.

    Extracts goal signals from the ``<code>`` segment of ``prompt`` and injects
    them into the GoalScope probe ("What are you doing at this line?") to produce
    a verbalization per token position.

    Args:
        prompt: Full text (prompt + generation) to analyse.
        mt: ModelandTokenizer.
        selected_heads: ``(layer, head_idx)`` pairs from the trained head mask.
        steer_layer_range: ``(start, end)`` — layers ``range(start, end)`` receive
            the additive steering injection.
        batch_size: Token positions per ``generate_with_patch`` call.
        alpha: Steering strength multiplier.
        max_new_tokens: Max tokens generated per position.
        title: HTML visualization title.
        show: Display the HTML visualization inline when ``True``.

    Returns:
        List of dicts with keys ``"pos"``, ``"token"``, ``"patched_gen"`` — one
        entry per token position inside the code segment.
    """
    from tqdm.auto import tqdm

    from src.functional import PatchSpec, generate_with_patch
    from src.goals.head_loc_attn import collect_head_steering_vecs
    from src.tokens import TokenizerOutput, find_token_range, prepare_input

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
        if "</code>" in prompt:
            _, code_end = find_token_range(
                string=prompt, substring="</code>", tokenizer=mt.tokenizer,
                occurrence=1, offset_mapping=src_offsets,
            )
        else:
            code_end = len(src_ids)
    except Exception:
        code_start, code_end = 0, len(src_ids)

    seg_positions = list(range(code_start, code_end))

    # Steering vectors: (n_seg, n_embd)
    steering_vecs = collect_head_steering_vecs(
        mt=mt, src_tokenized=src_tokenized,
        selected_heads=selected_heads, token_positions=seg_positions,
    )

    # Tokenize GoalScope dest once
    dest_tok = prepare_input(prompts=goalscope_prompt, tokenizer=mt)
    dest_ids = dest_tok["input_ids"]
    dest_attn = dest_tok["attention_mask"]

    layer_start, layer_end = steer_layer_range
    steer_layers = list(range(layer_start, layer_end))
    n_seg = len(seg_positions)
    results: list[dict] = []

    for chunk_start in tqdm(range(0, n_seg, batch_size), desc="GoalScope"):
        chunk_end = min(chunk_start + batch_size, n_seg)
        B = chunk_end - chunk_start

        sv_batch = alpha * steering_vecs[chunk_start:chunk_end]  # (B, n_embd)

        tiled = TokenizerOutput(data=dict(
            input_ids=dest_ids.expand(B, -1).clone(),
            attention_mask=dest_attn.expand(B, -1).clone(),
        ))

        patches = [
            PatchSpec(
                location=(mt.layer_name_format.format(li), -1),
                patch=sv_batch, strategy="add", on_pass="all",
            )
            for li in steer_layers
        ]

        gens = generate_with_patch(
            mt, tiled, max_new_tokens=max_new_tokens, patches=patches,
            do_sample=False, patch_interface="baukit",
            remove_prefix=True, skip_special_tokens=True,
        )

        for i in range(B):
            results.append({
                "pos": seg_positions[chunk_start + i],
                "token": text_tokens[seg_positions[chunk_start + i]],
                "patched_gen": gens[i],
            })

    if show:
        show_patchscope(results, title=title)

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
