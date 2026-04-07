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
    _stop = stop_ids or set()
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
            logits = forward_with_src_replacement(mt, inputs, src_head_vals, mask, layers)
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

    return [mt.tokenizer.decode(g, skip_special_tokens=True).rstrip() for g in generated]


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
    position: relative; cursor: default;
    border-radius: 3px; padding: 1px 0;
    transition: background 0.12s;
}
.ps-tok:hover { background: rgba(255,210,80,0.22); outline: 1px solid rgba(255,210,80,0.4); }
.ps-tok .ps-tip {
    display: none;
    position: absolute;
    bottom: 130%; left: 50%; transform: translateX(-50%);
    background: #1a1a2e; border: 1px solid #555;
    color: #e8e8e8; font-size: 12px;
    padding: 6px 10px; border-radius: 5px;
    white-space: pre-wrap; min-width: 240px; max-width: 440px;
    z-index: 9999; box-shadow: 0 4px 14px rgba(0,0,0,0.6);
    pointer-events: none;
}
.ps-tok:hover .ps-tip { display: block; }
.ps-legend { font-size: 11px; color: #777; margin-bottom: 8px; }
"""

_PY_KEYWORDS = set(_kw.kwlist) | {"True", "False", "None"}
_PY_BUILTINS = {
    "print", "len", "range", "int", "str", "float", "list",
    "dict", "set", "tuple", "bool", "type", "isinstance",
}
_OPERATORS = {
    "+", "-", "*", "/", "//", "**", "%", "=", "==", "!=",
    "<", ">", "<=", ">=", "+=", "-=", "&", "|", "^", "~",
    "<<", ">>", "->", "...",
}
_PUNCTUATION = {"(", ")", "[", "]", "{", "}", ",", ":", ";", "."}
_STR_PREFIXES = (
    '"', "'", 'f"', "f'", 'b"', "b'", 'r"', "r'",
    '"""', "'''", 'f"""', "f'''",
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
        tok_html = _html.escape(r["token"])
        tip_html = _html.escape(r["patched_gen"].strip() or "(empty)")
        cls = ("ps-tok " + _prism_class(r["token"])).strip()
        spans.append(
            f'<span class="{cls}">'
            + tok_html
            + f'<span class="ps-tip">pos {r["pos"]}[{r["token"]!r}]: {tip_html}</span>'
            + "</span>"
        )
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
        + "\n</div>\n</body>\n</html>"
    )


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
