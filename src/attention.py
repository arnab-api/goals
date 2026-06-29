import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from circuitsvis.tokens import colored_tokens
from dataclasses_json import DataClassJsonMixin
from IPython.display import display

from src.functional import (
    PatchSpec,
    patch_with_baukit,
    prepare_input,
)
from src.models import ModelandTokenizer
from src.tokens import find_token_range
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


def _safe_token(t: str) -> str:
    """Replace HTML angle-bracket characters with fullwidth Unicode look-alikes.

    circuitsvis passes tokens as JSON to a React component which renders them
    as text nodes.  Raw ``<`` / ``>`` characters inside the JSON get parsed as
    HTML tags by the browser before React runs, corrupting special tokens such
    as ``<|im_start|>``.  Fullwidth variants (U+FF1C / U+FF1E) are visually
    identical but are never treated as HTML markup.
    """
    return t.replace("<", "\uff1c").replace(">", "\uff1e")


@dataclass(frozen=False)
class AttentionInformation(DataClassJsonMixin):
    tokenized_prompt: list[str]
    attention_matrices: np.ndarray
    logits: torch.Tensor | None = None

    def _init__(
        self, prompt: str, tokenized_prompt: list[str], attention_matrices: torch.tensor
    ):
        assert len(tokenized_prompt) == attention_matrices.shape[-1], (
            "Tokenized prompt and attention matrices must have the same length"
        )
        assert len(attention_matrices.shape) == 4, (
            "Attention matrices must be of shape (layers, heads, tokens, tokens)"
        )
        assert attention_matrices.shape[-1] == attention_matrices.shape[-2], (
            "Attention matrices must be square"
        )

        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt
        self.attention_matrices = attention_matrices

    def get_attn_matrix(self, layer: int, head: int) -> np.ndarray:
        return self.attention_matrices[layer, head]

    def get_layer_attn(self, layer: int, agg: str = "mean") -> np.ndarray:
        """Aggregate attention across all heads for a single layer.

        Args:
            layer: Layer index.
            agg: Aggregation method — ``"mean"`` (default) or ``"sum"``.

        Returns:
            ``(seq, seq)`` float32 array of aggregated attention weights.
        """
        if agg == "mean":
            return self.attention_matrices[layer].mean(axis=0)
        elif agg == "sum":
            return self.attention_matrices[layer].sum(axis=0)
        else:
            raise ValueError(f"Unknown agg={agg!r}; expected 'mean' or 'sum'")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.inference_mode()
def get_attention_matrices(
    input: str | list[str] | TokenizerOutput,
    mt: ModelandTokenizer,
    value_weighted: bool = False,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    patch_interface: callable = patch_with_baukit,
) -> "AttentionInformation | list[AttentionInformation]":
    """
    Parameters:
        input: str, list[str], or TokenizerOutput.
            - str            => single prompt; returns one AttentionInformation
            - list[str]      => batched prompts; returns list[AttentionInformation]
            - TokenizerOutput=> already tokenized (batch dim required); returns list (or single
              AttentionInformation if batch size is 1).
        mt: ModelandTokenizer, model and tokenizer
        value_weighted: bool.
            - False => returns attention masks after softmax (the attention pattern produced inside the model)
            - True  => weights attention by ||v_k|| to reflect the actual contribution of source tokens
    Returns:
        AttentionInformation if a single str was passed, else list[AttentionInformation].
        Each AttentionInformation has attention_matrices of shape (layers, heads, tokens, tokens).
    """
    single_input = False
    if isinstance(input, str):
        input = prepare_input(prompts=input, tokenizer=mt)
        single_input = True
    elif isinstance(input, list):
        input = prepare_input(prompts=input, tokenizer=mt)
    else:
        assert isinstance(input, TokenizerOutput), (
            "input must be a str, list[str], or TokenizerOutput object"
        )
        single_input = input.input_ids.shape[0] == 1

    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    output = patch_interface(
        mt=mt,
        inputs=input,
        patches=patches if patches is not None else [],
        model_kwargs=dict(output_attentions=True, use_cache=True),
    )

    # logits: (batch, seq, vocab) -> last-position logits per sample
    logits = output.logits[:, -1, :]  # (batch, vocab)

    # attentions: list of (batch, heads, seq, seq) per layer
    output.attentions = [attn.cuda() for attn in output.attentions]
    # stack along new layer axis -> (layers, batch, heads, seq, seq) -> (batch, layers, heads, seq, seq)
    attentions = torch.stack(output.attentions, dim=0).permute(1, 0, 2, 3, 4).contiguous()

    if value_weighted:
        pkv = output.past_key_values
        # transformers ≥ 4.36 returns a Cache object; the internal structure
        # changed again in later versions.  Try the known access patterns:
        #   - new layered API  (DynamicCache.layers[i].values)
        #   - mid-era API      (.value_cache list)
        #   - legacy tuple     ([i][1])
        if hasattr(pkv, "layers") and hasattr(pkv.layers[0], "values"):
            value_list = [layer.values.cuda() for layer in pkv.layers]
        elif hasattr(pkv, "value_cache"):
            value_list = [v.cuda() for v in pkv.value_cache]
        else:
            value_list = [pkv[i][1].cuda() for i in range(mt.config.num_hidden_layers)]
        # each layer value: (batch, kv_heads, seq, head_dim)
        values = torch.stack(value_list, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        # values: (batch, layers, kv_heads, seq, head_dim)
        n_rep = mt.config.num_attention_heads // mt.config.num_key_value_heads
        b, L, kvh, s, d = values.shape
        values = repeat_kv(values.reshape(b * L, kvh, s, d), n_rep=n_rep)
        values = values.reshape(b, L, mt.config.num_attention_heads, s, d)
        # attentions: (b, L, H, q, s); value norms: (b, L, H, s) -> broadcast over q
        attentions = torch.einsum("blhqs,blhs->blhqs", attentions, values.norm(dim=-1))

    attentions_np = attentions.detach().cpu().to(torch.float32).numpy()
    logits_cpu = logits.detach().cpu()
    batch_size = input.input_ids.shape[0]

    infos = [
        AttentionInformation(
            tokenized_prompt=[
                _safe_token(mt.tokenizer.decode(tok)) for tok in input.input_ids[i]
            ],
            attention_matrices=attentions_np[i],
            logits=logits_cpu[i],
        )
        for i in range(batch_size)
    ]
    return infos[0] if single_input else infos


def visualize_attn_matrix(
    attn_matrix: torch.Tensor,
    tokens: list[str],
    q_index: int = -1,
    start_from: int = 0,
    end_at: Optional[int] = None,
    vis_args: dict = {},
):
    assert len(tokens) == attn_matrix.shape[-1]
    if "start_from" in vis_args:
        start_from = vis_args.pop("start_from")
    if "end_at" in vis_args:
        end_at = vis_args.pop("end_at")
    if "q_index" in vis_args:
        q_index = vis_args.pop("q_index")
    attn_matrix = (
        attn_matrix.squeeze()[q_index][start_from:]
        if end_at is None
        else attn_matrix.squeeze()[q_index][start_from:end_at]
    )
    tokens = tokens[start_from:] if end_at is None else tokens[start_from:end_at]
    tokens = [_safe_token(t) for t in tokens]
    assert len(tokens) == attn_matrix.shape[0], (
        f"Shape mismatch: {len(tokens)=} | {attn_matrix.shape=}"
    )
    display(colored_tokens(tokens=tokens, values=attn_matrix, **vis_args))


def visualize_average_attn_matrix(
    mt: ModelandTokenizer,
    attn_matrices: dict,
    prompt: str,
    tokenized: Optional[TokenizerOutput] = None,
    layer_window: list | None = None,
    q_index: int = -1,
    remove_bos: bool = True,
    start_from: int | str | None = None,
):
    if tokenized is None:
        tokenized = prepare_input(
            prompts=prompt, tokenizer=mt, return_offsets_mapping=True
        )
    if start_from is None:
        start_from = (
            1
            if remove_bos and tokenized["input_ids"][0][0] == mt.tokenizer.bos_token_id
            else 0
        )
    elif isinstance(start_from, str):
        offset_mapping = (
            tokenized.pop("offset_mapping")[0]
            if "offset_mapping" in tokenized
            else None
        )
        start_from = (
            find_token_range(
                string=prompt.prompt,
                substring="#",
                tokenizer=mt,
                offset_mapping=offset_mapping,
                occurrence=-1,
            )[1]
            - 1
        )

    # print(f"{start_from=}")

    for layer in layer_window:
        print(f"{layer=}")
        if isinstance(attn_matrices, AttentionInformation):
            avg_attn_module_matrix = torch.Tensor(
                attn_matrices.attention_matrices[layer]
            ).mean(dim=0)[q_index]
        else:
            avg_attn_module_matrix = torch.stack(
                [
                    attn_matrices[layer][h_idx].squeeze()
                    for h_idx in range(mt.config.num_attention_heads)
                ]
            ).mean(dim=0)[q_index]

        # print(avg_attn_module_matrix.shape)

        tokens = [
            mt.tokenizer.decode(t, skip_special_tokens=False)
            for t in tokenized["input_ids"][0]
        ][start_from:]
        tokens = [_safe_token(t) for t in tokens]

        display(
            colored_tokens(tokens=tokens, values=avg_attn_module_matrix[start_from:])
        )
        print("-" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive attention visualization (standalone HTML, no JS deps).
# ──────────────────────────────────────────────────────────────────────────────

_GLYPH_NL = "↵"
_GLYPH_TAB = "⇥"
_GLYPH_DOT = "·"


def _display_tok(tok: str) -> str:
    """Un-escape _safe_token HTML subs and replace whitespace with visible glyphs."""
    tok = tok.replace("＜", "<").replace("＞", ">")
    tok = tok.replace("\n", _GLYPH_NL).replace("\r", _GLYPH_NL)
    tok = tok.replace("\t", _GLYPH_TAB)
    if tok.strip() == "" and len(tok) > 0:
        tok = _GLYPH_DOT * len(tok)
    return tok


def _is_newline_tok(tok: str) -> bool:
    return "\n" in tok or "\r" in tok


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def aggregate_attention(
    attn_info: AttentionInformation,
    selected_heads: list[tuple[int, int]],
    agg: str = "mean",
) -> np.ndarray:
    """Aggregate attention across a chosen set of (layer, head) pairs.

    Args:
        attn_info: AttentionInformation holding (layers, heads, seq, seq).
        selected_heads: list of (layer, head) tuples to aggregate over.
        agg: "mean" | "sum" | "max" over the head axis.

    Returns:
        (seq, seq) float32 ndarray.
    """
    if len(selected_heads) == 0:
        raise ValueError("selected_heads must be non-empty")
    rows = np.stack(
        [attn_info.attention_matrices[layer, head] for (layer, head) in selected_heads],
        axis=0,
    ).astype(np.float32)
    if agg == "mean":
        return rows.mean(axis=0)
    if agg == "sum":
        return rows.sum(axis=0)
    if agg == "max":
        return rows.max(axis=0)
    raise ValueError(f"Unknown agg={agg!r}; expected 'mean' | 'sum' | 'max'")


def interactive_attention_html(
    attn_info: AttentionInformation,
    selected_heads: list[tuple[int, int]],
    start_from: int = 0,
    end_at: Optional[int] = None,
    agg: str = "mean",
    normalize: str = "row_max",
    cmap: str = "Blues",
    title: str = "Interactive attention",
    save_path: Optional[str] = None,
    intervened_positions: Optional[list[int]] = None,
) -> str:
    """Build a standalone interactive HTML for attention patterns.

    Hover over a token (query q) to color tokens k ≤ q by attention A[q, k].
    Tokens k > q (future positions) are greyed out. Click a token to pin/unpin
    the view; click again to release.

    Args:
        attn_info: AttentionInformation from get_attention_matrices.
        selected_heads: list of (layer, head) tuples to aggregate.
        start_from / end_at: token index slice applied to both axes.
        agg: head-axis aggregation ("mean" | "sum" | "max").
        normalize: per-row max ("row_max") or global max ("global_max") or "none".
        cmap: matplotlib colormap name for non-future cells.
        title: HTML <title> and header.
        save_path: if given, write the HTML to disk.
        intervened_positions: token positions (in the FULL, un-sliced sequence — same index
            space as ``attn_info.tokenized_prompt``; negatives allowed) where an intervention
            (steering / head patch) was applied.  Marked with a persistent purple underline +
            badge so the viz shows where the transplant landed.  Positions outside the
            ``[start_from, end_at)`` slice are dropped.

    Returns:
        The HTML string.
    """
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    A = aggregate_attention(attn_info, selected_heads, agg=agg)  # (S, S)
    if end_at is None:
        end_at = A.shape[0]
    A = A[start_from:end_at, start_from:end_at]
    tokens = attn_info.tokenized_prompt[start_from:end_at]
    n = A.shape[0]
    assert len(tokens) == n, f"token/attn length mismatch: {len(tokens)} vs {n}"

    # Intervened token positions: map from full-sequence (possibly negative) index space into
    # the displayed slice [start_from, end_at); drop any that fall outside the window.
    seq_full = len(attn_info.tokenized_prompt)
    interv_set: set[int] = set()
    for p in intervened_positions or []:
        pf = p if p >= 0 else seq_full + p
        if start_from <= pf < end_at:
            interv_set.add(pf - start_from)

    # Mask future positions (causal): A[q, k] for k > q is set to NaN-marker (-1).
    mask_future = np.triu(np.ones_like(A, dtype=bool), k=1)
    A = np.where(mask_future, 0.0, A)

    if normalize == "row_max":
        row_max = A.max(axis=1, keepdims=True)
        row_max = np.where(row_max <= 0, 1.0, row_max)
        Anorm = A / row_max
    elif normalize == "global_max":
        gm = A.max() if A.max() > 0 else 1.0
        Anorm = A / gm
    elif normalize == "none":
        Anorm = A.copy()
    else:
        raise ValueError(f"Unknown normalize={normalize!r}")

    cmap_obj = mcm.get_cmap(cmap)

    # Pre-compute color (#rrggbb) and text-color (black/white) per (q, k).
    # k > q tokens use a neutral grey background.
    colors: list[list[str]] = []
    text_colors: list[list[str]] = []
    GREY = "#e9ecef"
    GREY_TEXT = "#888"
    for q in range(n):
        row_colors = []
        row_text = []
        for k in range(n):
            if k > q:
                row_colors.append(GREY)
                row_text.append(GREY_TEXT)
                continue
            v = float(np.clip(Anorm[q, k], 0.0, 1.0))
            rgba = cmap_obj(v)
            hex_color = mcolors.to_hex(rgba)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            row_colors.append(hex_color)
            row_text.append("#111" if brightness > 0.55 else "#fafafa")
        colors.append(row_colors)
        text_colors.append(row_text)

    # Token spans with optional <br/> after newline tokens for natural flow.
    token_spans = []
    for i, tok in enumerate(tokens):
        disp = _display_tok(tok)
        is_nl = _is_newline_tok(tok)
        interv = i in interv_set
        cls = "tok interv" if interv else "tok"
        title_extra = ", INTERVENED" if interv else ""
        # base_title is the static tooltip; on hover JS appends the live attn score A[q, i]
        # (and resets back to this base when unhovered). Stored in data-base so it survives.
        base_title = f"idx={i}, tok={_html_escape(repr(tok))}{title_extra}"
        token_spans.append(
            f'<span class="{cls}" data-idx="{i}" data-base="{base_title}" '
            f'title="{base_title}">'
            f"{_html_escape(disp)}</span>"
        )
        if is_nl:
            token_spans.append("<br/>")

    tokens_html = "".join(token_spans)

    # Heads summary for header.
    heads_summary = (
        f"{len(selected_heads)} heads · agg={agg} · norm={normalize}"
    )
    head_pairs_short = ", ".join(
        f"({layer},{head})" for layer, head in selected_heads[:12]
    )
    if len(selected_heads) > 12:
        head_pairs_short += f", … (+{len(selected_heads) - 12} more)"

    intervened_legend = (
        " · <span style=\"border-bottom:3px solid #8e44ad;font-weight:700;\">intervened</span>"
        f" position{'s' if len(interv_set) != 1 else ''} ({len(interv_set)})"
        if interv_set else ""
    )

    import json as _json
    import uuid as _uuid
    colors_json = _json.dumps(colors)
    text_colors_json = _json.dumps(text_colors)
    # Aggregated, causal-masked attention actually being plotted (A[q, k]); shown live in the
    # tooltip on hover. Future cells (k > q) are 0 here and flagged as "(future)" in JS.
    scores_json = _json.dumps(np.round(A, 6).tolist())

    # Unique root id so all CSS selectors and JS DOM queries are scoped — this
    # chunk is safe to drop into a Jupyter cell without leaking styles onto
    # other outputs (e.g. body { background } turning the whole notebook white).
    root = f"iaviz-{_uuid.uuid4().hex[:8]}"
    toks_id = f"{root}-toks"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{_html_escape(title)}</title>
<style>
 #{root} {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #fafafa; color: #111; padding: 12px; border-radius: 6px; }}
 #{root} h2 {{ margin: 4px 0 6px 0; font-size: 16px; font-weight: 600; color: #111; }}
 #{root} .meta {{ color: #555; font-size: 12px; margin-bottom: 10px; }}
 #{root} .head-list {{ color: #777; font-size: 11px; margin-bottom: 14px; font-family: monospace; }}
 #{root} .tokens {{
   font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
   font-size: 13px;
   line-height: 1.85;
   background: #fff;
   border: 1px solid #ddd;
   border-radius: 6px;
   padding: 14px 16px;
   white-space: pre-wrap;
   word-break: break-word;
 }}
 #{root} .tok {{
   display: inline-block;
   padding: 2px 4px;
   margin: 1px 0;
   border-radius: 3px;
   border: 1px solid transparent;
   background: #fff;
   color: #111;
   cursor: pointer;
   transition: background 60ms, color 60ms;
 }}
 #{root} .tok.query {{ outline: 2px solid #d62728; outline-offset: -1px; }}
 #{root} .tok.interv {{ border-bottom: 3px solid #8e44ad; font-weight: 700; }}
 #{root} .legend {{ margin-top: 12px; font-size: 11px; color: #666; }}
 #{root} .swatch {{ display: inline-block; width: 14px; height: 12px; vertical-align: middle;
            border: 1px solid #999; margin: 0 4px; border-radius: 2px; }}
</style></head>
<body>
<div id="{root}">
<h2>{_html_escape(title)}</h2>
<div class="meta">{_html_escape(heads_summary)} · {n} tokens (slice [{start_from}, {end_at}))</div>
<div class="head-list">heads: {_html_escape(head_pairs_short)}</div>
<div class="tokens" id="{toks_id}">{tokens_html}</div>
<div class="legend">
  Hover a token to view its attention row (score shown in tooltip) · click to pin/unpin ·
  <span class="swatch" style="background:#e9ecef;"></span> future (causal mask) ·
  query token outlined in <span style="color:#d62728;font-weight:600;">red</span>{intervened_legend}
</div>
</div>
<script>
(function() {{
  const COLORS = {colors_json};
  const TEXT_COLORS = {text_colors_json};
  const SCORES = {scores_json};
  const container = document.getElementById("{toks_id}");
  if (!container) return;
  const toks = container.querySelectorAll(".tok");
  let pinned = null;

  function paint(q) {{
    if (q == null) {{
      toks.forEach(t => {{
        t.style.background = "";
        t.style.color = "";
        t.classList.remove("query");
        t.title = t.dataset.base;
      }});
      return;
    }}
    const row = COLORS[q];
    const trow = TEXT_COLORS[q];
    const srow = SCORES[q];
    toks.forEach(t => {{
      const k = +t.dataset.idx;
      t.style.background = row[k];
      t.style.color = trow[k];
      t.classList.toggle("query", k === q);
      t.title = (k <= q)
        ? t.dataset.base + " · attn[q" + q + "->k" + k + "]=" + srow[k].toFixed(4)
        : t.dataset.base + " · (future)";
    }});
  }}

  toks.forEach(t => {{
    t.addEventListener("mouseover", () => {{
      if (pinned == null) paint(+t.dataset.idx);
    }});
    t.addEventListener("click", () => {{
      const q = +t.dataset.idx;
      if (pinned === q) {{ pinned = null; paint(null); }}
      else {{ pinned = q; paint(q); }}
    }});
  }});

  container.addEventListener("mouseleave", () => {{
    if (pinned == null) paint(null);
    else paint(pinned);
  }});
}})();
</script>
</body></html>
"""

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Saved interactive attention HTML → {save_path}")

    return html


def show_interactive_attention(
    attn_info: AttentionInformation,
    selected_heads: list[tuple[int, int]],
    start_from: int = 0,
    end_at: Optional[int] = None,
    agg: str = "mean",
    normalize: str = "row_max",
    cmap: str = "Blues",
    title: str = "Interactive attention",
    save_path: Optional[str] = None,
    height: Optional[int] = 520,
    intervened_positions: Optional[list[int]] = None,
):
    """Convenience wrapper: build the HTML and display inline (Jupyter).

    height:
        - int  → wrap in an iframe of fixed pixel height (default 520).
        - None → render inline (no iframe); the cell expands to fit all tokens.
                 Use when you don't want to scroll inside a fixed-height box.
    intervened_positions:
        token positions (full-sequence index space, negatives allowed) where steering /
        head-patching was applied — marked with a purple underline.  See
        ``interactive_attention_html``.
    """
    from IPython.display import HTML, display

    html = interactive_attention_html(
        attn_info=attn_info,
        selected_heads=selected_heads,
        start_from=start_from,
        end_at=end_at,
        agg=agg,
        normalize=normalize,
        cmap=cmap,
        title=title,
        save_path=save_path,
        intervened_positions=intervened_positions,
    )
    if height is None:
        display(HTML(html))
    else:
        iframe = (
            f'<iframe srcdoc="{_html_escape(html)}" '
            f'style="width:100%; height:{height}px; border:1px solid #ddd; border-radius:6px;">'
            f"</iframe>"
        )
        display(HTML(iframe))
    return html
