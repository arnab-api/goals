"""Shared primitives for the DIFF-mechanism notebooks (01/02/03).

Lifted out of ``00_diff.ipynb`` (cells 1/3/4/6) so the three experiment notebooks
single-source the model, head sets, prompt builders, q/k/v transplant machinery and
readouts.

Usage in a notebook::

    import sys; sys.path.append("..")            # so `_diff_primitives` is importable
    import _diff_primitives as P
    P.setup("Qwen/Qwen2.5-Coder-32B-Instruct")   # loads model + head masks (once)
    from _diff_primitives import *                # pull the helper functions into scope
    mt, GOAL_HEADS, BOUNDARY_HEADS = P.mt, P.GOAL_HEADS, P.BOUNDARY_HEADS

The helper functions look up the module-global ``mt`` (and ``GROUP``, head sets, …) that
``setup()`` binds, so ``from … import *`` before ``setup()`` is fine — the functions
resolve those names at call time.
"""

import logging
import os
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

import baukit
import numpy as np
import torch
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── path bootstrap: make `src` importable regardless of the caller's cwd ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from circuitsvis.tokens import colored_tokens
from IPython.display import display

from src.attention import _safe_token
from src.goals.goalscope import apply_goalscope
from src.goals.head_loc_attn import HeadMask, collect_head_steering_vecs
from src.goals.oid_gaze import cache_root
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils, logging_utils

logging_utils.configure(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console(theme=Theme({"default": "white"}), highlight=False)


def rich_print(*a, **k):
    console.print(*a, **k)


RES = env_utils.DEFAULT_RESULTS_DIR

# ── module globals bound by setup() ──
mt = None
MODEL_NAME = None
GROUP = None
N_KV = None
N_LAYER = None
N_HEADS = None
GOAL_HEADS = None
BOUNDARY_HEADS = None
UNION_HEADS = None
GOAL_MASK = None
BOUNDARY_MASK = None
READOUT_HEADS = None
gsp = None  # optional GoalScope probe override (apply_goalscope reads globals().get("gsp"))


@dataclass
class Ctx:
    """Bundled run context — the model + derived config + trained head sets. Hold ONE in the
    notebook (``ctx = setup(...)``) and pass it to helpers as ``ctx=``; it lives in the notebook
    namespace, so it survives ``%autoreload`` of this module (unlike the setup()-bound module
    globals above, which reset to None on reload). Helpers fall back to those globals when ``ctx``
    is omitted, so old call sites keep working during the transition."""

    mt: object
    model_name: str = None
    group: int = None
    n_kv: int = None
    n_layer: int = None
    n_heads: int = None
    goal_heads: list = None
    boundary_heads: list = None
    union_heads: list = None
    goal_mask: object = None
    boundary_mask: object = None
    readout_heads: list = None
    gsp: object = None
    res: str = None

    @classmethod
    def from_live(cls, ns):
        """Rebuild a Ctx from a namespace of live values (e.g. a notebook's globals() after
        ``from _diff_primitives import *``) — WITHOUT re-running setup() / reloading the model."""
        g = ns.get
        return cls(
            mt=g("mt"), model_name=g("MODEL_NAME"), group=g("GROUP"), n_kv=g("N_KV"),
            n_layer=g("N_LAYER"), n_heads=g("N_HEADS"), goal_heads=g("GOAL_HEADS"),
            boundary_heads=g("BOUNDARY_HEADS"), union_heads=g("UNION_HEADS"),
            goal_mask=g("GOAL_MASK"), boundary_mask=g("BOUNDARY_MASK"),
            readout_heads=g("READOUT_HEADS"), gsp=g("gsp"), res=g("RES"),
        )


def mask_bin(heads, ctx=None):
    _nl = ctx.n_layer if ctx is not None else N_LAYER
    _nh = ctx.n_heads if ctx is not None else N_HEADS
    m = HeadMask(_nl, _nh)
    with torch.no_grad():
        m.values.zero_()
        for l, h in heads:
            m.values[l, h] = 1.0
    return m


def setup(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    cuda_visible_devices="0, 2, 3",
    goal_threshold=0.9,
    boundary_cos_thr=0.7,
    boundary_sel_thr=0.8,
):
    """Load the model + trained head masks once; bind module globals. Returns a namespace."""
    global mt, MODEL_NAME, GROUP, N_KV, N_LAYER, N_HEADS
    global \
        GOAL_HEADS, \
        BOUNDARY_HEADS, \
        UNION_HEADS, \
        GOAL_MASK, \
        BOUNDARY_MASK, \
        READOUT_HEADS

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    MODEL_NAME = model_name

    mt = ModelandTokenizer(
        model_key=model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    N_KV = mt.model.config.num_key_value_heads
    GROUP = mt.n_heads // N_KV
    N_LAYER = mt.n_layer
    N_HEADS = mt.model.config.num_attention_heads
    print(mt.name)
    print(
        f"n_heads={mt.n_heads}  n_kv_heads={N_KV}  group={GROUP}  "
        f"head_dim={mt.head_dim}  n_layer={mt.n_layer}"
    )

    # GOAL heads (READOUT) — trained localization mask (lam20), mid-band, thresholded.
    _gckpt = torch.load(
        os.path.join(
            RES, f"localization/{model_name.split('/')[-1]}/lam20/head_mask.pt"
        ),
        map_location="cpu",
    )
    GOAL_LAYERS = _gckpt["layers"]
    _goal_trained = HeadMask(len(GOAL_LAYERS), _gckpt["mask_values"].shape[1])
    with torch.no_grad():
        _goal_trained.values.copy_(_gckpt["mask_values"])
        if model_name.split("/")[-1] == "Qwen2.5-Coder-32B-Instruct":
            _goal_trained.values[0:25] = 0
            _goal_trained.values[56:] = 0
    GOAL_HEADS = sorted(
        _goal_trained.selected_heads(GOAL_LAYERS, threshold=goal_threshold)
    )
    GOAL_MASK = mask_bin(GOAL_HEADS)

    # BOUNDARY heads (DECISION) — move-on/stay separation (cos < thr) from clock/15's aff_sep.
    _bnd_root = os.path.join(cache_root(model_name, "conditions"), "completion_mask")
    _bnd_dir = max(
        (
            t
            for t in os.listdir(_bnd_root)
            if os.path.isdir(os.path.join(_bnd_root, t)) and "lam10.0" in t
        ),
        key=lambda t: os.path.getmtime(os.path.join(_bnd_root, t)),
    )
    _DCK = torch.load(
        os.path.join(_bnd_root, _bnd_dir, "mask.pt"),
        map_location="cpu",
        weights_only=False,
    )
    _dec_layers = [int(x) for x in _DCK["layers"]]
    _dmv = _DCK["mask_values"].numpy()
    _ = sorted(
        (_dec_layers[li], int(h)) for li, h in np.argwhere(_dmv > boundary_sel_thr)
    )  # BOUNDARY_ALL
    _aff = torch.load(
        os.path.join(RES, "notebook_caches/15_order_dependency/aff_sep__d5fbad08ba.pt"),
        map_location="cpu",
        weights_only=False,
    )["value"]
    BOUNDARY_HEADS = sorted(lh for lh, c in _aff.items() if c < boundary_cos_thr)
    BOUNDARY_MASK = mask_bin(BOUNDARY_HEADS)

    UNION_HEADS = sorted(set(GOAL_HEADS) | set(BOUNDARY_HEADS))
    READOUT_HEADS = GOAL_HEADS
    print(
        f"GOAL_HEADS     : {len(GOAL_HEADS):3d}  "
        f"(layers {min(l for l, _ in GOAL_HEADS)}–{max(l for l, _ in GOAL_HEADS)})"
    )
    print(
        f"BOUNDARY_HEADS : {len(BOUNDARY_HEADS):3d}  "
        f"(layers {min(l for l, _ in BOUNDARY_HEADS)}–{max(l for l, _ in BOUNDARY_HEADS)})"
    )
    print(f"UNION_HEADS    : {len(UNION_HEADS):3d}")

    return Ctx(
        mt=mt, model_name=MODEL_NAME, group=GROUP, n_kv=N_KV, n_layer=N_LAYER, n_heads=N_HEADS,
        goal_heads=GOAL_HEADS, boundary_heads=BOUNDARY_HEADS, union_heads=UNION_HEADS,
        goal_mask=GOAL_MASK, boundary_mask=BOUNDARY_MASK, readout_heads=READOUT_HEADS,
        gsp=gsp, res=RES,
    )


# ══════════════════════════════════════════════════════════════════════════════
# prompt display / spans  (00_diff.ipynb cell 4)
# ══════════════════════════════════════════════════════════════════════════════
def exact_prompt(prompt, tail_tokens=None, patch_pos=[-1], emphasize_tokens=None, ctx=None):
    """Prompt EXACTLY as the model sees it: patch_pos (pivot) in salmon, emphasize_tokens in yellow."""
    _mt = ctx.mt if ctx is not None else mt
    ids = prepare_input(prompts=prompt, tokenizer=_mt).input_ids[0].tolist()
    piv = {len(ids) + p if p < 0 else p for p in patch_pos}
    emph = set(emphasize_tokens or ())
    lo = 0 if tail_tokens is None else max(0, len(ids) - tail_tokens)
    OPEN, CLOSE, EOPEN, ECLOSE = "\x00<", "\x00>", "\x01<", "\x01>"
    parts = []
    for i in range(lo, len(ids)):
        s = _mt.tokenizer.decode([ids[i]]).replace("\n", "\\n\n")
        if i in piv:
            s = OPEN + s + CLOSE
        elif i in emph:
            s = EOPEN + s + ECLOSE
        parts.append(s)
    txt = escape("".join(parts))
    txt = (
        txt.replace(OPEN, "[bold black on salmon1]")
        .replace(CLOSE, "[/bold black on salmon1]")
        .replace(EOPEN, "[yellow]")
        .replace(ECLOSE, "[/yellow]")
    )
    head = f"[grey42]… {lo} earlier tokens elided …[/grey42]\n" if lo else ""
    return head + txt


def pivot_token(prompt, patch_pos=[-1], ctx=None):
    _mt = ctx.mt if ctx is not None else mt
    ids = prepare_input(prompts=prompt, tokenizer=_mt).input_ids[0].tolist()
    return [_mt.tokenizer.decode([ids[len(ids) + p if p < 0 else p]]) for p in patch_pos]


def token_span(prompt, substring, ctx=None):
    """[lo, hi) token indices of `substring` inside the tokenization of `prompt` (via char offsets)."""
    _mt = ctx.mt if ctx is not None else mt
    enc = prepare_input(prompts=prompt, tokenizer=_mt, return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    ci = prompt.index(substring)
    cj = ci + len(substring)
    toks = [k for k, (a, b) in enumerate(offs) if b > a and a < cj and b > ci]
    assert toks, f"substring not found in tokenization: {substring!r}"
    return min(toks), max(toks) + 1


def show_prompt(tag, prompt, span, ctx=None):
    _mt = ctx.mt if ctx is not None else mt
    rich_print(
        Panel(
            Text.from_markup(
                exact_prompt(prompt, patch_pos=[-1], emphasize_tokens=range(*span), ctx=ctx)
            ),
            title=f"[bold]{tag}[/bold]  TASK span {span}  ·  pivot={pivot_token(prompt, ctx=ctx)!r}  ·  "
            f"L={prepare_input(prompts=prompt, tokenizer=_mt)['input_ids'].shape[1]}",
            title_align="left",
            border_style="grey42",
            padding=(1, 1),
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# prompt builder + q/k/v patch primitives + readouts  (00_diff.ipynb cell 6)
# ══════════════════════════════════════════════════════════════════════════════
def get_prompt(question, prefill, ctx=None):
    _mt = ctx.mt if ctx is not None else mt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant proficient in programming. "
            "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain.",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    return _mt.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        continue_final_message=True,
        tokenize=False,
        chat_template_kwargs={"enable_thinking": False},
    )


def _proj_name(l, which, mt):  # which in {'q','k','v'}
    return mt.attn_module_name_format.format(l) + f".{which}_proj"


def cache_src_proj(prompt, positions, layers, which, ctx=None):
    _mt = ctx.mt if ctx is not None else mt
    """{layer: (len(positions), width)} — source {which}_proj outputs at `positions` (pre-RoPE for q/k)."""
    tok = prepare_input(prompts=prompt, tokenizer=_mt)
    names = {l: _proj_name(l, which, _mt) for l in layers}
    with (
        torch.no_grad(),
        baukit.TraceDict(
            _mt._model, list(names.values()), retain_input=False, retain_output=True
        ) as tr,
    ):
        _mt._model(
            **{k: v for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        )
    return {l: tr[names[l]].output[0, positions].detach().float().cpu() for l in layers}


def heads_per_layer(head_set, which, layers=None, ctx=None):
    _group = ctx.group if ctx is not None else GROUP
    """{layer: set(head)} — q-head index for which=="q", else kv-head index (h // GROUP)."""
    d = defaultdict(set)
    for l, h in head_set:
        if layers is None or l in layers:
            d[l].add(h if which == "q" else h // _group)
    return dict(d)


def _layers_of(head_set, ctx=None):
    _union = ctx.union_heads if ctx is not None else UNION_HEADS
    hs = _union if head_set == "ALL" else head_set
    return sorted({l for l, _ in hs})


@contextmanager
def proj_patch_hooks(mt, patch_specs):
    """Replace {which}_proj-output head-slices at `positions` with cached source vectors, for this forward.
    Positions absent from the current forward (e.g. a prompt position during a 1-token decode step) are
    skipped — so a prefill spec is a no-op on decode, and the effect then rides the KV cache."""
    hd = mt.head_dim
    handles = []
    try:
        for spec in patch_specs:
            which, cache, headmap, positions = (
                spec["which"],
                spec["cache"],
                spec["heads"],
                spec["positions"],
            )
            for layer, hset in headmap.items():
                mod = baukit.get_module(mt._model, _proj_name(layer, which, mt))
                src_l = cache[layer]  # (K, width)

                def make(hset, src_l, positions):
                    def hook(module, inp, out):
                        seq = out.shape[1]
                        pos = [p if p >= 0 else seq + p for p in positions]
                        keep = [(r, p) for r, p in enumerate(pos) if 0 <= p < seq]
                        if not keep:
                            return out
                        o = out.clone()
                        for h in hset:
                            sl = slice(h * hd, (h + 1) * hd)
                            for r, p in keep:
                                o[:, p, sl] = src_l[r, sl].to(o.dtype).to(o.device)
                        return o

                    return hook

                handles.append(mod.register_forward_hook(make(hset, src_l, positions)))
        yield
    finally:
        for h in handles:
            h.remove()


def make_patch_interface(mt, patch_specs):
    def iface(mt, inputs, patches=None, model_kwargs=None):
        with proj_patch_hooks(mt, patch_specs):
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    return iface


def pivot_attn(info, heads, positions, pivot):
    """mean over `heads` of value-weighted attention from `pivot` onto `positions` -> (len(positions),)."""
    A = info.attention_matrices  # (L, H, q, s)
    return np.stack([A[l, h, pivot, positions] for (l, h) in heads]).mean(0)


def attn_table(title, toks, a_ref, a_patch, ref_name):
    tbl = Table(title=title, highlight=False)
    tbl.add_column("task token")
    tbl.add_column(ref_name, justify="right")
    tbl.add_column("patched", justify="right")
    tbl.add_column("Δ", justify="right")
    for t, c, p in zip(toks, a_ref, a_patch):
        d = p - c
        col = "green" if d > 0 else "red"
        tbl.add_row(escape(repr(t)), f"{c:.3f}", f"{p:.3f}", f"[{col}]{d:+.3f}[/{col}]")
    rich_print(tbl)
    print(
        f"total mass on span:  {ref_name}={a_ref.sum():.3f}  patched={a_patch.sum():.3f}"
    )


def goalscope_readout(
    tag,
    prompt,
    pivot,
    heads=None,
    iface=None,
    strategy="additive_norm",
    alpha=0.1,
    layers=(35, 46),
    maxnew=128,
    ctx=None,
):
    _mt = ctx.mt if ctx is not None else mt
    _readout = ctx.readout_heads if ctx is not None else READOUT_HEADS
    _gsp = ctx.gsp if ctx is not None else globals().get("gsp")
    res = apply_goalscope(
        prompt=prompt,
        mt=_mt,
        selected_heads=heads or _readout,
        strategy=strategy,
        alpha=alpha,
        steer_layer_range=layers,
        batch_size=1,
        max_new_tokens=maxnew,
        goalscope_prompt=_gsp,
        token_positions=[pivot],
        include_noncode=False,
        show=False,
        patch_interface=iface,
    )
    for r in res:
        rich_print(
            f"[bold]{tag}[/bold]  pos={r['pos']} {r['token']!r} => "
            f"[italic]{escape(r['patched_gen'])}[/italic]"
        )
    return res


def greedy(tok, prefill_specs=None, decode_specs=None, n_steer=1, max_new=64, ctx=None):
    """Greedy decode. `prefill_specs` patched on the prefill (rides the KV cache after); `decode_specs`
    re-applied at the current token for the first `n_steer` forward passes (prefill = pass 1)."""
    _mt = ctx.mt if ctx is not None else mt
    ids, am = tok["input_ids"], tok["attention_mask"]
    ctx0 = proj_patch_hooks(_mt, prefill_specs) if prefill_specs else nullcontext()
    with torch.no_grad(), ctx0:
        out = _mt._model(input_ids=ids, attention_mask=am, use_cache=True)
    pkv = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [int(nxt)]
    with torch.no_grad():
        for step in range(1, max_new):
            am = torch.cat([am, torch.ones_like(nxt)], 1)
            use = decode_specs if (decode_specs and step < n_steer) else None
            _ck = proj_patch_hooks(_mt, use) if use else nullcontext()
            with _ck:
                out = _mt._model(
                    input_ids=nxt,
                    attention_mask=am,
                    past_key_values=pkv,
                    use_cache=True,
                )
            pkv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            t = int(nxt)
            if t == _mt.tokenizer.eos_token_id:
                break
            gen.append(t)
    return _mt.tokenizer.decode(gen, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
# GQA-bleed-corrected k/v transplant — per-query-head routing (single trajectory)
# ══════════════════════════════════════════════════════════════════════════════
# The per-kv-head k/v patch (proj_patch_hooks) feeds the SOURCE k/v to ALL `GROUP`
# query heads that share a patched kv-head — so a "goal-head" transplant actually
# drives up to GROUP× more query heads (the GQA bleed). This corrects it: at each
# patched layer, ONLY the selected query heads read the source span; every other head
# reads the clean dest k/v. Because it is one trajectory with the correct (propagated)
# query at every layer, it fixes both the token-divergence and the query-propagation
# concerns. Implemented by wrapping the eager attention forward of the patched layers.
#
# Mechanics inside the wrapped forward (per patched layer):
#   • compute q, k, v as usual; k/v come from the CLEAN cache (span = dest)
#   • out_clean   = eager_attention(q, k, v)                         # all heads, clean
#   • out_patched = eager_attention(q, k⊕src_span, v⊕src_span)       # all heads, source span
#   • out = out_clean; out[:, :, selected_qheads] = out_patched[...] # route per query head
# The span is fixed (prefill positions), so the source k (RoPE'd at dest positions) and
# source v are precomputed once. The cache is never patched → clean rides it; routing is
# re-applied inside every forward. Only the patched layers are wrapped.

def _attn_module(mt, layer):
    return baukit.get_module(mt._model, mt.attn_module_name_format.format(layer))


def _rope_source_span(mt, k_cache, v_cache, dest_positions, patched_kvheads):
    """RoPE the (pre-RoPE) source-span keys at the DEST positions; keep source values as-is.
    Returns src_k, src_v as {layer: {kv_head: (len_span, head_dim)}} on the model's device/dtype."""
    import importlib

    hd = mt.head_dim
    ref = next(mt._model.parameters())
    dev, dt = ref.device, ref.dtype
    layers = sorted(patched_kvheads)
    sample = _attn_module(mt, layers[0])
    apply_rope = importlib.import_module(type(sample).__module__).apply_rotary_pos_emb
    base = mt.layer_name_format.split(".layers")[0]                 # e.g. "model"
    rot = baukit.get_module(mt._model, base + ".rotary_emb")

    pos = torch.tensor([list(dest_positions)], device=dev)         # (1, S)
    dummy = torch.zeros(1, len(dest_positions), 1, device=dev, dtype=dt)
    cos, sin = rot(dummy, pos)                                     # (1, S, hd)
    cos, sin = cos.to(dev), sin.to(dev)  # rotary_emb may live on another shard under device_map="auto"

    src_k, src_v = {}, {}
    for L in layers:
        kk, vv = {}, {}
        for kvh in patched_kvheads[L]:
            k_slice = k_cache[L][:, kvh * hd:(kvh + 1) * hd].to(dev, dt)[None, None]  # (1,1,S,hd)
            _, k_roped = apply_rope(torch.zeros_like(k_slice), k_slice, cos, sin)     # RoPE @ dest pos
            kk[kvh] = k_roped[0, 0].to(dt)                          # (S, hd)
            vv[kvh] = v_cache[L][:, kvh * hd:(kvh + 1) * hd].to(dev, dt)  # (S, hd) — values not RoPE'd
        src_k[L], src_v[L] = kk, vv
    return src_k, src_v


def _make_headwise_forward(sample_module, state):
    import importlib

    m = importlib.import_module(type(sample_module).__module__)
    eager = m.eager_attention_forward
    apply_rope = m.apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(
                k, v, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position})
        sw = getattr(self, "sliding_window", None)
        out_clean, w = eager(self, q, k, v, attention_mask, dropout=0.0,
                             scaling=self.scaling, sliding_window=sw, **kwargs)
        gh = state["goal_qheads"].get(self.layer_idx)
        if gh:
            span = state["span"]
            src_k, src_v = state["src_k"][self.layer_idx], state["src_v"][self.layer_idx]
            k2, v2 = k.clone(), v.clone()
            for kvh in src_k:                                      # replace span for patched kv-heads only
                k2[:, kvh, span, :] = src_k[kvh].to(k2.device, k2.dtype)
                v2[:, kvh, span, :] = src_v[kvh].to(v2.device, v2.dtype)
            out_patched, _ = eager(self, q, k2, v2, attention_mask, dropout=0.0,
                                   scaling=self.scaling, sliding_window=sw, **kwargs)
            out = out_clean.clone()
            out[:, :, gh, :] = out_patched[:, :, gh, :]            # route per query head
        else:
            out = out_clean
        attn_output = out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), w

    return forward


@contextmanager
def headwise_kv_patch(mt, patch_heads, k_cache, v_cache, dest_positions, route_group=False):
    """Wrap the patched layers' attention so ONLY `patch_heads` (query heads) read the source span k/v;
    all other heads (in-group bled + out-of-group) read the clean dest k/v. Fixes the GQA bleed.

    Everything is derived from `mt` and the passed `patch_heads` list of (layer, query_head) — no module
    globals — so it is independent of setup()/autoreload state and safe across notebooks.

    route_group=True routes ALL query heads of each patched group (= reproduces the raw bled patch) —
    used only for the equivalence self-test (should match greedy(prefill_specs=patch_specs))."""
    n_kv = getattr(mt, "n_kv_heads", None) or mt.config.num_key_value_heads
    group = mt.n_heads // n_kv                                     # GQA group size, from mt (not module GROUP)
    sel_q, patched_kv = defaultdict(set), defaultdict(set)
    for (l, h) in patch_heads:                                    # patch_heads: list of (layer, query_head)
        sel_q[l].add(h)
        patched_kv[l].add(h // group)
    patched_kvheads = {l: sorted(kvs) for l, kvs in patched_kv.items()}
    if route_group:                                               # every query head in a patched group
        goal_qheads = {l: sorted(g * group + i for g in kvs for i in range(group))
                       for l, kvs in patched_kvheads.items()}
    else:                                                         # only the selected query heads
        goal_qheads = {l: sorted(hs) for l, hs in sel_q.items()}
    src_k, src_v = _rope_source_span(mt, k_cache, v_cache, dest_positions, patched_kvheads)
    state = {"goal_qheads": goal_qheads, "span": list(dest_positions), "src_k": src_k, "src_v": src_v}
    originals = {}
    try:
        for L in goal_qheads:
            mod = _attn_module(mt, L)
            originals[L] = mod.forward
            mod.forward = types.MethodType(_make_headwise_forward(mod, state), mod)
        yield state
    finally:
        for L, f in originals.items():
            _attn_module(mt, L).forward = f


def make_headwise_interface(mt, patch_heads, k_cache, v_cache, dest_positions):
    """A goalscope / get_attention_matrices-style ``patch_interface`` that runs the forward under the
    per-query-head *headwise* routing (GQA-bleed-corrected). Drop it in as the ``iface`` for
    ``goalscope_readout`` (or ``collect_head_steering_vecs`` / ``get_o_proj_inputs_batch``) so the
    goal-head write is EXTRACTED under the corrected single-forward intervention — only PATCH_HEADS
    read the source span — instead of the raw bled k/v patch. Same shape as ``make_patch_interface``,
    so nothing in apply_goalscope changes."""

    def iface(mt, inputs, patches=None, model_kwargs=None):
        with headwise_kv_patch(mt, patch_heads, k_cache, v_cache, dest_positions):
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    return iface


def show_delta_tokens(tokens, delta, label=""):
    """Diverging red(neg)/green(pos) per-token coloring of an attention delta, symmetric about 0."""
    d = np.asarray(delta, dtype=float)
    M = float(np.abs(d).max()) or 1.0
    if label:
        rich_print(f"[grey58]{label}   (green +Δ / red −Δ, |Δ|max={M:.3f})[/grey58]")
    display(
        colored_tokens(
            tokens=[_safe_token(t) for t in tokens],
            values=d,
            min_value=-M,
            max_value=M,
            negative_color="red",
            positive_color="green",
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# NEW helpers for the DIFF experiments (query arithmetic · OV-write · cosines)
# ══════════════════════════════════════════════════════════════════════════════
def q_lincomb(terms):
    """Linear combination of {layer: (K, width)} proj caches.  `terms` = list of (coef, cache).

    All caches must share the same layer keys and shapes (same K, width). Used to synthesise a
    query cache, e.g. q({a})+q({b})-q({}) for the compositional-injection test (B2)."""
    layers = list(terms[0][1].keys())
    return {l: sum(coef * cache[l] for coef, cache in terms) for l in layers}


def head_write(prompt, heads, positions, iface=None, ctx=None):
    """Aggregated residual-stream write of `heads` at `positions` — the SAME object GoalScope reads.

    Wraps ``collect_head_steering_vecs`` (sum over heads of o_proj_input[h] @ W_o[:, h].T).
    Returns (len(positions), n_embd) float32 CPU.  `iface` runs the extraction under an intervention."""
    _mt = ctx.mt if ctx is not None else mt
    tok = prepare_input(prompts=prompt, tokenizer=_mt)
    vecs = collect_head_steering_vecs(
        _mt, tok, list(heads), list(positions), patch_interface=iface
    )
    return vecs[0]  # (n_pos, n_embd)


def write_norm(prompt, heads, pivot, iface=None, ctx=None):
    """L2 norm of the aggregated head write at a single pivot position."""
    return float(head_write(prompt, heads, [pivot], iface=iface, ctx=ctx)[0].norm())


def per_head_cosine(cache_a, cache_b, heads, which, ctx=None):
    """Per-head cosine between two proj caches (both {layer:(1,width)}), sliced to `heads`.

    `which`=="q" slices q-head h; otherwise slices kv-head h//GROUP.  Returns {(layer,head): cos}."""
    _mt = ctx.mt if ctx is not None else mt
    _group = ctx.group if ctx is not None else GROUP
    hd = _mt.head_dim
    out = {}
    for l, h in heads:
        if l not in cache_a or l not in cache_b:
            continue
        idx = h if which == "q" else h // _group
        sl = slice(idx * hd, (idx + 1) * hd)
        va = cache_a[l][0, sl].float()
        vb = cache_b[l][0, sl].float()
        denom = (va.norm() * vb.norm()).clamp_min(1e-8)
        out[(l, h)] = float((va @ vb) / denom)
    return out


def cos(u, v):
    """Cosine between two 1-D tensors/arrays."""
    u = torch.as_tensor(np.asarray(u), dtype=torch.float32).flatten()
    v = torch.as_tensor(np.asarray(v), dtype=torch.float32).flatten()
    return float((u @ v) / (u.norm() * v.norm()).clamp_min(1e-8))
