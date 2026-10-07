"""Interventions, generations, and debugging/visualization utilities for the DIFF
hypothesis — the ``src.diff`` submodule's functional core.

Originally lifted out of ``00_diff.ipynb`` (cells 1/3/4/6) as
``notebooks/diff/_diff_primitives.py``; promoted to ``src/diff/functional.py`` so the
experiment notebooks (00/01/02/03) and the scaled-up scripts single-source the model,
head sets, prompt builders, q/k/v transplant machinery and readouts.

Sibling modules in this package:
    ``src.diff.data``  — data generation for Spec-1 (query-transfer) and Spec-2 (kv-transfer).
    ``src.diff.eval``  — evaluation utils for the q-transfer and kv-transfer interventions.

Usage in a notebook::

    import sys; sys.path.append("../..")             # repo root, so `src` is importable
    import src.diff.functional as P
    P.setup("Qwen/Qwen2.5-Coder-32B-Instruct")       # loads model + head masks (once)
    from src.diff.functional import *                # pull the helper functions into scope
    mt, GOAL_HEADS, BOUNDARY_HEADS = P.mt, P.GOAL_HEADS, P.BOUNDARY_HEADS

The helper functions look up the module-global ``mt`` (and ``GROUP``, head sets, …) that
``setup()`` binds, so ``from … import *`` before ``setup()`` is fine — the functions
resolve those names at call time. Prefer passing an explicit ``ctx=`` (a :class:`Ctx`
from ``setup()``) in library code: it survives ``%autoreload`` and avoids the globals.
"""

import logging
import os
import sys
import types
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass

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
            mt=g("mt"),
            model_name=g("MODEL_NAME"),
            group=g("GROUP"),
            n_kv=g("N_KV"),
            n_layer=g("N_LAYER"),
            n_heads=g("N_HEADS"),
            goal_heads=g("GOAL_HEADS"),
            boundary_heads=g("BOUNDARY_HEADS"),
            union_heads=g("UNION_HEADS"),
            goal_mask=g("GOAL_MASK"),
            boundary_mask=g("BOUNDARY_MASK"),
            readout_heads=g("READOUT_HEADS"),
            gsp=g("gsp"),
            res=g("RES"),
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
    device_map="auto",
):
    """Load the model + trained head masks once; bind module globals. Returns a namespace.

    ``device_map`` is forwarded to ``ModelandTokenizer`` — pass a custom dict (e.g. the balanced map
    from ``h1h2_mask.build_balanced_device_map``) to rebalance layers across GPUs; default ``"auto"``."""
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
        device_map=device_map,
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
        mt=mt,
        model_name=MODEL_NAME,
        group=GROUP,
        n_kv=N_KV,
        n_layer=N_LAYER,
        n_heads=N_HEADS,
        goal_heads=GOAL_HEADS,
        boundary_heads=BOUNDARY_HEADS,
        union_heads=UNION_HEADS,
        goal_mask=GOAL_MASK,
        boundary_mask=BOUNDARY_MASK,
        readout_heads=READOUT_HEADS,
        gsp=gsp,
        res=RES,
    )


# ══════════════════════════════════════════════════════════════════════════════
# prompt display / spans  (00_diff.ipynb cell 4)
# ══════════════════════════════════════════════════════════════════════════════
def exact_prompt(
    prompt, tail_tokens=None, patch_pos=[-1], emphasize_tokens=None, ctx=None
):
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
    return [
        _mt.tokenizer.decode([ids[len(ids) + p if p < 0 else p]]) for p in patch_pos
    ]


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


def show_prompt(tag, prompt, span, ctx=None, neg_shift=-1):
    _mt = ctx.mt if ctx is not None else mt
    rich_print(
        Panel(
            Text.from_markup(
                exact_prompt(
                    prompt, patch_pos=[neg_shift], emphasize_tokens=range(*span), ctx=ctx
                )
            ),
            title=f"[bold]{tag}[/bold]  TASK span {span}  ·  pivot={pivot_token(prompt, patch_pos=[neg_shift], ctx=ctx)!r}  ·  "
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
    """{layer: (len(positions), width)} — source {which}_proj outputs at `positions` (pre-RoPE for q/k).
    `prompt` may be a string OR a pre-tokenized TokenizerOutput (e.g. a token-spliced pad alignment)."""
    tok = prepare_input(prompts=prompt, tokenizer=_mt) if isinstance(prompt, str) else prompt
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
    skipped — so a prefill spec is a no-op on decode, and the effect then rides the KV cache.

    Optional ``spec["proj"] = {layer: {head: P (d, d)}}`` makes the patch PARTIAL for those heads: the
    head slice becomes ``x + (src - x) @ P`` (P = M diag(m) M^-1, an SVD-component mask of W_QK — see
    ``src.diff.nrgs_svd_mask``) instead of ``src``. Heads without a P are fully replaced as before."""
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
            projmap = spec.get("proj") or {}
            for layer, hset in headmap.items():
                mod = baukit.get_module(mt._model, _proj_name(layer, which, mt))
                src_l = cache[layer]  # (K, width)
                pl = projmap.get(layer) or {}
                ph = [h for h in hset if h in pl]                       # partially patched heads
                fh = [h for h in hset if h not in pl]                   # fully replaced heads
                P = torch.stack([pl[h] for h in ph]).float() if ph else None   # (n_ph, d, d)

                def make(fh, ph, P, src_l, positions):
                    def hook(module, inp, out):
                        seq = out.shape[1]
                        pos = [p if p >= 0 else seq + p for p in positions]
                        keep = [(r, p) for r, p in enumerate(pos) if 0 <= p < seq]
                        if not keep:
                            return out
                        o = out.clone()
                        for h in fh:
                            sl = slice(h * hd, (h + 1) * hd)
                            for r, p in keep:
                                o[:, p, sl] = src_l[r, sl].to(o.dtype).to(o.device)
                        if ph:
                            Pd = P.to(o.device)
                            for r, p in keep:
                                x = o[:, p].float().view(o.shape[0], -1, hd)[:, ph]          # (B, n_ph, d)
                                s = src_l[r].float().to(o.device).view(-1, hd)[ph]          # (n_ph, d)
                                new = x + torch.einsum("bhd,hde->bhe", s.unsqueeze(0) - x, Pd)
                                for k, h in enumerate(ph):
                                    o[:, p, h * hd:(h + 1) * hd] = new[:, k].to(o.dtype)
                        return o

                    return hook

                handles.append(mod.register_forward_hook(make(fh, ph, P, src_l, positions)))
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
    base = mt.layer_name_format.split(".layers")[0]  # e.g. "model"
    rot = baukit.get_module(mt._model, base + ".rotary_emb")

    pos = torch.tensor([list(dest_positions)], device=dev)  # (1, S)
    dummy = torch.zeros(1, len(dest_positions), 1, device=dev, dtype=dt)
    cos, sin = rot(dummy, pos)  # (1, S, hd)
    cos, sin = (
        cos.to(dev),
        sin.to(dev),
    )  # rotary_emb may live on another shard under device_map="auto"

    src_k, src_v = {}, {}
    for L in layers:
        kk, vv = {}, {}
        for kvh in patched_kvheads[L]:
            k_slice = k_cache[L][:, kvh * hd : (kvh + 1) * hd].to(dev, dt)[
                None, None
            ]  # (1,1,S,hd)
            _, k_roped = apply_rope(
                torch.zeros_like(k_slice), k_slice, cos, sin
            )  # RoPE @ dest pos
            kk[kvh] = k_roped[0, 0].to(dt)  # (S, hd)
            vv[kvh] = v_cache[L][:, kvh * hd : (kvh + 1) * hd].to(
                dev, dt
            )  # (S, hd) — values not RoPE'd
        src_k[L], src_v[L] = kk, vv
    return src_k, src_v


def _make_headwise_forward(sample_module, state):
    import importlib

    m = importlib.import_module(type(sample_module).__module__)
    eager = m.eager_attention_forward
    apply_rope = m.apply_rotary_pos_emb

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(
                k,
                v,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        sw = getattr(self, "sliding_window", None)
        out_clean, w = eager(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=sw,
            **kwargs,
        )
        gh = state["goal_qheads"].get(self.layer_idx)
        if gh:
            span = state["span"]
            src_k, src_v = (
                state["src_k"][self.layer_idx],
                state["src_v"][self.layer_idx],
            )
            k2, v2 = k.clone(), v.clone()
            for kvh in src_k:  # replace span for patched kv-heads only
                k2[:, kvh, span, :] = src_k[kvh].to(k2.device, k2.dtype)
                v2[:, kvh, span, :] = src_v[kvh].to(v2.device, v2.dtype)
            out_patched, _ = eager(
                self,
                q,
                k2,
                v2,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                sliding_window=sw,
                **kwargs,
            )
            out = out_clean.clone()
            out[:, :, gh, :] = out_patched[:, :, gh, :]  # route per query head
        else:
            out = out_clean
        attn_output = out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), w

    return forward


@contextmanager
def headwise_kv_patch(
    mt, patch_heads, k_cache, v_cache, dest_positions, route_group=False
):
    """Wrap the patched layers' attention so ONLY `patch_heads` (query heads) read the source span k/v;
    all other heads (in-group bled + out-of-group) read the clean dest k/v. Fixes the GQA bleed.

    Everything is derived from `mt` and the passed `patch_heads` list of (layer, query_head) — no module
    globals — so it is independent of setup()/autoreload state and safe across notebooks.

    route_group=True routes ALL query heads of each patched group (= reproduces the raw bled patch) —
    used only for the equivalence self-test (should match greedy(prefill_specs=patch_specs))."""
    n_kv = getattr(mt, "n_kv_heads", None) or mt.config.num_key_value_heads
    group = mt.n_heads // n_kv  # GQA group size, from mt (not module GROUP)
    sel_q, patched_kv = defaultdict(set), defaultdict(set)
    for l, h in patch_heads:  # patch_heads: list of (layer, query_head)
        sel_q[l].add(h)
        patched_kv[l].add(h // group)
    patched_kvheads = {l: sorted(kvs) for l, kvs in patched_kv.items()}
    if route_group:  # every query head in a patched group
        goal_qheads = {
            l: sorted(g * group + i for g in kvs for i in range(group))
            for l, kvs in patched_kvheads.items()
        }
    else:  # only the selected query heads
        goal_qheads = {l: sorted(hs) for l, hs in sel_q.items()}
    src_k, src_v = _rope_source_span(
        mt, k_cache, v_cache, dest_positions, patched_kvheads
    )
    state = {
        "goal_qheads": goal_qheads,
        "span": list(dest_positions),
        "src_k": src_k,
        "src_v": src_v,
    }
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


@contextmanager
def boost_head_ov(mt, heads, alpha, positions=(-1,)):
    """Scale the OV contribution of `heads` (query-head (layer, h)) by `alpha` for this forward — i.e.
    amplify how much those heads WRITE into the residual stream. o_proj is linear (no bias in Qwen2),
    so scaling a head's o_proj-INPUT slice by `alpha` scales its residual contribution by `alpha`.
    Applied at `positions` of each forward (default last token = the decision point each decode step;
    pass `positions=None` for every position). Composable with `headwise_kv_patch` — stack the two
    `with` blocks. `alpha == 1.0` is a no-op."""
    hd = mt.head_dim
    by_layer = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)
    handles = []
    try:
        for layer, hs in by_layer.items():
            mod = baukit.get_module(mt._model, _proj_name(layer, "o", mt))

            def make(hs):
                def pre_hook(module, args):
                    x = args[0]
                    seq = x.shape[1]
                    pos = (list(range(seq)) if positions is None
                           else [p if p >= 0 else seq + p for p in positions if -seq <= p < seq])
                    if not pos or alpha == 1.0:
                        return None
                    x = x.clone()
                    for h in hs:
                        x[:, pos, h * hd:(h + 1) * hd] *= alpha
                    return (x,) + tuple(args[1:])

                return pre_hook

            handles.append(mod.register_forward_pre_hook(make(hs)))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


@contextmanager
def restrict_head_attn(mt, heads, block_key_positions, keep=None):
    """Attention knockout ablation (ABLATION_STRAT="restrict"): let `heads` attend to ALL context
    tokens EXCEPT the key columns in `block_key_positions` (the <task_span>). For each (layer,
    query_head) in `heads`, the pre-softmax attention scores at those key columns are set to -inf,
    so the head can no longer read the task-description span — but it keeps its normal write and can
    still look everywhere else. Contrast with the "zero" strategy (:func:`boost_head_ov` with
    alpha=0), which kills the head's OV write at every position.

    Implemented by expanding the additive attention bias to per-head shape (batch, n_heads, q, k) and
    writing the dtype min at the blocked key columns for the selected heads only. Composes with
    :func:`headwise_kv_patch` (which forwards the same `attention_mask` straight into `eager`), so the
    patch heads still read the transplanted span while the ablate heads are blind to it. `keep` heads
    (default = none) are never blocked — pass PATCH_HEADS so a patched head is never knocked out.

    `block_key_positions` are ABSOLUTE key indices into the (padded) prompt; they stay valid through
    cached generation since the task span sits in the prompt prefix."""
    keep = set(keep or [])
    cols = sorted({int(p) for p in block_key_positions})
    by_layer = defaultdict(list)
    for l, h in heads:
        if (l, h) not in keep:
            by_layer[l].append(h)
    if not cols or not by_layer:
        yield
        return

    handles = []
    try:
        for layer, hs in by_layer.items():
            mod = _attn_module(mt, layer)

            def make(hs):
                def pre_hook(module, args, kwargs):
                    am = kwargs.get("attention_mask", None)
                    pos_idx = None
                    if am is None and len(args) >= 3:  # (hidden, pos_emb, attention_mask, ...)
                        am, pos_idx = args[2], 2
                    if am is None or am.dim() != 4:
                        return None  # non-eager / no additive mask — cannot knock out here
                    B, H, Q, K = am.shape
                    blk = [c for c in cols if c < K]
                    if not blk:
                        return None
                    am = (am.expand(B, mt.n_heads, Q, K).clone() if H == 1 else am.clone())
                    minval = torch.finfo(am.dtype).min
                    for h in hs:
                        am[:, h, :, blk] = minval  # this head is blind to the task span
                    if pos_idx is not None:
                        new = list(args)
                        new[pos_idx] = am
                        return (tuple(new), kwargs)
                    return (args, {**kwargs, "attention_mask": am})

                return pre_hook

            handles.append(mod.register_forward_pre_hook(make(hs), with_kwargs=True))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


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


# ══════════════════════════════════════════════════════════════════════════════
# q-transfer intervention runner (single source for the 04 notebook + batch eval)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Closed-loop query transfer — regenerate the goal heads' query at every decode step
# ══════════════════════════════════════════════════════════════════════════════
CL_INSTRUCTION = {
    "python": "You are implementing a simple Python function. Just write the function body, without any "
              "comments or explanations.",
    "cpp": "You are implementing a simple C++ function. Just write the function body, without any comments "
           "or explanations. Assume all the necessary headers are already included.",
}


def closed_loop_probe_prompt(mt, datatype, var_name, lang, pivot="indent"):
    """Information-less probe for the closed loop: DST's signature, then the bare trailing indent
    (``pivot="indent"``) or ``    return`` (``pivot="var_eq"``). Returns ``(prompt, prefill)``."""
    from src.diff.data import _signature
    from src.goals.goalscope import GOALSCOPE_SYS

    prefill = _signature(datatype, var_name, lang) + ("\n    return" if pivot == "var_eq" else "\n    ")
    msgs = [{"role": "system", "content": GOALSCOPE_SYS}, {"role": "user", "content": CL_INSTRUCTION[lang]},
            {"role": "assistant", "content": prefill}]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        text = mt.tokenizer.apply_chat_template(msgs, chat_template_kwargs={"enable_thinking": False}, **kw)
    except TypeError:
        text = mt.tokenizer.apply_chat_template(msgs, **kw)
    if not text.endswith(prefill) and text.endswith(prefill.rstrip()):   # templates that rstrip the turn
        text += prefill[len(prefill.rstrip()):]
    return text, prefill


def statement_done(text):
    """True once ``text`` contains a newline at which every ``()``, ``[]``, ``{}`` opened so far is closed — the end
    of the first statement. Works for one-line Python and for C++ statements whose lambda body spans lines."""
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "\n" and depth <= 0 and text[:i].strip():   # ignore newlines before any code
            return True
    return False


@contextmanager
def _goal_steer(mt, goal_unit, layers, alpha, positions):
    """additive_norm: h <- h + alpha * ||h|| * goal_unit at ``positions`` of the residual after each layer in
    ``layers`` (the same rule as GoalScope's ``add_norm`` patch)."""

    def edit(output, layer):
        h = output[0] if isinstance(output, tuple) else output
        for p in positions:
            v = h[:, p, :]
            h[:, p, :] = v + alpha * v.float().norm(dim=-1, keepdim=True).to(v.dtype) * goal_unit.to(v.device, v.dtype)
        return output

    with baukit.TraceDict(mt._model, [mt.layer_name_format.format(l) for l in layers], edit_output=edit):
        yield


@torch.no_grad()
def closed_loop_generate(
    mt,
    dst_tok,
    prefill_specs,
    patch_heads,
    goal_vec,
    probe_tok,
    *,
    steer_layers=(35, 46),
    alpha=0.1,
    probe_steer="all",
    max_new=64,
    stop_at_statement=False,
    patched_ctx=None,
    component_proj=None,
    ref_q=None,
):
    """Closed-loop query transfer (notebook ``04.4``).

    1. Patched DST prefill with ``prefill_specs`` (the SRC query at the pivot) -> ``t1``.
    2. Probe (``probe_tok``, information-less, ends at its pivot) steered with ``goal_vec`` (additive_norm, ``alpha``,
       residual after ``range(*steer_layers)``) at its pivot and, with ``probe_steer="all"``, at every appended token.
    3. For k >= 1: the probe consumes ``t_k``; its query at ``t_k`` on ``patch_heads`` (``q_caus``) replaces the
       patched run's goal-head query at ``t_k`` -> ``t_{k+1}``. Runs for ``max_new`` tokens (or EOS); with
       ``stop_at_statement=True`` it also stops at the end of the first statement (:func:`statement_done`).

    ``patched_ctx()`` (optional) returns a context manager entered around EVERY patched-run forward (e.g. an OV
    boost / attention restriction); the probe forwards never get it. ``ref_q`` ``{layer: (1, width)}`` (optional)
    adds the per-step mean cosine of ``q_caus`` to it over ``patch_heads``. Returns
    ``dict(text, ids, steps=[{k, tok, cos_ref}])``."""
    q_layers = sorted({l for l, _ in patch_heads})
    qh = heads_per_layer(patch_heads, "q")
    q_names = {l: _proj_name(l, "q", mt) for l in q_layers}
    layers = list(range(*steer_layers))
    g_unit = (goal_vec.float() / goal_vec.float().norm().clamp_min(1e-6))
    eos = mt.tokenizer.eos_token_id
    pctx = patched_ctx or nullcontext
    hd = mt.head_dim

    def _ext(am):
        return torch.cat([am, torch.ones_like(am[:, :1])], 1)

    def _cos(q):
        if ref_q is None:
            return None
        vals = [float(torch.nn.functional.cosine_similarity(q[l][0, h * hd:(h + 1) * hd], ref_q[l][0, h * hd:(h + 1) * hd], dim=0))
                for l, h in patch_heads]
        return float(np.mean(vals))

    ids, am = dst_tok["input_ids"], dst_tok["attention_mask"]
    with pctx(), proj_patch_hooks(mt, prefill_specs):
        out = mt._model(input_ids=ids, attention_mask=am, use_cache=True)
    pkv_d, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)

    p_am = probe_tok["attention_mask"]
    piv_probe = probe_tok["input_ids"].shape[1] - 1
    with _goal_steer(mt, g_unit, layers, alpha, [piv_probe]):
        pout = mt._model(input_ids=probe_tok["input_ids"], attention_mask=p_am, use_cache=True)
    pkv_p = pout.past_key_values

    gen, steps = [int(nxt)], []
    while (len(gen) < max_new and int(nxt) != eos
           and not (stop_at_statement and statement_done(mt.tokenizer.decode(gen)))):
        p_am = _ext(p_am)
        sctx = _goal_steer(mt, g_unit, layers, alpha, [-1]) if probe_steer == "all" else nullcontext()
        with sctx, baukit.TraceDict(mt._model, list(q_names.values()), retain_output=True) as tr:
            pout = mt._model(input_ids=nxt, attention_mask=p_am, past_key_values=pkv_p, use_cache=True)
        pkv_p = pout.past_key_values
        q_caus = {l: tr[n].output[0, -1:].detach().float().cpu() for l, n in q_names.items()}
        am = _ext(am)
        spec = [dict(which="q", cache=q_caus, heads=qh, positions=[-1], proj=component_proj)]
        with pctx(), proj_patch_hooks(mt, spec):
            out = mt._model(input_ids=nxt, attention_mask=am, past_key_values=pkv_d, use_cache=True)
        pkv_d = out.past_key_values
        steps.append(dict(k=len(gen), tok=mt.tokenizer.decode(gen[-1:]), cos_ref=_cos(q_caus)))
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen.append(int(nxt))
    return dict(text=mt.tokenizer.decode([t for t in gen if t != eos], skip_special_tokens=True),
                ids=gen, steps=steps)


def run_q_transfer_intervention(
    mt,
    inst,
    patch_heads,
    readout_heads=None,
    *,
    n_steer=16,
    max_new=96,
    alpha=1.0,
    n_pivot=1,
    compute_gaze=True,
    compute_readout=True,
    restrict_to_task=False,
    component_proj=None,
    closed_loop=None,
    kv=None,
    ctx=None,
):
    """Run one q-transfer (or composed-filter) intervention and return all RAW artifacts as
    JSON-native types. Caches the SRC pivot query over ``patch_heads`` and injects it at the DST
    pivot (prefill + re-clamped for ``n_steer`` decode steps); collects clean-SRC / clean-DST /
    patched-DST generations, the three goalscope readouts (over ``readout_heads``, default =
    ``patch_heads``), and value-weighted pivot→TASK attention (clean & patched) for the gaze eval.

    ``n_pivot`` > 1 transplants the SRC query over the last ``n_pivot`` pivot tokens on the PREFILL
    (aligned by offset-from-pivot; only safe where those trailing tokens coincide on SRC/DST — e.g.
    ``postfix="targeted"``); the decode steer always re-clamps the last token only. ``alpha`` != 1.0
    boosts the OV write of ``patch_heads`` at the decision token during the PATCHED generation and
    readout (amplifying how much they write to the residual). Attention (the gaze signal) is always
    measured under the PLAIN patch — an OV boost scales the write, not the attention weights.

    ``compute_readout=False`` skips the three goalscope readout generations (``ro_*`` -> None).

    ``component_proj`` ``{layer: {head: P (d, d)}}`` makes the query transplant PARTIAL per head
    (``q_dst + (q_src - q_dst) @ P``, an SVD-component mask; :func:`src.diff.nrgs_svd_mask.component_projectors`)
    for the heads it lists; ``patch_heads`` must be exactly those heads.

    ``closed_loop`` (dict, optional) replaces the re-clamp PATCHED generation with :func:`closed_loop_generate`:
    keys ``probe_steer`` ("all" | "pivot"), ``max_new`` (64), ``alpha`` (0.1), ``steer_layers`` ((35, 46)). The goal
    encoding is the ``readout_heads`` OV write at the DST pivot under the prefill patch (+ boost); the probe is
    :func:`closed_loop_probe_prompt` for the DST language. Adds ``cl_steps`` / ``cl_probe_prefill`` to the result.
    Readouts and gaze are unchanged (they read the pivot under the prefill patch).

    ``kv`` (dict, optional) stacks a kv-transfer under every PATCHED forward (notebooks 04.3 / 04.4): the task-list
    key+value of ``kv["prompt"]`` (its listed predicates ``kv["predicates"]``) on ``kv["heads"]`` (default
    ``patch_heads``) replace those on DST's task-list span (DST's predicates ``kv["dst_predicates"]``), GQA-corrected
    head-wise routing, ``kv["align"]`` alignment (default "pad"). Patched runs use the aligned DST; the clean DST
    stays unpadded; the OV boost covers ``patch_heads`` ∪ kv heads. Adds a kv-only run (``gen_kv``, ``ro_kv``).
    Gaze is skipped under kv (the task span then holds the kv prompt's items) and ``restrict_to_task`` is not
    supported with it.

    Works for both ``QTransferInstance`` (TASK = numbered predicate list) and
    ``FilterTransferInstance`` (TASK = the composed phrase); the span is chosen accordingly.
    Returns a dict consumed directly by :func:`src.diff.eval.eval_q_transfer`."""
    from src.attention import get_attention_matrices
    from src.diff.data import materialize

    readout_heads = readout_heads or patch_heads
    m = materialize(inst, ctx=ctx)
    src_prompt, dst_prompt = m["src_prompt"], m["dst_prompt"]

    def _piv(p):
        return prepare_input(prompts=p, tokenizer=mt)["input_ids"].shape[1] - 1

    piv_src, piv_dst = _piv(src_prompt), _piv(dst_prompt)
    src_tok = prepare_input(prompts=src_prompt, tokenizer=mt)
    dst_tok = prepare_input(prompts=dst_prompt, tokenizer=mt)

    # optional kv-transfer: patched runs use the ALIGNED dst_tok / piv_dst; the clean DST keeps its own tokens
    dst_tok_clean, piv_dst_clean = dst_tok, piv_dst
    kv_ctx, kv_heads, kv_extra = nullcontext, [], {}
    if kv:
        assert not restrict_to_task, "restrict_to_task is not supported with kv"
        kv_heads = list(kv.get("heads") or patch_heads)

        def _list_span(prompt, preds):
            sp = [token_span(prompt, p.description, ctx=ctx) for p in preds]
            return min(a for a, _ in sp), max(b for _, b in sp)

        kv_tok, dst_tok, kv_pos, dst_pos, _, _ = align_kv_spans(
            mt, kv["prompt"], dst_prompt, _list_span(kv["prompt"], kv["predicates"]),
            _list_span(dst_prompt, kv["dst_predicates"]), mode=kv.get("align", "pad"), ctx=ctx)
        piv_dst = dst_tok["input_ids"].shape[1] - 1
        _kvh = heads_per_layer(kv_heads, "k", layers=set(_layers_of(kv_heads, ctx=ctx)), ctx=ctx)
        _k = cache_src_proj(kv_tok, kv_pos, sorted(_kvh), "k", ctx=ctx)
        _v = cache_src_proj(kv_tok, kv_pos, sorted(_kvh), "v", ctx=ctx)

        def kv_ctx():
            return headwise_kv_patch(mt, kv_heads, _k, _v, dst_pos)

        compute_gaze = False
        kv_extra = dict(kv_prompt=kv["prompt"], kv_n_tokens=len(kv_pos))

    # TASK span (gaze only): composed phrase (filter) or the union over the numbered predicate list
    # (q). N/A when the instance has no in-prompt SRC-target span (multi-predicate) — skip gaze.
    if (compute_gaze or restrict_to_task) and (("task_text" in m) or hasattr(inst, "task")):
        if "task_text" in m:
            lo, hi = token_span(dst_prompt, m["task_text"], ctx=ctx)
        else:
            spans = [token_span(dst_prompt, p.description, ctx=ctx) for p in inst.task.predicates]
            lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        task_positions = list(range(lo, hi))
        task_tokens = [mt.tokenizer.decode([int(dst_tok["input_ids"][0][j])]) for j in task_positions]
    else:
        task_positions, task_tokens = None, None

    # query transplant specs — single-pivot (n_pivot=1) or last-n_pivot window on the prefill
    q_layers = sorted({l for l, _ in patch_heads})
    qh = heads_per_layer(patch_heads, "q", ctx=ctx)
    if n_pivot <= 1:
        q_src = cache_src_proj(src_prompt, [piv_src], q_layers, "q", ctx=ctx)
        pf = [dict(which="q", cache=q_src, heads=qh, positions=[piv_dst], proj=component_proj)]
        dc = [dict(which="q", cache=q_src, heads=qh, positions=[-1], proj=component_proj)]
    else:
        src_win = list(range(piv_src - n_pivot + 1, piv_src + 1))  # cache row r -> dst_win[r]
        dst_win = list(range(piv_dst - n_pivot + 1, piv_dst + 1))
        q_src = cache_src_proj(src_prompt, src_win, q_layers, "q", ctx=ctx)
        pf = [dict(which="q", cache=q_src, heads=qh, positions=dst_win, proj=component_proj)]
        dcache = {l: q_src[l][-1:] for l in q_src}  # pivot-only (last row) for the decode steer
        dc = [dict(which="q", cache=dcache, heads=qh, positions=[-1], proj=component_proj)]
    iface = make_patch_interface(mt, pf)  # PLAIN patch — used for attention (gaze) regardless of alpha

    # OV boost of patch_heads (∪ kv heads) at the decision token, active only on the PATCHED gen + readout
    _boost_heads = sorted(set(patch_heads) | set(kv_heads))

    def _boost_ctx():
        return boost_head_ov(mt, _boost_heads, alpha, positions=(-1,)) if alpha != 1.0 else nullcontext()

    def _patched_iface():
        if alpha == 1.0 and not kv:
            return iface

        def _if(mt, inputs, patches=None, model_kwargs=None):
            with kv_ctx(), proj_patch_hooks(mt, pf), _boost_ctx():
                return mt._model(**dict(inputs), **dict(model_kwargs or {}))

        return _if

    def _kv_iface(mt, inputs, patches=None, model_kwargs=None):   # kv-only (+ boost), for its readout
        with kv_ctx(), _boost_ctx():
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    # optional: blind the patch (goal) heads to everything EXCEPT the task span, so the transplanted
    # query can only read the task-description list (BOS/pos-0 kept as a sink to avoid all-masked rows).
    def _restrict_ctx():
        if not restrict_to_task or not task_positions:
            return nullcontext()
        keepset = set(task_positions)
        blk = [p for p in range(piv_dst + 1) if p not in keepset and p != 0]
        return restrict_head_attn(mt, patch_heads, blk, keep=[])

    # generations
    gen_src = greedy(src_tok, max_new=max_new, ctx=ctx)
    gen_dst = greedy(dst_tok_clean, max_new=max_new, ctx=ctx)
    if kv:
        with kv_ctx(), _boost_ctx():
            kv_extra["gen_kv"] = greedy(dst_tok, max_new=max_new, ctx=ctx)
    cl_extra = {}
    if closed_loop:
        # goal encoding: readout heads' OV write at the DST pivot under the prefill patch (+ boost)
        goal_vec = collect_head_steering_vecs(mt=mt, src_tokenized=dst_tok, selected_heads=list(readout_heads),
                                              token_positions=[piv_dst], patch_interface=_patched_iface())[0][0]
        datatype = getattr(inst, "datatype", None) or inst.task.datatype
        dst_lang = getattr(inst, "dst_lang", "python")
        cl_pivot = "var_eq" if dst_prompt.rstrip().endswith("=") else "indent"
        probe_prompt, probe_prefill = closed_loop_probe_prompt(mt, datatype, inst.var_name, dst_lang, cl_pivot)

        def _pctx():
            s = ExitStack()
            s.enter_context(kv_ctx())
            s.enter_context(_boost_ctx())
            s.enter_context(_restrict_ctx())
            return s

        cl = closed_loop_generate(
            mt, dst_tok, pf, patch_heads, goal_vec, prepare_input(prompts=probe_prompt, tokenizer=mt),
            steer_layers=tuple(closed_loop.get("steer_layers", (35, 46))), alpha=closed_loop.get("alpha", 0.1),
            probe_steer=closed_loop.get("probe_steer", "all"), max_new=closed_loop.get("max_new", 64),
            patched_ctx=_pctx, component_proj=component_proj, ref_q={l: q_src[l][-1:] for l in q_src},
        )
        gen_patched = cl["text"]
        cl_extra = dict(cl_steps=cl["steps"], cl_probe_prefill=probe_prefill)
    else:
        with kv_ctx(), _boost_ctx(), _restrict_ctx():
            gen_patched = greedy(
                dst_tok, prefill_specs=pf, decode_specs=dc, n_steer=n_steer, max_new=max_new, ctx=ctx
            )

    # goalscope readouts (source / destination / patched under the same query patch + boost)
    def _ro(prompt, piv, iface=None):
        return goalscope_readout("", prompt, piv, heads=readout_heads, iface=iface, ctx=ctx)[0][
            "patched_gen"
        ]

    if compute_readout:
        ro_src = _ro(src_prompt, piv_src)
        ro_dst = _ro(dst_prompt, piv_dst_clean)
        ro_patched = _ro(dst_tok if kv else dst_prompt, piv_dst, iface=_patched_iface())
        if kv:
            kv_extra["ro_kv"] = _ro(dst_tok, piv_dst, iface=_kv_iface)
    else:
        ro_src = ro_dst = ro_patched = None

    # value-weighted pivot→TASK attention over the readout (goal) heads, clean vs patched. Under
    # restrict_to_task the patched attention is measured WITH the task-span restriction active, so the
    # ATTN_VIS shows where the (transplanted-query) goal heads focus WITHIN the task list.
    if task_positions is not None:
        if restrict_to_task:
            keepset = set(task_positions)
            _blk = [p for p in range(piv_dst + 1) if p not in keepset and p != 0]

            def _attn_iface(mt, inputs, patches=None, model_kwargs=None):
                with proj_patch_hooks(mt, pf), restrict_head_attn(mt, patch_heads, _blk, keep=[]):
                    return mt._model(**dict(inputs), **dict(model_kwargs or {}))
        else:
            _attn_iface = iface
        clean_i = get_attention_matrices(dst_tok, mt, value_weighted=True)
        patched_i = get_attention_matrices(dst_tok, mt, value_weighted=True, patch_interface=_attn_iface)
        a_clean = pivot_attn(clean_i, list(readout_heads), task_positions, piv_dst)
        a_patched = pivot_attn(patched_i, list(readout_heads), task_positions, piv_dst)
        attn = dict(
            attn_clean=[float(x) for x in a_clean],
            attn_patched=[float(x) for x in a_patched],
            attn_delta=[float(x) for x in (a_patched - a_clean)],
        )
    else:
        attn = dict(attn_clean=None, attn_patched=None, attn_delta=None)

    return dict(
        src_prompt=src_prompt,
        dst_prompt=dst_prompt,
        piv_src=piv_src,
        piv_dst=piv_dst,
        task_positions=task_positions,
        task_tokens=task_tokens,
        gen_src=gen_src,
        gen_dst=gen_dst,
        gen_patched=gen_patched,
        ro_src=ro_src,
        ro_dst=ro_dst,
        ro_patched=ro_patched,
        **attn,
        **cl_extra,
        **kv_extra,
    )


def align_kv_spans(mt, src_prompt, dst_prompt, src_span, dst_span, mode="right",
                   src_text=None, dst_text=None, ctx=None):
    """Choose the transplant token windows for a Spec-2 k/v span transplant (the A4 ``ALIGN_KNOB``).
    Tokenizes both prompts and returns ``(src_tok, dst_tok, src_pos, dst_pos, src_span, dst_span)`` —
    ``*_tok`` are :class:`TokenizerOutput`s (mode="pad" splices tokens, so there is no clean string).

    ``mode="right"`` — right-edge align: ``K = min(|src|, |dst|)``, the last ``K`` tokens of each span
    move; token sequences unchanged (the FRONT of the longer span is left out).
    ``mode="pad"`` — full overwrite: splice ``|len_s − len_d|`` **newline tokens** immediately BEFORE
    the SHORTER target span (at the token-id level — Qwen merges consecutive ``\\n`` in text, so a
    string insert would collapse to one token) so both spans become length ``K = max`` and EVERY token
    in each span is transplanted (no residue). Newlines are neutral padding (vs the EOS pad token,
    whose semantics bleed). Works cross-language."""
    _mt = ctx.mt if ctx is not None else mt
    src_tok = prepare_input(prompts=src_prompt, tokenizer=_mt)
    dst_tok = prepare_input(prompts=dst_prompt, tokenizer=_mt)
    len_s, len_d = src_span[1] - src_span[0], dst_span[1] - dst_span[0]
    if mode == "right":
        K = min(len_s, len_d)
        return (src_tok, dst_tok,
                list(range(src_span[1] - K, src_span[1])),
                list(range(dst_span[1] - K, dst_span[1])),
                src_span, dst_span)
    if mode == "pad":
        nl = _mt.tokenizer("\n", add_special_tokens=False)["input_ids"][-1]  # atomic newline id

        def _splice(tok, span, n):  # insert n newline ids right before span start (mutate in place)
            ids, am = tok["input_ids"], tok["attention_mask"]
            t0 = span[0]
            row = torch.full((ids.shape[0], n), nl, dtype=ids.dtype, device=ids.device)
            ones = torch.ones((am.shape[0], n), dtype=am.dtype, device=am.device)
            tok["input_ids"] = torch.cat([ids[:, :t0], row, ids[:, t0:]], dim=1)
            tok["attention_mask"] = torch.cat([am[:, :t0], ones, am[:, t0:]], dim=1)
            return tok, (span[0], span[1] + n)  # span grows by n -> length K = max

        if len_s < len_d:
            src_tok, src_span = _splice(src_tok, src_span, len_d - len_s)
        elif len_d < len_s:
            dst_tok, dst_span = _splice(dst_tok, dst_span, len_s - len_d)
        # both spans now length K = max; transplant EVERY token in each
        return (src_tok, dst_tok, list(range(*src_span)), list(range(*dst_span)), src_span, dst_span)
    raise ValueError(f"align mode must be 'right'|'pad', got {mode!r}")


def run_kv_transfer_intervention(
    mt,
    inst,
    patch_heads,
    readout_heads=None,
    *,
    max_new=96,
    alpha=1.0,
    align="right",
    ablate_heads=None,
    ablation_strat="zero",
    ctx=None,
):
    """Run one kv-transfer (Spec-2) intervention and return RAW artifacts as JSON-native types.

    ``SRC.task != DST.task`` but both share the same prefilled progress. Cache the SRC
    task-DESCRIPTION span key+value over ``patch_heads`` and transplant it onto the DST task span
    (right-edge aligned — Spec-2 convention), via the GQA-corrected headwise routing
    (:func:`headwise_kv_patch`, so only ``patch_heads`` read the source span). Collects clean-SRC /
    clean-DST / patched-DST generations and the three goalscope readouts (over ``readout_heads``,
    default = ``patch_heads``).

    **No gaze.** In Spec-2 the SRC target never appears in the DST prompt, so there is no SRC-target
    span to gaze at — the eval is causal + readout only (:func:`src.diff.eval.eval_kv_transfer`).

    ``alpha`` != 1.0 boosts the OV write of ``patch_heads`` at the decision token during the PATCHED
    generation and readout (as in :func:`run_q_transfer_intervention`). ``align`` ("right" | "pad";
    see :func:`align_kv_spans`) selects the transplant window: right-edge (drop the longer span's
    front) or left-pad the shorter span so the dest span is fully overwritten. ``ablate_heads``
    (query-head ``(layer, head)`` list) are ablated during the PATCHED GENERATION only (never a patched
    head) — kills the 'backup' goal computation the k/v inconsistency can trigger. ``ablation_strat``
    selects HOW: ``"zero"`` ZEROes their OV write at every position (:func:`boost_head_ov`);
    ``"restrict"`` keeps their write but knocks out their attention to the DST task span
    (:func:`restrict_head_attn`) — they can look everywhere except the task description. The readout is
    NOT ablated (goalscope injects into a neutral probe, no inconsistency).
    Returns a dict consumed directly by :func:`src.diff.eval.eval_kv_transfer` (pass ``patched_gen``
    and ``readouts``)."""
    from src.diff.data import materialize

    readout_heads = readout_heads or patch_heads
    m = materialize(inst, ctx=ctx)
    src_prompt, dst_prompt = m["src_prompt"], m["dst_prompt"]
    src_span = token_span(src_prompt, m["src_task_text"], ctx=ctx)
    dst_span = token_span(dst_prompt, m["dst_task_text"], ctx=ctx)

    # align the two task spans per ``align`` ("right" | "pad"); "pad" token-splices the shorter side,
    # so align returns TokenizerOutputs (not strings) used directly downstream.
    src_tok, dst_tok, src_pos, dst_pos, src_span, dst_span = align_kv_spans(
        mt, src_prompt, dst_prompt, src_span, dst_span, mode=align,
        src_text=m["src_task_text"], dst_text=m["dst_task_text"], ctx=ctx)
    piv_src = src_tok["input_ids"].shape[1] - 1
    piv_dst = dst_tok["input_ids"].shape[1] - 1

    # cache SRC key+value over the patched kv-heads (positions from the alignment above)
    kvheads = heads_per_layer(patch_heads, "k", layers=set(_layers_of(patch_heads, ctx=ctx)), ctx=ctx)
    k_cache = cache_src_proj(src_tok, src_pos, sorted(kvheads), "k", ctx=ctx)
    v_cache = cache_src_proj(src_tok, src_pos, sorted(kvheads), "v", ctx=ctx)
    iface = make_headwise_interface(mt, patch_heads, k_cache, v_cache, dst_pos)  # GQA-corrected
    # heads whose OV write is ZEROED on the PATCHED run (never a patched head)
    ablate = [h for h in (ablate_heads or []) if h not in set(patch_heads)]

    # The PATCHED-run context: k/v transplant + (optional) OV boost of patch_heads at the pivot.
    # ``with_ablate`` adds the OV ablation of `ablate` heads at ALL positions — ON for the GENERATION
    # (counters the k/v inconsistency), OFF for the READOUT (goalscope injects the goal-head write into
    # a neutral, information-less probe, which has no cross-context inconsistency to counter, so the
    # readout logic stays identical across source/dest/patched). Rebuilt per forward (single-use CMs).
    def _patched_stack(with_ablate):
        s = ExitStack()
        s.enter_context(headwise_kv_patch(mt, patch_heads, k_cache, v_cache, dst_pos))
        if alpha != 1.0:
            s.enter_context(boost_head_ov(mt, patch_heads, alpha, positions=(-1,)))
        if with_ablate and ablate:
            if ablation_strat == "restrict":
                s.enter_context(restrict_head_attn(
                    mt, ablate, range(dst_span[0], dst_span[1]), keep=patch_heads))
            else:  # "zero"
                s.enter_context(boost_head_ov(mt, ablate, 0.0, positions=None))
        return s

    def _patched_iface():  # readout iface — transplant (+ boost), NO ablation
        if alpha == 1.0:
            return iface

        def _if(mt, inputs, patches=None, model_kwargs=None):
            with _patched_stack(with_ablate=False):
                return mt._model(**dict(inputs), **dict(model_kwargs or {}))

        return _if

    # generations (patched gen counters the k/v inconsistency with the OV ablation)
    gen_src = greedy(src_tok, max_new=max_new, ctx=ctx)
    gen_dst = greedy(dst_tok, max_new=max_new, ctx=ctx)
    with _patched_stack(with_ablate=True):
        gen_patched = greedy(dst_tok, max_new=max_new, ctx=ctx)

    # goalscope readouts (source / destination / patched under the k+v transplant + boost)
    def _ro(tok, piv, iface=None):
        return goalscope_readout("", tok, piv, heads=readout_heads, iface=iface, ctx=ctx)[0][
            "patched_gen"
        ]

    ro_src = _ro(src_tok, piv_src)
    ro_dst = _ro(dst_tok, piv_dst)
    ro_patched = _ro(dst_tok, piv_dst, iface=_patched_iface())

    return dict(
        src_prompt=src_prompt,
        dst_prompt=dst_prompt,
        piv_src=piv_src,
        piv_dst=piv_dst,
        src_span=list(src_span),
        dst_span=list(dst_span),
        gen_src=gen_src,
        gen_dst=gen_dst,
        gen_patched=gen_patched,
        ro_src=ro_src,
        ro_dst=ro_dst,
        ro_patched=ro_patched,
    )
