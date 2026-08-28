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

import os
import sys
import types
import logging
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import numpy as np
import baukit
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

# ── path bootstrap: make `src` importable regardless of the caller's cwd ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from src.utils import logging_utils, env_utils
from src.tokens import prepare_input
from src.models import ModelandTokenizer
from src.goals.head_loc_attn import HeadMask, collect_head_steering_vecs
from src.goals.goalscope import apply_goalscope, head_additive_generate
from src.goals.oid_gaze import cache_root
from src.attention import get_attention_matrices, _safe_token
from circuitsvis.tokens import colored_tokens
from IPython.display import display

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


def mask_bin(heads):
    m = HeadMask(N_LAYER, N_HEADS)
    with torch.no_grad():
        m.values.zero_()
        for l, h in heads:
            m.values[l, h] = 1.0
    return m


def setup(model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
          cuda_visible_devices="0, 2, 3",
          goal_threshold=0.9,
          boundary_cos_thr=0.7,
          boundary_sel_thr=0.8):
    """Load the model + trained head masks once; bind module globals. Returns a namespace."""
    global mt, MODEL_NAME, GROUP, N_KV, N_LAYER, N_HEADS
    global GOAL_HEADS, BOUNDARY_HEADS, UNION_HEADS, GOAL_MASK, BOUNDARY_MASK, READOUT_HEADS

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
    print(f"n_heads={mt.n_heads}  n_kv_heads={N_KV}  group={GROUP}  "
          f"head_dim={mt.head_dim}  n_layer={mt.n_layer}")

    # GOAL heads (READOUT) — trained localization mask (lam20), mid-band, thresholded.
    _gckpt = torch.load(
        os.path.join(RES, f"localization/{model_name.split('/')[-1]}/lam20/head_mask.pt"),
        map_location="cpu")
    GOAL_LAYERS = _gckpt["layers"]
    _goal_trained = HeadMask(len(GOAL_LAYERS), _gckpt["mask_values"].shape[1])
    with torch.no_grad():
        _goal_trained.values.copy_(_gckpt["mask_values"])
        if model_name.split("/")[-1] == "Qwen2.5-Coder-32B-Instruct":
            _goal_trained.values[0:25] = 0
            _goal_trained.values[56:] = 0
    GOAL_HEADS = sorted(_goal_trained.selected_heads(GOAL_LAYERS, threshold=goal_threshold))
    GOAL_MASK = mask_bin(GOAL_HEADS)

    # BOUNDARY heads (DECISION) — move-on/stay separation (cos < thr) from clock/15's aff_sep.
    _bnd_root = os.path.join(cache_root(model_name, "conditions"), "completion_mask")
    _bnd_dir = max((t for t in os.listdir(_bnd_root)
                    if os.path.isdir(os.path.join(_bnd_root, t)) and "lam10.0" in t),
                   key=lambda t: os.path.getmtime(os.path.join(_bnd_root, t)))
    _DCK = torch.load(os.path.join(_bnd_root, _bnd_dir, "mask.pt"),
                      map_location="cpu", weights_only=False)
    _dec_layers = [int(x) for x in _DCK["layers"]]
    _dmv = _DCK["mask_values"].numpy()
    _ = sorted((_dec_layers[li], int(h)) for li, h in np.argwhere(_dmv > boundary_sel_thr))  # BOUNDARY_ALL
    _aff = torch.load(os.path.join(RES, "notebook_caches/15_order_dependency/aff_sep__d5fbad08ba.pt"),
                      map_location="cpu", weights_only=False)["value"]
    BOUNDARY_HEADS = sorted(lh for lh, c in _aff.items() if c < boundary_cos_thr)
    BOUNDARY_MASK = mask_bin(BOUNDARY_HEADS)

    UNION_HEADS = sorted(set(GOAL_HEADS) | set(BOUNDARY_HEADS))
    READOUT_HEADS = GOAL_HEADS
    print(f"GOAL_HEADS     : {len(GOAL_HEADS):3d}  "
          f"(layers {min(l for l, _ in GOAL_HEADS)}–{max(l for l, _ in GOAL_HEADS)})")
    print(f"BOUNDARY_HEADS : {len(BOUNDARY_HEADS):3d}  "
          f"(layers {min(l for l, _ in BOUNDARY_HEADS)}–{max(l for l, _ in BOUNDARY_HEADS)})")
    print(f"UNION_HEADS    : {len(UNION_HEADS):3d}")

    return types.SimpleNamespace(
        mt=mt, GROUP=GROUP, N_KV=N_KV, N_LAYER=N_LAYER, N_HEADS=N_HEADS,
        GOAL_HEADS=GOAL_HEADS, BOUNDARY_HEADS=BOUNDARY_HEADS, UNION_HEADS=UNION_HEADS,
        GOAL_MASK=GOAL_MASK, BOUNDARY_MASK=BOUNDARY_MASK)


# ══════════════════════════════════════════════════════════════════════════════
# prompt display / spans  (00_diff.ipynb cell 4)
# ══════════════════════════════════════════════════════════════════════════════
def exact_prompt(prompt, tail_tokens=None, patch_pos=[-1], emphasize_tokens=None):
    """Prompt EXACTLY as the model sees it: patch_pos (pivot) in salmon, emphasize_tokens in yellow."""
    ids = prepare_input(prompts=prompt, tokenizer=mt).input_ids[0].tolist()
    piv = {len(ids) + p if p < 0 else p for p in patch_pos}
    emph = set(emphasize_tokens or ())
    lo = 0 if tail_tokens is None else max(0, len(ids) - tail_tokens)
    OPEN, CLOSE, EOPEN, ECLOSE = "\x00<", "\x00>", "\x01<", "\x01>"
    parts = []
    for i in range(lo, len(ids)):
        s = mt.tokenizer.decode([ids[i]]).replace("\n", "\\n\n")
        if i in piv:
            s = OPEN + s + CLOSE
        elif i in emph:
            s = EOPEN + s + ECLOSE
        parts.append(s)
    txt = escape("".join(parts))
    txt = (txt.replace(OPEN, "[bold black on salmon1]").replace(CLOSE, "[/bold black on salmon1]")
              .replace(EOPEN, "[yellow]").replace(ECLOSE, "[/yellow]"))
    head = f"[grey42]… {lo} earlier tokens elided …[/grey42]\n" if lo else ""
    return head + txt


def pivot_token(prompt, patch_pos=[-1]):
    ids = prepare_input(prompts=prompt, tokenizer=mt).input_ids[0].tolist()
    return [mt.tokenizer.decode([ids[len(ids) + p if p < 0 else p]]) for p in patch_pos]


def token_span(prompt, substring):
    """[lo, hi) token indices of `substring` inside the tokenization of `prompt` (via char offsets)."""
    enc = prepare_input(prompts=prompt, tokenizer=mt, return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].tolist()
    ci = prompt.index(substring)
    cj = ci + len(substring)
    toks = [k for k, (a, b) in enumerate(offs) if b > a and a < cj and b > ci]
    assert toks, f"substring not found in tokenization: {substring!r}"
    return min(toks), max(toks) + 1


def show_prompt(tag, prompt, span):
    rich_print(Panel(
        Text.from_markup(exact_prompt(prompt, patch_pos=[-1], emphasize_tokens=range(*span))),
        title=f"[bold]{tag}[/bold]  TASK span {span}  ·  pivot={pivot_token(prompt)!r}  ·  "
              f"L={prepare_input(prompts=prompt, tokenizer=mt)['input_ids'].shape[1]}",
        title_align="left", border_style="grey42", padding=(1, 1)))


# ══════════════════════════════════════════════════════════════════════════════
# prompt builder + q/k/v patch primitives + readouts  (00_diff.ipynb cell 6)
# ══════════════════════════════════════════════════════════════════════════════
def get_prompt(question, prefill):
    messages = [
        {"role": "system", "content": "You are a helpful assistant proficient in programming. "
         "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    return mt.tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, continue_final_message=True,
        tokenize=False, chat_template_kwargs={"enable_thinking": False})


def _proj_name(l, which):  # which in {'q','k','v'}
    return mt.attn_module_name_format.format(l) + f".{which}_proj"


def cache_src_proj(prompt, positions, layers, which):
    """{layer: (len(positions), width)} — source {which}_proj outputs at `positions` (pre-RoPE for q/k)."""
    tok = prepare_input(prompts=prompt, tokenizer=mt)
    names = {l: _proj_name(l, which) for l in layers}
    with torch.no_grad(), baukit.TraceDict(mt._model, list(names.values()),
                                           retain_input=False, retain_output=True) as tr:
        mt._model(**{k: v for k, v in tok.items() if k in ("input_ids", "attention_mask")})
    return {l: tr[names[l]].output[0, positions].detach().float().cpu() for l in layers}


def heads_per_layer(head_set, which, layers=None):
    """{layer: set(head)} — q-head index for which=="q", else kv-head index (h // GROUP)."""
    d = defaultdict(set)
    for (l, h) in head_set:
        if layers is None or l in layers:
            d[l].add(h if which == "q" else h // GROUP)
    return dict(d)


def _layers_of(head_set):
    hs = UNION_HEADS if head_set == "ALL" else head_set
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
            which, cache, headmap, positions = spec["which"], spec["cache"], spec["heads"], spec["positions"]
            for layer, hset in headmap.items():
                mod = baukit.get_module(mt._model, _proj_name(layer, which))
                src_l = cache[layer]           # (K, width)

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
    A = info.attention_matrices                # (L, H, q, s)
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
    print(f"total mass on span:  {ref_name}={a_ref.sum():.3f}  patched={a_patch.sum():.3f}")


def goalscope_readout(tag, prompt, pivot, heads=None, iface=None,
                      strategy="additive_norm", alpha=0.1, layers=(35, 46), maxnew=128):
    res = apply_goalscope(prompt=prompt, mt=mt, selected_heads=heads or READOUT_HEADS,
                          strategy=strategy, alpha=alpha, steer_layer_range=layers, batch_size=1,
                          max_new_tokens=maxnew, goalscope_prompt=globals().get("gsp"),
                          token_positions=[pivot], include_noncode=False, show=False,
                          patch_interface=iface)
    for r in res:
        rich_print(f"[bold]{tag}[/bold]  pos={r['pos']} {r['token']!r} => "
                   f"[italic]{escape(r['patched_gen'])}[/italic]")
    return res


def greedy(tok, prefill_specs=None, decode_specs=None, n_steer=1, max_new=64):
    """Greedy decode. `prefill_specs` patched on the prefill (rides the KV cache after); `decode_specs`
    re-applied at the current token for the first `n_steer` forward passes (prefill = pass 1)."""
    ids, am = tok["input_ids"], tok["attention_mask"]
    ctx0 = proj_patch_hooks(mt, prefill_specs) if prefill_specs else nullcontext()
    with torch.no_grad(), ctx0:
        out = mt._model(input_ids=ids, attention_mask=am, use_cache=True)
    pkv = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [int(nxt)]
    with torch.no_grad():
        for step in range(1, max_new):
            am = torch.cat([am, torch.ones_like(nxt)], 1)
            use = decode_specs if (decode_specs and step < n_steer) else None
            ctx = proj_patch_hooks(mt, use) if use else nullcontext()
            with ctx:
                out = mt._model(input_ids=nxt, attention_mask=am, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            t = int(nxt)
            if t == mt.tokenizer.eos_token_id:
                break
            gen.append(t)
    return mt.tokenizer.decode(gen, skip_special_tokens=True)


def show_delta_tokens(tokens, delta, label=""):
    """Diverging red(neg)/green(pos) per-token coloring of an attention delta, symmetric about 0."""
    d = np.asarray(delta, dtype=float)
    M = float(np.abs(d).max()) or 1.0
    if label:
        rich_print(f"[grey58]{label}   (green +Δ / red −Δ, |Δ|max={M:.3f})[/grey58]")
    display(colored_tokens(tokens=[_safe_token(t) for t in tokens], values=d,
                           min_value=-M, max_value=M, negative_color="red", positive_color="green"))


# ══════════════════════════════════════════════════════════════════════════════
# NEW helpers for the DIFF experiments (query arithmetic · OV-write · cosines)
# ══════════════════════════════════════════════════════════════════════════════
def q_lincomb(terms):
    """Linear combination of {layer: (K, width)} proj caches.  `terms` = list of (coef, cache).

    All caches must share the same layer keys and shapes (same K, width). Used to synthesise a
    query cache, e.g. q({a})+q({b})-q({}) for the compositional-injection test (B2)."""
    layers = list(terms[0][1].keys())
    return {l: sum(coef * cache[l] for coef, cache in terms) for l in layers}


def head_write(prompt, heads, positions, iface=None):
    """Aggregated residual-stream write of `heads` at `positions` — the SAME object GoalScope reads.

    Wraps ``collect_head_steering_vecs`` (sum over heads of o_proj_input[h] @ W_o[:, h].T).
    Returns (len(positions), n_embd) float32 CPU.  `iface` runs the extraction under an intervention."""
    tok = prepare_input(prompts=prompt, tokenizer=mt)
    vecs = collect_head_steering_vecs(mt, tok, list(heads), list(positions), patch_interface=iface)
    return vecs[0]  # (n_pos, n_embd)


def write_norm(prompt, heads, pivot, iface=None):
    """L2 norm of the aggregated head write at a single pivot position."""
    return float(head_write(prompt, heads, [pivot], iface=iface)[0].norm())


def per_head_cosine(cache_a, cache_b, heads, which):
    """Per-head cosine between two proj caches (both {layer:(1,width)}), sliced to `heads`.

    `which`=="q" slices q-head h; otherwise slices kv-head h//GROUP.  Returns {(layer,head): cos}."""
    hd = mt.head_dim
    out = {}
    for (l, h) in heads:
        if l not in cache_a or l not in cache_b:
            continue
        idx = h if which == "q" else h // GROUP
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
