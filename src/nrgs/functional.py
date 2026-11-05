"""Interventions, generations and readouts for the q-transfer (``scripts/nrgs/q_transfer_eval.py``) and the
pivot-onwards KV transfer (``src.nrgs.kv_transfer``) — the ``src.nrgs`` submodule's functional core.

Sibling modules in this package:
    ``src.nrgs.data``  — data generation (task banks, instances, prompt materialization).
    ``src.nrgs.eval``  — the oracle judges.

``setup()`` loads the model and binds the module-global ``mt``, which the helpers fall back to when no ``ctx=`` (the
:class:`Ctx` returned by ``setup()``) is passed.
"""

import logging
import os
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import baukit
import numpy as np
import torch
from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

# ── path bootstrap: make `src` importable regardless of the caller's cwd ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from src.goals.goalscope import apply_goalscope
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import logging_utils

logging_utils.configure(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console(theme=Theme({"default": "white"}), highlight=False)


def rich_print(*a, **k):
    console.print(*a, **k)


# ── module globals bound by setup() ──
mt = None
gsp = None  # optional GoalScope probe override (apply_goalscope reads globals().get("gsp"))
# GoalScope readout config used by goalscope_readout when its args are left None (default = the paper's Coder-32B
# config); the eval scripts override it from --readout-strategy / --readout-alpha / --readout-layers
READOUT = dict(strategy="additive_norm", alpha=0.1, layers=(35, 46))


def readout_tag():
    """'' for the default readout config, else e.g. '__ros-replace-a2-L35-46' (keeps non-default runs in own dirs)."""
    r, d = READOUT, dict(strategy="additive_norm", alpha=0.1, layers=(35, 46))
    if (r["strategy"], float(r["alpha"]), tuple(r["layers"])) == (d["strategy"], d["alpha"], d["layers"]):
        return ""
    return f"__ros-{r['strategy']}-a{r['alpha']:g}-L{r['layers'][0]}-{r['layers'][1]}"


@dataclass
class Ctx:
    """Run context: the model, plus the GoalScope readout heads / probe the callers set."""

    mt: object
    model_name: str = None
    readout_heads: list = None
    gsp: object = None


def setup(model_name="Qwen/Qwen2.5-Coder-32B-Instruct", device_map="auto"):
    """Load the model once and bind the module-global ``mt``. Returns a :class:`Ctx`.

    ``device_map`` is forwarded to ``ModelandTokenizer`` (a custom dict rebalances layers across GPUs)."""
    global mt

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mt = ModelandTokenizer(
        model_key=model_name,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
    )
    n_kv = mt.model.config.num_key_value_heads
    print(mt.name)
    print(
        f"n_heads={mt.n_heads}  n_kv_heads={n_kv}  group={mt.n_heads // n_kv}  "
        f"head_dim={mt.head_dim}  n_layer={mt.n_layer}"
    )
    return Ctx(mt=mt, model_name=model_name)


# ══════════════════════════════════════════════════════════════════════════════
# spans + prompt builder + q/k/v patch primitives + readouts
# ══════════════════════════════════════════════════════════════════════════════
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


def get_prompt(question, prefill, ctx=None, system_extra=""):
    _mt = ctx.mt if ctx is not None else mt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant proficient in programming. "
            "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain."
            + system_extra,
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


def _proj_name(l, which, mt):  # which in {'q','k','v','o'}
    base = mt.attn_module_name_format.format(l)
    if which == "q" and _norm_scope(baukit.get_module(mt._model, base)) == "full":
        return base + ".q_norm"  # OLMo-2/3: q is normalized over ALL heads, so cache / patch it after q_norm (exact)
    return base + f".{which}_proj"


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


def heads_per_layer(head_set):
    """{layer: set(query head)}."""
    d = defaultdict(set)
    for l, h in head_set:
        d[l].add(h)
    return dict(d)


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


def goalscope_readout(
    tag,
    prompt,
    pivot,
    heads=None,
    iface=None,
    strategy=None,
    alpha=None,
    layers=None,
    maxnew=128,
    ctx=None,
):
    strategy = READOUT["strategy"] if strategy is None else strategy
    alpha = READOUT["alpha"] if alpha is None else alpha
    layers = tuple(READOUT["layers"]) if layers is None else layers
    _mt = ctx.mt if ctx is not None else mt
    _readout = ctx.readout_heads if ctx is not None else None
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
        patch_interface=iface,
    )
    for r in res:
        rich_print(
            f"[bold]{tag}[/bold]  pos={r['pos']} {r['token']!r} => "
            f"[italic]{escape(r['patched_gen'])}[/italic]"
        )
    return res


def greedy(tok, max_new=64, ctx=None):
    """Greedy decode of up to ``max_new`` tokens (stops at EOS)."""
    _mt = ctx.mt if ctx is not None else mt
    ids, am = tok["input_ids"], tok["attention_mask"]
    with torch.no_grad():
        out = _mt._model(input_ids=ids, attention_mask=am, use_cache=True)
    pkv = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    gen = [int(nxt)]
    with torch.no_grad():
        for step in range(1, max_new):
            am = torch.cat([am, torch.ones_like(nxt)], 1)
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
# A per-kv-head k/v patch feeds the SOURCE k/v to ALL query heads that share a patched kv-head (the GQA
# bleed). Here, at each patched layer, ONLY the selected query heads read the source span; every other head
# reads the clean dest k/v. Implemented by wrapping the eager attention forward of the patched layers:
#   • compute q, k, v as usual; k/v come from the CLEAN cache (span = dest)
#   • out_clean   = eager_attention(q, k, v)                         # all heads, clean
#   • out_patched = eager_attention(q, k⊕src_span, v⊕src_span)       # all heads, source span
#   • out = out_clean; out[:, last query, selected_qheads] = out_patched[...]
# The span is fixed (prefill positions), so the source k (RoPE'd at dest positions) and
# source v are precomputed once. The cache is never patched → clean rides it; routing is
# re-applied inside every forward. Only the patched layers are wrapped.


def _attn_module(mt, layer):
    return baukit.get_module(mt._model, mt.attn_module_name_format.format(layer))


def _norm_scope(attn):
    """Where an attention module applies its q / k RMSNorm: None (Qwen2, Llama), "head" (Qwen3: weight of size
    head_dim, applied per head) or "full" (OLMo-2/3: weight over the whole projection, applied before the split)."""
    qn = getattr(attn, "q_norm", None)
    if qn is None:
        return None
    return "head" if qn.weight.shape[0] == attn.head_dim else "full"


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
        attn_L = _attn_module(mt, L)
        k_full = k_cache[L]  # (S, n_kv * hd), the pre-k_norm k_proj output
        if _norm_scope(attn_L) == "full":  # OLMo-2/3: k_norm over the whole projection, before the head split
            k_full = attn_L.k_norm(k_full.to(attn_L.k_norm.weight.device, attn_L.k_norm.weight.dtype))
        for kvh in patched_kvheads[L]:
            k_slice = k_full[:, kvh * hd : (kvh + 1) * hd].to(dev, dt)[
                None, None
            ]  # (1,1,S,hd)
            if _norm_scope(attn_L) == "head":  # Qwen3: per-head k_norm
                k_slice = attn_L.k_norm(k_slice.to(attn_L.k_norm.weight.device)).to(dev, dt)
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
        q, k = self.q_proj(hidden_states), self.k_proj(hidden_states)
        scope = _norm_scope(self)
        if scope == "full":  # OLMo-2/3: RMSNorm over the whole q / k projection, before the head split
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.view(hidden_shape), k.view(hidden_shape)
        if scope == "head":  # Qwen3: per-head RMSNorm on q / k (Qwen2 / Llama have none)
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
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
        gh = state["qheads"].get(self.layer_idx)
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
            if state["v_scale"] != 1.0:  # x the transplanted SOURCE values on the span
                for kvh in src_v:
                    v2[:, kvh, span, :] = v2[:, kvh, span, :] * state["v_scale"]
            out_patched, _ = eager(self, q, k2, v2, attention_mask, dropout=0.0, scaling=self.scaling,
                                   sliding_window=sw, **kwargs)
            # only the last query of each forward (the pivot on the prompt pass, the new token on each decode
            # step) reads the source, so every earlier position stays bit-identical to the clean DST run
            out = out_clean.clone()
            out[:, -1:, gh, :] = out_patched[:, -1:, gh, :]
        else:
            out = out_clean
        attn_output = out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), w

    return forward


@contextmanager
def headwise_kv_patch(mt, heads, k_cache, v_cache, dest_positions, v_scale=1.0):
    """Wrap the patched layers' attention so ONLY ``heads`` (query heads ``(layer, head)``) read the source span k/v
    (``k_cache`` / ``v_cache``: ``{layer: (len(dest_positions), width)}``, pre-RoPE keys) on ``dest_positions``, at the
    last query of each forward: the pivot on the prompt pass, then each new token on the decode steps, so the
    pre-pivot latents are identical to the clean DST run. All other heads (in-group bled + out-of-group) read the
    clean dest k/v (fixes the GQA bleed). ``v_scale`` multiplies the transplanted source values (keys / attention
    pattern unchanged)."""
    n_kv = getattr(mt, "n_kv_heads", None) or mt.config.num_key_value_heads
    group = mt.n_heads // n_kv  # GQA group size
    qheads, patched_kv = defaultdict(set), defaultdict(set)
    for l, h in heads:
        qheads[l].add(h)
        patched_kv[l].add(h // group)
    qheads = {l: sorted(hs) for l, hs in qheads.items()}
    src_k, src_v = _rope_source_span(
        mt, k_cache, v_cache, dest_positions, {l: sorted(kvs) for l, kvs in patched_kv.items()}
    )
    state = {"qheads": qheads, "span": list(dest_positions), "src_k": src_k, "src_v": src_v, "v_scale": v_scale}
    originals = {}
    try:
        for L in qheads:
            mod = _attn_module(mt, L)
            originals[L] = mod.forward
            mod.forward = types.MethodType(_make_headwise_forward(mod, state), mod)
        yield state
    finally:
        for L, f in originals.items():
            _attn_module(mt, L).forward = f


@contextmanager
def boost_head_ov(mt, heads, alpha):
    """Scale the OV contribution of `heads` (query-head (layer, h)) by `alpha` at the last token of each forward (the
    decision point) — i.e. amplify how much those heads WRITE into the residual stream. o_proj is linear (no bias in
    Qwen2), so scaling a head's o_proj-INPUT slice by `alpha` scales its residual contribution by `alpha`.
    `alpha == 1.0` is a no-op."""
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
                    if alpha == 1.0:
                        return None
                    x = x.clone()
                    for h in hs:
                        x[:, [x.shape[1] - 1], h * hd:(h + 1) * hd] *= alpha
                    return (x,) + tuple(args[1:])

                return pre_hook

            handles.append(mod.register_forward_pre_hook(make(hs)))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


# ══════════════════════════════════════════════════════════════════════════════
# q-transfer intervention runner (gaze-select-restrict)
# ══════════════════════════════════════════════════════════════════════════════
@contextmanager
def whitelist_head_attn(mt, heads, keep, q_from, keep_from=None):
    """clock/09-style attention restriction (``generate_restricted_to_span``), hook-based and KV-cache safe: for each
    ``(layer, head)`` in ``heads``, query rows at ABSOLUTE position >= ``q_from`` may attend ONLY to the key positions
    in ``keep`` and to every key at position >= ``keep_from`` (e.g. the generated tokens; None = none); all other keys
    get the dtype min pre-softmax. Query rows < ``q_from`` (the prompt before the pivot) are untouched. Works for the
    prefill (Q = K) and cached decode (Q = 1, row position = K - 1). Needs eager attention (4-D additive mask)."""
    keep = sorted({int(k) for k in keep})
    by_layer = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)

    def make(hs):
        def pre_hook(module, args, kwargs):
            am = kwargs.get("attention_mask", None)
            pos_idx = None
            if am is None and len(args) >= 3:
                am, pos_idx = args[2], 2
            if am is None or am.dim() != 4:
                raise RuntimeError("whitelist_head_attn needs eager attention with a 4-D additive mask")
            B, H, Q, K = am.shape
            rows = [i for i in range(Q) if K - Q + i >= q_from]
            if not rows:
                return None
            keep_k = {k for k in keep if k < K} | (set(range(keep_from, K)) if keep_from is not None else set())
            blk = [c for c in range(K) if c not in keep_k]
            if not blk:
                return None
            am = am.expand(B, mt.n_heads, Q, K).clone() if H == 1 else am.clone()
            r = torch.tensor(rows, device=am.device)[:, None]
            c = torch.tensor(blk, device=am.device)[None, :]
            for h in hs:
                am[:, h][:, r, c] = torch.finfo(am.dtype).min
            if pos_idx is not None:
                new = list(args)
                new[pos_idx] = am
                return (tuple(new), kwargs)
            return (args, {**kwargs, "attention_mask": am})

        return pre_hook

    handles = []
    try:
        for l, hs in by_layer.items():
            handles.append(_attn_module(mt, l).register_forward_pre_hook(make(hs), with_kwargs=True))
        yield
    finally:
        for hnd in handles:
            hnd.remove()


def run_q_transfer_intervention(
    mt,
    inst,
    patch_heads,
    readout_heads=None,
    *,
    max_new=96,
    alpha=1.0,
    compute_gaze=True,
    compute_readout=True,
    gsr_max_new=64,
    ctx=None,
):
    """Run one q-transfer intervention (a ``DiffTriggerInstance``, pivot = the frontier's ``{var} =`` stub) and return
    all RAW artifacts as JSON-native types. The SRC pivot query over ``patch_heads`` is cached and injected at the DST
    pivot on the prefill (``iface``).

    The PATCHED generation is gaze-select-restrict: (1) the readout heads' value-weighted pivot gaze under the query
    patch (no boost) scores each of DST's listed tasks (mean over its description tokens) and selects the top one;
    (2) a SEPARATE run with NO query transfer, in which, from the pivot on, the patch (goal) heads may read only the
    selected task's span, the pivot itself and the generated tokens (:func:`whitelist_head_attn`), for
    ``gsr_max_new`` tokens.

    Also collects clean-SRC / clean-DST generations (``max_new``), the three goalscope readouts (over
    ``readout_heads``, default = ``patch_heads``; the patched one under the query patch + an ``alpha`` OV boost of
    ``patch_heads`` at the decision token; ``compute_readout=False`` -> None), and value-weighted pivot→TASK attention
    (clean & under the plain query patch) for the gaze eval. Returns a dict consumed directly by
    :func:`src.nrgs.eval.eval_q_transfer`."""
    from src.attention import get_attention_matrices
    from src.nrgs.data import materialize

    readout_heads = readout_heads or patch_heads
    m = materialize(inst, ctx=ctx)
    src_prompt, dst_prompt = m["src_prompt"], m["dst_prompt"]

    def _piv(p):
        return prepare_input(prompts=p, tokenizer=mt)["input_ids"].shape[1] - 1

    piv_src, piv_dst = _piv(src_prompt), _piv(dst_prompt)
    src_tok = prepare_input(prompts=src_prompt, tokenizer=mt)
    dst_tok = prepare_input(prompts=dst_prompt, tokenizer=mt)

    # TASK span (gaze only): the union over the numbered predicate list
    if compute_gaze:
        spans = [token_span(dst_prompt, p.description, ctx=ctx) for p in inst.task.predicates]
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        task_positions = list(range(lo, hi))
        task_tokens = [mt.tokenizer.decode([int(dst_tok["input_ids"][0][j])]) for j in task_positions]
    else:
        task_positions, task_tokens = None, None

    # query transplant spec: SRC pivot query -> DST pivot, on the prefill
    q_layers = sorted({l for l, _ in patch_heads})
    q_src = cache_src_proj(src_prompt, [piv_src], q_layers, "q", ctx=ctx)
    pf = [dict(which="q", cache=q_src, heads=heads_per_layer(patch_heads), positions=[piv_dst])]
    iface = make_patch_interface(mt, pf)  # PLAIN patch — used for attention (gaze) regardless of alpha

    # OV boost of patch_heads at the decision token, active only on the PATCHED readout
    _boost_heads = sorted(set(patch_heads))

    def _patched_iface():
        if alpha == 1.0:
            return iface

        def _if(mt, inputs, patches=None, model_kwargs=None):
            with proj_patch_hooks(mt, pf), boost_head_ov(mt, _boost_heads, alpha):
                return mt._model(**dict(inputs), **dict(model_kwargs or {}))

        return _if

    # generations
    gen_src = greedy(src_tok, max_new=max_new, ctx=ctx)
    gen_dst = greedy(dst_tok, max_new=max_new, ctx=ctx)

    # (1) selection: pivot gaze (readout heads, value-weighted) under the query patch, no boost
    sel_preds = list(inst.task.predicates)
    sel_items = {p.id: list(range(*token_span(dst_prompt, p.description, ctx=ctx))) for p in sel_preds}
    gi = get_attention_matrices(dst_tok, mt, value_weighted=True, patch_interface=iface)
    gi = gi[0] if isinstance(gi, list) else gi
    span = sorted({q for v in sel_items.values() for q in v})
    at = dict(zip(span, pivot_attn(gi, list(readout_heads), span, piv_dst)))
    scores = {pid: float(np.mean([at[q] for q in ps])) for pid, ps in sel_items.items()}
    selected = next(p for p in sel_preds if p.id == max(scores, key=scores.get))

    # (2) restricted run (clock/09 §5): NO query transfer; from the pivot on, the patch (goal) heads read only the
    # selected task's span, the pivot ` =` (the anchor) and the generated tokens
    keep = sorted(set(sel_items[selected.id]) | {piv_dst})
    gen_from = dst_tok["input_ids"].shape[1]
    with torch.no_grad():
        ids_, am_ = dst_tok["input_ids"], dst_tok["attention_mask"]
        with whitelist_head_attn(mt, patch_heads, keep, piv_dst, keep_from=gen_from):
            out_ = mt._model(input_ids=ids_, attention_mask=am_, use_cache=True)
        pkv_, nxt_, gen_ = out_.past_key_values, out_.logits[:, -1].argmax(-1, keepdim=True), []
        while len(gen_) < gsr_max_new and int(nxt_) != mt.tokenizer.eos_token_id:
            gen_.append(int(nxt_))
            am_ = torch.cat([am_, torch.ones_like(am_[:, :1])], 1)
            with whitelist_head_attn(mt, patch_heads, keep, piv_dst, keep_from=gen_from):
                out_ = mt._model(input_ids=nxt_, attention_mask=am_, past_key_values=pkv_, use_cache=True)
            pkv_, nxt_ = out_.past_key_values, out_.logits[:, -1].argmax(-1, keepdim=True)
    gen_patched = mt.tokenizer.decode(gen_, skip_special_tokens=True)
    cl_extra = dict(cl_probe_mode="gaze_select_restrict", cl_inject=False, cl_restrict_keep=keep,
                    cl_select_scores=scores, cl_selected=selected.id, cl_select_options=[p.id for p in sel_preds])

    # goalscope readouts (source / destination / patched under the query patch + boost)
    def _ro(prompt, piv, iface=None):
        return goalscope_readout("", prompt, piv, heads=readout_heads, iface=iface, ctx=ctx)[0][
            "patched_gen"
        ]

    if compute_readout:
        ro_src = _ro(src_prompt, piv_src)
        ro_dst = _ro(dst_prompt, piv_dst)
        ro_patched = _ro(dst_prompt, piv_dst, iface=_patched_iface())
    else:
        ro_src = ro_dst = ro_patched = None

    # value-weighted pivot→TASK attention over the readout (goal) heads, clean vs patched
    if task_positions is not None:
        clean_i = get_attention_matrices(dst_tok, mt, value_weighted=True)
        patched_i = get_attention_matrices(dst_tok, mt, value_weighted=True, patch_interface=iface)
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
    )


def align_kv_spans(mt, src_prompt, dst_prompt, src_span, dst_span, ctx=None):
    """Transplant token windows for a k/v span transplant ("A1" alignment). Tokenizes both prompts and splices
    ``|len_s − len_d|`` **newline tokens** immediately BEFORE the SHORTER span (at the token-id level — Qwen merges
    consecutive ``\\n`` in text, so a string insert would collapse to one token), so both spans become length
    ``K = max`` and EVERY token in each span is transplanted. Newlines are neutral padding (vs the EOS pad token,
    whose semantics bleed). Works cross-language. Returns ``(src_tok, dst_tok, src_pos, dst_pos)`` — ``*_tok`` are
    :class:`TokenizerOutput`s (the splice leaves no clean string)."""
    _mt = ctx.mt if ctx is not None else mt
    src_tok = prepare_input(prompts=src_prompt, tokenizer=_mt)
    dst_tok = prepare_input(prompts=dst_prompt, tokenizer=_mt)
    len_s, len_d = src_span[1] - src_span[0], dst_span[1] - dst_span[0]
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
    return src_tok, dst_tok, list(range(*src_span)), list(range(*dst_span))
