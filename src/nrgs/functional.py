"""Interventions, generations and readouts for the q-transfer (``scripts/nrgs/q_transfer_eval.py``) and the
pivot-onwards KV transfer (``src.nrgs.kv_transfer``) — the ``src.nrgs`` submodule's functional core.

Sibling modules in this package:
    ``src.nrgs.data``  — data generation (task banks, instances, prompt materialization).
    ``src.nrgs.eval``  — the oracle judges.

``setup()`` loads the model and binds the module-global ``mt``, which the helpers fall back to when no ``ctx=`` (the
:class:`Ctx` returned by ``setup()``) is passed.

Every intervention here is an :class:`Intervention`: a callable ``iv(mt, step=0)`` that edits nnterp values inside a
trace. :func:`greedy` calls it at every step of ``mt.generate`` (step 0 = the prompt pass); a single-forward reader
(``apply_goalscope``'s source trace, ``get_attention_matrices``) takes it as ``interventions=`` and calls ``iv(mt)``.
Interventions compose with ``+``, which keeps their edits in forward order.
"""

import logging
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

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

from src.functional import _LOOP_CUT_SHORT
from src.goals.goalscope import apply_goalscope
from src.models import ModelandTokenizer
from src.tokens import chat_prompt, prepare_input
from src.utils import logging_utils

logging_utils.configure(level=logging.WARNING)
logger = logging.getLogger(__name__)
console = Console(theme=Theme({"default": "white"}), highlight=False)


def rich_print(*a, **k):
    console.print(*a, **k)


# ── module globals bound by setup() ──
mt = None
gsp = None  # optional GoalScope probe override (apply_goalscope reads globals().get("gsp"))


# The GoalScope readout's steer layers (half-open) on the paper's 64-layer 32B models.
READOUT_LAYERS = (35, 46)
READOUT_ALPHA = 0.1  # the GoalScope readout's additive_norm alpha (the paper's)


def readout_layers_for(n_layer: int) -> tuple[int, int]:
    """:data:`READOUT_LAYERS` scaled by depth: ``(round(L * 35 / 64), round(L * 46 / 64))``, the paper's (35, 46) on a
    64-layer model (``scripts/cross_family_check.py`` scales them the same way)."""
    return round(n_layer * READOUT_LAYERS[0] / 64), round(n_layer * READOUT_LAYERS[1] / 64)


def readout_layers_tag(layers, default, readout=True) -> str:
    """Run-tag suffix of the NRGS scripts for the readout's steer layers: ``__rl<a>-<b>`` when they differ from the
    model's default ``default`` (:func:`readout_layers_for` its depth), else (and without readouts) empty, so the
    paper's run dirs keep their names."""
    return f"__rl{layers[0]}-{layers[1]}" if readout and tuple(layers) != tuple(default) else ""


def readout_alpha_tag(alpha, readout=True) -> str:
    """Run-tag suffix for the readout's alpha: ``__ra<alpha>`` when it is not :data:`READOUT_ALPHA`, else (and
    without readouts) empty, so the paper's run dirs keep their names."""
    return f"__ra{alpha:g}" if readout and alpha is not None and alpha != READOUT_ALPHA else ""


@dataclass
class Ctx:
    """Run context: the model, plus the GoalScope readout heads / probe / steer layers the callers set
    (``readout_layers``: :func:`setup` sets :func:`readout_layers_for` the model's depth)."""

    mt: object
    model_name: str = None
    readout_heads: list = None
    gsp: object = None
    readout_layers: tuple = None
    readout_alpha: float = None  # None: READOUT_ALPHA


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
    return Ctx(mt=mt, model_name=model_name, readout_layers=readout_layers_for(mt.n_layer))


# ══════════════════════════════════════════════════════════════════════════════
# the intervention vocabulary
# ══════════════════════════════════════════════════════════════════════════════
# An nnsight invoke must touch values in forward order: an earlier value touched after a later one raises
# ``OutOfOrderError`` in a trace and, inside ``tracer.iter``, silently binds the next step's occurrence. Where a value
# sits in the forward is nnterp's to say: ``mt.rank(name, layer)`` (``mt.order(layer)`` lists a block's values in the
# order the forward reaches them, measured by one probe scan per model). An Intervention sorts its edits by it when it
# is built, outside any trace.


@dataclass(frozen=True)
class Edit:
    """One piece of an :class:`Intervention`: ``fn(mt, step)`` touches the value ``name`` of block ``layer``, named
    as ``mt.support(layer)`` / ``mt.rank`` name it (``"self_attn.attention_scores"``, ``"layer_output"``); ``layer``
    is ``None`` for a root value (``"token_embeddings"``). ``first_only`` edits run on the prompt pass (step 0) only."""

    layer: int | None
    name: str
    fn: Callable
    first_only: bool = False


class Intervention:
    """An NRGS intervention: ``iv(mt, step=0)`` applies its :class:`Edit` s, in forward order, inside a trace.

    - Under ``mt.generate``, call it once per step (``for step in tracer.iter[:n]: iv(mt, step)``); :func:`greedy`
      does, for ``n = max_new`` when :attr:`every_step` and ``n = 1`` otherwise.
    - In a single forward, pass it as ``interventions=`` to ``apply_goalscope`` / ``collect_head_steering_vecs`` /
      ``get_o_proj_inputs_batch`` / ``get_attention_matrices``, which call ``iv(mt)`` (step 0).
    - ``a + b`` merges two interventions; edits at one location (equal rank) keep the order ``a`` then ``b``.
      ``len(iv)`` is its number of edits, so an empty intervention is falsy.

    The edits bind their envoys when the intervention is built, so build one per model (``mt``, which ranks them).
    An intervention that keeps state between its edits within a step (the KV transfer) must not run in two invokes
    of one trace."""

    def __init__(self, edits=(), mt=None):
        edits = list(edits)
        if edits and mt is None:
            raise ValueError("an Intervention with edits takes the model they bind to (mt=), which orders them")
        self.mt = mt
        self.edits = sorted(edits, key=lambda e: mt.rank(e.name, e.layer))

    def __call__(self, mt, step=0):
        for e in self.edits:
            if step == 0 or not e.first_only:
                e.fn(mt, step)

    @property
    def every_step(self) -> bool:
        """Whether any edit runs on the decode steps (else only the prompt pass needs a step)."""
        return any(not e.first_only for e in self.edits)

    def __add__(self, other: "Intervention") -> "Intervention":
        return Intervention(list(self.edits) + list(other.edits), self.mt if self.mt is not None else other.mt)

    def __len__(self):
        return len(self.edits)


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
    return chat_prompt(_mt, messages, add_generation_prompt=False, continue_final_message=True)


# The queries and keys before the rotary embedding (after any query / key norm), and the values: what a transplant to
# another position writes, since the block then rotates it for the position it lands at.
QKV = {"q": "attention_queries_prerope", "k": "attention_keys_prerope", "v": "attention_values"}


def cache_src_qkv(prompt, positions, layers, which, ctx=None):
    """{layer: (heads, len(positions), head_dim)} — the source's ``self_attn.<QKV[which]>`` at `positions` (CPU
    float32): ``which="q"`` the queries before the rotary embedding (``num_heads`` wide), ``"k"`` the keys before it,
    ``"v"`` the values (``num_kv_heads`` wide). `prompt` may be a string OR a pre-tokenized TokenizerOutput (e.g. a
    token-spliced pad alignment)."""
    _mt = ctx.mt if ctx is not None else mt
    tok = prepare_input(prompts=prompt, tokenizer=_mt) if isinstance(prompt, str) else prompt
    layers = sorted(layers)
    saves = {}
    with torch.no_grad(), _mt.trace(dict(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])):
        for l in layers:
            saves[l] = getattr(_mt.layers[l].self_attn, QKV[which])[0, :, positions].save()
    return {l: saves[l].detach().float().cpu() for l in layers}


def heads_per_layer(head_set):
    """{layer: set(query head)}."""
    d = defaultdict(set)
    for l, h in head_set:
        d[l].add(h)
    return dict(d)


def qkv_patch(mt, patch_specs) -> Intervention:
    """Replace heads of ``self_attn.<QKV[which]>`` (the queries or keys before the rotary embedding, or the values)
    at prompt `positions` with cached source vectors, on the prompt pass only (the effect then rides the KV cache).

    ``patch_specs``: dicts with ``which`` (``"q"`` | ``"k"`` | ``"v"``), ``cache`` (``{layer: (heads,
    len(positions), head_dim)}``, e.g. :func:`cache_src_qkv`), ``heads`` (``{layer: {head}}``; kv heads for k / v)
    and ``positions`` (absolute prompt positions; negative ones count from the prompt's end). The block rotates a
    query or key written before its rotary embedding for the position it lands at, so a source vector moved to
    another position is rotated there by the model itself."""
    edits = []
    for spec in patch_specs:
        which, cache, headmap, positions = spec["which"], spec["cache"], spec["heads"], spec["positions"]
        for layer, hset in headmap.items():
            attn = mt.layers[layer].self_attn

            def edit(mt_, step, attn=attn, name=QKV[which], hs=sorted(hset), src_l=cache[layer], positions=positions):
                x = getattr(attn, name)  # (B, heads, seq, head_dim), edited in place
                seq = x.shape[2]
                pos = [p if p >= 0 else seq + p for p in positions]
                for r, p in enumerate(pos):
                    if 0 <= p < seq:
                        x[:, hs, p] = src_l[hs, r].to(x.device, x.dtype)

            edits.append(Edit(layer, f"self_attn.{QKV[which]}", edit, first_only=True))
    return Intervention(edits, mt)


def pivot_attn(info, heads, positions, pivot):
    """mean over `heads` of value-weighted attention from `pivot` onto `positions` -> (len(positions),)."""
    A = info.attention_matrices  # (L, H, q, s)
    return np.stack([A[l, h, pivot, positions] for (l, h) in heads]).mean(0)


def goalscope_readout(
    tag,
    prompt,
    pivot,
    heads=None,
    interventions=None,
    strategy="additive_norm",
    alpha=None,
    layers=None,
    maxnew=128,
    ctx=None,
):
    """GoalScope readout at `pivot`; the source vector is read under `interventions` (an :class:`Intervention`,
    e.g. the KV stack or a query patch), the readout generation runs without it. ``layers`` (the steer layers,
    half-open) defaults to ``ctx.readout_layers``, else :func:`readout_layers_for` the model's depth; ``alpha`` to
    ``ctx.readout_alpha``, else :data:`READOUT_ALPHA`."""
    _mt = ctx.mt if ctx is not None else mt
    if alpha is None:
        alpha = ctx.readout_alpha if ctx is not None and ctx.readout_alpha is not None else READOUT_ALPHA
    if layers is None:
        layers = ctx.readout_layers if ctx is not None and ctx.readout_layers else readout_layers_for(_mt.n_layer)
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
        interventions=interventions,
    )
    for r in res:
        rich_print(
            f"[bold]{tag}[/bold]  pos={r['pos']} {r['token']!r} => "
            f"[italic]{escape(r['patched_gen'])}[/italic]"
        )
    return res


def _generate(mt, tok, max_new, interventions=None, stopping_criteria=None):
    """The decode driver: ``mt.generate`` + ``tracer.iter``, batch 1, plain greedy argmax, stops at
    ``mt.eos_token_ids`` (and wherever ``stopping_criteria``, a ``StoppingCriteriaList``, says);
    ``interventions(mt, step)`` at every step (step 0 = the prompt pass), or at step 0 only when no edit of it runs
    on the decode steps. Returns the new tokens, decoded without special tokens."""
    n_prompt = tok["input_ids"].shape[1]
    # an empty Intervention is falsy; a plain callable may edit any step
    n_steps = 0 if not interventions else (max_new if getattr(interventions, "every_step", True) else 1)
    with torch.no_grad(), warnings.catch_warnings():
        # A run that stops at EOS before step n_steps cuts the step loop short (nnsight warns); the edits of the
        # steps that ran have landed, and tracer.result is read in its own invoke so it is never dropped.
        warnings.filterwarnings("ignore", message=_LOOP_CUT_SHORT, category=UserWarning)
        # repetition_penalty overrides the checkpoint's generation_config (Qwen2.5: 1.05-1.1), so this is the plain
        # argmax loop the hand-written decoders ran
        with mt.generate(max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                         eos_token_id=mt.eos_token_ids, stopping_criteria=stopping_criteria) as tracer:
            with tracer.invoke(dict(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])):
                if n_steps:
                    for step in tracer.iter[:n_steps]:
                        interventions(mt, step)
            with tracer.invoke():
                out = tracer.result.save()
    return mt.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)


def greedy(tok, max_new=64, ctx=None, interventions=None):
    """Greedy decode of up to ``max_new`` tokens (stops at EOS), under ``interventions`` (an :class:`Intervention`
    applied at every step) when given."""
    _mt = ctx.mt if ctx is not None else mt
    return _generate(_mt, tok, max_new, interventions)


# ══════════════════════════════════════════════════════════════════════════════
# GQA-bleed-corrected k/v transplant — per-query-head routing (single trajectory)
# ══════════════════════════════════════════════════════════════════════════════
# A per-kv-head k/v patch feeds the SOURCE k/v to ALL query heads that share a patched kv-head (the GQA bleed). Here,
# at each patched layer, ONLY the selected query heads read the source span; every other head reads the clean dest
# k/v. nnterp serves the attention's inputs (attention_queries / attention_keys / attention_values, keys and values
# before repeat_kv, the whole cache on a decode step), the attention function the block called (self_attn.interface,
# bound to this step's mask and scale) and its per-head outputs (attention_head_outputs), so per step and patched
# layer:
#   • read the interface and q, k, v (clean: the cache is never patched, so clean rides it)
#   • call the interface on copies of k and v with the span of the patched kv heads replaced by the source's (values
#     x v_scale)
#   • write the selected heads' row of attention_head_outputs at the LAST query (the pivot on the prompt pass, the
#     new token on each decode step), so every earlier position stays bit-identical to the clean DST run.
# The call is the model's own attention (a logit softcap, a sink, the sliding window included) on every query head and
# every row, the forward's own GEMM shapes, so the selected heads' last row is bit for bit what the block computes over
# the replaced k / v. The span is fixed (prefill positions), so the source keys (rotated for the DEST positions by the
# block's own rotary embedding) and values are computed once.


def headwise_kv_patch(mt, heads, k_cache, v_cache, dest_positions, v_scale=1.0) -> Intervention:
    """ONLY ``heads`` (query heads ``(layer, head)``) read the source span k/v (``k_cache`` / ``v_cache``:
    ``{layer: (num_kv_heads, len(dest_positions), head_dim)}``, :func:`cache_src_qkv`'s keys before the rotary
    embedding and values) on ``dest_positions``, at the last query of each forward: the pivot on the prompt pass, then
    each new token on the decode steps, so the pre-pivot latents are identical to the clean DST run. All other heads
    (in-group bled + out-of-group) read the clean dest k/v (fixes the GQA bleed), and the KV cache stays clean.
    ``v_scale`` multiplies the transplanted source values (keys / attention pattern unchanged). Runs on every step;
    eager attention only (it calls ``self_attn.interface`` and writes ``attention_head_outputs``)."""
    group = mt.num_heads // mt.num_kv_heads  # GQA group size; query head h reads kv head h // group
    qheads, patched_kv = defaultdict(set), defaultdict(set)
    for l, h in heads:
        qheads[l].add(h)
        patched_kv[l].add(h // group)
    dt = mt.embed_tokens.weight.dtype

    edits = []
    for L in sorted(qheads):
        reason = mt.support(layer=L)["self_attn.interface"]
        if reason is not None:
            raise ValueError(f"headwise_kv_patch calls the block's attention function: {reason}")
        attn = mt.layers[L].self_attn
        dev = attn.o_proj.weight.device
        kv = sorted(patched_kv[L])
        # the source keys rotated for the DEST positions, outside any trace; the values carry no position. (A rope
        # scaling that adapts to the sequence, Phi-3's longrope, picks its frequencies from these positions, as the
        # model does from the whole prompt: the two agree while the prompt is shorter than the scaling's
        # original_max_position_embeddings.)
        keys = attn.rotate(k_cache[L][kv].to(dev, dt), list(dest_positions))
        values = v_cache[L][kv].to(dev, dt)
        if v_scale != 1.0:  # x the transplanted SOURCE values on the span
            values = values * v_scale
        edits += _headwise_kv_edits(attn, L, sorted(qheads[L]), kv, keys, values,
                                    torch.tensor(list(dest_positions), device=dev))
    return Intervention(edits, mt)


def _headwise_kv_edits(attn, layer, qheads, kvheads, keys, values, span):
    """The two edits of one patched layer: at the attention call, run the block's attention function on the clean
    q and on copies of k / v with the source span in the patched kv heads; at the head outputs, write the selected
    heads' last row of the result."""
    kv_idx = torch.tensor(kvheads, device=span.device)[:, None]  # (n_kv, 1), broadcasts with span (S,)
    stash = {}

    def call(mt_, step):
        interface = attn.interface  # this step's call; read once, with q / k / v, and called before the scores
        q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
        # (B, H, S, d); (B, KV, K, d) x 2, the whole cache on a decode step
        k2, v2 = k.clone(), v.clone()  # never edit k / v in place: they are the cache
        k2[:, kv_idx, span] = keys
        v2[:, kv_idx, span] = values
        stash["heads"], _ = interface(q, k2, v2)  # (B, S, H, d), the layout of attention_head_outputs

    def write(mt_, step):
        attn.attention_head_outputs[:, -1, qheads] = stash.pop("heads")[:, -1, qheads]

    return [Edit(layer, "self_attn.attention_queries", call), Edit(layer, "self_attn.attention_head_outputs", write)]


def boost_head_ov(mt, heads, alpha) -> Intervention:
    """Scale the OV contribution of `heads` (query-head (layer, h)) by `alpha` at the last token of each forward (the
    decision point) — i.e. amplify how much those heads WRITE into the residual stream: ``attention_premix[:, -1,
    heads] *= alpha``, the heads' slices of the o_proj input. o_proj is linear (no bias in Qwen2), so this scales the
    heads' residual contribution by `alpha`. Every step. `alpha == 1.0` is a no-op (an empty intervention)."""
    if alpha == 1.0:
        return Intervention()
    by_layer = defaultdict(set)
    for l, h in heads:
        by_layer[l].add(h)

    def make(attn, hs):
        def edit(mt_, step):
            attn.attention_premix[:, -1, hs] *= alpha  # (B, seq, H, hd)

        return edit

    return Intervention([Edit(l, "self_attn.attention_premix", make(mt.layers[l].self_attn, sorted(hs)))
                         for l, hs in by_layer.items()], mt)


# ══════════════════════════════════════════════════════════════════════════════
# q-transfer intervention runner (gaze-select-restrict)
# ══════════════════════════════════════════════════════════════════════════════
def whitelist_head_attn(mt, heads, keep, q_from, keep_from=None) -> Intervention:
    """clock/09-style attention restriction (``generate_restricted_to_span``), KV-cache safe: for each ``(layer,
    head)`` in ``heads``, query rows at ABSOLUTE position >= ``q_from`` may attend ONLY to the key positions in
    ``keep`` and to every key at position >= ``keep_from`` (e.g. the generated tokens; None = none); all other keys
    get the dtype min in ``attention_scores`` (a masked fill, so the causal mask stays). Query rows < ``q_from`` (the
    prompt before the pivot) are untouched. Works for the prefill (Q = K) and cached decode (Q = 1, row position =
    K - 1); every step. Needs eager attention."""
    keep = sorted({int(k) for k in keep})
    by_layer = defaultdict(set)
    for l, h in heads:
        by_layer[l].add(h)

    def make(attn, hs):
        def edit(mt_, step):
            scores = attn.attention_scores  # (B, H, Q, K), masked and scaled
            _, _, Q, K = scores.shape
            rows = [i for i in range(Q) if K - Q + i >= q_from]
            keep_k = {k for k in keep if k < K} | (set(range(keep_from, K)) if keep_from is not None else set())
            blk = [c for c in range(K) if c not in keep_k]
            if not rows or not blk:
                return
            dev = scores.device
            h = torch.tensor(hs, device=dev)[:, None, None]
            r = torch.tensor(rows, device=dev)[None, :, None]
            c = torch.tensor(blk, device=dev)[None, None, :]
            scores[:, h, r, c] = torch.finfo(scores.dtype).min

        return edit

    return Intervention([Edit(l, "self_attn.attention_scores", make(mt.layers[l].self_attn, sorted(hs)))
                         for l, hs in by_layer.items()], mt)


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
    on_gaze: Optional[Callable] = None,
):
    """Run one q-transfer intervention (a ``DiffTriggerInstance``, pivot = the frontier's ``{var} =`` stub) and return
    all RAW artifacts as JSON-native types. The SRC pivot query over ``patch_heads`` (``attention_queries_prerope``,
    before the rotary embedding) is cached and written at the DST pivot on the prefill (:func:`qkv_patch`).

    The PATCHED generation is gaze-select-restrict: (1) the readout heads' value-weighted pivot gaze under the query
    patch (no boost) scores each of DST's listed tasks (mean over its description tokens) and selects the top one;
    (2) a SEPARATE run with NO query transfer, in which, from the pivot on, the patch (goal) heads may read only the
    selected task's span, the pivot itself and the generated tokens (:func:`whitelist_head_attn`), for
    ``gsr_max_new`` tokens.

    Also collects clean-SRC / clean-DST generations (``max_new``), the three goalscope readouts (over
    ``readout_heads``, default = ``patch_heads``; the patched one under the query patch + an ``alpha`` OV boost of
    ``patch_heads`` at the decision token; ``compute_readout=False`` -> None), and value-weighted pivot→TASK attention
    (clean & under the plain query patch) for the gaze eval. Returns a dict consumed directly by
    :func:`src.nrgs.eval.eval_q_transfer`.

    ``on_gaze(kind, info)``, when given, receives each ``AttentionInformation`` the run computes: ``"select"`` (the
    selection pass, under the query patch), then, with ``compute_gaze``, ``"clean"`` and ``"patched"``."""
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

    # query transplant: SRC pivot query -> DST pivot, on the prefill
    q_layers = sorted({l for l, _ in patch_heads})
    q_src = cache_src_qkv(src_prompt, [piv_src], q_layers, "q", ctx=ctx)
    pf = [dict(which="q", cache=q_src, heads=heads_per_layer(patch_heads), positions=[piv_dst])]
    q_patch = qkv_patch(mt, pf)  # PLAIN patch — used for attention (gaze) regardless of alpha
    # + the OV boost of patch_heads at the decision token, active only on the PATCHED readout
    patched_iv = q_patch + boost_head_ov(mt, sorted(set(patch_heads)), alpha)

    def _gaze(kind, iv=None):
        info = get_attention_matrices(dst_tok, mt, value_weighted=True, interventions=iv)
        info = info[0] if isinstance(info, list) else info
        if on_gaze is not None:
            on_gaze(kind, info)
        return info

    # generations
    gen_src = greedy(src_tok, max_new=max_new, ctx=ctx)
    gen_dst = greedy(dst_tok, max_new=max_new, ctx=ctx)

    # (1) selection: pivot gaze (readout heads, value-weighted) under the query patch, no boost
    sel_preds = list(inst.task.predicates)
    sel_items = {p.id: list(range(*token_span(dst_prompt, p.description, ctx=ctx))) for p in sel_preds}
    gi = _gaze("select", q_patch)
    span = sorted({q for v in sel_items.values() for q in v})
    at = dict(zip(span, pivot_attn(gi, list(readout_heads), span, piv_dst)))
    scores = {pid: float(np.mean([at[q] for q in ps])) for pid, ps in sel_items.items()}
    selected = next(p for p in sel_preds if p.id == max(scores, key=scores.get))

    # (2) restricted run (clock/09 §5): NO query transfer; from the pivot on, the patch (goal) heads read only the
    # selected task's span, the pivot ` =` (the anchor) and the generated tokens
    keep = sorted(set(sel_items[selected.id]) | {piv_dst})
    gen_from = dst_tok["input_ids"].shape[1]
    restrict = whitelist_head_attn(mt, patch_heads, keep, piv_dst, keep_from=gen_from)
    gen_patched = _generate(mt, dst_tok, gsr_max_new, restrict)
    cl_extra = dict(cl_probe_mode="gaze_select_restrict", cl_inject=False, cl_restrict_keep=keep,
                    cl_select_scores=scores, cl_selected=selected.id, cl_select_options=[p.id for p in sel_preds])

    # goalscope readouts (source / destination / patched under the query patch + boost)
    def _ro(prompt, piv, iv=None):
        return goalscope_readout("", prompt, piv, heads=readout_heads, interventions=iv, ctx=ctx)[0][
            "patched_gen"
        ]

    if compute_readout:
        ro_src = _ro(src_prompt, piv_src)
        ro_dst = _ro(dst_prompt, piv_dst)
        ro_patched = _ro(dst_prompt, piv_dst, iv=patched_iv)
    else:
        ro_src = ro_dst = ro_patched = None

    # value-weighted pivot→TASK attention over the readout (goal) heads, clean vs patched
    if task_positions is not None:
        clean_i = _gaze("clean")
        patched_i = _gaze("patched", q_patch)
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
