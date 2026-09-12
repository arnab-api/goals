"""Query-mask training to dissociate H1 (induction ordering) from H2 (set difference) — see
``agent_reports/claude/h1h2_mask_design.md`` and the dataset in ``src/diff/h1h2.py``.

We learn a sparse soft mask over the QUERIES of candidate heads (layers ``< L_READ``): for a masked
head the pivot query is blended toward the SRC pivot query,
``q_h = m_h * q_src_h + (1 - m_h) * q_dst_h``. The mask is optimized (per hypothesis) so that the
COLLAPSED goal-head gaze at ``h^{L_READ}`` (all goal heads read the residual after block ``L_READ-1``,
value-weighted pivot->span attention) lands on the hypothesis's target predicate span
(COPY->D, H1->B, H2->E).

Key structural fact used throughout: under the collapse every goal head reads the SAME ``h_read``, so
the gaze is a CLOSED-FORM differentiable function of ``h_read`` (input_layernorm -> W_{q,k,v} -> RoPE ->
softmax -> value-norm weighting), computed directly per goal head with no full-model forward. The three
interventions differ only in how ``h_read(mask)`` is produced:

  * ``block_head_communication`` (#3): ``h_read = clean_h_read`` with ``sum_h m_h * delta_h`` added at the
    pivot; ``delta_h`` are CONSTANT per-head OV deltas (q_src, clean-DST k/v) -> linear in the mask, no
    forward per step.
  * ``free_form`` (#1): one grad-enabled DST forward with the q-blend hooks; everything below L_READ
    recomputes.
  * ``mlps_frozen`` (#2): as #1 but every MLP output below L_READ is repatched from the clean-DST run
    (3 total forwards: src queries, clean-DST MLPs, patched).
"""
import hashlib
import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import baukit
import torch

from src.goals.head_loc_attn import HeadMask
from src.tokens import prepare_input

logger = logging.getLogger(__name__)

INTERVENTIONS = ("free_form", "mlps_frozen", "block_head_communication")


def build_balanced_device_map(model_name, l_read, n_gpu=2, max_gpu="79GiB"):
    """Device map that splits the GRAD-COMPUTED layers (``0..l_read-1``) EVENLY across ``n_gpu`` GPUs.

    ``device_map="auto"`` fills sequentially (layers 0..~N/2 on GPU0, rest on GPU1). With the forward
    truncated at ``l_read`` (see ``h_read_forward``), only layers ``0..l_read-1`` compute, so under the
    auto map GPU0 carries almost the whole backward graph (e.g. 33 of 40 grad-layers for l_read=40 on a
    64-layer model) and OOMs while GPU1 sits idle holding the never-run layers ``l_read..``. Splitting
    the computed layers evenly balances the graph; the weight-only layers ``l_read..`` are then packed
    to equalise total layers per GPU. Non-layer modules keep accelerate's inferred placement."""
    import torch as _torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_name)
    n_layer = cfg.num_hidden_layers
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=_torch.bfloat16)
    # non-layer modules, explicit (never rely on a '' catch-all): embed + rotary feed the first block
    # (GPU0); norm + lm_head are post-all-layers and never run under truncation, so pack them on the
    # last GPU. Only the ones this arch actually has are emitted.
    named = dict(m.named_modules())
    dm = {}
    for name, gpu in [("model.embed_tokens", 0), ("model.rotary_emb", 0),
                      ("model.norm", n_gpu - 1), ("lm_head", n_gpu - 1)]:
        if name in named:
            dm[name] = gpu
    comp = list(range(l_read))                                    # grad-computed layers -> split EVENLY
    assign, per = {}, len(comp) // n_gpu
    for i in range(n_gpu):
        for l in comp[i * per: (i + 1) * per if i < n_gpu - 1 else len(comp)]:
            assign[l] = i
    counts = [sum(1 for v in assign.values() if v == i) for i in range(n_gpu)]
    for l in range(l_read, n_layer):                              # weight-only layers -> equalise totals
        i = min(range(n_gpu), key=lambda g: counts[g])
        assign[l] = i; counts[i] += 1
    for l, i in assign.items():
        dm[f"model.layers.{l}"] = i
    return dm


# ══════════════════════════════════════════════════════════════════════════════
# module-name helpers + RoPE
# ══════════════════════════════════════════════════════════════════════════════
def _proj(mt, layer, which):  # which in {q,k,v,o}
    return mt.attn_module_name_format.format(layer) + f".{which}_proj"


def _ln_name(mt, layer):
    return mt.layer_name_format.format(layer) + ".input_layernorm"


def _mlp_name(mt, layer):
    return mt.layer_name_format.format(layer) + ".mlp"


def _rope_cos_sin(mt, positions, device=None):
    """(cos, sin) for absolute ``positions`` -> each (1, len, head_dim). ``device`` targets the shard
    that will consume them (device_map='auto' splits layers across GPUs)."""
    ref = next(mt._model.parameters())
    dev = device or ref.device
    base = mt.layer_name_format.split(".layers")[0]
    rot = baukit.get_module(mt._model, base + ".rotary_emb")
    pos = torch.tensor([list(positions)], device=ref.device)
    dummy = torch.zeros(1, len(positions), 1, device=ref.device, dtype=ref.dtype)
    cos, sin = rot(dummy, pos)
    return cos.to(dev), sin.to(dev)


def _apply_rope(mt, layer):
    """The model's own ``apply_rotary_pos_emb`` (family-specific), located via any attn module."""
    sample = baukit.get_module(mt._model, mt.attn_module_name_format.format(layer))
    return importlib.import_module(type(sample).__module__).apply_rotary_pos_emb


# ══════════════════════════════════════════════════════════════════════════════
# goal-head readers — closed-form collapsed gaze
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class _GoalReaders:
    """Per-goal-head projection weights (sliced), grouped by layer, for the closed-form collapse gaze."""
    mt: object
    goal_heads: list
    hd: int
    group: int
    by_layer: dict = field(default_factory=dict)  # layer -> dict(q_idx=[...], kv_idx=[...], Wq,Wk,Wv,ln)

    @classmethod
    def build(cls, mt, goal_heads):
        hd, group = mt.head_dim, mt.n_heads // mt.n_kv_heads
        self = cls(mt=mt, goal_heads=list(goal_heads), hd=hd, group=group)
        layers = sorted({l for l, _ in goal_heads})
        for L in layers:
            js = sorted(h for l, h in goal_heads if l == L)
            Wq = baukit.get_module(mt._model, _proj(mt, L, "q")).weight  # (n_heads*hd, n_embd)
            Wk = baukit.get_module(mt._model, _proj(mt, L, "k")).weight  # (n_kv*hd, n_embd)
            Wv = baukit.get_module(mt._model, _proj(mt, L, "v")).weight
            ln = baukit.get_module(mt._model, _ln_name(mt, L)).weight     # (n_embd,)
            self.by_layer[L] = dict(js=js, Wq=Wq, Wk=Wk, Wv=Wv, ln=ln)
        return self


def _rmsnorm(x, weight, eps=1e-6):
    """Llama/Qwen RMSNorm (no bias, no mean-subtraction), in float32 for the variance."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * xf.to(x.dtype))


def collapse_gaze_per_pos(readers: _GoalReaders, h_read, pivot, eps=1e-6):
    """Value-weighted attention from the pivot onto every position, averaged over goal heads, when all
    goal heads read ``h_read`` (seq, n_embd). Differentiable in ``h_read``. Returns (seq,) float.

    Exactly the collapse Tool-1 gaze of ``eval_collapse_layerwise`` computed in closed form: per goal
    head, input_layernorm(h_read) -> W_{q,k,v} -> RoPE -> softmax(q_pivot . k / sqrt(d)) -> weight by
    ||v||. Causal: only positions <= pivot receive mass."""
    mt, hd = readers.mt, readers.hd
    S = h_read.shape[0]
    acc = torch.zeros(S, device=h_read.device, dtype=torch.float32)
    nheads = 0
    for L, d in readers.by_layer.items():
        dev = d["Wq"].device                                # this layer's shard
        apply_rope = _apply_rope(mt, L)
        cos_all, sin_all = _rope_cos_sin(mt, range(S), device=dev)
        x = _rmsnorm(h_read.to(dev), d["ln"], eps).to(d["Wq"].dtype)   # (S, n_embd), match weight dtype
        q_all = (x @ d["Wq"].T)                              # (S, n_heads*hd)
        k_all = (x @ d["Wk"].T)                              # (S, n_kv*hd)
        v_all = (x @ d["Wv"].T)
        for j in d["js"]:
            kv = j // readers.group
            q = q_all[:, j * hd:(j + 1) * hd].unsqueeze(0).unsqueeze(0)      # (1,1,S,hd)
            k = k_all[:, kv * hd:(kv + 1) * hd].unsqueeze(0).unsqueeze(0)
            v = v_all[:, kv * hd:(kv + 1) * hd]                              # (S,hd)
            q, k = apply_rope(q, k, cos_all, sin_all)                        # RoPE @ absolute pos
            qp = q[0, 0, pivot]                                              # (hd,)
            kk = k[0, 0]                                                     # (S,hd)
            scores = (kk @ qp) / (hd ** 0.5)                                 # (S,)
            causal = torch.arange(S, device=scores.device) > pivot          # future positions
            scores = scores.masked_fill(causal, float("-inf"))              # causal (no in-place on grad)
            attn = torch.softmax(scores.float(), dim=-1)                    # (S,)
            acc = acc + (attn * v.float().norm(dim=-1)).to(acc.device)      # value-weighted
            nheads += 1
    return acc / max(nheads, 1)


def span_mass(per_pos, spans, target_idx):
    """(target mass fraction, per-span masses) — per_pos summed over each predicate span, normalized
    over all spans. ``spans`` = list of (lo, hi); ``target_idx`` indexes the target predicate."""
    masses = torch.stack([per_pos[lo:hi].sum() for (lo, hi) in spans])
    frac = masses[target_idx] / masses.sum().clamp_min(1e-9)
    return frac, masses


def gaze_objective(per_pos, spans, target_idx, comp_weight=1.0, eps=1e-9):
    """Per-hypothesis gaze loss that REWARDS the target span and explicitly PENALIZES competing spans.

    Over the T predicate spans, ``p_j = mass_j / sum_k mass_k``:
        loss = -log(p_target)            (cross-entropy pull toward the target span)
             + comp_weight * (1 - p_t)   (explicit competitor penalty; 1 - p_t = total competing prob)
    Returns (loss, p_target, per-span masses). ``p_target`` is the same span-normalized fraction the
    collapse eval reports, so it is comparable across hypotheses/interventions."""
    masses = torch.stack([per_pos[lo:hi].sum() for (lo, hi) in spans])
    p = masses / masses.sum().clamp_min(eps)
    p_t = p[target_idx]
    loss = -(p_t.clamp_min(eps)).log() + comp_weight * (1.0 - p_t)
    return loss, p_t, masses


# ══════════════════════════════════════════════════════════════════════════════
# caching — one contrastive pair
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PairCache:
    """Everything the interventions need for one SRC/DST pair, at a fixed ``L_READ``."""
    spec: object
    l_read: int
    dst_ids: torch.Tensor           # (1, seq) DST prompt ids
    dst_attn: torch.Tensor
    dst_pivot: int
    clean_h_read: torch.Tensor      # (seq, n_embd) residual AFTER block l_read-1, DST clean
    spans: list                     # DST predicate description spans [(lo,hi)]
    q_src_pivot: dict               # {layer: (n_heads*hd,) SRC pivot q_proj output, pre-RoPE}, layers < l_read
    q_dst_pivot: dict               # {layer: (n_heads*hd,)} clean DST pivot q_proj output, pre-RoPE
    deltas: Optional[torch.Tensor] = None       # (l_read, n_heads, n_embd) per-head OV pivot deltas (#3)
    clean_mlp: Optional[dict] = None            # {layer: (seq, n_embd)} clean DST MLP outputs (#2)
    P: Optional[dict] = None                    # build_prompts output (prompts, insts, outcomes)

    def to(self, device):
        """Move all tensors to ``device`` (deltas/mlp stay lazily on CPU — moved per-use)."""
        self.dst_ids = self.dst_ids.to(device)
        self.dst_attn = self.dst_attn.to(device)
        self.clean_h_read = self.clean_h_read.to(device)
        self.q_src_pivot = {l: v.to(device) for l, v in self.q_src_pivot.items()}
        self.q_dst_pivot = {l: v.to(device) for l, v in self.q_dst_pivot.items()}
        return self

    def cpu_(self):
        """Move everything to CPU in-place (for disk serialization)."""
        self.dst_ids = self.dst_ids.cpu(); self.dst_attn = self.dst_attn.cpu()
        self.clean_h_read = self.clean_h_read.cpu()
        self.q_src_pivot = {l: v.cpu() for l, v in self.q_src_pivot.items()}
        self.q_dst_pivot = {l: v.cpu() for l, v in self.q_dst_pivot.items()}
        if self.deltas is not None:
            self.deltas = self.deltas.cpu()
        if self.clean_mlp is not None:
            self.clean_mlp = {l: v.cpu() for l, v in self.clean_mlp.items()}
        return self


@torch.no_grad()
def _forward_capture(mt, ids, attn, layers_resid, layers_qkv=None, layers_mlp=None, stop_after=None):
    """One forward capturing: residual-after-block for ``layers_resid`` (layer_name_format output),
    q/k/v proj outputs for ``layers_qkv``, mlp outputs for ``layers_mlp``. Returns dicts.

    ``stop_after`` (a layer index): truncate the forward right after that decoder block's output, so
    layers above it are never computed (baukit ``stop=True`` on the last-registered name). All captured
    modules must live at layers <= ``stop_after`` for their outputs to be populated before the halt."""
    unt = lambda x: x[0] if isinstance(x, tuple) else x
    names = [_proj(mt, l, w) for l in (layers_qkv or []) for w in ("q", "k", "v")]
    names += [_mlp_name(mt, l) for l in (layers_mlp or [])]
    names += [mt.layer_name_format.format(l) for l in layers_resid]   # block outputs LAST (deepest fires last)
    if stop_after is not None:
        names.append(mt.layer_name_format.format(stop_after))         # ensure the stop target is the last name
    with baukit.TraceDict(mt._model, names, retain_output=True, stop=stop_after is not None) as tr:
        mt._model(input_ids=ids, attention_mask=attn)
    resid = {l: unt(tr[mt.layer_name_format.format(l)].output)[0].float() for l in layers_resid}
    qkv = {l: {w: unt(tr[_proj(mt, l, w)].output)[0].float() for w in ("q", "k", "v")}
           for l in (layers_qkv or [])}
    mlp = {l: unt(tr[_mlp_name(mt, l)].output)[0].float() for l in (layers_mlp or [])}
    return resid, qkv, mlp


@torch.no_grad()
def build_pair_cache(mt, spec, l_read, intervention, build_prompts_fn, stub="var_eq",
                     src_targ_in_dst=True):
    """Cache SRC queries + clean-DST artifacts for one pair. ``build_prompts_fn`` = ``h1h2.build_prompts``."""
    from src.tokens import find_token_range
    P = build_prompts_fn(mt, spec, stub=stub, src_targ_in_dst=src_targ_in_dst)
    src_tok = prepare_input(prompts=P["src_prompt"], tokenizer=mt)
    dst_tok = prepare_input(prompts=P["dst_prompt"], tokenizer=mt)
    src_piv = src_tok["input_ids"].shape[1] - 1
    dst_piv = dst_tok["input_ids"].shape[1] - 1
    cand_layers = list(range(l_read))
    spans = [find_token_range(P["dst_prompt"], pr.description, mt.tokenizer)
             for pr in P["dst_inst"].task.predicates]

    # SRC forward: pivot q_proj outputs (pre-RoPE) for candidate layers — truncate past l_read
    _, src_qkv, _ = _forward_capture(mt, src_tok["input_ids"], src_tok["attention_mask"],
                                     layers_resid=[], layers_qkv=cand_layers, stop_after=l_read - 1)
    q_src_pivot = {l: src_qkv[l]["q"][src_piv] for l in cand_layers}

    # clean DST forward: residual@l_read, candidate q/k/v, MLPs (for #2) — truncate past l_read
    need_mlp = cand_layers if intervention == "mlps_frozen" else []
    resid, dst_qkv, mlp = _forward_capture(
        mt, dst_tok["input_ids"], dst_tok["attention_mask"],
        layers_resid=[l_read - 1], layers_qkv=cand_layers, layers_mlp=need_mlp, stop_after=l_read - 1)
    clean_h_read = resid[l_read - 1]                              # (seq, n_embd)
    q_dst_pivot = {l: dst_qkv[l]["q"][dst_piv] for l in cand_layers}

    cache = PairCache(
        spec=spec, l_read=l_read, dst_ids=dst_tok["input_ids"], dst_attn=dst_tok["attention_mask"],
        dst_pivot=dst_piv, clean_h_read=clean_h_read, spans=spans,
        q_src_pivot=q_src_pivot, q_dst_pivot=q_dst_pivot,
        clean_mlp={l: mlp[l] for l in need_mlp} if need_mlp else None,
    )
    if intervention == "block_head_communication":
        cache.deltas = _precompute_ov_deltas(mt, cache, dst_qkv, l_read, dst_piv)
    cache.P = P
    return cache


def _cache_key(slug, datatype, lang, n_pairs, l_read, stub, src_targ_in_dst, seed, intervention):
    desc = (f"{slug}|{datatype}|{lang}|n{n_pairs}|L{l_read}|{stub}"
            f"|sd{int(src_targ_in_dst)}|seed{seed}|{intervention}")
    return hashlib.md5(desc.encode()).hexdigest()[:12], desc


# Only ``block_head_communication`` is disk-cached: its per-head OV deltas are expensive to COMPUTE
# (~n_heads * L_READ isolated head-attentions per pair) yet compact to store. ``free_form`` needs only
# SRC queries (cheap) and ``mlps_frozen`` needs the clean-DST MLP outputs — huge on disk (~164 MB/pair)
# but a single cheap forward to recompute — so both are always rebuilt in memory.
_DISK_CACHED = ("block_head_communication",)


def load_or_build_caches(mt, specs, l_read, intervention, build_prompts_fn, cache_dir,
                         datatype, lang, seed, stub="var_eq", src_targ_in_dst=True, overwrite=False):
    """Per-pair caches for one intervention, all kept on **CPU** (never accumulated on the GPU). Each
    cache is built then immediately moved to CPU, so the build phase never piles N caches on one shard
    (the #3 512-pair delta build otherwise OOMs `cuda:0` at ~20GB on top of the weights). During
    training every h_read path moves what it needs to the device per-pair: #1/#2 move inputs + hook
    slices, #3 moves the per-pair OV deltas (``deltas.to(mask.device)``) — all small, all per-use.
    ``block_head_communication`` (#3) is additionally disk-persisted, keyed by the full config."""

    def _build_cpu():                                          # build + move each cache to CPU at once
        out = []
        for s in specs:
            c = build_pair_cache(mt, s, l_read, intervention, build_prompts_fn, stub, src_targ_in_dst)
            c.cpu_()
            out.append(c)
        return out

    if intervention not in _DISK_CACHED:
        caches = _build_cpu()
        logger.info(f"built {len(caches)} caches on CPU (no disk cache for {intervention})")
        return caches

    slug = mt.name.split("/")[-1]
    key, desc = _cache_key(slug, datatype, lang, len(specs), l_read, stub, src_targ_in_dst, seed, intervention)
    path = os.path.join(cache_dir, f"h1h2_caches_{key}.pt")
    if os.path.exists(path) and not overwrite:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        caches = blob["caches"]                                # already CPU (map_location); keep on CPU
        logger.info(f"loaded {len(caches)} caches <- {path}")
        return caches
    caches = _build_cpu()
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"desc": desc, "caches": caches}, path)
    logger.info(f"built + saved {len(caches)} caches -> {path} "
                f"({os.path.getsize(path) / 1e9:.2f} GB)")
    return caches


@torch.no_grad()
def _precompute_ov_deltas(mt, cache, dst_qkv, l_read, dst_piv):
    """(#3) Per candidate head: delta_h = OV(q_src, k_dst, v_dst) - OV(q_dst, k_dst, v_dst) at the pivot.
    Returns (l_read, n_heads, n_embd). Keys/values are CLEAN DST (no inter-head communication); only the
    query is swapped -> the pure isolated query-transplant write of each head."""
    hd, group = mt.head_dim, mt.n_heads // mt.n_kv_heads
    deltas = torch.zeros(l_read, mt.n_heads, mt._model.config.hidden_size, dtype=torch.float32)
    for L in range(l_read):
        Wo = baukit.get_module(mt._model, _proj(mt, L, "o")).weight.float()   # (n_embd, n_heads*hd)
        dev = Wo.device                                                       # this layer's shard
        apply_rope = _apply_rope(mt, L)
        S = cache.clean_h_read.shape[0]
        cos_all, sin_all = _rope_cos_sin(mt, range(S), device=dev)
        cos_piv, sin_piv = cos_all[:, dst_piv:dst_piv + 1], sin_all[:, dst_piv:dst_piv + 1]
        k_all = dst_qkv[L]["k"].to(dev)   # (S, n_kv*hd) pre-RoPE
        v_all = dst_qkv[L]["v"].to(dev)
        for j in range(mt.n_heads):
            kv = j // group
            k = k_all[:, kv * hd:(kv + 1) * hd].unsqueeze(0).unsqueeze(0)
            v = v_all[:, kv * hd:(kv + 1) * hd]                               # (S,hd)
            _, k_r = apply_rope(torch.zeros_like(k), k, cos_all, sin_all)
            k_r = k_r[0, 0]                                                   # (S,hd) RoPE'd
            for tag, qsrc in (("src", cache.q_src_pivot[L]), ("dst", cache.q_dst_pivot[L])):
                q = qsrc[j * hd:(j + 1) * hd].to(dev).view(1, 1, 1, hd)
                q_r, _ = apply_rope(q, torch.zeros_like(q), cos_piv, sin_piv)
                qp = q_r[0, 0, 0]                                             # (hd,) RoPE'd @ pivot
                sc = (k_r @ qp) / (hd ** 0.5)
                sc[dst_piv + 1:] = float("-inf")
                attn = torch.softmax(sc.float(), -1)
                out = (attn @ v.float())                                      # (hd,)
                ov = Wo[:, j * hd:(j + 1) * hd] @ out                         # (n_embd,)
                if tag == "src":
                    ov_src = ov
                else:
                    ov_dst = ov
            deltas[L, j] = (ov_src - ov_dst).cpu()
    return deltas


# ══════════════════════════════════════════════════════════════════════════════
# h_read(mask) for each intervention
# ══════════════════════════════════════════════════════════════════════════════
def h_read_block_comm(mask: HeadMask, cache: PairCache):
    """#3: clean h_read with sum_h m_h * delta_h added at the pivot. Linear in the mask; no forward."""
    dev = mask.values.device                                    # keep grad graph on the mask's device
    d = cache.deltas.to(dev)                                     # (l_read, n_heads, n_embd)
    agg = torch.einsum("lh,lhe->e", mask.values[:cache.l_read], d)   # (n_embd,), grad -> mask
    hdev = cache.clean_h_read.device
    agg = agg.to(hdev)
    S = cache.clean_h_read.shape[0]
    onehot = torch.zeros(S, device=hdev, dtype=agg.dtype)
    onehot[cache.dst_pivot] = 1.0                                # constant selector (no grad needed)
    return cache.clean_h_read + torch.outer(onehot, agg)        # add agg at pivot row, no in-place


def _q_blend_hooks(mt, mask: HeadMask, cache: PairCache):
    """forward-pre? no — a forward hook on each candidate q_proj that blends the pivot query slice:
    q[pivot, h] <- m_{l,h} * q_src_{l,h} + (1 - m_{l,h}) * q[pivot, h]. Grad flows to the mask."""
    hd = mt.head_dim
    handles, piv = [], cache.dst_pivot
    for L in range(cache.l_read):
        mod = baukit.get_module(mt._model, _proj(mt, L, "q"))
        qsrc = cache.q_src_pivot[L]

        def make(L, qsrc):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                seq = o.shape[1]
                if not (0 <= piv < seq):
                    return out
                qs = qsrc.to(o.dtype).to(o.device)                       # (n_heads*hd,)
                m_full = mask.values[L].repeat_interleave(hd).to(o.device).to(o.dtype)  # (n_heads*hd,)
                new_piv = m_full * qs + (1 - m_full) * o[0, piv]         # (n_heads*hd,) blended pivot row
                # rebuild the sequence out-of-place (no versioned in-place -> autograd-safe)
                o_new = torch.cat([o[:, :piv], new_piv.view(1, 1, -1), o[:, piv + 1:]], dim=1)
                return (o_new,) + tuple(out[1:]) if isinstance(out, tuple) else o_new
            return hook
        handles.append(mod.register_forward_hook(make(L, qsrc)))
    return handles


def _mlp_freeze_hooks(mt, cache: PairCache):
    """#2: replace each candidate-layer MLP output with the cached clean-DST output (kills MLP-mediated
    propagation of the query intervention)."""
    handles = []
    for L, clean in (cache.clean_mlp or {}).items():
        mod = baukit.get_module(mt._model, _mlp_name(mt, L))

        def make(clean):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if o.shape[1] != clean.shape[0]:
                    return out
                new = clean.to(o.dtype).to(o.device).unsqueeze(0)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            return hook
        handles.append(mod.register_forward_hook(make(clean)))
    return handles


def h_read_forward(mt, mask: HeadMask, cache: PairCache, freeze_mlp: bool):
    """#1 / #2: grad-enabled DST forward with the q-blend hooks (+ MLP freeze for #2), returning the
    residual after block l_read-1 at every position. Differentiable in the mask.

    Truncated: ``stop=True`` halts the forward right after block l_read-1, so layers >= l_read are never
    computed (no activations retained for them). Inputs are moved to the model device here, so the pair
    caches can live on CPU (the q-blend / MLP-freeze hooks move their own slices per-use)."""
    unt = lambda x: x[0] if isinstance(x, tuple) else x
    name = mt.layer_name_format.format(cache.l_read - 1)
    dev = next(mt._model.parameters()).device
    ids, attn = cache.dst_ids.to(dev), cache.dst_attn.to(dev)
    handles = _q_blend_hooks(mt, mask, cache)
    if freeze_mlp:
        handles += _mlp_freeze_hooks(mt, cache)
    try:
        with baukit.TraceDict(mt._model, [name], retain_output=True, stop=True) as tr:
            mt._model(input_ids=ids, attention_mask=attn)
        return unt(tr[name].output)[0].float()
    finally:
        for h in handles:
            h.remove()


def h_read(mt, mask, cache, intervention):
    if intervention == "block_head_communication":
        return h_read_block_comm(mask, cache)
    return h_read_forward(mt, mask, cache, freeze_mlp=(intervention == "mlps_frozen"))


# ══════════════════════════════════════════════════════════════════════════════
# training — per hypothesis
# ══════════════════════════════════════════════════════════════════════════════
HYPOTHESIS_ROLE = {"copy": "src_targ", "h1": "dst_h1", "h2": "dst_targ", "noop": "frontier"}


def target_span_idx(cache: PairCache, hypothesis: str):
    """Index (into cache.spans / DST predicate list) of the hypothesis's target predicate."""
    role = HYPOTHESIS_ROLE[hypothesis]
    tgt_id = getattr(cache.spec, role)
    ids = [p.id for p in cache.P["dst_inst"].task.predicates]
    return ids.index(tgt_id)


def train_query_mask(mt, caches, l_read, hypothesis, intervention,
                     n_epochs=100, lr=1e-2, lambda_sparse=1e-2, comp_weight=1.0, batch_size=8, seed=0,
                     checkpoint_dir=None, checkpoint_every=5, config=None):
    """Learn a sparse query mask (layers < l_read) pushing the collapsed goal-head gaze onto the
    ``hypothesis`` target span, under ``intervention``. Per-pair loss = ``gaze_objective`` (reward
    target span + penalize competing spans); total = mean_pair(loss) + lambda_sparse * mean(mask).
    Projected GD (clamp to [0,1]). Returns (HeadMask, history).

    Checkpointing (convention from ``scripts/clock/train_completion_mask.py``): every
    ``checkpoint_every`` epochs (and the last), write ``checkpoint_dir/epochs/mask_epochNN.pt`` and
    overwrite ``checkpoint_dir/mask.pt`` with {mask_values, layers, selected_heads, history, config}.
    ``checkpoint_every=0`` disables."""
    assert intervention in INTERVENTIONS and hypothesis in HYPOTHESIS_ROLE
    readers = _GoalReaders.build(mt, mt_goal_heads(mt))
    mask = HeadMask(l_read, mt.n_heads, mask_layers=list(range(l_read))).to(
        next(mt._model.parameters()).device)
    opt = torch.optim.AdamW(mask.parameters(), lr=lr)
    tgt_idx = [target_span_idx(c, hypothesis) for c in caches]
    hist = {"loss": [], "task": [], "sparse": [], "mass": [], "n_active": []}
    epochs_dir = os.path.join(checkpoint_dir, "epochs") if checkpoint_dir else None
    if epochs_dir:
        os.makedirs(epochs_dir, exist_ok=True)

    def _save_ckpt(epoch):
        full = mask.to_full_depth(list(range(l_read)), mt.n_layer)
        sel = sorted(full.selected_heads(list(range(mt.n_layer)), threshold=0.5))
        blob = {"mask_values": mask.values.detach().cpu(), "layers": list(range(l_read)),
                "selected_heads": sel, "n_selected": len(sel), "l_read": l_read, "epoch": epoch,
                "hypothesis": hypothesis, "intervention": intervention,
                "history": hist, "config": config or {}}
        torch.save(blob, os.path.join(epochs_dir, f"mask_epoch{epoch:02d}.pt"))
        torch.save(blob, os.path.join(checkpoint_dir, "mask.pt"))

    g = torch.Generator().manual_seed(seed)
    for ep in range(n_epochs):
        order = torch.randperm(len(caches), generator=g).tolist()
        ep_loss = ep_task = ep_mass = 0.0
        for s in range(0, len(caches), batch_size):
            batch = order[s:s + batch_size]
            opt.zero_grad()
            task = 0.0
            for i in batch:
                c = caches[i]
                hr = h_read(mt, mask, c, intervention)
                per = collapse_gaze_per_pos(readers, hr, c.dst_pivot)
                ploss, p_t, _ = gaze_objective(per, c.spans, tgt_idx[i], comp_weight=comp_weight)
                task = task + ploss.to(mask.values.device)
                ep_mass += float(p_t.detach())
            task = task / len(batch)
            sparse = lambda_sparse * mask.values.mean()
            loss = task + sparse
            loss.backward()
            opt.step()
            mask.clamp_()
            b_task, b_sparse, b_loss = float(task.detach()), float(sparse.detach()), float(loss.detach())
            logger.debug(f"[{hypothesis}/{intervention}] ep {ep:3d} batch {s // batch_size:3d} "
                         f"target={b_task:.4f} sparse={b_sparse:.4f} total={b_loss:.4f}")
            ep_loss += b_loss * len(batch)
            ep_task += b_task * len(batch)
        n = len(caches)
        hist["loss"].append(ep_loss / n); hist["task"].append(ep_task / n)
        hist["sparse"].append(float(sparse.detach())); hist["mass"].append(ep_mass / n)
        hist["n_active"].append(int((mask.values > 0.5).sum()))
        logger.info(f"[{hypothesis}/{intervention}] ep {ep:3d} loss={hist['loss'][-1]:.4f} "
                    f"task={hist['task'][-1]:.4f} sparse={hist['sparse'][-1]:.4f} "
                    f"mass={hist['mass'][-1]:.3f} active={hist['n_active'][-1]}")
        if checkpoint_dir and checkpoint_every and (ep % checkpoint_every == 0 or ep == n_epochs - 1):
            _save_ckpt(ep)
    return mask, hist


def mt_goal_heads(mt):
    """Goal heads from the diff.functional module globals (set by FUNC.setup), else raise."""
    import src.diff.functional as FUNC
    if getattr(FUNC, "GOAL_HEADS", None) is None:
        raise RuntimeError("goal heads unset — call src.diff.functional.setup(...) first")
    return FUNC.GOAL_HEADS
