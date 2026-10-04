"""Sparse query-head mask for the NRGS common-prefix setup (``src/diff/nrgs_prefix.py``).

For each mined instance, SRC's pre-RoPE `` =`` query is blended into DST's LAST token (the end of the
shared prefix, where b and d diverge), per head::

    q_dst[-1, h] <- m[l, h] * q_src[l, h] + (1 - m[l, h]) * q_dst[-1, h]        (every candidate layer l)

and the mask ``m`` (a :class:`src.goals.head_loc_attn.HeadMask`, init all-ones, clamped to [0, 1]) is
trained so the patched next token moves from b to d::

    loss = w_kl  * KL(P_patched || P_DST*)                    # match the "B done" counterfactual
         + w_dir * dir(logp[d] - logp[b])                     # directional: d must beat b
         + w_ctrl * relu(margin_ctrl - margin_clean)          # optional: SRC' query must NOT flip (progress
                                                              #   specificity; order_diff only)
         + lambda * mean(m)

``dir`` is ``softplus(-x)`` ("logistic", saturates once d wins) or ``-x`` ("margin", as in
``scripts/clock/train_completion_mask.py``). The KL mirrors ``head_loc_attn.train_head_mask`` exactly.
Queries are cached for ALL layers, so any candidate-layer band reuses one cache.
"""
from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import baukit
import torch
import torch.nn.functional as F

from src.diff.query_mask_ops import q_blend_hooks

import src.diff.nrgs_prefix as N
from src.goals.head_loc_attn import HeadMask
from src.tokens import prepare_input

logger = logging.getLogger(__name__)


@dataclass
class NRGSCache:
    key: str
    datatype: str
    pair: tuple                   # (datatype, b, d) — for per-pair eval breakdowns
    split: str                    # "train" | "val"
    dst_prompt: str
    b_id: int
    d_id: int
    q_src: torch.Tensor           # (n_layer, width) float32 CPU — SRC `` =`` query, all layers
    q_ctrl: Optional[torch.Tensor]  # (n_layer, width) — SRC' query; None when the config has no SRC'
    log_p_target: torch.Tensor    # (vocab,) float32 CPU — DST* next-token log-probs
    clean_margin: float           # clean DST: logp[d] - logp[b]
    config: str = "order_diff_1"  # contrast config (nrgs_prefix.CONFIGS); old caches predate the field


def _q_proj(mt, layer):
    return mt.attn_module_name_format.format(layer) + ".q_proj"


@torch.no_grad()
def q_last_batch(mt, prompts: list[str]) -> torch.Tensor:
    """(B, n_layer, width) float32 CPU — pre-RoPE q_proj output at each prompt's last token."""
    layers = list(range(mt.n_layer))
    names = [_q_proj(mt, l) for l in layers]
    inp = prepare_input(prompts=prompts, tokenizer=mt)          # left-padded: [:, -1] is the last real token
    with baukit.TraceDict(mt._model, names, retain_input=False, retain_output=True) as tr:
        mt._model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], use_cache=False)
    return torch.stack([tr[n].output[:, -1].float().cpu() for n in names], dim=1)


def build_caches(mt, ctx, records: list[dict], batch_size: int = 8) -> list[NRGSCache]:
    """One cache per mined record (see ``scripts/diff/mine_nrgs_prefix.py``). Re-checks that the clean
    DST top-1 is still ``b_id`` (it is the loss target) and warns when it is not."""
    out, t0 = [], time.time()
    for s in range(0, len(records), batch_size):
        chunk = records[s: s + batch_size]
        insts = [N.from_record(r) for r in chunk]
        ps = [x.prompts(ctx) for x in insts]
        q_src = q_last_batch(mt, [p["src"] for p in ps])
        ic = [i for i, p in enumerate(ps) if "src_ctrl" in p]                  # configs with an SRC' control
        qc = q_last_batch(mt, [ps[i]["src_ctrl"] for i in ic]) if ic else None
        q_ctrl = {i: qc[j] for j, i in enumerate(ic)}
        lp_star = N.last_token_logprobs(mt, [p["dst_star"] for p in ps])
        lp_dst = N.last_token_logprobs(mt, [p["dst"] for p in ps])
        for i, (r, p) in enumerate(zip(chunk, ps)):
            b, d = r["b_id"], r["d_id"]
            if int(lp_dst[i].argmax()) != b:
                logger.warning(f"clean DST top-1 != mined b_id for {r['key']} (batching / bf16 drift)")
            out.append(NRGSCache(
                key=r["key"], datatype=r["datatype"], pair=(r["datatype"], r["b"], r["d"]), split=r["split"],
                dst_prompt=p["dst"], b_id=b, d_id=d, q_src=q_src[i],
                q_ctrl=q_ctrl.get(i),
                log_p_target=lp_star[i], clean_margin=float(lp_dst[i][d] - lp_dst[i][b]),
                config=r.get("config", r.get("variant", "order_diff_1")),
            ))
        el = time.time() - t0
        logger.info(f"cache: {len(out)}/{len(records)}  ({el:.0f}s, ETA {el / len(out) * (len(records) - len(out)):.0f}s)")
    return out




@contextmanager
def q_blend(mt, mask_values, layers, caches: list[NRGSCache], use_ctrl: bool = False):
    """Keep the blend hooks registered for the whole block — forward AND backward, so gradient
    checkpointing's backward-time recompute is blended too."""
    qs = torch.stack([(c.q_ctrl if use_ctrl else c.q_src)[layers] for c in caches])   # (B, L, width)
    handles = q_blend_hooks(mt, mask_values, layers, qs)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def dst_logprobs(mt, caches: list[NRGSCache]) -> torch.Tensor:
    """(B, vocab) next-token log-probs at DST's last token (call inside :func:`q_blend`)."""
    inp = prepare_input(prompts=[c.dst_prompt for c in caches], tokenizer=mt)
    out = mt._model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    use_cache=False, logits_to_keep=1)
    return out.logits[:, -1].float().log_softmax(-1)


def patched_logprobs(mt, mask_values, layers, caches: list[NRGSCache], use_ctrl: bool = False):
    """Forward-only convenience (eval): blend, forward, unhook."""
    with q_blend(mt, mask_values, layers, caches, use_ctrl):
        return dst_logprobs(mt, caches)


def _dir_term(margin, kind):
    return F.softplus(-margin) if kind == "logistic" else -margin


@dataclass
class History:
    loss: list = field(default_factory=list)
    kl: list = field(default_factory=list)
    dir: list = field(default_factory=list)
    ctrl: list = field(default_factory=list)
    sparse: list = field(default_factory=list)
    n_active: list = field(default_factory=list)
    val: list = field(default_factory=list)       # per-epoch dicts from evaluate()


@torch.no_grad()
def evaluate(mt, mask_values, layers, caches, batch_size=8, use_ctrl=False) -> dict:
    """flip = top-1 becomes d_id (the target); margin_gain = mean (logp[d] - logp[b]) - clean; kl vs DST*.
    ``use_ctrl`` evaluates the SRC' query, on the instances whose config defines it. Returns the pooled
    scores plus ``by_config``."""
    if use_ctrl:
        caches = [c for c in caches if c.q_ctrl is not None]
    rows = []
    for s in range(0, len(caches), batch_size):
        cs = caches[s: s + batch_size]
        lp = patched_logprobs(mt, mask_values, layers, cs, use_ctrl=use_ctrl)
        for i, c in enumerate(cs):
            rows.append((c.config, int(lp[i].argmax()) == c.d_id,
                         float(lp[i][c.d_id] - lp[i][c.b_id]) - c.clean_margin,
                         float(F.kl_div(c.log_p_target.to(lp.device), lp[i], reduction="sum", log_target=True))))
    def agg(rs):
        n = max(1, len(rs))
        return dict(n=len(rs), flip=sum(r[1] for r in rs) / n, margin_gain=sum(r[2] for r in rs) / n,
                    kl=sum(r[3] for r in rs) / n)
    out = agg(rows)
    out["by_config"] = {cf: agg([r for r in rows if r[0] == cf]) for cf in sorted({r[0] for r in rows})}
    return out


def train_nrgs_mask(mt, train: list[NRGSCache], val: list[NRGSCache], layers: list[int], *,
                    n_epochs=50, lr=1e-2, lambda_sparse=20.0, w_kl=1.0, w_dir=1.0, w_ctrl=0.0,
                    dir_loss="logistic", batch_size=64, micro_batch=8, seed=0, threshold=0.9, on_epoch=None):
    """Projected-GD training of a (len(layers), n_heads) HeadMask. Defaults = the recipe behind the 122
    goal heads (``results/localization/<model>/lam20``: lambda 20, lr 1e-2, 50 epochs, 64 per optimizer
    step, all layers, threshold 0.9). ``batch_size`` instances per optimizer step, gradients accumulated
    over forwards of ``micro_batch`` (a 64-wide backward through a 32B model does not fit). ``threshold``
    only affects the logged ``n_active``. ``on_epoch(epoch, mask, hist)`` runs after every epoch."""
    if w_ctrl > 0 and not any(c.q_ctrl is not None for c in train):
        raise ValueError("w_ctrl > 0 needs SRC' queries; no training instance's config defines one")
    for p in mt._model.parameters():
        p.requires_grad_(False)
    dev = next(mt._model.parameters()).device
    mask = HeadMask(len(layers), mt.n_heads, mask_layers=layers).to(dev)
    opt = torch.optim.AdamW(mask.parameters(), lr=lr)
    rng, hist = random.Random(seed), History()

    def _accumulate(batch: list[NRGSCache], micro: int) -> dict:
        """zero_grad + backward of one optimizer step's loss over ``batch``, in forwards of ``micro``."""
        opt.zero_grad()
        loss_sparse = lambda_sparse * mask.values.mean()                       # once per step
        loss_sparse.backward()
        step = dict(kl=0.0, dir=0.0, ctrl=0.0, sparse=float(loss_sparse.detach()))
        n_ctrl = sum(c.q_ctrl is not None for c in batch)                       # SRC' term: mean over these
        for m in range(0, len(batch), micro):                                  # grad accumulation
            cs = batch[m: m + micro]
            w = len(cs) / len(batch)                                           # -> batch-mean over the step
            b = torch.tensor([c.b_id for c in cs])
            d = torch.tensor([c.d_id for c in cs])
            ar = torch.arange(len(cs))
            with q_blend(mt, mask.values, layers, cs):                         # SRC query: flip b -> d
                lp = dst_logprobs(mt, cs)
                tgt = torch.stack([c.log_p_target for c in cs]).to(lp.device)
                loss_kl = F.kl_div(tgt, lp, reduction="batchmean", log_target=True)   # KL(P_patched || P_DST*)
                ar, b, d = ar.to(lp.device), b.to(lp.device), d.to(lp.device)
                loss_dir = _dir_term(lp[ar, d] - lp[ar, b], dir_loss).mean()
                (w * (w_kl * loss_kl + w_dir * loss_dir)).backward()
            step["kl"] += w * float(loss_kl.detach())
            step["dir"] += w * float(loss_dir.detach())
            cc = [c for c in cs if c.q_ctrl is not None]
            if w_ctrl > 0 and cc:                                              # SRC' query: must not flip
                wc = len(cc) / n_ctrl
                with q_blend(mt, mask.values, layers, cc, use_ctrl=True):
                    lpc = dst_logprobs(mt, cc)
                    clean = torch.tensor([c.clean_margin for c in cc], device=lpc.device)
                    arc = torch.arange(len(cc), device=lpc.device)
                    bc = torch.tensor([c.b_id for c in cc], device=lpc.device)
                    dc = torch.tensor([c.d_id for c in cc], device=lpc.device)
                    loss_ctrl = F.relu(lpc[arc, dc] - lpc[arc, bc] - clean).mean()
                    (wc * w_ctrl * loss_ctrl).backward()
                step["ctrl"] += wc * float(loss_ctrl.detach())
        return step

    steps_per_epoch = -(-len(train) // batch_size)
    logger.info(f"train_nrgs_mask: {len(train)} train / {len(val)} val · {len(layers)} layers x {mt.n_heads} heads · "
                f"{n_epochs} epochs x {steps_per_epoch} steps (batch {batch_size}, micro {micro_batch}) · lr={lr} "
                f"lambda={lambda_sparse} w_kl={w_kl} w_dir={w_dir} w_ctrl={w_ctrl} dir={dir_loss}")
    t_start = time.time()
    for epoch in range(n_epochs):
        t_ep = time.time()
        _reset_peak_mem()
        order = list(range(len(train)))
        rng.shuffle(order)
        agg = dict(loss=0.0, kl=0.0, dir=0.0, ctrl=0.0, sparse=0.0)
        n_steps = 0
        for s in range(0, len(order), batch_size):                            # one optimizer step per batch
            step_idx = order[s: s + batch_size]
            while True:                                                        # OOM -> halve micro-batch, redo step
                oom = False
                try:
                    step = _accumulate([train[i] for i in step_idx], micro_batch)
                except torch.cuda.OutOfMemoryError:
                    if micro_batch == 1:
                        raise
                    oom = True
                if not oom:
                    break
                micro_batch = max(1, micro_batch // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                logger.warning(f"CUDA OOM -> micro_batch={micro_batch}; redoing e{epoch + 1} step {n_steps + 1} "
                               f"(exact accumulation: the optimizer step is unchanged)")
            opt.step()
            mask.clamp_()
            step["loss"] = w_kl * step["kl"] + w_dir * step["dir"] + w_ctrl * step["ctrl"] + step["sparse"]
            for k in agg:
                agg[k] += step[k]
            n_steps += 1
            logger.info(f"  e{epoch + 1:03d} step {n_steps}/{steps_per_epoch}  loss={step['loss']:.3f}  "
                        f"kl={step['kl']:.3f}  dir={step['dir']:.3f}  ctrl={step['ctrl']:.3f}  "
                        f"mask_mean={float(mask.values.detach().mean()):.3f}  n>{threshold}={mask.n_active(threshold)}  "
                        f"({time.time() - t_ep:.0f}s into epoch)")
        for k in agg:
            getattr(hist, k).append(agg[k] / max(1, n_steps))
        hist.n_active.append(mask.n_active(threshold))
        t_tr = time.time() - t_ep
        v = evaluate(mt, mask.values.detach(), layers, val, micro_batch) if val else {}
        hist.val.append(v)
        el = time.time() - t_start
        eta = el / (epoch + 1) * (n_epochs - epoch - 1)
        logger.info(f"e{epoch + 1:03d}/{n_epochs}  loss={hist.loss[-1]:.3f}  kl={hist.kl[-1]:.3f}  dir={hist.dir[-1]:.3f}  "
                    f"ctrl={hist.ctrl[-1]:.3f}  n_active={hist.n_active[-1]}  "
                    f"val_flip={v.get('flip', float('nan')):.2f}  val_gain={v.get('margin_gain', float('nan')):+.2f}  "
                    f"val_kl={v.get('kl', float('nan')):.2f} | train {t_tr:.0f}s + val {time.time() - t_ep - t_tr:.0f}s · "
                    f"peak mem {_peak_mem()} · elapsed {el / 60:.1f}m · ETA {eta / 60:.1f}m")
        if on_epoch is not None:
            on_epoch(epoch, mask, hist)
    return mask, hist


def _reset_peak_mem():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def _peak_mem() -> str:
    return " ".join(f"cuda{i}:{torch.cuda.max_memory_allocated(i) / 2**30:.0f}G"
                    for i in range(torch.cuda.device_count())) or "cpu"


def binarized(mask_values: torch.Tensor, threshold=0.5) -> torch.Tensor:
    return (mask_values.detach() > threshold).float()


def mask_from_heads(heads, layers, n_heads, device=None) -> torch.Tensor:
    """0/1 (len(layers), n_heads) tensor selecting ``heads`` — to evaluate a fixed head set (e.g. goal
    heads) with the same blend (a 1 row = hard replacement)."""
    m = torch.zeros(len(layers), n_heads, device=device)
    row = {l: i for i, l in enumerate(layers)}
    for l, h in heads:
        if l in row:
            m[row[l], h] = 1.0
    return m
