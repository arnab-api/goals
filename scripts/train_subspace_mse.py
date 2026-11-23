"""A residual subspace that reproduces the goal-head write: MSE regression onto the GoalScope vector.

Target: at a pivot, ``v_g`` = the summed residual-stream contribution of the 122 goal heads
(``collect_head_steering_vecs``) — the exact vector GoalScope steers with. Input: the residual stream at the
same position, after each layer of a read band (default 40-49). Model: ``o = Σ_l W_lᵀ W_l h_l`` with ``W_l``
(r orthonormal rows per layer, projected gradient descent with QR); loss (default ``--loss dir``) MSE between the unit
vectors ``o/|o|`` and ``v_g/|v_g|`` (direction only, as GoalScope's additive_norm uses it); ``--loss mse`` = raw MSE.
Reference: the unconstrained least-squares map from the same residuals (concatenated) to ``v_g`` — the best any
linear read-out of these layers can do.

Data: source prompts from the corpus' task bank (``QTASKS``; 2-4 sub-tasks, Python, letters naming, prefilled to the
frontier's ``{var} =``) at the 4 pre-implementation pivots (the prompt's last 4 tokens), with every instance that
duplicates a spec of the locked eval corpus removed. 80/20 split by source instance.

    python -m scripts.train_subspace_mse cache    # model needed once: h (per read layer) + v_g at every pivot
    python -m scripts.train_subspace_mse train    # no model: fit W (and the least-squares reference), export a probe

Outputs: results/subspace_mse/<model>/<tag>/{cache.pt, subspace.pt, probe.pt, metrics.json}
(``probe.pt`` is a Subspace Probe v2 checkpoint, readable by scripts/evaluation/eval_subspace_probe_v2.py.)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.append(".")

from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
PIVOTS = ["prev_end", "indent", "var", "eq"]


def args_():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["cache", "train"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--goal_heads", default="head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json")
    ap.add_argument("--read_lo", type=int, default=40)
    ap.add_argument("--read_hi", type=int, default=50, help="exclusive (40-50 = layers 40..49)")
    ap.add_argument("--cache_layers", type=int, nargs="+", default=None,
                    help="cache stage: an explicit list of read layers (overrides --read_lo/--read_hi)")
    ap.add_argument("--loss", default="dir", choices=["dir", "mse"],
                    help="dir (default): MSE between unit vectors, o/|o| vs v_g/|v_g| (= 2 - 2cos; GoalScope's "
                         "additive_norm only uses the direction); mse: raw MSE(o, v_g)")
    ap.add_argument("--n_per_datatype", type=int, default=256, help="source instances per datatype (before dedup)")
    ap.add_argument("--seed", type=int, default=2027, help="instance sampling (the eval corpus used 123)")
    ap.add_argument("--batch", type=int, default=16, help="prompts per forward while caching")
    ap.add_argument("--r", type=int, default=64, help="directions per read layer")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_batch", type=int, default=256)
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge for the least-squares reference (× mean diag)")
    ap.add_argument("--tag", default=None, help="default: L<lo>-<hi>")
    ap.add_argument("--use_layers", type=int, nargs="+", default=None,
                    help="train stage: fit only these cached read layers (e.g. one layer); outputs go to "
                         "<tag>/fit_L<layers>_r<r>/")
    ap.add_argument("--no_export", action="store_true", help="train stage: skip writing probe.pt")
    ap.add_argument("--init", default="target", choices=["target", "random"],
                    help="target (default): every W_l starts as the top-r right singular vectors of the training targets "
                         "(the span the read-out must hit; nested across r); random: a random orthonormal basis")
    ap.add_argument("--fit_tag", default="", help="suffix for the fit directory (keeps earlier fits)")
    ap.add_argument("--cuda_train", default="0", help="device for the (model-free) fit")
    return ap.parse_args()


# ── stage 1: cache h and v_g ─────────────────────────────────────────────────────────────────────
def cache(a, out_dir: Path):
    import baukit
    from scripts.evaluation.eval_goalscope import (ALL_DATATYPES, DEFAULT_DATASET, build_prompt,
                                                   load_locked_datapoints)
    from scripts.evaluation.gen_teacher_targets_v2 import sample_datapoints
    from src.goals.head_loc_attn import collect_head_steering_vecs
    from src.goals.head_sets import load_json_heads
    from src.models import ModelandTokenizer
    from src.tokens import prepare_input

    mt = ModelandTokenizer(model_key=a.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    torch.set_grad_enabled(False)
    heads = load_json_heads(a.goal_heads, mt.n_layer, mt.n_heads)
    layers = list(a.cache_layers) if a.cache_layers else list(range(a.read_lo, a.read_hi))
    key = lambda inst: (inst.meta["datatype"], tuple(p.id for p in inst.task.predicates), tuple(inst.src_progress),
                        inst.expected_next)
    corpus = {key(inst) for _, inst in load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)[0]}
    insts = []
    for dt in ALL_DATATYPES:
        pool = [x for x in sample_datapoints(dt, [2, 3, 4], a.n_per_datatype, "python", a.seed) if key(x) not in corpus]
        logger.info(f"[{dt}] {len(pool)} source instances after removing corpus duplicates")
        insts += pool
    H, V, meta = [], [], []
    for i in range(0, len(insts), a.batch):
        grp = insts[i:i + a.batch]
        prompts = [build_prompt(mt, x)[0] for x in grp]
        tok = prepare_input(prompts, tokenizer=mt)          # left-padded: the last 4 tokens are the pivots in every row
        T = tok["input_ids"].shape[1]
        pos = [T - 4, T - 3, T - 2, T - 1]
        v = collect_head_steering_vecs(mt, tok, heads, pos)  # (B, 4, d)
        cap = {}
        hooks = []
        for l in layers:
            def hk(m, inp, o, l=l):
                h = o[0] if isinstance(o, tuple) else o
                cap[l] = h[:, pos].float().cpu()
            hooks.append(baukit.get_module(mt._model, mt.layer_name_format.format(l)).register_forward_hook(hk))
        try:
            mt._model(**tok)
        finally:
            for hd in hooks:
                hd.remove()
        H.append(torch.stack([cap[l] for l in layers], dim=2))   # (B, 4, L, d)
        V.append(v)
        meta += [dict(idx=i + j, datatype=x.meta["datatype"], frontier=x.expected_next) for j, x in enumerate(grp)]
        logger.info(f"cached {min(i + a.batch, len(insts))}/{len(insts)}")
    torch.save(dict(H=torch.cat(H).half(), V=torch.cat(V), layers=layers, pivots=PIVOTS, meta=meta,
                    model=a.model, goal_heads=a.goal_heads), out_dir / "cache.pt")
    logger.info(f"saved {out_dir / 'cache.pt'}: {len(insts)} instances × {len(PIVOTS)} pivots × {len(layers)} layers")


# ── stage 2: fit ─────────────────────────────────────────────────────────────────────────────────
def metrics(o, v):
    err = (o - v).pow(2).sum(-1)
    return dict(mse=float(err.mean() / v.shape[-1]), rel_err=float((err / v.pow(2).sum(-1)).mean()),
                cos=float(torch.nn.functional.cosine_similarity(o, v, dim=-1).mean()))


def train(a, out_dir: Path):
    from src.goals.subspace_probe_v2 import SubspaceV2Config
    dev = torch.device(f"cuda:{a.cuda_train}" if torch.cuda.is_available() else "cpu")
    c = torch.load(out_dir / "cache.pt", map_location="cpu", weights_only=False)
    H, V, layers = c["H"].float(), c["V"].float(), c["layers"]          # (N,4,L,d), (N,4,d)
    if a.use_layers:
        sel = [layers.index(l) for l in a.use_layers]
        H, layers = H[:, :, sel], list(a.use_layers)
        out_dir = out_dir / f"fit_L{'-'.join(map(str, layers))}_r{a.r}_{a.loss}{a.fit_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
    N = H.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    tr, va = perm[: int(0.8 * N)], perm[int(0.8 * N):]                    # split by source instance
    if a.loss == "dir":                                                    # direction only: unit target
        V = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    flat = lambda idx: (H[idx].reshape(-1, H.shape[2], H.shape[3]).to(dev), V[idx].reshape(-1, V.shape[-1]).to(dev))
    Htr, Vtr = flat(tr)
    Hva, Vva = flat(va)
    L, d = Htr.shape[1], Htr.shape[2]
    logger.info(f"rows: train {Htr.shape[0]}, val {Hva.shape[0]} | read layers {layers} | r={a.r}/layer | loss={a.loss}")

    # least-squares reference: v ≈ [h_40 … h_49] A (ridge), the best linear read-out of these residuals
    X, Xv = Htr.reshape(len(Htr), -1), Hva.reshape(len(Hva), -1)
    mu_x, mu_v = X.mean(0), Vtr.mean(0)
    Xc = X - mu_x
    G = Xc.T @ Xc if Xc.shape[1] <= Xc.shape[0] else None
    if G is not None:
        lam = a.ridge * G.diagonal().mean()
        A = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev), Xc.T @ (Vtr - mu_v))
    else:                                                                 # more features than rows: dual (kernel) form
        K = Xc @ Xc.T
        lam = a.ridge * K.diagonal().mean()
        A = Xc.T @ torch.linalg.solve(K + lam * torch.eye(K.shape[0], device=dev), Vtr - mu_v)
    ls_val = metrics((Xv - mu_x) @ A + mu_v, Vva)
    logger.info(f"least-squares reference (val): {ls_val}")
    del X, Xc, A

    # subspace: o = Σ_l W_lᵀ W_l h_l, MSE(o, v)
    if a.init == "target":   # the read-out lives in span(W): start W on the principal directions of the targets
        _, _, Vh = torch.linalg.svd(Vtr, full_matrices=False)
        W0 = Vh[: a.r]                                                    # (r, d), orthonormal rows
        W = W0.unsqueeze(0).repeat(L, 1, 1).contiguous().requires_grad_()
    else:
        W = torch.stack([torch.linalg.qr(torch.randn(d, a.r, generator=g))[0].T for _ in range(L)]).to(dev).requires_grad_()
    opt = torch.optim.Adam([W], lr=a.lr)
    readout = lambda Hb: torch.einsum("bkr,krd->bd", torch.einsum("bkd,krd->bkr", Hb, W), W)
    for ep in range(a.epochs):
        idx = torch.randperm(len(Htr), device=dev)
        for s in range(0, len(idx), a.train_batch):
            b = idx[s:s + a.train_batch]
            o = readout(Htr[b])
            if a.loss == "dir":
                o = o / o.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            loss = (o - Vtr[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for k in range(L):
                    W[k].copy_(torch.linalg.qr(W[k].T)[0].T)
        if (ep + 1) % 25 == 0 or ep == 0:
            with torch.no_grad():
                logger.info(f"epoch {ep + 1}: train {metrics(readout(Htr), Vtr)} | val {metrics(readout(Hva), Vva)}")
    with torch.no_grad():
        sub_val, sub_tr = metrics(readout(Hva), Vva), metrics(readout(Htr), Vtr)
        per_pivot = {p: metrics(readout(H[va][:, i].to(dev)), V[va][:, i].to(dev)) for i, p in enumerate(PIVOTS)}
        zero = metrics(torch.zeros_like(Vva), Vva)
    W = W.detach().cpu()
    torch.save(dict(W=W, layers=layers, r=a.r, metrics=dict(val=sub_val, train=sub_tr)), out_dir / "subspace.pt")
    cfg = SubspaceV2Config(d_model=d, r=a.r, read_layer=layers[-1], read_layers=layers,
                           steer_layers=list(range(35, 46)), alpha=0.1)
    if not a.no_export:
        torch.save(dict(W=W, config=cfg, epoch=a.epochs, read_layer=layers[-1], read_layers=layers,
                        steer_layers=cfg.steer_layers, model=c["model"], source="subspace_mse"), out_dir / "probe.pt")
    m = dict(subspace_val=sub_val, subspace_train=sub_tr, subspace_val_per_pivot=per_pivot,
             least_squares_val=ls_val, zero_baseline_val=zero, n_rows_train=len(Htr), n_rows_val=len(Hva),
             layers=layers, r=a.r, epochs=a.epochs, lr=a.lr, loss=a.loss, init=a.init)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    logger.info(f"subspace (val): {sub_val} | least squares (val): {ls_val} | zero: {zero}")
    logger.info(f"per pivot (val): {per_pivot}")
    logger.info(f"saved {out_dir / 'probe.pt'}")


def main():
    a = args_()
    tag = a.tag or (f"L{'-'.join(map(str, a.cache_layers))}" if a.cache_layers and len(a.cache_layers) <= 3
                    else f"L{a.cache_layers[0]}-{a.cache_layers[-1]}e{a.cache_layers[1] - a.cache_layers[0]}"
                    if a.cache_layers else f"L{a.read_lo}-{a.read_hi}")
    out_dir = Path(DEFAULT_RESULTS_DIR) / "subspace_mse" / a.model.split("/")[-1] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (cache if a.stage == "cache" else train)(a, out_dir)


if __name__ == "__main__":
    main()
