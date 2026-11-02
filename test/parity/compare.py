#!/usr/bin/env python
"""Compare two parity output directories (written by ``run.py``) case by case.

    python test/parity/compare.py <ref_dir> <new_dir>            # cross-stack tolerances (bf16)
    python test/parity/compare.py <ref_dir> <new_dir> --exact    # bit-identity (same env, same GPU)

Strings are compared exactly in both modes (with a token-level summary of the first divergence). Tensors are
compared against per-kind tolerances (``TOL`` below; the README explains each). Cases marked ``padded`` (left-padded
batches) report deviations as WARN in tolerance mode, because the old and new stacks are allowed to number the
positions of padded rows differently; in ``--exact`` mode every deviation fails. Exit status 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Tolerances for bf16 inference compared ACROSS stacks (old baukit/nnsight-0.7 code vs the port). All relative errors
# are Frobenius: ||new - ref|| / ||ref||. A wrong position, head slice, layer or scale gives errors of order 1, so
# these bounds separate "same computation, different kernels / op order" from "different computation".
TOL = {
    "act": dict(rel=2e-2, cos=0.9995),  # activations read from the bf16 model (hidden states, o_proj / q / k / v)
    "vec": dict(rel=2e-2, cos=0.9995),  # fp32 vectors derived from activations (steering, OV, goal encodings)
    "attn": dict(max_abs=1e-2),  # attention probabilities in [0, 1] (bf16 ulp at 1.0 is 2^-8 = 0.0039)
    "attn_vw": dict(rel=2e-2),  # attention x ||v||
    "logits": dict(rel=2e-2, max_abs=0.5),  # bf16 logits of magnitude ~10-40 (ulp 0.0625-0.25)
    "probs": dict(max_abs=5e-3),
    "grad": dict(rel=5e-2, cos=0.998),  # head-mask gradient: a bf16 backward through 28 layers
    "topk": dict(overlap=0.8, max_abs_logits=0.5, max_abs_probs=5e-3),  # + top-1 id must match
    "json_float": dict(abs=1e-3, rel=2e-2),  # floats in the json payload (losses, gaze scores, cosines)
}
EXACT_KINDS = ("topk_ids",)
# re-tokenizations of generated strings that are compared as strings already (one failure per divergence, not two)
DERIVED_KEYS = {"ids", "gen_ids", "ro_ids", "completion_ids", "base_ids", "boost_ids", "gen_src_ids", "gen_dst_ids",
                "gen_patched_ids"}


class Report:
    def __init__(self, case):
        self.case = case
        self.n = 0
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.worst_rel = 0.0
        self.n_str_diff = 0

    def check(self, ok: bool, msg: str, soft: bool = False):
        self.n += 1
        if not ok:
            (self.warns if soft else self.fails).append(msg)

    @property
    def status(self):
        return "FAIL" if self.fails else ("WARN" if self.warns else "PASS")


# ── strings ─────────────────────────────────────────────────────────────────────────────────────
_TOK = None


def _tokenizer(model_id):
    global _TOK
    if _TOK is None:
        try:
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoTokenizer

            _TOK = AutoTokenizer.from_pretrained(model_id)
        except Exception:  # noqa: BLE001
            _TOK = False
    return _TOK or None


def token_diff(a: str, b: str, model_id: str) -> str:
    """Where two generations first diverge, in tokens (character fallback without a tokenizer)."""
    tok = _tokenizer(model_id)
    if tok is not None:
        ta = tok(a, add_special_tokens=False)["input_ids"]
        tb = tok(b, add_special_tokens=False)["input_ids"]
        k = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        ctx = tok.decode(ta[max(0, k - 4):k])
        ra = tok.decode(ta[k:k + 4]) if k < len(ta) else "<end>"
        rb = tok.decode(tb[k:k + 4]) if k < len(tb) else "<end>"
        return f"first divergence at token {k} (ref {len(ta)} / new {len(tb)} tokens) after {ctx!r}: ref {ra!r} vs new {rb!r}"
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return f"first divergence at char {k}: ref {a[k:k + 24]!r} vs new {b[k:k + 24]!r}"


# ── json payload ────────────────────────────────────────────────────────────────────────────────
def _floats_close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    t = TOL["json_float"]
    return abs(a - b) <= t["abs"] + t["rel"] * abs(a)


def compare_json(rep: Report, ref, new, path: str, exact: bool, soft: bool, model_id: str):
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            rep.check(False, f"{path}: type {type(new).__name__} != dict", soft)
            return
        for k, v in ref.items():
            if str(k).startswith("_meta") or (k in DERIVED_KEYS and not exact):
                continue
            if k not in new:
                rep.check(False, f"{path}/{k}: missing in new", soft)
                continue
            compare_json(rep, v, new[k], f"{path}/{k}", exact, soft, model_id)
        extra = [k for k in new if k not in ref and not str(k).startswith("_meta")]
        if extra:
            rep.warns.append(f"{path}: keys only in new: {extra}")
        return
    if isinstance(ref, list):
        if not isinstance(new, list) or len(new) != len(ref):
            n = len(new) if isinstance(new, list) else type(new).__name__
            msg = f"{path}: list length {len(ref)} != {n}"
            if isinstance(new, list):
                k = next((i for i in range(min(len(ref), len(new))) if ref[i] != new[i]), min(len(ref), len(new)))
                msg += f" (first differing index {k})"
            rep.check(False, msg, soft)
            return
        for i, (a, b) in enumerate(zip(ref, new)):
            compare_json(rep, a, b, f"{path}[{i}]", exact, soft, model_id)
        return
    if isinstance(ref, str):
        ok = ref == new
        if not ok:
            rep.n_str_diff += 1
            msg = f"{path}: string differs; " + (token_diff(ref, new, model_id) if isinstance(new, str) else
                                                 f"new is {type(new).__name__}")
            rep.check(False, msg, soft)
        else:
            rep.check(True, "")
        return
    if isinstance(ref, bool) or ref is None or isinstance(ref, int) and not isinstance(new, float):
        rep.check(ref == new, f"{path}: {ref!r} != {new!r}", soft)
        return
    if isinstance(ref, (int, float)):
        if not isinstance(new, (int, float)) or isinstance(new, bool):
            rep.check(False, f"{path}: {ref!r} != {new!r}", soft)
            return
        ok = (float(ref) == float(new)) if exact else _floats_close(float(ref), float(new))
        rep.check(ok, f"{path}: {ref!r} vs {new!r} (diff {abs(float(ref) - float(new)):.3g})", soft)
        return
    rep.check(ref == new, f"{path}: {ref!r} != {new!r}", soft)


# ── tensors ─────────────────────────────────────────────────────────────────────────────────────
def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    d = b - a
    na = a.norm().item()
    return dict(max_abs=d.abs().max().item() if d.numel() else 0.0,
                rel=(d.norm().item() / na) if na > 0 else d.norm().item(),
                cos=torch.nn.functional.cosine_similarity(a, b, dim=0).item() if na > 0 and b.norm() > 0 else 1.0)


def compare_tensor(rep: Report, key: str, kind: str, a: torch.Tensor, b: torch.Tensor, exact: bool, soft: bool):
    if tuple(a.shape) != tuple(b.shape):
        rep.check(False, f"{key}: shape {tuple(a.shape)} != {tuple(b.shape)}", soft)
        return
    if exact or kind in EXACT_KINDS:
        ok = torch.equal(a, b) if a.dtype == b.dtype else torch.equal(a.double(), b.double())
        m = _metrics(a, b) if not ok and a.is_floating_point() else {}
        rep.check(ok, f"{key} [{kind}]: not bit-identical {_fmt(m)}", soft)
        if m:
            rep.worst_rel = max(rep.worst_rel, m["rel"])
        return
    m = _metrics(a, b)
    rep.worst_rel = max(rep.worst_rel, m["rel"])
    t = TOL[kind]
    bad = [f"{name} {m[name]:.3g} > {lim:g}" for name, lim in t.items() if name in ("rel", "max_abs") and m[name] > lim]
    bad += [f"cos {m['cos']:.6f} < {t['cos']:g}" for _ in [0] if "cos" in t and m["cos"] < t["cos"]]
    rep.check(not bad, f"{key} [{kind}]: " + "; ".join(bad) + f"  {_fmt(m)}", soft)


def compare_topk(rep: Report, key: str, of: str, ids_a, vals_a, ids_b, vals_b, exact: bool, soft: bool):
    if exact:
        ok = torch.equal(ids_a, ids_b) and torch.equal(vals_a, vals_b)
        rep.check(ok, f"{key} [topk {of}]: not bit-identical", soft)
        return
    if ids_a.shape != ids_b.shape:
        rep.check(False, f"{key} [topk]: shape {tuple(ids_a.shape)} != {tuple(ids_b.shape)}", soft)
        return
    ia, ib = ids_a.reshape(-1, ids_a.shape[-1]), ids_b.reshape(-1, ids_b.shape[-1])
    va, vb = vals_a.reshape(ia.shape), vals_b.reshape(ib.shape)
    t = TOL["topk"]
    lim = t["max_abs_probs"] if of == "probs" else t["max_abs_logits"]
    msgs = []
    for row in range(ia.shape[0]):
        if ia[row, 0] != ib[row, 0]:
            msgs.append(f"row {row}: top-1 id {int(ia[row, 0])} != {int(ib[row, 0])}")
        sa, sb = ia[row].tolist(), ib[row].tolist()
        common = set(sa) & set(sb)
        ov = len(common) / len(sa)
        if ov < t["overlap"]:
            msgs.append(f"row {row}: top-{len(sa)} overlap {ov:.2f} < {t['overlap']}")
        dv = max((abs(float(va[row, sa.index(i)]) - float(vb[row, sb.index(i)])) for i in common), default=0.0)
        if dv > lim:
            msgs.append(f"row {row}: max |dvalue| over shared ids {dv:.3g} > {lim:g}")
    rep.check(not msgs, f"{key} [topk {of}]: " + "; ".join(msgs[:4]), soft)


def _fmt(m: dict) -> str:
    return "(" + ", ".join(f"{k}={v:.3g}" for k, v in m.items()) + ")" if m else ""


# ── driver ──────────────────────────────────────────────────────────────────────────────────────
def compare_case(name: str, ref_dir: Path, new_dir: Path, exact: bool, model_id: str) -> Report:
    rep = Report(name)
    rj = json.loads((ref_dir / f"{name}.json").read_text())
    nj_path = new_dir / f"{name}.json"
    if not nj_path.exists():
        rep.check(False, "case missing in new run")
        return rep
    nj = json.loads(nj_path.read_text())
    if rj.get("error"):
        rep.check(False, "reference run errored: " + rj["error"].strip().splitlines()[-1])
    if nj.get("error"):
        rep.check(False, "new run errored: " + nj["error"].strip().splitlines()[-1])
        return rep
    soft = bool(rj.get("padded")) and not exact
    compare_json(rep, rj["data"], nj["data"], "data", exact, soft, model_id)
    man = rj.get("manifest", {})
    if man:
        ta = torch.load(ref_dir / f"{name}.pt", weights_only=True)
        tb_path = new_dir / f"{name}.pt"
        if not tb_path.exists():
            rep.check(False, "tensor file missing in new run")
            return rep
        tb = torch.load(tb_path, weights_only=True)
        for key, info in man.items():
            kind = info["kind"]
            if kind == "topk_vals":
                continue  # handled with its ids
            if key not in tb:
                rep.check(False, f"{key}: tensor missing in new run", soft)
                continue
            if kind == "topk_ids":
                base = key[: -len("/top_ids")]
                vk = base + "/top_vals"
                if vk not in tb:
                    rep.check(False, f"{vk}: tensor missing in new run", soft)
                    continue
                compare_topk(rep, base, man[vk].get("of", "logits"), ta[key], ta[vk], tb[key], tb[vk], exact, soft)
                continue
            a, b = ta[key], tb[key]
            if not exact:
                a, b = a.float(), b.float()
            elif a.dtype != b.dtype:  # stored dtype may differ when one side was not bf16-representable
                a, b = a.double(), b.double()
            compare_tensor(rep, key, kind, a, b, exact, soft)
        extra = [k for k in tb if k not in man]
        if extra:
            rep.warns.append(f"tensors only in new: {extra}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--exact", action="store_true", help="require bit-identical tensors and floats (self-check)")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--max-details", type=int, default=12, help="failure / warning lines printed per case")
    args = ap.parse_args()

    env_a = json.loads((args.ref_dir / "env.json").read_text())
    env_b = json.loads((args.new_dir / "env.json").read_text()) if (args.new_dir / "env.json").exists() else {}
    model_id = env_a.get("model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    print(f"mode: {'EXACT (bit-identity)' if args.exact else 'tolerance (cross-stack)'}")
    for k in ("git_sha", "torch", "transformers", "nnsight", "gpu", "model_snapshot"):
        a, b = env_a.get(k), env_b.get(k)
        print(f"  {k:15s} ref={a}  new={b}{'' if a == b else '   <- differs'}")

    names = env_a.get("cases") or sorted(p.stem for p in args.ref_dir.glob("*.json") if p.stem != "env")
    if args.cases:
        names = [n for n in names if n in args.cases]
    reps = [compare_case(n, args.ref_dir, args.new_dir, args.exact, model_id) for n in names]

    w = max(len(r.case) for r in reps)
    print(f"\n{'case':{w}s}  status  checks  fails  warns  str-diffs  worst-rel")
    for r in reps:
        print(f"{r.case:{w}s}  {r.status:6s}  {r.n:6d}  {len(r.fails):5d}  {len(r.warns):5d}  {r.n_str_diff:9d}  "
              f"{r.worst_rel:9.3g}")
    for r in reps:
        if r.fails or r.warns:
            print(f"\n== {r.case} ({r.status})")
            for m in r.fails[: args.max_details]:
                print(f"  FAIL {m}")
            if len(r.fails) > args.max_details:
                print(f"  ... {len(r.fails) - args.max_details} more failures")
            for m in r.warns[: args.max_details]:
                print(f"  WARN {m}")
            if len(r.warns) > args.max_details:
                print(f"  ... {len(r.warns) - args.max_details} more warnings")
    n_fail = sum(r.status == "FAIL" for r in reps)
    print(f"\n{len(reps)} cases: {sum(r.status == 'PASS' for r in reps)} pass, "
          f"{sum(r.status == 'WARN' for r in reps)} warn, {n_fail} fail")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
