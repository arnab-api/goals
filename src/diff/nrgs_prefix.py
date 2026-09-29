"""NRGS common-prefix contrastive setup — localize query-carried PROGRESS by a next-token loss.

Order-diff query transfer (as in :func:`src.diff.data.diff_trigger_setups`), except the DST frontier
line is prefilled with an implementation prefix SHARED by b (DST's native frontier) and d (the
NRGS-predicted one), so b and d diverge on the NEXT token after the prefix::

    SRC   [A, C, B]  prefill {A, B}  ->  ``c =``            frontier C   (query cached at `` =``)
    DST   [A, B, D]  prefill {A}     ->  ``b = <prefix>``   next token: b_tok  (native)
    DST*  [A, B, D]  prefill {A, B}  ->  ``c = <prefix>``   next token: d_tok  (target distribution)
    SRC'  [A, C, B]  prefill {A}     ->  ``b =``            frontier C, b NOT done  (control query)

Patch = the selected heads' pre-RoPE query at DST's LAST token (the end of the prefix, where b and d
diverge) <- SRC's `` =`` query. NRGS predicts the flip b_tok -> d_tok under SRC's query but not under
SRC''s, which carries the same "look for C" frontier without the "B done" progress.

Letters are SEQUENTIAL by appearance (:func:`src.diff.data._seq_letter`). The (b, d, prefix) triples
come from the goal-head localization pivots (``data/head_loc/pairs/<model>.json``): b = the pair's
destination predicate (``dest_top_token`` = b_tok), d = its source predicate (``target_top_token`` =
d_tok). A and C are drawn from the same 7-predicate bank so variable names and semantics match the
contexts the pivots were mined in.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch

import src.diff.data as D

VARIANTS = ("order_diff", "order_matched", "c_in_dst")
PIVOT_POOL = "data/head_loc/pairs/{model}.json"
PRED_DIR = "data/head_loc/predicates"
AUG_WRAPPERS = ("bool(", "int(", "not not ", "True and ", "False or ")  # = head_loc_oracle.AUGMENTATION_WRAPPERS


# ── the goal-head localization bank (data/head_loc/predicates) as one-line Python impls ──────────
VAR_NAME = {"str": "s", "int": "n", "list[int]": "lst"}


def _p(id, description, expr):
    # cpp impl not written yet — _check_lang refuses cpp until these one-liners get C++ twins
    return D._P(id, description, id, "{res} = " + expr, None)


BANK: dict[str, dict[str, D.Predicate]] = {
    dt: {p.id: p for p in preds}
    for dt, preds in {
        "str": (
            _p("ends_with_vowel", "whether the string ends with a vowel (case-insensitive)",
               "bool({arg}) and {arg}[-1].lower() in 'aeiou'"),
            _p("has_spaces", "whether the string contains a space", "' ' in {arg}"),
            _p("is_digit_string", "whether every character is a digit", "{arg}.isdigit()"),
            _p("is_lower", "whether all alphabetic characters are lowercase", "{arg} == {arg}.lower()"),
            _p("is_palindrome", "whether the string is a palindrome", "{arg} == {arg}[::-1]"),
            _p("is_upper", "whether all alphabetic characters are uppercase", "{arg} == {arg}.upper()"),
            _p("starts_with_vowel", "whether the string starts with a vowel (case-insensitive)",
               "bool({arg}) and {arg}[0].lower() in 'aeiou'"),
        ),
        "int": (
            _p("is_divisible_by_three", "whether the number is divisible by three", "{arg} % 3 == 0"),
            _p("is_even", "whether the number is even", "{arg} % 2 == 0"),
            _p("is_odd", "whether the number is odd", "{arg} % 2 != 0"),
            _p("is_perfect_square", "whether the number is a perfect square",
               "{arg} >= 0 and int({arg} ** 0.5) ** 2 == {arg}"),
            _p("is_positive", "whether the number is positive", "{arg} > 0"),
            _p("is_power_of_two", "whether the number is a power of two",
               "{arg} > 0 and ({arg} & ({arg} - 1)) == 0"),
            _p("is_prime", "whether the number is prime",
               "{arg} > 1 and all({arg} % i for i in range(2, int({arg} ** 0.5) + 1))"),
        ),
        "list[int]": (
            _p("all_even", "whether every element is even", "all(x % 2 == 0 for x in {arg})"),
            _p("all_positive", "whether every element is positive", "all(x > 0 for x in {arg})"),
            _p("all_same", "whether all elements are equal", "len(set({arg})) <= 1"),
            _p("has_duplicates", "whether any element appears more than once", "len(set({arg})) != len({arg})"),
            _p("is_sorted", "whether the list is sorted in non-decreasing order",
               "all({arg}[i] <= {arg}[i + 1] for i in range(len({arg}) - 1))"),
            _p("is_strictly_decreasing", "whether the list is strictly decreasing",
               "all({arg}[i] > {arg}[i + 1] for i in range(len({arg}) - 1))"),
            _p("is_strictly_sorted", "whether the list is strictly increasing",
               "all({arg}[i] < {arg}[i + 1] for i in range(len({arg}) - 1))"),
        ),
    }.items()
}

# near-duplicate predicates share a family; with distinct_families, A and C avoid every other role's family
FAMILY = {
    "is_lower": "case", "is_upper": "case",
    "starts_with_vowel": "vowel_pos", "ends_with_vowel": "vowel_pos",
    "is_even": "parity", "is_odd": "parity",
    "is_sorted": "order", "is_strictly_sorted": "order", "is_strictly_decreasing": "order",
}


def _fam(pid: str) -> str:
    return FAMILY.get(pid, pid)


def validate_bank(pred_dir: str = PRED_DIR) -> list[dict]:
    """Run every one-liner against its head_loc unit tests. One row per predicate."""
    rows = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        j = json.load(open(f))
        dt, pid, arg = j["variable_type"], j["predicate_name"], j["variable_name"]
        assert arg == VAR_NAME[dt], f"{pid}: head_loc var {arg!r} != bank var {VAR_NAME[dt]!r}"
        code = BANK[dt][pid].impl["python"][len("{res} = "):].replace("{arg}", arg)
        fails = []
        for inp, exp in j["unit_tests"]:
            # arg goes in GLOBALS: generator-expression bodies can't see eval() locals
            got = eval(code, {"__builtins__": __builtins__, arg: inp})
            if got != exp:
                fails.append((inp, exp, got))
        rows.append(dict(datatype=dt, id=pid, impl=code, n_tests=len(j["unit_tests"]),
                         n_fail=len(fails), fails=fails))
    return rows


# ── pivot pool: (b, d, shared prefix) from the goal-head localization pairs ──────────────────────
@dataclass(frozen=True)
class Pivot:
    datatype: str
    b: str        # DST native frontier  (head_loc destination predicate)
    d: str        # NRGS target          (head_loc source predicate)
    prefix: str   # non-empty implementation prefix shared by b and d
    b_tok: str    # head_loc next token after the prefix, destination side
    d_tok: str    # head_loc next token after the prefix, target side


def _core(prefix: str) -> Optional[str]:
    for w in AUG_WRAPPERS:
        if prefix.startswith(w):
            return prefix[len(w):]
    return None


def load_pivots(model_name: str, datatypes=None, core_only: bool = True) -> list[Pivot]:
    """Non-empty-prefix pivots. ``core_only`` drops a wrapper augmentation (``bool(s == s``,
    ``True and s == s``, ...) when its unwrapped prefix is already in the same pair's pool."""
    path = PIVOT_POOL.format(model=model_name.split("/")[-1])
    out = []
    for pr in json.load(open(path))["pairs"]:
        dt = pr["variable_type"]
        if dt not in BANK or (datatypes and dt not in datatypes):
            continue
        pis = [pi for pi in pr["pivot_instances"] if pi["pivot_prefix"].strip()]
        pool = {pi["pivot_prefix"] for pi in pis}
        for pi in pis:
            core = _core(pi["pivot_prefix"])
            if core_only and core is not None and core in pool:
                continue
            out.append(Pivot(dt, pr["destination_predicate"], pr["source_predicate"],
                             pi["pivot_prefix"], pi["dest_top_token"], pi["target_top_token"]))
    return out


# ── the contrastive instance ──────────────────────────────────────────────────────────────────
def _with_prefix(part, prefix):
    q, code = part
    return q, code + " " + prefix


@dataclass
class NRGSPrefixInstance:
    pivot: Pivot
    a: str
    c: str
    variant: str = "order_diff"
    src_lang: str = "python"
    dst_lang: str = "python"
    presentation: str = "listed"

    def __post_init__(self):
        assert self.variant in VARIANTS, self.variant
        for lang in (self.src_lang, self.dst_lang):
            if lang != "python":
                raise NotImplementedError("the head_loc bank has Python one-liners only; add C++ twins first")
        ids = [self.a, self.pivot.b, self.c, self.pivot.d]
        assert len(set(ids)) == 4, f"A, B, C, D must be distinct: {ids}"

    @property
    def datatype(self) -> str:
        return self.pivot.datatype

    @property
    def key(self) -> str:
        p = self.pivot
        return (f"{self.variant}|{p.datatype}|A={self.a}|B={p.b}|C={self.c}|D={p.d}"
                f"|prefix={p.prefix}|{self.src_lang}-{self.dst_lang}")

    @property
    def base(self) -> D.DiffTriggerInstance:
        """The SRC/DST pair as a regular diff-trigger instance (same renderer as the eval pipeline)."""
        dt, bank = self.datatype, BANK[self.datatype]
        A, B, C, Dp = (bank[x] for x in (self.a, self.pivot.b, self.c, self.pivot.d))
        if self.variant == "c_in_dst":
            src_order, dst_order = (A, B, C, Dp), (A, B, Dp, C)
        elif self.variant == "order_matched":
            src_order, dst_order = (A, B, C), (A, B, Dp)
        else:
            src_order, dst_order = (A, C, B), (A, B, Dp)
        return D.DiffTriggerInstance(
            datatype=dt, src_order=src_order, dst_order=dst_order,
            src_progress=[A.id, B.id], dst_progress=[A.id],
            diff_target=Dp.id, dst_native=B.id, src_target=C.id,
            var_name=VAR_NAME[dt], presentation=self.presentation,
            src_lang=self.src_lang, dst_lang=self.dst_lang,
            naming="letters", letter_stub="var_eq",
            meta={"variant": self.variant, "dst_next": B.id, "diff_target": Dp.id,
                  "src_target": C.id, "datatype": dt, "prefix": self.pivot.prefix},
        )

    def parts(self) -> dict[str, tuple[str, str]]:
        """role -> (question, code prefill). Roles: src, dst, dst_star, and src_ctrl (order_diff only:
        SRC' needs C listed before B so that C stays the frontier with B un-prefilled)."""
        i = self.base
        A, B = i.src_progress
        dstar = i._parts(i.dst_lang, i.dst_order, [A, B])
        assert dstar[1].rstrip().endswith("c ="), "DST* stub must be the third line (frontier D)"
        out = {
            "src": i.src_parts(),
            "dst": _with_prefix(i.dst_parts(), self.pivot.prefix),
            "dst_star": _with_prefix(dstar, self.pivot.prefix),
        }
        if self.variant == "order_diff":
            out["src_ctrl"] = i._parts(i.src_lang, i.src_order, [A])
        return out

    def prompts(self, ctx) -> dict[str, str]:
        from src.diff.functional import get_prompt

        return {role: get_prompt(q, D._finalize_prefill(code), ctx=ctx)
                for role, (q, code) in self.parts().items()}


def to_record(inst: NRGSPrefixInstance) -> dict:
    """JSON-able instance (pivot fields + roles) — round-trips through :func:`from_record`."""
    p = inst.pivot
    return dict(key=inst.key, datatype=p.datatype, a=inst.a, b=p.b, c=inst.c, d=p.d, prefix=p.prefix,
                b_tok=p.b_tok, d_tok=p.d_tok, variant=inst.variant, src_lang=inst.src_lang,
                dst_lang=inst.dst_lang, presentation=inst.presentation)


def from_record(r: dict) -> NRGSPrefixInstance:
    pv = Pivot(r["datatype"], r["b"], r["d"], r["prefix"], r["b_tok"], r["d_tok"])
    return NRGSPrefixInstance(pv, r["a"], r["c"], r["variant"], r["src_lang"], r["dst_lang"],
                              r.get("presentation", "listed"))


def build_instances(pivots, variant="order_diff", n_per_pivot=1, seed=0, distinct_families=True,
                    src_lang="python", dst_lang="python") -> list[NRGSPrefixInstance]:
    """For each pivot (b, d, prefix), draw ``n_per_pivot`` (A, C) orderings from the rest of the bank."""
    rng = random.Random(seed)
    out = []
    for pv in pivots:
        used = {_fam(pv.b), _fam(pv.d)}
        pool = [p for p in BANK[pv.datatype]
                if p not in (pv.b, pv.d) and (not distinct_families or _fam(p) not in used)]
        pairs = [(a, c) for a in pool for c in pool
                 if a != c and (not distinct_families or _fam(a) != _fam(c))]
        rng.shuffle(pairs)
        out += [NRGSPrefixInstance(pv, a, c, variant, src_lang, dst_lang) for a, c in pairs[:n_per_pivot]]
    return out


def pair_id(inst: NRGSPrefixInstance) -> tuple:
    """Unordered (b, d) pair — (b, d) and (d, b) share predicates and tokens, so split on this."""
    return (inst.datatype, frozenset({inst.pivot.b, inst.pivot.d}))


def stratified_sample(instances, n, seed=0):
    """``n`` instances balanced across datatypes and, within a datatype, spread round-robin over the
    (b, d) pairs (random order within a pair) — so no datatype or pair dominates a small sample."""
    import collections

    rng = random.Random(seed)
    by_dt = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in instances:
        by_dt[x.datatype][pair_id(x)].append(x)
    queues = {}
    for dt, pairs in by_dt.items():
        groups = [rng.sample(v, len(v)) for _, v in sorted(pairs.items(), key=lambda kv: sorted(kv[0][1]))]
        rng.shuffle(groups)
        q = []
        while any(groups):
            for g in groups:
                if g:
                    q.append(g.pop())
        queues[dt] = q
    out, dts = [], sorted(queues)
    while len(out) < n and any(queues[d] for d in dts):
        for d in dts:
            if queues[d] and len(out) < n:
                out.append(queues[d].pop(0))
    return out


def split_by_pair(instances, val_frac=0.2, seed=0):
    """Train/val split with DISJOINT (b, d) pairs, so val measures generalization to unseen pairs."""
    pids = sorted({pair_id(x) for x in instances}, key=lambda t: (t[0], sorted(t[1])))
    random.Random(seed).shuffle(pids)
    val = set(pids[: max(1, round(len(pids) * val_frac))])
    return [x for x in instances if pair_id(x) not in val], [x for x in instances if pair_id(x) in val]


# ── model-side checks (validation notebook + later training) ─────────────────────────────────
def load_ctx(model_name: str, device_map="auto"):
    """Minimal, model-agnostic run context for the NRGS scripts: the model + what ``get_prompt`` /
    ``heads_per_layer`` read (``mt``, ``group``). Unlike ``src.diff.functional.setup`` it loads no
    model-specific goal / boundary masks, so it works for any size (e.g. the 14B smoke test)."""
    from types import SimpleNamespace

    from src.models import ModelandTokenizer

    mt = ModelandTokenizer(model_key=model_name, dtype=torch.bfloat16, device_map=device_map,
                           attn_implementation="eager")
    cfg = mt._model.config
    return SimpleNamespace(mt=mt, model_name=model_name, n_layer=mt.n_layer, n_heads=mt.n_heads,
                           group=cfg.num_attention_heads // cfg.num_key_value_heads)


@torch.no_grad()
def last_token_logits(mt, prompts: list[str]) -> torch.Tensor:
    """(n, vocab) float32 CPU raw logits of the next token (left-padded batch)."""
    from src.tokens import prepare_input

    inp = prepare_input(prompts=prompts, tokenizer=mt)
    out = mt._model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    return out.logits[:, -1].float().cpu()


def last_token_logprobs(mt, prompts: list[str]) -> torch.Tensor:
    """(n, vocab) float32 CPU log-probs of the next token (left-padded batch)."""
    return last_token_logits(mt, prompts).log_softmax(-1)


def indirect_effect(l_patched, l_dst, l_star, targ: int) -> float:
    """Normalized indirect effect on ``targ``'s LOGIT: (M(dst | <- q_src) - M(dst)) / (M(dst*) - M(dst)).
    1 = the patch moves targ as far as the "B done" counterfactual DST*; 0 = no effect."""
    return float((l_patched[targ] - l_dst[targ]) / (l_star[targ] - l_dst[targ]))


def gate(mt, ctx, instances, batch_size=8) -> list[dict]:
    """Divergence gate in the NRGS context. b_id / d_id are the MODEL's top-1 after the prefix in DST /
    DST* (the loss targets); ``match_b`` / ``match_d`` say whether they equal the head_loc tokens."""
    from src.goals.head_loc_oracle import _tokens_differ_meaningfully

    dec = lambda i: mt.tokenizer.decode([i])  # noqa: E731
    rows = []
    for s in range(0, len(instances), batch_size):
        chunk = instances[s: s + batch_size]
        ps = [x.prompts(ctx) for x in chunk]
        lp_dst = last_token_logprobs(mt, [p["dst"] for p in ps])
        lp_str = last_token_logprobs(mt, [p["dst_star"] for p in ps])
        for x, ld, ls in zip(chunk, lp_dst, lp_str):
            b_id, d_id = int(ld.argmax()), int(ls.argmax())
            tb, td = dec(b_id), dec(d_id)
            diverge = b_id != d_id and _tokens_differ_meaningfully(tb, td)
            rows.append(dict(
                key=x.key, datatype=x.datatype, b=x.pivot.b, d=x.pivot.d, prefix=x.pivot.prefix,
                b_id=b_id, d_id=d_id, top1_dst=tb, top1_dst_star=td,
                p_dst_b=float(ld[b_id].exp()), p_dst_d=float(ld[d_id].exp()),
                p_star_b=float(ls[b_id].exp()), p_star_d=float(ls[d_id].exp()),
                match_b=tb == x.pivot.b_tok, match_d=td == x.pivot.d_tok,
                diverge=diverge, passed=diverge and tb == x.pivot.b_tok and td == x.pivot.d_tok,
            ))
    return rows


def q_patch_spec(mt, ctx, inst, heads, src_role="src") -> list[dict]:
    """``proj_patch_hooks`` spec: ``heads``' pre-RoPE queries cached at the ``src_role`` prompt's LAST
    token (the `` =`` pivot), written into position -1 of whatever runs under the hooks (DST's last
    token = the end of the shared prefix). Hard replacement; use as
    ``with FUNC.proj_patch_hooks(mt, spec): ...``."""
    import src.diff.functional as FUNC
    from src.tokens import prepare_input

    src_prompt = inst.prompts(ctx)[src_role]
    n_src = prepare_input(prompts=src_prompt, tokenizer=mt)["input_ids"].shape[1]
    layers = sorted({l for l, _ in heads})
    cache = FUNC.cache_src_proj(src_prompt, [n_src - 1], layers, "q", ctx=ctx)
    return [{"which": "q", "cache": cache, "heads": FUNC.heads_per_layer(heads, "q", ctx=ctx),
             "positions": [-1]}]


@torch.no_grad()
def q_patch_logits(mt, ctx, inst, heads, src_role="src", ov_boost=1.0) -> torch.Tensor:
    """(vocab,) raw next-token logits at DST's last token under :func:`q_patch_spec`. ``ov_boost`` also
    scales the SAME heads' OV write (o_proj-input slice) at that token — as the paper's q-transfer runs
    stack ``boost_head_ov`` on the query patch (``--alphas 2``). 1.0 = query patch only."""
    import src.diff.functional as FUNC

    with FUNC.proj_patch_hooks(mt, q_patch_spec(mt, ctx, inst, heads, src_role)), \
            FUNC.boost_head_ov(mt, heads, ov_boost, positions=(-1,)):
        return last_token_logits(mt, [inst.prompts(ctx)["dst"]])[0]


def q_patch_logprobs(mt, ctx, inst, heads, src_role="src", ov_boost=1.0) -> torch.Tensor:
    """(vocab,) next-token log-probs under :func:`q_patch_logits` (non-differentiable baseline)."""
    return q_patch_logits(mt, ctx, inst, heads, src_role, ov_boost).log_softmax(-1)
