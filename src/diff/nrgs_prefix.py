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
import itertools
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch

import src.diff.data as D

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


# ── contrastive configs (notation: ``p: a'bc' -> b`` = spec order a, b, c; x' prefilled; arrow = frontier) ──
@dataclass(frozen=True)
class Config:
    """One SRC/DST contrast. Letters are tasks (shared letters = the same task in SRC and DST). ``target`` is
    the q-transfer target: the frontier of DST* (DST with every task listed before ``target`` prefilled),
    whose next-token distribution after the shared prefix is the training target. The shared prefix is
    between the implementations of DST's native frontier and ``target`` (a goal-head localization pivot)."""
    name: str
    group: str                       # order_diff | order_matched | diff_vs_prog
    src: tuple
    src_prefill: frozenset
    src_next: str                    # SRC frontier (the notation's arrow; validated)
    dst: tuple
    dst_prefill: frozenset
    native: str                      # DST frontier (the notation's arrow; validated)
    target: str
    note: str = ""
    hypothesis: str = "H1"           # which query-content hypothesis ``target`` is the prediction of (HYPOTHESES)

    @property
    def letters(self) -> tuple:
        return tuple(sorted(set(self.src) | set(self.dst)))

    @property
    def predictions(self) -> dict:
        """DST letter each query-content hypothesis predicts for this contrast (:func:`predict`)."""
        return predict(self.src, self.src_prefill, self.dst, self.dst_prefill)

    @property
    def others(self) -> tuple:       # letters filled from the bank (native / target come from the pivot)
        return tuple(x for x in self.letters if x not in (self.native, self.target))

    @property
    def dst_star_prefill(self) -> frozenset:
        return frozenset(self.dst[: self.dst.index(self.target)])

    @property
    def src_ctrl_prefill(self):
        """SRC' = SRC with every prefilled task listed AFTER the SRC frontier un-prefilled: same frontier
        ("look for" the same task), but not the progress that makes ``target`` pending. ``None`` when no
        such task exists (the control is then undefined for this config)."""
        after = {x for x in self.src[self.src.index(self.src_next) + 1:] if x in self.src_prefill}
        if not after:
            return None
        ctrl = self.src_prefill - after
        # not a negative when the hypothesis predicts the target under SRC' too (e.g. H2: same frontier slot)
        if predict(self.src, ctrl, self.dst, self.dst_prefill).get(self.hypothesis) == self.target:
            return None
        return ctrl

    @property
    def notation(self) -> str:
        f = lambda order, pre, nxt: "".join(x + ("'" if x in pre else "") for x in order) + f" -> {nxt}"  # noqa: E731
        return (f"src {f(self.src, self.src_prefill, self.src_next)} | dst {f(self.dst, self.dst_prefill, self.native)}"
                f" | target {self.target} ({self.hypothesis})")


def _first_open(order, prefill):
    return next(x for x in order if x not in prefill)


# ── query-content hypotheses (agent_reports/claude/analysis/q_transfer_query_content_hypotheses_design.md) ──
HYPOTHESES = ("H1", "H1c", "H2", "H3")


def predict(src, src_pre, dst, dst_pre) -> dict:
    """DST letter each hypothesis predicts when SRC's pivot query is transplanted into DST (None = undefined).

    H0  transfer fails            -> DST native
    H1  set progress D_src         -> first DST item not in D_src
    H1c count |D_src|              -> DST spec slot |D_src|
    H2  order id pos_src(t_src)    -> DST spec slot pos_src(t_src)
    H3  induction "I follow x", x = spec predecessor of t_src -> the item after x in DST spec;
        defined only for an ORDER-MATCHED SRC (everything before t_src done, nothing after), where x is
        also the last-written item — in an order-broken SRC the target's key ("I follow x") and the last
        written line disagree, so there is no induction target."""
    t = _first_open(src, src_pre)
    i = src.index(t)
    matched = all(x in src_pre for x in src[:i]) and not any(x in src_pre for x in src[i:])
    at = lambda k: dst[k] if 0 <= k < len(dst) else None  # noqa: E731
    x = src[i - 1] if i > 0 else None
    return {"H0": _first_open(dst, dst_pre),
            "H1": next((y for y in dst if y not in src_pre), None),
            "H1c": at(len(src_pre)),
            "H2": at(i),
            "H3": (dst[dst.index(x) + 1] if matched and x in dst and dst.index(x) + 1 < len(dst) else None)}


def _parse(side: str):
    body, nxt = (t.strip() for t in side.split("->"))
    order, pre = [], set()
    for ch in body:
        if ch == "'":
            pre.add(order[-1])
        elif ch.strip():
            order.append(ch)
    return tuple(order), frozenset(pre), nxt


def _cfg(name, group, src, dst, target, note="", hypothesis="H1", strict=False):
    """``strict`` (the H1c / H2 / H3 configs): the target must be that hypothesis's UNIQUE prediction — not
    the native, not predicted by any other (defined) hypothesis, never seen in SRC (a letter SRC wrote or
    listed could be picked up by a copy / completeness bias) — and SRC's own frontier must be absent from
    DST (else the transplanted target identity, App I.3, would dominate)."""
    so, sp, sn = _parse(src)
    do, dp, dn = _parse(dst)
    c = Config(name, group, so, sp, sn, do, dp, dn, target, note, hypothesis)
    assert hypothesis in HYPOTHESES, f"{name}: unknown hypothesis {hypothesis}"
    assert _first_open(so, sp) == sn, f"{name}: SRC frontier is {_first_open(so, sp)}, notation says {sn}"
    assert _first_open(do, dp) == dn, f"{name}: DST frontier is {_first_open(do, dp)}, notation says {dn}"
    assert target in do and target not in dp and target != dn, f"{name}: bad target {target}"
    assert _first_open(do, c.dst_star_prefill) == target, f"{name}: DST* frontier != target"
    assert len(set(so)) == len(so) and len(set(do)) == len(do), f"{name}: repeated task"
    P = c.predictions
    assert P[hypothesis] == target, f"{name}: {hypothesis} predicts {P[hypothesis]}, target is {target}"
    if strict:
        rivals = {h: v for h, v in P.items() if h not in ("H0", hypothesis) and v is not None}
        assert target not in rivals.values(), f"{name}: target {target} also predicted by {rivals}"
        assert target not in so, f"{name}: target {target} appears in SRC"
        assert sn not in do, f"{name}: SRC frontier {sn} appears in DST"
    return c


CONFIGS: dict[str, Config] = {c.name: c for c in (
    # ── H1 set progress (the original q-transfer targets; other hypotheses may coincide) ──
    _cfg("order_diff_1", "order_diff", "a'cb' -> c", "a'bd -> b", "d", "the original order-diff"),
    _cfg("order_diff_2", "order_diff", "a'bc'd' -> b", "a'dec -> d", "e", "count ('3 done -> 4th') would predict c"),
    _cfg("order_diff_3", "order_diff", "a'b'cd' -> c", "a'bde -> b", "e"),
    _cfg("order_matched_1", "order_matched", "a'b'c -> c", "a'bd -> b", "d"),
    _cfg("order_matched_2", "order_matched", "a'b'c'd -> d", "a'bce -> b", "e"),
    _cfg("diff_vs_prog_1", "diff_vs_prog", "a'b'cd -> c", "a'bdc -> b", "d",
         "d = the output of the progress; c = the SRC frontier (the 'next op')"),
    # ── H1c count progress: order-broken SRC with |D| != pos(t_src) (5 tasks) ──
    _cfg("h1c_1", "h1c", "a'bc' -> b", "a'de -> d", "e", "2 done -> slot 2; H1 = H2 = native d",
         hypothesis="H1c", strict=True),
    _cfg("h1c_2", "h1c", "a'bc' -> b", "a'dec -> d", "e", "as h1c_1, plus a SRC-done task c pending in DST (H1 skips it)",
         hypothesis="H1c", strict=True),
    # ── H2 order id: order-broken SRC, target slot = pos(t_src) (6 tasks: needs 6 distinct families) ──
    _cfg("h2_1", "h2", "a'b'cd' -> c", "a'efb -> e", "f", "pos(c) = 2 -> slot 2; H1 = native e, H1c -> b",
         hypothesis="H2", strict=True),
    _cfg("h2_2", "h2", "a'b'cd' -> c", "a'efd -> e", "f", "as h2_1 with H1c -> d (a SRC-done task)",
         hypothesis="H2", strict=True),
    # ── H3 induction: order-matched SRC, target = the task after pred(t_src) = b in DST (5 tasks) ──
    _cfg("h3_1", "h3", "a'b'c -> c", "a'ebd -> e", "d", "'I follow b' -> d; H1 = native e, H1c = H2 -> b",
         hypothesis="H3", strict=True),
    _cfg("h3_2", "h3", "a'b'c -> c", "a'dbe -> d", "e", "'I follow b' -> e; H1 = native d, H1c = H2 -> b",
         hypothesis="H3", strict=True),
)}
ALIASES = {"order_diff": "order_diff_1", "order_matched": "order_matched_1", "c_in_dst": "diff_vs_prog_1"}
VARIANTS = tuple(CONFIGS) + tuple(ALIASES)       # accepted names (CLI choices)
BY_HYPOTHESIS = {h: [n for n, c in CONFIGS.items() if c.hypothesis == h] for h in HYPOTHESES}


def canonical(name: str) -> str:
    name = ALIASES.get(name, name)
    assert name in CONFIGS, f"unknown config {name!r}; one of {sorted(CONFIGS)} or aliases {sorted(ALIASES)}"
    return name


def resolve_configs(names) -> list[str]:
    """``None`` / ``['all']`` -> every config (all hypotheses — targets differ across them!); a hypothesis
    name (``H1`` / ``H1c`` / ``H2`` / ``H3``, case-insensitive) -> its configs; group names (order_diff, ...)
    expand only via their alias."""
    if not names or list(names) == ["all"]:
        return list(CONFIGS)
    hyp = {h.lower(): h for h in HYPOTHESES}
    out = []
    for n in names:
        for c in (BY_HYPOTHESIS[hyp[n.lower()]] if n.lower() in hyp else [canonical(n)]):
            if c not in out:
                out.append(c)
    return out


def _with_prefix(part, prefix):
    q, code = part
    return q, code + " " + prefix


@dataclass
class NRGSPrefixInstance:
    pivot: Pivot                      # b = DST native predicate, d = target predicate, shared prefix
    assign: dict                      # config letter -> predicate id, for Config.others
    config: str = "order_diff_1"
    src_lang: str = "python"
    dst_lang: str = "python"
    presentation: str = "listed"

    def __post_init__(self):
        self.config = canonical(self.config)
        for lang in (self.src_lang, self.dst_lang):
            if lang != "python":
                raise NotImplementedError("the head_loc bank has Python one-liners only; add C++ twins first")
        cfg = self.cfg
        assert set(self.assign) == set(cfg.others), f"{cfg.name}: need letters {cfg.others}, got {sorted(self.assign)}"
        ids = list(self.roles.values())
        assert len(set(ids)) == len(ids), f"{cfg.name}: tasks must be distinct: {self.roles}"

    @property
    def cfg(self) -> Config:
        return CONFIGS[self.config]

    @property
    def roles(self) -> dict:          # letter -> predicate id, all letters
        return {**self.assign, self.cfg.native: self.pivot.b, self.cfg.target: self.pivot.d}

    # backward-compatible names (order_diff_1 letters)
    @property
    def variant(self) -> str:
        return self.config

    @property
    def a(self):
        return self.roles.get("a")

    @property
    def c(self):
        return self.roles.get("c")

    @property
    def datatype(self) -> str:
        return self.pivot.datatype

    @property
    def key(self) -> str:
        r = self.roles
        return (f"{self.config}|{self.datatype}|" + "|".join(f"{x}={r[x]}" for x in sorted(r))
                + f"|prefix={self.pivot.prefix}|{self.src_lang}-{self.dst_lang}")

    @property
    def base(self) -> D.DiffTriggerInstance:
        """The SRC/DST pair as a regular diff-trigger instance (same renderer as the eval pipeline)."""
        dt, bank, cfg, r = self.datatype, BANK[self.datatype], self.cfg, self.roles
        return D.DiffTriggerInstance(
            datatype=dt, src_order=tuple(bank[r[x]] for x in cfg.src), dst_order=tuple(bank[r[x]] for x in cfg.dst),
            src_progress=[r[x] for x in cfg.src if x in cfg.src_prefill],
            dst_progress=[r[x] for x in cfg.dst if x in cfg.dst_prefill],
            diff_target=r[cfg.target], dst_native=r[cfg.native], src_target=r[cfg.src_next],
            var_name=VAR_NAME[dt], presentation=self.presentation,
            src_lang=self.src_lang, dst_lang=self.dst_lang,
            naming="letters", letter_stub="var_eq",
            meta={"variant": self.config, "config": self.config, "dst_next": r[cfg.native],
                  "diff_target": r[cfg.target], "src_target": r[cfg.src_next], "datatype": dt,
                  "prefix": self.pivot.prefix},
        )

    def parts(self) -> dict[str, tuple[str, str]]:
        """role -> (question, code prefill). Roles: src, dst, dst_star, and src_ctrl when the config has an
        SRC' control (:attr:`Config.src_ctrl_prefill`)."""
        i, cfg, r = self.base, self.cfg, self.roles
        star_pre = [r[x] for x in cfg.dst if x in cfg.dst_star_prefill]
        dstar = i._parts(i.dst_lang, i.dst_order, star_pre)
        assert dstar[1].rstrip().endswith(f"{chr(ord('a') + len(star_pre))} ="), "DST* stub must be the target's line"
        out = {
            "src": i.src_parts(),
            "dst": _with_prefix(i.dst_parts(), self.pivot.prefix),
            "dst_star": _with_prefix(dstar, self.pivot.prefix),
        }
        if cfg.src_ctrl_prefill is not None:
            out["src_ctrl"] = i._parts(i.src_lang, i.src_order, [r[x] for x in cfg.src if x in cfg.src_ctrl_prefill])
        return out

    def prompts(self, ctx) -> dict[str, str]:
        from src.diff.functional import get_prompt

        return {role: get_prompt(q, D._finalize_prefill(code), ctx=ctx)
                for role, (q, code) in self.parts().items()}


def to_record(inst: NRGSPrefixInstance) -> dict:
    """JSON-able instance — round-trips through :func:`from_record`."""
    p = inst.pivot
    return dict(key=inst.key, config=inst.config, group=inst.cfg.group, hypothesis=inst.cfg.hypothesis,
                datatype=p.datatype, assign=dict(inst.assign),
                b=p.b, d=p.d, prefix=p.prefix, b_tok=p.b_tok, d_tok=p.d_tok,
                src_lang=inst.src_lang, dst_lang=inst.dst_lang, presentation=inst.presentation)


def from_record(r: dict) -> NRGSPrefixInstance:
    """Also reads legacy (pre-registry) records: ``variant`` + ``a`` / ``c`` (order_diff_1 letters)."""
    pv = Pivot(r["datatype"], r["b"], r["d"], r["prefix"], r["b_tok"], r["d_tok"])
    if "config" in r:
        return NRGSPrefixInstance(pv, dict(r["assign"]), r["config"], r["src_lang"], r["dst_lang"],
                                  r.get("presentation", "listed"))
    return NRGSPrefixInstance(pv, {"a": r["a"], "c": r["c"]}, r["variant"], r["src_lang"], r["dst_lang"],
                              r.get("presentation", "listed"))


def load_dataset(path, configs=None, split=None) -> list[dict]:
    """Records of a mined ``dataset.json``, optionally one or more config shards and/or one split."""
    recs = json.load(open(path))["instances"]
    keep = set(resolve_configs(configs)) if configs else None
    return [x for x in recs
            if (keep is None or canonical(x.get("config", x.get("variant"))) in keep)
            and (split is None or x["split"] == split)]


def build_instances(pivots, variant=None, n_per_pivot=1, seed=0, distinct_families=True,
                    src_lang="python", dst_lang="python", configs=None) -> list[NRGSPrefixInstance]:
    """For each pivot (native b, target d, prefix) and config, draw ``n_per_pivot`` assignments of the
    config's other letters from the rest of the bank. ``variant`` (one config) is kept for old callers."""
    cfgs = [canonical(variant)] if (variant and not configs) else resolve_configs(configs)
    rng = random.Random(seed)
    out = []
    for pv in pivots:
        used = {_fam(pv.b), _fam(pv.d)}
        pool = [p for p in BANK[pv.datatype]
                if p not in (pv.b, pv.d) and (not distinct_families or _fam(p) not in used)]
        for name in cfgs:
            others = CONFIGS[name].others
            combos = [c for c in itertools.permutations(pool, len(others))
                      if not distinct_families or len({_fam(x) for x in c}) == len(c)]
            rng.shuffle(combos)
            out += [NRGSPrefixInstance(pv, dict(zip(others, c)), name, src_lang, dst_lang) for c in combos[:n_per_pivot]]
    return out


def pair_id(inst: NRGSPrefixInstance) -> tuple:
    """Unordered (native, target) pair — shared across configs and directions, so split on this."""
    return (inst.datatype, frozenset({inst.pivot.b, inst.pivot.d}))


def stratified_sample(instances, n, seed=0):
    """``n`` instances balanced across (config, datatype) strata and, within one, spread round-robin over
    the (native, target) pairs (random order within a pair) — no stratum or pair dominates a small sample."""
    import collections

    rng = random.Random(seed)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in instances:
        by[(x.config, x.datatype)][pair_id(x)].append(x)
    queues = {}
    for st, pairs in by.items():
        groups = [rng.sample(v, len(v)) for _, v in sorted(pairs.items(), key=lambda kv: sorted(kv[0][1]))]
        rng.shuffle(groups)
        q = []
        while any(groups):
            for g in groups:
                if g:
                    q.append(g.pop())
        queues[st] = q
    out, sts = [], sorted(queues)
    while len(out) < n and any(queues[k] for k in sts):
        for k in sts:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop(0))
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
    from src.goals.pivot_token_filter import (
        TOKEN_FILTER_VERSION, tokens_differ_meaningfully,
    )

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
            # No generated full reference is available in this next-token-only
            # gate: normalize the known input alias, preserve unknown local names.
            diverge = b_id != d_id and tokens_differ_meaningfully(
                tb, td, prefix=x.pivot.prefix,
                variable_names=(VAR_NAME[x.datatype],),
            )
            rows.append(dict(
                key=x.key, config=x.config, datatype=x.datatype, b=x.pivot.b, d=x.pivot.d, prefix=x.pivot.prefix,
                b_id=b_id, d_id=d_id, top1_dst=tb, top1_dst_star=td,
                p_dst_b=float(ld[b_id].exp()), p_dst_d=float(ld[d_id].exp()),
                p_star_b=float(ls[b_id].exp()), p_star_d=float(ls[d_id].exp()),
                match_b=tb == x.pivot.b_tok, match_d=td == x.pivot.d_tok,
                token_filter_version=TOKEN_FILTER_VERSION,
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
