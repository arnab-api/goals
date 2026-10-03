"""Multi-hierarchy kv-transfer data: the kv-(task-)transfer setup one level DOWN the goal hierarchy.

A frontier sub-goal is a filter over a list whose condition is a 2-way conjunction (``and``) or
disjunction (``or``) of SUB-SUB-GOALS drawn from the filter-condition bank (:data:`D.FILTER_TASKS`).
The shared first condition ``x`` is prefilled up to the connective, so the pivot is the ``and`` / ``or``
token and the model is about to write the second sub-sub-goal::

    p_dst      : a'(x'y) -> y      (DST spec: "... that <x> and <y>")
    p_src_match: a'(x'z) -> z      (SRC spec: "... that <x> and <z>")      kv-transfer target: z
    p_src_diff : a'(zx') -> z      (SRC spec: "... that <z> and <x>", code still writes x first)

``a`` is one fully-prefilled sub-goal from the datatype's q-bank (:data:`D.QTASKS`), chosen so it does not
overlap x / y / z (:data:`A_OVERLAP`). The intervention is the unchanged kv-transfer
(:func:`src.diff.functional.run_kv_transfer_intervention`: SRC task-list k+v -> DST) and the eval the
unchanged :func:`src.diff.eval.judge_causal` / :func:`src.diff.eval.eval_readouts`; :func:`rule_verdict`
adds a cheap execution-based check of the first condition the model writes.

Any side may be C++ (py->cpp, cpp->py, cpp->cpp): a C++ side's frontier is a range-for filter loop
``std::vector<T> b;  for (const auto& w : words) if (<x_cpp> &&`` (pivot ``" &&"`` / ``" ||"``). Emptiness
guards (``!w.empty() && …`` in the C++ bank) of any of x / y / z are PREFILLED before x, so the model only
writes the bare test; a compound x (``a && b`` ranges) is parenthesised where precedence needs it; the two
``[&]{…}()`` lambda conditions are excluded (:data:`CPP_EXCLUDE`).

This module only READS the existing banks — it never edits them (existing results depend on them).
"""

import random
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import src.diff.data as D

Connective = Literal["and", "or"]
Lang = Literal["python", "cpp"]
CPP_OP = {"and": "&&", "or": "||"}
CPP_ELEM_TYPE = {"list[str]": "std::string", "list[int]": "int"}
CPP_EXCLUDE = {"perfect_square", "is_prime"}   # [&]{ ... }() lambdas — not a clean single condition
_GUARD = "!{elem}.empty() && "


def cpp_split(cond: D.Predicate) -> tuple[bool, str]:
    """(has_emptiness_guard, bare C++ test template over {elem})."""
    t = cond.impl["cpp"]
    return (True, t[len(_GUARD):]) if t.startswith(_GUARD) else (False, t)
Order = Literal["match", "diff"]
HIER_DATATYPES = ("list[str]", "list[int]")

# q-bank sub-goal `a` -> filter conditions it overlaps with (so `a` never restates x / y / z)
A_OVERLAP: dict[str, dict[str, set[str]]] = {
    "list[str]": {
        "has_palindrome": {"palindrome", "same_reversed"},
        "total_length": {"long", "short", "length_five"},
        "count_long": {"long", "short", "length_five"},
        "count_capitalized": {"capitalized", "all_upper", "all_lower"},
        "count_starts_vowel": {"vowel_start", "starts_a"},
        "count_with_digit": {"contains_digit", "last_is_digit", "all_alpha"},
        "count_all_lowercase": {"all_lower", "all_upper", "capitalized"},
        "count_short": {"short", "long", "length_five"},
        "longest_word_length": {"long", "short", "length_five"},
        "shortest_word_length": {"long", "short", "length_five"},
        "total_vowels": {"vowel_start", "contains_e", "starts_a"},
        "all_unique_lengths": {"long", "short", "length_five"},
        "count_ends_consonant": {"ends_s", "last_is_digit"},
        "total_uppercase_letters": {"all_upper", "capitalized", "all_lower"},
    },
    "list[int]": {
        "has_even": {"even", "odd", "divisible_by_four", "multiple_of_ten"},
        "count_even": {"even", "odd", "divisible_by_four", "multiple_of_ten"},
        "count_odd": {"even", "odd", "divisible_by_four", "multiple_of_ten"},
        "count_positive": {"positive", "negative", "non_negative"},
        "count_negative": {"positive", "negative", "non_negative"},
        "all_positive": {"positive", "negative", "non_negative"},
        "contains_zero": {"positive", "negative", "non_negative"},
    },
}

# execution pools for :func:`rule_verdict` — every bank condition is true on some and false on others
_WORD_POOL = [
    "level", "noon", "aba", "racecar", "Anna", "kayak", "apple", "Apple", "APPLE", "banana", "cats",
    "dogs", "tree", "bee", "book", "moon", "a1b", "x9", "42", "abc7", "HELLO", "WORLD", "Hello",
    "World", "egg", "ice", "orange", "umbrella", "sky", "fly", "hymns", "zz", "aa", "abcde", "quiet",
    "seven", "eagle", "Echo", "is", "as", "bus", "glass", "ab", "x", "I", "OK", "ok", "e", "ss",
    "abba", "tenet", "refer", "stats", "lemon", "melon", "grape", "plum", "kiwi", "mango", "peach",
]
_INT_POOL = list(range(-120, 121))
POOLS = {"list[str]": _WORD_POOL, "list[int]": _INT_POOL}


@dataclass(frozen=True)
class HierKVInstance(D.MultiPredicateInstance):
    """A :class:`D.MultiPredicateInstance` (shared = (a,), letters naming, ``var_eq`` stub) whose
    frontier prefill continues past ``b =`` with the filter comprehension up to the connective.
    ``src_frontier`` / ``dst_frontier`` are the rendered filter sub-goals; the second conditions (the
    sub-sub-goals being contrasted) are ``src_cond`` (z, the transplant target) and ``dst_cond`` (y)."""

    x: Optional[D.Predicate] = None          # shared first condition (prefilled)
    src_cond: Optional[D.Predicate] = None   # z — SRC's next sub-sub-goal (transplant target)
    dst_cond: Optional[D.Predicate] = None   # y — clean DST's next sub-sub-goal
    connective: Connective = "and"
    order: Order = "match"                   # "diff": SRC's spec lists z BEFORE x
    elem: str = "w"
    noun: str = "words"

    def _comprehension_prefix(self, lang) -> str:
        """The frontier line(s) after ``b`` up to the connective: python ``= [w for w in … if <x> and`` |
        c++ ``;`` + the range-for loop ``for (const auto& w : words) if (<guard &&> <x> &&``."""
        e, arg = self.elem, self.var_name
        if lang == "python":
            x = D._fmt(self.x.impl["python"], arg=arg, elem=e)
            return f"[{e} for {e} in {arg} if {x} {self.connective}"
        guarded = any(cpp_split(c)[0] for c in (self.x, self.src_cond, self.dst_cond))
        x = D._fmt(cpp_split(self.x)[1], arg=arg, elem=e)
        if "&&" in x or "||" in x:
            x = f"({x})"                                          # compound x (ranges): keep it one operand
        guard = D._fmt(_GUARD, arg=arg, elem=e) if guarded else ""
        if self.connective == "or" and guard:
            return f"for (const auto& {e} : {arg}) if ({guard}({x} {CPP_OP['or']}"   # guard && (x || …
        return f"for (const auto& {e} : {arg}) if ({guard}{x} {CPP_OP[self.connective]}"

    def _parts(self, lang, frontier):
        q, pf = super()._parts(lang, frontier)  # python: ends at "    b =" ; c++: ends at "    auto b ="
        if lang == "python":
            return q, pf + " " + self._comprehension_prefix(lang)
        stub = "\n    auto b ="
        assert pf.endswith(stub), pf[-40:]
        return q, (pf[: -len(stub)] + f"\n    std::vector<{CPP_ELEM_TYPE[self.datatype]}> b;\n    "
                   + self._comprehension_prefix(lang))

    @property
    def expected_next_desc(self) -> str:
        return f"keep only the {self.noun} that {self.src_cond.description}"

    @property
    def dst_next_desc(self) -> str:
        return f"keep only the {self.noun} that {self.dst_cond.description}"

    @property
    def task_desc(self) -> str:
        so_far = (f"b = {self._comprehension_prefix('python')}" if self.dst_lang == "python"
                  else self._comprehension_prefix("cpp"))
        return (f"{self.shared[0].description}; then the list of {self.noun} in `{self.var_name}` filtered "
                f"by two conditions — the code so far is `{so_far}` "
                f"(the first condition, that they {self.x.description}, is already written)")


def _frontier(ft: D.FilterTask, arg: str, c1: D.Predicate, c2: D.Predicate, conn: str) -> D.Predicate:
    desc = f"the list of {ft.noun} in `{arg}` that {c1.description} {conn} {c2.description}"
    return D.Predicate(id=f"{c1.id}-{conn}-{c2.id}", description=desc, result_name="b",
                       impl={"python": "", "cpp": ""})


def _filter(cond_ids, conn, dt, conds) -> list:
    """Reference result of ``[e for e in POOL if c1 <conn> c2]`` on the datatype's pool."""
    fns = [eval(f"lambda e: {D._fmt(conds[c].impl['python'], arg='', elem='e')}") for c in cond_ids]
    op = all if conn == "and" else any
    return [e for e in POOLS[dt] if op(f(e) for f in fns)]


def hier_kv_setups(
    datatype: str,
    n_samples: int = 32,
    seed: int = 0,
    connective: Connective = "and",
    order: Order = "match",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
) -> list[HierKVInstance]:
    """Sample ``n_samples`` (a, x, y, z) instances. x, y, z are distinct, pairwise compatible filter
    conditions (:func:`D.filter_compatible`) with ``x <conn> y`` and ``x <conn> z`` giving different
    results on the execution pool (so the contrast is observable); ``a`` is a q-bank sub-goal that
    overlaps none of them. Deterministic in ``seed``; a larger ``n_samples`` extends the same prefix."""
    assert datatype in HIER_DATATYPES, f"{datatype} not supported (filter bank: {HIER_DATATYPES})"
    ft = D.FILTER_TASKS[datatype]
    conds = {c.id: c for c in ft.conditions}
    a_bank = list(D.QTASKS[datatype].predicates)
    arg = ft.var_names[0]
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 500):
        if len(out) >= n_samples:
            break
        x, y, z = rng.sample(list(conds), 3)
        if "cpp" in (src_lang, dst_lang) and {x, y, z} & CPP_EXCLUDE:
            continue
        if not all(D.filter_compatible(p, q) for p, q in ((x, y), (x, z), (y, z))):
            continue
        if (x, y, z) in seen:
            continue
        ref_y, ref_z = _filter((x, y), connective, datatype, conds), _filter((x, z), connective, datatype, conds)
        if ref_y == ref_z or not ref_y or not ref_z:
            continue
        a_ok = [p for p in a_bank if not (A_OVERLAP[datatype].get(p.id, set()) & {x, y, z})]
        a = rng.choice(a_ok)
        seen.add((x, y, z))
        src_spec = (conds[z], conds[x]) if order == "diff" else (conds[x], conds[z])
        out.append(HierKVInstance(
            datatype=datatype,
            shared=(a,),
            src_frontier=_frontier(ft, arg, *src_spec, connective),
            dst_frontier=_frontier(ft, arg, conds[x], conds[y], connective),
            var_name=arg,
            presentation="listed",
            src_lang=src_lang,
            dst_lang=dst_lang,
            naming="letters",
            letter_stub="var_eq",
            x=conds[x], src_cond=conds[z], dst_cond=conds[y],
            connective=connective, order=order, elem=ft.elem, noun=ft.noun,
            meta=dict(datatype=datatype, a=a.id, x=x, y=y, z=z, connective=connective, order=order,
                      src_lang=src_lang, dst_lang=dst_lang),
        ))
    return out


def inst_key(inst: HierKVInstance) -> str:
    m = inst.meta
    lang = "" if m.get("dst_lang", "python") == "python" else f"|{m['src_lang']}-{m['dst_lang']}"   # py-py keys unchanged
    return f"hier|{m['datatype']}|{m['connective']}|{m['order']}|a={m['a']}|x={m['x']}|y={m['y']}|z={m['z']}{lang}"


# ══════════════════════════════════════════════════════════════════════════════
# cheap rule-based causal verdict (execution) — to check agreement with the oracle
# ══════════════════════════════════════════════════════════════════════════════
def rule_verdict(inst: HierKVInstance, gen: str) -> dict:
    """Complete the prefilled comprehension with the FIRST line of ``gen`` and execute it on the pool.
    ``src`` = matches ``x <conn> z`` (transplant target), ``dst`` = matches ``x <conn> y`` (native),
    ``x_only`` = the second condition is vacuous, ``neither`` = runs but matches none, ``error`` = does
    not parse/run. Restricted to the comprehension: anything after its closing ``]`` is dropped.
    Python DST only — a C++ DST returns ``label="na"`` (oracle only)."""
    if inst.dst_lang != "python":
        return dict(label="na", code="", why="c++ destination: rule check not implemented")
    conds = {c.id: c for c in D.FILTER_TASKS[inst.datatype].conditions}
    m = inst.meta
    line = gen.split("\n")[0]
    expr = f"[{inst.elem} for {inst.elem} in POOL if {D._fmt(inst.x.impl['python'], arg='', elem=inst.elem)} {inst.connective}{line}"
    depth, cut = 0, None                                  # cut at the bracket closing the comprehension
    for i, ch in enumerate(expr):
        depth += ch in "[({"
        depth -= ch in "])}"
        if depth == 0:
            cut = i + 1
            break
    if cut is None:
        return dict(label="error", code=expr, why="unclosed comprehension")
    expr = expr[:cut]
    try:
        got = eval(expr, {"__builtins__": __builtins__, "POOL": POOLS[inst.datatype]}, {})
    except Exception as e:  # noqa: BLE001 — any failure of model-written code is an "error" verdict
        return dict(label="error", code=expr, why=f"{type(e).__name__}: {e}")
    refs = {
        "src": _filter((m["x"], m["z"]), inst.connective, inst.datatype, conds),
        "dst": _filter((m["x"], m["y"]), inst.connective, inst.datatype, conds),
        "x_only": _filter((m["x"],), inst.connective, inst.datatype, conds),
    }
    label = next((k for k, v in refs.items() if got == v), "neither")
    return dict(label=label, code=expr, why="")
