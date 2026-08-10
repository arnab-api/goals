"""Dependency-carrying variants of the `check_conditions` corpus.

`condition_data` / `multiline_prompts` build steps that are mutually **independent**: any order is
correct, so the only thing that can govern the order is the prompt. These graphs add real
dependencies between steps, which lets us ask the two questions the independent corpus cannot:

  * behavioural — with a dependency present, does the LM still follow the prompt's listing order,
    or does it reorder to satisfy the graph?
  * causal — does the move-on signal still advance when the next step's dependencies are UNMET?

Design decisions, and why:

**Every edge is referential.** A step's NL text names the producing variable (`the third smallest
element of divs`), so a violation is a `NameError` and is detectable by parsing rather than by
judgement. The tempting alternative — "a = the third smallest divisor of n" alongside "b = the list
of all divisors" — is *not* a dependency at all: `a` is fully specified in terms of `n`, nothing
obliges the model to use `b`, and an inline implementation is perfectly correct. That would measure
whether the model chooses to *reuse* an intermediate, which is a different question.

**One value-producing step, boolean consumers.** The surface form is kept identical to the
multiline corpus — same stub, one variable per step, single aggregating print, same instruction —
so the move-on vector stays in distribution and a null result in the causal test is attributable to
the dependency rather than to a change of domain.

**Templates, not fixed strings.** Each graph carries parameter slots, so ~10 graphs expand into
hundreds of instances; a fixed-string corpus would cap the causal test at 10.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DepStep:
    """One step of a dependency graph.

    `text`, `single` and `multi` are `str.format` templates over the graph's params. `requires`
    lists the variables this step *references* — the edges of the DAG, and the thing a violation
    check looks at. `multi` (optional) is the multi-line implementation; only steps that have one
    can serve as the forced-prefill pivot, since the pivot is an incomplete first line.
    """

    var: str
    text: str
    requires: tuple[str, ...] = ()
    single: str = ""
    multi: tuple[str, ...] = ()

    def render_text(self, p: dict) -> str:
        return self.text.format(**p)

    def render_single(self, p: dict) -> str:
        return f"    {self.var} = {self.single.format(**p)}"

    def render_multi(self, p: dict) -> list[str]:
        assert self.multi, f"step {self.var!r} has no multi-line form"
        return [ln.format(**p) for ln in self.multi]


@dataclass(frozen=True)
class DepGraph:
    """A named set of steps plus a parameter sampler.

    `steps` are given in *canonical* (dependency-respecting) order; the prompt's listing order is
    chosen separately, which is exactly the manipulation we care about.
    """

    name: str
    steps: tuple[DepStep, ...]
    params: dict = field(default_factory=dict)

    def sample_params(self, rng: random.Random) -> dict:
        return {k: (v(rng) if callable(v) else v) for k, v in self.params.items()}

    @property
    def vars(self) -> list[str]:
        return [s.var for s in self.steps]

    def edges(self) -> list[tuple[str, str]]:
        """(consumer, producer) pairs — consumer must be written after producer."""
        return [(s.var, r) for s in self.steps for r in s.requires]

    def producers(self) -> list[str]:
        return sorted({r for s in self.steps for r in s.requires})

    def pivotable(self) -> list[str]:
        return [s.var for s in self.steps if s.multi]

    def step(self, var: str) -> DepStep:
        return next(s for s in self.steps if s.var == var)


# ── the resource steps ────────────────────────────────────────────────────────
# Each produces a value the boolean steps then reference. Multi-line forms exist so a resource can
# also be the pivot (its first line is an incomplete assignment).
_DIVS = DepStep(
    var="divs",
    text="the list of all divisors of n, in ascending order",
    single="[d for d in range(1, abs(n) + 1) if n % d == 0]",
    multi=("    divs = []",
           "    for d in range(1, abs(n) + 1):",
           "        if n % d == 0:",
           "            divs.append(d)"),
)
_DIGITS = DepStep(
    var="digits",
    text="the list of n's decimal digits, most significant first",
    single="[int(c) for c in str(abs(n))]",
    multi=("    digits = []",
           "    for c in str(abs(n)):",
           "        digits.append(int(c))"),
)
_BITS = DepStep(
    var="bits",
    text="the list of n's binary digits, as integers",
    single="[int(c) for c in bin(abs(n))[2:]]",
    multi=("    bits = []",
           "    for c in bin(abs(n))[2:]:",
           "        bits.append(int(c))"),
)


# Chain heads with a multi-line form: these can serve as the PIVOT, which the met/unmet pair in
# the causal test needs (the pivot must be a step that only *some* later steps depend on).
_TOTAL = DepStep(
    var="total", text="the sum of the elements of divs", requires=("divs",), single="sum(divs)",
    multi=("    total = 0", "    for d in divs:", "        total += d"),
)
_DSUM = DepStep(
    var="dsum", text="the sum of the elements of digits", requires=("digits",),
    single="sum(digits)",
    multi=("    dsum = 0", "    for d in digits:", "        dsum += d"),
)


def _rint(lo, hi):
    return lambda rng: rng.randint(lo, hi)


# ── graphs ────────────────────────────────────────────────────────────────────
# Shapes: fork (one producer, independent consumers), chain (producer -> consumer -> consumer),
# diamond (fork + join), and two-producer. The join steps are what make an ordering *forced*
# beyond the producer.
GRAPHS: list[DepGraph] = [
    DepGraph(
        "divs_fork",
        (_DIVS,
         DepStep("many", "divs has more than {k} elements", ("divs",), "len(divs) > {k}"),
         DepStep("big_last", "the last element of divs is greater than {m}", ("divs",),
                 "divs[-1] > {m}"),
         DepStep("even_count", "the number of elements in divs is even", ("divs",),
                 "len(divs) % 2 == 0")),
        {"k": _rint(2, 6), "m": _rint(5, 40)},
    ),
    DepGraph(
        "divs_chain",
        (_DIVS,
         _TOTAL,
         DepStep("abundant", "total is greater than twice n", ("total",), "total > 2 * n"),
         DepStep("total_big", "total is greater than {m}", ("total",), "total > {m}")),
        {"m": _rint(10, 60)},
    ),
    DepGraph(
        "divs_diamond",
        (_DIVS,
         DepStep("many", "divs has more than {k} elements", ("divs",), "len(divs) > {k}"),
         DepStep("big_last", "the last element of divs is greater than {m}", ("divs",),
                 "divs[-1] > {m}"),
         DepStep("both", "many and big_last are both true", ("many", "big_last"),
                 "many and big_last")),
        {"k": _rint(2, 6), "m": _rint(5, 40)},
    ),
    DepGraph(
        "digits_fork",
        (_DIGITS,
         DepStep("all_even", "every element of digits is even", ("digits",),
                 "all(d % 2 == 0 for d in digits)"),
         DepStep("dsum_div", "the sum of digits is divisible by {k}", ("digits",),
                 "sum(digits) % {k} == 0"),
         DepStep("has_zero", "digits contains a zero", ("digits",), "0 in digits")),
        {"k": _rint(2, 9)},
    ),
    DepGraph(
        "digits_chain",
        (_DIGITS,
         _DSUM,
         DepStep("dsum_big", "dsum is greater than {m}", ("dsum",), "dsum > {m}"),
         DepStep("dsum_div", "dsum is divisible by {k}", ("dsum",), "dsum % {k} == 0")),
        {"m": _rint(5, 25), "k": _rint(2, 9)},
    ),
    DepGraph(
        "digits_diamond",
        (_DIGITS,
         DepStep("dsum", "the sum of the elements of digits", ("digits",), "sum(digits)"),
         DepStep("dmax", "the largest element of digits", ("digits",), "max(digits)"),
         DepStep("spread", "dsum is more than {k} times dmax", ("dsum", "dmax"),
                 "dsum > {k} * dmax")),
        {"k": _rint(2, 4)},
    ),
    DepGraph(
        "bits_fork",
        (_BITS,
         DepStep("ones", "the number of ones in bits is greater than {k}", ("bits",),
                 "sum(bits) > {k}"),
         DepStep("bit_len", "bits has more than {m} elements", ("bits",), "len(bits) > {m}"),
         DepStep("ends_one", "the last element of bits is 1", ("bits",), "bits[-1] == 1")),
        {"k": _rint(1, 5), "m": _rint(2, 8)},
    ),
    DepGraph(
        "bits_chain",
        (_BITS,
         DepStep("ones", "the number of ones in bits", ("bits",), "sum(bits)"),
         DepStep("ones_gt", "ones is greater than {k}", ("ones",), "ones > {k}"),
         DepStep("ones_even", "ones is even", ("ones",), "ones % 2 == 0")),
        {"k": _rint(1, 5)},
    ),
    DepGraph(
        "two_producers",
        (_DIVS, _DIGITS,
         DepStep("many", "divs has more than {k} elements", ("divs",), "len(divs) > {k}"),
         DepStep("all_even", "every element of digits is even", ("digits",),
                 "all(d % 2 == 0 for d in digits)"),
         DepStep("both", "many and all_even are both true", ("many", "all_even"),
                 "many and all_even")),
        {"k": _rint(2, 6)},
    ),
    DepGraph(
        "deep_chain",
        (_DIGITS,
         DepStep("dsum", "the sum of the elements of digits", ("digits",), "sum(digits)"),
         DepStep("dsum_sq", "dsum multiplied by itself", ("dsum",), "dsum * dsum"),
         DepStep("sq_big", "dsum_sq is greater than {m}", ("dsum_sq",), "dsum_sq > {m}")),
        {"m": _rint(20, 200)},
    ),
]

GRAPHS.append(DepGraph(
    # The met/unmet workhorse: pivot mid-`total`, and then
    #   `many`     requires divs  -> already complete  -> MET
    #   `abundant` requires total -> being written now -> UNMET
    # Same prefill and same pivot; only which step the listing puts next changes.
    "mixed_dep",
    (_DIVS,
     _TOTAL,
     DepStep("many", "divs has more than {k} elements", ("divs",), "len(divs) > {k}"),
     DepStep("abundant", "total is greater than twice n", ("total",), "total > 2 * n")),
    {"k": _rint(2, 6)},
))

GRAPHS_BY_NAME = {g.name: g for g in GRAPHS}


# ── ordering utilities ────────────────────────────────────────────────────────
def is_valid_order(graph: DepGraph, order: list[str]) -> bool:
    """True iff every consumer appears after all of its producers."""
    pos = {v: i for i, v in enumerate(order)}
    return all(pos.get(c, -1) > pos.get(p, len(order)) for c, p in graph.edges()
               if c in pos and p in pos)


def violations(graph: DepGraph, order: list[str]) -> list[tuple[str, str]]:
    """Edges the order breaks — (consumer, producer) written in the wrong relative order."""
    pos = {v: i for i, v in enumerate(order)}
    return [(c, p) for c, p in graph.edges()
            if c in pos and p in pos and pos[c] < pos[p]]


def constrained_pairs(graph: DepGraph) -> set[tuple[str, str]]:
    """Ordered pairs (u, v) with u before v in EVERY valid order — the transitive closure."""
    reach: dict[str, set[str]] = {v: set() for v in graph.vars}
    for c, p in graph.edges():
        reach[p].add(c)
    changed = True
    while changed:                                     # transitive closure
        changed = False
        for v in graph.vars:
            add = {y for x in reach[v] for y in reach[x]} - reach[v]
            if add:
                reach[v] |= add
                changed = True
    return {(p, c) for p in graph.vars for c in reach[p]}


def free_pairs(graph: DepGraph) -> set[frozenset]:
    """Unordered pairs the DAG does NOT constrain — where prompt order is free to govern."""
    con = constrained_pairs(graph)
    out = set()
    for i, u in enumerate(graph.vars):
        for v in graph.vars[i + 1:]:
            if (u, v) not in con and (v, u) not in con:
                out.add(frozenset((u, v)))
    return out


def topological_orders(graph: DepGraph, limit: int = 64) -> list[list[str]]:
    """Up to `limit` valid orders (for checking how much freedom a graph actually leaves)."""
    out: list[list[str]] = []
    need = {s.var: set(s.requires) for s in graph.steps}

    def rec(done: list[str]):
        if len(out) >= limit:
            return
        if len(done) == len(graph.vars):
            out.append(list(done))
            return
        for v in graph.vars:
            if v not in done and need[v] <= set(done):
                rec(done + [v])

    rec([])
    return out
