"""Controllable "check-N-against-independent-conditions" dataset for the clock /
instrumental-goal probing experiments.

A sample asks a code-LM to write a function `check_conditions(n)` that checks an
ordered list of *mutually independent* integer predicates and prints a
pass/fail message after each one, in order. Unlike the file-handling clock data,
the predicates have no data dependencies, so each "check predicate P" step is an
independent instrumental goal — ideal for probing.

Why this design (see brainstorm with A.S.):
  * Controllable: vary datatype, predicate identity/params, count, order, lang,
    modularity → arbitrarily large dataset.
  * Independent predicates → clean step boundaries, swappable for minimal pairs.
  * Constant structural landmark tokens (`def`/`if`/`print`) once per step → the
    "which predicate" signal must live in the hidden state, not the token id
    (kills the token-feature confound that plagued the line-end probes).
  * Same (predicate-set, order) rendered across {modular, monolithic} and across
    languages → clean implementation-invariance test.

N is *symbolic*: `n` is the function argument, so the LM must implement each
predicate's logic; pass/fail is a runtime property (impossible condition sets are
fine — the checker just prints "failed" at runtime). Ground-truth oracles here
are only for building expected stdout / unit tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from dataclasses_json import DataClassJsonMixin

Lang = Literal["python", "cpp", "java", "javascript"]
Modularity = Literal["modular", "monolithic"]


# ──────────────────────────────────────────────────────────────────────────────
# Per-language config. To add a language: change the opener word, the entry
# signature stub, the comment token, and the boolean type — nothing else.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LangSpec:
    display: str          # used in the instruction opener ("Write a {display} program…")
    fn_name: str          # entry function name as it appears in code
    mono_stub: str        # prefill stub for MONOLITHIC (opens the entry fn; model fills body)
    comment: str          # line-comment token (only used if comments are requested)
    bool_type: str        # boolean return type word, for the modular instruction


LANGS: dict[Lang, LangSpec] = {
    "python": LangSpec(
        display="Python",
        fn_name="check_conditions",
        mono_stub="def check_conditions(n: int):\n",
        comment="#",
        bool_type="bool",
    ),
    "cpp": LangSpec(
        display="C++",
        fn_name="check_conditions",
        mono_stub="void check_conditions(int n) {\n",
        comment="//",
        bool_type="bool",
    ),
    "java": LangSpec(
        display="Java",
        fn_name="checkConditions",
        # Java needs a class wrapper; open class + method for monolithic.
        mono_stub="public class Solution {\n    public static void checkConditions(int n) {\n",
        comment="//",
        bool_type="boolean",
    ),
    "javascript": LangSpec(
        display="JavaScript",
        fn_name="checkConditions",
        mono_stub="function checkConditions(n) {\n",
        comment="//",
        bool_type="boolean",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Predicate bank (int). Oracles live here (not serialized); a sample stores only
# predicate id + params + rendered spec text, and looks the oracle up by id.
# ──────────────────────────────────────────────────────────────────────────────


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _digit_product(n: int) -> int:
    p = 1
    for c in str(abs(n)):
        p *= int(c)
    return p


def _is_perfect_cube(n: int) -> bool:
    c = round(abs(n) ** (1 / 3))
    return any((c + d) ** 3 == abs(n) for d in (-1, 0, 1))


def _is_triangular(n: int) -> bool:
    if n < 0:
        return False
    m = int((2 * n) ** 0.5)
    return any(mm >= 0 and (mm * (mm + 1)) // 2 == n for mm in (m - 1, m, m + 1))


@dataclass(frozen=True)
class Predicate:
    id: str
    spec_template: str                 # NL spec, e.g. "n is divisible by {k}"
    oracle: Callable[..., bool]        # oracle(n, **params) -> bool (ground truth)
    param_space: dict[str, list] = field(default_factory=dict)
    # Most-informative substrings of spec_template (may contain {param} placeholders);
    # used to locate the spec tokens to track attention back to.
    # e.g. "n is a perfect square" -> ["perfect", "square"].
    id_substrs: list[str] = field(default_factory=list)
    # Forced-prefill body: monolithic-Python lines (own indentation) implementing the
    # predicate as an explicit multi-line block.  EVERY assigned variable — intermediates
    # included — is `{var}` (reused across lines), so substituting the condition's result
    # letter (a/b/c/…) renames the whole block in one shot.  `{param}` placeholders are
    # filled from sampled params.  Empty ⇒ predicate has no canonical multi-line form.
    multiline_impl: list[str] = field(default_factory=list)
    # Single-line forced-prefill form: ONE complete `{var} = <boolean>` line that fully checks
    # the condition (no intermediate, nothing left to finish). Same `{var}`/`{param}` rules.
    singleline_impl: str = ""

    @property
    def is_multiline(self) -> bool:
        return bool(self.multiline_impl)

    def sample_params(self, rng: random.Random) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, choices in self.param_space.items():
            out[key] = rng.choice(choices)
        # in_range needs a <= b
        if {"a", "b"} <= out.keys() and out["a"] > out["b"]:
            out["a"], out["b"] = out["b"], out["a"]
        return out

    def render(self, params: dict[str, Any]) -> str:
        return self.spec_template.format(**params)

    def render_substrs(self, params: dict[str, Any]) -> list[str]:
        """Render the informative substrings with params (e.g. '{k}' -> '18')."""
        return [s.format(**params) for s in self.id_substrs]

    def render_multiline(self, params: dict[str, Any], var: str) -> list[str]:
        """Forced-prefill body lines with `{var}` -> the condition's result letter and
        `{param}` placeholders filled in.  `var` is substituted first (it is not a
        param), then params are formatted."""
        return [ln.replace("{var}", var).format(**params) for ln in self.multiline_impl]

    def render_singleline(self, params: dict[str, Any], var: str) -> str:
        """Single-line forced-prefill: `{var}` -> the result letter, `{param}` filled in."""
        return self.singleline_impl.replace("{var}", var).format(**params)

    def evaluate(self, n: int, params: dict[str, Any]) -> bool:
        return bool(self.oracle(n, **params))


INT_PREDICATES: list[Predicate] = [
    # ── trivially single-line (no canonical multi-line form) ──
    Predicate("positive", "n is positive", lambda n: n > 0, id_substrs=["positive"],
              singleline_impl="    {var} = n > 0"),
    Predicate("le_bound", "n is less than or equal to {b}", lambda n, b: n <= b,
              {"b": [50, 100, 200, 500, 1000]}, id_substrs=["less than or equal", "{b}"],
              singleline_impl="    {var} = n <= {b}"),
    Predicate("ge_bound", "n is greater than or equal to {a}", lambda n, a: n >= a,
              {"a": [0, 1, 10, 50, 100]}, id_substrs=["greater than or equal", "{a}"],
              singleline_impl="    {var} = n >= {a}"),
    Predicate("in_range", "n is between {a} and {b} (inclusive)",
              lambda n, a, b: a <= n <= b, {"a": [0, 10, 50], "b": [100, 200, 500]},
              id_substrs=["between", "{a}", "{b}"],
              singleline_impl="    {var} = {a} <= n <= {b}"),

    # ── "compute remainder/quantity, then compare" (forced 2-line) ──
    Predicate("even", "n is even", lambda n: n % 2 == 0, id_substrs=["even"],
              multiline_impl=["    {var} = n % 2", "    {var} = {var} == 0"],
              singleline_impl="    {var} = n % 2 == 0"),
    Predicate("odd", "n is odd", lambda n: n % 2 == 1, id_substrs=["odd"],
              multiline_impl=["    {var} = n % 2", "    {var} = {var} == 1"],
              singleline_impl="    {var} = n % 2 == 1"),
    Predicate("divisible", "n is divisible by {k}", lambda n, k: k != 0 and n % k == 0,
              {"k": list(range(2, 21))}, id_substrs=["divisible", "{k}"],
              multiline_impl=["    {var} = n % {k}", "    {var} = {var} == 0"],
              singleline_impl="    {var} = n % {k} == 0"),
    Predicate("perfect_square", "n is a perfect square",
              lambda n: n >= 0 and math.isqrt(n) ** 2 == n, id_substrs=["perfect", "square"],
              multiline_impl=["    {var} = int(abs(n) ** 0.5)",
                              "    {var} = n >= 0 and {var} * {var} == n"],
              singleline_impl="    {var} = n >= 0 and int(abs(n) ** 0.5) ** 2 == n"),
    Predicate("perfect_cube", "n is a perfect cube", lambda n: _is_perfect_cube(n),
              id_substrs=["perfect", "cube"],
              multiline_impl=["    {var} = round(abs(n) ** (1 / 3))",
                              "    {var} = {var} ** 3 == abs(n)"],
              singleline_impl="    {var} = round(abs(n) ** (1 / 3)) ** 3 == abs(n)"),
    Predicate("triangular", "n is a triangular number", lambda n: _is_triangular(n),
              id_substrs=["triangular"],
              multiline_impl=["    {var} = int((2 * abs(n)) ** 0.5)",
                              "    {var} = n >= 0 and {var} * ({var} + 1) // 2 == n"],
              singleline_impl="    {var} = n >= 0 and int((8 * n + 1) ** 0.5) ** 2 == 8 * n + 1"),
    Predicate("palindrome", "n is a palindrome (its digits read the same forwards and backwards)",
              lambda n: str(abs(n)) == str(abs(n))[::-1], id_substrs=["palindrome"],
              multiline_impl=["    {var} = str(abs(n))",
                              "    {var} = {var} == {var}[::-1]"],
              singleline_impl="    {var} = str(abs(n)) == str(abs(n))[::-1]"),
    Predicate("reversed_greater", "n is smaller than the number obtained by reversing its digits",
              lambda n: int(str(abs(n))[::-1]) > abs(n), id_substrs=["reversing", "digits"],
              multiline_impl=["    {var} = int(str(abs(n))[::-1])",
                              "    {var} = {var} > abs(n)"],
              singleline_impl="    {var} = int(str(abs(n))[::-1]) > abs(n)"),
    Predicate("num_digits_ge", "n has at least {k} digits",
              lambda n, k: len(str(abs(n))) >= k, {"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "digits"],
              multiline_impl=["    {var} = len(str(abs(n)))",
                              "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = len(str(abs(n))) >= {k}"),
    Predicate("set_bits_gt", "n has more than {t} ones in its binary representation",
              lambda n, t: bin(abs(n)).count("1") > t, {"t": [1, 2, 3, 4]},
              id_substrs=["more than", "{t}", "ones", "binary"],
              multiline_impl=['    {var} = bin(abs(n)).count("1")',
                              "    {var} = {var} > {t}"],
              singleline_impl='    {var} = bin(abs(n)).count("1") > {t}'),
    Predicate("power_of_two", "n is a power of two",
              lambda n: n > 0 and (n & (n - 1)) == 0, id_substrs=["power", "two"],
              multiline_impl=['    {var} = bin(n).count("1")',
                              "    {var} = n > 0 and {var} == 1"],
              singleline_impl="    {var} = n > 0 and (n & (n - 1)) == 0"),
    Predicate("coprime_with", "n is coprime with {k}", lambda n, k: math.gcd(n, k) == 1,
              {"k": [6, 10, 12, 15]}, id_substrs=["coprime", "{k}"],
              multiline_impl=["    {var} = math.gcd(n, {k})",
                              "    {var} = {var} == 1"],
              singleline_impl="    {var} = math.gcd(n, {k}) == 1"),

    # ── digit-sum family ──
    Predicate("digit_sum_gt", "the sum of n's digits is greater than {t}",
              lambda n, t: _digit_sum(n) > t, {"t": [5, 10, 15, 20]},
              id_substrs=["sum", "digits", "{t}"],
              multiline_impl=["    {var} = sum(int(d) for d in str(abs(n)))",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = sum(int(d) for d in str(abs(n))) > {t}"),
    Predicate("digit_sum_divisible", "the sum of n's digits is divisible by {k}",
              lambda n, k: _digit_sum(n) % k == 0, {"k": [3, 4, 5, 9]},
              id_substrs=["sum", "digits", "divisible", "{k}"],
              multiline_impl=["    {var} = sum(int(d) for d in str(abs(n)))",
                              "    {var} = {var} % {k} == 0"],
              singleline_impl="    {var} = sum(int(d) for d in str(abs(n))) % {k} == 0"),

    # ── accumulate over digits/divisors, then test (loop / comprehension) ──
    Predicate("digit_product_gt", "the product of n's digits is greater than {t}",
              lambda n, t: _digit_product(n) > t, {"t": [0, 5, 10, 20]},
              id_substrs=["product", "digits", "{t}"],
              multiline_impl=["    {var} = 1",
                              "    for d in str(abs(n)):",
                              "        {var} *= int(d)",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = math.prod(int(d) for d in str(abs(n))) > {t}"),
    Predicate("all_digits_even", "all of n's digits are even",
              lambda n: all(int(c) % 2 == 0 for c in str(abs(n))),
              id_substrs=["all", "digits", "even"],
              multiline_impl=["    {var} = [int(c) for c in str(abs(n))]",
                              "    {var} = all(d % 2 == 0 for d in {var})"],
              singleline_impl="    {var} = all(int(c) % 2 == 0 for c in str(abs(n)))"),
    Predicate("num_divisors_gt", "n has more than {t} divisors",
              lambda n, t: n != 0 and sum(1 for d in range(1, abs(n) + 1) if n % d == 0) > t,
              {"t": [2, 3, 4, 6]}, id_substrs=["more than", "{t}", "divisors"],
              multiline_impl=["    {var} = sum(1 for d in range(1, abs(n) + 1) if n % d == 0)",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = sum(1 for d in range(1, abs(n) + 1) if n % d == 0) > {t}"),
    Predicate("abundant", "the sum of n's proper divisors is greater than n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) > n,
              id_substrs=["sum", "proper divisors", "greater"],
              multiline_impl=["    {var} = sum(d for d in range(1, abs(n)) if n % d == 0)",
                              "    {var} = n > 0 and {var} > n"],
              singleline_impl="    {var} = n > 0 and sum(d for d in range(1, abs(n)) if n % d == 0) > n"),

    # ── prime via trial division ──
    Predicate("prime", "n is a prime number", lambda n: _is_prime(n), id_substrs=["prime"],
              multiline_impl=["    {var} = all(n % d != 0 for d in range(2, int(abs(n) ** 0.5) + 1))",
                              "    {var} = {var} and n > 1"],
              singleline_impl="    {var} = n > 1 and all(n % d != 0 for d in range(2, int(abs(n) ** 0.5) + 1))"),
]

PREDICATES_BY_ID: dict[str, Predicate] = {p.id: p for p in INT_PREDICATES}


# ──────────────────────────────────────────────────────────────────────────────
# Sample types.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ConditionSpec(DataClassJsonMixin):
    """Language/modularity-agnostic core: an ordered predicate list + test inputs.
    Matched (lang, modularity) variants share `spec_id`."""

    spec_id: str
    predicate_ids: list[str]
    predicate_params: list[dict]
    spec_texts: list[str]            # rendered NL, in order
    id_substrs: list[list[str]]      # rendered informative substrings per predicate, in order
    test_inputs: list[int]           # n values for unit-testing the generated checker

    @property
    def num_conditions(self) -> int:
        return len(self.predicate_ids)

    def expected_results(self, n: int) -> tuple[list[bool], bool]:
        """Oracle-derived (per-condition booleans in order, collective-all) for input n.
        Logical reference for eyeballing — the LM chooses the print format, so this is
        not an exact stdout matcher."""
        individuals = [
            PREDICATES_BY_ID[pid].evaluate(n, params)
            for pid, params in zip(self.predicate_ids, self.predicate_params)
        ]
        return individuals, all(individuals)


@dataclass
class ConditionSample(DataClassJsonMixin):
    """A concrete, renderable task: a spec under one (lang, modularity)."""

    spec: ConditionSpec
    lang: Lang = "python"
    modularity: Modularity = "modular"
    without_comment: bool = True

    @property
    def id_substrs(self) -> list[list[str]]:
        """Informative spec substrings per condition (rendered), in order — handy for
        tracking attention back to each predicate's key tokens."""
        return self.spec.id_substrs

    @property
    def steps(self) -> list[str]:
        """Condition spec texts — alias so this duck-types as MultiStepSample for the
        oracle labeler (`label_generation`)."""
        return self.spec.spec_texts

    @property
    def name(self) -> str:
        """Unique id for on-disk artifacts / LabeledGeneration.sample_name."""
        return f"{self.spec.spec_id}__{self.lang}__{self.modularity}"

    # --- rendering ---------------------------------------------------------
    def render_instruction(self) -> str:
        ls = LANGS[self.lang]
        body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.spec.spec_texts))
        opener = (
            f"Write a {ls.display} program that implements a function "
            f"`{ls.fn_name}` which, given an integer n, checks the following "
            f"conditions:"
        )
        # Force a named result variable per condition (so each check is a
        # separable statement, not hidden inside a print) + a single aggregation
        # print at the end (individual + collective pass).
        assign_rule = (
            "For each condition, in the order listed, assign the boolean result of "
            "the check to its own local variable, using single lowercase letters in "
            "order — `a` for the first condition, `b` for the second, and so on. "
            "After all conditions have been checked, use a SINGLE print statement that, "
            "using those variables, reports whether each condition passed individually "
            "and whether all of them passed collectively."
        )
        if self.modularity == "modular":
            mod_rule = (
                f"Implement each condition as a separate function that returns a "
                f"{ls.bool_type}; inside `{ls.fn_name}`, call these functions in the "
                f"order listed to compute the per-condition result variables."
            )
        else:
            mod_rule = (
                f"Compute each condition's boolean result "
                f"inside `{ls.fn_name}` (one local variable per condition); do not "
                f"define any helper functions."
            )
        parts = [opener, "", body, "", assign_rule, mod_rule]
        if self.without_comment:
            parts.append("Do not include any comments in your code.")
        return "\n".join(parts)

    def render_code_stub(self) -> str:
        """Prefill stub. Monolithic seeds the entry signature; modular seeds empty
        (helpers-first — required so C++/Java compile, harmless for Python/JS)."""
        if self.modularity == "monolithic":
            return LANGS[self.lang].mono_stub
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Sampling.
# ──────────────────────────────────────────────────────────────────────────────


def sample_specs(
    n_samples: int,
    num_conditions: list[int],
    seed: int = 0,
    predicates: list[Predicate] = INT_PREDICATES,
    n_test_inputs: int = 6,
    allow_repeats: bool = True,
) -> list[ConditionSpec]:
    """Draw `n_samples` specs. Each gets a count drawn uniformly from
    `num_conditions`, then that many predicates drawn (with replacement if
    `allow_repeats`) from `predicates`, each with sampled params."""
    rng = random.Random(seed)
    specs: list[ConditionSpec] = []
    for i in range(n_samples):
        k = rng.choice(num_conditions)
        if allow_repeats:
            chosen = [rng.choice(predicates) for _ in range(k)]
        else:
            chosen = rng.sample(predicates, k=min(k, len(predicates)))
        params = [p.sample_params(rng) for p in chosen]
        texts = [p.render(pr) for p, pr in zip(chosen, params)]
        substrs = [p.render_substrs(pr) for p, pr in zip(chosen, params)]
        test_inputs = [rng.randint(-5, 1000) for _ in range(n_test_inputs)]
        specs.append(ConditionSpec(
            spec_id=f"cond_{i:04d}",
            predicate_ids=[p.id for p in chosen],
            predicate_params=params,
            spec_texts=texts,
            id_substrs=substrs,
            test_inputs=test_inputs,
        ))
    return specs


def materialize(
    specs: list[ConditionSpec],
    langs: list[Lang],
    modularities: list[Modularity],
    without_comment: bool = True,
) -> list[ConditionSample]:
    """Cross specs × langs × modularities into concrete samples (matched variants
    share spec.spec_id)."""
    out: list[ConditionSample] = []
    for spec in specs:
        for lang in langs:
            for mod in modularities:
                out.append(ConditionSample(spec=spec, lang=lang, modularity=mod,
                                           without_comment=without_comment))
    return out


@dataclass
class ConditionDataset(DataClassJsonMixin):
    """A saved collection of concrete (lang, modularity) samples + provenance meta."""

    samples: list[ConditionSample]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ConditionDataset":
        return cls.from_json(Path(path).read_text())
