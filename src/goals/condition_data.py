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
from typing import Any, Callable, Literal, Optional

from dataclasses_json import DataClassJsonMixin

Lang = Literal["python", "cpp", "java", "javascript"]
Modularity = Literal["modular", "monolithic"]
# The argument datatype a predicate operates on.  A single prompt/spec mixes only predicates
# of ONE datatype: "int" -> check_conditions(n: int); "str" -> check_conditions(s: str).
DataType = Literal["int", "str"]


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
    oracle: Callable[..., bool]        # oracle(arg, **params) -> bool (ground truth)
    param_space: dict[str, list] = field(default_factory=dict)
    # Datatype of the checked value.  "int" -> impls/oracle take `n`; "str" -> they take `s`.
    # All predicates in one spec share a single datatype (see `DataType`).  Kept AFTER
    # param_space so the existing positional `Predicate(id, spec, oracle, param_space, ...)`
    # call sites (int bank) are unaffected; chunky predicates pass `datatype=` as a kwarg.
    datatype: DataType = "int"
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
    # OPTIONAL — `None` ⇒ predicate has no single-line form (e.g. "chunky" predicates whose
    # implementation is inherently multi-line). Such predicates can only be rendered multiline;
    # the "moved_on" (complete-target) state is produced from the full multiline block instead.
    singleline_impl: Optional[str] = None
    # C++ forced-prefill forms.  Same shape and rules as the Python versions.  Because
    # C++ statements are typed, multiline impls typically declare an intermediate
    # `int {var}_v = …;` on line 1 and then `bool {var} = {var}_v <op> …;` on line 2 —
    # the `{var}` substitution ripples through so distinct condition letters produce
    # distinct intermediate names.  Empty ⇒ predicate has no C++ multi-line form.
    multiline_impl_cpp: list[str] = field(default_factory=list)
    singleline_impl_cpp: Optional[str] = None
    # Canonical natural variable name for this predicate's boolean result
    # (e.g. "is_even"); used by build_example(naming="natural"). Populated from
    # _PREDICATE_VAR_NAMES at module load; falls back to `id` if left blank.
    variable_name: str = ""

    @property
    def var_name(self) -> str:
        """Natural result-variable name (`variable_name`, else `id`)."""
        return self.variable_name or self.id

    @property
    def is_multiline(self) -> bool:
        return bool(self.multiline_impl)

    def _multiline_for(self, lang: str) -> list[str]:
        if lang == "cpp":
            return self.multiline_impl_cpp
        if lang == "python":
            return self.multiline_impl
        raise ValueError(f"Predicate has no forced-prefill impl for lang={lang!r}")

    def _singleline_for(self, lang: str) -> Optional[str]:
        if lang == "cpp":
            return self.singleline_impl_cpp
        if lang == "python":
            return self.singleline_impl
        raise ValueError(f"Predicate has no forced-prefill impl for lang={lang!r}")

    def is_multiline_for(self, lang: str = "python") -> bool:
        """Whether this predicate has a multi-line forced-prefill form in `lang`."""
        return bool(self._multiline_for(lang))

    def has_singleline_for(self, lang: str = "python") -> bool:
        """Whether this predicate has a single-line forced-prefill form in `lang`.
        `False` for "chunky" (inherently multi-line) predicates — callers that need a
        completed-target rendering must fall back to the full multiline block."""
        return bool(self._singleline_for(lang))

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

    def render_multiline(self, params: dict[str, Any], var: str, lang: str = "python") -> list[str]:
        """Forced-prefill body lines with `{var}` -> the condition's result letter and
        `{param}` placeholders filled in.  `var` is substituted first (it is not a
        param), then params are formatted."""
        return [ln.replace("{var}", var).format(**params) for ln in self._multiline_for(lang)]

    def render_singleline(self, params: dict[str, Any], var: str, lang: str = "python") -> str:
        """Single-line forced-prefill: `{var}` -> the result letter, `{param}` filled in.
        Raises if this predicate has no single-line form in `lang` (see `has_singleline_for`)."""
        impl = self._singleline_for(lang)
        if not impl:
            raise ValueError(
                f"predicate {self.id!r} has no single-line impl for lang={lang!r} "
                f"(multiline-only 'chunky' predicate); use render_multiline instead"
            )
        return impl.replace("{var}", var).format(**params)

    def evaluate(self, n: int, params: dict[str, Any]) -> bool:
        return bool(self.oracle(n, **params))


INT_PREDICATES: list[Predicate] = [
    # ── trivially single-line (no canonical multi-line form) ──
    Predicate("positive", "n is positive", lambda n: n > 0, id_substrs=["positive"],
              singleline_impl="    {var} = n > 0",
              singleline_impl_cpp="    bool {var} = n > 0;"),
    Predicate("le_bound", "n is less than or equal to {b}", lambda n, b: n <= b,
              {"b": [50, 100, 200, 500, 1000]}, id_substrs=["less than or equal", "{b}"],
              singleline_impl="    {var} = n <= {b}",
              singleline_impl_cpp="    bool {var} = n <= {b};"),
    Predicate("ge_bound", "n is greater than or equal to {a}", lambda n, a: n >= a,
              {"a": [0, 1, 10, 50, 100]}, id_substrs=["greater than or equal", "{a}"],
              singleline_impl="    {var} = n >= {a}",
              singleline_impl_cpp="    bool {var} = n >= {a};"),
    Predicate("in_range", "n is between {a} and {b} (inclusive)",
              lambda n, a, b: a <= n <= b, {"a": [0, 10, 50], "b": [100, 200, 500]},
              id_substrs=["between", "{a}", "{b}"],
              singleline_impl="    {var} = {a} <= n <= {b}",
              singleline_impl_cpp="    bool {var} = n >= {a} && n <= {b};"),

    # ── "compute remainder/quantity, then compare" (forced 2-line) ──
    Predicate("even", "n is even", lambda n: n % 2 == 0, id_substrs=["even"],
              multiline_impl=["    {var} = n % 2", "    {var} = {var} == 0"],
              singleline_impl="    {var} = n % 2 == 0",
              multiline_impl_cpp=["    int {var}_v = n % 2;",
                                  "    bool {var} = {var}_v == 0;"],
              singleline_impl_cpp="    bool {var} = n % 2 == 0;"),
    Predicate("odd", "n is odd", lambda n: n % 2 == 1, id_substrs=["odd"],
              multiline_impl=["    {var} = n % 2", "    {var} = {var} == 1"],
              singleline_impl="    {var} = n % 2 == 1",
              multiline_impl_cpp=["    int {var}_v = n % 2;",
                                  "    bool {var} = {var}_v == 1;"],
              singleline_impl_cpp="    bool {var} = n % 2 == 1;"),
    Predicate("divisible", "n is divisible by {k}", lambda n, k: k != 0 and n % k == 0,
              {"k": list(range(2, 21))}, id_substrs=["divisible", "{k}"],
              multiline_impl=["    {var} = n % {k}", "    {var} = {var} == 0"],
              singleline_impl="    {var} = n % {k} == 0",
              multiline_impl_cpp=["    int {var}_v = n % {k};",
                                  "    bool {var} = {var}_v == 0;"],
              singleline_impl_cpp="    bool {var} = n % {k} == 0;"),
    Predicate("perfect_square", "n is a perfect square",
              lambda n: n >= 0 and math.isqrt(n) ** 2 == n, id_substrs=["perfect", "square"],
              multiline_impl=["    {var} = int(abs(n) ** 0.5)",
                              "    {var} = n >= 0 and {var} * {var} == n"],
              singleline_impl="    {var} = n >= 0 and int(abs(n) ** 0.5) ** 2 == n",
              multiline_impl_cpp=["    int {var}_v = (int)std::sqrt((double)std::abs(n));",
                                  "    bool {var} = n >= 0 && {var}_v * {var}_v == n;"],
              singleline_impl_cpp="    bool {var} = n >= 0 && (int)std::sqrt((double)std::abs(n)) * (int)std::sqrt((double)std::abs(n)) == n;"),
    Predicate("perfect_cube", "n is a perfect cube", lambda n: _is_perfect_cube(n),
              id_substrs=["perfect", "cube"],
              multiline_impl=["    {var} = round(abs(n) ** (1 / 3))",
                              "    {var} = {var} ** 3 == abs(n)"],
              singleline_impl="    {var} = round(abs(n) ** (1 / 3)) ** 3 == abs(n)",
              multiline_impl_cpp=["    int {var}_v = (int)std::round(std::cbrt((double)std::abs(n)));",
                                  "    bool {var} = {var}_v * {var}_v * {var}_v == std::abs(n);"],
              singleline_impl_cpp="    bool {var} = (int)std::round(std::cbrt((double)std::abs(n))) * (int)std::round(std::cbrt((double)std::abs(n))) * (int)std::round(std::cbrt((double)std::abs(n))) == std::abs(n);"),
    Predicate("triangular", "n is a triangular number", lambda n: _is_triangular(n),
              id_substrs=["triangular"],
              multiline_impl=["    {var} = int((2 * abs(n)) ** 0.5)",
                              "    {var} = n >= 0 and {var} * ({var} + 1) // 2 == n"],
              singleline_impl="    {var} = n >= 0 and int((8 * n + 1) ** 0.5) ** 2 == 8 * n + 1",
              multiline_impl_cpp=["    int {var}_v = (int)std::sqrt(2.0 * (double)std::abs(n));",
                                  "    bool {var} = n >= 0 && {var}_v * ({var}_v + 1) / 2 == n;"],
              singleline_impl_cpp="    bool {var} = n >= 0 && (int)std::sqrt(2.0 * (double)n) * ((int)std::sqrt(2.0 * (double)n) + 1) / 2 == n;"),
    Predicate("palindrome", "n is a palindrome (its digits read the same forwards and backwards)",
              lambda n: str(abs(n)) == str(abs(n))[::-1], id_substrs=["palindrome"],
              multiline_impl=["    {var} = str(abs(n))",
                              "    {var} = {var} == {var}[::-1]"],
              singleline_impl="    {var} = str(abs(n)) == str(abs(n))[::-1]",
              multiline_impl_cpp=["    std::string {var}_s = std::to_string(std::abs(n));",
                                  "    bool {var} = {var}_s == std::string({var}_s.rbegin(), {var}_s.rend());"],
              singleline_impl_cpp="    bool {var} = std::to_string(std::abs(n)) == std::string(std::to_string(std::abs(n)).rbegin(), std::to_string(std::abs(n)).rend());"),
    Predicate("reversed_greater", "n is smaller than the number obtained by reversing its digits",
              lambda n: int(str(abs(n))[::-1]) > abs(n), id_substrs=["reversing", "digits"],
              multiline_impl=["    {var} = int(str(abs(n))[::-1])",
                              "    {var} = {var} > abs(n)"],
              singleline_impl="    {var} = int(str(abs(n))[::-1]) > abs(n)",
              multiline_impl_cpp=["    std::string {var}_s = std::to_string(std::abs(n));",
                                  "    bool {var} = std::stoi(std::string({var}_s.rbegin(), {var}_s.rend())) > std::abs(n);"],
              singleline_impl_cpp="    bool {var} = std::stoi(std::string(std::to_string(std::abs(n)).rbegin(), std::to_string(std::abs(n)).rend())) > std::abs(n);"),
    Predicate("num_digits_ge", "n has at least {k} digits",
              lambda n, k: len(str(abs(n))) >= k, {"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "digits"],
              multiline_impl=["    {var} = len(str(abs(n)))",
                              "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = len(str(abs(n))) >= {k}",
              multiline_impl_cpp=["    int {var}_v = (int)std::to_string(std::abs(n)).length();",
                                  "    bool {var} = {var}_v >= {k};"],
              singleline_impl_cpp="    bool {var} = std::to_string(std::abs(n)).length() >= {k};"),
    Predicate("set_bits_gt", "n has more than {t} ones in its binary representation",
              lambda n, t: bin(abs(n)).count("1") > t, {"t": [1, 2, 3, 4]},
              id_substrs=["more than", "{t}", "ones", "binary"],
              multiline_impl=['    {var} = bin(abs(n)).count("1")',
                              "    {var} = {var} > {t}"],
              singleline_impl='    {var} = bin(abs(n)).count("1") > {t}',
              multiline_impl_cpp=["    int {var}_v = __builtin_popcount(std::abs(n));",
                                  "    bool {var} = {var}_v > {t};"],
              singleline_impl_cpp="    bool {var} = __builtin_popcount(std::abs(n)) > {t};"),
    Predicate("power_of_two", "n is a power of two",
              lambda n: n > 0 and (n & (n - 1)) == 0, id_substrs=["power", "two"],
              multiline_impl=['    {var} = bin(n).count("1")',
                              "    {var} = n > 0 and {var} == 1"],
              singleline_impl="    {var} = n > 0 and (n & (n - 1)) == 0",
              multiline_impl_cpp=["    int {var}_v = __builtin_popcount(n);",
                                  "    bool {var} = n > 0 && {var}_v == 1;"],
              singleline_impl_cpp="    bool {var} = n > 0 && (n & (n - 1)) == 0;"),
    Predicate("coprime_with", "n is coprime with {k}", lambda n, k: math.gcd(n, k) == 1,
              {"k": [6, 10, 12, 15]}, id_substrs=["coprime", "{k}"],
              multiline_impl=["    {var} = math.gcd(n, {k})",
                              "    {var} = {var} == 1"],
              singleline_impl="    {var} = math.gcd(n, {k}) == 1",
              multiline_impl_cpp=["    int {var}_v = std::gcd(n, {k});",
                                  "    bool {var} = {var}_v == 1;"],
              singleline_impl_cpp="    bool {var} = std::gcd(n, {k}) == 1;"),

    # ── digit-sum family ──
    Predicate("digit_sum_gt", "the sum of n's digits is greater than {t}",
              lambda n, t: _digit_sum(n) > t, {"t": [5, 10, 15, 20]},
              id_substrs=["sum", "digits", "{t}"],
              multiline_impl=["    {var} = sum(int(d) for d in str(abs(n)))",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = sum(int(d) for d in str(abs(n))) > {t}",
              multiline_impl_cpp=["    int {var}_s = 0;",
                                  "    for (char c : std::to_string(std::abs(n))) {var}_s += c - '0';",
                                  "    bool {var} = {var}_s > {t};"],
              singleline_impl_cpp="    bool {var} = std::accumulate(std::to_string(std::abs(n)).begin(), std::to_string(std::abs(n)).end(), 0, [](int acc, char c) {{ return acc + (c - '0'); }}) > {t};"),
    Predicate("digit_sum_divisible", "the sum of n's digits is divisible by {k}",
              lambda n, k: _digit_sum(n) % k == 0, {"k": [3, 4, 5, 9]},
              id_substrs=["sum", "digits", "divisible", "{k}"],
              multiline_impl=["    {var} = sum(int(d) for d in str(abs(n)))",
                              "    {var} = {var} % {k} == 0"],
              singleline_impl="    {var} = sum(int(d) for d in str(abs(n))) % {k} == 0",
              multiline_impl_cpp=["    int {var}_s = 0;",
                                  "    for (char c : std::to_string(std::abs(n))) {var}_s += c - '0';",
                                  "    bool {var} = {var}_s % {k} == 0;"],
              singleline_impl_cpp="    bool {var} = std::accumulate(std::to_string(std::abs(n)).begin(), std::to_string(std::abs(n)).end(), 0, [](int acc, char c) {{ return acc + (c - '0'); }}) % {k} == 0;"),

    # ── accumulate over digits/divisors, then test (loop / comprehension) ──
    Predicate("digit_product_gt", "the product of n's digits is greater than {t}",
              lambda n, t: _digit_product(n) > t, {"t": [0, 5, 10, 20]},
              id_substrs=["product", "digits", "{t}"],
              multiline_impl=["    {var} = 1",
                              "    for d in str(abs(n)):",
                              "        {var} *= int(d)",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = math.prod(int(d) for d in str(abs(n))) > {t}",
              multiline_impl_cpp=["    int {var}_p = 1;",
                                  "    for (char c : std::to_string(std::abs(n))) {var}_p *= (c - '0');",
                                  "    bool {var} = {var}_p > {t};"],
              singleline_impl_cpp="    bool {var} = std::accumulate(std::to_string(std::abs(n)).begin(), std::to_string(std::abs(n)).end(), 1, [](int acc, char c) {{ return acc * (c - '0'); }}) > {t};"),
    Predicate("all_digits_even", "all of n's digits are even",
              lambda n: all(int(c) % 2 == 0 for c in str(abs(n))),
              id_substrs=["all", "digits", "even"],
              multiline_impl=["    {var} = [int(c) for c in str(abs(n))]",
                              "    {var} = all(d % 2 == 0 for d in {var})"],
              singleline_impl="    {var} = all(int(c) % 2 == 0 for c in str(abs(n)))",
              multiline_impl_cpp=["    std::string {var}_s = std::to_string(std::abs(n));",
                                  "    bool {var} = std::all_of({var}_s.begin(), {var}_s.end(), [](char c) {{ return (c - '0') % 2 == 0; }});"],
              singleline_impl_cpp="    bool {var} = std::all_of(std::to_string(std::abs(n)).begin(), std::to_string(std::abs(n)).end(), [](char c) {{ return (c - '0') % 2 == 0; }});"),
    Predicate("num_divisors_gt", "n has more than {t} divisors",
              lambda n, t: n != 0 and sum(1 for d in range(1, abs(n) + 1) if n % d == 0) > t,
              {"t": [2, 3, 4, 6]}, id_substrs=["more than", "{t}", "divisors"],
              multiline_impl=["    {var} = sum(1 for d in range(1, abs(n) + 1) if n % d == 0)",
                              "    {var} = {var} > {t}"],
              singleline_impl="    {var} = sum(1 for d in range(1, abs(n) + 1) if n % d == 0) > {t}",
              multiline_impl_cpp=["    int {var}_c = 0;",
                                  "    for (int d = 1; d <= std::abs(n); ++d) if (n % d == 0) {var}_c++;",
                                  "    bool {var} = {var}_c > {t};"],
              singleline_impl_cpp="    bool {var} = [n]() {{ int c = 0; for (int d = 1; d <= std::abs(n); ++d) if (n % d == 0) c++; return c; }}() > {t};"),
    Predicate("abundant", "the sum of n's proper divisors is greater than n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) > n,
              id_substrs=["sum", "proper divisors", "greater"],
              multiline_impl=["    {var} = sum(d for d in range(1, abs(n)) if n % d == 0)",
                              "    {var} = n > 0 and {var} > n"],
              singleline_impl="    {var} = n > 0 and sum(d for d in range(1, abs(n)) if n % d == 0) > n",
              multiline_impl_cpp=["    int {var}_s = 0;",
                                  "    for (int d = 1; d < std::abs(n); ++d) if (n % d == 0) {var}_s += d;",
                                  "    bool {var} = n > 0 && {var}_s > n;"],
              singleline_impl_cpp="    bool {var} = n > 0 && [n]() {{ int s = 0; for (int d = 1; d < std::abs(n); ++d) if (n % d == 0) s += d; return s; }}() > n;"),

    # ── prime via trial division ──
    Predicate("prime", "n is a prime number", lambda n: _is_prime(n), id_substrs=["prime"],
              multiline_impl=["    {var} = all(n % d != 0 for d in range(2, int(abs(n) ** 0.5) + 1))",
                              "    {var} = {var} and n > 1"],
              singleline_impl="    {var} = n > 1 and all(n % d != 0 for d in range(2, int(abs(n) ** 0.5) + 1))",
              multiline_impl_cpp=["    bool {var} = n > 1;",
                                  "    for (int d = 2; d * d <= std::abs(n); ++d) if (n % d == 0) {var} = false;"],
              singleline_impl_cpp="    bool {var} = n > 1 && [n]() {{ for (int d = 2; d * d <= std::abs(n); ++d) if (n % d == 0) return false; return true; }}();"),
]

PREDICATES_BY_ID: dict[str, Predicate] = {p.id: p for p in INT_PREDICATES}

# Canonical natural variable names per predicate, for build_example(naming="natural").
# Kept as a table (not inline constructor kwargs) to keep the predicate list readable.
_PREDICATE_VAR_NAMES: dict[str, str] = {
    "positive": "is_positive",
    "le_bound": "is_le_bound",
    "ge_bound": "is_ge_bound",
    "in_range": "is_in_range",
    "even": "is_even",
    "odd": "is_odd",
    "divisible": "is_divisible",
    "perfect_square": "is_perfect_square",
    "perfect_cube": "is_perfect_cube",
    "triangular": "is_triangular",
    "palindrome": "is_palindrome",
    "reversed_greater": "is_reverse_greater",
    "num_digits_ge": "has_min_digits",
    "set_bits_gt": "has_many_set_bits",
    "power_of_two": "is_power_of_two",
    "coprime_with": "is_coprime",
    "digit_sum_gt": "has_large_digit_sum",
    "digit_sum_divisible": "digit_sum_divisible",
    "digit_product_gt": "has_large_digit_product",
    "all_digits_even": "all_digits_even",
    "num_divisors_gt": "has_many_divisors",
    "abundant": "is_abundant",
    "prime": "is_prime",
}
assert set(_PREDICATE_VAR_NAMES) == set(PREDICATES_BY_ID), (
    "predicate var-name table out of sync with INT_PREDICATES"
)
# frozen dataclass → fill the field via object.__setattr__ at module load.
for _p in INT_PREDICATES:
    object.__setattr__(_p, "variable_name", _PREDICATE_VAR_NAMES[_p.id])


# ──────────────────────────────────────────────────────────────────────────────
# "Chunky" predicates — inherently multi-line (explicit loops / accumulators / flags), with
# NO single-line form (`singleline_impl` left None → multiline-only, see `has_singleline_for`).
# Unlike INT_PREDICATES these span both datatypes: `datatype="int"` (arg `n`) and `"str"`
# (arg `s`).  A spec is built from chunky predicates of ONE datatype.  Seed set only — kept
# small to test model behavior before scaling.  `variable_name` is set inline here (the
# frozen dataclass still accepts it in __init__), so no separate name table is needed.
# `{var}` reuse convention matches INT_PREDICATES: every assigned variable — accumulators and
# flags included — is `{var}` (or `{var}_<suffix>` for a distinct intermediate), so swapping
# the result letter renames the whole block; loop variables stay plain (`d`, `ch`, `i`).
CHUNKY_PREDICATES: list[Predicate] = [
    # ── int (arg `n`) ──
    Predicate("prime_chunky", "n is a prime number", lambda n: _is_prime(n),
              datatype="int", id_substrs=["prime"], variable_name="is_prime",
              multiline_impl=["    {var} = True",
                              "    for d in range(2, int(abs(n) ** 0.5) + 1):",
                              "        if n % d == 0:",
                              "            {var} = False",
                              "            break",
                              "    {var} = {var} and n > 1"]),
    Predicate("digit_sum_gt_chunky", "the sum of n's digits is greater than {t}",
              lambda n, t: _digit_sum(n) > t, datatype="int", param_space={"t": [5, 10, 15, 20]},
              id_substrs=["sum", "digits", "{t}"], variable_name="has_large_digit_sum",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d)",
                              "    {var} = {var} > {t}"]),
    Predicate("perfect_number_chunky", "the sum of n's proper divisors equals n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) == n,
              datatype="int", id_substrs=["proper divisors", "equals"], variable_name="is_perfect",
              multiline_impl=["    {var} = 0",
                              "    for d in range(1, abs(n)):",
                              "        if n % d == 0:",
                              "            {var} += d",
                              "    {var} = n > 0 and {var} == n"]),
    Predicate("all_digits_even_chunky", "all of n's digits are even",
              lambda n: all(int(c) % 2 == 0 for c in str(abs(n))),
              datatype="int", id_substrs=["all", "digits", "even"], variable_name="all_digits_even",
              multiline_impl=["    {var} = True",
                              "    for d in str(abs(n)):",
                              "        if int(d) % 2 != 0:",
                              "            {var} = False",
                              "            break"]),

    # ── str (arg `s`) ──
    Predicate("vowel_count_ge_chunky", "s contains at least {k} vowels",
              lambda s, k: sum(1 for ch in s.lower() if ch in "aeiou") >= k,
              datatype="str", param_space={"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "vowels"], variable_name="has_min_vowels",
              multiline_impl=["    {var} = 0",
                              "    for ch in s.lower():",
                              "        if ch in 'aeiou':",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("palindrome_chunky", "s reads the same forwards and backwards",
              lambda s: s == s[::-1], datatype="str",
              id_substrs=["forwards", "backwards"], variable_name="is_palindrome",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(s) // 2):",
                              "        if s[i] != s[len(s) - 1 - i]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("has_digit_char_chunky", "s contains at least one digit character",
              lambda s: any(ch.isdigit() for ch in s), datatype="str",
              id_substrs=["digit", "character"], variable_name="has_digit",
              multiline_impl=["    {var} = False",
                              "    for ch in s:",
                              "        if ch.isdigit():",
                              "            {var} = True",
                              "            break"]),
    Predicate("no_repeated_chars_chunky", "no character in s appears more than once",
              lambda s: len(set(s)) == len(s), datatype="str",
              id_substrs=["no character", "more than once"], variable_name="no_repeated_chars",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for ch in s:",
                              "        if ch in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.append(ch)"]),
]

CHUNKY_PREDICATES_BY_ID: dict[str, Predicate] = {p.id: p for p in CHUNKY_PREDICATES}

# Unified id → Predicate lookup across the int bank + chunky bank (ids are disjoint: chunky
# ids are suffixed `_chunky`).  Used wherever a spec resolves predicate ids to oracles, since a
# spec may draw from either bank.
ALL_PREDICATES_BY_ID: dict[str, Predicate] = {**PREDICATES_BY_ID, **CHUNKY_PREDICATES_BY_ID}
assert len(ALL_PREDICATES_BY_ID) == len(PREDICATES_BY_ID) + len(CHUNKY_PREDICATES_BY_ID), (
    "predicate id collision between INT_PREDICATES and CHUNKY_PREDICATES"
)

# invariants for the seed set: every chunky predicate is multiline-only in python, has a
# result-variable name, and a valid datatype.
for _p in CHUNKY_PREDICATES:
    assert _p.is_multiline_for("python"), f"chunky predicate {_p.id!r} has no python multiline_impl"
    assert not _p.has_singleline_for("python"), f"chunky predicate {_p.id!r} should be multiline-only"
    assert _p.variable_name, f"chunky predicate {_p.id!r} missing variable_name"
    assert _p.datatype in ("int", "str"), f"chunky predicate {_p.id!r} bad datatype {_p.datatype!r}"


def chunky_predicates_for(datatype: DataType) -> list[Predicate]:
    """Chunky predicates for one datatype (a spec mixes only a single datatype)."""
    return [p for p in CHUNKY_PREDICATES if p.datatype == datatype]


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

    @property
    def datatype(self) -> DataType:
        """The checked-value datatype for this spec (all predicates share one).
        `int` for the classic int bank; may be `str` for chunky specs."""
        if not self.predicate_ids:
            return "int"
        return ALL_PREDICATES_BY_ID[self.predicate_ids[0]].datatype

    def expected_results(self, n) -> tuple[list[bool], bool]:
        """Oracle-derived (per-condition booleans in order, collective-all) for input `n`
        (an int, or a str for str-datatype specs).  Logical reference for eyeballing — the LM
        chooses the print format, so this is not an exact stdout matcher."""
        individuals = [
            ALL_PREDICATES_BY_ID[pid].evaluate(n, params)
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
    def render_instruction(self, name_rule: str = "letters", multistep: bool = False) -> str:
        """`name_rule="letters"` mandates lowercase-letter result vars (`a`, `b`, …);
        any other value drops the naming constraint entirely (lets the model/prefill
        choose), used by build_example(naming="natural"|"indexed").

        `multistep=True` adds a clause asking the model to implement each check step-by-step
        with explicit loops/intermediate variables rather than a one-line expression or
        comprehension — used for the chunky dataset, where we WANT multi-line implementations
        (the simple predicates otherwise collapse to one-liners)."""
        ls = LANGS[self.lang]
        body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.spec.spec_texts))
        subject = "a string s" if self.spec.datatype == "str" else "an integer n"
        opener = (
            f"Write a {ls.display} program that implements a function "
            f"`{ls.fn_name}` which, given {subject}, checks the following "
            f"conditions:"
        )
        # Force a named result variable per condition (so each check is a
        # separable statement, not hidden inside a print) + a single aggregation
        # print at the end (individual + collective pass).
        if name_rule == "letters":
            assign_rule = (
                "For each condition, in the order listed, assign the boolean result of "
                "the check to its own local variable, using single lowercase letters in "
                "order — `a` for the first condition, `b` for the second, and so on. "
                "After all conditions have been checked, use a SINGLE print statement that, "
                "using those variables, reports whether each condition passed individually "
                "and whether all of them passed collectively."
            )
        else:
            assign_rule = (
                "For each condition, in the order listed, assign the boolean result of "
                "the check to its own local variable. "
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
        if multistep:
            parts.append(
                "Implement each condition's check step by step using explicit statements — "
                "loops and intermediate variables where appropriate — rather than a single "
                "one-line expression or comprehension."
            )
        if self.without_comment:
            parts.append("Do not include any comments in your code.")
        return "\n".join(parts)

    def render_code_stub(self) -> str:
        """Prefill stub. Monolithic seeds the entry signature; modular seeds empty
        (helpers-first — required so C++/Java compile, harmless for Python/JS).
        The Python monolithic stub is datatype-aware: `n: int` vs `s: str`."""
        if self.modularity != "monolithic":
            return ""
        ls = LANGS[self.lang]
        if self.lang == "python":
            arg, ann = ("s", "str") if self.spec.datatype == "str" else ("n", "int")
            return f"def {ls.fn_name}({arg}: {ann}):\n"
        return ls.mono_stub  # non-python langs: int-only for now


# ──────────────────────────────────────────────────────────────────────────────
# Sampling.
# ──────────────────────────────────────────────────────────────────────────────

_STR_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def sample_test_inputs(rng: random.Random, datatype: DataType, k: int = 6) -> list:
    """Eyeball/unit-test inputs for a spec: ints for `int`, short random strings for `str`."""
    if datatype == "str":
        return ["".join(rng.choice(_STR_ALPHABET) for _ in range(rng.randint(3, 8)))
                for _ in range(k)]
    return [rng.randint(-5, 1000) for _ in range(k)]


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
    `allow_repeats`) from `predicates`, each with sampled params.

    `predicates` must be a single-datatype pool (all int, or all str — a spec never mixes
    datatypes).  Pass e.g. `chunky_predicates_for("str")` for a string-datatype dataset."""
    rng = random.Random(seed)
    specs: list[ConditionSpec] = []
    for i in range(n_samples):
        k = rng.choice(num_conditions)
        if allow_repeats:
            chosen = [rng.choice(predicates) for _ in range(k)]
        else:
            chosen = rng.sample(predicates, k=min(k, len(predicates)))
        dts = {p.datatype for p in chosen}
        assert len(dts) == 1, f"spec {i} mixes datatypes {dts}; a spec must be single-datatype"
        params = [p.sample_params(rng) for p in chosen]
        texts = [p.render(pr) for p, pr in zip(chosen, params)]
        substrs = [p.render_substrs(pr) for p, pr in zip(chosen, params)]
        test_inputs = sample_test_inputs(rng, next(iter(dts)), n_test_inputs)
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
