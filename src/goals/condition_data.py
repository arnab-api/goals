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
# of ONE datatype: e.g. "int" -> check_conditions(n: int); "list[int]" -> (nums: list[int]).
DataType = Literal["int", "str", "list[int]", "list[str]", "list[list[int]]", "dict[str, int]"]

_STR_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_WORD_POOL = ["cat", "dog", "tree", "sky", "code", "data", "loop", "node", "edge",
              "time", "list", "map", "key", "sum", "max", "arc", "bit", "row"]


def _s_int(rng: random.Random):
    return rng.randint(-5, 1000)


def _s_str(rng: random.Random):
    return "".join(rng.choice(_STR_ALPHABET) for _ in range(rng.randint(3, 8)))


def _s_list_int(rng: random.Random):
    return [rng.randint(-10, 50) for _ in range(rng.randint(2, 7))]


def _s_list_str(rng: random.Random):
    return [rng.choice(_WORD_POOL) for _ in range(rng.randint(2, 6))]


def _s_matrix(rng: random.Random):
    k = rng.randint(2, 4)                        # square so symmetric/diagonal checks are meaningful
    return [[rng.randint(-5, 9) for _ in range(k)] for _ in range(k)]


def _s_dict(rng: random.Random):
    keys = rng.sample(_WORD_POOL, rng.randint(2, 5))
    return {k: rng.randint(-3, 20) for k in keys}


@dataclass(frozen=True)
class DataTypeSpec:
    """Per-datatype rendering config: the argument name used in impls/oracle, its type
    annotation for the stub signature, the instruction subject phrase, and a random test-input
    sampler.  A predicate's `multiline_impl` and `oracle` must reference `arg` by name."""
    key: str
    arg: str
    annotation: str
    subject: str
    sampler: Callable[[random.Random], Any]


DATATYPES: dict[str, DataTypeSpec] = {d.key: d for d in [
    DataTypeSpec("int", "n", "int", "an integer n", _s_int),
    DataTypeSpec("str", "s", "str", "a string s", _s_str),
    DataTypeSpec("list[int]", "nums", "list[int]", "a list of integers nums", _s_list_int),
    DataTypeSpec("list[str]", "words", "list[str]", "a list of strings words", _s_list_str),
    DataTypeSpec("list[list[int]]", "grid", "list[list[int]]",
                 "a square integer matrix grid (a list of equal-length integer rows)", _s_matrix),
    DataTypeSpec("dict[str, int]", "counts", "dict[str, int]",
                 "a dictionary counts mapping strings to integers", _s_dict),
]}


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
# Simple, INDEPENDENT predicate banks for the non-int datatypes.
#
# Same shape and contract as INT_PREDICATES — *not* chunky: each has a canonical two-step
# `multiline_impl` (compute an intermediate, then compare it) AND a `singleline_impl`, so the
# stay-vs-move-on test has both states available. Loops are deliberately avoided: a multi-line
# `for` body can host more than one sub-task, which makes "did it move on" unreadable off the
# next line. Comprehensions and genexps are fine — one statement cannot host a second sub-task.
#
# INDEPENDENCE: every impl reads only the input argument (`s` / `nums` / `words`) and its own
# `{var}`. No predicate can consume another's result, so any ordering of a spec is correct and
# the only thing that can govern the order is the prompt.
#
# `{var}` reuse convention matches INT_PREDICATES: every assigned variable, intermediates
# included, is `{var}`, so substituting the condition's result letter renames the whole block.
# C++ impls are intentionally left empty — these banks exist for the (python-only) move-on
# evaluation; `_multiline_preds_for("cpp")` simply yields a smaller pool.
# ──────────────────────────────────────────────────────────────────────────────

STR_PREDICATES: list[Predicate] = [
    Predicate("str_len_gt", "s is longer than {k} characters", lambda s, k: len(s) > k,
              {"k": [2, 3, 4, 5, 6]}, datatype="str", id_substrs=["longer", "{k}"],
              variable_name="is_long",
              multiline_impl=["    {var} = len(s)", "    {var} = {var} > {k}"],
              singleline_impl="    {var} = len(s) > {k}"),
    Predicate("str_vowel_count_ge", "s contains at least {k} vowels",
              lambda s, k: sum(ch in "aeiou" for ch in s.lower()) >= k,
              {"k": [1, 2, 3, 4]}, datatype="str", id_substrs=["vowels", "{k}"],
              variable_name="has_min_vowels",
              multiline_impl=['    {var} = sum(ch in "aeiou" for ch in s.lower())',
                              "    {var} = {var} >= {k}"],
              singleline_impl='    {var} = sum(ch in "aeiou" for ch in s.lower()) >= {k}'),
    Predicate("str_upper_count_gt", "s contains more than {k} uppercase letters",
              lambda s, k: sum(ch.isupper() for ch in s) > k,
              {"k": [0, 1, 2, 3]}, datatype="str", id_substrs=["uppercase", "{k}"],
              variable_name="has_many_uppercase",
              multiline_impl=["    {var} = sum(ch.isupper() for ch in s)",
                              "    {var} = {var} > {k}"],
              singleline_impl="    {var} = sum(ch.isupper() for ch in s) > {k}"),
    Predicate("str_digit_count_ge", "s contains at least {k} digits",
              lambda s, k: sum(ch.isdigit() for ch in s) >= k,
              {"k": [1, 2, 3]}, datatype="str", id_substrs=["digits", "{k}"],
              variable_name="has_min_digits",
              multiline_impl=["    {var} = sum(ch.isdigit() for ch in s)",
                              "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = sum(ch.isdigit() for ch in s) >= {k}"),
    Predicate("str_palindrome", "s reads the same forwards and backwards",
              lambda s: s == s[::-1], datatype="str", id_substrs=["forwards", "backwards"],
              variable_name="is_palindrome",
              multiline_impl=["    {var} = s[::-1]", "    {var} = {var} == s"],
              singleline_impl="    {var} = s == s[::-1]"),
    Predicate("str_distinct_gt", "s uses more than {k} distinct characters",
              lambda s, k: len(set(s)) > k, {"k": [2, 3, 4, 5]}, datatype="str",
              id_substrs=["distinct", "{k}"], variable_name="has_many_distinct_chars",
              multiline_impl=["    {var} = set(s)", "    {var} = len({var}) > {k}"],
              singleline_impl="    {var} = len(set(s)) > {k}"),
    Predicate("str_starts_vowel", "s starts with a vowel",
              lambda s: s[:1].lower() in "aeiou" and s[:1] != "", datatype="str",
              id_substrs=["starts", "vowel"], variable_name="starts_with_vowel",
              multiline_impl=["    {var} = s[:1].lower()",
                              '    {var} = {var} != "" and {var} in "aeiou"'],
              singleline_impl='    {var} = s[:1].lower() != "" and s[:1].lower() in "aeiou"'),
    Predicate("str_ends_digit", "s ends with a digit", lambda s: s[-1:].isdigit(),
              datatype="str", id_substrs=["ends", "digit"], variable_name="ends_with_digit",
              multiline_impl=["    {var} = s[-1:]", "    {var} = {var}.isdigit()"],
              singleline_impl="    {var} = s[-1:].isdigit()"),
    Predicate("str_alpha_majority", "more than half the characters of s are letters",
              lambda s: sum(ch.isalpha() for ch in s) * 2 > len(s), datatype="str",
              id_substrs=["half", "letters"], variable_name="mostly_letters",
              multiline_impl=["    {var} = sum(ch.isalpha() for ch in s)",
                              "    {var} = {var} * 2 > len(s)"],
              singleline_impl="    {var} = sum(ch.isalpha() for ch in s) * 2 > len(s)"),
    Predicate("str_contains_char", 's contains the character "{c}"',
              lambda s, c: c in s, {"c": ["a", "e", "i", "o", "u"]}, datatype="str",
              id_substrs=["contains", "{c}"], variable_name="contains_char",
              multiline_impl=['    {var} = s.count("{c}")', "    {var} = {var} > 0"],
              singleline_impl='    {var} = s.count("{c}") > 0'),
]

LIST_INT_PREDICATES: list[Predicate] = [
    Predicate("li_sum_gt", "the sum of nums is greater than {k}", lambda nums, k: sum(nums) > k,
              {"k": [0, 20, 50, 100]}, datatype="list[int]", id_substrs=["sum", "{k}"],
              variable_name="has_large_sum",
              multiline_impl=["    {var} = sum(nums)", "    {var} = {var} > {k}"],
              singleline_impl="    {var} = sum(nums) > {k}"),
    Predicate("li_sum_divisible", "the sum of nums is divisible by {k}",
              lambda nums, k: sum(nums) % k == 0, {"k": [2, 3, 4, 5, 7]},
              datatype="list[int]", id_substrs=["sum", "divisible", "{k}"],
              variable_name="sum_divisible",
              multiline_impl=["    {var} = sum(nums) % {k}", "    {var} = {var} == 0"],
              singleline_impl="    {var} = sum(nums) % {k} == 0"),
    Predicate("li_max_ge", "the largest element of nums is at least {k}",
              lambda nums, k: max(nums) >= k, {"k": [0, 10, 25, 40]}, datatype="list[int]",
              id_substrs=["largest", "{k}"], variable_name="has_large_max",
              multiline_impl=["    {var} = max(nums)", "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = max(nums) >= {k}"),
    Predicate("li_min_lt", "the smallest element of nums is less than {k}",
              lambda nums, k: min(nums) < k, {"k": [0, 5, 10, 20]}, datatype="list[int]",
              id_substrs=["smallest", "{k}"], variable_name="has_small_min",
              multiline_impl=["    {var} = min(nums)", "    {var} = {var} < {k}"],
              singleline_impl="    {var} = min(nums) < {k}"),
    Predicate("li_len_ge", "nums has at least {k} elements", lambda nums, k: len(nums) >= k,
              {"k": [2, 3, 4, 5]}, datatype="list[int]", id_substrs=["at least", "{k}", "elements"],
              variable_name="has_min_length",
              multiline_impl=["    {var} = len(nums)", "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = len(nums) >= {k}"),
    Predicate("li_all_even", "every element of nums is even",
              lambda nums: all(x % 2 == 0 for x in nums), datatype="list[int]",
              id_substrs=["every", "even"], variable_name="all_even",
              multiline_impl=["    {var} = [x for x in nums if x % 2 != 0]",
                              "    {var} = len({var}) == 0"],
              singleline_impl="    {var} = all(x % 2 == 0 for x in nums)"),
    Predicate("li_distinct", "all elements of nums are distinct",
              lambda nums: len(set(nums)) == len(nums), datatype="list[int]",
              id_substrs=["distinct"], variable_name="all_distinct",
              multiline_impl=["    {var} = set(nums)", "    {var} = len({var}) == len(nums)"],
              singleline_impl="    {var} = len(set(nums)) == len(nums)"),
    Predicate("li_sorted_asc", "nums is sorted in non-decreasing order",
              lambda nums: list(nums) == sorted(nums), datatype="list[int]",
              id_substrs=["sorted", "non-decreasing"], variable_name="is_sorted",
              multiline_impl=["    {var} = sorted(nums)", "    {var} = {var} == nums"],
              singleline_impl="    {var} = nums == sorted(nums)"),
    Predicate("li_range_gt", "the difference between the largest and smallest element of nums "
                             "is greater than {k}",
              lambda nums, k: max(nums) - min(nums) > k, {"k": [5, 10, 20, 30]},
              datatype="list[int]", id_substrs=["difference", "largest", "smallest", "{k}"],
              variable_name="has_wide_range",
              multiline_impl=["    {var} = max(nums) - min(nums)", "    {var} = {var} > {k}"],
              singleline_impl="    {var} = max(nums) - min(nums) > {k}"),
    Predicate("li_count_positive_ge", "at least {k} elements of nums are positive",
              lambda nums, k: sum(x > 0 for x in nums) >= k, {"k": [1, 2, 3, 4]},
              datatype="list[int]", id_substrs=["at least", "{k}", "positive"],
              variable_name="has_min_positives",
              multiline_impl=["    {var} = sum(x > 0 for x in nums)", "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = sum(x > 0 for x in nums) >= {k}"),
]

LIST_STR_PREDICATES: list[Predicate] = [
    Predicate("ls_len_ge", "words has at least {k} entries", lambda words, k: len(words) >= k,
              {"k": [2, 3, 4, 5]}, datatype="list[str]", id_substrs=["at least", "{k}", "entries"],
              variable_name="has_min_entries",
              multiline_impl=["    {var} = len(words)", "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = len(words) >= {k}"),
    Predicate("ls_all_lower", "every word in words is entirely lowercase",
              lambda words: all(w == w.lower() for w in words), datatype="list[str]",
              id_substrs=["every", "lowercase"], variable_name="all_lowercase",
              multiline_impl=["    {var} = [w for w in words if w != w.lower()]",
                              "    {var} = len({var}) == 0"],
              singleline_impl="    {var} = all(w == w.lower() for w in words)"),
    Predicate("ls_total_chars_gt", "the words in words have more than {k} characters in total",
              lambda words, k: sum(len(w) for w in words) > k, {"k": [10, 15, 20, 30]},
              datatype="list[str]", id_substrs=["more than", "{k}", "in total"],
              variable_name="has_many_characters",
              multiline_impl=["    {var} = sum(len(w) for w in words)", "    {var} = {var} > {k}"],
              singleline_impl="    {var} = sum(len(w) for w in words) > {k}"),
    Predicate("ls_longest_ge", "the longest word in words has at least {k} characters",
              lambda words, k: max(len(w) for w in words) >= k, {"k": [3, 4, 5, 6]},
              datatype="list[str]", id_substrs=["longest", "{k}"], variable_name="has_long_word",
              multiline_impl=["    {var} = max(len(w) for w in words)", "    {var} = {var} >= {k}"],
              singleline_impl="    {var} = max(len(w) for w in words) >= {k}"),
    Predicate("ls_distinct", "all words in words are distinct",
              lambda words: len(set(words)) == len(words), datatype="list[str]",
              id_substrs=["distinct"], variable_name="all_distinct",
              multiline_impl=["    {var} = set(words)", "    {var} = len({var}) == len(words)"],
              singleline_impl="    {var} = len(set(words)) == len(words)"),
    Predicate("ls_sorted_alpha", "the words in words are in alphabetical order",
              lambda words: list(words) == sorted(words), datatype="list[str]",
              id_substrs=["alphabetical", "order"], variable_name="is_sorted_alphabetically",
              multiline_impl=["    {var} = sorted(words)", "    {var} = {var} == words"],
              singleline_impl="    {var} = words == sorted(words)"),
    Predicate("ls_same_length", "all words in words have the same length",
              lambda words: len({len(w) for w in words}) <= 1, datatype="list[str]",
              id_substrs=["same", "length"], variable_name="all_same_length",
              multiline_impl=["    {var} = set(len(w) for w in words)",
                              "    {var} = len({var}) <= 1"],
              singleline_impl="    {var} = len(set(len(w) for w in words)) <= 1"),
    Predicate("ls_starts_vowel_ge", "at least {k} words in words start with a vowel",
              lambda words, k: sum(w[:1].lower() in "aeiou" and w != "" for w in words) >= k,
              {"k": [1, 2, 3]}, datatype="list[str]", id_substrs=["at least", "{k}", "vowel"],
              variable_name="has_min_vowel_starts",
              multiline_impl=['    {var} = sum(w != "" and w[:1].lower() in "aeiou" for w in words)',
                              "    {var} = {var} >= {k}"],
              singleline_impl='    {var} = sum(w != "" and w[:1].lower() in "aeiou" for w in words) >= {k}'),
    Predicate("ls_has_long_word", "words contains a word longer than {k} characters",
              lambda words, k: any(len(w) > k for w in words), {"k": [3, 4, 5, 6]},
              datatype="list[str]", id_substrs=["longer than", "{k}"],
              variable_name="contains_long_word",
              multiline_impl=["    {var} = [w for w in words if len(w) > {k}]",
                              "    {var} = len({var}) > 0"],
              singleline_impl="    {var} = any(len(w) > {k} for w in words)"),
    Predicate("ls_shares_initial", "two of the words in words start with the same letter",
              lambda words: len({w[:1].lower() for w in words}) < len(words),
              datatype="list[str]", id_substrs=["same", "letter"],
              variable_name="shares_initial",
              multiline_impl=["    {var} = set(w[:1].lower() for w in words)",
                              "    {var} = len({var}) < len(words)"],
              singleline_impl="    {var} = len(set(w[:1].lower() for w in words)) < len(words)"),
]

# One simple (non-chunky) bank per datatype. `int` keeps its historical name.
SIMPLE_PREDICATE_BANKS: dict[str, list[Predicate]] = {
    "int": INT_PREDICATES,
    "str": STR_PREDICATES,
    "list[int]": LIST_INT_PREDICATES,
    "list[str]": LIST_STR_PREDICATES,
}

for _dt, _bank in SIMPLE_PREDICATE_BANKS.items():
    for _p in _bank:
        assert _p.datatype == _dt, f"{_p.id!r} in the {_dt!r} bank but datatype={_p.datatype!r}"
        assert _p.variable_name, f"simple predicate {_p.id!r} missing variable_name"
        assert _p.has_singleline_for("python"), f"{_p.id!r} needs a python singleline_impl"

SIMPLE_PREDICATES_BY_ID: dict[str, Predicate] = {
    p.id: p for bank in SIMPLE_PREDICATE_BANKS.values() for p in bank
}
assert len(SIMPLE_PREDICATES_BY_ID) == sum(len(b) for b in SIMPLE_PREDICATE_BANKS.values()), (
    "duplicate predicate id across the simple banks"
)


def predicates_for(datatype: DataType) -> list[Predicate]:
    """Simple (non-chunky) predicates for one datatype — the counterpart of
    `chunky_predicates_for`. A spec never mixes datatypes."""
    bank = SIMPLE_PREDICATE_BANKS.get(datatype)
    assert bank, (f"no simple predicate bank for datatype={datatype!r}; "
                  f"have {sorted(SIMPLE_PREDICATE_BANKS)}")
    return bank


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
# helper functions used by chunky-predicate oracles (auto-collected)
def _collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


CHUNKY_PREDICATES: list[Predicate] = [
    # ── int (arg n) ──
    Predicate("int_prime_chunky", "n is a prime number",
              lambda n: n > 1 and all(n % d for d in range(2, int(abs(n) ** 0.5) + 1)),
              datatype="int", id_substrs=["prime"], variable_name="is_prime",
              multiline_impl=["    {var} = True",
                              "    for d in range(2, int(abs(n) ** 0.5) + 1):",
                              "        if n % d == 0:",
                              "            {var} = False",
                              "            break",
                              "    {var} = {var} and n > 1"]),
    Predicate("int_perfect_number_chunky", "the sum of n's proper divisors equals n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) == n,
              datatype="int", id_substrs=["proper divisors", "equals"],
              variable_name="is_perfect_number",
              multiline_impl=["    {var} = 0",
                              "    for d in range(1, abs(n)):",
                              "        if n % d == 0:",
                              "            {var} += d",
                              "    {var} = n > 0 and {var} == n"]),
    Predicate("int_abundant_chunky", "the sum of n's proper divisors is greater than n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) > n,
              datatype="int", id_substrs=["proper divisors", "greater than"],
              variable_name="is_abundant",
              multiline_impl=["    {var} = 0",
                              "    for d in range(1, abs(n)):",
                              "        if n % d == 0:",
                              "            {var} += d",
                              "    {var} = n > 0 and {var} > n"]),
    Predicate("int_deficient_chunky", "the sum of n's proper divisors is less than n",
              lambda n: n > 0 and sum(d for d in range(1, n) if n % d == 0) < n,
              datatype="int", id_substrs=["proper divisors", "less than"],
              variable_name="is_deficient",
              multiline_impl=["    {var} = 0",
                              "    for d in range(1, abs(n)):",
                              "        if n % d == 0:",
                              "            {var} += d",
                              "    {var} = n > 0 and {var} < n"]),
    Predicate("int_armstrong_chunky",
              "n is an Armstrong number (equal to the sum of its digits each raised to the power of the digit count)",
              lambda n: n >= 0 and sum(int(c) ** len(str(abs(n))) for c in str(abs(n))) == n,
              datatype="int", id_substrs=["Armstrong"], variable_name="is_armstrong",
              multiline_impl=["    {var} = 0",
                              "    {var}_k = len(str(abs(n)))",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d) ** {var}_k",
                              "    {var} = n >= 0 and {var} == n"]),
    Predicate("int_harshad_chunky", "n is a Harshad number (divisible by the sum of its digits)",
              lambda n: n > 0 and n % sum(int(c) for c in str(abs(n))) == 0,
              datatype="int", id_substrs=["Harshad"], variable_name="is_harshad",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d)",
                              "    {var} = n > 0 and n % {var} == 0"]),
    Predicate("int_digit_sum_gt_chunky", "the sum of n's digits is greater than {t}",
              lambda n, t: sum(int(c) for c in str(abs(n))) > t,
              datatype="int", param_space={"t": [5, 10, 15, 20]},
              id_substrs=["sum", "digits", "{t}"], variable_name="has_large_digit_sum",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d)",
                              "    {var} = {var} > {t}"]),
    Predicate("int_digit_sum_divisible_chunky", "the sum of n's digits is divisible by {k}",
              lambda n, k: sum(int(c) for c in str(abs(n))) % k == 0,
              datatype="int", param_space={"k": [3, 4, 5, 9]},
              id_substrs=["sum", "digits", "divisible", "{k}"], variable_name="digit_sum_divisible",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d)",
                              "    {var} = {var} % {k} == 0"]),
    Predicate("int_digit_product_gt_chunky", "the product of n's digits is greater than {t}",
              lambda n, t: math.prod(int(c) for c in str(abs(n))) > t,
              datatype="int", param_space={"t": [0, 5, 10, 20]},
              id_substrs=["product", "digits", "{t}"], variable_name="has_large_digit_product",
              multiline_impl=["    {var} = 1",
                              "    for d in str(abs(n)):",
                              "        {var} *= int(d)",
                              "    {var} = {var} > {t}"]),
    Predicate("int_num_digits_gt_chunky", "the number of digits in n is greater than {k}",
              lambda n, k: len(str(abs(n))) > k,
              datatype="int", param_space={"k": [1, 2, 3]},
              id_substrs=["number of digits", "{k}"], variable_name="has_many_digits",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += 1",
                              "    {var} = {var} > {k}"]),
    Predicate("int_all_digits_even_chunky", "all of n's digits are even",
              lambda n: all(int(c) % 2 == 0 for c in str(abs(n))),
              datatype="int", id_substrs=["all", "digits", "even"], variable_name="all_digits_even",
              multiline_impl=["    {var} = True",
                              "    for d in str(abs(n)):",
                              "        if int(d) % 2 != 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_all_digits_odd_chunky", "all of n's digits are odd",
              lambda n: all(int(c) % 2 == 1 for c in str(abs(n))),
              datatype="int", id_substrs=["all", "digits", "odd"], variable_name="all_digits_odd",
              multiline_impl=["    {var} = True",
                              "    for d in str(abs(n)):",
                              "        if int(d) % 2 != 1:",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_strictly_increasing_digits_chunky",
              "n's digits are strictly increasing from left to right",
              lambda n: all(int(a) < int(b) for a, b in zip(str(abs(n)), str(abs(n))[1:])),
              datatype="int", id_substrs=["strictly increasing", "digits"],
              variable_name="digits_increasing",
              multiline_impl=["    {var} = True",
                              "    {var}_s = str(abs(n))",
                              "    for i in range(1, len({var}_s)):",
                              "        if int({var}_s[i]) <= int({var}_s[i - 1]):",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_strictly_decreasing_digits_chunky",
              "n's digits are strictly decreasing from left to right",
              lambda n: all(int(a) > int(b) for a, b in zip(str(abs(n)), str(abs(n))[1:])),
              datatype="int", id_substrs=["strictly decreasing", "digits"],
              variable_name="digits_decreasing",
              multiline_impl=["    {var} = True",
                              "    {var}_s = str(abs(n))",
                              "    for i in range(1, len({var}_s)):",
                              "        if int({var}_s[i]) >= int({var}_s[i - 1]):",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_has_repeated_digit_chunky", "n has at least one repeated digit",
              lambda n: len(set(str(abs(n)))) != len(str(abs(n))),
              datatype="int", id_substrs=["repeated digit"], variable_name="has_repeated_digit",
              multiline_impl=["    {var} = False",
                              "    {var}_seen = []",
                              "    for d in str(abs(n)):",
                              "        if d in {var}_seen:",
                              "            {var} = True",
                              "            break",
                              "        {var}_seen.append(d)"]),
    Predicate("int_palindrome_chunky", "n reads the same forwards and backwards",
              lambda n: str(abs(n)) == str(abs(n))[::-1],
              datatype="int", id_substrs=["forwards", "backwards"], variable_name="is_palindrome",
              multiline_impl=["    {var} = True",
                              "    {var}_s = str(abs(n))",
                              "    for i in range(len({var}_s) // 2):",
                              "        if {var}_s[i] != {var}_s[len({var}_s) - 1 - i]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_perfect_square_chunky", "n is a perfect square",
              lambda n: n >= 0 and math.isqrt(n) ** 2 == n,
              datatype="int", id_substrs=["perfect", "square"], variable_name="is_perfect_square",
              multiline_impl=["    {var} = False",
                              "    {var}_r = int(abs(n) ** 0.5)",
                              "    for i in range({var}_r - 1, {var}_r + 2):",
                              "        if i >= 0 and i * i == n:",
                              "            {var} = True",
                              "            break"]),
    Predicate("int_perfect_cube_chunky", "n is a perfect cube",
              lambda n: any((round(abs(n) ** (1 / 3)) + d) ** 3 == abs(n) for d in (-1, 0, 1)),
              datatype="int", id_substrs=["perfect", "cube"], variable_name="is_perfect_cube",
              multiline_impl=["    {var} = False",
                              "    {var}_c = round(abs(n) ** (1 / 3))",
                              "    for d in (-1, 0, 1):",
                              "        if ({var}_c + d) ** 3 == abs(n):",
                              "            {var} = True",
                              "            break"]),
    Predicate("int_triangular_chunky", "n is a triangular number",
              lambda n: n >= 0 and any(m >= 0 and m * (m + 1) // 2 == n
                                       for m in range(int((2 * n) ** 0.5) - 1, int((2 * n) ** 0.5) + 2)),
              datatype="int", id_substrs=["triangular"], variable_name="is_triangular",
              multiline_impl=["    {var} = False",
                              "    if n >= 0:",
                              "        {var}_m = int((2 * n) ** 0.5)",
                              "        for m in range({var}_m - 1, {var}_m + 2):",
                              "            if m >= 0 and m * (m + 1) // 2 == n:",
                              "                {var} = True",
                              "                break"]),
    Predicate("int_power_of_two_chunky", "n is a power of two",
              lambda n: n > 0 and (n & (n - 1)) == 0,
              datatype="int", id_substrs=["power", "two"], variable_name="is_power_of_two",
              multiline_impl=["    {var} = n > 0",
                              "    {var}_m = n",
                              "    while {var}_m > 1:",
                              "        if {var}_m % 2 != 0:",
                              "            {var} = False",
                              "            break",
                              "        {var}_m //= 2"]),
    Predicate("int_set_bits_gt_chunky", "n has more than {k} ones in its binary representation",
              lambda n, k: bin(abs(n)).count("1") > k,
              datatype="int", param_space={"k": [1, 2, 3, 4]},
              id_substrs=["more than", "{k}", "ones", "binary"], variable_name="has_many_set_bits",
              multiline_impl=["    {var} = 0",
                              "    {var}_m = abs(n)",
                              "    while {var}_m > 0:",
                              "        {var} += {var}_m % 2",
                              "        {var}_m //= 2",
                              "    {var} = {var} > {k}"]),
    Predicate("int_divisible_by_each_digit_chunky", "n is divisible by each of its nonzero digits",
              lambda n: n != 0 and all(n % int(c) == 0 for c in str(abs(n)) if c != "0"),
              datatype="int", id_substrs=["divisible", "nonzero digits"],
              variable_name="divisible_by_digits",
              multiline_impl=["    {var} = True",
                              "    for d in str(abs(n)):",
                              "        if int(d) != 0 and n % int(d) != 0:",
                              "            {var} = False",
                              "            break",
                              "    {var} = {var} and n != 0"]),
    Predicate("int_sum_squares_digits_gt_chunky",
              "the sum of the squares of n's digits is greater than {t}",
              lambda n, t: sum(int(c) ** 2 for c in str(abs(n))) > t,
              datatype="int", param_space={"t": [10, 50, 100, 200]},
              id_substrs=["sum of the squares", "digits", "{t}"],
              variable_name="has_large_digit_sq_sum",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        {var} += int(d) ** 2",
                              "    {var} = {var} > {t}"]),
    Predicate("int_num_divisors_gt_chunky", "n has more than {k} divisors",
              lambda n, k: n != 0 and sum(1 for d in range(1, abs(n) + 1) if n % d == 0) > k,
              datatype="int", param_space={"k": [2, 3, 4, 6]},
              id_substrs=["more than", "{k}", "divisors"], variable_name="has_many_divisors",
              multiline_impl=["    {var} = 0",
                              "    for d in range(1, abs(n) + 1):",
                              "        if n % d == 0:",
                              "            {var} += 1",
                              "    {var} = {var} > {k}"]),
    Predicate("int_contains_digit_chunky", "n contains the digit {k}",
              lambda n, k: str(k) in str(abs(n)),
              datatype="int", param_space={"k": [0, 3, 7, 9]},
              id_substrs=["contains", "digit {k}"], variable_name="contains_digit",
              multiline_impl=["    {var} = False",
                              "    for d in str(abs(n)):",
                              "        if int(d) == {k}:",
                              "            {var} = True",
                              "            break"]),
    Predicate("int_multiple_of_chunky", "n is a multiple of {k}",
              lambda n, k: n % k == 0,
              datatype="int", param_space={"k": [3, 4, 6, 7, 11]},
              id_substrs=["multiple", "{k}"], variable_name="is_multiple",
              multiline_impl=["    {var} = abs(n)",
                              "    while {var} >= {k}:",
                              "        {var} -= {k}",
                              "    {var} = {var} == 0"]),
    Predicate("int_alternating_parity_digits_chunky",
              "n's digits alternate in parity between even and odd",
              lambda n: all(int(a) % 2 != int(b) % 2 for a, b in zip(str(abs(n)), str(abs(n))[1:])),
              datatype="int", id_substrs=["alternate", "parity"], variable_name="alternating_parity",
              multiline_impl=["    {var} = True",
                              "    {var}_s = str(abs(n))",
                              "    for i in range(1, len({var}_s)):",
                              "        if int({var}_s[i]) % 2 == int({var}_s[i - 1]) % 2:",
                              "            {var} = False",
                              "            break"]),
    Predicate("int_collatz_steps_gt_chunky",
              "the number of Collatz steps needed to reduce n to 1 is greater than {k}",
              lambda n, k: n >= 1 and _collatz_steps(n) > k,
              datatype="int", param_space={"k": [5, 10, 15]},
              id_substrs=["Collatz steps", "{k}"], variable_name="many_collatz_steps",
              multiline_impl=["    {var} = 0",
                              "    {var}_m = n",
                              "    while {var}_m > 1:",
                              "        if {var}_m % 2 == 0:",
                              "            {var}_m //= 2",
                              "        else:",
                              "            {var}_m = 3 * {var}_m + 1",
                              "        {var} += 1",
                              "    {var} = n >= 1 and {var} > {k}"]),
    Predicate("int_digit_product_divisible_chunky", "the product of n's digits is divisible by {k}",
              lambda n, k: math.prod(int(c) for c in str(abs(n))) % k == 0,
              datatype="int", param_space={"k": [2, 3, 5]},
              id_substrs=["product", "digits", "divisible", "{k}"],
              variable_name="digit_product_divisible",
              multiline_impl=["    {var} = 1",
                              "    for d in str(abs(n)):",
                              "        {var} *= int(d)",
                              "    {var} = {var} % {k} == 0"]),
    Predicate("int_count_even_digits_gt_chunky", "the count of n's even digits is greater than {t}",
              lambda n, t: sum(1 for c in str(abs(n)) if int(c) % 2 == 0) > t,
              datatype="int", param_space={"t": [0, 1, 2]},
              id_substrs=["count", "even digits", "{t}"], variable_name="has_many_even_digits",
              multiline_impl=["    {var} = 0",
                              "    for d in str(abs(n)):",
                              "        if int(d) % 2 == 0:",
                              "            {var} += 1",
                              "    {var} = {var} > {t}"]),
    # ── str (arg s) ──
    Predicate("str_vowel_count_ge_chunky", "s contains at least {k} vowels",
              lambda s, k: sum(1 for ch in s.lower() if ch in "aeiou") >= k,
              datatype="str", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "vowels"], variable_name="has_min_vowels",
              multiline_impl=["    {var} = 0",
                              "    for ch in s.lower():",
                              "        if ch in 'aeiou':",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_consonant_count_ge_chunky", "s contains at least {k} consonants",
              lambda s, k: sum(1 for ch in s.lower() if ch.isalpha() and ch not in "aeiou") >= k,
              datatype="str", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "consonants"], variable_name="has_min_consonants",
              multiline_impl=["    {var} = 0",
                              "    for ch in s.lower():",
                              "        if ch.isalpha() and ch not in 'aeiou':",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_digit_count_ge_chunky", "s contains at least {k} digit characters",
              lambda s, k: sum(1 for ch in s if ch.isdigit()) >= k,
              datatype="str", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "digit"], variable_name="has_min_digits",
              multiline_impl=["    {var} = 0",
                              "    for ch in s:",
                              "        if ch.isdigit():",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_palindrome_chunky", "s reads the same forwards and backwards",
              lambda s: s == s[::-1], datatype="str",
              id_substrs=["forwards", "backwards"], variable_name="is_palindrome",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(s) // 2):",
                              "        if s[i] != s[len(s) - 1 - i]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_has_digit_char_chunky", "s contains at least one digit character",
              lambda s: any(ch.isdigit() for ch in s), datatype="str",
              id_substrs=["one digit", "character"], variable_name="has_digit",
              multiline_impl=["    {var} = False",
                              "    for ch in s:",
                              "        if ch.isdigit():",
                              "            {var} = True",
                              "            break"]),
    Predicate("str_has_uppercase_chunky", "s contains at least one uppercase letter",
              lambda s: any(ch.isupper() for ch in s), datatype="str",
              id_substrs=["one uppercase", "letter"], variable_name="has_uppercase",
              multiline_impl=["    {var} = False",
                              "    for ch in s:",
                              "        if ch.isupper():",
                              "            {var} = True",
                              "            break"]),
    Predicate("str_has_lowercase_chunky", "s contains at least one lowercase letter",
              lambda s: any(ch.islower() for ch in s), datatype="str",
              id_substrs=["one lowercase", "letter"], variable_name="has_lowercase",
              multiline_impl=["    {var} = False",
                              "    for ch in s:",
                              "        if ch.islower():",
                              "            {var} = True",
                              "            break"]),
    Predicate("str_all_alpha_chunky", "every character in s is an alphabetic letter",
              lambda s: len(s) > 0 and all(ch.isalpha() for ch in s), datatype="str",
              id_substrs=["every character", "alphabetic"], variable_name="all_alpha",
              multiline_impl=["    {var} = len(s) > 0",
                              "    for ch in s:",
                              "        if not ch.isalpha():",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_all_alnum_chunky", "every character in s is a letter or a digit",
              lambda s: len(s) > 0 and all(ch.isalnum() for ch in s), datatype="str",
              id_substrs=["every character", "letter or a digit"], variable_name="all_alnum",
              multiline_impl=["    {var} = len(s) > 0",
                              "    for ch in s:",
                              "        if not ch.isalnum():",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_no_uppercase_chunky", "s contains no uppercase letters",
              lambda s: all(not ch.isupper() for ch in s), datatype="str",
              id_substrs=["no uppercase", "letters"], variable_name="has_no_uppercase",
              multiline_impl=["    {var} = True",
                              "    for ch in s:",
                              "        if ch.isupper():",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_no_repeated_chars_chunky", "no character in s appears more than once",
              lambda s: len(set(s)) == len(s), datatype="str",
              id_substrs=["no character", "more than once"], variable_name="no_repeated_chars",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for ch in s:",
                              "        if ch in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.append(ch)"]),
    Predicate("str_adjacent_equal_chunky", "s contains two adjacent equal characters",
              lambda s: any(s[i] == s[i + 1] for i in range(len(s) - 1)), datatype="str",
              id_substrs=["two adjacent equal", "characters"], variable_name="has_adjacent_equal",
              multiline_impl=["    {var} = False",
                              "    for i in range(len(s) - 1):",
                              "        if s[i] == s[i + 1]:",
                              "            {var} = True",
                              "            break"]),
    Predicate("str_non_decreasing_chunky", "the characters of s are in non-decreasing order",
              lambda s: all(s[i] <= s[i + 1] for i in range(len(s) - 1)), datatype="str",
              id_substrs=["non-decreasing order"], variable_name="is_non_decreasing",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(s) - 1):",
                              "        if s[i] > s[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_non_increasing_chunky", "the characters of s are in non-increasing order",
              lambda s: all(s[i] >= s[i + 1] for i in range(len(s) - 1)), datatype="str",
              id_substrs=["non-increasing order"], variable_name="is_non_increasing",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(s) - 1):",
                              "        if s[i] < s[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_ascii_sum_gt_chunky", "the sum of the ASCII codes of s's characters is greater than {t}",
              lambda s, t: sum(ord(ch) for ch in s) > t,
              datatype="str", param_space={"t": [200, 400, 600]},
              id_substrs=["sum", "ASCII codes", "greater than", "{t}"], variable_name="has_large_ascii_sum",
              multiline_impl=["    {var} = 0",
                              "    for ch in s:",
                              "        {var} += ord(ch)",
                              "    {var} = {var} > {t}"]),
    Predicate("str_run_of_k_identical_chunky", "s contains a run of at least {k} identical characters in a row",
              lambda s, k: any(all(i + j < len(s) and s[i + j] == s[i] for j in range(k)) for i in range(len(s))),
              datatype="str", param_space={"k": [2, 3]},
              id_substrs=["run of at least", "{k}", "identical characters"], variable_name="has_identical_run",
              multiline_impl=["    {var} = False",
                              "    {var}_run = 1",
                              "    for i in range(1, len(s)):",
                              "        if s[i] == s[i - 1]:",
                              "            {var}_run += 1",
                              "        else:",
                              "            {var}_run = 1",
                              "        if {var}_run >= {k}:",
                              "            {var} = True"]),
    Predicate("str_distinct_chars_ge_chunky", "s contains at least {k} distinct characters",
              lambda s, k: len(set(s)) >= k,
              datatype="str", param_space={"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "distinct characters"], variable_name="has_min_distinct",
              multiline_impl=["    {var} = 0",
                              "    {var}_seen = []",
                              "    for ch in s:",
                              "        if ch not in {var}_seen:",
                              "            {var}_seen.append(ch)",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_starts_ends_same_chunky", "s starts and ends with the same character",
              lambda s: len(s) > 0 and s[0] == s[-1], datatype="str",
              id_substrs=["starts and ends", "same character"], variable_name="starts_ends_same",
              multiline_impl=["    {var} = False",
                              "    if len(s) > 0:",
                              "        {var} = s[0] == s[len(s) - 1]"]),
    Predicate("str_alternating_letter_digit_chunky", "s strictly alternates between letters and digits",
              lambda s: all(s[i].isdigit() != s[i + 1].isdigit() for i in range(len(s) - 1)),
              datatype="str", id_substrs=["strictly alternates", "letters and digits"],
              variable_name="is_alternating",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(s) - 1):",
                              "        if s[i].isdigit() == s[i + 1].isdigit():",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_more_letters_than_digits_chunky", "s contains more letters than digit characters",
              lambda s: sum(1 for ch in s if ch.isalpha()) > sum(1 for ch in s if ch.isdigit()),
              datatype="str", id_substrs=["more letters than", "digit characters"],
              variable_name="more_letters_than_digits",
              multiline_impl=["    {var}_letters = 0",
                              "    {var}_digits = 0",
                              "    for ch in s:",
                              "        if ch.isalpha():",
                              "            {var}_letters += 1",
                              "        elif ch.isdigit():",
                              "            {var}_digits += 1",
                              "    {var} = {var}_letters > {var}_digits"]),
    Predicate("str_more_vowels_than_consonants_chunky", "s contains more vowels than consonants",
              lambda s: sum(1 for ch in s.lower() if ch in "aeiou") > sum(1 for ch in s.lower() if ch.isalpha() and ch not in "aeiou"),
              datatype="str", id_substrs=["more vowels than", "consonants"],
              variable_name="more_vowels_than_consonants",
              multiline_impl=["    {var}_vowels = 0",
                              "    {var}_consonants = 0",
                              "    for ch in s.lower():",
                              "        if ch in 'aeiou':",
                              "            {var}_vowels += 1",
                              "        elif ch.isalpha():",
                              "            {var}_consonants += 1",
                              "    {var} = {var}_vowels > {var}_consonants"]),
    Predicate("str_starts_with_vowel_chunky", "s starts with a vowel",
              lambda s: len(s) > 0 and s[0].lower() in "aeiou", datatype="str",
              id_substrs=["starts with a vowel"], variable_name="starts_with_vowel",
              multiline_impl=["    {var} = False",
                              "    if len(s) > 0:",
                              "        {var} = s[0].lower() in 'aeiou'"]),
    Predicate("str_ends_with_digit_chunky", "s ends with a digit character",
              lambda s: len(s) > 0 and s[-1].isdigit(), datatype="str",
              id_substrs=["ends with a digit"], variable_name="ends_with_digit",
              multiline_impl=["    {var} = False",
                              "    if len(s) > 0:",
                              "        {var} = s[len(s) - 1].isdigit()"]),
    Predicate("str_two_consecutive_vowels_chunky", "s contains two consecutive vowels",
              lambda s: any(s[i].lower() in "aeiou" and s[i + 1].lower() in "aeiou" for i in range(len(s) - 1)),
              datatype="str", id_substrs=["two consecutive vowels"],
              variable_name="has_consecutive_vowels",
              multiline_impl=["    {var} = False",
                              "    for i in range(len(s) - 1):",
                              "        if s[i].lower() in 'aeiou' and s[i + 1].lower() in 'aeiou':",
                              "            {var} = True",
                              "            break"]),
    Predicate("str_letter_a_count_ge_chunky", "s contains the letter a at least {k} times",
              lambda s, k: sum(1 for ch in s.lower() if ch == "a") >= k,
              datatype="str", param_space={"k": [1, 2]},
              id_substrs=["letter a", "at least", "{k}"], variable_name="has_min_a",
              multiline_impl=["    {var} = 0",
                              "    for ch in s.lower():",
                              "        if ch == 'a':",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_length_ge_chunky", "s has at least {k} characters",
              lambda s, k: len(s) >= k,
              datatype="str", param_space={"k": [3, 4, 5]},
              id_substrs=["at least", "{k}", "characters"], variable_name="has_min_length",
              multiline_impl=["    {var} = 0",
                              "    for ch in s:",
                              "        {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("str_all_chars_same_chunky", "all characters in s are identical",
              lambda s: len(s) > 0 and all(ch == s[0] for ch in s), datatype="str",
              id_substrs=["all characters", "identical"], variable_name="all_chars_same",
              multiline_impl=["    {var} = len(s) > 0",
                              "    for ch in s:",
                              "        if ch != s[0]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("str_digit_value_sum_gt_chunky", "the sum of the digit characters in s is greater than {t}",
              lambda s, t: sum(int(ch) for ch in s if ch.isdigit()) > t,
              datatype="str", param_space={"t": [5, 10, 15]},
              id_substrs=["sum", "digit characters", "greater than", "{t}"], variable_name="has_large_digit_sum",
              multiline_impl=["    {var} = 0",
                              "    for ch in s:",
                              "        if ch.isdigit():",
                              "            {var} += int(ch)",
                              "    {var} = {var} > {t}"]),
    Predicate("str_has_upper_and_lower_chunky", "s contains both an uppercase and a lowercase letter",
              lambda s: any(ch.isupper() for ch in s) and any(ch.islower() for ch in s), datatype="str",
              id_substrs=["both an uppercase", "lowercase letter"], variable_name="has_upper_and_lower",
              multiline_impl=["    {var}_upper = False",
                              "    {var}_lower = False",
                              "    for ch in s:",
                              "        if ch.isupper():",
                              "            {var}_upper = True",
                              "        if ch.islower():",
                              "            {var}_lower = True",
                              "    {var} = {var}_upper and {var}_lower"]),
    Predicate("str_has_vowel_and_digit_chunky", "s contains both a vowel and a digit",
              lambda s: any(ch.lower() in "aeiou" for ch in s) and any(ch.isdigit() for ch in s), datatype="str",
              id_substrs=["both a vowel and a digit"], variable_name="has_vowel_and_digit",
              multiline_impl=["    {var}_vowel = False",
                              "    {var}_digit = False",
                              "    for ch in s:",
                              "        if ch.lower() in 'aeiou':",
                              "            {var}_vowel = True",
                              "        if ch.isdigit():",
                              "            {var}_digit = True",
                              "    {var} = {var}_vowel and {var}_digit"]),
    Predicate("str_first_before_last_chunky", "s's first character comes before its last character alphabetically",
              lambda s: len(s) >= 2 and s[0] < s[-1], datatype="str",
              id_substrs=["first character comes before", "last character"], variable_name="first_before_last",
              multiline_impl=["    {var} = False",
                              "    if len(s) >= 2:",
                              "        {var} = s[0] < s[len(s) - 1]"]),
    # ── list[int] (arg nums) ──
    Predicate("listint_all_positive_chunky", "every integer in nums is positive", lambda nums: all(x > 0 for x in nums),
              datatype="list[int]", id_substrs=["every", "positive"], variable_name="all_positive",
              multiline_impl=["    {var} = True",
                              "    for x in nums:",
                              "        if x <= 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_all_even_chunky", "every integer in nums is even", lambda nums: all(x % 2 == 0 for x in nums),
              datatype="list[int]", id_substrs=["every", "even"], variable_name="all_even",
              multiline_impl=["    {var} = True",
                              "    for x in nums:",
                              "        if x % 2 != 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_all_distinct_chunky", "all elements in nums are distinct",
              lambda nums: len(set(nums)) == len(nums),
              datatype="list[int]", id_substrs=["all", "distinct"], variable_name="all_distinct",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for x in nums:",
                              "        if x in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.append(x)"]),
    Predicate("listint_is_sorted_ascending_chunky", "nums is sorted in ascending order",
              lambda nums: all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)),
              datatype="list[int]", id_substrs=["sorted", "ascending"], variable_name="is_sorted_ascending",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(nums) - 1):",
                              "        if nums[i] > nums[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_is_sorted_descending_chunky", "nums is sorted in descending order",
              lambda nums: all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1)),
              datatype="list[int]", id_substrs=["sorted", "descending"], variable_name="is_sorted_descending",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(nums) - 1):",
                              "        if nums[i] < nums[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_strictly_increasing_chunky", "nums is strictly increasing",
              lambda nums: all(nums[i] < nums[i + 1] for i in range(len(nums) - 1)),
              datatype="list[int]", id_substrs=["strictly", "increasing"], variable_name="is_strictly_increasing",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(nums) - 1):",
                              "        if nums[i] >= nums[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_has_duplicate_chunky", "nums contains at least one duplicate value",
              lambda nums: len(set(nums)) != len(nums),
              datatype="list[int]", id_substrs=["duplicate"], variable_name="has_duplicate",
              multiline_impl=["    {var} = False",
                              "    {var}_seen = []",
                              "    for x in nums:",
                              "        if x in {var}_seen:",
                              "            {var} = True",
                              "            break",
                              "        {var}_seen.append(x)"]),
    Predicate("listint_contains_value_chunky", "nums contains the value {target}",
              lambda nums, target: target in nums,
              datatype="list[int]", param_space={"target": [-2, 0, 1, 3, 5]},
              id_substrs=["contains", "value", "{target}"], variable_name="contains_value",
              multiline_impl=["    {var} = False",
                              "    for x in nums:",
                              "        if x == {target}:",
                              "            {var} = True",
                              "            break"]),
    Predicate("listint_sum_gt_chunky", "the sum of nums is greater than {t}",
              lambda nums, t: sum(nums) > t,
              datatype="list[int]", param_space={"t": [0, 20, 50, 100]},
              id_substrs=["sum", "greater than", "{t}"], variable_name="sum_is_large",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        {var}_acc += x",
                              "    {var} = {var}_acc > {t}"]),
    Predicate("listint_max_gt_chunky", "the maximum element of nums is greater than {t}",
              lambda nums, t: len(nums) > 0 and max(nums) > t,
              datatype="list[int]", param_space={"t": [0, 10, 20, 40]},
              id_substrs=["maximum", "greater than", "{t}"], variable_name="max_is_large",
              multiline_impl=["    {var}_acc = None",
                              "    for x in nums:",
                              "        if {var}_acc is None or x > {var}_acc:",
                              "            {var}_acc = x",
                              "    {var} = {var}_acc is not None and {var}_acc > {t}"]),
    Predicate("listint_min_lt_chunky", "the minimum element of nums is less than {t}",
              lambda nums, t: len(nums) > 0 and min(nums) < t,
              datatype="list[int]", param_space={"t": [-5, 0, 5, 10]},
              id_substrs=["minimum", "less than", "{t}"], variable_name="min_is_small",
              multiline_impl=["    {var}_acc = None",
                              "    for x in nums:",
                              "        if {var}_acc is None or x < {var}_acc:",
                              "            {var}_acc = x",
                              "    {var} = {var}_acc is not None and {var}_acc < {t}"]),
    Predicate("listint_count_even_ge_chunky", "nums contains at least {k} even numbers",
              lambda nums, k: sum(1 for x in nums if x % 2 == 0) >= k,
              datatype="list[int]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "even"], variable_name="has_enough_evens",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        if x % 2 == 0:",
                              "            {var}_acc += 1",
                              "    {var} = {var}_acc >= {k}"]),
    Predicate("listint_product_gt_chunky", "the product of nums is greater than {t}",
              lambda nums, t: math.prod(nums) > t,
              datatype="list[int]", param_space={"t": [0, 10, 100, 1000]},
              id_substrs=["product", "greater than", "{t}"], variable_name="product_is_large",
              multiline_impl=["    {var}_acc = 1",
                              "    for x in nums:",
                              "        {var}_acc *= x",
                              "    {var} = {var}_acc > {t}"]),
    Predicate("listint_has_negative_chunky", "nums contains at least one negative number",
              lambda nums: any(x < 0 for x in nums),
              datatype="list[int]", id_substrs=["negative"], variable_name="has_negative",
              multiline_impl=["    {var} = False",
                              "    for x in nums:",
                              "        if x < 0:",
                              "            {var} = True",
                              "            break"]),
    Predicate("listint_range_gt_chunky", "the range of nums (maximum minus minimum) is greater than {t}",
              lambda nums, t: len(nums) > 0 and (max(nums) - min(nums)) > t,
              datatype="list[int]", param_space={"t": [5, 10, 20, 30]},
              id_substrs=["range", "greater than", "{t}"], variable_name="range_is_large",
              multiline_impl=["    {var}_hi = None",
                              "    {var}_lo = None",
                              "    for x in nums:",
                              "        if {var}_hi is None or x > {var}_hi:",
                              "            {var}_hi = x",
                              "        if {var}_lo is None or x < {var}_lo:",
                              "            {var}_lo = x",
                              "    {var} = {var}_hi is not None and {var}_hi - {var}_lo > {t}"]),
    Predicate("listint_adjacent_equal_chunky", "nums has two adjacent equal elements",
              lambda nums: any(nums[i] == nums[i + 1] for i in range(len(nums) - 1)),
              datatype="list[int]", id_substrs=["adjacent", "equal"], variable_name="has_adjacent_equal",
              multiline_impl=["    {var} = False",
                              "    for i in range(len(nums) - 1):",
                              "        if nums[i] == nums[i + 1]:",
                              "            {var} = True",
                              "            break"]),
    Predicate("listint_alternating_parity_chunky", "the parities of consecutive elements in nums alternate",
              lambda nums: all(nums[i] % 2 != nums[i + 1] % 2 for i in range(len(nums) - 1)),
              datatype="list[int]", id_substrs=["parities", "consecutive", "alternate"],
              variable_name="parities_alternate",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(nums) - 1):",
                              "        if nums[i] % 2 == nums[i + 1] % 2:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_multiple_present_chunky", "nums contains at least one multiple of {k}",
              lambda nums, k: sum(1 for x in nums if x % k == 0) >= 1,
              datatype="list[int]", param_space={"k": [2, 3, 4, 5]},
              id_substrs=["at least one", "multiple", "{k}"], variable_name="has_multiple_of_k",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        if x % {k} == 0:",
                              "            {var}_acc += 1",
                              "    {var} = {var}_acc >= 1"]),
    Predicate("listint_all_in_range_chunky", "every element of nums is between 0 and {t} inclusive",
              lambda nums, t: all(0 <= x <= t for x in nums),
              datatype="list[int]", param_space={"t": [10, 30, 50]},
              id_substrs=["between", "{t}", "inclusive"], variable_name="all_in_range",
              multiline_impl=["    {var} = True",
                              "    for x in nums:",
                              "        if x < 0 or x > {t}:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_palindromic_chunky", "nums reads the same forwards and backwards",
              lambda nums: nums == nums[::-1],
              datatype="list[int]", id_substrs=["forwards", "backwards"], variable_name="is_palindromic",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(nums) // 2):",
                              "        if nums[i] != nums[len(nums) - 1 - i]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_sum_even_gt_chunky", "the sum of the even elements of nums is greater than {t}",
              lambda nums, t: sum(x for x in nums if x % 2 == 0) > t,
              datatype="list[int]", param_space={"t": [0, 5, 10, 20]},
              id_substrs=["sum", "even", "greater than", "{t}"], variable_name="even_sum_is_large",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        if x % 2 == 0:",
                              "            {var}_acc += x",
                              "    {var} = {var}_acc > {t}"]),
    Predicate("listint_run_increasing_chunky", "nums has a run of {k} consecutive increasing elements",
              lambda nums, k: any(all(nums[i + j] < nums[i + j + 1] for j in range(k - 1))
                                  for i in range(len(nums) - k + 1)),
              datatype="list[int]", param_space={"k": [2, 3, 4]},
              id_substrs=["run", "{k}", "consecutive", "increasing"], variable_name="has_increasing_run",
              multiline_impl=["    {var}_run = 0",
                              "    {var}_best = 0",
                              "    for i in range(len(nums)):",
                              "        if i > 0 and nums[i] > nums[i - 1]:",
                              "            {var}_run += 1",
                              "        else:",
                              "            {var}_run = 1",
                              "        if {var}_run > {var}_best:",
                              "            {var}_best = {var}_run",
                              "    {var} = {var}_best >= {k}"]),
    Predicate("listint_distinct_count_ge_chunky", "nums has at least {k} distinct values",
              lambda nums, k: len(set(nums)) >= k,
              datatype="list[int]", param_space={"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "distinct"], variable_name="has_enough_distinct",
              multiline_impl=["    {var}_seen = []",
                              "    for x in nums:",
                              "        if x not in {var}_seen:",
                              "            {var}_seen.append(x)",
                              "    {var} = len({var}_seen) >= {k}"]),
    Predicate("listint_contains_zero_chunky", "nums contains a zero",
              lambda nums: any(x == 0 for x in nums),
              datatype="list[int]", id_substrs=["contains", "zero"], variable_name="contains_zero",
              multiline_impl=["    {var} = False",
                              "    for x in nums:",
                              "        if x == 0:",
                              "            {var} = True",
                              "            break"]),
    Predicate("listint_first_equals_last_chunky", "the first element of nums equals its last element",
              lambda nums: len(nums) > 0 and nums[0] == nums[-1],
              datatype="list[int]", id_substrs=["first element", "last element"],
              variable_name="first_equals_last",
              multiline_impl=["    {var} = False",
                              "    if len(nums) > 0:",
                              "        {var}_first = nums[0]",
                              "        {var}_last = nums[-1]",
                              "        {var} = {var}_first == {var}_last"]),
    Predicate("listint_mean_gt_chunky", "the mean of nums is greater than {t}",
              lambda nums, t: len(nums) > 0 and sum(nums) / len(nums) > t,
              datatype="list[int]", param_space={"t": [-1, 0, 5, 10]},
              id_substrs=["mean", "greater than", "{t}"], variable_name="mean_is_large",
              multiline_impl=["    {var} = False",
                              "    if len(nums) > 0:",
                              "        {var}_acc = 0",
                              "        for x in nums:",
                              "            {var}_acc += x",
                              "        {var} = {var}_acc / len(nums) > {t}"]),
    Predicate("listint_all_multiples_chunky", "every element of nums is a multiple of {k}",
              lambda nums, k: all(x % k == 0 for x in nums),
              datatype="list[int]", param_space={"k": [2, 3, 5]},
              id_substrs=["every", "multiple", "{k}"], variable_name="all_multiples_of_k",
              multiline_impl=["    {var} = True",
                              "    for x in nums:",
                              "        if x % {k} != 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("listint_count_positive_ge_chunky", "nums contains at least {k} positive numbers",
              lambda nums, k: sum(1 for x in nums if x > 0) >= k,
              datatype="list[int]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "positive"], variable_name="has_enough_positives",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        if x > 0:",
                              "            {var}_acc += 1",
                              "    {var} = {var}_acc >= {k}"]),
    Predicate("listint_sum_divisible_chunky", "the sum of nums is divisible by {k}",
              lambda nums, k: sum(nums) % k == 0,
              datatype="list[int]", param_space={"k": [2, 3, 5]},
              id_substrs=["sum", "divisible", "{k}"], variable_name="sum_is_divisible",
              multiline_impl=["    {var}_acc = 0",
                              "    for x in nums:",
                              "        {var}_acc += x",
                              "    {var} = {var}_acc % {k} == 0"]),
    Predicate("listint_all_less_than_chunky", "every element of nums is less than {t}",
              lambda nums, t: all(x < t for x in nums),
              datatype="list[int]", param_space={"t": [10, 30, 50, 60]},
              id_substrs=["every", "less than", "{t}"], variable_name="all_less_than_t",
              multiline_impl=["    {var} = True",
                              "    for x in nums:",
                              "        if x >= {t}:",
                              "            {var} = False",
                              "            break"]),
    # ── list[str] (arg words) ──
    Predicate("liststr_all_lowercase_chunky", "every word in words is entirely lowercase",
              lambda words: all(w == w.lower() and w != "" for w in words),
              datatype="list[str]", id_substrs=["every word", "lowercase"], variable_name="all_lowercase",
              multiline_impl=["    {var} = True",
                              "    for w in words:",
                              "        if w == '' or w != w.lower():",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_same_length_chunky", "all words in words have the same length",
              lambda words: len(set(len(w) for w in words)) <= 1,
              datatype="list[str]", id_substrs=["all words", "same length"], variable_name="all_same_length",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(words)):",
                              "        if len(words[i]) != len(words[0]):",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_sorted_alphabetically_chunky", "the words in words are sorted alphabetically (non-decreasing)",
              lambda words: all(words[i] <= words[i + 1] for i in range(len(words) - 1)),
              datatype="list[str]", id_substrs=["sorted alphabetically", "non-decreasing"],
              variable_name="is_sorted_alphabetically",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(words) - 1):",
                              "        if words[i] > words[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_has_duplicate_chunky", "words contains at least one duplicate word",
              lambda words: len(set(words)) != len(words),
              datatype="list[str]", id_substrs=["at least one duplicate"], variable_name="has_duplicate",
              multiline_impl=["    {var} = False",
                              "    {var}_seen = set()",
                              "    for w in words:",
                              "        if w in {var}_seen:",
                              "            {var} = True",
                              "            break",
                              "        {var}_seen.add(w)"]),
    Predicate("liststr_count_ge_chunky", "words contains at least {k} words",
              lambda words, k: len(words) >= k,
              datatype="list[str]", param_space={"k": [2, 3, 4]},
              id_substrs=["at least", "{k}", "words"], variable_name="has_enough_words",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("liststr_total_chars_gt_chunky", "the total number of characters across all words is greater than {t}",
              lambda words, t: sum(len(w) for w in words) > t,
              datatype="list[str]", param_space={"t": [5, 10, 15]},
              id_substrs=["total number of characters", "greater than", "{t}"],
              variable_name="has_many_characters",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        {var} += len(w)",
                              "    {var} = {var} > {t}"]),
    Predicate("liststr_longest_word_gt_chunky", "the longest word in words has more than {t} characters",
              lambda words, t: max((len(w) for w in words), default=0) > t,
              datatype="list[str]", param_space={"t": [2, 3, 4]},
              id_substrs=["longest word", "more than", "{t}", "characters"],
              variable_name="has_long_word",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        if len(w) > {var}:",
                              "            {var} = len(w)",
                              "    {var} = {var} > {t}"]),
    Predicate("liststr_all_distinct_chunky", "all words in words are distinct",
              lambda words: len(set(words)) == len(words),
              datatype="list[str]", id_substrs=["all words", "distinct"], variable_name="all_distinct",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = set()",
                              "    for w in words:",
                              "        if w in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.add(w)"]),
    Predicate("liststr_has_palindrome_chunky", "at least one word in words is a palindrome",
              lambda words: any(w == w[::-1] for w in words),
              datatype="list[str]", id_substrs=["at least one word", "palindrome"],
              variable_name="has_palindrome",
              multiline_impl=["    {var} = False",
                              "    for w in words:",
                              "        if w == w[::-1]:",
                              "            {var} = True",
                              "            break"]),
    Predicate("liststr_all_start_vowel_chunky", "every word in words starts with a vowel",
              lambda words: all(w != "" and w[0] in "aeiou" for w in words),
              datatype="list[str]", id_substrs=["every word", "starts with a vowel"],
              variable_name="all_start_with_vowel",
              multiline_impl=["    {var} = True",
                              "    for w in words:",
                              "        if w == '' or w[0] not in 'aeiou':",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_increasing_length_chunky", "the words in words have strictly increasing length",
              lambda words: all(len(words[i]) < len(words[i + 1]) for i in range(len(words) - 1)),
              datatype="list[str]", id_substrs=["strictly increasing length"],
              variable_name="has_increasing_length",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(words) - 1):",
                              "        if len(words[i]) >= len(words[i + 1]):",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_count_with_e_ge_chunky", "at least {k} words in words contain the letter 'e'",
              lambda words, k: sum(1 for w in words if "e" in w) >= k,
              datatype="list[str]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "letter 'e'"], variable_name="has_words_with_e",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        if 'e' in w:",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("liststr_all_len_le_chunky", "every word in words has length at most {t}",
              lambda words, t: all(len(w) <= t for w in words),
              datatype="list[str]", param_space={"t": [2, 3, 4]},
              id_substrs=["every word", "length at most", "{t}"], variable_name="all_short_enough",
              multiline_impl=["    {var} = True",
                              "    for w in words:",
                              "        if len(w) > {t}:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_all_first_letters_distinct_chunky", "all first letters of the words in words are distinct",
              lambda words: len(set(w[0] for w in words if w)) == len([w for w in words if w]),
              datatype="list[str]", id_substrs=["all first letters", "distinct"],
              variable_name="first_letters_distinct",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = set()",
                              "    for w in words:",
                              "        if w == '':",
                              "            continue",
                              "        if w[0] in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.add(w[0])"]),
    Predicate("liststr_count_long_words_ge_chunky", "at least {k} words in words are longer than {t} characters",
              lambda words, k, t: sum(1 for w in words if len(w) > t) >= k,
              datatype="list[str]", param_space={"k": [1, 2], "t": [2, 3]},
              id_substrs=["at least", "{k}", "longer than", "{t}"], variable_name="has_enough_long_words",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        if len(w) > {t}:",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("liststr_reverse_pair_chunky", "words contains a word that is the reverse of another word",
              lambda words: any(i != j and words[i] == words[j][::-1]
                                for i in range(len(words)) for j in range(len(words))),
              datatype="list[str]", id_substrs=["reverse of another word"], variable_name="has_reverse_pair",
              multiline_impl=["    {var} = False",
                              "    for i in range(len(words)):",
                              "        for j in range(len(words)):",
                              "            if i != j and words[i] == words[j][::-1]:",
                              "                {var} = True"]),
    Predicate("liststr_no_repeated_char_words_chunky", "every word in words has no repeated character",
              lambda words: all(len(set(w)) == len(w) for w in words),
              datatype="list[str]", id_substrs=["every word", "no repeated character"],
              variable_name="all_chars_unique",
              multiline_impl=["    {var} = True",
                              "    for w in words:",
                              "        if len(set(w)) != len(w):",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_distinct_first_letters_ge_chunky", "the number of distinct first letters in words is at least {k}",
              lambda words, k: len(set(w[0] for w in words if w)) >= k,
              datatype="list[str]", param_space={"k": [2, 3]},
              id_substrs=["distinct first letters", "at least", "{k}"],
              variable_name="has_many_first_letters",
              multiline_impl=["    {var}_seen = set()",
                              "    for w in words:",
                              "        if w != '':",
                              "            {var}_seen.add(w[0])",
                              "    {var} = len({var}_seen) >= {k}"]),
    Predicate("liststr_first_precedes_last_chunky", "the first word alphabetically precedes the last word in words",
              lambda words: len(words) >= 2 and words[0] < words[-1],
              datatype="list[str]", id_substrs=["first word", "precedes the last word"],
              variable_name="first_precedes_last",
              multiline_impl=["    {var} = False",
                              "    if len(words) >= 2:",
                              "        {var}_a = words[0]",
                              "        {var}_b = words[len(words) - 1]",
                              "        {var} = {var}_a < {var}_b"]),
    Predicate("liststr_all_contain_vowel_chunky", "every word in words contains at least one vowel",
              lambda words: all(any(c in "aeiou" for c in w) for w in words),
              datatype="list[str]", id_substrs=["every word", "at least one vowel"],
              variable_name="all_have_vowel",
              multiline_impl=["    {var} = True",
                              "    for w in words:",
                              "        {var}_has = False",
                              "        for c in w:",
                              "            if c in 'aeiou':",
                              "                {var}_has = True",
                              "                break",
                              "        if not {var}_has:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_total_vowels_ge_chunky", "the total number of vowels across all words is at least {k}",
              lambda words, k: sum(1 for w in words for c in w if c in "aeiou") >= k,
              datatype="list[str]", param_space={"k": [2, 3, 4]},
              id_substrs=["total number of vowels", "at least", "{k}"], variable_name="has_enough_vowels",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        for c in w:",
                              "            if c in 'aeiou':",
                              "                {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("liststr_any_word_repeats_letter_chunky", "at least one word in words has a repeated character",
              lambda words: any(len(set(w)) != len(w) for w in words),
              datatype="list[str]", id_substrs=["at least one word", "repeated character"],
              variable_name="has_word_with_repeat",
              multiline_impl=["    {var} = False",
                              "    for w in words:",
                              "        if len(set(w)) != len(w):",
                              "            {var} = True",
                              "            break"]),
    Predicate("liststr_consecutive_duplicate_chunky", "words contains two equal words next to each other",
              lambda words: any(words[i] == words[i + 1] for i in range(len(words) - 1)),
              datatype="list[str]", id_substrs=["two equal words next to each other"],
              variable_name="has_consecutive_duplicate",
              multiline_impl=["    {var} = False",
                              "    for i in range(len(words) - 1):",
                              "        if words[i] == words[i + 1]:",
                              "            {var} = True",
                              "            break"]),
    Predicate("liststr_shortest_word_lt_chunky", "the shortest word in words has fewer than {t} characters",
              lambda words, t: len(words) > 0 and min(len(w) for w in words) < t,
              datatype="list[str]", param_space={"t": [3, 4]},
              id_substrs=["shortest word", "fewer than", "{t}", "characters"],
              variable_name="has_short_word",
              multiline_impl=["    {var}_min = -1",
                              "    for w in words:",
                              "        if {var}_min == -1 or len(w) < {var}_min:",
                              "            {var}_min = len(w)",
                              "    {var} = {var}_min != -1 and {var}_min < {t}"]),
    Predicate("liststr_contains_word_len_eq_chunky", "words contains a word of exactly {t} characters",
              lambda words, t: any(len(w) == t for w in words),
              datatype="list[str]", param_space={"t": [2, 3, 4]},
              id_substrs=["a word of exactly", "{t}", "characters"], variable_name="has_word_of_length",
              multiline_impl=["    {var} = False",
                              "    for w in words:",
                              "        if len(w) == {t}:",
                              "            {var} = True",
                              "            break"]),
    Predicate("liststr_all_end_same_letter_chunky", "all words in words end with the same letter",
              lambda words: len(set(w[-1] for w in words if w)) <= 1,
              datatype="list[str]", id_substrs=["all words", "end with the same letter"],
              variable_name="all_end_same",
              multiline_impl=["    {var} = True",
                              "    {var}_last = None",
                              "    for w in words:",
                              "        if w == '':",
                              "            continue",
                              "        if {var}_last is None:",
                              "            {var}_last = w[-1]",
                              "        elif w[-1] != {var}_last:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_count_starting_c_ge_chunky", "at least {k} words in words start with the letter 'c'",
              lambda words, k: sum(1 for w in words if w.startswith("c")) >= k,
              datatype="list[str]", param_space={"k": [1, 2]},
              id_substrs=["at least", "{k}", "start with the letter 'c'"],
              variable_name="has_words_starting_c",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        if w.startswith('c'):",
                              "            {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("liststr_reverse_sorted_chunky", "the words in words are sorted in non-increasing alphabetical order",
              lambda words: all(words[i] >= words[i + 1] for i in range(len(words) - 1)),
              datatype="list[str]", id_substrs=["non-increasing alphabetical order"],
              variable_name="is_reverse_sorted",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(words) - 1):",
                              "        if words[i] < words[i + 1]:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_total_length_even_chunky", "the total number of characters across all words is even",
              lambda words: sum(len(w) for w in words) % 2 == 0,
              datatype="list[str]", id_substrs=["total number of characters", "is even"],
              variable_name="total_length_is_even",
              multiline_impl=["    {var} = 0",
                              "    for w in words:",
                              "        {var} += len(w)",
                              "    {var} = {var} % 2 == 0"]),
    Predicate("liststr_adjacent_share_letter_chunky", "every adjacent pair of words in words shares at least one common letter",
              lambda words: all(len(set(words[i]) & set(words[i + 1])) > 0 for i in range(len(words) - 1)),
              datatype="list[str]", id_substrs=["every adjacent pair", "common letter"],
              variable_name="adjacent_share_letter",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(words) - 1):",
                              "        {var}_common = set(words[i]) & set(words[i + 1])",
                              "        if len({var}_common) == 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("liststr_count_distinct_ge_chunky", "the number of distinct words in words is at least {k}",
              lambda words, k: len(set(words)) >= k,
              datatype="list[str]", param_space={"k": [2, 3]},
              id_substrs=["number of distinct words", "at least", "{k}"], variable_name="has_enough_distinct",
              multiline_impl=["    {var}_seen = set()",
                              "    for w in words:",
                              "        {var}_seen.add(w)",
                              "    {var} = len({var}_seen) >= {k}"]),
    Predicate("liststr_has_word_with_letter_chunky", "at least one word in words contains the letter '{c}'",
              lambda words, c: any(c in w for w in words),
              datatype="list[str]", param_space={"c": ["a", "e", "o", "t"]},
              id_substrs=["at least one word", "letter", "'{c}'"], variable_name="has_word_with_letter",
              multiline_impl=["    {var} = False",
                              "    for w in words:",
                              "        if '{c}' in w:",
                              "            {var} = True",
                              "            break"]),
    # ── list[list[int]] (arg grid) ──
    Predicate("matrix_symmetric_chunky", "grid is symmetric (grid[i][j] equals grid[j][i] for all i, j)",
              lambda grid: all(grid[i][j] == grid[j][i] for i in range(len(grid)) for j in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["symmetric"], variable_name="is_symmetric",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        for j in range(len(grid)):",
                              "            if grid[i][j] != grid[j][i]:",
                              "                {var} = False"]),
    Predicate("matrix_all_positive_chunky", "all entries of grid are positive",
              lambda grid: all(x > 0 for row in grid for x in row),
              datatype="list[list[int]]", id_substrs=["all entries", "positive"], variable_name="all_positive",
              multiline_impl=["    {var} = True",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x <= 0:",
                              "                {var} = False"]),
    Predicate("matrix_contains_zero_chunky", "grid contains a zero entry",
              lambda grid: any(x == 0 for row in grid for x in row),
              datatype="list[list[int]]", id_substrs=["contains a zero"], variable_name="has_zero",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x == 0:",
                              "                {var} = True"]),
    Predicate("matrix_contains_negative_chunky", "grid contains a negative entry",
              lambda grid: any(x < 0 for row in grid for x in row),
              datatype="list[list[int]]", id_substrs=["contains a negative"], variable_name="has_negative",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x < 0:",
                              "                {var} = True"]),
    Predicate("matrix_diagonal_all_positive_chunky", "the main diagonal of grid is all positive",
              lambda grid: all(grid[i][i] > 0 for i in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["main diagonal", "positive"], variable_name="diagonal_all_positive",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        if grid[i][i] <= 0:",
                              "            {var} = False"]),
    Predicate("matrix_trace_gt_chunky", "the trace of grid is greater than {t}",
              lambda grid, t: sum(grid[i][i] for i in range(len(grid))) > t,
              datatype="list[list[int]]", param_space={"t": [0, 5, 10]},
              id_substrs=["trace", "{t}"], variable_name="trace_exceeds",
              multiline_impl=["    {var} = 0",
                              "    for i in range(len(grid)):",
                              "        {var} += grid[i][i]",
                              "    {var} = {var} > {t}"]),
    Predicate("matrix_has_zero_row_chunky", "grid has a row that is all zeros",
              lambda grid: any(all(x == 0 for x in row) for row in grid),
              datatype="list[list[int]]", id_substrs=["row", "all zeros"], variable_name="has_zero_row",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        {var}_all = True",
                              "        for x in row:",
                              "            if x != 0:",
                              "                {var}_all = False",
                              "        if {var}_all:",
                              "            {var} = True"]),
    Predicate("matrix_has_zero_column_chunky", "grid has a column that is all zeros",
              lambda grid: any(all(grid[i][j] == 0 for i in range(len(grid))) for j in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["column", "all zeros"], variable_name="has_zero_column",
              multiline_impl=["    {var} = False",
                              "    for j in range(len(grid)):",
                              "        {var}_all = True",
                              "        for i in range(len(grid)):",
                              "            if grid[i][j] != 0:",
                              "                {var}_all = False",
                              "        if {var}_all:",
                              "            {var} = True"]),
    Predicate("matrix_every_row_sorted_chunky", "every row of grid is sorted in ascending order",
              lambda grid: all(row[k] <= row[k + 1] for row in grid for k in range(len(row) - 1)),
              datatype="list[list[int]]", id_substrs=["every row", "ascending"], variable_name="rows_sorted",
              multiline_impl=["    {var} = True",
                              "    for row in grid:",
                              "        for k in range(len(row) - 1):",
                              "            if row[k] > row[k + 1]:",
                              "                {var} = False"]),
    Predicate("matrix_every_column_sorted_chunky", "every column of grid is sorted in ascending order",
              lambda grid: all(grid[i][j] <= grid[i + 1][j] for j in range(len(grid)) for i in range(len(grid) - 1)),
              datatype="list[list[int]]", id_substrs=["every column", "ascending"], variable_name="columns_sorted",
              multiline_impl=["    {var} = True",
                              "    for j in range(len(grid)):",
                              "        for i in range(len(grid) - 1):",
                              "            if grid[i][j] > grid[i + 1][j]:",
                              "                {var} = False"]),
    Predicate("matrix_all_row_sums_equal_chunky", "all row sums of grid are equal",
              lambda grid: all(sum(row) == sum(grid[0]) for row in grid),
              datatype="list[list[int]]", id_substrs=["all row sums", "equal"], variable_name="row_sums_equal",
              multiline_impl=["    {var} = True",
                              "    {var}_target = None",
                              "    for row in grid:",
                              "        {var}_sum = 0",
                              "        for x in row:",
                              "            {var}_sum += x",
                              "        if {var}_target is None:",
                              "            {var}_target = {var}_sum",
                              "        elif {var}_sum != {var}_target:",
                              "            {var} = False"]),
    Predicate("matrix_all_column_sums_equal_chunky", "all column sums of grid are equal",
              lambda grid: all(sum(grid[i][j] for i in range(len(grid))) == sum(grid[i][0] for i in range(len(grid))) for j in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["all column sums", "equal"], variable_name="column_sums_equal",
              multiline_impl=["    {var} = True",
                              "    {var}_target = None",
                              "    for j in range(len(grid)):",
                              "        {var}_sum = 0",
                              "        for i in range(len(grid)):",
                              "            {var}_sum += grid[i][j]",
                              "        if {var}_target is None:",
                              "            {var}_target = {var}_sum",
                              "        elif {var}_sum != {var}_target:",
                              "            {var} = False"]),
    Predicate("matrix_max_entry_gt_chunky", "the maximum entry of grid is greater than {t}",
              lambda grid, t: any(x > t for row in grid for x in row),
              datatype="list[list[int]]", param_space={"t": [3, 5, 7]},
              id_substrs=["maximum entry", "{t}"], variable_name="max_exceeds",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x > {t}:",
                              "                {var} = True"]),
    Predicate("matrix_strictly_increasing_diagonal_chunky", "the main diagonal of grid is strictly increasing",
              lambda grid: all(grid[i][i] < grid[i + 1][i + 1] for i in range(len(grid) - 1)),
              datatype="list[list[int]]", id_substrs=["main diagonal", "strictly increasing"], variable_name="diagonal_increasing",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid) - 1):",
                              "        if grid[i][i] >= grid[i + 1][i + 1]:",
                              "            {var} = False"]),
    Predicate("matrix_anti_diagonal_sum_gt_chunky", "the anti-diagonal sum of grid is greater than {t}",
              lambda grid, t: sum(grid[i][len(grid) - 1 - i] for i in range(len(grid))) > t,
              datatype="list[list[int]]", param_space={"t": [0, 5, 8]},
              id_substrs=["anti-diagonal sum", "{t}"], variable_name="anti_diagonal_large",
              multiline_impl=["    {var} = 0",
                              "    for i in range(len(grid)):",
                              "        {var} += grid[i][len(grid) - 1 - i]",
                              "    {var} = {var} > {t}"]),
    Predicate("matrix_contains_value_chunky", "grid contains the value {k}",
              lambda grid, k: any(x == k for row in grid for x in row),
              datatype="list[list[int]]", param_space={"k": [0, 1, 2, 5]},
              id_substrs=["contains the value", "{k}"], variable_name="contains_value",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x == {k}:",
                              "                {var} = True"]),
    Predicate("matrix_all_rows_distinct_sums_chunky", "all rows of grid have distinct sums",
              lambda grid: len(set(sum(row) for row in grid)) == len(grid),
              datatype="list[list[int]]", id_substrs=["all rows", "distinct sums"], variable_name="rows_distinct_sums",
              multiline_impl=["    {var}_sums = []",
                              "    for row in grid:",
                              "        {var}_s = 0",
                              "        for x in row:",
                              "            {var}_s += x",
                              "        {var}_sums.append({var}_s)",
                              "    {var} = len(set({var}_sums)) == len({var}_sums)"]),
    Predicate("matrix_upper_triangle_zero_chunky", "every entry above the main diagonal of grid is zero",
              lambda grid: all(grid[i][j] == 0 for i in range(len(grid)) for j in range(i + 1, len(grid))),
              datatype="list[list[int]]", id_substrs=["above the main diagonal", "zero"], variable_name="upper_triangle_zero",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        for j in range(i + 1, len(grid)):",
                              "            if grid[i][j] != 0:",
                              "                {var} = False"]),
    Predicate("matrix_lower_triangle_zero_chunky", "every entry below the main diagonal of grid is zero",
              lambda grid: all(grid[i][j] == 0 for i in range(len(grid)) for j in range(i)),
              datatype="list[list[int]]", id_substrs=["below the main diagonal", "zero"], variable_name="lower_triangle_zero",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        for j in range(i):",
                              "            if grid[i][j] != 0:",
                              "                {var} = False"]),
    Predicate("matrix_identity_like_chunky", "grid is an identity-like matrix (1 on the diagonal, 0 elsewhere)",
              lambda grid: all(grid[i][j] == (1 if i == j else 0) for i in range(len(grid)) for j in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["identity-like"], variable_name="is_identity_like",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        for j in range(len(grid)):",
                              "            {var}_expected = 1 if i == j else 0",
                              "            if grid[i][j] != {var}_expected:",
                              "                {var} = False"]),
    Predicate("matrix_every_row_palindrome_chunky", "every row of grid is a palindrome",
              lambda grid: all(row == row[::-1] for row in grid),
              datatype="list[list[int]]", id_substrs=["every row", "palindrome"], variable_name="rows_palindromic",
              multiline_impl=["    {var} = True",
                              "    for row in grid:",
                              "        for k in range(len(row) // 2):",
                              "            if row[k] != row[len(row) - 1 - k]:",
                              "                {var} = False"]),
    Predicate("matrix_count_positive_ge_chunky", "grid has at least {k} positive entries",
              lambda grid, k: sum(1 for row in grid for x in row if x > 0) >= k,
              datatype="list[list[int]]", param_space={"k": [2, 3, 4, 5]},
              id_substrs=["at least", "{k}", "positive"], variable_name="enough_positives",
              multiline_impl=["    {var} = 0",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x > 0:",
                              "                {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("matrix_all_diagonal_equal_chunky", "all main-diagonal entries of grid are equal",
              lambda grid: all(grid[i][i] == grid[0][0] for i in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["main-diagonal entries", "equal"], variable_name="diagonal_all_equal",
              multiline_impl=["    {var} = True",
                              "    {var}_target = None",
                              "    for i in range(len(grid)):",
                              "        if {var}_target is None:",
                              "            {var}_target = grid[i][i]",
                              "        elif grid[i][i] != {var}_target:",
                              "            {var} = False"]),
    Predicate("matrix_some_row_strictly_increasing_chunky", "some row of grid is strictly increasing",
              lambda grid: any(all(row[k] < row[k + 1] for k in range(len(row) - 1)) for row in grid),
              datatype="list[list[int]]", id_substrs=["some row", "strictly increasing"], variable_name="some_row_increasing",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        {var}_inc = True",
                              "        for k in range(len(row) - 1):",
                              "            if row[k] >= row[k + 1]:",
                              "                {var}_inc = False",
                              "        if {var}_inc:",
                              "            {var} = True"]),
    Predicate("matrix_count_zero_ge_chunky", "grid has at least {k} zero entries",
              lambda grid, k: sum(1 for row in grid for x in row if x == 0) >= k,
              datatype="list[list[int]]", param_space={"k": [1, 2]},
              id_substrs=["at least", "{k}", "zero entries"], variable_name="enough_zeros",
              multiline_impl=["    {var} = 0",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x == 0:",
                              "                {var} += 1",
                              "    {var} = {var} >= {k}"]),
    Predicate("matrix_antisymmetric_chunky", "grid equals the negation of its transpose (grid[i][j] equals -grid[j][i])",
              lambda grid: all(grid[i][j] == -grid[j][i] for i in range(len(grid)) for j in range(len(grid))),
              datatype="list[list[int]]", id_substrs=["negation of its transpose"], variable_name="is_antisymmetric",
              multiline_impl=["    {var} = True",
                              "    for i in range(len(grid)):",
                              "        for j in range(len(grid)):",
                              "            if grid[i][j] != -grid[j][i]:",
                              "                {var} = False"]),
    Predicate("matrix_constant_grid_chunky", "all entries of grid are equal",
              lambda grid: all(x == grid[0][0] for row in grid for x in row),
              datatype="list[list[int]]", id_substrs=["all entries", "equal"], variable_name="all_equal",
              multiline_impl=["    {var} = True",
                              "    {var}_target = None",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if {var}_target is None:",
                              "                {var}_target = x",
                              "            elif x != {var}_target:",
                              "                {var} = False"]),
    Predicate("matrix_row_sums_all_even_chunky", "every row sum of grid is even",
              lambda grid: all(sum(row) % 2 == 0 for row in grid),
              datatype="list[list[int]]", id_substrs=["every row sum", "even"], variable_name="row_sums_even",
              multiline_impl=["    {var} = True",
                              "    for row in grid:",
                              "        {var}_sum = 0",
                              "        for x in row:",
                              "            {var}_sum += x",
                              "        if {var}_sum % 2 != 0:",
                              "            {var} = False"]),
    Predicate("matrix_all_entries_distinct_chunky", "all entries of grid are distinct",
              lambda grid: len(set(x for row in grid for x in row)) == sum(len(row) for row in grid),
              datatype="list[list[int]]", id_substrs=["all entries", "distinct"], variable_name="all_distinct",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for row in grid:",
                              "        for x in row:",
                              "            if x in {var}_seen:",
                              "                {var} = False",
                              "            {var}_seen.append(x)"]),
    Predicate("matrix_max_row_sum_gt_chunky", "the maximum row sum of grid is greater than {t}",
              lambda grid, t: any(sum(row) > t for row in grid),
              datatype="list[list[int]]", param_space={"t": [0, 5, 10]},
              id_substrs=["maximum row sum", "{t}"], variable_name="max_row_sum_large",
              multiline_impl=["    {var} = False",
                              "    for row in grid:",
                              "        {var}_sum = 0",
                              "        for x in row:",
                              "            {var}_sum += x",
                              "        if {var}_sum > {t}:",
                              "            {var} = True"]),
    # ── dict[str, int] (arg counts) ──
    Predicate("dict_all_values_positive_chunky", "every value in counts is positive",
              lambda counts: all(v > 0 for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["every value", "positive"],
              variable_name="all_values_positive",
              multiline_impl=["    {var} = True",
                              "    for v in counts.values():",
                              "        if v <= 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("dict_sum_values_gt_chunky", "the sum of the values in counts is greater than {t}",
              lambda counts, t: sum(counts.values()) > t,
              datatype="dict[str, int]", param_space={"t": [0, 5, 10, 20, 30]},
              id_substrs=["sum", "values", "greater than", "{t}"],
              variable_name="sum_values_large",
              multiline_impl=["    {var}_total = 0",
                              "    for v in counts.values():",
                              "        {var}_total += v",
                              "    {var} = {var}_total > {t}"]),
    Predicate("dict_max_value_gt_chunky", "the largest value in counts is greater than {t}",
              lambda counts, t: len(counts) > 0 and max(counts.values()) > t,
              datatype="dict[str, int]", param_space={"t": [0, 5, 10, 15]},
              id_substrs=["largest value", "greater than", "{t}"],
              variable_name="max_value_large",
              multiline_impl=["    {var}_max = None",
                              "    for v in counts.values():",
                              "        if {var}_max is None or v > {var}_max:",
                              "            {var}_max = v",
                              "    {var} = {var}_max is not None and {var}_max > {t}"]),
    Predicate("dict_min_value_lt_chunky", "the smallest value in counts is less than {t}",
              lambda counts, t: len(counts) > 0 and min(counts.values()) < t,
              datatype="dict[str, int]", param_space={"t": [-1, 0, 1, 5]},
              id_substrs=["smallest value", "less than", "{t}"],
              variable_name="min_value_small",
              multiline_impl=["    {var}_min = None",
                              "    for v in counts.values():",
                              "        if {var}_min is None or v < {var}_min:",
                              "            {var}_min = v",
                              "    {var} = {var}_min is not None and {var}_min < {t}"]),
    Predicate("dict_num_keys_ge_chunky", "counts has at least {k} keys",
              lambda counts, k: len(counts) >= k,
              datatype="dict[str, int]", param_space={"k": [1, 2, 3, 4]},
              id_substrs=["at least", "{k}", "keys"],
              variable_name="has_enough_keys",
              multiline_impl=["    {var}_count = 0",
                              "    for k in counts:",
                              "        {var}_count += 1",
                              "    {var} = {var}_count >= {k}"]),
    Predicate("dict_has_value_eq_key_length_chunky", "some key in counts has a value equal to the length of that key",
              lambda counts: any(v == len(k) for k, v in counts.items()),
              datatype="dict[str, int]", id_substrs=["value equal to the length", "key"],
              variable_name="has_value_matching_key_length",
              multiline_impl=["    {var} = False",
                              "    for k, v in counts.items():",
                              "        if v == len(k):",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_all_values_distinct_chunky", "all values in counts are distinct",
              lambda counts: len(set(counts.values())) == len(counts),
              datatype="dict[str, int]", id_substrs=["all values", "distinct"],
              variable_name="all_values_distinct",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for v in counts.values():",
                              "        if v in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.append(v)"]),
    Predicate("dict_has_negative_value_chunky", "counts has at least one negative value",
              lambda counts: any(v < 0 for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["at least one negative value"],
              variable_name="has_negative_value",
              multiline_impl=["    {var} = False",
                              "    for v in counts.values():",
                              "        if v < 0:",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_all_keys_short_chunky", "every key in counts has length at most {t}",
              lambda counts, t: all(len(k) <= t for k in counts),
              datatype="dict[str, int]", param_space={"t": [3, 4, 5]},
              id_substrs=["every key", "length at most", "{t}"],
              variable_name="all_keys_short",
              multiline_impl=["    {var} = True",
                              "    for k in counts:",
                              "        if len(k) > {t}:",
                              "            {var} = False",
                              "            break"]),
    Predicate("dict_sum_values_even_chunky", "the sum of the values in counts is even",
              lambda counts: sum(counts.values()) % 2 == 0,
              datatype="dict[str, int]", id_substrs=["sum", "values", "even"],
              variable_name="sum_values_even",
              multiline_impl=["    {var}_total = 0",
                              "    for v in counts.values():",
                              "        {var}_total += v",
                              "    {var} = {var}_total % 2 == 0"]),
    Predicate("dict_count_positive_ge_chunky", "at least {k} values in counts are positive",
              lambda counts, k: sum(1 for v in counts.values() if v > 0) >= k,
              datatype="dict[str, int]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "positive"],
              variable_name="enough_positive_values",
              multiline_impl=["    {var}_count = 0",
                              "    for v in counts.values():",
                              "        if v > 0:",
                              "            {var}_count += 1",
                              "    {var} = {var}_count >= {k}"]),
    Predicate("dict_has_value_zero_chunky", "counts has at least one value equal to zero",
              lambda counts: any(v == 0 for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["value equal to zero"],
              variable_name="has_zero_value",
              multiline_impl=["    {var} = False",
                              "    for v in counts.values():",
                              "        if v == 0:",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_all_values_even_chunky", "every value in counts is even",
              lambda counts: all(v % 2 == 0 for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["every value", "even"],
              variable_name="all_values_even",
              multiline_impl=["    {var} = True",
                              "    for v in counts.values():",
                              "        if v % 2 != 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("dict_has_long_key_chunky", "counts has a key longer than {t} characters",
              lambda counts, t: any(len(k) > t for k in counts),
              datatype="dict[str, int]", param_space={"t": [2, 3, 4]},
              id_substrs=["key longer than", "{t}", "characters"],
              variable_name="has_long_key",
              multiline_impl=["    {var} = False",
                              "    for k in counts:",
                              "        if len(k) > {t}:",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_mean_value_gt_chunky", "the mean of the values in counts is greater than {t}",
              lambda counts, t: len(counts) > 0 and sum(counts.values()) > t * len(counts),
              datatype="dict[str, int]", param_space={"t": [0, 5, 10]},
              id_substrs=["mean", "values", "greater than", "{t}"],
              variable_name="mean_value_large",
              multiline_impl=["    {var}_total = 0",
                              "    {var}_count = 0",
                              "    for v in counts.values():",
                              "        {var}_total += v",
                              "        {var}_count += 1",
                              "    {var} = {var}_count > 0 and {var}_total > {t} * {var}_count"]),
    Predicate("dict_value_range_gt_chunky", "the range of the values in counts (largest minus smallest) is greater than {t}",
              lambda counts, t: len(counts) > 0 and (max(counts.values()) - min(counts.values())) > t,
              datatype="dict[str, int]", param_space={"t": [0, 5, 10]},
              id_substrs=["range of the values", "greater than", "{t}"],
              variable_name="value_range_large",
              multiline_impl=["    {var}_max = None",
                              "    {var}_min = None",
                              "    for v in counts.values():",
                              "        if {var}_max is None or v > {var}_max:",
                              "            {var}_max = v",
                              "        if {var}_min is None or v < {var}_min:",
                              "            {var}_min = v",
                              "    {var} = {var}_max is not None and {var}_max - {var}_min > {t}"]),
    Predicate("dict_at_least_k_exceed_t_chunky", "at least {k} values in counts exceed {t}",
              lambda counts, k, t: sum(1 for v in counts.values() if v > t) >= k,
              datatype="dict[str, int]", param_space={"k": [1, 2], "t": [5, 10]},
              id_substrs=["at least", "{k}", "exceed", "{t}"],
              variable_name="enough_large_values",
              multiline_impl=["    {var}_count = 0",
                              "    for v in counts.values():",
                              "        if v > {t}:",
                              "            {var}_count += 1",
                              "    {var} = {var}_count >= {k}"]),
    Predicate("dict_all_keys_distinct_length_chunky", "all keys in counts have distinct lengths",
              lambda counts: len(set(len(k) for k in counts)) == len(counts),
              datatype="dict[str, int]", id_substrs=["all keys", "distinct lengths"],
              variable_name="all_keys_distinct_length",
              multiline_impl=["    {var} = True",
                              "    {var}_seen = []",
                              "    for k in counts:",
                              "        if len(k) in {var}_seen:",
                              "            {var} = False",
                              "            break",
                              "        {var}_seen.append(len(k))"]),
    Predicate("dict_total_divisible_chunky", "the total of the values in counts is divisible by {k}",
              lambda counts, k: sum(counts.values()) % k == 0,
              datatype="dict[str, int]", param_space={"k": [2, 3, 4, 5]},
              id_substrs=["total", "values", "divisible by", "{k}"],
              variable_name="total_divisible",
              multiline_impl=["    {var}_total = 0",
                              "    for v in counts.values():",
                              "        {var}_total += v",
                              "    {var} = {var}_total % {k} == 0"]),
    Predicate("dict_some_value_eq_num_keys_chunky", "some value in counts equals the number of keys",
              lambda counts: any(v == len(counts) for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["some value", "number of keys"],
              variable_name="value_equals_num_keys",
              multiline_impl=["    {var}_count = 0",
                              "    for k in counts:",
                              "        {var}_count += 1",
                              "    {var} = False",
                              "    for v in counts.values():",
                              "        if v == {var}_count:",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_sum_positive_gt_chunky", "the sum of the positive values in counts is greater than {t}",
              lambda counts, t: sum(v for v in counts.values() if v > 0) > t,
              datatype="dict[str, int]", param_space={"t": [5, 10, 20]},
              id_substrs=["sum of the positive values", "greater than", "{t}"],
              variable_name="sum_positive_large",
              multiline_impl=["    {var}_total = 0",
                              "    for v in counts.values():",
                              "        if v > 0:",
                              "            {var}_total += v",
                              "    {var} = {var}_total > {t}"]),
    Predicate("dict_product_values_gt_chunky", "the product of the values in counts is greater than {t}",
              lambda counts, t: math.prod(counts.values()) > t,
              datatype="dict[str, int]", param_space={"t": [0, 1, 10, 100]},
              id_substrs=["product", "values", "greater than", "{t}"],
              variable_name="product_values_large",
              multiline_impl=["    {var}_prod = 1",
                              "    for v in counts.values():",
                              "        {var}_prod *= v",
                              "    {var} = {var}_prod > {t}"]),
    Predicate("dict_num_even_values_ge_chunky", "at least {k} values in counts are even",
              lambda counts, k: sum(1 for v in counts.values() if v % 2 == 0) >= k,
              datatype="dict[str, int]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "even"],
              variable_name="enough_even_values",
              multiline_impl=["    {var}_count = 0",
                              "    for v in counts.values():",
                              "        if v % 2 == 0:",
                              "            {var}_count += 1",
                              "    {var} = {var}_count >= {k}"]),
    Predicate("dict_all_values_le_chunky", "every value in counts is at most {t}",
              lambda counts, t: all(v <= t for v in counts.values()),
              datatype="dict[str, int]", param_space={"t": [5, 10, 15]},
              id_substrs=["every value", "at most", "{t}"],
              variable_name="all_values_le_bound",
              multiline_impl=["    {var} = True",
                              "    for v in counts.values():",
                              "        if v > {t}:",
                              "            {var} = False",
                              "            break"]),
    Predicate("dict_has_key_starting_vowel_chunky", "counts has a key that starts with a vowel",
              lambda counts: any(len(k) > 0 and k[0] in 'aeiou' for k in counts),
              datatype="dict[str, int]", id_substrs=["key that starts with a vowel"],
              variable_name="has_vowel_key",
              multiline_impl=["    {var} = False",
                              "    for k in counts:",
                              "        if len(k) > 0 and k[0] in 'aeiou':",
                              "            {var} = True",
                              "            break"]),
    Predicate("dict_count_distinct_values_ge_chunky", "counts has at least {k} distinct values",
              lambda counts, k: len(set(counts.values())) >= k,
              datatype="dict[str, int]", param_space={"k": [1, 2, 3]},
              id_substrs=["at least", "{k}", "distinct values"],
              variable_name="enough_distinct_values",
              multiline_impl=["    {var}_seen = []",
                              "    for v in counts.values():",
                              "        if v not in {var}_seen:",
                              "            {var}_seen.append(v)",
                              "    {var} = len({var}_seen) >= {k}"]),
    Predicate("dict_sum_key_lengths_gt_chunky", "the total length of all keys in counts is greater than {t}",
              lambda counts, t: sum(len(k) for k in counts) > t,
              datatype="dict[str, int]", param_space={"t": [5, 10, 15]},
              id_substrs=["total length of all keys", "greater than", "{t}"],
              variable_name="key_length_total_large",
              multiline_impl=["    {var}_total = 0",
                              "    for k in counts:",
                              "        {var}_total += len(k)",
                              "    {var} = {var}_total > {t}"]),
    Predicate("dict_all_values_nonzero_chunky", "no value in counts is equal to zero",
              lambda counts: all(v != 0 for v in counts.values()),
              datatype="dict[str, int]", id_substrs=["no value", "equal to zero"],
              variable_name="all_values_nonzero",
              multiline_impl=["    {var} = True",
                              "    for v in counts.values():",
                              "        if v == 0:",
                              "            {var} = False",
                              "            break"]),
    Predicate("dict_more_positive_than_negative_chunky", "counts has more positive values than negative values",
              lambda counts: sum(1 for v in counts.values() if v > 0) > sum(1 for v in counts.values() if v < 0),
              datatype="dict[str, int]", id_substrs=["more positive values than negative values"],
              variable_name="more_positive_than_negative",
              multiline_impl=["    {var}_pos = 0",
                              "    {var}_neg = 0",
                              "    for v in counts.values():",
                              "        if v > 0:",
                              "            {var}_pos += 1",
                              "        if v < 0:",
                              "            {var}_neg += 1",
                              "    {var} = {var}_pos > {var}_neg"]),
    Predicate("dict_sum_negative_values_lt_chunky", "the sum of the negative values in counts is less than {t}",
              lambda counts, t: sum(v for v in counts.values() if v < 0) < t,
              datatype="dict[str, int]", param_space={"t": [-3, -1, 0]},
              id_substrs=["sum of the negative values", "less than", "{t}"],
              variable_name="sum_negative_small",
              multiline_impl=["    {var}_total = 0",
                              "    for v in counts.values():",
                              "        if v < 0:",
                              "            {var}_total += v",
                              "    {var} = {var}_total < {t}"]),
    Predicate("dict_has_value_gt_key_length_chunky", "some key in counts has a value greater than its length",
              lambda counts: any(v > len(k) for k, v in counts.items()),
              datatype="dict[str, int]", id_substrs=["value greater than its length", "key"],
              variable_name="has_value_over_key_length",
              multiline_impl=["    {var} = False",
                              "    for k, v in counts.items():",
                              "        if v > len(k):",
                              "            {var} = True",
                              "            break"]),
]

CHUNKY_PREDICATES_BY_ID: dict[str, Predicate] = {p.id: p for p in CHUNKY_PREDICATES}
assert len(CHUNKY_PREDICATES_BY_ID) == len(CHUNKY_PREDICATES), (
    "duplicate id within CHUNKY_PREDICATES: "
    + str([i for i in {p.id for p in CHUNKY_PREDICATES} if [q.id for q in CHUNKY_PREDICATES].count(i) > 1])
)

# Unified id → Predicate lookup across the int bank + chunky bank (ids are disjoint: chunky
# ids are suffixed `_chunky`).  Used wherever a spec resolves predicate ids to oracles, since a
# spec may draw from either bank.
# SIMPLE_PREDICATES_BY_ID already covers INT_PREDICATES plus the str / list[int] / list[str]
# banks, so a spec built from any simple bank resolves here (ConditionSpec.datatype reads it).
ALL_PREDICATES_BY_ID: dict[str, Predicate] = {**SIMPLE_PREDICATES_BY_ID,
                                              **CHUNKY_PREDICATES_BY_ID}
assert len(ALL_PREDICATES_BY_ID) == len(SIMPLE_PREDICATES_BY_ID) + len(CHUNKY_PREDICATES_BY_ID), (
    "predicate id collision between the simple banks and CHUNKY_PREDICATES"
)

# invariants for the seed set: every chunky predicate is multiline-only in python, has a
# result-variable name, and a valid datatype.
for _p in CHUNKY_PREDICATES:
    assert _p.is_multiline_for("python"), f"chunky predicate {_p.id!r} has no python multiline_impl"
    assert not _p.has_singleline_for("python"), f"chunky predicate {_p.id!r} should be multiline-only"
    assert _p.variable_name, f"chunky predicate {_p.id!r} missing variable_name"
    assert _p.datatype in DATATYPES, f"chunky predicate {_p.id!r} bad datatype {_p.datatype!r}"


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
    def render_instruction(self, name_rule: str = "letters", multistep: bool = False,
                           letter_start: str = "a") -> str:
        """`name_rule="letters"` mandates lowercase-letter result vars (`a`, `b`, …);
        any other value drops the naming constraint entirely (lets the model/prefill
        choose), used by build_example(naming="natural"|"indexed").

        `letter_start` shifts that alphabet: `"p"` mandates `p`, `q`, `r`, … . The instruction
        text has to move with the prefill, otherwise the prompt asks for `a, b, c` while the
        forced prefill writes `p, q, r` — an instruction/prefill *mismatch*, which is a
        different experiment from an out-of-distribution naming scheme. Default `"a"`
        reproduces the previous string byte-for-byte.

        `multistep=True` adds a clause asking the model to implement each check step-by-step
        with explicit loops/intermediate variables rather than a one-line expression or
        comprehension — used for the chunky dataset, where we WANT multi-line implementations
        (the simple predicates otherwise collapse to one-liners)."""
        ls = LANGS[self.lang]
        body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.spec.spec_texts))
        subject = DATATYPES[self.spec.datatype].subject
        opener = (
            f"Write a {ls.display} program that implements a function "
            f"`{ls.fn_name}` which, given {subject}, checks the following "
            f"conditions:"
        )
        # Force a named result variable per condition (so each check is a
        # separable statement, not hidden inside a print) + a single aggregation
        # print at the end (individual + collective pass).
        if name_rule == "letters":
            _l0 = letter_start
            _l1 = chr(ord(letter_start) + 1)
            assign_rule = (
                "For each condition, in the order listed, assign the boolean result of "
                "the check to its own local variable, using single lowercase letters in "
                f"order — `{_l0}` for the first condition, `{_l1}` for the second, and so on. "
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
            dt = DATATYPES[self.spec.datatype]
            return f"def {ls.fn_name}({dt.arg}: {dt.annotation}):\n"
        return ls.mono_stub  # non-python langs: int-only for now


# ──────────────────────────────────────────────────────────────────────────────
# Sampling.
# ──────────────────────────────────────────────────────────────────────────────


def sample_test_inputs(rng: random.Random, datatype: DataType, k: int = 6) -> list:
    """Eyeball/unit-test inputs for a spec, per the datatype's sampler (see `DATATYPES`)."""
    return [DATATYPES[datatype].sampler(rng) for _ in range(k)]


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
