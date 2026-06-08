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

    def evaluate(self, n: int, params: dict[str, Any]) -> bool:
        return bool(self.oracle(n, **params))


INT_PREDICATES: list[Predicate] = [
    Predicate("positive", "n is positive", lambda n: n > 0, id_substrs=["positive"]),
    Predicate("even", "n is even", lambda n: n % 2 == 0, id_substrs=["even"]),
    Predicate("odd", "n is odd", lambda n: n % 2 == 1, id_substrs=["odd"]),
    Predicate("prime", "n is a prime number", lambda n: _is_prime(n), id_substrs=["prime"]),
    Predicate("perfect_square", "n is a perfect square",
              lambda n: n >= 0 and math.isqrt(n) ** 2 == n, id_substrs=["perfect", "square"]),
    Predicate("palindrome", "n is a palindrome (its digits read the same forwards and backwards)",
              lambda n: str(abs(n)) == str(abs(n))[::-1], id_substrs=["palindrome"]),
    Predicate("le_bound", "n is less than or equal to {b}", lambda n, b: n <= b,
              {"b": [50, 100, 200, 500, 1000]}, id_substrs=["less than or equal", "{b}"]),
    Predicate("ge_bound", "n is greater than or equal to {a}", lambda n, a: n >= a,
              {"a": [0, 1, 10, 50, 100]}, id_substrs=["greater than or equal", "{a}"]),
    Predicate("in_range", "n is between {a} and {b} (inclusive)",
              lambda n, a, b: a <= n <= b, {"a": [0, 10, 50], "b": [100, 200, 500]},
              id_substrs=["between", "{a}", "{b}"]),
    Predicate("divisible", "n is divisible by {k}", lambda n, k: k != 0 and n % k == 0,
              {"k": list(range(2, 21))}, id_substrs=["divisible", "{k}"]),
    Predicate("digit_sum_gt", "the sum of n's digits is greater than {t}",
              lambda n, t: _digit_sum(n) > t, {"t": [5, 10, 15, 20]},
              id_substrs=["sum", "digits", "{t}"]),
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
