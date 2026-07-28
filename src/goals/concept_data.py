"""Data types for goal-concept experiments (e.g. iterative-vs-recursive).

Each GoalConceptSample holds a language-neutral problem definition (instruction,
unit tests, docstring) plus per-language stubs that do NOT include the docstring.
Call get_code_stub(lang, add_docstring=True) to obtain the complete stub with the
docstring inserted in the idiomatic style for that language.

Supported languages: python, cpp, java, haskell, rust, prolog.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from dataclasses_json import DataClassJsonMixin, config


logger = logging.getLogger(__name__)

Language = Literal["python", "cpp", "java", "haskell", "rust", "prolog"]
# Free-form string — each concept registers its own valid strategy keys in
# _STRATEGY_DIRECTIVES; full_instruction validates against that table at runtime.
StrategyDirective = str

_NO_COMMENT_STR = (
    "Write your code without any comments or explanations, just the implementation."
)

# ---------------------------------------------------------------------------
# Per-concept strategy directives
# ---------------------------------------------------------------------------
# When ``preferred_strategy`` is set on a sample, the matching directive is
# appended to the instruction.  Looked up as _STRATEGY_DIRECTIVES[concept][strategy].
# Add new concepts here; sample.full_instruction raises a clear error if a
# (concept, strategy) pair is missing.

_STRATEGY_DIRECTIVES: dict[str, dict[str, str]] = {
    "iterative_vs_recursive": {
        "recursive": "Implement the function recursively.",
        "iterative": "Implement the function iteratively using a loop. **Do not use recursion**.",
    },
    "monolithic_vs_modular": {
        "monolithic": (
            "Implement everything inside a single function. **Do not introduce "
            "helper functions, classes, or split the logic across multiple "
            "scopes.**"
        ),
        "modular": (
            "Decompose the solution into small, well-named helper functions or "
            "classes. Each helper should do one thing."
        ),
    },
    "design_pattern": {
        "singleton":  "Use the **Singleton** design pattern.",
        "observer":   "Use the **Observer** design pattern.",
        "strategy":   "Use the **Strategy** design pattern.",
        "factory":    "Use the **Factory** design pattern.",
        "decorator":  "Use the **Decorator** design pattern.",
        "visitor":    "Use the **Visitor** design pattern.",
        "iterator":   "Use the **Iterator** design pattern.",
        "if_else":    "Use a flat if/else dispatch — **no classes, no polymorphism**.",
        "callback_list": (
            "Use a plain callback list (a dict or list of functions) — **do not "
            "introduce an Observer/Subscriber class hierarchy**."
        ),
        "subclassing": "Add the new behavior by **subclassing** the existing class.",
    },
}


# ---------------------------------------------------------------------------
# Per-language docstring formatters: (docstring, stub) -> stub_with_docstring
# ---------------------------------------------------------------------------

def _fmt_python(docstring: str, stub: str) -> str:
    """Insert a triple-quoted docstring after the def line.

    stub = "def foo(n: int) -> int:\\n    "
    result= "def foo(n: int) -> int:\\n    \\"\\"\\"docstring\\"\\"\\"\\n    "
    """
    nl = stub.index("\n") + 1
    return stub[:nl] + f'    """{docstring}"""\n    '


def _fmt_cpp(docstring: str, stub: str) -> str:
    """Prepend a // comment before the C++ function signature."""
    return f"// {docstring}\n" + stub


def _fmt_java(docstring: str, stub: str) -> str:
    """Prepend a Javadoc /** … */ comment before the method."""
    return f"/** {docstring} */\n" + stub


def _fmt_haskell(docstring: str, stub: str) -> str:
    """Prepend a Haddock -- | comment before the type signature."""
    return f"-- | {docstring}\n" + stub


def _fmt_rust(docstring: str, stub: str) -> str:
    """Prepend a /// doc-comment before the fn definition."""
    return f"/// {docstring}\n" + stub


def _fmt_prolog(docstring: str, stub: str) -> str:
    """Prepend a % comment before the clause head."""
    return f"% {docstring}\n" + stub


_DOCSTRING_FORMATTERS: dict[str, Callable[[str, str], str]] = {
    "python":  _fmt_python,
    "cpp":     _fmt_cpp,
    "java":    _fmt_java,
    "haskell": _fmt_haskell,
    "rust":    _fmt_rust,
    "prolog":  _fmt_prolog,
}


# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------

def _to_camel_case(snake: str) -> str:
    """Convert snake_case to camelCase (for C++, Java, Haskell)."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


_CAMEL_CASE_LANGS = {"cpp", "java", "haskell"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GoalConceptSample(DataClassJsonMixin):
    """A single goal-concept problem, language-neutral except for code stubs.

    Fields
    ------
    function_name : canonical snake_case identifier, e.g. "fibonacci". May be
                    "" for design-pattern / probing-only samples where there is
                    no single target function and the code will never be
                    compiled.
    concept       : concept tag, e.g. "iterative_vs_recursive"
    instruction   : language-neutral task description shown to the model
    unit_tests    : [(input, expected), ...]; input may be a scalar or tuple for
                    multi-arg functions (e.g. gcd: input = (12, 4)).  Empty list
                    for probing-only samples (no compilation / execution).
    pivot_note    : human-readable note about what the fork looks like
    stubs         : {language: code stub WITHOUT docstring}; stub must end with
                    the indentation cursor ready for the model to continue (e.g.
                    "def foo(n):\\n    " or "int foo(int n) {\\n    ").  May be
                    ``{}`` when ``prompt_stub`` is used instead.
    prompt_stub   : a single language-agnostic free-form stub used when
                    ``stubs`` is empty.  Typical values: ``""`` (empty — model
                    starts fresh after the instruction), ``"class "``,
                    ``"def "``, ``"# "``.  Used for probing samples where
                    forcing a specific function signature would itself bias the
                    pattern choice.
    docstring     : language-neutral one-line description inserted by
                    get_code_stub when add_docstring=True (only when
                    per-language ``stubs`` are present — ``prompt_stub`` is
                    emitted verbatim).
    without_comments : mirrors IncompleteCodeSample; when True, full_instruction
                    appends a no-comment directive and get_code_stub ignores
                    add_docstring
    preferred_strategy : optional categorical preference (e.g. "recursive" /
                    "iterative" / "singleton") to advertise in the instruction.
                    Looked up in _STRATEGY_DIRECTIVES[concept][preferred_strategy].
                    When ``None`` (default), the prompt stays underspecified —
                    the setting used for proactive-choice probing.
    """

    function_name: str = ""
    concept: str = ""
    instruction: str = ""
    unit_tests: list[tuple[Any, Any]] = field(default_factory=list)
    pivot_note: str = ""
    stubs: dict[str, str] = field(default_factory=dict)
    prompt_stub: Optional[str] = None
    docstring: str = ""
    without_comments: bool = False
    preferred_strategy: Optional[StrategyDirective] = None
    no_comment_str: str = field(
        default=_NO_COMMENT_STR,
        metadata=config(exclude=lambda v: v == _NO_COMMENT_STR),
    )

    def __post_init__(self) -> None:
        self.unit_tests = [
            t if isinstance(t, tuple) else tuple(t) for t in self.unit_tests
        ]

    # ------------------------------------------------------------------
    # Instruction helpers
    # ------------------------------------------------------------------

    @property
    def full_instruction(self) -> str:
        """Instruction with optional strategy + no-comment suffixes appended.

        Composition order:
            1. base instruction
            2. strategy directive (if preferred_strategy is set)
            3. no-comment directive (if without_comments is True)

        Raises ValueError if preferred_strategy is set but no directive is
        registered for (concept, preferred_strategy).
        """
        text = self.instruction

        if self.preferred_strategy is not None:
            directives = _STRATEGY_DIRECTIVES.get(self.concept, {})
            directive = directives.get(self.preferred_strategy)
            if directive is None:
                available = list(directives) or list(_STRATEGY_DIRECTIVES)
                raise ValueError(
                    f"No directive for concept={self.concept!r}, "
                    f"preferred_strategy={self.preferred_strategy!r}. "
                    f"Available: {available}"
                )
            text = text.rstrip(". ") + ". " + directive

        if self.without_comments:
            text = text.rstrip(". ") + ". " + self.no_comment_str

        return text

    # ------------------------------------------------------------------
    # Stub helpers
    # ------------------------------------------------------------------

    def get_code_stub(self, lang: str, add_docstring: bool = True) -> str:
        """Return the code stub for *lang*, optionally with the docstring inserted.

        Resolution order:
          1. ``self.stubs[lang]`` if present (with docstring inserted per lang
             when ``add_docstring`` and not ``without_comments``).
          2. ``self.prompt_stub`` (returned verbatim, no docstring insertion —
             prompt_stub is intentionally language-agnostic and free-form).

        Raises KeyError if neither route yields a stub.
        """
        if lang in self.stubs:
            stub = self.stubs[lang]
            if add_docstring and not self.without_comments and self.docstring:
                fmt = _DOCSTRING_FORMATTERS.get(lang)
                if fmt is not None:
                    stub = fmt(self.docstring, stub)
            return stub.strip()

        if self.prompt_stub is not None:
            # Note: returned verbatim; no .strip() so callers can request
            # prompt_stub="" or prompt_stub="class " and get exactly that.
            return self.prompt_stub

        available = list(self.stubs)
        raise KeyError(
            f"No stub for language {lang!r} and no prompt_stub set. "
            f"Available stubs: {available}"
        )

    def get_function_name(self, lang: str) -> str:
        """Return the idiomatic function name for *lang*.

        Python / Rust / Prolog use snake_case (the canonical function_name).
        C++ / Java / Haskell use camelCase derived from function_name.
        Returns "" if function_name itself is empty (probing-only samples).
        """
        if not self.function_name:
            return ""
        if lang in _CAMEL_CASE_LANGS:
            return _to_camel_case(self.function_name)
        return self.function_name

    @property
    def languages(self) -> list[str]:
        """Languages for which a per-language stub is available.

        Empty when only ``prompt_stub`` is set — ``get_code_stub(lang)`` will
        still succeed for any *lang* via the prompt_stub fallback.
        """
        return list(self.stubs)


@dataclass
class GoalConceptDataset:
    """Collection of GoalConceptSamples loaded from a JSON file."""

    samples: list[GoalConceptSample]

    @classmethod
    def load(cls, path: str) -> "GoalConceptDataset":
        with open(path) as f:
            data = json.load(f)
        return cls(samples=[GoalConceptSample.from_dict(d) for d in data])

    def filter_language(self, lang: str) -> "GoalConceptDataset":
        """Return a new dataset containing only samples that have a stub for *lang*."""
        return GoalConceptDataset(
            [s for s in self.samples if lang in s.stubs]
        )

    def get(self, function_name: str) -> GoalConceptSample | None:
        for s in self.samples:
            if s.function_name == function_name:
                return s
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx: int) -> GoalConceptSample:
        return self.samples[idx]
