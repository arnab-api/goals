"""Augmentation pipeline for multi-predicate and incomplete-code samples.

Generates synthetic variants by applying one surface-level perturbation per
sample, keeping goal-relevant structure invariant:
  - Variable rename       (e.g. s → text)
  - Function rename       (e.g. is_palindrome → check_palindrome)
  - Type-hint toggle      (remove param annotations, keep -> bool)
  - Docstring predicate swap  (multi-predicate only)

Augmented variants are derived only from originals (augmentation == None) to
prevent combinatorial blowup.  Each variant records what was changed in its
augmentation dict.

Typical usage:
    from src.goals.augment import generate_variants, extract_short_descriptions

    # First time: populate short_description fields
    for tp in ds.groups:
        extract_short_descriptions(tp)
        tp.save()

    # Generate variants
    for tp in ds.groups:
        new_samples = generate_variants(tp)
        tp.samples.extend(new_samples)
        tp.save()
"""

from __future__ import annotations

import copy
import logging
import re
from typing import TypeVar

from src.goals.data import IncompleteCodeSample
from src.goals.predicate_data import MultiPredicateSample, TargetPredicate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Variable name alternatives keyed by original name.
_VAR_VOCAB: dict[str, list[str]] = {
    "s": ["text", "input_str", "string"],
    "n": ["num", "number", "val"],
    "lst": ["items", "elements", "data"],
    "pair": ["words", "strs", "strings"],
}

# Function name prefix replacements (only applied when fn starts with "is_").
_FN_PREFIXES = ["check_", "verify_", "has_"]

S = TypeVar("S", IncompleteCodeSample, MultiPredicateSample)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def extract_var_names_from_code(code: str) -> list[str]:
    """Extract parameter names from the first def line in code.

    Example:
        "def is_valid_palindrome(s: str) -> bool:" -> ["s"]
        "def validate(host: str, port: int) -> bool:" -> ["host", "port"]
    """
    m = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if not m:
        return []
    params_str = m.group(1)
    var_names: list[str] = []
    for param in params_str.split(","):
        param = param.strip()
        if not param or param.startswith("*"):
            continue
        name = re.split(r"[:\s=]", param)[0].strip()
        if name and name.isidentifier():
            var_names.append(name)
    return var_names


def adapt_text_var(text: str, old_var: str, new_var: str) -> str:
    """Word-boundary substitution of old_var → new_var in text."""
    return re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, text)


def adapt_prefills(prefills: dict[str, list[str]], old_var: str, new_var: str) -> dict[str, list[str]]:
    """Apply adapt_text_var to every prefill string in every category."""
    return {
        category: [adapt_text_var(p, old_var, new_var) for p in plist]
        for category, plist in prefills.items()
    }


def _toggle_type_hints_in_def(def_line: str) -> str:
    """Remove type annotations from parameters only (keep -> bool).

    'def foo(s: str, n: int) -> bool:' -> 'def foo(s, n) -> bool:'
    """
    # Find the parameter list span
    m = re.search(r"def\s+\w+\s*\(", def_line)
    if not m:
        return def_line
    paren_start = m.end() - 1  # position of '('

    # Find matching closing ')'
    depth = 0
    paren_end = -1
    for i in range(paren_start, len(def_line)):
        if def_line[i] == "(":
            depth += 1
        elif def_line[i] == ")":
            depth -= 1
            if depth == 0:
                paren_end = i
                break
    if paren_end == -1:
        return def_line

    params_str = def_line[paren_start + 1 : paren_end]

    # Remove ': <type>' from each parameter (not the return annotation)
    def _strip_annotation(param: str) -> str:
        # Match param_name followed by optional whitespace, colon, type
        m2 = re.match(r"(\s*\*{0,2}\w+)\s*:.*", param)
        if m2:
            return m2.group(1)
        return param

    cleaned_params = ", ".join(_strip_annotation(p).strip() for p in params_str.split(","))
    return def_line[: paren_start + 1] + cleaned_params + def_line[paren_end:]


def _parse_docstring_clauses(code: str) -> list[str] | None:
    """Extract the two predicate clauses from the docstring.

    Returns [clause_1, clause_2] or None if parsing fails.
    Convention: clause_1 = p_minus description, clause_2 = p_plus description.
    """
    m = re.search(r'"""Return True iff (.+?)"""', code, re.DOTALL)
    if not m:
        return None
    compound = m.group(1).strip()

    # Split on first top-level ' and ' (bracket-aware)
    depth = 0
    i = 0
    while i < len(compound):
        ch = compound[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and compound[i : i + 5] == " and ":
            clause_1 = compound[:i].strip()
            clause_2 = compound[i + 5 :].strip()
            return [clause_1, clause_2]
        i += 1
    return None  # no top-level ' and ' found


# ---------------------------------------------------------------------------
# extract_short_descriptions (in-place)
# ---------------------------------------------------------------------------


def extract_short_descriptions(tp: TargetPredicate) -> None:
    """Populate short_description on tp.p_plus and each sample.p_minus (in-place).

    Only fills in empty short_description fields; existing values are left alone.
    """
    p_plus_desc_seen: str | None = None

    for sample_idx, sample in enumerate(tp.samples):
        if sample.augmentation is not None:
            continue  # skip augmented copies

        clauses = _parse_docstring_clauses(sample.code)
        if clauses is None:
            logger.warning(
                f"[{tp.name}] sample {sample_idx}: could not parse docstring clauses — skipping"
            )
            continue

        clause_1, clause_2 = clauses
        var_names = extract_var_names_from_code(sample.code)
        primary_var = var_names[0] if var_names else None

        def _to_template(clause: str, var: str | None) -> str:
            if var is None:
                return clause
            return re.sub(r"\b" + re.escape(var) + r"\b", "{var}", clause)

        p_minus_template = _to_template(clause_1, primary_var)
        p_plus_template = _to_template(clause_2, primary_var)

        if sample.p_minus and not sample.p_minus.short_description:
            sample.p_minus.short_description = p_minus_template

        # p_plus is shared — take first sample's value; warn on inconsistency
        if not tp.p_plus.short_description:
            tp.p_plus.short_description = p_plus_template
            p_plus_desc_seen = p_plus_template
        elif p_plus_desc_seen is not None and p_plus_template != p_plus_desc_seen:
            logger.warning(
                f"[{tp.name}] sample {sample_idx}: p_plus short_description mismatch: "
                f"{p_plus_template!r} vs {p_plus_desc_seen!r}"
            )


# ---------------------------------------------------------------------------
# Core augmentation functions
# ---------------------------------------------------------------------------


def rename_variable(sample: S, old_var: str, new_var: str) -> S:
    """Return a copy of sample with old_var renamed to new_var.

    Applied to: code, instruction, prefills, p_minus.short_description.
    NOT applied to: unit_tests, anchor_patterns, function_name.
    """
    s = copy.deepcopy(sample)
    s.code = adapt_text_var(s.code, old_var, new_var)
    s.instruction = adapt_text_var(s.instruction, old_var, new_var)
    s.prefills = adapt_prefills(s.prefills, old_var, new_var)
    if isinstance(s, MultiPredicateSample) and s.p_minus and s.p_minus.short_description:
        s.p_minus.short_description = adapt_text_var(
            s.p_minus.short_description, old_var, new_var
        )
    s.augmentation = {"type": "variable_rename", "mapping": {old_var: new_var}}
    return s


def rename_function(sample: S, new_name: str) -> S:
    """Return a copy of sample with the function name changed to new_name.

    Applied to: code (def line only), function_name.
    NOT applied to: instruction, prefills, unit_tests, short_description.
    """
    s = copy.deepcopy(sample)
    old_name = s.function_name or ""
    s.code = re.sub(r"def\s+\w+\s*\(", f"def {new_name}(", s.code, count=1)
    s.function_name = new_name
    s.augmentation = {"type": "function_rename", "old_name": old_name, "new_name": new_name}
    return s


def toggle_type_hints(sample: S) -> S:
    """Return a copy of sample with param type annotations removed from the def line.

    The return annotation '-> bool' is preserved.
    """
    s = copy.deepcopy(sample)
    # Apply _toggle_type_hints_in_def to the first line of code
    lines = s.code.split("\n")
    # Find the def line (may span multiple lines in theory, but always first here)
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            lines[i] = _toggle_type_hints_in_def(line)
            break
    s.code = "\n".join(lines)
    s.augmentation = {"type": "toggle_type_hints", "action": "remove"}
    return s


def swap_predicate_order(
    sample: MultiPredicateSample, tp: TargetPredicate
) -> MultiPredicateSample | None:
    """Return a copy of sample with the docstring predicate order swapped.

    Returns None if short_description is missing on either p_plus or p_minus.
    Does NOT swap aligned_p_plus / aligned_p_minus prefills (they track goal
    alignment, not docstring position).
    """
    if not tp.p_plus.short_description:
        logger.warning(
            f"[{tp.name}]: tp.p_plus.short_description is empty — cannot swap"
        )
        return None
    if sample.p_minus is None or not sample.p_minus.short_description:
        logger.warning(
            f"[{tp.name}]: sample.p_minus.short_description is empty — cannot swap"
        )
        return None

    var_names = extract_var_names_from_code(sample.code)
    var = var_names[0] if var_names else None

    # Render templates with concrete variable name
    def _render(template: str, v: str | None) -> str:
        if v is None:
            return template
        return template.replace("{var}", v)

    p_plus_clause = _render(tp.p_plus.short_description, var)
    p_minus_clause = _render(sample.p_minus.short_description, var)

    # Build swapped compound predicate: p_plus first, then p_minus
    swapped_compound = f"{p_plus_clause} and {p_minus_clause}"

    s = copy.deepcopy(sample)

    # Replace docstring clause in code
    s.code = re.sub(
        r'("""Return True iff ).+?(""")',
        lambda m: m.group(1) + swapped_compound + "." + m.group(2),
        s.code,
        count=1,
        flags=re.DOTALL,
    )

    # Replace compound predicate in instruction
    # Template: "Write a Python function that returns True iff {compound}.\n\n"
    s.instruction = re.sub(
        r"(returns True iff ).+?(\.\n)",
        lambda m: m.group(1) + swapped_compound + m.group(2),
        s.instruction,
        count=1,
        flags=re.DOTALL,
    )

    s.augmentation = {"type": "swap_predicate_order"}
    return s


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------


def generate_variants(tp: TargetPredicate) -> list[MultiPredicateSample]:
    """Generate all augmented variants for every original sample in tp.

    Variants per original sample:
      1. Variable rename (up to 3 alternatives from _VAR_VOCAB)
      2. Function rename (up to 3 prefix swaps if fn starts with "is_")
      3. Type hint toggle (1 variant)
      4. Docstring predicate swap (1 variant, if short_descriptions populated)

    Returns a flat list of new samples (not appended to tp.samples yet).
    """
    new_samples: list[MultiPredicateSample] = []

    for sample in tp.samples:
        if sample.augmentation is not None:
            continue  # only augment originals

        var_names = extract_var_names_from_code(sample.code)
        primary_var = var_names[0] if var_names else None

        # 1. Variable rename
        if primary_var is not None:
            for new_var in _VAR_VOCAB.get(primary_var, [])[:3]:
                new_samples.append(rename_variable(sample, primary_var, new_var))

        # 2. Function rename
        fn = sample.function_name or ""
        if fn.startswith("is_"):
            suffix = fn[len("is_"):]
            for prefix in _FN_PREFIXES:
                new_samples.append(rename_function(sample, prefix + suffix))

        # 3. Type hint toggle
        new_samples.append(toggle_type_hints(sample))

        # 4. Docstring predicate swap
        swapped = swap_predicate_order(sample, tp)
        if swapped is not None:
            new_samples.append(swapped)

    return new_samples


def generate_variants_incomplete(sample: IncompleteCodeSample) -> list[IncompleteCodeSample]:
    """Generate augmented variants for a single IncompleteCodeSample.

    No docstring swap (these samples have no docstring).
    Variants: variable rename, function rename, type hint toggle.
    """
    if sample.augmentation is not None:
        return []

    new_samples: list[IncompleteCodeSample] = []

    var_names = extract_var_names_from_code(sample.code)
    primary_var = var_names[0] if var_names else None

    # 1. Variable rename
    if primary_var is not None:
        for new_var in _VAR_VOCAB.get(primary_var, [])[:3]:
            new_samples.append(rename_variable(sample, primary_var, new_var))

    # 2. Function rename
    fn = sample.function_name or ""
    if fn.startswith("is_"):
        suffix = fn[len("is_"):]
        for prefix in _FN_PREFIXES:
            new_samples.append(rename_function(sample, prefix + suffix))

    # 3. Type hint toggle
    new_samples.append(toggle_type_hints(sample))

    return new_samples
