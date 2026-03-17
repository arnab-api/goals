"""Augmentation pipeline for multi-predicate and incomplete-code samples.

Generates synthetic variants by applying one surface-level perturbation per
sample, keeping goal-relevant structure invariant:
  - Variable rename       (e.g. s → text)
  - Function rename       (e.g. is_palindrome → check_palindrome)
  - Type-hint toggle      (remove param annotations, keep -> bool)
  - Docstring predicate swap  (multi-predicate only)

Augmentation is **online (in-memory)**. JSON files remain clean (originals only).
Augmented variants are derived only from originals (is_original == True) to
prevent combinatorial blowup.  Each variant appends an entry to its
augmentation list so stacking is supported:
    sample.augment("toggle_type_hints").augment("function_rename", {"new_name": "check_foo"})

Typical usage:
    from src.goals.predicate_data import MultiPredicateDataset

    ds = MultiPredicateDataset.load_dataset(augment=True)
    # or: ds.augment() after loading
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


def adapt_prefills(
    prefills: dict[str, list[str]], old_var: str, new_var: str
) -> dict[str, list[str]]:
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

    cleaned_params = ", ".join(
        _strip_annotation(p).strip() for p in params_str.split(",")
    )
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
    """Populate short_description on tp.p_plus, each sample.p_minus, and
    sample.p_plus_short_description (in-place).

    Only fills in empty short_description fields; existing values are left alone.
    """
    p_plus_desc_seen: str | None = None

    for sample_idx, sample in enumerate(tp.samples):
        if not sample.is_original:
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

        # Per-sample p_plus_short_description (from this sample's own docstring)
        if not sample.p_plus_short_description:
            sample.p_plus_short_description = p_plus_template

        # tp-level p_plus is shared — take first sample's value; warn on inconsistency
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
    if (
        isinstance(s, MultiPredicateSample)
        and s.p_minus
        and s.p_minus.short_description
    ):
        s.p_minus.short_description = adapt_text_var(
            s.p_minus.short_description, old_var, new_var
        )
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [{"type": "variable_rename", "mapping": {old_var: new_var}}]
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
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [
        {"type": "function_rename", "old_name": old_name, "new_name": new_name}
    ]
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
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [{"type": "toggle_type_hints", "action": "remove"}]
    return s


def _derive_p_plus_short_description(sample: MultiPredicateSample) -> str | None:
    """Derive the p_plus short_description template from the sample's own docstring.

    Returns a template string (with {var} placeholder where the variable name
    appeared), or None if the docstring cannot be parsed.
    """
    clauses = _parse_docstring_clauses(sample.code)
    if clauses is None:
        return None
    var_names = extract_var_names_from_code(sample.code)
    primary_var = var_names[0] if var_names else None
    clause_2 = clauses[1]
    if primary_var:
        clause_2 = re.sub(r"\b" + re.escape(primary_var) + r"\b", "{var}", clause_2)
    return clause_2


def swap_predicate_order(
    sample: MultiPredicateSample,
) -> MultiPredicateSample | None:
    """Return a copy of sample with the docstring predicate order swapped.

    If p_plus_short_description is empty, it is derived automatically from the
    sample's docstring (supports direct .augment() calls without needing
    extract_short_descriptions to have been called first).

    Returns None if p_minus.short_description is missing or the docstring
    cannot be parsed.
    Does NOT swap aligned_p_plus / aligned_p_minus prefills (they track goal
    alignment, not docstring position).
    """
    # Auto-derive p_plus_short_description from docstring if not already set
    p_plus_short_desc = sample.p_plus_short_description
    if not p_plus_short_desc:
        p_plus_short_desc = _derive_p_plus_short_description(sample)
        if not p_plus_short_desc:
            logger.warning(
                f"Cannot derive p_plus_short_description for {sample.function_name!r}: "
                f"docstring could not be parsed"
            )
            return None

    if sample.p_minus is None or not sample.p_minus.short_description:
        logger.warning(
            f"sample.p_minus.short_description is empty for {sample.function_name!r} — cannot swap"
        )
        return None

    var_names = extract_var_names_from_code(sample.code)
    var = var_names[0] if var_names else None

    p_plus_tmpl = p_plus_short_desc
    p_minus_tmpl = sample.p_minus.short_description

    if "{var}" not in p_plus_tmpl:
        logger.warning(
            f"p_plus_short_description missing {{var}} placeholder for {sample.function_name!r}: {p_plus_tmpl!r}"
        )
    if "{var}" not in p_minus_tmpl:
        logger.warning(
            f"p_minus.short_description missing {{var}} placeholder for {sample.function_name!r}: {p_minus_tmpl!r}"
        )

    # Docstring compound: render {var} → concrete variable name from the def line
    def _render(template: str, v: str | None) -> str:
        return template.replace("{var}", v) if v is not None else template

    swapped_code_compound = (
        f"{_render(p_plus_tmpl, var)} and {_render(p_minus_tmpl, var)}"
    )

    # Instruction compound: render {var} → "the input" (instruction uses natural language)
    swapped_instr_compound = (
        f"{p_plus_tmpl.replace('{var}', 'the input')} and "
        f"{p_minus_tmpl.replace('{var}', 'the input')}"
    )

    s = copy.deepcopy(sample)

    # Replace docstring clause in code
    s.code = re.sub(
        r'("""Return True iff ).+?(""")',
        lambda m: m.group(1) + swapped_code_compound + "." + m.group(2),
        s.code,
        count=1,
        flags=re.DOTALL,
    )

    # Replace compound predicate in instruction
    # Template: "Write a Python function that returns True iff {compound}.\n\n"
    s.instruction = re.sub(
        r"(returns True iff ).+?(\.\n)",
        lambda m: m.group(1) + swapped_instr_compound + m.group(2),
        s.instruction,
        count=1,
        flags=re.DOTALL,
    )

    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [{"type": "swap_predicate_order"}]
    return s


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def augment_sample(sample: S, augmentation_type: str, augmentation_config: dict) -> S:
    """Dispatch to the appropriate augmentation function by type string.

    This is the entry point for sample.augment(type, config) calls.
    """
    if augmentation_type == "variable_rename":
        return rename_variable(
            sample, augmentation_config["old_var"], augmentation_config["new_var"]
        )
    elif augmentation_type == "function_rename":
        return rename_function(sample, augmentation_config["new_name"])
    elif augmentation_type == "toggle_type_hints":
        return toggle_type_hints(sample)
    elif augmentation_type == "swap_predicate_order":
        if not isinstance(sample, MultiPredicateSample):
            raise TypeError("swap_predicate_order requires a MultiPredicateSample")
        result = swap_predicate_order(sample)
        if result is None:
            raise ValueError(
                f"swap_predicate_order failed: p_plus_short_description missing on sample {sample.function_name!r}"
            )
        return result  # type: ignore[return-value]
    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type!r}")


# ---------------------------------------------------------------------------
# Variant generators
# ---------------------------------------------------------------------------


def generate_variants(tp: TargetPredicate) -> list[MultiPredicateSample]:
    """Generate all augmented variants for every original sample in tp.

    Calls extract_short_descriptions(tp) first to ensure p_plus_short_description
    is set on all originals (idempotent — skips already-populated fields).

    Variants per original sample:
      1. Variable rename (up to 3 alternatives from _VAR_VOCAB)
      2. Function rename (up to 3 prefix swaps if fn starts with "is_")
      3. Type hint toggle (1 variant)
      4. Docstring predicate swap (1 variant, if short_descriptions populated)

    Returns a flat list of new samples (not appended to tp.samples yet).
    """
    extract_short_descriptions(tp)

    new_samples: list[MultiPredicateSample] = []

    for sample in tp.samples:
        if not sample.is_original:
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
            suffix = fn[len("is_") :]
            for prefix in _FN_PREFIXES:
                new_samples.append(rename_function(sample, prefix + suffix))

        # 3. Type hint toggle
        new_samples.append(toggle_type_hints(sample))

        # 4. Docstring predicate swap
        swapped = swap_predicate_order(sample)
        if swapped is not None:
            new_samples.append(swapped)

    return new_samples


def generate_variants_incomplete(
    sample: IncompleteCodeSample,
) -> list[IncompleteCodeSample]:
    """Generate augmented variants for a single IncompleteCodeSample.

    No docstring swap (these samples have no docstring).
    Variants: variable rename, function rename, type hint toggle.
    """
    if not sample.is_original:
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
        suffix = fn[len("is_") :]
        for prefix in _FN_PREFIXES:
            new_samples.append(rename_function(sample, prefix + suffix))

    # 3. Type hint toggle
    new_samples.append(toggle_type_hints(sample))

    return new_samples
