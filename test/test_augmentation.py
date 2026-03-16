"""Minimal tests for src/goals/augment.py.

Tests run without any LM — pure data transformation checks.
"""

import sys
sys.path.insert(0, ".")

from src.goals.augment import (
    adapt_text_var,
    extract_var_names_from_code,
    extract_short_descriptions,
    generate_variants,
    generate_variants_incomplete,
    rename_variable,
    rename_function,
    toggle_type_hints,
    swap_predicate_order,
    _toggle_type_hints_in_def,
    _parse_docstring_clauses,
)
from src.goals.predicate_data import MultiPredicateDataset, MultiPredicateSample, PredicateSpec, TargetPredicate
from src.goals.data import IncompleteCodeSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mp_sample(fn="is_foo", var="s") -> MultiPredicateSample:
    return MultiPredicateSample(
        instruction=f"Write a Python function that returns True iff {var} has no spaces and {var} is a palindrome.\n\n",
        code=(
            f"def {fn}({var}: str) -> bool:\n"
            f'    """Return True iff {var} has no spaces and {var} is a palindrome."""\n'
            f"    return "
        ),
        unit_tests=[("aba", True), ("a b", False)],
        prefills={
            "aligned_p_plus": [f"{var} == {var}"],
            "aligned_p_minus": [f"' ' not in {var}"],
            "adversarial": ["not ("],
            "inconsequential": [""],
        },
        function_name=fn,
        p_minus=PredicateSpec(
            description=f"{var} has no spaces",
            short_description="",
            anchor_patterns=[r"' ' not in \w+"],
        ),
    )


def _make_incomplete_sample() -> IncompleteCodeSample:
    return IncompleteCodeSample(
        instruction="Write a function in python that checks if a number is odd.\n\n",
        code="def is_odd(n: int) -> bool:\n    return True if ",
        function_name="is_odd",
        unit_tests=[(1, True), (2, False)],
        prefills={"aligned": ["n %"], "adversarial": ["not"]},
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_extract_var_names():
    assert extract_var_names_from_code("def foo(s: str) -> bool:") == ["s"]
    assert extract_var_names_from_code("def foo(s: str, n: int) -> bool:") == ["s", "n"]
    assert extract_var_names_from_code("def foo(lst: list) -> bool:") == ["lst"]
    assert extract_var_names_from_code("no def here") == []
    print("PASS test_extract_var_names")


def test_adapt_text_var():
    assert adapt_text_var("s == s[::-1]", "s", "text") == "text == text[::-1]"
    # word boundary: 'lst' should not match inside 'list'
    assert adapt_text_var("isinstance(n, list)", "lst", "items") == "isinstance(n, list)"
    # single-char var: 'n' should not replace inside 'int'
    assert adapt_text_var("n > 0 and isinstance(n, int)", "n", "num") == "num > 0 and isinstance(num, int)"
    print("PASS test_adapt_text_var")


def test_toggle_type_hints_in_def():
    assert _toggle_type_hints_in_def("def foo(s: str) -> bool:") == "def foo(s) -> bool:"
    assert _toggle_type_hints_in_def("def foo(s: str, n: int) -> bool:") == "def foo(s, n) -> bool:"
    # already no annotations
    result = _toggle_type_hints_in_def("def foo(s) -> bool:")
    assert "s" in result and "->" in result
    print("PASS test_toggle_type_hints_in_def")


def test_parse_docstring_clauses():
    code = 'def foo(s: str) -> bool:\n    """Return True iff s has no spaces and s is a palindrome."""\n    return '
    clauses = _parse_docstring_clauses(code)
    assert clauses is not None
    assert len(clauses) == 2
    assert "no spaces" in clauses[0]
    assert "palindrome" in clauses[1]
    print("PASS test_parse_docstring_clauses")


def test_rename_variable():
    sample = _make_mp_sample(var="s")
    new = rename_variable(sample, "s", "text")
    assert "text" in new.code
    assert "s: str" not in new.code  # annotation replaced too
    assert "text" in new.instruction
    assert any("text" in p for p in new.prefills["aligned_p_plus"])
    # unit_tests and anchor_patterns unchanged
    assert new.unit_tests == sample.unit_tests
    assert new.p_minus.anchor_patterns == sample.p_minus.anchor_patterns
    assert new.function_name == sample.function_name
    assert new.augmentation == {"type": "variable_rename", "mapping": {"s": "text"}}
    # original unchanged
    assert "s: str" in sample.code
    print("PASS test_rename_variable")


def test_rename_function():
    sample = _make_mp_sample(fn="is_foo")
    new = rename_function(sample, "check_foo")
    assert "def check_foo(" in new.code
    assert new.function_name == "check_foo"
    assert new.augmentation == {"type": "function_rename", "old_name": "is_foo", "new_name": "check_foo"}
    # instruction unchanged
    assert new.instruction == sample.instruction
    # original unchanged
    assert "def is_foo(" in sample.code
    print("PASS test_rename_function")


def test_toggle_type_hints():
    sample = _make_mp_sample()
    new = toggle_type_hints(sample)
    assert "s: str" not in new.code
    assert "-> bool" in new.code
    assert new.augmentation == {"type": "toggle_type_hints", "action": "remove"}
    assert "s: str" in sample.code  # original unchanged
    print("PASS test_toggle_type_hints")


def test_swap_predicate_order():
    sample = _make_mp_sample(var="s")
    tp = TargetPredicate(
        p_plus=PredicateSpec(
            description="is a palindrome",
            short_description="is a palindrome",
        ),
        samples=[sample],
        name="test_tp",
    )
    sample.p_minus.short_description = "{var} has no spaces"

    swapped = swap_predicate_order(sample, tp)
    assert swapped is not None
    # p_plus clause should now appear first in docstring
    assert "is a palindrome and s has no spaces" in swapped.code
    assert swapped.augmentation == {"type": "swap_predicate_order"}
    # original code unchanged
    assert "has no spaces and" in sample.code
    print("PASS test_swap_predicate_order")


def test_swap_predicate_order_missing_desc():
    sample = _make_mp_sample()
    tp = TargetPredicate(
        p_plus=PredicateSpec(description="is a palindrome", short_description=""),
        samples=[sample],
        name="test_tp",
    )
    result = swap_predicate_order(sample, tp)
    assert result is None  # should return None when short_description missing
    print("PASS test_swap_predicate_order_missing_desc")


def test_extract_short_descriptions():
    sample = _make_mp_sample(var="s")
    tp = TargetPredicate(
        p_plus=PredicateSpec(description="is a palindrome", short_description=""),
        samples=[sample],
        name="test_tp",
    )
    extract_short_descriptions(tp)
    assert tp.p_plus.short_description != ""
    assert "palindrome" in tp.p_plus.short_description
    assert sample.p_minus.short_description != ""
    assert "spaces" in sample.p_minus.short_description
    # should use {var} placeholder, not 's'
    assert "{var}" in sample.p_minus.short_description or "s" not in sample.p_minus.short_description
    print("PASS test_extract_short_descriptions")


def test_generate_variants_multi():
    sample = _make_mp_sample(fn="is_foo", var="s")
    sample.p_minus.short_description = "{var} has no spaces"
    tp = TargetPredicate(
        p_plus=PredicateSpec(
            description="is a palindrome",
            short_description="is a palindrome",
        ),
        samples=[sample],
        name="test_tp",
    )

    variants = generate_variants(tp)
    types = [(v.augmentation or {}).get("type") for v in variants]

    assert types.count("variable_rename") == 3   # text, input_str, string
    assert types.count("function_rename") == 3   # check_, verify_, has_
    assert types.count("toggle_type_hints") == 1
    assert types.count("swap_predicate_order") == 1
    assert len(variants) == 8

    # No augmented sample should itself produce variants
    variants2 = generate_variants(tp)
    tp.samples.extend(variants)
    variants3 = generate_variants(tp)
    assert len(variants3) == len(variants2)  # same count — augmented skipped

    print("PASS test_generate_variants_multi")


def test_generate_variants_incomplete():
    sample = _make_incomplete_sample()
    variants = generate_variants_incomplete(sample)
    types = [(v.augmentation or {}).get("type") for v in variants]

    assert types.count("variable_rename") == 3   # num, number, val
    assert types.count("function_rename") == 3   # check_, verify_, has_
    assert types.count("toggle_type_hints") == 1
    assert len(variants) == 7

    # augmented sample produces no further variants
    assert generate_variants_incomplete(variants[0]) == []
    print("PASS test_generate_variants_incomplete")


def test_no_var_vocab_match():
    """Samples whose primary var has no _VAR_VOCAB entry get no rename variants."""
    sample = IncompleteCodeSample(
        instruction="check x",
        code="def is_foo(x: int) -> bool:\n    return True if ",
        function_name="is_foo",
        unit_tests=[],
        prefills={},
    )
    variants = generate_variants_incomplete(sample)
    types = [(v.augmentation or {}).get("type") for v in variants]
    assert "variable_rename" not in types
    assert types.count("function_rename") == 3
    assert types.count("toggle_type_hints") == 1
    print("PASS test_no_var_vocab_match")


def test_loaded_dataset_counts():
    """Sanity check: loaded dataset has expected originals/augmented counts."""
    ds = MultiPredicateDataset.load_dataset()
    orig = sum(1 for tp in ds.groups for s in tp.samples if s.augmentation is None)
    aug = sum(1 for tp in ds.groups for s in tp.samples if s.augmentation is not None)
    assert orig == 23, f"expected 23 originals, got {orig}"
    assert aug == 184, f"expected 184 augmented, got {aug}"
    print(f"PASS test_loaded_dataset_counts  (orig={orig}, aug={aug})")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_extract_var_names()
    test_adapt_text_var()
    test_toggle_type_hints_in_def()
    test_parse_docstring_clauses()
    test_rename_variable()
    test_rename_function()
    test_toggle_type_hints()
    test_swap_predicate_order()
    test_swap_predicate_order_missing_desc()
    test_extract_short_descriptions()
    test_generate_variants_multi()
    test_generate_variants_incomplete()
    test_no_var_vocab_match()
    test_loaded_dataset_counts()
    print("\nAll tests passed.")
