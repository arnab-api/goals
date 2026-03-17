"""Joint augmentation for (dest, target) SinglePredicateSample pairs.

Step 5 of the head localization pipeline: augment dest and target jointly so
their pivot prefix tokens stay identical. Source (C++) is augmented independently.

Design invariant: the pivot prefix `return <pivot_prefix>` must be token-identical
across dest and target after augmentation. This is guaranteed for variable_rename
(same new_var applied to both) and toggle_type_hints (identical transformation).
Function names do not appear in the pivot prefix so they may differ between dest
and target.
"""

from __future__ import annotations

import copy
import logging
import re

from src.goals.augment import (
    _VAR_VOCAB,
    _toggle_type_hints_in_def,
    adapt_text_var,
)
from src.goals.head_loc_data import SinglePredicateSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SinglePredicateSample-aware helpers
# ---------------------------------------------------------------------------


def _rename_variable_sp(
    sample: SinglePredicateSample, old_var: str, new_var: str
) -> SinglePredicateSample:
    """Return a copy of ``sample`` with ``old_var`` renamed to ``new_var``.

    Applied to: ``python_incomplete_code``, ``cpp_incomplete_code``,
    ``variable_name``, and ``docstring`` (so the docstring hint remains consistent
    with the renamed variable).  C++ code uses the same word-boundary regex as
    Python — safe for typical single-character or short variable names.

    Args:
        sample: Original SinglePredicateSample (not mutated).
        old_var: Current variable name to replace.
        new_var: Replacement variable name.

    Returns:
        New SinglePredicateSample with the variable renamed throughout.
    """
    s = copy.deepcopy(sample)
    s.variable_name = new_var
    s.python_incomplete_code = adapt_text_var(s.python_incomplete_code, old_var, new_var)
    s.cpp_incomplete_code = adapt_text_var(s.cpp_incomplete_code, old_var, new_var)
    if s.docstring:
        s.docstring = adapt_text_var(s.docstring, old_var, new_var)
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [{"type": "variable_rename", "mapping": {old_var: new_var}}]
    return s


def _toggle_type_hints_sp(sample: SinglePredicateSample) -> SinglePredicateSample:
    """Return a copy of ``sample`` with parameter type annotations removed from the
    Python def line.

    The return annotation ``-> bool`` is preserved.  C++ code is not modified.
    The ``docstring`` field is left unchanged since it does not contain type hints.

    Args:
        sample: Original SinglePredicateSample (not mutated).

    Returns:
        New SinglePredicateSample with param annotations stripped from the def line.
    """
    s = copy.deepcopy(sample)
    lines = s.python_incomplete_code.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            lines[i] = _toggle_type_hints_in_def(line)
            break
    s.python_incomplete_code = "\n".join(lines)
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [{"type": "toggle_type_hints", "action": "remove"}]
    return s


def _rename_function_sp(sample: SinglePredicateSample, new_name: str) -> SinglePredicateSample:
    """Return a copy of ``sample`` with the Python function name changed to ``new_name``.

    Applied to: ``python_incomplete_code`` (def line only) and ``function_name``.
    The ``predicate_name``, ``docstring``, and C++ code are not modified.

    Args:
        sample: Original SinglePredicateSample (not mutated).
        new_name: New Python function name.

    Returns:
        New SinglePredicateSample with the Python function renamed.
    """
    s = copy.deepcopy(sample)
    old_name = s.function_name or s.predicate_name
    s.python_incomplete_code = re.sub(
        r"def\s+\w+\s*\(", f"def {new_name}(", s.python_incomplete_code, count=1
    )
    s.function_name = new_name
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [
        {"type": "function_rename", "old_name": old_name, "new_name": new_name}
    ]
    return s


def _rename_function_cpp(sample: SinglePredicateSample, new_name: str) -> SinglePredicateSample:
    """Return a copy of ``sample`` with the C++ function name changed to ``new_name``.

    Uses a C++-aware regex matching ``<old_name>(`` to avoid spurious replacements
    inside variable names or comments.  Python code is not modified by this helper
    (use ``_rename_function_sp`` for that).

    Args:
        sample: Original SinglePredicateSample (not mutated).
        new_name: New C++ function name.

    Returns:
        New SinglePredicateSample with the C++ function renamed.
    """
    s = copy.deepcopy(sample)
    old_name = s.function_name or s.predicate_name
    s.cpp_incomplete_code = re.sub(
        r"\b" + re.escape(old_name) + r"\s*\(",
        f"{new_name}(",
        s.cpp_incomplete_code,
        count=1,
    )
    prev = list(s.augmentation) if s.augmentation else []
    s.augmentation = prev + [
        {"type": "function_rename_cpp", "old_name": old_name, "new_name": new_name}
    ]
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_pair(
    dest: SinglePredicateSample,
    target: SinglePredicateSample,
    augmentation_type: str,
    config: dict | None = None,
) -> tuple[SinglePredicateSample, SinglePredicateSample]:
    """Apply the same augmentation to both ``dest`` and ``target`` jointly.

    Joint application preserves the invariant that the pivot prefix tokens are
    identical across dest and target after augmentation — required for the head
    localization experiment.

    Supported augmentation types:

    ``"variable_rename"`` — ``config: {"new_var": str}``
        Renames the variable using the SAME ``new_var`` for both samples.
        ``old_var`` is read from ``dest.variable_name`` (assumed identical on
        both by construction of the head-loc dataset).  Applied to
        ``python_incomplete_code``, ``cpp_incomplete_code``, ``variable_name``,
        and ``docstring``.

    ``"toggle_type_hints"`` — no config needed
        Removes parameter type annotations from the Python def line of both
        samples identically.  Preserves the ``-> bool`` return annotation and
        the ``docstring`` field.

    ``"function_rename"`` — ``config: {"dest_name": str, "target_name": str}``
        Renames the Python function.  Names can differ between dest and target
        because function names do not appear in the pivot prefix (``return …``).

    Args:
        dest: Destination SinglePredicateSample (not mutated).
        target: Target SinglePredicateSample (not mutated).
        augmentation_type: One of ``"variable_rename"``, ``"toggle_type_hints"``,
            ``"function_rename"``.
        config: Augmentation-specific configuration dict.  See above.

    Returns:
        ``(augmented_dest, augmented_target)`` tuple.

    Raises:
        ValueError: If ``augmentation_type`` is not recognised.
        KeyError: If a required key is missing from ``config``.
    """
    if config is None:
        config = {}

    if augmentation_type == "variable_rename":
        new_var = config["new_var"]
        old_var = dest.variable_name
        return (
            _rename_variable_sp(dest, old_var, new_var),
            _rename_variable_sp(target, old_var, new_var),
        )

    elif augmentation_type == "toggle_type_hints":
        return (
            _toggle_type_hints_sp(dest),
            _toggle_type_hints_sp(target),
        )

    elif augmentation_type == "function_rename":
        dest_name = config.get("dest_name", dest.function_name or dest.predicate_name)
        targ_name = config.get("target_name", target.function_name or target.predicate_name)
        return (
            _rename_function_sp(dest, dest_name),
            _rename_function_sp(target, targ_name),
        )

    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type!r}")


def augment_source(
    source: SinglePredicateSample,
    augmentation_type: str,
    config: dict | None = None,
) -> SinglePredicateSample:
    """Augment the C++ source predicate independently of the (dest, target) pair.

    Source augmentation is independent because the C++ code serves as the
    activation source in head-patching experiments and need not share pivot-prefix
    tokens with the Python samples.

    Supported augmentation types:

    ``"variable_rename"`` — ``config: {"new_var": str}``
        Renames the variable in both ``python_incomplete_code``,
        ``cpp_incomplete_code``, ``variable_name``, and ``docstring``.

    ``"toggle_type_hints"`` — no config needed
        Removes Python parameter type annotations from the def line.
        C++ has no Python-style type hints; this only affects
        ``python_incomplete_code``.

    ``"function_rename"`` — ``config: {"new_name": str}``
        Renames the function in both Python (``_rename_function_sp``) and C++
        (``_rename_function_cpp``) using the same ``new_name``.

    Args:
        source: Source SinglePredicateSample (not mutated).
        augmentation_type: One of ``"variable_rename"``, ``"toggle_type_hints"``,
            ``"function_rename"``.
        config: Augmentation-specific configuration dict.  See above.

    Returns:
        New SinglePredicateSample with the augmentation applied.

    Raises:
        ValueError: If ``augmentation_type`` is not recognised.
    """
    if config is None:
        config = {}

    if augmentation_type == "variable_rename":
        new_var = config["new_var"]
        old_var = source.variable_name
        return _rename_variable_sp(source, old_var, new_var)

    elif augmentation_type == "toggle_type_hints":
        # C++ has no Python-style type hints — apply Python toggle only
        return _toggle_type_hints_sp(source)

    elif augmentation_type == "function_rename":
        new_name = config["new_name"]
        s = _rename_function_sp(source, new_name)
        # Also update the C++ name
        return _rename_function_cpp(s, new_name)

    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type!r}")


def generate_pair_variants(
    dest: SinglePredicateSample,
    target: SinglePredicateSample,
    var_vocab: list[str] | None = None,
) -> list[tuple[SinglePredicateSample, SinglePredicateSample]]:
    """Generate the standard set of joint augmentation variants for a (dest, target) pair.

    Variants produced:
    - Up to 3 variable renames using alternatives from ``_VAR_VOCAB`` (or a
      custom ``var_vocab``).  The same new name is applied to both samples.
    - 1 type-hint toggle applied identically to both.

    Function renaming is intentionally excluded here because dest and target
    functions have different semantic roles; callers can apply ``augment_pair``
    with ``"function_rename"`` directly if needed.

    Args:
        dest: Destination SinglePredicateSample.
        target: Target SinglePredicateSample.
        var_vocab: Alternative variable names to try.  If None, uses the
            built-in ``_VAR_VOCAB`` lookup keyed on ``dest.variable_name``.

    Returns:
        List of ``(dest_variant, target_variant)`` tuples, one per variant.
    """
    variants: list[tuple[SinglePredicateSample, SinglePredicateSample]] = []

    old_var = dest.variable_name

    # Variable renames
    if var_vocab is None:
        var_vocab = _VAR_VOCAB.get(old_var, [])

    for new_var in var_vocab[:3]:
        if new_var == old_var:
            continue
        d, t = augment_pair(dest, target, "variable_rename", {"new_var": new_var})
        variants.append((d, t))

    # Type hint toggle
    d, t = augment_pair(dest, target, "toggle_type_hints")
    variants.append((d, t))

    return variants
