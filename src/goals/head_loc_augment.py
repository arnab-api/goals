"""Joint augmentation for (dest, target) SinglePredicateSample pairs.

Step 5 of the head localization pipeline: augment dest and target jointly so
their pivot prefix tokens stay identical. Source (C++) is augmented independently.

Design invariant: the pivot prefix `return <pivot_prefix>` must be token-identical
across dest and target after augmentation. This is guaranteed for variable_rename
(same new_var applied to both) and toggle_type_hints (identical transformation).
"""

from __future__ import annotations

import copy
import logging

from src.goals.augment import (
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

    Args:
        dest: Destination SinglePredicateSample (not mutated).
        target: Target SinglePredicateSample (not mutated).
        augmentation_type: One of ``"variable_rename"``, ``"toggle_type_hints"``.
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

    Args:
        source: Source SinglePredicateSample (not mutated).
        augmentation_type: One of ``"variable_rename"``, ``"toggle_type_hints"``.
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

    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type!r}")


