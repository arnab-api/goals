"""Surface-form helpers shared by the head-localization augmentation
(``src.goals.head_loc_augment`` and ``src.goals.head_loc_attn``): the variable-name
vocabulary, word-boundary variable substitution, and the type-hint toggle.
"""

from __future__ import annotations

import logging
import re

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def adapt_text_var(text: str, old_var: str, new_var: str) -> str:
    """Word-boundary substitution of old_var → new_var in text."""
    return re.sub(r"\b" + re.escape(old_var) + r"\b", new_var, text)


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
