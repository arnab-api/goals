"""Intent block identification for code-LM generated Python.

Identifies hierarchical "intent blocks" in a Python source string using the
``ast`` module.  An intent block is a contiguous code region corresponding to
a single sub-goal of the overall task.

Design decisions (see ``claude_reports/goalscope.md`` for full rationale):
  D1  Input is clean Python code only — code extraction is a separate step.
  D2  Output is a typed dataclass tree with parent/children pointers.
  D3  No condition sub-scopes: ``while queue:`` condition tokens belong to
      the ``while`` block, not a dedicated ``while_condition`` child.
  D4  ``else``/``elif`` are siblings of ``if``, not children.
  D5  Every statement is a first-class intent block.
  D6  Innermost block is default for probing; full ancestor chain available.
  D7  Character spans only; no tokenizer dependency.
  D8  All granularity levels L1-L5 implemented together, controlled by
      the ``level`` parameter.

Public API
----------
``identify_intent_blocks(code, level=4)`` → root ``IntentBlock``
``get_blocks_at_char(root, char_pos)``    → ``list[IntentBlock]``  (root→innermost)
``extract_code_from_tags(text)``          → ``str | None``
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntentBlock:
    """A contiguous code region corresponding to a single sub-goal.

    Attributes:
        kind:       Block type string, one of:
                    "module", "function", "class",
                    "if", "elif", "else", "while", "for", "with",
                    "try", "except", "try_else", "finally",
                    "match", "case",
                    "listcomp", "setcomp", "dictcomp", "genexpr",
                    "stmt",
                    "boolop_operand", "compare_chain", "ifexp"
        char_start: Inclusive absolute character offset into the source string.
        char_end:   Exclusive absolute character offset.
        parent:     Parent block (None for root).
        children:   Ordered list of direct child blocks.
        label:      Sub-goal description, filled post-hoc by LM labeling.
    """

    kind: str
    char_start: int
    char_end: int
    parent: Optional["IntentBlock"] = field(default=None, repr=False)
    children: list["IntentBlock"] = field(default_factory=list, repr=False)
    label: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"IntentBlock(kind={self.kind!r}, "
            f"span=({self.char_start}, {self.char_end}), "
            f"label={self.label!r}, "
            f"n_children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_COMP_KINDS: dict[type, str] = {
    ast.ListComp: "listcomp",
    ast.SetComp: "setcomp",
    ast.DictComp: "dictcomp",
    ast.GeneratorExp: "genexpr",
}

_COMP_TYPES = tuple(_COMP_KINDS.keys())

# Simple statements that become "stmt" blocks at L4
_SIMPLE_STMT_TYPES = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Return,
    ast.Expr,
    ast.Delete,
    ast.Raise,
    ast.Assert,
    ast.Global,
    ast.Nonlocal,
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Import,
    ast.ImportFrom,
)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _line_offsets(code: str) -> list[int]:
    """Return list where ``offs[i]`` is the absolute char offset of line ``i``.

    Uses 1-based line indexing (``offs[0]`` is a placeholder 0).
    ``offs[1] == 0`` always (line 1 starts at char 0).
    """
    offs = [0]  # index 0 unused; pad so offs[1] = position of line 1
    pos = 0
    for line in code.split("\n"):
        offs.append(pos)
        pos += len(line) + 1  # +1 for the newline character
    return offs


def _node_span(node: ast.AST, offs: list[int]) -> tuple[int, int]:
    """Return ``(char_start, char_end)`` for an AST node."""
    start = offs[node.lineno] + node.col_offset  # type: ignore[attr-defined]
    end = offs[node.end_lineno] + node.end_col_offset  # type: ignore[attr-defined]
    return start, end


def _make_child(
    parent: IntentBlock,
    kind: str,
    node: ast.AST,
    offs: list[int],
) -> IntentBlock:
    """Create a child ``IntentBlock`` from an AST node and link it to parent."""
    start, end = _node_span(node, offs)
    block = IntentBlock(kind=kind, char_start=start, char_end=end, parent=parent)
    parent.children.append(block)
    return block


def _make_child_raw(
    parent: IntentBlock,
    kind: str,
    char_start: int,
    char_end: int,
) -> IntentBlock:
    """Create a child ``IntentBlock`` from raw char offsets and link to parent."""
    block = IntentBlock(
        kind=kind, char_start=char_start, char_end=char_end, parent=parent
    )
    parent.children.append(block)
    return block


def _find_keyword_before(code: str, char_pos: int, keyword: str) -> int:
    """Find the last occurrence of ``keyword`` as a whole word before ``char_pos``.

    Returns the start char of the match, or ``char_pos`` if not found.
    Used to locate ``else:``/``finally:`` keywords whose positions are not
    directly stored in Python AST nodes.
    """
    matches = list(re.finditer(r"\b" + re.escape(keyword) + r"\b", code[:char_pos]))
    return matches[-1].start() if matches else char_pos


def _find_innermost_containing(block: IntentBlock, start: int, end: int) -> IntentBlock:
    """Return the innermost existing child of ``block`` that fully contains ``[start, end)``.

    If no child fully contains the span, returns ``block`` itself.
    """
    for child in block.children:
        if child.char_start <= start and end <= child.char_end:
            return _find_innermost_containing(child, start, end)
    return block


# ---------------------------------------------------------------------------
# Expression block visitor (L3 comprehensions, L5 sub-expressions)
# ---------------------------------------------------------------------------


def _add_expr_blocks(
    expr_root: ast.AST,
    parent_block: IntentBlock,
    offs: list[int],
    level: int,
) -> None:
    """Walk ``expr_root``'s sub-expressions and add L3/L5 blocks to the tree.

    Uses ``ast.walk`` (BFS) so outer comprehensions are encountered before
    inner ones, ensuring correct nesting via ``_find_innermost_containing``.

    Args:
        expr_root:    Root AST node whose sub-expressions to inspect.
        parent_block: The block that will contain the newly created children.
        offs:         Line offsets from ``_line_offsets``.
        level:        Granularity level (3 = comprehensions, 5 = sub-expressions).
    """
    if level < 3:
        return

    for node in ast.walk(expr_root):
        if node is expr_root:
            continue

        # L3 — comprehension nodes
        if isinstance(node, _COMP_TYPES):
            c_start, c_end = _node_span(node, offs)
            container = _find_innermost_containing(parent_block, c_start, c_end)
            _make_child_raw(container, _COMP_KINDS[type(node)], c_start, c_end)
            continue  # children of this comp will be found in later walk iterations

        # L5 — sub-expression intent blocks
        if level < 5:
            continue

        if isinstance(node, ast.BoolOp):
            for val in node.values:
                v_start, v_end = _node_span(val, offs)
                container = _find_innermost_containing(parent_block, v_start, v_end)
                _make_child_raw(container, "boolop_operand", v_start, v_end)

        elif isinstance(node, ast.Compare):
            for part in [node.left, *node.comparators]:
                p_start, p_end = _node_span(part, offs)
                container = _find_innermost_containing(parent_block, p_start, p_end)
                _make_child_raw(container, "compare_chain", p_start, p_end)

        elif isinstance(node, ast.IfExp):
            for part in [node.body, node.test, node.orelse]:
                p_start, p_end = _node_span(part, offs)
                container = _find_innermost_containing(parent_block, p_start, p_end)
                _make_child_raw(container, "ifexp", p_start, p_end)


# ---------------------------------------------------------------------------
# Statement visitors
# ---------------------------------------------------------------------------


def _add_else_sibling(
    orelse_stmts: list[ast.stmt],
    parent: IntentBlock,
    code: str,
    offs: list[int],
    level: int,
    kind: str = "else",
) -> None:
    """Create an ``else`` block as a sibling of the preceding if/while/for."""
    orelse_first_start = _node_span(orelse_stmts[0], offs)[0]
    else_char = _find_keyword_before(code, orelse_first_start, "else")
    else_end = _node_span(orelse_stmts[-1], offs)[1]
    else_block = _make_child_raw(parent, kind, else_char, else_end)
    _visit_stmts(orelse_stmts, else_block, code, offs, level)


def _visit_if(
    node: ast.If,
    parent: IntentBlock,
    code: str,
    offs: list[int],
    level: int,
    is_elif: bool = False,
) -> None:
    """Visit an ``if``/``elif`` node (D4: else/elif are siblings of if).

    The block span ends at the last statement in the *if body*, not at the
    end of the entire if/elif/else construct — so sibling elif/else blocks
    do not overlap with the if block (which would confuse ``get_blocks_at_char``).
    """
    kind = "elif" if is_elif else "if"
    # Span: keyword start → end of this branch's own body only (not elif/else)
    start = _node_span(node, offs)[0]
    body_end = _node_span(node.body[-1], offs)[1]
    block = _make_child_raw(parent, kind, start, body_end)

    # D3: condition tokens belong to the if block, no sub-scope for condition.
    # But L3/L5 expression blocks within the condition are still created as
    # children of the if block.
    _add_expr_blocks(node.test, block, offs, level)

    # Body
    _visit_stmts(node.body, block, code, offs, level)

    # orelse — siblings of the if block (D4)
    if node.orelse:
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            # elif: recurse with parent (sibling), not block (child)
            _visit_if(node.orelse[0], parent, code, offs, level, is_elif=True)
        else:
            # else clause
            _add_else_sibling(node.orelse, parent, code, offs, level)


def _visit_try(
    node: ast.Try,
    parent: IntentBlock,
    code: str,
    offs: list[int],
    level: int,
) -> None:
    """Visit a ``try`` statement, creating sibling blocks for each clause."""
    # try body — span ends at body[-1] so sibling except/else/finally don't overlap
    if node.body:
        try_start = _node_span(node, offs)[0]
        try_end = _node_span(node.body[-1], offs)[1]
        try_block = _make_child_raw(parent, "try", try_start, try_end)
        _visit_stmts(node.body, try_block, code, offs, level)

    # except handlers (siblings)
    for handler in node.handlers:
        exc_block = _make_child(parent, "except", handler, offs)
        _visit_stmts(handler.body, exc_block, code, offs, level)

    # else clause (after try body, only when no exception raised)
    if node.orelse:
        _add_else_sibling(node.orelse, parent, code, offs, level, kind="try_else")

    # finally clause
    if node.finalbody:
        fin_first_start = _node_span(node.finalbody[0], offs)[0]
        fin_char = _find_keyword_before(code, fin_first_start, "finally")
        fin_end = _node_span(node.finalbody[-1], offs)[1]
        fin_block = _make_child_raw(parent, "finally", fin_char, fin_end)
        _visit_stmts(node.finalbody, fin_block, code, offs, level)


def _visit_stmt(
    node: ast.stmt,
    parent: IntentBlock,
    code: str,
    offs: list[int],
    level: int,
) -> None:
    """Dispatch a single statement node, creating child blocks on ``parent``."""

    # ── L1: function / class definitions ─────────────────────────────────────
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if level >= 1:
            block = _make_child(parent, "function", node, offs)
            _visit_stmts(node.body, block, code, offs, level)
        # level < 1: skip entirely (no blocks below module level)
        return

    if isinstance(node, ast.ClassDef):
        if level >= 1:
            block = _make_child(parent, "class", node, offs)
            _visit_stmts(node.body, block, code, offs, level)
        return

    # ── L2: control-flow ─────────────────────────────────────────────────────
    if isinstance(node, ast.If):
        if level >= 2:
            _visit_if(node, parent, code, offs, level, is_elif=False)
        else:
            # Don't create a block but still recurse
            _visit_stmts(node.body, parent, code, offs, level)
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    _visit_stmt(node.orelse[0], parent, code, offs, level)
                else:
                    _visit_stmts(node.orelse, parent, code, offs, level)
        return

    if isinstance(node, ast.While):
        if level >= 2:
            # Span ends at body end so the sibling else block doesn't overlap
            start = _node_span(node, offs)[0]
            body_end = _node_span(node.body[-1], offs)[1]
            block = _make_child_raw(parent, "while", start, body_end)
            _add_expr_blocks(node.test, block, offs, level)  # D3 + L3/L5 in cond
            _visit_stmts(node.body, block, code, offs, level)
            if node.orelse:
                _add_else_sibling(node.orelse, parent, code, offs, level)
        else:
            _visit_stmts(node.body, parent, code, offs, level)
            if node.orelse:
                _visit_stmts(node.orelse, parent, code, offs, level)
        return

    if isinstance(node, ast.For):
        if level >= 2:
            start = _node_span(node, offs)[0]
            body_end = _node_span(node.body[-1], offs)[1]
            block = _make_child_raw(parent, "for", start, body_end)
            _add_expr_blocks(node.iter, block, offs, level)
            _visit_stmts(node.body, block, code, offs, level)
            if node.orelse:
                _add_else_sibling(node.orelse, parent, code, offs, level)
        else:
            _visit_stmts(node.body, parent, code, offs, level)
            if node.orelse:
                _visit_stmts(node.orelse, parent, code, offs, level)
        return

    if isinstance(node, ast.With):
        if level >= 2:
            block = _make_child(parent, "with", node, offs)
            for item in node.items:
                _add_expr_blocks(item.context_expr, block, offs, level)
            _visit_stmts(node.body, block, code, offs, level)
        else:
            _visit_stmts(node.body, parent, code, offs, level)
        return

    if isinstance(node, ast.Try):
        if level >= 2:
            _visit_try(node, parent, code, offs, level)
        else:
            _visit_stmts(node.body, parent, code, offs, level)
            for h in node.handlers:
                _visit_stmts(h.body, parent, code, offs, level)
            if node.orelse:
                _visit_stmts(node.orelse, parent, code, offs, level)
            if node.finalbody:
                _visit_stmts(node.finalbody, parent, code, offs, level)
        return

    # Python 3.10+ match statement (treat like a compound at L2)
    if hasattr(ast, "Match") and isinstance(node, ast.Match):  # type: ignore[attr-defined]
        if level >= 2:
            block = _make_child(parent, "match", node, offs)
            for case in node.cases:  # type: ignore[attr-defined]
                case_block = _make_child(parent, "case", case, offs)
                _visit_stmts(case.body, case_block, code, offs, level)
        return

    # Python 3.11+ TryStar (except*)
    if hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):  # type: ignore[attr-defined]
        if level >= 2:
            _visit_try(node, parent, code, offs, level)  # type: ignore[arg-type]
        return

    # ── L4: simple statements ─────────────────────────────────────────────────
    if isinstance(node, _SIMPLE_STMT_TYPES):
        if level >= 4:
            block = _make_child(parent, "stmt", node, offs)
            _add_expr_blocks(node, block, offs, level)
        elif level >= 3:
            # No stmt block, but still identify comprehensions in the expression
            _add_expr_blocks(node, parent, offs, level)
        return

    # Fallback: unknown statement type — recurse if it has a body
    for field_name, value in ast.iter_fields(node):
        if field_name == "body" and isinstance(value, list):
            _visit_stmts(value, parent, code, offs, level)


def _visit_stmts(
    stmts: list[ast.stmt],
    parent: IntentBlock,
    code: str,
    offs: list[int],
    level: int,
) -> None:
    """Visit a list of statement nodes, adding child blocks to ``parent``."""
    for stmt in stmts:
        _visit_stmt(stmt, parent, code, offs, level)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_intent_blocks(code: str, level: int = 5) -> IntentBlock:
    """Parse Python source and return the root ``IntentBlock`` tree.

    Args:
        code:  Well-formed Python source string (D1: no extraction needed).
        level: Granularity level:
               1 = functions only
               2 = + control-flow (if/while/for/with/try)
               3 = + comprehensions (list/set/dict/gen)
               4 = + statements (assign/return/expr/…)  [default]
               5 = + sub-expressions (boolop operands, compare chains, ifexp)

    Returns:
        Root ``IntentBlock`` with ``kind="module"`` spanning the entire code.

    Raises:
        SyntaxError: If ``code`` is not valid Python.
    """
    tree = ast.parse(code)
    offs = _line_offsets(code)
    root = IntentBlock(kind="module", char_start=0, char_end=len(code))
    _visit_stmts(tree.body, root, code, offs, level)
    return root


def get_blocks_at_char(root: IntentBlock, char_pos: int) -> list[IntentBlock]:
    """Return the ancestor chain ``[root, …, innermost]`` for ``char_pos``.

    Each block in the chain fully contains ``char_pos``.
    Returns an empty list if ``char_pos`` is outside ``root``'s span.

    Args:
        root:     Root of the ``IntentBlock`` tree.
        char_pos: Absolute character offset into the source string.

    Returns:
        List of blocks from root to innermost, or ``[]`` if not contained.
    """
    if not (root.char_start <= char_pos < root.char_end):
        return []

    chain: list[IntentBlock] = [root]
    node = root
    while True:
        found = False
        for child in node.children:
            if child.char_start <= char_pos < child.char_end:
                chain.append(child)
                node = child
                found = True
                break
        if not found:
            break
    return chain


def extract_code_from_tags(text: str) -> Optional[str]:
    """Extract Python source from the first ``<code>…</code>`` block.

    Args:
        text: Raw text possibly containing ``<code>…</code>`` tags.

    Returns:
        Content inside the tags (stripped), or ``None`` if not found.
    """
    m = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if m is None:
        return None
    return m.group(1).strip()
