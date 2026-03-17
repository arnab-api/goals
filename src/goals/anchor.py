"""Anchor token extraction and predicate span labeling for multi-predicate MCTS trees.

Provides two main capabilities:

1. **Anchor extraction** — oracle (Claude) identifies short code substrings that
   uniquely signal P+ vs P- at the branching point of generation.

2. **Span labeling** — given anchor tokens, labels every node in an MCTS tree
   with which predicate (P+ first / P- first / mixed / unlabeled) the subtree
   rooted at that node is heading toward.  Labels live in a returned dict
   (MCTSNode stays clean).

Typical usage:

    from src.goals.anchor import (
        get_green_completions,
        extract_anchor_tokens,
        label_predicate_spans,
        show_span_annotation,
        print_labeled_tree,
    )

    # ── Step 1: extract anchor tokens (oracle) ───────────────────────────────
    p_plus_anchors, p_minus_anchors_per_sample = extract_anchor_tokens(
        tp, completions_per_sample, use_cache=True,
    )
    tp.p_plus.anchor_tokens = p_plus_anchors
    for i, sample in enumerate(tp.samples):
        if sample.p_minus:
            sample.p_minus.anchor_tokens = p_minus_anchors_per_sample[i]
    tp.save()

    # ── Step 2: label spans in a tree ────────────────────────────────────────
    labels = label_predicate_spans(
        root=root,
        p_plus_anchors=tp.p_plus.anchor_tokens,
        p_minus_anchors=tp.samples[sample_idx].p_minus.anchor_tokens,
    )

    # ── Step 3: inspect ───────────────────────────────────────────────────────
    print_labeled_tree(root, labels)
    for leaf, path_text, label in get_labeled_leaves(root, labels):
        print(show_span_annotation(leaf, tp.p_plus.anchor_tokens,
                                   tp.samples[sample_idx].p_minus.anchor_tokens))
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from src.goals.predicate_data import MultiPredicateSample, TargetPredicate
from src.utils.oracle_llms import ask_claude

if TYPE_CHECKING:
    from src.goals.sampling import MCTSNode

logger = logging.getLogger(__name__)


def _re_search_safe(pattern: str, text: str) -> "re.Match | None":
    """re.search with graceful handling of invalid regex patterns."""
    try:
        return re.search(pattern, text)
    except re.error:
        return None


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_ANCHOR_PROMPT = """\
You are labeling branches in a code-generation search tree. A language model is completing a Python function stub that must implement two independent boolean predicates:

- **P+** (target predicate): {p_plus_description}
- **P-** (companion predicate): {p_minus_description}

Function stub:
```python
{code_stub}
```

One valid green completion (passes all unit tests):
```
return {completion}
```

**Task**: Identify *anchor patterns* — Python regex patterns that match the earliest distinctive code fragment signaling which predicate is being implemented at the branching point (right after `return`).

- **P+ anchor patterns**: regex(es) that match the earliest code fragment(s) uniquely indicating P+ is being implemented. Must NOT normally match a P--only implementation.
- **P- anchor patterns**: same for P-. Must NOT normally match a P+-only implementation.

Rules:
- Focus on fragments near the *start* of each predicate's code in the completion.
- Prefer short, distinctive patterns (e.g. `\\[::-1\\]`, `len\\(`, `\\.isalpha\\(\\)`).
- **Do NOT include specific variable names** from the function signature. Use `\\w+` in their place (e.g. use `\\w+\\[::-1\\]` instead of `s\\[::-1\\]`, use `len\\(\\w+\\)` instead of `len\\(s\\)`).
- Include 1–3 patterns per predicate — only the most distinctive ones.
- Each pattern must be a valid Python regex string.

Respond with ONLY a JSON object, no markdown fences, no explanation:
{{"p_plus_anchor_patterns": ["...", "..."], "p_minus_anchor_patterns": ["...", "..."]}}"""


# ---------------------------------------------------------------------------
# PredicateLabel and ClauseSpan dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PredicateLabel:
    """Label assigned to one MCTSNode describing the predicate ordering in its subtree.

    Attributes:
        label:               "p_plus_first"  — all reachable green leaves implement P+ before P-.
                             "p_minus_first" — all reachable green leaves implement P- before P+.
                             "mixed"         — green leaves disagree on ordering (contrastive point).
                             "unlabeled"     — no green leaves reachable, or no anchor matched.
        p_plus_count:        # green leaf descendants where a P+ anchor was found.
        p_minus_count:       # green leaf descendants where a P- anchor was found.
        p_plus_start_depth:  Meaningful only for green leaf nodes.  Depth of the first node
                             whose token causes the P+ anchor to appear in the path text
                             (prefix included).  -1 = P+ anchor not found in this path.
        p_minus_start_depth: Same as above for P-.
    """

    label: str
    p_plus_count: int
    p_minus_count: int
    p_plus_start_depth: int = -1  # leaf-only: depth where P+ span begins
    p_minus_start_depth: int = -1  # leaf-only: depth where P- span begins

    def __repr__(self) -> str:
        return (
            f"PredicateLabel({self.label!r}, "
            f"p+={self.p_plus_count}, p-={self.p_minus_count})"
        )


@dataclass
class ClauseSpan:
    """The full predicate clause identified for one predicate within a green leaf path.

    Attributes:
        text:        Stripped clause text, e.g. "timeout > 0" or "host != ''".
        start_depth: Depth of the first node whose token boundary passes the clause start.
                     -1 means the clause start falls inside the prefix (before MCTS tokens).
        end_depth:   Depth of the first node whose token boundary passes the clause end.
                     -1 means the clause end falls inside the prefix.
    """

    text: str
    start_depth: int
    end_depth: int

    def __repr__(self) -> str:
        return f"ClauseSpan({self.text!r}, depth {self.start_depth}→{self.end_depth})"


# ---------------------------------------------------------------------------
# Internal helpers — anchor extraction
# ---------------------------------------------------------------------------


def _build_prompt(
    sample: MultiPredicateSample,
    tp: TargetPredicate,
    completion: str,
) -> str:
    p_minus_desc = sample.p_minus.description if sample.p_minus else "(none)"
    return _ANCHOR_PROMPT.format(
        p_plus_description=tp.p_plus.description,
        p_minus_description=p_minus_desc,
        code_stub=sample.code.rstrip(),
        completion=completion.strip(),
    )


def _parse_response(response: str) -> tuple[list[str], list[str]]:
    """Extract anchor token lists from oracle response.

    Handles both raw JSON and JSON wrapped in markdown code fences.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in oracle response:\n{response}")
    data = json.loads(match.group(0))
    # Support both old key names (cache compat) and new names
    p_plus = data.get("p_plus_anchor_patterns", data.get("p_plus_anchor_tokens", []))
    p_minus = data.get("p_minus_anchor_patterns", data.get("p_minus_anchor_tokens", []))
    if not isinstance(p_plus, list) or not isinstance(p_minus, list):
        raise ValueError(f"Expected lists in oracle response, got: {data}")
    return [str(t) for t in p_plus], [str(t) for t in p_minus]


# ---------------------------------------------------------------------------
# Internal helpers — span labeling
# ---------------------------------------------------------------------------


def _get_path_nodes(leaf: "MCTSNode") -> list["MCTSNode"]:
    """Return non-root nodes from root-child down to leaf (inclusive)."""
    nodes: list[MCTSNode] = []
    node = leaf
    while node is not None and not node.is_root():
        nodes.append(node)
        node = node.parent
    nodes.reverse()
    return nodes


def _label_leaf(
    leaf: "MCTSNode",
    prefix: str,
    p_plus_anchors: list[str],
    p_minus_anchors: list[str],
) -> tuple[int, int]:
    """Scan (prefix + root→leaf tokens) to find where P+ and P- spans start.

    Returns:
        (p_plus_start_depth, p_minus_start_depth)
        Each is the depth of the first MCTSNode whose token causes the anchor
        to appear in the cumulative text (prefix included).
        -1 means the anchor was not found anywhere in this path.

    Note: if an anchor first appears inside the prefix text itself (char_pos <
    len(prefix)), the returned depth is the depth of the first path node — i.e.
    the predicate was already established before the MCTS tokens began.
    """
    path_nodes = _get_path_nodes(leaf)
    full_text = prefix + "".join(n.token_str for n in path_nodes)

    def _first_char_pos(patterns: list[str]) -> int | None:
        best: int | None = None
        for p in patterns:
            try:
                m = re.search(p, full_text)
            except re.error:
                continue
            if m is not None and (best is None or m.start() < best):
                best = m.start()
        return best

    p_plus_char = _first_char_pos(p_plus_anchors)
    p_minus_char = _first_char_pos(p_minus_anchors)

    def _depth_for(char_pos: int) -> int:
        """Depth of the first node whose token boundary passes char_pos."""
        cumlen = len(prefix)
        for node in path_nodes:
            cumlen += len(node.token_str)
            if cumlen > char_pos:
                return node.depth
        return leaf.depth  # shouldn't reach here

    p_plus_depth = _depth_for(p_plus_char) if p_plus_char is not None else -1
    p_minus_depth = _depth_for(p_minus_char) if p_minus_char is not None else -1
    return p_plus_depth, p_minus_depth


def _propagate(
    node: "MCTSNode",
    leaf_depths: dict["MCTSNode", tuple[int, int]],
    result: dict["MCTSNode", PredicateLabel],
) -> PredicateLabel:
    """Post-order DFS: aggregate leaf span info bottom-up into PredicateLabel entries."""
    if node.is_leaf():
        if node in leaf_depths:
            p_plus_depth, p_minus_depth = leaf_depths[node]
            found_p = p_plus_depth != -1
            found_m = p_minus_depth != -1
            if found_p and found_m:
                label = (
                    "p_plus_first" if p_plus_depth <= p_minus_depth else "p_minus_first"
                )
            elif found_p:
                label = "p_plus_first"
            elif found_m:
                label = "p_minus_first"
            else:
                label = "unlabeled"
            pl = PredicateLabel(
                label=label,
                p_plus_count=1 if found_p else 0,
                p_minus_count=1 if found_m else 0,
                p_plus_start_depth=p_plus_depth,
                p_minus_start_depth=p_minus_depth,
            )
        else:
            # red / orange leaf — not labeled
            pl = PredicateLabel(label="unlabeled", p_plus_count=0, p_minus_count=0)
        result[node] = pl
        return pl

    total_p_plus = 0
    total_p_minus = 0
    for child in node.children:
        child_pl = _propagate(child, leaf_depths, result)
        total_p_plus += child_pl.p_plus_count
        total_p_minus += child_pl.p_minus_count

    if total_p_plus > 0 and total_p_minus > 0:
        label = "mixed"
    elif total_p_plus > 0:
        label = "p_plus_first"
    elif total_p_minus > 0:
        label = "p_minus_first"
    else:
        label = "unlabeled"

    pl = PredicateLabel(
        label=label,
        p_plus_count=total_p_plus,
        p_minus_count=total_p_minus,
    )
    result[node] = pl
    return pl


# ---------------------------------------------------------------------------
# Internal helpers — clause splitting
# ---------------------------------------------------------------------------


def _extract_boolean_expr(full_text: str) -> tuple[str, int] | None:
    """Return (expr_text, expr_start_char) for the first boolean expression found.

    Handles these patterns (in precedence order):
      'if <expr>:'      terminates at first unbracketed ':'
      'return <expr>'   terminates at first unbracketed newline or 'else'
      'while <expr>:'   terminates at first unbracketed ':'
      'assert <expr>'   terminates at first unbracketed newline or ','

    A depth-0 'else' keyword always marks the default branch of a Python ternary
    expression ('X if COND else Y') and is never part of the boolean predicate.
    Stopping before it gives the correct condition for all patterns:
      'return X if COND else Y'  → condition = 'X if COND'
      'return COND else Y'       → condition = 'COND'

    If no terminator is found (partial generation), returns everything after
    the keyword so partial expressions can still be split into clauses.
    Returns None if none of the keywords are found.
    """
    m = re.search(r"\b(if|return|while|assert)\s+", full_text)
    if m is None:
        return None
    keyword = m.group(1)
    expr_start = m.end()
    rest = full_text[expr_start:]

    colon_terminated = keyword in ("if", "while")
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            if colon_terminated and ch == ":":
                return rest[:i], expr_start
            elif not colon_terminated and ch == "\n":
                return rest[:i], expr_start
            elif keyword == "assert" and ch == ",":
                return rest[:i], expr_start
            # Stop at depth-0 'else': terminates ternary-if default branch
            # ('X if COND else Y' → return 'X if COND', never includes 'else Y')
            elif ch == "e" and rest[i : i + 4] == "else":
                prev_ok = i == 0 or not (rest[i - 1].isalnum() or rest[i - 1] == "_")
                next_ok = i + 4 >= len(rest) or not (
                    rest[i + 4].isalnum() or rest[i + 4] == "_"
                )
                if prev_ok and next_ok:
                    return rest[:i].rstrip(), expr_start

    # No terminator found — return everything (partial/non-terminal node)
    return rest, expr_start


def _split_clauses(
    condition: str,
    cond_start: int,
) -> list[tuple[str, int, int]]:
    """Split a condition string by top-level 'and'/'or' keywords.

    Respects bracket nesting so that 'and'/'or' inside parentheses are ignored.
    Returns a list of (clause_text, abs_start, abs_end) where abs_* are char
    positions in the original full_text (cond_start already added).
    """
    # Collect (start, end) positions of top-level 'and'/'or' within `condition`
    sep_spans: list[tuple[int, int]] = []
    depth = 0
    i = 0
    while i < len(condition):
        ch = condition[i]
        if ch in "([{":
            depth += 1
            i += 1
        elif ch in ")]}":
            depth -= 1
            i += 1
        elif depth == 0:
            matched = False
            for kw in ("and", "or"):
                end = i + len(kw)
                if condition[i:end].lower() == kw:
                    before_ok = i == 0 or not (
                        condition[i - 1].isalnum() or condition[i - 1] == "_"
                    )
                    after_ok = end >= len(condition) or not (
                        condition[end].isalnum() or condition[end] == "_"
                    )
                    if before_ok and after_ok:
                        sep_spans.append((i, end))
                        i = end
                        matched = True
                        break
            if not matched:
                i += 1
        else:
            i += 1

    # Build clause list from the gaps between separators
    clauses: list[tuple[str, int, int]] = []
    prev = 0
    for sep_start, sep_end in sep_spans:
        raw = condition[prev:sep_start]
        text = raw.strip()
        if text:
            lstrip = len(raw) - len(raw.lstrip())
            clauses.append((text, cond_start + prev + lstrip, cond_start + sep_start))
        prev = sep_end
    raw = condition[prev:]
    text = raw.strip()
    if text:
        lstrip = len(raw) - len(raw.lstrip())
        clauses.append((text, cond_start + prev + lstrip, cond_start + len(condition)))
    return clauses


def _char_to_depth(
    char_pos: int,
    prefix_len: int,
    path_nodes: list,
) -> int:
    """Map a char position in (prefix + path_tokens) to the depth of the covering node.

    Returns -1 if char_pos falls entirely within the prefix (before any MCTS token).
    """
    if char_pos < prefix_len:
        return -1
    cumlen = prefix_len
    for node in path_nodes:
        cumlen += len(node.token_str)
        if cumlen > char_pos:
            return node.depth
    return path_nodes[-1].depth if path_nodes else -1


# ---------------------------------------------------------------------------
# Public — anchor extraction
# ---------------------------------------------------------------------------


def get_green_completions(root: "MCTSNode", prefix: str) -> list[str]:
    """Return the completion text (after prefix) for every green leaf node.

    Args:
        root:   Root MCTSNode after evaluate_mcts_tree has been called.
        prefix: The prompt string passed as the MCTS root context
                (typically instruction + code stub up to `return `).

    Returns:
        List of completion strings (text after prefix), one per green leaf.
    """
    results: list[str] = []

    def _walk(node: "MCTSNode") -> None:
        if node.is_leaf() and node.color == "green":
            full_text = prefix + node.get_path_text()
            results.append(full_text[len(prefix) :])
        for child in node.children:
            _walk(child)

    _walk(root)
    return results


def extract_anchor_tokens(
    tp: TargetPredicate,
    completions_per_sample: list[dict[str, list[str]]],
    use_cache: bool = True,
    temperature: float = 0.0,
) -> tuple[list[str], list[list[str]]]:
    """Ask oracle to identify P+ / P- anchor tokens from green completions.

    For each green completion, prompts the oracle (Claude) to identify the short
    code substrings that uniquely signal whether the model is implementing P+ or
    P- at the branching point of the generation tree.

    This function is **read-only** with respect to `tp` — it never modifies
    anchor_tokens.  The caller decides what to do with the returned lists.

    Args:
        tp: TargetPredicate (used for descriptions; not modified).
        completions_per_sample: completions_per_sample[i] is a dict mapping
            prefix → list of unique green completions for sample i.  Each
            (root, prefix) pair is a separate MCTS run; a sample may have
            multiple runs if you ran MCTS under several prefills.
            Build each entry using get_green_completions(root, prefix).
        use_cache: Whether to read/write cached oracle responses (default True).
        temperature: Sampling temperature for oracle (0 = deterministic JSON).

    Returns:
        (p_plus_anchors, p_minus_anchors_per_sample) — both as sorted lists.
        - p_plus_anchors: aggregated P+ anchors across all samples and prefixes.
        - p_minus_anchors_per_sample[i]: P- anchors for sample i.
    """
    if len(completions_per_sample) != len(tp.samples):
        raise ValueError(
            f"completions_per_sample has {len(completions_per_sample)} entries "
            f"but tp '{tp.name}' has {len(tp.samples)} samples."
        )

    all_p_plus: set[str] = set(tp.p_plus.anchor_patterns)
    p_minus_per_sample: list[set[str]] = [
        set(sample.p_minus.anchor_patterns if sample.p_minus else [])
        for sample in tp.samples
    ]

    def _identified(completion: str, patterns: set[str]) -> bool:
        return bool(patterns) and any(_re_search_safe(p, completion) for p in patterns)

    for sample_idx, (sample, prefix_to_comps) in enumerate(
        zip(tp.samples, completions_per_sample)
    ):
        if sample.p_minus is None:
            logger.warning(
                f"[{tp.name}] sample {sample_idx} has no p_minus — skipping."
            )
            continue

        for prefix, completions in prefix_to_comps.items():
            for completion in completions:
                if not completion.strip():
                    continue

                if _identified(completion, all_p_plus) and _identified(
                    completion, p_minus_per_sample[sample_idx]
                ):
                    logger.debug(
                        f"[{tp.name}] sample={sample_idx}: both predicates already "
                        f"identified in completion — skipping oracle call."
                    )
                    continue

                prompt = _build_prompt(sample, tp, completion)
                try:
                    response = ask_claude(
                        prompt=prompt,
                        max_tokens=256,
                        temperature=temperature,
                        use_cache=use_cache,
                    )
                    p_plus_new, p_minus_new = _parse_response(response)
                    all_p_plus.update(p_plus_new)
                    p_minus_per_sample[sample_idx].update(p_minus_new)
                    logger.debug(
                        f"[{tp.name}] sample={sample_idx} prefix={prefix[:30]!r}: "
                        f"p+={p_plus_new}  p-={p_minus_new}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{tp.name}] sample={sample_idx} prefix={prefix[:30]!r}: "
                        f"oracle call failed — {e}"
                    )

    # ── Prune anchors that don't appear in any completion ────────────────────
    all_comps_flat: list[str] = [
        c.strip()
        for ptc in completions_per_sample
        for cs in ptc.values()
        for c in cs
        if c.strip()
    ]
    per_sample_comps: list[list[str]] = [
        [c.strip() for cs in ptc.values() for c in cs if c.strip()]
        for ptc in completions_per_sample
    ]

    valid_p_plus = {
        p for p in all_p_plus if any(_re_search_safe(p, c) for c in all_comps_flat)
    }
    pruned = all_p_plus - valid_p_plus
    if pruned:
        logger.info(
            f"[{tp.name}] pruned p+ patterns (no completion match): {sorted(pruned)}"
        )
    all_p_plus = valid_p_plus

    for i, sample_comps in enumerate(per_sample_comps):
        valid = {
            p
            for p in p_minus_per_sample[i]
            if any(_re_search_safe(p, c) for c in sample_comps)
        }
        pruned = p_minus_per_sample[i] - valid
        if pruned:
            logger.info(
                f"[{tp.name}] sample {i}: pruned p- patterns (no completion match): {sorted(pruned)}"
            )
        p_minus_per_sample[i] = valid

    p_plus_anchors = sorted(all_p_plus)
    p_minus_anchors = [sorted(s) for s in p_minus_per_sample]

    logger.info(f"[{tp.name}] p+ patterns (final): {p_plus_anchors}")
    for i, anchors in enumerate(p_minus_anchors):
        if anchors:
            logger.info(f"[{tp.name}] sample {i} p- patterns (final): {anchors}")

    return p_plus_anchors, p_minus_anchors


# ---------------------------------------------------------------------------
# Public — conversion utility
# ---------------------------------------------------------------------------


def convert_anchor_tokens_to_patterns(
    anchor_tokens: list[str],
    var_names: list[str],
) -> list[str]:
    """Convert literal anchor token strings to variable-agnostic regex patterns.

    For each anchor token:
    1. Replace whole-word occurrences of each var_name with a NUL placeholder.
    2. re.escape() the remaining literal text.
    3. Replace placeholders with \\w+.

    This makes patterns robust to variable renaming: "s[::-1]" with var "s"
    becomes "\\w+\\[::-1\\]", which still matches "text[::-1]" after renaming.

    Tokens that contain no var_names are simply re.escape()'d (safe default).

    Args:
        anchor_tokens: List of literal anchor strings (current JSON format).
        var_names:     List of variable names to strip (typically the function
                       parameter names extracted from the code stub).

    Returns:
        List of regex pattern strings, same length as anchor_tokens.

    Example:
        >>> convert_anchor_tokens_to_patterns(["s[::-1]", "len(s)", "[::-1]"], ["s"])
        ['\\\\w+\\\\[::-1\\\\]', 'len\\\\(\\\\w+\\\\)', '\\\\[::-1\\\\]']
    """
    PLACEHOLDER = "\x00VAR\x00"
    results: list[str] = []
    for token in anchor_tokens:
        s = token
        # Replace longest variable names first to avoid partial matches
        for var in sorted(var_names, key=len, reverse=True):
            s = re.sub(r"\b" + re.escape(var) + r"\b", PLACEHOLDER, s)
        parts = s.split(PLACEHOLDER)
        results.append(r"\w+".join(re.escape(p) for p in parts))
    return results


# ---------------------------------------------------------------------------
# Public — span labeling
# ---------------------------------------------------------------------------


def label_predicate_spans(
    root: "MCTSNode",
    p_plus_anchors: list[str],
    p_minus_anchors: list[str],
    prefix: str = "",
) -> dict["MCTSNode", PredicateLabel]:
    """Label every node in the MCTS tree with predicate-span information.

    Two passes:
      1. For each green leaf, scan (prefix + root→leaf token text) to find the
         start depth of the P+ span AND the P- span within that branch.
         Both spans are tracked independently — a branch typically implements
         both predicates, just in different order.
      2. Post-order DFS propagates leaf info upward: a node is "mixed" when its
         green descendants disagree on ordering — these are the contrastive
         branching points of interest.

    Only green leaves contribute to propagation; red/orange leaves are ignored.

    Args:
        root:            Root MCTSNode (must be evaluated — colors set).
        p_plus_anchors:  Anchor tokens for P+ (from extract_anchor_tokens).
        p_minus_anchors: Anchor tokens for P- (from extract_anchor_tokens).
        prefix:          Text prepended before the MCTS tokens (the full prompt
                         up to the root, i.e. run.prefix).  Needed so that
                         anchors present in the prefill are correctly detected.

    Returns:
        dict mapping every MCTSNode → PredicateLabel.
        For green leaf nodes, p_plus_start_depth / p_minus_start_depth record
        where each predicate's span begins (-1 if not found).
        For internal nodes, "mixed" marks contrastive branching points.
    """
    # ── Pass 1: find span start depths for every green leaf ──────────────────
    leaf_depths: dict["MCTSNode", tuple[int, int]] = {}

    def _collect_green_leaves(node: "MCTSNode") -> None:
        if node.is_leaf():
            if node.color == "green":
                leaf_depths[node] = _label_leaf(
                    node, prefix, p_plus_anchors, p_minus_anchors
                )
            return
        for child in node.children:
            _collect_green_leaves(child)

    _collect_green_leaves(root)

    n_both = sum(1 for dp, dm in leaf_depths.values() if dp != -1 and dm != -1)
    n_p_only = sum(1 for dp, dm in leaf_depths.values() if dp != -1 and dm == -1)
    n_m_only = sum(1 for dp, dm in leaf_depths.values() if dp == -1 and dm != -1)
    n_none = sum(1 for dp, dm in leaf_depths.values() if dp == -1 and dm == -1)
    logger.debug(
        f"label_predicate_spans: {len(leaf_depths)} green leaves — "
        f"both={n_both}, p+_only={n_p_only}, p-_only={n_m_only}, unlabeled={n_none}"
    )

    # ── Pass 2: propagate upward ──────────────────────────────────────────────
    result: dict["MCTSNode", PredicateLabel] = {}
    _propagate(root, leaf_depths, result)
    return result


# ---------------------------------------------------------------------------
# Inspection utilities
# ---------------------------------------------------------------------------

_LABEL_STYLES = {
    "p_plus_first": ("bold bright_green", "P+"),
    "p_minus_first": ("bold magenta", "P-"),
    "mixed": ("bold bright_yellow", "±"),
    "unlabeled": ("dim white", "?"),
}


def show_span_annotation(
    leaf: "MCTSNode",
    p_plus_anchors: list[str],
    p_minus_anchors: list[str],
    prefix: str = "",
    label: "PredicateLabel | None" = None,
) -> str:
    """Return a multi-line string annotating where P+ and P- spans start in a leaf path.

    Searches in (prefix + path_text) so that anchors in the prefill are shown.

    Example output:
        Path (after prefix): ' n % 2 == 0'
          P+ anchor '%' at char 2  (depth 3):  n [%] 2 == 0
          P- anchor 'not' at char 0 (depth 1): [not] found in prefix
          First predicate: P-  (both found)

    Args:
        leaf:            A green MCTSNode leaf.
        p_plus_anchors:  P+ anchor token list.
        p_minus_anchors: P- anchor token list.
        prefix:          Prompt text prepended before the MCTS path.
        label:           Optional PredicateLabel from label_predicate_spans()
                         (used to show start-depth alongside the char position).

    Returns:
        Formatted annotation string (plain text, no rich markup).
    """
    path_text = leaf.get_path_text()
    full_text = prefix + path_text
    lines = [f"  Path (after prefix): {path_text!r}"]

    def _find(patterns: list[str], pred_name: str) -> int | None:
        best_pos: int | None = None
        best_pattern: str = ""
        best_match_len: int = 0
        for pattern in patterns:
            m = _re_search_safe(pattern, full_text)
            if m is not None and (best_pos is None or m.start() < best_pos):
                best_pos = m.start()
                best_pattern = pattern
                best_match_len = len(m.group(0))
        if best_pos is not None:
            in_prefix = best_pos < len(prefix)
            rel_pos = best_pos - len(prefix)  # position within path_text
            start = max(0, best_pos - 8)
            snippet = full_text[start : best_pos + best_match_len + 12]
            marker = " " * (best_pos - start) + "^" * best_match_len
            depth_str = ""
            if label is not None:
                d = (
                    label.p_plus_start_depth
                    if pred_name == "P+"
                    else label.p_minus_start_depth
                )
                depth_str = f" (depth {d})" if d != -1 else ""
            loc = (
                f"in prefix at char {best_pos}"
                if in_prefix
                else f"at char {rel_pos} in path"
            )
            lines.append(f"  {pred_name} pattern {best_pattern!r} — {loc}{depth_str}:")
            lines.append(f"    {snippet!r}")
            lines.append(f"    {marker}")
        else:
            lines.append(f"  {pred_name} pattern: (none found)")
        return best_pos

    p_plus_pos = _find(p_plus_anchors, "P+")
    p_minus_pos = _find(p_minus_anchors, "P-")

    if p_plus_pos is None and p_minus_pos is None:
        lines.append("  → unlabeled (no anchor found)")
    elif p_plus_pos is None:
        lines.append("  → P- only (P+ anchor not found)")
    elif p_minus_pos is None:
        lines.append("  → P+ only (P- anchor not found)")
    elif p_plus_pos <= p_minus_pos:
        lines.append("  → P+ first, then P-  (both spans present)")
    else:
        lines.append("  → P- first, then P+  (both spans present)")

    return "\n".join(lines)


def extract_predicate_clauses(
    leaf: "MCTSNode",
    prefix: str,
    p_plus_anchors: list[str],
    p_minus_anchors: list[str],
) -> tuple["ClauseSpan | None", "ClauseSpan | None"]:
    """Identify the full P+ and P- predicate clauses within a green leaf path.

    Splits the if-condition on top-level 'and'/'or' separators and finds which
    clause contains each predicate's anchor token.  This gives the whole clause
    text (e.g. "timeout > 0") and the node depth range that generates it,
    rather than just the single anchor token.

    Assumes a single-line condition (multi-line conditions are handled only
    partially — the first 'if … :' found is used).

    Args:
        leaf:            A green MCTSNode leaf.
        prefix:          Full prompt text prepended before the MCTS path
                         (i.e. run.prefix).
        p_plus_anchors:  P+ anchor token list.
        p_minus_anchors: P- anchor token list.

    Returns:
        (p_plus_clause, p_minus_clause) — either may be None if the
        corresponding predicate could not be matched to any clause.
    """
    path_nodes = _get_path_nodes(leaf)
    full_text = prefix + "".join(n.token_str for n in path_nodes)

    result = _extract_boolean_expr(full_text)
    if result is None:
        return None, None
    condition, cond_start = result
    clauses = _split_clauses(condition, cond_start)
    if not clauses:
        return None, None

    def _discriminative(anchors: list[str]) -> list[str]:
        """Keep only anchors that match exactly one clause for this path."""
        return [
            anchor for anchor in anchors
            if sum(1 for ct, _, _ in clauses if _re_search_safe(anchor, ct)) == 1
        ]

    def _match(anchors: list[str]) -> "ClauseSpan | None":
        # Try each pattern; use the clause whose start comes earliest in full_text
        best: tuple[int, str, int, int] | None = None
        for anchor in anchors:
            for clause_text, abs_start, abs_end in clauses:
                if _re_search_safe(anchor, clause_text):
                    if best is None or abs_start < best[0]:
                        best = (abs_start, clause_text, abs_start, abs_end)
        if best is None:
            return None
        _, text, abs_start, abs_end = best
        return ClauseSpan(
            text=text,
            start_depth=_char_to_depth(abs_start, len(prefix), path_nodes),
            end_depth=_char_to_depth(
                max(abs_start, abs_end - 1), len(prefix), path_nodes
            ),
        )

    disc_p_plus  = _discriminative(p_plus_anchors)
    disc_p_minus = _discriminative(p_minus_anchors)
    return _match(disc_p_plus), _match(disc_p_minus)


def _highlight_spans_in_path(
    path_text: str,
    spans: list[tuple[int, int, str]],
) -> str:
    """Build a Rich-markup string from path_text with highlighted substrings.

    Args:
        path_text: Raw path text.
        spans:     List of (start, end, rich_style) in path_text char coords,
                   sorted by start.  Non-overlapping spans only.

    Returns:
        Rich-markup string suitable for Console.print().
    """
    from rich.markup import escape as _esc

    result = ""
    cursor = 0
    for start, end, style in spans:
        result += _esc(path_text[cursor:start])
        result += f"[{style}]{_esc(path_text[start:end])}[/{style}]"
        cursor = end
    result += _esc(path_text[cursor:])
    return result


def show_clause_spans(
    leaf: "MCTSNode",
    prefix: str,
    p_plus_anchors: list[str],
    p_minus_anchors: list[str],
) -> str:
    """Return a multi-line Rich-markup string showing P+ and P- clause spans for a leaf.

    The Path: line highlights the matched P+ span in bright green and the P-
    span in magenta directly within the path text.

    Example output (rendered):
        Path: ' timeout > 0 and host != "" and port > 0:\\n    return True'
                ^^^^^^^^^^^           ^^^^^^^^^
          P+: 'timeout > 0'          depth 1 → 4
          P-: 'host != ""'           depth 5 → 8
          All clauses: ['timeout > 0', 'host != ""', 'port > 0']

    Args:
        leaf:            A green MCTSNode leaf.
        prefix:          run.prefix — prepended before MCTS path when scanning.
        p_plus_anchors:  P+ anchor token list.
        p_minus_anchors: P- anchor token list.

    Returns:
        Formatted string with Rich markup (pass to Console.print()).
    """
    from rich.markup import escape as _esc

    path_nodes = _get_path_nodes(leaf)
    full_text = prefix + "".join(n.token_str for n in path_nodes)
    path_text = leaf.get_path_text()

    p_plus_clause, p_minus_clause = extract_predicate_clauses(
        leaf, prefix, p_plus_anchors, p_minus_anchors
    )

    # Mutual exclusivity check: if both resolve to the same clause the leaf is
    # not usable for contrastive pairs — treat P- as unresolved.
    conflict = (
        p_plus_clause is not None
        and p_minus_clause is not None
        and p_plus_clause.text == p_minus_clause.text
    )
    if conflict:
        p_minus_clause = None

    # Build highlighted Path line — find each clause's position within path_text.
    _clause_styles = [
        (p_plus_clause, "bold cyan"),
        (p_minus_clause, "bold magenta"),
    ]
    raw_spans: list[tuple[int, int, str]] = []
    for clause, style in _clause_styles:
        if clause is not None:
            pos = path_text.find(clause.text)
            if pos >= 0:
                raw_spans.append((pos, pos + len(clause.text), style))
    raw_spans.sort(key=lambda x: x[0])

    if raw_spans:
        highlighted = _highlight_spans_in_path(path_text, raw_spans)
    else:
        highlighted = _esc(path_text)
    lines = [f"  Path: '{highlighted}'"]

    # Show all clauses found in the condition
    cond_result = _extract_boolean_expr(full_text)
    if cond_result is not None:
        condition, cond_start = cond_result
        all_clauses = [t for t, _, _ in _split_clauses(condition, cond_start)]
        lines.append(f"  All clauses: {_esc(str(all_clauses))}")
    else:
        lines.append("  All clauses: (no if/return/while/assert found)")

    for name, clause, style in (
        ("P+", p_plus_clause, "bold cyan"),
        ("P-", p_minus_clause, "bold magenta"),
    ):
        if clause is None:
            if name == "P-" and conflict:
                lines.append(
                    f"  [{style}]{name}[/{style}]: "
                    f"[dim](conflicted — same clause as P+, excluded from contrastive pairs)[/dim]"
                )
            else:
                lines.append(f"  {name}: (not matched to any clause)")
        else:
            lines.append(
                f"  [{style}]{name}[/{style}]: {_esc(repr(clause.text)):40s}"
                f"  depth {clause.start_depth} → {clause.end_depth}"
            )

    return "\n".join(lines)


def get_labeled_leaves(
    root: "MCTSNode",
    labels: dict["MCTSNode", PredicateLabel],
) -> list[tuple["MCTSNode", str, str]]:
    """Return (leaf, path_text, label_str) for every green leaf in the tree.

    Useful for quickly iterating over what the labeler decided for each leaf.

    Returns:
        List of (leaf_node, path_text, label) sorted by label then path_text.
    """
    results = []

    def _walk(node: "MCTSNode") -> None:
        if node.is_leaf() and node.color == "green":
            lbl = labels.get(node)
            results.append(
                (node, node.get_path_text(), lbl.label if lbl else "unlabeled")
            )
            return
        for child in node.children:
            _walk(child)

    _walk(root)
    results.sort(key=lambda t: (t[2], t[1]))
    return results


def print_labeled_tree(
    root: "MCTSNode",
    labels: dict["MCTSNode", PredicateLabel],
    max_depth: int | None = None,
    show_counts: bool = True,
) -> None:
    """Print the MCTS tree with predicate-label annotations at every node.

    Color coding:
        green  (bold) — p_plus_first
        magenta       — p_minus_first
        yellow        — mixed  (contrastive branching point)
        dim white     — unlabeled

    Args:
        root:        Root MCTSNode.
        labels:      Output of label_predicate_spans().
        max_depth:   Truncate display below this depth (None = full tree).
        show_counts: If True, append (p+=N, p-=M) counts after each label tag.
    """
    lines: list[str] = []

    def _collect(node: "MCTSNode", prefix: str, is_last: bool) -> None:
        if max_depth is not None and node.depth > max_depth:
            return

        pl = labels.get(node)
        style, short = _LABEL_STYLES.get(
            pl.label if pl else "unlabeled", ("dim white", "?")
        )
        count_str = ""
        if show_counts and pl is not None and (pl.p_plus_count + pl.p_minus_count) > 0:
            count_str = f" (p+={pl.p_plus_count}, p-={pl.p_minus_count})"
        label_tag = f"[{style}][{short}]{count_str}[/{style}]"

        connector = "└── " if is_last else "├── "

        if node.is_root():
            lines.append(f"[ROOT] [dim]v={node.visits}[/dim]  {label_tag}")
        else:
            tok = node.token_str
            is_leaf_node = node.is_leaf()
            leaf_suffix = ""
            if is_leaf_node:
                path_repr = repr(node.get_path_text())[1:-1]
                term = " [TERM]" if node.is_terminal else ""
                leaf_suffix = f"[dim]{term} ('{path_repr}')[/dim]"
            lines.append(
                f"{prefix}{connector}[bold]{tok!r}[/bold] "
                f"[dim](p={node.prior:.3f}, v={node.visits})[/dim]  "
                f"{label_tag} {leaf_suffix}"
            )

        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _collect(child, child_prefix, i == len(node.children) - 1)

    _collect(root, "", True)
    Console(highlight=False, width=320).print("\n".join(lines))
