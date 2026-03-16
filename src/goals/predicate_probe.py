"""Contrastive pair sampling for the multi-predicate goal-direction setup.

Implements the dual-window cross-tree pairing strategy:

  For each green leaf path that implements both P+ and P-, split the path
  into two windows at the 'and' boundary:

    P+-first branch:  [pursuing P+] and [pursuing P-]
    P--first branch:  [pursuing P-] and [pursuing P+]

  Both windows contribute labeled nodes.  Pairs are formed across trees
  (different prefills) at the same absolute token position, so the probe
  cannot exploit prefix-specific artifacts or positional encoding differences.

Typical usage:

    from src.goals.predicate_probe import collect_labeled_nodes, sample_multipred_pairs

    labeled = collect_labeled_nodes(tp_results, target_predicate, mt)
    pairs_by_pos = sample_multipred_pairs(labeled)

    print_node_label_summary(labeled)
    print_pairs_sample(pairs_by_pos, n=5)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich import box

from src.goals.anchor import extract_predicate_clauses, _get_path_nodes
from src.goals.probe import compute_outcome_probs

if TYPE_CHECKING:
    from src.goals.sampling import MCTSNode
    from src.goals.anchor import ClauseSpan
    from src.models import ModelandTokenizer
    from src.goals.predicate_sampling import MCTSSearchResults_for_TargetPredicate
    from src.goals.predicate_data import TargetPredicate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LabeledNode:
    """A single MCTSNode annotated with its predicate label and alignment info.

    Attributes:
        node:          The MCTSNode itself.
        label:         "p_plus" — node is within the P+ clause span of its branch.
                       "p_minus" — node is within the P- clause span.
        window:        "first"  — this predicate is being pursued first in the branch.
                       "second" — this predicate comes after 'and' in the branch.
        abs_position:  Absolute token position in the full sequence.
                       = prompt_token_len + node.depth - 1.
                       Used as the alignment key for cross-tree pairing.
        depth:         MCTS depth (root = 0, first generated token = 1).
        tree_id:       run.prefix — uniquely identifies the source MCTS tree.
        sample_idx:    Index of the MultiPredicateSample this node comes from.
        prefill_type:  e.g. "aligned_p_plus", "aligned_p_minus", "inconsequential".
    """

    node: "MCTSNode"
    label: str           # "p_plus" | "p_minus"
    window: str          # "first" | "second"
    abs_position: int
    depth: int
    tree_id: str
    sample_idx: int
    prefill_type: str
    prefill: str = ""     # the actual prefill string (e.g. "n %", "not ", ...)
    p_green: float = 1.0  # fraction of probability mass from this node flowing to green leaves

    def path_text(self) -> str:
        return self.node.get_path_text()

    def __repr__(self) -> str:
        return (
            f"LabeledNode({self.label!r}, win={self.window!r}, "
            f"p_green={self.p_green:.2f}, abs_pos={self.abs_position}, depth={self.depth}, "
            f"tree={self.tree_id[-20:]!r})"
        )


@dataclass
class MultiPredContrastivePair:
    """A contrastive pair for the multi-predicate setup.

    Both nodes are from green branches but from different MCTS trees (by
    default), ensuring the contrast reflects predicate intent, not the
    specific prefill context.

    Attributes:
        positive:     LabeledNode pursuing P+.
        negative:     LabeledNode pursuing P-.
        abs_position: Absolute token position used for alignment (= positive's abs_pos).
    """

    positive: LabeledNode
    negative: LabeledNode
    abs_position: int

    def __repr__(self) -> str:
        return (
            f"MultiPredContrastivePair("
            f"pos=[{self.positive.label}/{self.positive.window}] "
            f"'{self.positive.path_text()[:25]}...', "
            f"neg=[{self.negative.label}/{self.negative.window}] "
            f"'{self.negative.path_text()[:25]}...', "
            f"abs_pos={self.abs_position})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _label_path_nodes(
    leaf: "MCTSNode",
    p_plus_clause: "ClauseSpan | None",
    p_minus_clause: "ClauseSpan | None",
) -> list[tuple["MCTSNode", str, str]]:
    """Label each non-root node in the leaf path.

    Returns list of (node, predicate_label, window) in root→leaf order.

    predicate_label is one of:
        "p_plus"      — node depth falls within the P+ clause span.
        "p_minus"     — node depth falls within the P- clause span.
        "transition"  — node is outside both spans (e.g. the 'and' token,
                        'return True', closing characters).  Excluded from pairing.

    window is "first" or "second" — which clause is being pursued first in
    this branch.  Nodes in the first window are "transition" between pivot
    and 'and'; nodes in the second window come after 'and'.
    """
    path_nodes = _get_path_nodes(leaf)  # depth 1..leaf, root excluded

    def _start(clause: "ClauseSpan | None") -> int:
        """Effective start depth; -1 (in prefix) treated as 1."""
        if clause is None:
            return int(1e9)   # never matches
        return 1 if clause.start_depth <= 0 else clause.start_depth

    def _end(clause: "ClauseSpan | None") -> int:
        """Effective end depth; -1 treated as matching nothing."""
        if clause is None:
            return -1
        if clause.end_depth == -1:
            return max(1, clause.start_depth)
        return clause.end_depth

    p_plus_start  = _start(p_plus_clause)
    p_plus_end    = _end(p_plus_clause)
    p_minus_start = _start(p_minus_clause)
    p_minus_end   = _end(p_minus_clause)

    # Which predicate is first in this branch?
    if p_plus_clause is not None and p_minus_clause is not None:
        first_label = "p_plus" if p_plus_start <= p_minus_start else "p_minus"
    elif p_plus_clause is not None:
        first_label = "p_plus"
    elif p_minus_clause is not None:
        first_label = "p_minus"
    else:
        first_label = "p_plus"  # fallback, won't matter (no clauses → all transition)

    result: list[tuple["MCTSNode", str, str]] = []
    for node in path_nodes:
        if node is leaf:          # terminal node: goal already done, exclude from probing
            continue
        d = node.depth
        in_p_plus  = p_plus_clause  is not None and p_plus_start  <= d <= p_plus_end
        in_p_minus = p_minus_clause is not None and p_minus_start <= d <= p_minus_end

        if in_p_plus and not in_p_minus:
            pred_label = "p_plus"
        elif in_p_minus and not in_p_plus:
            pred_label = "p_minus"
        else:
            pred_label = "transition"

        window = (
            "first"  if pred_label == first_label else
            "second" if pred_label != "transition" else
            "transition"
        )
        result.append((node, pred_label, window))

    return result


# ---------------------------------------------------------------------------
# Public — node collection
# ---------------------------------------------------------------------------


def collect_labeled_nodes(
    tp_results: "MCTSSearchResults_for_TargetPredicate",
    tp: "TargetPredicate",
    mt: "ModelandTokenizer",
    sample_indices: list[int] | None = None,
    min_p_green: float = 0.5,
) -> list[LabeledNode]:
    """Walk all green leaf paths across all trees and samples; label nodes by predicate.

    For each green leaf:
      1. Extract P+ and P- clause spans with extract_predicate_clauses().
      2. Walk the path and label each node as "p_plus", "p_minus", or "transition".
      3. Filter: keep only nodes where compute_outcome_probs(node).green >= min_p_green.
         This ensures we only include nodes from which the model robustly generates
         correct code — not just nodes that happen to be on one lucky green path.
      4. Compute abs_position = len(tokenize(run.prompt)) + node.depth - 1.
      5. Emit a LabeledNode for every surviving non-transition node.

    Args:
        tp_results:     MCTSSearchResults_for_TargetPredicate (all samples).
        tp:             TargetPredicate — provides anchor tokens per sample.
        mt:             ModelandTokenizer — used to tokenize run.prompt for abs_position.
        sample_indices: Only process these sample indices.  None = all samples.
        min_p_green:    Minimum fraction of probability mass flowing to green leaves
                        required to include a node.  Default 0.5.  Set to 0.0 to
                        disable the filter (include all nodes on green leaf paths).

    Returns:
        Flat list of LabeledNode (transition nodes and low-p_green nodes excluded).
    """
    indices = sample_indices if sample_indices is not None else range(len(tp_results.sample_results))
    collected: list[LabeledNode] = []

    for sample_idx in indices:
        sr = tp_results.sample_results[sample_idx]
        sample = tp.samples[sample_idx]

        p_plus_anchors  = tp.p_plus.anchor_patterns
        p_minus_anchors = sample.p_minus.anchor_patterns if sample.p_minus else []
        if not p_plus_anchors or not p_minus_anchors:
            logger.warning(f"Sample {sample_idx}: missing anchor tokens — skipping.")
            continue

        for run in sr.runs:
            root = sr._trees.get(run.prefix)
            if root is None:
                logger.debug(f"Sample {sample_idx}, prefix {run.prefix[:30]!r}: no tree in memory.")
                continue

            # Tokenize once per run to get absolute position offset
            prompt_token_len = len(mt.tokenizer.encode(run.prompt))

            n_leaves = 0
            n_labeled = 0
            n_filtered = 0

            def _walk(node: "MCTSNode") -> None:
                nonlocal n_leaves, n_labeled
                if node.is_leaf():
                    if node.color == "green":
                        n_leaves += 1
                        _process_leaf(node)
                    return
                for child in node.children:
                    _walk(child)

            def _process_leaf(leaf: "MCTSNode") -> None:
                nonlocal n_labeled, n_filtered
                p_plus_clause, p_minus_clause = extract_predicate_clauses(
                    leaf, run.prefix, p_plus_anchors, p_minus_anchors
                )
                # Need at least one clause to label anything
                if p_plus_clause is None and p_minus_clause is None:
                    return

                labeled_path = _label_path_nodes(leaf, p_plus_clause, p_minus_clause)
                for node, pred_label, window in labeled_path:
                    # "transition" covers pre-clause pivot tokens (e.g. "return") and
                    # inter-clause connectors (e.g. "and") — neither reflects a single
                    # predicate goal, so we skip them.
                    if pred_label == "transition":
                        continue
                    p_green = compute_outcome_probs(node).green
                    if p_green < min_p_green:
                        n_filtered += 1
                        continue
                    abs_pos = prompt_token_len + node.depth - 1
                    collected.append(LabeledNode(
                        node=node,
                        label=pred_label,
                        window=window,
                        abs_position=abs_pos,
                        depth=node.depth,
                        tree_id=run.prefix,
                        sample_idx=sample_idx,
                        prefill_type=run.prefill_type,
                        prefill=run.prefill,
                        p_green=p_green,
                    ))
                    n_labeled += 1

            _walk(root)
            logger.debug(
                f"Sample {sample_idx}, prefill {run.prefill!r}: "
                f"{n_leaves} green leaves → {n_labeled} labeled nodes "
                f"({n_filtered} dropped by p_green < {min_p_green})."
            )

    logger.info(
        f"collect_labeled_nodes: {len(collected)} total labeled nodes "
        f"across {len(set(n.tree_id for n in collected))} trees."
    )
    return collected


# ---------------------------------------------------------------------------
# Public — pairing
# ---------------------------------------------------------------------------


def sample_multipred_pairs(
    labeled_nodes: list[LabeledNode],
    allow_same_tree: bool = False,
    abs_pos_tolerance: int = 0,
    seed: int = 42,
) -> dict[int, list[MultiPredContrastivePair]]:
    """Form contrastive pairs from labeled nodes, grouped by absolute token position.

    Pairs P+-labeled nodes against P--labeled nodes.  By default only pairs
    nodes from DIFFERENT source trees to prevent the probe from learning
    tree-level (prefill-specific) artifacts.

    With abs_pos_tolerance > 0, nodes whose abs_positions differ by at most
    that many tokens can be paired (useful when prefill lengths vary by 1–2
    tokens).  The pair's abs_position is set to the positive node's value.

    Args:
        labeled_nodes:       Output of collect_labeled_nodes().
        allow_same_tree:     If False (default), only pair nodes from different trees.
        abs_pos_tolerance:   Max abs_position difference for a valid pair (0 = exact).
        seed:                Random seed for reproducible 1:1 matching.

    Returns:
        dict[abs_position → list[MultiPredContrastivePair]]
        Keyed by the positive node's abs_position.
    """
    rng = random.Random(seed)

    # Group by abs_position
    by_pos: dict[int, dict[str, list[LabeledNode]]] = defaultdict(lambda: defaultdict(list))
    for ln in labeled_nodes:
        by_pos[ln.abs_position][ln.label].append(ln)

    pairs_by_pos: dict[int, list[MultiPredContrastivePair]] = {}

    sorted_positions = sorted(by_pos.keys())
    for abs_pos in sorted_positions:
        group = by_pos[abs_pos]
        p_plus_nodes  = group.get("p_plus",  [])
        p_minus_nodes = group.get("p_minus", [])

        if not p_plus_nodes or not p_minus_nodes:
            continue

        # Expand candidate pool with tolerance: include nodes from nearby positions
        if abs_pos_tolerance > 0:
            extra_minus: list[LabeledNode] = []
            for nearby in sorted_positions:
                if nearby == abs_pos:
                    continue
                if abs(nearby - abs_pos) <= abs_pos_tolerance:
                    extra_minus.extend(by_pos[nearby].get("p_minus", []))
            p_minus_nodes = p_minus_nodes + extra_minus

        pairs: list[MultiPredContrastivePair] = []
        p_minus_pool = p_minus_nodes[:]
        rng.shuffle(p_minus_pool)

        for pos_node in p_plus_nodes:
            candidates = [
                n for n in p_minus_pool
                if allow_same_tree or n.tree_id != pos_node.tree_id
            ]
            if not candidates:
                continue
            neg_node = rng.choice(candidates)
            pairs.append(MultiPredContrastivePair(
                positive=pos_node,
                negative=neg_node,
                abs_position=abs_pos,
            ))

        if pairs:
            pairs_by_pos[abs_pos] = pairs

    total = sum(len(v) for v in pairs_by_pos.values())
    logger.info(
        f"sample_multipred_pairs: {total} pairs across {len(pairs_by_pos)} positions."
    )
    return pairs_by_pos


# ---------------------------------------------------------------------------
# Inspection utilities
# ---------------------------------------------------------------------------

_LABEL_STYLE = {
    "p_plus":  "bold bright_green",
    "p_minus": "bold magenta",
}
_WINDOW_STYLE = {
    "first":  "cyan",
    "second": "yellow",
}


def print_node_label_summary(
    labeled_nodes: list[LabeledNode],
    by_window: bool = True,
) -> None:
    """Print a summary table of labeled node counts broken down by label and window.

    Args:
        labeled_nodes: Output of collect_labeled_nodes().
        by_window:     If True, break down by window ("first"/"second") as well.
    """
    console = Console(highlight=False, width=120)

    # Counts
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for ln in labeled_nodes:
        counts[(ln.label, ln.window, ln.prefill_type)] += 1

    table = Table(box=box.SIMPLE_HEAVY, title="Labeled Node Summary", show_lines=False)
    table.add_column("label",        width=10)
    table.add_column("window",       width=8)
    table.add_column("prefill_type", width=20)
    table.add_column("count", justify="right", width=8)

    for (label, window, ptype), count in sorted(counts.items()):
        ls = _LABEL_STYLE.get(label, "")
        ws = _WINDOW_STYLE.get(window, "")
        table.add_row(
            f"[{ls}]{label}[/{ls}]",
            f"[{ws}]{window}[/{ws}]",
            ptype,
            str(count),
        )

    total_p_plus  = sum(n.label == "p_plus"  for n in labeled_nodes)
    total_p_minus = sum(n.label == "p_minus" for n in labeled_nodes)
    n_trees = len(set(n.tree_id for n in labeled_nodes))
    n_pos   = len(set(n.abs_position for n in labeled_nodes))
    console.print(table)
    console.print(
        f"Total: {len(labeled_nodes)} nodes  |  "
        f"[bold bright_green]p_plus={total_p_plus}[/bold bright_green]  "
        f"[bold magenta]p_minus={total_p_minus}[/bold magenta]  |  "
        f"{n_trees} trees  |  {n_pos} distinct abs_positions"
    )


def print_pairs_sample(
    pairs_by_pos: dict[int, list[MultiPredContrastivePair]],
    n: int = 5,
) -> None:
    """Print a sample of contrastive pairs showing path text and metadata.

    Args:
        pairs_by_pos: Output of sample_multipred_pairs().
        n:            Max number of pairs to show.
    """
    console = Console(highlight=False, width=200)

    total = sum(len(v) for v in pairs_by_pos.values())
    n_pos = len(pairs_by_pos)
    console.print(f"\n[bold]Contrastive pairs:[/bold] {total} total across {n_pos} abs_positions\n")

    shown = 0
    for abs_pos in sorted(pairs_by_pos.keys()):
        for pair in pairs_by_pos[abs_pos]:
            pos = pair.positive
            neg = pair.negative
            console.print(
                f"  abs_pos=[bold]{abs_pos}[/bold]  "
                f"[dim](pos depth={pos.depth}, {pos.prefill_type!r}  |  "
                f"neg depth={neg.depth}, {neg.prefill_type!r})[/dim]"
            )
            console.print(
                f"    [bold bright_green]+[/bold bright_green] [{_WINDOW_STYLE[pos.window]}]{pos.window}[/{_WINDOW_STYLE[pos.window]}] "
                f"{pos.path_text()!r}"
            )
            console.print(
                f"    [bold magenta]-[/bold magenta] [{_WINDOW_STYLE[neg.window]}]{neg.window}[/{_WINDOW_STYLE[neg.window]}] "
                f"{neg.path_text()!r}"
            )
            console.print()
            shown += 1
            if shown >= n:
                remaining = total - shown
                if remaining > 0:
                    console.print(f"  [dim]... {remaining} more pairs not shown[/dim]")
                return


def pairs_position_histogram(
    pairs_by_pos: dict[int, list[MultiPredContrastivePair]],
    width: int = 60,
) -> None:
    """Print a simple ASCII histogram of pair counts by absolute position.

    Args:
        pairs_by_pos: Output of sample_multipred_pairs().
        width:        Max bar width in characters.
    """
    console = Console(highlight=False, width=width + 40)
    if not pairs_by_pos:
        console.print("[dim](no pairs)[/dim]")
        return

    positions = sorted(pairs_by_pos.keys())
    counts = [len(pairs_by_pos[p]) for p in positions]
    max_count = max(counts)

    console.print("[bold]Pair count by absolute token position:[/bold]")
    for pos, count in zip(positions, counts):
        bar_len = max(1, int(count / max_count * width))
        bar = "█" * bar_len
        console.print(f"  {pos:5d}  [bright_green]{bar}[/bright_green] {count}")
