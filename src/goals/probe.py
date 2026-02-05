"""
Concept probing utilities for extracting goal representations from MCTS trees.

This module implements the contrastive sampling approach inspired by Schut et al.'s
dynamic concept extraction (Section 4.1.2). The key idea is to contrast "optimal"
paths (leading to correct code) with "suboptimal" paths (leading to incorrect code)
at various depths in the MCTS tree.

Usage:
    from src.goals.probe import compute_outcome_probs, sample_contrastive_pairs

    # After running MCTS and evaluate_mcts_tree()
    pairs = sample_contrastive_pairs(root, depth=5)
    for pos_node, neg_node in pairs:
        # Extract hidden states and use for contrastive learning
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.goals.sampling import MCTSNode


@dataclass
class OutcomeProbs:
    """Probability mass flowing toward each outcome type from a node."""

    green: float  # Leads to PASSED (all unit tests pass)
    red: float  # Leads to FAILED or ERROR (complete but incorrect)
    uncertain: float  # Leads to SYNTAX_ERROR or incomplete (ambiguous)

    @property
    def is_confident_green(self) -> bool:
        """Node is confidently on a successful path."""
        return self.green > 0.5 and self.red < 0.2

    @property
    def is_confident_red(self) -> bool:
        """Node is confidently on a failing path."""
        return self.red > 0.5 and self.green < 0.1

    def __repr__(self) -> str:
        return f"OutcomeProbs(green={self.green:.3f}, red={self.red:.3f}, uncertain={self.uncertain:.3f})"


@dataclass
class ContrastivePair:
    """A pair of nodes for contrastive learning."""

    positive: "MCTSNode"  # Node on path leading to success
    negative: "MCTSNode"  # Node on path leading to failure
    depth: int  # Depth at which these nodes were sampled
    pos_probs: OutcomeProbs  # Outcome probabilities for positive node
    neg_probs: OutcomeProbs  # Outcome probabilities for negative node

    def __repr__(self) -> str:
        pos_text = self.positive.get_path_text()[:30]
        neg_text = self.negative.get_path_text()[:30]
        return (
            f"ContrastivePair(depth={self.depth}, "
            f"pos='{pos_text}...' {self.pos_probs}, "
            f"neg='{neg_text}...' {self.neg_probs})"
        )


def compute_outcome_probs(node: "MCTSNode") -> OutcomeProbs:
    """
    Compute probability mass flowing toward each outcome type.

    For a node, this computes what fraction of the probability mass
    (weighted by child priors) leads to green/red/uncertain outcomes.

    For leaf nodes:
        - green if color == "green"
        - red if color in ("red", "orange")  # orange = valid but failed tests
        - uncertain if color is None or SYNTAX_ERROR

    For internal nodes:
        - Recursively compute children's outcome probs
        - Weight by each child's prior probability
        - Sum to get this node's outcome distribution

    Args:
        node: MCTSNode with colors already assigned (after evaluate_mcts_tree)

    Returns:
        OutcomeProbs with probability mass for each outcome type
    """
    # Leaf node: return based on color
    if node.is_leaf():
        if node.color == "green":
            return OutcomeProbs(green=1.0, red=0.0, uncertain=0.0)
        elif node.color in ("red", "orange"):
            # Both red and orange are "complete but incorrect" for contrastive purposes
            return OutcomeProbs(green=0.0, red=1.0, uncertain=0.0)
        else:
            # None or SYNTAX_ERROR - uncertain/incomplete
            return OutcomeProbs(green=0.0, red=0.0, uncertain=1.0)

    # Internal node: aggregate children weighted by prior
    total_green = 0.0
    total_red = 0.0
    total_uncertain = 0.0
    total_prior = 0.0

    for child in node.children:
        child_probs = compute_outcome_probs(child)
        prior = child.prior

        total_green += prior * child_probs.green
        total_red += prior * child_probs.red
        total_uncertain += prior * child_probs.uncertain
        total_prior += prior

    # Normalize by total prior (should be ~1.0 but may not be exact)
    if total_prior > 0:
        return OutcomeProbs(
            green=total_green / total_prior,
            red=total_red / total_prior,
            uncertain=total_uncertain / total_prior,
        )
    else:
        # No children with priors (shouldn't happen for non-leaf)
        return OutcomeProbs(green=0.0, red=0.0, uncertain=1.0)


def collect_nodes_at_depth(root: "MCTSNode", target_depth: int) -> list["MCTSNode"]:
    """
    Collect all nodes at a specific depth in the tree.

    Args:
        root: Root node of the MCTS tree (depth=0)
        target_depth: The depth to collect nodes from

    Returns:
        List of MCTSNode at the target depth
    """
    nodes = []
    _collect_at_depth_recursive(root, target_depth, nodes)
    return nodes


def _collect_at_depth_recursive(
    node: "MCTSNode", target_depth: int, result: list["MCTSNode"]
) -> None:
    """Recursive helper for collect_nodes_at_depth."""
    if node.depth == target_depth:
        result.append(node)
        return

    # Don't go deeper than target
    if node.depth > target_depth:
        return

    for child in node.children:
        _collect_at_depth_recursive(child, target_depth, result)


def sample_contrastive_pairs(
    root: "MCTSNode",
    depth: int,
    min_green_confidence: float = 0.5,
    max_green_in_negative: float = 0.1,
    min_red_confidence: float = 0.5,
    max_red_in_positive: float = 0.2,
    max_uncertain: float = 0.5,
) -> list[ContrastivePair]:
    """
    Sample contrastive pairs (positive, negative) at a given depth.

    A positive node is one where the model's probability mass mostly leads to
    successful completions (green). A negative node is one where the probability
    mass mostly leads to failures (red).

    This filtering ensures we're capturing what the model "knows" - branches where
    the model's own preferences align with eventual correctness/incorrectness.

    Only samples from internal nodes (not leaves or terminals). At terminal/leaf
    nodes the goal pursuit is already resolved, so the model's next-token objective
    is likely doing something other than pursuing the goal.

    Args:
        root: Root node of evaluated MCTS tree (after evaluate_mcts_tree)
        depth: Tree depth to sample from (root=0)
        min_green_confidence: Minimum p_green for positive nodes (default: 0.5)
        max_green_in_negative: Maximum p_green allowed in negative nodes (default: 0.1)
        min_red_confidence: Minimum p_red for negative nodes (default: 0.5)
        max_red_in_positive: Maximum p_red allowed in positive nodes (default: 0.2)
        max_uncertain: Maximum p_uncertain allowed in either (default: 0.5)

    Returns:
        List of ContrastivePair objects
    """
    nodes_at_depth = collect_nodes_at_depth(root, depth)

    positive_nodes: list[tuple["MCTSNode", OutcomeProbs]] = []
    negative_nodes: list[tuple["MCTSNode", OutcomeProbs]] = []

    for node in nodes_at_depth:
        # Skip terminal and leaf nodes - goal pursuit is already resolved there
        if node.is_terminal or node.is_leaf():
            continue

        probs = compute_outcome_probs(node)

        # Skip nodes with too much uncertainty (SYNTAX_ERROR / incomplete)
        if probs.uncertain > max_uncertain:
            continue

        # Positive: high green, low red
        if probs.green >= min_green_confidence and probs.red <= max_red_in_positive:
            positive_nodes.append((node, probs))

        # Negative: high red, low green
        if probs.red >= min_red_confidence and probs.green <= max_green_in_negative:
            negative_nodes.append((node, probs))

    # Create all pairs
    pairs = []
    for pos_node, pos_probs in positive_nodes:
        for neg_node, neg_probs in negative_nodes:
            pairs.append(
                ContrastivePair(
                    positive=pos_node,
                    negative=neg_node,
                    depth=depth,
                    pos_probs=pos_probs,
                    neg_probs=neg_probs,
                )
            )

    return pairs


def sample_multidepth_pairs(
    root: "MCTSNode",
    min_depth: int = 1,
    max_depth: int | None = None,
    **kwargs,
) -> dict[int, list[ContrastivePair]]:
    """
    Sample contrastive pairs at multiple depths.

    This implements the multi-depth contrast from Schut et al. Figure 3,
    where we find suboptimal rollouts at different depths to generate
    constraints across the trajectory.

    Args:
        root: Root node of evaluated MCTS tree
        min_depth: Minimum depth to sample from (default: 1)
        max_depth: Maximum depth to sample from (default: tree's max depth)
        **kwargs: Additional arguments passed to sample_contrastive_pairs

    Returns:
        Dict mapping depth -> list of ContrastivePair at that depth
    """
    # Find max depth if not specified
    if max_depth is None:
        max_depth = _find_max_depth(root)

    all_pairs = {}
    for d in range(min_depth, max_depth + 1):
        pairs = sample_contrastive_pairs(root, d, **kwargs)
        if pairs:
            all_pairs[d] = pairs

    return all_pairs


def _find_max_depth(node: "MCTSNode") -> int:
    """Find the maximum depth in the tree."""
    if node.is_leaf():
        return node.depth

    max_child_depth = node.depth
    for child in node.children:
        child_max = _find_max_depth(child)
        if child_max > max_child_depth:
            max_child_depth = child_max

    return max_child_depth


def summarize_contrastive_pairs(
    pairs_by_depth: dict[int, list[ContrastivePair]],
) -> dict:
    """
    Summarize contrastive pairs across depths.

    Args:
        pairs_by_depth: Output from sample_multidepth_pairs

    Returns:
        Summary statistics dict
    """
    total_pairs = sum(len(pairs) for pairs in pairs_by_depth.values())
    depths_with_pairs = len(pairs_by_depth)

    # Collect unique positive and negative nodes
    all_pos_nodes = set()
    all_neg_nodes = set()
    for pairs in pairs_by_depth.values():
        for pair in pairs:
            all_pos_nodes.add(id(pair.positive))
            all_neg_nodes.add(id(pair.negative))

    return {
        "total_pairs": total_pairs,
        "depths_with_pairs": depths_with_pairs,
        "depth_range": (min(pairs_by_depth.keys()), max(pairs_by_depth.keys()))
        if pairs_by_depth
        else (0, 0),
        "unique_positive_nodes": len(all_pos_nodes),
        "unique_negative_nodes": len(all_neg_nodes),
        "pairs_per_depth": {d: len(pairs) for d, pairs in pairs_by_depth.items()},
    }


def print_contrastive_pairs(
    pairs_by_depth: dict[int, list[ContrastivePair]],
    max_pairs_per_depth: int = 3,
    max_text_length: int = 50,
) -> None:
    """
    Pretty print contrastive pairs for inspection.

    Args:
        pairs_by_depth: Output from sample_multidepth_pairs
        max_pairs_per_depth: Maximum pairs to show per depth
        max_text_length: Maximum path text length to display
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    summary = summarize_contrastive_pairs(pairs_by_depth)
    console.print("\n[bold]Contrastive Pairs Summary[/bold]")
    console.print(f"  Total pairs: {summary['total_pairs']}")
    console.print(f"  Depths with pairs: {summary['depths_with_pairs']}")
    console.print(f"  Depth range: {summary['depth_range']}")
    console.print(f"  Unique positive nodes: {summary['unique_positive_nodes']}")
    console.print(f"  Unique negative nodes: {summary['unique_negative_nodes']}")

    for depth in sorted(pairs_by_depth.keys()):
        pairs = pairs_by_depth[depth]
        console.print(f"\n[bold cyan]Depth {depth}[/bold cyan] ({len(pairs)} pairs)")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Positive Path", style="green")
        table.add_column("P(green)", justify="right")
        table.add_column("Negative Path", style="red")
        table.add_column("P(red)", justify="right")

        for pair in pairs[:max_pairs_per_depth]:
            pos_text = pair.positive.get_path_text()
            neg_text = pair.negative.get_path_text()

            # Truncate
            if len(pos_text) > max_text_length:
                pos_text = pos_text[:max_text_length] + "..."
            if len(neg_text) > max_text_length:
                neg_text = neg_text[:max_text_length] + "..."

            # Escape rich markup
            pos_text = pos_text.replace("[", "\\[")
            neg_text = neg_text.replace("[", "\\[")

            table.add_row(
                repr(pos_text),
                f"{pair.pos_probs.green:.2f}",
                repr(neg_text),
                f"{pair.neg_probs.red:.2f}",
            )

        console.print(table)

        if len(pairs) > max_pairs_per_depth:
            console.print(f"  ... and {len(pairs) - max_pairs_per_depth} more pairs")
