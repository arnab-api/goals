"""
Monte Carlo Tree Search for LM code completion exploration.

This module implements a bounded MCTS algorithm that explores different
token continuations from an LM, building a tree of possible completions.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from dataclasses_json import DataClassJsonMixin
from rich.console import Console

from src.models import ModelandTokenizer
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)

# Color mapping for evaluation results (bright/bold for evaluated nodes)
COLOR_STYLES = {
    "green": "bold bright_green",
    "orange": "bold bright_yellow",  # bright_yellow is more visible than orange1
    "red": "bold bright_red",
    None: "white",  # unevaluated nodes: plain white (not bold)
}


def _get_console() -> Console:
    """Create a fresh Console instance for colored output.

    Created on demand (not at module level) to avoid interference from
    notebooks that override the builtin print with rich.Console.print.
    """
    return Console(highlight=False, width=310)


@dataclass
class MCTSNode:
    """
    A single node in the MCTS tree, representing one token.

    Attributes:
        token_id: The token ID from the vocabulary (None for root)
        token_str: Decoded string representation of the token
        prior: LM probability of this token given parent context
        parent: Parent node (None for root)
        children: List of child nodes
        visits: Number of times this node was visited during search
        is_terminal: Whether this node ends a completion (</code>, EOS, newline, or max depth)
        terminal_reason: Why the node is terminal (if applicable)
        depth: Depth in the tree (root = 0)
        color: Evaluation color (green=passed, orange=valid but failed, red=error, None=unevaluated)
        eval_result: Evaluation result for leaf nodes (SingleCodeEval or None)
    """

    token_id: int | None = None
    token_str: str = ""
    prior: float = 0.0
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    is_terminal: bool = False
    terminal_reason: str | None = None
    depth: int = 0
    color: str | None = None
    eval_result: Any = None

    def is_leaf(self) -> bool:
        """A leaf node has no children (not yet expanded)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Root node has no parent."""
        return self.parent is None

    def get_path_tokens(self) -> list[int]:
        """Get all token IDs from root to this node."""
        tokens = []
        node = self
        while node is not None:
            if node.token_id is not None:
                tokens.append(node.token_id)
            node = node.parent
        return list(reversed(tokens))

    def get_path_text(self) -> str:
        """Get the full text from root to this node."""
        texts = []
        node = self
        while node is not None:
            if node.token_str:
                texts.append(node.token_str)
            node = node.parent
        return "".join(reversed(texts))

    def __repr__(self) -> str:
        return (
            f"MCTSNode(token={self.token_str!r}, prior={self.prior:.4f}, "
            f"visits={self.visits}, depth={self.depth}, "
            f"children={len(self.children)}, terminal={self.is_terminal})"
        )


@dataclass
class MCTSResult(DataClassJsonMixin):
    """
    Results from a single MCTS search + evaluation + contrastive sampling.
    Serializable via DataClassJsonMixin (.to_dict() / .from_dict()).

    The MCTS tree itself is not serialized (MCTSNode objects are not JSON-friendly).
    Instead, we store the contrastive pairs as text paths + outcome probs,
    and optionally save the tree as HTML via search.save_tree().
    """

    # Identification
    model_name: str = ""
    sample_idx: int = 0
    prefill: str = ""
    prefill_type: str = ""
    function_name: str = "func"

    # Prompt info
    prompt: str = ""  # full prompt sent to LM
    prefix: str = ""  # sample.code + prefill (for eval)

    # MCTS config & stats
    mcts_config: dict[str, Any] = field(default_factory=dict)
    mcts_stats: dict[str, Any] = field(default_factory=dict)

    # Evaluation counts {"green": N, "orange": N, "red": N, "total": N}
    eval_counts: dict[str, int] = field(default_factory=dict)

    # Contrastive pairs: depth (str key) -> list of {pos_path, neg_path, pos_probs, neg_probs}
    contrastive_pairs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Sampling thresholds used
    sampling_config: dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MCTSResult":
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    @property
    def total_pairs(self) -> int:
        return sum(len(p) for p in self.contrastive_pairs.values())

    @property
    def depths_with_pairs(self) -> list[int]:
        return sorted(
            int(d) for d in self.contrastive_pairs if self.contrastive_pairs[d]
        )


class MCTSSearch:
    """
    Monte Carlo Tree Search for exploring LM continuations.

    The search builds a tree where each node is a single token. It uses
    PUCT (Predictor + UCB for Trees) for selection to balance exploration
    and exploitation based on LM priors and visit counts.

    Args:
        mt: ModelandTokenizer instance for generation
        prompt: Initial prompt text or TokenizerOutput
        max_depth: Maximum number of tokens per path (default: 15)
        top_k: Maximum children per node expansion (default: 5)
        prob_threshold: Minimum probability to consider a token (default: 0.03)
        stop_tokens: Additional stop strings beyond </code> (default: None)
        c_puct: Exploration constant for PUCT formula (default: 1.0)
        temperature: Temperature for logit scaling before softmax (default: 1.0).
            Values > 1.0 flatten the distribution, increasing branching diversity.
        min_children_up_to_depth: Force at least `min_children` children for nodes
            at depth <= this value, ignoring prob_threshold (default: 0, disabled).
            With depth_first=True, all forced children are fully expanded.
        min_children: Minimum children to force at shallow depths (default: 2).
            Only active when min_children_up_to_depth > 0.
    """

    def __init__(
        self,
        mt: ModelandTokenizer,
        prompt: str | TokenizerOutput,
        max_depth: int = 15,
        top_k: int = 5,
        prob_threshold: float = 0.03,
        stop_tokens: list[str] | None = None,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        min_children_up_to_depth: int = 0,
        min_children: int = 2,
    ):
        self.mt = mt
        self.max_depth = max_depth
        self.top_k = top_k
        self.prob_threshold = prob_threshold
        self.c_puct = c_puct
        self.temperature = temperature
        self.min_children_up_to_depth = min_children_up_to_depth
        self.min_children = min_children

        # Process prompt
        if isinstance(prompt, str):
            self.prompt_text = prompt
            self.prompt_tokens = mt.tokenizer.encode(prompt, return_tensors="pt")
        else:
            self.prompt_text = mt.tokenizer.decode(prompt.input_ids[0])
            self.prompt_tokens = prompt.input_ids

        # Move prompt to model device
        self.prompt_tokens = self.prompt_tokens.to(mt.device)

        # Setup stop conditions
        self.stop_token_ids: set[int] = set()
        self.stop_strings: list[str] = ["</code>", "\n", "\r\n", "\r"]
        if stop_tokens:
            self.stop_strings.extend(stop_tokens)

        # Add EOS token
        if mt.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(mt.tokenizer.eos_token_id)

        # Try to encode stop strings to token IDs (for single-token stops)
        # Only add if the token decodes back to exactly the stop string
        for stop_str in self.stop_strings:
            encoded = mt.tokenizer.encode(stop_str, add_special_tokens=False)
            if len(encoded) == 1:
                # Verify round-trip to avoid false positives
                decoded = mt.tokenizer.decode([encoded[0]])
                if stop_str in decoded:
                    self.stop_token_ids.add(encoded[0])

        # Initialize root node
        self.root = MCTSNode(depth=0)

        # Track statistics
        self.total_iterations = 0
        self.total_nodes = 1  # root

    def run(self, n_iterations: int = 100, depth_first: bool = False) -> MCTSNode:
        """
        Run MCTS for a specified number of iterations.

        Each iteration:
        1. SELECT: Traverse from root to a leaf using PUCT
        2. EXPAND: Generate children for the leaf (if not terminal)
        3. BACKPROPAGATE: Update visit counts back to root

        Args:
            n_iterations: Number of select-expand-backprop cycles
            depth_first: If True, after expanding a node, immediately select and
                        expand its highest-prior child until reaching terminal/max_depth.
                        This ensures branches are fully explored before backtracking.

        Returns:
            The root node of the expanded tree
        """
        for i in range(n_iterations):
            # SELECT: Find a leaf node using PUCT
            leaf = self._select(self.root)

            # EXPAND: Generate children if not terminal
            if not leaf.is_terminal:
                children = self._expand(leaf)

                # Depth-first: keep expanding down the highest-prior path
                if depth_first and children:
                    # At shallow depths with forced children, depth-first expand
                    # ALL forced children so each gets a complete code path.
                    if leaf.depth < self.min_children_up_to_depth:
                        children_to_expand = children[: self.min_children]
                    else:
                        children_to_expand = children[:1]

                    for child in children_to_expand:
                        current = child
                        while not current.is_terminal and current.depth < self.max_depth:
                            new_children = self._expand(current)
                            if not new_children:
                                break
                            # Backprop from current before going deeper
                            self._backpropagate(current)
                            current = new_children[0]
                        # Final backprop from the deepest node
                        self._backpropagate(current)
                else:
                    # BACKPROPAGATE: Update visits from leaf to root
                    self._backpropagate(leaf)
            else:
                self._backpropagate(leaf)

            self.total_iterations += 1

            if (i + 1) % 50 == 0:
                logger.debug(
                    f"MCTS iteration {i + 1}/{n_iterations}, "
                    f"total nodes: {self.total_nodes}"
                )

        return self.root

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node by traversing the tree using PUCT scores.

        PUCT formula: score = prior * c_puct * sqrt(parent_visits) / (1 + visits)

        Higher prior (LM confidence) → higher score
        Lower visits (less explored) → higher score
        """
        while not node.is_leaf() and not node.is_terminal:
            node = self._select_child(node)
        return node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest PUCT score."""
        best_score = -float("inf")
        best_child = None

        parent_visits = node.visits
        sqrt_parent = math.sqrt(parent_visits) if parent_visits > 0 else 1.0

        for child in node.children:
            # PUCT score: exploration bonus based on prior and visits
            exploration = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            score = exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else node.children[0]

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Expand a leaf node by generating top-k children from the LM.

        Args:
            node: The leaf node to expand

        Returns:
            List of newly created child nodes
        """
        # Check depth limit
        if node.depth >= self.max_depth:
            node.is_terminal = True
            node.terminal_reason = "max_depth"
            return []

        # Build context: prompt + path tokens
        path_tokens = node.get_path_tokens()
        if path_tokens:
            context = torch.cat(
                [
                    self.prompt_tokens,
                    torch.tensor([path_tokens], device=self.mt.device),
                ],
                dim=1,
            )
        else:
            context = self.prompt_tokens

        # logger.debug(context)
        # Get next token distribution from LM
        with torch.no_grad():
            outputs = self.mt._model(context)
            logits = outputs.logits[0, -1, :]  # Last token logits
            if self.temperature != 1.0:
                logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)

        # Get top-k tokens above probability threshold
        top_probs, top_indices = torch.topk(probs, min(self.top_k * 2, probs.shape[0]))

        # At shallow depths, force at least min_children regardless of prob_threshold
        force_children = (
            node.depth <= self.min_children_up_to_depth
            and self.min_children > 0
        )

        children = []
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            # Filter by probability threshold (relaxed if forcing children at shallow depth)
            if prob < self.prob_threshold:
                if not force_children or len(children) >= self.min_children:
                    break

            # Stop at top_k children
            if len(children) >= self.top_k:
                break

            # Decode token
            token_str = self.mt.tokenizer.decode([token_id])

            # Check if terminal
            is_terminal = False
            terminal_reason = None

            if token_id in self.stop_token_ids:
                is_terminal = True
                terminal_reason = "stop_token"
            else:
                # Check if token string contains stop string
                for stop_str in self.stop_strings:
                    if stop_str in token_str:
                        is_terminal = True
                        terminal_reason = f"stop_string:{stop_str}"
                        break

            # Check if next depth would exceed max
            if node.depth + 1 >= self.max_depth:
                is_terminal = True
                terminal_reason = "max_depth"

            # logger.debug(
            #     f"Created child: {token_str!r} (id={token_id}, prob={prob:.4f}, "
            #     f"depth={node.depth + 1}, terminal={is_terminal}, reason={terminal_reason})"
            # )

            # Create child node
            child = MCTSNode(
                token_id=token_id,
                token_str=token_str,
                prior=prob,
                parent=node,
                depth=node.depth + 1,
                is_terminal=is_terminal,
                terminal_reason=terminal_reason,
            )
            children.append(child)
            self.total_nodes += 1

        node.children = children

        # If no valid children, mark as terminal
        if not children:
            node.is_terminal = True
            node.terminal_reason = "no_valid_children"

        return children

    def _backpropagate(self, node: MCTSNode) -> None:
        """
        Update visit counts from the given node back to the root.
        """
        while node is not None:
            node.visits += 1
            node = node.parent

    def get_terminal_nodes(self) -> list[MCTSNode]:
        """
        Get all terminal nodes in the tree.

        Returns:
            List of terminal nodes (completed paths)
        """
        terminals = []
        self._collect_terminals(self.root, terminals)
        return terminals

    def _collect_terminals(self, node: MCTSNode, terminals: list[MCTSNode]) -> None:
        """Recursively collect terminal nodes."""
        if node.is_terminal:
            terminals.append(node)
        for child in node.children:
            self._collect_terminals(child, terminals)

    def get_all_paths(
        self,
        terminal_only: bool = False,
        include_intermediate: bool = False,
        contains: str | list[str] | None = None,
        depth: int | None = None,
        min_depth: int | None = None,
        max_depth: int | None = None,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> list[dict]:
        """
        Get all traversed paths in the tree with flexible filtering and sorting.

        By default, returns only "fleshed out" complete paths - those that are either:
        - Terminal nodes (reached stop token, EOS, newline, max_depth, or no valid children)
        - Leaf nodes (frontier nodes that haven't been expanded yet)

        Args:
            terminal_only: If True, only return paths ending at terminal nodes.
            include_intermediate: If True, also include intermediate nodes (nodes that
                                 have been expanded but aren't terminal). Default False.
            contains: Filter paths that contain this string (or any string in list).
                     Example: "n & 1" or ["n & 1", "n % 2"]
            depth: Filter paths with exactly this depth.
            min_depth: Filter paths with depth >= this value.
            max_depth: Filter paths with depth <= this value.
            sort_by: Sort results by this key. Options:
                    - "visits": Sort by visit count
                    - "prob": Sort by cumulative probability
                    - "depth": Sort by path depth
                    - None: No sorting (tree traversal order)
            descending: If True, sort in descending order (default).

        Returns:
            List of dicts with keys:
                - "node": The MCTSNode at the end of this path
                - "text": Full text of the path
                - "prob": Cumulative probability (product of priors)
                - "log_prob": Log of cumulative probability
                - "visits": Visit count of the end node
                - "depth": Depth of the end node
                - "is_terminal": Whether the end node is terminal
                - "terminal_reason": Reason for termination (if terminal)
        """
        # Collect all nodes
        all_nodes: list[MCTSNode] = []
        self._collect_all_nodes(self.root, all_nodes)

        # Build path info for each node
        paths = []
        for node in all_nodes:
            # Skip root node (no tokens)
            if node.is_root():
                continue

            # Filter by terminal/leaf status
            # By default, only return "fleshed out" paths (terminal or leaf nodes)
            if terminal_only:
                # Only terminal nodes
                if not node.is_terminal:
                    continue
            elif not include_intermediate:
                # Default: only fleshed out paths (terminal OR leaf)
                if not node.is_terminal and not node.is_leaf():
                    continue
            # If include_intermediate=True, include all nodes

            # Filter by depth
            if depth is not None and node.depth != depth:
                continue
            if min_depth is not None and node.depth < min_depth:
                continue
            if max_depth is not None and node.depth > max_depth:
                continue

            # Get path text
            text = node.get_path_text()

            # Filter by contains
            if contains is not None:
                if isinstance(contains, str):
                    contains = [contains]
                if not any(s in text for s in contains):
                    continue

            # Compute cumulative log probability
            log_prob = 0.0
            current = node
            while current.parent is not None:
                log_prob += math.log(current.prior + 1e-10)
                current = current.parent

            paths.append(
                {
                    "node": node,
                    "text": text,
                    "prob": math.exp(log_prob),
                    "log_prob": log_prob,
                    "visits": node.visits,
                    "depth": node.depth,
                    "is_terminal": node.is_terminal,
                    "terminal_reason": node.terminal_reason,
                }
            )

        # Sort if requested
        if sort_by is not None:
            if sort_by == "visits":
                paths.sort(key=lambda p: p["visits"], reverse=descending)
            elif sort_by == "prob":
                paths.sort(key=lambda p: p["prob"], reverse=descending)
            elif sort_by == "depth":
                paths.sort(key=lambda p: p["depth"], reverse=descending)
            else:
                raise ValueError(
                    f"Unknown sort_by value: {sort_by}. Use 'visits', 'prob', or 'depth'."
                )

        return paths

    def _collect_all_nodes(self, node: MCTSNode, nodes: list[MCTSNode]) -> None:
        """Recursively collect all nodes in the tree."""
        nodes.append(node)
        for child in node.children:
            self._collect_all_nodes(child, nodes)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search tree.

        Returns:
            Dictionary with tree statistics
        """
        terminals = self.get_terminal_nodes()
        depths = [t.depth for t in terminals]

        return {
            "total_iterations": self.total_iterations,
            "total_nodes": self.total_nodes,
            "terminal_count": len(terminals),
            "max_depth_reached": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "root_visits": self.root.visits,
            "terminal_reasons": self._count_terminal_reasons(terminals),
        }

    def _count_terminal_reasons(self, terminals: list[MCTSNode]) -> dict[str, int]:
        """Count terminal nodes by reason."""
        reasons: dict[str, int] = {}
        for t in terminals:
            reason = t.terminal_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def get_stop_tokens_info(self) -> dict:
        """
        Get information about stop tokens for debugging.

        Returns:
            Dictionary with stop token IDs and their decoded strings
        """
        info = {
            "stop_token_ids": {},
            "stop_strings": self.stop_strings,
        }
        for tok_id in self.stop_token_ids:
            decoded = self.mt.tokenizer.decode([tok_id])
            info["stop_token_ids"][tok_id] = repr(decoded)
        return info

    def debug_node(self, node: MCTSNode) -> dict:
        """
        Get debug information about a specific node.

        Args:
            node: The node to debug

        Returns:
            Dictionary with node info and why it might not be expanded
        """
        info = {
            "token_str": node.token_str,
            "token_id": node.token_id,
            "depth": node.depth,
            "visits": node.visits,
            "prior": node.prior,
            "is_terminal": node.is_terminal,
            "terminal_reason": node.terminal_reason,
            "is_leaf": node.is_leaf(),
            "num_children": len(node.children),
            "path_text": node.get_path_text(),
        }

        # Check why it might be terminal
        if node.token_id is not None:
            info["in_stop_token_ids"] = node.token_id in self.stop_token_ids
            info["contains_stop_string"] = any(
                s in node.token_str for s in self.stop_strings
            )
            info["at_max_depth"] = node.depth >= self.max_depth

        return info

    def find_node_by_text(self, text: str) -> MCTSNode | None:
        """
        Find a node whose path text ends with the given text.

        Args:
            text: Text to search for at end of path

        Returns:
            The node if found, None otherwise
        """
        all_nodes: list[MCTSNode] = []
        self._collect_all_nodes(self.root, all_nodes)
        for node in all_nodes:
            if node.get_path_text().endswith(text):
                return node
        return None

    def force_expand(self, node: MCTSNode) -> list[MCTSNode]:
        """
        Force expansion of a specific node (for debugging).

        Args:
            node: The node to expand

        Returns:
            List of created children
        """
        if node.is_terminal:
            print(f"Node is terminal (reason: {node.terminal_reason}), cannot expand")
            return []
        if not node.is_leaf():
            print(f"Node already has {len(node.children)} children")
            return node.children

        children = self._expand(node)
        print(f"Created {len(children)} children:")
        for child in children:
            print(
                f"  {child.token_str!r} (prob={child.prior:.4f}, terminal={child.is_terminal})"
            )
        return children

    def get_puct_scores(self, node: MCTSNode) -> list[tuple[MCTSNode, float]]:
        """
        Get PUCT scores for all children of a node (for debugging selection).

        Args:
            node: The parent node

        Returns:
            List of (child, score) tuples sorted by score descending
        """
        if node.is_leaf():
            return []

        parent_visits = node.visits
        sqrt_parent = math.sqrt(parent_visits) if parent_visits > 0 else 1.0

        scores = []
        for child in node.children:
            score = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            scores.append((child, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def print_tree(
        self, max_depth: int | None = None, show_colors: bool = True
    ) -> None:
        """
        Print a text representation of the tree.

        Args:
            max_depth: Maximum depth to print (None for full tree)
            show_colors: If True, use rich to show evaluation colors (default True)
        """
        # Collect all lines first, then print once to avoid
        # Jupyter creating separate output cells per console.print() call
        lines: list[str] = []
        self._collect_tree_lines(self.root, "", True, max_depth, show_colors, lines)

        if show_colors:
            # Single rich print call with all lines joined
            console = _get_console()
            console.print("\n".join(lines))
        else:
            # Use builtins.print to avoid notebook overrides
            _print = (
                __builtins__["print"]
                if isinstance(__builtins__, dict)
                else __builtins__.print
            )
            _print("\n".join(lines))

    def _collect_tree_lines(
        self,
        node: MCTSNode,
        prefix: str,
        is_last: bool,
        max_depth: int | None,
        show_colors: bool,
        lines: list[str],
    ) -> None:
        """Recursively collect tree lines into a list."""
        if max_depth is not None and node.depth > max_depth:
            return

        connector = "└── " if is_last else "├── "
        color_style = COLOR_STYLES.get(node.color, "white")

        # Build the leaf suffix: full path text shown at leaf nodes (terminal or non-terminal)
        is_leaf_node = node.is_leaf() and not node.is_root()
        leaf_suffix = ""
        if is_leaf_node:
            path_text = node.get_path_text()
            # Use repr to show \n, \t, etc. but strip the surrounding quotes
            path_repr = repr(path_text)[1:-1]
            if node.is_terminal:
                leaf_suffix = f" [TERMINAL] ('{path_repr}')"
            else:
                leaf_suffix = f" ('{path_repr}')"

        if node.is_root():
            if show_colors and node.color:
                lines.append(
                    f"[{color_style}]\\[ROOT][/{color_style}] "
                    f"[dim]visits={node.visits}[/dim]"
                )
            elif show_colors:
                lines.append(f"\\[ROOT] [dim]visits={node.visits}[/dim]")
            else:
                lines.append(f"[ROOT] visits={node.visits}")
        else:
            if show_colors and node.color:
                if is_leaf_node:
                    # Escape brackets in leaf_suffix for rich markup
                    safe_suffix = leaf_suffix.replace("[", "\\[")
                    lines.append(
                        f"{prefix}{connector}[{color_style}]{node.token_str!r}[/{color_style}] "
                        f"[dim](p={node.prior:.3f}, v={node.visits}){safe_suffix}[/dim]"
                    )
                else:
                    lines.append(
                        f"{prefix}{connector}[{color_style}]{node.token_str!r}[/{color_style}] "
                        f"[dim](p={node.prior:.3f}, v={node.visits})[/dim]"
                    )
            elif show_colors:
                if is_leaf_node:
                    safe_suffix = leaf_suffix.replace("[", "\\[")
                    lines.append(
                        f"{prefix}{connector}{node.token_str!r} "
                        f"[dim](p={node.prior:.3f}, v={node.visits}){safe_suffix}[/dim]"
                    )
                else:
                    lines.append(
                        f"{prefix}{connector}{node.token_str!r} "
                        f"[dim](p={node.prior:.3f}, v={node.visits})[/dim]"
                    )
            else:
                lines.append(
                    f"{prefix}{connector}{node.token_str!r} "
                    f"(p={node.prior:.3f}, v={node.visits}){leaf_suffix}"
                )

        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self._collect_tree_lines(
                    child, child_prefix, is_last_child, max_depth, show_colors, lines
                )

    def save_tree(
        self,
        filepath: str,
        max_depth: int | None = None,
        show_colors: bool = True,
        dark_mode: bool = True,
        print_tree: bool = False,
    ) -> None:
        """
        Save the tree to a file. Supports HTML (with colors) and plain text.

        Args:
            filepath: Path to save the tree. Use .html extension for colored output,
                     or .txt for plain text.
            max_depth: Maximum depth to include (None for full tree)
            show_colors: If True and saving to HTML, include evaluation colors
            dark_mode: If True (default), use dark background for HTML output

        Example:
            search.save_tree("mcts_tree.html")  # Colored HTML with dark background
            search.save_tree("mcts_tree.txt", show_colors=False)  # Plain text
        """
        from rich.console import Console
        from rich.terminal_theme import MONOKAI

        lines: list[str] = []
        self._collect_tree_lines(self.root, "", True, max_depth, show_colors, lines)
        content = "\n".join(lines)

        if filepath.endswith(".html"):
            # Save as HTML with colors.
            # We need console.print() to populate the recording buffer for export_html().
            # To suppress visible output: redirect to devnull and disable Jupyter display.
            import os as _os

            console = Console(
                record=True,
                highlight=False,
                width=310,
                force_jupyter=False,
                file=open(_os.devnull, "w"),
            )
            console.print(content)

            # Use a dark theme
            theme = MONOKAI if dark_mode else None

            # Export HTML and customize for dark mode
            html = console.export_html(theme=theme, inline_styles=True)

            if dark_mode:
                # Inject custom CSS for dark background and white tree edges
                custom_css = """
<style>
body {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    padding: 20px;
}
pre {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
</style>
"""
                # Insert custom CSS after <head> or at the beginning
                if "<head>" in html:
                    html = html.replace("<head>", "<head>" + custom_css)
                else:
                    html = custom_css + html

            with open(filepath, "w") as f:
                f.write(html)
            logger.info(f"Saved tree to {filepath} (HTML with colors)")
        else:
            # Save as plain text (strip rich markup if present)
            if show_colors:
                # Re-collect without colors for plain text
                lines = []
                self._collect_tree_lines(self.root, "", True, max_depth, False, lines)
                content = "\n".join(lines)
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Saved tree to {filepath} (plain text)")
