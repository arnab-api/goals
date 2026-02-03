"""
Monte Carlo Tree Search for LM code completion exploration.

This module implements a bounded MCTS algorithm that explores different
token continuations from an LM, building a tree of possible completions.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from src.models import ModelandTokenizer
from src.utils.typing import TokenizerOutput

logger = logging.getLogger(__name__)


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
    ):
        self.mt = mt
        self.max_depth = max_depth
        self.top_k = top_k
        self.prob_threshold = prob_threshold
        self.c_puct = c_puct

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
        self.stop_strings: list[str] = ["</code>"]
        if stop_tokens:
            self.stop_strings.extend(stop_tokens)

        # Add EOS token
        if mt.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(mt.tokenizer.eos_token_id)

        # Try to encode stop strings to token IDs (for single-token stops)
        for stop_str in self.stop_strings:
            encoded = mt.tokenizer.encode(stop_str, add_special_tokens=False)
            if len(encoded) == 1:
                self.stop_token_ids.add(encoded[0])

        # Newline tokens
        for nl in ["\n", "\r\n", "\r"]:
            encoded = mt.tokenizer.encode(nl, add_special_tokens=False)
            for tok_id in encoded:
                self.stop_token_ids.add(tok_id)

        # Initialize root node
        self.root = MCTSNode(depth=0)

        # Track statistics
        self.total_iterations = 0
        self.total_nodes = 1  # root

    def run(self, n_iterations: int = 100) -> MCTSNode:
        """
        Run MCTS for a specified number of iterations.

        Each iteration:
        1. SELECT: Traverse from root to a leaf using PUCT
        2. EXPAND: Generate children for the leaf (if not terminal)
        3. BACKPROPAGATE: Update visit counts back to root

        Args:
            n_iterations: Number of select-expand-backprop cycles

        Returns:
            The root node of the expanded tree
        """
        for i in range(n_iterations):
            # SELECT: Find a leaf node using PUCT
            leaf = self._select(self.root)

            # EXPAND: Generate children if not terminal
            if not leaf.is_terminal:
                self._expand(leaf)

            # BACKPROPAGATE: Update visits from leaf to root
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
                [self.prompt_tokens, torch.tensor([path_tokens], device=self.mt.device)],
                dim=1,
            )
        else:
            context = self.prompt_tokens

        # Get next token distribution from LM
        with torch.no_grad():
            outputs = self.mt.model(context)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = F.softmax(logits, dim=-1)

        # Get top-k tokens above probability threshold
        top_probs, top_indices = torch.topk(probs, min(self.top_k * 2, probs.shape[0]))

        children = []
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            # Filter by probability threshold
            if prob < self.prob_threshold:
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

    def get_all_paths(self) -> list[tuple[str, float, int]]:
        """
        Get all paths from root to terminal nodes.

        Returns:
            List of (text, cumulative_prior, visits) tuples
        """
        paths = []
        for terminal in self.get_terminal_nodes():
            text = terminal.get_path_text()
            # Compute cumulative log probability
            log_prob = 0.0
            node = terminal
            while node.parent is not None:
                log_prob += math.log(node.prior + 1e-10)
                node = node.parent
            paths.append((text, math.exp(log_prob), terminal.visits))
        return paths

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

    def _count_terminal_reasons(
        self, terminals: list[MCTSNode]
    ) -> dict[str, int]:
        """Count terminal nodes by reason."""
        reasons: dict[str, int] = {}
        for t in terminals:
            reason = t.terminal_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def print_tree(self, max_depth: int | None = None) -> None:
        """
        Print a text representation of the tree.

        Args:
            max_depth: Maximum depth to print (None for full tree)
        """
        self._print_node(self.root, "", True, max_depth)

    def _print_node(
        self,
        node: MCTSNode,
        prefix: str,
        is_last: bool,
        max_depth: int | None,
    ) -> None:
        """Recursively print a node and its children."""
        if max_depth is not None and node.depth > max_depth:
            return

        connector = "└── " if is_last else "├── "
        if node.is_root():
            print(f"[ROOT] visits={node.visits}")
        else:
            terminal_marker = " [TERMINAL]" if node.is_terminal else ""
            print(
                f"{prefix}{connector}{node.token_str!r} "
                f"(p={node.prior:.3f}, v={node.visits}){terminal_marker}"
            )

        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                self._print_node(child, child_prefix, is_last_child, max_depth)
