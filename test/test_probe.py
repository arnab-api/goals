#!/usr/bin/env python
"""
Test script for probe.py - contrastive pair sampling from MCTS trees.

Tests:
1. compute_outcome_probs() correctly computes probability mass toward outcomes
2. collect_nodes_at_depth() collects nodes at correct depths
3. sample_contrastive_pairs() filters by confidence thresholds
4. Leaf/terminal nodes are excluded from sampling
5. Uncertain (SYNTAX_ERROR) nodes are handled correctly
6. sample_multidepth_pairs() samples across depths

Usage:
    conda run -n goals python test/test_probe.py
"""

import os
import sys

# Add project root to path for src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.panel import Panel

from src.goals.probe import (
    OutcomeProbs,
    ContrastivePair,
    compute_outcome_probs,
    collect_nodes_at_depth,
    sample_contrastive_pairs,
    sample_multidepth_pairs,
    summarize_contrastive_pairs,
    print_contrastive_pairs,
)
from src.goals.sampling import MCTSNode

console = Console()


def print_section(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def print_result(test_name: str, passed: bool, message: str = ""):
    """Print a test result."""
    status = (
        "[bold green]PASSED[/bold green]" if passed else "[bold red]FAILED[/bold red]"
    )
    console.print(f"  {test_name}: {status}")
    if message:
        console.print(f"    {message}")


def create_mock_tree():
    """
    Create a simple mock MCTS tree for testing.

    Structure (priors in parentheses):

    ROOT (depth=0)
    ├── A (0.6, depth=1) -> eventually GREEN
    │   ├── A1 (0.7) -> GREEN leaf
    │   └── A2 (0.3) -> RED leaf
    └── B (0.4, depth=1) -> eventually RED
        ├── B1 (0.5) -> RED leaf
        └── B2 (0.5) -> ORANGE leaf (valid but failed)
    """
    root = MCTSNode(token_id=None, token_str="", depth=0)

    # Branch A - leads to green (but has one red child)
    node_a = MCTSNode(token_id=1, token_str=" n", prior=0.6, parent=root, depth=1)
    root.children.append(node_a)

    node_a1 = MCTSNode(token_id=2, token_str=" % 2", prior=0.7, parent=node_a, depth=2)
    node_a1.color = "green"  # PASSED
    node_a.children.append(node_a1)

    node_a2 = MCTSNode(token_id=3, token_str=" + 1", prior=0.3, parent=node_a, depth=2)
    node_a2.color = "red"  # ERROR
    node_a.children.append(node_a2)

    # Branch B - all red/orange
    node_b = MCTSNode(token_id=4, token_str=" True", prior=0.4, parent=root, depth=1)
    root.children.append(node_b)

    node_b1 = MCTSNode(token_id=5, token_str=" if", prior=0.5, parent=node_b, depth=2)
    node_b1.color = "red"  # ERROR
    node_b.children.append(node_b1)

    node_b2 = MCTSNode(
        token_id=6, token_str="</code>", prior=0.5, parent=node_b, depth=2
    )
    node_b2.color = "orange"  # FAILED (valid but wrong)
    node_b.children.append(node_b2)

    return root, {
        "A": node_a,
        "B": node_b,
        "A1": node_a1,
        "A2": node_a2,
        "B1": node_b1,
        "B2": node_b2,
    }


def test_compute_outcome_probs():
    """Test outcome probability computation."""
    print_section("Test 1: compute_outcome_probs")

    root, nodes = create_mock_tree()
    all_passed = True

    # Test leaf nodes
    probs_a1 = compute_outcome_probs(nodes["A1"])
    console.print(f"A1 (green leaf): {probs_a1}")
    test_passed = probs_a1.green == 1.0 and probs_a1.red == 0.0
    print_result("Green leaf returns green=1.0", test_passed)
    all_passed &= test_passed

    probs_a2 = compute_outcome_probs(nodes["A2"])
    console.print(f"A2 (red leaf): {probs_a2}")
    test_passed = probs_a2.red == 1.0 and probs_a2.green == 0.0
    print_result("Red leaf returns red=1.0", test_passed)
    all_passed &= test_passed

    probs_b2 = compute_outcome_probs(nodes["B2"])
    console.print(f"B2 (orange leaf): {probs_b2}")
    test_passed = probs_b2.red == 1.0
    print_result("Orange leaf treated as red", test_passed)
    all_passed &= test_passed

    # Test internal node A: 0.7*green + 0.3*red
    probs_a = compute_outcome_probs(nodes["A"])
    console.print(f"A (internal, has green child): {probs_a}")
    test_passed = abs(probs_a.green - 0.7) < 0.01 and abs(probs_a.red - 0.3) < 0.01
    print_result(
        "Internal node A: p_green=0.7, p_red=0.3",
        test_passed,
        f"Got green={probs_a.green:.3f}, red={probs_a.red:.3f}",
    )
    all_passed &= test_passed

    # Test internal node B: 0.5*red + 0.5*red = 1.0 red
    probs_b = compute_outcome_probs(nodes["B"])
    console.print(f"B (internal, all red/orange): {probs_b}")
    test_passed = abs(probs_b.red - 1.0) < 0.01 and abs(probs_b.green - 0.0) < 0.01
    print_result(
        "Internal node B: p_red=1.0",
        test_passed,
        f"Got green={probs_b.green:.3f}, red={probs_b.red:.3f}",
    )
    all_passed &= test_passed

    # Test root: 0.6*(0.7g+0.3r) + 0.4*(1.0r) = 0.42g + 0.58r
    probs_root = compute_outcome_probs(root)
    console.print(f"Root: {probs_root}")
    expected_green = 0.6 * 0.7
    expected_red = 0.6 * 0.3 + 0.4 * 1.0
    test_passed = (
        abs(probs_root.green - expected_green) < 0.01
        and abs(probs_root.red - expected_red) < 0.01
    )
    print_result(
        f"Root: p_green={expected_green:.2f}, p_red={expected_red:.2f}",
        test_passed,
        f"Got green={probs_root.green:.3f}, red={probs_root.red:.3f}",
    )
    all_passed &= test_passed

    return all_passed


def test_collect_nodes_at_depth():
    """Test collecting nodes at specific depths."""
    print_section("Test 2: collect_nodes_at_depth")

    root, nodes = create_mock_tree()
    all_passed = True

    depth_0 = collect_nodes_at_depth(root, 0)
    test_passed = len(depth_0) == 1 and depth_0[0] is root
    print_result("Depth 0: 1 node (root)", test_passed, f"Got {len(depth_0)} nodes")
    all_passed &= test_passed

    depth_1 = collect_nodes_at_depth(root, 1)
    test_passed = len(depth_1) == 2
    print_result(
        "Depth 1: 2 nodes",
        test_passed,
        f"Got {len(depth_1)} nodes: {[n.token_str for n in depth_1]}",
    )
    all_passed &= test_passed

    depth_2 = collect_nodes_at_depth(root, 2)
    test_passed = len(depth_2) == 4
    print_result(
        "Depth 2: 4 nodes",
        test_passed,
        f"Got {len(depth_2)} nodes: {[n.token_str for n in depth_2]}",
    )
    all_passed &= test_passed

    depth_3 = collect_nodes_at_depth(root, 3)
    test_passed = len(depth_3) == 0
    print_result("Depth 3: 0 nodes", test_passed, f"Got {len(depth_3)} nodes")
    all_passed &= test_passed

    return all_passed


def test_sample_contrastive_pairs():
    """Test contrastive pair sampling."""
    print_section("Test 3: sample_contrastive_pairs")

    root, nodes = create_mock_tree()
    all_passed = True

    # At depth 1:
    # Node A: p_green=0.7, p_red=0.3 -> won't qualify with strict max_red_in_positive=0.2
    # Node B: p_green=0.0, p_red=1.0 -> qualifies as negative

    # With default thresholds, A won't qualify as positive (p_red=0.3 > 0.2)
    pairs_strict = sample_contrastive_pairs(root, depth=1)
    test_passed = len(pairs_strict) == 0
    print_result(
        "Strict thresholds: no pairs (A has p_red=0.3 > 0.2)",
        test_passed,
        f"Got {len(pairs_strict)} pairs",
    )
    all_passed &= test_passed

    # Relax thresholds to allow A as positive
    pairs_relaxed = sample_contrastive_pairs(
        root,
        depth=1,
        min_green_confidence=0.5,
        max_red_in_positive=0.35,  # Allow A with p_red=0.3
    )
    test_passed = len(pairs_relaxed) == 1
    print_result(
        "Relaxed thresholds: 1 pair",
        test_passed,
        f"Got {len(pairs_relaxed)} pairs",
    )
    all_passed &= test_passed

    if pairs_relaxed:
        pair = pairs_relaxed[0]
        test_passed = (
            pair.positive.token_str == " n" and pair.negative.token_str == " True"
        )
        print_result(
            "Correct pair: pos=' n', neg=' True'",
            test_passed,
            f"Got pos={pair.positive.token_str!r}, neg={pair.negative.token_str!r}",
        )
        all_passed &= test_passed

    return all_passed


def test_leaf_terminal_exclusion():
    """Test that leaf and terminal nodes are excluded from sampling."""
    print_section("Test 4: Leaf/terminal node exclusion")

    # Create a deeper tree where depth 3 has leaf nodes
    root = MCTSNode(token_id=None, token_str="", depth=0)

    # Branch A -> green path (internal at depth 1,2; leaf at depth 3)
    node_a = MCTSNode(token_id=1, token_str=" n", prior=0.6, parent=root, depth=1)
    root.children.append(node_a)
    node_a1 = MCTSNode(token_id=2, token_str=" %", prior=0.8, parent=node_a, depth=2)
    node_a.children.append(node_a1)
    node_a1a = MCTSNode(token_id=3, token_str=" 2", prior=0.9, parent=node_a1, depth=3)
    node_a1a.color = "green"  # leaf
    node_a1.children.append(node_a1a)
    node_a1b = MCTSNode(token_id=4, token_str=" 3", prior=0.1, parent=node_a1, depth=3)
    node_a1b.color = "orange"  # leaf
    node_a1.children.append(node_a1b)

    # Branch B -> red path (internal at depth 1; leaf at depth 2)
    node_b = MCTSNode(token_id=5, token_str=" True", prior=0.4, parent=root, depth=1)
    root.children.append(node_b)
    node_b1 = MCTSNode(token_id=6, token_str="</code>", prior=1.0, parent=node_b, depth=2)
    node_b1.color = "red"  # leaf
    node_b.children.append(node_b1)

    all_passed = True

    # Depth 3 should have no pairs (both nodes are leaves)
    pairs_d3 = sample_contrastive_pairs(root, depth=3, max_red_in_positive=0.35)
    test_passed = len(pairs_d3) == 0
    print_result(
        "Depth 3: no pairs (all leaves)",
        test_passed,
        f"Got {len(pairs_d3)} pairs",
    )
    all_passed &= test_passed

    # Depth 2 should have no pairs (B1 is a leaf, only candidate for negative)
    pairs_d2 = sample_contrastive_pairs(root, depth=2, max_red_in_positive=0.35)
    test_passed = len(pairs_d2) == 0
    print_result(
        "Depth 2: no pairs (negative candidate is leaf)",
        test_passed,
        f"Got {len(pairs_d2)} pairs",
    )
    all_passed &= test_passed

    # Depth 1 should have 1 pair (both A and B are internal)
    pairs_d1 = sample_contrastive_pairs(root, depth=1, max_red_in_positive=0.35)
    test_passed = len(pairs_d1) == 1
    print_result(
        "Depth 1: 1 pair (both internal)",
        test_passed,
        f"Got {len(pairs_d1)} pairs",
    )
    all_passed &= test_passed

    return all_passed


def test_uncertain_nodes():
    """Test handling of SYNTAX_ERROR (uncertain) nodes."""
    print_section("Test 5: Uncertain nodes handling")

    root = MCTSNode(token_id=None, token_str="", depth=0)

    # Node C - all children are SYNTAX_ERROR (uncertain)
    node_c = MCTSNode(token_id=7, token_str=" x", prior=0.5, parent=root, depth=1)
    root.children.append(node_c)
    node_c1 = MCTSNode(token_id=8, token_str=" =", prior=1.0, parent=node_c, depth=2)
    node_c1.color = None  # SYNTAX_ERROR / incomplete
    node_c.children.append(node_c1)

    # Node D - has green child
    node_d = MCTSNode(token_id=9, token_str=" y", prior=0.5, parent=root, depth=1)
    root.children.append(node_d)
    node_d1 = MCTSNode(token_id=10, token_str=" + 1", prior=1.0, parent=node_d, depth=2)
    node_d1.color = "green"
    node_d.children.append(node_d1)

    all_passed = True

    probs_c = compute_outcome_probs(node_c)
    test_passed = probs_c.uncertain == 1.0
    print_result(
        "Node C (uncertain children): p_uncertain=1.0",
        test_passed,
        f"Got {probs_c}",
    )
    all_passed &= test_passed

    probs_d = compute_outcome_probs(node_d)
    test_passed = probs_d.green == 1.0
    print_result(
        "Node D (green child): p_green=1.0",
        test_passed,
        f"Got {probs_d}",
    )
    all_passed &= test_passed

    # C should be excluded due to high uncertainty, D has no negative to pair with
    pairs = sample_contrastive_pairs(root, depth=1, max_uncertain=0.5)
    test_passed = len(pairs) == 0
    print_result(
        "No pairs (C excluded, no negative for D)",
        test_passed,
        f"Got {len(pairs)} pairs",
    )
    all_passed &= test_passed

    return all_passed


def test_multidepth_sampling():
    """Test multi-depth sampling."""
    print_section("Test 6: sample_multidepth_pairs")

    root, nodes = create_mock_tree()
    all_passed = True

    # Use relaxed thresholds
    pairs_by_depth = sample_multidepth_pairs(
        root,
        min_depth=1,
        max_depth=2,
        min_green_confidence=0.5,
        max_red_in_positive=0.35,
    )

    console.print(f"Depths with pairs: {list(pairs_by_depth.keys())}")

    # Only depth 1 should have pairs (depth 2 nodes are all leaves)
    test_passed = list(pairs_by_depth.keys()) == [1]
    print_result(
        "Only depth 1 has pairs",
        test_passed,
        f"Got depths: {list(pairs_by_depth.keys())}",
    )
    all_passed &= test_passed

    summary = summarize_contrastive_pairs(pairs_by_depth)
    console.print(f"Summary: {summary}")

    test_passed = summary["total_pairs"] == 1
    print_result(
        "Total pairs = 1",
        test_passed,
        f"Got {summary['total_pairs']}",
    )
    all_passed &= test_passed

    return all_passed


def run_all_tests():
    """Run all tests and report summary."""
    console.print(
        Panel.fit(
            "[bold]Probe Module Test Suite[/bold]\n"
            "Testing contrastive pair sampling from MCTS trees",
            title="Test Script",
        )
    )

    results = []
    results.append(("compute_outcome_probs", test_compute_outcome_probs()))
    results.append(("collect_nodes_at_depth", test_collect_nodes_at_depth()))
    results.append(("sample_contrastive_pairs", test_sample_contrastive_pairs()))
    results.append(("leaf/terminal exclusion", test_leaf_terminal_exclusion()))
    results.append(("uncertain nodes", test_uncertain_nodes()))
    results.append(("multidepth sampling", test_multidepth_sampling()))

    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[green]PASS[/green]" if p else "[red]FAIL[/red]"
        console.print(f"  {name}: {status}")

    console.print()
    if passed == total:
        console.print(f"[bold green]All {total} tests passed![/bold green]")
    else:
        console.print(f"[bold red]{passed}/{total} tests passed[/bold red]")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
