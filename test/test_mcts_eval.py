#!/usr/bin/env python
"""
Test script for MCTS tree evaluation and color backpropagation.

Tests:
1. evaluate_mcts_tree() correctly evaluates leaf nodes
2. Color assignment based on evaluation results (green/orange/red)
3. Color backpropagation with correct priority (green > orange > red)
4. print_tree() displays colors correctly

Usage:
    conda run -n goals python test/test_mcts_eval.py
"""

import os
import sys

# Add project root to path for src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import fixtures directly (same directory)
from fixtures import (
    cleanup,
    create_test_prompt,
    get_test_model,
    get_test_sample,
)
from rich.console import Console
from rich.panel import Panel

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


def test_evaluate_mcts_tree():
    """
    Test the evaluate_mcts_tree() function with a simple MCTS search.

    This test:
    1. Runs a small MCTS search on the is_odd problem
    2. Evaluates the tree with evaluate_mcts_tree()
    3. Verifies colors are assigned correctly
    4. Verifies backpropagation works (parent colors reflect children)
    """
    print_section("Test: evaluate_mcts_tree()")

    # Import modules under test
    from src.goals.eval import clear_mcts_colors, evaluate_mcts_tree
    from src.goals.sampling import MCTSSearch

    # -------------------------------------------------------------------------
    # Setup: Load model and create a simple MCTS tree
    # -------------------------------------------------------------------------
    console.print("[dim]Loading model and sample...[/dim]")
    mt = get_test_model()
    sample = get_test_sample(index=0)  # is_odd function

    console.print(f"[dim]Sample: {sample.function_name}[/dim]")
    console.print(f"[dim]Code prefix: {repr(sample.code)}[/dim]")

    # Use a simple prefill for testing
    prefill = ""
    prompt = create_test_prompt(mt, sample, prefill)

    console.print(
        "\n[dim]Running small MCTS search (max_depth=8, iterations=50)...[/dim]"
    )

    search = MCTSSearch(
        mt=mt,
        prompt=prompt,
        max_depth=8,  # Small depth for quick test
        top_k=3,  # Fewer branches
        prob_threshold=0.05,
    )

    root = search.run(n_iterations=50, depth_first=True)

    stats = search.get_stats()
    console.print(
        f"[dim]Tree stats: {stats['total_nodes']} nodes, {stats['terminal_count']} terminals[/dim]"
    )

    # -------------------------------------------------------------------------
    # Test 1: Evaluate tree and check counts
    # -------------------------------------------------------------------------
    print_section("Test 1: Basic evaluation and color assignment")

    prefix = sample.code + prefill
    counts = evaluate_mcts_tree(
        root=root,
        prefix=prefix,
        unit_tests=sample.unit_tests,
        function_name=sample.function_name,
        terminal_only=False,  # Evaluate all leaves
    )

    console.print(f"Evaluation counts: {counts}")

    # Verify counts add up
    total_check = counts["green"] + counts["orange"] + counts["red"]
    test1_passed = total_check == counts["total"]
    print_result(
        "Counts add up correctly",
        test1_passed,
        f"green({counts['green']}) + orange({counts['orange']}) + red({counts['red']}) = {total_check}, total = {counts['total']}",
    )

    # -------------------------------------------------------------------------
    # Test 2: Verify leaf nodes have colors
    # -------------------------------------------------------------------------
    print_section("Test 2: Leaf nodes have colors assigned")

    def collect_leaves(node, leaves):
        if node.is_leaf():
            leaves.append(node)
        for child in node.children:
            collect_leaves(child, leaves)

    leaves = []
    collect_leaves(root, leaves)

    leaves_with_colors = sum(1 for leaf in leaves if leaf.color is not None)
    test2_passed = leaves_with_colors == len(leaves)
    print_result(
        "All leaves have colors",
        test2_passed,
        f"{leaves_with_colors}/{len(leaves)} leaves have colors",
    )

    # Show some example leaves
    console.print("\n[dim]Sample leaf nodes:[/dim]")
    for leaf in leaves[:5]:
        console.print(
            f"  [{leaf.color or 'white'}]{repr(leaf.get_path_text()[:40])}...[/{leaf.color or 'white'}] "
            f"-> {leaf.color}"
        )

    # -------------------------------------------------------------------------
    # Test 3: Verify color backpropagation
    # -------------------------------------------------------------------------
    print_section("Test 3: Color backpropagation")

    # Check that internal nodes have colors based on children
    def verify_backprop(node):
        """Verify node's color matches expected based on children."""
        if node.is_leaf():
            return True  # Leaves are evaluated directly

        child_colors = [child.color for child in node.children if child.color]

        expected_color = None
        if "green" in child_colors:
            expected_color = "green"
        elif "orange" in child_colors:
            expected_color = "orange"
        elif "red" in child_colors:
            expected_color = "red"

        if node.color != expected_color:
            return False

        # Recursively check children
        return all(verify_backprop(child) for child in node.children)

    test3_passed = verify_backprop(root)
    print_result(
        "Backpropagation follows priority (green > orange > red)",
        test3_passed,
        f"Root color: {root.color}",
    )

    # -------------------------------------------------------------------------
    # Test 4: Print colored tree
    # -------------------------------------------------------------------------
    print_section("Test 4: Print colored tree")

    console.print("[dim]Colored tree output (max_depth=4):[/dim]\n")
    search.print_tree(max_depth=4, show_colors=True)

    test4_passed = True  # Visual inspection
    print_result("Tree prints with colors", test4_passed, "(visual inspection)")

    # -------------------------------------------------------------------------
    # Test 5: Clear colors and re-evaluate
    # -------------------------------------------------------------------------
    print_section("Test 5: Clear and re-evaluate")

    clear_mcts_colors(root)

    # Verify colors are cleared
    all_cleared = root.color is None
    for leaf in leaves:
        if leaf.color is not None:
            all_cleared = False
            break

    print_result("Colors cleared successfully", all_cleared)

    # Re-evaluate with terminal_only=True
    counts2 = evaluate_mcts_tree(
        root=root,
        prefix=prefix,
        unit_tests=sample.unit_tests,
        function_name=sample.function_name,
        terminal_only=True,
    )

    console.print(f"Re-evaluation counts (terminal_only=True): {counts2}")
    test5_passed = all_cleared and counts2["total"] <= counts["total"]
    print_result(
        "Re-evaluation works",
        test5_passed,
        f"Terminal-only evaluated {counts2['total']} vs all leaves {counts['total']}",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print_section("Test Summary")

    all_tests = [test1_passed, test2_passed, test3_passed, test4_passed, test5_passed]
    passed = sum(all_tests)
    total = len(all_tests)

    if passed == total:
        console.print(f"[bold green]All {total} tests passed![/bold green]")
    else:
        console.print(f"[bold red]{passed}/{total} tests passed[/bold red]")

    return passed == total


if __name__ == "__main__":
    console.print(
        Panel.fit(
            "[bold]MCTS Evaluation Test Suite[/bold]\n"
            "Testing evaluate_mcts_tree() and color backpropagation",
            title="Test Script",
        )
    )

    try:
        success = test_evaluate_mcts_tree()
        sys.exit(0 if success else 1)
    except Exception:
        console.print("\n[bold red]Test failed with exception:[/bold red]")
        console.print_exception()
        sys.exit(1)
    finally:
        # Cleanup
        cleanup()
