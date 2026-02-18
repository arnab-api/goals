import logging
import warnings
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from src.goals.data import (
    CodeGenerations,
    EvaluationSummary,
    SingleCodeEval,
    UnitTestResult,
)

if TYPE_CHECKING:
    from src.goals.sampling import MCTSNode

logger = logging.getLogger(__name__)


def evaluate_single_code(
    full_code: str, unit_tests: list[tuple[Any, Any]], function_name: str = "func"
) -> SingleCodeEval:
    """
    Evaluate a single code snippet against unit tests.

    Returns a SingleCodeEval that captures verdict, warnings, error details,
    and structured UnitTestResult entries for each test case.
    """
    namespace: dict[str, Any] = {}
    warnings_captured: list[str] = []
    unit_test_results: list[UnitTestResult] = []

    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            exec(full_code, namespace)
        warnings_captured.extend(
            f"{warning.category.__name__}: {warning.message}"
            for warning in warning_records
        )
    except SyntaxError as exc:
        return SingleCodeEval(
            code=full_code,
            verdict="SYNTAX_ERROR",
            warnings=warnings_captured,
            unit_test_results=unit_test_results,
            error_message=str(exc),
        )
    except Exception as exc:
        return SingleCodeEval(
            code=full_code,
            verdict="ERROR",
            warnings=warnings_captured,
            unit_test_results=unit_test_results,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if function_name not in namespace:
        return SingleCodeEval(
            code=full_code,
            verdict="FAILED",
            warnings=warnings_captured,
            unit_test_results=unit_test_results,
            error_message=f"{function_name} function not defined",
        )

    func = namespace[function_name]

    for input_val, expected in unit_tests:
        test_result = UnitTestResult(input=input_val, expected=expected)
        try:
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                #! need to do this weird str() conversion in case the func returns a builtin function
                actual = str(func(input_val))
                # actual = func(input_val)
            warnings_captured.extend(
                f"{warning.category.__name__}: {warning.message}"
                for warning in warning_records
            )
            test_result.actual = actual
            test_result.passed = actual == str(expected)
        except Exception as exc:
            test_result.error = f"{type(exc).__name__}: {exc}"

        unit_test_results.append(test_result)

    verdict = "PASSED" if all(t.passed for t in unit_test_results) else "FAILED"

    return SingleCodeEval(
        code=full_code,
        verdict=verdict,
        warnings=warnings_captured,
        unit_test_results=unit_test_results,
    )


def run_evaluation(
    result: CodeGenerations,
    function_name: str | None = None,
    code_stop_tag: str = "</code>",
) -> CodeGenerations:
    """
    Run evaluation on multiple code completions.

    Args:
        result: CodeGenerations containing the prompt sample and model outputs.
        function_name: Optional override for the function name to validate. Defaults
            to the sample's function name when provided.
        code_stop_tag: String used to truncate generations before evaluation.

    Returns the updated CodeGenerations with embedded evaluation details and
    refreshed EvaluationSummary statistics.
    """
    sample = result.sample
    base_code = sample.code
    unit_tests = list(sample.unit_tests)
    target_function = (
        function_name or result.function_name or sample.function_name or "func"
    )
    result.function_name = target_function

    for code_entry in result.generations:
        prefill = code_entry.prefill or ""
        p_type = code_entry.prefill_type or ""
        out = code_entry.lm_generation
        # Clean up the output
        if code_stop_tag:
            #! Fallback: most of our code is expected to end at the first newline
            cur_stop_tag = code_stop_tag if code_stop_tag in out else "\n"
            out = out.split(cur_stop_tag)[0].rstrip()

        code_entry.lm_generation = out
        full_code = f"{base_code}{prefill}{out}"

        code_entry.source_code = full_code
        code_entry.function_name = target_function

        single_result = evaluate_single_code(full_code, unit_tests, target_function)
        single_result.prefill = (prefill, p_type)
        single_result.output = out

        code_entry.eval = single_result

    pass_count = sum(
        1
        for code_entry in result.generations
        if code_entry.eval and code_entry.eval.verdict == "PASSED"
    )
    fail_count = len(result.generations) - pass_count
    result.summary = EvaluationSummary(
        pass_count=pass_count,
        fail_count=fail_count,
        total=len(result.generations),
    )

    return result


def pretty_print_eval_results(
    eval_results: CodeGenerations, output_file: str | None = None
) -> None:
    """
    Print evaluation results in a nicely formatted table.

    Args:
        eval_results: Output from run_evaluation()
        output_file: Optional file path to write output to
    """
    # Always create a console for building the output
    console = Console(record=True, width=200)  # record=True captures output

    results_table = Table(title="Code Completion Results", show_lines=True)
    results_table.add_column("Prefill", style="white")
    results_table.add_column("Logic", style="white")
    results_table.add_column("Verdict", style="cyan")
    results_table.add_column("Unit Tests", style="white")
    results_table.add_column("Logs", style="dim", overflow="fold")

    code = eval_results.sample.code
    function_name = eval_results.function_name or "func"

    console.print('"""' + code + '"""')

    for code_entry in eval_results.generations:
        if code_entry.eval is None:
            raise ValueError(
                "Generation missing eval payload; run run_evaluation first."
            )

        eval_payload = code_entry.eval
        prefill = code_entry.prefill or ""
        p_type = code_entry.prefill_type or ""
        out = eval_payload.output

        verdict = eval_payload.verdict
        warns = eval_payload.warnings
        error_msg = eval_payload.error_message or ""

        # Format verdict
        if verdict == "PASSED":
            verdict_msg = "[bright_green]PASSED[/bright_green]"
        else:
            verdict_msg = f"[bright_red]{verdict}[/bright_red]"

        # Format unit test results
        unit_test_strs = []
        for t in eval_payload.unit_test_results:
            if t.error:
                unit_test_strs.append(f"{function_name}({t.input}) raised {t.error}")
            elif t.passed:
                unit_test_strs.append(f"{function_name}({t.input})={t.actual} ✓")
            else:
                unit_test_strs.append(
                    f"{function_name}({t.input})={t.actual}, expected {t.expected} ✗"
                )
        unit_test_str = "\n".join(unit_test_strs)

        # Format logs
        log_parts = []
        if warns:
            log_parts.append("[yellow]" + "\n".join(warns) + "[/yellow]")
        if error_msg:
            log_parts.append(error_msg)
        if verdict not in ["PASSED", "FAILED"]:
            log_parts.append(f"[dim]{eval_payload.code}[/dim]")
        log_str = "\n".join(log_parts)

        results_table.add_row(
            f'{p_type}: "{prefill}"',
            f'"[bright_yellow]{prefill}[/bright_yellow][white]{out}[/white]"',
            verdict_msg,
            unit_test_str if verdict != "PASSED" else "",
            log_str,
        )

    # Print summary
    if eval_results.summary is None:
        raise ValueError("Evaluation summary missing; run run_evaluation first.")
    summary = eval_results.summary
    console.print(results_table)
    console.print(
        f"\n[bold]Summary:[/bold] {summary.pass_count}/{summary.total} passed, {summary.fail_count}/{summary.total} failed"
    )

    # Export to file or print to stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write(console.export_text())  # Plain text with table structure preserved


def evaluate_mcts_paths(
    prefix: str,
    paths: list[dict],
    unit_tests: list[tuple[Any, Any]],
    function_name: str = "func",
    code_stop_tag: str = "</code>",
) -> list[dict]:
    """
    Evaluate a list of MCTS paths against unit tests.

    Takes paths from MCTSSearch.get_all_paths() and evaluates each by
    concatenating the prefix with the path text.

    Args:
        prefix: The code prefix (e.g., sample.code + prefill) that the MCTS
               expanded from.
        paths: List of path dicts from MCTSSearch.get_all_paths().
              Each dict should have at least "text" key.
        unit_tests: List of (input, expected_output) tuples for testing.
        function_name: The function name to test (default: "func").

    Returns:
        The same list of path dicts, with added keys:
            - "full_code": The complete code (prefix + path)
            - "eval": SingleCodeEval result from evaluation
    """
    for path in paths:
        path_text = path.get("text", "")
        if code_stop_tag:
            # Fallback: most of our code is expected to end at the first newline
            cur_stop_tag = code_stop_tag if code_stop_tag in path_text else "\n"
            path_text = path_text.split(cur_stop_tag)[0].rstrip()

        full_code = f"{prefix}{path_text}"

        # Evaluate the code
        eval_result = evaluate_single_code(full_code, unit_tests, function_name)

        # Add results to the path dict
        path["full_code"] = full_code
        path["eval"] = eval_result

    return paths


def pretty_print_mcts_eval(
    prefix: str,
    paths: list[dict],
    function_name: str = "func",
    output_file: str | None = None,
    show_passed_tests: bool = False,
    max_paths: int | None = None,
) -> None:
    """
    Pretty print evaluation results for MCTS paths.

    Args:
        prefix: The code prefix used for evaluation.
        paths: List of path dicts with "eval" key from evaluate_mcts_paths().
        function_name: The function name being tested (default: "func").
        output_file: Optional file path to write output to.
        show_passed_tests: If True, show unit test details even for passed tests.
        max_paths: Maximum number of paths to display (None for all).
    """
    console = Console(record=True, width=200)

    # Print the prefix
    console.print(f'[dim]Prefix:[/dim] "{prefix}"')
    console.print()

    # Create results table
    results_table = Table(title="MCTS Path Evaluation Results", show_lines=True)
    results_table.add_column("#", style="dim", width=4)
    results_table.add_column("Path Text", style="white", max_width=50)
    results_table.add_column("Depth", style="cyan", justify="right", width=6)
    results_table.add_column("Visits", style="cyan", justify="right", width=7)
    results_table.add_column("Prob", style="cyan", justify="right", width=10)
    results_table.add_column("Verdict", style="white", width=12)
    results_table.add_column("Unit Tests / Errors", style="white", overflow="fold")

    # Limit paths if requested
    display_paths = paths[:max_paths] if max_paths else paths

    # Count results for summary
    passed_count = 0
    failed_count = 0
    error_count = 0

    for idx, path in enumerate(display_paths):
        eval_result: SingleCodeEval | None = path.get("eval")

        if eval_result is None:
            results_table.add_row(
                str(idx + 1),
                repr(path.get("text", "")),
                str(path.get("depth", "?")),
                str(path.get("visits", "?")),
                f"{path.get('prob', 0):.2e}",
                "[yellow]NOT EVALUATED[/yellow]",
                "",
            )
            continue

        verdict = eval_result.verdict
        path_text = path.get("text", "")
        depth = path.get("depth", "?")
        visits = path.get("visits", "?")
        prob = path.get("prob", 0)

        # Format verdict with color
        if verdict == "PASSED":
            verdict_str = "[bright_green]PASSED[/bright_green]"
            passed_count += 1
        elif verdict in ["SYNTAX_ERROR", "ERROR"]:
            verdict_str = f"[bright_red]{verdict}[/bright_red]"
            error_count += 1
        else:
            verdict_str = f"[yellow]{verdict}[/yellow]"
            failed_count += 1

        # Format unit test results / error message
        details_parts = []

        if eval_result.error_message:
            details_parts.append(f"[red]{eval_result.error_message}[/red]")

        if eval_result.unit_test_results:
            if verdict != "PASSED" or show_passed_tests:
                for t in eval_result.unit_test_results:
                    if t.error:
                        details_parts.append(
                            f"[red]{function_name}({t.input}) raised {t.error}[/red]"
                        )
                    elif t.passed:
                        if show_passed_tests:
                            details_parts.append(
                                f"[green]{function_name}({t.input})={t.actual} ✓[/green]"
                            )
                    else:
                        details_parts.append(
                            f"[red]{function_name}({t.input})={t.actual}, "
                            f"expected {t.expected} ✗[/red]"
                        )

        details_str = "\n".join(details_parts)

        results_table.add_row(
            str(idx + 1),
            f'"{path_text}"',
            str(depth),
            str(visits),
            f"{prob:.2e}",
            verdict_str,
            details_str,
        )

    console.print(results_table)

    # Print summary
    total = len(display_paths)
    console.print()
    console.print(
        f"[bold]Summary:[/bold] "
        f"[green]{passed_count} passed[/green], "
        f"[yellow]{failed_count} failed[/yellow], "
        f"[red]{error_count} errors[/red] "
        f"(out of {total} paths)"
    )

    if max_paths and len(paths) > max_paths:
        console.print(f"[dim](Showing {max_paths} of {len(paths)} total paths)[/dim]")

    # Export to file or print to stdout
    if output_file:
        with open(output_file, "w") as f:
            f.write(console.export_text())


def evaluate_mcts_tree(
    root: "MCTSNode",
    prefix: str,
    unit_tests: list[tuple[Any, Any]],
    function_name: str = "func",
    code_stop_tag: str = "</code>",
    terminal_only: bool = False,
) -> dict[str, int]:
    """
    Evaluate all leaf nodes in an MCTS tree and backpropagate colors.

    This function:
    1. Finds all leaf nodes (or terminal-only if specified)
    2. Evaluates each leaf by constructing full code from prefix + path
    3. Assigns colors based on evaluation result:
       - green: all unit tests passed
       - orange: valid code but some tests failed
       - red: code has errors (syntax, runtime, function not found)
    4. Backpropagates colors up the tree with priority: green > orange > red

    Args:
        root: The root node of the MCTS tree (from MCTSSearch.root)
        prefix: The code prefix that the MCTS expanded from
                (e.g., sample.code + prefill)
        unit_tests: List of (input, expected_output) tuples for testing
        function_name: The function name to test (default: "func")
        code_stop_tag: String used to truncate generations (default: "</code>")
        terminal_only: If True, only evaluate terminal nodes.
                      If False (default), evaluate all leaf nodes.

    Returns:
        Dictionary with counts: {"green": N, "orange": N, "red": N, "total": N}
    """
    # Collect leaf nodes to evaluate
    leaves: list["MCTSNode"] = []
    _collect_leaves(root, leaves, terminal_only)

    # Evaluate each leaf
    counts = {"green": 0, "orange": 0, "red": 0, "total": len(leaves)}

    for leaf in leaves:
        path_text = leaf.get_path_text()

        # Clean up the path text
        if code_stop_tag and code_stop_tag in path_text:
            path_text = path_text.split(code_stop_tag)[0].rstrip()

        full_code = f"{prefix}{path_text}"

        # Evaluate the code
        eval_result = evaluate_single_code(full_code, unit_tests, function_name)
        leaf.eval_result = eval_result

        # Assign color based on verdict
        verdict = eval_result.verdict
        if verdict == "PASSED":
            leaf.color = "green"
            counts["green"] += 1
        elif verdict == "FAILED":
            # Valid code but tests failed
            leaf.color = "orange"
            counts["orange"] += 1
        else:
            # SYNTAX_ERROR, ERROR, or other issues
            leaf.color = "red"
            counts["red"] += 1

    # Backpropagate colors up the tree
    _backpropagate_colors(root)

    return counts


def _collect_leaves(
    node: "MCTSNode", leaves: list["MCTSNode"], terminal_only: bool
) -> None:
    """Recursively collect leaf nodes for evaluation."""
    if terminal_only:
        # Only collect terminal nodes
        if node.is_terminal:
            leaves.append(node)
    else:
        # Collect all leaves (nodes with no children)
        if node.is_leaf():
            leaves.append(node)

    for child in node.children:
        _collect_leaves(child, leaves, terminal_only)


def _backpropagate_colors(node: "MCTSNode") -> str | None:
    """
    Recursively backpropagate colors from leaves to root.

    Priority: green > orange > red > None

    Returns the color assigned to this node.
    """
    # If this is a leaf, return its color (already set during evaluation)
    if node.is_leaf():
        return node.color

    # Collect colors from all children
    child_colors = []
    for child in node.children:
        child_color = _backpropagate_colors(child)
        if child_color:
            child_colors.append(child_color)

    # Determine this node's color based on children (priority: green > orange > red)
    if "green" in child_colors:
        node.color = "green"
    elif "orange" in child_colors:
        node.color = "orange"
    elif "red" in child_colors:
        node.color = "red"
    else:
        node.color = None

    return node.color


def clear_mcts_colors(root: "MCTSNode") -> None:
    """
    Clear all colors and eval_results from an MCTS tree.

    Useful for re-evaluating with different parameters.

    Args:
        root: The root node of the MCTS tree
    """
    root.color = None
    root.eval_result = None
    for child in root.children:
        clear_mcts_colors(child)
