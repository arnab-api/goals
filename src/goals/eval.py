import json
import logging
import pathlib
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


# ── Multi-predicate dataset evaluation ───────────────────────────────────────

import io as _io

from src.goals.predicate_data import MultiPredicateDataset, MultiPredicateSample  # noqa: E402

_PREFILL_ORDER = ["aligned_p_plus", "aligned_p_minus", "adversarial", "inconsequential"]


def _sorted_ptypes(ptypes: "list[str]") -> list[str]:
    known = [p for p in _PREFILL_ORDER if p in ptypes]
    rest = sorted(p for p in ptypes if p not in _PREFILL_ORDER)
    return known + rest


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "–"
    return f"{100 * numerator / denominator:.1f}%"


def _blank_accum() -> dict:
    return {"pass": 0, "total": 0, "by_prefill_type": {}}


def _merge_into(accum: dict, source: dict) -> None:
    """Add source pass/total into accum in-place; merge by_prefill_type."""
    accum["pass"] += source["pass"]
    accum["total"] += source["total"]
    for ptype, data in source.get("by_prefill_type", {}).items():
        if ptype not in accum["by_prefill_type"]:
            accum["by_prefill_type"][ptype] = {"pass": 0, "total": 0}
        accum["by_prefill_type"][ptype]["pass"] += data["pass"]
        accum["by_prefill_type"][ptype]["total"] += data["total"]


def _finalise(accum: dict) -> dict:
    """Compute pass_rate fields from accumulated pass/total counts."""
    accum["pass_rate"] = accum["pass"] / accum["total"] if accum["total"] else 0.0
    for v in accum["by_prefill_type"].values():
        v["pass_rate"] = v["pass"] / v["total"] if v["total"] else 0.0
    return accum


def _aggregate_code_gen(code_gen: CodeGenerations) -> dict:
    """
    Summarise all evaluated completions in one CodeGenerations object.

    Each Code entry must already have an eval payload (i.e. run_evaluation()
    was called beforehand). Entries without eval are skipped with a warning.

    Each result entry stores full per-completion detail for use at the
    ``"sample"`` verbosity level of ``_print_multi_predicate_report``.

    Returns:
        {pass, total, pass_rate,
         by_prefill_type: {type: {pass, total, pass_rate, results: [
             {prefill, lm_generation, verdict, passed,
              unit_test_results: [{input, expected, actual, passed, error}],
              warnings, error_message}
         ]}}}
    """
    by_type: dict[str, dict] = {}
    total_pass, total_count = 0, 0

    for code in code_gen.generations:
        if code.eval is None:
            logger.warning(
                f"Generation (prefill='{code.prefill}') has no eval payload; "
                "call run_evaluation() on each CodeGenerations before passing to "
                "evaluate_multi_predicate_dataset()."
            )
            continue
        ptype = code.prefill_type or "unknown"
        if ptype not in by_type:
            by_type[ptype] = {"pass": 0, "total": 0, "results": []}
        passed = code.eval.verdict == "PASSED"
        by_type[ptype]["pass"] += int(passed)
        by_type[ptype]["total"] += 1
        by_type[ptype]["results"].append(
            {
                "prefill": code.prefill or "",
                "lm_generation": code.lm_generation or "",
                "verdict": code.eval.verdict,
                "passed": passed,
                "unit_test_results": [
                    {
                        "input": t.input,
                        "expected": t.expected,
                        "actual": t.actual,
                        "passed": t.passed,
                        "error": t.error,
                    }
                    for t in (code.eval.unit_test_results or [])
                ],
                "warnings": code.eval.warnings or [],
                "error_message": code.eval.error_message,
            }
        )
        total_pass += int(passed)
        total_count += 1

    for v in by_type.values():
        v["pass_rate"] = v["pass"] / v["total"] if v["total"] else 0.0

    return {
        "pass": total_pass,
        "total": total_count,
        "pass_rate": total_pass / total_count if total_count else 0.0,
        "by_prefill_type": by_type,
    }


def evaluate_multi_predicate_dataset(
    generations: "list[list[CodeGenerations]]",
    dataset: MultiPredicateDataset,
    save_json: "str | pathlib.Path | None" = None,
) -> dict:
    """
    Aggregate LM performance across a MultiPredicateDataset.

    Computes the report dict only — display and file saving are handled
    separately by ``_print_multi_predicate_report``.

    Args:
        generations: Nested list ``[group_idx][sample_idx]`` of CodeGenerations,
                     each already passed through ``run_evaluation()``.
                     Must match the structure of ``dataset.groups``.
        dataset:     The corresponding MultiPredicateDataset (supplies P+/P- metadata).
        save_json:   If given, write the full report as JSON to this path.

    Returns:
        Report dict with structure::

            {
              "overall":  {pass, total, pass_rate, by_prefill_type},
              "by_group": {group_name: {pass, total, pass_rate}},
              "groups": [
                {
                  "name", "p_plus",
                  "pass", "total", "pass_rate", "by_prefill_type",
                  "samples": [
                    {
                      "function_name", "p_minus",
                      "pass", "total", "pass_rate",
                      "by_prefill_type": {
                        type: {pass, total, pass_rate, results: [
                          {prefill, lm_generation, verdict, passed,
                           unit_test_results, warnings, error_message}
                        ]}
                      }
                    }
                  ]
                }
              ]
            }
    """
    if len(generations) != len(dataset.groups):
        raise ValueError(
            f"generations has {len(generations)} group(s) but "
            f"dataset has {len(dataset.groups)} group(s)"
        )

    overall_accum = _blank_accum()
    group_entries: list[dict] = []

    for group, group_gens in zip(dataset.groups, generations):
        if len(group_gens) != len(group.samples):
            raise ValueError(
                f"Group '{group.name}': {len(group_gens)} CodeGenerations provided "
                f"but group has {len(group.samples)} sample(s)"
            )

        group_accum = _blank_accum()
        sample_entries: list[dict] = []

        for sample, code_gen in zip(group.samples, group_gens):
            metrics = _aggregate_code_gen(code_gen)
            _merge_into(group_accum, metrics)

            p_minus_desc = (
                sample.p_minus.description
                if isinstance(sample, MultiPredicateSample) and sample.p_minus
                else (sample.function_name or "?")
            )
            sample_entries.append(
                {
                    "function_name": sample.function_name or "",
                    "p_minus": p_minus_desc,
                    "pass": metrics["pass"],
                    "total": metrics["total"],
                    "pass_rate": metrics["pass_rate"],
                    "by_prefill_type": metrics["by_prefill_type"],
                }
            )

        _finalise(group_accum)
        _merge_into(overall_accum, group_accum)

        group_entries.append(
            {
                "name": group.name,
                "p_plus": group.p_plus.description if group.p_plus else "",
                "pass": group_accum["pass"],
                "total": group_accum["total"],
                "pass_rate": group_accum["pass_rate"],
                "by_prefill_type": group_accum["by_prefill_type"],
                "samples": sample_entries,
            }
        )

    _finalise(overall_accum)

    report = {
        "overall": overall_accum,
        "by_group": {
            g["name"]: {
                "pass": g["pass"],
                "total": g["total"],
                "pass_rate": g["pass_rate"],
            }
            for g in group_entries
        },
        "groups": group_entries,
    }

    if save_json is not None:
        p = pathlib.Path(save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"JSON report saved to {p}")

    return report


def _print_multi_predicate_report(
    report: dict,
    level: str = "predicate",
    save_txt: "str | pathlib.Path | None" = None,
    show: bool = True,
) -> None:
    """
    Render a multi-predicate evaluation report as rich tables.

    Args:
        report:    Dict returned by ``evaluate_multi_predicate_dataset()``.
        level:     Verbosity level — one of:

                   ``"summary"``
                       Least verbose. Only the two aggregate tables:
                       Overall Summary (by target predicate) and
                       Performance by Prefill Type.

                   ``"predicate"``  *(default)*
                       Per-predicate aggregation tables (P- companions ×
                       prefill types, with subtotals and group totals)
                       followed by the summary tables.

                   ``"sample"``
                       Most verbose. Before each predicate table, shows
                       one detailed table per MultiPredicateSample with
                       per-completion rows: prefill type, completed logic,
                       verdict, and notes (failing unit tests / compile
                       errors / warnings).

        save_txt:  If given, write the rendered output as plain text to
                   this path.  Directory is created if it does not exist.
        show:      If ``True`` *(default)*, print to stdout (captured by
                   Jupyter as cell output).  If ``False``, output is only
                   written to ``save_txt`` (requires ``save_txt`` to be set).
    """
    if level not in ("summary", "predicate", "sample"):
        raise ValueError(f"level must be 'summary', 'predicate', or 'sample'; got {level!r}")
    if not show and save_txt is None:
        logger.warning("_print_multi_predicate_report: show=False but no save_txt given — nothing will be output.")
        return

    # Route output: show=True → stdout (captured by Jupyter), show=False → silent buffer.
    # force_jupyter=False is required: without it, Rich detects the Jupyter kernel and
    # routes output through IPython.display.display(), which bypasses `file` entirely
    # and always appears in the cell regardless of what file we pass.
    console = Console(
        record=True,
        highlight=False,
        width=220,
        force_jupyter=False,
        file=None if show else _io.StringIO(),
    )

    # ── Sample-level detail ───────────────────────────────────────────────────
    if level == "sample":
        for group in report["groups"]:
            console.rule(f"[bold]{group['name']}[/bold]  [dim](sample detail)[/dim]")
            console.print(f"[dim]P+:[/dim] {group['p_plus']}")
            console.print()

            for sample in group["samples"]:
                console.print(
                    f"[bold]{sample['function_name']}[/bold]"
                    f"  [dim]P-:[/dim] {sample['p_minus']}"
                )
                tbl = Table(show_lines=True, header_style="bold magenta", expand=False)
                tbl.add_column("Prefill Type", style="white", min_width=18)
                tbl.add_column("Logic", style="white", min_width=44, overflow="fold")
                tbl.add_column("Verdict", min_width=10)
                tbl.add_column("Notes", style="dim", min_width=36, overflow="fold")

                for ptype in _sorted_ptypes(list(sample["by_prefill_type"].keys())):
                    for res in sample["by_prefill_type"][ptype]["results"]:
                        logic = f'[yellow]{res["prefill"]}[/yellow]{res["lm_generation"]}'
                        if res["verdict"] == "PASSED":
                            verdict_str = "[bright_green]PASSED[/bright_green]"
                            notes = ""
                        else:
                            verdict_str = f"[bright_red]{res['verdict']}[/bright_red]"
                            note_parts = []
                            if res.get("error_message"):
                                note_parts.append(f"[red]{res['error_message']}[/red]")
                            for t in res.get("unit_test_results", []):
                                if t.get("error"):
                                    note_parts.append(f"[red]input={t['input']} raised {t['error']}[/red]")
                                elif not t.get("passed"):
                                    note_parts.append(
                                        f"input={t['input']} → got {t['actual']}, expected {t['expected']}"
                                    )
                            for w in res.get("warnings", []):
                                note_parts.append(f"[yellow]{w}[/yellow]")
                            notes = "\n".join(note_parts)

                        tbl.add_row(ptype, logic, verdict_str, notes)

                console.print(tbl)
                console.print()

    # ── Predicate-level aggregation ───────────────────────────────────────────
    if level in ("predicate", "sample"):
        for group in report["groups"]:
            console.rule(f"[bold]{group['name']}[/bold]")
            console.print(f"[dim]P+:[/dim] {group['p_plus']}")
            console.print()

            tbl = Table(show_lines=True, header_style="bold cyan", expand=False)
            tbl.add_column("P- Companion", style="white", min_width=32)
            tbl.add_column("Prefill Type", style="white", min_width=20)
            tbl.add_column("Pass", justify="right", style="green", min_width=6)
            tbl.add_column("Total", justify="right", min_width=6)
            tbl.add_column("Pass Rate", justify="right", min_width=10)

            for sample in group["samples"]:
                by_type = sample["by_prefill_type"]
                ptypes = _sorted_ptypes(list(by_type.keys()))
                first = True
                for ptype in ptypes:
                    data = by_type[ptype]
                    tbl.add_row(
                        sample["p_minus"] if first else "",
                        ptype,
                        str(data["pass"]),
                        str(data["total"]),
                        _pct(data["pass"], data["total"]),
                    )
                    first = False
                tbl.add_row(
                    "",
                    "[dim]── subtotal ──[/dim]",
                    f"[bold]{sample['pass']}[/bold]",
                    f"[bold]{sample['total']}[/bold]",
                    f"[bold]{_pct(sample['pass'], sample['total'])}[/bold]",
                )

            tbl.add_row(
                "[bold]GROUP TOTAL[/bold]",
                "",
                f"[bold cyan]{group['pass']}[/bold cyan]",
                f"[bold cyan]{group['total']}[/bold cyan]",
                f"[bold cyan]{_pct(group['pass'], group['total'])}[/bold cyan]",
            )
            console.print(tbl)
            console.print()

    # ── Overall summary ───────────────────────────────────────────────────────
    console.rule("[bold]Overall Summary[/bold]")
    console.print()

    summary_tbl = Table(show_lines=True, header_style="bold cyan", expand=False)
    summary_tbl.add_column("Target Predicate", style="white", min_width=24)
    summary_tbl.add_column("Pass", justify="right", style="green", min_width=6)
    summary_tbl.add_column("Total", justify="right", min_width=6)
    summary_tbl.add_column("Pass Rate", justify="right", min_width=10)

    for group in report["groups"]:
        summary_tbl.add_row(
            group["name"],
            str(group["pass"]),
            str(group["total"]),
            _pct(group["pass"], group["total"]),
        )

    ov = report["overall"]
    summary_tbl.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold cyan]{ov['pass']}[/bold cyan]",
        f"[bold cyan]{ov['total']}[/bold cyan]",
        f"[bold cyan]{_pct(ov['pass'], ov['total'])}[/bold cyan]",
    )
    console.print(summary_tbl)
    console.print()

    # ── By-prefill-type summary ───────────────────────────────────────────────
    console.rule("[bold]Performance by Prefill Type[/bold]")
    console.print()

    ptype_tbl = Table(show_lines=True, header_style="bold cyan", expand=False)
    ptype_tbl.add_column("Prefill Type", style="white", min_width=22)
    ptype_tbl.add_column("Pass", justify="right", style="green", min_width=6)
    ptype_tbl.add_column("Total", justify="right", min_width=6)
    ptype_tbl.add_column("Pass Rate", justify="right", min_width=10)

    for ptype in _sorted_ptypes(list(ov["by_prefill_type"].keys())):
        data = ov["by_prefill_type"][ptype]
        ptype_tbl.add_row(
            ptype,
            str(data["pass"]),
            str(data["total"]),
            _pct(data["pass"], data["total"]),
        )
    console.print(ptype_tbl)

    # ── Save to file ──────────────────────────────────────────────────────────
    if save_txt is not None:
        p = pathlib.Path(save_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(console.export_text())
        logger.info(f"Plain-text report saved to {p}")
