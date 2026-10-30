import logging
import warnings
from typing import Any

from src.goals.data import (
    SingleCodeEval,
    UnitTestResult,
)

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


