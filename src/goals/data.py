import json
import logging
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config


logger = logging.getLogger(__name__)


_NO_COMMENT_STR = (
    "Write your code without any comments or explanations, just the implementation."
)


@dataclass
class IncompleteCodeSample(DataClassJsonMixin):
    instruction: str
    code: str
    unit_tests: list[tuple[Any, Any]] = field(default_factory=list)
    prefills: dict[str, list[str]] = field(default_factory=dict)
    function_name: str | None = None
    augmentation: list[dict] | None = None
    without_comments: bool = False
    no_comment_str: str = field(
        default=_NO_COMMENT_STR,
        metadata=config(exclude=lambda v: v == _NO_COMMENT_STR),
    )


    def __str__(self) -> str:
        return f"""Instruction: {self.instruction}
{self.code}
Unit Tests: {self.unit_tests}
Prefills: {json.dumps(self.prefills, indent=2)}"""

    def __post_init__(self) -> None:
        # ensure unit tests are stored as tuples for consistency
        self.unit_tests = [
            test if isinstance(test, tuple) else tuple(test) for test in self.unit_tests
        ]


@dataclass
class UnitTestResult(DataClassJsonMixin):
    input: Any
    expected: Any
    actual: Any | None = None
    passed: bool = False
    error: str | None = None

    def __str__(self) -> str:
        status = "(✓)" if self.passed else "(✗)"
        return f"Input: {self.input}, Expected: {self.expected}, Actual: {self.actual}, Status: {status}, Error: {self.error}"


@dataclass
class SingleCodeEval(DataClassJsonMixin):
    code: str
    verdict: str
    unit_test_results: list[UnitTestResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    prefill: tuple[str, str] | None = None
    output: str = ""
