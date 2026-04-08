import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

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

    @property
    def full_instruction(self) -> str:
        """Instruction with optional no-comment suffix appended."""
        if self.without_comments:
            return self.instruction.rstrip(". ") + ". " + self.no_comment_str
        return self.instruction

    @property
    def is_original(self) -> bool:
        return not self.augmentation  # True for None and []

    def augment(
        self, augmentation_type: str, augmentation_config: dict | None = None
    ) -> "IncompleteCodeSample":
        from src.goals.augment import augment_sample

        return augment_sample(self, augmentation_type, augmentation_config or {})

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
class IncompleteCodeDataset(DataClassJsonMixin):
    samples: list[IncompleteCodeSample] = field(default_factory=list)

    def augment(self) -> None:
        from src.goals.augment import generate_variants_incomplete

        new = [
            v
            for s in self.samples
            if s.is_original
            for v in generate_variants_incomplete(s)
        ]
        self.samples.extend(new)

    @staticmethod
    def load_dataset(
        path: PathLike = os.path.join(DEFAULT_DATA_DIR, "incomplete_code.json"),
        augment: bool = False,
    ) -> "IncompleteCodeDataset":
        with open(path, "r") as f:
            data = json.load(f)
        samples = [IncompleteCodeSample.from_dict(sample_dict) for sample_dict in data]

        logger.info(f"Loaded {len(samples)} samples from {path}")
        ds = IncompleteCodeDataset(samples=samples)
        if augment:
            ds.augment()
        return ds

    def find_by_function_name(self, name: str) -> "IncompleteCodeSample | None":
        """Return the first sample whose function_name matches name, or None."""
        for sample in self.samples:
            if sample.function_name == name:
                return sample
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> IncompleteCodeSample:
        return self.samples[idx]


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


@dataclass
class EvaluationSummary(DataClassJsonMixin):
    pass_count: int = 0
    fail_count: int = 0
    total: int = 0


@dataclass
class Code(DataClassJsonMixin):
    prefill: str | None = None
    prefill_type: str | None = None
    full_prompt: str = ""  # full prompt sent to the LM
    lm_generation: str = ""  # completion by the LM
    source_code: str | None = None  # code to execute
    function_name: str | None = None
    eval: SingleCodeEval | None = None  # None means not evaluated yet


@dataclass
class CodeGenerations(DataClassJsonMixin):
    sample: IncompleteCodeSample
    generations: list[Code] = field(default_factory=list)
    summary: EvaluationSummary | None = None
    function_name: str | None = None

    def __str__(self) -> str:
        return f"""{self.sample.code}
Generations:
{chr(10).join([f'({code.prefill_type or ""}) "{code.prefill or ""}" => "{code.lm_generation}"' for code in self.generations])}"""

    def __post_init__(self) -> None:
        # Normalize generations to Code objects (legacy tuples or dicts -> Code)
        normalized: list[Code] = []
        for entry in self.generations:
            if isinstance(entry, Code):
                normalized.append(entry)
                continue
            if isinstance(entry, dict):
                normalized.append(Code.from_dict(entry))
                continue
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                prefill_info, generation_text = entry
                prefill_value = None
                prefill_type = None
                if isinstance(prefill_info, (list, tuple)) and len(prefill_info) == 2:
                    prefill_value, prefill_type = prefill_info
                elif isinstance(prefill_info, str):
                    prefill_value = prefill_info
                normalized.append(
                    Code(
                        prefill=prefill_value,
                        prefill_type=prefill_type,
                        lm_generation=generation_text
                        if isinstance(generation_text, str)
                        else str(generation_text),
                    )
                )
                continue
            normalized.append(Code(lm_generation=str(entry)))

        self.generations = normalized

        if self.function_name is None:
            self.function_name = self.sample.function_name

        if self.summary is None:
            self.summary = EvaluationSummary(total=len(self.generations))


# # Backwards compatibility alias for older imports
# CodeGenerationResult = CodeGenerations
