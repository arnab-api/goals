"""Create SinglePredicateSample JSON files for the head localization dataset.

For each predicate, calls Claude to generate:
  - Python incomplete code trimmed at `return`
  - C++ one-liner incomplete code trimmed at `return`
  - Unit tests

Results are cached via the standard LLM cache (MD5-keyed), so re-runs cost nothing.

Usage:
    # Generate all predicates defined in PREDICATE_SPECS
    conda run -n goals python scripts/create_head_loc_predicates.py

    # Generate a specific subset
    conda run -n goals python scripts/create_head_loc_predicates.py \\
        --predicates is_palindrome is_upper

    # Regenerate even if the JSON already exists
    conda run -n goals python scripts/create_head_loc_predicates.py --overwrite
"""

import argparse
import json
import logging
import os
import re
import sys

sys.path.append(".")

from src.goals.head_loc_data import HEAD_LOC_DATA_DIR, SinglePredicateSample
from src.utils.oracle_llms import ask_claude

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate registry
# Each entry: variable_type, variable_name, description
# Add new predicates here; the script handles everything else.
# ---------------------------------------------------------------------------

PREDICATE_SPECS: dict[str, dict] = {
    "is_palindrome": {
        "variable_type": "str",
        "variable_name": "s",
        "description": "Returns True if the string s reads the same forwards and backwards.",
    },
    "is_upper": {
        "variable_type": "str",
        "variable_name": "s",
        "description": "Returns True if all cased characters in s are uppercase (and there is at least one cased character).",
    },
    "is_odd": {
        "variable_type": "int",
        "variable_name": "n",
        "description": "Returns True if the integer n is odd.",
    },
    "is_power_of_two": {
        "variable_type": "int",
        "variable_name": "n",
        "description": "Returns True if n is a positive integer that is a power of two (1, 2, 4, 8, 16, ...).",
    },
    "is_sorted": {
        "variable_type": "list[int]",
        "variable_name": "lst",
        "description": "Returns True if the list lst is sorted in non-decreasing order.",
    },
    "has_duplicates": {
        "variable_type": "list[int]",
        "variable_name": "lst",
        "description": "Returns True if the list lst contains any duplicate elements.",
    },
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are generating data for a code interpretability research experiment.

Generate a Python function and a C++ equivalent for the following predicate:

  Predicate name : {predicate_name}
  Variable type  : {variable_type}  (Python type annotation)
  Variable name  : {variable_name}
  Description    : {description}

Requirements
------------
1. Python: a one-liner that uses a single `return` expression with type hints.
   The function body must be EXACTLY one line: `    return <expression>`.

2. C++: a one-liner in the form:
       bool fnName(CppType var) {{ return <expression>; }}
   BUT trimmed so it ends right before the expression, i.e.:
       bool fnName(CppType var) {{ return
   No expression, no closing brace, no semicolon after `return` — we append those ourselves.
   Use `string` for str, `int` for int, `vector<int>` for list[int].

3. Unit tests: 6–8 diverse cases covering edge cases (empty input, negatives, etc.).
   For list inputs encode them as JSON arrays.

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "function_name": "<snake_case Python function name>",
  "python_incomplete_code": "<full def line + newline + '    return'>",
  "cpp_incomplete_code": "<bool fnName(type var) {{ return>",
  "unit_tests": [[<input>, <expected_bool>], ...]
}}

Example for is_palindrome / str / s:
{{
  "function_name": "is_palindrome",
  "python_incomplete_code": "def is_palindrome(s: str) -> bool:\\n    return",
  "cpp_incomplete_code": "bool isPalindrome(string s) {{ return",
  "unit_tests": [["racecar", true], ["hello", false], ["", true], ["a", true], ["ab", false], ["aba", true]]
}}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict:
    """Extract the first {...} block from raw LLM output."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{raw}")
    return json.loads(match.group())


def _validate(data: dict, predicate_name: str) -> None:
    required = {"function_name", "python_incomplete_code", "cpp_incomplete_code", "unit_tests"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}")
    if not data["python_incomplete_code"].rstrip().endswith("return"):
        raise ValueError(
            f"python_incomplete_code must end with 'return', got: {data['python_incomplete_code']!r}"
        )
    if not data["cpp_incomplete_code"].rstrip().endswith("return"):
        raise ValueError(
            f"cpp_incomplete_code must end with 'return', got: {data['cpp_incomplete_code']!r}"
        )
    if not data["unit_tests"]:
        raise ValueError("unit_tests is empty")


def generate_predicate(predicate_name: str, spec: dict) -> SinglePredicateSample:
    prompt = PROMPT_TEMPLATE.format(predicate_name=predicate_name, **spec)
    logger.info(f"Calling Claude for '{predicate_name}' ...")
    raw = ask_claude(prompt, max_tokens=1024, temperature=0.2, use_cache=True)
    data = _extract_json(raw)
    _validate(data, predicate_name)

    # Strip any accidental trailing whitespace from the code strings
    data["python_incomplete_code"] = data["python_incomplete_code"].rstrip()
    data["cpp_incomplete_code"] = data["cpp_incomplete_code"].rstrip()

    return SinglePredicateSample(
        predicate_name=predicate_name,
        variable_type=spec["variable_type"],
        variable_name=spec["variable_name"],
        python_incomplete_code=data["python_incomplete_code"],
        cpp_incomplete_code=data["cpp_incomplete_code"],
        unit_tests=data["unit_tests"],
        function_name=data["function_name"],
    )


def save_predicate(sample: SinglePredicateSample, overwrite: bool = False) -> str:
    predicates_dir = os.path.join(HEAD_LOC_DATA_DIR, "predicates")
    os.makedirs(predicates_dir, exist_ok=True)
    path = os.path.join(predicates_dir, f"{sample.predicate_name}.json")
    if os.path.exists(path) and not overwrite:
        logger.info(f"  Already exists, skipping: {path}")
        return path
    with open(path, "w") as f:
        json.dump(sample.to_dict(), f, indent=2)
    logger.info(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate head_loc predicate JSON files.")
    parser.add_argument(
        "predicates",
        nargs="*",
        help="Predicate names to generate (default: all in PREDICATE_SPECS)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files",
    )
    args = parser.parse_args()

    names = args.predicates if args.predicates else list(PREDICATE_SPECS.keys())
    unknown = [n for n in names if n not in PREDICATE_SPECS]
    if unknown:
        logger.error(f"Unknown predicates: {unknown}. Available: {list(PREDICATE_SPECS)}")
        sys.exit(1)

    for name in names:
        try:
            sample = generate_predicate(name, PREDICATE_SPECS[name])
            save_predicate(sample, overwrite=args.overwrite)
        except Exception as e:
            logger.error(f"Failed to generate '{name}': {e}")
            raise


if __name__ == "__main__":
    main()
