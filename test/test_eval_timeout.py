"""Test that _eval_with_timeout properly interrupts hanging code.

Tests the _EvalTimeout fix: because _EvalTimeout inherits from BaseException
(not Exception), it propagates through the two broad `except Exception` guards
inside evaluate_single_code and aborts the entire evaluation instead of being
silently swallowed by the first unit test, leaving subsequent unit tests hung.

Key subtlety: in non-interactive mode (no real TTY), input() raises EOFError
immediately without blocking. We patch builtins.input to actually block using
a threading.Event, which faithfully reproduces the interactive terminal scenario
where the original bug caused hangs.

Run with:
    python test/test_eval_timeout.py
"""

import builtins
import contextlib
import sys
import threading
import time

sys.path.insert(0, ".")

from scripts.causality.adversarial_persistence import _eval_with_timeout

# ── Patch builtins.input to block forever ────────────────────────────────────
# In non-interactive mode input() raises EOFError immediately — the alarm never
# fires and the timeout machinery is never exercised. We replace input() with a
# version that blocks on a threading.Event so SIGALRM is the only way out.

_never = threading.Event()  # never set → wait() blocks forever


@contextlib.contextmanager
def blocking_input_patch():
    """Context manager: replace builtins.input with a version that blocks."""
    original = builtins.input

    def _blocking(prompt=""):
        print(prompt, end="", flush=True)
        _never.wait()  # blocks indefinitely

    builtins.input = _blocking
    try:
        yield
    finally:
        builtins.input = original

TIMEOUT_SECS = 2
# Allow 1s slack for process overhead on top of the alarm timeout.
MAX_WALL_SECS = TIMEOUT_SECS + 1.0

# Minimal unit tests: (input, expected_output)
UNIT_TESTS = [(1, "1"), (2, "2")]
FUNCTION_NAME = "f"


def _run(label: str, code: str) -> dict:
    """Call _eval_with_timeout and return timing + verdict info."""
    t0 = time.monotonic()
    result = _eval_with_timeout(
        full_code=code,
        unit_tests=UNIT_TESTS,
        function_name=FUNCTION_NAME,
        timeout_secs=TIMEOUT_SECS,
        timeout_log=None,
        context={"test": label},
    )
    elapsed = time.monotonic() - t0
    return {"verdict": result.verdict, "elapsed": elapsed, "error": result.error_message}


def check(label: str, info: dict, expected_verdict: str, must_not_hang: bool = True):
    ok = True
    verdict = info["verdict"]
    elapsed = info["elapsed"]

    verdict_ok = verdict == expected_verdict
    timing_ok = elapsed <= MAX_WALL_SECS if must_not_hang else True

    status = "PASS" if (verdict_ok and timing_ok) else "FAIL"
    print(
        f"[{status}] {label}: verdict={verdict!r} ({elapsed:.2f}s)"
        + (f" | expected={expected_verdict!r}" if not verdict_ok else "")
        + (f" | TOO SLOW ({elapsed:.1f}s > {MAX_WALL_SECS}s)" if not timing_ok else "")
        + (f" | error: {info['error']}" if info["error"] else "")
    )
    if status == "FAIL":
        ok = False
    return ok


def main():
    all_ok = True

    # ── 1. Normal code — should PASS within timeout ───────────────────────────
    normal_code = "def f(n):\n    return str(n)\n"
    info = _run("normal code", normal_code)
    all_ok &= check("normal code", info, expected_verdict="PASSED")

    # ── 2. input() at module level (caught by exec() guard in evaluate_single_code)
    #       Without the fix: _EvalTimeout would be swallowed here, alarm disarmed,
    #       next statement would hang. With fix: propagates out immediately.
    module_input_code = (
        "x = input('Enter value: ')\n"
        "def f(n):\n"
        "    return str(n)\n"
    )
    with blocking_input_patch():
        info = _run("input() at module level", module_input_code)
    all_ok &= check("input() at module level", info, expected_verdict="TIMEOUT")

    # ── 3. input() inside function body (caught by func(input_val) guard)
    #       This is the primary failure mode: first unit test triggers the alarm,
    #       _EvalTimeout would have been swallowed, second test hangs forever.
    func_input_code = (
        "def f(n):\n"
        "    answer = input('Reverse input? (0/1): ')\n"
        "    return str(n) if not int(answer) else str(-n)\n"
    )
    with blocking_input_patch():
        info = _run("input() inside function", func_input_code)
    all_ok &= check("input() inside function", info, expected_verdict="TIMEOUT")

    # ── 4. Infinite loop inside function ─────────────────────────────────────
    inf_loop_code = (
        "def f(n):\n"
        "    while True:\n"
        "        pass\n"
    )
    info = _run("infinite loop", inf_loop_code)
    all_ok &= check("infinite loop", info, expected_verdict="TIMEOUT")

    # ── 5. Infinite loop at module level ─────────────────────────────────────
    inf_loop_module_code = (
        "while True:\n"
        "    pass\n"
        "def f(n):\n"
        "    return str(n)\n"
    )
    info = _run("infinite loop at module level", inf_loop_module_code)
    all_ok &= check("infinite loop at module level", info, expected_verdict="TIMEOUT")

    # ── 6. Infinite recursion — should raise RecursionError (ERROR verdict), not hang
    #       RecursionError is a subclass of Exception, so it IS caught by the
    #       func(input_val) guard and stored as test_result.error. Verdict = FAILED
    #       (not all tests passed). Should complete quickly without hanging.
    inf_recursion_code = (
        "def f(n):\n"
        "    return f(n + 1)\n"
    )
    info = _run("infinite recursion", inf_recursion_code)
    # RecursionError is caught by except Exception → each test gets an error entry
    # → verdict is FAILED (not all passed). Either FAILED or ERROR is acceptable
    # as long as it doesn't hang.
    timing_ok = info["elapsed"] <= MAX_WALL_SECS
    verdict_ok = info["verdict"] in ("FAILED", "ERROR", "TIMEOUT")
    status = "PASS" if (timing_ok and verdict_ok) else "FAIL"
    print(
        f"[{status}] infinite recursion: verdict={info['verdict']!r} ({info['elapsed']:.2f}s)"
        + (" | TOO SLOW" if not timing_ok else "")
        + (" | unexpected verdict" if not verdict_ok else "")
    )
    all_ok &= status == "PASS"

    # ── 7. Multiple unit tests: input() in function, verify ALL are aborted.
    #       Use 3 unit tests to confirm alarm fires once and all 3 are aborted.
    #       Without the fix: alarm fires on test 1, TimeoutError is swallowed,
    #       tests 2 and 3 hang forever (alarm not re-armed).
    multi_ut_code = (
        "def f(n):\n"
        "    v = input('val: ')\n"
        "    return v\n"
    )
    three_tests = [(1, "1"), (2, "2"), (3, "3")]
    t0 = time.monotonic()
    with blocking_input_patch():
        result = _eval_with_timeout(
            full_code=multi_ut_code,
            unit_tests=three_tests,
            function_name=FUNCTION_NAME,
            timeout_secs=TIMEOUT_SECS,
            timeout_log=None,
            context={"test": "multi-ut input()"},
        )
    elapsed = time.monotonic() - t0
    timing_ok = elapsed <= MAX_WALL_SECS
    verdict_ok = result.verdict == "TIMEOUT"
    status = "PASS" if (timing_ok and verdict_ok) else "FAIL"
    print(
        f"[{status}] multi-unit-test input(): verdict={result.verdict!r} ({elapsed:.2f}s)"
        + (" | TOO SLOW" if not timing_ok else "")
        + (" | expected TIMEOUT" if not verdict_ok else "")
    )
    all_ok &= status == "PASS"

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if all_ok:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
