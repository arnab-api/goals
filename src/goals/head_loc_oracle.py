"""Oracle LLM-based pivot prefix mining — Case 3.

Addresses the core limitation of Case 2 (MCTS-based LCP): MCTS finds LCPs
across *all* passing implementations of the destination predicate, so when dest
has diverse implementations (e.g. ``s.isupper()`` and ``s == s.upper()``), the
LCP collapses to just ``s`` — too short to be a meaningful pivot.

**Strategy**: give the oracle *both* predicates and ask it to directly identify
shared prefixes where the two functions would diverge.  The oracle has semantic
understanding and can reason about structural shared prefixes without needing to
enumerate implementations and compute cross-LCPs mechanically.

Example oracle output for (is_upper, is_palindrome):
  PREFIX: s == s.
  A: s == s.upper()
  B: s == s[::-1]

Pipeline:
  1. Call ``get_oracle_lcps(dest_sample, targ_sample)`` — oracle returns candidate
     pivot prefix strings directly.
  2. Completion gate: greedily generate from ``return {lcp}`` for both dest and
     target; both must pass all unit tests (confirms the prefix is semantically
     valid for each side).
  3. Divergence gate: ``dest_top_token != target_top_token`` at ``return {lcp}``.
  4. Emit ``PivotInstance`` for each passing (pair, lcp).

Single-predicate helpers (``build_oracle_prompt``, ``get_oracle_implementations``)
are kept as utilities for notebook inspection.
"""

from __future__ import annotations

import logging

import torch

from src.goals.eval import evaluate_single_code
from src.goals.head_loc_data import (
    HeadLocDataset,
    PivotInstance,
    PredicatePair,
    SinglePredicateSample,
)
from src.goals.head_loc_pivot import (
    build_python_prompt,
    get_next_token_probs,
    get_top_token,
)
from src.tokens import prepare_input
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-predicate oracle (inspection / legacy)
# ---------------------------------------------------------------------------

_ORACLE_PROMPT_TEMPLATE = """\
Given a Python predicate function ending at `return`, list up to {n} distinct \
idiomatic single-expression return values (what follows `return `). Include diverse \
styles — operator-based, built-in-based, boolean-based — so that together they \
cover the common ways a programmer might implement the predicate. \
One expression per line, prefixed with `- `.

=== Example ===

def is_even(n: int) -> bool:
    return
- n % 2 == 0
- not n % 2
- n & 1 == 0

def is_positive(n: int) -> bool:
    return
- n > 0
- n >= 1

=== Your turn ===

{python_incomplete_code}
    return
"""


def build_oracle_prompt(sample: SinglePredicateSample, n: int = 5) -> str:
    """Build the single-predicate oracle prompt (used for inspection)."""
    return _ORACLE_PROMPT_TEMPLATE.format(
        python_incomplete_code=sample.python_incomplete_code,
        n=n,
    )


def parse_oracle_response(response: str) -> list[str]:
    """Extract ``- <expr>`` lines from a single-predicate oracle response."""
    impls: list[str] = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("- "):
            expr = line[2:].strip()
            if expr and not expr.startswith("#") and "\n" not in expr:
                impls.append(expr)
    return impls


def get_oracle_implementations(
    sample: SinglePredicateSample,
    n: int = 5,
    temperature: float = 0.5,
) -> list[str]:
    """Query the oracle for up to ``n`` implementations of a single predicate (cached)."""
    prompt = build_oracle_prompt(sample, n=n)
    response = ask_claude(prompt, temperature=temperature, max_tokens=512)
    logger.debug(f"Oracle response for {sample.predicate_name!r}:\n{response}")
    impls = parse_oracle_response(response)
    seen: set[str] = set()
    deduped = [i for i in impls if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]
    logger.info(
        f"Oracle implementations for {sample.predicate_name!r} "
        f"({len(deduped)} unique): {deduped}"
    )
    return deduped


# ---------------------------------------------------------------------------
# Pair-based oracle: ask for divergence prefixes directly
# ---------------------------------------------------------------------------

_ORACLE_PAIR_PROMPT_TEMPLATE = """\
You are given two Python predicate functions that both end at `return`. \
Find up to {n} short shared prefixes (what immediately follows `return `) such that:
  1. Both predicates can be validly completed starting from that prefix.
  2. The two predicates would naturally complete the prefix *differently*.

For each prefix, also show the expected completion for each function.
Format strictly — one block per prefix, no extra text:

PREFIX: <prefix text>
A: <how function A completes it>
B: <how function B completes it>

=== Example ===

Function A:
def is_divisible_by_three(n: int) -> bool:
    return

Function B:
def is_divisible_by_five(n: int) -> bool:
    return

PREFIX: n %
A: n % 3 == 0
B: n % 5 == 0

PREFIX: not n %
A: not n % 3
B: not n % 5

=== Your turn ===

Function A (destination):
{dest_code}
    return

Function B (target):
{targ_code}
    return
"""


def build_oracle_pair_prompt(
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n: int = 5,
) -> str:
    """Build the pair-based oracle prompt asking for divergence prefixes directly.

    Args:
        dest_sample: Destination predicate.
        targ_sample: Target predicate.
        n: Maximum number of prefixes to request.

    Returns:
        Prompt string ready for ``ask_claude``.
    """
    return _ORACLE_PAIR_PROMPT_TEMPLATE.format(
        dest_code=dest_sample.python_incomplete_code,
        targ_code=targ_sample.python_incomplete_code,
        n=n,
    )


def parse_oracle_pair_response(response: str) -> list[str]:
    """Extract ``PREFIX: <text>`` lines from an oracle pair response.

    Args:
        response: Raw oracle LLM response.

    Returns:
        List of prefix strings (deduplicated, order-preserved).
    """
    prefixes: list[str] = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("PREFIX:"):
            prefix = line[len("PREFIX:"):].strip()
            if prefix:
                prefixes.append(prefix)
    seen: set[str] = set()
    return [p for p in prefixes if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]


def get_oracle_lcps(
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n: int = 5,
    temperature: float = 0.5,
) -> list[str]:
    """Ask the oracle LLM for shared divergence prefixes for a (dest, target) pair.

    The oracle sees both predicates and reasons about where they structurally
    diverge — more targeted than computing cross-LCPs from independent
    implementations.  Results are cached by prompt hash.

    Args:
        dest_sample: Destination predicate.
        targ_sample: Target predicate.
        n: Number of prefixes to request.
        temperature: Oracle sampling temperature.

    Returns:
        List of candidate pivot prefix strings (deduplicated, order-preserved).
    """
    prompt = build_oracle_pair_prompt(dest_sample, targ_sample, n=n)
    response = ask_claude(prompt, temperature=temperature, max_tokens=512)
    logger.debug(
        f"Oracle pair response for "
        f"({dest_sample.predicate_name}, {targ_sample.predicate_name}):\n{response}"
    )
    lcps = parse_oracle_pair_response(response)
    logger.info(
        f"Oracle LCPs for ({dest_sample.predicate_name}, {targ_sample.predicate_name})"
        f" ({len(lcps)}): {lcps}"
    )
    return lcps


# ---------------------------------------------------------------------------
# Gate: successful completion
# ---------------------------------------------------------------------------


def _greedy_complete_and_eval(
    mt,
    sample: SinglePredicateSample,
    lcp: str,
    include_docstring: bool,
    max_new_tokens: int,
    stop_ids: set[int],
) -> bool:
    """Generate greedily from ``return {lcp}`` and check unit tests for ``sample``."""
    pivot_prompt = build_python_prompt(
        mt, sample, pivot_prefix=lcp, include_docstring=include_docstring
    )
    inputs = prepare_input(prompts=[pivot_prompt], tokenizer=mt)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = mt._model(input_ids=input_ids, attention_mask=attention_mask)
            next_tok = int(out.logits[0, -1, :].argmax())
            generated_ids.append(next_tok)
            if next_tok in stop_ids:
                break
            next_tok_t = torch.tensor([[next_tok]], device=input_ids.device)
            ones = torch.ones(
                (1, 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            input_ids = torch.cat([input_ids, next_tok_t], dim=1)
            attention_mask = torch.cat([attention_mask, ones], dim=1)

    generated_text = mt.tokenizer.decode(generated_ids, skip_special_tokens=True)
    code_stop_tag = "</code>"
    if code_stop_tag in generated_text:
        generated_text = generated_text.split(code_stop_tag)[0]
    else:
        generated_text = generated_text.split("\n")[0]
    generated_text = generated_text.rstrip()

    full_code = sample.python_incomplete_code + " " + lcp + generated_text
    result = evaluate_single_code(
        full_code,
        sample.unit_tests,
        function_name=sample.function_name or sample.predicate_name,
    )
    logger.debug(
        f"  eval [{sample.predicate_name}] lcp={lcp!r}"
        f"  generated={generated_text!r}  verdict={result.verdict}"
    )
    return result.verdict == "green"


def _completion_gate(
    mt,
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    lcp: str,
    include_docstring: bool = False,
    max_new_tokens: int = 32,
) -> bool:
    """True if greedy generation from ``return {lcp}`` yields passing completions
    for **both** the destination and target predicates.

    A valid LCP must be reachable and semantically correct for both sides:
    dest must produce a valid dest implementation, and target must produce a valid
    target implementation, each starting from the same prefix.

    Short-circuits: dest is checked first; target is skipped if dest fails.

    Args:
        mt: ModelandTokenizer (frozen, used for generation only).
        dest_sample: Destination predicate — code prefix and unit tests.
        targ_sample: Target predicate — code prefix and unit tests.
        lcp: Candidate pivot prefix string (goes after ``return ``).
        include_docstring: Passed to ``build_python_prompt``.
        max_new_tokens: Maximum tokens to generate before giving up.

    Returns:
        True if both dest and target completions pass all unit tests.
    """
    nl_ids = set(mt.tokenizer.encode("\n", add_special_tokens=False))
    stop_ids = {mt.tokenizer.eos_token_id} | nl_ids

    dest_ok = _greedy_complete_and_eval(
        mt, dest_sample, lcp, include_docstring, max_new_tokens, stop_ids
    )
    if not dest_ok:
        return False

    return _greedy_complete_and_eval(
        mt, targ_sample, lcp, include_docstring, max_new_tokens, stop_ids
    )


# ---------------------------------------------------------------------------
# Case 3 — Oracle-based pivot mining
# ---------------------------------------------------------------------------


def case3_pivot_instances(
    mt,
    ds: HeadLocDataset,
    pairs: list[PredicatePair] | None = None,
    include_docstring: bool = False,
    n_oracle: int = 5,
    oracle_temperature: float = 0.5,
    max_completion_tokens: int = 32,
) -> list[PivotInstance]:
    """Mine Case 3 pivot instances using oracle-generated divergence prefixes.

    For each pair:
      1. Query the oracle with both predicates; it returns candidate pivot prefixes
         where the two functions would naturally diverge (``get_oracle_lcps``).
      2. Completion gate: greedily generate from ``return {lcp}`` for both dest and
         target; both must pass all unit tests (confirms the prefix is valid for
         each side).
      3. Divergence gate: ``dest_top_token != target_top_token`` at ``return {lcp}``.

    Args:
        mt: ModelandTokenizer to probe.
        ds: HeadLocDataset providing samples and pairs.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        include_docstring: Passed to ``build_python_prompt``.
        n_oracle: Number of divergence prefixes requested from the oracle per pair.
        oracle_temperature: Sampling temperature for oracle LLM.
        max_completion_tokens: Max tokens for greedy generation in the completion gate.

    Returns:
        List of PivotInstances where both gates pass.
    """
    target_pairs = pairs if pairs is not None else ds.pairs
    instances: list[PivotInstance] = []

    for pair in target_pairs:
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            logger.warning(
                f"Missing sample for pair "
                f"{pair.source_predicate} -> {pair.destination_predicate} — skipping"
            )
            continue

        logger.info(
            f"Case 3: pair {pair.destination_predicate} -> {pair.source_predicate}"
        )

        oracle_lcps = get_oracle_lcps(
            dest_sample, targ_sample, n=n_oracle, temperature=oracle_temperature
        )

        if not oracle_lcps:
            logger.warning("  Oracle returned no prefixes — skipping")
            continue

        for lcp in oracle_lcps:
            # Gate 1: completion — both sides must produce passing code from this prefix
            if not _completion_gate(
                mt,
                dest_sample,
                targ_sample,
                lcp,
                include_docstring=include_docstring,
                max_new_tokens=max_completion_tokens,
            ):
                logger.debug(f"  Completion gate FAILED for lcp={lcp!r}")
                continue

            # Gate 2: divergence — top tokens must differ at the pivot position
            dest_pivot_prompt = build_python_prompt(
                mt, dest_sample, pivot_prefix=lcp, include_docstring=include_docstring
            )
            targ_pivot_prompt = build_python_prompt(
                mt, targ_sample, pivot_prefix=lcp, include_docstring=include_docstring
            )

            try:
                p_dest = get_next_token_probs(mt, dest_pivot_prompt)
                p_targ = get_next_token_probs(mt, targ_pivot_prompt)
            except Exception as e:
                logger.error(f"  Forward pass failed for lcp={lcp!r}: {e}")
                continue

            dest_tok = get_top_token(mt, p_dest)
            targ_tok = get_top_token(mt, p_targ)

            logger.debug(
                f"  lcp={lcp!r} | dest_top={dest_tok!r} | targ_top={targ_tok!r}"
                f" | diverge={dest_tok != targ_tok}"
            )

            if dest_tok != targ_tok:
                instances.append(
                    PivotInstance(
                        source_predicate=pair.source_predicate,
                        destination_predicate=pair.destination_predicate,
                        target_predicate=pair.source_predicate,
                        pivot_prefix=lcp,
                        dest_top_token=dest_tok,
                        target_top_token=targ_tok,
                    )
                )
                logger.info(
                    f"  ✓ New pivot: {pair.destination_predicate} -> {pair.source_predicate}"
                    f" prefix={lcp!r} | dest={dest_tok!r} targ={targ_tok!r}"
                )

    logger.info(f"Case 3: {len(instances)} diverging pivot instances")
    return instances
