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

import builtins
import keyword
import logging
import re

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

_ORACLE_PAIR_PROMPT_TEMPLATE = '''\
You are given two Python predicate functions that both end at `return`.

Find up to {n} shared prefixes P (what immediately follows `return `) such that:
  1. Both predicates can be validly completed starting from that prefix.
  2. The very next token after P is different between the two predicates.
  3. That diverging token directly encodes the functional difference between
     the two predicates — it must be the specific operator, method name, or
     index that makes A and B semantically distinct.

     GOOD divergences: an operator like `<` vs `>`, an index like `0` vs `-1`,
     a method name like `.upper` vs `.lower`.  These tokens, on their own, tell
     you which predicate is being implemented.
     BAD divergences: a variable name like `c` vs `i` (naming convention, not
     semantics), or a wrapper like `not not` that adds no new information.

Reason freely. Each time you confirm a valid prefix, emit it immediately in this
XML tag before continuing to the next candidate:

<pivot>
PREFIX: <prefix text>
A: <how function A completes it>
B: <how function B completes it>
DIVERGES: A=<first new token for A after P>  B=<first new token for B after P>
</pivot>

Only emit a <pivot> block once you have verified:
  - A starts with PREFIX, B starts with PREFIX
  - The first token after PREFIX differs between A and B
  - The diverging token is semantically meaningful (not just a variable name)

=== Example 1 (index-level divergence) ===

Function A:
def starts_with_vowel(s: str) -> bool:
    """Return True if the first character of s is a vowel."""
    return

Function B:
def ends_with_vowel(s: str) -> bool:
    """Return True if the last character of s is a vowel."""
    return

Consider `s[0].lower() in 'aeiou'` vs `s[-1].lower() in 'aeiou'` — LCP is `s[`,
next token `0` vs `-` encodes first vs last character. Valid.

<pivot>
PREFIX: s[
A: s[0].lower() in 'aeiou'
B: s[-1].lower() in 'aeiou'
DIVERGES: A='0'  B='-'
</pivot>

Wrap with `not`: same divergence applies.

<pivot>
PREFIX: not s[
A: not s[0].lower() in 'aeiou'
B: not s[-1].lower() in 'aeiou'
DIVERGES: A='0'  B='-'
</pivot>

=== Example 2 (operator-level divergence) ===

Function A:
def is_strictly_increasing(lst: list[int]) -> bool:
    """Return True if each element is strictly greater than the previous."""
    return

Function B:
def is_strictly_decreasing(lst: list[int]) -> bool:
    """Return True if each element is strictly less than the previous."""
    return

Consider `all(lst[i] < lst[i+1] ...)` vs `all(lst[i] > lst[i+1] ...)` — LCP is
`all(lst[i]`, next token `<` vs `>` encodes increasing vs decreasing. Valid.

<pivot>
PREFIX: all(lst[i]
A: all(lst[i] < lst[i+1] for i in range(len(lst)-1))
B: all(lst[i] > lst[i+1] for i in range(len(lst)-1))
DIVERGES: A='<'  B='>'
</pivot>

=== Your turn ===

Function A (destination):
{dest_code}

Function B (target):
{targ_code}
'''


def _code_with_docstring(sample: SinglePredicateSample) -> str:
    """Return ``python_incomplete_code`` with docstring inserted after the def line."""
    code = sample.python_incomplete_code
    if not sample.docstring:
        return code
    lines = code.split("\n")
    def_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("def ")), None
    )
    if def_idx is not None:
        lines.insert(def_idx + 1, f'    """{sample.docstring}"""')
    return "\n".join(lines)


def build_oracle_pair_prompt(
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n: int = 5,
    include_docstring: bool = True,
) -> str:
    """Build the pair-based oracle prompt asking for divergence prefixes directly.

    Args:
        dest_sample: Destination predicate.
        targ_sample: Target predicate.
        n: Maximum number of prefixes to request.
        include_docstring: When True (default), insert each sample's docstring
            after the ``def`` line so the oracle has semantic context.

    Returns:
        Prompt string ready for ``ask_claude``.
    """
    dest_code = (
        _code_with_docstring(dest_sample)
        if include_docstring
        else dest_sample.python_incomplete_code
    )
    targ_code = (
        _code_with_docstring(targ_sample)
        if include_docstring
        else targ_sample.python_incomplete_code
    )
    return _ORACLE_PAIR_PROMPT_TEMPLATE.format(
        dest_code=dest_code,
        targ_code=targ_code,
        n=n,
    )


def parse_oracle_pair_response(response: str) -> list[str]:
    """Extract prefixes from ``<pivot>`` blocks in an oracle pair response.

    Only content inside ``<pivot>...</pivot>`` tags is parsed — the oracle's
    free-form reasoning outside these tags is ignored.  Each block may contain
    PREFIX, A, B, and DIVERGES lines; only PREFIX is extracted here (the gates
    handle the actual verification).

    Args:
        response: Raw oracle LLM response.

    Returns:
        List of prefix strings (deduplicated, order-preserved).
    """

    def _strip_delimiters(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] in ('"', "'", "`") and s[0] == s[-1]:
            return s[1:-1]
        return s

    prefixes: list[str] = []
    for block in re.findall(r"<pivot>(.*?)</pivot>", response, re.DOTALL):
        current_prefix: str | None = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("PREFIX:"):
                current_prefix = _strip_delimiters(line[len("PREFIX:") :].strip())
            elif line.startswith("DIVERGES:") and current_prefix:
                logger.debug(
                    f"  Oracle DIVERGES hint for prefix={current_prefix!r}: {line}"
                )
        if current_prefix:
            prefixes.append(current_prefix)

    seen: set[str] = set()
    return [p for p in prefixes if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]


def get_oracle_lcps(
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n: int = 5,
    temperature: float = 0.5,
    include_docstring: bool = True,
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
        include_docstring: When True (default), insert docstrings into the prompt
            so the oracle has semantic context about each function.

    Returns:
        List of candidate pivot prefix strings (deduplicated, order-preserved).
    """
    prompt = build_oracle_pair_prompt(
        dest_sample, targ_sample, n=n, include_docstring=include_docstring
    )
    response = ask_claude(prompt, temperature=temperature, max_tokens=4096)
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
) -> bool:
    """Generate greedily from ``return {lcp}`` and check unit tests for ``sample``."""
    pivot_prompt = build_python_prompt(
        mt, sample, pivot_prefix=lcp, include_docstring=include_docstring
    )
    # logger.debug(f"'''{pivot_prompt=}'''")
    inputs = prepare_input(prompts=[pivot_prompt], tokenizer=mt)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = mt._model(input_ids=input_ids, attention_mask=attention_mask)
            next_tok = int(out.logits[0, -1, :].argmax())
            generated_ids.append(next_tok)
            # if next_tok in stop_ids:
            #     break
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

    # logger.debug(f"Generated text for lcp={lcp!r}: {generated_text!r}")

    full_code = sample.python_incomplete_code + " " + lcp + generated_text
    # logger.debug(f"Full code to evaluate:\n'''{full_code}'''")
    result = evaluate_single_code(
        full_code,
        sample.unit_tests,
        function_name=sample.function_name or sample.predicate_name,
    )
    verdict_msg = result.verdict
    if result.verdict == "FAILED":
        verdict_msg += f" (failed tests: {[str(unit_test) for unit_test in result.unit_test_results if unit_test.passed is False]})"
    logger.debug(
        f"  eval [{sample.predicate_name}] lcp={lcp!r}"
        f"  generated={generated_text!r}  verdict={verdict_msg}"
    )
    return result.verdict == "PASSED"


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
    dest_ok = _greedy_complete_and_eval(
        mt, dest_sample, lcp, include_docstring, max_new_tokens
    )
    if not dest_ok:
        return False

    return _greedy_complete_and_eval(
        mt, targ_sample, lcp, include_docstring, max_new_tokens
    )


# ---------------------------------------------------------------------------
# Case 3 — Oracle-based pivot mining
# ---------------------------------------------------------------------------


def case3_pivot_instances(
    mt,
    ds: HeadLocDataset,
    pairs: list[PredicatePair] | None = None,
    include_docstring: bool = False,
    oracle_include_docstring: bool = True,
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
        include_docstring: Whether to insert docstrings into the LM prompts used
            for the completion and divergence gates.
        oracle_include_docstring: Whether to insert docstrings into the oracle
            prompt so the oracle LLM has semantic context about each function.
        n_oracle: Number of divergence prefixes requested from the oracle per pair.
        oracle_temperature: Sampling temperature for oracle LLM.
        max_completion_tokens: Max tokens for greedy generation in the completion gate.

    Returns:
        List of (pair, PivotInstance) tuples where both gates pass.
    """
    target_pairs = pairs if pairs is not None else ds.pairs
    instances: list[tuple[PredicatePair, PivotInstance]] = []

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
            dest_sample,
            targ_sample,
            n=n_oracle,
            temperature=oracle_temperature,
            include_docstring=oracle_include_docstring,
        )

        if not oracle_lcps:
            logger.warning("  Oracle returned no prefixes — skipping")
            continue

        for lcp in oracle_lcps:
            pi = _validate_candidate(
                mt,
                dest_sample,
                targ_sample,
                lcp,
                include_docstring=include_docstring,
                max_completion_tokens=max_completion_tokens,
            )
            if pi:
                instances.append((pair, pi))
                logger.info(
                    f"  ✓ New pivot: {pair.destination_predicate} -> {pair.source_predicate}"
                    f" prefix={lcp!r} | dest={pi.dest_top_token!r} targ={pi.target_top_token!r}"
                )

    logger.info(f"Case 3: {len(instances)} diverging pivot instances")
    return instances


# ---------------------------------------------------------------------------
# Case 3b — Per-predicate oracle + mechanical LCP pivot mining
# ---------------------------------------------------------------------------


def _mine_case3b_raw(
    mt,
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    n_implementations: int = 8,
    oracle_temperature: float = 0.5,
    min_prefix_len: int = 3,
) -> list[str]:
    """Return deduplicated candidate LCPs via per-predicate oracle + mechanical cross-product.

    Queries the oracle independently for dest and target implementations, then
    computes token-level LCPs across all (dest_impl, targ_impl) pairs. Filters by
    minimum prefix length, prefix-relationship check, and ``_tokens_differ_meaningfully``
    on the diverging token pair.

    Args:
        mt: ModelandTokenizer (tokenizer used for LCP computation).
        dest_sample: Destination predicate.
        targ_sample: Target predicate.
        n_implementations: Number of oracle implementations to request per predicate.
        oracle_temperature: Sampling temperature for oracle LLM.
        min_prefix_len: Minimum character length for an LCP to be considered.

    Returns:
        Deduplicated list of candidate LCP strings (order-preserved).
    """
    from src.goals.head_loc_pivot import _lcp_tokens

    dest_impls = get_oracle_implementations(
        dest_sample, n=n_implementations, temperature=oracle_temperature
    )
    targ_impls = get_oracle_implementations(
        targ_sample, n=n_implementations, temperature=oracle_temperature
    )

    if not dest_impls or not targ_impls:
        logger.warning(
            f"  Oracle returned no implementations for "
            f"{dest_sample.predicate_name!r} or {targ_sample.predicate_name!r}"
        )
        return []

    candidate_lcps: dict[str, None] = {}  # ordered set

    for impl_dest in dest_impls:
        for impl_targ in targ_impls:
            lcp = _lcp_tokens(mt, impl_dest, impl_targ)
            if len(lcp) < min_prefix_len:
                continue

            tok_lcp = mt.tokenizer.encode(lcp, add_special_tokens=False)
            tok_dest = mt.tokenizer.encode(impl_dest, add_special_tokens=False)
            tok_targ = mt.tokenizer.encode(impl_targ, add_special_tokens=False)
            n = len(tok_lcp)
            if n >= len(tok_dest) or n >= len(tok_targ):
                continue  # one string is a prefix of the other

            div_tok_dest = mt.tokenizer.decode([tok_dest[n]])
            div_tok_targ = mt.tokenizer.decode([tok_targ[n]])

            if not _tokens_differ_meaningfully(div_tok_dest, div_tok_targ):
                logger.debug(
                    f"  Skipping lcp={lcp!r}: diverging tokens differ only in naming"
                    f" ({div_tok_dest!r} vs {div_tok_targ!r})"
                )
                continue

            if lcp not in candidate_lcps:
                candidate_lcps[lcp] = None
                logger.debug(
                    f"  Candidate lcp={lcp!r} | div: {div_tok_dest!r} / {div_tok_targ!r}"
                )

    logger.info(
        f"  {len(candidate_lcps)} candidate LCPs for "
        f"{dest_sample.predicate_name!r} -> {targ_sample.predicate_name!r}"
    )
    return list(candidate_lcps)


def _validate_candidate(
    mt,
    dest_sample: SinglePredicateSample,
    targ_sample: SinglePredicateSample,
    lcp: str,
    include_docstring: bool = False,
    max_completion_tokens: int = 32,
) -> "PivotInstance | None":
    """Run completion gate + divergence gate for a candidate LCP.

    Returns a ``PivotInstance`` if both gates pass, otherwise ``None``.

    Args:
        mt: ModelandTokenizer to probe.
        dest_sample: Destination predicate sample.
        targ_sample: Target predicate sample.
        lcp: Candidate pivot prefix string.
        include_docstring: Passed to prompt builders and ``_completion_gate``.
        max_completion_tokens: Max tokens for greedy generation in the completion gate.

    Returns:
        ``PivotInstance`` if both gates pass, else ``None``.
    """
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
        return None

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
        return None

    dest_tok = get_top_token(mt, p_dest)
    targ_tok = get_top_token(mt, p_targ)

    logger.debug(
        f"  lcp={lcp!r} | dest_top={dest_tok!r} | targ_top={targ_tok!r}"
        f" | diverge={dest_tok != targ_tok}"
    )

    if dest_tok != targ_tok:
        if not _tokens_differ_meaningfully(dest_tok, targ_tok):
            logger.debug(
                f"  Skipping lcp={lcp!r}: diverging tokens differ only in naming"
                f" ({dest_tok!r} vs {targ_tok!r})"
            )
            return None
        return PivotInstance(
            pivot_prefix=lcp, dest_top_token=dest_tok, target_top_token=targ_tok
        )
    return None


_KEYWORDS_AND_BUILTINS: frozenset[str] = frozenset(
    keyword.kwlist + dir(builtins)
)
_IDENTIFIER_RE = re.compile(r"[a-zA-Z_]\w*")


def _normalize_token(token: str) -> str:
    """Replace non-keyword, non-builtin identifiers with a placeholder.

    ``len``, ``not``, ``True`` etc. are preserved; bare variable names like
    ``s``, ``c``, ``result`` are replaced with ``VAR``.
    """
    return _IDENTIFIER_RE.sub(
        lambda m: m.group(0) if m.group(0) in _KEYWORDS_AND_BUILTINS else "VAR",
        token,
    )


def _tokens_differ_meaningfully(tok_dest: str, tok_targ: str) -> bool:
    """True if the two diverging tokens encode a real semantic difference.

    Normalizes both tokens by replacing variable-name identifiers with ``VAR``,
    then checks whether they still differ.  Tokens that differ only in variable
    naming convention (e.g. ``(s`` vs ``(c``, ``[item`` vs ``[elem``) become
    identical after normalization and are rejected.

    Examples:
        ``(s``  vs ``(c``   → both → ``(VAR``        → False  (naming only)
        ``<``   vs ``>``    → unchanged               → True   (operator)
        ``[-``  vs ``[``    → unchanged               → True   (index structure)
        ``not`` vs ``len(`` → ``not`` vs ``len(``     → True   (keyword vs builtin)
        ``(len(`` vs ``(s`` → ``(len(`` vs ``(VAR``  → True   (meaningful)
    """
    return _normalize_token(tok_dest) != _normalize_token(tok_targ)


def case3b_pivot_instances(
    mt,
    ds: HeadLocDataset,
    pairs: list[PredicatePair] | None = None,
    include_docstring: bool = False,
    n_implementations: int = 8,
    oracle_temperature: float = 0.5,
    min_prefix_len: int = 3,
    max_completion_tokens: int = 32,
) -> list[PivotInstance]:
    """Mine Case 3b pivot instances via per-predicate oracle + mechanical LCP.

    Complementary to Case 3 (pair-based oracle).  For each pair:
      1. Call ``get_oracle_implementations`` separately for dest and target.
      2. Cross-product all (impl_dest, impl_targ) pairs and compute token-level
         LCPs via ``_lcp_tokens``.
      3. Filter: LCP must be at least ``min_prefix_len`` characters.
      4. Extract diverging tokens at position ``len(tokenized_lcp)``; both must
         pass ``_tokens_differ_meaningfully`` (rejects naming-only divergences).
      5. Deduplicate LCPs across cross-product pairs.
      6. Completion gate + divergence gate (same as Case 3).
      7. Emit ``PivotInstance`` for passing pairs.

    Args:
        mt: ModelandTokenizer to probe.
        ds: HeadLocDataset providing samples and pairs.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        include_docstring: Whether to insert docstrings into LM prompts for
            the completion and divergence gates.
        n_implementations: Number of oracle implementations requested per
            predicate.
        oracle_temperature: Sampling temperature for the oracle LLM.
        min_prefix_len: Minimum character length for an LCP to be considered.
        max_completion_tokens: Max tokens for greedy generation in the
            completion gate.

    Returns:
        List of (pair, PivotInstance) tuples where both gates pass.
    """
    target_pairs = pairs if pairs is not None else ds.pairs
    instances: list[tuple[PredicatePair, PivotInstance]] = []

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
            f"Case 3b: pair {pair.destination_predicate} -> {pair.source_predicate}"
        )

        candidate_lcps = _mine_case3b_raw(
            mt,
            dest_sample,
            targ_sample,
            n_implementations=n_implementations,
            oracle_temperature=oracle_temperature,
            min_prefix_len=min_prefix_len,
        )

        for lcp in candidate_lcps:
            pi = _validate_candidate(
                mt,
                dest_sample,
                targ_sample,
                lcp,
                include_docstring=include_docstring,
                max_completion_tokens=max_completion_tokens,
            )
            if pi:
                instances.append((pair, pi))
                logger.info(
                    f"  ✓ New pivot (3b): {pair.destination_predicate} -> {pair.source_predicate}"
                    f" prefix={lcp!r} | dest={pi.dest_top_token!r} targ={pi.target_top_token!r}"
                )

    logger.info(f"Case 3b: {len(instances)} diverging pivot instances")
    return instances


# ---------------------------------------------------------------------------
# Semantic-preserving augmentation of validated pivot instances
# ---------------------------------------------------------------------------

#: Wrappers that preserve boolean semantics: ``return W + P`` is semantically
#: equivalent to ``return P`` for any boolean expression P.
AUGMENTATION_WRAPPERS: list[str] = [
    "bool(",
    "int(",
    "not not ",
    "True and ",
    "False or ",
]


def augment_pivot_instances(
    mt,
    instances: list[tuple[PredicatePair, PivotInstance]],
    ds: HeadLocDataset,
    wrappers: list[str] | None = None,
    include_docstring: bool = False,
    max_completion_tokens: int = 32,
) -> list[tuple[PredicatePair, PivotInstance]]:
    """Augment validated pivot instances with semantic-preserving prefix wrappers.

    For each validated ``(pair, PivotInstance)`` and each wrapper W, the augmented prefix
    ``W + pivot_prefix`` is passed through both gates:

    - **Completion gate**: greedy generation from ``return W+P`` must produce
      passing code for both dest and target.  (Re-checked because the wrapper
      changes the LM context and the model may complete differently.)
    - **Divergence gate**: dest and target top tokens at ``return W+P`` must still
      differ.  Nearly always holds by construction but checked to obtain the actual
      token values and catch edge cases.

    Args:
        mt: ModelandTokenizer (frozen).
        instances: Already-validated ``(pair, PivotInstance)`` tuples (output of
            ``case3_pivot_instances`` or ``get_oracle_pivot_instances`` Stage 2).
        ds: HeadLocDataset for looking up dest/target samples.
        wrappers: Prefix wrappers to try.  Defaults to ``AUGMENTATION_WRAPPERS``.
        include_docstring: Passed to prompt builders and ``_completion_gate``.
        max_completion_tokens: Max tokens for greedy generation.

    Returns:
        New ``(pair, PivotInstance)`` tuples for each (instance, wrapper) pair that
        passes both gates.  The original instances are not included.
    """
    if wrappers is None:
        wrappers = AUGMENTATION_WRAPPERS

    augmented: list[tuple[PredicatePair, PivotInstance]] = []

    for pair, pi in instances:
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            logger.warning(
                f"Missing sample for {pair.destination_predicate}/{pair.source_predicate}"
                " — skipping augmentation"
            )
            continue

        for wrapper in wrappers:
            aug_prefix = wrapper + pi.pivot_prefix

            # Gate 1: completion
            if not _completion_gate(
                mt,
                dest_sample,
                targ_sample,
                aug_prefix,
                include_docstring=include_docstring,
                max_new_tokens=max_completion_tokens,
            ):
                logger.debug(
                    f"  Augmentation completion FAILED: wrapper={wrapper!r}"
                    f" base={pi.pivot_prefix!r}"
                )
                continue

            # Gate 2: divergence (also retrieves the token values)
            dest_prompt = build_python_prompt(
                mt,
                dest_sample,
                pivot_prefix=aug_prefix,
                include_docstring=include_docstring,
            )
            targ_prompt = build_python_prompt(
                mt,
                targ_sample,
                pivot_prefix=aug_prefix,
                include_docstring=include_docstring,
            )
            try:
                p_dest = get_next_token_probs(mt, dest_prompt)
                p_targ = get_next_token_probs(mt, targ_prompt)
            except Exception as e:
                logger.error(
                    f"  Forward pass failed for aug_prefix={aug_prefix!r}: {e}"
                )
                continue

            dest_tok = get_top_token(mt, p_dest)
            targ_tok = get_top_token(mt, p_targ)

            if dest_tok != targ_tok:
                augmented.append(
                    (
                        pair,
                        PivotInstance(
                            pivot_prefix=aug_prefix,
                            dest_top_token=dest_tok,
                            target_top_token=targ_tok,
                        ),
                    )
                )
                logger.info(
                    f"  ✓ Augmented pivot: {pair.destination_predicate} ->"
                    f" {pair.source_predicate} prefix={aug_prefix!r}"
                    f" | dest={dest_tok!r} targ={targ_tok!r}"
                )
            else:
                logger.debug(
                    f"  Augmentation divergence FAILED: wrapper={wrapper!r}"
                    f" base={pi.pivot_prefix!r} both predict {dest_tok!r}"
                )

    logger.info(
        f"Augmentation: {len(augmented)} new instances from"
        f" {len(instances)} base × {len(wrappers)} wrappers"
    )
    return augmented


# ---------------------------------------------------------------------------
# Main pipeline wrapper
# ---------------------------------------------------------------------------


def get_oracle_pivot_instances(
    mt,
    ds: HeadLocDataset,
    pairs: list[PredicatePair] | None = None,
    include_case3: bool = True,
    include_case3b: bool = True,
    include_augmentation: bool = True,
    wrappers: list[str] | None = None,
    include_docstring: bool = True,
    oracle_include_docstring: bool = True,
    # Case 3 params
    n_oracle: int = 5,
    oracle_temperature: float = 0.5,
    max_completion_tokens: int = 32,
    # Case 3b params
    n_implementations: int = 8,
    min_prefix_len: int = 3,
    # I/O
    save: bool = True,
) -> list[PivotInstance]:
    """Mine, validate, augment, and save oracle pivot instances.

    Four named stages:
      1. **Mine** — collect raw candidate LCPs from Case 3 (pair oracle) and/or
         Case 3b (per-predicate oracle + mechanical cross-product) for each pair,
         then deduplicate by string equality.
      2. **Gate Check 1** — completion gate + divergence gate via
         ``_validate_candidate``; passing candidates become ``PivotInstance`` objects.
      3. **Augment** — ``augment_pivot_instances`` wraps validated instances with
         semantic-preserving prefixes (Gate Check 2 is internal to that function).
      4. **Combine + save** — deduplicate base + augmented instances by
         (destination, target, prefix) and optionally save via
         ``ds.save_pivot_instances``.

    Args:
        mt: ModelandTokenizer to probe.
        ds: HeadLocDataset providing samples, pairs, and save path.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        include_case3: Include Case 3 (pair-based oracle) mining.
        include_case3b: Include Case 3b (per-predicate oracle + cross-product) mining.
        include_augmentation: Apply wrapper augmentation to validated instances.
        wrappers: Wrappers passed to ``augment_pivot_instances``.  Defaults to
            ``AUGMENTATION_WRAPPERS``.
        include_docstring: Whether to insert docstrings into LM prompts (both gates
            and augmentation).
        oracle_include_docstring: Whether to insert docstrings into the Case 3 oracle
            pair prompt.
        n_oracle: Number of divergence prefixes to request from the oracle per pair
            (Case 3).
        oracle_temperature: Sampling temperature for oracle LLM.
        max_completion_tokens: Max tokens for greedy generation in the completion gate.
        n_implementations: Number of oracle implementations per predicate (Case 3b).
        min_prefix_len: Minimum character length for an LCP to be considered (Case 3b).
        save: If True, save combined instances via ``ds.save_pivot_instances``.

    Returns:
        Deduplicated list of all validated (base + augmented) ``PivotInstance`` objects.
    """
    target_pairs = pairs if pairs is not None else ds.pairs

    # ------------------------------------------------------------------
    # Stage 1 — Mine raw candidates
    # ------------------------------------------------------------------
    candidate_lcps: dict[str, list[str]] = {}  # pair key -> deduped LCPs

    for pair in target_pairs:
        pair_key = f"{pair.destination_predicate}->{pair.source_predicate}"
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            logger.warning(
                f"Missing sample for pair "
                f"{pair.source_predicate} -> {pair.destination_predicate} — skipping"
            )
            candidate_lcps[pair_key] = []
            continue

        raw: list[str] = []

        if include_case3:
            case3_lcps = get_oracle_lcps(
                dest_sample,
                targ_sample,
                n=n_oracle,
                temperature=oracle_temperature,
                include_docstring=oracle_include_docstring,
            )
            raw.extend(case3_lcps)

        if include_case3b:
            case3b_lcps = _mine_case3b_raw(
                mt,
                dest_sample,
                targ_sample,
                n_implementations=n_implementations,
                oracle_temperature=oracle_temperature,
                min_prefix_len=min_prefix_len,
            )
            raw.extend(case3b_lcps)

        # Dedup by string equality, preserving order
        seen: dict[str, None] = {}
        for lcp in raw:
            if lcp not in seen:
                seen[lcp] = None
        candidate_lcps[pair_key] = list(seen)
        logger.info(
            f"Stage 1 [{pair_key}]: {len(raw)} raw → {len(seen)} deduped candidates"
        )

    # ------------------------------------------------------------------
    # Stage 2 — Gate Check 1 (completion + divergence)
    # ------------------------------------------------------------------
    valid_instances: list[tuple[PredicatePair, PivotInstance]] = []

    for pair in target_pairs:
        pair_key = f"{pair.destination_predicate}->{pair.source_predicate}"
        lcps = candidate_lcps.get(pair_key, [])
        if not lcps:
            continue

        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            continue  # already warned in Stage 1

        for lcp in lcps:
            pi = _validate_candidate(
                mt,
                dest_sample,
                targ_sample,
                lcp,
                include_docstring=include_docstring,
                max_completion_tokens=max_completion_tokens,
            )
            if pi:
                valid_instances.append((pair, pi))
                logger.info(
                    f"  ✓ Validated: {pair_key} prefix={lcp!r}"
                    f" | dest={pi.dest_top_token!r} targ={pi.target_top_token!r}"
                )

    logger.info(f"Stage 2: {len(valid_instances)} validated pivot instances")

    # ------------------------------------------------------------------
    # Stage 3 — Augmentation (Gate Check 2 is internal)
    # ------------------------------------------------------------------
    aug_instances: list[tuple[PredicatePair, PivotInstance]] = []
    if include_augmentation and valid_instances:
        aug_instances = augment_pivot_instances(
            mt,
            valid_instances,
            ds,
            wrappers=wrappers,
            include_docstring=include_docstring,
            max_completion_tokens=max_completion_tokens,
        )
        logger.info(f"Stage 3: {len(aug_instances)} augmented pivot instances")

    # ------------------------------------------------------------------
    # Stage 4 — Combine, dedup, insert into pairs, save, return
    # ------------------------------------------------------------------
    combined_raw = valid_instances + aug_instances

    # Dedup by (destination_predicate, source_predicate, pivot_prefix)
    seen_keys: dict[tuple[str, str, str], None] = {}
    combined: list[tuple[PredicatePair, PivotInstance]] = []
    for pair, pi in combined_raw:
        key = (pair.destination_predicate, pair.source_predicate, pi.pivot_prefix)
        if key not in seen_keys:
            seen_keys[key] = None
            combined.append((pair, pi))

    logger.info(
        f"Stage 4: {len(combined_raw)} total → {len(combined)} after dedup"
        f" ({len(valid_instances)} base + {len(aug_instances)} augmented)"
    )

    # Insert into the pair objects and persist
    if save:
        # Clear existing pivot_instances for affected pairs, then repopulate
        affected_pairs = {id(pair): pair for pair, _ in combined}
        for pair in affected_pairs.values():
            pair.pivot_instances = []
        for pair, pi in combined:
            pair.pivot_instances.append(pi)
        model_name = mt.name.split("/")[-1]
        ds.save_pairs(model_name)
        logger.info(
            f"Saved {len(combined)} pivot instances into pairs/{model_name}.json"
        )

    return combined


# ---------------------------------------------------------------------------
# Global pair generation — oracle-verified semantic compatibility
# ---------------------------------------------------------------------------

_PAIR_ORACLE_PROMPT = """\
You are given two Python predicate functions that take the same input type.
Determine whether they are semantically distinct enough to be useful contrastive
pairs in a model probing experiment.

A GOOD pair (answer YES) satisfies all of:
  - They check for clearly different properties of their input.
  - A programmer writing one would make clearly different token-level choices
    than writing the other (e.g. different operators, method names, or indices).
  - They are not trivially related by negation or a single threshold change
    (e.g. is_positive vs is_negative is too easy — just `not`).

A BAD pair (answer NO) if any of:
  - They are near-identical predicates differing only in a single constant
    (e.g. is_divisible_by_two vs is_divisible_by_three — same structure,
    just swap the literal; the pivot is always the same position).
  - One is a strictly stronger/weaker version of the other and shares
    almost all implementation tokens (e.g. is_sorted vs is_strictly_sorted).
  - Their most natural implementations share such a long common prefix that
    no meaningful divergence point exists before the end of the expression.

Function A:
{code_a}

Function B:
{code_b}

Respond with exactly one line: VERDICT: YES or VERDICT: NO, followed by a
brief (one sentence) reason.
"""


def _oracle_verify_pair(
    sample_a: SinglePredicateSample,
    sample_b: SinglePredicateSample,
    temperature: float = 0.0,
) -> tuple[bool, str]:
    """Ask the oracle whether (A, B) is a semantically useful contrastive pair.

    Returns (verified: bool, reason: str).
    """
    prompt = _PAIR_ORACLE_PROMPT.format(
        code_a=_code_with_docstring(sample_a),
        code_b=_code_with_docstring(sample_b),
    )
    response = ask_claude(prompt, temperature=temperature, max_tokens=128)
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    verified = "VERDICT: YES" in first_line.upper()
    reason = response.strip()
    logger.debug(
        f"Oracle pair ({sample_a.predicate_name}, {sample_b.predicate_name}): "
        f"verified={verified}  response={reason!r}"
    )
    return verified, reason


def generate_global_pairs(
    ds: HeadLocDataset,
    dir_path=None,
    oracle_temperature: float = 0.0,
    save: bool = True,
) -> list[PredicatePair]:
    """Oracle-verify all same-type predicate combinations and save to ``global_pairs.json``.

    For each unordered same-type pair (A, B), makes one oracle call to determine
    if the pair is semantically distinct enough for contrastive probing.  If
    verified, both directed pairs (A→B) and (B→A) are added.

    This is a **one-time, model-agnostic** operation.  Re-run only when new
    predicates are added.  Results are saved to ``global_pairs.json`` and can
    be used to bootstrap any model-specific ``pairs/{model_name}.json`` via
    ``HeadLocDataset.initialize_model_pairs()``.

    Args:
        ds: HeadLocDataset providing the predicate samples.
        dir_path: Root head_loc data directory. Defaults to HEAD_LOC_DATA_DIR.
        oracle_temperature: Sampling temperature for the oracle LLM (0.0 = greedy).
        save: If True, save results to ``global_pairs.json`` and update ``ds.pairs``.

    Returns:
        List of oracle-verified ``PredicatePair`` objects (both directions).
    """
    from collections import defaultdict
    from itertools import combinations

    from src.goals.head_loc_data import HEAD_LOC_DATA_DIR

    if dir_path is None:
        dir_path = HEAD_LOC_DATA_DIR

    # Group by variable_type
    by_type: dict[str, list] = defaultdict(list)
    for sample in ds.samples:
        by_type[sample.variable_type].append(sample)

    verified_pairs: list[PredicatePair] = []
    total_calls = sum(len(v) * (len(v) - 1) // 2 for v in by_type.values())
    call_num = 0

    for var_type, samples in by_type.items():
        logger.info(
            f"Type '{var_type}': {len(samples)} predicates → "
            f"{len(samples) * (len(samples) - 1) // 2} oracle calls"
        )
        for sample_a, sample_b in combinations(samples, 2):
            call_num += 1
            logger.info(
                f"  [{call_num}/{total_calls}] "
                f"({sample_a.predicate_name}, {sample_b.predicate_name})"
            )
            verified, reason = _oracle_verify_pair(
                sample_a, sample_b, temperature=oracle_temperature
            )
            logger.info(
                f"    → {'YES' if verified else 'NO'}  {reason.splitlines()[0]}"
            )

            if verified:
                for src, dest in [(sample_a, sample_b), (sample_b, sample_a)]:
                    verified_pairs.append(
                        PredicatePair(
                            source_predicate=src.predicate_name,
                            destination_predicate=dest.predicate_name,
                            variable_type=var_type,
                            oracle_verified=True,
                        )
                    )

    logger.info(
        f"generate_global_pairs: {call_num} oracle calls → "
        f"{len(verified_pairs) // 2} verified pairs ({len(verified_pairs)} directed)"
    )

    if save:
        ds.pairs = verified_pairs
        ds.save_global_pairs(dir_path)

    return verified_pairs
