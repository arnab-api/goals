"""Prompt builders and next-token utilities for head-localization pivot mining.

Pivot instances are mined by the oracle pipeline in ``src.goals.head_loc_oracle``
(``scripts/mine_pivot_instances.py``); this module provides the per-language prompt
builders (Python destination, C++ / Haskell / Prolog / Java / Rust sources), the
next-token probability helpers used by its gates, and ``_lcp_tokens``.

Docstring control: pass ``include_docstring=True`` to insert the sample's
``docstring`` field as a Python docstring between the def line and ``return``.
This gives the model a semantic hint and changes which token it predicts — useful
for studying how the docstring interacts with cross-predicate head patching.
"""

from __future__ import annotations

import logging

import torch

from src.goals.functional import apply_chat_template
from src.tokens import prepare_input

logger = logging.getLogger(__name__)

GENERIC_INSTRUCTION  = "Complete the Python function below."
CPP_INSTRUCTION      = "Complete the C++ function below."
HASKELL_INSTRUCTION  = "Complete the Haskell function below."
PROLOG_INSTRUCTION   = "Complete the Prolog predicate below."
JAVA_INSTRUCTION     = "Complete the Java method below."
RUST_INSTRUCTION     = "Complete the Rust function below."


# ---------------------------------------------------------------------------
# Type-mapping helpers
# ---------------------------------------------------------------------------


def _to_camel_case(snake: str) -> str:
    """Convert ``snake_case`` identifier to ``camelCase`` (for Haskell / Java)."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _py_type_to_haskell(py_type: str) -> str:
    """Map a Python type annotation string to its Haskell equivalent."""
    _scalar = {"str": "String", "int": "Int", "bool": "Bool", "float": "Double"}
    if py_type.startswith("list[") and py_type.endswith("]"):
        inner = _py_type_to_haskell(py_type[5:-1])
        return f"[{inner}]"
    return _scalar.get(py_type, py_type.capitalize())


def _py_type_to_java(py_type: str) -> str:
    """Map a Python type annotation string to its Java equivalent (parameter position)."""
    _scalar  = {"str": "String", "int": "int",     "bool": "boolean", "float": "double"}
    _boxed   = {"str": "String", "int": "Integer",  "bool": "Boolean", "float": "Double"}
    if py_type.startswith("list[") and py_type.endswith("]"):
        inner = py_type[5:-1]
        return f"List<{_boxed.get(inner, inner.capitalize())}>"
    return _scalar.get(py_type, py_type)


def _py_type_to_rust(py_type: str) -> str:
    """Map a Python type annotation string to its Rust equivalent (borrowed)."""
    _scalar = {"str": "&str", "int": "i64", "bool": "bool", "float": "f64"}
    if py_type.startswith("list[") and py_type.endswith("]"):
        inner = _py_type_to_rust(py_type[5:-1])
        return f"&[{inner}]"
    return _scalar.get(py_type, py_type)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_cpp_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build the source C++ prompt ending at ``return ``.

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample whose ``cpp_incomplete_code`` ends with
            ``return`` in one-liner brace form (e.g. ``bool isPalindrome(string s) { return``).
        include_docstring: When True and ``sample.docstring`` is non-empty, appends
            the docstring as a C++ inline comment before ``return``.

    Returns:
        Full prompt string ready for tokenization.
    """
    code_at_pivot = sample.cpp_incomplete_code + " "
    if include_docstring and sample.docstring:
        code_at_pivot = f"// {sample.docstring}\n{code_at_pivot}"
    return apply_chat_template(mt, CPP_INSTRUCTION, code_at_pivot)


def build_python_prompt(
    mt,
    sample,
    pivot_prefix: str = "",
    include_docstring: bool = False,
) -> str:
    """Build the full prompt for a predicate sample ending at ``return <pivot_prefix>``.

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample whose ``python_incomplete_code`` ends with
            ``return`` (no trailing space).
        pivot_prefix: Token string appended after ``return ``.
        include_docstring: When True and ``sample.docstring`` is non-empty, insert
            a Python docstring between the def line and ``return``.  This changes
            the model's prediction at the pivot — use consistently across dest and
            target within a single experiment.

    Returns:
        Full prompt string ready for tokenization.
    """
    code_at_pivot = sample.python_incomplete_code + " " + pivot_prefix

    if include_docstring and sample.docstring:
        lines = code_at_pivot.split("\n")
        def_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("def ")), None
        )
        if def_idx is not None:
            indent = "    "
            lines.insert(def_idx + 1, f'{indent}"""{sample.docstring}"""')
            code_at_pivot = "\n".join(lines)

    return apply_chat_template(mt, GENERIC_INSTRUCTION, code_at_pivot)


def build_haskell_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build a Haskell source prompt ending just after the ``=`` definition pivot.

    Generates the type signature and function head from sample metadata:
        isPalindrome :: String -> Bool
        isPalindrome s =·

    The trailing space after ``=`` is the pivot — the model's next token is the
    first token of the expression (e.g. ``s``, ``null``, ``True`` …).

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample providing predicate_name, variable_name,
            variable_type, and optionally docstring.
        include_docstring: When True and ``sample.docstring`` is non-empty,
            prepends a Haddock ``-- |`` comment above the type signature.

    Returns:
        Full prompt string ready for tokenization.
    """
    fn       = _to_camel_case(sample.predicate_name)
    arg      = sample.variable_name
    arg_type = _py_type_to_haskell(sample.variable_type)

    lines = []
    if include_docstring and sample.docstring:
        lines.append(f"-- | {sample.docstring}")
    lines.append(f"{fn} :: {arg_type} -> Bool")
    lines.append(f"{fn} {arg} = ")

    code_at_pivot = "\n".join(lines)
    return apply_chat_template(mt, HASKELL_INSTRUCTION, code_at_pivot)


def build_prolog_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build a Prolog source prompt ending just after the ``:-`` neck pivot.

    Generates the clause head from sample metadata:
        is_palindrome(S) :-·

    Prolog variables must be capitalised, so ``variable_name`` is upper-cased.
    The trailing space after ``:-`` is the pivot — the next token is the first
    subgoal (e.g. ``string_codes``, ``atom_chars`` …).

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample providing predicate_name, variable_name,
            and optionally docstring.
        include_docstring: When True and ``sample.docstring`` is non-empty,
            prepends a ``%`` line comment above the clause.

    Returns:
        Full prompt string ready for tokenization.
    """
    pred = sample.predicate_name          # already snake_case
    arg  = sample.variable_name.capitalize()  # Prolog vars must be upper-case

    lines = []
    if include_docstring and sample.docstring:
        lines.append(f"% {sample.docstring}")
    lines.append(f"{pred}({arg}) :- ")

    code_at_pivot = "\n".join(lines)
    return apply_chat_template(mt, PROLOG_INSTRUCTION, code_at_pivot)


def build_java_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build a Java source prompt ending just after the ``return`` pivot.

    Wraps the method in a ``Solution`` class for full syntactic context:
        public class Solution {
            public static boolean isPalindrome(String s) {
                return·

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample providing predicate_name, variable_name,
            variable_type, and optionally docstring.
        include_docstring: When True and ``sample.docstring`` is non-empty,
            inserts a Javadoc ``/** … */`` comment above the method signature.

    Returns:
        Full prompt string ready for tokenization.
    """
    fn       = _to_camel_case(sample.predicate_name)
    arg      = sample.variable_name
    arg_type = _py_type_to_java(sample.variable_type)

    method_lines = []
    if include_docstring and sample.docstring:
        method_lines.append(f"    /** {sample.docstring} */")
    method_lines.append(f"    public static boolean {fn}({arg_type} {arg}) {{")
    method_lines.append("        return ")

    code_at_pivot = "public class Solution {\n" + "\n".join(method_lines)
    return apply_chat_template(mt, JAVA_INSTRUCTION, code_at_pivot)


def build_rust_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build a Rust source prompt ending just after the ``return`` pivot.

    Generates an ``fn`` definition from sample metadata:
        fn is_palindrome(s: &str) -> bool {
            return·

    Uses borrowed types (``&str``, ``&[i64]``) for parameters.

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample providing predicate_name, variable_name,
            variable_type, and optionally docstring.
        include_docstring: When True and ``sample.docstring`` is non-empty,
            prepends a ``///`` doc-comment above the function.

    Returns:
        Full prompt string ready for tokenization.
    """
    fn       = sample.predicate_name          # keep snake_case (idiomatic Rust)
    arg      = sample.variable_name
    arg_type = _py_type_to_rust(sample.variable_type)

    lines = []
    if include_docstring and sample.docstring:
        lines.append(f"/// {sample.docstring}")
    lines.append(f"fn {fn}({arg}: {arg_type}) -> bool {{")
    lines.append("    return ")

    code_at_pivot = "\n".join(lines)
    return apply_chat_template(mt, RUST_INSTRUCTION, code_at_pivot)


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_next_token_probs(mt, prompt: str) -> torch.Tensor:
    """Return the softmax probability distribution over the full vocabulary for the
    next token after ``prompt``.

    Args:
        mt: ModelandTokenizer instance.
        prompt: Full prompt string ending at the pivot position.

    Returns:
        Float32 CPU tensor of shape ``(vocab_size,)`` with probabilities summing to 1.
    """
    inputs = prepare_input(prompts=[prompt], tokenizer=mt)
    with mt.trace(inputs):
        logits = mt.lm_head.output[0, -1, :].save()
    return torch.softmax(logits.float(), dim=-1).cpu()


@torch.inference_mode()
def get_next_token_probs_batch(mt, prompts: list[str]) -> torch.Tensor:
    """Return softmax probabilities for the next token after each prompt.

    Args:
        mt: ModelandTokenizer instance.
        prompts: List of prompt strings.

    Returns:
        Float32 CPU tensor of shape ``(batch, vocab_size)`` with probabilities
        summing to 1 along dim=-1.  Left-padding ensures ``[:, -1, :]`` is the
        last real token for every sequence.
    """
    inputs = prepare_input(prompts=prompts, tokenizer=mt)
    with mt.trace(inputs):
        logits = mt.lm_head.output[:, -1, :].save()  # (batch, vocab)
    return torch.softmax(logits.float(), dim=-1).cpu()


def get_top_token(mt, probs: torch.Tensor) -> str:
    """Decode the argmax token from a probability distribution.

    Args:
        mt: ModelandTokenizer instance (for its tokenizer).
        probs: Vocabulary probability tensor, shape ``(vocab_size,)``.

    Returns:
        Decoded string for the highest-probability token.
    """
    token_id = int(probs.argmax().item())
    return mt.tokenizer.decode([token_id])


# ---------------------------------------------------------------------------
# Token-level LCP
# ---------------------------------------------------------------------------


def _lcp_tokens(mt, gen_a: str, gen_b: str) -> str:
    """Compute the token-level longest common prefix (LCP) of two generated strings.

    Tokenizes both strings independently, finds the longest prefix where all token
    ids match, then decodes back to a string.

    Args:
        mt: ModelandTokenizer instance (for its tokenizer).
        gen_a: First generated completion string.
        gen_b: Second generated completion string.

    Returns:
        The decoded LCP string, or ``""`` if the first tokens differ.
    """
    tok_a = mt.tokenizer.encode(gen_a, add_special_tokens=False)
    tok_b = mt.tokenizer.encode(gen_b, add_special_tokens=False)

    common_len = 0
    for ta, tb in zip(tok_a, tok_b):
        if ta != tb:
            break
        common_len += 1

    if common_len == 0:
        return ""

    return mt.tokenizer.decode(tok_a[:common_len])
