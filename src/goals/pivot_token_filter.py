"""Conservative lexical filtering of next-token contrasts in Python expressions.

A model token is not necessarily a whole Python token. Compare ``prefix + token``
so that split attributes, names, and literals are interpreted in context. Remove
formatting outside literals and normalize only *known bound variable roles*.
Unknown identifiers, attributes, calls, operators, and literal contents survive.

This is a rejection heuristic, not a semantic-equivalence or correctness oracle.
Keep the existing completion/unit-test and model-token-divergence checks.
"""
from __future__ import annotations

import ast
import io
import keyword
import tokenize
from collections.abc import Collection

# Bump when changing acceptance semantics. NRGS resume records carry this version.
TOKEN_FILTER_VERSION = "python-context-v1"

_IGNORED = frozenset({
    tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT,
    tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT,
})


def _variable_roles(code: str | None, variable_names: Collection[str]) -> dict[str, str]:
    """Find bound roles in one reference implementation, never arbitrary names.

    ``variable_names`` supplies aliases of the single input argument in the
    predicate-mining contract. Complete reference code, when available, separates
    input arguments from distinct local bindings (including comprehension targets).
    Ambiguous shadowed names are left unnormalized rather than guessed.
    """
    roles = {name: "arg:0" for name in variable_names}
    if code is None:
        return roles
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return roles

    # The mining inputs are a single predicate function or expression. Do not
    # claim alpha-equivalence across nested functions, lambdas, or class scopes.
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) > 1 or any(isinstance(n, (ast.Lambda, ast.ClassDef)) for n in ast.walk(tree)):
        return roles
    if functions:
        fn = functions[0]
        args = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
        args += [a for a in (fn.args.vararg, fn.args.kwarg) if a is not None]
        # Full function metadata supersedes fallback aliases from the other side.
        roles = {a.arg: f"arg:{i}" for i, a in enumerate(args)}

    stores = sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)),
        key=lambda n: (n.lineno, n.col_offset),
    )
    counts: dict[str, int] = {}
    for n in stores:
        counts[n.id] = counts.get(n.id, 0) + 1
    local_index = 0
    for n in stores:
        if counts[n.id] != 1 or n.id in roles:
            # A reassigned argument or repeated/shadowed local is not safely
            # normalized by a scope-free map.
            roles.pop(n.id, None)
            continue
        roles[n.id] = f"local:{local_index}"
        local_index += 1
    return roles


def _offset(text: str, position: tuple[int, int]) -> int:
    row, col = position
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[: max(0, row - 1)])) + col


def _signature(text: str, roles: dict[str, str]) -> tuple:
    """Lex an incomplete expression without treating string contents as names.

    Open brackets are normal at a pivot. An unfinished quoted literal is kept as
    a literal tail, including its spaces. On unfamiliar lexical errors, retain
    raw text instead of deleting potentially meaningful information.
    """
    result = []
    previous = None
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            kind, value = item.type, item.string
            if kind in _IGNORED:
                continue
            if kind == tokenize.ERRORTOKEN:
                if value.isspace():
                    continue
                if value in ("'", '"'):
                    result.append(("literal_tail", text[_offset(text, item.start):]))
                    break
                result.append(("error", value))
            elif kind == tokenize.NAME:
                if previous != "." and not keyword.iskeyword(value) and value in roles:
                    result.append(("bound_name", roles[value]))
                else:
                    result.append(("name", value))
            elif kind == tokenize.STRING:
                # Keep literal content intact; quote/escape canonicalization is
                # deliberately outside this small filter's contract.
                result.append(("literal", value))
            else:
                result.append((kind, value))
            previous = value
    except tokenize.TokenError as exc:
        message, position = exc.args
        if "multi-line string" in message:
            result.append(("literal_tail", text[_offset(text, position):]))
        elif "multi-line statement" not in message:
            return (("unparsed", text),)
        # EOF in a multi-line statement is the usual unmatched-bracket prefix;
        # all meaningful tokens preceding EOF have already been yielded.
    except (IndentationError, SyntaxError):
        return (("unparsed", text),)
    return tuple(result)


def tokens_differ_meaningfully(
    tok_dest: str,
    tok_targ: str,
    *,
    prefix: str = "",
    variable_names: Collection[str] = (),
    dest_code: str | None = None,
    targ_code: str | None = None,
) -> bool:
    """Whether a candidate contrast survives contextual lexical normalization.

    ``prefix`` is the exact common expression prefix, not stripped or retokenized.
    ``dest_code``/``targ_code`` can be the already-generated reference completions;
    they supply variable binding roles without further model calls. When absent,
    only explicitly supplied input aliases are normalized. Unknown names remain
    distinct rather than being assumed to be variables.

    This signature remains callable with two positional tokens, but callers must
    pass the prefix to disambiguate token fragments. No claim of semantic validity
    follows solely from returning True.
    """
    if tok_dest == tok_targ:
        return False
    # Formatting-only candidates must not become a positive just because the
    # reference implementations assign different binding roles to the prefix.
    raw_left = _signature(prefix + tok_dest, {})
    raw_right = _signature(prefix + tok_targ, {})
    if raw_left == raw_right:
        return False
    left_roles = _variable_roles(dest_code, variable_names)
    right_roles = _variable_roles(targ_code, variable_names)
    # If the maps already disagree on the shared prefix, they cannot establish
    # that the new difference is merely alpha-renaming. Keep it conservatively.
    if _signature(prefix, left_roles) != _signature(prefix, right_roles):
        return True
    return _signature(prefix + tok_dest, left_roles) != _signature(prefix + tok_targ, right_roles)
