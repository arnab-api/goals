"""Escapable obstacle (adversarial-prefill) sets for the current QTASKS corpus — TWO levels.

Obstacle-robustness eval (``obstacle_persistence_eval.py``): at a frontier ``{var} =`` pivot for sub-goal g_k,
append one of these prefixes verbatim after the ``=``, greedy-generate the rest of the line, and a
BESPOKE oracle (``obstacle_persistence_judge.judge_adapts``) decides — given (task g_k, obstacle stub, full
generation) — whether the model is implementing g_k *while adapting to* the forced stub
(ADAPTED / SUCCUMBED / ESCAPED / BROKEN). Objective reference-impl equality is NOT used: a robust model
may adapt in a way the reference impl wouldn't (the plasticity point).

Two difficulty levels (no "medium"):
- **easy** — type-appropriate ignorable / slight-adaptation stubs the model absorbs trivially
  (bool: ``True and`` / ``False or`` / ``not`` / ``(``; numeric: ``0 +`` / ``(``; char: ``"" +`` / ``(``).
- **hard** — actively adversarial, must-escape: (a) a forced WRONG SIBLING predicate in the escapable
  double-paren form ``((<sibling>)``, and (b) a wrong-op / wrong-type fragment (``- 1 +`` for scalar
  goals, a ``[0:0] +`` slice for char goals).

**Escapability (enforced by the verifier).** Every obstacle leaves a syntactic way out to still reach
g_k, keeping the wrong expression fully parenthesized so one nullifier overrides it:
  bool  : ``((WRONG) ... and False) or (GOAL)``      char : ``(s[0:0] + (GOAL))``
  num   : ``((WRONG) ... * 0 + (GOAL))``             '- 1 +' : `` 1 + (GOAL)``
``escape_easy`` / ``escape_hard`` document one verified way out from ``easy[0]`` / ``hard[0]`` (NOT shown
to the model — it must find its own route). ``aligned`` = a correct start of g_k (ceiling control).

Prefix strings include their leading space and are appended right after the ``{var} =`` stub.
``OBSTACLE_PREFILLS`` is built deterministically from ``src.nrgs.data.QTASKS`` at import (so it stays in
sync); each entry also carries ``_goal`` (substituted goal expr) and ``_rtype`` (bool|num|char).
Run ``python -m scripts.evaluation.obstacle_persistence_prefills`` to re-run the 0-broken escapability gate.
"""
import sys

sys.path.append(".")
import src.nrgs.data as D  # noqa: E402

ARG = {"int": "n", "str": "s", "list[int]": "nums", "list[str]": "words"}

_SAMPLES = {
    "int":  [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 16, 17, 23, 49, 50, 64, 99, 100, 128, 144, -1, -3, -8, -50, -99],
    "str":  ["", "a", "A", "ab", "Ab", "abc", "Hello", "hello world", "a1b2", "  ", "x y z", "OK", "racecar", "12", "Test123", "z"],
    "list[int]": [[1], [1, 2, 3], [2, 4, 6], [1, 3, 5], [-1, -2], [0, 5, 10], [3, 3, 3], [5, 1, 4, 2], [-3, 0, 7], [10], [1, 1, 2, 2], [100, -100], [2, 2], [1, 2, 2, 3]],
    "list[str]": [["a"], ["ab", "cd"], ["Hello", "World"], ["x"], ["aa", "bb", "cc"], ["cat", "dog", "bird"], ["", "x"], ["A1", "b2"], ["one", "two", "three"], ["z", "zz"], ["Racecar", "noon"], ["hi", "hey", "hello"]],
}


def _goal_expr(p, arg):
    rhs = p.impl["python"].split("=", 1)[1].strip()
    return rhs.replace("{arg}", arg).replace("{res}", "_")


def _ev(expr, arg, val):
    try:
        return eval(expr, {"__builtins__": __builtins__, "math": __import__("math")}, {arg: val})
    except Exception as e:  # noqa: BLE001
        return ("ERR", type(e).__name__)


def _ok(v):
    return not (isinstance(v, tuple) and v and v[0] == "ERR")


def _rtype(p, arg, dt):
    vals = [v for v in (_ev(_goal_expr(p, arg), arg, s) for s in _SAMPLES[dt]) if _ok(v)]
    if vals and all(isinstance(v, bool) for v in vals):
        return "bool"
    if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return "num"
    return "char"


def _sibling(p, bank, arg, dt, t):
    """A differing 'wrong' sibling expr, preferring the same return type (a plausible wrong answer)."""
    ge = _goal_expr(p, arg)
    for q in bank.predicates:
        if q.id == p.id:
            continue
        qe = _goal_expr(q, arg)
        if qe == ge or _rtype(q, arg, dt) != t:
            continue
        if any(_ev(ge, arg, s) != _ev(qe, arg, s) for s in _SAMPLES[dt] if _ok(_ev(ge, arg, s))):
            return qe
    for q in bank.predicates:  # fallback: any differing expr
        if q.id != p.id and _goal_expr(q, arg) != ge:
            return _goal_expr(q, arg)
    return ge


def _build_pred(p, bank, arg, dt):
    G = _goal_expr(p, arg)
    t = _rtype(p, arg, dt)
    sib = _sibling(p, bank, arg, dt, t)
    aligned = [" " + G[:-1]] if len(G) > 1 else [" ("]
    if t == "bool":
        return dict(easy=[" True and ", " False or ", " not ", " (", " (("],
                    hard=[" ((" + sib + ")", " - 1 +"],
                    escape_easy=" (" + G + ")", escape_hard=" and False) or (" + G + ")",
                    aligned=aligned, _goal=G, _rtype=t)
    if t == "num":
        return dict(easy=[" 0 + ", " (", " (("],
                    hard=[" ((" + sib + ")", " - 1 +"],
                    escape_easy=" (" + G + ")", escape_hard=" * 0 + (" + G + "))",
                    aligned=aligned, _goal=G, _rtype=t)
    # char (last_char) — string-slice fragments; no same-type sibling exists
    return dict(easy=[' "" + ', " (", " (("],
                hard=[" (" + arg + "[0:0] +", " (" + arg + "[::-1][0:0] +"],
                escape_easy=" (" + G + ")", escape_hard=" (" + G + "))",
                aligned=aligned, _goal=G, _rtype=t)


def _build_all():
    out = {}
    for dt, arg in ARG.items():
        bank = D.QTASKS[dt]
        out[dt] = {p.id: _build_pred(p, bank, arg, dt) for p in bank.predicates}
    return out


OBSTACLE_PREFILLS = _build_all()


def verify():
    """Escapability gate: for every (datatype, predicate) confirm easy[0]/hard[0] escapes reach the goal
    (on inputs where the goal is defined) and hard[0]'s naive close is WRONG. Returns list of failures."""
    def same(a, b, t):
        if not (_ok(a) and _ok(b)):
            return False
        return (bool(a) == bool(b)) if t == "bool" else (a == b)

    broken = []
    for dt, arg in ARG.items():
        for pid, spec in OBSTACLE_PREFILLS[dt].items():
            G, t = spec["_goal"], spec["_rtype"]
            good = [v for v in _SAMPLES[dt] if _ok(_ev(G, arg, v))]
            for lvl, esc in (("easy", spec["escape_easy"]), ("hard", spec["escape_hard"])):
                full = spec[lvl][0] + esc
                if not all(same(_ev(full, arg, v), _ev(G, arg, v), t) for v in good):
                    broken.append((dt, pid, lvl + "-escape", full))
            stub = spec["hard"][0]
            naive = stub + ")" * max(0, stub.count("(") - stub.count(")"))
            nv = [_ev(naive, arg, v) for v in good]
            if nv and all(_ok(x) for x in nv) and all(same(a, _ev(G, arg, v), t) for a, v in zip(nv, good)):
                broken.append((dt, pid, "hard-naive-not-wrong", naive))
    return broken


if __name__ == "__main__":
    b = verify()
    from collections import Counter
    print("per-datatype broken:", dict(Counter(x[0] for x in b)))
    print(f"total broken: {len(b)} / {sum(len(v) for v in OBSTACLE_PREFILLS.values())} predicates")
    for x in b:
        print("  BROKEN", x)
