"""Oracle-generated verifier dataset (wrong_task + incomplete), validated.

Asks Claude for diverse self-contained Python tasks, each with a correct impl, a
`wrong_task` impl (runs but does something unrelated), and an `incomplete` impl
(on-task, parses, runs, ends clean, but unfinished). Every candidate is then
validated locally: parses, single named function, clean line-end, no imports/IO,
runs without raising on a provided sample call, and the buggy outputs differ from
the correct output. Hand-authored seeds (from build_verifier_stubs) are prepended
as a guaranteed-quality core.

Run (full):   conda run -n goals python scripts/clock/build_verifier_oracle.py --calls 30 --k 15 --target 200
Run (smoke):  conda run -n goals python scripts/clock/build_verifier_oracle.py --smoke
Writes: data/verifier/wrong_task.json, data/verifier/incomplete.json
"""

import os
import re
import sys
import json
import signal
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.oracle_llms import ask_claude
from scripts.clock.build_verifier_stubs import W as SEED_W, I as SEED_I

SYSTEM = ("You are a meticulous Python instructor generating coding exercises and "
          "deliberately-buggy variants for an interpretability dataset. You output only JSON.")

THEMES = [
    "integer/number predicates and arithmetic", "string manipulation and parsing",
    "list aggregation and filtering", "dictionary / counting tasks",
    "recursion and sequences", "search and sorting", "geometry and simple math",
    "date / calendar logic", "validation and formatting", "boolean logic and conditions",
    "set operations", "matrix / nested-list basics",
]

PROMPT_TMPL = """Generate {k} DISTINCT, simple, self-contained Python function tasks. \
Focus this batch on: {theme}. (batch #{batch})

For each task give these fields:
1. "name": the function name (a valid identifier).
2. "task": a one-sentence spec that names the function, e.g. "Write a Python function `is_prime(n)` returning True iff n is prime."
3. "correct": a CORRECT implementation — a single complete `def` (1-8 lines), no imports.
4. "wrong_task": a COMPLETE impl that PARSES and RUNS without error but computes something clearly UNRELATED to the task (answers a different question). Same name/params. e.g. for is_palindrome, check whether the first character is lowercase.
5. "incomplete": a COMPLETE impl that PARSES and RUNS without error, is ON-TASK, but does NOT finish: it handles only SOME required cases, or performs only the FIRST step(s) of a multi-step task and returns early. NO syntax errors, NOT cut off mid-line. Same name/params. e.g. is_prime that only checks divisibility by 2 and 3.
6. "sample_call": one expression calling the function with concrete literal args, valid for all three impls, e.g. "is_prime(7)".

Hard constraints:
- No imports, no randomness, no input()/print(), no file/network access.
- All three impls are a single top-level function with the SAME name and parameters.
- wrong_task and incomplete must return a value without raising on sample_call, and must give a DIFFERENT result than correct.
- Do NOT reuse any of these already-used names: {used}.

Return ONLY a JSON array of {k} objects with keys name, task, correct, wrong_task, incomplete, sample_call. No markdown, no prose."""

FORBIDDEN = ("import ", "input(", "print(", "open(", "exec(", "eval(", "__")
BAD_END = (",", "(", "[", "{", "+", "-", "*", "/", "=", "%", "<", ">", ":", "\\", "&", "|", "and", "or")


class _Timeout(Exception):
    pass


def _run(code, call, timeout=2):
    def _h(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        ns = {}
        exec(code, ns)
        return eval(call, ns)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    a, b = t.find("["), t.rfind("]")
    return json.loads(t[a:b + 1])


def _single_fn(code, name):
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    return len(tree.body) == 1 and len(fns) == 1 and fns[0].name == name


def _clean(code):
    last = code.rstrip().splitlines()[-1].rstrip()
    return not last.endswith(BAD_END)


def _ok_code(code, name):
    return (name in code and not any(f in code for f in FORBIDDEN)
            and _single_fn(code, name) and _clean(code))


def validate(obj):
    """Return (wrong_row, incomplete_row) or (None, None) if the candidate fails any gate."""
    try:
        name = obj["name"]; task = obj["task"]; call = obj["sample_call"]
        cor, wr, inc = obj["correct"], obj["wrong_task"], obj["incomplete"]
    except (KeyError, TypeError):
        return None, None
    if not re.match(r"^[A-Za-z_]\w*$", str(name)):
        return None, None
    for code in (cor, wr, inc):
        if not _ok_code(code, name):
            return None, None
    try:
        c_out = _run(cor, call)
        w_out = _run(wr, call)
        i_out = _run(inc, call)
    except (_Timeout, Exception):
        return None, None

    def _differs(a, b):
        try:
            return bool(a != b)
        except Exception:
            return True
    wrong_row = inc_row = None
    if _differs(w_out, c_out):
        wrong_row = dict(lang="python", name=name, task=task, correct=cor.strip("\n"),
                         incorrect=wr.strip("\n"), bug="blatant", blatant_kind="wrong_task")
    if _differs(i_out, c_out):
        inc_row = dict(lang="python", name=name, task=task, correct=cor.strip("\n"),
                       incorrect=inc.strip("\n"), bug="incomplete", sub_kind="oracle")
    return wrong_row, inc_row


def _write(items, path):
    for i, e in enumerate(items):
        e["id"] = f"{e['lang']}_{i:03d}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--smoke", action="store_true", help="1 call, k=6, no write")
    args = ap.parse_args()
    if args.smoke:
        args.calls, args.k = 1, 6

    seed_names = {e["name"] for e in SEED_W} | {e["name"] for e in SEED_I}
    wrong = {e["name"]: e for e in SEED_W}
    incomplete = {e["name"]: e for e in SEED_I}
    used = set(seed_names)
    raw_seen = n_valid = 0

    for c in range(args.calls):
        prompt = PROMPT_TMPL.format(k=args.k, theme=THEMES[c % len(THEMES)], batch=c,
                                    used=", ".join(sorted(used)) or "(none)")
        try:
            text = ask_claude(prompt, max_tokens=8000, temperature=args.temp,
                              model_name=args.model, system_prompt=SYSTEM)
            objs = _parse_json(text)
        except Exception as e:
            print(f"  [call {c}] FAILED: {type(e).__name__}: {e}")
            continue
        for obj in objs:
            raw_seen += 1
            nm = obj.get("name")
            if not nm or nm in used:
                continue
            wr_row, inc_row = validate(obj)
            if wr_row or inc_row:
                n_valid += 1
                used.add(nm)
            if wr_row:
                wrong[nm] = wr_row
            if inc_row:
                incomplete[nm] = inc_row
        print(f"  [call {c}] theme='{THEMES[c % len(THEMES)]}'  "
              f"wrong={len(wrong)}  incomplete={len(incomplete)}  (raw_seen={raw_seen})")
        if len(wrong) >= args.target and len(incomplete) >= args.target:
            print("  reached target for both types.")
            break

    W = list(wrong.values())
    I = list(incomplete.values())
    print(f"\nvalidated {n_valid}/{raw_seen} raw candidates")
    print(f"wrong_task={len(W)}  incomplete={len(I)}  (seeds: {len(SEED_W)}/{len(SEED_I)})")
    if args.smoke:
        print("[smoke] not writing; sample wrong_task:")
        if W:
            print(json.dumps(W[-1], indent=2)[:600])
        return
    out = os.path.join("data", "verifier")
    _write(W, os.path.join(out, "wrong_task.json"))
    _write(I, os.path.join(out, "incomplete.json"))
    print(f"wrote {len(W)} -> {out}/wrong_task.json ; {len(I)} -> {out}/incomplete.json")


if __name__ == "__main__":
    main()
