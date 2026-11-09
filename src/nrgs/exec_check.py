"""Execution check for NRGS transfers: does the destination's generated code COMPILE, and does its frontier statement
implement a sub-goal COMPLETELY (same value as the bank's reference implementation on every test input)?

The LM judges (``src.nrgs.eval``) say WHICH sub-goal a generation goes for; this checks that the code actually works.

    from src.nrgs import exec_check as XC
    r = XC.check(dst_prompt, gen_patched, datatype="int", targets=["is_odd", "absolute_value"])
    r["compiles"]                      # the whole generated function compiles (python: compile(); C++: g++ -fsyntax-only)
    r["targets"]["is_odd"]["complete"]  # the frontier statement == the reference on every test input
    XC.render(r)                       # one-line summary

How: the code is the prompt's code prefix (after the last ``<code>\\n``, ending in the frontier stub ``    c =`` /
``    auto c =``) + the generation (up to ``</code>``).
- ``compiles``: the whole function, as generated.
- frontier value: the prefix + the generation's frontier block (its lines up to the next line at the stub's
  indentation that returns or assigns another variable; multi-statement implementations included) +
  ``return <var>``, run on the test inputs (python in a subprocess, C++ compiled with g++ and run once per input, so a
  crash or UB affects one input only). Isolating the block makes the check independent of how the generation orders
  its ``return`` (Q transfers reorder it).
- reference: the bank's Python implementation of each target (``QTASKS[datatype]`` predicates, ``impl["python"]``),
  on the inputs where the reference function of every listed sub-task is defined.
- comparison: canonical value strings; a sub-goal whose reference values are all ``bool`` is compared by truth value
  (``n % 2`` for "is odd" counts); ``ALT_REFS`` lists equally valid readings of a description that the reference is
  stricter than (``vowel_count``: upper-case vowels count too). ``complete_typical`` repeats the check on
  non-negative ints and non-empty strings / lists, where language conventions (C++ ``-7 % 10 == -7``, Python
  ``-7 % 10 == 3``) do not differ.

Needs ``g++`` (C++17) on PATH for C++ destinations. No model, no API calls.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from src.nrgs.data import QTASKS

INPUTS: dict[str, list] = {
    "int": [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 15, 16, 25, 27, 49, 64, 97, 99, 100, 101, 121, 128, 255, 1000, 12321,
            123456, -1, -7, -10, -12, -99, -128],
    "str": ["", "a", "A", "abc", "Hello", "racecar", "Hello World", "hello world", "abc123", "AbC", "  ", "12345",
            "aeiou", "xyz", "Madam", "noon", "a b c", "Python3", "UPPER", "lower", "Zebra crossing"],
    "list[int]": [[], [0], [1], [-1], [1, 2, 3], [3, 2, 1], [1, 1, 2], [0, 5, -3], [2, 4, 6], [1, 3, 5], [-5, -2, -9],
                  [10, 20, 30, 40], [7, 7, 7], [1, -1, 0, 2], [100, 3, 57, 8], [4, 4, 9, 9, 2]],
    "list[str]": [[], ["a"], [""], ["apple", "banana"], ["Apple", "Banana", "cherry"], ["level", "noon", "abc"],
                  ["a1", "b2", "c"], ["x", "yy", "zzz"], ["same", "same"], ["Elephant", "owl", "Ice", "umbrella"],
                  ["", "abc"], ["longword", "tiny", "enormous"], ["Racecar", "deed", "x9y"]],
}


def typical(datatype: str, x) -> bool:
    """Inputs where Python / C++ conventions agree: non-negative ints, non-empty strings / lists (of non-empty words)."""
    if datatype == "int":
        return x >= 0
    if datatype == "list[int]":
        return len(x) > 0 and all(v >= 0 for v in x)
    if datatype == "list[str]":
        return len(x) > 0 and all(len(w) > 0 for w in x)
    return len(x) > 0


PRED = {dt: {p.id: p for p in t.predicates} for dt, t in QTASKS.items()}
# Equally valid readings of a sub-task's description that the bank's reference is stricter than; a frontier value
# counts when it matches the reference OR one of these (per input).
ALT_REFS: dict[tuple[str, str], list[str]] = {
    ("str", "vowel_count"): ["{res} = sum(1 for c in {arg} if c.lower() in 'aeiou')"],   # "the number of vowels"
}
CPP_HEAD = "#include <bits/stdc++.h>\nusing namespace std;\n"   # the prompts say "assume all libraries are imported"
_CPP_CANON = r'''
static string jq(const string& s){string o="\"";for(char c:s){if(c=='"'||c=='\\'){o+='\\';o+=c;}else o+=c;}return o+"\"";}
static string canon(bool v){return v?"1":"0";}
static string canon(char c){return jq(string(1,c));}
static string canon(const string& s){return jq(s);}
static string canon(const char* s){return jq(s);}
template<class T> static typename enable_if<is_integral<T>::value && !is_same<T,bool>::value && !is_same<T,char>::value,string>::type canon(T v){return to_string(v);}
template<class T> static typename enable_if<is_floating_point<T>::value,string>::type canon(T v){char b[64];snprintf(b,64,"%.6g",(double)v);return b;}
template<class T> static string canon(const vector<T>& v){string o="[";for(size_t i=0;i<v.size();i++){if(i)o+=",";o+=canon(v[i]);}return o+"]";}
'''
_PY_RUNNER = '''
import json, signal, sys
def canon(v):
    if isinstance(v, bool): return str(int(v))
    if isinstance(v, int): return str(v)
    if isinstance(v, float): return format(v, ".6g")
    if isinstance(v, str): return json.dumps(v)
    if isinstance(v, (list, tuple)): return "[" + ",".join(canon(x) for x in v) + "]"
    return "UNSUPPORTED:" + type(v).__name__
def _alarm(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
for x in json.loads(sys.argv[1]):
    try:
        signal.alarm(5); v = func(x); signal.alarm(0); print(canon(v))
    except BaseException as e:
        signal.alarm(0); print("ERR:" + type(e).__name__)
'''


def canon(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return format(v, ".6g")
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canon(x) for x in v) + "]"
    return repr(v)


# ── parsing ──────────────────────────────────────────────────────────────────
def code_prefix(prompt: str) -> str:
    """The prefilled code: everything after the last ``<code>\\n`` (ends in the frontier stub)."""
    return prompt.rsplit("<code>\n", 1)[1]


def language(prefix: str) -> str:
    return "python" if prefix.lstrip().startswith("def ") else "cpp"


def frontier_var(prefix: str) -> str:
    m = re.search(r"(?:auto\s+)?([A-Za-z_]\w*)\s*=\s*$", prefix.rsplit("\n", 1)[-1])
    if not m:
        raise ValueError(f"prefix does not end in a frontier stub: {prefix[-40:]!r}")
    return m.group(1)


def arg_name(prefix: str) -> str:
    m = re.search(r"func\(\s*(?:const\s+)?(?:[\w:<>]+\s*&?\s+)?([A-Za-z_]\w*)\s*(?::[^)]*)?\)", prefix)
    if not m:
        raise ValueError("no `func(<arg>)` signature in the prefix")
    return m.group(1)


def _strip_gen(gen: str) -> str:
    """The generation up to ``</code>``; a closing tag cut off by the token budget (``…}\\n</code``) is dropped too."""
    g = gen.split("</code>")[0]
    for k in range(len("</code>") - 1, 0, -1):
        if g.endswith("</code>"[:k]):
            return g[:-k]
    return g


def truncated(gen: str) -> bool:
    """True when the generation stops before closing its code block (the token budget ran out mid-code)."""
    g = gen.split("</code>")[0]
    return "</code>" not in gen and not any(g.endswith("</code>"[:k]) for k in range(2, len("</code>")))


def _depth(s: str) -> int:
    """Bracket depth at the end of ``s`` (string / char literals skipped)."""
    depth, i, q = 0, 0, None
    while i < len(s):
        ch = s[i]
        if q:
            if ch == "\\":
                i += 1
            elif ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        i += 1
    return depth


def frontier_block(gen: str, lang: str, prefix: str) -> str | None:
    """The generation's code for the frontier variable: from the stub up to (not including) the next line at the
    stub's indentation that returns or assigns another variable, or closes the function. Covers multi-statement
    implementations (``auto c = 0; for (...) if (...) c++;``) and leaves later sub-goals / the return out."""
    g, var = _strip_gen(gen), frontier_var(prefix)
    ind = re.match(r"\s*", prefix.rsplit("\n", 1)[-1]).group(0)
    bound = re.compile(rf"^{ind}(?:return\b|\}}|(?:[\w:<>,*&\s]+\s+)?(?!{re.escape(var)}\b)[A-Za-z_]\w*\s*(?:[-+*/%|&^]|<<|>>)?=(?!=))")
    lines = g.split("\n")
    block = [lines[0]]
    for line in lines[1:]:
        cur = "\n".join(block)
        if _depth(cur) == 0 and (bound.match(line) or (lang == "cpp" and line.startswith("}"))):
            break
        block.append(line)
    block = "\n".join(block).rstrip()
    if lang == "python":
        try:
            compile(prefix + block + f"\n{ind}return {var}\n", "<frontier>", "exec")
            return block
        except SyntaxError:
            return first_statement(gen, lang, prefix)
    if _depth(block) != 0 or not block.endswith((";", "}")):
        return first_statement(gen, lang, prefix)
    return block


def first_statement(gen: str, lang: str, prefix: str = "") -> str | None:
    """The generation's first statement (the frontier's right-hand side). C++: up to the first ``;`` outside
    brackets / literals. Python: the shortest run of lines that compiles after the prefix."""
    g = _strip_gen(gen)
    if lang == "cpp":
        depth, i, q = 0, 0, None
        while i < len(g):
            ch = g[i]
            if q:
                if ch == "\\":
                    i += 1
                elif ch == q:
                    q = None
            elif ch in "\"'":
                q = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth < 0:
                    return None
            elif ch == ";" and depth == 0:
                return g[: i + 1]
            i += 1
        return None
    lines = g.split("\n")
    var = frontier_var(prefix)
    for k in range(1, min(len(lines), 12) + 1):
        cand = "\n".join(lines[:k])
        try:
            compile(prefix + cand + f"\n    return {var}\n", "<frontier>", "exec")
            return cand
        except SyntaxError:
            continue
    return None


# ── references ───────────────────────────────────────────────────────────────
def reference(datatype: str, pid: str, arg: str, x):
    loc = {arg: x}
    exec(PRED[datatype][pid].impl["python"].format(res="_r", arg=arg), {}, loc)
    return loc["_r"]


def test_inputs(datatype: str, arg: str, pids: list[str]) -> list:
    """INPUTS[datatype] restricted to the inputs on which every given sub-task's reference is defined."""
    out = []
    for x in INPUTS[datatype]:
        try:
            for p in pids:
                reference(datatype, p, arg, x)
            out.append(x)
        except Exception:
            pass
    return out


# ── running ──────────────────────────────────────────────────────────────────
def _cpp_lit(datatype: str, x) -> str:
    if datatype == "int":
        return str(x)
    if datatype == "str":
        return "std::string(" + json.dumps(x) + ")"
    if datatype == "list[int]":
        return "std::vector<int>{" + ",".join(map(str, x)) + "}"
    return "std::vector<std::string>{" + ",".join(json.dumps(s) for s in x) + "}"


def _first_error(stderr: str) -> str:
    line = next((l for l in stderr.splitlines() if "error" in l), stderr.strip()[:200])
    return line.split("error:", 1)[-1].strip()[:200]


def compiles(code: str, lang: str) -> tuple[bool, str]:
    if lang == "python":
        try:
            compile(code, "<gen>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg}"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "f.cpp")
        open(f, "w").write(CPP_HEAD + code + "\nint main(){}\n")
        p = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", f], capture_output=True, text=True, timeout=120)
        return p.returncode == 0, ("" if p.returncode == 0 else _first_error(p.stderr))


def run_frontier(fn_code: str, lang: str, datatype: str, xs: list) -> tuple[list[str] | None, str]:
    """Run ``func`` (which returns the frontier value) on each input; canonical value strings or ``ERR:...``."""
    with tempfile.TemporaryDirectory() as d:
        if lang == "python":
            f = os.path.join(d, "m.py")
            open(f, "w").write(fn_code + "\n" + _PY_RUNNER)
            try:
                p = subprocess.run([sys.executable, "-I", f, json.dumps(xs)], capture_output=True, text=True,
                                   timeout=60 + 6 * len(xs))
            except subprocess.TimeoutExpired:
                return ["ERR:Timeout"] * len(xs), "timeout"
            lines = p.stdout.splitlines()
            return (lines + ["ERR:crash"] * len(xs))[: len(xs)], ("" if len(lines) == len(xs) else p.stderr[-200:])
        cases = "\n".join(f"  if(i=={i}){{ cout << canon(func({_cpp_lit(datatype, x)})) << endl; }}"
                          for i, x in enumerate(xs))
        f, exe = os.path.join(d, "h.cpp"), os.path.join(d, "h")
        open(f, "w").write(CPP_HEAD + _CPP_CANON + fn_code + "\nint main(int argc,char**argv){ int i=atoi(argv[1]);\n"
                           + cases + "\n  return 0; }\n")
        c = subprocess.run(["g++", "-std=c++17", "-O0", "-o", exe, f], capture_output=True, text=True, timeout=180)
        if c.returncode != 0:
            return None, _first_error(c.stderr)
        outs = []
        for i in range(len(xs)):
            try:
                p = subprocess.run([exe, str(i)], capture_output=True, text=True, timeout=10)
                outs.append(p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else f"ERR:rc{p.returncode}")
            except subprocess.TimeoutExpired:
                outs.append("ERR:Timeout")
        return outs, ""


def _truthy(s: str) -> str:
    if s.startswith("ERR:"):
        return s
    try:
        return "1" if float(s) != 0 else "0"
    except ValueError:
        return s


# ── main entry ───────────────────────────────────────────────────────────────
def check(prompt: str, gen: str, *, datatype: str, targets: list[str], listed: list[str] | None = None) -> dict:
    """Compile + execution check of one generation.

    prompt   the destination prompt (its code prefix ends in the frontier stub); or the code prefix itself
    gen      the destination's generation after the prompt
    targets  sub-task ids (``QTASKS[datatype]`` predicates) to test the frontier statement against
    listed   all sub-task ids of the destination's task list (test inputs must be valid for each); default = targets
    """
    prefix = code_prefix(prompt) if "<code>\n" in prompt else prompt
    lang, var, arg = language(prefix), frontier_var(prefix), arg_name(prefix)
    whole = prefix + _strip_gen(gen)
    ok, err = compiles(whole, lang)
    stmt = frontier_block(gen, lang, prefix)
    xs = test_inputs(datatype, arg, list(dict.fromkeys((listed or []) + list(targets))))
    out = dict(lang=lang, var=var, compiles=ok, compile_error=err, truncated=truncated(gen), statement=stmt,
               n_tests=len(xs), targets={}, run_error="")
    outs = None
    if stmt is None:
        out["run_error"] = "no complete frontier statement in the generation"
    else:
        fn = prefix + stmt + (f"\n    return {var}\n" if lang == "python" else f"\n    return {var};\n}}\n")
        outs, out["run_error"] = run_frontier(fn, lang, datatype, xs)
    for t in targets:
        refs = [reference(datatype, t, arg, x) for x in xs]
        is_bool = all(isinstance(v, bool) for v in refs)
        want = [canon(v) for v in refs]
        got = outs or ["ERR:not run"] * len(xs)
        if is_bool:
            want, got = [_truthy(w) for w in want], [_truthy(g) for g in got]
        alts = []
        for impl in ALT_REFS.get((datatype, t), []):
            vals = []
            for x in xs:
                loc = {arg: x}
                exec(impl.format(res="_r", arg=arg), {}, loc)
                vals.append(canon(loc["_r"]))
            alts.append([_truthy(v) for v in vals] if is_bool else vals)
        hit = [g == w or any(g == a[i] for a in alts) for i, (g, w) in enumerate(zip(got, want))]
        typ = [h for h, x in zip(hit, xs) if typical(datatype, x)]
        out["targets"][t] = dict(
            n_pass=sum(hit), n=len(xs), complete=bool(outs) and all(hit),
            n_pass_typical=sum(typ), n_typical=len(typ), complete_typical=bool(outs) and all(typ),
            failures=[(x, g, w) for x, g, w, h in zip(xs, got, want, hit) if not h][:5],
        )
    return out


def render(r: dict) -> str:
    """One-line summary for notebooks."""
    s = f"compiles: {'yes' if r['compiles'] else 'NO (' + r['compile_error'][:80] + ')'}"
    if r.get("truncated"):
        s += " [generation cut off by the token budget]"
    for t, v in r["targets"].items():
        s += (f" | {t}: {v['n_pass']}/{v['n']} inputs"
              f"{' (complete)' if v['complete'] else ' (complete on typical inputs)' if v['complete_typical'] else ''}")
    if r["run_error"]:
        s += f" | run: {r['run_error'][:80]}"
    return s
