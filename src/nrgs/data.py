"""Data generation for the DIFF-hypothesis experiments (q-transfer / pivot-onwards kv-transfer).

One function-writing task made of several INDEPENDENT, COMMUTATIVE sub-predicates (the per-datatype :data:`QTASKS`
banks), listed as a numbered task. Contrastive SRC / DST prompts differ in the listed predicates and / or the prefilled
PROGRESS, so at the shared pivot the source is "about to write" a different next sub-predicate than the destination:

* :class:`DiffTriggerInstance` (:func:`diff_trigger_setups`) — query transfer (``scripts/nrgs/q_transfer_eval.py``);
* :class:`MultiPredicateInstance` (:func:`multi_predicate_setups`) — kv transfer of the task span
  (``scripts/nrgs/kv_transfer_eval.py``); the multi-hierarchy variant over :data:`FILTER_TASKS` conditions lives in
  :mod:`src.nrgs.hier_data`.

Generalization axes:
    • datatype     — int | str | list[int] | list[str]
    • language     — python | cpp, independently for src and dst (cross-language transfer).

Predicate implementations are templates over the placeholders ``{arg}`` (the function's parameter name), ``{res}``
(the result variable name of THIS predicate) and ``{elem}`` (a filter condition's element).

Prompts are materialized with the chat template via :func:`src.nrgs.functional.get_prompt`
(imported lazily so plain dataset inspection needs no model).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

Lang = Literal["python", "cpp"]
Presentation = Literal["description", "listed"]

# ── datatype → (question phrase, python annotation, c++ type) ──
DATATYPES: dict[str, dict[str, str]] = {
    "int": {"phrase": "an integer", "py": "int", "cpp": "int"},
    "str": {"phrase": "a string", "py": "str", "cpp": "std::string"},
    "list[int]": {
        "phrase": "a list of integers",
        "py": "list[int]",
        "cpp": "std::vector<int>",
    },
    "list[str]": {
        "phrase": "a list of strings",
        "py": "list[str]",
        "cpp": "std::vector<std::string>",
    },
}
_LANG_NAME = {"python": "Python", "cpp": "C++"}


# ══════════════════════════════════════════════════════════════════════════════
# building blocks
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Predicate:
    """One sub-computation. ``impl[lang]`` is a code template over {arg}/{res}/{prog}.

    For q-transfer predicates and kv common-progress: ``impl`` is an ASSIGNMENT block that
    binds {res} (may span multiple statements). For kv-transfer TASKS: ``impl`` is a single
    EXPRESSION (used after ``return``), typically referencing {prog}."""

    id: str
    description: str  # natural-language, language-agnostic (may use {arg})
    result_name: str  # natural name for {res} / {prog}
    impl: dict[str, str]  # lang -> code template


@dataclass(frozen=True)
class QTransferTask:
    """A multi-predicate task: N independent, commutative predicates over one parameter."""

    id: str
    datatype: str
    var_names: tuple[str, ...]
    predicates: tuple[Predicate, ...]


@dataclass(frozen=True)
class FilterTask:
    """A single filter that COMPOSES independent, commutative element-wise conditions inside ONE
    return expression, e.g. "all the words that are palindromes and start with a vowel". Unlike
    :class:`QTransferTask` (one assignment per sub-goal, statement-boundary pivot), the pivot here
    sits MID-EXPRESSION — right after ``… if <cond> and`` — so the model completes the same
    comprehension by adding the next condition (:mod:`src.nrgs.hier_data`).

    Each condition is a boolean template over ``{elem}`` (the loop/comprehension variable) and, if
    needed, ``{arg}``; ``description`` is the NL fragment ("are palindromes")."""

    id: str
    datatype: str  # list[str] | list[int]
    var_names: tuple[str, ...]
    elem: str  # comprehension element variable, e.g. "w"
    noun: str  # NL plural noun for the phrase ("words", "values", "characters")
    conditions: tuple[
        Predicate, ...
    ]  # description = NL fragment; impl[lang] = boolean expr over {elem}


# ══════════════════════════════════════════════════════════════════════════════
# rendering helpers
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(template: str, arg: str, res: str = "", prog: str = "", elem: str = "") -> str:
    return (
        template.replace("{arg}", arg)
        .replace("{res}", res)
        .replace("{prog}", prog)
        .replace("{elem}", elem)
    )


def _signature(datatype: str, arg: str, lang: Lang) -> str:
    dt = DATATYPES[datatype]
    if lang == "python":
        return f"def func({arg}: {dt['py']}):"
    return f"auto func({dt['cpp']} {arg})" + "{"  # c++ opens the body brace


CPP_NOTE = "Assume that all the libraries have been already imported."
# True: the C++ note moves from the end of the user turn to the system prompt (C++ prompts only), so in every
# language the task list is followed by just its closing `\n` before <|im_end|>
CPP_NOTE_IN_SYSTEM = False


def _question(
    descriptions: list[str],
    datatype: str,
    arg: str,
    lang: Lang,
    presentation: Presentation,
) -> str:
    dt = DATATYPES[datatype]
    head = (
        f"Write a {_LANG_NAME[lang]} program that implements a function `func` which, "
        f"given {dt['phrase']} `{arg}`, computes:\n"
    )
    if presentation == "listed":
        body = "\n".join(f"({i + 1}) {d}" for i, d in enumerate(descriptions))
    else:  # description / prose
        body = "; ".join(descriptions)
    tail = f"\n{CPP_NOTE}\n" if lang == "cpp" and not CPP_NOTE_IN_SYSTEM else "\n"
    return head + body + tail


def _assign_line(
    pred: Predicate,
    arg: str,
    lang: Lang,
    prog: str = "",
    res_name: Optional[str] = None,
) -> str:
    """Render a predicate's assignment block as flat 4-space-indented code line(s).
    Templates may carry their own newlines for multi-statement c++ blocks; each statement is
    re-indented to a single level (they are sequential, not nested). ``res_name`` overrides the
    result variable ({res}) — used to switch between natural names and letter names."""
    code = _fmt(pred.impl[lang], arg=arg, res=res_name or pred.result_name, prog=prog)
    return "\n".join("    " + ln.strip() for ln in code.split("\n") if ln.strip())


# C++ type prefix (optionally qualified / templated / ref), used to rewrite a result-var declaration
# to ``auto`` for the var_eq letter stub. Deliberately conservative: matched only at a line start and
# only when immediately followed by the exact result-variable name.
_CPP_TYPE = (
    r"(?:unsigned\s+|signed\s+|const\s+|long\s+|short\s+)*"
    r"[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*(?:\s*<[^;{}\n]*>)?\s*[\*&]?"
)


def _cpp_autoify(cpp_block: str, var: str) -> str:
    """Rewrite the declaration of result variable ``var`` in a rendered C++ block to ``auto`` — so the
    prefilled shared lines match the ``auto {var} =`` pivot of the ``var_eq`` letter stub. Only the
    result variable's own declaration is touched; helper vars (``{var}_s`` etc.) and later
    reassignments (no type prefix) are left untouched (anchored at line start; must be immediately
    followed by ``var`` then ``=``/``;``)."""
    pat = re.compile(
        r"(^[ \t]*)" + _CPP_TYPE + r"\s+(" + re.escape(var) + r")(\s*[=;])",
        re.MULTILINE,
    )
    return pat.sub(lambda m: f"{m.group(1)}auto {m.group(2)}{m.group(3)}", cpp_block)


# ══════════════════════════════════════════════════════════════════════════════
# instances
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QTransferInstance:
    """A single q-transfer contrastive setup (same task, different progress)."""

    task: QTransferTask
    var_name: str
    presentation: Presentation
    src_lang: Lang
    dst_lang: Lang
    src_progress: list[str]  # predicate ids prefilled as done in SRC
    dst_progress: list[str]  # predicate ids prefilled as done in DST
    expected_next: str  # predicate id SRC is about to write (transplant target)
    naming: Literal["natural", "letters"] = (
        "natural"  # result var names: descriptive vs a/b/c/d
    )
    meta: dict = field(default_factory=dict)

    def var_of(self, pred_id: str) -> str:
        """Result variable name for a predicate — descriptive ("natural") or positional letter."""
        by_id = {p.id: p for p in self.task.predicates}
        if self.naming == "letters":
            idx = [p.id for p in self.task.predicates].index(pred_id)
            return chr(ord("a") + idx)
        return by_id[pred_id].result_name

    def _parts(self, lang: Lang, progress_ids: list[str]) -> tuple[str, str]:
        descs = [p.description for p in self.task.predicates]
        q = _question(descs, self.task.datatype, self.var_name, lang, self.presentation)
        by_id = {p.id: p for p in self.task.predicates}
        lines = [
            _assign_line(by_id[i], self.var_name, lang, res_name=self.var_of(i))
            for i in progress_ids
        ]
        prefill = (
            "<code>\n"
            + _signature(self.task.datatype, self.var_name, lang)
            + ("\n" + "\n".join(lines) if lines else "")
            + "\n    "
        )  # trailing indent = pivot
        return q, prefill

    def src_parts(self):
        return self._parts(self.src_lang, self.src_progress)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_progress)

    @property
    def expected_next_desc(self) -> str:
        return next(
            p.description for p in self.task.predicates if p.id == self.expected_next
        )

    @property
    def dst_next_desc(self) -> str:
        did = self.meta.get("dst_next")
        return next(p.description for p in self.task.predicates if p.id == did)

    @property
    def task_desc(self) -> str:
        return "; ".join(p.description for p in self.task.predicates)


@dataclass(frozen=True)
class MultiPredicateInstance:
    """MULTI-PREDICATE, divergent-frontier setup (the kv-transfer setup).

    SRC lists predicates ``[*shared, src_frontier]`` (e.g. T1, T2, T3); DST lists
    ``[*shared, dst_frontier]`` (e.g. T1, T2, T4). Both PREFILL the SHARED prefix (T1, T2) as
    independent assignment lines, then hit the same trailing-indent pivot — so SRC is about to write
    its frontier sub-goal (T3) and clean DST its frontier (T4). Mechanically identical to
    :class:`QTransferInstance` (independent assignments, indent pivot, query transplant), but the two
    sides pursue GENUINELY DIFFERENT next sub-goals rather than differing only in progress. Because the
    SRC target (T3) never appears in the DST prompt, GAZE is N/A here (as in kv-transfer) — eval is
    causal + readout only.

    The test: after transplanting SRC's pivot query, does clean DST write T3 instead of T4?"""

    datatype: str
    shared: tuple[
        Predicate, ...
    ]  # predicates prefilled in BOTH (the common progress: T1, T2, …)
    src_frontier: Predicate  # SRC's next sub-goal = transplant target (T3)
    dst_frontier: Predicate  # clean DST's next sub-goal (T4)
    var_name: str
    presentation: Presentation
    src_lang: Lang
    dst_lang: Lang
    naming: Literal["natural", "letters"] = "natural"
    letter_stub: Literal["indent", "var_eq"] = (
        "indent"  # pivot shape: "indent" = trailing indent (about to write the whole frontier line);
    )
    # "var_eq" = also prefill the frontier's variable name up to ``=`` (python ``    d =`` / c++
    # ``    auto d =`` for k=4), so the model writes only the frontier's VALUE. Intended with
    # naming="letters" (positional a/b/c/d); the prefilled c++ shared decls are rewritten to ``auto``
    # too so they match the ``auto d =`` pivot.
    meta: dict = field(default_factory=dict)

    def _preds_for(self, frontier: Predicate) -> tuple[Predicate, ...]:
        return self.shared + (frontier,)

    def var_of(self, preds: tuple[Predicate, ...], pred_id: str) -> str:
        """Result variable name for a predicate within a side's list — descriptive or positional letter."""
        if self.naming == "letters":
            return chr(ord("a") + [p.id for p in preds].index(pred_id))
        return {p.id: p for p in preds}[pred_id].result_name

    def full_task_text(self, frontier: Predicate) -> str:
        """The WHOLE rendered task list (shared T1,T2 + the frontier) exactly as it appears in the
        question body — the k/v TRANSPLANT span for the multi-predicate setup (cache SRC's full list,
        replace DST's full list; the ``pad`` alignment inserts the length-matching newlines at the
        START of this span, i.e. before ``(1)``). Mirrors ``_question``'s body construction."""
        descs = [p.description for p in self._preds_for(frontier)]
        if self.presentation == "listed":
            return "\n".join(f"({i + 1}) {d}" for i, d in enumerate(descs))
        return "; ".join(descs)

    def _parts(self, lang: Lang, frontier: Predicate) -> tuple[str, str]:
        preds = self._preds_for(frontier)
        q = _question(
            [p.description for p in preds],
            self.datatype,
            self.var_name,
            lang,
            self.presentation,
        )
        var_eq = self.letter_stub == "var_eq"
        lines = []
        for p in self.shared:
            rn = self.var_of(preds, p.id)
            ln = _assign_line(p, self.var_name, lang, res_name=rn)
            if var_eq and lang == "cpp":  # match the ``auto d =`` pivot style
                ln = _cpp_autoify(ln, rn)
            lines.append(ln)
        base = (
            "<code>\n"
            + _signature(self.datatype, self.var_name, lang)
            + ("\n" + "\n".join(lines) if lines else "")
        )
        if var_eq:  # prefill the frontier's var name up to ``=`` (python ``d =`` / c++ ``auto d =``)
            fv = self.var_of(preds, frontier.id)
            stub = f"auto {fv} =" if lang == "cpp" else f"{fv} ="
            prefill = base + "\n    " + stub
        else:
            prefill = base + "\n    "  # trailing indent = pivot
        return q, prefill

    def src_parts(self):
        return self._parts(self.src_lang, self.src_frontier)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_frontier)

    @property
    def expected_next(self) -> str:
        return self.src_frontier.id

    @property
    def dst_next(self) -> str:
        return self.dst_frontier.id

    @property
    def expected_next_desc(self) -> str:
        return self.src_frontier.description

    @property
    def dst_next_desc(self) -> str:
        return self.dst_frontier.description

    @property
    def task_desc(self) -> str:
        """Oracle context line: the SHARED prefilled sub-goals (T3 vs T4 are the A/B candidates)."""
        return "; ".join(p.description for p in self.shared)


def _finalize_prefill(prefill: str) -> str:
    """Drop a DANGLING trailing space so the pivot lands on a real token, NOT whitespace.

    The intended pivot is the last token of the prefill. Two shapes exist:
      • indent pivot (q-transfer, kv ``pivot="indent"``): the final line is pure indentation
        (``…\\n    ``) — this whitespace IS the pivot ("about to write the next line"); KEEP it.
      • keyword pivot (filter ``… if ``, kv/filter ``… return ``): a content line ending in a
        stray space — the pivot should be ``if`` / ``return``, so strip the trailing space(s)/tabs.
    Distinguishing rule: strip trailing spaces/tabs only when the final line has non-space content."""
    last_nl = prefill.rfind("\n")
    last_line = prefill[last_nl + 1 :]
    if (
        last_line.strip() == ""
    ):  # pure-indent final line -> the indent IS the pivot, keep as-is
        return prefill
    return prefill.rstrip(" \t")


def materialize(instance, ctx=None) -> dict:
    """Apply the chat template to an instance's src/dst (question, prefill) parts.

    Returns ``{"src_prompt", "dst_prompt", ...}`` (+ ``src_task_text``/``dst_task_text`` for the multi-predicate
    and multi-hierarchy instances). ``ctx`` is a :class:`src.nrgs.functional.Ctx` (else module-global
    ``mt``). Prefills are finalized (``_finalize_prefill``) so a dangling trailing space never
    becomes the pivot. Imported lazily so dataset inspection needs no model."""
    from src.nrgs.functional import get_prompt

    qs, pfs = instance.src_parts()
    qd, pfd = instance.dst_parts()
    sys_extra = lambda lang: f" {CPP_NOTE}" if CPP_NOTE_IN_SYSTEM and lang == "cpp" else ""
    out = {
        "src_prompt": get_prompt(qs, _finalize_prefill(pfs), ctx=ctx, system_extra=sys_extra(instance.src_lang)),
        "dst_prompt": get_prompt(qd, _finalize_prefill(pfd), ctx=ctx, system_extra=sys_extra(instance.dst_lang)),
    }
    if isinstance(instance, MultiPredicateInstance):
        # the transplant span is the WHOLE rendered task list (shared T1,T2 + the frontier), NOT just
        # the frontier — cache SRC's full list, replace DST's full list. The ``pad`` alignment then
        # inserts the length-matching newlines at the START of this span (before ``(1)``, right after
        # ``computes:\n``), and every token in the span (pads + T1 + T2 + frontier) is transplanted.
        out["src_task_text"] = instance.full_task_text(instance.src_frontier)
        out["dst_task_text"] = instance.full_task_text(instance.dst_frontier)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# predicate banks — one q-task per datatype
# ══════════════════════════════════════════════════════════════════════════════
def _P(id, description, result_name, py, cpp):
    return Predicate(
        id=id,
        description=description,
        result_name=result_name,
        impl={"python": py, "cpp": cpp},
    )


QTASKS: dict[str, QTransferTask] = {
    "list[str]": QTransferTask(
        "wordflags",
        "list[str]",
        ("words", "arr"),
        (
            _P(
                "has_palindrome",
                "whether any word is a palindrome",
                "has_palindrome",
                "{res} = any(w == w[::-1] for w in {arg})",
                "bool {res} = false;\n    for (auto& w : {arg}) { std::string rv(w.rbegin(), w.rend()); if (w == rv) { {res} = true; break; } }",
            ),
            _P(
                "total_length",
                "the total number of characters across all words",
                "total_length",
                "{res} = sum(len(w) for w in {arg})",
                "int {res} = 0;\n    for (auto& w : {arg}) {res} += w.size();",
            ),
            _P(
                "count_long",
                "the number of words longer than five characters",
                "count_long",
                "{res} = sum(1 for w in {arg} if len(w) > 5)",
                "int {res} = 0;\n    for (auto& w : {arg}) if (w.size() > 5) {res}++;",
            ),
            _P(
                "count_capitalized",
                "the number of capitalized words",
                "count_capitalized",
                "{res} = sum(1 for w in {arg} if w[:1].isupper())",
                "int {res} = 0;\n    for (auto& w : {arg}) if (!w.empty() && std::isupper((unsigned char)w[0])) {res}++;",
            ),
            _P(
                "distinct_count",
                "the number of distinct words",
                "distinct_count",
                "{res} = len(set({arg}))",
                "std::set<std::string> {res}_s({arg}.begin(), {arg}.end());\n    int {res} = {res}_s.size();",
            ),
            _P(
                "count_starts_vowel",
                "the number of words that start with a vowel",
                "starts_vowel_count",
                '{res} = sum(1 for w in {arg} if w and w[0].lower() in "aeiou")',
                "int {res} = 0;\nfor(auto &{res}_w : {arg}){ if(!{res}_w.empty()){ char {res}_c = std::tolower((unsigned char){res}_w[0]); if({res}_c=='a'||{res}_c=='e'||{res}_c=='i'||{res}_c=='o'||{res}_c=='u') {res}++; } }",
            ),
            _P(
                "count_with_digit",
                "the number of words that contain a digit",
                "words_with_digit",
                "{res} = sum(1 for w in {arg} if any(c.isdigit() for c in w))",
                "int {res} = 0;\nfor(auto &{res}_w : {arg}){ bool {res}_f = false; for(char {res}_c : {res}_w) if(std::isdigit((unsigned char){res}_c)) {res}_f = true; if({res}_f) {res}++; }",
            ),
            _P(
                "count_all_lowercase",
                "the number of words written entirely in lowercase letters",
                "all_lower_count",
                "{res} = sum(1 for w in {arg} if w.islower())",
                "int {res} = 0;\nfor(auto &{res}_w : {arg}){ bool {res}_has = false, {res}_ok = true; for(char {res}_c : {res}_w){ if(std::isalpha((unsigned char){res}_c)){ {res}_has = true; if(std::isupper((unsigned char){res}_c)) {res}_ok = false; } } if({res}_has && {res}_ok) {res}++; }",
            ),
            _P(
                "count_short",
                "the number of words shorter than three characters",
                "short_word_count",
                "{res} = sum(1 for w in {arg} if len(w) < 3)",
                "int {res} = 0;\nfor(auto &{res}_w : {arg}) if({res}_w.size() < 3) {res}++;",
            ),
            _P(
                "longest_word_length",
                "the length of the longest word",
                "longest_len",
                "{res} = max(len(w) for w in {arg})",
                "int {res} = 0;\nfor(auto &{res}_w : {arg}) if((int){res}_w.size() > {res}) {res} = (int){res}_w.size();",
            ),
            _P(
                "shortest_word_length",
                "the length of the shortest word",
                "shortest_len",
                "{res} = min(len(w) for w in {arg})",
                "int {res} = INT_MAX;\nfor(auto &{res}_w : {arg}) if((int){res}_w.size() < {res}) {res} = (int){res}_w.size();",
            ),
            _P(
                "total_vowels",
                "the total number of vowels across all words",
                "vowel_total",
                '{res} = sum(1 for w in {arg} for c in w if c.lower() in "aeiou")',
                "int {res} = 0;\nfor(auto &{res}_w : {arg}) for(char {res}_c : {res}_w){ char {res}_l = std::tolower((unsigned char){res}_c); if({res}_l=='a'||{res}_l=='e'||{res}_l=='i'||{res}_l=='o'||{res}_l=='u') {res}++; }",
            ),
            _P(
                "any_empty",
                "whether any word is empty",
                "has_empty_word",
                "{res} = any(len(w) == 0 for w in {arg})",
                "bool {res} = false;\nfor(auto &{res}_w : {arg}) if({res}_w.empty()) {res} = true;",
            ),
            _P(
                "all_unique_lengths",
                "whether all words have distinct lengths",
                "unique_lengths",
                "{res} = len(set(len(w) for w in {arg})) == len({arg})",
                "std::set<int> {res}_s;\nfor(auto &{res}_w : {arg}) {res}_s.insert((int){res}_w.size());\nbool {res} = {res}_s.size() == {arg}.size();",
            ),
            _P(
                "count_ends_consonant",
                "the number of words that end with a consonant",
                "ends_consonant_count",
                '{res} = sum(1 for w in {arg} if w and w[-1].isalpha() and w[-1].lower() not in "aeiou")',
                "int {res} = 0;\nfor(auto &{res}_w : {arg}){ if(!{res}_w.empty()){ char {res}_c = {res}_w.back(); if(std::isalpha((unsigned char){res}_c)){ char {res}_l = std::tolower((unsigned char){res}_c); if(!({res}_l=='a'||{res}_l=='e'||{res}_l=='i'||{res}_l=='o'||{res}_l=='u')) {res}++; } } }",
            ),
            _P(
                "total_uppercase_letters",
                "the total number of uppercase letters across all words",
                "uppercase_total",
                "{res} = sum(1 for w in {arg} for c in w if c.isupper())",
                "int {res} = 0;\nfor(auto &{res}_w : {arg}) for(char {res}_c : {res}_w) if(std::isupper((unsigned char){res}_c)) {res}++;",
            ),
        ),
    ),
    "list[int]": QTransferTask(
        "numflags",
        "list[int]",
        ("nums", "arr"),
        (
            _P(
                "has_even",
                "whether any value is even",
                "has_even",
                "{res} = any(x % 2 == 0 for x in {arg})",
                "bool {res} = false;\n    for (int x : {arg}) if (x % 2 == 0) {res} = true;",
            ),
            _P(
                "total_sum",
                "the sum of all values",
                "total_sum",
                "{res} = sum({arg})",
                "int {res} = 0;\n    for (int x : {arg}) {res} += x;",
            ),
            _P(
                "max_val",
                "the largest value",
                "max_val",
                "{res} = max({arg})",
                "int {res} = *std::max_element({arg}.begin(), {arg}.end());",
            ),
            _P(
                "count_positive",
                "the number of positive values",
                "count_positive",
                "{res} = sum(1 for x in {arg} if x > 0)",
                "int {res} = 0;\n    for (int x : {arg}) if (x > 0) {res}++;",
            ),
            _P(
                "has_duplicate",
                "whether any value appears more than once",
                "has_duplicate",
                "{res} = len(set({arg})) != len({arg})",
                "std::set<int> {res}_s({arg}.begin(), {arg}.end());\n    bool {res} = {res}_s.size() != {arg}.size();",
            ),
            _P(
                "count_even",
                "the number of even values",
                "even_count",
                "{res} = sum(1 for x in {arg} if x % 2 == 0)",
                "int {res} = count_if({arg}.begin(), {arg}.end(), [](int x){ return x % 2 == 0; });",
            ),
            _P(
                "count_odd",
                "the number of odd values",
                "odd_count",
                "{res} = sum(1 for x in {arg} if x % 2 != 0)",
                "int {res} = count_if({arg}.begin(), {arg}.end(), [](int x){ return x % 2 != 0; });",
            ),
            _P(
                "count_negative",
                "the number of negative values",
                "negative_count",
                "{res} = sum(1 for x in {arg} if x < 0)",
                "int {res} = count_if({arg}.begin(), {arg}.end(), [](int x){ return x < 0; });",
            ),
            _P(
                "min_val",
                "the minimum value",
                "min_value",
                "{res} = min({arg})",
                "int {res} = *min_element({arg}.begin(), {arg}.end());",
            ),
            _P(
                "product_all",
                "the product of all values",
                "product_value",
                "{res} = __import__('math').prod({arg})",
                "long long {res} = 1;\nfor(int x : {arg}) {res} *= x;",
            ),
            _P(
                "value_range",
                "the difference between the maximum and minimum values",
                "range_value",
                "{res} = max({arg}) - min({arg})",
                "int {res} = *max_element({arg}.begin(), {arg}.end()) - *min_element({arg}.begin(), {arg}.end());",
            ),
            _P(
                "mean_int",
                "the integer mean of the values",
                "mean_value",
                "{res} = sum({arg}) // len({arg})",
                "long long {res}_s = accumulate({arg}.begin(), {arg}.end(), 0LL);\nint {res} = (int)({res}_s / (long long){arg}.size());",
            ),
            _P(
                "is_sorted",
                "whether the values are sorted in non-decreasing order",
                "is_sorted_flag",
                "{res} = all({arg}[i] <= {arg}[i+1] for i in range(len({arg})-1))",
                "bool {res} = is_sorted({arg}.begin(), {arg}.end());",
            ),
            _P(
                "all_positive",
                "whether all values are positive",
                "all_positive_flag",
                "{res} = all(x > 0 for x in {arg})",
                "bool {res} = all_of({arg}.begin(), {arg}.end(), [](int x){ return x > 0; });",
            ),
            _P(
                "contains_zero",
                "whether any value is zero",
                "contains_zero_flag",
                "{res} = 0 in {arg}",
                "bool {res} = (find({arg}.begin(), {arg}.end(), 0) != {arg}.end());",
            ),
            _P(
                "distinct_count",
                "the number of distinct values",
                "distinct_value_count",
                "{res} = len(set({arg}))",
                "set<int> {res}_st({arg}.begin(), {arg}.end());\nint {res} = (int){res}_st.size();",
            ),
        ),
    ),
    "int": QTransferTask(
        "intflags",
        "int",
        ("n", "x"),
        (
            _P(
                "is_even",
                "whether the number is even",
                "is_even",
                "{res} = ({arg} % 2 == 0)",
                "bool {res} = ({arg} % 2 == 0);",
            ),
            _P(
                "is_positive",
                "whether the number is positive",
                "is_positive",
                "{res} = ({arg} > 0)",
                "bool {res} = ({arg} > 0);",
            ),
            _P(
                "digit_count",
                "the number of digits",
                "digit_count",
                "{res} = len(str(abs({arg})))",
                "int {res} = std::to_string(std::abs({arg})).size();",
            ),
            _P(
                "is_multiple_of_3",
                "whether the number is a multiple of three",
                "is_multiple_of_3",
                "{res} = ({arg} % 3 == 0)",
                "bool {res} = ({arg} % 3 == 0);",
            ),
            _P(
                "is_two_digit",
                "whether the number has exactly two digits",
                "is_two_digit",
                "{res} = (10 <= abs({arg}) <= 99)",
                "bool {res} = (std::abs({arg}) >= 10 && std::abs({arg}) <= 99);",
            ),
            _P(
                "is_prime",
                "whether the number is prime",
                "is_prime",
                "{res} = {arg} > 1 and all({arg} % i != 0 for i in range(2, int({arg}**0.5) + 1))",
                "bool {res} = {arg} > 1;\nfor(long long i = 2; i*i <= {arg}; i++) if({arg} % i == 0) {res} = false;",
            ),
            _P(
                "is_perfect_square",
                "whether the number is a perfect square",
                "is_perfect_square",
                "{res} = {arg} >= 0 and int({arg}**0.5)**2 == {arg}",
                "bool {res} = false;\nif({arg} >= 0){ long long {res}_r = (long long)sqrt((double){arg}); for(long long d=-1; d<=1; d++) if(({res}_r+d)*({res}_r+d) == {arg}) {res} = true; }",
            ),
            _P(
                "is_power_of_two",
                "whether the number is a power of two",
                "is_power_of_two",
                "{res} = {arg} > 0 and ({arg} & ({arg} - 1)) == 0",
                "bool {res} = ({arg} > 0) && (({arg} & ({arg} - 1)) == 0);",
            ),
            _P(
                "digit_sum",
                "the sum of the digits",
                "digit_sum",
                "{res} = sum(int(d) for d in str(abs({arg})))",
                "int {res} = 0;\nint {res}_x = abs({arg});\ndo { {res} += {res}_x % 10; {res}_x /= 10; } while({res}_x != 0);",
            ),
            _P(
                "is_palindrome",
                "whether the number reads the same forwards and backwards",
                "is_palindrome",
                "{res} = str(abs({arg})) == str(abs({arg}))[::-1]",
                "string {res}_s = to_string(abs({arg}));\nstring {res}_r = {res}_s;\nreverse({res}_r.begin(), {res}_r.end());\nbool {res} = ({res}_s == {res}_r);",
            ),
            _P(
                "last_digit",
                "the last decimal digit",
                "last_digit",
                "{res} = abs({arg}) % 10",
                "int {res} = abs({arg}) % 10;",
            ),
            _P(
                "is_odd",
                "whether the number is odd",
                "is_odd",
                "{res} = {arg} % 2 != 0",
                "bool {res} = ({arg} % 2 != 0);",
            ),
            _P(
                "absolute_value",
                "the absolute value",
                "absolute_value",
                "{res} = abs({arg})",
                "int {res} = abs({arg});",
            ),
            _P(
                "sign",
                "the sign of the number",
                "sign",
                "{res} = ({arg} > 0) - ({arg} < 0)",
                "int {res} = ({arg} > 0) - ({arg} < 0);",
            ),
            _P(
                "is_negative",
                "whether the number is negative",
                "is_negative",
                "{res} = {arg} < 0",
                "bool {res} = ({arg} < 0);",
            ),
            _P(
                "count_set_bits",
                "the number of set bits in the binary representation",
                "count_set_bits",
                '{res} = bin(abs({arg})).count("1")',
                "int {res} = __builtin_popcount((unsigned)abs({arg}));",
            ),
        ),
    ),
    "str": QTransferTask(
        "strflags",
        "str",
        ("s", "text"),
        (
            _P(
                "is_palindrome",
                "whether the string is a palindrome",
                "is_palindrome",
                "{res} = ({arg} == {arg}[::-1])",
                "std::string {res}_r({arg}.rbegin(), {arg}.rend());\n    bool {res} = ({arg} == {res}_r);",
            ),
            _P(
                "length",
                "the length of the string",
                "length",
                "{res} = len({arg})",
                "int {res} = {arg}.size();",
            ),
            _P(
                "vowel_count",
                "the number of vowels",
                "vowel_count",
                "{res} = sum(1 for c in {arg} if c in 'aeiou')",
                'int {res} = 0;\n    for (char c : {arg}) if (std::string("aeiou").find(c) != std::string::npos) {res}++;',
            ),
            _P(
                "has_digit",
                "whether the string contains a digit",
                "has_digit",
                "{res} = any(c.isdigit() for c in {arg})",
                "bool {res} = false;\n    for (char c : {arg}) if (std::isdigit((unsigned char)c)) {res} = true;",
            ),
            _P(
                "starts_upper",
                "whether the string starts with an uppercase letter",
                "starts_upper",
                "{res} = {arg}[:1].isupper()",
                "bool {res} = (!{arg}.empty() && std::isupper((unsigned char){arg}[0]));",
            ),
            _P(
                "count_letters",
                "the number of alphabetic characters",
                "letter_count",
                "{res} = sum(1 for c in {arg} if c.isalpha())",
                "int {res} = 0;\nfor(char c : {arg}) if(isalpha((unsigned char)c)) {res}++;",
            ),
            _P(
                "count_digits",
                "the number of digit characters",
                "digit_count",
                "{res} = sum(1 for c in {arg} if c.isdigit())",
                "int {res} = 0;\nfor(char c : {arg}) if(isdigit((unsigned char)c)) {res}++;",
            ),
            _P(
                "count_spaces",
                "the number of space characters",
                "space_count",
                "{res} = {arg}.count(' ')",
                "int {res} = 0;\nfor(char c : {arg}) if(c==' ') {res}++;",
            ),
            _P(
                "count_uppercase",
                "the number of uppercase letters",
                "uppercase_count",
                "{res} = sum(1 for c in {arg} if c.isupper())",
                "int {res} = 0;\nfor(char c : {arg}) if(isupper((unsigned char)c)) {res}++;",
            ),
            _P(
                "count_lowercase",
                "the number of lowercase letters",
                "lowercase_count",
                "{res} = sum(1 for c in {arg} if c.islower())",
                "int {res} = 0;\nfor(char c : {arg}) if(islower((unsigned char)c)) {res}++;",
            ),
            _P(
                "all_lowercase",
                "whether the string has letters and all of them are lowercase",
                "is_all_lowercase",
                "{res} = {arg}.islower()",
                "bool {res}_has = false;\nbool {res}_ok = true;\nfor(char c : {arg}){ if(isupper((unsigned char)c)) {res}_ok=false; if(islower((unsigned char)c)) {res}_has=true; }\nbool {res} = {res}_has && {res}_ok;",
            ),
            _P(
                "has_space",
                "whether the string contains a space",
                "contains_space",
                "{res} = ' ' in {arg}",
                "bool {res} = {arg}.find(' ') != std::string::npos;",
            ),
            _P(
                "is_alnum",
                "whether every character is a letter or a digit",
                "is_alphanumeric",
                "{res} = {arg}.isalnum()",
                "bool {res} = !{arg}.empty();\nfor(char c : {arg}) if(!isalnum((unsigned char)c)) {res} = false;",
            ),
            _P(
                "first_is_letter",
                "whether the first character is a letter",
                "first_is_letter",
                "{res} = {arg}[0].isalpha()",
                "bool {res} = !{arg}.empty() && isalpha((unsigned char){arg}[0]);",
            ),
            _P(
                "last_char",
                "the last character",
                "last_character",
                "{res} = {arg}[-1]",
                'std::string {res} = {arg}.empty() ? std::string("") : std::string(1, {arg}.back());',
            ),
            _P(
                "word_count",
                "the number of whitespace-separated words",
                "word_count",
                "{res} = len({arg}.split())",
                "int {res} = 0;\nbool {res}_in = false;\nfor(char c : {arg}){ if(isspace((unsigned char)c)) {res}_in=false; else if(!{res}_in){ {res}_in=true; {res}++; } }",
            ),
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# trivial frontiers — target conditions the no-goal causality default already emits
# ══════════════════════════════════════════════════════════════════════════════
# The causality read-out steers an information-less C++ stub ``bool check(auto value){``. With NO
# goal signal the model defaults to a scalar sign/zero comparison (``value > 0``, ``value == 0``,
# ``value < 0``, or a boolean combination of these). Any frontier whose gold predicate IS such a
# comparison is therefore matched by that default output *regardless of transfer*, inflating both
# causality and faithfulness with false positives. Exclude these from frontier / target selection
# (they may still appear as prefilled CONTEXT — only the measured frontier must be non-trivial).
# Keyed per datatype; extend freely. Mirrors the ``is_trivial_causal_readout`` sanitizer in
# ``scripts/evaluation/eval_goalscope.py`` (target-side vs read-out-side of the same idea).
TRIVIAL_FRONTIERS: dict[str, frozenset[str]] = {
    "int": frozenset({"is_positive", "is_negative", "sign"}),
    "list[int]": frozenset(
        {"all_positive", "count_positive", "count_negative", "contains_zero"}
    ),
    "list[str]": frozenset(),
    "str": frozenset(),
}


def nontrivial_frontiers(datatype: str) -> list[Predicate]:
    """The QTASKS predicates for ``datatype`` eligible to be a *measured* frontier / target
    condition — i.e. the bank minus :data:`TRIVIAL_FRONTIERS` (the sign/zero comparisons the
    no-goal causality default trivially matches)."""
    trivial = TRIVIAL_FRONTIERS.get(datatype, frozenset())
    return [p for p in QTASKS[datatype].predicates if p.id not in trivial]


def _build_incompatible(
    implies: list[tuple[str, str]],
    mutex_groups: list[tuple[str, ...]],
) -> dict[str, list[str]]:
    """Derive a symmetric, implication-closed incompatibility mapping from authored rules.

    ``implies``      — ``(a, b)`` pairs meaning ``a ⟹ b`` (composing/contrasting a with b is redundant);
                       transitively closed. Duplicates ``a ⟺ b`` are two entries, one each way.
    ``mutex_groups`` — every pair *within* a tuple can never be true together (or is a near-duplicate we
                       want excluded); expanded to all intra-group pairs.

    Returns ``{id: sorted[ids it must NOT be paired with]}``. Reused by the multi-predicate
    (q-bank) and filter-composition compatibility maps."""
    impl: dict[str, set[str]] = {}
    for a, b in implies:
        impl.setdefault(a, set()).add(b)
    for _ in range(len(impl) + 2):  # transitive closure of ⟹
        for a in list(impl):
            for b in list(impl[a]):
                impl[a] |= impl.get(b, set())

    mutex: dict[str, set[str]] = {}

    def _m(a: str, b: str) -> None:
        if a != b:
            mutex.setdefault(a, set()).add(b)
            mutex.setdefault(b, set()).add(a)

    for grp in mutex_groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                _m(grp[i], grp[j])
    # mutex is inherited up implications: if a ⟹ x and x ⊥ y, then a ⊥ y
    for _ in range(len(impl) + 2):
        for a, xs in list(impl.items()):
            for x in list(xs):
                for y in list(mutex.get(x, ())):
                    _m(a, y)

    inc: dict[str, set[str]] = {}
    keys = set(impl) | set(mutex) | {b for _, b in implies}
    for a in keys:
        s = set(mutex.get(a, set())) | impl.get(a, set())
        s |= {b for b, bs in impl.items() if a in bs}  # reverse implications (b ⟹ a)
        s.discard(a)
        if s:
            inc[a] = sorted(s)
    return dict(sorted(inc.items()))


def _C(id, fragment, py, cpp):  # a filter condition (boolean expr over {elem})
    return Predicate(
        id=id, description=fragment, result_name="", impl={"python": py, "cpp": cpp}
    )


FILTER_TASKS: dict[str, FilterTask] = {
    "list[str]": FilterTask(
        "word_filter",
        "list[str]",
        ("words", "arr"),
        "w",
        "words",
        (
            _C(
                "palindrome",
                "are palindromes",
                "{elem} == {elem}[::-1]",
                "{elem} == std::string({elem}.rbegin(), {elem}.rend())",
            ),
            _C(
                "vowel_start",
                "start with a vowel",
                "{elem}[:1].lower() in 'aeiou'",
                '!{elem}.empty() && std::string("aeiouAEIOU").find({elem}[0]) != std::string::npos',
            ),
            _C(
                "long",
                "are longer than three characters",
                "len({elem}) > 3",
                "{elem}.size() > 3",
            ),
            _C(
                "all_upper",
                "are all uppercase letters",
                "{elem}.isupper()",
                "!{elem}.empty() && std::all_of({elem}.begin(), {elem}.end(), [](unsigned char c){ return std::isupper(c); })",
            ),
            _C(
                "all_lower",
                "are all lowercase letters",
                "{elem}.islower()",
                "!{elem}.empty() && std::all_of({elem}.begin(), {elem}.end(), [](unsigned char c){ return std::islower(c); })",
            ),
            _C(
                "capitalized",
                "start with an uppercase letter",
                "{elem}[:1].isupper()",
                "!{elem}.empty() && std::isupper((unsigned char){elem}[0])",
            ),
            _C(
                "short",
                "are shorter than four characters",
                "len({elem}) < 4",
                "{elem}.size() < 4",
            ),
            _C(
                "length_five",
                "have exactly five characters",
                "len({elem}) == 5",
                "{elem}.size() == 5",
            ),
            _C(
                "contains_digit",
                "contain a digit",
                "any(c.isdigit() for c in {elem})",
                "std::any_of({elem}.begin(), {elem}.end(), [](unsigned char c){ return std::isdigit(c); })",
            ),
            _C(
                "all_alpha",
                "consist only of letters",
                "{elem}.isalpha()",
                "!{elem}.empty() && std::all_of({elem}.begin(), {elem}.end(), [](unsigned char c){ return std::isalpha(c); })",
            ),
            _C(
                "ends_s",
                "end with the letter s",
                "{elem}.endswith('s')",
                "!{elem}.empty() && {elem}.back() == 's'",
            ),
            _C(
                "contains_e",
                "contain the letter e",
                "'e' in {elem}",
                "{elem}.find('e') != std::string::npos",
            ),
            _C(
                "starts_a",
                "start with the letter a",
                "{elem}.startswith('a')",
                "!{elem}.empty() && {elem}.front() == 'a'",
            ),
            _C(
                "has_double",
                "contain a repeated adjacent character",
                "any({elem}[i] == {elem}[i+1] for i in range(len({elem})-1))",
                "std::adjacent_find({elem}.begin(), {elem}.end()) != {elem}.end()",
            ),
            _C(
                "last_is_digit",
                "end with a digit",
                "{elem}[-1:].isdigit()",
                "!{elem}.empty() && std::isdigit((unsigned char){elem}.back())",
            ),
            _C(
                "same_reversed",
                "are unchanged when reversed",
                "{elem} == {elem}[::-1]",
                "{elem} == std::string({elem}.rbegin(), {elem}.rend())",
            ),
        ),
    ),
    "list[int]": FilterTask(
        "num_filter",
        "list[int]",
        ("nums", "arr"),
        "x",
        "values",
        (
            _C("even", "are even", "{elem} % 2 == 0", "{elem} % 2 == 0"),
            _C("positive", "are positive", "{elem} > 0", "{elem} > 0"),
            _C("big", "are greater than ten", "{elem} > 10", "{elem} > 10"),
            _C("odd", "are odd", "{elem} % 2 != 0", "{elem} % 2 != 0"),
            _C("negative", "are negative", "{elem} < 0", "{elem} < 0"),
            _C("non_negative", "are zero or positive", "{elem} >= 0", "{elem} >= 0"),
            _C(
                "multiple_of_three",
                "are multiples of three",
                "{elem} % 3 == 0",
                "{elem} % 3 == 0",
            ),
            _C(
                "multiple_of_five",
                "are multiples of five",
                "{elem} % 5 == 0",
                "{elem} % 5 == 0",
            ),
            _C(
                "multiple_of_ten",
                "are multiples of ten",
                "{elem} % 10 == 0",
                "{elem} % 10 == 0",
            ),
            _C(
                "divisible_by_four",
                "are divisible by four",
                "{elem} % 4 == 0",
                "{elem} % 4 == 0",
            ),
            _C(
                "single_digit",
                "have a single digit in absolute value",
                "-9 <= {elem} <= 9",
                "{elem} >= -9 && {elem} <= 9",
            ),
            _C(
                "two_digit",
                "have exactly two digits in absolute value",
                "10 <= abs({elem}) <= 99",
                "abs({elem}) >= 10 && abs({elem}) <= 99",
            ),
            _C(
                "in_range_ten_hundred",
                "are between ten and one hundred inclusive",
                "10 <= {elem} <= 100",
                "{elem} >= 10 && {elem} <= 100",
            ),
            _C(
                "abs_over_fifty",
                "are greater than fifty in absolute value",
                "abs({elem}) > 50",
                "abs({elem}) > 50",
            ),
            _C(
                "perfect_square",
                "are perfect squares",
                "{elem} >= 0 and __import__('math').isqrt({elem}) ** 2 == {elem}",
                "[&]{ if ({elem} < 0) return false; long r = (long)llround(sqrt((double){elem})); return r * r == (long){elem}; }()",
            ),
            _C(
                "is_prime",
                "are prime numbers",
                "{elem} > 1 and all({elem} % d != 0 for d in range(2, int({elem} ** 0.5) + 1))",
                "[&]{ if ({elem} < 2) return false; for (int d = 2; (long)d * d <= {elem}; ++d) if ({elem} % d == 0) return false; return true; }()",
            ),
        ),
    ),
}


def _seq_letter(order, progress, pred_id) -> str:
    """``letters`` naming is SEQUENTIAL (by appearance): prefilled lines are lettered a, b, c, ... in the
    order they are WRITTEN (list order restricted to the prefilled set) and the frontier stub takes the
    next letter. Never positional (spec-slot): a slot letter names the frontier's list position in the
    stub, e.g. order-diff SRC [A,C,B] with {A,B} prefilled would render ``a=.. c=.. b =``. Identical to
    slot naming whenever the prefill is a prefix of the list."""
    written = [p.id for p in order if p.id in set(progress)]
    return chr(ord("a") + (written.index(pred_id) if pred_id in written else len(written)))


# ══════════════════════════════════════════════════════════════════════════════
# diff-trigger q-transfer — transplant an ABSTRACT "advance past completed sub-tasks" query
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DiffTriggerInstance:
    """Q-transfer where SRC and DST share two sub-tasks {A, B} but each has a UNIQUE third one
    (SRC has C, DST has D). Transplanting SRC's pivot goal-head query into DST tests whether the
    query carries the DIFF *relative to progress* (abstract/relational) rather than the literal
    next sub-task::

        order-matched:  SRC.task = [A, B, C], prefill {A, B}  (SRC frontier = C)
                        DST.task = [A, B, D], prefill {A}      (DST native  = B)
        order-diff:     SRC.task = [A, C, B], prefill {A, B}  (SRC frontier = C, listed 2nd)
                        DST.task = [A, B, D], prefill {A}      (DST native  = B; D listed 3rd)

    Transplant SRC's pivot query into DST and read what DST writes:
      • **D** -> the query carried an abstract ``{A,B} done, do the remaining`` DIFF, resolved
        against DST's own list -> D (the DIFF hypothesis).
      • **B** -> no transfer (DST keeps its native frontier).
      • **C** -> literal content copy (SRC's target verbatim, though C is not in DST's spec).
    ``order-matched`` confounds D with a positional "do the 3rd-listed" copy (C and D are both 3rd);
    ``order-diff`` breaks that — C is 2nd in SRC, D is 3rd in DST, so a positional copy predicts B.

    Scoring reuses :func:`src.nrgs.eval.eval_q_transfer` / :func:`judge_causal`: ``expected_next_desc``
    = D (the DIFF-predicted DST target = "triggered"), ``dst_next_desc`` = B (native), ``task_desc`` =
    the DST task; ``src_target_desc`` = C is exposed for a separate literal-copy check. Gaze uses the
    DST task predicates via the ``task`` property. Prompt: the listed task, independent assignment lines named by
    sequential letters (:func:`_seq_letter`), and the frontier's ``{var} =`` stub as the pivot (single-position q
    transplant)."""

    datatype: str
    src_order: tuple[Predicate, ...]   # SRC listed predicates (contains A, B, C)
    dst_order: tuple[Predicate, ...]   # DST listed predicates (contains A, B, D)
    src_progress: list[str]            # prefilled in SRC  ([A, B])
    dst_progress: list[str]            # prefilled in DST  ([A])
    diff_target: str                   # D id — DIFF-predicted DST outcome (scoring target)
    dst_native: str                    # B id — DST's native next
    src_target: str                    # C id — SRC transplant frontier / literal-copy reference
    var_name: str
    src_lang: Lang
    dst_lang: Lang
    meta: dict = field(default_factory=dict)

    def _parts(self, lang, order, progress):
        q = _question([p.description for p in order], self.datatype, self.var_name, lang, "listed")
        prog = [p for p in order if p.id in set(progress)]          # prefilled lines, in list order
        lines = []
        for p in prog:
            rn = _seq_letter(order, progress, p.id)
            ln = _assign_line(p, self.var_name, lang, res_name=rn)
            if lang == "cpp":
                ln = _cpp_autoify(ln, rn)
            lines.append(ln)
        base = ("<code>\n" + _signature(self.datatype, self.var_name, lang)
                + ("\n" + "\n".join(lines) if lines else ""))
        # eq pivot: emit the frontier's ``{var} =`` stub so the query is read at the ``=`` token
        # (the strong goal-head pivot). Frontier = the first un-prefilled predicate in list order.
        frontier = next(p for p in order if p.id not in set(progress))
        rn = _seq_letter(order, progress, frontier.id)
        stub = f"auto {rn} =" if lang == "cpp" else f"{rn} ="
        return q, base + f"\n    {stub}"

    def src_parts(self):
        return self._parts(self.src_lang, self.src_order, self.src_progress)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_order, self.dst_progress)

    def _desc(self, pred_id):
        for p in (*self.src_order, *self.dst_order):   # C only in src_order, D only in dst_order
            if p.id == pred_id:
                return p.description
        raise KeyError(pred_id)

    @property
    def expected_next(self) -> str:      # DIFF-predicted DST target (transplant "success")
        return self.diff_target

    @property
    def expected_next_desc(self) -> str:
        return self._desc(self.diff_target)

    @property
    def dst_next_desc(self) -> str:
        return self._desc(self.dst_native)

    @property
    def src_target_desc(self) -> str:    # SRC's literal transplant frontier (C) — literal-copy check
        return self._desc(self.src_target)

    @property
    def task_desc(self) -> str:
        return "; ".join(p.description for p in self.dst_order)

    @property
    def task(self):                      # gaze: DST task predicates -> their description spans in dst_prompt
        from types import SimpleNamespace
        return SimpleNamespace(predicates=self.dst_order)


# Semantic predicate families — used ONLY by the diff-trigger q-transfer to avoid sampling A/B/C/D
# sets with overlapping sub-tasks (e.g. two "length" ops, or even/odd complements), which confound the
# "did DST switch to D vs stay on B" test. See agent_reports/claude/analysis/appI.2__diff_trigger_predicate_families.md.
# NOT used by the other setups (the overlap issue is minor there). Any predicate id not listed falls
# back to a singleton family (its own id).
DIFF_TRIGGER_FAMILIES: dict[str, dict[str, list[str]]] = {
    "int": {
        "decimal_digits": ["digit_count", "digit_sum", "last_digit", "is_palindrome", "is_two_digit"],
        "parity": ["is_even", "is_odd"],
        "divisibility": ["is_multiple_of_3"],
        "sign_magnitude": ["is_positive", "is_negative", "sign", "absolute_value"],
        "mult_structure": ["is_prime", "is_perfect_square", "is_power_of_two"],
        "bits": ["count_set_bits"],
    },
    "str": {
        "charclass_count": ["vowel_count", "count_letters", "count_digits", "count_spaces",
                            "count_uppercase", "count_lowercase"],
        "charclass_bool": ["has_digit", "has_space", "is_alnum", "all_lowercase", "starts_upper",
                           "first_is_letter"],
        "length_words": ["length", "word_count"],
        "palindrome": ["is_palindrome"],
        "last_char": ["last_char"],
    },
    "list[int]": {
        "parity": ["has_even", "count_even", "count_odd"],
        "sign": ["count_positive", "count_negative", "all_positive", "contains_zero"],
        "extremes": ["max_val", "min_val", "value_range"],
        "aggregate": ["total_sum", "product_all", "mean_int"],
        "uniq_order": ["has_duplicate", "distinct_count", "is_sorted"],
    },
    "list[str]": {
        "word_length": ["total_length", "count_long", "count_short", "longest_word_length",
                        "shortest_word_length", "all_unique_lengths"],
        "vowel_consonant": ["count_starts_vowel", "count_ends_consonant", "total_vowels"],
        "case": ["count_capitalized", "count_all_lowercase", "total_uppercase_letters"],
        "content": ["count_with_digit", "has_palindrome", "any_empty", "distinct_count"],
    },
}


def _diff_family(datatype: str, pred_id: str) -> str:
    """Family of a predicate for the diff-trigger constraint; unmapped ids -> their own id."""
    for fam, members in DIFF_TRIGGER_FAMILIES.get(datatype, {}).items():
        if pred_id in members:
            return fam
    return pred_id


def diff_trigger_setups(
    datatype: str,
    variant: Literal["order_matched", "order_diff", "c_in_dst"] = "order_matched",
    n_samples: int = 32,
    seed: int = 0,
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
) -> list[DiffTriggerInstance]:
    """Sample ``n_samples`` diff-trigger instances (:class:`DiffTriggerInstance`). Each draws 4
    pairwise-compatible predicates A, B, C, D from the datatype's :data:`QTASKS` bank and builds the
    shared-{A,B} contrast. ``variant="order_matched"`` lists SRC as [A,B,C] (C in the 3rd slot, like
    DST's D); ``variant="order_diff"`` lists SRC as [A,C,B] (C in the 2nd slot) to dissociate the DIFF
    outcome (D) from a positional "copy the 3rd-listed" account.

    ``variant="c_in_dst"`` is the *target-present* setup: BOTH sides list all four predicates so that
    SRC's transplant target C is also available in DST — SRC=[A,B,C,D] prefill{A,B} (frontier C),
    DST=[A,B,D,C] prefill{A} (native B). Transplanting SRC's pivot query then has three legible
    outcomes: **B** (no-op, DST's native next), **C** (query is content-addressed / "already knows what
    it's looking for" — C now exists in DST), or **D** (query carries only progress ⟨A,B⟩, so DST's
    slot after ⟨A,B⟩ is D = the DIFF prediction). Scored via the usual roles: causal→D, literal→C,
    native→B. Deduped by the ordered 4-tuple; reproducible via ``seed``."""
    assert variant in ("order_matched", "order_diff", "c_in_dst")
    bank = QTASKS[datatype]
    preds = list(bank.predicates)
    assert len(preds) >= 4, f"need >=4 predicates in the {datatype!r} bank, have {len(preds)}"
    trivial = TRIVIAL_FRONTIERS.get(datatype, frozenset())
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 400):
        if len(out) >= n_samples:
            break
        A, B, C, D = rng.sample(preds, 4)
        quad = (A.id, B.id, C.id, D.id)
        if quad in seen:
            continue
        # diff-trigger overlap controls (scoped to this experiment):
        #  - C (SRC target) and D (DST remaining) must be non-trivial frontiers, and
        #  - B (native), C, D must come from three DIFFERENT semantic families,
        # so "did DST switch to D vs stay on B" is not confounded by look-alike sub-tasks.
        if C.id in trivial or D.id in trivial:
            continue
        if len({_diff_family(datatype, B.id),
                                      _diff_family(datatype, C.id),
                                      _diff_family(datatype, D.id)}) < 3:
            continue
        if not all(qpred_compatible(datatype, a.id, b.id)
                   for i, a in enumerate((A, B, C, D))
                   for b in (A, B, C, D)[i + 1:]):
            continue
        seen.add(quad)
        if variant == "c_in_dst":
            # target-present: both sides list all four; SRC target C is also in DST (last slot).
            src_order = (A, B, C, D)
            dst_order = (A, B, D, C)
        else:
            src_order = (A, B, C) if variant == "order_matched" else (A, C, B)
            dst_order = (A, B, D)
        out.append(DiffTriggerInstance(
            datatype=datatype,
            src_order=src_order,
            dst_order=dst_order,
            src_progress=[A.id, B.id],
            dst_progress=[A.id],
            diff_target=D.id,
            dst_native=B.id,
            src_target=C.id,
            var_name=bank.var_names[0],
            src_lang=src_lang, dst_lang=dst_lang,
            meta={"datatype": datatype, "variant": variant,
                  "dst_next": B.id, "diff_target": D.id, "src_target": C.id},
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# composition rules — when two filter sub-tasks may NOT be composed
# ══════════════════════════════════════════════════════════════════════════════
# A composed-filter instance pairs two *commutative, independent* sub-tasks. A pair is DISALLOWED when
# composing it is degenerate — either the two conditions cannot both hold, or one trivially entails the
# other (so the composition is redundant and often shares wording). Authored below as a readable
# source-of-truth; ``FILTER_INCOMPATIBLE`` is the derived symmetric ``{condition_id: [ids it must not be
# composed with]}`` mapping (implication-closed) that :func:`filter_compatible` consults.
#
# Three flavours:
#   (i)   MUTUALLY EXCLUSIVE  — can never be true together        (all_upper / all_lower)
#   (ii)  IMPLICATION / DUP   — a ⟹ b, so composing is redundant   (starts_a ⟹ vowel_start)
#   (iii) CONTRADICTION       — a special case of (i)             (all_alpha / contains_digit)

# a ⟹ b  (composing a with b is redundant; also covers duplicates a ⟺ b as one entry each way)
_FILTER_IMPLIES: list[tuple[str, str]] = [
    # ── list[str] ──
    ("same_reversed", "palindrome"),
    ("palindrome", "same_reversed"),  # identical meaning
    ("starts_a", "vowel_start"),  # 'a' is a vowel (+ shared "start with" phrasing)
    ("all_upper", "capitalized"),  # all-caps ⟹ starts with an uppercase letter
    ("length_five", "long"),  # exactly five ⟹ longer than three
    ("last_is_digit", "contains_digit"),  # ends with a digit ⟹ contains a digit
    # ── list[int] ──
    ("positive", "non_negative"),
    ("in_range_ten_hundred", "positive"),
    ("multiple_of_ten", "multiple_of_five"),
    ("multiple_of_ten", "even"),
    ("divisible_by_four", "even"),
]

# mutually-exclusive pairs/groups (every pair within a tuple can never be true together)
_FILTER_MUTEX: list[tuple[str, ...]] = [
    # ── list[str] ──
    ("short", "long"),
    ("length_five", "short"),
    ("all_upper", "all_lower"),
    ("all_lower", "capitalized"),
    ("all_alpha", "contains_digit"),
    ("all_alpha", "last_is_digit"),
    ("ends_s", "last_is_digit"),
    # ── list[int] ──
    ("even", "odd"),
    ("positive", "negative"),
    ("negative", "non_negative"),
    ("single_digit", "two_digit"),
    ("single_digit", "big"),
    ("single_digit", "in_range_ten_hundred"),
    ("single_digit", "abs_over_fifty"),
    ("big", "negative"),
    ("in_range_ten_hundred", "negative"),
    ("odd", "multiple_of_ten"),
    ("odd", "divisible_by_four"),
    ("perfect_square", "is_prime"),
    ("perfect_square", "negative"),
    ("is_prime", "negative"),
]

def _build_filter_incompatible() -> dict[str, list[str]]:
    """Derive the symmetric, implication-closed incompatibility mapping for filter conditions."""
    return _build_incompatible(_FILTER_IMPLIES, _FILTER_MUTEX)


#: ``{condition_id: [ids it must NOT be composed with]}`` — consulted by :func:`filter_compatible`.
FILTER_INCOMPATIBLE: dict[str, list[str]] = _build_filter_incompatible()


# ══════════════════════════════════════════════════════════════════════════════
# multi-predicate compatibility — when two q-bank predicates are too similar to CONTRAST
# ══════════════════════════════════════════════════════════════════════════════
# The multi-predicate template (:class:`MultiPredicateInstance`) contrasts two frontier sub-goals
# (T3 vs T4) sampled from a datatype's :data:`QTASKS` bank. A contrast is DEGENERATE when one frontier
# is redundant with / implied by the other (so "did it do T3 not T4?" is ill-posed) — e.g.
# ``value_range = max - min`` overlaps ``max_val``; ``is_two_digit`` ⟹ ``digit_count == 2``. Unlike the
# filter rules, *opposites* (max_val vs min_val, count_even vs count_odd) are LEGITIMATE, clearly
# distinguishable contrasts and are NOT excluded. Keyed per-datatype (q-bank ids repeat across
# datatypes, e.g. ``distinct_count``). Authored as a readable source-of-truth; extend freely.

# a ⟹ b  (contrasting a against b is redundant — knowing a's computation largely determines b's)
_QPRED_IMPLIES: dict[str, list[tuple[str, str]]] = {
    "list[str]": [
        (
            "count_capitalized",
            "total_uppercase_letters",
        ),  # capitalized words ⟹ uppercase letters present
    ],
    "list[int]": [
        ("value_range", "max_val"),
        ("value_range", "min_val"),  # range = max - min
        ("mean_int", "total_sum"),
        ("total_sum", "mean_int"),  # mean = sum / n (mutually determining)
        ("all_positive", "count_positive"),  # all-positive ⟹ count == len
    ],
    "int": [
        ("is_positive", "sign"),
        ("is_negative", "sign"),  # sign encodes positivity/negativity
        ("is_two_digit", "digit_count"),  # two-digit ⟹ digit_count == 2
        ("is_power_of_two", "count_set_bits"),  # power-of-two ⟺ exactly one set bit
    ],
    "str": [
        ("has_digit", "count_digits"),  # has a digit ⟺ digit count > 0
        ("has_space", "count_spaces"),  # has a space ⟺ space count > 0
        ("starts_upper", "count_uppercase"),  # starts uppercase ⟹ ≥1 uppercase
        ("starts_upper", "first_is_letter"),  # uppercase start ⟹ first char is a letter
        (
            "all_lowercase",
            "count_lowercase",
        ),  # all-lowercase overlaps the lowercase count
    ],
}

# near-duplicate pairs to exclude that aren't a clean implication (rare — opposites are kept)
_QPRED_MUTEX: dict[str, list[tuple[str, ...]]] = {
    "list[str]": [],
    "list[int]": [
        ("has_even", "count_even")
    ],  # has-an-even ⟺ even-count > 0 (same signal)
    "int": [],
    "str": [],
}

#: ``{datatype: {pred_id: [ids too similar to CONTRAST with]}}`` — consulted by ``multi_predicate_setups``.
QPRED_INCOMPATIBLE: dict[str, dict[str, list[str]]] = {
    dt: _build_incompatible(_QPRED_IMPLIES.get(dt, []), _QPRED_MUTEX.get(dt, []))
    for dt in QTASKS
}


def qpred_compatible(datatype: str, a_id: str, b_id: str) -> bool:
    """True iff two q-bank predicates form a NON-degenerate contrast (not redundant / mutually implied)."""
    return b_id not in QPRED_INCOMPATIBLE.get(datatype, {}).get(a_id, ())


def filter_compatible(a_id: str, b_id: str) -> bool:
    """True iff the two filter sub-tasks may be composed (not mutually exclusive / redundant)."""
    return b_id not in FILTER_INCOMPATIBLE.get(a_id, ())


def multi_predicate_setups(
    datatype: str,
    n_conditions: "int | list[int]" = 3,
    n_samples: int = 8,
    seed: int = 0,
    presentation: Presentation = "listed",
    naming: Literal["natural", "letters"] = "natural",
    letter_stub: Literal["indent", "var_eq"] = "indent",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
) -> list[MultiPredicateInstance]:
    """Sample ``n_samples`` multi-predicate divergent-frontier instances (:class:`MultiPredicateInstance`).

    Each instance draws ``k + 1`` DISTINCT, pairwise-COMPATIBLE predicates from the datatype's
    :data:`QTASKS` bank (compatible = a non-degenerate contrast, see :func:`qpred_compatible`) — ``k`` is
    ``n_conditions`` if an int, else a per-instance random pick from the ``n_conditions`` list (task SIZE
    mixes over, e.g. {2, 3}, without multiplying configs). The first ``k - 1`` are the SHARED prefix
    prefilled in both sides, and the last two are the SRC and DST frontiers. With ``k=3``: shared =
    (T1, T2), SRC frontier = T3, DST frontier = T4 — SRC lists (T1, T2, T3) & DST lists (T1, T2, T4), both
    prefill (T1, T2). The test: after transplanting SRC's pivot query, does clean DST write T3 instead of T4?

    Deduped by the ORDERED frontier concept pair (src_frontier, dst_frontier), so ``n_samples`` counts
    distinct frontier contrasts; reproducible via ``seed`` (fixed draw sequence — resume-stable).

    ``letter_stub`` controls the pivot shape (see :class:`MultiPredicateInstance`): "indent" (default)
    ends the prefill at the trailing indent; "var_eq" additionally prefills the frontier's variable name
    up to ``=`` (with ``k=4``: python ``    d =`` / c++ ``    auto d =``), so the model writes only the
    frontier's VALUE. Intended with ``naming="letters"``."""
    bank = QTASKS[datatype]
    preds = list(bank.predicates)
    ks = [n_conditions] if isinstance(n_conditions, int) else list(n_conditions)
    assert ks and all(k >= 2 for k in ks), (
        "n_conditions must be >= 2 (>=1 shared predicate + the divergent frontier)"
    )
    assert all(k + 1 <= len(preds) for k in ks), (
        f"n_conditions+1 (max {max(ks) + 1}) exceeds bank size ({len(preds)})"
    )
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 400):
        if len(out) >= n_samples:
            break
        k = rng.choice(ks)  # task size for this instance
        n_pick = k + 1  # (k - 1) shared + 2 frontiers
        n_shared = k - 1
        chosen = rng.sample(preds, n_pick)
        # every pair in the chosen set must be a non-degenerate contrast (shared incl.) — avoids a
        # prefilled predicate that is redundant with a frontier, and a degenerate T3-vs-T4 contrast
        if not all(
            qpred_compatible(datatype, a.id, b.id)
            for i, a in enumerate(chosen)
            for b in chosen[i + 1 :]
        ):
            continue
        shared = tuple(chosen[:n_shared])
        src_frontier, dst_frontier = chosen[n_shared], chosen[n_shared + 1]
        ckey = (src_frontier.id, dst_frontier.id)  # dedup unit = ordered frontier pair
        if ckey in seen:
            continue
        seen.add(ckey)
        out.append(
            MultiPredicateInstance(
                datatype=datatype,
                shared=shared,
                src_frontier=src_frontier,
                dst_frontier=dst_frontier,
                var_name=bank.var_names[0],
                presentation=presentation,
                src_lang=src_lang,
                dst_lang=dst_lang,
                naming=naming,
                letter_stub=letter_stub,
                meta={
                    "datatype": datatype,
                    "dst_next": dst_frontier.id,
                    "n_conditions": k,
                    "letter_stub": letter_stub,
                },
            )
        )
    return out
