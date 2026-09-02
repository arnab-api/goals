"""Data generation for the DIFF-hypothesis experiments (q-transfer / kv-transfer).

Two contrastive "specs", one per transplant mechanism:

Spec-1 — QUERY transfer (``01`` A-experiments style).
    ``SRC.task == DST.task``, ``SRC.progress != DST.progress``. One function-writing task
    made of several INDEPENDENT, COMMUTATIVE sub-predicates. A different subset is prefilled
    as completed PROGRESS in the source vs the destination, so at the shared pivot the source
    is "about to write" a different next sub-predicate than the destination. The intervention
    transplants the source pivot QUERY; test = does DST now pursue SRC's next sub-goal?

Spec-2 — KEY+VALUE transfer (``00_diff`` Spec-2 style).
    ``SRC.task != DST.task``, ``SRC.progress == DST.progress``. The two tasks are distinct but
    share a COMMON PROGRESS (a prerequisite sub-computation), so both prefill the same code.
    Tasks are therefore CLUSTERED by their common progress. The intervention transplants the
    source TASK-DESCRIPTION key+value onto the destination; test = does DST now implement
    SRC's task?

Generalization axes (both specs):
    • datatype     — int | str | list[int] | list[str]
    • presentation — "description" (prose task) | "listed" (numbered multi-predicate)
    • language     — python | cpp, independently for src and dst (cross-language transfer:
                     cache the task/progress in one language, trigger it in the other).

Predicate implementations are templates over three placeholders:
    {arg}  = the function's parameter name (chosen from the task/cluster's ``var_names``)
    {res}  = the result variable name of THIS predicate            (q-transfer / kv common progress)
    {prog} = the common-progress result variable name              (kv-transfer tasks only)

Prompts are materialized with the chat template via :func:`src.diff.functional.get_prompt`
(imported lazily so plain dataset inspection needs no model). Each rendered instance carries
the exact SRC/DST prompt strings plus the pivot / task-span metadata the intervention needs.
"""

from __future__ import annotations

import dataclasses
import itertools
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
class KVCluster:
    """A family of distinct tasks sharing one common-progress prerequisite."""

    id: str
    datatype: str
    var_names: tuple[str, ...]
    common_progress: Predicate  # binds {prog}
    tasks: tuple[Predicate, ...]  # each is an EXPRESSION using {prog}


@dataclass(frozen=True)
class FilterTask:
    """A single filter that COMPOSES independent, commutative element-wise conditions inside ONE
    return expression, e.g. "all the words that are palindromes and start with a vowel". Unlike
    :class:`QTransferTask` (one assignment per sub-goal, statement-boundary pivot), the pivot here
    sits MID-EXPRESSION — right after ``… if <cond> and`` — so the model completes the same
    comprehension by adding the next condition. This is the original Spec-1 framing (00_diff TASK_1).

    Each condition is a boolean template over ``{elem}`` (the loop/comprehension variable) and, if
    needed, ``{arg}``; ``description`` is the NL fragment ("are palindromes")."""

    id: str
    datatype: str  # list[str] | list[int] | str (chars)
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
    tail = (
        "\nAssume that all the libraries have been already imported.\n"
        if lang == "cpp"
        else "\n"
    )
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


# ══════════════════════════════════════════════════════════════════════════════
# instances
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QTransferInstance:
    """A single Spec-1 contrastive setup (same task, different progress)."""

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


@dataclass
class KVTransferInstance:
    """A single Spec-2 contrastive setup (different task, same common progress)."""

    cluster: KVCluster
    var_name: str
    presentation: Presentation
    src_lang: Lang
    dst_lang: Lang
    src_task: str  # task id for SRC
    dst_task: str  # task id for DST
    pivot: Literal["return", "indent"] = "return"
    meta: dict = field(default_factory=dict)

    def _task(self, task_id: str) -> Predicate:
        return next(t for t in self.cluster.tasks if t.id == task_id)

    def task_text(self, task_id: str) -> str:
        """The task-description substring as it appears in the question (span anchor)."""
        return self._task(task_id).description

    def _parts(self, lang: Lang, task_id: str) -> tuple[str, str]:
        prog = self.cluster.common_progress.result_name
        q = _question(
            [self.task_text(task_id)],
            self.cluster.datatype,
            self.var_name,
            lang,
            self.presentation,
        )
        prog_line = _assign_line(
            self.cluster.common_progress, self.var_name, lang, prog=prog
        )
        base = (
            "<code>\n"
            + _signature(self.cluster.datatype, self.var_name, lang)
            + "\n"
            + prog_line
        )
        if self.pivot == "return":
            prefill = base + "\n    return "  # model writes the task EXPRESSION
        else:
            prefill = base + "\n    "  # trailing indent
        return q, prefill

    def src_parts(self):
        return self._parts(self.src_lang, self.src_task)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_task)

    @property
    def expected_next_desc(self) -> str:
        return self.task_text(self.src_task)


def _vec_decl(dt: str, name: str) -> str:
    return {
        "list[str]": f"std::vector<std::string> {name};",
        "list[int]": f"std::vector<int> {name};",
        "str": f"std::string {name};",
    }[dt]


@dataclass
class FilterTransferInstance:
    """A single composed-filter contrastive setup (00_diff TASK_1 style). The task is TWO
    independent, COMMUTATIVE filter sub-tasks (e.g. T1 = palindromes, T2 = starts-with-a-vowel).
    SRC and DST share the same task but prefill DIFFERENT single sub-tasks as a first ``res = […]``
    filter, then hit a common ``postfix`` pivot — so SRC is about to apply its remaining sub-task
    and DST its remaining (the OTHER one):

        SRC: res = [w for w in words if <src_fill>] ; postfix   -> SRC about to apply expected_next
        DST: res = [w for w in words if <dst_fill>] ; postfix   -> DST about to apply dst_next

    ``pair`` is the two sub-tasks in play (defines the shared task text). Query-transfer (Spec-1)
    mechanically — same intervention as :class:`QTransferInstance`.

    ``postfix``: "return" (loose — bare ``return``) or "targeted" (a filter opener that forces the
    next token to be a condition: python ``return [w for w in words if``; c++ a fresh filtered loop)."""

    task: FilterTask
    pair: tuple[str, str]  # the two commutative sub-task (condition) ids in this task
    var_name: str
    src_lang: Lang
    dst_lang: Lang
    src_fill: str  # condition id prefilled in the SRC res-line
    dst_fill: str  # condition id prefilled in the DST res-line
    expected_next: (
        str  # SRC's remaining sub-task (transplant target) = other(pair, src_fill)
    )
    postfix: Literal["return", "targeted"] = "targeted"
    meta: dict = field(default_factory=dict)

    def _cond(self, cid: str) -> Predicate:
        return next(c for c in self.task.conditions if c.id == cid)

    @property
    def task_text(self) -> str:
        """The shared NL phrase over the TWO sub-tasks — the TASK span anchor."""
        frags = " and ".join(self._cond(cid).description for cid in self.pair)
        return f"all the {self.task.noun} in `{self.var_name}` that {frags}"

    def _postfix(self, lang: Lang) -> str:
        elem, dt = self.task.elem, self.task.datatype
        if self.postfix == "return":
            return "return "
        if lang == "python":
            return f"return [{elem} for {elem} in res if "  # filter the T1-result -> T1 & T2
        return (
            _vec_decl(dt, "out") + f"\n    for (auto& {elem} : res) if"
        )  # c++ targeted: filter the T1-result (`res`) -> T1 & T2

    def _parts(self, lang: Lang, fill_id: str) -> tuple[str, str]:
        elem, arg, dt = self.task.elem, self.var_name, self.task.datatype
        q = _question([self.task_text], dt, arg, lang, "description")
        cond = _fmt(self._cond(fill_id).impl[lang], arg=arg, elem=elem)
        sig = _signature(dt, arg, lang)
        if lang == "python":
            res_line = f"    res = [{elem} for {elem} in {arg} if {cond}]"
        else:  # c++ res built by a loop
            res_line = (
                f"    {_vec_decl(dt, 'res')}\n"
                f"    for (auto& {elem} : {arg}) if ({cond}) res.push_back({elem});"
            )
        prefill = f"<code>\n{sig}\n{res_line}\n    {self._postfix(lang)}"
        return q, prefill  # pivot = trailing token of the postfix

    def src_parts(self):
        return self._parts(self.src_lang, self.src_fill)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_fill)

    @property
    def expected_next_desc(self) -> str:
        return self._cond(self.expected_next).description


def materialize(instance, ctx=None) -> dict:
    """Apply the chat template to an instance's src/dst (question, prefill) parts.

    Returns ``{"src_prompt", "dst_prompt", ...}`` (+ ``src_task_text``/``dst_task_text`` for kv,
    ``task_text`` for filter). ``ctx`` is a :class:`src.diff.functional.Ctx` (else module-global
    ``mt``). Imported lazily so dataset inspection needs no model."""
    from src.diff.functional import get_prompt

    qs, pfs = instance.src_parts()
    qd, pfd = instance.dst_parts()
    out = {
        "src_prompt": get_prompt(qs, pfs, ctx=ctx),
        "dst_prompt": get_prompt(qd, pfd, ctx=ctx),
    }
    if isinstance(instance, KVTransferInstance):
        out["src_task_text"] = instance.task_text(instance.src_task)
        out["dst_task_text"] = instance.task_text(instance.dst_task)
    elif isinstance(instance, FilterTransferInstance):
        out["task_text"] = instance.task_text
    return out


# ══════════════════════════════════════════════════════════════════════════════
# predicate banks — one q-task + one kv-cluster per datatype
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
        ),
    ),
}

KVCLUSTERS: dict[str, KVCluster] = {
    "list[str]": KVCluster(
        "sorted_by_len",
        "list[str]",
        ("words", "arr"),
        _P(
            "sort_by_len",
            "the words sorted by length",
            "sorted_words",
            "{prog} = sorted({arg}, key=len)",
            "std::vector<std::string> {prog} = {arg};\n    std::stable_sort({prog}.begin(), {prog}.end(), [](const std::string& a, const std::string& b){ return a.size() < b.size(); });",
        ),
        (
            _P(
                "shortest_vowel_start",
                "returns true if the shortest word starts with a vowel",
                "",
                "{prog}[0][0].lower() in 'aeiou'",
                '!{prog}.empty() && std::string("aeiouAEIOU").find({prog}[0][0]) != std::string::npos',
            ),
            _P(
                "longest_palindrome",
                "returns true if the longest word is a palindrome",
                "",
                "{prog}[-1] == {prog}[-1][::-1]",
                "[&]{ auto& w = {prog}.back(); std::string rv(w.rbegin(), w.rend()); return w == rv; }()",
            ),
            _P(
                "longest_len",
                "the length of the longest word",
                "",
                "len({prog}[-1])",
                "(int){prog}.back().size()",
            ),
            _P(
                "shortest_len",
                "the length of the shortest word",
                "",
                "len({prog}[0])",
                "(int){prog}.front().size()",
            ),
        ),
    ),
    "list[int]": KVCluster(
        "sorted_asc",
        "list[int]",
        ("nums", "arr"),
        _P(
            "sort_asc",
            "the values sorted in ascending order",
            "sorted_nums",
            "{prog} = sorted({arg})",
            "std::vector<int> {prog} = {arg};\n    std::sort({prog}.begin(), {prog}.end());",
        ),
        (
            _P(
                "median_is_even",
                "returns true if the median value is even",
                "",
                "{prog}[len({prog})//2] % 2 == 0",
                "{prog}[{prog}.size()/2] % 2 == 0",
            ),
            _P(
                "range_span",
                "the difference between the largest and smallest value",
                "",
                "{prog}[-1] - {prog}[0]",
                "{prog}.back() - {prog}.front()",
            ),
            _P(
                "second_largest",
                "the second largest value",
                "",
                "{prog}[-2]",
                "{prog}[{prog}.size()-2]",
            ),
            _P(
                "smallest_positive",
                "the smallest value greater than zero",
                "",
                "next((x for x in {prog} if x > 0), -1)",
                "[&]{ for (int x : {prog}) if (x > 0) return x; return -1; }()",
            ),
        ),
    ),
    "int": KVCluster(
        "digits",
        "int",
        ("n", "x"),
        _P(
            "to_digits",
            "the list of decimal digits",
            "digits",
            "{prog} = [int(c) for c in str(abs({arg}))]",
            "std::vector<int> {prog};\n    for (char c : std::to_string(std::abs({arg}))) {prog}.push_back(c - '0');",
        ),
        (
            _P(
                "digit_sum",
                "the sum of the digits",
                "",
                "sum({prog})",
                "std::accumulate({prog}.begin(), {prog}.end(), 0)",
            ),
            _P(
                "max_digit",
                "the largest digit",
                "",
                "max({prog})",
                "*std::max_element({prog}.begin(), {prog}.end())",
            ),
            _P(
                "count_even_digits",
                "the number of even digits",
                "",
                "sum(1 for d in {prog} if d % 2 == 0)",
                "[&]{ int c = 0; for (int d : {prog}) if (d % 2 == 0) c++; return c; }()",
            ),
            _P(
                "all_distinct_digits",
                "returns true if all digits are distinct",
                "",
                "len(set({prog})) == len({prog})",
                "std::set<int>({prog}.begin(), {prog}.end()).size() == {prog}.size()",
            ),
        ),
    ),
    "str": KVCluster(
        "chars_sorted",
        "str",
        ("s", "text"),
        _P(
            "sort_chars",
            "the characters sorted alphabetically",
            "sorted_chars",
            "{prog} = sorted({arg})",
            "std::string {prog} = {arg};\n    std::sort({prog}.begin(), {prog}.end());",
        ),
        (
            _P(
                "first_char_vowel",
                "returns true if the alphabetically smallest character is a vowel",
                "",
                "{prog}[0] in 'aeiou'",
                'std::string("aeiou").find({prog}[0]) != std::string::npos',
            ),
            _P(
                "last_char",
                "the alphabetically largest character",
                "",
                "{prog}[-1]",
                "{prog}.back()",
            ),
            _P(
                "has_repeat",
                "returns true if any character repeats",
                "",
                "any({prog}[i] == {prog}[i+1] for i in range(len({prog}) - 1))",
                "[&]{ for (size_t i = 0; i + 1 < {prog}.size(); ++i) if ({prog}[i] == {prog}[i+1]) return true; return false; }()",
            ),
            _P(
                "distinct_chars",
                "the number of distinct characters",
                "",
                "len(set({prog}))",
                "std::set<char>({prog}.begin(), {prog}.end()).size()",
            ),
        ),
    ),
}


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
        ),
    ),
    "str": FilterTask(
        "char_filter",
        "str",
        ("s", "text"),
        "c",
        "characters",
        (
            _C(
                "vowel",
                "are vowels",
                "{elem} in 'aeiou'",
                'std::string("aeiou").find({elem}) != std::string::npos',
            ),
            _C(
                "upper",
                "are uppercase",
                "{elem}.isupper()",
                "std::isupper((unsigned char){elem})",
            ),
            _C(
                "alpha",
                "are alphabetic",
                "{elem}.isalpha()",
                "std::isalpha((unsigned char){elem})",
            ),
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# setup generators — a few contrastive instances per (datatype, presentation, language)
# ══════════════════════════════════════════════════════════════════════════════
def q_transfer_setups(
    datatype: str,
    presentation: Presentation = "listed",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
    n_conditions: Optional[int] = None,
    naming: Literal["natural", "letters"] = "natural",
    max_setups: int = 4,
) -> list[QTransferInstance]:
    """Contrastive q-transfer setups (01 A-experiment style). Both sides prefill a CONTIGUOUS
    prefix of the predicate list; SRC's prefix is strictly LONGER than DST's, so their frontiers
    (= the next predicate to write) differ by construction:

        DST progress = ids[:dst_len]  -> DST frontier = ids[dst_len]
        SRC progress = ids[:src_len]  -> SRC frontier = ids[src_len] = expected_next   (src_len > dst_len)

    A non-contiguous "all-but-X" scheme is WRONG here: it leaves a gap that collides with DST's
    frontier (both sides then write the same left-out predicate — no contrast). Enumerates
    (dst_len, src_len) pairs up to ``max_setups``. ids[0] is the always-prefilled style anchor.

    Knobs:
        n_conditions : number of predicates the task exposes (uses the first ``n_conditions`` of the
                       datatype's bank; ``None`` = all). Needs >= 3 for a genuine 3-way contrast.
        naming       : result variable names — "natural" (descriptive) or "letters" (a/b/c/d, as in
                       the 01 multi-predicate setup). Affects the prefilled code, not the NL task list.
    """
    task = QTASKS[datatype]
    if n_conditions is not None:
        assert 2 <= n_conditions <= len(task.predicates), (
            f"n_conditions must be in [2, {len(task.predicates)}]"
        )
        task = dataclasses.replace(task, predicates=task.predicates[:n_conditions])
    ids = [p.id for p in task.predicates]
    n = len(ids)
    out = []
    for dst_len in range(1, n - 1):  # DST prefix length (>=1 = anchor)
        for src_len in range(
            dst_len + 1, n
        ):  # SRC prefix strictly longer; SRC writes ids[src_len]
            out.append(
                QTransferInstance(
                    task=task,
                    var_name=task.var_names[0],
                    presentation=presentation,
                    src_lang=src_lang,
                    dst_lang=dst_lang,
                    src_progress=ids[:src_len],
                    dst_progress=ids[:dst_len],
                    expected_next=ids[src_len],
                    naming=naming,
                    meta={"datatype": datatype, "dst_next": ids[dst_len]},
                )
            )
            if len(out) >= max_setups:
                return out
    return out


def kv_transfer_setups(
    datatype: str,
    presentation: Presentation = "description",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
    pivot: Literal["return", "indent"] = "return",
    n: int = 4,
) -> list[KVTransferInstance]:
    """Contrastive kv-transfer setups for one datatype's cluster: distinct (src_task, dst_task)
    pairs sharing the common progress. Yields up to ``n`` pairs (cyclic src→next)."""
    cluster = KVCLUSTERS[datatype]
    tids = [t.id for t in cluster.tasks]
    pairs = [(tids[i], tids[(i + 1) % len(tids)]) for i in range(len(tids))][:n]
    return [
        KVTransferInstance(
            cluster=cluster,
            var_name=cluster.var_names[0],
            presentation=presentation,
            src_lang=src_lang,
            dst_lang=dst_lang,
            src_task=s,
            dst_task=d,
            pivot=pivot,
            meta={"datatype": datatype},
        )
        for s, d in pairs
    ]


def filter_transfer_setups(
    datatype: str,
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
    postfix: Literal["return", "targeted"] = "targeted",
    max_setups: int = 4,
) -> list[FilterTransferInstance]:
    """Contrastive composed-filter setups (00_diff TASK_1 style). The task is a pair of commutative
    filter sub-tasks; SRC and DST prefill DIFFERENT single sub-tasks (as the first ``res = […]``
    filter), then hit a shared ``postfix`` pivot — so SRC is about to apply its remaining sub-task
    and clean DST its remaining (the complementary one). Enumerates 2-subsets of the datatype's
    condition bank × both swap directions, up to ``max_setups``.

        src_fill=T1, dst_fill=T2  ->  expected_next=T2 (SRC's remaining), dst_next=T1 (clean DST's).
    """
    task = FILTER_TASKS[datatype]
    conds = list(task.conditions)
    out = []
    for ca, cb in itertools.combinations(conds, 2):  # each unordered pair of sub-tasks
        for src_fill, dst_fill in (
            (ca.id, cb.id),
            (cb.id, ca.id),
        ):  # both swap directions
            out.append(
                FilterTransferInstance(
                    task=task,
                    pair=(ca.id, cb.id),
                    var_name=task.var_names[0],
                    src_lang=src_lang,
                    dst_lang=dst_lang,
                    src_fill=src_fill,
                    dst_fill=dst_fill,
                    expected_next=dst_fill,  # SRC's remaining == the sub-task DST prefilled
                    postfix=postfix,
                    meta={"datatype": datatype, "dst_next": src_fill},
                )
            )
            if len(out) >= max_setups:
                return out
    return out


def all_q_transfer_setups(**kw) -> list[QTransferInstance]:
    return list(
        itertools.chain.from_iterable(q_transfer_setups(dt, **kw) for dt in QTASKS)
    )


def all_kv_transfer_setups(**kw) -> list[KVTransferInstance]:
    return list(
        itertools.chain.from_iterable(kv_transfer_setups(dt, **kw) for dt in KVCLUSTERS)
    )


def all_filter_transfer_setups(**kw) -> list[FilterTransferInstance]:
    return list(
        itertools.chain.from_iterable(
            filter_transfer_setups(dt, **kw) for dt in FILTER_TASKS
        )
    )
