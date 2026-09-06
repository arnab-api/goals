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
import random
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

    @property
    def dst_next_desc(self) -> str:
        did = self.meta.get("dst_next")
        return next(p.description for p in self.task.predicates if p.id == did)

    @property
    def task_desc(self) -> str:
        return "; ".join(p.description for p in self.task.predicates)


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
    pivot: Literal["auto", "return", "indent"] = (
        "auto"  # "auto" = per-language (default):
    )
    # python -> ``return`` keyword pivot (the model writes the task EXPRESSION); c++ -> trailing
    # indent, because a prefilled ``return`` tends to yield weird c++ completions. "return"/"indent"
    # force a single style for BOTH languages.
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
        pivot = self.pivot
        if (
            pivot == "auto"
        ):  # per-language: python -> return, c++ -> indent (return breaks c++)
            pivot = "return" if lang == "python" else "indent"
        if pivot == "return":
            prefill = base + "\n    return "  # model writes the task EXPRESSION
        else:
            prefill = base + "\n    "  # trailing indent
        return q, prefill

    def src_parts(self):
        return self._parts(self.src_lang, self.src_task)

    def dst_parts(self):
        return self._parts(self.dst_lang, self.dst_task)

    @property
    def expected_next(self) -> str:
        """Task id SRC pursues (the transplant target) — mirrors QTransferInstance.expected_next so
        the same oracle-eval + script record path works unchanged."""
        return self.src_task

    @property
    def dst_next(self) -> str:
        """Task id clean DST pursues."""
        return self.dst_task

    @property
    def expected_next_desc(self) -> str:
        return self.task_text(self.src_task)

    @property
    def dst_next_desc(self) -> str:
        return self.task_text(self.dst_task)

    @property
    def task_desc(self) -> str:
        """Oracle context line: the SHARED common-progress prerequisite both tasks build on (the two
        candidate sub-goals A/B are src_task vs dst_task, so the neutral context is the progress)."""
        return self.cluster.common_progress.description


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

    @property
    def dst_next_desc(self) -> str:
        return self._cond(self.meta["dst_next"]).description

    @property
    def task_desc(self) -> str:
        return self.task_text


@dataclass(frozen=True)
class MultiPredicateInstance:
    """Spec-2 MULTI-PREDICATE, divergent-frontier setup (multi-predicate analog of kv-transfer).

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
        lines = [
            _assign_line(p, self.var_name, lang, res_name=self.var_of(preds, p.id))
            for p in self.shared
        ]
        prefill = (
            "<code>\n"
            + _signature(self.datatype, self.var_name, lang)
            + ("\n" + "\n".join(lines) if lines else "")
            + "\n    "  # trailing indent = pivot
        )
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

    Returns ``{"src_prompt", "dst_prompt", ...}`` (+ ``src_task_text``/``dst_task_text`` for kv,
    ``task_text`` for filter). ``ctx`` is a :class:`src.diff.functional.Ctx` (else module-global
    ``mt``). Prefills are finalized (``_finalize_prefill``) so a dangling trailing space never
    becomes the pivot. Imported lazily so dataset inspection needs no model."""
    from src.diff.functional import get_prompt

    qs, pfs = instance.src_parts()
    qd, pfd = instance.dst_parts()
    out = {
        "src_prompt": get_prompt(qs, _finalize_prefill(pfs), ctx=ctx),
        "dst_prompt": get_prompt(qd, _finalize_prefill(pfd), ctx=ctx),
    }
    if isinstance(instance, KVTransferInstance):
        out["src_task_text"] = instance.task_text(instance.src_task)
        out["dst_task_text"] = instance.task_text(instance.dst_task)
    elif isinstance(instance, MultiPredicateInstance):
        # the transplant span is the WHOLE rendered task list (shared T1,T2 + the frontier), NOT just
        # the frontier — cache SRC's full list, replace DST's full list. The ``pad`` alignment then
        # inserts the length-matching newlines at the START of this span (before ``(1)``, right after
        # ``computes:\n``), and every token in the span (pads + T1 + T2 + frontier) is transplanted.
        out["src_task_text"] = instance.full_task_text(instance.src_frontier)
        out["dst_task_text"] = instance.full_task_text(instance.dst_frontier)
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


def _build_incompatible(
    implies: list[tuple[str, str]],
    mutex_groups: list[tuple[str, ...]],
    char_class: "dict[str, str] | None" = None,
) -> dict[str, list[str]]:
    """Derive a symmetric, implication-closed incompatibility mapping from authored rules.

    ``implies``      — ``(a, b)`` pairs meaning ``a ⟹ b`` (composing/contrasting a with b is redundant);
                       transitively closed. Duplicates ``a ⟺ b`` are two entries, one each way.
    ``mutex_groups`` — every pair *within* a tuple can never be true together (or is a near-duplicate we
                       want excluded); expanded to all intra-group pairs.
    ``char_class``   — optional ``{id: category-string}``; ids with DISJOINT category letters are made
                       mutually exclusive (e.g. a letter-class ⊥ a digit-class), avoiding an O(n²) listing.

    Returns ``{id: sorted[ids it must NOT be paired with]}``. Reused by the kv-cluster, multi-predicate
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
    if char_class:
        _cls = list(char_class.items())
        for i in range(len(_cls)):
            for j in range(i + 1, len(_cls)):
                (a, ca), (b, cb) = _cls[i], _cls[j]
                if not (set(ca) & set(cb)):
                    _m(a, b)
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


KVCLUSTERS: dict[str, list[KVCluster]] = {
    "list[str]": [
        KVCluster(
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
                    'std::string("aeiouAEIOU").find({prog}[0][0]) != std::string::npos',
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
                _P(
                    "median_word_len",
                    "the length of the median word when the words are ordered by length",
                    "",
                    "len({prog}[len({prog})//2])",
                    "(int){prog}[{prog}.size()/2].size()",
                ),
                _P(
                    "longest_starts_upper",
                    "returns true if the longest word starts with an uppercase letter",
                    "",
                    "{prog}[-1][0].isupper()",
                    "{prog}.back()[0] >= 'A' && {prog}.back()[0] <= 'Z'",
                ),
            ),
        ),
        KVCluster(
            "long_words",
            "list[str]",
            ("words", "arr"),
            _P(
                "long_words",
                "the words that have at least two characters",
                "long_words",
                "{prog} = [w for w in {arg} if len(w) > 1]",
                "std::vector<std::string> {prog};\n    for (auto& w : {arg}) if (w.size() > 1) {prog}.push_back(w);",
            ),
            (
                _P(
                    "count_palindromes",
                    "the number of words with at least two characters that are palindromes",
                    "",
                    "sum(1 for w in {prog} if w == w[::-1])",
                    "[&]{ int c = 0; for (auto& w : {prog}) { std::string rv(w.rbegin(), w.rend()); if (w == rv) c++; } return c; }()",
                ),
                _P(
                    "count_all_distinct",
                    "the number of words with at least two characters whose letters are all distinct",
                    "",
                    "sum(1 for w in {prog} if len(set(w)) == len(w))",
                    "[&]{ int c = 0; for (auto& w : {prog}) { std::set<char> s(w.begin(), w.end()); if (s.size() == w.size()) c++; } return c; }()",
                ),
                _P(
                    "count_same_ends",
                    "the number of words with at least two characters that start and end with the same letter",
                    "",
                    "sum(1 for w in {prog} if w[0] == w[-1])",
                    "[&]{ int c = 0; for (auto& w : {prog}) if (w.front() == w.back()) c++; return c; }()",
                ),
            ),
        ),
        KVCluster(
            "lowercased",
            "list[str]",
            ("words", "arr"),
            _P(
                "lowercased",
                "the words converted to lowercase",
                "lowered",
                "{prog} = [w.lower() for w in {arg}]",
                "std::vector<std::string> {prog};\n    for (auto w : {arg}) { for (auto& ch : w) ch = (char)std::tolower((unsigned char)ch); {prog}.push_back(w); }",
            ),
            (
                _P(
                    "count_ci_palindromes",
                    "the number of words that are palindromes when letter case is ignored",
                    "",
                    "sum(1 for w in {prog} if w == w[::-1])",
                    "[&]{ int c = 0; for (auto& w : {prog}) { std::string rv(w.rbegin(), w.rend()); if (w == rv) c++; } return c; }()",
                ),
                _P(
                    "num_distinct_ci",
                    "the number of distinct words when letter case is ignored",
                    "",
                    "len(set({prog}))",
                    "(int)std::set<std::string>({prog}.begin(), {prog}.end()).size()",
                ),
                _P(
                    "count_contains_a",
                    "the number of words that contain the letter A in either case",
                    "",
                    "sum(1 for w in {prog} if 'a' in w)",
                    "[&]{ int c = 0; for (auto& w : {prog}) if (w.find('a') != std::string::npos) c++; return c; }()",
                ),
                _P(
                    "any_ci_duplicate",
                    "returns true if two of the words are equal when letter case is ignored",
                    "",
                    "len(set({prog})) < len({prog})",
                    "(int)std::set<std::string>({prog}.begin(), {prog}.end()).size() < (int){prog}.size()",
                ),
            ),
        ),
    ],
    "list[int]": [
        KVCluster(
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
                    "median_val",
                    "the median value",
                    "",
                    "{prog}[len({prog})//2]",
                    "{prog}[{prog}.size()/2]",
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
                    "second_smallest",
                    "the second smallest value",
                    "",
                    "{prog}[1]",
                    "{prog}[1]",
                ),
                _P(
                    "max_gap",
                    "the largest gap between consecutive values when they are sorted",
                    "",
                    "max(({prog}[i+1] - {prog}[i] for i in range(len({prog}) - 1)), default=0)",
                    "[&]{ int g = 0; for (size_t i = 0; i + 1 < {prog}.size(); ++i) g = std::max(g, {prog}[i+1] - {prog}[i]); return g; }()",
                ),
            ),
        ),
        KVCluster(
            "positives",
            "list[int]",
            ("nums", "arr"),
            _P(
                "positives",
                "the values that are greater than zero",
                "positives",
                "{prog} = [x for x in {arg} if x > 0]",
                "std::vector<int> {prog};\n    for (int x : {arg}) if (x > 0) {prog}.push_back(x);",
            ),
            (
                _P(
                    "sum_positives",
                    "the sum of the values that are greater than zero",
                    "",
                    "sum({prog})",
                    "std::accumulate({prog}.begin(), {prog}.end(), 0)",
                ),
                _P(
                    "count_positives",
                    "how many values are greater than zero",
                    "",
                    "len({prog})",
                    "(int){prog}.size()",
                ),
                _P(
                    "product_positives",
                    "the product of the values that are greater than zero",
                    "",
                    "math.prod({prog})",
                    "[&]{ long long p = 1; for (int x : {prog}) p *= x; return p; }()",
                ),
                _P(
                    "smallest_positive",
                    "the smallest value that is greater than zero",
                    "",
                    "min({prog}, default=0)",
                    "[&]{ if ({prog}.empty()) return 0; return *std::min_element({prog}.begin(), {prog}.end()); }()",
                ),
                _P(
                    "all_even_positives",
                    "returns true if every value greater than zero is even",
                    "",
                    "all(x % 2 == 0 for x in {prog})",
                    "[&]{ for (int x : {prog}) if (x % 2 != 0) return false; return true; }()",
                ),
            ),
        ),
        KVCluster(
            "prefix_sums",
            "list[int]",
            ("nums", "arr"),
            _P(
                "prefix_sums",
                "the running cumulative sums of the values from left to right",
                "prefix",
                "{prog} = [sum({arg}[:i+1]) for i in range(len({arg}))]",
                "std::vector<long long> {prog};\n    { long long s = 0; for (int x : {arg}) { s += x; {prog}.push_back(s); } }",
            ),
            (
                _P(
                    "total_sum",
                    "the total of all the values",
                    "",
                    "{prog}[-1] if {prog} else 0",
                    "[&]{ return {prog}.empty() ? 0LL : {prog}.back(); }()",
                ),
                _P(
                    "max_prefix",
                    "the largest value reached by the running total as the values are added left to right",
                    "",
                    "max({prog}, default=0)",
                    "[&]{ if ({prog}.empty()) return 0LL; return *std::max_element({prog}.begin(), {prog}.end()); }()",
                ),
                _P(
                    "count_positive_prefixes",
                    "the number of positions where the running total so far is positive",
                    "",
                    "sum(1 for p in {prog} if p > 0)",
                    "[&]{ int c = 0; for (long long p : {prog}) if (p > 0) c++; return c; }()",
                ),
                _P(
                    "first_positive_index",
                    "the index of the first position where the running total becomes positive, or -1 if it never does",
                    "",
                    "next((i for i, p in enumerate({prog}) if p > 0), -1)",
                    "[&]{ for (size_t i = 0; i < {prog}.size(); ++i) if ({prog}[i] > 0) return (int)i; return -1; }()",
                ),
            ),
        ),
    ],
    "int": [
        KVCluster(
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
                    "min_digit",
                    "the smallest digit",
                    "",
                    "min({prog})",
                    "*std::min_element({prog}.begin(), {prog}.end())",
                ),
                _P(
                    "count_even_digits",
                    "the number of even digits",
                    "",
                    "sum(1 for d in {prog} if d % 2 == 0)",
                    "[&]{ int c = 0; for (int d : {prog}) if (d % 2 == 0) c++; return c; }()",
                ),
                _P(
                    "count_odd_digits",
                    "the number of odd digits",
                    "",
                    "sum(1 for d in {prog} if d % 2 == 1)",
                    "[&]{ int c = 0; for (int d : {prog}) if (d % 2 == 1) c++; return c; }()",
                ),
                _P(
                    "is_ascending",
                    "returns true if the digits are in non-decreasing order",
                    "",
                    "all({prog}[i] <= {prog}[i+1] for i in range(len({prog}) - 1))",
                    "[&]{ for (size_t i = 0; i + 1 < {prog}.size(); ++i) if ({prog}[i] > {prog}[i+1]) return false; return true; }()",
                ),
                _P(
                    "has_zero_digit",
                    "returns true if any digit is zero",
                    "",
                    "0 in {prog}",
                    "[&]{ for (int d : {prog}) if (d == 0) return true; return false; }()",
                ),
                _P(
                    "digit_product",
                    "the product of the digits",
                    "",
                    "math.prod({prog})",
                    "[&]{ long long p = 1; for (int d : {prog}) p *= d; return p; }()",
                ),
            ),
        ),
        KVCluster(
            "divisors",
            "int",
            ("n", "x"),
            _P(
                "divisors",
                "the list of positive divisors of the number",
                "divisors",
                "{prog} = [d for d in range(1, abs({arg}) + 1) if {arg} % d == 0]",
                "std::vector<int> {prog};\n    for (int d = 1; d <= std::abs({arg}); ++d) if ({arg} % d == 0) {prog}.push_back(d);",
            ),
            (
                _P(
                    "count_divisors",
                    "the number of positive divisors of the number",
                    "",
                    "len({prog})",
                    "(int){prog}.size()",
                ),
                _P(
                    "sum_divisors",
                    "the sum of the positive divisors of the number",
                    "",
                    "sum({prog})",
                    "std::accumulate({prog}.begin(), {prog}.end(), 0)",
                ),
                _P(
                    "largest_proper_divisor",
                    "the largest divisor of the number that is smaller than the number itself",
                    "",
                    "max((d for d in {prog} if d < abs({arg})), default=1)",
                    "[&]{ int b = 1; for (int d : {prog}) if (d < std::abs({arg})) b = std::max(b, d); return b; }()",
                ),
                _P(
                    "is_prime",
                    "returns true if the number is prime",
                    "",
                    "len({prog}) == 2",
                    "{prog}.size() == 2",
                ),
                _P(
                    "is_perfect",
                    "returns true if the number equals the sum of its proper divisors",
                    "",
                    "sum(d for d in {prog} if d < abs({arg})) == abs({arg})",
                    "[&]{ int s = 0; for (int d : {prog}) if (d < std::abs({arg})) s += d; return s == std::abs({arg}); }()",
                ),
            ),
        ),
        KVCluster(
            "binary",
            "int",
            ("n", "x"),
            _P(
                "binary_digits",
                "the binary digits of the number, most significant first",
                "bits",
                "{prog} = [int(b) for b in bin(abs({arg}))[2:]]",
                "std::vector<int> {prog};\n    { int m = std::abs({arg}); if (m == 0) {prog}.push_back(0); else { std::vector<int> t; while (m) { t.push_back(m & 1); m >>= 1; } for (auto it = t.rbegin(); it != t.rend(); ++it) {prog}.push_back(*it); } }",
            ),
            (
                _P(
                    "popcount",
                    "the number of one-bits in the binary representation of the number",
                    "",
                    "sum({prog})",
                    "std::accumulate({prog}.begin(), {prog}.end(), 0)",
                ),
                _P(
                    "bit_length",
                    "the number of bits in the binary representation of the number",
                    "",
                    "len({prog})",
                    "(int){prog}.size()",
                ),
                _P(
                    "is_power_of_two",
                    "returns true if the number is a power of two",
                    "",
                    "sum({prog}) == 1",
                    "std::accumulate({prog}.begin(), {prog}.end(), 0) == 1",
                ),
                _P(
                    "trailing_zeros",
                    "the number of trailing zero bits in the binary representation of the number",
                    "",
                    "next((i for i, b in enumerate(reversed({prog})) if b == 1), len({prog}))",
                    "[&]{ int c = 0; for (auto it = {prog}.rbegin(); it != {prog}.rend(); ++it) { if (*it == 1) break; c++; } return c; }()",
                ),
            ),
        ),
    ],
    "str": [
        KVCluster(
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
                    "second_smallest_char",
                    "the alphabetically second-smallest character",
                    "",
                    "{prog}[1]",
                    "{prog}[1]",
                ),
                _P(
                    "char_range",
                    "the distance between the largest and smallest character codes",
                    "",
                    "ord({prog}[-1]) - ord({prog}[0])",
                    "(int){prog}.back() - (int){prog}.front()",
                ),
                _P(
                    "median_char",
                    "the median character in alphabetical order",
                    "",
                    "{prog}[len({prog})//2]",
                    "{prog}[{prog}.size()/2]",
                ),
                _P(
                    "has_repeat",
                    "returns true if any character occurs more than once",
                    "",
                    "any({prog}[i] == {prog}[i+1] for i in range(len({prog}) - 1))",
                    "[&]{ for (size_t i = 0; i + 1 < {prog}.size(); ++i) if ({prog}[i] == {prog}[i+1]) return true; return false; }()",
                ),
            ),
        ),
        KVCluster(
            "vowels",
            "str",
            ("s", "text"),
            _P(
                "vowels",
                "the vowels in the string, in order",
                "vowels",
                "{prog} = [c for c in {arg} if c in 'aeiou']",
                'std::string {prog};\n    for (char c : {arg}) if (std::string("aeiou").find(c) != std::string::npos) {prog}.push_back(c);',
            ),
            (
                _P(
                    "count_vowels",
                    "the number of vowels in the string",
                    "",
                    "len({prog})",
                    "(int){prog}.size()",
                ),
                _P(
                    "first_vowel",
                    "the first vowel in the string, or '?' if there are none",
                    "",
                    "{prog}[0] if {prog} else '?'",
                    "[&]{ return {prog}.empty() ? '?' : {prog}.front(); }()",
                ),
                _P(
                    "all_same_vowel",
                    "returns true if all the vowels in the string are the same letter",
                    "",
                    "len(set({prog})) <= 1",
                    "[&]{ std::set<char> s({prog}.begin(), {prog}.end()); return s.size() <= 1; }()",
                ),
                _P(
                    "distinct_vowels",
                    "the number of distinct vowels in the string",
                    "",
                    "len(set({prog}))",
                    "(int)std::set<char>({prog}.begin(), {prog}.end()).size()",
                ),
            ),
        ),
        KVCluster(
            "char_counts",
            "str",
            ("s", "text"),
            _P(
                "char_counts",
                "how many times each character occurs",
                "counts",
                "{prog} = {c: {arg}.count(c) for c in set({arg})}",
                "std::map<char,int> {prog};\n    for (char c : {arg}) {prog}[c]++;",
            ),
            (
                _P(
                    "most_frequent_char",
                    "the character that occurs most frequently",
                    "",
                    "max({prog}, key={prog}.get)",
                    "[&]{ char b = 0; int mx = -1; for (auto& kv : {prog}) if (kv.second > mx) { mx = kv.second; b = kv.first; } return b; }()",
                ),
                _P(
                    "count_repeated_chars",
                    "the number of characters that occur more than once",
                    "",
                    "sum(1 for v in {prog}.values() if v > 1)",
                    "[&]{ int c = 0; for (auto& kv : {prog}) if (kv.second > 1) c++; return c; }()",
                ),
                _P(
                    "any_repeat",
                    "returns true if some character occurs more than once",
                    "",
                    "any(v > 1 for v in {prog}.values())",
                    "[&]{ for (auto& kv : {prog}) if (kv.second > 1) return true; return false; }()",
                ),
                _P(
                    "distinct_char_count",
                    "the number of distinct characters",
                    "",
                    "len({prog})",
                    "(int){prog}.size()",
                ),
            ),
        ),
    ],
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
            _C(
                "lower",
                "are lowercase",
                "{elem}.islower()",
                "islower((unsigned char){elem})",
            ),
            _C(
                "digit",
                "are digits",
                "{elem}.isdigit()",
                "isdigit((unsigned char){elem})",
            ),
            _C(
                "alnum",
                "are alphanumeric",
                "{elem}.isalnum()",
                "isalnum((unsigned char){elem})",
            ),
            _C(
                "whitespace",
                "are whitespace",
                "{elem}.isspace()",
                "isspace((unsigned char){elem})",
            ),
            _C(
                "consonant",
                "are consonants",
                "{elem}.isalpha() and {elem}.lower() not in 'aeiou'",
                'isalpha((unsigned char){elem}) && std::string("aeiou").find(tolower((unsigned char){elem})) == std::string::npos',
            ),
            _C(
                "hex_digit",
                "are hexadecimal digits",
                "{elem} in '0123456789abcdefABCDEF'",
                "isxdigit((unsigned char){elem})",
            ),
            _C(
                "upper_vowel",
                "are uppercase vowels",
                "{elem} in 'AEIOU'",
                'std::string("AEIOU").find({elem}) != std::string::npos',
            ),
            _C(
                "first_half_alpha",
                "are letters between a and m",
                "{elem}.isalpha() and {elem}.lower() >= 'a' and {elem}.lower() <= 'm'",
                "isalpha((unsigned char){elem}) && tolower((unsigned char){elem}) >= 'a' && tolower((unsigned char){elem}) <= 'm'",
            ),
            _C(
                "second_half_alpha",
                "are letters between n and z",
                "{elem}.isalpha() and {elem}.lower() >= 'n' and {elem}.lower() <= 'z'",
                "isalpha((unsigned char){elem}) && tolower((unsigned char){elem}) >= 'n' && tolower((unsigned char){elem}) <= 'z'",
            ),
            _C(
                "letter_e",
                "are the letter e in any case",
                "{elem}.lower() == 'e'",
                "tolower((unsigned char){elem}) == 'e'",
            ),
            _C(
                "even_digit",
                "are even digits",
                "{elem}.isdigit() and int({elem}) % 2 == 0",
                "isdigit((unsigned char){elem}) && ({elem} - '0') % 2 == 0",
            ),
            _C(
                "odd_digit",
                "are odd digits",
                "{elem}.isdigit() and int({elem}) % 2 == 1",
                "isdigit((unsigned char){elem}) && ({elem} - '0') % 2 == 1",
            ),
            _C(
                "comma_or_period",
                "are commas or periods",
                "{elem} in ',.'",
                "{elem} == ',' || {elem} == '.'",
            ),
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# setup generators — a few contrastive instances per (datatype, presentation, language)
# ══════════════════════════════════════════════════════════════════════════════
def q_transfer_setups(
    datatype: str,
    n_conditions: "int | list[int]" = (3, 4, 5),
    n_samples: int = 8,
    seed: int = 0,
    src_n_prefill: Optional[int] = None,
    dst_n_prefill: int = 1,
    presentation: Presentation = "listed",
    naming: Literal["natural", "letters"] = "natural",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
) -> list[QTransferInstance]:
    """Sample ``n_samples`` contrastive q-transfer instances (01 A-experiment style). SRC.task ==
    DST.task; only the amount of prefilled PROGRESS differs. Each instance:

      1. draws a random TASK of ``k`` predicates — ``k`` is ``n_conditions`` if int, else a
         per-instance random pick from the ``n_conditions`` list (task SIZE mixes over, e.g. {3,4,5},
         without multiplying configs). Independent, commutative predicates, sampled without
         replacement in random order from the datatype's bank; and
      2. uses a FIXED contrastive split (randomness is only in *which* predicates):
             SRC prefills ``src_n_prefill`` (default k-1) -> SRC frontier = ids[src_n_prefill]
                 (default = the LAST condition = the transplant target)
             DST prefills ``dst_n_prefill`` (default 1)   -> DST frontier = ids[dst_n_prefill]
                 (default = only the first condition prefilled)
         The test: after patching, does the LM's IMMEDIATE next computation jump to SRC's target?

    Requires ``dst_n_prefill < src_n_prefill < k`` and ``dst_n_prefill >= 1`` for every sampled k.
    Only the CONCEPT PAIR (SRC target = ids[src_n_prefill], DST frontier = ids[dst_n_prefill]) is the
    unit of interest — the anchor and other prefilled conditions are nuisance fillers, sampled at
    random. Instances are therefore deduped by the ORDERED concept pair (src_target, dst_frontier),
    so ``n_samples`` counts distinct concept pairs (≤ len(bank)*(len(bank)-1)); reproducible via
    ``seed`` (fixed draw sequence — resume-stable).

        naming : result variable names — "natural" (descriptive) or "letters" (a/b/c/d).
    """
    bank = QTASKS[datatype]
    preds = list(bank.predicates)
    ks = [n_conditions] if isinstance(n_conditions, int) else list(n_conditions)
    assert ks and all(3 <= k <= len(preds) for k in ks), (
        f"n_conditions must be in [3, {len(preds)}]"
    )
    for k in ks:  # validate the fixed split against every task size
        sp = k - 1 if src_n_prefill is None else src_n_prefill
        assert 1 <= dst_n_prefill < sp < k, (
            f"need 1 <= dst_n_prefill({dst_n_prefill}) < src_n_prefill({sp}) < n_conditions({k})"
        )
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 200):
        if len(out) >= n_samples:
            break
        k = rng.choice(ks)  # task size for this instance
        src_len = (
            k - 1 if src_n_prefill is None else src_n_prefill
        )  # SRC target position (default last)
        dst_len = dst_n_prefill  # DST frontier position (default 1)
        src_target, dst_frontier = rng.sample(preds, 2)  # the CONCEPT PAIR (dedup unit)
        ckey = (src_target.id, dst_frontier.id)
        if ckey in seen:
            continue
        seen.add(ckey)
        # place the concept pair at their split positions; fill the rest with random nuisance preds
        others = [p for p in preds if p.id not in (src_target.id, dst_frontier.id)]
        fillers = iter(rng.sample(others, k - 2))
        slots = [None] * k
        slots[src_len], slots[dst_len] = src_target, dst_frontier
        subset = [s if s is not None else next(fillers) for s in slots]
        ids = [p.id for p in subset]
        task = dataclasses.replace(bank, id=f"{bank.id}_k{k}", predicates=tuple(subset))
        out.append(
            QTransferInstance(
                task=task,
                var_name=bank.var_names[0],
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
    return out


# ══════════════════════════════════════════════════════════════════════════════
# kv-cluster compatibility — when two cluster tasks are too similar to CONTRAST
# ══════════════════════════════════════════════════════════════════════════════
# Same idea as the multi-predicate rule, over :data:`KVCLUSTERS` task ids (keyed per CLUSTER). A
# (src_task, dst_task) contrast is DEGENERATE when one task is redundant with / implied by the other
# (shared wording, e.g. "the median value" vs "whether the median is even"). Opposites (max vs min,
# longest vs shortest) are legitimate and kept.

# ── degenerate contrasts, keyed per CLUSTER id (task ids may repeat across clusters). A pair is
# degenerate when one task is redundant with / implied by the other (shared wording / one reads the
# other). Opposites (max vs min, longest vs shortest) are legitimate and kept.
_KV_IMPLIES: dict[str, list[tuple[str, str]]] = {
    "digits": [("has_zero_digit", "min_digit"), ("min_digit", "has_zero_digit")],
}

_KV_MUTEX: dict[str, list[tuple[str, ...]]] = {
    "lowercased": [("num_distinct_ci", "any_ci_duplicate")],
    "prefix_sums": [("count_positive_prefixes", "first_positive_index")],
    "divisors": [("count_divisors", "is_prime"), ("sum_divisors", "is_perfect")],
    "binary": [("popcount", "is_power_of_two")],
    "chars_sorted": [],
    "vowels": [
        ("count_vowels", "distinct_vowels"),
        ("all_same_vowel", "distinct_vowels"),
    ],
    "char_counts": [("any_repeat", "count_repeated_chars")],
}

_ALL_KV_CLUSTER_IDS = [c.id for cls in KVCLUSTERS.values() for c in cls]

#: ``{cluster_id: {task_id: [ids too similar to CONTRAST with]}}`` — consulted by ``kv_transfer_setups``.
KV_INCOMPATIBLE: dict[str, dict[str, list[str]]] = {
    cid: _build_incompatible(_KV_IMPLIES.get(cid, []), _KV_MUTEX.get(cid, []))
    for cid in _ALL_KV_CLUSTER_IDS
}


def kv_compatible(cluster_id: str, a_id: str, b_id: str) -> bool:
    """True iff two tasks in the SAME cluster form a NON-degenerate contrast (not redundant / implied)."""
    return b_id not in KV_INCOMPATIBLE.get(cluster_id, {}).get(a_id, ())


def kv_transfer_setups(
    datatype: str,
    presentation: Presentation = "description",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
    pivot: Literal["auto", "return", "indent"] = "auto",
    n_samples: "int | None" = None,
    seed: int = 0,
) -> list[KVTransferInstance]:
    """Contrastive kv-transfer setups for one datatype: distinct (src_task, dst_task) pairs drawn from
    WITHIN each of the datatype's clusters (both tasks share that cluster's common progress). Enumerates
    all non-degenerate ordered pairs per cluster (:func:`kv_compatible`), concatenates across clusters,
    shuffles reproducibly by ``seed``, and returns the first ``n_samples`` (all if ``None``).
    ``meta`` records ``datatype``, ``cluster`` (id) and ``dst_next`` (clean-DST task)."""
    out: list[KVTransferInstance] = []
    for cluster in KVCLUSTERS[datatype]:
        tids = [t.id for t in cluster.tasks]
        for s in tids:
            for d in tids:
                if s != d and kv_compatible(cluster.id, s, d):
                    out.append(
                        KVTransferInstance(
                            cluster=cluster,
                            var_name=cluster.var_names[0],
                            presentation=presentation,
                            src_lang=src_lang,
                            dst_lang=dst_lang,
                            src_task=s,
                            dst_task=d,
                            pivot=pivot,
                            meta={
                                "datatype": datatype,
                                "cluster": cluster.id,
                                "dst_next": d,
                            },
                        )
                    )
    random.Random(seed).shuffle(out)
    if n_samples is not None:
        out = out[:n_samples]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# composition rules — when two filter sub-tasks may NOT be composed
# ══════════════════════════════════════════════════════════════════════════════
# A composed-filter instance pairs two *commutative, independent* sub-tasks. A pair is DISALLOWED when
# composing it is degenerate — either the two conditions cannot both hold, or one trivially entails the
# other (so the composition is redundant and often shares wording). Authored below as a readable
# source-of-truth; ``FILTER_INCOMPATIBLE`` is the derived symmetric ``{condition_id: [ids it must not be
# composed with]}`` mapping (implication-closed) that ``filter_transfer_setups`` consults.
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
    # ── str (single char) ──
    ("vowel", "alpha"),
    ("consonant", "alpha"),
    ("upper", "alpha"),
    ("lower", "alpha"),
    ("upper_vowel", "upper"),
    ("upper_vowel", "vowel"),
    ("first_half_alpha", "alpha"),
    ("second_half_alpha", "alpha"),
    ("letter_e", "vowel"),
    ("letter_e", "first_half_alpha"),
    ("alpha", "alnum"),
    ("digit", "alnum"),
    ("hex_digit", "alnum"),
    ("digit", "hex_digit"),
    ("even_digit", "digit"),
    ("odd_digit", "digit"),
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
    # ── str (single char) ──
    ("upper", "lower"),
    ("vowel", "consonant"),
    ("even_digit", "odd_digit"),
    ("first_half_alpha", "second_half_alpha"),
]

# str single-char conditions each hold a character CATEGORY; disjoint categories ⇒ mutually exclusive
# (L=letter, D=digit, S=whitespace, P=punctuation; "LD" = either). Generates e.g. vowel⊥digit,
# alpha⊥whitespace, hex_digit⊥whitespace, comma_or_period⊥alnum, … without listing each pair.
_STR_CHAR_CLASS: dict[str, str] = {
    "vowel": "L",
    "consonant": "L",
    "alpha": "L",
    "upper": "L",
    "lower": "L",
    "upper_vowel": "L",
    "first_half_alpha": "L",
    "second_half_alpha": "L",
    "letter_e": "L",
    "digit": "D",
    "even_digit": "D",
    "odd_digit": "D",
    "whitespace": "S",
    "comma_or_period": "P",
    "alnum": "LD",
    "hex_digit": "LD",
}


def _build_filter_incompatible() -> dict[str, list[str]]:
    """Derive the symmetric, implication-closed incompatibility mapping for filter conditions."""
    return _build_incompatible(_FILTER_IMPLIES, _FILTER_MUTEX, _STR_CHAR_CLASS)


#: ``{condition_id: [ids it must NOT be composed with]}`` — consulted by ``filter_transfer_setups``.
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


def filter_transfer_setups(
    datatype: str,
    n_samples: int = 8,
    seed: int = 0,
    postfix: Literal["return", "targeted"] = "targeted",
    src_lang: Lang = "python",
    dst_lang: Lang = "python",
    n_conditions: int = 2,
) -> list[FilterTransferInstance]:
    """Sample ``n_samples`` composed-filter instances (00_diff TASK_1 style). Each task is
    ``n_conditions`` (fixed at 2 — the two commutative sub-tasks) filter conditions sampled at random
    from the datatype's bank; SRC and DST prefill DIFFERENT single sub-tasks (as the first
    ``res = […]`` filter), then hit a shared ``postfix`` pivot — so SRC is about to apply its
    remaining sub-task and clean DST its remaining (the complementary one). A random swap direction
    is drawn per instance. Deduped by (pair, src_fill), reproducible via ``seed`` (drawn in a fixed
    sequence so a larger ``n_samples`` extends the same prefix — resume-stable).

        src_fill=T1, dst_fill=T2  ->  expected_next=T2 (SRC's remaining), dst_next=T1 (clean DST's).
    """
    assert n_conditions == 2, "filter tasks are exactly 2 commutative sub-tasks"
    ft = FILTER_TASKS[datatype]
    conds = list(ft.conditions)
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n_samples * 200):
        if len(out) >= n_samples:
            break
        ca, cb = rng.sample(conds, 2)  # random 2-subset (the two sub-tasks)
        if not filter_compatible(
            ca.id, cb.id
        ):  # skip degenerate pairs (see FILTER_INCOMPATIBLE)
            continue
        src_fill, dst_fill = (ca.id, cb.id) if rng.random() < 0.5 else (cb.id, ca.id)
        key = ((ca.id, cb.id), src_fill)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            FilterTransferInstance(
                task=ft,
                pair=(ca.id, cb.id),
                var_name=ft.var_names[0],
                src_lang=src_lang,
                dst_lang=dst_lang,
                src_fill=src_fill,
                dst_fill=dst_fill,
                expected_next=dst_fill,  # SRC's remaining == the sub-task DST prefilled
                postfix=postfix,
                meta={"datatype": datatype, "dst_next": src_fill},
            )
        )
    return out


def multi_predicate_setups(
    datatype: str,
    n_conditions: "int | list[int]" = 3,
    n_samples: int = 8,
    seed: int = 0,
    presentation: Presentation = "listed",
    naming: Literal["natural", "letters"] = "natural",
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
    distinct frontier contrasts; reproducible via ``seed`` (fixed draw sequence — resume-stable)."""
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
                meta={
                    "datatype": datatype,
                    "dst_next": dst_frontier.id,
                    "n_conditions": k,
                },
            )
        )
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


def all_multi_predicate_setups(**kw) -> list[MultiPredicateInstance]:
    return list(
        itertools.chain.from_iterable(multi_predicate_setups(dt, **kw) for dt in QTASKS)
    )
