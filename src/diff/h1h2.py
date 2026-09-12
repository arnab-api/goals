"""Contrastive setup to dissociate INDUCTION-ORDERING (H1) from SET-DIFFERENCE (H2) as the mechanism
that selects the next sub-task at the pivot.

Five DISTINCT predicates fill five structural roles (letters match the design note in
``logs/diff_loc.md``: SRC={A,C,D,B}, DST={A,C,B,E}):

  anchor   (A): pos-1 in both lists; the only DST-prefilled task; its successor is DST's native target.
  frontier (C): pos-2 in both; the linchpin — its successor DIVERGES: -> src_targ in SRC, -> dst_h1 in DST.
  src_targ (D): SRC pos-3 (the SRC target); ABSENT from DST                 => the COPY outcome.
  dst_h1   (B): SRC pos-4 (gap-filler, done); DST pos-3 (successor of C)     => the H1 / induction outcome.
  dst_targ (E): DST pos-4; ABSENT from SRC                                   => the H2 / set-difference outcome.

    SRC.tasks = [A, C, D, B]   prefill = {A, C, B}   targ = D
    DST.tasks = [A, C, B, E]   prefill = {A}         targ = C   (native)

Why this construction:
  * D is BOTH "successor of C" (H1) AND the set-complement {A,C,D,B}\\{A,C,B} (H2) -> SRC is neutral.
  * The GAP (B done at pos-4 while D undone at pos-3) makes the done-set non-contiguous, so naive
    positional counting ("3 done -> do the 4th slot") would pick B (already done), not D -> counting
    is ruled out, and an E at DST is cleanly attributable to set-difference (not counting).
  * Patching SRC's pivot query into DST then splits the hypotheses:
        H1  -> B   (successor-of-C resolves to B in DST)
        H2  -> E   (set-complement of {A,C,B} in DST)
        COPY-> D   (SRC target verbatim; D is not even in DST)
        NO-OP-> C  (DST native)

Prompts use the PLAIN ``computes:`` instruction (NOT ``computes in sequential order:`` — the
out-of-order SRC prefill would contradict that) and SEQUENTIAL-BY-APPEARANCE letter naming: the
prefilled lines read ``a``, ``b``, ``c`` in the order they are written and the target gets the next
letter (``d =``), independent of task-list position. So SRC renders naturally as
``a=.. b=.. c=.. d=`` even though its done-set is out of task order; the gap lives in the numbered
task list, not in the variable labels.
"""
import dataclasses
import math
import random
from dataclasses import dataclass

import src.diff.data as DATA

ROLES = ("anchor", "frontier", "src_targ", "dst_h1", "dst_targ")  # A, C, D, B, E


@dataclass(frozen=True)
class H1H2Spec:
    """One contrastive pair: five predicate ids assigned to the five roles."""
    datatype: str
    anchor: str      # A
    frontier: str    # C
    src_targ: str    # D
    dst_h1: str      # B
    dst_targ: str    # E
    lang: str = "python"

    @property
    def ids(self):
        return (self.anchor, self.frontier, self.src_targ, self.dst_h1, self.dst_targ)


# ── instance builders ──────────────────────────────────────────────────────────
class _SeqQTI(DATA.QTransferInstance):
    """Name result vars sequentially by APPEARANCE in the prefill (progress order, then the target),
    NOT by task-list position. So the prefilled lines read a, b, c, ... in the order they are written
    and the target gets the next letter (`d =`), regardless of where the tasks sit in the numbered
    list. The task-list order — and hence the induction/gap structure — is untouched; only the labels
    change. Assumes src_progress == dst_progress (``_instance`` sets them equal)."""
    def var_of(self, pred_id):
        if self.naming != "letters":
            return super().var_of(pred_id)
        order = list(self.src_progress) + [self.expected_next]
        for pr in self.task.predicates:
            if pr.id not in order:
                order.append(pr.id)
        return chr(ord("a") + order.index(pred_id))


def _instance(bank, order_ids, progress_ids, target_id, lang, meta):
    by = {p.id: p for p in bank.predicates}
    task = dataclasses.replace(bank, id=f"{bank.id}_h1h2", predicates=tuple(by[i] for i in order_ids))
    return _SeqQTI(
        task=task, var_name=bank.var_names[0], presentation="listed",
        src_lang=lang, dst_lang=lang, src_progress=list(progress_ids), dst_progress=list(progress_ids),
        expected_next=target_id, naming="letters", meta=meta,
    )


def src_instance(spec: H1H2Spec):
    """SRC = [A, C, D, B], prefill {A, B, C} written IN THAT ORDER, target D.

    The prefill is written A -> B -> C so the LAST completed task is the frontier C. This is what gives
    D the induction "I follow C" signal: at the pivot the "what-comes-next" query keys on the last-done
    task (C), and D immediately follows C in the task list -> H1 can reach D. (If B were written last,
    "successor of B" = nothing, so only H2 could reach D and SRC would not be neutral.) The done-SET is
    still {A,B,C} so H2 (set-complement = {D}) reaches D too. The GAP (D at list-pos 3 undone while B at
    pos 4 is done) still rules out naive positional counting."""
    bank = DATA.QTASKS[spec.datatype]
    return _instance(
        bank,
        order_ids=[spec.anchor, spec.frontier, spec.src_targ, spec.dst_h1],       # A C D B  (task list order)
        progress_ids=[spec.anchor, spec.dst_h1, spec.frontier],                    # A B C  (C written LAST => "follow C")
        target_id=spec.src_targ,                                                   # D
        lang=spec.lang, meta={"datatype": spec.datatype, "side": "src", **{r: getattr(spec, r) for r in ROLES}},
    )


def dst_instance(spec: H1H2Spec, src_targ_in_dst: bool = False):
    """DST = [A, C, B, E], prefill {A}, native target C.

    ``src_targ_in_dst=True`` appends the SRC target D -> DST = [A, C, B, E, D]. This gives the COPY
    hypothesis a token span to gaze at: if the patched ``h_src`` has ALREADY decided D, the goal heads
    can look at D's description in DST (with D absent they'd have nowhere to land). It also splits the
    set-complement into {E, D}: gaze on D => "decision already committed in h_src" (COPY); gaze on E =>
    set-difference recomputed in DST (H2); gaze on B => induction (H1)."""
    bank = DATA.QTASKS[spec.datatype]
    order = [spec.anchor, spec.frontier, spec.dst_h1, spec.dst_targ]                # A C B E
    if src_targ_in_dst:
        order = order + [spec.src_targ]                                            # A C B E D
    return _instance(
        bank,
        order_ids=order,
        progress_ids=[spec.anchor],                                                # A
        target_id=spec.frontier,                                                   # C (native)
        lang=spec.lang, meta={"datatype": spec.datatype, "side": "dst",
                              "src_targ_in_dst": src_targ_in_dst, **{r: getattr(spec, r) for r in ROLES}},
    )


# ── prompt rendering (PLAIN 'computes:', positional letters) ─────────────────────
def _chat_prompt(mt, question, prefill):
    """Chat-templated prefilled prompt — same as eval_ov_reader._chat_prompt, WITHOUT the
    'in sequential order' nudge (kept plain for this experiment)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant proficient in programming. "
         "Always wrap your code inside <code></code> blocks. Just the code, no need to comment or explain."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": prefill},
    ]
    try:
        return mt.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True,
            tokenize=False, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        return mt.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True, tokenize=False)


def _stub_parts(inst, side, stub, lang):
    """(question, prefill) honoring the pivot ``stub``: 'indent' (trailing-indent, model still chooses the
    variable) or 'var_eq' (end at the frontier's ``{var} =``). NO 'in sequential order' rewrite."""
    q, pf = inst.src_parts() if side == "src" else inst.dst_parts()
    if stub != "var_eq":
        return q, pf
    var = inst.var_of(inst.expected_next)
    body = pf.rstrip("\n ")
    if lang == "cpp":
        prog = inst.src_progress if side == "src" else inst.dst_progress
        for pid in prog:
            body = DATA._cpp_autoify(body, inst.var_of(pid))
        return q, body + f"\n    auto {var} ="
    return q, body + f"\n    {var} ="


def build_prompts(mt, spec: H1H2Spec, stub="var_eq", src_targ_in_dst=False):
    """Render SRC and DST prompts + metadata for one contrastive pair.

    ``src_targ_in_dst`` (see :func:`dst_instance`) appends D to DST so the COPY hypothesis has a gaze
    target. Returns a dict:
      src_prompt / dst_prompt          : the chat-templated prefilled strings
      src_inst / dst_inst              : the underlying QTransferInstances
      outcomes                         : {label: (role, pred_id, description)} for the graded conditions
      src_targ_desc / dst_native_desc  : gold descriptions of the SRC target (D) and DST native (C)
    """
    si, di = src_instance(spec), dst_instance(spec, src_targ_in_dst=src_targ_in_dst)
    sq, spf = _stub_parts(si, "src", stub, spec.lang)
    dq, dpf = _stub_parts(di, "dst", stub, spec.lang)
    by = {p.id: p for p in DATA.QTASKS[spec.datatype].predicates}
    desc = lambda pid: by[pid].description
    outcomes = {
        "H1 (induction)  -> B": ("dst_h1", spec.dst_h1, desc(spec.dst_h1)),
        "H2 (set-diff)   -> E": ("dst_targ", spec.dst_targ, desc(spec.dst_targ)),
        "COPY            -> D": ("src_targ", spec.src_targ, desc(spec.src_targ)),
        "NO-OP (native)  -> C": ("frontier", spec.frontier, desc(spec.frontier)),
    }
    return dict(
        spec=spec, stub=stub, src_targ_in_dst=src_targ_in_dst,
        src_prompt=_chat_prompt(mt, sq, DATA._finalize_prefill(spf)),
        dst_prompt=_chat_prompt(mt, dq, DATA._finalize_prefill(dpf)),
        src_inst=si, dst_inst=di,
        src_targ_desc=desc(spec.src_targ), dst_native_desc=desc(spec.frontier),
        outcomes=outcomes,
    )


# ── enumeration / counting ───────────────────────────────────────────────────────
def count_pairs(datatype, all_nontrivial=False):
    """Number of DISTINCT contrastive pairs for ``datatype``.

    Roles B/C/D/E (the four graded OUTCOMES) must be NON-trivial predicates; A (anchor, only ever
    prefilled context) may be any predicate. -> P(NT, 4) * (N - 4).  ``all_nontrivial=True`` forces A
    non-trivial too -> P(NT, 5)."""
    bank = DATA.QTASKS[datatype]
    N = len(bank.predicates)
    NT = N - len(DATA.TRIVIAL_FRONTIERS.get(datatype, frozenset()))
    return math.perm(NT, 5) if all_nontrivial else math.perm(NT, 4) * (N - 4)


def sample_pairs(datatype, n, lang="python", seed=123, all_nontrivial=False):
    """``n`` distinct H1H2Specs, deterministic in ``seed`` (resume-stable). B/C/D/E non-trivial; A any
    (or non-trivial if ``all_nontrivial``)."""
    bank = DATA.QTASKS[datatype]
    ids = [p.id for p in bank.predicates]
    triv = DATA.TRIVIAL_FRONTIERS.get(datatype, frozenset())
    nt = [i for i in ids if i not in triv]
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n * 400):
        if len(out) >= n:
            break
        dst_h1, frontier, src_targ, dst_targ = rng.sample(nt, 4)   # B, C, D, E
        pool_a = [i for i in (nt if all_nontrivial else ids) if i not in (dst_h1, frontier, src_targ, dst_targ)]
        if not pool_a:
            continue
        anchor = rng.choice(pool_a)                                 # A
        key = (anchor, frontier, src_targ, dst_h1, dst_targ)
        if key in seen:
            continue
        seen.add(key)
        out.append(H1H2Spec(datatype, anchor, frontier, src_targ, dst_h1, dst_targ, lang=lang))
    return out
