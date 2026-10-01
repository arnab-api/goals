"""Generation-level q-transfer eval for the query-content hypotheses (H1 / H1c / H2 / H3).

Design doc: ``agent_reports/claude/analysis/q_transfer_query_content_hypotheses_design.md``. The mask
localization (``scripts/diff/train_nrgs_mask.py``) scores a next-token flip on the 7-predicate head-loc
bank; this eval scores what the patched model actually WRITES, on the 16-predicate q-transfer banks
(:data:`src.diff.data.QTASKS`, 4 datatypes), with an exclusive multi-letter judge:

* instances — :func:`hypothesis_setups` draws one predicate per letter of a registry config
  (:data:`src.diff.nrgs_prefix.CONFIGS`) and renders the SRC / DST prompts with the regular
  :class:`src.diff.data.DiffTriggerInstance` (sequential letter naming, ``var_eq`` / indent pivot);
* judge — :func:`judge_letters` asks which ONE of DST's pending sub-tasks (plus SRC's own frontier, which
  a literal copy would write) the first new computation implements, or NEITHER. Options are shown in a
  per-instance shuffled order (position-bias control) and mapped back to their letters;
* scoring — :func:`score` maps the judged letter onto every hypothesis via ``Config.predictions``:
  ``hit[h] = (label == predictions[h])``. The same judge runs on the clean DST continuation, so each
  hypothesis gets a clean rate, a patched rate and their difference (the shift).
"""
from __future__ import annotations

import hashlib
import random
import re

import src.diff.data as D
import src.diff.nrgs_prefix as N
from src.diff.eval import CONCEPT_SYSTEM

SRC_TARGET = "src_target"            # label: SRC's own frontier (absent from DST in the strict configs)


def _roles_ok(dt, ids, judged, trivial, distinct_families):
    if any(i in trivial for i in judged):
        return False
    if distinct_families and len({D._diff_family(dt, i) for i in judged}) < len(judged):
        return False
    return all(D.qpred_compatible(dt, a, b) for k, a in enumerate(ids) for b in ids[k + 1:])


def hypothesis_setups(datatype, config, n_samples=32, seed=0, letter_stub="var_eq", naming="letters",
                      src_lang="python", dst_lang="python", distinct_families=True, nontrivial=True):
    """``n_samples`` :class:`DiffTriggerInstance` s for one registry config. Every letter gets a distinct
    predicate from ``QTASKS[datatype]``; the JUDGED options (DST's pending tasks + SRC's frontier) must be
    non-trivial frontiers from pairwise-distinct semantic families, and all predicates pairwise compatible
    (no implication / mutex pairs). ``meta`` carries the letter -> predicate map and the per-hypothesis
    predicted letters."""
    cfg = N.CONFIGS[N.canonical(config)]
    bank = D.QTASKS[datatype]
    preds = {p.id: p for p in bank.predicates}
    letters = list(cfg.letters)
    pending = [x for x in cfg.dst if x not in cfg.dst_prefill]
    judged = list(dict.fromkeys(pending + ([cfg.src_next] if cfg.src_next not in cfg.dst else [])))
    trivial = D.TRIVIAL_FRONTIERS.get(datatype, frozenset()) if nontrivial else frozenset()
    rng = random.Random(seed + int(hashlib.md5(f"{cfg.name}/{datatype}".encode()).hexdigest()[:8], 16))
    out, seen = [], set()
    for _ in range(n_samples * 2000):
        if len(out) >= n_samples:
            break
        ids = rng.sample(list(preds), len(letters))
        role = dict(zip(letters, ids))
        if tuple(ids) in seen or not _roles_ok(datatype, ids, [role[x] for x in judged], trivial, distinct_families):
            continue
        seen.add(tuple(ids))
        out.append(D.DiffTriggerInstance(
            datatype=datatype,
            src_order=tuple(preds[role[x]] for x in cfg.src),
            dst_order=tuple(preds[role[x]] for x in cfg.dst),
            src_progress=[role[x] for x in cfg.src if x in cfg.src_prefill],
            dst_progress=[role[x] for x in cfg.dst if x in cfg.dst_prefill],
            diff_target=role[cfg.target], dst_native=role[cfg.native], src_target=role[cfg.src_next],
            var_name=bank.var_names[0], presentation="listed", src_lang=src_lang, dst_lang=dst_lang,
            naming=naming, letter_stub=letter_stub,
            meta={"datatype": datatype, "variant": cfg.name, "config": cfg.name, "hypothesis": cfg.hypothesis,
                  "roles": role, "pending": pending, "judged": judged, "predictions": cfg.predictions,
                  "dst_next": role[cfg.native], "diff_target": role[cfg.target], "src_target": role[cfg.src_next]},
        ))
    return out


def inst_key(inst) -> str:
    r = inst.meta["roles"]
    return (f"{inst.meta['config']}|{inst.datatype}|" + "|".join(f"{x}={r[x]}" for x in sorted(r))
            + f"|{inst.src_lang}-{inst.dst_lang}")


# ── the exclusive multi-letter judge ──────────────────────────────────────────────────────────────
def _letters_prompt(text, options, task_desc, done_desc, kind):
    opts = "\n".join(f"  ({i + 1}): {desc}" for i, (_, desc) in enumerate(options))
    codes = "\n".join(f"  {i + 1:<8} - pursues ({i + 1})" for i in range(len(options)))
    return f"""A language model is writing code for this task:
  TASK: {task_desc}
Already implemented (do not count these): {done_desc or "nothing"}

Candidate sub-goals it might pursue next:
{opts}

Below is {kind}. Decide which ONE of the candidate sub-goals it pursues. Treat a non-trivial partial
implementation or an explicit mention as pursuing that sub-goal.

IMPORTANT — judge by the actual COMPUTATION / logic the code performs, NOT by variable, function, or
identifier names. Names can be arbitrary, autocompleted, or outright misleading, so a statement that
*computes* sub-goal X is pursuing X even when its variable is named after something else.

--- BEGIN ---
{text}
--- END ---

Classify as exactly one of:
{codes}
  NEITHER  - none of the candidates (unrelated, empty, generic, or re-implements an already-done sub-goal)
If it clearly computes more than one, pick the one it performs FIRST.

Answer with: <verdict>{"|".join(str(i + 1) for i in range(len(options)))}|NEITHER</verdict> then a one-line <why>...</why>."""


def judge_letters(inst, text, model_name="claude-sonnet-4-6", use_cache=True, kind=None) -> dict:
    """Exclusive verdict over the judged letters of ``inst`` (DST's pending tasks + SRC's frontier).
    Returns ``{"label": <letter> | "neither" | "parse_error", "why", "raw", "order"}``."""
    m = inst.meta
    letters = list(m["judged"])
    random.Random(inst_key(inst)).shuffle(letters)                      # deterministic per instance
    desc = {p.id: p.description for p in (*inst.src_order, *inst.dst_order)}
    options = [(x, desc[m["roles"][x]]) for x in letters]
    if not text or not text.strip():
        return {"label": "neither", "why": "empty", "raw": "", "order": letters}
    done = "; ".join(desc[i] for i in inst.dst_progress)
    from src.utils.oracle_llms import ask_claude

    raw = ask_claude(
        _letters_prompt(text, options, inst.task_desc, done,
                        kind or "the code the model generated to continue the function after the pivot. "
                                "Judge ONLY the FIRST computation/statement it performs (ignore anything after)"),
        system_prompt=CONCEPT_SYSTEM, max_tokens=768, use_cache=use_cache, model_name=model_name)
    v = re.search(r"<verdict>\s*(\d+|NEITHER)\s*</verdict>", raw, re.I)
    w = re.search(r"<why>\s*(.*?)\s*</why>", raw, re.I | re.S)
    if not v:
        label = "parse_error"
    elif v.group(1).upper() == "NEITHER":
        label = "neither"
    else:
        k = int(v.group(1)) - 1
        label = letters[k] if 0 <= k < len(letters) else "parse_error"
    return {"label": label, "why": w.group(1).strip() if w else "", "raw": raw, "order": letters}


def score(inst, label: str) -> dict:
    """{hypothesis: hit} for one judged letter; ``src_target`` marks a literal copy of SRC's frontier."""
    P = inst.meta["predictions"]
    out = {h: (P.get(h) is not None and label == P[h]) for h in ("H0", "H1", "H1c", "H2", "H3")}
    out[SRC_TARGET] = label == N.CONFIGS[inst.meta["config"]].src_next
    out["neither"] = label in ("neither", "parse_error")
    return out
