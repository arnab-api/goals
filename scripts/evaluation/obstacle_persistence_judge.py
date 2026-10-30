"""Bespoke oracle judge for the obstacle-robustness eval (``obstacle_persistence_eval.py``) and the
obstacle OV-boost eval (``obstacle_persistence_ov_boost_eval.py``).

A code model was forced to begin a one-line implementation with a prefix it could not delete (the
obstacle stub), then greedy-generated the rest. ``judge_adapts`` asks an oracle whether the model
implemented the target sub-goal *while adapting to* that forced prefix, returning one of four verdicts:

  ADAPTED   — the full RHS correctly implements the sub-goal, working AROUND the forced stub
              (any valid implementation counts — plasticity). The robustness success.
  SUCCUMBED — the model followed the stub's pull and implemented the WRONG operation.
  ESCAPED   — the RHS does compute the sub-goal, but by degenerately discarding the stub (e.g. a
              comment/newline then a fresh restart, or a trivial bypass) rather than adapting to it.
  BROKEN    — syntactically invalid, incomplete, or unrelated to the sub-goal.

Mirrors ``ablation_common.judge_implements`` for the ask_claude call + caching pattern.
"""
import re

from src.utils.oracle_llms import ask_claude

ORACLE_MODEL = "claude-sonnet-4-6"

JUDGE_SYS = (
    "You grade a code model's robustness to a misleading start. The model was implementing a single "
    "sub-goal as one right-hand-side expression, but its output was FORCED to begin with a given prefix "
    "that it could not delete; it then generated the rest of the line. Judge, from the FULL right-hand "
    "side, whether the model still implemented the intended sub-goal while adapting to that forced "
    "prefix. Any correct implementation counts — it need not match a canonical one. Reply with exactly "
    "one <verdict> tag and a one-line <rationale>.\n"
    "Verdicts:\n"
    "- ADAPTED: the full expression correctly computes the sub-goal, working around / overriding the "
    "forced prefix.\n"
    "- SUCCUMBED: it followed the prefix and computes the WRONG operation (not the sub-goal).\n"
    "- ESCAPED: it does compute the sub-goal, but by degenerately discarding the prefix (e.g. restarting "
    "on a new statement/line, commenting it out, or a trivial bypass) rather than genuinely adapting.\n"
    "- BROKEN: syntactically invalid, incomplete, or unrelated to the sub-goal."
)

def judge_adapts(task_desc: str, obstacle_stub: str, full_rhs: str) -> dict:
    """Return {"verdict": one of ADAPTED/SUCCUMBED/ESCAPED/BROKEN/UNPARSED, "rationale": str, "response": str}."""
    if not full_rhs or not full_rhs.strip():
        return dict(verdict="BROKEN", rationale="empty generation", response=None)
    prompt = (
        f"Target sub-goal (what the expression should compute): {task_desc}\n"
        f"The right-hand side was FORCED to begin with this prefix (the model could not delete it): "
        f"`{obstacle_stub.strip()}`\n"
        f"The model's FULL resulting right-hand side is:\n`{full_rhs.strip()}`\n\n"
        f"Is the model implementing the target sub-goal while adapting to the forced prefix?\n"
        f"Answer with <verdict>ADAPTED|SUCCUMBED|ESCAPED|BROKEN</verdict> and "
        f"<rationale>one line</rationale>."
    )
    resp = ask_claude(prompt=prompt, system_prompt=JUDGE_SYS, max_tokens=256,
                      temperature=0.0, use_cache=True, model_name=ORACLE_MODEL)
    verdict = "UNPARSED"
    m = re.search(r"<verdict>\s*(ADAPTED|SUCCUMBED|ESCAPED|BROKEN)\s*</verdict>", resp, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
    else:  # fallback: first bare mention
        hits = re.findall(r"\b(ADAPTED|SUCCUMBED|ESCAPED|BROKEN)\b", resp.upper())
        if hits:
            verdict = hits[0]
    rm = re.search(r"<rationale>\s*(.*?)\s*</rationale>", resp, re.IGNORECASE | re.DOTALL)
    rationale = rm.group(1).strip() if rm else resp.strip()[:200]
    return dict(verdict=verdict, rationale=rationale, response=resp)
