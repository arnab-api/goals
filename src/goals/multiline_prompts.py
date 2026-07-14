"""Forced-prefill prompt builders for the multiline/singleline 'stop vs continue' setup.

Lifted from `notebooks/clock/04.2_multiline.ipynb` (cell 4) so notebooks 04.2 / 04.3 / sweeps can
share one source of truth. The construction:

  - A `check_conditions(n)`-style stub over `n_conditions` predicate conditions (a, b, c, ...).
  - PRIOR conditions (slots < target) are force-filled COMPLETE — a random single/multi-line mix,
    but with >=1 multiline — so the LM has in-context precedent that a variable MAY span lines.
  - The TARGET condition (slot `target_slot`) is the *test* variable:
        style="multiline":  forced at its INCOMPLETE first line  -> hypothesis 'continue' (stay).
        style="singleline": forced as one COMPLETE line          -> hypothesis 'moved_on'.

Two prompt modes: "chat" (instruction + <code> assistant turn) and "fim" (Qwen FIM with a fixed
result-printing suffix that references every var, forcing each to be defined in the middle).
"""

import random
from typing import Callable, Optional

from src.goals.condition_data import (
    INT_PREDICATES,
    ConditionSample,
    ConditionSpec,
    DataType,
    Lang,
    chunky_predicates_for,
    sample_test_inputs,
)
from src.goals.functional import apply_chat_template
from src.models import ModelandTokenizer

# multiline-capable predicates (have a `render_multiline`), keyed by language.
# Populated lazily so a language without impls (e.g. cpp before we add impls to
# every predicate) simply yields a smaller pool rather than a KeyError.
def _multiline_preds_for(lang: str) -> list:
    return [p for p in INT_PREDICATES if p.is_multiline_for(lang)]

MULTILINE_PREDS = _multiline_preds_for("python")  # backward-compat alias (python only)
VAR_LETTERS = "abcdefghij"
FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"


def result_print(letters: list[str]) -> str:
    """Fixed FIM suffix: an f-string `print` referencing EVERY result var (so to make the suffix
    valid the model must define each one in the middle)."""
    conds = ", ".join("Condition %d: {%s}" % (i + 1, v) for i, v in enumerate(letters))
    all_expr = " and ".join(letters)
    return '    print(f"' + conds + ", ALL: {" + all_expr + '}")\n'


def replace_in_code_newline_with_tabnewline(prompt: str) -> str:
    """Turn each `\\n` inside the final <code> block into `    \\n` so trailing-whitespace
    tokenization is stable (used before tokenizing for latent extraction / patching)."""
    code_split = prompt.split("<code>")
    code_segment = code_split[-1]
    prefix = "<code>".join(code_split[:-1])
    code_segment = code_segment.replace("\n", "    \n")
    return prefix + "<code>" + code_segment


def build_example(
    mt: ModelandTokenizer,
    rng: random.Random,
    target_slot: int,
    n_conditions: int,
    mode: str = "chat",
    style: str = "multiline",
    naming: str = "letters",
    aug: Optional[Callable[[str], str]] = None,
    lang: Lang = "python",
    chunky: bool = False,
    datatype: DataType = "int",
    prior_style: Optional[str] = None,
) -> dict:
    """One forced-prefill example for the 'stay vs move-on' test.

    `target_slot >= 1` (need >=1 prior to set the multiline precedent) and `target_slot <
    n_conditions` (need a following condition for 'moved_on' to be possible).

    `naming` controls the per-condition result variable names (returned as `var_names`):
        "letters" — positional `a`, `b`, `c`, …  (default; instruction mandates them)
        "natural" — each predicate's `var_name` (e.g. `is_even`); instruction drops the
                    naming constraint. Names are unique because predicate classes are distinct.
        "indexed" — `var_1`, `var_2`, …; instruction drops the naming constraint.
    Predicate CLASSES are always distinct within an example (no repeated `id`, even with
    different params), so e.g. `divisible by 20` and `divisible by 26` can't co-occur.

    `lang` picks the forced-prefill impl (`python` | `cpp`).  FIM mode is python-only for
    now: `result_print` writes a python `print(...)` suffix; a C++-flavoured suffix would
    need a matching `std::cout` writer.

    `chunky=True` draws from `CHUNKY_PREDICATES` (inherently multi-line, no single-line form)
    for the given `datatype` (`int` -> `check_conditions(n: int)`, `str` -> `(s: str)`).  Since
    chunky predicates have no single-line form, the 'moved_on' (complete-target) state is the
    FULL multiline block (pivot after its last line) rather than a one-line impl; priors are
    always rendered as full blocks.  Chat mode only.
    """
    assert 1 <= target_slot < n_conditions
    assert naming in ("letters", "natural", "indexed"), f"unknown naming={naming!r}"
    if mode == "fim" and lang != "python":
        raise NotImplementedError(f"mode='fim' is python-only; got lang={lang!r}")
    if chunky:
        if mode == "fim":
            raise NotImplementedError("mode='fim' is not supported for chunky predicates")
        pool = chunky_predicates_for(datatype)
        assert pool, f"no chunky predicates for datatype={datatype!r}"
    else:
        pool = _multiline_preds_for(lang)
    assert n_conditions <= len(pool), (
        f"n_conditions={n_conditions} exceeds {len(pool)} available predicates "
        f"(chunky={chunky}, datatype={datatype!r}, lang={lang!r})"
    )
    # distinct predicate classes per example (sample WITHOUT replacement)
    preds = rng.sample(pool, n_conditions)
    params = [p.sample_params(rng) for p in preds]
    # per-condition result variable names
    if naming == "natural":
        var_names = [p.var_name for p in preds]
    elif naming == "indexed":
        var_names = [f"var_{i + 1}" for i in range(n_conditions)]
    else:
        var_names = [VAR_LETTERS[i] for i in range(n_conditions)]
    spec = ConditionSpec(
        spec_id=f"ml_s{target_slot}_{style[:3]}_{rng.randrange(10 ** 6):06d}",
        predicate_ids=[p.id for p in preds],
        predicate_params=params,
        spec_texts=[p.render(pr) for p, pr in zip(preds, params)],
        id_substrs=[p.render_substrs(pr) for p, pr in zip(preds, params)],
        test_inputs=sample_test_inputs(rng, datatype if chunky else "int"),
    )
    sample = ConditionSample(
        spec=spec, lang=lang, modularity="monolithic", without_comment=True
    )
    stub = sample.render_code_stub()

    def render_complete(pred, prm, var, prefer_singleline: bool) -> list[str]:
        """Lines that fully implement `pred` at a slot: the single-line form when available and
        requested, else the full multiline block (chunky predicates are always full blocks)."""
        if prefer_singleline and pred.has_singleline_for(lang):
            return [pred.render_singleline(prm, var, lang=lang)]
        return pred.render_multiline(prm, var, lang=lang)

    # priors: force-filled COMPLETE. `prior_style` pins every prior's style:
    #   None (default) — random single/multi mix, but >=1 multiline (the 'when' setup, so the
    #                    model has in-context precedent that a var MAY span lines);
    #   "singleline"  — every prior is single-line (the 'where' setup, so a multi-line prior
    #                    doesn't nudge the target var toward a multi-line implementation);
    #   "multiline"   — every prior is a full multi-line block.
    # A prior predicate lacking a single-line form (chunky) still renders multi-line regardless.
    if prior_style in ("singleline", "multiline"):
        prior_styles = [prior_style] * target_slot
    elif chunky:
        prior_styles = ["multiline"] * target_slot
    else:
        prior_styles = [rng.choice(["multiline", "singleline"]) for _ in range(target_slot)]
        if "multiline" not in prior_styles:
            prior_styles[rng.randrange(target_slot)] = "multiline"
    body = []
    for j in range(target_slot):
        body += render_complete(preds[j], params[j], var_names[j],
                                prefer_singleline=(prior_styles[j] == "singleline"))

    # target: the test variable
    tp, tvar, tprm = preds[target_slot], var_names[target_slot], params[target_slot]
    if style == "multiline":
        target_impl = tp.render_multiline(tprm, tvar, lang=lang)
        target_first = target_impl[0]          # incomplete first line -> "continue"
        target_lines = [target_first]
        expected_rest = "\n".join(target_impl[1:])
        expected_verdict = "continue"
    else:
        # 'moved_on': the target var is COMPLETE at the pivot.  Single-line form if the predicate
        # has one; otherwise (chunky) the full multiline block, with the pivot after its last line.
        if tp.has_singleline_for(lang):
            target_lines = [tp.render_singleline(tprm, tvar, lang=lang)]
        else:
            target_lines = tp.render_multiline(tprm, tvar, lang=lang)
        target_first = target_lines[-1]        # pivot line = the completed statement
        expected_rest = ""                     # complete; nothing left for this var
        expected_verdict = "moved_on"
    body += target_lines

    fim_suffix = result_print(var_names) if lang == "python" else None
    if mode == "fim":
        doc = (
            '    """\n'
            + "".join(f"    {i + 1}. {t}\n" for i, t in enumerate(spec.spec_texts))
            + '    """\n'
        )
        prefix_code = stub + doc + "\n".join(body) + "\n"
        full_prompt = FIM_PREFIX + prefix_code + FIM_SUFFIX + fim_suffix + FIM_MIDDLE
    else:
        prefix_code = stub + "\n".join(body) + "\n"
        name_rule = "letters" if naming == "letters" else "free"
        full_prompt = apply_chat_template(
            mt=mt,
            user_query=sample.render_instruction(name_rule=name_rule).strip(),
            incomplete_code=prefix_code,
        )

    full_prompt = aug(full_prompt) if aug else full_prompt
    next_slot = target_slot + 1
    return {
        "spec_id": spec.spec_id,
        "spec": spec,  # the ConditionSpec (ordered conditions + id_substrs) — for gaze tracking
        "mode": mode,
        "lang": lang,
        "prefill_style": style,  # the TARGET's style (test variable)
        "prior_styles": prior_styles,  # complete priors, >=1 multiline
        "full_prompt": full_prompt,
        "prefill_code": prefix_code,
        "fim_suffix": fim_suffix.rstrip("\n") if (mode == "fim" and fim_suffix) else None,
        "target_slot": target_slot,
        "var_names": var_names,
        "target_var": var_names[target_slot],
        "next_var": var_names[next_slot] if next_slot < n_conditions else None,
        "target_pred": preds[target_slot].id,
        "target_first_line": target_first.strip(),
        "expected_rest": expected_rest,
        "expected_verdict": expected_verdict,
        "n_conditions": n_conditions,
    }
