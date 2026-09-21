# Obstacle robustness of Qwen2.5-Coder-32B-Instruct (behavioral)

*2026-09-20. How robustly does the model still implement a sub-goal when its frontier is forced to start
with a misleading prefix? Behavioral (generation-level) check; the goal-**encoding** persistence test is
separate. Notebook: `notebooks/persistance/03_obstacle_robustness.ipynb`. Machinery:
`scripts/evaluation/obstacle_prefills.py` (obstacles), `scripts/evaluation/obstacle_eval.py`
(`judge_adapts`). Cache: `results/notebook_caches/obstacle_robustness/Qwen2.5-Coder-32B-Instruct/`
(409 per-trial files + `rows.json`).*

## Method

**Task.** Current QTASKS corpus — 4 datatypes (int / str / list[int] / list[str]), 16 predicates each.
Each trial builds a **two-condition prompt with the first condition prefilled** (prior progress), and
obstructs the second (the target sub-goal `g_k`):

```
def func(n: int):
    a = len(str(abs(n)))     # a prior sub-goal, PREFILLED (a different predicate)
    b = <OBSTACLE PREFIX>    # frontier: target g_k, forced to start with the prefix
```

The model greedy-generates the rest of the `b =` line; we grade the completed right-hand side.

**Obstacles (appended verbatim after `b =`).** Three categories, all *escapable* — there is always a
syntactic way to still reach `g_k` despite the forced start (verified 0/64: the wrong expression is kept
fully parenthesized so one nullifier can override it):
- **aligned** — a correct start of `g_k` (ceiling control).
- **easy** — type-appropriate ignorable / slight-adaptation stubs the model should absorb trivially
  (bool: `True and` / `False or` / `not` / `(`; numeric: `0 +` / `(`; char: `"" +` / `(`).
- **hard** — actively adversarial: a forced **wrong sibling** predicate `((<sibling>)` and a wrong-op /
  wrong-type fragment (`- 1 +`). The model must recognize and work around a committed wrong start.

**Scoring — bespoke oracle, plasticity-aware.** `judge_adapts(task, obstacle_stub, full_rhs)` →
**ADAPTED** (implements `g_k`, working around the forced stub), **SUCCUMBED** (followed the stub to the
wrong op), **ESCAPED** (reaches `g_k` but by degenerately discarding the stub), **BROKEN**
(invalid/incomplete). We deliberately do **not** score reference-impl equality — a robust model may adapt
via a different valid implementation (the plasticity point). **ADAPTED rate = obstacle robustness.**

**Scale.** All 64 predicates × up to 4 prefixes/category (`N_PER=4`) = **409 trials**, greedy decoding,
per-trial cached/resumable. Oracle: `claude-sonnet-4-6`, temperature 0.

## Results

**ADAPTED rate by level × datatype** (n=409):

| level | int | str | list[int] | list[str] | **all** | n |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| aligned (ceiling) | 0.94 | 1.00 | 1.00 | 0.94 | **0.97** | 64 |
| easy | 0.72 | 0.75 | 0.53 | 0.51 | **0.63** | 217 |
| hard | 0.16 | 0.19 | 0.09 | 0.16 | **0.15** | 128 |

**Verdict breakdown by level** (fraction; ESCAPED omitted — it never fired):

| level | ADAPTED | SUCCUMBED | BROKEN |
|---|:-:|:-:|:-:|
| aligned | 0.97 | 0.02 | 0.02 |
| easy | 0.63 | 0.19 | 0.15 |
| hard | 0.15 | **0.76** | 0.02 |

## Findings

1. **The model is not very obstacle-robust at the frontier.** A forced *wrong-sibling* start makes it
   **succumb 76%** of the time; it finds the escape only **15%**. The ceiling (aligned 0.97) confirms the
   harness and judge are calibrated — the collapse is real, not an artifact.
2. **Even "easy" perturbations hurt more than expected (0.63).** Not because the model can't syntactically
   ignore them, but because stubs like `not` / `True and` nudge it into *subtly wrong or simplified*
   expressions: e.g. `is_positive` + `not ` → `not 0 > n` (= n≥0, off-by-boundary); `is_prime` + `True and `
   → `True and 1 < n` (drops the primality test, just `n>1`). The prefix perturbs the *content*, not just
   the syntax.
3. **The capability exists but is rarely deployed under hard obstacles.** The 15% hard-ADAPTED cases are
   genuine creative escapes: `is_odd` + `(((n % 2 == 0))` → `(((n % 2 == 0)) == False)`; `count_spaces` +
   `((len(s))` → `((len(s)) - len(s.replace(" ","")))`; `sign` + `- 1 +` → `- 1 + (n >= 0) * 2`. So the
   model *can* work around a committed wrong start — it just usually doesn't.
4. **Datatype gradient (hard):** list[int] weakest (ADAPTED 0.09, SUCCUMBED 0.88), str strongest (0.19);
   all low.

## Caveats

- **Behavioral, not representational.** This measures the *model's* obstacle robustness (generation), not
  the *goal-encoding's* persistence. The encoding-level obstacle test (does the goal-head readout stay on
  `g_k` under the obstacle; does encoding-persistence mediate recovery) is the separate leg-(iii) work.
- **UNPARSED = 16 (~4%).** The judge's output didn't match the verdict tags on 16 complex RHSs; they sit
  in the denominator but count as no verdict, so all rates are marginally *under*-stated. Worth a parser
  fix / re-judge.
- **ESCAPED never fired** — either genuinely absent (forced prefixes can't be truly discarded) or the
  judge doesn't separate it from ADAPTED/SUCCUMBED. Treat the 4-way as effectively 3-way here.
- **Judge is the arbiter** (plasticity-lenient). Most calls look right; a few ADAPTED are slightly off
  (`sign` escape mishandles 0) and a few SUCCUMBED are borderline-creative — the rates are oracle
  judgments, not ground truth.
- **Prompt is a 2-condition minimal frontier**, not the full multi-predicate context of the main paper.

## Takeaway

Qwen2.5-Coder-32B-Instruct exhibits **weak obstacle robustness** at a sub-goal frontier: near-ceiling with
a correct start, **~0.63** under mild (ignorable) perturbations, and **~0.15** under a committed wrong
start (76% succumb). Robustness *is* achievable — a real fraction of hard cases show creative escapes —
but it is not the model's default. This mirrors the earlier prefill-probing finding (`0_lm_performance.md`,
~74% adversarial on the older single-line set) in spirit, and sharpens it with a graded, escapable,
plasticity-aware obstacle design. For the paper's persistence(ii) leg, this is the behavioral half; the
representational half (goal-encoding persistence under the same obstacles) remains to be run.
