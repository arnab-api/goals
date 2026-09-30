# Subspace-probe read-outs on the GoalScope eval

_2026-09-14. Evaluates the old subspace probe (`notebooks/11_goalscope_subspace.ipynb`,
`src/goals/subspace_probe.py`) with the **same** faithfulness/causality check goal-head GoalScope uses,
on the locked 253-spec corpus. Script: `scripts/evaluation/eval_subspace_probe.py`._

## What the subspace probe is
An alternative intent read-out (not the goal-head OV path). It learns, for Qwen2.5-Coder-32B:
- per-layer low-rank projections `W_l ∈ R^{r×d}` at layers **35–45** (`r=256`; kept orthonormal via QR),
- a shared **soft prompt** (`soft_tokens`, the implicit "what are you doing?" question).

Read-out = inject the source residual's in-subspace component `h + (h_src−h)·W_lᵀW_l` at layers 35–45
while greedy-generating from `[before_ids, soft_tokens, after_ids]` → decode. Trained by cross-entropy
against labeled intent blocks (`well-known-tasks/labels`), 5 epochs. Checkpoints:
`results/subspace_probe/Qwen2.5-Coder-32B-Instruct/{r=256,r=128}/probe.pt`.

## Eval setup (identical to GoalScope)
`hard_eval(readout, impl_span, mode)` is agnostic to how the read-out was produced, so the whole
GoalScope harness is reused (locked corpus `data/goalscope_eval.json`, all 4 datatypes → **253
datapoints × 5 pivots = 1265 records**, `span=frontier`, the exact frontier `impl_span`, the 5 anchors
`prev_end/indent/var/eq/first_impl`, the oracle grader, the summary). **Only the read-out producer is
swapped**: per pivot, `cache_source_residuals(full_text, pivot, layers 35–45)` → `verbalize_batch(probe,…)`.
Faithfulness uses the NL "helpful assistant" dest template the probe was trained on; causality prefills
the C++ stub (`CAUSAL_PREFIX`) as the assistant turn.

## Results — subspace probe (r=256, layers 35–45, ep4), n=253, hard_yes pooled

| pivot | faithfulness | causality |
|------------|:-----------:|:---------:|
| prev_end   | 0.040 | 0.008 |
| indent     | 0.281 | 0.004 |
| var        | 0.107 | 0.000 |
| eq         | **0.569** | 0.079 |
| first_impl | **0.605** | **0.198** |
| **OVERALL**| **0.320** | **0.058** |

(An 8-datapoint smoke gave 0.30, matching the full 0.32.)

## vs. goal-head GoalScope (same corpus, Axis (a) of `appE_appF__goalscope_eval_axis.md`)

| read-out | faithfulness | causality |
|----------|:-----------:|:---------:|
| **goal heads (122, `additive_norm`, α=0.1)** | **0.879** | **0.792** |
| **subspace probe (r=256)** | 0.320 | 0.058 |

**The subspace read-out is substantially weaker than the goal-head OV read-out** on this corpus
(0.32 vs 0.88 faithful). It *does* recover intent, but only near the pivot: `eq`/`first_impl` reach
~0.57–0.60, while `var`/`prev_end` are near floor. Two reasons: (1) it's a low-rank subspace + soft
prompt trained on a **different** label set (`well-known-tasks` sub-goal labels), so transfer to the
locked frontier corpus is lossy; (2) the goal-head path selects the *heads that do the work* and steers
their OV write, which Axis (a)/(b) already showed "does nearly all the work."

## Causality is off-distribution — not a fair comparison
Causality (0.058) is near floor because the probe's soft prompt was trained **only** on the NL template,
never the C++ transfer stub; reusing it under `CAUSAL_PREFIX` is out-of-distribution. This measures the
train/eval mismatch, not the subspace's intrinsic causal transfer. The goal-head `replace α=1.0` path is
the apples-to-apples causality baseline.

## Qualitative (smoke read-outs)
The probe emits coherent short intents; correctness is pivot- and task-dependent:
- **Palindrome tasks read cleanly at the good pivots**: `b = s == s[::-1]` → `eq`/`first_impl`/`indent`
  → *"check if string equals its reverse"* ✓ (and `list[str]` → *"check if any string equals its reverse"*).
- **Even-check tasks often misfire to a sibling predicate**: `b = (n % 2 == 0)` → *"non-negative" /
  "positive" / "perfect square"* ✗ — though `n % 2 == 0` sometimes reads correctly as *"last bit is set"* ✓.
- **The `var` pivot is uninformative** (near-garbage), hence 0.107 / 0.000.

## Notes
- `r=128` not run — expected to be no better (the bottleneck is the read-out mechanism + label
  mismatch, not subspace rank).
- Results: `results/evaluation/subspace_probe/Qwen2.5-Coder-32B-Instruct/{faithfulness,causality}/r256__L35_45__ep4/nspec253/`
  (`records.jsonl`, `samples.jsonl`, `summary.json`, `config.json`).
- Re-run: `python scripts/evaluation/eval_subspace_probe.py --mode {faithfulness|causality} [--probe … --cuda N]`.
