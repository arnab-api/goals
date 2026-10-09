# Pre-commitment ablation (Qwen2.5-Coder-32B)

**Question.** §3 argues the frontier sub-goal $g_k$ is *pre-committed*: GoalScope already decodes it at
`prev_end` (~60%) and `indent` (0.90+), before the implementation starts. Decodable-early is not the same as
*used*-early. Is the encoding at the pre-implementation tokens **needed** for the model to implement $g_k$
and for the goal heads at `eq` to carry it?

**Answer: no.** Making `prev_end`, `indent` and `var` context-free at every layer leaves frontier
implementation unchanged (0.921 → 0.921, 0/253 verdict flips) and costs the `eq` readout 4 pts
(0.961 → 0.921). Everything $g_k$ needs at `eq` can be (re)computed from the spec and earlier code.

## Setup

- Script: `scripts/evaluation/eval_precommit_ablation.py` (commit `64893a8`); job
  `job_list/goalscope_eval/precommit_ablation.json`, run `precommit_ky_20261008` on kyoto (≈58 min).
- Corpus: locked 253-spec Python corpus (`data/goalscope_eval.json`), frontier $j\in\{1,2,3\}$. The prompt
  ends at `    {var} =`; its last four tokens are the pivots `prev_end, indent, var, eq`.
- **Ablation:** at `prev_end`, `indent`, `var`, overwrite the residual stream at **every layer** with the
  latents of the same token in a context-free prompt (16 pad tokens + the bare token; pads keep it off the
  attention-sink position). Since every layer is replaced, the K/V that later tokens read at those positions
  are context-free too. `eq` is left intact.
- **Measures** (clean vs ablated, same spec):
  1. *Frontier implementation* — greedy first line (≤64 tok), Section-4 judge `judge_implements`.
  2. *GoalScope readout at `eq`* — 122 goal heads, additive_norm α=0.1, steer L35–46 (paper faithfulness
     config), faithfulness judge `hard_eval` against the **clean** implementation.
- Results: `results/evaluation/precommit_ablation/Qwen2.5-Coder-32B-Instruct/ablate=prev_end+indent+var__npad=16__ro=goal122-additive_norm0.1-L35-46__nspec=253__seed=123/`

## Results

| subset | n | impl clean | impl ablated | readout clean | readout ablated |
|---|---|---|---|---|---|
| **all** | 253 | 0.921 | 0.921 | 0.961 | 0.921 |
| int | 65 | 1.000 | 1.000 | 1.000 | 0.908 |
| str | 64 | 0.922 | 0.922 | 1.000 | 1.000 |
| list[int] | 60 | 0.750 | 0.750 | 0.900 | 0.883 |
| list[str] | 64 | 1.000 | 1.000 | 0.938 | 0.891 |
| $j=1$ | 151 | 0.927 | 0.927 | 0.954 | 0.934 |
| $j=2$ | 73 | 0.904 | 0.904 | 0.959 | 0.890 |
| $j=3$ | 29 | 0.931 | 0.931 | 1.000 | 0.931 |

**The ablation is real, not a no-op.**
- Cosine(clean, context-free) at L32: median 0.26 (`prev_end`), 0.31 (`indent`), 0.35 (`var`), tight IQRs;
  norm ratio ≈ 1 (0.98 / 1.08 / 1.05). The states are replaced at matched scale.
- Downstream text moves: the readout text at `eq` changes in 189/253 specs, the frontier generation in 18/253.

**…but the judged content barely moves.**
- Implementation: 0/253 verdict flips. The 18 changed generations are alternative correct (or equally
  wrong) implementations — implementation plasticity, same sub-goal.
- Readout: 16/253 verdict flips — 13 YES → not-YES, 3 not-YES → YES. The losses are mostly YES → NO_INFO,
  i.e. the readout goes content-free rather than reporting a *different* goal (e.g. clean "I'm checking if
  the number is even by using the modulus operator…" → ablated "Sure, please provide the line of code…").
  Net −10 specs; largest on `int` (−9 pts) and $j=2$ (−7 pts).

**14B smoke (n=8, not for the paper).** Same ablation: no change (1.0 / 1.0). Adding `eq` to the ablated
set: impl 1.0 → 0.0, readout 1.0 → 0.375 — ablating the token whose prediction *starts* the implementation
breaks everything, as expected, so it does not isolate pre-commitment.

## Interpretation

1. **Early decodability ≠ causal pre-commitment.** The $g_k$ encoding visible at `prev_end`/`indent` is
   not a state the model *relies on* later: removing every layer's context at those positions changes
   neither what the model writes nor (mostly) what the goal heads carry at `eq`. The goal is
   **recomputed** at `eq` from the spec and the earlier code — consistent with the non-recurrent framing
   in the teaser (no state is carried token-to-token; it is re-derived where needed) and with the TODO
   in `01_intro.tex` that "pre-commitment in its traditional definition is not feasible in LLMs".
2. **Suggested wording for §3.** Keep "decodable before implementation" as a *representational* claim;
   avoid implying the early tokens *carry* the commitment forward. E.g.: "The frontier sub-goal is
   decodable several tokens before implementation, but it is not stored there: making those tokens
   context-free leaves both the implementation (92.1% → 92.1%) and the goal-head readout at `eq`
   (96.1% → 92.1%) essentially intact. The model re-derives the sub-goal at each position from the
   specification rather than propagating a committed state."
3. The small readout drop is the one place the early tokens help: they make the `eq` encoding slightly
   more legible (mostly content-loss, not content-change).

## Caveats

- One model (Coder-32B), one corpus; readout judged against the clean implementation, single LLM judge.
- Cos / norm diagnostics are logged at one layer only (L32 = n_layer/2).
- `eq` and everything before `prev_end` are intact — the model has full access to the spec and prior
  lines, so this tests *the last three tokens'* necessity, not "any early commitment".

## Possible follow-ups

- Ablate the **entire preceding line** (or the whole previous sub-goal's span) instead of three tokens, to
  test whether any carried state matters beyond the spec.
- Block the goal heads' *attention* from `eq` to `prev_end..var` (cheaper, head-specific) instead of the
  full residual.
- Replicate on one more family (Llama-70B / Olmo-32B) for the appendix.

## Figures & table (paper style, `figstyle`)

Generator: `notebooks/fig_n_table_generators/precommit_ablation.ipynb` → `notebooks/Figures/precommit_ablation/`

- `precommit_ablation_main.pdf` — (a) frontier implementation, (b) GoalScope readout at `eq`; clean
  (grey) vs ablated (firebrick), pooled + per datatype.
  *Suggested caption:* **Early pivots are not needed to pursue the sub-goal.** Replacing the residual stream
  at `prev_end`, `indent` and `var` at every layer with context-free latents leaves (a) the model's
  frontier implementation unchanged and (b) the goal-head readout at `eq` nearly intact
  (Qwen2.5-Coder-32B, $n=253$). The sub-goal is re-derived at `eq` rather than carried from earlier tokens.
- `precommit_ablation_diagnostics.pdf` (appendix) — (a) cosine between clean and context-free latents at
  L32 per ablated pivot; (b) their norm ratio; (c) readout verdict transitions clean → ablated.
- `precommit_ablation_table.tex` — the table above, booktabs, for `\input`.
