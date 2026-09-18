# Appendix A–E review — September 16, 2026

Author baseline: Overleaf commit `5c4c138` (“Appendix A, B, C”). Starting source/PDF copies are under `before/`. Changes are uncommitted.

## Edits

- A–C: spelling, agreement, punctuation, and small sentence-level repairs. Paragraph structure, voice, claims, figures, and commented-out material were retained. No old prose or tables that the author removed were reinstated.
- A: resolved the request to add Python stubs to the corpus table. All twelve author-selected sub-tasks remain. Stubs come directly from `src/diff/data.py`, instantiated with readable input/result names. Corrected “words longer than 5 words” to “words longer than 5 characters.” Removed the caption's verbatim-description claim because some author descriptions paraphrase the bank.
- C: consistently used `v_g` and `mathcal G` in the additive formula, corrected head-index membership in the replacement formula, and answered the recursion comment with one sentence. An assignment arrow reads the pre-injection state and overwrites it; it is not a self-referential equation.
- D: filled in the baseline comparison and head-budget subsection. Recomputed baseline bars using all 253 specifications at the four pre-implementation pivots. Added a four-panel head-budget figure including the assignment pivot.
- Explicit main-figure exception: changed the head-budget panel to the requested average over four pre-implementation pivots, and updated its caption. The right-hand ablation results are unchanged numerically. No main-body prose was edited; its assignment-pivot percentages remain correct and can be read against the new appendix breakdown.
- E: populated size, code-specialization, and OLMo instruction/SFT comparisons, with three pairwise figures, a four-pivot aggregate table, and four complete matched OLMo readout examples. The existing later model-family appendix remains intact as the five-position/configuration reference.

## Substantive issues for the author — not silently rewritten in A–C

1. **B: learned masks are not binary during optimization.** `src/goals/head_loc_attn.py:98` defines a real-valued parameter, clamped to `[0,1]` after each update. Exporting a discrete selected set is a later threshold/ranking operation. The sentence calling the jointly learned values “binary” was left for author revision.
2. **B: the loss matches distributions by minimizing KL.** “Match the KL divergence” is imprecise: the implementation minimizes `KL(P_patched || P_target)` plus the mean-mask sparsity term (`src/goals/head_loc_attn.py:1623`). The displayed KL direction is consistent with that implementation; the wording is the issue.
3. **C: replacement text says “destination run,” but the donor is the source run.** The equation already uses `a_src`. The prose claim was left unchanged, apart from its possessive punctuation.
4. **E: “almost always” is too strong for the inspected OLMo readouts.** Literal `I cannot` / `I can't` appears in 402/1,012 pre-implementation instruction-model readouts (39.7%), versus 11/1,012 SFT readouts (1.1%). At the assignment pivot it appears in 40/253 (15.8%); at previous-line end, 192/253 (75.9%). This is a case-insensitive phrase diagnostic, not a comprehensive refusal classifier. The new prose reports the actual counts. Twenty-two of the 402 phrase-containing responses receive `YES` under the partial-content rubric.
5. **E: training comparisons are not isolated causal effects of size/code training/RL.** Models use independently selected head sets and different checkpoints. The author's subsection headings are retained; the new paragraphs identify what improves, what declines, and what is held fixed. OLMo causality increases while faithfulness decreases.

An additional wording tension in A is that the figure caption calls the prefilled code “the LM's generated code,” while the table caption correctly describes curated implementation stubs. The completed-prefix code comes from the templates; the frontier continuation used for judging is generated. I did not rewrite that claim in the figure caption.

## Results and provenance

All new aggregate rates exclude `first_impl`: 253 specifications × four pivots = 1,012 observations per mode. Matching specification/pivot keys were asserted across source baselines, head budgets, models, and modes. These are repeated observations, not 1,012 independent specifications.

| Comparison | Faithfulness | Causality |
|---|---:|---:|
| Additive: 122 heads | 85.3% | 75.0% |
| Additive: full attention, L40–49 | 32.6% | 29.2% |
| Replace: Qwen Coder 14B | 51.9% | 30.5% |
| Replace: Qwen Coder 32B | 78.9% | 73.5% |
| Replace: Qwen non-Coder 32B | 23.8% | 44.1% |
| Replace: OLMo SFT | 52.9% | 45.2% |
| Replace: OLMo Instruct | 43.5% | 62.2% |

`results.json` records per-pivot and pooled rates. `sources.json` records SHA-256 hashes of read configurations and verdict records. The main source-comparison records use additive normalization (the residual runner calls this `resid_norm`); model comparisons all use replacement with alpha 1, matching the existing model-comparison setup. No new inference or oracle calls were made.

The four OLMo examples were selected to show generic refusal, refusal containing correct task information, and a normal instruction-model readout where SFT fails. They are labeled illustrative and are not the basis of the aggregate rates. Exact identifiers and strings are in `olmo_selected_examples.json`.

## Build and scope checks

The updated PDF is `Overleaf-draft/Goals-Overleaf/main.pdf`, also copied here as `revised_draft.pdf`. Figures and table layouts were visually checked after rebuilding with `latexmk`.

`verify.py` checks byte preservation of all existing text outside the authorized six files, preservation of A–C's commented material, four-pivot arithmetic, and LaTeX overflow/reference warnings. All main-body source files remain byte-identical. The only main-figure source change is the explicitly requested sparsity caption/panel.

Existing reference issues from the author baseline remain: `app:pivots` has been commented out, the oracle appendix input defining `app:oracle_prompts` is disabled, `fig:goal-eval` is still absent, and the two new free-form examples share `fig:freeform-matching`. These were left outside this A–E prose/data pass rather than undoing the author's structure changes.

`build_results.py` can regenerate the quantitative assets from saved records. `edit_draft.py` records the initial textual pass against the author snapshot; do not rerun it over subsequent edits. The final layout additionally begins Appendix D on a new page.
