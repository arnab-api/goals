# Q-transfer update in the Overleaf draft — TODOs

*Opened 2026-10-12 · draft `Overleaf-draft/Goals-Overleaf` @ `291d9ad` (rebuilt clean: 45 pp, 1 undefined ref
`app:task_transfer`, 10 undefined citations) · results: `gaze_select_restrict.md` (gsr), `alt_hypotheses_set_diff.md`*

## Plan

Main text §5.1.2 (`sec:diff:q-transfer`) presents and discusses **order-diff only**. The caveats move to
Appendix I.2 (`app:transfer:q`):
- order-matched gives higher accuracy;
- the query consistently selects the source frontier when it is present in p_dst (I.3, `app:transfer:expb`);
- several mechanisms appear to contribute to selecting by progress (order, set-difference, …).

## TODO 1 — figures (order-diff, gsr)

Decisions (2026-10-12): **re-clamp is removed** from the text and figures; **kv-off only** (kv-on adds ≤ 0.04).

- [x] `q_crosslang_gsr.pdf`: order-diff, gsr, kv-off, causal-shift (solid) and readout-shift (hatched) by language
  direction. Overall .576 / .441; py→py .68/.55, py→cpp .55/.40, cpp→py .51/.39, cpp→cpp .56/.43.
- [x] `q_barcharts_gsr.pdf`: order-matched and order-diff rows, by direction and by datatype. Order-matched overall
  .871 / .709.
- [x] `q_gsr_matched.tex`, `q_gsr_diff.tex`: 4×4 grids for `tab:q-matched` / `tab:q-diff`.
- Generator: `notebooks/fig_n_table_generators/q_transfer_gsr.ipynb`. `transfer.ipynb` and the old
  `q_crosslang.pdf` / `q_barcharts.pdf` are untouched.
- Metrics match kv-transfer: both use `src/diff/eval.py`. Causal-shift = `causal_trigger_rate` (`judge_causal`,
  first computation, D vs native, `both` = hit). Readout-shift = `readout_shift_rate`. Aggregates are unweighted
  cell means, as in `transfer.ipynb`.
- Readout caveat (for the I.2 text): the gsr readout is taken at the pivot from the selection pass (query transfer
  + α boost, before the restriction), the same procedure as the q-transfer readout. It is not a readout of the
  restricted run. Order-matched: identical to the re-clamp run in 16/16 cells. Order-diff: 10/16 cells identical;
  97/512 readout texts differ (run-to-run noise).
- [x] Recompose `Figures/transfer/q-transfer-main-crop.pdf` with `q_crosslang_gsr` as the right-hand panel. The
  left panels illustrate re-clamp q-transfer and need a gsr version (ties into TODO 2).
- [x] Update the main-text numbers: the draft's "~53 %" pooled both orderings under re-clamp; order-diff gsr is
  57.6 % causal / 44.1 % readout.
- [x] Remove re-clamp from I.2 (`fig:q-barcharts`, `tab:q-matched`, `tab:q-diff` → the gsr versions above).

## TODO 2 — gsr in the main text + step-by-step visualization

- [ ] One or two sentences in §5.1.2: query transfer *selects* D (gaze picks it 51–90 % of the time), and a span
  restriction executes the selection almost losslessly (P(D | selected D) = 0.97). Details in I.2.
- [ ] Step-by-step figure for I.2:
  1. Transfer the SRC query at the pivot ` =`.
  2. Score the listed tasks by value-weighted gaze and select one.
  3. Restrict the goal heads to the selected span plus the pivot and the generated tokens.
  4. Generate.
  Show the gaze scores over the task list for one example (`cl_select_scores`) and the whitelist mask.

## TODO 3 — rewrite Appendix I.2 (`app:transfer:q`; currently marked "need a full rewrite")

- [ ] Setup: query transfer at the pivot (pre-RoPE), the gsr procedure (TODO 2), and the order-diff vs
  order-matched definitions. Re-clamp is not presented.
- [ ] Caveats moved from the main text: order-matched > order-diff, and the multiple-mechanism reading.
- [ ] Selection fidelity table (P(D | selD) = .972; no-query control 0/1024).
- [ ] Readout caveat: the readout is taken before the restriction (see TODO 1).
- [ ] kv arm dropped (decided 2026-10-12).
- [ ] Replace `tab:q-matched` / `tab:q-diff` and `fig:q-barcharts` with `q_gsr_matched.tex`, `q_gsr_diff.tex` and
  `q_barcharts_gsr.pdf`.
- [ ] Fix the self-reference "Similar to the kv-transfer in \Cref{app:transfer:q}" → `app:transfer:kv`.

## TODO 4 — alternative-hypothesis experiments in I.2

- [ ] Source: `agent_reports/claude/analysis/alt_hypotheses_set_diff.md`. SET_DIFF vs ORDER-ID / COUNT / INDUCTION;
  only ORDER-ID predicts a difference between order-diff and order-matched. The designs are validated but **not yet
  run**, so this needs runs first (or present the designs and the order-diff/matched gap only).
- [ ] The kv set-vs-count test is confounded in the current design (p_kv's 3rd slot is D; see gsr report §6). It
  needs a redesigned p_kv before it can go in.

## TODO 5 — document the intervention evaluation (not in the draft)

The draft only says an oracle judges causal and readout shift (§5.1.1 `06_mechanism.tex:89`, Appendix I intro
`task_transfer.tex:8`). `app:oracle_prompts` covers only the GoalScope faithfulness/causality judges. The commented
note `% \arnab{include full oracle prompts in the appendix.}` (`06_mechanism.tex:90`) is still open.

- [ ] Appendix I intro: one paragraph on the shared protocol, used by kv-transfer, q-transfer (gsr), multi-hierarchy
  kv and the I.3 setup.
  - Judge: `classify_concept` in `src/diff/eval.py` (`claude-sonnet-4-6`). Inputs are the task and two candidate
    sub-goals (A) and (B); the verdict is one of A / B / BOTH / NEITHER. It judges the computation, not identifier
    names; partial implementations and explicit mentions count.
  - Slot assignment per setup:
    - kv-transfer: A = source frontier c, B = destination frontier d;
    - q-transfer: A = NRGS-predicted D, B = native next task B;
    - multi-hierarchy: A = source's pending sub-sub-task y, B = the destination's own pending sub-sub-task z.
  - Causal-shift: `judge_causal` on the patched continuation, **first computation only** (later statements are
    ignored). Hit = A or BOTH.
  - Readout-shift: the same judge on the patched GoalScope readout at the pivot. Hit = A or BOTH. We report the
    **unconditional** rate. The `valid` flag (clean source readout = A and clean destination readout = B) is
    near-zero for q-transfer by design, because the source's own frontier C is neither A nor B.
  - Aggregation: unweighted mean over the 4×4 datatype × direction cells (n = 32 each); bars are means over one axis.
- [ ] q-transfer specifics (I.2):
  - the gsr readout is taken from the selection pass (query transfer + α boost), before the restriction;
  - a literal copy of the source's target C counts as NEITHER under the binary judge;
  - the 4-way `judge_joint` (B / C / D / neither) is used by the gsr report and, judging by its columns, the I.3
    table (`tab:expb`; verify the run).
    Say which judge each number uses.
- [ ] `app:oracle_prompts`: add the verbatim `CONCEPT_SYSTEM` + `_concept_prompt` (and `_joint_prompt` if `tab:expb`
  keeps the 4-way labels), plus the verdict parsing (`PARSE_ERROR` kept separate from NEITHER).
- [ ] Check the multi-hierarchy readout numbers in the I.1.1 rewrite against this protocol (0.80 unconditional;
  0.91 among valid cases), and say which one is reported.
