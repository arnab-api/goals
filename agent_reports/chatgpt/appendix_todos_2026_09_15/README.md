# Appendix population — September 15, 2026

Starting Overleaf commit: `737f0e3` (`appendix suggestions.`). The starting text and PDF are preserved in `before/`. Changes are uncommitted.

The revised paper is saved as `Overleaf-draft/Goals-Overleaf/main.pdf`, with a review copy in this directory as `revised_draft.pdf`.

## Completed requests

| Author request | Result |
|---|---|
| Datatype counts and full example task descriptions | Replaced the distribution table with four datatype groups, specification counts, and three verbatim bank descriptions per datatype. |
| Tie evaluation corpus to results | Added section references; distinguished the locked 253-specification corpus from the archived 256-specification OVReader cohort and transfer-pair datasets. |
| Put capture positions and scoring in evaluation | Moved both to Readout Evaluation; restored the enlarged position/readout illustration there. |
| Full faithfulness and causality prompts | Both generation prompts appear in matching boxes, with literal `\n` markers. Existing raw oracle prompts and their display copies are unchanged; the oracle section remains at the end. |
| Steering choices | Added a comparison table for norm-relative addition and head replacement; stated coefficients, layers, duration, and the absence of a systematic hyperparameter search. Distinguished steering alpha from localization sparsity lambda. |
| More localization detail | Added the actual KL-plus-mean-mask objective, ordered training/validation splits, optimizer settings, augmentation policy, and two tables of checkpoint/export metadata. All five exported selections were checked against their checkpoints. |
| Source comparison as a bar chart | Recomputed all eight bars from the saved 1,265-record runs. Corrected displayed layer endpoints to the inclusive equivalents of the configured half-open ranges. |
| Separate necessity and sufficiency | Created separate appendix sections/files. Each has its own setup illustration, result table, and exact saved continuations. Necessity includes task-only, task-plus-progress, and sink examples. |
| KV aggregate, then datatype plots | Replaced the datatype tables with two bar charts, recomputed from 512 unique saved pairs. Moved the free-form illustration and query example after these results. |
| OVReader derivation and alternatives | Added the implemented per-head normalized OV construction, read/write/eigen alternatives, and a full-text layer-40 qualitative table from the notebook. Cropped the prompt separately from the slide's truncated comparison table. |
| Separate OVReader faithfulness and causality plots | Added two five-pivot figures, each with faint datatype curves, a mean curve, and the actual-head reference. |
| Functional dissociation figures | Located and incorporated the existing goal-versus-impulse overlap and move-on plots, with the actual task/metric and unmatched head-budget caveat. |
| Missing figures | Added a clearly labeled MLP path-patching placeholder, preserving the author's experiment TODO. No matching result figure was located. |

The programming-language subsection now points to the populated cross-language results and free-form examples rather than remaining empty. Existing model-family tables and plots were retained.

## Evidence and limits

- `results.json` contains recomputed source, ablation, KV, OVReader, and localization values. `sources.json` records SHA-256 hashes of the input records, configs, checkpoints, selections, and additional notebook/figure sources used.
- OVReader plots deliberately retain the complete archived rule-span cohort used by the unchanged main-paper plot. The newer causality rerun was incomplete when inspected. Neither partial runs nor different cohorts were silently pooled.
- The 250-head sufficiency comparison spares 236 goal heads versus 170 random heads inside layers 20–63. This mismatch remains explicit beside the table.
- The notebook's eigen variant performs QR on real and imaginary eigenvector components without rank truncation. Redundant/zero columns can add QR completion directions. The appendix states this limitation; a rank-aware rerun would be needed before interpreting this as a strict leading-eigenspace comparison. No model runs were launched.
- The notebook displays verdicts truncated to three characters. `NO_` was expanded to `NO_INFO`, the unique matching verdict in its declared verdict mapping. Generated descriptions were recovered from its full-output cell, not the truncated slide table.
- The task-only and task-plus-progress necessity spans include the token immediately before the frontier description, as documented by the implementation. They differ in whether the blocked span ends at that description or continues to the prompt's final token.
- Existing main-paper references to `fig:goal-eval` are unresolved because the corresponding figure input is commented out in the author baseline. These were left unchanged under the main-paper protection instruction.

## Validation and files

- Compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/goals-appendix-todos-build main.tex`.
- Inspected rendered appendix pages, including all new plots, paired ablation pages, prompts, localization tables, OVReader examples, and dissociation figures.
- `verify.py` checks byte-for-byte preservation of main-paper text, main figure includes, `main.tex`, `packages.tex`, the existing appendix driver, and original oracle/prompt files. It also checks figure paths, prompt newline rendering, overflow warnings, and unresolved references. Its output is saved in `verification.json`.
- New charts and reused dissociation assets are under `Figures/appendix/`. Table bodies are under `Text/Appendix/tables/`. Former `mechanistic_details.tex` now includes the separate sufficiency, necessity, transfer, and OVReader files.
- `build_assets.py` rebuilds quantitative charts and data tables without inference. `populate_appendix.py` records the initial population pass against the preserved starting snapshot; subsequent layout edits are in the final TeX files. **Do not rerun that initial population script over later author edits.**
- The vector crop for `ov_example_prompt.pdf` uses `pdfcrop --bbox '0 109 245 265' Figures/slides/ov_example.pdf Figures/appendix/ov_example_prompt.pdf` from the Overleaf directory.

No main-paper prose, equations, captions, or main figure assets were edited. No GPU jobs, oracle calls, commits, or pushes were performed.
