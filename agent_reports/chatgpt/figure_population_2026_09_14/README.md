# Figure population and review

Updated September 15, 2026, including the new `Overleaf-draft/REF/goal-figures.pdf`.

[Compiled review draft](revised_draft.pdf). The same PDF is saved as `Overleaf-draft/Goals-Overleaf/main.pdf`. [Figure numbers and pages](figure_locations.json) and [validation](validation.json) are generated from the final build.

## Scope and preservation

The main paper's prose, equations, title, section structure, and comments outside replaced floats are preserved. Thirteen top-level text files are compared against the pre-task snapshot, excluding figure/table environments and the new figure-input lines. `main.tex`, `packages.tex`, and every exact prompt/oracle file remain byte-identical. Four main-text files receive only figure inputs or replacement of existing floats. The original ablation table moves to the appendix; its existing main-text references now resolve to the paired chart.

The only author instruction removed from the visible main draft was inside the replaced evaluation caption: “this figure should show the pivot and the brief faithfulness and causality prompts on the left, followed by the actual results on the right.” That instruction is fulfilled. All original files, including that caption, are preserved in `before/`.

No model runs, oracle calls, commits, pushes, or changes to experiment results were made.

## What is populated

| Location | Figure/content | Source |
|---|---|---|
| Introduction | GoalScope teaser: source pivot → cached head contributions → aggregate → verbalization | New figures, p. 2; caption follows the author's proposal |
| Readout evaluation | Brief faithfulness/causality prompts with all five pivot highlights, beside results | New figures, p. 3 + matched 253-specification results |
| Goal-head section | Head-budget curve and native ablation bars, side by side | Saved GoalScope and native-ablation evaluations |
| KV-transfer section | Actual Python evenness → C++ divisibility example; 2×2 language score matrix | Saved 250-head KV run, no collateral ablation |
| OVReader section | Faithfulness and causality together, datatype SD, and the 122-head layer distribution | Complete archived OVReader cohort |
| Corpus appendix | Author's illustrative prompt, alongside the existing exact saved prompt and distribution tables | New figures, p. 4; existing byte-exact prompt retained |
| Localization appendix | Contrastive source/destination/target example and learned-mask heatmap | New figures, p. 5 |
| GoalScope appendix | Enlarged pivot/probe diagram | New figures, p. 3 |
| Ablation appendix | Full prompt-span diagrams, exact aggregate rates, extra span restrictions, and an observed continuation example | New figures, p. 7 + saved ablation records |
| Transfer appendix | Shared sorting-step KV example, datatype matrices, query-order contrast | New figures, p. 9; old slides, p. 18; saved KV results |
| OVReader appendix | Other four pivots and exploratory projection example | Saved archived results; old slides, p. 19 |
| Gaze appendix | Qualitative task-description highlights | Old slides, p. 18 |
| Model appendix | Existing matched model-family/size figures and tables retained | Existing verified draft artifacts |
| Unspecified-task appendix | Stable matching and string-to-integer readouts | New figures, p. 6; old slides, p. 4 |
| Non-code appendix | Musician/instrument and antonym examples | Old slides, p. 22 |
| Boundary appendix | Forward- and reversed-order examples | Old slides, p. 23 |

The revised teaser uses the clearer distinct-word example and only faithfulness, following the new drawing. The adjacent evaluation figure carries the implementation-transfer demonstration. The new slide deck's blank ablation-effects page is represented by a saved continuation example; its blank query-transfer page is supplemented with the older task-order example.

## Corrections to illustrations and captions

The reference PDFs themselves are unchanged. Corrections appear only in derived paper assets or typeset captions.

- New teaser caption: changed the named pivot from `a` to the visibly highlighted `b`.
- New task and ablation prompts: corrected the uppercase comprehension to `sum(1 for w in words for c in w if c.isupper())`. The original ordering references `w` before binding it. The illustrated prompt is explicitly distinguished from the byte-exact saved evaluation example.
- New sufficiency diagram: moved the query-position outline from the preceding `b =` to the final `c =`.
- New ablation caption: layers **20–63**, including layer 20; generic `<bos>` denotes key position zero, which is `<|im_start|>` in Qwen. Sink restriction retains that value contribution and is not zeroing.
- New free-form KV illustration: removed the accidentally duplicated C++ function declaration. Task descriptions, continuations, and readouts otherwise remain as provided.
- New KV caption: omitted the “swap once into the KV-cache” implementation claim. The evaluated GQA-corrected implementation precomputes the source span once, retains destination cache values, and routes source-span attention outputs to selected query heads on every forward pass. See `src/diff/functional.py`, `headwise_kv_patch` and `run_kv_transfer_intervention`.
- Old query slide: the destination has completed only A, so its native next goal is **B**, not the slide's labeled D. D is the prediction after transferring the *source's completed-task set*. The incorrect annotation is excluded; the appendix explains the contrast.
- Code after each highlighted source pivot is described as context, not text available to the earlier cached state.

## Quantitative provenance

`build_results.py` reads saved records and recomputes all newly generated numerical panels. `result_sources.json` hashes 393 input files; `results.json` stores the aggregate rates.

**GoalScope and sparsity.** The plots use `results/evaluation/goalscope/Qwen2.5-Coder-32B-Instruct/{faithfulness,causality}/additive_norm`, with `span_source=frontier`, 253 specifications, α=.1, and the same specification/position identifiers in every included run. At `=`, the 122-head scores are 97.2% faithfulness and 95.7% causality; 25-head scores are 93.7% and 90.5%. The earlier main evaluation plot used a different 256-example cohort, explaining the author's mismatch comment.

**Native ablation.** Eight complete 253-specification runs supply the main bars and appendix tables. Clean success is 92.5%. Preserving 250 selected heads' access gives 91.3%, versus 9.5% for the random condition; sink restriction gives 15.8%, versus 92.1% for random heads. The qualitative example is exactly `int_python_000`, also used in the existing exact-prompt appendix: evenness survives goal-head span preservation, whereas sink restriction produces `nn\`` and is judged MALFORMED.

**KV transfer.** The matrix uses all 512 records under `kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad`, at `var_eq`. Each language cell has 128 examples, 32 per datatype. Causality counts source-only or both-goal verdicts; braces count exclusively source-directed patched readouts, using all examples in the denominator. These differ conceptually from a shift rate conditional on correct original readouts. `kv_example.json` preserves the exact source/destination prompts, generated continuations, and readouts used in the main schematic.

**OVReader.** The current `results/evaluation/ov-reader` run was incomplete during this work. The figure therefore uses the complete `ov-reader__backup_span-rule_pre-locked_2026-09-14` archive: 19 layer settings plus the head-output reference, two modes, four datatypes, 64 examples each, and five positions. All 160 included runs have matching specification/position identifiers within each datatype. Bands are ±1 sample SD across the four datatype rates, clipped to [0,1]; they are not uncertainty intervals across seeds. The archived cohort is explicitly distinguished from the matched 253-specification cohort. OVReader uses α=.15, versus .1 for the head-output reference.

## Main-text issues deliberately left for the author

1. **OVReader's linear formulation differs from the evaluated operator.** `OVReader.resid_ov_vec` in `src/goals/goalscope.py` unit-normalizes each transformed head vector before averaging. It cannot generally be represented as one sum of linear matrices. The appendix describes the implementation accurately; the main equations remain untouched.
2. **The broader sufficiency control does not match effective in-band head count.** The 250-head goal condition spares 236 heads in layers 20–63; its random condition spares 170 (1,524 versus 1,590 blocked). The 122-head comparison does match. The caption/appendix define the intervention and report the counts; the main interpretation is preserved for review.
3. **OVReader accessibility is not itself a causal time of goal selection.** Selected heads span layers 28–55, overlapping the effective readout depths. The plot shows where this probe reads information, rather than proving the first commitment event.
4. **KV notation typo and placeholders remain.** The main text introduces destination tasks `(A,D)` but subsequently calls the frontier C. Literal red “Figure/Table X” references, wording TODOs, and the author's undecided query subsection are retained. Actual figures have stable labels for a later prose pass.
5. **Exploratory examples are not aggregate correctness results.** The free-form, query-order, non-code, and boundary crops remain qualitative. I did not import the old slide's KV “restrict” scores into the no-ablation matrix, or replace the existing matched model comparisons with older slide charts.

## Rebuilding

Run from the project root:

```bash
python agent_reports/chatgpt/figure_population_2026_09_14/crop_slides.py
python agent_reports/chatgpt/figure_population_2026_09_14/crop_updated_figures.py
python agent_reports/chatgpt/figure_population_2026_09_14/build_results.py
python agent_reports/chatgpt/figure_population_2026_09_14/build_kv_example.py
```

Then, from `Overleaf-draft/Goals-Overleaf`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/goals-figure-population-build main.tex
```

Finally, from the project root:

```bash
python agent_reports/chatgpt/figure_population_2026_09_14/verify_and_export.py
```

`populate.py` records the one-off document integration against `before/`; it is **not** a general rebuild step after further author edits. Rebuilding the assets and compiling the current TeX does not require running it. PDF crops preserve vector drawing/text where present in the source; screenshots already embedded in the slides retain their original raster resolution.
