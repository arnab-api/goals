# Main-text writing-style revision

Continued the style revision through Sections 2–8 of the active Overleaf draft. The style reference remains the author's edits in commit `8a85662` and the filter, function-vector, and dual-route papers: introduce a concrete phenomenon, explain the intervention that tests it, and state the finding directly.

[Read the compiled draft](revised_draft.pdf). [This pass's diff](revision.diff) compares against the working copy saved immediately before editing. Prior source files and the PDF are under [before/](before/).

## Changes

- **Setup:** introduced the actual digit-count/evenness code prefix from the appendix before defining the frontier and pivots. Clarified that the dataset's term “predicate” includes counts and transformations as well as Boolean tests.
- **GoalScope:** kept localization before readout, made the head-output caching and repeated probe injection explicit, and separated faithfulness from implementation transfer. Led the results with the assignment-pivot scores already reported in the sparsity section; retained the pooled results and the distinction between evaluation corpora.
- **Goal heads:** organized sparsity and access restrictions around the questions they answer. Explained the ablation outcomes using percentages from the existing table. Preserved the intervention definitions, controls, and non-nested-selection qualification.
- **Mechanism:** developed gaze, KV transfer, query transfer, and OVReader as a sequence of experiments. Retained the unequal-progress task-order example, source-or-both scoring definition, patch schedule, and limits on localizing progress or the origin of goal selection.
- **Generalization:** led with the saved few-shot antonym example. Retained the timing of both captures, the adapted probe, and nearby failures. Described free-form code as an extension requiring examples rather than referring readers to a completed analysis in an empty appendix.
- **Related work and discussion:** connected prior portable representations to the upcoming sub-goal and removed an unrelated failure-mode cross-reference. Reworked the discussion around abstraction, selection, and scope rather than repeating an experiment inventory.

## Remaining draft material

The gaze appendix and free-form-code appendix are still incomplete. Source comments mark the missing plots, analysis, and examples. Existing related-work citation notes and localization configuration TODOs remain. The earlier Abstract/Introduction revision and author notes are preserved. Exact appendix prompts, data tables, and figure assets were not edited in this pass.

## Validation

Compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error` into `/tmp/goals-main-text-style-review`. The final log has no undefined references/citations, multiply defined labels, or overfull boxes. Inspected result, ablation, OVReader, generalization, and discussion pages. Existing package and float-placement warnings remain. The changed source files pass `git diff --check`; an existing whitespace warning in `packages.tex` is outside this pass. Numerical table bodies were checked against the saved pre-edit copy. No experiments were rerun.
