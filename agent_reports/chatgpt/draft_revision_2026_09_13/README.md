# Draft revision — 2026-09-13

Updated the active Overleaf source in `Overleaf-draft/Goals-Overleaf`; no Overleaf push was performed. [Read the compiled draft](revised_draft.pdf). Prior source files and the prior PDF are preserved under `before/`; [revision.diff](revision.diff) records textual edits.

## Main changes

- Rewrote abstract/introduction around readable and transferable subgoals; retained the title.
- Section 2 distinguishes capture positions, projected head contributions, intervention types, and repeated decoding interventions.
- Section 3 introduces localization before GoalScope and distinguishes readout faithfulness from implementation transfer.
- Section 4 combines head-budget results with corrected 122/250-head span-restriction and sink-attention results. Corrected the head denominator to 2,560 and stated the non-nested head-set limitation.
- Section 5 presents gaze, task-span transfer, query transfer, and OVReader positively, without a DIFF rejection narrative or a claimed unique selection algorithm. Corrected the qualitative query example to match the saved figure's unequal progress.
- Section 6 includes a concrete few-shot antonym example with actual saved readouts, exact capture positions, and domain-adapted probe wording. Quantitative non-code claims remain pending.
- Moved source/model comparisons to an appendix. Added intervention/OVReader details and a supporting boundary summary. Updated discussion and GoalScope appendix.
- Replaced the first-implementation OVReader plot with the assignment-pivot plot copied from Claude's analysis figures; caption states its corpus and scoring convention.

## Validation

Built with `latexmk -pdf -interaction=nonstopmode -halt-on-error` into `/tmp/goals-overleaf-review`, then copied the PDF into the Overleaf directory and this report directory. Final log has no undefined references/citations, duplicate-label warnings, or overfull boxes. Visually inspected representative result and main-text pages. Remaining warnings are typographic/package warnings (including margin-note placement). No experiments or model evaluations were rerun.

## Remaining author-facing work

- Choose/finalize the teaser and localization schematic; this revision does not silently select an unapproved teaser variant.
- Consolidate localization training/validation splits, optimization settings, and selection-file hashes. A completion note remains in the localization appendix.
- Quantify non-code relation readouts; the main example is explicitly qualitative. Saved notebook observations include successful relation readouts and nearby vague/mismatched outputs.
- Complete free-form code examples, gaze quantification details, and remaining pre-existing appendix stubs. These were not populated with invented results.
- Finalize the evaluation figure on the locked corpus. The available position-wise figure is explicitly labeled as the distinct 64-per-datatype OVReader corpus; its metrics are not presented as the locked-corpus breakdown.
- Existing related-work citation TODOs remain. Bibliography was not independently audited.

## Evidence provenance

Tables were transcribed from the current `agent_reports/claude/analysis/{goalscope_eval_axis,ablation_analysis,diff_eval_analysis,ov_reader_probe}.md` reports, with intervention semantics checked against the implementation. This revision is an editorial/evidence-alignment pass, not a fresh per-record statistical audit.

Few-shot example: `notebooks/clock/19_goalscope_out_of_domain.ipynb`, cell 12 saved HTML, positions 57 and 70 (Identifying antonyms), with contrasting positions 60 and 71. Probe configuration is in cells 7 and 9. The notebook scopes the continued generated sequence; these are not readouts before its first answer. No notebook or experiment files were modified by this revision.
