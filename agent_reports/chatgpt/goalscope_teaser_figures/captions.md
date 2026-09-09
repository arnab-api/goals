# Poster-style GoalScope teaser revisions

Only the faithfulness/readout context is shown. No equations or causal implementation branch. Each gold tile represents a selected attention head's contribution projected into residual-stream coordinates at the highlighted source pivot. The contributions are summed and injected into a separate run of the same model. Tiles are schematic, not an actual count or layer arrangement.

**Recommended: T1**, closest to the poster's direct source-to-readout composition. T2 moves the cache to a pivot-linked inset. T3 is a more explicit compact sequence, showing only the source prefix.

**Proposed caption.** GoalScope reads out the sub-step encoded at a code position. At the highlighted pivot, we cache the selected attention heads' contributions and sum them into a steering vector. Injecting this vector into a separate run of the same model, prompted to describe its current activity, yields a natural-language readout. The target prompt does not include the source instructions or code. In T1/T2, gray text shows the later continuation, which is not part of the captured prefix. Gold tiles schematically denote projected head contributions. Prompt wording is abbreviated and the example readout is illustrative; replace it with a recorded example before publication.

7-inch paper figures with 7–8.5 pt text. Editable SVG, embedded-font PDF, PNG, combined PDF, and HTML comparison included. Reference: Overleaf-draft/REF/goal-poster.pdf, central GoalScope panel. Previous options remain unchanged. Regenerate with `python agent_reports/chatgpt/goalscope_teaser_figures/generate.py`.
