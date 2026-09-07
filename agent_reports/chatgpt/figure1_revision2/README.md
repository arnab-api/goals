# Figure 1 revision 2: show the transplant in the prompt

Read all six pages of `Overleaf-draft/REF/goal-diff.pdf`; visually inspected the concrete examples on slides 1, 3, 4, 5 and 6. These revisions replace abstract input boxes with prompt/code excerpts and activation arrows attached directly to tokens or instruction spans.

Open `index.html`. Both options have SVG, vector PDF, and PNG exports.

## D: code-first three-panel overview

Retains the chosen Option 1 structure: natural DIFF, query transplant, KV transplant. The natural example and query example use the same palindrome/vowel task. The KV example follows slide 5's shared sorting progress, with shortest-word/vowel and longest-word/palindrome instructions.

All displayed code and continuations in D are normalized, explanatory examples. They are not quoted experimental outputs. Assumes nonempty lowercase words and a nonempty input list. Query transplantation does not rewrite the destination prefix: the shown palindrome continuation repeats work already done there. Causal redirection of the next computation is distinct from correctness of the complete program.

## E: closer to the slide evidence

Two wide prompt-to-prompt panels, using slides 3 and 5. The Q arrow connects the final `if` tokens; the KV arrow connects the highlighted instruction spans. Python-to-C++ KV transfer is retained here as a second layout option, even though it could be moved to a later figure.

Prompt boilerplate is omitted. Q prompt whitespace and quoting are normalized. The C++ comparator is abbreviated with an explicit comment. Reported patched continuation expressions are retained, with line wrapping changed and trailing closing code-tag/function delimiters omitted. GoalScope labels are semantic summaries, not quotations. No attention heatmaps or numerical results are invented.

The slide 3 patched continuation uses `word`, while the displayed prefix binds `w`; it also repeats a palindrome test. E deliberately retains that reported expression. This is a reason to select a verified saved run before using the example as the final paper figure. Slide 4 illustrates malformed generation under repeated Q patching, reinforcing the importance of stating intervention duration.

Slide 5 says to patch KV in place of `DST.progress`, but its shaded regions and the repository implementation indicate the destination **task-description span**. Both figures depict the task-span intervention. The slide's unrelated source continuation is omitted by stopping the prompt at `return`.

## Remaining paper-figure decisions

- Select an actual saved run for the final Q and KV continuations. D emphasizes explanatory clarity; E preserves the supplied evidence more closely.
- State the selected head set, Q prefill/decode schedule, KV token-span alignment, output amplification, and any other-head ablations in the final caption or methods reference.
- Mark attention arrows as schematic unless replacing them with measured attention.
- The Q and K/V outlines identify sites, not edits to visible text. Token IDs and padding are abstracted in these shortened prompt diagrams.

Regenerate with `python agent_reports/chatgpt/figure1_revision2/build.py`, then `/usr/bin/python3 agent_reports/chatgpt/figure1_revision2/render_figures.py`.
