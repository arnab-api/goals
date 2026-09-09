# GoalScope paper figure drafts

These revisions follow the central GoalScope panel in [the poster](../../../Overleaf-draft/REF/goal-poster.pdf): capture selected-head outputs at a marked code position, then steer a new context to describe or implement the sub-step. Figures are sized for a paper (7-inch full-width or 3.35-inch single-column), with 7–8.5 pt body text, no presentation title, and editable SVG/PDF output.

The code and outputs are schematic, not newly measured examples. The source task follows the poster's binary-count example, but the C++ illustration uses the matching threshold >2. The poster displays >5, so its exact output should not be presented as complete task transfer without discussing that mismatch. The faithfulness illustration concerns the current operation; the implementation illustration conveys the intended full predicate. Prompt text is abbreviated. Use actual saved continuations before publication.

## P1 — Describe and implement (recommended)

**Suggested caption.** GoalScope captures a sub-step signal and transfers it to new contexts. (a) At the marked source token t, we sum the selected attention heads' contributions in residual-stream coordinates to obtain g_t. The extracted signal steers separate runs of the same model to (b) describe the sub-step under a verbalization prompt or (c) implement it under a code-generation prompt. Source code is not included as text in either target prompt. Colored target text denotes schematic generated continuations; ⊕ denotes activation injection, not an input token. Generation is steered at the target's current position across a chosen layer band. Examples are illustrative.

The fork is the key message; source capture and target injections must remain visually distinct. This is the closest to the poster's high-level framing.

## P2 — Goal readout before implementation

**Suggested caption.** GoalScope verbalizes the sub-step encoded before its implementation appears in generated text. (a) We capture selected-head outputs at the highlighted assignment token. The gray continuation is shown only to identify the upcoming computation; it is causally downstream of the captured state. (b) Their summed direction steers a separate model run with a fixed verbalization prompt. In the norm-relative additive variant, r_{ℓ,s} ← r_{ℓ,s} + α‖r_{ℓ,s}‖ĝ_t at selected probe layers and decoding positions. Examples are schematic; the figure explains the extraction/injection procedure, not a measured success rate.

This is narrower than P1 and is suitable if the paper figure should introduce only the GoalScope readout. Do not infer early-encoding performance from the illustrative example. The diagram depicts the norm-relative variant used in DIFF evaluation; the poster also reports an unnormalized additive setting. Keep their alpha values and settings separate.

## P3 — Single-column describe/implement

Use the P1 caption. This version stacks the branches for a 3.35-inch column and retains readable text without shrinking a wide figure. Right-side colored arrows indicate the two target contexts; they are not a sequential dependency between target runs.

## Provenance and implementation

Reference inspected: Overleaf-draft/REF/goal-poster.pdf, page 1, central GoalScope panel. Code consulted in the preceding drafts: src/goals/goalscope.py and src/goals/head_loc_attn.py. The current drafts intentionally omit head counts, alpha values, layer ranges, and performance bars; these depend on the specific experiment and belong in the caption/methods once the result to illustrate is selected. The original poster has multiple source capture positions; the drafts focus on one early position to keep the procedure readable.

Run `python agent_reports/chatgpt/goalscope_paper_figures/generate.py` to regenerate. PDFs preserve physical dimensions and embed fonts; SVG text remains editable.
