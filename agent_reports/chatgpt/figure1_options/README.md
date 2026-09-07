# DIFF Figure 1 concepts

Open `index.html` to compare all three options. Each has editable SVG, vector PDF, and PNG exports.

- **B_worked_example**: recommended starting point. Code and pivot at top; two causal edits below. Strongest explanation, but tallest.
- **A_parallel_panels**: three symmetric panels, each with inputs, DIFF, and continuation. Easy comparison, with more repetition.
- **C_compact_comparison**: shortest layout; best for a first-page teaser. Code and GoalScope would be explained in a later figure.

All three develop the chosen Option 1. Running-progress and cross-language figures are reserved for later sections rather than added here.

## Interpretation

These are conceptual mockups. No model run or oracle call supplied the displayed continuations or readout. Code assumes nonempty lowercase words. The central DIFF box summarizes selection by the relevant attention heads; it does not denote one physical head or layer, or literal vector subtraction.

The clean destination has implemented the palindrome filter. In the Q intervention, its actual code prefix stays unchanged, but the transplanted query comes from a context where the vowel filter is implemented. The illustrative palindrome continuation therefore repeats a computation already present in the destination. That illustrates changed subgoal selection, not successful completion of the whole task. In A and C, the blue patched entry describes the donor progress represented by Q, not an edit to the destination's code.

The KV intervention keeps the palindrome computation and replaces the task representation with a palindrome-plus-length requirement. Donor examples are chosen to explain the factorization; their exact match to a final experiment needs verification.

The final paper caption should state the settings of the actual displayed runs, including steering duration, patch span/head set, amplification, and any other-head ablations. Current diagrams intentionally abstract those settings. For the final empirical version, replace illustrative code/readouts with saved outputs and retain distinctions between subgoal pursuit and full correctness.

## Visual references

Inspected the local Figure 1 pages from:

- Function Vectors in Large Language Models (`2310.15213v2.pdf`): compact transfer arrows and concrete changed outputs.
- The Dual-Route Model of Induction (`2504.03022v2.pdf`): parallel mechanism schematics.
- LLMs Process Lists with General Filter Heads (`2510.26784v3 (2).pdf`): concrete prompt, query intervention, and changed attention/output.

## Regeneration

```bash
python agent_reports/chatgpt/figure1_options/build_figures.py
/usr/bin/python3 agent_reports/chatgpt/figure1_options/render_figures.py
```

The renderer uses the installed Linux librsvg/cairo libraries. System Python avoids the current conda environment's libffi conflict. SVG text remains text, so an SVG editor can adjust labels and layout.
