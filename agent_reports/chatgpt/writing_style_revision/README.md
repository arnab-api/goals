# Abstract and Introduction: writing-style revision

Revised the active draft's Abstract and Introduction using the author's edits in Overleaf commit `8a85662`, the working-tree diff against that commit, and the Abstracts and Introductions of `filter.pdf`, `function.pdf`, and `dual-route.pdf` in `Overleaf-draft/REF/papers`.

[Compiled draft](revised_draft.pdf). The previous Abstract, Introduction, and PDF are saved under [before/](before/). [This pass's diff](revision.diff) compares against that saved working copy, so it isolates these edits from earlier draft changes.

## What to preserve in future revisions

- Begin with a concrete example or a direct scientific question. The author's opening asks whether the model knows what it intends to do next; the filter paper opens with finding fruit, and the dual-route paper with copying a sequence.
- Explain the abstraction through familiar computation. Retain the broader goal / instrumental sub-goal / implementation hierarchy. This revision uses merge sort for both the motivating example and the hierarchy.
- Name the mechanism, describe what it carries, and demonstrate what can be done with it. The filter and function-vector papers develop compact representations through extraction, causal intervention, and reuse across contexts.
- Organize the contribution paragraphs around scientific findings. Restore the author's sequence: verbalizable upcoming sub-goals; abstract and portable encodings; mediation of implementation.
- Use direct, affirmative prose and concrete verbs. Avoid turning the opening into a catalogue of experimental qualifications or implementation details. State claims at the scope the evidence supports, with detailed qualifications in the relevant experimental sections.
- Preserve author notes that remain unresolved. The terminology and teaser notes are restored; the request for a better opening example is addressed with merge sort.

## Factual distinctions retained

Cross-model findings are described as replications, while activation transfer is across programming languages within a model. Non-code and free-form extensions remain exploratory. The causal contribution describes preserving access to a task/code span and restricting a broader head set, rather than removing all other attention computation or asserting a unique necessary-and-sufficient bottleneck. The opening does not claim that verbalizability establishes a native natural-language plan or that OVReader establishes irreversible commitment upstream of every goal head.

## Validation

Compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`, using `/tmp/goals-writing-style-review` for build files. Final log has no undefined references/citations, multiply defined labels, or overfull boxes. Inspected the first two PDF pages. Existing package, float-placement, and margin-note warnings remain. This pass changes only the two opening sections and the compiled PDF in the active Overleaf tree.
