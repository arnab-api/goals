# Figure 1: editable refinement of the supplied reference

[HTML preview](index.html) · [SVG](goal_pursuit_reference_refined.svg) · [PDF](goal_pursuit_reference_refined.pdf) · [PNG](goal_pursuit_reference_refined.png)

Reference: [user draft](../../../Overleaf-draft/REF/fig1_draft.png). This is a new vector drawing that follows its layout; the supplied image, previous proposals, and Overleaf manuscript have not been changed.

## Preserved structure

- Three phases with aligned functional and neural rows.
- A context tape between those rows.
- Context-read paths entering the current computation and a distinct next-token write path.
- Head-output samples spanning pre-implementation and implementation positions.
- A transfer branch from the pre-implementation sample into a C++ continuation.

## Refinements

- “About to do” replaces “gonna do”; the first vertical divider marks action onset rather than a proved commitment event.
- The source expression is `b = max(len(w) for w in words)`. The diagram depicts a current step about to emit `in`; `words)` is muted future text and has no read path. Token tiles are schematic display units, not tokenizer-verified boundaries. The source assumes a nonempty collection; the C++ illustration also supplies an empty-input default.
- The selected pre-implementation capture is at `=`. No unverified three-token distance is claimed.
- Read paths are blue; the next-token write arrow is red-brown. The completion test remains explicitly schematic.
- The output glyphs use different patterns with the same mean intensity, avoiding a suggestion that an affinity curve was measured from these values.
- The C++ function returns `std::size_t`, and the accumulator matches `w.size()` for string inputs. The function uses a C++20 abbreviated template and assumes ordinary string collections and appropriate headers. The snippet is illustrative, not a measured transfer result.
- The sampled head-output glyphs are abstract projections/contributions; no vector is shown being passed unchanged to later positions.

## Proposed caption

**Functional attributes of goal pursuit and their neural counterparts in an LLM.** (i) A schematic functional account: the system forms an intention to perform sub-goal \(g_k\), pursues it, and moves on after completion. (ii) We investigate corresponding representations during code generation. **Before action:** the upcoming sub-goal—here, “get the length of the longest word”—is readable in selected attention-head outputs before implementation begins. **Persistence:** goal-related information remains represented across sampled positions during implementation, while the generated tokens and the activations carrying that information change. This is compatible with repeatedly reconstructing the current sub-goal from the evolving context. The context arrows illustrate this proposed account; the completion test is schematic. **Plasticity:** an encoding captured before implementation can be transplanted into another programming-language context to guide a different realization of the same operation. Here, the captured encoding steers a C++ continuation toward computing the maximum word length.

The figure itself is illustrative. Before publication, replace snippets and captures with a matched experimental example and calibrate the persistence wording to the final temporal analysis. Neither the dashed neural boxes nor their connecting arrows establish a finite-state controller or a specific progress-comparison algorithm in the LM. The context tape includes ordinary attention-accessible information, not only token-mediated influence; no exclusive maintenance route is asserted.

## Rebuild

```sh
.venv/bin/python agent_reports/chatgpt/figure1_reference_refinement/generate.py
```

SVG text remains editable and PDF fonts are embedded. The HTML page includes a collapsible comparison with the supplied reference.
