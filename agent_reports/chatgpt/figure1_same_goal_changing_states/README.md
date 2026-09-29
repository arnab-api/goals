# Figure 1 — Same sub-goal, changing actions and activations

[HTML preview](index.html) · [Editable SVG](same_goal_changing_states.svg) · [PDF](same_goal_changing_states.pdf) · [PNG](same_goal_changing_states.png)

This revision implements the agreed simpler layout: one concrete goal, an aligned functional row and LLM row, changing local actions, a small context-feedback connection, and one cross-language branch. Earlier artifacts and Overleaf sources remain unchanged.

## Proposed caption

**The sub-goal stays the same while actions and activations change.** (a) Functionally, a primality-checking goal can organize several distinct actions before completion. (b) We investigate corresponding goal information in selected attention-head outputs before and during code generation. A pre-implementation encoding can guide a realization in another programming language. Repeated goal information at later positions is compatible with reconstructing the current sub-goal from the evolving context, illustrated by dashed arrows.

## Reading the figure

- The gold goal band is an abstract description, not a neural memory register.
- The three gold vectors depict different selected-head contributions. Their labels illustrate the same enclosing sub-goal at distinct stages; they are not verified per-position readouts.
- The first capture is at the assignment operator, before `n >= 2`. That later code is displayed to explain what follows the capture. Exact tokenizer positions and boundaries are not specified.
- The code boxes are excerpts from an illustrative primality implementation. They are selected stages of a continuation, not single forward-pass outputs. Intermediate tokens and local operations are omitted.
- The arrow from an encoding toward a code stage depicts a proposed action-guiding role. Other model components also participate in generation.
- Dashed arrows emphasize feedback through growing context. They do not establish that the goal encoding at one stage causes the later encoding, nor that token-mediated feedback is the only available route. Task instructions and previous latent states may also contribute.
- The purple branch depicts transfer from the first captured encoding to a separate C++ generation. Its snippet is illustrative and incomplete by design, rather than an execution-verified result.
- “Complete” denotes the functional endpoint after the final stage; no semantic completion detector or boundary-signal circuit is asserted.

Before publication, select a saved trajectory that supports the enclosing primality goal at the depicted capture positions, and replace the example transfer with its actual output. If readouts instead shift to local implementation steps, the diagram must reflect that result.

## Rebuild

```sh
.venv/bin/python agent_reports/chatgpt/figure1_same_goal_changing_states/generate.py
```

SVG text is editable; PDF embeds fonts. The HTML preview includes the earlier figure for comparison.
