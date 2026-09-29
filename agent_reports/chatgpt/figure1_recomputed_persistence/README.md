# Figure 1 revision: functional persistence through recomputation

September 22, 2026. The [earlier figure](../figure1_functional_vs_llm/index.html) remains unchanged. No Overleaf files edited.

[HTML preview](index.html) · [SVG](recomputed_persistence.svg) · [PDF](recomputed_persistence.pdf) · [PNG](recomputed_persistence.png)

## What changed

The previous version showed three separate representations without explaining how continued pursuit could arise. This revision introduces the missing feedback connection: action changes context, and that context informs the next computation. The functional panel shows a proposed goal-reconstruction loop; the LM panel unrolls generation with explicit arrows from emitted tokens to the growing prefix. There is no arrow from one extracted head-output vector to another.

The figure focuses on persistence. The readout/C++ transfer panels are left in the earlier version so they do not compete with the feedback path in this revision. The selected head outputs and first capture pivot remain visible.

## Proposed caption

**Functional persistence can be compatible with repeated reconstruction of a goal.** (a) In this proposed account, an agent identifies the current sub-goal from the task and its progress, takes another action, and updates the context for the next computation. The same unfinished sub-goal can recur while actions change. (b) During LLM generation, emitted tokens extend the available prefix. Each sampled forward computation produces distinct head outputs. We capture those contributions at selected positions and test for recurring goal-related information.

## Interpretation

- This is a conceptual illustration, not a measured trajectory. Code fragments, capture indices, and vector patterns are schematic. The three snapshots omit intermediate tokens; each illustrated forward computation predicts only its next token.
- Recurring content is distinct from causal maintenance. The temporal experiments do not identify which combination of instructions, generated code, or earlier latent states supplies the later goal representation.
- Blue arrows highlight the generated-text route. They do not exclude access to earlier latent states through attention/KV caching or imply reprocessing the whole prefix from scratch.
- “Progress” in panel (a) denotes the situation after previous actions. It does not assert a separate progress register, a subtraction operation, or the DIFF hypothesis.
- Selected head outputs appear inside the current forward computation. They are not portrayed as the only components determining the next token.
- The top panel is a possible implementation of functional persistence, not a claim that every goal-pursuing system must reconstruct its goal this way.

## Rebuild

```sh
.venv/bin/python agent_reports/chatgpt/figure1_recomputed_persistence/generate.py
```

SVG labels remain editable and PDF fonts are embedded. The HTML preview includes the older figure in a collapsed comparison panel.
