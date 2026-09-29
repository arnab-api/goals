# Figure 1: functional description and LLM implementation

September 22, 2026. A schematic figure for discussion. No Overleaf edits.

- [HTML preview](index.html)
- [Editable SVG](functional_vs_llm.svg)
- [PDF](functional_vs_llm.pdf)
- [PNG](functional_vs_llm.png)

The top row describes one sub-goal through selection, action, and continued pursuit. The bottom shows three distinct computations at selected positions of a Python continuation. Highlighted tiles are selected attention heads inside each model run; the gold vectors are their aggregated output contributions. Only the first captured vector supplies the two separate readout/transfer runs.

The code, vectors, and readout are illustrative, not experimental records. Position labels identify schematic snapshots and do not specify exact tokenizer boundaries or token distances. The vectors' shared color indicates related goal content; their changing patterns prevent an implication of literal state identity. There are no arrows between vectors asserting causal propagation. Prefixes include the earlier context, supplied through ordinary transformer computation and, when used, its KV cache; the figure does not assume full prefix recomputation or resolve the mechanism of goal maintenance.

## Proposed caption

**A sub-goal can remain represented while the neural states carrying it change.** (a) A functional description connects a selected goal with its implementation and continued pursuit. (b) At three illustrative positions, the LM processes the available prefix, contributes selected head outputs to the computation, and predicts a next token. We capture those outputs separately at each position. The pre-implementation encoding is cached for injection into separate runs that describe the operation or implement it in another programming language. Comparing representations over positions investigates temporal persistence of goal-related information.

Before publication, replace the schematic source/continuation, verbalization, and C++ transfer with a matched saved example. Temporal persistence here concerns goal-related content, not a demonstrated causal maintenance path. No obstacle-recovery or semantic-completion detector is depicted. The functional row is our synthesis rather than a reproduction of one philosopher's diagram; citation choices are in the [state-machine reference report](../figure1_goal_state_machine_references_2026_09_22.md).

## Regenerate

From the project root:

```sh
.venv/bin/python agent_reports/chatgpt/figure1_functional_vs_llm/generate.py
```

SVG text remains editable; PDF fonts are embedded. PNG is a 200-dpi preview.
