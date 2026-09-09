# GoalScope teaser: explicit model-run capture

Reference inspected: Figure 1, page 1, Overleaf-draft/REF/papers/filter.pdf. Its dotted pivot-to-internals magnification is adapted here to show output capture rather than query transplantation.

**U1** keeps the poster's source-code/faithfulness-readout composition, adding a pivot-linked view of the source model run underneath. **U2** makes the model-run → cache → readout sequence more explicit.

**Proposed caption.** GoalScope reads out a sub-step from selected attention-head outputs. The highlighted source token is expanded into a schematic view of the source model run: rows indicate layers and the gold column identifies the same pivot position across those layers. We cache selected heads' output contributions at that position, sum them in residual-stream coordinates, and inject the resulting signal into a separate run of the same model with a fixed verbalization prompt. The probe does not receive the source code as text. In U1, gray code shows the later continuation, which is not part of the captured prefix. Head counts, layers, and token columns are schematic; tiles represent projected contributions, not raw queries or keys. The readout is illustrative and should be replaced with a saved example for publication.

Both are 7-inch paper figures, with editable SVG and embedded-font PDF. No equations or causality branch. The small repeated gold tiles show selected contributions across layers at one token, not contributions across multiple source positions.
