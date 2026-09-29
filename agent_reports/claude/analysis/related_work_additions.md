# Related Work — assessment + gap-filling drafts

*2026-09-24. A full `Text/06_related_works.tex` already exists (7 paragraphs, `\input` in the Appendix,
main-text input commented in `main.tex`). It is strong and largely aligned with the current intro. This
doc does **not** rewrite it; it (a) assesses alignment with the rewritten intro, (b) drafts the two
paragraphs it is missing, and (c) gives integration/ordering notes. LaTeX-ready; real bib keys used where
they exist, `[CITE: …]` placeholders otherwise (reconcile with `references.bib`).*

## What the existing section already covers (keep)

1. Functional accounts of goals/intentions (Dennett, Bratman, Williams, Lee).
2. Persistence, completion, memory for goals (Cohen–Levesque, Goschke, Marsh, Altmann).
3. Anticipation/planning in LMs (Future Lens, circuit tracing / biology-of-LLMs).
4. Portable task/operation representations (task vectors, function vectors, concept-induction, filter heads).
5. Reading/steering internal representations (logit/tuned lens, Patchscope/SelfIE, activation addition).
6. Causal interventions & mechanistic explanation (causal mediation, causal abstraction, backup/Hydra).
7. Generalization beyond explicit coding tasks (function composition, task superposition, emergent misalignment).

This maps cleanly onto the intro's contributions. Two gaps below.

## Gap 1 — depth-bounded computation & recomputation from context (the intro's distinctive thesis)

*The intro now argues persistence is **recomputation, not storage** (a fixed-depth transformer can't carry
a deep encoding forward), yet no related-work paragraph situates this. Add one.*

```latex
\paragraph{Bounded-depth computation and recomputation from context.}
A transformer applies a fixed number of layers per token, which bounds the serial computation available
before it must emit \citep{weiss2021thinking}. This has a direct consequence for goal maintenance: a
representation that only becomes available in the upper layers cannot be written once and reused many
tokens later, because a later token can read it only in its own remaining layers. Rather than a stored,
continuously updated state, the natural strategy is to \emph{recompute} the relevant quantity from
information that is always in context --- a view consistent with treating generated tokens (e.g.\
chain-of-thought or scratchpad traces) as externalized working memory the model re-reads
\citep[CITE: Nye et al.\ 2021 scratchpad; Wei et al.\ 2022 CoT]{}. We make this concrete for goal
pursuit: the next sub-goal is re-derived at each step from the task specification and the code written so
far, so the persistence we observe is the stability of that recomputation rather than a maintained
set-point (\Cref{sec:persistence,sec:diff}).
```

## Gap 2 — measuring goal-directedness & high-level interpretability (motivation)

*The intro opens on "do apparent goals correspond to an explicit internal representation" and frames the
read-out as an oversight primitive. Situate this against behavioral goal-directedness measurement and the
call for high-level interpretability.*

```latex
\paragraph{Measuring goal-directedness and high-level interpretability.}
Whether a system is goal-directed has been argued to be partly observer-relative, measurable only from
behavior \citep[CITE: MacDermott et al.\ 2024; Rajcic \& S\o gaard 2025]{}. Recent commentary calls for
\emph{high-level} interpretability of how models pursue goals --- built from interventions on sets of
components rather than single features --- as promising but neglected \citep{greenblatt2026huggingface,
greenblatt2024highlevel}. We take up this program representationally, \emph{inside the forward pass}:
instead of scoring goal-directedness from outputs, we localize a causal, decodable encoding of the next
sub-goal and test it against functional criteria. This complements behavioral measures with a mechanistic
correlate, and yields a pre-action read-out (\goalscope) relevant to oversight of agentic systems.
```

## Integration notes

- **Placement.** The existing section is currently in the Appendix (`main.tex` input commented). If it is
  promoted to the main text under a page limit, keep paragraphs 1, 3, 4, 6 (closest to the contributions),
  fold 2 and 5 into one sentence each, and add **Gap 1** (it carries the paper's distinctive thesis). Add
  **Gap 2** near the top since it is the motivation.
- **Ordering to match the intro.** Lead with Gap 2 (motivation: goal-directedness / high-level interp) →
  functional accounts (¶1) → portable representations (¶4) → reading/steering (¶5) → anticipation/planning
  (¶3) → Gap 1 (recomputation/depth) right before persistence/memory (¶2) → causal interventions (¶6) →
  generalization (¶7).
- **Bib fixes:** `\citep{feucht2025dual}` in the existing ¶4 has **no matching key** in `references.bib`
  (the dual-route / concept-induction-heads paper) — add the entry or fix the key. New keys to add for the
  gaps: scratchpad/CoT (Nye 2021, Wei 2022), goal-directedness measurement (MacDermott 2024, Rajcic &
  S\o gaard 2025). `weiss2021thinking`, `greenblatt2026huggingface`, `greenblatt2024highlevel` already
  exist in the bib.
- **Consistency:** the Gap-1 wording matches `analysis/diff_persistence_recomputation_revision.md` and the
  short conclusion (`discussion_draft.md` §1) — keep the "recomputation, not storage" phrasing uniform.
```
