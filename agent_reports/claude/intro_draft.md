# Introduction — draft (reframed: functional three-attribute framing)

*2026-09-20. A fresh Introduction draft on the **pre-commitment · plasticity · persistence** spine (per
`intro_reframing_plan.md`). Does NOT overwrite that plan, and does not follow the current Overleaf
`01_intro.tex`. Opens on the question "Do LLMs have goals?" (observer → functional definition →
mechanistic correlate). Persistence per the user's steer: temporal retention = a brief, real result;
obstacle-robustness = limited (the 32B model itself is only partly robust) → appendix, qualitative.
`\citep{...}` keys are placeholders to reconcile with the `.bib`. Terminology: **sub-goal** primary.*

---

\section{Introduction}
\label{sec:intro}

Do large language models have goals? When a model plans several tokens ahead or recovers after a misstep,
it is natural to say it is \emph{trying} to achieve something~\citep{greenblatt2026huggingface}. But such
descriptions are, by default, an \emph{observer's}: Dennett's \emph{intentional stance}, in which we
predict a system \emph{as if} it had beliefs and goals, with no commitment about its internal
machinery~\citep{dennett1987}. For AI this posture is slippery — goal-directedness has been argued to be
``in the eye of the beholder,'' admitting many equally predictive ascriptions~\citep{rajcic2025goaldirectedness} —
and because LLMs are trained to emit words that mimic humans, fluent goal-talk is no evidence of a goal.
The scientific question is whether an apparent intention corresponds to an actual computation: does the
model hold an explicit internal representation of what it intends to do next, or is it improvising token by
token? Answering it demands two things — a test that separates a genuine goal from a shadow of next-token
competence, and a definition of ``goal'' that does not smuggle in the mental states we are trying to test
for.

We give such a definition and test it, taking a \emph{functional} view. Following
\citet{bratman1987intention}'s planning theory, we treat an intention not as a subjective feeling but as a
functional state: a resource-bounded agent \emph{commits} to it before acting, it controls the conduct that
follows, and it is stable enough to guide behavior over time. This lets us define a \emph{sub-goal} by the
functional roles its internal correlate must play, and then look for that correlate. We specialize to the
\emph{next instrumental sub-goal} during multi-step code generation, where the target is concrete and
checkable: implementing merge sort, for instance, decomposes into splitting the array, sorting each half,
and merging the two halves; before writing a single token of the merge, a goal-directed model must already
represent \emph{that} sub-goal — abstractly enough that it could be realized by different loops, variable
names, or even a different programming language. Concretely, we ask whether the model's encoding of this
next sub-goal exhibits the signatures the planning and philosophy-of-biology literatures use to
characterize goal-directed behavior: \textbf{(i) pre-commitment} — the state is present and controls the
action \emph{before} the action is produced~\citep{bratman1987intention}; \textbf{(ii) plasticity} — the
\emph{same} goal is reached by different means; and \textbf{(iii) persistence} — the goal is maintained
over the course of its execution. Plasticity and persistence are the two classical hallmarks of
goal-directedness (\citealp{sommerhoff1950}; \citealp{nagel1979}), made quantitative by
\citet{lee2020operationalizing}: plasticity is \emph{equifinality}, reaching the goal along different
paths, and persistence is maintaining it despite the passage of time and intervening perturbations. Two
points make this the right lens for an LLM. First, an LLM is resource-bounded in a strong sense — a
fixed-depth forward pass per token — which is exactly the regime in which committing to a sub-goal
\emph{in advance} is computationally useful rather than re-deriving it at every step. Second, we make no
claim about subjective experience: we show that the intentional stance one naturally takes toward such a
model has an explicit \emph{mechanistic correlate}.

We study this correlate by dissecting \texttt{Qwen2.5-Coder-32B-Instruct} as it executes multi-step coding
tasks, with replications across model families and sizes (\Cref{app:model_axes}). Using causal mediation
analysis, we localize the next sub-goal to a compact set of attention heads, which we call the
\goalheads, and we find that this encoding satisfies the three functional criteria:

\paragraph{Pre-commitment — the next sub-goal is encoded, and readable, before it is written.}
We introduce \goalscope, a lightweight probe that turns the \goalheads' contribution to the residual stream
into a natural-language description of the upcoming operation. At the pivot where the model is about to
write a sub-goal — but has emitted none of its code — \goalscope already names it. Tracing the encoding
across depth with \textbf{OVReader}, we find the sub-goal is committed early (by the middle layers), before
most of the \goalheads act and before implementation begins.

\paragraph{Plasticity — the encoding is abstract and portable across realizations.}
Transplanting the \goalheads' encoding from one context into a fresh one causes the model to pursue the
\emph{same} sub-goal, realized through a \emph{different} sequence of tokens — including across programming
languages: an encoding captured from Python triggers the operation in C++, and vice versa. This
equifinality — one goal, many concrete paths — is the property that most cleanly separates a goal
representation from a prediction of \emph{these} output tokens, which would not survive transplantation
into a foreign context.

\paragraph{Persistence — the encoding is retained across execution.}
Once committed at the pivot, the sub-goal encoding is \emph{retained} across the tokens that implement it,
rather than recomputed line by line: its direction is sustained through the implementation span and the
same upcoming sub-goal is read out at successive pre-implementation positions (\Cref{sec:persistence}).
This temporal retention is a distinct kind of evidence from the faithfulness and causality measures above,
and we report it as a targeted check. A stronger form of persistence — remaining on the goal despite an
adversarial \emph{obstacle} placed in the way of the implementation — is only partially exhibited: the
model itself is not strongly robust to such perturbations at this scale, and we discuss this behavior with
qualitative examples in \Cref{app:obstacles} rather than as a headline claim.

\paragraph{The goal heads are a causal bottleneck.}
Beyond being readable, the \goalheads are load-bearing. Preserving only their access to the task
specification — while blinding every other head in the upper two-thirds of the network — largely preserves
the model's ability to carry out the pre-committed sub-goal, whereas blinding the \goalheads themselves
sharply impairs it; a size-matched random head set has no such effect. Redirecting the \goalheads'
key/value or query states changes \emph{which} sub-goal is implemented, and \goalscope tracks the change.

Read together, these results give concrete, testable evidence that a code LLM carries a mechanistic
analogue of ``about to do $X$'': a compact, decodable, and portable representation of the next sub-goal
that is committed before the action, invariant to how the action is realized, and retained across its
execution. We further find the same \goalheads verbalize inferred operations beyond enumerated code steps
and in non-code few-shot settings (\Cref{sec:generalization}), indicating the mechanism is not an artifact
of a single model or the coding domain.

\paragraph{Contributions.}
\begin{itemize}
  \item A \textbf{functional, testable definition} of an LLM sub-goal — pre-committed, plastic, and
    persistent — grounded in planning theory and the goal-directedness literature, and evidence that a
    code LLM's internal representation satisfies it.
  \item \textbf{Localization} of the next sub-goal to a compact set of \emph{goal heads} via causal
    mediation, and \goalscope, a probe that \textbf{verbalizes} the upcoming sub-goal from their
    residual-stream contribution.
  \item A \textbf{portability / transplantation} result establishing plasticity: the encoding triggers the
    same sub-goal in a new context and a new programming language, realized by different tokens.
  \item A \textbf{temporal-persistence} check (the encoding is retained, not recomputed, across execution)
    and a \textbf{causal-bottleneck} characterization of the goal heads (necessity and sufficiency of their
    task access), with replication across model families/sizes and read-outs in non-code settings.
\end{itemize}

---

## Notes for iteration (not part of the draft)
- **Opening** now leads with "Do LLMs have goals?" → observer problem (Dennett; Rajcic & Søgaard) →
  functional definition (Bratman) → mechanistic correlate, per `intro_reframing_plan.md` §1/§5. Merge-sort
  is now a concrete instantiation inside the functional-definition paragraph, not the opener.
- **Persistence section** (`\Cref{sec:persistence}`): brief, built from
  `notebooks/clock/14_goal_retention.ipynb` + `notebooks/persistance/01_temporal_persistance.ipynb`
  (affinity-retention curve + token-by-token readout); framed as a *retention* measure, distinct from
  faithfulness/causality.
- **Obstacle appendix** (`\Cref{app:obstacles}`): qualitative examples from
  `notebooks/persistance/03_obstacle_robustness.ipynb` / `obstacle_robustness_analysis.md` — 32B is only
  ~0.15 ADAPTED under hard obstacles (76% succumb): a limitation/nuance, not a claim.
- **Open calls:** keep Greenblatt's call-to-action here or move to Discussion; name the equifinality
  backbone explicitly vs. leave it folded; "pre-commitment" vs decision-theory "precommitment" (Strotz) —
  we mean Bratman's future-directed intention.
- Citation keys are placeholders: `bratman1987intention`, `dennett1987`, `rajcic2025goaldirectedness`,
  `sommerhoff1950`, `nagel1979`, `lee2020operationalizing`, `greenblatt2026huggingface`;
  plasticity/planning precedents (`todd2024function`, `dong2025emergent`, `lindsey2025biology`).
