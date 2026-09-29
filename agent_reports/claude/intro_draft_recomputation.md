# Introduction — draft (recomputation reframe)

*2026-09-22. New Introduction draft on the user's steer: **pre-commitment in its traditional
"written-once, read-later" sense is not feasible in a fixed-depth transformer** — the model would run out
of layers. The goal is instead **recomputed from context at every generated token**; what looks like a
persistent goal is a *stable recomputation*. This is supported by the per-position GoalScope readout,
OVReader's mid-layer commitment at each token, and is exactly what the DIFF mechanism (\Cref{sec:diff},
goal = broader task − progress) predicts. Supersedes the "retained, not recomputed" persistence line in the
older `intro_draft.md` / `intro_reframing_plan.md`.*

*Keeps the three-attribute spine (pre-commitment · plasticity · persistence) but reinterprets each
mechanistically. Aligns with the current teaser (Fig 1): a state machine (gonna-do → doing → done? →
move-on) whose "doing" state is re-instantiated by re-reading the context tape at every token, with
decodability fading across the implementation (panel a). Citation keys are placeholders — reconcile with
`references.bib`. Section refs use the paper's existing labels: `sec:what`, `sec:heads`, `sec:mechanism`,
`sec:diff`, `sec:persistence`, `sec:generalization`, `app:model_axes`, `app:ovreader`.*

---

## Framing changes vs. the current `01_intro.tex`

1. **Persistence is no longer "a single encoding retained across tokens."** It is *apparent* persistence
   produced by **recomputation**: the same goal direction is re-derived from the prompt at each step and
   stays stable, so the model behaves *as if* one encoding persists. The temporal-retention result
   (\Cref{sec:persistence}) becomes evidence that the *recomputation is stable*, not that a vector is
   carried.
2. **New middle beat: the depth-budget argument.** A goal written into token *t*'s residual at layer ℓ can
   only be *used* by a later token *t+k* at layers ≥ ℓ; a deep goal leaves almost no layers to act on it,
   and the further ahead the token the less budget remains. A one-time deep commitment therefore cannot
   drive a whole multi-token implementation → the model must recompute from the (early, always-readable)
   task tokens in the prompt.
3. **Pre-commitment is retained as a claim but recast:** the *decision precedes the action* (the sub-goal
   is computed and decodable at the pivot, before any implementation token is written) — realized freshly
   each step rather than as a one-time write.
4. **Consistency with \Cref{sec:diff}:** recomputation *is* what a difference computation (task − progress)
   does — the frontier is re-selected every step, so one mechanism explains both selection and apparent
   persistence.

---

## `\section{Introduction}` (LaTeX-ready)

```latex
\section{Introduction}
\label{sec:intro}

\input{Text/Figures/teaser}

Do large language models have goals? When a model plans several tokens ahead, or recovers after a
misstep, it is natural to say it is \emph{trying} to achieve something~\citep{greenblatt2026huggingface}.
But such descriptions are, by default, an \emph{observer's} --- Dennett's \emph{intentional stance}, in
which we predict a system \emph{as if} it had beliefs and goals with no commitment about its internal
machinery~\citep{dennett1987}. For AI this posture is slippery: goal-directedness has been argued to be
``in the eye of the beholder,'' admitting many equally predictive ascriptions~\citep{rajcic2025goaldirectedness},
and because LLMs are trained to emit words that mimic humans, fluent goal-talk is no evidence of a goal.
The scientific question is whether an apparent intention has an actual \emph{computational} correlate:
does the model hold an explicit representation of what it intends to do next, or is it improvising token by
token? Answering it needs a definition of ``goal'' that does not smuggle in the mental states we are
testing for, and a test that separates a genuine goal from a shadow of next-token competence.

We take a \emph{functional} view. Following \citet{bratman1987intention}'s planning theory, we treat an
intention not as a subjective feeling but as a functional state that a resource-bounded agent
\emph{commits to before acting}, that controls the conduct which follows, and that is stable enough to
guide behavior over time. This lets us define a \emph{sub-goal} by the roles its internal correlate must
play, and then look for that correlate. We specialize to the \emph{next instrumental sub-goal} during
multi-step code generation, where the target is concrete and checkable. Implementing a multi-predicate
function, for instance, decomposes into a sequence of sub-tasks --- \emph{check the timeout}, then
\emph{validate the host}, then \emph{the port} --- and before writing a single token of one of them a
goal-directed model must already represent \emph{that} sub-goal, abstractly enough that it could be
realized by different expressions, variable names, or even a different programming language. We ask whether
the model's encoding of this next sub-goal shows the signatures the planning and philosophy-of-biology
literatures use to characterize goal-directedness: \textbf{(i)~pre-commitment} --- the state is present and
controls the action \emph{before} the action is produced~\citep{bratman1987intention}; \textbf{(ii)~plasticity}
--- the \emph{same} goal is reached by different means~\citep{lee2020operationalizing}; and
\textbf{(iii)~persistence} --- the goal is maintained over the course of its execution.

A naive reading of pre-commitment and persistence is that the model writes a goal into its residual stream
once and then \emph{holds} it, reading the same vector back over the many tokens that implement it. In a
fixed-depth transformer this is not how it can work. For an encoding placed in token $t$ at layer $\ell$ to
steer a later token $t{+}k$, that token must read it through attention --- but attention at layer $m$ only
exposes source states computed up to layer $m$, so $t{+}k$ can use $t$'s layer-$\ell$ state only within its
own layers $\ge \ell$. The upcoming sub-goal becomes decodable only in the upper layers (we find it
committed around the middle-to-late layers, \Cref{sec:mechanism}); a token that inherits such a deep
encoding has almost no depth left to act on it, and tokens further into the implementation have less still.
A single deep commitment therefore cannot drive a whole multi-token implementation --- the model would run
out of layers. What the model does instead is \textbf{recompute} the sub-goal at \emph{every} generated
token. The task specification lives in the prompt, at early positions that every later token can read at
every layer; so the model can cheaply re-derive ``what I am doing now'' from the specification and the code
written so far, again and again. On this view the apparent persistence of a goal is not a stored set-point
but a \emph{stable recomputation}: the same sub-goal direction is re-formed at each step, so the system
behaves \emph{as if} one encoding persisted (\Cref{fig:teaser}). This reframes all three attributes ---
pre-commitment becomes ``the sub-goal is decided before it is written, freshly at each step''; persistence
becomes ``the recomputed goal stays stable across the span'' --- and, as we show, it is exactly what a
\emph{difference} mechanism that re-contrasts the broader task against the progress so far would produce
(\Cref{sec:diff}).

We study this correlate by dissecting \texttt{Qwen2.5-Coder-32B-Instruct} as it executes multi-step coding
tasks, with replications across model families and sizes (\Cref{app:model_axes}). Using causal mediation
analysis we localize the next sub-goal to a compact set of attention heads, which we call the
\goalheads~\citep{todd2024function, sensharma2025filter}, and we find that their encoding satisfies the
three functional criteria under the recomputation view:

\paragraph{Pre-commitment --- the next sub-goal is decided, and readable, before it is written.}
We introduce \goalscope, a lightweight probe that turns the \goalheads' contribution to the residual
stream into a natural-language description of the upcoming operation~\citep{ghandeharioun2024patchscope}. At
the pivot where the model is about to write a sub-goal --- but has emitted none of its code --- \goalscope
already names it. Tracing the encoding across depth with \textbf{OVReader}, a weights-only read-out built
from the \goalheads' value and output projections, we find the sub-goal is committed by the middle layers,
\emph{before} most of the \goalheads act and before implementation begins (\Cref{app:ovreader}). Because
this holds at each pre-implementation position, the ``commitment'' is a decision that precedes the action
and is re-made at every step, not a one-time write.

\paragraph{Plasticity --- the encoding is abstract and portable across realizations.}
Transplanting the \goalheads' encoding from one context into a fresh one causes the model to pursue the
\emph{same} sub-goal, realized through a \emph{different} sequence of tokens --- including across
programming languages: an encoding captured from Python triggers the operation in C++, and vice versa
(\Cref{sec:what}). This equifinality --- one goal, many concrete paths --- is the property that most
cleanly separates a goal representation from a prediction of \emph{these} output tokens, which would not
survive transplantation into a foreign context.

\paragraph{Persistence --- a stable recomputation, not a carried vector.}
Across the tokens that implement a sub-goal, the recomputed goal direction stays close to the one formed at
the pivot and far from that of any other sub-goal, decaying only gently until the sub-goal is finished
(\Cref{sec:persistence}). The model thus \emph{behaves as if} a single encoding persisted, even though it
is re-derived at each step, and it \emph{retires} the encoding once the sub-goal is done to make room for
the next one~\citep{cohen1990intention}. A stronger, behavioral form of persistence --- staying on the
sub-goal when its implementation is forced to begin with a misleading expression --- is only partially
exhibited at this scale, and we report it as a supporting diagnostic rather than a headline claim.

Beyond reading the encoding, we ask whether the \goalheads are \emph{load-bearing}. Preserving only their
access to the task specification, while blinding every other head in the upper two-thirds of the network,
largely preserves the model's ability to carry out the next sub-goal; conversely, blinding a broader set of
\goalheads sharply impairs it, whereas a matched set of random later-layer heads has little effect
(\Cref{sec:heads}). The encoding we read is therefore used, not epiphenomenal. Finally, we give a simple
account of \emph{how} the next sub-goal is computed: as a \emph{difference} between the broader task
specification and a running encoding of the progress made so far, re-evaluated at each step, and we test the
concrete claims of this mechanism --- and its hypothesized implementation as an attention read --- with
targeted causal interventions (\Cref{sec:mechanism}). Recomputation is a natural consequence of such a
difference: because the frontier is re-selected every token, the same machinery that \emph{chooses} the
sub-goal also produces its apparent persistence.

We replicate the readout and transfer findings across model families and sizes, and explore whether the
same tools illuminate LLM goals in non-code settings (\Cref{sec:generalization}). Together, our results
give concrete, testable evidence that a code LLM carries a mechanistic analogue of \emph{going to do X}:
decodable before the action, portable across its realizations, and sustained --- by continual
recomputation --- across the action it guides; \emph{as if} the model had a goal.
```

---

## Downstream reframing this implies (not done here — flagged for the main text)

- **\Cref{sec:persistence} (Temporal Persistence):** currently says the encoding is "retained ... rather
  than recomputed line by line." Flip to: the recomputed direction is *stable* across the span (retention
  = stability of recomputation), consistent with \Cref{sec:diff}. The appendix already frames the affinity
  as "the LM re-encodes the goal at each token."
- **\Cref{sec:what} / abstract:** drop any wording implying a single carried encoding; say the sub-goal is
  *decoded before it is written* (decision precedes action), re-formed each step.
- **\Cref{sec:mechanism} / \Cref{sec:diff}:** make explicit that a difference computation entails per-token
  recomputation, and that this is *why* the goal appears persistent — one paragraph tying DIFF to the
  persistence result.
- **OVReader (\Cref{app:ovreader}):** already supports the reframe (commitment by ~layer 40 at the pivot,
  at each position). Add one sentence: mid-layer commitment at *each* token is the signature of
  recomputation, and the depth budget above the commitment layer is what forces it.
- **Teaser caption (Fig 1):** already states "(b) at every token, the LM re-encodes the goal G from
  context ... behaves as if a single encoding persists." Consistent; no change needed.
- **Abstract:** the closing line should shift from "encode ... before acting" (implying storage) to
  "decodable before the action and sustained by recomputation."
```
