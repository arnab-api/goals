# Conclusion / Discussion — drafts

*2026-09-23 (updated 2026-09-24). Two versions:*
- ***Short conclusion** (~half page) for the main text `Text/07_discussion.tex` — the initial submission is
  page-limited, so this is the one that goes in the body.*
- ***Extended discussion** (~1 page) for the **end of the Appendix** — the full themed version.*

*LaTeX-ready, using the paper's macros (`\goalscope`, `\goalheads`) and labels (`sec:what`, `sec:heads`,
`sec:diff`, `sec:persistence`, `sec:noncode`, `app:*`). Numbers as they currently stand (readout ~90%+ at
`indent`, cross-language causal transfer ~86%, kv 99.2%/96.1%, q ~0.52 aggregated, OVReader peak ~L40, 122
sufficient / 250 necessary). Not applied to any `.tex`; citation keys are placeholders. The main-text
conclusion uses `\label{sec:conclusion}`; the appendix version uses `\label{app:discussion}` to avoid a
clash.*

---

## 1. Short conclusion — MAIN TEXT (~half page)

```latex
\section{Conclusion}
\label{sec:conclusion}

We asked whether a code LLM's apparent goal-directed behavior corresponds to an explicit internal
representation, operationalizing a \emph{goal} by three functional attributes: pre-commitment, plasticity,
and persistence \citep{bratman1987intention, lee2020operationalizing}. In \texttt{Qwen2.5-Coder-32B-Instruct}
we localized a representation of the next sub-goal to a sparse set of attention heads --- the \goalheads ---
and showed it satisfies all three: it is decodable \emph{before} the sub-goal is written (\Cref{sec:what}),
it transplants across contexts and programming languages to trigger the same operation through a different
token sequence (\Cref{sec:what}), and it is retained across the tokens that implement it
(\Cref{sec:persistence}); the same heads form a causal bottleneck on the model's own execution
(\Cref{sec:heads}). A simple \emph{DIFF} account --- the model re-reads the broader goal and its progress
and advances to what remains --- ties these together (\Cref{sec:diff}).

Our central claim is that this goal is sustained by \emph{recomputation, not storage}. A depth-bounded
transformer cannot carry an encoding that becomes readable only in its upper layers (our \goalheads
concentrate in layers 40--50) forward to steer many later tokens; instead the model re-derives the next
sub-goal at each step from information always present in context --- the specification in the prompt and
the code it has already written. Selection and persistence are then two faces of one per-token computation.
Two results separate a genuine goal from a shadow of next-token competence: cross-language \emph{plasticity}
(a Python-captured encoding triggers the operation in C++), and a \emph{causal bottleneck} that a matched
random-head control does not reproduce. Because the \goalheads expose intent \emph{before} action, \goalscope
gives a cheap, causal, pre-action read of what a model is about to do --- a promising primitive for
oversight of agentic systems. The account is deliberately partial (the progress/query signal is weaker than
the goal/key--value one, and behavioral robustness to adversarial obstacles is limited; see
\Cref{app:discussion}), and we release the \goalheads and \goalscope as a substrate for studying how models
represent, pursue, and hand off goals.
```

---

## 2. Extended discussion — APPENDIX (`app:discussion`, ~1 page)

```latex
\section{Extended Discussion}
\label{app:discussion}

We asked whether a code LLM's apparent goal-directed behavior corresponds to an explicit internal
representation, and operationalized a \emph{goal} by three functional attributes: pre-commitment,
plasticity, and persistence \citep{bratman1987intention, lee2020operationalizing}. In
\texttt{Qwen2.5-Coder-32B-Instruct} we localized such a representation of the next sub-goal to a sparse set
of attention heads --- the \goalheads --- and showed that it satisfies all three: it is decodable
\emph{before} the sub-goal is written (\Cref{sec:what}), it transplants across contexts and programming
languages to trigger the same operation through a different token sequence (\Cref{sec:what}), it is
retained across the tokens that implement it (\Cref{sec:persistence}), and the heads are a causal
bottleneck for the model's own execution (\Cref{sec:heads}). A simple \emph{DIFF} account --- the model
re-reads the broader goal and its progress and advances to what remains --- ties these together and is
partially supported by causal key/value and query transplants (\Cref{sec:diff}).

\paragraph{A goal sustained by recomputation, not storage.}
A depth-bounded transformer cannot literally \emph{hold} a goal it forms at one token and reuse it many
tokens later: an encoding that becomes readable only in the upper layers (our \goalheads concentrate in
layers 40--50) leaves a downstream token too few layers to act on it. The persistence we measure is
therefore better read not as a stored set-point but as a \emph{stable recomputation}: the model re-derives
the next sub-goal at every step from information that is always available in context --- the specification
in the prompt and the progress it has already externalized as code. This reframing unifies our results.
It is why the retention curve stays high while a sub-goal is written and decays as the model finishes it
(progress advances, the recomputed target flips); it is why OVReader can read the committed sub-goal from
mid-layer residual states at \emph{each} token (\Cref{app:ovreader}); and it is exactly what a DIFF over
(goal, progress) evaluated per token would produce. On this view, selection (\Cref{sec:diff}) and
persistence (\Cref{sec:persistence}) are two faces of one per-token computation rather than two mechanisms.

\paragraph{Separating a goal from next-token competence.}
A representation that merely predicts the model's own next tokens could be a shadow of competence rather
than a goal. Two results pull the two apart. First, \emph{plasticity}: an encoding captured in Python
triggers the same operation in a C++ continuation ($\sim$86\% of the time), realized in a different token
sequence --- an implementation-invariance a next-token predictor would not survive. Second, the
\emph{causal bottleneck}: preserving only the \goalheads' access to the task specification largely
preserves the model's ability to execute the next sub-goal, while blinding a broader set sharply impairs
it, and a matched later-layer random-head control does neither (\Cref{sec:heads}). The encoding we read
is thus \emph{used}, not epiphenomenal, and it is abstract over its realizations.

\paragraph{Relation to accounts of goal-directedness.}
Our operationalization follows the functional tradition --- Bratman's planning intention as a
conduct-controlling state a resource-bounded agent commits to before acting, and Dennett's intentional
stance as a predictive posture with no built-in mechanistic claim \citep{bratman1987intention, dennett1987}.
Where behavioral work argues goal-directedness is measured only from the outside, we exhibit a
\emph{mechanistic correlate} inside the forward pass, with the classical hallmarks of equifinality
(plasticity) and maintenance over time (persistence) \citep{lee2020operationalizing}. Methodologically we
build on portable-encoding findings --- function vectors \citep{todd2024function} and filtering predicates
\citep{sensharma2025filter} --- and on activation read-outs \citep{ghandeharioun2024patchscope}; the
release of a completed sub-goal to make room for the next echoes intention-activation dynamics from
cognitive psychology \citep{goschke1993representation, marsh1998activation, altmann2002memory}, and the
redundancy we see under ablation is consistent with reported backup pathways in LLMs
\citep{mcgrath2023hydra, wang2022interpretability}.

\paragraph{Reading intent before action.}
Because the \goalheads expose the next sub-goal \emph{before} it is written, \goalscope offers a handle on
\emph{what a model is about to do} as generation proceeds, in natural language and from the model's own
weights. That the same heads --- localized purely on code --- also verbalize implicit rules in in-context
learning and surface intent in chat settings (\Cref{sec:noncode}) suggests the read-out is not a coding
artifact but taps a more general goal-verbalization capacity. This is a promising primitive for oversight
of agentic systems: a cheap, causal, pre-action read of intent. We stress its current limits below; it is
a research tool, not yet a safety guarantee.

\paragraph{Limitations.}
Our principal setting uses \emph{explicit} task specifications with a known frontier, which makes the
sub-goal identifiable but does not establish how the model behaves when it must \emph{infer} the
decomposition; the free-form and non-code results (\Cref{sec:noncode}) are preliminary and depend on
choices of sub-goal granularity. Success is judged by an oracle for nontrivial semantic agreement,
including partial implementations, and should not be read as executable-program correctness. The DIFF
account is deliberately partial: the query (progress) transplant is far weaker than the key/value (goal)
transplant (\Cref{sec:diff}), it does not explain how the model chooses the \emph{first} among several
undone sub-goals, and the attention-read implementation is a high-level abstraction, not a verified
circuit. The behavioral form of persistence is limited --- models at this scale do not robustly work
around an adversarial obstacle placed at a sub-goal's start (\Cref{app:persistence:obstacle}).
Cross-model replications use each model's own localized heads, so differences confound checkpoint,
head-selection, and generation effects. Finally, verbalizability shows the content is \emph{readable} in
language; it does not establish that natural language is the representation's native format.

\paragraph{Future work.}
Several questions follow directly. How does the model select the first undone sub-goal, and what marks the
\emph{boundary} where one sub-goal is retired and the next begins (our exploratory boundary-signal
interventions, \Cref{app:boundary}, are a start)? How few heads suffice as the intervention scales --- our
head-budget sweeps for read-out and for the causal transfers begin to answer this. Can the read-out be run
\emph{throughout} an implementation to detect when an intended operation silently fails or drifts? And does
the recomputation picture --- a goal re-derived each step from an externalized, growing context --- extend
to long-horizon agentic settings, where ``progress'' is spread across tool calls and turns rather than
tokens of a single function? We see \goalscope and the \goalheads as a concrete substrate for these
studies of how models represent, pursue, and hand off goals.
```

---

## Notes for the author

- **Two versions, one job each:** the short **Conclusion** (§1) is the main-text body paragraph; the
  **Extended Discussion** (§2) goes at the end of the Appendix (`app:discussion`). The short version is a
  strict condensation — no claim in it that isn't in the extended one.
- **Label clash avoided:** main = `\label{sec:conclusion}`, appendix = `\label{app:discussion}`. If the
  main text already `\Cref{sec:conclusion}`s from elsewhere, it will resolve to the short version.
- **Numbers to reconcile before pasting:** cross-language causal transfer (`~86%`), readout at `indent`
  (`~90%+`), kv 99.2%/96.1%, q **~0.52 aggregated** (matched 0.65 / diff 0.39 → appendix), OVReader peak
  layer, ICL 36.2%. Re-check against final figures/tables.
- **`app:boundary`** is currently commented out in `201_Appendix.tex`; either re-enable it or change that
  sentence in §2 to "left for future work" (drop the `\Cref`).
- **`app:persistence:obstacle`** and **`sec:noncode`** are referenced — both exist.
- Keep §1/§3/§5 language aligned with the **recomputation-not-storage** thesis (see
  `analysis/1_3.4_5__diff_persistence_recomputation_revision.md`); the short conclusion leads with it.
```
