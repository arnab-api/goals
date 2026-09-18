# Abstract & Introduction — draft sections (ready to port to Overleaf)

*Draft, 2026-09-17. Strategy A (functional-goal, portability-forward); "sub-goal" terminology; abstract
does not name GoalScope. Defaults taken: title = candidate #1; persistence kept secondary; Greenblatt in
the intro only. LaTeX `\citep{...}` keys are placeholders — reconcile with the `.bib`.*

---

## Title

**LLMs Encode Abstract Sub-Goals Before Acting on Them**

*Understated alternative:* Abstract Sub-Goal Encodings in LLMs
*If a subtitle is wanted:* LLMs Encode Abstract Sub-Goals Before Acting on Them: Decodable, Portable
Representations of the Next Coding Step

---

## Abstract

When a language model writes code for a multi-step specification, does it commit to *what it will do next*
before it writes it? We take a functional view: following planning theories of intention, we treat a
*sub-goal* as an internal state a resource-bounded agent must commit to before it can act, and we ask
whether code LLMs contain such a state — one that is decodable, causally effective, and portable. Using
causal mediation analysis over multi-step coding tasks, we find that a small number of attention heads,
which we call \emph{goal heads}, encode a compact representation of the next instrumental sub-goal before
its implementation is emitted. This encoding is \emph{decodable}: a lightweight probe reads out, in natural
language, the upcoming sub-task from what these heads write to the residual stream. Crucially, it is
\emph{abstract and portable} — it can be extracted from one function and transplanted to trigger the
\emph{same} sub-goal in a different context and a different programming language, an
implementation-invariance that separates a genuine goal representation from a byproduct of competence. The
heads act as a causal bottleneck on execution: preserving their access to the task specification largely
preserves the model's ability to carry out the next sub-task, while restricting it sharply impairs
execution. We replicate the readout and transfer across model families and sizes, and examine the same
heads in non-code settings. Our results give concrete, testable evidence that LLMs encode abstract
sub-goals before acting on them.

---

## 1. Introduction

When a programmer begins to fill in \texttt{validate\_config(timeout, host, port)} and pauses at the first
blank line, they have usually already decided \emph{what to check first} — say, that the timeout is
positive — before typing a single character of \texttt{timeout > 0}. The commitment to a sub-goal precedes
the code that realizes it. Does a language model, generating the same function, likewise settle on its next
sub-goal before it writes the implementation?

It is common for human observers to analyze an AI in terms of its ``goals,'' noting that an AI can seem to
``try'' to take steps toward an apparent goal even when it is unable~\citep{greenblatt2026huggingface}. Yet
because LLMs are trained to emit words that mimic humans, we must be careful not to misattribute motivation
by anthropomorphizing their words. The scientific question is whether the \emph{apparent} intention
corresponds to the \emph{actual} computation: does the model contain an explicit representation of what it
intends to do next, or is it figuring out the code token by token? This is not settled by showing that some
internal state predicts the model's own next tokens — such a state could be a mere shadow of competence. A
convincing answer needs a test that pulls ``goal'' apart from ``prediction.''

This is a recognized gap. Recent commentary argues that \emph{high-level} interpretability of how LLMs
``make decisions in pursuit of goals'' — built from interventions such as activation patching and
fine-tuning subsets of components, and aimed at a higher level of abstraction than individual features — is
promising but neglected, and that existing work too often stops short of ``[doing] the
thing''~\citep{greenblatt2026huggingface}. We take up this challenge for a concrete, checkable case: the
next instrumental sub-goal during multi-step code generation.

Rather than argue about whether LLMs ``have goals,'' we give an operational definition and test it. Planning
theories of intention treat an intention not as a feeling but as a \emph{functional} state that a
resource-bounded agent commits to \emph{before} acting — a state that sets the agenda for what to do next,
filters the options considered, and is stable enough to guide action over time~\citep{bratman1987}. Adopting
this stance, we say a model represents a \emph{sub-goal} if the corresponding internal state is
(i)~\textbf{committed before the action}, (ii)~\textbf{decodable}, (iii)~\textbf{causally constraining} on
what the model does next, and (iv)~\textbf{portable} across contexts. Two points make this the right lens
for an LLM. First, LLMs are resource-bounded in a strong sense — a fixed-depth forward pass per token —
which is precisely the condition under which committing to a sub-goal \emph{before} acting is
computationally useful, rather than deliberating over every option at every step. Second, we make no claim
about subjective experience: we show that the intentional stance~\citep{dennett1987} one naturally takes
toward such a model has an explicit \emph{mechanistic correlate}.

Using causal mediation analysis on multi-step coding tasks, we localize the next sub-goal to a small set of
attention heads, which we call \emph{goal heads}. These heads encode a compact representation of the
upcoming sub-task \emph{before} its implementation is written (property~i). We introduce \goalscope, a
lightweight probe that reads what the goal heads write to the residual stream and \emph{decodes} it into a
natural-language description of the upcoming sub-task (property~ii). Ablating the heads' access to the task
specification shows they are a \emph{causal bottleneck}: preserving access largely preserves the model's
ability to carry out the next sub-task, while restricting it sharply impairs execution (property~iii).

The decisive result is \textbf{portability}. The sub-goal encoding is abstract: it can be extracted from one
function and transplanted into a different context — and even a different programming language — where it
triggers the \emph{same} sub-goal (property~iv). Portability is the property that most cleanly separates a
goal representation from a prediction of the output, because a prediction of \emph{these} tokens would not
survive transplantation into a foreign context; and it is the classical signature of goal-directed behavior
across cybernetics, biology, and cognitive science — equifinality, the realization of the same goal along
different paths. We find the same goal heads and the same decodable, transferable encoding across model
families and sizes, and we observe related read-outs beyond code, indicating that the mechanism is not an
artifact of a single model or the coding domain.

\paragraph{Contributions.}
\begin{itemize}
  \item A \textbf{functional, testable definition} of an LLM sub-goal — committed before action, decodable,
    causally constraining, and portable — and evidence that a code LLM satisfies it.
  \item \textbf{Localization} of the next instrumental sub-goal to a small set of \emph{goal heads} via
    causal mediation analysis.
  \item \goalscope, a \textbf{decoding probe} that verbalizes the upcoming sub-task from the goal heads'
    residual-stream writes.
  \item A \textbf{portability / transplantation} result: the sub-goal encoding is implementation- and
    language-invariant, triggering the same sub-goal when moved to a new context.
  \item A \textbf{causal-bottleneck} characterization of the heads (necessity and sufficiency of their task
    access), with \textbf{replication} across model families and sizes and read-outs in \textbf{non-code}
    settings.
\end{itemize}
