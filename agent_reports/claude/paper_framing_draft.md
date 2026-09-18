# Paper framing draft — title, abstract, intro (Strategy A: functional-goal, portability-forward)

*Working doc, 2026-09-17. For the coding-sub-goals paper (current Overleaf title: "Intent Before
Action: Decodable and Portable Sub-Goal Representations in LLMs"). Decisions locked with the user:*
- *Framing = **(A) functional-goal, portability-forward**.*
- *Operational anchor = **Bratman's planning theory of intention**, used as inspiration, not as a
  metaphysical claim: a **sub-goal is an internal state a resource-bounded agent must commit to
  before it can act on it**. We measure that functional role; we do **not** claim subjective experience.*
- *Consistent terminology: **"sub-goal"** throughout (see §4).*
- *Abstract must **not** name GoalScope (refer to it as "a lightweight readout probe").*
- *Style modeled on lab papers in `Overleaf-draft/REF/papers/` (function vectors, filter heads,
  dual-route) — see §5.*

---

## 1. The framing in one paragraph

Human observers routinely describe AI systems as "trying" to reach goals, but because LLMs are trained
to imitate human text, apparent intent may be projection rather than a property of the computation. We
refuse to settle this by philosophy and settle it **operationally**. Borrowing from planning theories of
intention (Bratman, 1987), we treat a **sub-goal** as a *functional* state: the thing a resource-bounded
agent must **commit to before it can act** — a commitment that (i) is formed *before* the action, (ii) is
*decodable* from the computation, (iii) *causally constrains* what the agent does next, and (iv) is
*portable*: the same commitment can be lifted out of one situation and made to drive the same behavior in
another. We show a code LLM contains exactly such a state for its next instrumental sub-goal, localized to
a small set of attention heads, and that it satisfies all four properties — with **portability** (transplanting
the sub-goal into a new context and language to trigger the same behavior) as the decisive test that it is a
genuine goal representation rather than a byproduct of competence.

**Why Bratman licenses the strong word.** Bratman's core move is that intention is *not* reducible to
belief+desire and is *not* about phenomenology — it is a distinct **functional** state defined by its roles
for a **resource-bounded** agent: it poses the next problem (sets the agenda), filters options, and stays
stable enough to coordinate action across time. LLMs are resource-bounded in a sharp sense — a fixed-depth
forward pass per token — so the same computational pressure that makes commit-before-acting useful for
people is a principled reason to expect it in an LLM. This answers the "you're overclaiming / it's just a
billiard ball going to strike" objection directly: the billiard ball has no *decodable, transplantable*
internal state that commits it before it acts; our sub-goal does, and we measure it. (Cf. Dennett's
intentional stance — we show the stance has a **mechanistic correlate**.)

---

## 2. The functional operationalization (the paper's spine)

We define a sub-goal by four properties, each mapped to an experiment and to a classical diagnostic of
goal-directedness (equifinality / internal-reference / perturbation-robustness — the criteria that recur
across cybernetics, biology, and cognitive science; see `lit_review/ood_readings.md`).

| Functional property (our claim) | Experiment | Bratman role | Goal-directedness diagnostic |
|---|---|---|---|
| **Committed before action** (proactive / pre-decisional) | the next sub-goal is decodable at the pivot, *before* its implementation is written; goal-head **queries already carry** the computed next sub-goal | future-directed intention; sets the agenda, filters options | an internal *reference signal* held before the outcome |
| **Decodable** | a lightweight probe verbalizes the upcoming sub-task in natural language from what the goal heads write | (intention is available to further reasoning) | the reference signal is *readable*, not just present |
| **Portable / abstract** (implementation- & language-invariant) | **transplant** the encoding to trigger the *same* sub-goal in a different context and programming language | partial plan: the sub-goal is committed, the implementation is a placeholder filled in later | **equifinality** — same goal, many realizations |
| **Causally constraining** (a bottleneck on execution) | preserving the heads' access to the task preserves execution of the next sub-task; restricting it impairs execution | commitment *filters* and *drives* subsequent action | interventionist causation (Woodward): moving the state moves behavior |
| **Persistent / tracked** *(secondary)* | the sub-goal is tracked/updated across many generated tokens | stability + coordination over time | perturbation-robust maintenance of a set-point |

**Portability is the linchpin.** The transplant experiment is the strongest evidence that the sub-goal is a
*portable object* and not an epiphenomenon of the current context or a competence artifact: lifting "intend
to check the timeout" out of one function and dropping it into a neutral stub in a *different language* still
makes the model do the timeout check. This is the LLM analog of the classic goal-directedness tests —
equifinality (same outcome via different paths) and an internal reference that survives a change of substrate.
It is also the property that most cleanly separates *goal* from *prediction*: a mere prediction of the output
would not transplant.

---

## 3. Title options

Recommended direction (per the user): promote the headline claim, keep "sub-goal", don't name the probe.

1. **LLMs Encode Abstract Sub-Goals Before Acting on Them** ← user candidate; strongest as a plain,
   falsifiable claim. The abstract's last sentence can echo it verbatim (lab habit — cf. function vectors).
2. **Abstract Sub-Goal Encodings in LLMs** ← user candidate; noun-phrase, understated, matches the
   dual-route/function-vector cadence.
3. **Committing Before Acting: Portable Sub-Goal Representations in Code LLMs** ← foregrounds the Bratman
   "commit-before-act" operationalization + portability.
4. **Portable Sub-Goals: Decoding and Transplanting an LLM's Next Coding Step** ← portability-forward.

**My pick: #1** as the title (bold, testable, matches the abstract's closing line), optionally with a
portability-flavored subtitle if a subtitle is wanted (e.g. "*LLMs Encode Abstract Sub-Goals Before Acting
on Them: Decodable, Portable Representations of the Next Coding Step*"). #2 is the safe understated
alternative.

---

## 4. Terminology (locked)

- **"sub-goal"** — the primary term throughout (the abstract instrumental step the model is about to
  implement, distinct from the concrete implementation). Use consistently in technical sections.
- **"goal heads"** — the localized attention heads (keep; parallels "filter heads", "function vector").
- **"commit / commitment"** — for the *before-action* property (Bratman); use in framing.
- **"intention"** — only in the framing/motivation, always in the explicit *functional* sense, with a
  one-line Bratman footnote disclaiming phenomenology. Never load-bearing in results.
- **avoid "prediction"** — it concedes the competence confound the portability result exists to rule out;
  a prediction of the output is exactly what a sub-goal representation must be shown to be *more than*.
- **avoid unqualified "goal"** in technical sections; fine in the title/motivation for rhetorical force.

---

## 5. Draft abstract (lab style; "sub-goal"; no probe name)

> When a language model writes code for a multi-step specification, does it commit to *what it will do
> next* before it writes it? We take a functional view: following planning theories of intention, we treat a
> **sub-goal** as an internal state a resource-bounded agent must commit to *before* it can act, and ask
> whether code LLMs contain such a state — and whether it is decodable, causally effective, and portable.
> Using causal mediation analysis over multi-step coding tasks, we find that a small number of attention
> heads, which we call **goal heads**, encode a compact representation of the next instrumental sub-goal
> before its implementation is emitted. We show this encoding is **decodable**: a lightweight probe reads
> out, in natural language, the upcoming sub-task from what these heads write to the residual stream.
> Crucially, the encoding is **abstract and portable** — it can be extracted from one function and
> transplanted to trigger the *same* sub-goal in a different context and a different programming language,
> an implementation-invariance that distinguishes a genuine goal representation from a byproduct of
> competence. The heads act as a **causal bottleneck** on execution: preserving their access to the task
> specification preserves the model's ability to carry out the next sub-task, while restricting it sharply
> impairs execution. We replicate the readout and transfer across model families and sizes, and examine the
> same heads in non-code settings. Our results give concrete, testable evidence that **LLMs encode abstract
> sub-goals before acting on them.**

*(≈200 words. Mirrors the filter-heads/function-vector arc: claim → causal mediation → named heads →
"decodable" → "portable: extracted and reapplied across context/language" → causal-bottleneck nuance →
replication → closing significance line that echoes the title.)*

---

## 6. Draft introduction (paragraph plan + prose)

**Opening (concrete scenario — lab habit).**
> When a programmer starts to fill in `validate_config(timeout, host, port)` and pauses at the first blank
> line, they have usually already decided *what to check first* — say, that the timeout is positive — even
> before typing a single character of `timeout > 0`. The commitment to the sub-goal precedes its
> implementation. Does a language model, generating the same function, likewise settle on its next sub-goal
> before it writes the code that realizes it?

**Beat 2 — the skeptic / why this is non-trivial.**
It is common to describe an AI as "trying" to take steps toward an apparent goal, even when it is
unable~\citep{greenblatt2026huggingface}. But LLMs are trained to emit words that mimic humans, so we must
not misattribute motivation by anthropomorphizing their words. The scientific question is whether the
*apparent* intention corresponds to the *actual computation*: **does the model contain an explicit
representation of the sub-goal, or is it figuring out the code token by token?** A representation that merely
predicts the model's own next tokens would not settle this — it could be a shadow of competence. We need a
test that pulls "goal" apart from "prediction."

**Beat 3 — the field explicitly asks for this (Greenblatt call-to-action).**
This is a recognized gap. Greenblatt~\citep{greenblatt2026huggingface} argues that *high-level*
interpretability of "how LLMs make decisions in pursuit of goals" — built from interventions like
activation patching and fine-tuning subsets of components, aimed at a higher level of abstraction — is
promising but neglected, and that existing work too often "[doesn't] do the thing." We do the thing for a
concrete, checkable case: the next instrumental sub-goal during multi-step code generation.

**Beat 4 — our operationalization (functional, Bratman-inspired).**
Rather than argue about whether LLMs "have goals," we give an operational definition and test it.
Following planning theories of intention~\citep{bratman1987}, which treat an intention not as a feeling but
as a *functional* state a resource-bounded agent commits to before acting, we say a model represents a
sub-goal if the state is **(i) committed before the action, (ii) decodable, (iii) causally constraining, and
(iv) portable across contexts.** LLMs are resource-bounded in a strong sense — a fixed-depth forward pass
per token — which is precisely the condition under which commit-before-acting is computationally useful.
We make no claim about subjective experience; we show the intentional stance~\citep{dennett1987} has a
mechanistic correlate.

**Beat 5 — what we find (roadmap = the four properties + portability crux).**
Using causal mediation analysis we localize the sub-goal to a small set of **goal heads** that encode it
*before* implementation (property i); a lightweight probe **decodes** it to natural language (ii); ablations
show the heads are a **causal bottleneck** on executing the next sub-task (iii); and — the decisive result —
the encoding is **portable**: extracted from one function and transplanted, it triggers the *same* sub-goal
in a different context and programming language (iv). Portability is the property that most cleanly separates
a goal representation from a prediction of the output, and it is the classical signature of goal-directedness
(equifinality: the same goal realized along different paths). We replicate across model families and sizes and
probe the same heads beyond code.

**Beat 6 — contributions (bulleted).**
- A **functional, testable definition** of an LLM sub-goal (commit-before-act; decodable; causal; portable),
  and evidence a code LLM satisfies it.
- **Localization** of the next sub-goal to a small set of goal heads via causal mediation.
- A **decoding probe** that verbalizes the upcoming sub-task from the heads' residual-stream writes.
- **Portability / transplantation** across implementation and programming language — the implementation-
  invariance test.
- A **causal-bottleneck** characterization (necessity/sufficiency of head access), plus **replication**
  across families/sizes and a look at **non-code** settings.

---

## 7. Open choices to confirm before I port this to Overleaf
- **Title:** #1 ("LLMs Encode Abstract Sub-Goals Before Acting on Them") vs #2 (understated) vs a
  title+subtitle. (Default: #1.)
- **"intent" in the current title/section names:** switch to "sub-goal"/"commit" per §4? (Default: yes,
  for consistency.)
- **Persistence:** promote it to a first-class fourth property in the abstract, or keep it secondary /
  discussion-only? (Default: secondary — the abstract already runs long; the h1/h2 tracking result can
  live in the body.)
- **Greenblatt in the abstract?** Currently only in the intro. (Default: intro only.)
