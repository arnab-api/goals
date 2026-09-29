# Revision: DIFF theory ⟷ functional persistence through repeated recomputation

*2026-09-23. Proposed text revisions for the main draft (§1 intro bridge, §3 Temporal Persistence, §5
DIFF) to present a single position: **a fixed-depth transformer cannot hold a goal, so it re-derives it
every token by contrasting the fixed task (prompt) against its progress-so-far (generated code);
"persistence" is the stability of that recomputation, and DIFF is the recomputation.** This makes
persistence a corollary of DIFF rather than a separate phenomenon, and resolves the `running` vs
`repeat-computing` fork flagged in `06_mechanism.tex`.*

*These are drop-in replacements for existing prose. Macros/labels/citation keys match the current draft
(`\goalheads`, `\goalscope`, `\pivot`, `\M`, `sec:persistence`, `sec:diff`, `sec:heads:localize`,
`fig:temporal-retention`, `app:ovreader`, `app:persistence:temporal`). `\arnab{}` scaffolding notes are
dropped. I have **not** edited any `.tex`.*

---

## The position in one sentence

The LM recomputes the next sub-goal as `DIFF(G, progress)` at every token; because `G` is fixed and
`progress` advances one sub-goal at a time, the *same* frontier is re-derived across the whole
implementation span — which is exactly the temporal persistence we measure. The depth budget of a
fixed-depth transformer is *why* it must recompute rather than carry.

---

## 1. Intro — the persistence → recomputation bridge

**Where:** replace the paragraph at `Text/01_intro.tex:40` ("The encoding captured by \goalheads also
let's us check ...").

**Why:** name recomputation, introduce the depth intuition, and set persistence up as *stable
recomputation* so §3/§5 can pay it off.

```latex
The same encoding lets us ask whether the sub-goal is \emph{sustained} across the many tokens that
implement it (\textbf{persistence}\footnote{In this work we focus on \emph{temporal} persistence, the
retention of a goal through time. Literature also defines persistence as the state's robust maintenance
under planted/unexpected obstacles; the LMs at the scale we study do not robustly show such obstacle
persistence in their local continuations, which we discuss separately in \Cref{app:persistence:obstacle}.}).
In \Cref{sec:persistence} we find that it is: the frontier encoding stays close to its pivot value
throughout the implementation. But a depth-bounded transformer cannot literally \emph{hold} a state across
generation steps --- a representation formed in the upper layers at one token has too few layers left to
steer a token many steps later. We argue the LM instead \emph{re-derives} the frontier at every step, by
contrasting the fixed task specification (in the prompt) against the progress it has already externalized
(the code written so far). On this view persistence is not a carried set-point but the \emph{stability of a
repeated recomputation}. \Cref{sec:diff} makes this precise as the \emph{DIFF} theory --- the LM recomputes
the next sub-goal as the difference between the broader goal $G$ and its current progress --- proposes an
attention-read implementation, and tests its concrete claims with targeted causal interventions.
```

---

## 2. §3 — Temporal Persistence (reinterpret the curve as stable recomputation)

**Where:** replace the two prose paragraphs in `Text/persistance.tex` after the figure (from "An idealized
goal-pursuing agent ..." to the end).

**Why:** the retention curve currently reads as "a vector being carried." Reframe it as the goal heads
*re-deriving the same frontier each token*, and make the end-of-span decay a *mechanistic* release
(progress advancing), not only a memory-activation analogy.

```latex
An idealized goal-directed agent forms an intention to pursue $g_k$ and keeps it active until $g_k$ is
achieved \citep{bratman1987intention, cohen1990intention}. A depth-bounded transformer cannot maintain such
a state \emph{continuously}: what it computes at one token is not privately carried to the next. We
therefore ask a weaker, testable question --- does the LM \emph{functionally} maintain the frontier $g_k$
throughout its execution? The encoding captured by the \goalheads lets us check directly. For a frontier
$g_k$ we cache the goal encoding at the last pre-implementation pivot (\texttt{eq}) and let the LM greedily
generate $g_k$'s implementation; at every generated token we take the encoding in the goal heads' outputs
and measure its cosine similarity to the pivot encoding.

\Cref{fig:temporal-retention} plots this similarity as the LM realizes $g_k$ in tokens, against three
controls: the same sub-goal from a different context and programming language (ii), a different sub-goal in
the same context but appearing in a different order (iii), and a different sub-goal from a different context
(iv). The same-sub-goal curves (teal) start high near the pivot and remain above the different-sub-goal
curves (orange) throughout, decaying only gently. We read this not as a single vector being \emph{carried}
forward, but as the goal heads \emph{re-deriving the same frontier at each token}: while $g_k$ is being
written the task specification is fixed and the progress-so-far grows by only one token at a time, so the
recomputed encoding stays close to its pivot value. The gap to the different-sub-goal baselines shrinks
toward the end of the span --- as $g_k$ nears completion the model's progress begins to tick over to
$g_{k+1}$, and the recomputed frontier drifts away from $g_k$. This mirrors the view that a pending
intention stays highly active until it is executed and is then \emph{released} to avoid interfering with the
next one \citep{goschke1993representation, marsh1998activation, altmann2002memory} --- here realized
mechanistically, as a change in what the recomputation returns once progress advances. A matched control
head set, the full setup, and \goalscope readouts across the implementation of a few examples are in
\Cref{app:persistence:temporal}. \Cref{sec:diff} explains how this repeated recomputation is carried out
within a depth-bounded transformer.
```

---

## 3. §5 — retitled, gaze subsection removed, depth argument folded into the opening

**Structural changes to `Text/06_mechanism.tex`:**
- Retitle the section (adopting your header): `\section{Deciding the Next Sub-Step From Goal and Progress}`.
- **Delete the `\subsection{Gaze to relevant task descriptions}` (`sec:mechanism:gaze`)** — we don't have
  room to introduce *gaze* before it earns its motivation. Keep only a one-clause `\citep{gandikota2026gaze}`
  where it motivates the KV read (see 3b). Remove any `\Cref{sec:mechanism:gaze}` references.
- Fold the depth-budget argument directly into the section opening (below), replacing the old
  "What information do the \goalheads use ..." lead-in.

### 3a. Section opening (your draft, lightly edited: typos, precise attention mechanics, refs)

**Where:** the first paragraph of `sec:diff`, replacing the old opening and the gaze subsection.

```latex
\section{Deciding the Next Sub-Step From Goal and Progress}
\label{sec:diff}

In the preceding sections we identified a sub-goal encoding that can trigger the action and that remains
recognizable throughout its execution. A depth-bounded transformer, however, cannot literally \emph{carry}
the same sub-goal encoding forward. An encoding written to the residual state $h_t^{\ell}$ at layer $\ell$ of
token $t$ can influence a later token $t' > t$ only through attention, and a query at layer $\ell'$ attends to
keys and values built from the residual after layer $\ell'{-}1$; so token $t'$ can read the content at
$h_t^{\ell}$ only at its own layers $\ell' \ge \ell{+}1$. We observe that the \goalheads --- and hence the
sub-goal encoding --- are concentrated in layers 40--50 (\Cref{fig:localization}), leaving a later token only
a handful of layers to act on an inherited encoding, and a token deep into the implementation fewer still. A
single pre-committed encoding of $g_k$ therefore cannot guide the full multi-token implementation, which
often spans 30+ tokens (\Cref{app:persistence:temporal}). What \emph{is} available at every generation step
is the context tape (\Cref{fig:teaser}): the goal specification sits in the prompt and, once encoded by the
early layers, is readable by later layers at every position, while the progress-so-far can be read from the
growing tape. At each position the next \emph{sub-goal} $g_k$ --- the abstract next action whose pre-commitment,
plasticity, and persistence we established in \Cref{sec:what,sec:persistence} --- is then computed by
contrasting the re-derived broader goal $G$ with the progress so far.
```

*Edits vs. your draft:* fixed "the the"; made the attention statement precise (a query at layer $\ell'$ reads
$h^{\ell'-1}$, so content at $h_t^\ell$ is visible only at $\ell' \ge \ell{+}1$ — "only the next later
$\ell{+}1$" understates it, it's $\ell{+}1$ **and above**); `h^{l}`→`h^{\ell}`; kept your `\Cref{fig:localization}`
and 30+-token / `\Cref{app:persistence:temporal}` refs (add the impl-length analysis there per your TODO).
Double-check `fig:localization` is the head-density figure's label; if it currently lives in the appendix,
point to that label instead.

### 3b. DIFF-theory statement (`running` → *re-derived*, weld to persistence, one-clause gaze, OVReader)

**Where:** the DIFF-theory paragraph, after the opening above (replaces "We hypothesize a simple account ...
a {\color{red}running} progress encoding $E_\mathrm{prog}$ ...").

**Why:** commit to *recomputation* (not a carried accumulator), state the persistence corollary (the weld),
and motivate the attention-read implementation with a single in-passing gaze mention instead of a subsection.

```latex
We hypothesize the \emph{DIFF} theory: at each token the LM re-reads the broader goal $G$ and an encoding of
its progress $E_\mathrm{prog}$ --- the sub-goals it has already externalized in code --- and computes their
\emph{difference} to identify what remains undone, the next frontier $g_k$. Crucially, $E_\mathrm{prog}$ is
\emph{re-derived from the generated tokens at each step} rather than privately accumulated across steps; the
frontier is a per-token function of the current context. This makes the functional persistence of
\Cref{sec:persistence} a direct consequence: because $G$ is fixed and $E_\mathrm{prog}$ advances by only one
sub-goal at a time, the difference $G - E_\mathrm{prog}$ --- and hence the recomputed frontier --- stays
stable across the tokens that implement $g_k$, and changes only once a sub-goal is completed. Consistent with
per-token recomputation, the next sub-goal is already decodable from intermediate residual states by the
middle layers at \emph{each} token (OVReader, \Cref{app:ovreader}).

In our code-generation setting the sub-goals are specified in the prompt, and the \goalheads' attention
concentrates on the prompt tokens that describe the upcoming predicate (a \emph{gaze}-like pattern,
\citealp{gandikota2026gaze}), suggesting they read $G$ from the task span. A natural attention-read
implementation of the DIFF computation follows: the task-span keys and values carry the encoding of $G$,
while the query carries the (negated) progress $-E_\mathrm{prog}$, so that completed sub-goals are suppressed
and the unfinished one is selected. This makes two causally testable claims --- $G$ in the task-span
key/value states (\Cref{sec:diff:kv-transfer}) and progress in the pivot query state
(\Cref{sec:diff:q-transfer}).
```

*(This absorbs the old separate "In our code-generation setting ..." paragraph, now with the one-clause gaze
mention folded in — so the gaze subsection can be deleted entirely.)*

### 3c. Figure note (the schema TODO in §5)

Replace the FSM-only schema with a **per-token recomputation loop**: at token $t$, `read G (prompt)` +
`read progress (code so far)` → `DIFF` → `emit token` → advance one token, repeat. Keep the functional
"gonna-do → doing → done? → move-on" states from Fig 1 as the *behavioral* view, but annotate that each
"doing" box is **re-instantiated from context** every token (the teaser caption already says this) — so the
two figures are consistent: Fig 1 = functional/behavioral, the §5 schema = the per-token mechanism.

---

## 4. How the results line up under this framing (no text change, for the rebuttal/your reference)

| evidence | role under "persistence = stable recomputation / DIFF" |
|---|---|
| kv-transfer (99% causal / 96% readout) | `G` is read from the task span — the "what to read" |
| q-transfer (0.65 matched / 0.45 diff; causal rises with head budget in the sweep) | progress in the query — the "what to look for"; order-diff level = content/DIFF component, matched−diff ≈ ordinal component |
| temporal retention (§3, teal ≫ orange, gentle decay) | the *stability* of the recomputation while progress is unchanged |
| OVReader (committed by mid-layers, every token) | cleanest evidence that the frontier is *recomputed each token*, not carried |
| depth budget (goal heads L28–55 / 64) | *why* carrying is impossible → recomputation is forced |

---

## 5. Objections to preempt (keep these in mind when tightening)

- **"Re-reading the code is just carrying state by another name."** No — the state is *externalized* to the
  token stream and *re-contrasted* each step; that is why the fixed prompt `G`, not a hidden register,
  drives behavior (kv-transfer confirms `G` is read from the prompt).
- **Retention decays, it is not flat.** This is *predicted*: recomputation is stable only while progress is
  constant; the decay is progress advancing. A perfectly flat curve would argue *against* the release
  mechanism.
- **Don't overclaim a complete FSM.** §5's existing hedge ("underdeveloped, partial account") is compatible
  with the recomputation framing — q-transfer need only be non-zero and content-bearing (it is), not
  perfect.

---

## 6. Smoothing the sub-goal ↔ DIFF granularity transition

**The seam.** §3 studies the *sub-goal* $g_k$ (its pre-commitment / plasticity / persistence); §5's DIFF
talks about the *broader goal* $G$ minus progress. It reads like a level jump, but it isn't: both are at the
`G → g_k` level. DIFF(`G`, progress), evaluated at every token, is one operation with two faces — it
**selects** $g_k$ at the pivot and **keeps returning** $g_k$ while it is unfinished (the persistence of §3),
flipping to $g_{k+1}$ only when progress advances. Fix = state the hierarchy + your "study the abstract
level" rationale *once* (early), then present selection and persistence as the same recomputation.

### 6a. Hierarchy + granularity rationale (state it once)

**Where:** end of `\subsection{Sequential Implementation of Multiple Sub-Goals}` in `Text/02_notations.tex`
(the definitional home for $G$ and $g_k$); optionally echo one clause in the intro merge-sort paragraph.

**Why:** makes explicit *why* we probe the next action at the sub-goal granularity, so §5 does not feel like a
jump — it is the selection of the very object §3 characterizes.

```latex
A multi-step task is hierarchical: a broader goal $G$ decomposes into sub-goals $g_1,\ldots,g_K$, each of
which is realized by a sequence of concrete operations and, ultimately, tokens. At any position the model has
a \emph{next action}, but the granularity at which that next action is \emph{abstract} --- decoupled from the
immediate next token --- is what interests us. The next operation \emph{within} a sub-goal is typically
pinned to the next token or two, leaving little gap between intent and realization. The next \emph{sub-goal}
(the next action relative to $G$) instead spans many tokens and admits many realizations. We therefore study
the next action at the sub-goal granularity: the frontier $g_k$ is the abstract next-action whose
representation and plasticity (\Cref{sec:what}) and persistence (\Cref{sec:persistence}) we probe, and whose
selection from $G$ and progress we explain (\Cref{sec:diff}).
```

### 6b. Selection ↔ persistence, one recomputation (strengthen the weld in 3b)

**Where:** append to the DIFF-theory paragraph (3b), right after the persistence-corollary sentence.

```latex
The same difference, evaluated at every token, thus accounts for both \emph{selection} and \emph{persistence}
with one rule: it returns $g_k$ at the pivot, keeps returning $g_k$ while the sub-goal is unfinished
(\Cref{sec:persistence}), and flips to $g_{k+1}$ only once progress advances --- so tracking $g_k$'s encoding
through its implementation (\Cref{sec:persistence}) and deciding $g_k$ at the pivot are two views of the same
per-token computation, not two mechanisms.
```

### 6c. Alternative framings (if you don't want the explicit hierarchy paragraph)

- **Minimal bridge only.** Skip 6a; add a single sentence opening §5: *"Having shown the next sub-goal $g_k$
  is abstract, pre-committed, and persistent (\Cref{sec:what,sec:persistence}), we now ask how it is
  \emph{selected} — a question we necessarily pose one level up, at the broader goal $G$, since $g_k$ is the
  model's next action relative to $G$."* Cheapest, but doesn't justify the granularity choice.
- **Recursion as a feature.** Lean into the hierarchy: note the same DIFF rule could in principle operate at
  every level (`G`→sub-goal, sub-goal→operation), and we test it where the next-action is most abstract
  (`G`→$g_k$). One sentence in the discussion; risks over-claiming without evidence at the lower level.
- **Recommended:** 6a (hierarchy + rationale, once in §2) **+** 6b (unify selection/persistence in §5). This
  removes the seam without adding a speculative recursion claim.
