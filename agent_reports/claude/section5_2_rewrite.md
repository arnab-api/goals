# §5.2 (DIFF theory: kv-transfer & q-transfer) — readability rewrite

*Suggestion only — does not touch the Overleaf draft. Source: `Text/06_mechanism.tex` (compiles as §5,
"How the Upcoming Sub-Goal is Decided?"), Fig 3 `fig:kv-transfer`, Fig 4 `fig:q-transfer`.*

## 1. The setup, as I understand it (from the text + Fig 3/4)

**Shared notation.** A task-prompt is written `p: a̲bc̲ → b` — letters are sub-tasks, an underline marks a
sub-task **prefilled** (already written) in the prompt, and `→ x` marks the **frontier** (the next,
not-yet-written sub-task the model is about to implement at the pivot `π`).

**DIFF theory (the claim being tested).** While pursuing a broader goal `G`, the LM maintains a running
**progress encoding** `E_prog`. At a pivot it compares `E_prog` against `G` to decide what is still
undone and selects the next operation `g_k`. In a transformer this can be realized by an **attention
read**: the task description supplies `G` through its key/value states; the **query** surfaces the
undone part (e.g. by carrying `−E_prog`, suppressing what is done). This yields two testable, dissociable
predictions — and the two experiments target one each:

**§5.2.1 kv-transfer — "what the heads *read*" (Fig 3).** Swap the *content* the goal heads read.
- `p_src: a̲b̲c → c` and `p_dst: a̲b̲d → d` (shared prefix `a,b` prefilled; frontiers differ: `c` vs `d`).
- At the pivot, replace the **goal heads' key/value states over the task span** with the source's; non-goal
  heads keep the destination's (two KV caches; goal heads read the corrupted one — the GQA footnote).
- Prediction: DST now implements the **source** frontier `c` (and the readout verbalizes `c`).
- Result (250 heads): near-perfect **causal-shift ≈ 0.99** and **readout-shift ≈ 0.96**, even across
  languages.

**§5.2.2 q-transfer — "what the heads *look for*" (Fig 4).** Swap the *query* (the progress pointer).
- `p_dst: a̲bd → b` (only `a` prefilled → native next is `b`).
- Source carries progress `{a,b} done`: order-diff `p_src: a̲c̲b → c`, order-matched `p_src: a̲b̲c → c`.
- At the pivot, replace the goal heads' **query** with the source's. If the query carries progress, DST
  should skip its native `b` and jump to the operation undone *given `{a,b}`* — namely **`d`**.
- Result: causal-shift **≈ 0.50** (order-diff) / **≈ 0.68–0.76** (order-matched). [Draft currently shows
  placeholder 45% / 65% in red — real aggregated no-restrict numbers: order_diff 122·a4 0.539 / 250·a2
  0.500; order_matched 122·a4 0.758 / 250·a2 0.680.]

**Control (both experiments).** Repeating each intervention with an equal-size random head set drawn from
the later half of the model produces no shift — q-transfer causal-shift ≈ 0.06–0.11 (vs 0.50–0.76),
kv-transfer causal/readout = 0.00 (vs 0.99/0.96). The effects are goal-head-specific.

## 2. Issues in the current text

- **Notation bug (kv, line 23):** `p_src` is written `a̲b̲d → c`; it should be **`a̲b̲c → c`** (the very
  next sentence says the frontier `c` vs `d` is "the only difference"). The `d` is a typo.
- **Two intros do the same work.** The "A DIFF theory" prose (¶s at lines 12–16) and each subsubsection's
  first paragraph both re-explain `E_prog`/attention-read. Tighten the theory to one crisp paragraph and
  let each experiment open with its own contrast, not a re-derivation.
- **Order of presentation.** kv (read) then q (look-for) is the right order, but the transition is
  implicit. State the read-vs-look-for split *once* at the top of §5.2 so each experiment lands as "the
  other half."
- **Results buried in method.** kv results sit in a paragraph that also (re)describes the metric; the
  q-transfer results paragraph fuses number + control + a long limitations hedge. Split: setup →
  result → control, and move the "signs of life / underdeveloped" reflection to a short closing note.
- **Grammar / typos:** "the only different in the frontier" → "the only difference"; "interverntion";
  "a equivalent number" → "an equivalent number"; "the order of c are d are different" → "the order of
  `c` and `d` differ"; "sub-goal not in the source progress" reads awkwardly.
- **Placeholders:** `{\color{red}45\%/65\%}`, "Appendix X", "Figure X", and several `\arnab{...}` notes are
  still inline. Real numbers are available (above).
- **Metric definition repeated** in both subsubsections. Define *causal-shift* and *readout-shift* once
  (they're identical across both experiments).

## 3. Suggested rewrite

Drop-in prose (keeps `\goalheads`, `\goalscope`, `\Cref`, the notation, and the GQA footnote). Numbers
in `\todo{}` where the figures/tables should supply them.

---

### §5.2 A *DIFF* theory of computing the next sub-goal

Section~\ref{sec:mechanism:gaze} showed that the \goalheads look back to the task description, so some
mechanism selects *which* task to implement next. We propose a simple account, the **DIFF** theory. While
pursuing a broader goal $G$, the LM maintains a running **progress encoding** $E_\mathrm{prog}$, updated as
it writes. At a pivot it effectively computes the *difference* between $G$ and $E_\mathrm{prog}$ — what
remains undone — and advances to the next operation $g_k$.

In our setting the sub-goals are named in the prompt, so a transformer can realize this with an
**attention read**: the task description carries $G$ in its key/value states, and the **query** selects
the undone part through key–query affinity (one way to suppress the done parts is for the query to carry
$-E_\mathrm{prog}$). This gives two dissociable, causally testable predictions, which we probe with
complementary interventions on a single set of \goalheads: swapping *what the heads read* (the task-span
key/values, §\ref{sec:diff:kv-transfer}) versus *what they look for* (the query, §\ref{sec:diff:q-transfer}).

We write a task-prompt as $p:\mathrm{\underline{a}b\underline{c}\rightarrow b}$: letters are sub-tasks
(\Cref{tab:data-distribution}), an underline marks a sub-task prefilled in the prompt, and
$\rightarrow\!\mathrm{b}$ marks the frontier — the next sub-task, still to be written at the pivot $\pi_k$.
For every intervention we report two oracle-judged measures on the post-pivot rollout: **causal-shift**,
whether the implemented operation changes as predicted, and **readout-shift**, whether the \goalscope
verbalization changes with it. We use the 250-head set, the more causally effective one
(§\ref{sec:mechanism:ablation}).

### §5.2.1 Reading the goal encoding from the task span (kv-transfer)

\input{Text/Figures/kv_transfer}

If the \goalheads read $G$ from the task description, then changing *what they read* should change *what
the model does*. We take a destination prompt $p_\mathrm{dst}:\mathrm{\underline{a}\underline{b}d
\rightarrow d}$ and a source prompt $p_\mathrm{src}:\mathrm{\underline{a}\underline{b}c\rightarrow c}$ that
share the prefilled prefix $\mathrm{a},\mathrm{b}$ and differ only in the frontier ($\mathrm{d}$ vs
$\mathrm{c}$). At the pivot we replace the \goalheads' **key/value states over the task span** with the
source's; the non-goal heads keep reading the destination.\footnote{[GQA footnote unchanged.]} The patch
is applied once and carried forward by the KV cache. \Cref{fig:kv-transfer} illustrates the setup.

If the heads truly read the goal from the task span, this should make the model implement the *source*
frontier $\mathrm{c}$ instead of $\mathrm{d}$. It does: with the 250-head set the kv-transfer triggers the
source frontier at near-perfect rates (causal-shift \todo{0.99}), and the \goalscope readout flips to
$\mathrm{c}$ just as consistently (readout-shift \todo{0.96}) — the verbalization faithfully tracks the
intervention. The effect holds across all four language directions, producing a destination-language
realization of the source operation. Configurations and per-cell scores are in \Cref{app:task_transfer}.

### §5.2.2 The progress encoding dictates what to read (q-transfer)

\input{Text/Figures/q-transfer}

The kv-transfer changed the read; we now change the **query** — what the heads *look for*. Consider a
destination prompt $p_\mathrm{dst}:\mathrm{\underline{a}bd\rightarrow b}$ with only $\mathrm{a}$ prefilled,
so the model's native next step is $\mathrm{b}$. We replace the \goalheads' query at the pivot with the
query from a source run in which $\mathrm{a}$ *and* $\mathrm{b}$ are already done. If the query carries the
progress $\{\mathrm{a},\mathrm{b}\}$, transplanting it should make the destination skip $\mathrm{b}$ and
jump to the operation that is undone given that progress — namely $\mathrm{d}$.

We use two source specifications (\Cref{fig:q-transfer}): **order-matched**,
$p_\mathrm{src}:\mathrm{\underline{a}\underline{b}c\rightarrow c}$, where a shift to $\mathrm{d}$ could
reflect either transferred progress *or* copied task order (both point at the third slot); and
**order-diff**, $p_\mathrm{src}:\mathrm{\underline{a}c\underline{b}\rightarrow c}$, which places $\mathrm{c}$
and $\mathrm{d}$ at different positions across the two prompts, isolating progress from order.

The query transplant does shift the frontier to $\mathrm{d}$: causal-shift \todo{$\approx$0.50} in the
confound-free order-diff setup, and \todo{$\approx$0.68} in order-matched — the larger order-matched value
shows that task-specification *order* is also encoded in the query and influences selection. The shift is
weaker than for kv-transfer, but non-trivial and in the predicted direction, supporting a DIFF mechanism
realized through attention read. A complementary contrast (\Cref{app:...}) shows the next sub-goal is
already decided in the query by the time the \goalheads act, and is legible with \ovreader
(\Cref{fig:ovreader}).

**Control.** Repeating both interventions with an equal-size random head set drawn from the later half of
the model produces no measurable causal- or readout-shift (q-transfer causal-shift \todo{$\approx$0.06},
kv-transfer \todo{0.00}), confirming the effects are specific to the \goalheads.

**Scope.** The partial success of q-transfer suggests the DIFF theory captures part, but not all, of how
the model decides what remains to be done — order information and other parallel signals likely
contribute. Our interventions also fix a single target sub-task; how the model chooses the *first* among
several undone sub-goals is beyond what this setup can resolve. We read these as encouraging but
incomplete evidence for the theory.

---

## 4. Notes for whoever applies this
- Fix the `a̲b̲d → c` → `a̲b̲c → c` typo in the kv source spec.
- Fill the four `\todo{}` numbers from the figures/`app:task_transfer` (real values in §1 above).
- The metric definition now lives once in §5.2; delete the repeats in each subsubsection.
- Resolve the inline `\arnab{}` notes (oracle prompts → appendix; the readout-shift "should it be in main
  paper" question; the state-machine schema for the theory).
- Consider promoting the **control** to its own sentence in each figure caption too (both captions have
  `TODO: fill out with results (core + control)`).
