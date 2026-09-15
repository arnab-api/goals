# Subspace Probe v2 — causal-transfer objective, structured corpus, no soft prompt

*Design doc, 2026-09-15. Supersedes the training setup in `subspace_probe_design.md` /
`notebooks/11_goalscope_subspace.ipynb` / `src/goals/subspace_probe.py`. Companion evals:
`scripts/evaluation/eval_subspace_probe.py`, `eval_ov_reader.py`, `eval_goalscope.py`
(causality mode). Read the OVReader mechanism in `agent_reports/claude/analysis/ov_reader_probe.md`
first — this design reuses its read→steer→grade pipeline, swapping the OV transform for a learned W.*

---

## 0. What the original probe was (CONFIRMED)

The v1 subspace probe was trained on **oracle-written natural-language read-outs of specific code
segments**:

- **Data:** per-task generations in `GEN_DIR/{task}.json` + oracle-labeled block trees in
  `LABEL_DIR/{task}.json` (the "well-known-tasks" label set). Each labeled block gives an NL intent
  string (e.g. *"check if the string equals its reverse"*) attached to a code span.
- **Read-out target:** that NL label, teacher-forced (cross-entropy).
- **Mechanism:** learn per-layer `W_l ∈ R^{r×d}` (layers 35–45, r=256) **and** a shared **soft prompt**
  that replaces the user turn. Patch = subspace **projection-replace** into an NL "helpful assistant"
  destination: `r_patched = r_dest + WᵀW (h_src − r_dest)`, applied at all positions from the last
  prefix token on; then greedily verbalize and grade the NL description.
- **Result on the locked corpus** (`subspace_probe_eval.md`): faithfulness 0.320, causality 0.058 —
  weak, and causality was **off-distribution** (the soft prompt was trained only on the NL template,
  never the C++ transfer stub).

So v1 optimized **verbalization fidelity to an NL label**, on a **hand-labeled corpus**, with a **soft
prompt**. v2 changes all three.

---

## 1. Goals of v2

1. **Structured n-task corpus** instead of the hand-labeled well-known-tasks set — reuse the
   locked-prefix corpus machinery (`data/goalscope_eval.json` / `eval_ov_reader` corpus builder) and
   **random-sample more** prefix-to-frontier prompts for training.
2. **Optimize causal cross-language transfer**, not verbalization. The objective and metric become the
   **causality metric**: steer the read-out into an information-less C++ stub and require the C++
   continuation to implement the frontier goal `g_k`. This is the apples-to-apples target that v1's NL
   objective missed.
3. **Drop the soft prompt.** The destination user turn is a real (paraphrasable) instruction, not a
   learned embedding. The **only** trainable parameter is the subspace `W`.

---

## 2. What changes (v1 → v2)

| aspect | v1 (current) | v2 (this doc) |
|---|---|---|
| training corpus | oracle-labeled well-known-tasks blocks | structured n-task prefix-to-frontier (locked-corpus builder), random-sampled |
| supervision target | NL intent label (per block) | a **goal-informed teacher's** next-token distributions over its own C++ body continuation |
| destination prompt | NL "helpful assistant" + **soft prompt** replacing user turn | C++ causal stub (`GOALSCOPE_SYS` + paraphrased C++ instruction + `auto func(auto val){`), **no soft prompt** |
| injection | subspace **projection-replace** `r_dest + WᵀW(h_src−r_dest)` at layers 35–45 | subspace read-out `o = WᵀW·h`, injected via `head_additive_generate(strategy="additive_norm", alpha=0.1)` over a steer band |
| trainable params | `W_l` (per layer) **+ soft tokens** | **`W` only** |
| objective / metric | verbalization CE vs NL label | **KL distillation** — steered student matches the goal-informed teacher's per-token distribution; graded by the **causality** oracle |
| eval | faithfulness+causality (causality was OOD) | **causality metric**, in-distribution by construction |

---

## 3. The read-out mechanism

**Notation.** `d = d_model` (5120). Read layer `l`. Source (structured) prompt residual at the pivot
token = `h ∈ R^d` (after layer `l`). Learned subspace basis `W ∈ R^{r×d}` with orthonormal rows
(`W Wᵀ = I_r`), r a hyperparameter (start r=256).

**Steering vector** (the in-subspace component of the frontier residual):

$$
o \;=\; W^\top W\, h \;\in\; R^{d}.
$$

> **✅ LOCKED — form of `o` = (A) subspace projection.** `W ∈ R^{r×d}` with orthonormal rows,
> `o = WᵀW h` (d-dim, the projection of `h` onto span(W)). Low params (r·d), keeps the "subspace"
> semantics of v1, and `additive_norm` normalizes so only the projected **direction** matters.
> *(You wrote `o = W@h`; since `additive_norm` needs a d-dim direction, a bare `r×d` map can't be added —
> `WᵀW h` is the subspace reading of that.)*

**Injection (identical to GoalScope/OVReader).** Inject `o` into the C++ causal destination prompt over
a steer band (default the GoalScope band, layers 35–46) with

$$
h' \;=\; h_{\text{resid}} \;+\; \alpha\,\lVert h_{\text{resid}}\rVert\,\frac{o}{\lVert o\rVert},\qquad \alpha=0.1,
$$

applied at every position from the last prefix token onward (matching `on_pass="all"`). Because
`additive_norm` normalizes internally, only the **direction** of `o` matters — exactly why OVReader can
drop per-head magnitudes and the `1/rms` scalar. This makes v2 a **learned-`W` drop-in for OVReader's
fixed OV transform** `W_O W_V diag(g)`.

> **✅ LOCKED — read layer decoupled from steer band.** Read `h` at layer `l`, steer into a **fixed**
> band 35–46 (GoalScope convention), as OVReader does. **Start with a single `W` at the OVReader peak
> read layer L50**; optionally sweep `l` (train a `W_l` each) later to reproduce the layerwise emergence
> curve.

---

## 4. Training data

### 4a. Source side — structured n-task prompts (where `h` comes from)
Reuse the `eval_ov_reader` corpus builder (`D.QTASKS[datatype]`, prefill-to-frontier): sample a task of
`T` single-line predicates, prefill the first `j` as complete, end at the frontier's `{var} =`. The
frontier predicate is the **known goal `g_k`**. Read `h` at the pivot of the source prompt.

- **Pivots:** read at the strong pivots — `eq` (`c =`) and `first_impl` (these carry the goal cleanly;
  see `ov_reader_probe.md` §5.2). Optionally pool all 5, but `prev_end` is near-useless and would add
  noise. **Default: `{eq, first_impl}`.**
- **Random sampling for scale:** re-sample tasks / predicate subsets / `j` / condition-count `T` (and
  datatypes) to generate many more source instances than the fixed 253 (which is reserved for eval —
  see §7 on the split). Trivial sign/zero frontiers excluded (`nontrivial_frontiers`, as in the eval).

### 4b. Target side — the goal-informed **teacher** distribution (`caus(g_k)`) ✅ LOCKED (LM-generated)
Instead of discrete text targets, the supervision is a **soft target**: the next-token distribution of
the model itself when the goal `g_k` is stated **explicitly**. This is a **self-distillation** target —
"what would the model write here if you just told it the goal?" — and it is LM-generated (no oracle):

1. **Sample one causal stub** (augmented — func name, arg name, paraphrased system/user; §6) →
   `stub_aug`. System prompt instructs **code only, no comments**.
2. **Teacher prompt** = `stub_aug` **with the goal made explicit** (extend the user instruction with the
   `g_k` description, e.g. *"…implement a function that checks whether the input is a palindrome."*).
3. **Generate the teacher trajectory** `y = (y_1,…,y_T)` from the teacher prompt (greedy), stopping at
   **32 tokens, or `</code>` / end-of-body `}` / EOS** — whichever first. **Cache only the token
   sequence `y`** (tiny) together with its `(g_k, stub_aug)` — the per-step distributions are **not**
   cached; they're recomputed on the fly at train time (§5) so the KL is over the **full vocab**.

So a training example = `(source instance → h, stub_aug, teacher trajectory y)`. K>1 discrete targets are
unnecessary: the teacher's per-step distribution already encodes the plausible alternatives, and we align
against it exactly. (You review a sample of teacher generations before a full run.)

---

## 5. Objective & training loop — KL distillation ✅ LOCKED

**Setup.** For each training example: a source instance (→ `h` at read layer `l`), the **same** augmented
stub used for the teacher, and the cached teacher trajectory `y` (tokens only).

**Teacher forward — on the fly, no grad, full vocab:** teacher-force `[teacher_prompt (goal explicit), y]`
in **one no-grad pass** → full-vocab distributions `p_t = softmax(logits_t)`, `t = 1..T`. Recomputed each
step (frozen model, so `p_t` is stable); nothing cached beyond `y`. This buys **exact full-vocabulary KL**
at the cost of one extra 32B forward per example per step.

**Student forward — the only thing with grad (through `W`):**
1. Compute the read-out `o = WᵀW h` (§3).
2. Build the **student prompt** = `stub_aug` **without** the goal (bare `CAUSAL_INSTRUCTION`; same func/arg
   names + same system/user paraphrase as the teacher, so the token scaffolding matches). **No soft prompt.**
3. **Teacher-force the same trajectory `y`** under the `additive_norm` steering hook (inject `o` across
   the steer band 35–46 at every position from the prefill onward). One forward → student distributions
   `q_t = softmax(logits_t)` at `t = 1..T`.

**Loss (distillation KL, full vocab):**

$$
\mathcal{L} \;=\; \frac{1}{T}\sum_{t=1}^{T} \mathrm{KL}\!\big(p_t \,\Vert\, q_t\big) \;+\; \lambda_{\text{ortho}}\,\lVert W W^\top - I_r\rVert_F^2 .
$$

Forward KL (`teacher || student`, mode-covering) so the steered student must cover every token the
goal-informed model would place mass on. QR-reproject `W` to the Stiefel manifold after each step (v1's
`reorthogonalize`). No soft-token grads (removed).

**Why this is the right objective.** Teacher and student share the **same augmented C++ stub and the same
continuation `y`** at each position; the *only* difference is that the teacher **saw the goal in its
prompt** while the student **only receives it through the steered subspace read-out `o`**. Matching the
distributions therefore trains `W` to make the subspace read-out carry *exactly the information the
explicit goal statement would have supplied* — and the steering path is byte-for-byte the eval's
causality path, so train and eval are in-distribution (the failure that sank v1 causality). It also
avoids penalizing surface-form choices: aligning distributions grades *idea*, like the causality oracle.

**Concrete per-example recipe (both forwards share `y` and scaffolding):**
```
# ---- offline (once per example): cache only the teacher token trajectory ----
stub_aug ← sample_causal_stub(func∈…, arg∈…, sys/user paraphrase∈…)     # §6
teacher_prompt ← stub_aug + explicit(g_k)         # "…implement a function that {g_k}."
y ← generate(teacher_prompt, greedy, stop=32 | </code> | } | EOS)        # cache y (tokens) only
# ---- training step ----
{p_t} ← teacher_forward(teacher_prompt, y)         # on the fly, NO grad, FULL vocab
o ← Wᵀ W h                                         # h = source frontier residual @ L50  (grad via W)
{q_t} ← student_forward(stub_aug_no_goal, y, steer=additive_norm(o, α=0.1, band 35–46))
loss ← mean_t KL(p_t || q_t) + λ·‖WWᵀ−I‖²          # full-vocab KL
```

- **No distribution caching:** only the token trajectory `y` is cached; teacher distributions are
  recomputed each step (one extra no-grad 32B forward) → **exact full-vocab KL**, no truncation/renorm.
  *(Two 32B forwards per example per step — teacher no-grad + student grad — which you've accepted.)*
- **Batching:** batch over (source instance × stub augmentation). Each item carries its own `o` and its
  own `y`; hooks broadcast per-batch like `verbalize_batch`; pad `y` to the batch max and mask KL on pad.
  Teacher and student can share one batched forward call each.
- **Hyperparameters (start):** r=256, α=0.1, steer band 35–46, read layer L50, λ_ortho=0.1, greedy
  teacher, forward-KL (T=1), AdamW, ~5–10 epochs. Sweep read layer, r, and α after a working baseline.

> **Remaining small gaps to validate (defaults chosen):**
> - **#B teacher decoding** — greedy trajectory (deterministic, reproducible target). Default greedy;
>   alt: sample the trajectory for diversity (still distill full distributions).
> - **#C KL direction / temperature** — forward KL `p‖q`, T=1. Alt: reverse KL (mode-seeking) or T>1
>   softening. Default forward, T=1.
> - **#D stop condition** — 32 tokens ∨ `</code>` ∨ balanced `}` ∨ EOS. Confirm the body-end marker
>   (the stub is `auto func(auto val){`, so a depth-0 `}` closes the body).

---

## 6. Augmentation / generalization (destination side)
To keep `W` from latching onto surface scaffolding, randomize the **destination prompt** per example:

- **Function name:** `func` → {`check`, `f`, `g`, `predicate`, `test`}.
- **Argument name:** `val` → {`x`, `v`, `n`, `value`, `arg`}.
- **System / user paraphrase:** several rewrites of `GOALSCOPE_SYS` and the C++ instruction (keep the
  "write only the body, C++, no comments" intent).
- (Source side already varied by random sampling in §4a.)

The C++ stub itself (`auto func(auto val){`) already generalizes types via `auto` — matching your
sketch. Optionally also vary the return signature (`bool`/`auto`).

---

## 7. Evaluation

**Metric:** the **causality** metric, unchanged — `eval_subspace_probe.py --mode causality` (or a v2
variant), which reuses `eval_goalscope` grading: steer `o` into `CAUSAL_PREFIX`, greedy-decode, grade
with `hard_eval` (oracle `claude-sonnet-4.6`), `hard_yes` pooled over pivots; trivial-case sanitizer
(`is_trivial_causal_readout`) applies. Report the **locked 253-spec corpus** so numbers sit next to
OVReader / goal-head causality.

> **✅ LOCKED — held-out predicates.** Reserve a subset of predicates per datatype for **eval only**;
> train on random samples drawn from the remaining predicates. A predicate seen only at eval tests goal
> generalization (not spec memorization) — the stronger claim. Assert train/eval predicate sets are
> disjoint. *(The teacher distributions are computed per goal `g_k`, so this split also determines which
> goals ever appear as a teacher target.)*

**Baselines to beat / compare (same corpus, causality):**
- **OVReader** (learned-free OV transform) — the direct comparator; v2 asks "can a *learned* W beat the
  fixed OV read-out?"
- **goal-head GoalScope** `replace α=1.0` — the causal upper reference.
- **v1 subspace probe** (0.058) — the thing we're fixing.

**Success criterion:** v2 causality ≫ v1 (0.058) and approaching OVReader's causality peak
(~0.6–0.8 mid-stack, `ov_reader_probe.md` §5.1). Also worth the **layerwise curve** (train `W_l` per
read layer) to see whether a learned subspace pushes the emergence earlier than the fixed OV transform.

---

## 8. Implementation plan

- `src/goals/subspace_probe.py` — add a v2 path (keep v1):
  - `SubspaceProbe`: drop `soft_tokens` (gate `use_soft_tokens=False`); add `readout_vec(h)` = `WᵀW h`.
  - new `steered_kl_forward(...)` = teacher-forced **student** forward that injects `o` via the
    **`additive_norm` steering hook** into the C++ dest and returns student logits at the `y` positions
    (replaces `patched_lm_forward`'s projection-replace + soft prefix). Reuse `reorthogonalize`,
    `orthogonality_loss`.
- **teacher-target generator** (LM self-distillation): small script that, per held-out-train goal `g_k`,
  samples an augmented stub, generates the greedy trajectory `y` (stop 32 ∨ `</code>` ∨ `}` ∨ EOS), and
  caches **only `y` (tokens) + `(g_k, stub_aug)`** → `data/subspace_v2/teacher_targets/<...>.json`.
  Distributions are recomputed on the fly during training (full-vocab KL). **You review a sample before a
  full run.**
- corpus: reuse `eval_ov_reader` sampler; add `--random_sample N` + the **held-out predicate split**
  (§7); read source residuals at pivots `{eq, first_impl}`, layer L50.
- training script: `scripts/train_subspace_probe_v2.py` (analogous to `train_h1h2_mask.py`): KL loss (§5),
  AdamW, QR-reproject each step; checkpoints → `results/subspace_probe_v2/<model>/<slug>/probe.pt`.
- eval: extend `eval_subspace_probe.py` to load a v2 probe (no soft prompt; `o = WᵀW h`) and run
  `--mode causality` on the locked corpus (held-out predicates only).

**Status:** core design **LOCKED** — objective = KL self-distillation (§5), read-out = subspace
projection `o = WᵀW h` at L50 (§3), LM-generated teacher targets (§4b), held-out predicates (§7), no soft
prompt, teacher recomputed on the fly for **exact full-vocab KL**. Remaining small confirmations: **#B**
greedy teacher, **#C** forward KL / T=1, **#D** body-end stop marker — all defaulted in §5, flip any if
you disagree. Next
concrete step after your sign-off: build the teacher-target generator and show you a sample of `y`
completions.
