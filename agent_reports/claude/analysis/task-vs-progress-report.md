# Task vs progress: which span does the pivot read the goal from? (Qwen2.5-Coder-32B-Instruct)

*2026-10-26 · code `src/diff/task_vs_progress.py`, `scripts/diff/task_vs_progress_eval.py` · notebook
`notebooks/precommitment/task_vs_progress.ipynb` · job lists `job_list/diff_eval/task_vs_progress_32b{,_v2}.json` · results
(per-instance cache, shared with the notebook) `results/notebook_caches/task_vs_progress/Qwen2.5-Coder-32B-Instruct/<dt>__python-python__seed1123__mx64__ro0__task+gap/`
+ `<that dir>.summary.json`*

## TL;DR

The KV-transfer results argue that the model **re-reads the goal from the task span at the pivot**, rather than carrying a
goal marker forward in the code progress. This experiment fills the 2 × 2 grid directly: at the eq pivot ` =`, does the
destination read the **task span** or the **progress** from the source?

* **The task span carries the goal.** Task span from SRC, progress left as DST: **0.77** causal-shift at V×1 through the
  goal-250 heads, **0.93 / 0.97** at V×2 / V×4. Through all heads: 0.89 / 0.98 / 0.98.
* **The progress carries almost none of it.** Progress from SRC, task span left as DST: **0.00** through the goal heads at V×1,
  0.06 through all heads. Boosting the values helps somewhat but stays well short of the task span: **0.26** (goal-250, V×4) and
  **0.38** (all heads, V×4). The ~50 % a V×4 boost might have given is not reached. Even at V×4 most patched runs stay on d.
* **Both spans together** saturate: 0.98–0.99 in every scheme.
* **Residual transfer gives the same numbers as KV-transfer on all heads at V×1:** identical generations on all 384 (span, instance)
  pairs. The 0.89 (task only) holds with DST's progress latents pinned exactly to the clean DST run at every layer.

## Setup

Multi-predicate instances, the path-patching sample (`PP.sample_multi_instances`: k = 3, letters naming, `var_eq` pivot,
listed presentation, seed 1123), **Python → Python**, 32 per datatype × {int, str, list[int], list[str]} = **128 per cell**.

```
p_src: a'b'c -> c     source (k/v or residual donor)
p_dst: a'b'd -> d     the patched run (prompt ends at the eq pivot " =")
```

**Spans** (`TP.geometry`):

| span | tokens | SRC ↔ DST alignment |
|---|---|---|
| **task** | the task list `(1) … (3)` **plus its closing `\n`**, up to the user turn's `<|im_end|>` (excluded); `PP.build_geometry(span="list+gap")` | shorter list `\n`-padded at the front to equal length; every token pairs 1:1 |
| **progress** | `<|im_end|> … {var}` (pivot excluded) | token-identical in SRC and DST (asserted), aligned from the end |
| **pivot** ` =` | never overwritten | computes freely under the intervention |

The grid's four corners are none (= the clean DST run) / task / progress / task + progress.

**Interventions** (`TP.make_stack`):

| scheme | what is transferred |
|---|---|
| **kv · goal-250** | I3 only (pivot-onwards KV transfer): the 250 goal heads read SRC's k/v on the chosen span(s) from the pivot on (pivot and every generated token); values × V, keys / attention pattern unchanged. Every pre-pivot latent stays clean DST. |
| **kv · all heads** | the same, through all 64 × 40 heads |
| **residual** | the span's input embeddings and every decoder layer's output ← SRC's; every other position from the span start to pivot−1 is **pinned to clean DST** (so "task, no progress" keeps DST's progress latents exactly). No value boost. |

V ∈ {1, 2, 4} for both kv schemes. With all heads at V×1, kv is mathematically the same intervention as residual. Residual
also swaps the span's embeddings, because layer 0 reads k/v from the embeddings rather than a layer output.

**Eval.** Greedy rollout, 64 tokens. Oracle `claude-sonnet-4-6` (`PP.judge` → `judge_causal`, first computation only) labels
the patched rollout c / d / BOTH / NEITHER. *Causal-shift* = the share labelled **c**; **BOTH is a miss**. No GoalScope readouts in
this grid (causal only).

**Sanity checks** (notebook cells, every scheme):
- **kv:** every pre-pivot latent is bit-identical to clean DST.
- **residual:** the patched span equals SRC's residuals at every layer, and every non-patched pre-pivot span equals clean DST.
- **Smoke test of the batch driver:** it reproduced the notebook's cached generations exactly on all 14 configs that existed
  before the V×2 and task+progress V×4 additions.

## Results

### The 2 × 2 grid, per intervention (causal-shift, pooled, n = 128 per cell)

Each grid fixes the scheme and V, and varies which span the pivot reads from SRC. Rows give the task span's source, columns
the progress span's source. The top-left corner (both from DST) is the clean DST run, which implements d in 128 / 128.

#### Intervention: KV transfer (pivot-onwards) · 250 goal heads · V-boost ×1

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.00 |
| **task from SRC** | 0.77 | 0.98 |

---

#### Intervention: KV transfer (pivot-onwards) · 250 goal heads · V-boost ×2

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.04 |
| **task from SRC** | 0.93 | 0.98 |

---

#### Intervention: KV transfer (pivot-onwards) · 250 goal heads · V-boost ×4

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.26 |
| **task from SRC** | 0.97 | 0.98 |

---

#### Intervention: KV transfer (pivot-onwards) · all 2,560 heads · V-boost ×1

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.06 |
| **task from SRC** | 0.89 | 0.98 |

---

#### Intervention: KV transfer (pivot-onwards) · all 2,560 heads · V-boost ×2

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.16 |
| **task from SRC** | 0.98 | 0.98 |

---

#### Intervention: KV transfer (pivot-onwards) · all 2,560 heads · V-boost ×4

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.38 |
| **task from SRC** | 0.98 | 0.99 |

---

#### Intervention: residual transfer · every layer + input embeddings · no V-boost

*The span not taken from SRC is pinned to clean DST at every layer.*

| | progress from DST | progress from SRC |
|---|---|---|
| **task from DST** | 0.00 | 0.06 |
| **task from SRC** | 0.89 | 0.98 |

---

### By datatype (n = 32 per cell)

*Task only*

| scheme | V | int | str | list[int] | list[str] |
|---|---|---|---|---|---|
| kv · goal-250 | 1 | 0.75 | 0.78 | 0.78 | 0.78 |
| kv · goal-250 | 2 | 0.94 | 0.88 | 0.94 | 0.97 |
| kv · goal-250 | 4 | 0.97 | 0.94 | 0.97 | 1.00 |
| kv · all heads | 1 | 0.78 | 0.91 | 0.94 | 0.94 |
| kv · all heads | 2 | 0.97 | 0.97 | 0.97 | 1.00 |
| kv · all heads | 4 | 0.97 | 1.00 | 0.97 | 0.97 |
| residual | 1 | 0.78 | 0.91 | 0.94 | 0.94 |

*Progress only*

| scheme | V | int | str | list[int] | list[str] |
|---|---|---|---|---|---|
| kv · goal-250 | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| kv · goal-250 | 2 | 0.03 | 0.00 | 0.09 | 0.03 |
| kv · goal-250 | 4 | 0.34 | 0.25 | 0.28 | 0.16 |
| kv · all heads | 1 | 0.06 | 0.00 | 0.16 | 0.03 |
| kv · all heads | 2 | 0.19 | 0.12 | 0.22 | 0.12 |
| kv · all heads | 4 | 0.44 | 0.38 | 0.38 | 0.31 |
| residual | 1 | 0.06 | 0.00 | 0.16 | 0.03 |

*Task + progress*

| scheme | V | int | str | list[int] | list[str] |
|---|---|---|---|---|---|
| kv · goal-250 | 1 / 2 / 4 | 0.97 / 0.97 / 0.97 | 0.97 / 0.97 / 1.00 | 0.97 / 0.97 / 0.97 | 1.00 / 1.00 / 1.00 |
| kv · all heads | 1 / 2 / 4 | 0.97 / 0.97 / 1.00 | 1.00 / 1.00 / 1.00 | 0.97 / 0.97 / 0.97 | 1.00 / 1.00 / 1.00 |
| residual | 1 | 0.97 | 1.00 | 0.97 | 1.00 |

### Verdict mix (pooled, 128)

| config | c | d | BOTH | NEITHER |
|---|---|---|---|---|
| kv · goal-250 · task · V×1 | 99 | 22 | 0 | 7 |
| kv · goal-250 · task · V×4 | 124 | 1 | 1 | 2 |
| kv · goal-250 · progress · V×4 | 33 | 78 | 7 | 10 |
| kv · all · progress · V×4 | 48 | 69 | 3 | 8 |
| kv · all · task + progress · V×4 | 127 | 0 | 0 | 1 |

### Diagnostics

- **Residual = kv on all heads at V×1:** verdicts agree on 128 / 128 instances for every span, and the generations are identical
  token for token (384 / 384). This confirms the two implementations do the same thing.
- **Task-only failures shrink monotonically with V:** goal-250 fails on 29 instances at V×1 and on 4 at V×4. Those 4 are a
  subset of the 29.
- **Progress-only successes are mostly the easy cases:** the goal-250 V×4 progress-only hits (33) overlap the all-heads ones
  (48) on 30 instances. 23 of the 33 also succeed with the task span alone at V×1.

## Reading

1. **The goal is read from the task span at the pivot.** With DST's progress intact and pinned (residual) or simply left
   untouched (kv), the task span alone moves the frontier in 0.77–0.89 of cases at V×1 and ~0.97 with a modest boost. The
   progress span alone moves it essentially never at V×1 (0.00 / 0.06). This is the "re-read, don't carry" picture: a goal
   marker built up in the code progress and carried forward would show up as a progress-only effect, and there isn't one at
   natural scale.
2. **The progress carries a weak, boostable trace, not the goal.** Up-weighting the progress values ×4 pushes progress-only to
   0.26 (goal heads) / 0.38 (all heads), against 0.97 / 0.98 for the task span at the same boost. Most patched runs still
   implement d (78 / 69 of 128). So the source's progress positions hold some information about the source frontier, which is
   expected: each was computed while attending to SRC's task list. It is far weaker than the task span and only surfaces when
   amplified.
3. **The goal heads are most of the route, not all of it.** At V×1 the 250 goal heads get 0.77 of the all-heads 0.89 on the
   task span. Both converge with the boost (0.97 vs 0.98 at V×4).
4. **Both spans together saturate in every scheme** (0.98–0.99), with no boost needed.

## Relation to the other experiments

- **The closing `\n` matters.** The same grid with the old task span (the task list without its closing `\n`; cache
  `…__ro0/`, superseded) gave task only 0.40 / 0.84 (goal-250, V×1 / V×4) and 0.62 / 0.91 (all heads), and task + progress 0.82
  through goal-250 at V×1. That `\n` belongs to neither old span, so the old grid left it as DST's in every corner. Folding it
  into the task span raises task only to 0.77 / 0.97 and task + progress to 0.98. This is the same correction as the
  path-patching rerun (`path-patching-report.md` §2.5).
- **Path patching agrees exactly:** kv · goal-250 · task at V×1 here (0.75 / 0.78 / 0.78 / 0.78 by datatype) matches the
  path-patching rerun's py→py I3-only numbers instance for instance. It is the same intervention on the same sample.
- **Pre-commitment swap (`precommitment-swap-report.md`):** swapping the whole code-progress residual for SRC's, with the task
  span left as DST's, moved the frontier in 0.06 of cases. That is the residual · progress cell here (0.06 pooled, 0.03 for
  list[str]), reached by an independent implementation.

## Caveats

- **Python → Python only.** Cross-language directions are not run for this grid.
- **No readouts:** causal-shift only. Readout-shift would need every config recomputed with `--readout`.
- **The task span includes the closing `\n`.** That token attends to the whole task list and probably summarises it, so
  "task span" here means the task list plus the position that closes it. The old-span numbers above show how much that one
  token contributes.
- n = 32 per datatype cell; per-datatype differences of ±0.1 are within noise.

## Reproduce

```bash
# full grid (resumes from the shared cache; readout off)
python scripts/diff/task_vs_progress_eval.py --cuda 6 --datatypes int str 'list[int]' 'list[str]'
# smoke (fresh scratch cache) + reproduction check against the real cache
python scripts/diff/task_vs_progress_eval.py --cuda 6 --datatypes 'list[str]' --max-instances 2 --no-resume --cache-root results/_smoke_tvp
python test/test_task_vs_progress_smoke_compare.py results/_smoke_tvp/notebook_caches/task_vs_progress/Qwen2.5-Coder-32B-Instruct/list_str__python-python__seed1123__mx64__ro0__task+gap
```

The notebook reads the same cache: set `DATATYPE`, run the sweep cell (everything loads from cache), then the results and
inspect cells.
