# Path patching: what the goal heads read at the pivot (Qwen2.5-Coder-32B-Instruct)

*2026-10-17, updated 2026-10-26 (§2.5) · runs 2026-10-16/17, 2026-10-26 · code `src/diff/path_patch.py`, `scripts/diff/path_patch_eval.py` · job lists
`job_list/diff_eval/path_patch_32b{,_xlang,_vboost,_listgap}.json` · figures/tables `notebooks/fig_n_table_generators/path_patch.ipynb`
→ `notebooks/Figures/path_patch/` · prototype `notebooks/diff/05_kv_transfer.ipynb` §Path patching*

## TL;DR

> **Update 2026-10-26: the task span was missing its closing `\n` (§2.5).** Every I3 result below used a task span that
> stopped at the last task item, leaving out the `\n` that closes the list (the token right before the user turn's
> `<|im_end|>`). With that one token folded in (`--span list+gap`; the C++ note moved to the system prompt, `--cpp-note
> system`), I3 alone (I2 off) gives **0.72 / 0.87 / 0.94** causal-shift at V×1/2/4, against 0.37 / 0.62 / 0.80 before, and readout-shift
> 0.67 / 0.83 / 0.90. At V×4 that is within 0.05 of span + progress (0.99). The rest of this report is the old span. The sub-sub-goal
> grid and both controls are being rerun with the new span.

The paper's KV transfer is a **task-onwards KV transfer**: goal heads read the source task-span k/v at **every query from the
span on**, and it moves the frontier almost perfectly. Here the goal heads read the source task span **only from the pivot
` =` on** (I3, **pivot-onwards KV transfer**). That alone moves the
frontier in just **37 %** of cases: the destination's own **progress tokens** (`<|im_end|> … {var}`) still carry the
destination frontier d, and the goal heads read it there.

* **I2, pre-commitment-swap residual transfer**: replacing those progress latents with a third prompt's (p_res, which is
  pre-committed to its own frontier e, neither c nor d) raises it 0.37 → **0.61**. Part of the readout then follows the
  swapped-in pre-commitment to e (**24 %** of patched readouts).
* **Transplanting span + progress k/v**: **0.99**, with or without I2, and no readout goes to e.
* **Boosting the transplanted task-span values** (V×a, attention pattern unchanged): monotone gains. I2 off:
  0.37 → 0.62 → 0.80 → **0.86** (V×1/2/4/6). I2 on: 0.61 → 0.81 → 0.90 → **0.92**. Drift to e falls 0.24 → 0.02.
  The boost closes most of the gap to span + progress but not all of it. It works best py→py (0.94 / 0.98 at V×6)
  and is weakest with a C++ destination.
* **Random-head control (added 2026-10-19).** The paper configuration (I3 only, I2 off, V×4) routed through 250 random
  later-half non-goal heads instead of the goal heads gives **0.000 causal-shift** (0/512; 0/1024 one level down, at the
  connective), against 0.80 / 0.79 with the goal heads (§2.4).
* **Readout-shift tracks causal-shift** throughout but sits below it: 0.07–0.10 pooled, up to 0.21 for C++
  destinations at V×1, and persistently for `list[str]` (0.66 / 0.71 readout vs 0.88 / 0.92 causal at V×6).

## Setup

Multi-predicate instances (`multi_predicate_setups`, k = 3, letters naming, `var_eq` pivot, listed presentation),
32 per datatype × {int, str, list[int], list[str]} × 4 language directions = **128 per direction, 512 per config**.

```
p_src: a'b'c -> c     k/v donor           (task list a, b, c; a, b prefilled; next = c)
p_dst: a'b'd -> d     the patched run     (task list a, b, d; same prefilled a, b; pivot "c =")
p_res: a'b'e -> e     I2 residual donor   (frontier e ∉ {c, d}, compatible with a, b, c, d; rendered in DST's language,
                                          so its "<|im_end|> … {var}" prefill is token-identical to DST's, asserted)
```

**Terminology.** I1 = *task-onwards KV transfer* (heads read the source window k/v at every query from the window on,
so the destination's task-span latents change too); I2 = *pre-commitment-swap residual transfer* (if the next pending
operation is already decided in the progress latents, swapping M(p_dst)'s progress latents for M(p_res)'s swaps that
pre-commitment); I3 = *pivot-onwards KV transfer* (heads read the source window k/v only from the pivot on).

| intervention | here | what it does |
|---|---|---|
| I1 · task-onwards KV transfer | **off** | task heads read SRC's window k/v at every query from the window on (not used here) |
| I2 · pre-commitment-swap residual transfer | on / off | every decoder layer's OUTPUT residual on `<|im_end|> … {var}` (pivot ` =` excluded) ← p_res's, prompt pass only |
| I3 · pivot-onwards KV transfer | **on** | goal-250 heads read SRC's window k/v from ` =` on (query rule `last`: the pivot, then every generated token); all other heads read the patched run's own k/v |

* **Window** (what I3 transplants): the **task span** (the whole task list), or **task span + progress** (one
  contiguous window, task-span start → ` =` inclusive). SRC/DST windows of different token length are aligned by
  `\n`-padding the shorter one right before it (A1). This needs 0–8 pad tokens within a language and 9–135 across
  languages; all runs of that geometry (clean and patched) use the padded prompts.
* **V-boost** (`--i3-v-scale`): the I3 heads read the transplanted span **values × V**; keys, and so their attention
  pattern, are unchanged (route option `v_scale` in `headwise_kv_patch_routes`). No OV boost anywhere (α = 1).
* **Heads**: patch = readout = goal-250 (`head_sets/.../goal_topk/goal__lam20_top250.json`).
* **Eval**: the kv-transfer oracle (`claude-sonnet-4-6`), each verdict one of c / d / BOTH / NEITHER; only a clean **c**
  counts (BOTH is a miss, for both metrics). *Causal-shift* = the patched DST's **first** computation after ` =` (the
  frontier's value) pursues c; anything it writes afterwards is not judged. *Readout-shift* = the patched GoalScope
  readout at ` =` expresses c, as a share of **all** instances (paper convention, `readout_shift_rate`). *Validity* = clean SRC → c and clean DST → d readouts (0.95 pooled). With I2,
  the patched readout is also judged 3-way over {d, c, e} (*readout → e*) against the clean p_res readout (which reads
  out e in ≈ 0.97 of instances).
* V×1 numbers come from the base runs; the V-boost runs use the same seeded instances (all 512 matched).

## 1 · Overview: I2 off vs on

![V-boost trend](../../../notebooks/Figures/path_patch/vboost_trend.png)

*Figure 1: causal-shift (left) and readout-shift (right) vs the value boost V on the transplanted task span, pooled over
4 directions × 4 datatypes (n = 512 per point). Solid = I2 off, dashed = I2 on (p_res). Dotted = task span + progress k/v
(V×1; ≈ 0.99 causal / 0.98 readout for both I2 settings, overlapping).*

![base grid](../../../notebooks/Figures/path_patch/base_grid_bars.png)

*Figure 2: the V×1 base grid. Light = task span only, dark = task span + progress; blue = I2 off, vermillion = I2 on;
solid = causal-shift, hatched = readout-shift.*

## 2 · I2 off: the destination's own progress latents are left in place

The goal heads read SRC's task-span k/v from ` =` on; every other key (the code progress, the pivot itself) keeps the
destination's own latents. Pooled causal-shift climbs **0.37 → 0.62 → 0.80 → 0.86** with V×1/2/4/6 (readout
0.30 → 0.78), against **0.99 / 0.98** for task span + progress. py→py gets closest (0.94 / 0.84 at V×6). Destinations in
C++ and `str` stay lowest (py→cpp 0.81, cpp→cpp 0.82, str 0.80). `list[str]` moves causally like the others (0.88) but its
readout lags (0.66).

### 2.1 · Aggregated trend with the V-boost

Each cell: causal-shift / readout-shift.

**Pooled and by language direction** (n = 512 pooled, 128 per direction)

| V | pooled | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|---|---|---|---|
| ×1 | 0.37 / 0.30 | 0.40 / 0.38 | 0.35 / 0.27 | 0.38 / 0.36 | 0.36 / 0.21 |
| ×2 | 0.62 / 0.57 | 0.67 / 0.63 | 0.54 / 0.55 | 0.70 / 0.60 | 0.59 / 0.52 |
| ×4 | 0.80 / 0.72 | 0.84 / 0.76 | 0.70 / 0.67 | 0.84 / 0.77 | 0.81 / 0.70 |
| ×6 | 0.86 / 0.78 | 0.94 / 0.84 | 0.81 / 0.77 | 0.87 / 0.79 | 0.82 / 0.71 |
| span + progress (×1) | 0.99 / 0.98 | 0.98 / 1.00 | 0.99 / 0.98 | 0.99 / 0.98 | 0.98 / 0.95 |

**By datatype** (n = 128 per datatype, pooled over directions)

| V | int | str | list[int] | list[str] |
|---|---|---|---|---|
| ×1 | 0.38 / 0.41 | 0.34 / 0.27 | 0.35 / 0.35 | 0.41 / 0.20 |
| ×2 | 0.66 / 0.69 | 0.55 / 0.57 | 0.65 / 0.62 | 0.64 / 0.41 |
| ×4 | 0.85 / 0.78 | 0.73 / 0.74 | 0.80 / 0.80 | 0.81 / 0.57 |
| ×6 | 0.89 / 0.84 | 0.80 / 0.77 | 0.88 / 0.82 | 0.88 / 0.66 |
| span + progress (×1) | 0.98 / 0.98 | 0.99 / 0.98 | 0.98 / 1.00 | 1.00 / 0.95 |

### 2.2 · By language direction and datatype

![I2 off](../../../notebooks/Figures/path_patch/vboost_i2off.png)

*Figure 3: I2 off. Causal-shift (top) and readout-shift (bottom) vs V, by language direction (left) and by datatype
(right); black = pooled, dotted = task span + progress (pooled).*

### 2.3 · Full grids

**Full grids** (datatype × direction, n = 32 per cell; clean-readout validity in brackets)

*×1*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.44 / 0.50 [0.94] | 0.28 / 0.28 [0.97] | 0.38 / 0.47 [1.00] | 0.44 / 0.38 [0.94] | 0.38 / 0.41 [0.96] |
| str | 0.34 / 0.41 [0.94] | 0.38 / 0.25 [0.91] | 0.31 / 0.34 [0.94] | 0.34 / 0.06 [0.94] | 0.34 / 0.27 [0.93] |
| list[int] | 0.38 / 0.31 [1.00] | 0.34 / 0.38 [0.97] | 0.41 / 0.41 [1.00] | 0.28 / 0.31 [1.00] | 0.35 / 0.35 [0.99] |
| list[str] | 0.44 / 0.31 [1.00] | 0.41 / 0.16 [0.91] | 0.41 / 0.22 [0.94] | 0.38 / 0.09 [0.81] | 0.41 / 0.20 [0.91] |
| all | 0.40 / 0.38 [0.97] | 0.35 / 0.27 [0.94] | 0.38 / 0.36 [0.97] | 0.36 / 0.21 [0.92] | 0.37 / 0.30 [0.95] |

*×2*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.66 / 0.78 [0.94] | 0.47 / 0.53 [0.97] | 0.84 / 0.78 [1.00] | 0.66 / 0.66 [0.94] | 0.66 / 0.69 [0.96] |
| str | 0.69 / 0.59 [0.94] | 0.56 / 0.56 [0.91] | 0.47 / 0.56 [0.94] | 0.47 / 0.56 [0.94] | 0.55 / 0.57 [0.93] |
| list[int] | 0.69 / 0.66 [1.00] | 0.50 / 0.59 [0.97] | 0.81 / 0.66 [1.00] | 0.59 / 0.59 [1.00] | 0.65 / 0.62 [0.99] |
| list[str] | 0.66 / 0.50 [1.00] | 0.62 / 0.50 [0.91] | 0.66 / 0.41 [0.94] | 0.62 / 0.25 [0.81] | 0.64 / 0.41 [0.91] |
| all | 0.67 / 0.63 [0.97] | 0.54 / 0.55 [0.94] | 0.70 / 0.60 [0.97] | 0.59 / 0.52 [0.92] | 0.62 / 0.57 [0.95] |

*×4*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.88 / 0.88 [0.94] | 0.75 / 0.59 [0.97] | 0.97 / 0.84 [1.00] | 0.81 / 0.81 [0.94] | 0.85 / 0.78 [0.96] |
| str | 0.88 / 0.72 [0.94] | 0.72 / 0.78 [0.91] | 0.69 / 0.72 [0.94] | 0.66 / 0.75 [0.94] | 0.73 / 0.74 [0.93] |
| list[int] | 0.81 / 0.81 [1.00] | 0.59 / 0.72 [0.97] | 0.91 / 0.88 [1.00] | 0.88 / 0.81 [1.00] | 0.80 / 0.80 [0.99] |
| list[str] | 0.81 / 0.62 [1.00] | 0.75 / 0.59 [0.91] | 0.78 / 0.66 [0.94] | 0.91 / 0.41 [0.81] | 0.81 / 0.57 [0.91] |
| all | 0.84 / 0.76 [0.97] | 0.70 / 0.67 [0.94] | 0.84 / 0.77 [0.97] | 0.81 / 0.70 [0.92] | 0.80 / 0.72 [0.95] |

*×6*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 0.91 [0.94] | 0.81 / 0.72 [0.97] | 0.94 / 0.88 [1.00] | 0.84 / 0.88 [0.94] | 0.89 / 0.84 [0.96] |
| str | 0.91 / 0.81 [0.94] | 0.84 / 0.81 [0.91] | 0.78 / 0.78 [0.94] | 0.66 / 0.69 [0.94] | 0.80 / 0.77 [0.93] |
| list[int] | 0.91 / 0.88 [1.00] | 0.78 / 0.81 [0.97] | 0.91 / 0.78 [1.00] | 0.91 / 0.81 [1.00] | 0.88 / 0.82 [0.99] |
| list[str] | 0.97 / 0.75 [1.00] | 0.81 / 0.72 [0.91] | 0.84 / 0.72 [0.94] | 0.88 / 0.47 [0.81] | 0.88 / 0.66 [0.91] |
| all | 0.94 / 0.84 [0.97] | 0.81 / 0.77 [0.94] | 0.87 / 0.79 [0.97] | 0.82 / 0.71 [0.92] | 0.86 / 0.78 [0.95] |

*span + progress (×1)*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 1.00 [0.94] | 0.97 / 0.97 [1.00] | 1.00 / 0.97 [1.00] | 1.00 / 1.00 [0.94] | 0.98 / 0.98 [0.97] |
| str | 1.00 / 1.00 [0.94] | 1.00 / 0.94 [0.91] | 1.00 / 1.00 [0.94] | 0.97 / 1.00 [0.94] | 0.99 / 0.98 [0.93] |
| list[int] | 0.97 / 1.00 [1.00] | 1.00 / 1.00 [0.97] | 0.97 / 1.00 [1.00] | 0.97 / 1.00 [1.00] | 0.98 / 1.00 [0.99] |
| list[str] | 1.00 / 1.00 [1.00] | 1.00 / 1.00 [0.91] | 1.00 / 0.97 [0.94] | 1.00 / 0.81 [0.81] | 1.00 / 0.95 [0.91] |
| all | 0.98 / 1.00 [0.97] | 0.99 / 0.98 [0.95] | 0.99 / 0.98 [0.97] | 0.98 / 0.95 [0.92] | 0.99 / 0.98 [0.95] |

## 2.4 · Random-head control (V×4, the paper configuration)

*Added 2026-10-19.* The exact configuration reported in the paper (App. I.1 / I.1.1; Fig. 22, Table 7, Fig. 23): I1 off,
I2 off, I3 on, task-span window, V×4, α = 1, the same seeded instances. The only change is the head set: **250 heads drawn at
random from layers 32–63 with no goal head** (`head_sets/Qwen2.5-Coder-32B-Instruct/control/control250__laterhalf.json`)
read SRC's task-span k/v from the pivot on, in place of the goal-250 heads. Run without GoalScope readouts
(`--no-readout`), so causal-shift only.

**Sub-goal (multi-predicate; mirrors Fig. 22 / Table 7)** — causal-shift, n = 128 per row, 512 pooled

| | goal-250 | control-250 |
|---|---|---|
| **pooled** | **0.80** | **0.00** |
| py→py | 0.84 | 0.00 |
| py→cpp | 0.70 | 0.00 |
| cpp→py | 0.84 | 0.00 |
| cpp→cpp | 0.81 | 0.00 |
| int | 0.85 | 0.00 |
| str | 0.73 | 0.00 |
| list[int] | 0.80 | 0.00 |
| list[str] | 0.81 | 0.00 |

Patched verdicts under the control: 510 d, 2 neither, 0 c.

**Sub-sub-goal (multi-hierarchy, pivot = the connective; mirrors Fig. 23)** — causal-shift, n = 256 per direction,
512 per datatype / connective / order, 1024 pooled

| | goal-250 | control-250 |
|---|---|---|
| **pooled** | **0.79** | **0.00** |
| py→py / py→cpp / cpp→py / cpp→cpp | 0.80 / 0.76 / 0.84 / 0.76 | 0.00 / 0.00 / 0.00 / 0.00 |
| list[int] / list[str] | 0.74 / 0.83 | 0.00 / 0.00 |
| and / or | 0.83 / 0.75 | 0.00 / 0.00 |
| order match / diff | 0.97 / 0.61 | 0.00 / 0.00 |

Patched verdicts under the control: 1023 the destination's own sub-sub-goal, 1 neither, 0 the source's.

So the same k/v transplant with the same value boost has **no effect at all** unless the reading heads are the goal heads:
the redirection is specific to them, not a generic consequence of up-weighting the source task span in 250 later-layer heads.

Runs: `results/diff/<M>/path_patch{,_hier}/I1-off__I2-off__I3-on-V4__win-span__span-list__a1__th-control250_250__gh-control250_250__ro-off/`
· job list `job_list/diff_eval/path_patch_control_32b.json`.

## 2.5 · Task span + its closing `\n` (rerun 2026-10-26)

**What changed.** The task span ended at the last task item. The token after it, the `\n` that closes the list, was
outside the window, so at the pivot the goal heads still read DST's k/v there. The rerun uses
`PP.build_geometry(span="list+gap")`, where the task span runs from `(1)` up to the user turn's `<|im_end|>` (excluded). In every
instance that adds exactly one token, the `\n`, the same in SRC and DST (checked on all 512). C++ prompts ended the user turn with
`\nAssume that all the libraries have been already imported.\n`, which would otherwise fall inside the new span. That sentence now
goes at the end of the system prompt instead (`--cpp-note system`; C++ prompts only, Python prompts are byte-identical). Everything
else is unchanged: I1 off, I2 off, I3 on with goal-250 heads, task-span window, α = 1, the same seeded instances, readout on.

**Caveat.** Cross-language rows change two things at once (the `\n` and where the C++ note sits). py→py changes only the `\n`
and shows the same jump (0.40 → 0.77 at V×1), so most of the gain is the `\n`, but these runs don't separate the two.

Each cell: causal-shift / readout-shift. **New** (task span + `\n`) · old (§2.1), n = 512 pooled, 128 per direction / datatype.

| V | pooled | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|---|---|---|---|
| ×1 | **0.72 / 0.67** · 0.37 / 0.30 | **0.77 / 0.77** · 0.40 / 0.38 | **0.63 / 0.62** · 0.35 / 0.27 | **0.81 / 0.70** · 0.38 / 0.36 | **0.66 / 0.59** · 0.36 / 0.21 |
| ×2 | **0.87 / 0.83** · 0.62 / 0.57 | **0.93 / 0.90** · 0.67 / 0.63 | **0.78 / 0.77** · 0.54 / 0.55 | **0.91 / 0.90** · 0.70 / 0.60 | **0.85 / 0.77** · 0.59 / 0.52 |
| ×4 | **0.94 / 0.90** · 0.80 / 0.72 | **0.97 / 0.91** · 0.84 / 0.76 | **0.90 / 0.90** · 0.70 / 0.67 | **0.95 / 0.93** · 0.84 / 0.77 | **0.93 / 0.86** · 0.81 / 0.70 |

| V | int | str | list[int] | list[str] |
|---|---|---|---|---|
| ×1 | **0.71 / 0.69** · 0.38 / 0.41 | **0.69 / 0.73** · 0.34 / 0.27 | **0.70 / 0.70** · 0.35 / 0.35 | **0.79 / 0.56** · 0.41 / 0.20 |
| ×2 | **0.87 / 0.87** · 0.66 / 0.69 | **0.84 / 0.84** · 0.55 / 0.57 | **0.86 / 0.84** · 0.65 / 0.62 | **0.91 / 0.77** · 0.64 / 0.41 |
| ×4 | **0.92 / 0.90** · 0.85 / 0.78 | **0.91 / 0.93** · 0.73 / 0.74 | **0.95 / 0.93** · 0.80 / 0.80 | **0.97 / 0.84** · 0.81 / 0.57 |

**Full grids, new span** (datatype × direction, n = 32 per cell; clean-readout validity in brackets)

*×1*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.75 / 0.75 [0.94] | 0.53 / 0.56 [0.97] | 0.88 / 0.75 [1.00] | 0.69 / 0.69 [1.00] | 0.71 / 0.69 [0.98] |
| str | 0.78 / 0.81 [0.94] | 0.72 / 0.75 [0.94] | 0.69 / 0.75 [0.94] | 0.56 / 0.59 [0.94] | 0.69 / 0.73 [0.94] |
| list[int] | 0.78 / 0.78 [1.00] | 0.53 / 0.66 [1.00] | 0.84 / 0.75 [1.00] | 0.62 / 0.62 [0.97] | 0.70 / 0.70 [0.99] |
| list[str] | 0.78 / 0.72 [1.00] | 0.75 / 0.53 [0.94] | 0.84 / 0.53 [0.91] | 0.78 / 0.47 [0.84] | 0.79 / 0.56 [0.92] |
| all | 0.77 / 0.77 [0.97] | 0.63 / 0.62 [0.96] | 0.81 / 0.70 [0.96] | 0.66 / 0.59 [0.94] | 0.72 / 0.67 [0.96] |

*×2*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.94 / 0.91 [0.94] | 0.75 / 0.75 [0.97] | 0.94 / 0.97 [1.00] | 0.84 / 0.84 [1.00] | 0.87 / 0.87 [0.98] |
| str | 0.88 / 0.88 [0.94] | 0.84 / 0.88 [0.94] | 0.84 / 0.88 [0.94] | 0.78 / 0.75 [0.94] | 0.84 / 0.84 [0.94] |
| list[int] | 0.94 / 0.94 [1.00] | 0.69 / 0.72 [1.00] | 0.97 / 0.91 [1.00] | 0.84 / 0.81 [0.97] | 0.86 / 0.84 [0.99] |
| list[str] | 0.97 / 0.88 [1.00] | 0.84 / 0.72 [0.94] | 0.88 / 0.84 [0.91] | 0.94 / 0.66 [0.84] | 0.91 / 0.77 [0.92] |
| all | 0.93 / 0.90 [0.97] | 0.78 / 0.77 [0.96] | 0.91 / 0.90 [0.96] | 0.85 / 0.77 [0.94] | 0.87 / 0.83 [0.96] |

*×4*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 0.91 [0.94] | 0.84 / 0.84 [0.97] | 0.97 / 0.94 [1.00] | 0.91 / 0.91 [1.00] | 0.92 / 0.90 [0.98] |
| str | 0.94 / 0.94 [0.94] | 0.91 / 1.00 [0.94] | 0.91 / 0.94 [0.94] | 0.88 / 0.84 [0.94] | 0.91 / 0.93 [0.94] |
| list[int] | 0.97 / 0.91 [1.00] | 0.88 / 0.91 [1.00] | 0.97 / 0.97 [1.00] | 0.97 / 0.94 [0.97] | 0.95 / 0.93 [0.99] |
| list[str] | 1.00 / 0.91 [1.00] | 0.97 / 0.84 [0.94] | 0.94 / 0.88 [0.91] | 0.97 / 0.75 [0.84] | 0.97 / 0.84 [0.92] |
| all | 0.97 / 0.91 [0.97] | 0.90 / 0.90 [0.96] | 0.95 / 0.93 [0.96] | 0.93 / 0.86 [0.94] | 0.94 / 0.90 [0.96] |

**Reading.**
- **One token carries much of what §4.1 attributed to the progress.** With the closing `\n` from SRC, I3 at V×1 nearly doubles
  (0.37 → 0.72), and at V×4 it reaches 0.94, close to span + progress (0.99). So much of the "residual d" that the goal heads
  were reading in the destination sat on the `\n` that ends the task list. That position attends to the whole list and probably
  summarises it, rather than to the code progress.
- **The gain is largest where the old span was weakest:** `list[str]` (0.41 → 0.79 at V×1) and C++ destinations at low V.
  py→cpp is now the weakest direction (0.63 / 0.90 at V×1 / V×4).
- **Readout follows causal more closely:** the pooled gap is 0.04–0.05, against 0.07–0.10 before. `list[str]` readout still
  lags (0.56 vs 0.79 at V×1), most with C++ destinations (cpp→cpp 0.47, where clean-readout validity is lowest, 0.84).
- **Open:** whether the old §2.4 control (0.00) holds with the new span, and how the sub-sub-goal level (Fig. 23) moves. Both are
  being rerun (`job_list/diff_eval/path_patch_{hier,control}_32b_listgap.json`).

Runs: `results/diff/<M>/path_patch/I1-off__I2-off__I3-on{,-V2,-V4}__win-span__span-list+gap__cppnote-sys__a1__th-goal_250__gh-goal_250__ro-goal_250/`
· job list `job_list/diff_eval/path_patch_32b_listgap.json` · tables `scripts/diff/listgap_report_tables.py` · old results
backed up under `results/diff/<M>/_backups/path_patch__2026-10-26__pre-list+gap/`.

## 3 · I2 on: pre-commitment-swap residual transfer from p_res (a'b'e)

Every layer's residual on `<|im_end|> … {var}` is replaced by p_res's, whose progress carries neither c nor d. At V×1 this
alone lifts causal-shift from 0.37 to **0.61**, but **24 %** of patched readouts then express p_res's frontier e: the goal
heads take on what the restored progress carries (most when the destination is C++: py→cpp 0.32, cpp→cpp 0.30). Boosting the
span values both raises the shift (**0.61 → 0.81 → 0.90 → 0.92**; readout 0.51 → 0.83) and suppresses the drift to e
(0.24 → 0.10 → 0.04 → 0.02). py→py nearly saturates (0.97 at V×4, 0.98 at V×6); py→cpp stays lowest (0.88).

### 3.1 · Aggregated trend with the V-boost

Each cell: causal-shift / readout-shift · readout → e.

**Pooled and by language direction** (n = 512 pooled, 128 per direction)

| V | pooled | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|---|---|---|---|
| ×1 | 0.61 / 0.51 · e 0.24 | 0.64 / 0.62 · e 0.14 | 0.62 / 0.41 · e 0.32 | 0.62 / 0.55 · e 0.18 | 0.56 / 0.45 · e 0.30 |
| ×2 | 0.81 / 0.73 · e 0.10 | 0.88 / 0.81 · e 0.06 | 0.79 / 0.67 · e 0.18 | 0.78 / 0.72 · e 0.07 | 0.80 / 0.73 · e 0.10 |
| ×4 | 0.90 / 0.82 · e 0.04 | 0.97 / 0.90 · e 0.02 | 0.84 / 0.77 · e 0.06 | 0.90 / 0.82 · e 0.02 | 0.88 / 0.80 · e 0.05 |
| ×6 | 0.92 / 0.83 · e 0.02 | 0.98 / 0.89 · e 0.00 | 0.88 / 0.79 · e 0.04 | 0.91 / 0.85 · e 0.02 | 0.91 / 0.80 · e 0.03 |
| span + progress (×1) | 0.99 / 0.98 · e 0.00 | 0.98 / 1.00 · e 0.00 | 0.99 / 0.98 · e 0.00 | 0.99 / 0.98 · e 0.00 | 0.99 / 0.95 · e 0.00 |

**By datatype** (n = 128 per datatype, pooled over directions)

| V | int | str | list[int] | list[str] |
|---|---|---|---|---|
| ×1 | 0.67 / 0.62 · e 0.23 | 0.59 / 0.57 · e 0.16 | 0.56 / 0.49 · e 0.35 | 0.61 / 0.34 · e 0.20 |
| ×2 | 0.86 / 0.81 · e 0.08 | 0.80 / 0.82 · e 0.05 | 0.77 / 0.79 · e 0.12 | 0.82 / 0.51 · e 0.16 |
| ×4 | 0.91 / 0.88 · e 0.03 | 0.88 / 0.86 · e 0.02 | 0.85 / 0.89 · e 0.05 | 0.95 / 0.66 · e 0.05 |
| ×6 | 0.95 / 0.88 · e 0.02 | 0.91 / 0.85 · e 0.02 | 0.89 / 0.90 · e 0.02 | 0.92 / 0.71 · e 0.02 |
| span + progress (×1) | 0.98 / 0.99 · e 0.00 | 1.00 / 0.98 · e 0.00 | 0.98 / 1.00 · e 0.00 | 1.00 / 0.95 · e 0.00 |

### 3.2 · By language direction and datatype

![I2 on](../../../notebooks/Figures/path_patch/vboost_i2on.png)

*Figure 4: I2 on (p_res). Causal-shift (top) and readout-shift (bottom) vs V, by language direction (left) and by datatype
(right); black = pooled, dotted = task span + progress (pooled).*

![readout to e](../../../notebooks/Figures/path_patch/vboost_readout_to_e.png)

*Figure 5: I2 on. Share of patched readouts that express p_res's frontier e, vs V, by direction.*

### 3.3 · Full grids

**Full grids** (datatype × direction, n = 32 per cell; clean-readout validity in brackets)

*×1*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.84 / 0.78 · e 0.12 [0.94] | 0.56 / 0.47 · e 0.34 [0.97] | 0.66 / 0.66 · e 0.16 [1.00] | 0.62 / 0.59 · e 0.31 [0.94] | 0.67 / 0.62 · e 0.23 [0.96] |
| str | 0.62 / 0.69 · e 0.09 [0.94] | 0.62 / 0.47 · e 0.19 [0.91] | 0.62 / 0.69 · e 0.12 [0.94] | 0.50 / 0.44 · e 0.25 [0.94] | 0.59 / 0.57 · e 0.16 [0.93] |
| list[int] | 0.44 / 0.53 · e 0.22 [1.00] | 0.59 / 0.44 · e 0.41 [0.97] | 0.62 / 0.56 · e 0.28 [1.00] | 0.59 / 0.44 · e 0.50 [1.00] | 0.56 / 0.49 · e 0.35 [0.99] |
| list[str] | 0.66 / 0.47 · e 0.12 [1.00] | 0.69 / 0.25 · e 0.34 [0.91] | 0.56 / 0.31 · e 0.16 [0.94] | 0.53 / 0.31 · e 0.16 [0.81] | 0.61 / 0.34 · e 0.20 [0.91] |
| all | 0.64 / 0.62 · e 0.14 [0.97] | 0.62 / 0.41 · e 0.32 [0.94] | 0.62 / 0.55 · e 0.18 [0.97] | 0.56 / 0.45 · e 0.30 [0.92] | 0.61 / 0.51 · e 0.24 [0.95] |

*×2*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 0.88 · e 0.00 [0.94] | 0.75 / 0.75 · e 0.19 [0.97] | 0.88 / 0.81 · e 0.03 [1.00] | 0.84 / 0.81 · e 0.09 [0.94] | 0.86 / 0.81 · e 0.08 [0.96] |
| str | 0.84 / 0.88 · e 0.06 [0.94] | 0.75 / 0.72 · e 0.09 [0.91] | 0.78 / 0.88 · e 0.06 [0.94] | 0.81 / 0.81 · e 0.00 [0.94] | 0.80 / 0.82 · e 0.05 [0.93] |
| list[int] | 0.84 / 0.81 · e 0.12 [1.00] | 0.72 / 0.69 · e 0.16 [0.97] | 0.75 / 0.81 · e 0.09 [1.00] | 0.75 / 0.84 · e 0.12 [1.00] | 0.77 / 0.79 · e 0.12 [0.99] |
| list[str] | 0.84 / 0.69 · e 0.06 [1.00] | 0.94 / 0.53 · e 0.28 [0.91] | 0.72 / 0.38 · e 0.09 [0.94] | 0.78 / 0.44 · e 0.19 [0.81] | 0.82 / 0.51 · e 0.16 [0.91] |
| all | 0.88 / 0.81 · e 0.06 [0.97] | 0.79 / 0.67 · e 0.18 [0.94] | 0.78 / 0.72 · e 0.07 [0.97] | 0.80 / 0.73 · e 0.10 [0.92] | 0.81 / 0.73 · e 0.10 [0.95] |

*×4*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 0.94 · e 0.00 [0.94] | 0.78 / 0.84 · e 0.06 [0.97] | 0.94 / 0.84 · e 0.00 [1.00] | 0.94 / 0.88 · e 0.06 [0.94] | 0.91 / 0.88 · e 0.03 [0.96] |
| str | 0.94 / 0.91 · e 0.00 [0.94] | 0.88 / 0.84 · e 0.03 [0.91] | 0.88 / 0.91 · e 0.00 [0.94] | 0.84 / 0.78 · e 0.03 [0.94] | 0.88 / 0.86 · e 0.02 [0.93] |
| list[int] | 0.97 / 0.94 · e 0.06 [1.00] | 0.78 / 0.72 · e 0.09 [0.97] | 0.84 / 0.91 · e 0.03 [1.00] | 0.81 / 1.00 · e 0.03 [1.00] | 0.85 / 0.89 · e 0.05 [0.99] |
| list[str] | 1.00 / 0.81 · e 0.03 [1.00] | 0.94 / 0.69 · e 0.06 [0.91] | 0.94 / 0.62 · e 0.03 [0.94] | 0.94 / 0.53 · e 0.09 [0.81] | 0.95 / 0.66 · e 0.05 [0.91] |
| all | 0.97 / 0.90 · e 0.02 [0.97] | 0.84 / 0.77 · e 0.06 [0.94] | 0.90 / 0.82 · e 0.02 [0.97] | 0.88 / 0.80 · e 0.05 [0.92] | 0.90 / 0.82 · e 0.04 [0.95] |

*×6*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 1.00 / 0.94 · e 0.00 [0.94] | 0.88 / 0.84 · e 0.03 [0.97] | 0.97 / 0.88 · e 0.00 [1.00] | 0.94 / 0.84 · e 0.03 [0.94] | 0.95 / 0.88 · e 0.02 [0.96] |
| str | 0.94 / 0.91 · e 0.00 [0.94] | 0.97 / 0.81 · e 0.03 [0.91] | 0.91 / 0.91 · e 0.00 [0.94] | 0.84 / 0.78 · e 0.06 [0.94] | 0.91 / 0.85 · e 0.02 [0.93] |
| list[int] | 0.97 / 0.94 · e 0.00 [1.00] | 0.84 / 0.78 · e 0.06 [0.97] | 0.84 / 0.94 · e 0.03 [1.00] | 0.91 / 0.94 · e 0.00 [1.00] | 0.89 / 0.90 · e 0.02 [0.99] |
| list[str] | 1.00 / 0.78 · e 0.00 [1.00] | 0.84 / 0.72 · e 0.03 [0.91] | 0.91 / 0.69 · e 0.03 [0.94] | 0.94 / 0.66 · e 0.03 [0.81] | 0.92 / 0.71 · e 0.02 [0.91] |
| all | 0.98 / 0.89 · e 0.00 [0.97] | 0.88 / 0.79 · e 0.04 [0.94] | 0.91 / 0.85 · e 0.02 [0.97] | 0.91 / 0.80 · e 0.03 [0.92] | 0.92 / 0.83 · e 0.02 [0.95] |

*span + progress (×1)*

| datatype | py→py | py→cpp | cpp→py | cpp→cpp | all |
|---|---|---|---|---|---|
| int | 0.97 / 1.00 · e 0.00 [0.94] | 0.97 / 0.97 · e 0.00 [1.00] | 1.00 / 1.00 · e 0.00 [1.00] | 1.00 / 1.00 · e 0.00 [0.94] | 0.98 / 0.99 · e 0.00 [0.97] |
| str | 1.00 / 1.00 · e 0.00 [0.94] | 1.00 / 0.94 · e 0.00 [0.91] | 1.00 / 0.97 · e 0.00 [0.94] | 1.00 / 1.00 · e 0.00 [0.94] | 1.00 / 0.98 · e 0.00 [0.93] |
| list[int] | 0.97 / 1.00 · e 0.00 [1.00] | 1.00 / 1.00 · e 0.00 [0.97] | 0.97 / 1.00 · e 0.00 [1.00] | 0.97 / 1.00 · e 0.00 [1.00] | 0.98 / 1.00 · e 0.00 [0.99] |
| list[str] | 1.00 / 1.00 · e 0.00 [1.00] | 1.00 / 1.00 · e 0.00 [0.91] | 1.00 / 0.97 · e 0.00 [0.94] | 1.00 / 0.81 · e 0.00 [0.81] | 1.00 / 0.95 · e 0.00 [0.91] |
| all | 0.98 / 1.00 · e 0.00 [0.97] | 0.99 / 0.98 · e 0.00 [0.95] | 0.99 / 0.98 · e 0.00 [0.97] | 0.99 / 0.95 · e 0.00 [0.92] | 0.99 / 0.98 · e 0.00 [0.95] |

## 4 · Reading

1. **The pivot re-derives the frontier from the progress tokens as well as the task span.** With the source task span
   alone (I3, V×1) the goal heads write c only 37 % of the time. Their output at ` =` is `Σ_j a_j v_j` over **all** keys,
   and only the span keys are swapped. The destination's progress positions (and the pivot itself) keep values built while
   attending to DST's own task list, so they still carry d. The paper's KV transfer avoids this because its `"all"` query
   rule also rewrites those positions' latents.
2. **The pre-commitment swap helps but does not finish the job.** I2 with p_res raises causal-shift to 0.61 in every
   direction: it removes DST's pre-commitment to d from the progress, but does not supply one to c. Its signature is the
   readout → e drift. When the destination is C++ about 30 % of patched readouts land on e, the frontier p_res is
   pre-committed to. So the next pending operation is (partly) already decided in the progress latents, and the goal heads
   read it there.
3. **Transplanting span + progress k/v saturates.** That is 0.98–0.99 causal / 0.95–1.00 readout in every direction and
   datatype, independent of I2, with zero drift to e. Once the goal heads see a fully source-consistent context they
   follow it.
4. **Up-weighting the span's values is a cheap partial substitute.** V×a raises the source-span share of the goal-head
   output without changing where they attend. It climbs monotonically and saturates near 0.86 (I2 off) / 0.92 (I2 on)
   pooled, and suppresses the drift to e. It falls short of span + progress, most with a C++ destination (py→cpp 0.81 /
   0.88, cpp→cpp 0.82 / 0.91 at V×6, vs py→py 0.94 / 0.98).
5. **It is the goal heads specifically.** The same transplant routed through 250 random later-half non-goal heads moves
   nothing (0.00 at both hierarchy levels, §2.4).
6. **Readout lags causal.** Pooled, readout-shift is 0.07–0.10 below causal-shift; the gap is largest for `list[str]`
   (readout 0.66 / 0.71 at V×6 vs causal 0.88 / 0.92) and C++ destinations at low V (up to 0.21) (lower clean-readout validity there: 0.81 for cpp→cpp `list[str]`).

## 5 · Caveats

* Cross-language span + progress uses large A1 pads (9–135 `\n`). Clean validity there stays ≈ 0.95, so the padding does
  not visibly disturb the destination, but it is a different prompt from the unpadded task-span runs.
* p_res's frontier e is a seeded draw per instance (stable across configs). Readout → e is measured over all instances
  (its clean p_res readout validity ≈ 0.97).
* V×6 is the largest boost tried; the curves are still rising slightly in some directions.
* Pending: whether the remaining V-boost failures overlap the V×1 failures (inspect with `PP_MANUAL` + the failure finder
  in `05_kv_transfer.ipynb`); I1 (task-span latents via the task heads) has not been run at scale.
