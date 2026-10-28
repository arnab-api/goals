# Path patching: old vs new task span (task list without vs with its closing `\n`)

*2026-10-27 · Qwen2.5-Coder-32B-Instruct · every value below is on the **same seeded instances** in both versions
(512 per sub-goal config, 1024 per sub-sub-goal config) · tables `python3 scripts/diff/listgap_compare_tables.py` ·
current report (new span only): `path-patching-report.md` · previous report: `_backups/path-patching-report__2026-10-26__pre-list+gap.md`*

## What changed

The I3 (pivot-onwards KV transfer) task span used to stop at the last task item. The token after it, the `\n` that closes
the task list (the last token before the user turn's `<|im_end|>`), was outside the transplanted window, so at the pivot
the goal heads still read the destination's k/v at that position.

| | **old** (`--span list`) | **new** (`--span list+gap --cpp-note system`) |
|---|---|---|
| task span | `(1) … (3)` (the task items) | `(1) …` up to `<|im_end|>` (excluded) = the task items **+ the closing `\n`** (exactly 1 token, the same in SRC and DST) |
| C++ prompts | "Assume that all the libraries have been already imported." at the end of the user turn | the same sentence at the end of the **system** prompt (otherwise it would fall inside the new span) |
| Python prompts | — | byte-identical to old |

Everything else is identical: I1 off, goal-250 heads, query rule `last`, α = 1, readout heads, oracle, seeds, instances.
py→py rows change only the span; cross-language rows change both the span and the C++ note position. py→py shows the
same jump as the cross-language rows, so most of the change is the `\n`, but these runs do not separate the two.

## Headline

| | old | **new** |
|---|---|---|
| I3 only, V×1 (pooled causal / readout) | 0.37 / 0.30 | **0.72 / 0.67** |
| I3 only, V×4 (the paper's configuration) | 0.80 / 0.72 | **0.94 / 0.90** |
| + I2 (pre-commitment swap), V×1 · readout → e | 0.61 / 0.51 · 0.24 | **0.84 / 0.73 · 0.18** |
| span + progress, V×1 | 0.99 / 0.98 | 0.99 / 0.98 |
| sub-sub-goal, V×4 | 0.79 / 0.65 | 0.79 / 0.65 |
| sub-sub-goal, V×1 | 0.57 / 0.48 | **0.64 / 0.54** |
| control-250, V×4 (sub-goal / sub-sub-goal) | 0 / 512 · 0 / 1024 | 1 / 512 · 0 / 1024 |

## All axes

**Sub-goal · pooled + by language direction** — each cell: **new** · old (causal / readout; with I2 on, · e = readout → p_res's frontier e)

| config | all | py→py | py→cpp | cpp→py | cpp→cpp |
|---|---|---|---|---|---|
| task span · I2 off · V×1 | **0.72 / 0.67** · 0.37 / 0.30 | **0.77 / 0.77** · 0.40 / 0.38 | **0.63 / 0.62** · 0.35 / 0.27 | **0.81 / 0.70** · 0.38 / 0.36 | **0.66 / 0.59** · 0.36 / 0.21 |
| task span · I2 off · V×2 | **0.87 / 0.83** · 0.62 / 0.57 | **0.93 / 0.90** · 0.67 / 0.63 | **0.78 / 0.77** · 0.54 / 0.55 | **0.91 / 0.90** · 0.70 / 0.60 | **0.85 / 0.77** · 0.59 / 0.52 |
| task span · I2 off · V×4 | **0.94 / 0.90** · 0.80 / 0.72 | **0.97 / 0.91** · 0.84 / 0.76 | **0.90 / 0.90** · 0.70 / 0.67 | **0.95 / 0.93** · 0.84 / 0.77 | **0.93 / 0.86** · 0.81 / 0.70 |
| task span · I2 on · V×1 | **0.84 / 0.73 · e 0.18** · 0.61 / 0.51 · e 0.24 | **0.91 / 0.85 · e 0.10** · 0.64 / 0.62 · e 0.14 | **0.79 / 0.63 · e 0.27** · 0.62 / 0.41 · e 0.32 | **0.88 / 0.78 · e 0.14** · 0.62 / 0.55 · e 0.18 | **0.79 / 0.66 · e 0.21** · 0.56 / 0.45 · e 0.30 |
| task span · I2 on · V×2 | **0.94 / 0.90 · e 0.04** · 0.81 / 0.73 · e 0.10 | **0.98 / 0.90 · e 0.04** · 0.88 / 0.81 · e 0.06 | **0.91 / 0.88 · e 0.06** · 0.79 / 0.67 · e 0.18 | **0.94 / 0.93 · e 0.02** · 0.78 / 0.72 · e 0.07 | **0.93 / 0.88 · e 0.05** · 0.80 / 0.73 · e 0.10 |
| task span · I2 on · V×4 | **0.97 / 0.92 · e 0.01** · 0.90 / 0.82 · e 0.04 | **0.99 / 0.91 · e 0.00** · 0.97 / 0.90 · e 0.02 | **0.95 / 0.95 · e 0.02** · 0.84 / 0.77 · e 0.06 | **0.98 / 0.95 · e 0.01** · 0.90 / 0.82 · e 0.02 | **0.98 / 0.88 · e 0.01** · 0.88 / 0.80 · e 0.05 |
| task span + progress · I2 off · V×1 | **0.99 / 0.98** · 0.99 / 0.98 | **0.98 / 0.99** · 0.98 / 1.00 | **1.00 / 0.97** · 0.99 / 0.98 | **0.99 / 0.99** · 0.99 / 0.98 | **0.98 / 0.97** · 0.98 / 0.95 |
| task span + progress · I2 on · V×1 | **0.99 / 0.98 · e 0.00** · 0.99 / 0.98 · e 0.00 | **0.98 / 1.00 · e 0.00** · 0.98 / 1.00 · e 0.00 | **1.00 / 0.97 · e 0.00** · 0.99 / 0.98 · e 0.00 | **0.99 / 0.98 · e 0.00** · 0.99 / 0.98 · e 0.00 | **0.99 / 0.97 · e 0.00** · 0.99 / 0.95 · e 0.00 |

**Sub-goal · by datatype** — each cell: **new** · old (causal / readout; with I2 on, · e = readout → p_res's frontier e)

| config | all | int | str | list_int | list_str |
|---|---|---|---|---|---|
| task span · I2 off · V×1 | **0.72 / 0.67** · 0.37 / 0.30 | **0.71 / 0.69** · 0.38 / 0.41 | **0.69 / 0.73** · 0.34 / 0.27 | **0.70 / 0.70** · 0.35 / 0.35 | **0.79 / 0.56** · 0.41 / 0.20 |
| task span · I2 off · V×2 | **0.87 / 0.83** · 0.62 / 0.57 | **0.87 / 0.87** · 0.66 / 0.69 | **0.84 / 0.84** · 0.55 / 0.57 | **0.86 / 0.84** · 0.65 / 0.62 | **0.91 / 0.77** · 0.64 / 0.41 |
| task span · I2 off · V×4 | **0.94 / 0.90** · 0.80 / 0.72 | **0.92 / 0.90** · 0.85 / 0.78 | **0.91 / 0.93** · 0.73 / 0.74 | **0.95 / 0.93** · 0.80 / 0.80 | **0.97 / 0.84** · 0.81 / 0.57 |
| task span · I2 on · V×1 | **0.84 / 0.73 · e 0.18** · 0.61 / 0.51 · e 0.24 | **0.83 / 0.79 · e 0.13** · 0.67 / 0.62 · e 0.23 | **0.88 / 0.81 · e 0.11** · 0.59 / 0.57 · e 0.16 | **0.73 / 0.70 · e 0.24** · 0.56 / 0.49 · e 0.35 | **0.92 / 0.62 · e 0.24** · 0.61 / 0.34 · e 0.20 |
| task span · I2 on · V×2 | **0.94 / 0.90 · e 0.04** · 0.81 / 0.73 · e 0.10 | **0.93 / 0.91 · e 0.05** · 0.86 / 0.81 · e 0.08 | **0.97 / 0.90 · e 0.03** · 0.80 / 0.82 · e 0.05 | **0.88 / 0.90 · e 0.05** · 0.77 / 0.79 · e 0.12 | **0.98 / 0.88 · e 0.04** · 0.82 / 0.51 · e 0.16 |
| task span · I2 on · V×4 | **0.97 / 0.92 · e 0.01** · 0.90 / 0.82 · e 0.04 | **0.98 / 0.91 · e 0.01** · 0.91 / 0.88 · e 0.03 | **0.98 / 0.95 · e 0.00** · 0.88 / 0.86 · e 0.02 | **0.94 / 0.95 · e 0.02** · 0.85 / 0.89 · e 0.05 | **0.99 / 0.88 · e 0.00** · 0.95 / 0.66 · e 0.05 |
| task span + progress · I2 off · V×1 | **0.99 / 0.98** · 0.99 / 0.98 | **0.99 / 0.98** · 0.98 / 0.98 | **0.99 / 0.98** · 0.99 / 0.98 | **0.98 / 1.00** · 0.98 / 1.00 | **1.00 / 0.96** · 1.00 / 0.95 |
| task span + progress · I2 on · V×1 | **0.99 / 0.98 · e 0.00** · 0.99 / 0.98 · e 0.00 | **0.99 / 0.98 · e 0.00** · 0.98 / 0.99 · e 0.00 | **1.00 / 0.97 · e 0.00** · 1.00 / 0.98 · e 0.00 | **0.98 / 1.00 · e 0.00** · 0.98 / 1.00 · e 0.00 | **1.00 / 0.96 · e 0.00** · 1.00 / 0.95 · e 0.00 |

**Sub-sub-goal (connective pivot) · I2 off** — each cell: **new** · old (causal / readout)

| V | all | py→py | py→cpp | cpp→py | cpp→cpp | list_int | list_str | match | diff | and | or |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ×1 | **0.64 / 0.54** · 0.57 / 0.48 | **0.69 / 0.61** · 0.59 / 0.54 | **0.58 / 0.44** · 0.51 / 0.39 | **0.70 / 0.64** · 0.65 / 0.58 | **0.61 / 0.46** · 0.52 / 0.42 | **0.64 / 0.56** · 0.59 / 0.51 | **0.65 / 0.52** · 0.55 / 0.45 | **0.87 / 0.71** · 0.76 / 0.66 | **0.42 / 0.37** · 0.38 / 0.31 | **0.60 / 0.53** · 0.50 / 0.45 | **0.69 / 0.55** · 0.64 / 0.52 |
| ×2 | **0.76 / 0.62** · 0.72 / 0.61 | **0.80 / 0.66** · 0.75 / 0.64 | **0.72 / 0.58** · 0.67 / 0.56 | **0.79 / 0.69** · 0.78 / 0.67 | **0.73 / 0.57** · 0.68 / 0.57 | **0.74 / 0.61** · 0.71 / 0.62 | **0.78 / 0.64** · 0.73 / 0.61 | **0.96 / 0.79** · 0.91 / 0.75 | **0.56 / 0.46** · 0.53 / 0.47 | **0.77 / 0.65** · 0.71 / 0.63 | **0.75 / 0.60** · 0.73 / 0.60 |
| ×4 | **0.79 / 0.65** · 0.79 / 0.65 | **0.82 / 0.68** · 0.80 / 0.68 | **0.78 / 0.61** · 0.76 / 0.60 | **0.81 / 0.71** · 0.84 / 0.71 | **0.76 / 0.62** · 0.76 / 0.61 | **0.74 / 0.63** · 0.74 / 0.64 | **0.85 / 0.68** · 0.83 / 0.65 | **0.98 / 0.84** · 0.97 / 0.82 | **0.60 / 0.46** · 0.61 / 0.48 | **0.84 / 0.69** · 0.83 / 0.68 | **0.74 / 0.62** · 0.75 / 0.62 |

**Control-250 (V×4, no readout)** — causal-shift, new · old

- sub-goal: **1/512** ({'dst': 509, 'neither': 2, 'src': 1}) · 0/512 ({'dst': 510, 'neither': 2})
- sub-sub-goal: **0/1024** ({'dst': 1023, 'neither': 1}) · 0/1024 ({'dst': 1023, 'neither': 1})

## Which conclusions change

| claim (old report) | old | new | status |
|---|---|---|---|
| The task span alone moves the frontier in only ~37 % of cases at V×1 | 0.37 | **0.72** | **changed.** About half of what looked like "the destination's progress still carries d" sat on the list's closing `\n` |
| The pre-commitment swap (I2) is needed to reach ~0.6 at V×1 | +0.24 (0.37 → 0.61) | **+0.12** (0.72 → 0.84) | **weakened.** I2 still helps at every V, by about half as much |
| Readouts drift to p_res's frontier e under I2 | 0.24 (V×1) | **0.18** (V×1) | holds, smaller; ≤ 0.04 from V×2 |
| The V-boost is needed for a strong effect | V×4 for 0.80 | V×1 already 0.72; V×2 0.87 | **weakened**: the boost sharpens an effect that is already present |
| Span + progress saturates | 0.99 | 0.99 | unchanged (that window already contained the `\n`) |
| C++ destinations are weakest | py→cpp 0.35 / cpp→cpp 0.36 at V×1 | py→cpp 0.63 / cpp→cpp 0.66 | holds (still the two lowest directions) |
| `list[str]` readout lags causal | 0.41 causal / 0.20 readout at V×1 | 0.79 / 0.56 | holds, but `list[str]` gains the most causally (+0.38) |
| Readout-shift sits below causal-shift (pooled, task-span window) | 0.05–0.10 | 0.04–0.11 | holds; narrower except I2 on at V×1 |
| Sub-sub-goal transfer ~79 % at V×4, limited by spec order | 0.79; match 0.97 / diff 0.61 | 0.79; match 0.98 / diff 0.60 | **unchanged** at V×4; the `\n` helps only at low V (0.57 → 0.64 at V×1) |
| The effect is specific to the goal heads | 0 / 512, 0 / 1024 | 1 / 512, 0 / 1024 | unchanged |

## Reading

- **One token carries much of the goal signal at the sub-goal level.** The closing `\n` attends to the whole task list and
  plausibly summarises it. Leaving it as the destination's meant the goal heads still read a summary of DST's list at the
  pivot, which the old report attributed to the destination's code progress. The direct test of task span vs code progress
  (`task-vs-progress-report.md`) confirms the corrected picture: with the progress latents pinned to the destination, the
  source task span alone moves the frontier in 0.89 of cases (all layers) and in 0.77 through the goal heads (py→py, V×1),
  while the source code progress alone moves it in 0.00–0.06.
- **At the sub-sub-goal level the `\n` matters much less.** The transferred difference there is inside one task item (the
  second condition), and the limiting factor is the spec order, which the extra token does not address.
- **Paper impact** (done, `Overleaf-draft` commit `32d4c6d`): Fig. 5 right panel, Fig. 22, Fig. 23 right panel and Table 7
  regenerated from the new runs; text updated (~80 % → ~94 %, readout ~72 % → ~90 %, `list[str]` readout ~57 % → ~84 %;
  control 0 / 512 → 1 / 512). The sub-sub-goal ~79 % and the broader-goal ~77 % are unchanged.

## Where the runs are

- new: `results/diff/<M>/path_patch{,_hier}/…__span-list+gap__cppnote-sys__…`
- old: `results/diff/<M>/path_patch{,_hier}/…__span-list__…` (also backed up under `results/diff/<M>/_backups/path_patch__2026-10-26__pre-list+gap/`)
- job lists: `job_list/diff_eval/path_patch_32b_listgap{,_full}.json`, `path_patch_hier_32b_listgap.json`, `path_patch_control_32b_listgap.json`
