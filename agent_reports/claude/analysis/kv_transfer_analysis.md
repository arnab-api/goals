anot# KV-transfer: goal-head key/value transplant (no ablation)

*2026-09-19. Qwen2.5-Coder-32B-Instruct. Reports the **non-restrict** (ablation-off) runs only — at
250 goal heads the task-span ablation is unnecessary (`off ≈ restrict`), so the clean, no-intervention
run is the one to report.*

## Experiment

**KV-transfer (keys+values ⇒ goal).** At the frontier pivot of a partially-written function, replace
the goal heads' **keys+values over the task-list span** with the SOURCE prompt's, RoPE-realigned to the
destination positions (GQA-bleed-corrected: only the selected query heads read the source span). SRC
and DST share the prefilled predicates but differ in the frontier task; the transplant asks whether DST
now writes **SRC's frontier task** instead of its own. This changes *what the read finds* at the pivot.

**Best config (by aggregated causal-trigger rate): `multi_pred`.** Between the two templates, both at
250 heads / var_eq / α=2 / no ablation, multi_pred beats dual_comp:

| template | overall causal | overall readout-shift |
|---|:-:|:-:|
| **multi_pred** | **0.992** | **0.961** |
| dual_comp | 0.971 | 0.942 |

All numbers below are **multi_pred, 250 goal heads (patch+readout), `var_eq` pivot, α=2, `pad`
alignment, ablation OFF**, n=32 per cell, 16 cells (4 datatypes × 4 language directions). Oracle:
`claude-sonnet-4-6`. Run: `kv_transfer/multi_pred/ph-goal_250__ro-goal_250__abl-off__a2__pad`.

- **causal** = the patched DST's next computation switches to the SRC frontier task.
- **readout-shift** = GoalScope's patched readout (same 250 heads) verbalizes the SRC goal — the
  representational transfer signal, independent of whether the generation compiles.

## Overall

| metric | value |
|---|:-:|
| causal-trigger | **0.992** |
| readout-shift | **0.961** |

Near-ceiling on both: transplanting the goal heads' K/V over the task span makes the destination pursue
the source's sub-goal ~99% of the time, and the pivot readout names that sub-goal ~96% of the time.

## By language direction (mean over the 4 datatypes)

| direction | causal | readout-shift |
|---|:-:|:-:|
| py→py | 0.992 | 0.984 |
| py→cpp | 0.984 | 0.938 |
| cpp→py | 0.992 | 0.961 |
| cpp→cpp | 1.000 | 0.961 |

Causal is language-invariant (0.98–1.00 across all four directions). Readout-shift dips slightly when
the **destination** is cpp (py→cpp 0.938) but stays ≥0.94 everywhere — the transfer is robust
cross-lingually in both the generation and the verbalization.

## By datatype (mean over the 4 language directions)

| datatype | causal | readout-shift |
|---|:-:|:-:|
| int | 0.984 | 0.977 |
| str | 0.992 | 0.930 |
| list[int] | 0.992 | 0.992 |
| list[str] | 1.000 | 0.945 |

Uniformly strong; the only sub-0.95 readout is `str` (0.930), driven by the two cpp-destination cells
(cpp→py 0.91, cpp→cpp 0.91).

## Full grid (causal / readout-shift, n=32 each)

| datatype | py→py | py→cpp | cpp→py | cpp→cpp |
|---|:-:|:-:|:-:|:-:|
| int | 1.00 / 0.97 | 0.94 / 0.97 | 1.00 / 1.00 | 1.00 / 0.97 |
| str | 1.00 / 0.97 | 1.00 / 0.94 | 0.97 / 0.91 | 1.00 / 0.91 |
| list[int] | 0.97 / 1.00 | 1.00 / 0.97 | 1.00 / 1.00 | 1.00 / 1.00 |
| list[str] | 1.00 / 1.00 | 1.00 / 0.88 | 1.00 / 0.94 | 1.00 / 0.97 |

## Random-head control

Matched placebo: transplant the K/V of a **random later-half non-goal head set** (250 heads, layers
≥32, excluding the goal heads, seed 123) instead of the goal heads — patch = readout = the same random
set, everything else identical (var_eq, α=2, abl-off). py→py, n=32, 4 datatypes.

| run (py→py) | causal | readout-shift |
|---|:-:|:-:|
| GOAL (250 goal heads) | 0.992 | 0.984 |
| **CONTROL (random later-half)** | **0.000** | **0.000** |

The random-head transplant does **nothing** — complete collapse on both metrics. The kv-transfer effect
is entirely specific to the goal heads, not a generic consequence of overwriting *some* heads' K/V over
the task span. Run: `kv_transfer/multi_pred/ph-control250_250__ro-control250_250__abl-off__a2__pad`.

## Sparsity sweep (head budget)

*2026-09-24. Same main config (`multi_pred`, patch = readout = `<budget>` heads, `var_eq`, α=2, `pad`,
**ablation off**), swept over head budgets. 16 cells/budget (4 datatypes × 4 language directions),
n=32 → 512 samples/budget. Runs: `kv_transfer/multi_pred/ph-goal_<N>__ro-goal_<N>__abl-off__a2__pad`
(122 = the canonical `goal__lam20_thr0.9_band25-56` set). Aggregated from per-cell `meta.json`.*

| heads | causal | readout-shift |
|------:|:---:|:---:|
| 50  | 0.213 | 0.324 |
| 100 | 0.467 | 0.488 |
| 122 | 0.502 | 0.516 |
| 150 | 0.549 | 0.533 |
| 200 | **0.883** | **0.865** |
| 250 | **0.992** | **0.961** |

**KV-transfer needs many heads and saturates late.** The curve rises monotonically and only reaches
near-ceiling at 250, with a sharp jump between 150 (0.549) and 200 (0.883) → 250 (0.992). At 50 heads it
barely works (0.213). This is mechanistically expected: the intervention replaces the K/V the goal heads
*read* over the task span, so with few heads most goal-relevant task content is never transplanted.

**Causal and readout track each other** across the whole sweep (both scale with the patched head count),
unlike the q-transfer sweep where readout saturated early (~122) while causal kept climbing. So for KV,
one budget knob moves both the behavioral and the representational transfer together.

### By language direction (causal / readout-shift), per budget

| heads | py→py | py→cpp | cpp→py | cpp→cpp |
|------:|:---:|:---:|:---:|:---:|
| 50  | 0.305 / 0.422 | 0.133 / 0.242 | 0.250 / 0.398 | 0.164 / 0.234 |
| 100 | 0.570 / 0.570 | 0.359 / 0.445 | 0.500 / 0.547 | 0.438 / 0.391 |
| 122 | 0.570 / 0.586 | 0.414 / 0.484 | 0.500 / 0.555 | 0.523 / 0.438 |
| 150 | 0.609 / 0.594 | 0.453 / 0.461 | 0.547 / 0.586 | 0.586 / 0.492 |
| 200 | 0.906 / 0.891 | 0.859 / 0.820 | 0.875 / 0.875 | 0.891 / 0.875 |
| 250 | 0.992 / 0.984 | 0.984 / 0.938 | 0.992 / 0.961 | 1.000 / 0.961 |

At every budget the two `→cpp` directions (destination = C++) trail the `→py` ones, and the gap is
widest at low budgets (e.g. 50: py→py 0.305 vs py→cpp 0.133) — pruning heads hurts cross-language
realization the most. The language-invariance seen at 250 (0.98–1.00 across directions) is a
full-budget property; it erodes as heads are removed.

## Takeaway

At 250 goal heads, the key/value transplant drives the destination to the source's sub-goal at
near-ceiling (causal 0.992, readout-shift 0.961) **without any ablation** — the task-span ablation
(`restrict`) that earlier runs applied is unnecessary here (`off ≈ restrict`; off is if anything
marginally higher). The effect is uniform across datatypes and language directions, with only a small
destination-cpp readout dip. The goal heads' K/V over the task span carry a portable, decodable
sub-goal.

*Note: the on-disk top-level `summary.json` for this run was stale (aggregated only `list[str]`); the
tables here are recomputed from the per-cell `meta.json` (all 16 cells present).*
