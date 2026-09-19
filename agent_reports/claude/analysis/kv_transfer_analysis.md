# KV-transfer: goal-head key/value transplant (no ablation)

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

## Takeaway

At 250 goal heads, the key/value transplant drives the destination to the source's sub-goal at
near-ceiling (causal 0.992, readout-shift 0.961) **without any ablation** — the task-span ablation
(`restrict`) that earlier runs applied is unnecessary here (`off ≈ restrict`; off is if anything
marginally higher). The effect is uniform across datatypes and language directions, with only a small
destination-cpp readout dip. The goal heads' K/V over the task span carry a portable, decodable
sub-goal.

*Note: the on-disk top-level `summary.json` for this run was stale (aggregated only `list[str]`); the
tables here are recomputed from the per-cell `meta.json` (all 16 cells present).*
