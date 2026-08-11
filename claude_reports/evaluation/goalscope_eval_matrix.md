# GoalScope evaluation matrix — pending jobs, and why

*Started 2026-08-10. Job tracker first; becomes the write-up of (a), (b), (c) as runs land.*

Three questions about how good the goal-head encoding actually is:

| | question | manipulation |
|---|---|---|
| **a** | is the goal-head aggregate better than the raw residual? | source of the injected vector |
| **b** | how many heads does the encoding need? | head-set size |
| **c** | does it depend on scale / code-FT / RL-tuning? | model |

All runs use the same corpus and scoring (`n_specs=512`, `nc=3-4-5`, `seed=123`, `span=rule`,
`maxnew=64`, `nsteer=all`), and every one is run in both `--mode faithfulness` and
`--mode causality`. Results parse with `scripts.evaluation.faithfulness_io`.

---

## Status

Ten runs already on disk (`hard_yes`, OVERALL):

| model | mode | strategy | α | n_specs | hard_yes |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B | faithfulness | additive | 0.2 | 512 | 0.777 |
| Qwen2.5-Coder-32B | faithfulness | replace | 1.0 | 512 | 0.664 |
| Qwen2.5-Coder-32B | causality | additive | 0.2 | 512 | 0.841 |
| Qwen2.5-Coder-32B | causality | replace | 1.0 | 512 | 0.814 |
| Olmo-3.1-32B-Instruct | faithfulness | replace | 1.0 | 512 | 0.410 |
| Olmo-3.1-32B-Instruct | causality | replace | 1.0 | 512 | 0.715 |

(plus `nspec=256` / `nspec=8` partials for Qwen-Coder-32B, superseded by the 512 runs)

**Every `additive` number above is at risk.** See the magnitude caveat below — those runs
predate `additive_norm` and are not safe to compare across head sets or models.

---

## The magnitude caveat (applies to a and b)

`additive` injects `h + α·g_enc`: the added magnitude is `α·‖g_enc‖`, an **absolute** quantity,
while the thing it perturbs has magnitude `‖h‖`. Nothing holds the ratio fixed, and two terms move
independently of the thing under test:

* `‖h‖` grows with depth — so a single run applies a different relative perturbation at each of the
  11 steer layers;
* `‖g_enc‖` grows with head count, since it is a **sum** over heads — so **(b) at fixed α confounds
  "more heads" with "stronger injection"**, which is precisely the variable it means to isolate.

`additive_norm` (commit `283847e`) injects `h + α·‖h‖·ĝ` instead, so α means "add this fraction of
the local residual norm". **α does not carry over** — 0.2 has no meaning under the new units, hence
the α sweep in job group 0.

For **(a)** the same argument decides the strategy: `‖h_src‖` (a full residual) is far larger than
`‖g_enc‖` (a sum over ~120 heads), so a plain additive comparison at fixed α steers the residual arm
much harder and reads magnitude, not content. `eval_residual_patchscope.py` therefore uses
`resid_norm` (= `add_norm` semantics).

> **Open question for the spec.** The original brief for (a) said `strategy="additive", alpha=0.1`,
> exactly as the PatchScope paper. The script as built is norm-relative for the reason above, and is
> also an *additive* injection rather than the paper's single-layer **replace**. If the literal
> replication matters, two arms to add: `--source-agg last` (single layer, closest to the paper) and
> a plain-`additive` residual arm to show what the magnitude confound does. Not yet built.

---

## Job groups

### 0. Calibrate α for `additive_norm` *(prerequisite for a and b)*

α is in new units, so everything downstream needs a defensible value first. Qualitatively α=0.1
already looks right on causality (`12.1`: coherent C++, e.g. `first_impl` →
`for(i=1;i<=value/2;++i) if(value%i==0) sum+=i; return sum>value`), and the residual smoke test at
the same α produced *degenerate* text — so the two arms may not want the same α.

- [ ] `additive_norm` × {0.05, 0.1, 0.2} × faithfulness, goal heads, `n_specs=64` — pick α, then run
      the full grid at it
- [ ] `resid_norm` × {0.02, 0.05, 0.1} × faithfulness, `n_specs=64` — same, for the residual arm

Small `n_specs` on purpose: this is calibration, not a result.

### a. Residual-latent baseline

Isolates how much of the read-out quality comes from the **head selection** versus what the raw
residual at the same positions already carries. Source band = `GOAL_BAND` so both arms cover the
same depth.

- [ ] `resid_norm` faithfulness, α=TBD, `--source-agg mean`
- [ ] `resid_norm` causality, α=TBD, `--source-agg mean`
- [ ] `additive_norm` goal heads faithfulness, α=TBD ← the arm it is compared against
- [ ] `additive_norm` goal heads causality, α=TBD

In `job_list/goalscope_norm_vs_residual.json` (currently pinned at α=0.1; update after group 0).

Optional, only if the literal paper replication is wanted: `--source-agg last`, and a plain-additive
residual arm.

### b. Head sparsity

Top-k by mask value: **25, 50, 100, 150, 200, 300**. Export each with
`notebooks/clock/__head_inspection.ipynb` (`CFG["goal"]["topk"]`, writes
`head_sets/<model>/goal__lam20_top<k>.json`), then point `--heads` at the path.

- [ ] export the six json head sets
- [ ] 6 × faithfulness, `additive_norm`, α=TBD
- [ ] 6 × causality, `additive_norm`, α=TBD

Two things to watch:
* **must be `additive_norm`** — under `additive` this sweep measures ‖g_enc‖ growth (above);
* the goal mask **saturates**: 117 of Qwen-Coder-32B's heads sit at exactly 1.0, so k ≤ 117 cuts
  through a tie block and the selection is index order, not importance. `n_tied@K` in
  `__head_inspection` flags it. k = 25 / 50 / 100 are all inside that block — report them as
  "an arbitrary 25 of the 117 maximal heads", not "the top 25".

### c. Model axes

Three contrasts, one variable each:

| axis | models |
|---|---|
| scale | Qwen2.5-Coder-32B-Instruct vs **Qwen2.5-Coder-14B-Instruct** |
| code fine-tuning | Qwen2.5-Coder-32B-Instruct vs **Qwen2.5-32B-Instruct** |
| RL tuning | Olmo-3.1-32B-Instruct vs **Olmo-3.1-32B-Instruct-SFT** |

- [ ] Qwen2.5-Coder-14B faithfulness + causality
- [ ] **BLOCKED** Qwen2.5-32B-Instruct — no head mask
- [ ] **BLOCKED** Olmo-3.1-32B-Instruct-SFT — no head mask
- [ ] Olmo-3.1-32B-Instruct: only `replace` exists; add the `additive_norm` arm to match

**Blockers.** `results/localization/` currently holds: `Qwen2.5-Coder-32B-Instruct`,
`Qwen2.5-Coder-14B-Instruct`, `Olmo-3.1-32B-Instruct`, `Llama-3.3-70B-Instruct`,
`Qwen2.5-72B-Instruct`, `GLM-4-32B-0414`, `cwm`. **Two of the three axes cannot run**: the code-FT
and RL-tuning comparisons each need a mask trained for the second model
(`scripts/train_head_mask.py`) before any eval job is worth queuing. Both models *are* already
registered in `GOAL_BANDS` (`Qwen2.5-32B-Instruct` → (25,56), `Olmo-3.1-32B-Instruct-SFT` → (10,36)),
so only the mask is missing.

Also note `Qwen2.5-Coder-14B-Instruct` has **no `GOAL_BANDS` entry** — it will fall back to full
depth with a warning, which is not comparable to the 32B's (25,56). Look at its mask heatmap and set
a band before running, or the scale axis measures band width as much as scale.

---

## Comparability rules

* Never compare an `additive` number with an `additive_norm` one — different units.
* Within (b), all six head sets must share one α, and it must be `additive_norm`.
* Within (c), the steer band is currently fixed at (35, 46) for every model. That is the *Qwen-32B*
  band; for Olmo (whose goal band is (10, 36)) it sits outside the goal region entirely. Either set
  a per-model steer band or state plainly that the band is held fixed and the axis therefore
  confounds "where the goal heads live" with the model change.
