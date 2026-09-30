# Cross-model goal-head transfer: Coder-32B heads → Qwen2.5-32B-Instruct

> **Paper usage** (section numbers as of the 2026-09-29 build of `Overleaf-draft/Goals-Overleaf/main.pdf`)
> - App F.2 Effect of Code-finetuning and RL-tuning — `fig:comparison-code` (Coder-32B's 122 heads applied to Qwen2.5-32B-Instruct, additive_norm α=0.1).
> - Not cited in the main text.


*2026-09-21. Do the goal heads localized in `Qwen2.5-Coder-32B-Instruct` carry a goal signal that is
**faithful and causal in a different model** — the non-coder `Qwen2.5-32B-Instruct`? We apply the Coder's
122-head goal set directly to the non-coder and run GoalScope faithfulness + causality, then compare
against (a) the non-coder's **own** mined heads and (b) the Coder's own result (upper bound).*

## Setup

- **Transplanted head set:** `head_sets/coder32b_goal122.json` — the 122 goal heads of
  `Qwen2.5-Coder-32B-Instruct` (`goal__lam20_thr0.9_band25-56`; both models are 64-layer, so the layer
  indices map 1:1). Applied verbatim to `Qwen2.5-32B-Instruct`.
- **Target model:** `Qwen/Qwen2.5-32B-Instruct` (non-coder).
- **Eval:** `scripts/evaluation/eval_goalscope.py`, GoalScope readout, metric = OVERALL `hard_yes` oracle
  rate over 253 locked datapoints × 5 anchors (n=1265). Anchors: `prev_end`, `indent`, `var`, `eq`,
  `first_impl`.
- **Two intervention strategies:**
  - `replace` / α=1.0 (hard swap of the head contribution).
  - `additive_norm` / α=0.1 / layers 35–46 — the **core model's main-results hparams** (norm-relative
    injection `h + α·‖h‖·g/‖g‖`).
- **Baselines:** each model's **own** `heads=goal` set under the same strategy. Non-overwriting; all
  results saved under `results/evaluation/goalscope/<model>/<mode>/<strategy>/<alpha>/heads=…`.

## Results — OVERALL `hard_yes` (n=1265)

### additive_norm / α=0.1 / layers 35–46 (main-results hparams)

| head set applied to Qwen2.5-32B-Instruct | faithfulness | causality |
|---|:-:|:-:|
| own `heads=goal` (its own mined set) | 0.643 | 0.770 |
| **Coder-122 → Qwen2.5-32B-Instruct** (transfer) | **0.806** | **0.772** |
| *Coder-32B own `heads=goal` (source ceiling)* | *0.879* | *0.792* |

### replace / α=1.0

| head set applied to Qwen2.5-32B-Instruct | faithfulness | causality |
|---|:-:|:-:|
| own `heads=goal` (its own mined set) | 0.308 | 0.540 |
| **Coder-122 → Qwen2.5-32B-Instruct** (transfer) | **0.756** | **0.726** |
| *Coder-32B own `heads=goal` (source ceiling)* | *0.823* | *0.783* |

### Per-anchor `hard_yes` (transfer and own, both strategies)

| run | prev_end | indent | var | eq | first_impl | OVERALL |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| transfer · faith · replace       | 0.170 | 0.945 | 0.711 | 0.980 | 0.972 | **0.756** |
| transfer · causal · replace      | 0.154 | 0.909 | 0.640 | 0.972 | 0.953 | **0.726** |
| transfer · faith · additive_norm | 0.304 | 0.921 | 0.877 | 0.960 | 0.964 | **0.806** |
| transfer · causal · additive_norm| 0.300 | 0.929 | 0.715 | 0.957 | 0.960 | **0.772** |
| own · faith · additive_norm      | 0.186 | 0.834 | 0.427 | 0.862 | 0.905 | **0.643** |
| own · causal · additive_norm     | 0.328 | 0.874 | 0.692 | 0.976 | 0.980 | **0.770** |

*(Own · replace OVERALL values 0.308 / 0.540 read from stored `summary.json`; per-anchor not re-logged.)*

## Findings

1. **The transplanted Coder heads read out at-or-above the non-coder's own heads in every cell.** The gap
   is large except one tie: under `additive_norm`, causality is essentially even (own 0.770 ≈ transfer
   0.772). Everywhere else the transplant clearly wins — most dramatically **`replace` faithfulness
   (0.756 vs 0.308)**. The Coder-32B goal-head *positions* carry a genuine, portable goal signal in a
   different model.

2. **The transfer sits just under the Coder's own ceiling throughout** (within ~0.02–0.07). It is also
   remarkably strategy-robust: 0.756 / 0.726 / 0.806 / 0.772 all fall in a tight band, whereas the
   non-coder's own heads swing wildly with strategy.

3. **Strategy matters enormously for the non-coder's *own* heads, not for the transfer.** Switching
   `replace`→`additive_norm` lifts the own-heads readout from 0.308→0.643 (faithfulness) and 0.540→0.770
   (causality). The non-coder's own head set is fine; `replace` is simply a harsh readout for it. The
   transfer is largely indifferent to this.

4. **Anchor structure is preserved across the transplant.** `eq` / `first_impl` / `indent` stay high
   (~0.91–0.98), `var` mid, `prev_end` lowest (0.15–0.33) — the same profile seen within-model, indicating
   the transplanted heads engage the same readout structure rather than a degenerate always-on response.

## Takeaway

Applying `Qwen2.5-Coder-32B-Instruct`'s 122 goal heads directly to `Qwen2.5-32B-Instruct` yields a goal
readout that is **faithful and causal in the target model** — beating the target's own mined heads (tie in
one cell) and landing just below the Coder's own ceiling under both intervention strategies. This is
positive evidence that the goal-head localization is not idiosyncratic to the Coder: the same head
*positions* index a transferable goal encoding across models in the family. The one caveat is that this
transfer is between two 64-layer Qwen models with 1:1 layer alignment; extending to non-Qwen families or
mismatched depths is a separate test.

## Provenance

- Head set: `head_sets/coder32b_goal122.json` (122 heads).
- Transfer runs: `results/evaluation/goalscope/Qwen2.5-32B-Instruct/{faithfulness,causality}/{replace/1,additive_norm/0.1}/heads=json:coder32b_goal122__…/`
- Own-heads baselines: same tree with `heads=goal__…`.
- Coder ceiling: `results/evaluation/goalscope/Qwen2.5-Coder-32B-Instruct/{faithfulness,causality}/{replace/1,additive_norm/0.1}/heads=goal__…/`
- Logs: `logs/goalscope/xfer_*.log`, `logs/goalscope/own_*_an.log`.
