# Pre-commitment ablation: is the frontier's goal encoding at the pre-implementation pivots needed?

*Qwen2.5-Coder-32B-Instruct · 2026-10-08 → 2026-10-09 · script `scripts/evaluation/eval_precommit_ablation.py` · run
on kyoto (GPUs 0, 1) · results `results/evaluation/precommit_ablation/Qwen2.5-Coder-32B-Instruct/ablate=prev_end+indent+var__npad=16__ro=goal122-additive_norm0.1-L35-46__nspec=253__seed=123/`
· design notes `logs/pre-commitment.md`*

## Question

The LM encodes the upcoming sub-goal at the tokens before its implementation starts: `prev_end` (the token
holding the previous line's newline), `indent`, `var` and `eq` (the frontier line `    b =`). If the goal heads
re-read the goal from the task span at `eq` (repeated reading, NRGS), the earlier encodings may be redundant. Test:
remove everything those three tokens know about the context, leave `eq` intact, and check whether the LM still
implements the frontier.

## Setup

- **Corpus:** the locked 253-spec Python corpus (`data/goalscope_eval.json`, the paper's Section-3/4 corpus).
  Frontier = sub-task b / c / d (index j = 1 / 2 / 3: 151 / 73 / 29 specs). The prompt ends at the frontier's
  `    {var} =`, so the pivots are the last four prompt tokens:

  | pivot | tokens |
  |---|---|
  | prev_end | `)\n` (118), `\n` (31), `())\n` (29), `")\n` (19), `()\n` (19), … |
  | indent | 3 spaces |
  | var | ` b` / ` c` / ` d` |
  | eq | ` =` (left intact) |

- **Ablation** (filter paper Fig. 9 / App. B, adapted):
  - for each of `prev_end`, `indent`, `var`, overwrite the output of every decoder layer at that position with the
    latent of the same token read in a context-free prompt: 16 pad tokens followed by the bare token;
  - the pads keep the token off position 0, where it would become the attention sink;
  - bare latents are cached per token id, so each merged `prev_end` variant gets its own;
  - because every layer is overwritten, the keys/values that `eq` and the generated tokens read at those positions
    are context-free too.
- **Measures**, clean vs ablated, per spec:
  1. **Frontier implementation:** greedy decode of the frontier line (≤ 64 tokens), graded by the Section-4 ablation
     judge (`judge_implements`, `ORACLE_SYS_2`: does the code implement the frontier description?).
  2. **GoalScope readout at `eq`:** the paper's faithfulness config (122 goal heads, additive_norm α = 0.1, steer
     layers 35–46), read from the clean / ablated forward and graded with the faithfulness judge (`hard_eval`)
     against the clean frontier implementation.

## Results (n = 253)

| | implementation | readout at `eq` |
|---|---|---|
| clean | .921 | .961 |
| **pre-commitment ablated** | **.921** | **.921** |

- **Implementation is unchanged.** All 253 verdicts are identical between clean and ablated (233 YES→YES, 20
  NO_INFO→NO_INFO). The greedy code is token-identical in 235 specs. The other 18 are rewrites of the same operation:

  | clean | ablated |
  |---|---|
  | `(n % 2 == 0)` | `n % 2 == 0` |
  | `any(nums.count(x) > 1 for x in set(nums))` | `len(nums) != len(set(nums))` |
  | `max(len(w) for w in words) if words else 0` | `max(len(w) for w in words)` |

  The ablation changes how the frontier is written, never which task it implements.
- **The readout at `eq` is mostly unchanged** (.961 → .921; identical text in 64 specs, same verdict in 237):
  - **YES → NO_INFO: 11 specs.** 6 are the probe declining to answer ("Sure, please provide the line of code you'd
    like me to evaluate."), all on int frontiers (is_even, is_multiple_of_3). The rest are vaguer readouts on
    list[str] count frontiers ("Counting variables that meet a specific condition").
  - **YES → NO: 2 specs.**
  - **Not-YES → YES: 3 specs.**

  So the ablation slightly weakens what the goal heads carry at `eq` on some int / list[str] frontiers, but the
  model writes the right task anyway.
- **Per datatype and frontier position:**

  | | n | impl clean / ablated | readout clean / ablated |
  |---|---|---|---|
  | int | 65 | 1.00 / 1.00 | 1.00 / .91 |
  | str | 64 | .92 / .92 | 1.00 / 1.00 |
  | list[int] | 60 | .75 / .75 | .90 / .88 |
  | list[str] | 64 | 1.00 / 1.00 | .94 / .89 |
  | frontier b (j = 1) | 151 | .93 / .93 | .95 / .93 |
  | frontier c (j = 2) | 73 | .90 / .90 | .96 / .89 |
  | frontier d (j = 3) | 29 | .93 / .93 | 1.00 / .93 |

- **The bare latents are in range.** At the middle layer, the bare latent's size relative to the in-context one has
  medians 0.98 / 1.08 / 1.05 (prev_end / indent / var; range 0.87–1.21). Cosines are 0.26 / 0.31 / 0.35 (range
  0.19–0.39). The context is removed without out-of-range activations.

## Controls and sanity checks (Coder-14B smoke, 8 specs)

- **Main condition:** clean and ablated were 8/8 for both implementation and readout. The readout text changed in
  4/8, confirming the hook is active.
- **Positive control: also ablate `eq`.** Implementation falls to 0/8 and the readout to 3/8 (only one genuine
  hit). The implementation collapse is mechanical: overwriting `eq` at the last layer replaces the latent that
  produces the first generated token. The surviving readouts come from goal-head outputs at `eq`, which are
  computed from a context-free query over contextual keys and values before the overwrite. This control shows the
  intervention is potent; it does not isolate the goal signal.

## Interpretation

Removing the pre-commitment at `prev_end`, `indent` and `var` does not change which task the LM implements, or
whether it implements it correctly, on any of the 253 specs. At `eq`, the frontier is still readable from the goal
heads in 92 % of specs (96 % clean). The goal encoding the model computes eagerly at the earlier pivots is therefore
**not necessary**: the `eq` pivot recomputes the frontier on its own, consistent with goal heads re-reading the task
span. This shows necessity only. It does not show that the earlier encodings are unused when present (framing
deferred).

## Caveats

- **The judge underrates some correct clean code.** 15 of the 20 clean failures are `max(nums)`, `min(nums)` (10)
  and `1` (`product_all`, 5; a genuine failure, the accumulator's start value). The judge grades bare
  `max(nums)` / `min(nums)` as NO_INFO ("too generic"). They are identical in both conditions, so they don't affect
  the comparison. The clean rate (.921) matches the paper's ablation baseline (.925) up to one spec.
- **`prev_end` is often merged with the previous line's closing tokens** (`)\n`, `())\n`, …). Its bare latent still
  carries that local syntax, but no task context.
- **Only Python, and only greedy first-line decoding,** as in the Section-4 ablations.
- **No random-token control,** per the design decision.

## Files

- **Script:** `scripts/evaluation/eval_precommit_ablation.py`.
- **Job list:** `job_list/goalscope_eval/precommit_ablation.json`.
- **Results:** `results/evaluation/precommit_ablation/Qwen2.5-Coder-32B-Instruct/…` (copied from kyoto).
- **Smoke results:** `results/evaluation/precommit_ablation/Qwen2.5-Coder-14B-Instruct/` (main + eq positive
  control); logs `logs/precommit_smoke_14b{,_eq}.log`.
