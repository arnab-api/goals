# Pre-commitment swap: is the next operation already decided in the progress latents? (Qwen2.5-Coder-32B-Instruct)

*2026-10-17 · run 2026-10-17 19:03–22:02 (cuda:7) · code `src/diff/precommit_swap.py`, `scripts/diff/precommit_swap_eval.py` ·
notebook `notebooks/precommitment/replace_context.ipynb` · results
`results/diff/Qwen2.5-Coder-32B-Instruct/precommit_swap/` · proposed replacement for the "Pre-commitment Ablation" (Appendix J)*

## TL;DR

We swap the destination's **entire code-prefix context** for the source's, residual stream of every layer, from the end of
the user turn (`<|im_end|>`) up to the frontier variable `{var}`. The prompt then stops at the eq pivot ` =`, which is left clean. If the next
operation were already committed in those latents, the destination should now implement the source's frontier `c`.

* **It almost never does.** Swap alone (front off, n = 128, py→py): the patched run implements **d** (its own frontier) in
  **0.91**, **c** in only **0.06**, neither in 0.02. The GoalScope readout at ` =` moves to c in only **0.06**.
* **The patching logic is sound.** Also swapping the frontier description (SRC's c text → DST's d text; front on) flips the
  model to c in **0.98** and the readout in **1.00** (128/128).
* **The swap reaches the pivot but doesn't decide it.** On every record, positions before the window are bit-identical
  to clean DST and the window is exactly SRC's residuals. The pivot's latent is untouched at layer 0 and changes from
  layer 1 on (mean peak relative change 0.31 with the swap alone, mostly in layers ≥ 48). Yet the pivot still selects d:
  the goal is **re-derived from the task span at the pivot**, and the pre-commitment latents are redundant.

## Setup

Multi-predicate instances, the same sample as the path-patching batch (`PP.sample_multi_instances`: `multi_predicate_setups`,
k = 3, letters naming, `var_eq` pivot, listed presentation, seed 1123 + per-(datatype, direction) hash). 32 per datatype ×
{int, str, list[int], list[str]}, **Python → Python only** (128 instances).

```
p_src: a'b'c -> c     residual donor   (task list a, b, c; a, b prefilled; next = c)
p_dst: a'b'd -> d     the patched run  (task list a, b, d; same prefilled a, b; prompt ends at "{var} =")
```

Everything from `<|im_end|>` (end of the user turn) to the pivot is **token-identical** in the two prompts (asserted):

```
<|im_end|>\n<|im_start|>assistant\n<code>\ndef func(...):\n    a = ...\n    b = ...\n    {var}  |  " ="  <- pivot (never overwritten)
```

**Intervention** (= I2, the *pre-commitment-swap residual transfer*, with `p_src` itself as the donor): over that window
(`<|im_end|> … {var}`, ≈ 45 tokens), every decoder layer's **output** residual of DST is overwritten with SRC's on the
prompt pass. Generated tokens are not patched; they read the KV cache, whose window k/v come from the swapped residuals.
The DST task list (which names d) is untouched.

**Config axis `--src-front`:**

| config | window | role |
|---|---|---|
| **front off** | `<|im_end|> … {var}` | the experiment: swap the progress/pre-commitment latents only |
| **front on** | + the frontier description (SRC's c text → DST's d text; the shorter one `\n`-padded right before it, ≈ 1.9 pad tokens on average; padded prompts used for all three runs) | positive control: DST's task list itself now carries c, so a correct patch must flip to c |

**Why this replaces the context-free ablation (Appendix J).** The old ablation overwrote the pre-implementation pivots
(`prev_end`, `indent`, `var`) with *context-free* latents (the bare token after pad tokens), i.e. off-distribution states. Here
every swapped latent is a real, in-distribution latent of the **same tokens** from a prompt that is committed to a different
next operation. So "no effect" can't be blamed on broken or unnatural states, and an effect would have a clear
direction (towards c).

**Eval.**
- Greedy rollouts (64 tokens) for clean SRC, clean DST and patched DST.
- Causal: oracle `claude-sonnet-4-6` (`judge_causal`, first computation only) labels the patched rollout c / d / both /
  neither. **BOTH counts as a miss.**
- Readout: GoalScope at ` =` (goal-250 heads, additive_norm, α = 0.1, steer layers 35–46), judged source → destination →
  patched (`eval_readouts`). *Valid* means clean SRC → c and clean DST → d. Readout-shift is over all instances.
- Every record also stores `PS.pivot_sanity` (below).

## Sanity checks (every record, `PS.pivot_sanity`)

| check | front off (n = 128) | front on (n = 128) |
|---|---|---|
| positions before the window bit-identical to clean DST | 128/128 | 128/128 |
| swapped window == SRC's residuals | 128/128 | 128/128 |
| pivot ` =` unchanged at layer 0 (layer-0 attention reads embeddings) | 128/128 | 128/128 |
| pivot moves from layer 1 on | 128/128 | 128/128 |
| mean peak relative change ‖h_pat − h_dst‖ / ‖h_dst‖ at the pivot | 0.31 (L56) | 0.95 (L52) |
| pivot next-token top-1 == clean SRC's, among the 92 / 94 where SRC and DST top-1 differ | **14/92** | **94/94** |

Mean relative change of the pivot latent by layer:

| layer | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
|---|---|---|---|---|---|---|---|---|
| front off | 0.00 | 0.01 | 0.01 | 0.02 | 0.05 | 0.08 | 0.22 | 0.31 |
| front on | 0.00 | 0.02 | 0.02 | 0.04 | 0.21 | 0.41 | 0.77 | 0.90 |

So the pivot does read the swapped context. With the swap alone, its latent drifts mainly in the late layers, but its next token
stays DST's in 78 of the 92 cases where the two prompts start their frontier with different tokens.

## Results

| py→py, n = 32 each | c (pre-commitment effect) | d (no-op) | both | neither | readout → c | readout valid |
|---|---|---|---|---|---|---|
| **front off · all (128)** | **0.06** | **0.91** | 0.00 | 0.02 | **0.06** | 0.97 |
| `int` | 0.06 | 0.94 | 0.00 | 0.00 | 0.03 | 0.94 |
| `str` | 0.00 | 0.94 | 0.00 | 0.06 | 0.03 | 0.94 |
| `list[int]` | 0.16 | 0.81 | 0.00 | 0.03 | 0.09 | 1.00 |
| `list[str]` | 0.03 | 0.97 | 0.00 | 0.00 | 0.09 | 1.00 |
| **front on · all (128)** | **0.98** | 0.00 | 0.01 | 0.01 | **1.00** | 0.97 |
| `int` | 0.97 | 0.00 | 0.03 | 0.00 | 1.00 | 0.94 |
| `str` | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.94 |
| `list[int]` | 0.97 | 0.00 | 0.00 | 0.03 | 1.00 | 1.00 |
| `list[str]` | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

Patched readout labels, front off: d 108 · neither 10 · c 8 · both 2. Front on: c 128.

### The few cases that leak (front off)

8/128 rollouts implement c. Readout → c is also 8/128, but mostly **not the same instances**; only 3 have both.

| datatype | c | d | causal | patched readout | patched rollout |
|---|---|---|---|---|---|
| int | absolute_value | is_even | c | both | `abs(n)` |
| int | is_prime | last_digit | c | neither | `(n != 0 and all(n % i != 0 …` |
| list[int] | value_range | count_positive | c | c | `max(nums) - min(nums)` |
| list[int] | count_negative | mean_int | c | neither | `sum(1 for x in nums if x < 0)` |
| list[int] | max_val | contains_zero | c | d | `max(nums)` |
| list[int] | is_sorted | count_negative | c | c | `all(nums[i] <= nums[i + 1] …` |
| list[int] | is_sorted | count_positive | c | c | `all(nums[i] >= nums[i-1] …` |
| list[str] | has_palindrome | distinct_count | c | d | `any(w == w[::-1] for w in words)` |

`list[int]` leaks most (5/32). Two of its leaks are `is_sorted`, whose implementation shape (`all(nums[i] … nums[i+1])`)
differs sharply from the count-style destination frontiers. A few readouts move to c while the rollout stays on d
(e.g. `list[str]` total_length vs count_short).

## Interpretation

* **The next operation is not carried forward in the progress latents.** With the entire assistant-turn prefix swapped to a
  context committed to c, the model still implements d 91 % of the time. Whatever pre-commitment exists at
  `prev_end` / `indent` / `var` is **redundant**: at the eq pivot the goal heads re-read the task span (which still names d)
  and select the pending operation there. This fits the NRGS picture (re-derive, don't carry) and the path-patching result
  that pivot-onwards KV transfer of the task span moves the frontier.
* **This is not a failed patch.** The window carries SRC's exact residuals and the pivot's latent moves, yet it doesn't choose c.
  Swapping just the frontier description as well (≈ 7 more tokens) flips the choice almost perfectly. The decisive information is
  in the task span, not the progress.
* **Relation to path patching.** There, swapping the progress latents for a *third* prompt's (p_res → e) moved pivot-onwards
  KV-transfer success 0.37 → 0.61, and 24 % of patched readouts drifted to e. Here, without any KV transfer and with the task
  span intact, the progress swap alone moves only 6 %. So the progress latents matter *when the task-span read is
  perturbed or weak*, but on their own they don't override a clean task span.

## Caveats / open items

* **Python → Python only.** Cross-language directions (`--lang-transfers py-cpp cpp-py cpp-cpp`) were not run.
* **Readout heads are goal-250** (as in the path-patching runs), not the 122-head faithfulness set of the old Appendix J
  ablation. The steering config is otherwise the same.
* **The pivot is never swapped.** Including ` =` would make the first generated token SRC's by construction (last-layer
  residual), so it was excluded by design.
* **No figure/table for the paper yet.** The numbers above are from the per-cell `meta.json` / `results.json`.

## Reproduce

```bash
# smoke (2 instances x 2 datatypes, separate tag)
conda run -n goals python scripts/diff/precommit_swap_eval.py --cuda 7 --datatypes list[str] int --max-instances 2 --tag-suffix __smoke
# this run: 4 datatypes x 32, py-py, front off + on  (~1.7 min / instance for both configs)
conda run -n goals python scripts/diff/precommit_swap_eval.py --cuda 7
```

Results: `results/diff/Qwen2.5-Coder-32B-Instruct/precommit_swap/I2-src__front-{off,on}__layers-all__ro-goal_250/<dt>__python-python/{results,meta}.json`,
summary `summary__list_str__list_int__int__str__py-py__front-off-on__layers-all.json`. The notebook
`notebooks/precommitment/replace_context.ipynb` calls the same module (`PS.swap_geometry` / `swap_cache` / `swap_patch` /
`pivot_sanity` / `runs`) and oracle (`PP.judge`) for single-instance inspection.

**Speed note.** This run surfaced (and `daf183a` fixed) a slow path in `collect_head_steering_vecs`: it copied the full o_proj
matrix to CPU for every layer on every GoalScope readout (~26 s → ~0.3–4 s per readout; steering vectors equal to 1.6e-7).
