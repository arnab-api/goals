# MCTS Search & Contrastive Sampling: Design Report

## Overview

This report documents the MCTS-based exploration and contrastive pair sampling pipeline used to extract goal directions from Code-LLMs. The pipeline builds a tree of possible token continuations for a given code completion prompt, evaluates which branches lead to correct code, and extracts contrastive pairs (correct vs incorrect branches) at various depths for training goal direction probes.

---

## 1. MCTS Search (`MCTSSearch`)

### Core Algorithm

The search builds a tree where each node is a single token. Each iteration follows the standard MCTS cycle:

1. **SELECT**: Traverse from root to a leaf using PUCT scores
2. **EXPAND**: Generate top-k children from the LM's next-token distribution
3. **BACKPROPAGATE**: Update visit counts back to root

### PUCT Selection

```
score = c_puct * prior * sqrt(parent_visits) / (1 + visits)
```

- Higher `prior` (LM confidence) -> higher score
- Lower `visits` (less explored) -> higher score
- **No value signal** — only visit counts and priors. This means selection is entirely driven by the LM's own probability assignments and exploration pressure.
- `c_puct` (default 1.0) controls exploration vs exploitation. Higher values force exploration of low-prior children earlier.

### Node Expansion (`_expand`)

At each leaf, the LM produces next-token logits. The expansion:

1. Applies **temperature** scaling to logits before softmax (`temperature`, default 1.0). Values > 1.0 flatten the distribution, giving more tokens non-negligible probability.
2. Takes top-k tokens (default `top_k=10`) above `prob_threshold` (default 0.01).
3. Checks each token for stop conditions (`</code>`, EOS, newline, max depth) and marks terminal nodes.

### Forced Branching at Shallow Depths

**Problem**: When the model is very confident (e.g., 95% on one token), only 1-2 children survive the `prob_threshold` filter. This collapses the tree into a single dominant branch at early depths, preventing exploration of alternative code paths.

**Solution**: `min_children_up_to_depth` (default 0, disabled) and `min_children` (default 2).

- At depth <= `min_children_up_to_depth`, the `prob_threshold` check is skipped until at least `min_children` children are created. Tokens are still taken in probability order.
- With `depth_first=True`, **all** forced children (not just the highest-prior one) are depth-first expanded into complete code paths. This ensures every forced branch produces executable code that can be evaluated, not just a stub.

**Trade-off**: Each iteration at a shallow depth does N x the work (N = min_children depth-first expansions). But this is the whole point — ensuring enough diversity for contrastive pairs.

### Depth-First vs Breadth-First

- **`depth_first=True`** (recommended): After expanding a node, immediately follows the highest-prior child all the way down to a terminal node. Produces complete code paths early, which is important for evaluation. But can lead to narrow trees if `c_puct` is low.
- **`depth_first=False`**: Standard MCTS breadth-first exploration. Wider trees but incomplete branches until enough iterations accumulate.

### Terminal Conditions

A node becomes terminal when:
- Its token is a stop token (EOS, `</code>`)
- Its token string contains a stop string (newline `\n`)
- It reaches `max_depth`
- No valid children exist above the probability threshold

**Note**: Newline as a stop condition means branches terminate at the first line break. For multi-line code completions, this limits the tree to exploring the first line of the completion.

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_depth` | 15 | Maximum tokens per path. Deeper = more complete code but exponentially larger tree. |
| `top_k` | 5 | Max children per expansion. Higher = wider tree. |
| `prob_threshold` | 0.03 | Minimum token probability. Lower = more children per node. |
| `c_puct` | 1.0 | Exploration constant. Higher = more exploration of low-prior branches. |
| `temperature` | 1.0 | Logit temperature. >1.0 flattens distribution, increasing diversity. |
| `min_children_up_to_depth` | 0 | Force branching up to this depth. 0 = disabled. |
| `min_children` | 2 | How many children to force at shallow depths. |
| `n_iterations` | 1500 | Total MCTS iterations. More = better coverage. |

---

## 2. Tree Evaluation (`evaluate_mcts_tree`)

After MCTS builds the tree, each **leaf node** (terminal or unexpanded) is evaluated by:
1. Reconstructing the full code: `prefix + path_text`
2. Executing it and running unit tests

### Leaf Node Colors

| Color | Verdict | Meaning |
|-------|---------|---------|
| **green** | `PASSED` | Code compiles, runs, and all unit tests pass |
| **orange** | `FAILED` | Code compiles and runs, but one or more unit tests fail |
| **red** | `SYNTAX_ERROR` / `ERROR` | Code doesn't compile or crashes at runtime |
| **None** | (unevaluated) | Non-terminal node that wasn't expanded further |

### Internal Node Colors

Internal nodes inherit colors from children with priority: green > orange > red. This means an internal node is green if *any* descendant path passes all tests.

---

## 3. Contrastive Pair Sampling (`sample_multidepth_pairs`)

### Outcome Probability Propagation

For contrastive sampling, each node gets **outcome probabilities** (`OutcomeProbs`):
- `p_green`: probability mass leading to PASSED outcomes
- `p_red`: probability mass leading to FAILED or ERROR outcomes
- `p_uncertain`: probability mass leading to unevaluated/syntax error nodes

For leaf nodes, these are binary (1.0 for the matching category). For internal nodes, they're computed recursively by **weighting children's outcome probs by their LM prior probability**. This reflects "if the model samples from here according to its own distribution, what's the likely outcome?"

### Positive vs Negative Classification

At each depth, nodes are classified:

- **Positive** (goal-aligned): `p_green >= min_green_confidence` AND `p_red <= max_red_in_positive` AND `p_uncertain <= max_uncertain`
  - Default: `p_green >= 0.5`, `p_red <= 0.2`, `p_uncertain <= 0.5`

- **Negative** (goal-misaligned): `p_red >= min_red_confidence` AND `p_green <= max_green_in_negative` AND `p_uncertain <= max_uncertain`
  - Default: `p_red >= 0.5`, `p_green <= 0.1`, `p_uncertain <= 0.5`

All positive x negative pairs at each depth become contrastive pairs.

### Design Choice: Orange and Red Are Lumped Together

**Both orange (compiles but wrong) and red (doesn't compile) contribute to `p_red`.**

This means contrastive pairs can be:
- **(green, orange)**: correct code vs compiles-but-wrong-answer
- **(green, red)**: correct code vs syntax error/crash
- **(green, mixed)**: correct code vs a node whose subtree has both orange and red leaves

**Implication**: The learned goal direction captures "leads to correct code" vs "leads to any kind of failure." It does **not** distinguish between capability failures (can't write valid syntax) and goal failures (writes valid but incorrect code). This conflation may add noise to the learned direction if the latent representations for "wrong answer" and "syntax error" are different.

**Alternative**: Split into three classes or only use (green, orange) pairs to isolate the "correct vs incorrect but compilable" distinction. This would more cleanly target the goal signal but would reduce the number of available contrastive pairs.

### Filtering: Only Internal Nodes

Contrastive pairs are sampled only from **internal nodes** (nodes with children), not leaves or terminals. This is because:
- At leaf/terminal nodes, the token sequence is complete — there's no "choice point" where the model could have gone either way.
- Internal nodes represent decision points where the model's next-token choice determines whether the completion succeeds or fails.
- The hidden state at an internal node encodes the model's "plan" — the goal direction should be most detectable here.

### Multi-Depth Sampling

Pairs are collected across depths `min_depth` to `max_depth` (default 1-10). Shallower pairs capture higher-level decisions (e.g., choosing an approach), deeper pairs capture finer-grained decisions (e.g., specific operators or values).

---

## 4. Common Failure Modes

### Too Few Contrastive Pairs

**Symptoms**: `sample_multidepth_pairs` returns 0 or very few pairs for a prefill.

**Causes**:
1. **Tree too narrow**: Model is very confident, leading to a single dominant branch. Most nodes have only 1 child.
   - Fix: Increase `temperature`, `c_puct`, or use `min_children_up_to_depth`.
2. **Too many uncertain nodes**: Most leaves have syntax errors (code cut off at max_depth or newline), so `p_uncertain` is high and nodes are excluded.
   - Fix: Increase `max_depth`, or relax `max_uncertain` threshold.
3. **All branches succeed or all fail**: No divergence between green and red outcomes for a given prefill.
   - This is actually informative — it means the prefill strongly determines the outcome.
4. **Strict thresholds**: The default thresholds (`min_green_confidence=0.5`, `min_red_confidence=0.5`) exclude "ambiguous" nodes.
   - Fix: Relax thresholds, but at the cost of noisier pairs.

### Tree Terminates Too Early

**Symptoms**: Most branches are only 1-3 tokens deep.

**Causes**:
1. Newline `\n` in stop strings terminates branches at the first line break.
2. `</code>` appears early in many continuations.
3. `max_depth` is too low.

---

## 5. Pipeline Summary

```
Prompt + Prefill
     |
     v
[MCTSSearch.run()]  -- builds tree of token continuations
     |                  (temperature, top_k, prob_threshold,
     |                   min_children_up_to_depth control diversity)
     v
[evaluate_mcts_tree()]  -- executes code at each leaf, assigns colors
     |                     (green/orange/red/None)
     v
[compute_outcome_probs()]  -- propagates outcome probs up the tree
     |                        (prior-weighted aggregation)
     v
[sample_multidepth_pairs()]  -- extracts (positive, negative) pairs
     |                          at each depth (thresholds filter nodes)
     v
Contrastive Pairs  -- used for latent extraction and direction training
```
