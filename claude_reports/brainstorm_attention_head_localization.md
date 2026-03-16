# Brainstorm: Locating Goal-Mediating Attention Heads via Interchange Intervention

## 1. Why Attention Output, Not Queries

The filter heads paper patches **query states** because they have a specific mechanistic hypothesis: the predicate is encoded as a direction in query space, which creates selective attention over list items. Their intervention is: "if I give this head the query for 'find even numbers', it will attend to even numbers regardless of what list it's looking at."

For instrumental goals in code generation, the mechanism is likely different. The model isn't "attending to items matching a predicate" — it's deciding which subtask to pursue next based on context (function signature, already-written code, the current partial expression). The relevant computation could live in:

- **Query selectivity**: the head looks at different parts of the context depending on the goal
- **Value content**: the head always attends to the same things but extracts different features
- **OV circuit**: the head reads goal-relevant info and writes a goal-direction into the residual

Patching the **attention output** (the vector written to the residual stream) is agnostic to all three — it captures the net effect of whatever the head computes. This is the right choice for a *discovery* experiment. You're not testing a mechanism hypothesis; you're asking "which heads carry goal-relevant information?"

**Concrete target**: Patch the per-head output *before* `o_proj` concatenation. For a model with `H` heads and head dimension `d_h`:

```
head_out[ℓ, h] ∈ R^{d_h}  (output of head h at layer ℓ)
attn_out[ℓ] = O_proj(concat(head_out[ℓ, 0], ..., head_out[ℓ, H-1]))
```

Patching at the `head_out[ℓ, h]` level gives you head-level resolution. You already have `attn_module_name_format` and the head dimension extraction in `get_hs()` (lines 983-1044 of `functional.py`), so the infrastructure is mostly there.

**One caveat**: patching the full attention output at head granularity is clean. But if you later want to decompose *why* a head matters (is it the attention pattern or the value content?), you'd need separate Q/K/V patching as a follow-up. The output-level experiment comes first.


## 2. The Multi-Token Evaluation Problem

This is the central challenge. The filter paper has a beautiful setup: single-token output, clear target. Your setup requires generating 10-50 tokens that collectively implement a predicate. How do you define "success" for the interchange intervention?

### 2a. What exactly should the patching flip?

The interchange setup:
- **Source run**: prompt that naturally leads to implementing `is_palindrome` first (e.g., with `aligned_p_plus` prefill like `s == s`)
- **Destination run**: prompt that naturally leads to implementing `is_upper` first (e.g., with `aligned_p_minus` prefill like `s.is`)
- **Patched run**: destination prompt, but attention outputs from source injected at selected heads

**Success = the patched run starts implementing the source predicate instead of the destination predicate.**

But how do you measure this? Several options, from cheapest to most expensive:

### Option A: First-token logit shift (cheapest, most differentiable)

At the pivot point (right after `return `), look at the very first generated token. Measure whether patching shifts the distribution toward tokens that are consistent with `is_palindrome` implementations.

**Problem**: The first token is often ambiguous. Both `is_palindrome` and `is_upper` implementations might start with `s` or `not` or `(`. The discriminative tokens (anchor tokens like `[::-1]` or `.isupper()`) appear at position 2-5, not position 1.

**Verdict**: Too noisy as a sole signal. But could be a useful auxiliary.

### Option B: Next-token distribution matching (KL divergence) — RECOMMENDED STARTING POINT

This is the key insight: **you don't need anchor tokens or unit tests if you have the clean source distribution as a reference.**

1. Run the source prompt cleanly → get next-token distribution `P_src` at the pivot position
2. Run the destination prompt cleanly → get `P_dest`
3. Run the patched destination prompt → get `P_patched`

Measure: does `KL(P_patched || P_src) < KL(P_patched || P_dest)`? Or more precisely, train the mask to minimize:

```
L_causal = KL(P_patched || P_src)  # make patched look like source
```

This is:
- **Fully differentiable** (softmax outputs, KL is smooth)
- **Single forward pass** (no generation needed)
- **Anchor-free** (doesn't require knowing what tokens to look for)
- **Works at the pivot point** (where the goal decision is made)

The intuition: if the source run is "about to write palindrome-checking code", its next-token distribution will favor tokens like `s`, `(`, `reversed`, etc. in a specific mixture. If patching makes the destination look like the source, then the patched heads carry the goal signal.

**Potential issue**: KL at a single position might not capture enough. You could extend to a small window (positions pivot, pivot+1, pivot+2) by doing teacher-forced forward passes with the source's token choices. But starting with just the pivot position is cleanest.

### Option C: Anchor token logit boost

Measure whether patching increases the logit of anchor tokens for the source predicate and/or decreases logits of anchor tokens for the destination predicate.

```
L_anchor = -sum(log P_patched(a) for a in anchor_tokens_src)
           + sum(log P_patched(a) for a in anchor_tokens_dest)
```

**Advantage**: Very interpretable. Directly measures "does the model now want to write palindrome-related code?"

**Problems**:
- Anchor tokens may not be valid *first* tokens at the pivot. `[::-1]` is an anchor for palindrome, but it's never the first token after `return `.
- You'd need to identify which anchor tokens are first-position-valid. This is doable (filter anchors by whether they can appear as first token given the prefix) but adds complexity.
- Some anchor tokens are multi-token strings. You'd need to tokenize them and use only the first token, or do multi-step teacher-forcing.

**Verdict**: Useful as a *secondary metric* or sanity check, not as the primary training objective. The KL approach subsumes this — if patching makes `P_patched ≈ P_src`, then anchor token logits will naturally shift.

### Option D: Full generation + unit tests (RLVR-style)

Generate the full completion from the patched model, run unit tests for `is_palindrome`, check if it passes.

**The appeal**: Ground truth evaluation. No proxy, no approximation.

**The problems are severe**:
1. **Non-differentiable**: generation is discrete sampling. You'd need REINFORCE/PPO to get gradients through the mask.
2. **Extremely sparse reward**: binary pass/fail, with most random perturbations causing failure.
3. **Expensive**: each mask update requires full generation (50+ tokens) + code execution. With ~1000 heads and gradient estimation, this is orders of magnitude slower.
4. **Credit assignment nightmare**: even if you get a gradient signal, it's unclear whether the mask update reflects "this head carries the goal signal" vs "this head is generally important for code generation quality."

**How hard to implement?** Actually not that bad mechanistically — you already have `generate_with_patch()` and `evaluate_single_code()`. The loop would be:
```python
for step in range(num_steps):
    # Forward: generate with current mask
    patched_code = generate_with_patch(mt, prompt, patches_from_mask)
    # Evaluate
    reward = float(evaluate_single_code(patched_code, unit_tests).verdict == "green")
    # REINFORCE gradient
    log_probs = mask_distribution.log_prob(sampled_mask)
    loss = -reward * log_probs
    loss.backward()
    optimizer.step()
```

But the variance would be enormous. You'd need hundreds of rollouts per gradient step, and with your limited data, overfitting is guaranteed.

**Verdict**: Don't use this for mask training. But absolutely use it as a **validation metric** — after you've found the heads via KL-based training, verify that patching those heads actually flips the generated code.


## 3. Training the Head Mask

### Architecture

Following the filter paper, use a continuous relaxation of binary masks:

```python
# Learnable parameters: one scalar per head
logits = nn.Parameter(torch.zeros(n_layers, n_heads))

# During training: soft mask via sigmoid with temperature
mask = torch.sigmoid(logits / temperature)  # temperature → 0 for sharper masks

# mask[ℓ, h] ∈ (0, 1): interpolation weight
patched_head_out[ℓ, h] = mask[ℓ, h] * src_head_out[ℓ, h] + (1 - mask[ℓ, h]) * dest_head_out[ℓ, h]
```

### Loss function

```
L = L_causal + λ * L_sparsity

L_causal = KL(P_patched || P_src)    # push patched distribution toward source
L_sparsity = sum(sigmoid(logits))     # L1 on mask values, encourages sparsity
```

**λ tuning**: Start with λ that keeps ~10% of heads active, then increase λ gradually (curriculum) to find the minimal set. If L_causal degrades sharply at some sparsity level, that's the information bottleneck — the remaining heads are the critical ones.

### Training loop

```python
for src_prompt, dest_prompt in interchange_pairs:
    # 1. Clean forward passes (cached, no grad)
    with torch.no_grad():
        src_head_outs = get_all_head_outputs(mt, src_prompt)   # dict[(ℓ,h)] → tensor
        dest_head_outs = get_all_head_outputs(mt, dest_prompt)
        P_src = get_next_token_dist(mt, src_prompt)  # target distribution

    # 2. Patched forward pass (grad flows through mask only)
    mask = torch.sigmoid(logits / temperature)
    # Construct patched activations and run forward
    P_patched = forward_with_masked_heads(mt, dest_prompt, src_head_outs, dest_head_outs, mask)

    # 3. Loss
    loss = kl_div(P_patched, P_src) + lambda_ * mask.sum()
    loss.backward()
    optimizer.step()
```

**Key detail**: gradients flow through the mask parameters only, not through the model weights. The model is frozen. This makes the optimization very lightweight — you're learning ~1000 scalars (n_layers × n_heads), not finetuning.

### Position of intervention

The filter paper patches at the **final token position** (the position where the model outputs its answer). For your setup, the pivot position is the last token of the prompt (right before generation starts — i.e., the last token of `return `). This is where the model has processed all context and is about to commit to a goal.

But there's a subtlety: **should you patch at the same position across all layers?** In a causal transformer, the residual at position `t` at layer `ℓ` can only attend to positions `≤ t`. So the last position is the only one that has seen everything. Patching earlier positions would be an interesting follow-up (does the goal signal build up gradually across positions?), but start with last-position-only.

### What about patching at multiple positions?

In autoregressive generation, you could patch at every step (like your current `on_pass="all"` in `PatchSpec`). But for the mask-training objective, you only need the **first** generation step — the next-token distribution at the pivot. This is much cleaner because:
- Single forward pass, fully differentiable
- No autoregressive loop during training
- The goal decision is made at the pivot, not later


## 4. Data Scarcity

### The problem

The filter paper has effectively unlimited data: sample random lists, random predicates, and you get a new training instance. Your multi-predicate dataset has ~6 files × ~4 samples each ≈ 24 samples. Even with multiple prefills per sample, you're looking at ~100 interchange pairs total.

### Is this actually fatal?

**Maybe not.** Consider what you're learning: a binary mask over ~1000 heads (48 layers × 32 heads for Qwen-14B, roughly). The mask has ~1000 parameters. With 100 training pairs and strong sparsity pressure, this might be identifiable — especially if the causal heads are consistent across samples.

Think of it as feature selection, not neural network training. You're finding 5-10 heads out of 1000. Even with 30 samples, if the same heads light up consistently, you have a robust result.

### But robustness concerns are real

With small data, you risk:
1. **Overfitting to surface features**: the mask might find heads that distinguish these specific prompts, not heads that carry the general goal signal
2. **Sample-specific heads**: different samples might use different heads, and with few samples, you can't separate "this head is generally causal" from "this head matters for palindrome specifically"

### Augmentation strategies

**Strategy 1: Cross-prefill interchange (free, already available)**

Each sample has `aligned_p_plus`, `aligned_p_minus`, `adversarial`, and `inconsequential` prefills. Form interchange pairs:
- Source: sample with `aligned_p_plus` prefill (nudges toward P+)
- Destination: same sample with `aligned_p_minus` prefill (nudges toward P-)

This gives you multiple pairs per sample without needing new functions. And it's a clean test: the function is identical, only the prefill differs, so any head that flips the goal must be responding to the prefill context.

**Strategy 2: Cross-sample interchange (moderate effort)**

Pair different samples that share the same P+ predicate (e.g., both involve `is_palindrome` but in different functions). Source: sample_1 with `is_palindrome` prefill. Destination: sample_2 with `is_upper` prefill. The mask must find heads that carry the goal signal *across* different function contexts.

This is the real generalization test and you have the data for it — 4 samples per predicate file.

**Strategy 3: LLM-generated functions (higher effort, higher payoff)**

Use Claude to generate additional multi-predicate functions. You already have the pipeline for this (the data files were likely generated or curated with LLM help). Generate 20-30 more functions with known predicate pairs. This would bring you to ~150 samples, which is comfortable for mask learning.

**Risk**: generated functions might be too similar (same patterns, same variable names). Mitigate by specifying diverse domains (string processing, math, data validation, list operations).

**Strategy 4: Synthetic perturbations (cheap but limited)**

For each function:
- Variable renaming: `s` → `text`, `n` → `number`
- Docstring variation: rephrase the docstring
- Type hint variation: add/remove type hints
- Argument order: swap argument positions in the signature

These create surface-level diversity while preserving the semantic structure. Useful for preventing the mask from latching onto token-level features, but doesn't add genuinely new goal contexts.

### Recommended data strategy

1. Start with **cross-prefill interchange** within samples (free, ~100 pairs)
2. Add **cross-sample interchange** for shared predicates (~50 more pairs)
3. If results are noisy, generate **20-30 new functions** via Claude
4. Use surface perturbations for regularization, not as primary data


## 5. Critiques and Potential Issues

### 5a. The KL objective might be too easy to satisfy

If the source and destination distributions are very similar at the pivot (both might put high probability on `s`, `not`, `(`, etc.), then the mask could learn to patch nothing and still get low KL. This would happen if the goal decision doesn't actually manifest strongly in the next-token distribution at the pivot point.

**Mitigation**: Use prompts/prefills where the distributions *do* diverge clearly. The `aligned_p_plus` and `aligned_p_minus` prefills are designed for this — they push the model toward different first tokens. Verify empirically that `KL(P_src || P_dest) >> 0` before training.

If the KL between source and destination is already small, you have a bigger problem: the goal decision happens *later* than you think, or the model is genuinely uncertain at the pivot. In that case, you'd need to extend the objective to multi-step KL (teacher-forced over 2-3 tokens).

### 5b. Interchange direction asymmetry

Patching source→destination and destination→source might not be symmetric. Head `h` might be needed to flip from palindrome to upper but not vice versa.

**Solution**: Train the mask bidirectionally — include both `(src, dest)` and `(dest, src)` pairs. Heads that appear in both directions are the most robust.

### 5c. Head interactions / distributed representations

The goal signal might not be localized in a sparse set of heads. It might be distributed across many heads, each contributing a small piece. If so, the sparsity pressure will find *some* subset that works, but it won't be the "true" set (there is no true set — it's distributed).

**Diagnostic**: Plot the L_causal vs L_sparsity Pareto frontier. If there's a sharp elbow (5 heads get 90% of the effect), the signal is localized. If it's a gradual slope (you need 50+ heads for any effect), it's distributed. Both outcomes are interesting findings.

### 5d. Generation-time vs pivot-time heads

Heads that matter at the pivot (for the goal *decision*) might differ from heads that matter during generation (for *executing* the goal). The KL objective only finds pivot-time heads. To find execution-time heads, you'd need to patch during autoregressive generation and evaluate with unit tests — back to the RLVR problem.

**This is actually fine.** You're studying goal *representation* and *decision*, not execution. Finding the heads that determine "I will now implement palindrome checking" is exactly the research question. What happens downstream during execution is a separate (also interesting) question.

### 5e. Confound: attention heads that encode the prefill, not the goal

If the source uses prefill `s == s` and the destination uses `len(`, patching attention heads might just transfer "I saw `s == s` in the prefix" rather than "I intend to check palindrome." The head might be a simple copy head, not a goal head.

**This is the most important confound.** Mitigations:
1. Include `inconsequential` prefills (empty string) in training pairs — the goal direction must work even without a discriminative prefill
2. Cross-sample pairs (different functions, same predicate) test whether the signal generalizes beyond specific prefill tokens
3. After finding heads, inspect their attention patterns: are they attending to the prefill tokens specifically, or to broader context (function name, docstring, argument names)?


## 6. Concrete Experimental Protocol

### Phase 1: Validate the interchange setup

Before training any masks:
1. Pick one sample from `is_palindrome.json`
2. Get `P_src` (with `aligned_p_plus` prefill, e.g., `s == s`) and `P_dest` (with `aligned_p_minus` prefill, e.g., `len(`)
3. Verify that `KL(P_src || P_dest)` is substantial (> 0.5 nats)
4. Do brute-force single-head patching for a few candidate layers (e.g., layers 20, 30, 40) to see if any individual head shifts `P_dest` toward `P_src`

This sanity check tells you whether the interchange setup is viable before investing in the full mask optimization.

### Phase 2: Train the learned mask

1. Collect interchange pairs: cross-prefill + cross-sample
2. Cache all head outputs for source and destination runs (expensive but one-time)
3. Train mask with KL + sparsity objective
4. Anneal temperature and λ to find the minimal head set
5. Plot the Pareto frontier (number of active heads vs KL recovery)

### Phase 3: Validate with generation

1. Take the top-k heads identified by the mask
2. Do full interchange: run destination prompt but patch those k heads from source
3. Generate full completion
4. Run unit tests for the source predicate
5. Compare pass rates: no patching vs patching top-k vs patching random-k vs patching all heads

### Phase 4: Interpret the heads

1. For the identified goal heads: what layers are they in? Early (context parsing)? Middle (goal formation)? Late (goal execution)?
2. Attention pattern analysis: what are they attending to when the goal signal is being transmitted?
3. Cross-predicate test: do the same heads matter for `is_sorted` as for `is_palindrome`?


## 7. Implementation Notes

### Getting per-head outputs in nnsight

The key operation is intercepting the attention output before `o_proj` concatenation. In most architectures:

```python
# After the attention computation, before o_proj:
# attn_output shape: (batch, seq_len, num_heads, head_dim)
# or after reshape: (batch, seq_len, hidden_dim)

# To get per-head: intercept before o_proj
attn_module = model.layers[ℓ].self_attn
# In Qwen/LLaMA: attn_output = self.o_proj(attn_output)
# We want the input to o_proj, split by head dimension
```

The `get_hs()` function in `functional.py` already does something like this (lines 983-1044) by accessing `o_proj.input` and slicing by head dimension. This can be reused.

### Gradient flow through frozen model

The model is frozen — only mask parameters have `requires_grad=True`. But you still need the forward pass to be differentiable w.r.t. the mask. This means:
- The interpolation `mask[ℓ,h] * src + (1 - mask[ℓ,h]) * dest` must happen inside the computation graph
- You can't just swap tensors; you need to use hooks or nnsight proxies that maintain the graph

With nnsight, this should work via proxy arithmetic inside the trace context. With baukit, you'd use `TraceDict` hooks that perform the interpolation.

### Memory considerations

Caching all head outputs for all layers/heads is expensive:
- Per sample: 48 layers × 32 heads × 1 position × head_dim(128) × 2 bytes ≈ 400KB
- For 200 samples × 2 (src + dest) ≈ 160MB — totally fine

The bottleneck is the patched forward pass during training, which requires one full forward pass per training step. But since you're not doing generation (just next-token logits), this is a single forward pass — fast on A100.
