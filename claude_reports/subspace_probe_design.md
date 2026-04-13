# Intent Subspace Extraction: Objective, Plan, and Design Decisions

## 1. Motivation

Track 2 identified attention heads whose cross-language interchange transfers goal information (C++ → Python). The approach is effective but has two limitations:

1. **Discrete bottleneck**: heads are binary (selected / not selected). There is no continuous representation of how much goal information each head carries, and the selection threshold is a free parameter.
2. **Restricted to attention**: the head mask searches only within $o_\text{proj}$ inputs. Goal-relevant information may be distributed across the full residual stream, including MLP contributions.

The subspace probe is a generalization: instead of finding *which heads* carry intent, we learn *which directions in the residual stream* encode it. The result is a linear subspace per layer that can be read out, patched, and—in principle—steered, without any architectural assumption about where the information lives.

---

## 2. Research Objective

Learn a set of per-layer projection matrices

$$W_l \in \mathbb{R}^{r \times d}, \quad l \in \mathcal{L}$$

such that $W_l r_l$ captures the LM's *instrumental goal* at that token position — the sub-task it intends to execute next — independently of the specific implementation strategy it will use.

**Training signal**: given a token position in a code generation where the model is "about to implement goal G", and a frozen destination LM with a learned soft-prompt context, patching the subspace representation into the destination should cause it to verbalize G correctly.

**Claim if it works**: the $r$-dimensional subspace is a universal intent encoding — it fires for "about to check the sign condition" whether the implementation will be `< 0`, `<= -1`, or `& (1 << 31)`.

---

## 3. Method

### 3.1 Subspace and patching

For a residual $r_l \in \mathbb{R}^d$ at layer $l$, the subspace representation is:
$$s_l = W_l r_l \in \mathbb{R}^r$$

Patching the subspace of a destination run with a source representation:

$$r_l^{\text{patched}} = r_l^{\text{dest}} + W_l^T W_l \bigl(h_l^{\text{src}} - r_l^{\text{dest}}\bigr)$$

When rows of $W_l$ are orthonormal ($W_l W_l^T = I_r$), this is equivalent to projection-replacement: the subspace component of $r_l^{\text{dest}}$ is discarded and replaced with the subspace component of $h_l^{\text{src}}$, while everything outside the subspace is preserved:

$$r_l^{\text{patched}} = \underbrace{(I - W_l^T W_l)}_{\text{dest, complement}} r_l^{\text{dest}} + \underbrace{W_l^T W_l}_{\text{projection}} h_l^{\text{src}}$$

The additive form is used in practice because it makes gradient flow transparent and matches the existing patchscope infrastructure.

### 3.2 Soft prompt

The destination context is a learned soft prompt rather than a fixed natural-language question:

```
<|im_start|>system
You are a helpful assistant proficient in programming.<|im_end|>
<|im_start|>user
[e_1] [e_2] [e_3] [e_4]<|im_end|>
<|im_start|>assistant
```

where $e_1, \ldots, e_k \in \mathbb{R}^d$ are trainable embedding vectors. The soft tokens sit in the user turn (not the assistant turn) so that the verbalization is generated without a fixed context-biasing prefix. This choice keeps the verbalization more "natural" — the model is gently nudged to interpret the patched subspace, not told what to say.

### 3.3 Training objective

**Source positions**: the *first token* of each labeled intent block and the *immediately preceding token*. The preceding token is the pivot — the moment before commitment — mirroring the Track 2 pivot design. The first token of the block is the moment of maximum intent signal (the model has just committed). Using both gives two training instances per block.

**Teacher-forced forward pass**: the destination input is $[\text{prefix}][\text{label tokens}]$ in a single forward pass. The patch is applied at every position from the last prefix token onward (positions $p_0, p_0+1, \ldots$), which is equivalent to `on_pass="all"` in the autoregressive patchscope — at each generation step, the same source residual is injected at the current last position.

**Loss**:
$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}(\text{verbalizations}, \text{labels})}_{\text{main}} + \lambda_{\text{ortho}} \underbrace{\sum_l \|W_l W_l^T - I_r\|_F^2}_{\text{orthogonality}} + \underbrace{\text{weight decay}}_{\text{on } W_l, e_i}$$

**Orthogonality enforcement**: two-tier.
- *Soft*: the $\|W_l W_l^T - I_r\|_F^2$ loss term provides gradient pressure.
- *Hard*: after each optimizer step, each $W_l$ is projected onto the Stiefel manifold via QR decomposition (`reorthogonalize()`), analogous to `HeadMask.clamp_()`. This guarantees that the projection semantics are always well-defined, not just asymptotically.

---

## 4. Design Decisions

### D1 — Orthogonality loss: $\|W_l W_l^T - I_r\|_F^2$, not $\|W_l^T W_l - I_d\|_F^2$

$W_l^T W_l \in \mathbb{R}^{d \times d}$ has rank at most $r < d$, so $\|W_l^T W_l - I_d\|_F^2$ can never reach zero — it is a divergent loss. The correct constraint for orthonormal rows is $W_l W_l^T = I_r$, which has a well-defined minimum at zero. The two-tier enforcement (soft loss + hard QR) ensures this is satisfied throughout training, not just at convergence.

### D2 — Patch at all generated positions (mirrors `on_pass="all"`)

In autoregressive generation with `on_pass="all"`, the same source residual is injected at position $-1$ at every generation step. In teacher-forced training, the equivalent is to patch all positions $\geq p_0$ (last prefix token) in a single forward pass. The causal mask ensures correctness: for each position $t \geq p_0$, only residuals at positions $\leq t$ are visible, so simultaneous patching gives the same result as sequential step-by-step patching. This equivalence allows standard teacher-forced CE loss.

### D3 — Soft prompt in the user turn, not the assistant turn

Placing soft tokens at the start of the assistant turn would directly prime the first generated word (strong pressure toward a specific output format). Placing them in the user turn gives weaker, more distributed conditioning — the model must "interpret" the patched subspace and express it in its own words. This keeps the verbalization closer to introspection than to templated output.

### D4 — Source positions: first token of block + preceding token

The first token of an intent block is where the model has just committed to a sub-goal. The preceding token is the pivot — where it is *about to* commit. These are the two moments where the intent representation should be sharpest and most separable from the implementation. Using both doubles training instances cheaply and gives the model two views of the same intent.

### D5 — Source residuals cached once, no gradient

The source residuals $h_l^{\text{src}}$ are extracted from a frozen LM in a single forward pass per task and stored on disk. They do not participate in the computation graph. Gradients flow only through $W_l$ (via $W_l^T W_l (h_l^{\text{src}} - r_l^{\text{dest}})$, differentiable in $W_l$) and through the soft token embeddings $e_i$ (via the destination forward pass). This makes training efficient: only one forward pass + backward per instance per step, with no re-computation of source residuals.

### D6 — Per-layer independent subspaces

Each layer $l \in \mathcal{L}$ has its own $W_l$. This allows the subspace to be layer-specific, reflecting that goal information may be encoded differently at different depths. Sharing $W$ across layers would be more parameter-efficient but would force the subspace to be layer-invariant — a stronger assumption that is not warranted at this stage.

### D7 — $r = 256$, layers 35–45

These are initial choices motivated by the Track 2 results (the head mask concentrated on layers 35–45). $r = 256 \ll d = 8192$ enforces a meaningful bottleneck ($3\%$ of the residual dimension). Both are hyperparameters to tune if the proof-of-concept succeeds.

### D8 — Task-level train/test split

Train and test sets are split by task, not by instance. This tests generalization to unseen functions — the harder and more relevant criterion. Approximately 20 tasks are held out. Instance-level splitting would be misleading since instances from the same task (same code, different blocks) are correlated.

---

## 5. Training Data

- **Source**: `results/well-known-tasks/labels/<model>/<oracle>/` — labeled IntentBlock trees
- **Volume**: ~134 tasks × ~10 labeled blocks/task × 2 positions/block ≈ **~2,600 training instances** (before task split)
- **Test set**: ~20 tasks held out by task name (random seed fixed)
- **Augmentation**: none for the proof-of-concept; additional labeled generations can be added later

The labels are abstract descriptions like "strip leading and trailing whitespace" or "handle the edge case where the input is empty" — they do not name variables, following the annotation guidelines enforced during labeling.

---

## 6. Implementation

| File | Role |
|---|---|
| `src/goals/subspace_probe.py` | `SubspaceProbe` module, caching, pivot finding, patched forward pass |
| `scripts/cache_subspace_data.py` | One-time residual extraction → `cache.pt` |
| `scripts/train_subspace_probe.py` | Training loop, checkpointing |
| `notebooks/11_goalscope_subspace.ipynb` | Step-by-step verification before full run |

**Run order**:
```bash
# 1. Cache source residuals (GPU, ~1 forward pass per task)
python scripts/cache_subspace_data.py --model-key Qwen/Qwen2.5-Coder-32B-Instruct

# 2. Train probe
python scripts/train_subspace_probe.py \
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \
    --layer-range 35 46 --r 256 --n-soft-tokens 4 \
    --lambda-ortho 0.1 --lr 3e-4 --epochs 50
```

---

## 7. Evaluation Plan

**Quantitative**: validation CE loss on held-out tasks during training; should decrease relative to an untrained probe baseline and relative to a zero-source-residual ablation (no patching).

**Qualitative**: for each held-out task, generate verbalizations for several intent blocks and compare to ground-truth labels. Look for:
- Correct identification of the abstract sub-goal (e.g., "handle the carry" not "add the digits")
- Absence of variable names (per annotation Rule 6)
- Consistency across different token positions within the same block

**Ablations** (after proof-of-concept):
1. Random $W_l$ (no training) — tests whether patching alone does anything without learning
2. Identity patch ($r = d$, no compression) — upper bound; tests whether the bottleneck is necessary
3. Single layer vs multi-layer — which layers contribute most
4. Preceding token only vs first token only vs both — tests whether token position matters

---

## 8. Open Questions

- **Does $r = 256$ provide enough capacity?** If val CE plateaus high, try $r = 512$ or $r = 1024$.
- **Is the soft prompt sufficient?** If the model consistently ignores the patched subspace, consider adding a small learned linear head on top of the destination's last hidden state to extract the verbalization, bypassing the need for natural-language generation.
- **Cross-model transfer?** A direction trained on Qwen-32B should ideally transfer to Llama-70B (same sub-goal, different model). This would be strong evidence for universality.
- **Causal test**: after training, does steering along $W_l^T s$ in the source run (instead of patching) shift the generated code toward the intended sub-goal? This would connect the probe back to Track 1's causality question.
