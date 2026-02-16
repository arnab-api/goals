# PCA Visualization Recommendations for Goal Directions

## Problem

Standard PCA on contrastive latents (pos vs neg) is very noisy. PCA finds directions of **maximum variance**, which has no reason to align with the pass/fail distinction. The goal direction might explain 0.1% of total variance (dominated by token identity, position, prefill type, depth) and PCA will miss it entirely.

## Hypothesis Issues

**The core hypothesis**: pass branches have stronger goal signal, fail branches have weaker/absent goal signal.

This is reasonable but **incomplete** due to two confounds:

### 1. Capability vs. Goal Failure

A fail branch might have a *strong* goal signal but the model is "stuck" — the prefill committed it to a syntactic structure that can't express the correct solution. E.g., prefill `"1 +"` on `is_odd` — the model might *know* it needs to check oddness (goal intact) but can't find a valid completion starting with `1 +`.

**Validation approaches:**
- Probe goal direction at the **prefill position** (before any generation), not just at the end of the path. If goal signal is strong *before* generation but the branch still fails → capability failure, not goal loss.
- Compare goal signal strength at **depth 1 vs depth 10** within fail branches. If it starts strong and decays → goal overwritten by local context. If weak from the start → prefill disrupted the goal.
- Cross-reference with existing eval data: adversarial prefills that *some* models handle but others don't are likely capability failures. Universal-fail prefills ("breaking prefills") might be genuine goal disruption.

### 2. Negative Latents Are Not Homogeneous

Some fail branches are "almost right" (off-by-one, wrong operator) while others are "completely lost" (tautologies like `1+1==2`). These likely have very different latent signatures:
- "Almost right" failures probably *do* have the goal signal → capability failures
- Tautologies probably don't → goal failures

**Suggestion:** Stratify negative latents by failure severity. Split negatives into "near-miss" (partially correct logic) vs "total-miss" (tautology, syntax error, task-irrelevant). Check if the goal direction separates {positives + near-misses} from {total-misses}.

---

## Better Visualization Methods

### Option 1: Fisher's Linear Discriminant (LDA)

Find the projection maximizing between-class / within-class variance. One component only (2 classes), but combine with PCA on the residual for a 2D view: "goal axis" + "next most variable non-goal axis."

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(all_z, labels)  # labels: 1=pos, 0=neg
```

### Option 2: Contrastive PCA (cPCA)

[Abid et al., 2018](https://arxiv.org/abs/1709.06716). Finds directions with high variance in positives relative to negatives. Filters out shared variance (token identity, depth, etc.).

```python
# contrastive PCA: C_pos - alpha * C_neg
# eigendecompose to get directions enriched in positive-specific variance
```

### Option 3: Project onto Learned Contrastive Direction + Residual PCA (Recommended First)

Use the already-learned `contrastive_directions[layer_idx]` as one axis. Project all latents onto: (a) the contrastive goal direction, and (b) the top PCA component of the residual (after projecting out the goal direction).

```python
v = contrastive_directions[layer_idx].cpu().numpy()
proj_goal = all_z @ v  # scalar projection onto goal direction
residual = all_z - np.outer(proj_goal, v)
pca_residual = PCA(n_components=1).fit_transform(residual)
# 2D plot: x=goal projection, y=top residual PC
```

**Why this is best first**: directly answers whether the contrastive objective found something real. The residual axis shows what else is going on.

### Option 4: UMAP

Preserves local neighborhood structure, can reveal non-linear clusters. But:
- Non-linear → can't extract a "direction" (no steering interpretation)
- Hyperparameter-sensitive (`n_neighbors`, `min_dist`)
- Can create artificial clusters or merge real ones

Use as a **sanity check** after linear methods, not primary tool. If UMAP shows separation but linear methods don't → goal encoding might be non-linear.

### Option 5: Mean-Shifted PCA (PCA on Difference Vectors)

PCA on the per-pair difference vectors `z_pos - z_neg`. Removes shared component and shows structure of the *difference* itself.

```python
delta_z = z_pos - z_neg  # (n_pairs, hidden_dim)
pca_delta = PCA(n_components=2).fit_transform(delta_z)
```

---

## Recommended Implementation Order

1. **Option 3** (project onto contrastive direction + residual PCA) — most interpretable, connects to steering
2. **Option 5** (PCA on difference vectors) — shows structure of what separates pos from neg
3. **Option 1** (LDA) — if option 3 is noisy, LDA finds the best linear separator
4. **Option 4** (UMAP) — sanity check for non-linear structure
5. **Option 2** (cPCA) — if you need to disentangle shared vs class-specific variance
