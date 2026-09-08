# Residual-goal probe — the math (for review)

*Companion to the `residual-goal probe` cell in `notebooks/diff/6_disent_from_attn_out.ipynb`.
Question: does the residual stream `h` after an arbitrary layer already **contain the goal**, in the
subspace the goal heads' queries read from?*

## Notation

- $d = d_\text{model}$ (5120), $d_h = d_\text{head}$ (128).
- $h \in \mathbb{R}^{d}$ — the residual stream **after decoder layer $L$**, at the pivot token.
- Goal head $k=(\ell,j)$: query weight $W_Q^{(\ell,j)} \in \mathbb{R}^{d_h \times d}$ (head $j$'s row-slice of layer $\ell$'s `q_proj`).
- $g^{(\ell)} \in \mathbb{R}^{d}$ — layer $\ell$'s pre-attention RMSNorm gain (`input_layernorm.weight`; Qwen2 uses `weight *`, no `1+`).

## 1. The head's query transform

For any residual $x$, the head computes its query on the **normalized** residual:
$$
q \;=\; W_Q\,\mathrm{RMSNorm}_\ell(x) \;=\; W_Q\Big(\tfrac{1}{\mathrm{rms}(x)}\,g^{(\ell)}\odot x\Big).
$$
Fold the per-dimension gain into an **effective read matrix**
$$
Q \;:=\; W_Q\,\mathrm{diag}\!\big(g^{(\ell)}\big) \;\in\; \mathbb{R}^{d_h\times d},
\qquad\Longrightarrow\qquad
q \;=\; \frac{1}{\mathrm{rms}(x)}\,Q\,x .
$$
The factor $1/\mathrm{rms}(x)$ is a positive **scalar** (direction-preserving).

## 2. What the query reads = projection onto $\mathrm{rowspace}(Q)$

$q$ depends on $x$ **only** through $Qx$; any component of $x$ in $\ker(Q)$ is invisible to the query.
The part of $h$ the query "sees" is its orthogonal projection onto $\mathrm{rowspace}(Q)$:
$$
\boxed{\,h_q \;=\; P\,h \;=\; Q^{+}\!\big(Q\,h\big)\,},
\qquad
P \;:=\; Q^{+}Q .
$$
With $Q$ full row rank, $Q^{+} = Q^{\top}(QQ^{\top})^{-1}$, so
$$
P \;=\; Q^{\top}\big(QQ^{\top}\big)^{-1}Q .
$$

**Properties** (verified numerically): $P^2=P$, $P^{\top}=P$, $\operatorname{rank}(P)=d_h$, and
$Q(Ph)=Qh$ — the projection reproduces the same query, i.e. $Ph$ keeps exactly the read component of $h$.

**Why pseudo-inverse, not inverse.** $Q$ is $d_h\times d$ with $d_h < d$ (128 < 5120): non-square, so $Q^{-1}$
does not exist. $Q^{+}Q$ is the projector onto $\mathrm{rowspace}(Q)$ (and $QQ^{+}=I_{d_h}$).

## 3. Aggregation over goal heads

Fix the **same** $h$; apply each goal head's own transform $Q_k = W_Q^{(\ell,j)}\mathrm{diag}(g^{(\ell)})$ and project:
$$
h_\text{goal} \;=\; \operatorname{mean}_{k\,\in\,\mathrm{READOUT\_HEADS}}\;\frac{P_k\,h}{\lVert P_k\,h\rVert}.
$$
(Per-head unit-normalization so no single head dominates; only the resulting **direction** is used next.)

## 4. Steering (GoalScope, `additive_norm`, $\alpha=0.1$)

Inject $h_\text{goal}$ into the probe's residual at the steer band, magnitude set by the local norm:
$$
h' \;=\; h_\text{resid} \;+\; \alpha\,\lVert h_\text{resid}\rVert\;\hat g,
\qquad \hat g = \frac{h_\text{goal}}{\lVert h_\text{goal}\rVert},\quad \alpha=0.1 .
$$
Only the **direction** $\hat g$ matters, so the dropped $1/\mathrm{rms}(x)$ scalar (and any global scale) is irrelevant.

## 5. What RMSNorm contributes (why we fold $g$, drop $1/\mathrm{rms}$)

$$
\mathrm{RMSNorm}(x) \;=\; \underbrace{\tfrac{1}{\mathrm{rms}(x)}}_{\text{scalar} \;\to\; \text{ignore}}\;\cdot\;\underbrace{g\odot x}_{\text{per-dim gain} \;\to\; \text{fold into }Q}.
$$
The scalar is direction-preserving (safe to ignore for `additive_norm`); the per-dimension gain $g$ is **not**
uniform and rotates the direction, so it must be folded in — hence $Q=W_Q\,\mathrm{diag}(g)$, not $W_Q$.

## 6. Interpretation

If steering with $h_\text{goal}(L)$ makes the probe **verbalize the goal**, then the residual after layer $L$
already **contains the goal** within the goal heads' query-read subspace — present and readable by their
queries, independent of (before) their output. Sweeping $L$ shows the earliest layer at which this holds.

## Caveats to check

1. **Cross-layer application.** Head $(\ell,j)$'s transform uses its own layer's gain $g^{(\ell)}$, but is
   applied to a residual $h$ taken after a possibly *different* layer $L$ — i.e. "if $h$ were fed to head
   $j$'s query, what would it read." A modeling choice consistent with "apply $Q$ to $h$".
2. **RoPE** is applied *after* $W_Q$ (an orthogonal rotation) → does not change $\mathrm{rowspace}(Q)$ → ignored.
3. **Aggregation** is a mean of per-head unit projections (a heuristic "union" of read subspaces); an
   alternative is projecting onto the joint span $\mathrm{rowspace}\big(\big[\,Q_{k_1};\dots;Q_{k_m}\,\big]\big)$.
4. **Injection band** (steer layers 35–46) $\neq$ probe layer $L$; `additive_norm` makes this a direction test,
   not a same-site re-insertion.
