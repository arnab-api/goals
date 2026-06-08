# Step-lookback head score — equation

How `scripts/clock/score_multistep_heads.py` computes the per-head score with args:

```
--q_span=20 --key_agg=10 --compare_with="max" --version="v3"
```

(`--version="v3"` only redirects the cache path to `…/multistep_clock/v3/<model>/`; it does
not change the math.)

## Notation (per cached sample = one `(sample, modularity)`)

- $a_{\ell h}[q,s]$ — attention weight of head $(\ell,h)$ from query token $q$ to key token
  $s$ (the `attn.npz` array, shape $L\times H\times Q\times K$).
- Step $k$ has a **spec span** $P_k=[\,\text{prompt\_token\_range}\,)$ (instruction tokens,
  size $S_k$) and a **gen span** $[g_k^0,g_k^1)$ (its code tokens).
- $\mathcal{K}$ = steps with a spec span (denominator candidates); $\mathcal{T}$ = steps with
  both spans (query/target steps), minus the last step when `monolithic`.

## 1. Query window (`q_span=20`) — first 20 generated tokens of target step $k$

$$W_k=\big[\,g_k^0,\ \min(g_k^1,\ g_k^0+20)\,\big)$$

## 2. Window-summed attention to each spec token $s$ of step $j$

$$u^{k}_{\ell h}(s)=\sum_{q\in W_k} a_{\ell h}[q,s],\qquad s\in P_j$$

## 3. Per-step score with top-`key_agg`=10 key selection

(drop stopwords; mean of the 10 most-attended spec tokens of $j$, chosen per head)

$$
f^{k}_{\ell h}(j)=\frac{1}{\min(10,\,S_j)}\sum_{s\in\mathrm{Top}^{10}_{\ell h}(j;k)} u^{k}_{\ell h}(s),
\qquad
\mathrm{Top}^{10}_{\ell h}(j;k)=\underset{s\in P_j}{\arg\text{-top-}10}\ u^{k}_{\ell h}(s)
$$

(if $S_j\le 10$ it is just the mean over all of $P_j$).

## 4. Head score for target step $k$ (`compare_with="max"`, target excluded)

$$
r^{k}_{\ell h}=\frac{f^{k}_{\ell h}(k)}{\displaystyle\max_{j\in\mathcal{K},\,j\neq k} f^{k}_{\ell h}(j)}
\quad\text{(only when the denominator }>0\text{)}
$$

## 5. Final pooled score per head

(sum over targets across all samples / a modularity, ÷ count of contributing targets —
i.e. `sum_ratios / count`)

$$
\mathrm{Score}(\ell,h)=
\frac{\sum_{\text{sample}}\ \sum_{k\in\mathcal{T}}\ \mathbf{1}[\text{denom}>0]\; r^{k}_{\ell h}}
{\sum_{\text{sample}}\ \sum_{k\in\mathcal{T}}\ \mathbf{1}[\text{denom}>0]}
$$

## Notes

- The **same** target window $W_k$ is used for the numerator *and* every denominator term —
  they differ only in which step's spec tokens $P_j$ are attended to.
- $\mathrm{Top}^{10}$ is recomputed independently per $(\ell,h)$ and per candidate step $j$,
  so the 10 selected tokens differ across heads and steps.
- Interpretation: $\mathrm{Score}(\ell,h)>1$ ⇒ during the *first 20 tokens* of a step, head
  $(\ell,h)$ attends to that step's own (top-10) spec tokens more than to the most competitive
  *other* step's spec tokens.
