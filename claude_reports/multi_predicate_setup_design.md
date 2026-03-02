# Multi-Predicate Setup: Design Report

## 1. Motivation

### The Central Question

When an LM generates code, does it have a stable internal representation of *what it intends to do next*? Specifically, we are looking for **instrumental goal directions** in the hidden state: linear directions that capture "I am about to implement predicate P" regardless of which implementation strategy the model uses.

### The Three-Level Hierarchy

```
Broader goal:      implement the function correctly
Instrumental goal: check whether the input is a palindrome   <-- target of this research
Implementation:    s == s[::-1]   vs   s[::-1] == s   vs   two-pointer
```

The desired direction should fire for "about to implement palindrome check" **invariant to which specific implementation is used**. Implementation plasticity is noise; the instrumental goal representation is the signal.

### Why Not Correct-vs-Incorrect Contrast?

Our earlier setup contrasted **correct** (green) vs **incorrect** (orange/red) MCTS leaf nodes. A direction trained on that contrast risks capturing model *competence* -- "will this trajectory succeed?" -- rather than model *intent* -- "which instrumental goal is being pursued?" A competence direction and a goal direction, together, steer toward green-leading trajectories. But we need a setup where they are separable.

### Why Multi-Predicate?

If a function has two independent predicates (e.g., `is_palindrome` and `len >= 2`), the model must implement both but can choose which to do first. This creates two trajectory families -- palindrome-first and length-first -- that:

- Both lead to **correct code** (same broader goal)
- Differ only in **which instrumental goal is being pursued first**

Contrasting these two families gives a direction that captures *intent* rather than *competence*. There is no success/failure confound: both sides of the contrast are correct.

---

## 2. Concrete Steps to Extract a Goal Direction

### 2.1 Selecting a Target Predicate (P+)

P+ is the instrumental goal of interest (e.g., "check if string is a palindrome"). It defines a `TargetPredicate` group: a single `PredicateSpec` with a human-readable description and (eventually) oracle-extracted anchor tokens.

### 2.2 Sampling Distractor Predicates (P-)

For each P+, we sample multiple P- companion predicates -- independent predicates that can coexist with P+ in the same function. Multiple P- partners are needed so the trained direction captures "palindrome intent" specifically, not "not-P_i-intent."

Requirements for a good P-:
- **Logically independent** of P+ (not implied by, not implying P+)
- **Structurally distinct**: anchor tokens do not overlap with P+ anchors
- **Reasonable co-occurrence**: sounds natural in the same function as P+
- **Roughly symmetric difficulty**: neither ordering should be obviously preferred

Each (P+, P-) pair becomes one `MultiPredicateSample` -- an `IncompleteCodeSample` with the additional `p_minus: PredicateSpec` field.

### 2.3 Curating Samples with a Pivot Point

Each sample is a function stub whose docstring specifies *both* predicates. The code ends at a **pivot point** where the model is free to choose which predicate to implement first. Typically the pivot is right after `return `:

```python
def is_valid_palindrome(s: str) -> bool:
    """Return True iff s has at least 2 characters and is a palindrome."""
    return
```

At the pivot, the model's next tokens determine which branch it enters:
- `s ==` or `s[:` -> palindrome-first
- `len(` -> length-first

Unit tests evaluate the broader goal -- both predicates must be satisfied for a green leaf. They do **not** distinguish P+-first from P--first orderings (all orderings are correct).

Each sample also defines prefills of four types:

| Prefill type | Example | Effect |
|---|---|---|
| `aligned_p_plus` | `s == s`, `s[:` | Nudges into P+-first branch |
| `aligned_p_minus` | `len(` | Nudges into P--first branch |
| `adversarial` | `not (`, `False if` | Unusual; may still be correct |
| `inconsequential` | `""` (empty) | Model's natural preference |

### 2.4 Expanding MCTS Trees

For every `(sample x prefill)`, we run MCTS to explore the tree of possible continuations. Each tree is evaluated: leaves are colored green (correct, passes all unit tests), orange (compiles but fails tests), or red (syntax error / crash).

**Script**: `scripts/multipred_mcts.py`

Key MCTS config:
- `iterations=1500`, `max_depth=32`, `top_k=8`, `prob_threshold=0.01`
- Diversity: `c_puct=1.0`, `temperature=1.0`, `min_children_up_to_depth=3`, `min_children=2`

All trees for a `TargetPredicate` are serialized into a single JSON cache at `{save_dir}/{model}/{predicate.name}/trees/mcts.json`. Color-coded HTML trees are saved per `(P- description, prefill)` for visual inspection.

### 2.5 Identifying Predicate Spans with Anchor Tokens

Unit tests cannot distinguish which predicate is being implemented *first* -- they only evaluate overall correctness. Instead, branches are labeled by **anchor tokens**: short code substrings (2-8 chars) that uniquely signal which predicate's implementation is being written.

**Oracle extraction** (`src/goals/anchor.py`): An oracle LM (Claude, temperature=0) examines the function stub and green completions, and identifies 1-3 anchor tokens per predicate. The prompt asks for tokens that appear near the start of each predicate's code and do **not** normally appear in the other predicate's implementation. Anchors are pruned: any anchor that does not appear in any green completion is discarded.

Example anchors:
```
P+ (palindrome):  ["[::-1]", "reversed(", "== s["]
P- (length check): ["len("]
```

Anchor tokens are written back to the `TargetPredicate` JSON files on disk, so they persist across runs.

### 2.6 Labeling Nodes and Handling LM Ordering Preference

**The problem**: Often the LM has a strong ordering preference -- e.g., it *always* implements `len(...)` first in unprefilled trees. In that case, within any single tree, almost all green paths are P--first, and P+-first branches are scarce or absent. Within-tree pairing collapses.

**The solution**: Predicate-specific prefills. An `aligned_p_plus` prefill (e.g., `s ==`) forces the model into the P+-first branch; an `aligned_p_minus` prefill (e.g., `len(`) forces P--first. We then sample contrastive nodes **across different trees** -- P+-labeled nodes from a P+-nudged tree paired against P--labeled nodes from a P--nudged tree.

**Dual-window labeling** (`src/goals/predicate_probe.py`): For each green leaf path, we identify the `and` boundary (or equivalent clause separator) that separates the two predicate implementations. This splits each path into two windows:

```
P+-first branch: [pursuing P+] and [pursuing P-]
P--first branch: [pursuing P-] and [pursuing P+]
```

Both windows contribute labeled nodes: a node is labeled `"p_plus"` if it falls within the P+ clause span, `"p_minus"` if it falls within the P- clause span, or `"transition"` (excluded) if it falls outside both spans.

**Clause span detection** (`extract_predicate_clauses` in `anchor.py`): Extracts the boolean expression from `if/return/while/assert` statements, splits on top-level `and`/`or` (respecting bracket nesting), and matches each predicate's anchor tokens to the clause containing them. Returns start/end depths for each predicate's clause.

### 2.7 Contrastive Pairing Strategy

**Pairing unit**: nodes at the same **absolute token position** (`abs_position = len(tokenize(prompt)) + depth - 1`). This ensures the probe cannot exploit positional encoding differences between paired nodes.

**Cross-tree constraint** (default: `allow_same_tree=False`): Pairs are formed *across* different MCTS trees (different prefills). This prevents the probe from exploiting tree-internal structure (e.g., sibling topology) and forces it to learn from the hidden-state content alone.

**Filtering**: Only nodes with `p_green >= 0.5` (fraction of probability mass flowing to green descendants) survive. This ensures we only train on nodes in robustly correct trajectories.

**Matching**: 1:1 random matching at each absolute position. The positive pool is P+-labeled nodes and the negative pool is P--labeled nodes. Pool sizes vary by position; `abs_pos_tolerance=0` requires exact position match.

```python
CONTRASTIVE_PAIR_CONFIG = dict(
    min_p_green=0.5,
    allow_same_tree=False,
    abs_pos_tolerance=0,
)
```

**Train/test split**: 80/20 by tree ID (not by pair). All pairs from a given tree go entirely into train or test. This prevents leakage from seeing different nodes of the same tree in both splits.

---

## 3. Extracting Goal Directions

All activations are **L2-normalized** (projected onto the unit sphere) before training.

### 3.1 Mean-Difference Direction

The simplest baseline. For layer $\ell$:

$$\mathbf{v}_\ell^{\text{md}} = \frac{\bar{\mathbf{z}}_+^\ell - \bar{\mathbf{z}}_-^\ell}{\|\bar{\mathbf{z}}_+^\ell - \bar{\mathbf{z}}_-^\ell\|}$$

where $\bar{\mathbf{z}}_+^\ell = \frac{1}{|\mathcal{T}_+|}\sum_{i \in \mathcal{T}_+} \mathbf{z}_i^\ell$ is the mean of L2-normalized P+-labeled latents in the training set, and $\bar{\mathbf{z}}_-^\ell$ analogously for P-.

### 3.2 Contrastive (Hinge) Direction

A learned direction optimized via hinge loss with regularization. For layer $\ell$, we learn $\mathbf{v}_\ell$ to minimize:

$$\mathcal{L}(\mathbf{v}) = \frac{1}{N}\sum_{i=1}^{N} \max\!\Big(0,\; m - (\mathbf{z}_{+,i} - \mathbf{z}_{-,i})^\top \mathbf{v}\Big) + \lambda_1 \|\mathbf{v}\|_1 + \lambda_2 \|\mathbf{v}\|_2^2$$

where $({\mathbf{z}_{+,i}, \mathbf{z}_{-,i}})$ are contrastive pairs and $m$ is the margin.

Defaults: `margin=1.0`, `lr=1e-2`, `n_steps=1000`, `patience=50` (early stopping), `l2_lambda=1e-3`, `l1_lambda=0.0`.

The output is unit-normalized: $\mathbf{v}_\ell^{\text{cd}} = \mathbf{v} / \|\mathbf{v}\|$.

### 3.3 Evaluation Metrics on Train/Test

- **Constraint accuracy**: fraction of pairs where $\mathbf{z}_+^\top \mathbf{v} > \mathbf{z}_-^\top \mathbf{v}$
- **ROC-AUC**: AUC of the projection scores $\mathbf{z}^\top \mathbf{v}$ against binary labels (P+ = 1, P- = 0) on the test set

Both are computed per layer for both direction types.

---

## 4. Evaluation Protocol

### 4.1 In-Distribution Classification

Train on 80% of trees (from multiple (P+, P-) samples), test on the held-out 20%. This is the basic sanity check: can the direction separate P+-pursuing from P--pursuing nodes from the same `TargetPredicate`?

Metrics: constraint accuracy and ROC-AUC per layer.

### 4.2 Out-of-Distribution Generalization (Cross-P-)

Train the direction on samples with one set of P- companions (e.g., `(is_palindrome, len >= 2)`). Test on a held-out sample with a *different* P- companion (e.g., `(is_palindrome, s.isalpha())`).

If the direction still correctly identifies palindrome-first branches in the new P- context, it captures P+ intent specifically -- not "not-P_i-intent." This is the key test that the direction is about the target instrumental goal and not about a specific distractor.

### 4.3 Causality / Steerability

Three progressively stronger steering tests:

**4.3a. Nudge predicate ordering in multi-pred setup**: Steer generation with the P+ direction at the pivot point. Check whether the model shifts toward implementing P+ first, even when given a P--aligned prefill or no prefill. Success = the direction causally influences *which* instrumental goal gets pursued first.

**4.3b. Improve persistence against adversarial prefills in one-predicate setup**: Take a standard one-predicate task (e.g., `is_odd`) where an adversarial prefill (e.g., `1 +`) confuses the model. Steer with the goal direction trained from the multi-pred setup. Check whether the model's success rate on adversarial prefills improves -- i.e., the goal direction helps the model "remember" what it's supposed to do despite the misleading prefill.

**4.3c. Induce target predicate in arbitrary code**: The strongest test. Take an arbitrary code generation context that has *no mention of the target predicate in its instructions*. Steer with the P+ goal direction. Check whether the model spontaneously introduces P+-related code (e.g., inserts a palindrome check where none was requested). This would demonstrate that the direction encodes a generalizable intent to "do P+" that transfers beyond any specific function context.

### 4.4 Cross-Function Transfer (Strongest Generalization)

Train the P+ direction on `is_valid_palindrome(s)`. Test on a structurally different function that also involves the same instrumental goal:

```python
def are_all_palindromes(words: list[str]) -> bool:
    """Return True iff every word in the list is a palindrome."""
    return
```

If the direction fires here, the model has a generalizable `is_palindrome` goal representation -- not just a function-specific one.

---

## 5. Pipeline Summary

### Two-Script Architecture

| Script | Purpose | Output |
|---|---|---|
| `multipred_mcts.py` | MCTS expansion + tree evaluation + anchor token extraction | `trees/mcts.json`, HTML trees, populated anchor tokens in data JSONs |
| `multipred_probe.py` | Load trees, label nodes, pair, extract latents, train directions, evaluate | `mean_diff_directions.pt`, `contrastive_directions.pt`, `accuracies.json`, `config.json` |

The scripts share a `--save_dir` root (or can use separate `--tree_dir` / `--save_dir` in the probe script for decoupled storage).

### Artifact Layout

```
{save_dir}/
  {model_name}/
    {predicate.name}/
      trees/
        mcts.json                          # All (sample x prefill) trees, bundled
        {p_minus_desc}/
          {prefill}/
            mcts_tree.html                 # Color-coded tree (from multipred_mcts.py)
            probe_log.json                 # Labeled-node breakdown (from multipred_probe.py)
      mean_diff_directions.pt              # dict[layer_idx -> unit-norm Tensor]
      contrastive_directions.pt            # dict[layer_idx -> unit-norm Tensor]
      accuracies.json                      # Per-layer train/test accuracy + AUC
      contrastive_infos.json               # Per-layer training info
      config.json                          # Full run config
```

### Data Structures

```
MultiPredicateDataset
  groups: list[TargetPredicate]            # one per target instrumental goal

TargetPredicate
  name: str                                # e.g. "has_unique_elements"
  p_plus: PredicateSpec                    # target predicate (shared across samples)
  samples: list[MultiPredicateSample]      # one per (P+, P-) pair

PredicateSpec
  description: str                         # human-readable
  anchor_tokens: list[str]                 # oracle-extracted, e.g. ["[::-1]", "reversed("]

MultiPredicateSample (extends IncompleteCodeSample)
  code, instruction, unit_tests, prefills, function_name
  p_minus: PredicateSpec                   # companion predicate for this sample
```

---

## 6. Open Questions and Considerations

1. **`abs_pos_tolerance`**: Currently set to 0 (exact position match). If pairing is too sparse, relaxing to tolerance=1 would allow nearby-position pairs at the cost of introducing a small positional confound.

2. **Cross-tree prefill confound**: Pairs from different prefill trees come from different prompt contexts, so the hidden-state difference might partly reflect the prefill rather than the branch. Mitigated by (a) having multiple prefills of each type, (b) the cross-function transfer test which would expose prefill-specific artifacts, and (c) the within-tree pairs (when available) serving as a control.

3. **Multi-class / 3+ predicates**: The current setup is binary (P+ vs P-). With 3+ predicates in one function, one could train one-vs-rest directions or a multi-class probe. Out of scope for now -- once 2-predicate results are solid, this is the natural next stress test.

4. **`terminal_only=True` leaf scarcity**: Since MCTS branches stop at the first newline, most terminals are single-line completions. Orange leaves (compiles but wrong) can be rare for one-liners. The `min_p_green` filter on internal nodes helps by allowing use of non-leaf nodes on green-leading paths.

5. **Anchor token quality**: Anchors are extracted by an oracle LM and pruned against actual completions, but edge cases remain (e.g., an anchor for P+ that accidentally appears in a P- implementation). Manual inspection of anchor extractions is recommended for each new `TargetPredicate`.
