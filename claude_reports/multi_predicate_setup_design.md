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
- `iterations=1500`, `max_depth=48`, `top_k=8`, `prob_threshold=0.01`
- Diversity: `c_puct=1.0`, `temperature=1.0`, `min_children_up_to_depth=6`, `min_children=4`

Per-sample results are cached as `{save_dir}/{model}/{predicate.name}/{p_minus_desc}/mcts.json`. Color-coded HTML trees are saved in a `trees/` subdirectory alongside each `mcts.json`.

```
multipred_mcts.py :: process_predicate(tp):
    for sample in tp.samples:                        # each (P+, P-) pair
        sample_dir = {save_dir}/{model}/{tp.name}/{p_minus_desc}/
        if sample_dir/mcts.json exists and not --force_rerun_mcts:
            sr = load from cache
        else:
            prefills = flatten(sample.prefills)       # all 4 types
            for (pval, ptype) in prefills:
                prompt = apply_chat_template(instruction, code + pval)
                tree   = MCTSSearch(prompt, ...).run(n_iterations=1500)
                evaluate_mcts_tree(tree, unit_tests)  # color leaves green/orange/red
                save tree HTML → sample_dir/trees/{pval}.html
            sr.save(sample_dir/mcts.json)
        tp_results.sample_results.append(sr)
    print total green leaf count (informational)
```

### 2.5 Identifying Predicate Spans with Anchor Tokens

Unit tests cannot distinguish which predicate is being implemented *first* -- they only evaluate overall correctness. Instead, branches are labeled by **anchor tokens**: short code substrings (2-8 chars) that uniquely signal which predicate's implementation is being written.

**Oracle extraction** (`src/goals/anchor.py`): An oracle LM (Claude, temperature=0) examines the function stub and green completions, and identifies 1-3 anchor tokens per predicate. The prompt asks for tokens that appear near the start of each predicate's code and do **not** normally appear in the other predicate's implementation. Anchors are pruned: any anchor that does not appear in any green completion is discarded.

Example anchors:
```
P+ (palindrome):  ["[::-1]", "reversed(", "== s["]
P- (length check): ["len("]
```

Anchor tokens are written back to the `TargetPredicate` JSON files on disk, so they persist across runs.

**Ternary handling**: The extractor terminates the boolean expression at a top-level
`else` keyword (boundary-checked so it is not part of an identifier). This ensures that
`return True if cond else False` extracts only `cond`, not `cond else False`.
Without this, the `else` branch would be absorbed into the clause and corrupt anchor
matching for predicates that appear after `else`.

**Script**: `scripts/multipred_probe.py` (step 1.5, runs before labeling)

```
ensure_anchor_tokens(tp, tp_results, force_reextract):
    if all anchors already populated and not force_reextract:
        return  # no-op

    # Gather green completions grouped by sample
    completions_per_sample = [
        {prefix: [green_leaf_texts...]}   # for each sample
        for sr in tp_results.sample_results
    ]

    # Call oracle LM (Claude) to extract anchor tokens
    p_plus_anchors, p_minus_per_sample = extract_anchor_tokens(tp, completions_per_sample)
    # extract_anchor_tokens internally:
    #   1. Builds a prompt with the function stub + representative green completions
    #   2. Asks oracle: "what short code substrings (2-8 chars) uniquely signal each predicate?"
    #   3. Prunes: discard any anchor not found in any green completion

    tp.p_plus.anchor_tokens = sorted(p_plus_anchors)        # shared across all samples
    for i, sample in enumerate(tp.samples):
        sample.p_minus.anchor_tokens = sorted(p_minus_per_sample[i])  # per-sample

    tp.save()  # writes back to data/multi_predicate/{predicate.name}.json
```

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

```
collect_labeled_nodes(tp_results, tp, mt, sample_indices, min_p_green=0.5):
    for sample_idx in sample_indices:
        sr = tp_results.sample_results[sample_idx]
        p_plus_anchors  = tp.p_plus.anchor_tokens          # e.g. ["[::-1]"]
        p_minus_anchors = tp.samples[sample_idx].p_minus.anchor_tokens  # e.g. ["len("]

        for run in sr.runs:                                  # one run per prefill
            root = sr._trees[run.prefix]                     # in-memory MCTS tree
            prompt_token_len = len(tokenize(run.prompt))

            for each green leaf in tree:
                # 1. Find clause spans
                p_plus_clause, p_minus_clause = extract_predicate_clauses(
                    leaf, run.prefix, p_plus_anchors, p_minus_anchors
                )
                # extract_predicate_clauses internally:
                #   a. Reconstruct full text = prefix + root→leaf tokens
                #   b. Find the boolean expression (after return/if/while/assert)
                #   c. Split on top-level 'and'/'or' (respecting bracket nesting)
                #   d. Match each clause to P+/P- by checking which anchor appears in it
                #   e. Return ClauseSpan(text, start_depth, end_depth) for each

                if both clauses are None: skip leaf

                # 2. Label each non-root node on the path
                first_label = "p_plus" if p_plus starts first else "p_minus"
                for node in root→leaf path (depth 1..leaf):
                    if node is leaf:   # goal already done — exclude from probing
                        continue
                    if node.depth in [p_plus_clause.start .. p_plus_clause.end]:
                        label = "p_plus"
                    elif node.depth in [p_minus_clause.start .. p_minus_clause.end]:
                        label = "p_minus"
                    else:
                        label = "transition"  → skip

                    window = "first" if label == first_label else "second"

                    # 3. Filter by green probability mass
                    p_green = compute_outcome_probs(node).green
                    if p_green < min_p_green: skip

                    # 4. Compute absolute position
                    abs_pos = prompt_token_len + node.depth - 1

                    emit LabeledNode(node, label, window, abs_pos, depth,
                                     tree_id=run.prefix, sample_idx, prefill_type)

    return collected  # flat list, transition nodes excluded
```

**Terminal node exclusion**: The leaf node itself is excluded from labeled node
collection. At a terminal node the predicate implementation is already complete, so its
hidden state reflects *completion* rather than *pursuit* of the goal. Probing on
terminal nodes risks conflating goal intent with goal success. Only pre-terminal nodes
— where the model is actively generating within a predicate clause — are retained.

### 2.7 Contrastive Pairing Strategy

**Pairing scope**: Pairs are formed **per sample** (per P- companion). Nodes from different `(P+, P-)` pairs are never paired together — this prevents cross-P- contamination, where the probe could learn to distinguish "which P- is present" rather than "is P+ being pursued."

**Pairing unit**: nodes at the same **absolute token position** (`abs_position = len(tokenize(prompt)) + depth - 1`). This ensures the probe cannot exploit positional encoding differences between paired nodes.

**Cross-tree constraint** (default: `allow_same_tree=False`): Within each sample, pairs are formed *across* different MCTS trees (different prefills). This prevents the probe from exploiting tree-internal structure (e.g., sibling topology) and forces it to learn from the hidden-state content alone.

**Filtering**: Only nodes with `p_green >= 0.5` (fraction of probability mass flowing to green descendants) survive. This ensures we only train on nodes in robustly correct trajectories.

**Matching**: 1:1 random matching at each absolute position. The positive pool is P+-labeled nodes and the negative pool is P--labeled nodes. Pool sizes vary by position; `abs_pos_tolerance=0` requires exact position match.

```python
CONTRASTIVE_PAIR_CONFIG = dict(
    min_p_green=0.5,
    allow_same_tree=False,
    abs_pos_tolerance=0,
)
```

**Train/test split**: 70/30 by tree ID (not by pair). All pairs from a given tree go entirely into train or test. This prevents leakage from seeing different nodes of the same tree in both splits.

```
collect_and_split_pairs(tp_results, tp, mt, args):
    # ── Per-sample pairing ────────────────────────────────────────
    for sample_idx in range(len(tp.samples)):
        labeled_i = collect_labeled_nodes(
            tp_results, tp, mt,
            sample_indices=[sample_idx],     # THIS SAMPLE ONLY
            min_p_green=0.5,
        )

        # Form pairs WITHIN this sample (cross-tree, same abs_position)
        pairs_by_pos_i = sample_multipred_pairs(
            labeled_nodes=labeled_i,
            allow_same_tree=False,
            abs_pos_tolerance=0,
        )

        # sample_multipred_pairs internally:
        #   group labeled_i by abs_position
        #   for each abs_pos:
        #       p_plus_pool  = [ln for ln in group if ln.label == "p_plus"]
        #       p_minus_pool = [ln for ln in group if ln.label == "p_minus"]
        #       for each p_plus node:
        #           pick a p_minus node from a DIFFERENT tree (tree_id != pos.tree_id)
        #           emit MultiPredContrastivePair(positive, negative, abs_pos)

        flatten pairs_by_pos_i → pair_records (with pos_text, neg_text, metadata)

    # ── Train/test split by tree ID ───────────────────────────────
    all_tree_ids = unique tree IDs from pair_records
    shuffle, split 70/30
    train_idx = pairs whose pos_tree_id ∈ train_trees
    test_idx  = pairs whose pos_tree_id ∈ test_trees
    cap train_idx at args.n_train_pairs, test_idx at args.n_test_pairs

    return all_labeled, pair_records, train_idx, test_idx
```

---

## 3. Extracting Goal Directions

### 3.0 Latent Extraction

For each contrastive pair, reconstruct the full input text (prompt + path tokens up to the node) and extract the last-token hidden state at every probed layer.

```
extract_latents(mt, pair_records, layers, batch_size):
    pos_texts = [prompt + pos_node.get_path_text() for each pair]
    neg_texts = [prompt + neg_node.get_path_text() for each pair]
    all_texts = pos_texts + neg_texts

    for batch in chunks(all_texts, batch_size):
        hs = get_last_token_latents_batch(mt, batch, layers)
        # hs[layer_idx] = Tensor(batch_size, hidden_dim)
        # extracts hidden state at the LAST token position for each layer

    Z_pos[layer] = first half of concatenated latents   # (n_pairs, hidden_dim)
    Z_neg[layer] = second half                          # (n_pairs, hidden_dim)
    return Z_pos, Z_neg
```

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

```
train_directions(Z_pos, Z_neg, train_idx, test_idx, layers):
    # L2-normalize all activations
    Z_pos[li] = Z_pos[li] / ||Z_pos[li]||   for each layer li
    Z_neg[li] = Z_neg[li] / ||Z_neg[li]||

    for each layer li:
        z_pos_tr = Z_pos[li][train_idx]
        z_neg_tr = Z_neg[li][train_idx]

        # Mean-diff
        v_md = mean(z_pos_tr) - mean(z_neg_tr)
        v_md = v_md / ||v_md||

        # Contrastive (hinge)
        v_cd = randn(hidden_dim) * 0.01
        for step in 1..1000:
            delta = z_pos_tr - z_neg_tr
            loss = mean(clamp(margin - delta @ v_cd, min=0))
                 + l1 * mean(|v_cd|) + l2 * mean(v_cd²)
            v_cd -= lr * grad(loss)
            if no improvement for 50 steps: break
        v_cd = v_cd / ||v_cd||

        # Evaluate on test set
        constraint_acc = fraction of test pairs where z_pos @ v > z_neg @ v
        auc = roc_auc_score(labels=[1]*n_test + [0]*n_test,
                            scores=[z_pos @ v; z_neg @ v])

    return md_dirs, cd_dirs, accuracies, auc_results
```

### 3.3 Evaluation Metrics on Train/Test

- **Constraint accuracy**: fraction of pairs where $\mathbf{z}_+^\top \mathbf{v} > \mathbf{z}_-^\top \mathbf{v}$
- **ROC-AUC**: AUC of the projection scores $\mathbf{z}^\top \mathbf{v}$ against binary labels (P+ = 1, P- = 0) on the test set

Both are computed per layer for both direction types.

---

## 4. Evaluation Protocol

### 4.1 In-Distribution Classification

Train on 70% of trees (from multiple (P+, P-) samples), test on the held-out 30%. This is the basic sanity check: can the direction separate P+-pursuing from P--pursuing nodes from the same `TargetPredicate`?

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
| `multipred_mcts.py` | MCTS expansion + tree evaluation | `trees/mcts.json`, HTML trees |
| `multipred_probe.py` | Anchor token extraction, load trees, label nodes, pair, extract latents, train directions, evaluate | `mean_diff_directions.pt`, `contrastive_directions.pt`, `accuracies.json`, `config.json`, populated anchor tokens in data JSONs |

The scripts share a `--save_dir` root (or can use separate `--tree_dir` / `--save_dir` in the probe script for decoupled storage).

```
multipred_probe.py :: process_predicate(tp):
    1.   tp_results = load MCTS results from {tree_dir}/{model}/{tp.name}/*/mcts.json
    1.5  ensure_anchor_tokens(tp, tp_results)         # oracle call if missing
    2.   for each sample:                              # PER-SAMPLE pairing
             labeled_i = collect_labeled_nodes(sample_indices=[sample_idx])
             pairs_i = sample_multipred_pairs(labeled_i, allow_same_tree=False)
         → all_labeled, pair_records, train_idx, test_idx
    3.   save_tree_logs(labeled_nodes → per-run JSON logs)
    4.   layers = range(layer_from, layer_to, layer_step)
    5.   Z_pos, Z_neg = extract_latents(pair_records, layers)
    6.   train mean-diff + contrastive directions per layer
    7.   evaluate: constraint accuracy + ROC-AUC on test set
    8.   save directions, accuracies, config to {save_dir}/{model}/{tp.name}/
```

### Artifact Layout

```
{save_dir}/
  {model_name}/
    {predicate.name}/
      {p_minus_desc}/                        # One dir per (P+, P-) sample
        mcts.json                            # Per-sample MCTS results (from multipred_mcts.py)
        trees/
          {prefill}.html                     # Color-coded MCTS tree (from multipred_mcts.py)
        probe_log_{prefill}.json             # Labeled-node breakdown (from multipred_probe.py)
      mean_diff_directions.pt                # dict[layer_idx -> unit-norm Tensor]
      contrastive_directions.pt              # dict[layer_idx -> unit-norm Tensor]
      accuracies.json                        # Per-layer train/test accuracy + AUC
      contrastive_infos.json                 # Per-layer training info
      config.json                            # Full run config
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

---

## 7. Recent Updates

**`IncompleteCodeDataset.find_by_function_name(name)`** (`src/goals/data.py`): Added to
look up the standalone `IncompleteCodeSample` for any `TargetPredicate` by name. Used by
evaluation/probing code to retrieve the base single-predicate task for a given P+.

Three new entries were added to `data/incomplete_code.json` to cover all six current
`TargetPredicate` names: `is_anagram`, `is_perfect_square`, `has_unique_elements`
(joining the existing `is_sorted`, `is_palindrome`, `is_power_of_two`).
