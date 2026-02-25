# Multi-Predicate Setup: Design Report

## Why a New Setup?

### The Core Problem with the Current Setup

The current pipeline contrasts **correct** (green) vs **incorrect** (orange/red) MCTS nodes. A direction trained on this contrast may capture model *competence* — "will this trajectory succeed?" — rather than model *intent* — "which instrumental goal is the model currently pursuing?"

This matters because the central research claim is about **instrumental goal representations**, not success prediction. A competence direction and a goal direction *together* results in continuations/trajectories that land in a green-leading. We need a setup where they are separable.

### The Three-Level Hierarchy

```
Broader goal:      implement the function correctly
Instrumental goal: check whether the input is a palindrome   ← target
Implementation:    s == s[::-1]   vs   s[::-1] == s   vs   two-pointer
```

The desired direction should fire for "about to implement palindrome check" **invariant to which specific implementation is used**. Implementation plasticity is noise; the instrumental goal representation is the signal.

### Why Multi-Predicate?

If a function has two independent predicates (e.g., `is_palindrome` and `len >= 2`), the model must implement both but can choose which to do first. This creates two trajectories — palindrome-first and length-first — that:

- Both lead to **correct code** (same broader goal)
- Differ only in **which instrumental goal is being pursued first**

Contrasting these two trajectories gives a direction that captures *intent* rather than *competence*. There is no success/failure confound: both sides of the contrast are correct.

---

## The Setup: Concrete Design

### Target and Contrast Structure

- **P+** (positive predicate): the instrumental goal of interest, e.g., `is_palindrome`
- **P-** (negative predicate): a different, independent predicate paired with P+ in the same function

For each (P+, P-) pair, we create one `IncompleteCodeSample`. The MCTS tree for that sample has two main branch families: P+-first and P--first. Contrastive pairs are formed **across branches at the same depth**.

The direction trained on these pairs answers: "at this point in generation, is the model in a palindrome-pursuing trajectory or a length-pursuing trajectory?"

---

## Design Decisions

### 1. Pivot Point

The pivot is the point in the incomplete code where the model must choose which predicate to implement first. The natural pivot is right after `return `:

```python
def is_valid_palindrome(s: str) -> bool:
    """Return True iff s has at least 2 characters and is a palindrome."""
    return
```

At this point, the model's next token determines which branch it enters:
- `len(` → length-first
- `s ==` or `s[:` → palindrome-first

The pivot is fixed and explicit, making branch labeling clean.

### 2. Anchor Tokens

Unit tests cannot distinguish P+-first from P--first trajectories (all correct trajectories satisfy both predicates). Instead, branches are labeled by **anchor tokens** — distinctive tokens that uniquely identify which predicate's implementation is being written.

Anchor tokens are defined **per sample** and stored alongside the task definition:

```python
p_plus_anchors  = ["[::-1]", "reversed(", "== s["]   # palindrome
p_minus_anchors = ["len("]                             # length check
```

**Labeling algorithm**: walk the token path from root to node; the first anchor token hit determines the branch label. If no anchor has appeared yet, the node is labeled `unresolved`.

### 3. Depth Window: Using `and` as a Break Point

The instrumental goal signal is cleanest **after the model commits to a branch but before the first predicate is complete**. Once the model writes `and`, it transitions to the second predicate — the intent signal for P+ ends there and a new signal (for the second predicate) begins.

The `and` token is a concrete, detectable break point that defines two distinct intent windows per trajectory:

```
T1: [  P+-pursuing  ] and [  P--pursuing  ]
T2: [  P--pursuing  ] and [  P+-pursuing  ]
```

**Option A — Shallow fixed window (recommended for now)**: restrict pairs to depths 1–W from the pivot (suggested W=5). Simple, low noise, captures the first-predicate intent reliably. Noise: some W=5 nodes may already be past `and` for short implementations.

**Option B — `and`-bounded window**: detect the `and` token in each node's path and only include nodes that are (a) after the pivot and (b) before the first `and`. This gives a clean, variable-length window per trajectory. More implementation work but removes the fixed-W noise.

**Option C — Dual window (doubles training data)**: use both halves of each trajectory. In T1, nodes before `and` are positive for P+; nodes after `and` are negative (pursuing P- now). In T2, nodes before `and` are negative; nodes after `and` are positive for P+. This theoretically doubles training pairs.

The problem with Option C: labeling becomes trajectory-local rather than branch-global. The same node label can flip mid-trajectory (positive → negative at the `and`), requiring per-node rather than per-branch label assignment. Given that the `and` detection is already needed for Option B, Option C is reachable from there — but the added implementation complexity is not worth it at this stage. **Recommendation: start with Option A, migrate to Option B if the fixed window introduces too much noise.**

### 4. Cross-Branch Pairing: Within-Tree and Cross-Tree

Since both trajectories diverge at the pivot (depth 0), the LCA of any P+-labeled and P--labeled node within the same tree is always the root — **LCA-based matching is not applicable here**.

#### 4a. Within-Tree Pairing

If the model has a reasonably balanced prior over orderings, both P+-first and P--first branches will appear in the same MCTS tree. Pair them at each depth:

```
for each (prefill_idx, depth d in 1..W):
    pos_nodes = [n | label(n) == P+, depth(n) == d]
    neg_nodes = [n | label(n) == P-, depth(n) == d]
    pairs     = random 1:1 matching (allow neg reuse if len(pos) > len(neg))
```

#### 4b. Cross-Tree Pairing (necessary when LM has strong ordering preference)

If the model strongly prefers one ordering (e.g., always implements `len` first), then within any unprefilled tree almost all nodes will be P--labeled. The P+-branch may have zero or one node — within-tree pairing collapses.

**Solution**: pair across trees using prefill-forced branches.

- A P+-nudged prefill (e.g., `s ==`) forces most trajectories into the P+-first branch → predominantly P+-labeled nodes
- A P--nudged prefill (e.g., `len(`) forces most trajectories into the P--first branch → predominantly P--labeled nodes

Cross-tree pairs: at depth d, pair P+-labeled nodes from the P+-nudged tree against P--labeled nodes from the P--nudged tree.

```
for each (depth d in 1..W):
    pos_nodes = [n | label(n) == P+, from P+-nudged prefill trees]
    neg_nodes = [n | label(n) == P-, from P--nudged prefill trees]
    pairs     = random 1:1 matching
```

This is already partially supported by the existing prefill infrastructure. The key change: the (prefill_idx, depth) bucket definition must be relaxed so that nodes from *different* prefill trees can be paired together, as long as their branch labels are opposite.

**Concern**: cross-tree pairs come from different prompt contexts (different prefills), so the hidden state difference might partly reflect the prefill rather than the branch. This is mitigated by (a) having multiple prefills of each type so the direction can't overfit to a specific prompt, and (b) the cross-function transfer test, which would expose prefill-specific artifacts.

### 5. Choosing P- Predicates

Multiple P- partners are needed so the trained direction captures *palindrome intent* specifically, not *not-P-specific*. If the direction generalizes across P- partners, it is truly about P+.

Requirements for a good P-:
- **Logically independent** of P+ (not implied by, not implying P+)
- **Structurally distinct**: anchor tokens do not overlap with P+ anchors
- **Reasonable co-occurrence**: sounds natural in the same function as P+
- **Roughly symmetric difficulty**: neither ordering should be obviously preferred

Candidate P- predicates for `is_palindrome`:

| P- predicate | Example implementation | Anchor tokens |
|---|---|---|
| `len(s) >= 2` | `len(s) >= 2` | `["len("]` |
| `s.isalpha()` | `s.isalpha()` | `["isalpha"]` |
| `' ' not in s` | `' ' not in s` | `["not in"]` |
| `s[0].isupper()` | `s[0].isupper()` | `["isupper", "s[0]"]` |
| `s == s.lower()` | `s == s.lower()` | `["lower()"]` |

Start with 2–3 P- predicates. The cross-P- generalization test (see Evaluation) will reveal whether the direction is truly about P+.

### 6. Sample Construction

For each (P+, P-) pair, the `IncompleteCodeSample` contains:

**Function + docstring** (defines both predicates):
```python
def is_valid_palindrome(s: str) -> bool:
    """Return True iff s has at least 2 characters and is a palindrome."""
    return
```

**Unit tests** (evaluate the broader goal — both predicates must be satisfied):
```python
is_valid_palindrome("aba")   == True    # len OK, palindrome OK
is_valid_palindrome("ab")    == False   # len OK, not palindrome
is_valid_palindrome("a")     == False   # len fails
is_valid_palindrome("")      == False   # len fails
is_valid_palindrome("aa")    == True    # len OK, palindrome OK
is_valid_palindrome("abcba") == True
is_valid_palindrome("abc")   == False
```

**Prefills** (see below).

**Anchor token definitions** (new field, not in current `IncompleteCodeSample`):
```python
p_plus_anchors  = ["[::-1]", "reversed(", "== s["]
p_minus_anchors = ["len("]
```

### 7. Prefill Strategy

Prefills serve two roles in this setup: increasing training data and providing forced-branch examples for the generalization test.

| Prefill type | Example | Effect |
|---|---|---|
| Aligned for P+ | `s == s`, `s[:` | Nudges/forces palindrome-first branch |
| Aligned for P- | `len(` | Nudges/forces length-first branch |
| Adversarial | `not (`, `False if` | Unusual implementation, may still be correct |
| Inconsequential | `""` (empty) | Model's natural preference |

The aligned prefills are **forced-branch seeds**: they guarantee entry into a specific branch even when the model strongly prefers the other ordering. This solves the LM prior imbalance problem.

The generalization test: train direction on trees from forced-branch prefills, then test on the unprefilled tree where the model *chose* a branch. If the direction activates correctly, it captures intent rather than the effect of the prefill.

---

## Pipeline Changes vs Current Setup

| Step | Current pipeline | New pipeline |
|---|---|---|
| Sample definition | `code`, `unit_tests`, `prefills` | + `p_plus_anchors`, `p_minus_anchors` |
| MCTS | Unchanged | Unchanged |
| Tree evaluation | `evaluate_mcts_tree` via unit tests → green/orange/red | Same (still needed for filtering) |
| Branch labeling | — (not needed) | New: label each node P+/P-/unresolved via anchors |
| Pair sampling | Green vs red/orange nodes at same depth | P+-branch vs P--branch nodes at depths 1..W (within-tree or cross-tree) |
| Pairing matching | LCA-based unique positive | Random 1:1; cross-tree allowed when within-tree P+-/P-- balance is poor |
| Latent extraction | Unchanged | Unchanged |
| Direction training | Binary contrastive (correct vs incorrect) | Binary contrastive (P+-pursuing vs P--pursuing) |
| Evaluation | Steering causality test | + cross-P- generalization + cross-function transfer |

---

## Evaluation Protocol

### 1. Within-sample accuracy
Train on prefills 1..N-1, test on prefill N (same function). Baseline sanity check.

### 2. Cross-P- generalization (key test)
Train direction on `(is_palindrome, len >= 2)` samples. Test on `(is_palindrome, s.isalpha())` samples. If the direction still correctly identifies palindrome-first branches, it captures palindrome intent specifically — not "not-length-intent."

### 3. Causality / steering
Steer generation with the palindrome direction and check whether the model shifts towards palindrome-first implementations. Uses existing `test_causality.py` infrastructure with adapted prefills.

### 4. Cross-function transfer (strongest test)
Train on `is_valid_palindrome`. Test on a new function that also requires palindrome checking but in a different context:

```python
def are_all_palindromes(words: list[str]) -> bool:
    """Return True iff every word in the list is a palindrome."""
    return
```

If the direction activates here, the model has a generalizable `is_palindrome` goal representation — not just a function-specific one.

---

## Decisions and Open Questions

1. **Anchor tokens**: determine post-hoc. Run MCTS first, inspect the saved HTML trees to see what implementations actually appear, then define anchor tokens and label nodes from the saved JSONs. No need to pre-specify anchors before tree generation.

2. **Depth window**: use W=5 for now. Revisit with `and`-bounded window (Option B) if fixed-W noise is a problem empirically.

3. **Cross-tree pairing**: only pair P+-nudged trees against P--nudged trees (not unprefilled trees against each other). Defer implementation — generate trees first and check empirically whether within-tree branch balance is sufficient.

4. **Correctness filtering**: restrict contrastive pairs to nodes whose terminal descendants are predominantly green. Broken trajectories (only one predicate implemented) add noise to the intent direction.

5. **Multi-class / 3+ predicates**: out of scope for now. Once 2-predicate results are in hand, this is the natural next stress test.
