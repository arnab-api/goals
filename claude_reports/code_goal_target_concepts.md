# Code-Generation Goal Concepts: Target Taxonomy

**Date**: 2026-05-13
**Status**: design doc
**See also**: [`readings_causal_behavioral_goal_analysis.md`](readings_causal_behavioral_goal_analysis.md)

---

## 1. Motivation

We're trying to localize the representation of *goals* (proactive, future-oriented decisions) inside code-LMs. So far our investigation has aimed at the abstract notion of an "instrumental goal" — a representation that should fire when the LM is about to do X, invariant to which implementation strategy realizes X.

This framing is hard to make actionable for two reasons:

1. **No bounded concept set.** We don't have a fixed list of goals to probe for, the way the AlphaZero paper (McGrath et al. 2022) has Stockfish's sub-functions (`king_safety`, `material`, `mobility`, …) for board state.
2. **No automatic labels.** Without a closed set, we can't auto-score whether a particular generation pursued a particular goal.

The AlphaZero analysis sidesteps both problems by **delegating label generation to Stockfish**: each of ~25 sub-functions maps a board state to a numeric or binary value, giving free concept labels across 10⁵ positions.

We want the analogous move for code, with two adaptations imposed by our problem framing.

### Working definition of a goal

A **goal** is a proactive decision the LM makes that results in a future action.

- **Proactive**: the agent makes the decision. It's intrinsic to the agent, not forced by the prompt, syntax, or task.
- **Future-oriented**: it's about an action the agent *will* take, not a property of the current state of the partial program.

### Implications for concept selection

- Concepts must live at a **fork**: a point during generation where multiple legitimate continuations exist and the LM commits to one.
- The same partial program must admit **multiple goals**. If state S deterministically implies action A, then any direction predicting "will do A" is indistinguishable from a state representation of S — and we've recovered a world-state probe, not a goal probe.
- The choice must be observable in the generated tokens (so we can auto-label).
- Both options should be *legitimate*: if one option is strictly correct and the other is buggy, we're measuring competence, not goal-pursuit. The central confound we want to avoid.

We call this restricted concept set the **target concepts**.

---

## 2. What we explicitly excluded

**File-open → close anticipation.** Intuitively appealing as a "future commitment" test (does the LM at `open(...)` already represent the eventual `.close()`?). Rejected because it collapses to a cheaper proxy: detecting that a file is currently open in scope. Any direction predicting "will close" is indistinguishable from a state-property direction encoding "open file in scope." Fails the proactiveness axis.

**Surface stylistic choices** (snake_case vs camelCase, single vs double quotes, docstring style). Proactive in form, but these almost certainly reflect training-distribution bias rather than goal-pursuit toward a downstream computation.

**Syntactic obligations** (closing a paren, closing a bracket, indenting a block). Forced by the language, not a free choice.

**Anything determined by the prompt.** If the prompt says "implement quicksort," the LM doesn't *choose* quicksort — it's instruction-following. Prompts must be underspecified at the fork.

---

## 3. Target concepts

Three families:

- **Family A — Strategy choice at a fork.** Multiple valid algorithms / structures for the same underspecified task.
- **Family B — Multi-action ordering.** Multiple independent actions, all required, in any order.
- **Family C — Data-structure commitment.** A declarative choice at one token shapes the rest of the function.

Each concept below lists:
- **Fork**: where the LM commits
- **Choices**: the categorical options
- **Example**: a prompt that admits both
- **Auto-label**: how we score a generation

---

### Family A — Strategy choice at a fork

#### A1. Recursive vs Iterative

- **Fork**: function body start (after any base-case guard)
- **Choices**: {`recursive`, `iterative`}
- **Example**:
  ```python
  def count_leaves(node):
      """Return the number of leaf nodes in a binary tree."""
      # ← fork
  ```
  *Recursive*: `return 1 if node.is_leaf else count_leaves(node.left) + count_leaves(node.right)`
  *Iterative*: `stack = [node]; while stack: ...`
- **Auto-label**: AST — does the body contain a `Call` whose function name equals the enclosing function's name?

---

#### A2. Early return vs Nested if-else

- **Fork**: function body start, given a precondition / off-path case
- **Choices**: {`early_return`, `nested`}
- **Example**:
  ```python
  def normalize(vec):
      """Return vec scaled to unit L2 norm; if vec is the zero vector, return vec unchanged."""
      # ← fork
  ```
  *Early return*: `if not any(vec): return vec` then proceed flat.
  *Nested*: `if any(vec): ... else: return vec`.
- **Auto-label**: top-level `If` whose body is a single `Return` and that precedes the main computation.

---

#### A3. LBYL vs EAFP

- **Fork**: at an operation that may raise (dict access, attribute access, conversion)
- **Choices**: {`lbyl`, `eafp`}
- **Example**:
  ```python
  def get_user_age(db, user_id):
      """Return the user's age in years; if the user is not in db, return None."""
      # ← fork
  ```
  *LBYL*: `if user_id in db: return db[user_id].age; return None`
  *EAFP*: `try: return db[user_id].age except KeyError: return None`
- **Auto-label**: presence of `Try` node before the access, vs an `If` containing a membership test.

---

#### A4. Imperative loop vs Comprehension

- **Fork**: at the start of assembling an output collection
- **Choices**: {`imperative_loop`, `comprehension`}
- **Example**:
  ```python
  def squared_evens(xs):
      """Return the squares of the even numbers in xs."""
      # ← fork
  ```
  *Imperative*: `out = []; for x in xs: if x % 2 == 0: out.append(x * x); return out`
  *Comprehension*: `return [x * x for x in xs if x % 2 == 0]`
- **Auto-label**: presence of `ListComp` / `SetComp` / `DictComp` / `GeneratorExp` in the body.

---

#### A5. Generator vs List return

- **Fork**: function commits to its output protocol (very early — first non-trivial body token)
- **Choices**: {`generator`, `list_return`}
- **Example**:
  ```python
  def primes_up_to(n):
      """Produce all prime numbers ≤ n."""
      # ← fork
  ```
  *Generator*: `for i in range(2, n+1): if is_prime(i): yield i`
  *List*: `out = []; for i in range(2, n+1): if is_prime(i): out.append(i); return out`
- **Auto-label**: any `Yield` node anywhere in the function → generator; else list/return.

---

#### A6. Iteration paradigm (index vs item vs enumerate)

- **Fork**: at the start of the `for` loop header
- **Choices**: {`range_index`, `for_each`, `enumerate`}
- **Example**:
  ```python
  def label_items(items):
      """Return a list of strings of the form '<i>: <item>' for each element of items, where <i> is its 0-based position."""
      # ← fork
  ```
  *range_index*: `for i in range(len(items)): out.append(f"{i}: {items[i]}")`
  *enumerate*: `for i, item in enumerate(items): out.append(f"{i}: {item}")`
- **Auto-label**: parse `For.iter` — `Call(range, [Call(len, ...)])` → range_index, `Call(enumerate, ...)` → enumerate, plain `Name` → for_each.

---

### Family B — Multi-action ordering

#### B1. Multi-predicate validation order

- **Fork**: after the signature/docstring, before the first check
- **Choices**: which predicate is checked first (multi-class)
- **Example**:
  ```python
  def validate_config(timeout, host, port):
      """Raise ValueError if timeout ≤ 0, host is empty, or port is not in 1..65535."""
      # ← fork
  ```
  Three valid orderings, all correct. (This is the current `validate_config` line of work.)
- **Auto-label**: walk the body top-down → first `If` test → identify which input it inspects.

---

#### B2. Side-effect ordering

- **Fork**: function body start, given a set of required side effects
- **Choices**: order among the effects (multi-class)
- **Example**:
  ```python
  def setup_account(user, db, mailer, logger):
      """Insert user into db, send the welcome email via mailer, and log the signup via logger."""
      # ← fork
  ```
- **Auto-label**: AST — ordered list of top-level `Expr(Call)` nodes → which side effect is invoked first.

---

### Family C — Data-structure commitment

#### C1. Output container

- **Fork**: at the first appearance of the container being returned
- **Choices**: {`list`, `set`, `dict`, `tuple`, …}
- **Example**:
  ```python
  def unique_words(text):
      """Return the unique words in text."""
      # ← fork
  ```
  `set(text.split())` vs `list(dict.fromkeys(text.split()))` vs `list({...})`
- **Auto-label**: AST — the outermost constructor of the returned expression.

---

#### C2. Accumulator type

- **Fork**: at the declaration of an accumulator variable (commits to API for the rest of the function)
- **Choices**: {`dict`, `defaultdict_int`, `Counter`, …}
- **Example**:
  ```python
  def word_counts(text):
      """Return a mapping from each word in text to its count."""
      # ← fork
  ```
  `counts = {}` vs `counts = defaultdict(int)` vs `counts = Counter(text.split())`
- **Auto-label**: AST — the RHS of the first assignment to the accumulator (and presence of the relevant `Import`).

---

## 4. Goal-criteria check

| Concept | Proactive? | Future-oriented? | Multi-option from same state? |
|---|---|---|---|
| A1 recursive vs iterative      | algorithm-style choice  | commits whole body            | yes |
| A2 early return vs nested      | control-flow style      | shapes rest of body           | yes |
| A3 LBYL vs EAFP                | idiomatic style         | commits handler structure     | yes |
| A4 loop vs comprehension       | structural choice       | commits assembly pattern      | yes |
| A5 generator vs list           | major API choice        | commits whole function shape  | yes |
| A6 iteration paradigm          | structural choice       | shapes loop body              | yes |
| B1 multi-predicate order       | order is free           | predicts next ~5 tokens       | yes |
| B2 side-effect order           | order is free           | predicts next call            | yes |
| C1 output container            | declarative choice      | commits return value          | yes |
| C2 accumulator type            | declarative choice      | commits API throughout body   | yes |

All ten candidates satisfy the three criteria.

---

## 5. Recommended first round

Start with **2–3 concepts** with the cleanest auto-label rules and the strongest single-token fork:

1. **A2 — early return vs nested.** Single-token fork (`if guard` vs main logic), trivial AST rule, every sample provides supervision at exactly one well-defined position.
2. **A3 — LBYL vs EAFP.** Classic well-documented Python idiom; many natural prompts admit both; aligns with how training-data is split between idiom communities.
3. **A5 — generator vs list.** Very early commitment (`yield` vs `return [` / `out = []`); the choice constrains the rest of the function — high signal-per-sample.

Family B (the existing `validate_config` work) is the natural multi-class extension once binary concepts are validated.

---

## 6. Open questions

- **Prompt curation.** Some prompts strongly bias one strategy (e.g., "compute factorial" → recursive). We need a filtering step: keep only prompts where, under sampling, the LM picks each option in ≥ X% of completions. This is the dataset-construction equivalent of choosing chess positions where Stockfish's score is not extreme.
- **Cross-concept direction transfer.** Is "early return" the same hidden-state direction as "LBYL"? Both are about *exiting early on an off-path condition*. Cross-concept probe transfer is itself an interesting experiment — and a sharper analogue of the chess paper's question of whether `king_safety` and `mobility` are independently represented.
- **Sample-level confounds.** Prompts where the LM tends to pick "early return" may also be shorter, more procedural, etc. We need balanced datasets per concept and careful controls (random-concept regression baselines, à la §4.2 of the AlphaZero paper).
- **Behavioral signatures.** Beyond probe accuracy, we want goal-criteria evidence: **persistence** (does the LM stick with its chosen strategy after a mid-generation perturbation?) and **plasticity** (does it pursue the same goal across diverse phrasings of the prompt?). See §6.1 of the readings doc for the operationalization.
