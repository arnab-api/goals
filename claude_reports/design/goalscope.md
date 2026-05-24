# GoalScope: Decoding Sub-Goal Intent from Code-LM Latents

**Status:** Brainstorm / early design  
**Last updated:** 2026-04-09  
**Changelog:**
- 2026-04-09 — statements added as first-class intent blocks (L4)
- 2026-04-09 — scope identification design decisions finalised (see section below)

---

## Research Question

When a human programmer writes a line of code — whether opening a block (`while`, `if`, `for`) or writing a simple statement (`visited = set()`) — they already know *why* they are writing it: they have a pre-formed intent for what that unit should accomplish relative to the overall task. Does a code-LM have an analogous representation?

More precisely: **at the moment a code-LM generates the first token of any intent block (statement or control-flow construct), is the block's sub-goal already linearly encoded in the residual stream — strongly enough to be decoded or verbalized?**

---

## Terminology

**"Scope"** collides with the standard PL meaning (lexical binding / variable visibility). We use **"intent block"** to mean a contiguous code region corresponding to a single sub-goal of the overall task. Equivalently: **"goal region"** in writing, **"intent block"** informally.

An intent block is the code-level counterpart of an instrumental goal in the `g`/`c` framework:
- `g` signal — which predicate / high-level goal is being pursued
- `c` signal — how far along (in-progress vs not) within that goal
- **intent block signal** — what is the specific sub-goal of the current code block, relative to the enclosing task

---

## Example

```python
def bfs(graph, start):
    """Perform BFS on a graph starting from a given node."""
    visited = set()                       # intent: "initialize the visited set"
    queue = [start]                       # intent: "seed the BFS frontier with the start node"
    while queue:                          # intent: "iterate until frontier empty"
        vertex = queue.pop(0)             # intent: "process the frontmost element of the queue"
        if vertex not in visited:         # intent: "guard against revisiting"
            visited.add(vertex)           # intent: "mark vertex as visited"
            queue.extend(
                [node for node in         # intent: "collect unvisited neighbors"
                 graph[vertex]
                 if node not in visited]  # intent: "filter already-visited nodes"
            )
    return visited                        # intent: "return the set of all visited nodes"
```

Intent blocks are **hierarchical** — a token can belong to multiple nested blocks simultaneously. For any probing/training task the **innermost** block label is used by default, but the full ancestor chain is preserved.

Every statement has its own intent, even simple ones (`visited = set()` → "initialize the visited set"). Statements are first-class intent blocks, not just containers for other blocks.

Hierarchy for the `if node not in visited` token inside the comprehension:
```
function: bfs
  └─ while: iterate until frontier empty
       └─ stmt: process the frontmost element of the queue  [vertex = queue.pop(0)]
       └─ if: guard against revisiting
            └─ stmt: mark vertex as visited  [visited.add(vertex)]
            └─ stmt: expand frontier with unvisited neighbors  [queue.extend(...)]
                 └─ listcomp: collect unvisited neighbors
                      └─ listcomp_if: filter already-visited
```

---

## Sub-expression blocks

Standard control-flow boundaries miss sub-expression intent. Two examples:

- `s.islower() and s == s[::-1]` — single line, two semantically independent checks: "verify lowercase" AND "verify palindrome". Both are operands of `BoolOp(And)` in the AST.
- `return len(s) > 0 and all(c.isdigit() for c in s)` — length check goal vs digit-membership goal on one line.

At granularity level L5 (see below), `BoolOp` operands and `Compare` chains are treated as separate intent blocks.

---

## Scope Identification — Design Decisions

These decisions were confirmed before implementation.

**D1 — Input is clean Python code only.**
The scope identification module accepts a well-formed Python string. Extraction of code from model output (e.g. stripping everything outside `<code>…</code>` tags) is a separate, decoupled pre-processing step.

**D2 — Output is a typed dataclass tree.**
The primary output is a root `IntentBlock` dataclass with explicit `parent` / `children` pointers. A helper function `get_blocks_at_char(root, pos)` returns the flat ancestor chain (root → innermost) for any character position, making it easy to iterate during probing.

**D3 — Option A: no condition sub-scopes.**
For control-flow constructs (`while`, `if`, `for`, etc.), the condition/test expression tokens belong to the enclosing scope directly — they do **not** get a dedicated `while_condition` or `if_condition` sub-scope. Only the *body* of a construct can contain nested intent blocks.

Concretely, in `while queue:` the token `queue` is annotated as belonging to the `while` scope. The first sub-scope appears at the first statement inside the body.

**D4 — `else`/`elif` are siblings, not children.**
An `else` or `elif` block is its own `IntentBlock` at the same depth as the parent `if`, not nested inside it. In the AST, `elif` is an `If` node in `orelse` and `else` is a statement list in `orelse` — both are lifted to sibling depth.

**D5 — Statements are first-class intent blocks.**
Every statement (`Assign`, `AugAssign`, `Return`, `Expr`, etc.) is an `IntentBlock` in its own right, even when it contains no nested sub-blocks. `vertex = queue.pop(0)` is as much an intent block as `while queue:`.

**D6 — Hierarchy is always preserved; innermost is the default.**
A token belongs to *all* ancestor scopes simultaneously. The full chain is stored. For probing and training, the innermost block label is used by default, but the complete hierarchy is available.

**D7 — Character spans only; no tokenizer dependency.**
`IntentBlock` stores `char_start` / `char_end` offsets into the source string. Mapping to model token indices is a post-hoc operation left to the caller (via `find_token_range` or offset mapping).

**D8 — All granularity levels implemented together.**
No phased roll-out; L1–L5 are all handled in one implementation, controlled by a `level` parameter.

---

## Scope Identification — AST-Based

Use Python's `ast` module post-hoc on completed generations. Character offsets (`node.lineno`, `node.col_offset`, `node.end_lineno`, `node.end_col_offset`) map AST nodes → token spans via `find_token_range`.

**Configurable granularity levels:**

| Level | Node types added |
|---|---|
| L1 — function | `FunctionDef`, `AsyncFunctionDef` |
| L2 — control flow | + `If`, `While`, `For`, `With`, `Try`, `ExceptHandler` |
| L3 — comprehensions | + `ListComp`, `SetComp`, `DictComp`, `GeneratorExp` |
| L4 — statements | + `Assign`, `AugAssign`, `Return`, `Expr`, `Call` — each statement has its own intent even without sub-blocks |
| L5 — sub-expression | + `BoolOp` operands, `Compare` chains, `IfExp` |

Within L2 and above, **condition** and **body** are treated as separate sub-blocks (different goals). Example: `while queue:` (condition: "check if frontier non-empty") vs the while body (goal: "process next node").

**Timing problem:** `ast.parse` requires complete, valid Python. Since we always label post-hoc on completed generations this is not an issue. For future mid-generation analysis, `tree-sitter` handles partial/broken code.

---

## Goal Labeling

Two approaches, not mutually exclusive:

### Option A — LM self-labeling (preferred to start)

Prompt the same model with the full code + block boundaries, ask it to annotate each block in one phrase:

```
Given this Python function:
<code>
def bfs(graph, start): ...
</code>

For each marked block below, write one short phrase describing the programmer's intent:
  Block 1 [line 5-9]: while queue: ...
  Block 2 [line 7-9]: if vertex not in visited: ...
  ...
```

**Advantage:** Labels come from the same model whose latents are being probed. A mismatch between the explicit verbal label and what is decodable from the latents is itself informative — the model may "know" something in its latents that it does not express when asked directly.

### Option B — Oracle labeling (Claude / GPT-4)

Same prompt structure but using a stronger external model. Better label quality, but expensive and harder to iterate on. Use for final evaluation / ground truth.

---

## Patchscope Verification

### Core idea

Patch the latents at the **scope-opening token** of each intent block into a destination prompt designed to elicit a mid-code intent verbalization. If the block's sub-goal is pre-formed at the opening token, the verbalization should match the oracle label.

### Destination prompt design

The existing `def func(args): """` destination is good for "what function am I implementing" but weak for "what block am I about to open." A better destination:

```python
def func(args):
    # At this point in the code, I am about to:
    """
```

The comment creates a natural slot for verbalizing mid-code intent. The model fills in the docstring position.

### Comparison conditions

| Condition | Patch position | Expected verbalization |
|---|---|---|
| **Scope start** | Opening keyword (`while`, `if`, `[`, first token of return expr) | Should match block intent label |
| **Pre-scope** | Last token of preceding statement | Should verbalize preceding context, not upcoming block |
| **Mid-scope** | Token in middle of block body | Similar to start or stronger |
| **Post-scope** | Token after block closes | Continuation / cleanup |

If the scope-goal is pre-formed at block-open time: start condition verbalizes intent clearly; pre-scope does not; mid-scope is comparable or stronger.

### What to patch

Use `collect_head_steering_vecs` (the selected heads from the trained mask) at the scope-opening token position. Same setup as the existing per-token patchscope visualization in `09_patchscope.ipynb`.

---

## Evaluation

String match won't work for comparing verbalization vs oracle label. Two options:

1. **Embedding similarity** — encode both with a sentence encoder (e.g., `text-embedding-3-small`), compute cosine similarity. Threshold TBD.
2. **Oracle judgment** — prompt Claude/GPT-4: *"Does this verbalization correctly describe the intent of this code block? Answer yes/no with brief justification."* Cleaner for reporting in a paper; interpretable numbers without dependence on a specific embedding model.

Option 2 preferred for final results.

---

## Relationship to Existing Work

| Signal | Question | Measurement |
|---|---|---|
| `g` (goal direction) | Which predicate am I implementing? | Linear probe on residual at pivot |
| `c` (clock) | How far along in the current goal? | Binary/regression probe on in-progress spans |
| **Intent block signal** (this work) | What sub-goal is the current block pursuing? | Patchscope verbalization at block-start token |

The clock probe is about *position* along a known goal. The intent block signal is about *identity* of the goal. In principle, the clock probe is a special case — "is the decoded intent = P+?" — but this work asks for the richer semantic content of the intent, not just a binary label.

---

## Open Questions

- **What exactly is "the start" of a scope?** The keyword token (`while`) vs the condition (`queue`) vs the first body token. The keyword is when the model first commits to opening the block; the condition/test may carry more content about *why*.
- **Hierarchy in probing:** Should the probe be trained on innermost-only labels, or should the full ancestor chain be used (multi-label)? For a first pass, innermost is simpler.
- **Does the destination prompt matter a lot?** Need to ablate: `def func(args): """` vs the `# I am about to:` template vs other designs.
- **tree-sitter vs ast:** For tasks where the LM generates incomplete code (truncated generations), `ast.parse` will fail. Worth switching to `tree-sitter` early.
