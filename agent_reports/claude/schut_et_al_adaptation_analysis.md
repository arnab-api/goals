# Analysis: Adapting Schut et al. for Code LM Goal Representations

**Date**: 2026-02-05
**Paper**: Schut et al., "Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero"
**Context**: Evaluating a plan to adapt the dynamic concept extraction framework (Section 4.1.2) for extracting goal representations from code LMs during code generation.

## The Plan

| Schut et al. (Chess/AZ) | Proposed Adaptation (Code LMs) |
|---|---|
| AlphaZero policy-value network | Autoregressive Code LM (e.g., Qwen-Coder) |
| Board positions x_t | Partial token sequences (prompt + generated tokens) |
| MCTS rollouts in game tree | MCTS over token tree (`MCTSSearch`) |
| Optimal rollout X⁺ (highest value/visits) | Green paths (pass unit tests) |
| Subpar rollout X⁻ (lower value/visits) | Red/orange paths (fail unit tests) |
| Dynamic concept c | Code completion goal (e.g., "implement palindrome check") |
| Active planning (Scenario 1) | The model pursuing a goal across different prefills |
| Hidden representations z_{t,l} = f_l(x_t) | LM hidden states at layer l, position t |

## Potential Issues

### 1. State Representation Mismatch (Fundamental)

This is the most significant structural difference. In AlphaZero:
- Each state $x_t$ is a **self-contained** board position (8x8 grid)
- $z_{t,l} = f_l(x_t)$ is computed **independently** for each board state
- States at different depths are structurally identical (always a board position)

In an autoregressive LM:
- The "state" at token t is the **entire prefix** (prompt + all tokens up to t)
- $z_{t,l} = f_l(\text{tok}_1, \text{tok}_2, ..., \text{tok}_t)$ depends on **all** previous tokens, not just the current one
- States at different depths have **different lengths** and encode fundamentally different things (the hidden state after `n` vs after `n % 2 ==` serves different purposes)

**Implication**: When Schut et al. impose the constraint $v · z⁺_{t,l} >= v · z⁻_{t,l}$ for all t <= T, they're comparing representations of structurally similar objects (board positions) at each depth. In the code setting, you'd be comparing hidden states at position t along two diverging token paths. After the divergence point, the two paths have different context, which is fine conceptually, but the hidden states encode much more heterogeneous information (syntax state, semantic meaning, attention patterns to distant tokens) than chess board representations.

**Suggestion**: Consider extracting $z_{t,l}$ only at the **last token position** of each path prefix (the "current generation frontier"), rather than at every position along the path. This is more analogous to how AZ processes each board state as a whole.

### 2. X⁺/X⁻ Definition: External Evaluation vs. Internal Value

In the paper, optimal vs. subpar rollouts are defined by AZ's **own** value estimates and visit counts. The concept vector captures what the model *internally represents* as distinguishing good from bad plans.

The proposed setup defines X⁺/X⁻ via **external unit tests** (green vs. red/orange). This measures something subtly different:
- **Paper**: "What does the model think separates good from bad plans?" (model's own preferences)
- **Proposed setup**: "What separates actually-correct from actually-incorrect code in the model's representations?" (ground truth correctness)

These overlap when the model is well-calibrated (it prefers correct solutions), but diverge when:
- The model assigns high probability to an incorrect path (confident but wrong)
- The model assigns low probability to a correct path (underselling a good solution)

**This could actually be a feature**: If the research question is "does the model have representations that correspond to correct goal pursuit?", external evaluation is the right signal. But if the question is "does the model know where it's going?", you'd want the model's own ranking (e.g., cumulative log-probability or MCTS visit counts as in the original paper).

**Suggestion**: Try both variants:
1. X⁺/X⁻ defined by unit test results (current plan)
2. X⁺/X⁻ defined by the model's own visit counts / log-probs from MCTS (closer to original paper)

Comparing the resulting concept vectors would itself be informative about whether the model "knows" what's correct.

### 3. Circularity Risk: The Concept IS the Goal

In chess, concepts (piece activity, king safety, pawn structure) are **sub-strategies** that contribute to the terminal goal (winning). The concept is distinct from the objective. This is what makes concept discovery interesting -- you find intermediate strategic ideas the model uses.

In the proposed setup, the "concept" being extracted IS the goal itself: "implement this function correctly." This risks circularity:
- Of course correct completions have different representations than incorrect ones
- A linear probe distinguishing them is essentially a correctness classifier, not a "goal representation"

**The interesting question is whether this concept generalizes**:
- Does a concept vector extracted from `is_odd` also separate correct from incorrect paths for `is_palindrome`?
- If yes, you've found a general "goal pursuit" or "code correctness" direction -- genuinely interesting
- If no, you've just found problem-specific features (not a "goal" representation)

**Suggestion**: Frame the experiment around **cross-problem generalization**. Extract v from problem A, test constraints on problem B. This is the key test of whether you've found a general goal representation vs. a problem-specific classifier.

### 4. Missing Value Function and Scenario 3 Filtering

The paper filters out "random" concepts (Scenario 3) by checking that optimal and subpar rollouts have **sufficiently different** value estimates. If both paths end with similar values, the concept contrast is noise.

LMs don't have an explicit value head for code correctness. This means:
- You can't filter Scenario 3 internally
- Some green-vs-red contrasts might be "random" (the paths are superficially different but the model treats them similarly)

**Suggestion**: Use the model's cumulative log-probability (or MCTS visit counts) as a value proxy. Filter out contrast pairs where both paths have similar log-probs / visit counts.

### 5. Linearity Assumption for Abstract Goals

The paper validates linearity on relatively concrete chess concepts (piece presence achieves 0.99 constraint accuracy, Table 3). More abstract concepts (Stockfish evaluations) already drop to 0.75-0.77.

Code completion goals are highly abstract and compositional:
- "Check if palindrome" involves reversal, comparison, iteration, edge cases
- The "goal" is not a single feature but a complex conjunction of sub-goals
- This is closer to Stockfish-level complexity (or beyond) than to piece detection

**Risk**: The convex optimization might not find a sparse v with good constraint satisfaction for abstract code goals. You might get either:
- Very dense v (large L1 norm, violating the "unit of knowledge" minimality)
- Low constraint accuracy on test set

**Suggestion**: Start with simple goals where linearity is more plausible (e.g., "return True/False" as a static concept, "use modulo operator" as a simpler sub-goal). Build up to full function-level goals.

### 6. Scale of Contrast Data

The paper uses:
- 17,184 chess positions for spectral analysis
- Multiple subpar rollouts at different depths (Equation 5, up to T-bar = T-5)
- The convex optimization needs sufficient constraints to be well-conditioned

The current setup:
- ~50-500 MCTS iterations per problem
- Trees with 8-20 depth (= 8-20 tokens)
- A modest number of problems in `incomplete_code.json`

Each (z⁺, z⁻) pair at each depth generates one constraint. For a 14B model, hidden dimensions are ~5120. You need substantially more constraints than dimensions to avoid underdetermined optimization.

**Rough calculation**: If you have 10 problems, each with one green path and 5 red paths, and average depth 15, you get ~10 * 5 * 15 = 750 constraints for ~5120 dimensions. This is severely underdetermined.

**Suggestions**:
- Use many more problems or generate multiple MCTS trees per problem (different random seeds / temperatures)
- Consider dimensionality reduction (PCA on hidden states) before the convex optimization
- The paper works with d_l up to ~16K but has 17K+ chess positions -- their constraint count is adequate

### 7. Where to Probe: Token Position Ambiguity

In chess, there's one representation per board state. In LMs, each token position has its own hidden state. When extracting z_{t,l} along an MCTS path:

- **Option A**: Use the hidden state at the **t-th generated token** position (maps directly to MCTS depth t)
- **Option B**: Use the hidden state at the **last position** after feeding the path prefix (like a "state summary")
- **Option C**: Pool/average across positions

Option A is the most direct analog to the paper. But note that in an LM, the hidden state at position t already "knows" about everything before it (via attention). So each z_{t,l} is already a function of the full path prefix, unlike chess where x_t is just one board state fed independently.

**Suggestion**: Option A is most principled. Run the full prefix through the model once and extract z_{t,l} at each generated-token position.

## What Maps Well (Positive)

Despite the issues above, several things align nicely:

1. **Tree structure**: The MCTS token tree IS analogous to AZ's game tree. Paths diverge at choice points (tokens), just like moves.

2. **Depth-based contrasts**: The paper's key innovation (Figure 3) -- finding subpar rollouts at different depths -- maps directly to the token tree. At each internal node, you can find the green child's subtree vs. the red child's subtree.

3. **Prefill probing setup might be even better**: Different prefills create different "starting positions" for the same goal. This is analogous to testing the same concept across different chess positions. A concept vector that generalizes across prefills (aligned, adversarial, inconsequential) would be strong evidence of a consistent goal representation.

4. **Existing infrastructure**: `evaluate_mcts_tree()` already labels paths as green/orange/red, and `get_all_paths()` extracts complete paths. The main addition needed is hidden state extraction during MCTS expansion.

## Recommended Starting Point

Rather than jumping straight to dynamic concepts (Section 4.1.2), consider a **two-phase approach**:

### Phase 1 - Static concept validation
- For a set of code completion problems, collect many (prefix, completion) pairs
- Label them as correct/incorrect via unit tests
- Extract hidden states at the last prefix token
- Run the convex optimization (Equation 2) to find a concept vector
- Test cross-problem generalization (train on problems A,B,C -- test on D,E,F)
- This validates whether linearity holds at all for code goals

### Phase 2 - Dynamic concepts via MCTS
- Only proceed if Phase 1 shows cross-problem generalization
- Use the MCTS tree structure for the depth-based contrast (Equation 5)
- Extract hidden states at each depth along green vs. red paths
- Look for concept vectors that are consistent across depths (active planning signal)

This way you don't invest heavily in the MCTS-based pipeline before knowing if the fundamental linearity assumption holds for code goals.
