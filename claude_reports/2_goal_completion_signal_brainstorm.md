# Goal-Directedness Signal `g` and Goal-Completion Signal `c`: A Two-Signal Hypothesis

*Written 2026-03-26. Brainstorming note; not polished for publication.*

---

## The Hypothesis

When an LLM generates code, two separable internal signals may be present in the residual stream:

- **`g` — goal-directedness direction**: A linear direction (or sparse set of attention heads) encoding *which* instrumental goal is currently being pursued. In our setting: which predicate the model is about to implement. This is context-stable — it should fire regardless of whether the model writes `timeout > 0` or `timeout >= 1`, because both implement the same instrumental goal.

- **`c` — goal-completion signal**: A representation tracking *progress* toward the current goal. Stages might include:
  - Has the goal been started yet?
  - Is the model mid-implementation?
  - Is the goal complete?
  - Should the model transition to the next instrumental goal?

Think of it as a simple state machine. `g` is the state label; `c` is the clock that drives transitions.

**The intervention scheduling question**: We currently apply steering (patching head outputs, adding the `g` direction) at a constant strength for all generated tokens. Does `c` inform a smarter schedule — apply strong `g`-intervention at the pivot, then taper off as the goal completes?

---

## Supporting Evidence

### 1. `g` is real: goal-directedness directions exist

**The Linear Representation Hypothesis** (Park, Choe, Veitch — ICML 2024, [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)) formalizes the claim that high-level concepts are encoded as linear directions in activation space, connected to steering via a "causal inner product." This is the theoretical foundation for both the `g` direction and the contrastive probing methodology used in Track 1.

**Representation Engineering** (Zou et al. — arXiv 2023, [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) shows that population-level contrastive directions (LAT) capture cognitive states like honesty, emotion, and power-seeking, and causally steer behavior. Directly validates the methodology of Track 1 (contrastive pairs → mean-diff → causal steering). Does not address the `c` signal or scheduling.

**Function Vectors** (Todd et al. — ICLR 2024, [arXiv:2310.15213](https://arxiv.org/abs/2310.15213)) is perhaps the closest existing work to our `g` signal. A sparse set of mid-layer attention heads carries a compact "function vector" encoding which ICL task the model is pursuing. The FV causally triggers task execution in zero-shot settings. Crucially, FVs are not reducible to output information alone — they encode *intent beyond the answer*. This is our "goal direction vs. competence direction" distinction made concrete. Track 2's head localization approach is essentially finding the code-generation analog of function vectors.

**Emergent Response Planning** (Dong et al. — arXiv 2025, [arXiv:2502.06258](https://arxiv.org/abs/2502.06258)) shows that before generating a single token, LLMs encode response length, number of reasoning steps, and content choices in their hidden states, via linear probing. Behavior attributes stabilize early in layers; content attributes consolidate later. This directly supports the idea that `g` is set at the pivot point before the first generated token.

**Future Lens** (Pal et al. — CoNLL 2023, [arXiv:2311.04897](https://arxiv.org/abs/2311.04897)) shows that single hidden states in mid-layers of GPT-J-6B predict tokens 2+ positions ahead with >48% accuracy, confirmed causally. This is the empirical backbone of our pivot-point probing: the hidden state at the pivot encodes which predicate comes next.

**SCS-Code** (Wendlinger et al. — IEEE EuroS&P 2026, [arXiv:2603.11212](https://arxiv.org/abs/2603.11212)) demonstrates that code LLMs have internal representations of security sub-concepts during generation, extractable by contrastive probing, and steerable to promote secure code. The closest analog to our `g` signal in the code domain — subconcept identity (security vs. insecure pattern) rather than predicate identity, but the same structural claim.

---

### 2. `c` is plausible: state tracking and completion signals exist

**Othello-GPT** (Li et al. — ICLR 2023 oral, [arXiv:2210.13382](https://arxiv.org/abs/2210.13382)) is the gold standard for implicit state tracking. A GPT trained only on legal move sequences develops a linear internal representation of the full board state. Interventions on that representation causally shift predictions. This is `c` in the board-game domain: "what is the current state of the world?" The code analog would be a direction encoding "how much of predicate A has been written so far."

**"(How) Do Language Models Track State?"** (Li, Guo, Andreas — arXiv 2025, [arXiv:2503.02854](https://arxiv.org/abs/2503.02854)) finds two distinct mechanisms LMs use for state tracking: (1) associative scan (composing state transitions hierarchically per layer) and (2) permutation parity as a pre-filter. The associative scan mechanism generalizes better. This suggests `c` may not be a single clean direction but a structured computation across layers.

**FSA Inside Transformers with CoT** (Zhang et al. — arXiv 2025, [arXiv:2502.20129](https://arxiv.org/abs/2502.20129)) shows that with chain-of-thought, transformers implement an implicit FSA with late-layer MLP neurons responsible for state transitions. Crucially: state representations achieve near-100% compression/distinction metrics. This supports our state-machine framing directly. The locus (late-layer MLPs) is different from where `g` likely lives (mid-layer attention heads, per Function Vectors), suggesting `g` and `c` have different circuit footprints.

**States Hidden in Hidden States** (Chen et al. — arXiv 2024, [arXiv:2407.11421](https://arxiv.org/abs/2407.11421)) demonstrates implicit discrete state representations (IDSRs) in multi-digit arithmetic — the model tracks "how far along in the calculation" without chain-of-thought. The IDSRs are real but described as "far from lossless" in current open-source models, causing arithmetic errors. Cautionary note: `c` may exist but be noisy enough to make scheduling unreliable.

**Mechanistic Analysis of Multi-Step Reasoning** (Brinkmann et al. — arXiv 2024, [arXiv:2402.11917](https://arxiv.org/abs/2402.11917)) shows transformers solving symbolic multi-step reasoning implement "depth-bounded recurrent mechanisms that store intermediate results at selected token positions." This is a concrete proposed mechanism for `c`: intermediate progress is written to specific KV positions in the sequence.

**LLM Internal Code Correctness Representation** (Ribeiro et al. — ICSE 2026, [arXiv:2512.07404](https://arxiv.org/abs/2512.07404)) shows that LLMs encode an internal signal about code correctness (extractable by contrasting hidden states between correct and incorrect outputs). This is a *competence* signal rather than a completion signal — but it demonstrates that multiple distinct signals coexist in the residual stream during code generation. The representational space is rich enough for both `g` and `c`.

---

### 3. Scheduling is the gap: adaptive steering is nascent

**SADI** (Wang et al. — ICLR 2025, [arXiv:2410.12299](https://arxiv.org/abs/2410.12299)) adaptively scales steering element-wise based on the direction of input semantics, outperforming static steering. Closest existing work to scheduling, but adapts to *input* semantics rather than *generation-time* state (i.e., it's fixed once for the whole generation, computed from the prompt). It does not read the evolving `c` signal during generation.

**FASB** (Cheng et al. — NeurIPS 2025, [arXiv:2508.17621](https://arxiv.org/abs/2508.17621)) is the most direct analog: it tracks internal LLM states *during* generation, dynamically determining necessity and strength of intervention token-by-token. It also has a backtracking mechanism. The key difference from your hypothesis: FASB uses the full generation so far as context holistically, not a dedicated clean `c` probe. Your hypothesis proposes a cleaner two-signal decomposition: find `c` as a specific linear direction, read it cheaply at each step, and use it to gate `g`-steering.

**Progress Measures for Grokking** (Nanda et al. — ICLR 2023, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)) defines smooth progress measures by reverse-engineering a learned algorithm. This is the methodological blueprint for finding `c`: if you reverse-engineer the "check timeout" subroutine, you can define stages (has the model started writing the condition? written `timeout`? written `> 0`?) and track them as continuous linear signals.

---

## Contradictions and Skeptical Evidence

**The Pitfalls of Next-Token Prediction** (Bachmann & Nagarajan — ICML 2024, [arXiv:2403.06963](https://arxiv.org/abs/2403.06963)) is the strongest argument against the hypothesis. Teacher-forced NTP is inherently myopic — the model cannot see downstream consequences of current choices, leading to structural failures even on minimal planning tasks. Both Transformers and Mamba fail. If NTP is genuinely myopic at the objective level, stable goal representations should not emerge. **Counter-argument**: the paper's proposed fix (multi-token prediction) is necessary for short-context planning, but large code-pretrained models may have learned implicit multi-token planning through scale and data patterns. Papers 4 and 5 above show empirically that such anticipatory representations do emerge in large models. The tension is resolvable empirically in our setting.

**LLMs as Markov Chains** ([arXiv:2410.02724](https://arxiv.org/abs/2410.02724)) points out that autoregressive transformers are mathematically Markov chains over their context window. This is technically correct but does not preclude stable internal representations — the Markov "state" is the full KV-cache, which can encode `g` and `c` as linear functions. The Markov framing suggests each token is generated locally from history, which is *consistent* with `g` and `c` being read from that history at each step.

**Reintroducing Markov States** ([arXiv:2603.19987](https://arxiv.org/abs/2603.19987)) argues LLMs currently fail to distill noisy token history into clean internal states, and proposes explicit Markov states as a fix. This predicts that `g` and `c` signals, if they exist, are noisy. Consistent with Paper 10 (IDSRs are "far from lossless"). **Implication for our research**: use robust extraction methods (mean-diff, contrastive hinge, multiple instances) rather than assuming clean separability.

**Evaluating the Goal-Directedness of LLMs** (Everitt et al. — arXiv 2025, [arXiv:2504.11844](https://arxiv.org/abs/2504.11844)) finds most models are not fully goal-directed behaviorally. Models can have the capability without deploying it toward the goal. **Important distinction**: this is a behavioral result, not mechanistic. A model could have a stable internal `g` direction but fail behaviorally due to competing signals. However, it weakens the claim that `g` causally dominates generation.

---

## Brainstorm: How Would `c` Schedule the Intervention?

### The State Machine Model

Suppose the model is implementing `is_palindrome(s)` after already completing `is_lower(s)`. The state machine for implementing `is_palindrome` might look like:

```
PENDING → STARTED → PARTIALLY_DONE → COMPLETE → TRANSITION
```

- **PENDING**: At the pivot. `g` fires strongly (this is where we observe it). `c ≈ 0`. Apply maximum intervention.
- **STARTED**: The model has written `return s ==`. `g` still fires, `c` is rising. Maintain intervention.
- **PARTIALLY_DONE**: Written `return s == s[:`. Intervention may be creating incoherence — the model "knows" where it's going now. Taper.
- **COMPLETE**: Written the full `return s == s[::-1]`. `g` signal collapses. `c ≈ 1`. Zero out intervention to avoid interfering with next-predicate decision.
- **TRANSITION**: Model moves to next instrumental goal. `g` changes identity. `c` resets.

The scheduling policy: `strength(t) = g_coeff × f(c(t))` where `f` is a decreasing function (e.g., `f(c) = 1 - c`, or sigmoid-gated).

### Where to Look for `c`

Based on the literature:
- **Late-layer MLP neurons** (Paper 9: FSA in CoT transformers): State transition neurons
- **Specific token positions** (Paper 11: multi-step reasoning): Intermediate results stored in KV at designated positions — in code generation, natural positions are the end-of-line tokens `\n` after each statement
- **Residual stream at `\n` positions**: The end of each completed code line is a natural "checkpoint" where `c` might update

### The Practical Challenge

To use `c` for scheduling, we need to probe `c` cheaply at each generation step — one extra linear projection per step. If `c` is a single linear direction (the optimistic case), this is a trivial overhead. If `c` is a more complex structured computation (as Papers 8, 9 suggest), it may require a more expensive circuit read.

### A Minimal Experiment

Before building a full `c` probe, a simpler test:

1. Take generations where the model correctly completes predicate A. Mark the token positions: before starting A, mid-A, completing A.
2. Extract hidden states at these marked positions.
3. Train a 3-class linear probe: PENDING / IN_PROGRESS / COMPLETE.
4. Check if the probe generalizes across: (a) different orderings (A before B vs. B before A), (b) different implementations of A (`==` vs. `is`), (c) different functions.

If the probe generalizes, `c` is a genuine state signal. If it doesn't, `c` is context-dependent and scheduling based on it would require a context-conditional probe.

### Does Constant-Strength Intervention Actually Hurt?

It's worth asking: does the constant intervention create incoherence mid-generation? Possible effects:
- **Mid-goal**: The head suppression might be redundant (the model already "knows" it's writing predicate A based on what it has generated) or slightly helpful (reinforcing the direction against drift).
- **Post-goal**: If we keep patching `g` for predicate A while the model is now writing predicate B, we're actively pushing against the new correct goal. This would show up as lower pass rates on *later* predicates. This is testable.

**Proposed diagnostic**: Run the patched generation on a two-predicate function. Score each predicate's implementation independently. If predicate A improves but predicate B degrades with constant `g`-patching, that's strong evidence that `c`-scheduled intervention is needed.

---

## Summary Table

| Paper | Supports `g`? | Supports `c`? | Scheduling? | Contradicts? |
|---|---|---|---|---|
| Linear Representation Hypothesis | Strong | — | — | — |
| Representation Engineering | Strong | — | — | — |
| Function Vectors | Strong (attention heads) | — | — | — |
| Emergent Response Planning | Strong | Moderate | — | — |
| Future Lens | Strong | Moderate | — | — |
| Extracting Paragraphs | Moderate | — | — | — |
| Othello-GPT | — | Strong | — | — |
| (How) Do LMs Track State? | — | Strong | — | — |
| FSA in CoT Transformers | Moderate | Strong | — | — |
| States Hidden in Hidden States | — | Strong (noisy) | — | Noisy `c` |
| Mechanistic Multi-Step Reasoning | — | Moderate | — | — |
| SADI Dynamic Steering | — | — | Moderate (input-adaptive) | — |
| FASB Steering + Backtracking | — | — | Strong (generation-adaptive) | — |
| SCS-Code | Strong (code domain) | — | — | — |
| LLM Code Correctness | — | Moderate (competence) | — | — |
| Pitfalls of NTP | — | — | — | Strong (myopia) |
| LLMs as Markov Chains | — | — | — | Weak |
| Reintroducing Markov States | — | — | — | Moderate |
| Evaluating Goal-Directedness | — | — | — | Moderate (behavioral) |
| Progress Measures / Grokking | — | Moderate (method) | — | — |

---

## Open Questions

1. **Do `g` and `c` live in different circuit components?** The literature suggests `g`-like signals (function vectors) are in mid-layer attention heads; `c`-like signals (state transitions) are in late-layer MLP neurons. If true, they can be read and steered independently.

2. **Is `c` a single linear direction or a more complex computation?** The optimistic (and most experimentally tractable) case: a single direction in the residual stream at each token position increases monotonically from PENDING to COMPLETE. The pessimistic case: it requires reading specific MLP neuron states.

3. **Does constant `g`-intervention hurt later goals?** The proposed diagnostic (two-predicate function, score each predicate independently) would test this directly.

4. **Can we find `c` using the same contrastive methodology as `g`?** Contrastive pairs for `c`: (hidden state mid-predicate) vs. (hidden state at the pivot, before predicate starts). Both from the same model run, different time steps. The direction should fire as completion increases.

5. **Is the scheduling benefit worth the implementation complexity?** If `c` is noisy (Papers 10, 18), a scheduled intervention might not be clearly better than a constant-but-tapered intervention (e.g., just decay linearly after a fixed number of tokens). The simple baseline — apply full strength for first K tokens, then zero — might already capture most of the benefit.

---

## Conclusion

The two-signal hypothesis (`g` + `c`) is well-motivated by the mechanistic interpretability literature. `g` has strong direct support (Function Vectors, RepE, Future Lens); `c` has strong support in analogous domains (Othello-GPT, FSA-in-CoT, multi-step reasoning) but has not been directly demonstrated for code subgoal tracking. The scheduling question — using `c` to gate `g`-intervention — is a genuine research gap. FASB (NeurIPS 2025) is the closest existing work, but it uses holistic generation state rather than a clean two-signal decomposition.

The key experiment to run next: the two-predicate diagnostic (constant `g`-patching, score each predicate independently) to test whether constant intervention hurts later goals. If it does, that is the strongest motivation for `c`-scheduled steering.
