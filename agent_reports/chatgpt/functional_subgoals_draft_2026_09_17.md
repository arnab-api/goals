# LLMs Encode Abstract Sub-Goals Before Acting on Them

Working draft for discussion, September 17, 2026. No changes to the Overleaf source.

Other title options:

- **Abstract Sub-Goal Encodings in LLMs** — the shorter, more descriptive option.
- **LLMs Encode Abstract Sub-Goals Before Implementing Them** — makes the connection to code generation explicit.

I favor the working title: it states the finding directly, much like *LLMs Process Lists With General Filter Heads*. Here, “acting” refers to generating the implementation of the next sub-goal.

## Abstract

When an LLM writes code to satisfy a multi-step specification, does it represent the next sub-goal before beginning its implementation? We investigate this question through activation patching and find that code LLMs encode a compact, causal representation of the upcoming operation in the outputs of a small set of attention heads. These representations can be extracted before the implementation begins and used to generate a natural-language description of what the model is about to do. We demonstrate that the encoding is abstract and portable: transplanting it into a different code context can trigger an implementation of the same operation in another programming language. We also test the heads' role in the model's own generation. Preserving their access to task information largely preserves implementation performance when other upper-layer heads lose access, while restricting a broader set of these heads sharply impairs it. We replicate readout and implementation transfer across model families and sizes, and explore readouts of inferred operations in non-code tasks. Our results provide evidence that LLMs encode functional representations of upcoming sub-goals that can guide different concrete realizations of the same operation.

## Introduction

Do LLMs have goals? It is common for human observers to analyze an AI in terms of its “goals,” noting that an AI can seem to “try” to take steps toward an apparent goal even when unable to complete it (Greenblatt et al., 2026). Yet since LLMs have been trained to emit words that mimic humans, we must be careful not to misattribute motivation by simply anthropomorphizing their words. To assess whether apparent AI intentions correspond to the actual computational process, we ask: **does an AI contain an explicit representation of a goal?** In this paper we investigate this question by dissecting LLM execution of multi-step coding tasks. We ask whether, how and where a coding LLM represents goals by localizing representations that control future actions. We ask how to decode a goal representation. We ask how these representations are tracked and processed over a sequence of many tokens. And we ask whether the mechanisms that represent coding goals generalize to other settings.

Consider a programmer asked to implement *merge sort*, a task that decomposes into several sub-tasks: *split the array*, *sort each half*, and *merge the sorted halves*. Before writing the code for the final step, the programmer can already have a coherent goal: “merge the two halves.” This sub-goal admits many implementations, using different loops, variable names, or programming languages. We ask whether an LLM also holds a representation of the next sub-goal before it begins writing the corresponding code, and whether that representation can guide different implementations.

We take a functional approach to this question: a sub-goal representation specifies an intermediate operation and helps guide the computation that carries it out. The distinction between a goal and its implementation is motivated by accounts of intention in terms of its role in practical reasoning. In Bratman's planning theory, a partial plan can specify an end while leaving details of how to achieve it open; the plan then guides further reasoning about those details (Bratman, 1987; Bratman et al., 1988). We investigate a corresponding computational role in LLMs: whether an internal representation of an upcoming sub-goal can guide its realization across different code contexts. This gives us a concrete test of abstraction: can the same extracted representation trigger the same operation when its implementation requires different tokens?

Prior work has shown that LLMs encode information about tokens they have not yet emitted (Pal et al., 2023) and plan toward constraints several tokens in the future (Lindsey et al., 2025). Other work has identified portable representations of input–output functions and filtering predicates (Todd et al., 2024; Sen Sharma et al., 2026). We ask how such abstractions participate in an extended computation: as a model works through a multi-step specification, does it encode which sub-goal it is about to implement?

We investigate this question through activation patching on code tasks with multiple sub-goals. Our main experiments use Qwen2.5-Coder-32B-Instruct, with replications across model families and sizes. We first localize attention heads whose outputs can transfer an operation between programming languages, and then examine what these heads encode at positions before the next implementation begins. Our analysis yields three key findings.

**The next sub-goal is encoded before implementation, and is verbalizable.** We identify a small set of attention heads, which we call *goal heads*, whose contributions at a single token position carry a readable representation of the upcoming sub-goal. We introduce *GoalScope*, a probe that uses these contributions to generate a natural-language description of the operation. The probe receives the source task through the injected activations; the source instructions and code are not supplied as text. This allows us to inspect what the heads encode before the model writes the implementation.

**The encoding is abstract and portable.** We test whether the same extracted representation can guide code generation in another context. An encoding captured from Python can trigger the source operation in a C++ continuation, and vice versa. The operation can therefore be realized through a different sequence of tokens in the destination language. These interventions provide evidence that the representation carries sub-goal information that generalizes across concrete implementations.

**Goal heads mediate the model's implementation of the sub-goal.** We test whether the heads also play a causal role in the model's original generation. Preserving their access to task information largely preserves implementation performance when other upper-layer heads lose access to that information. Conversely, restricting a broader set of goal heads to the attention sink sharply reduces performance, while the same restriction on random heads has little effect. Changing the task information available through the heads' keys and values can also redirect the upcoming operation. Using their value and output projections, we trace where the sub-goal becomes readable in intermediate residual states.

We replicate readout and implementation transfer across model families and sizes, and explore whether heads localized on code can also describe operations inferred from non-code demonstrations. Together, these experiments provide concrete, testable evidence for functional sub-goal representations in LLMs: encodings that precede implementation and can guide different realizations of the same operation.

## Notes for discussion (not manuscript text)

- **Framing.** The central claim is that the representation specifies an upcoming operation, is causally involved in its implementation, and can guide another realization of that operation. The Bratman paragraph motivates the goal–implementation distinction. It does not claim that the experiments establish his broader account of intention, commitment, or persistence. Cross-language transfer also does not establish that implementation details were undecided in the source run.
- **Style.** I kept the existing opening almost verbatim and retained the question → concrete example → mechanism → three findings structure. The filter paper is the closest stylistic reference; the function-vector and dual-route papers also motivate abstractions through specific interventions. GoalScope appears only in the introduction, and OVReader is described without adding another method name there.
- **Scope.** “Before acting” means before generating the relevant implementation, not before encountering the task description or generating any code. The main controlled tasks explicitly list sub-goals. Transfer scores credit nontrivial partial implementations; they are not execution-based correctness scores. The abstract therefore makes no quantitative success claim. Readout heads and the broader set used for the strongest necessity result remain distinguished. Non-code evidence is described as exploratory.

## Citation notes

Author–year citations above are placeholders for the existing bibliography, except for the two Bratman references, which would need adding if retained.

- **Greenblatt et al. (2026):** existing key `greenblatt2026huggingface`, [METR investigation](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/). This preserves the citation in the current opening; the separate Greenblatt screenshot concerns high-level interpretability.
- **Bratman (1987):** *Intention, Plans, and Practical Reason*. Harvard University Press. Consulted the project's literature note for the book. The specific claim about partial plans and their role in further reasoning was checked against Bratman, Israel, and Pollack (1988), *Plans and Resource-Bounded Practical Reasoning*, especially Section 2 ([paper](https://www.emse.fr/~boissier/enseignement/defiia/up5/pdf/Bratman-PlansPracticalResoning.pdf)).
- **Pal et al. (2023):** `pal2023futurelens`.
- **Lindsey et al. (2025):** `lindsey2025biology`.
- **Todd et al. (2024):** `todd2024function`; style reference: [Function Vectors in Large Language Models](../../Overleaf-draft/REF/papers/function.pdf).
- **Sen Sharma et al. (2026):** `sensharma2025filterheads` (the existing key has 2025, but its entry is the ICLR 2026 version); style reference: [LLMs Process Lists With General Filter Heads](../../Overleaf-draft/REF/papers/filter.pdf).
- Additional style reference: [The Dual-Route Model of Induction](../../Overleaf-draft/REF/papers/dual-route.pdf).
