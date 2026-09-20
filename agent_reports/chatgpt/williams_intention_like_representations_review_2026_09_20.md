# Williams (2026): terminology, experimental mapping, and citation recommendations

September 20, 2026. Based on a full reading of the supplied 24-page [PDF](<../../Overleaf-draft/REF/intention-like reps in llms.pdf>), including the alternative explanations and conclusion. Publication metadata was checked against the [publisher's version of record](https://link.springer.com/article/10.1007/s11098-026-02605-y). The mappings to our experiments below are my analysis, not claims Williams makes about our work. No Overleaf files were edited.

**Paper:** Iwan Williams, *Intention-like representations in language models?* **Philosophical Studies**, published online September 17, 2026. DOI: [10.1007/s11098-026-02605-y](https://doi.org/10.1007/s11098-026-02605-y).

## Main recommendation

This is a close conceptual reference for the paper, not merely a source of convenient vocabulary. Williams explicitly uses **Function Vectors in Large Language Models** as a central case study, alongside work on output and planning features. He evaluates those representations against a functional profile of intentions rather than assuming that any internal representation is an intention.

The most useful terms for our framing are:

1. **Causal-representational approach** — the overall approach to locating representations that help produce actions.
2. **Directive causal role** — the connection between a representation's content and the behavior it guides.
3. **Proximal** — the temporal relationship between the captured representation and the imminent operation.
4. **Degrees of abstraction** — the distinction between an operation and the specific tokens implementing it.
5. **Means–end constraints** — a way to discuss how an operation constrains its realization, with stronger planning claims reserved for additional evidence.

Keep **sub-goal representation** as our main empirical term. Use these concepts to explain its functional properties. A concise formulation is:

> We investigate abstract, proximal sub-goal representations that causally guide code generation.

“Proximal” and “abstract” are compatible: an operation can begin immediately while admitting many implementations.

## 1. Is “mind-to-world” what faithfulness readout tests?

**No, not in the technical sense used in the paper.** The phrase is **mind-to-world**, not mind-to-word. It concerns a representation's *direction of fit*, not the process of converting an internal state into words. Williams introduces this distinction in §2.1, PDF page 4, and applies it in §4.1, PDF pages 10–13.

| Concept | Meaning in the paper | Relationship to our experiments |
| --- | --- | --- |
| **Mind-to-world direction of fit** | A descriptive representation is meant to conform to how things are. A prediction can describe the future as well as a belief describe the present. | A successful prediction or faithful description of the next operation does not by itself identify the source representation as directive. |
| **World-to-mind direction of fit** | A directive representation has the function of bringing about an outcome matching its content. | Transfer and native causal interventions are the closer experimental connection: changing the relevant activations or information access changes what the model implements. |
| **Faithfulness of a readout** | This is our measurement, not Williams's name for a direction of fit. | The generated description agrees with the operation in the source continuation. It provides evidence about semantic content. |

The distinction is about **what is supposed to fit what**, not the direction of information flow. It is therefore easy to misread the terminology: an internal state causing external behavior has a *world-to-mind* fit in this vocabulary, even though the causal influence runs from the internal state toward the output.

For our setup, separate three objects:

- **The source encoding:** the candidate representation whose content and causal role we investigate.
- **The verbalization:** a description generated in a separate probe run from the injected encoding.
- **The implementation:** the code whose relation to the representation is tested through ordinary continuation and intervention.

Faithfulness asks whether the description agrees with the implementation. A descriptive prediction, a representation of the task specification, or a directive representation could each support an accurate description. The intervention experiments provide the additional evidence that the representation can guide generation in accordance with that content.

Suggested wording:

> Faithful verbalization reveals the operation encoded by the selected heads; implementation transfer and interventions during native generation test its directive causal role.

This is a useful conceptual explanation of why we have both faithfulness and causality evaluations. It does **not** rename faithfulness as mind-to-world and causality as world-to-mind: the experimental metrics and philosophical distinction are related, but not equivalent.

## 2. Are we decoding proximal intentions?

**Our target is a good candidate for a proximal intention-like representation.** The reason is when it guides action, not the fact that we can verbalize it.

Williams distinguishes proximal intentions, which initiate an imminent action, from distal intentions, which concern action further in the future (§2.2, PDF page 5). For LLMs, he operationalizes this distance using token positions rather than wall-clock time (§4.2, PDF pages 13–14).

At the assignment pivot, the next generated tokens begin implementing the operation represented by the goal-head outputs. Relative to the beginning of that implementation, the representation is proximal. GoalScope supplies a natural-language description of its content. The causal experiments support interpreting it as action-guiding.

Recommended phrases, from most straightforward to most philosophical:

- **A representation of the imminent sub-goal.**
- **A proximal sub-goal representation.**
- **A proximal directive representation.**
- **A candidate proximal intention-like representation.**

I would normally use the first two in results and introduce the third in the conceptual framing. “Decoding proximal intentions” can be an interpretive description once that role has been established; it should not imply that verbalization alone identifies an intention or demonstrates introspective access by the source model.

Three distinctions help:

1. **Distance to onset differs from execution duration.** An operation that starts next token and lasts 100 tokens is still proximal relative to its onset. Sustaining its encoding during execution is a persistence question, not automatically evidence of a distal intention.
2. **Representation depth differs from temporal distance.** An encoding readable at an earlier layer of the same token position is not thereby more distal in Williams's token-position sense.
3. **Four pre-implementation pivots sample a short approach to onset.** They establish pre-implementation availability over those positions, not a broad range of long-horizon commitments.

For a quantitative extension, report capture-to-onset distance in actual tokens, implementation length separately, and how a captured state affects later generation. Continuous reinjection must be distinguished from the enduring effect of an early intervention.

Terminology note: Williams sometimes uses *prospective* as a synonym for *distal*. Our earlier phrase “prospective commitment” meant forward-looking more generally. If we adopt his vocabulary closely, “pre-implementation encoding” avoids that ambiguity.

## 3. Degrees of abstraction and different implementation paths

**This is the most direct match to the cross-language experiments.** Williams emphasizes that an intention can specify an end without specifying every detail of its realization (§2.3, PDF page 5), and discusses representations that generalize across inputs, formulations, and output languages (§4.3, PDF pages 14–15).

Our transfer experiment holds the operation fixed while changing the programming-language context. For example, the same predicate may be implemented with a Python expression or a sequence of C++ tokens. If a captured encoding guides both, its useful content generalizes across those concrete realizations.

Suggested wording:

> We test the abstraction of the encoding by transplanting it between programming-language contexts, where the same operation must be realized through a different sequence of tokens.

The relationship to our earlier terminology is:

- **Abstraction:** what level of detail the representation specifies or generalizes over.
- **Portability:** whether a captured representation remains effective in another context.
- **Plasticity:** whether pursuit of the same goal accommodates different means or conditions.

Cross-language transfer connects all three. They should not be used as interchangeable labels for every result. In particular, different output tokens establish different surface realizations; they do not necessarily establish different algorithms or distinct internal neural pathways. “Implementation” or “realization” is the clearer word unless the experiment directly distinguishes algorithms or circuits.

The relevant empirical claim is that goal information **generalizes across implementation details**. Transfer alone does not establish that the source state contains no implementation information, or that implementation details had not already been selected in the source run.

## 4. Terminology map

Page references below use the PDF viewer's page numbers, not the repeated publisher footer.

| Term | Where | Recommended use in our paper |
| --- | --- | --- |
| **Causal-representational view** | §2, pp. 3–4 | Main framing: identify a representation and test its role in producing behavior. Pair Williams with Bratman where appropriate. |
| **Directive representation / directive causal role** | §§2.1, 4.1, pp. 4, 10–13 | Explain what transfer and native interventions contribute beyond decoding content. This is the strongest terminology to borrow. |
| **Descriptive representation** | §§2.1, 5.2, pp. 4, 19–20 | Name alternatives such as encoding the prompt's task information or predicting the continuation. Useful in experimental interpretation. |
| **Direction of fit** | §2.1, p. 4 | A brief conceptual explanation if needed. Avoid using it as a new label for our evaluation metrics. |
| **Role-function** | §§2.1, 4.1 | Our experiments concern the representation's causal contribution in the tested computation. This supports a functional framing without a claim about training history. |
| **Proper function** | §§2.1, 4.1 | A function selected for through learning or selection. Do not infer this from activation patching alone, or treat checkpoint comparisons as establishing it. |
| **Proximal / distal** | §§2.2, 4.2, pp. 5, 13–14 | Qualify the relation between capture and action onset. Current frontier readouts are primarily proximal. |
| **Abstract / partial** | §§2.3, 4.3, pp. 5, 14–15 | Describe operation-level content that permits different realizations. “Partial” need not mean an incomplete or erroneous readout. |
| **Conduct-controlling** | §2.4, pp. 5–6 | Useful gloss on the action-guiding aspect of commitment. “Causally guides implementation” is clearer in most technical prose. |
| **Stability / inertia** | §§2.4, 4.4, pp. 5–6, 15–16 | Discuss continued influence and resistance to reconsideration. Distinguish this from cosine similarity across tokens. |
| **Goal shielding** | §4.4, p. 15 | A future experimental question: whether the selected goal resists competing, unchosen alternatives. It is narrower than generic robustness to noise. |
| **Means–end constraints** | §§2.5, 4.5, pp. 6–7, 16–18 | Relate an abstract operation to compatible implementation choices. Direct causal tests among goal and implementation representations would strengthen this connection. |
| **Horizontal constraints** | §§2.5, 4.5 | Compatibility constraints between different intentions. Completing a listed sequence alone does not demonstrate these. |
| **Soft constraints / competition** | §4.5, pp. 17–18 | Useful in discussion if multiple candidate goals remain influential. Do not infer their absence from a single greedy readout. |
| **Representational content versus vehicle** | §3.2, footnote 15, p. 9 | Distinguish the operation represented from the particular activation vector carrying it. Relevant to interpreting changing vectors with stable semantic readouts. |

These terms have broader philosophical histories. Williams is a good citation for their application to LLMs, not their original inventor. Bratman, Pacherie, Anscombe, and other original sources should be cited directly when their specific theories do substantive work and have been checked.

## 5. How this refines our existing framework

Williams's five dimensions are **directive function, distality, abstraction, commitment, and planning**. Our pre-commitment/plasticity/persistence framework is a related operational proposal, not a reproduction of his list.

| Our question | Closest part of Williams's framework | Evidence to emphasize |
| --- | --- | --- |
| Encoded before implementation | Proximity/distality plus the action-guiding aspect of commitment | Timing of readout, alongside causal interventions. |
| Different implementations of one operation | Abstraction, with a possible means–end interpretation | Cross-language implementation transfer. |
| Goal information during implementation | Temporal stability | Affinity and semantic readout over the span; maintained causal influence requires additional tests. |
| Recovery from obstacles or distraction | Stability and, for competing goals, shielding | Behavioral recovery combined with interventions on the proposed goal mechanism. |
| Moving on after completion | Regulation of pursuit and coordination between sub-goals | Boundary-signal interventions, with semantic-completion and readout checks. This mapping is our extension, not one of Williams's five named dimensions. |
| DIFF and selection of the next operation | Planning and coordination of sub-goals | Task/progress interventions and contrasts between rival selection mechanisms. |

Williams separates three aspects of commitment: guiding execution, settling between incompatible possibilities, and stability after a choice. Our present causal evidence addresses the first most directly. A readout naming one operation does not establish that competing operations have ceased to influence the model.

This distinction makes the paper more precise without making the introduction apologetic: foreground the functional roles we test, then investigate additional roles as concrete empirical questions.

## 6. Two substantive issues worth engaging

### Directive behavior versus descriptive task information

In §5.2, Williams considers task descriptions, procedural knowledge, and goal-satisfaction tracking as alternatives to the intention interpretation. He argues that these alternatives must explain how their content produces the observed actions, and that transfer into new contexts favors a directive interpretation.

This is directly relevant to our KV/Q experiments. Task-span information, progress information, and the action-guiding output of the heads need not all have the same functional status. Our strongest argument combines the semantic content of the readout, transfer to a new implementation, and causal involvement in the original run.

My methodological qualification: a descriptive task representation coupled to a standing instruction-following mechanism can also have content-specific causal effects when patched. Transfer is therefore strong evidence for an action-guiding role in the system, but does not uniquely settle every decomposition of that system into directive and descriptive components. Williams's discussion is useful for naming and testing alternatives, not as a theorem that patchability proves intention.

### Prompt processing versus generation

In §4.1, Williams emphasizes that predicting already supplied text differs from causing generated text. This is relevant to where we capture and intervene: we should make clear whether an activation describes supplied code, precedes a newly generated implementation, or guides a destination continuation.

However, his blanket separation between prompt-processing representations and directive effects deserves care. The final prompt-position computation normally supplies the logits for the first generated token, and earlier prompt representations can affect generation through cached states. The relevant distinction is whether a representation causally guides the model's own subsequent output, not merely whether its computation was labeled “prefill.” Our pre-implementation capture points and destination interventions let us address that question directly. This observation is our technical interpretation, not a claim that the experiments settle his broader philosophical objection.

## 7. Where and how to cite the paper

### Introduction: establish the approach

Use Williams once near the functional definition, rather than only in Related Work. For example:

```latex
We take a causal-representational approach to this question: we seek internal
representations whose content helps explain the actions they guide
\citep{bratman1987intention,williams2026intention}.
In our coding setting, we ask whether a representation of the next sub-goal
precedes its implementation and can guide different concrete realizations
of the same operation.
```

If the introduction explicitly introduces the temporal/abstraction distinction:

```latex
An intention can be proximal in when it is acted on yet abstract in how it
is realized \citep{williams2026intention}. We investigate this combination
in representations of the next coding sub-goal.
```

These are proposed sentences, not quotations from Williams. A single well-placed citation may suffice if both points appear in the same paragraph.

### Related Work: acknowledge the close conceptual connection

Suggested paragraph:

```latex
\paragraph{Intention-like representations.}
\citet{williams2026intention} evaluates function vectors and output/planning
features against the functional properties of intentions, including their
directive role, temporal distality, abstraction, commitment, and role in
planning. Our experiments examine these questions at the level of the
upcoming sub-goal in multi-step code generation. We verbalize its content,
test transfer across programming languages, and intervene on the attention
heads that mediate implementation.
```

Continue citing Todd et al. for function-vector experiments and Lindsey/Ameisen et al. for the empirical planning findings. Williams provides the philosophical synthesis; he is not the primary source for those experiments. We should not suggest that recognizing intention-like properties in portable vectors begins with our paper.

### Evaluation framing: an optional clarifying sentence

```latex
Verbalization tests the semantic content accessible in the encoding, while
implementation transfer tests whether that content can guide a new
realization of the operation---a directive causal role in the terminology
of \citet{williams2026intention}.
```

This citation is optional if the distinction is already established in the introduction. Avoid a repeated philosophy citation after every causal result.

### Discussion: distinguish the forms of commitment

```latex
\citet{williams2026intention} distinguishes the action-guiding aspect of
commitment from stability and the exclusion of incompatible alternatives.
Our interventions primarily address the first; temporal readouts and
perturbation experiments provide complementary tests of the others.
```

Do not describe proposed perturbation experiments as completed findings. The boundary experiments can then motivate the additional question of when a current sub-goal gives way to the next.

## 8. Bibliography entry

The current bibliography has no Williams entry for this paper. Use the published 2026 journal version, not the 2025 manuscript date or “under review” status still visible on some author/index pages. The publisher currently supplies no volume, issue, or page range in its citation; omit those fields rather than infer them from the PDF's repeated “13” footer.

```bibtex
@article{williams2026intention,
  author  = {Williams, Iwan},
  title   = {Intention-like representations in language models?},
  journal = {Philosophical Studies},
  year    = {2026},
  doi     = {10.1007/s11098-026-02605-y},
  url     = {https://doi.org/10.1007/s11098-026-02605-y}
}
```

Publication metadata: [publisher citation information](https://link.springer.com/article/10.1007/s11098-026-02605-y#citeas). Keep the question mark in the title.

## Suggested vocabulary to use consistently

**Main empirical object:** sub-goal representation.

**Its demonstrated role:** causally guides the upcoming implementation; candidate directive representation.

**Its temporal scope:** proximal to implementation onset.

**Its abstraction:** portable across concrete implementations and programming-language contexts.

**Its broader interpretation:** evidence bearing on a functional account of intention-like representations.

This vocabulary lets us make the positive finding directly while using Williams to specify exactly which aspects of intention the experiments investigate.

Related project report: [functional goal-representation brainstorm](goal_framing_research_brainstorm_2026_09_20.md).
