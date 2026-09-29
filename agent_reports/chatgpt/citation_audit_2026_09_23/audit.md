# Citation audit and Appendix A revision

September 23, 2026. Audited the current manuscript reachable from `main.tex`, including the appendix, figure captions, and table captions. Read the LaTeX sources and rebuilt the PDF. This audits citation placement and interpretation; it does not reproduce the paper's experiments.

## Scope and outcome

- Before revision: **41 reachable source files, 38 citation commands, 28 distinct citation keys**. Repeated renderings of a caption are not counted as additional source occurrences. Commented-out text and unreferenced source files are excluded.
- After the Appendix A draft: **57 citation commands, 32 distinct keys**, all present in the bibliography. Source inventories are saved in [inventory.json](inventory.json) and [inventory_after.json](inventory_after.json).
- Changed manuscript text only in [Text/06_related_works.tex](../../../Overleaf-draft/Goals-Overleaf/Text/06_related_works.tex). Added one bibliography entry, `chen2024selfie`, to `references.bib`. Existing disputed entries and citations elsewhere remain untouched.
- One clear wrong-work citation: activation addition cited for the provenance of an emergently misaligned fine-tune. Several additional citations are relevant but used for conclusions stronger or different than the source supports.
- The new Related Work section is approximately 850 words. It removes the untraceable Greenblatt interview entry from active use and adds verified discussion of functional intention, cognitive goal memory, lenses, SelfIE, task vectors, and causal abstraction.

**Verification limits.** I checked claim-relevant passages of primary papers, supplied PDFs, official proceedings, and author/publisher sources. This was not a cover-to-cover rereading of every paper. For Bratman's 1987 book, the assessment uses the publisher's description, Williams's direct discussion of the book, the project notes, and the original 1988 Bratman–Israel–Pollack paper; I did not obtain the full 1987 book. For Fodor I verified metadata and publisher framing, not the full argument. Goschke and Marsh were checked against the original studies' abstracts/first-page material and author records. Greenblatt's exact interview source remains unresolved. These limitations are reflected below.

## Citation issues requiring action outside Appendix A

### 1. Wrong Turner paper in the emergent-misalignment caption

**Location:** `Text/Appendix/non-code.tex:139`, caption of `fig:em_goalscope_all`.

The caption credits `turner2023steering` for a Qwen model fine-tuned on risky financial advice. That paper is by **Alexander Matt Turner et al.** and introduces inference-time activation addition. The relevant model-organisms paper is by **Edward Turner et al. (2025)**, already present as `turner2025modelorganisms`. Its §3.1 describes the financial-advice fine-tunes, including Qwen2.5-32B-Instruct. Replace the caption's 2023 key with the 2025 key; retain Soligo if the particular model/analysis also relies on that work. [Activation-engineering paper](https://arxiv.org/abs/2308.10248), [model-organisms paper, §3.1](https://arxiv.org/html/2506.11613v1#S3.SS1).

The new Related Work uses the 2023 paper for activation steering and the 2025 paper for EM model organisms.

### 2. The three-part definition is a synthesis, not Bratman's definition

**Location:** `Text/01_intro.tex:15`.

“Following Bratman's planning theory, we define goal ... as having three attributes” risks attributing this exact package to Bratman. His account concerns intentions and their roles in planning, including commitment, partial plans, and constraints on deliberation. Williams separately analyzes directive function, proximity, abstraction, commitment, and planning. Neither provides our exact operational definition. Suggested opening: **“Drawing on functional accounts of intention and goal-directedness, we examine three properties of a candidate sub-goal encoding.”** [Bratman, Israel, and Pollack (1988), §§2–3](https://www.emse.fr/~boissier/enseignement/defiia/up5/pdf/Bratman-PlansPracticalResoning.pdf), [Williams (2026), §§2.1–2.5](https://doi.org/10.1007/s11098-026-02605-y).

Relatedly, readable information before action does not establish all senses of commitment. Williams distinguishes action-guiding influence, settling among alternatives, and resistance to reconsideration. Our readout/transfer findings address parts of that profile. The new Appendix A states this distinction.

### 3. Lee and McShea do not define persistence as retention of a neural encoding

**Location:** `Text/01_intro.tex:15`, persistence criterion.

Their persistence metric concerns bias toward a goal following perturbation; plasticity concerns reaching a goal across starting conditions. They explicitly do not propose these as an exhaustive definition. Cross-language implementation is a reasonable adaptation of the plasticity idea, but cosine similarity through execution is our distinct temporal measure. Cite them for behavioral persistence/plasticity and label temporal representational continuity as our operationalization. [Original paper, §2 and discussion](https://sites.duke.edu/mcshearesearch/files/2023/11/Operationalizing-goal-directedness-Lee-and-McShea-2020.pdf).

### 4. Human intention-memory findings do not establish neural release in this LM

**Location:** `Text/persistance.tex:21`.

Goschke and Kuhl find facilitated recognition of material related to pending intentions. Marsh et al. find lexical-decision differences for completed versus pending or partially completed scripts. These are relevant comparisons, but their operational measure is accessibility of associated material, not continuous neural activation or representational cosine. Describe a resemblance in accessibility/continuity, rather than treating the LM affinity curve as the same measured phenomenon. [Goschke author-institution record and original abstract](https://fis.tu-dresden.de/portal/en/publications/representation-of-intentions%28a976e3ee-e4e1-402f-9b51-e9e5e1a7a422%29.html), [Marsh original first-page material](https://www.researchgate.net/publication/232558832_Activation_of_Completed_Uncompleted_and_Partially_Completed_Intentions).

The subsequent claim that the encoding is **released to prevent interference** is a larger problem. Altmann and Trafton model activation, decay, retrieval cues, and interference from old goals; decay can help prevent perseveration. They do not establish a completion-triggered release mechanism in this LM. Their model also questions special goal stores that remain active without maintenance. Recast the LM result as reduced similarity compatible with a change of operation, and reserve the functional purpose of release for a hypothesis. [Altmann and Trafton (2002), pp. 45–46, 61–64](https://www.interruptions.net/literature/Altmann-CogSci02.pdf).

### 5. “Until achieved” omits the release conditions in Cohen and Levesque

**Location:** `Text/persistance.tex:19`.

“Keeps it active until ... achieved” is a simplified gloss. The formal persistent-goal definition depends on the agent's beliefs, with termination upon believed achievement or impossibility; it is not a claim about constant neural activity. Suggested wording: **“Formal accounts characterize intentions through commitments that persist until appropriate release conditions, such as believed achievement or impossibility.”** [Cohen and Levesque (1990), §4, Definition 4.1](https://ai.stanford.edu/~epacuit/classes/lori-spr09/cohenlevesque-intention-aij90.pdf).

### 6. Dennett versus Fodor is an unnecessarily strong philosophical opposition

**Location:** `Text/01_intro.tex:13`.

The Dennett citation fits the opening about predicting behavior using goal/belief attribution. “Siding with Fodor” then implies that seeking internal representations decides a philosophical dispute. Dennett himself distinguishes successful intentional prediction from the further empirical question of the machinery implementing it. Fodor is relevant to representational explanation, but this experiment does not test his full account of mental representation. Replace the allegiance language with the empirical question, or introduce Fodor as broader motivation. [Dennett, chapter 2, concluding discussion](https://hri-class.github.io/mae6710/files/dennet1987-ch2.pdf), [Fodor publisher record](https://mitpress.mit.edu/9780262560528/psychosemantics/).

### 7. Greenblatt interview citation lacks a traceable source

**Original location:** `Text/06_related_works.tex:17`; **entry:** `greenblatt2024highlevel`.

The supplied [screenshot](../../../Overleaf-draft/REF/ryan_greenblatt.jpg) supports the idea about intervention-based, higher-level explanations. It does not identify a publication title, date, interview, URL, or timestamp. The bibliography title appears to be a description of the remarks, and I could not verify its 2024 date or exact source. This is an attribution/provenance problem, not evidence that Greenblatt never said it. The rewritten section uses traceable causal-mediation/abstraction literature instead. The original entry is left in the bibliography but is no longer cited by the revised section. To restore it, supply the actual interview title, date, URL, and timestamp.

### 8. Bibliographic cleanup

- **Two entries for the filter paper:** `sensharma2025filter` and `sensharma2025filterheads` generate separate 2025 and 2026 references for the preprint and conference versions. Both are real versions of the same work. Standardize on the ICLR entry unless intentionally contrasting versions. Main-text citations were left unchanged. [Author project page](https://filter.baulab.info/).
- **Khandelwal venue/link:** `khandelwal2026how` claims an ICML 2026 workshop venue. The linked OpenReview PDF identifies itself as a NeurIPS 2026 submission; the author page gives the 2025 arXiv preprint. I verified the paper and its 18-task collection, but not the claimed workshop provenance. Prefer a verified arXiv citation until that venue record is supplied. [Author bibliography](https://apoorvkh.com/), [current paper](https://arxiv.org/html/2510.01685), [linked OpenReview PDF](https://openreview.net/pdf?id=iXnaLHVuAj).
- **Bratman publisher:** missing; BibTeX warns. The 1987 edition is Harvard University Press. The publisher page for the CSLI reprint gives 1999, which should not be substituted as the original edition. [Reprint publisher record](https://press.uchicago.edu/ucp/books/book/distributed/I/bo3629095.html).
- **Turner 2023 title/version:** the entry uses the earlier activation-addition title; the current version is titled *Steering Language Models With Activation Engineering*. Pin the earlier version or update metadata consistently. This is separate from the wrong-use issue above. [Versioned source](https://arxiv.org/abs/2308.10248).
- **Mueller author/publisher fields:** the official record lists all 13 authors, including family name “Sen Sharma.” The entry truncates the author list and includes a postal address/ellipsis in `publisher`. Todd's entry also parses the family name as “Sharma.” These are bibliographic cleanup issues, not failures of the cited claims. [Official Mueller record](https://aclanthology.org/2026.cl-1.10/).
- **Other version metadata:** Betley now also has a Nature publication; citing the 2025 preprint remains legitimate. Wang's key contains 2022 while the entry correctly uses ICLR 2023; key names do not determine citation year. There is a duplicate **unused** bibliography key, `petroni2019language`; this is outside the active manuscript audit and did not prevent this build.

## Complete ledger of the 28 originally cited keys

“Supported” means that the source supports the particular use in this manuscript, not that it proves the manuscript's new experimental interpretation. Locations refer to source files before the Appendix A rewrite; repeated uses share a row, while the inventories retain every occurrence.

| Key | Locations and reason for use | Assessment and checked source |
| --- | --- | --- |
| `dennett1987` | Introduction: intentional attribution for prediction | **Supported.** Chapter 2 distinguishes the stance from an account of implementation. [Chapter](https://hri-class.github.io/mae6710/files/dennet1987-ch2.pdf). |
| `fodor1987` | Introduction: taking a representationalist side | **Qualified; limited primary access.** Relevant tradition, but stronger opposition than needed; publisher metadata checked. See issue 6. [Publisher](https://mitpress.mit.edu/9780262560528/psychosemantics/). |
| `bratman1987intention` | Introduction's definition/commitment/persistence; temporal subsection's idealized agent | **Relevant, overstated attribution.** The exact three-part definition and mere early decodability are ours. Book access limitation described above. See issues 2 and 5. [Original authors' 1988 account](https://www.emse.fr/~boissier/enseignement/defiia/up5/pdf/Bratman-PlansPracticalResoning.pdf). |
| `williams2026intention` | Introduction: pre-commitment | **Relevant but stronger than the test.** Checked supplied full PDF, especially §§2.1–2.5 and 4. Commitment includes more than an early action-related state. [Paper](https://doi.org/10.1007/s11098-026-02605-y). |
| `lee2020operationalizing` | Introduction: plasticity and persistence | **Plasticity motivation fits; temporal persistence needs qualification.** Behavioral perturbation/trajectory account. See issue 3. [Paper](https://sites.duke.edu/mcshearesearch/files/2023/11/Operationalizing-goal-directedness-Lee-and-McShea-2020.pdf). |
| `ghandeharioun2024patchscope` | Introduction, GoalScope method, old Related Work: natural-language readout through patching | **Supported.** Correct method lineage; official name is plural *Patchscopes*. It does not itself establish the correctness of our head selection or probe. [ICML paper](https://proceedings.mlr.press/v235/ghandeharioun24a.html). |
| `vig-etal-2020` | Notation: activation patching/causal mediation | **Supported.** Interventions on neural mediators are central to the study. [Original paper](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html). |
| `mueller2026quest` | Notation: mediation-based interpretability methods | **Supported.** Survey is appropriate methodological context, not an origin claim for patching. [CL record](https://aclanthology.org/2026.cl-1.10/). |
| `todd2024function` | Head localization motivation; portable vectors; single-relation ICL task provenance in main text and appendix | **Supported.** Supplied paper describes summed causal-head outputs, portability, and a collection of over 40 tasks. Our 39-task filtered subset is an adaptation documented by this manuscript. “Function vectors” is more precise than “function pointers.” [Local paper](../../../Overleaf-draft/REF/papers/function.pdf). |
| `sensharma2025filter` | Head-localization motivation | **Supported, duplicate version.** Same paper as next row. [Preprint](https://arxiv.org/abs/2510.26784). |
| `sensharma2025filterheads` | Related Work: query-encoded filtering predicates and transfer | **Supported.** Checked supplied conference PDF and Figure 1's query transfer. Consolidate version usage. [Local paper](../../../Overleaf-draft/REF/papers/filter.pdf). |
| `cohen1990intention` | Persistence: keeping intention active until achievement | **Qualified.** Belief-dependent release conditions, including impossibility; no constant-activation claim. See issue 5. [Paper §4](https://ai.stanford.edu/~epacuit/classes/lori-spr09/cohenlevesque-intention-aij90.pdf). |
| `goschke1993representation` | Persistence: activation of pending intentions | **Relevant analogy.** Recognition-latency experiment, not token-by-token neural persistence. Original abstract and author records checked. [Author record](https://fis.tu-dresden.de/portal/en/publications/representation-of-intentions%28a976e3ee-e4e1-402f-9b51-e9e5e1a7a422%29.html). |
| `marsh1998activation` | Persistence: decreased activation after completion | **Relevant analogy.** Lexical-decision evidence for differences involving completed and pending scripts; does not establish LM goal release. Original first-page text checked. [Paper preview](https://www.researchgate.net/publication/232558832_Activation_of_Completed_Uncompleted_and_Partially_Completed_Intentions). |
| `altmann2002memory` | Persistence: release to prevent interference | **Claim/source mismatch.** Activation/decay/retrieval model is relevant, but is not evidence of the asserted LM release mechanism. See issue 4. [Full paper](https://www.interruptions.net/literature/Altmann-CogSci02.pdf). |
| `mcgrath2023hydra` | Ablations: compensation/redundancy | **Supported precedent.** Shows adaptive compensation after ablations. It does not by itself show our surviving heads compensate dynamically. [Paper](https://arxiv.org/html/2307.15771). |
| `wang2022interpretability` | Ablations: backup pathways | **Supported.** §3.4 identifies backup name movers that respond to ablation. Our residual competence alone does not isolate that mechanism. [Paper](https://arxiv.org/html/2211.00593). |
| `khandelwal2026how` | Main/appendix: compositional ICL tasks | **Content supported; venue unverified.** Current Table 1 contains the 18 tasks and §2 specifies 10-shot Q/A prompts, matching the described source collection. See issue 8 for metadata. [Current paper](https://arxiv.org/html/2510.01685). |
| `xiong2025everything` | Main/appendix: multiple compatible ICL tasks | **Supported.** Appropriate task-superposition source. This paper explicitly marks its probabilistic mental model as non-rigorous; do not import a stronger neural-superposition theorem. [ICML record](https://proceedings.mlr.press/v267/xiong25a.html). |
| `betley2025emergentmisalignment` | Main/appendix: emergent-misalignment setting | **Supported.** Narrow fine-tuning can generalize into broader misalignment. Specific risky-finance model provenance belongs to the follow-up work. [Paper](https://arxiv.org/abs/2502.17424). |
| `pal2023futurelens` | Related Work: predicting future tokens from a state | **Supported.** Future-token information, not itself abstract-goal commitment. [CoNLL paper](https://aclanthology.org/2023.conll-1.37/). |
| `lindsey2025biology` | Related Work: planning toward later output constraints | **Supported.** Poetry case studies identify and intervene on rhyme-related planning; scope is selected examples. [Original report](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). |
| `feucht2025dual` | Related Work: language-independent lexical representations | **Supported.** Supplied COLM PDF explicitly studies translation mediated by concept-induction head outputs. [Local paper](../../../Overleaf-draft/REF/papers/dual-route.pdf). |
| `greenblatt2024highlevel` | Old Related Work: high-level interpretability agenda | **Source unresolved.** Screenshot supports the idea but not the bibliography metadata. Removed from active use in rewritten Related Work; entry unchanged. See issue 7. |
| `ainslie2023gqa` | Transfer appendix: multiple query heads sharing K/V | **Supported.** Matches grouped-query attention. [EMNLP paper](https://aclanthology.org/2023.emnlp-main.298/). |
| `turner2025modelorganisms` | Non-code appendix: fine-tune provenance and eight evaluation questions | **Supported.** §3.1 covers risky-finance Qwen fine-tunes; Appendix A reproduces the eight questions. The paper credits Betley for the evaluation lineage, so joint attribution would be most precise. [Paper](https://arxiv.org/html/2506.11613v1). |
| `soligo2025convergent` | Non-code appendix: EM organisms/representations | **Relevant follow-up.** Main minimal organism is Qwen-14B; use Turner for the particular Qwen-32B model unless its exact checkpoint documentation identifies this paper too. [Paper](https://arxiv.org/abs/2506.11618). |
| `turner2023steering` | Non-code figure: financial fine-tune provenance | **Wrong work at this location.** Correctly repurposed for activation addition in new Related Work. See issue 1. [Paper](https://arxiv.org/abs/2308.10248). |

## Additional citations used in the new Appendix A

All already existed in `references.bib` except SelfIE. These additions account for five new active keys; removal of the Greenblatt key yields the net increase from 28 to 32.

| Key | Verified role |
| --- | --- |
| `hendel2023context` | In-context examples induce task representations used on new inputs. [Official Findings paper](https://aclanthology.org/2023.findings-emnlp.624/). |
| `Nostalgebraist2020` | Direct intermediate-state projection into vocabulary space. [Original logit-lens post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). |
| `belrose2023eliciting` | Learned layer-specific transformations for decoding latent predictions. [Original paper](https://arxiv.org/abs/2303.08112). |
| `chen2024selfie` | Hidden-embedding interpretation using natural-language generation. New entry copied from official ICML metadata. [Proceedings record](https://proceedings.mlr.press/v235/chen24ao.html). |
| `geiger2021inducing` | Interchange interventions and correspondence to higher-level causal structure. This is the cited causal-structure paper, not an invented citation to Distributed Alignment Search. [Original paper](https://arxiv.org/html/2112.00826). |

## Major manuscript inconsistencies beyond citation placement

### A. Query-progress interpretation versus Appendix Experiment B

`Text/Appendix/task_transfer.tex`, `tab:expb`, reports source-frontier selection near **84%** when that frontier is present in the destination, versus approximately **5%** for the progress/order target. This is material to the main-text claim that query transfer validates progress content. It supports task-specific selection information in the transplanted query and challenges a pure-progress interpretation of these heads. It does not rule out an earlier or distributed task/progress computation. The main section should directly acknowledge this control, rather than allowing readers to infer that the source-absent transfer uniquely identifies progress.

The appendix calls the order-different setup “confound-free”; it only addresses the particular ordinal-position confound, and Experiment B demonstrates why that distinction matters. Related Work therefore describes task/context-dependent selection without treating DIFF as established.

### B. The layer-budget argument is still invalid as a necessity claim

`Text/06_mechanism.tex` still infers from late goal heads and long implementations that goal and progress must be re-derived. Earlier cached states can be read directly by many later tokens. Depth limits successive latent update chains, not the number of positions influenced by one earlier state. Generated tokens also provide an indirect influence path. The architectural facts motivate hypotheses about recomputation but do not force DIFF. The proposed opposite-sign implementation can remain explicitly hypothetical, conditional on the relevant progress/key alignment. It is not tested merely by successful Q/KV transfer. [Attention architecture](https://arxiv.org/html/1706.03762v7).

### C. Temporal evidence versus obstacle persistence and release

The abstract introduces persistence “in the face of obstacles,” while the reported obstacle evaluation finds weak recovery, especially on hard prefixes. Its later “We verify” phrasing should distinguish temporal evidence from obstacle behavior. Also, declining cosine similarity does not uniquely show release, recomputation, causal maintenance, or why a transition happens. The temporal table itself contains a mid-span readout of the following operation, so “at every token” in Figure 1 is stronger than the examples support.

The temporal-readout caption also contradicts itself: it first identifies a final-token shift to `is_sorted` in the range example, then calls both final readouts generic and unrelated. The actual table supports a goal-related shift in one case and a generic description in the other.

### D. Main DIFF results omit a material intervention detail

The transfer appendix specifies **amplification of goal-head outputs by alpha = 2**, in addition to query/KV replacement. The main section presents the effects mainly as transplants. Mention the amplification when introducing the quantitative results, since a reader otherwise interprets them as replacement alone. This is a reporting consistency issue, not an independently reproduced implementation check.

### E. KV summaries and plotted results use inconsistent cohorts

The main section reports 99.2%/96.1%. The appendix's own commented TODO says the run expanded beyond 32 examples per cell and gives updated aggregate rates around 0.97/0.94, while the active table/caption still state 32. Reconcile the run, denominator, aggregation, and figure together. I did not silently replace numbers.

### F. Remaining draft/build issues

- The Discussion section is empty apart from an author comment; its substantive paragraphs are commented out.
- Section 6 still prints internal drafting instructions such as where to put a figure.
- The fresh build has unresolved cross-references and a duplicate label. These predate the Related Work edit and are listed in [build_final.log](build_final.log). The main unresolved keys include `app:model_comparison`, `fig:goal-eval`, and `sec:mechanism`; `fig:freeform-matching` is duplicated.
- The bibliography versions and metadata issues above are left for review rather than being silently rewritten.

## Artifacts and validation

- [Rebuilt full paper](build/main.pdf), with Appendix A in place.
- [Original Related Work backup](related_work_before.tex) and [original bibliography backup](references_before.bib).
- [Source-level citation inventory before](inventory.json) and [after](inventory_after.json).
- [Build log](build_final.log) and [validation record](validation.json). The final 40-page PDF has no undefined citations; its 32 compiled citation keys exactly match the source inventory. Appendix A appears on pages 12–13, which were rendered and visually checked. An explicit file-hash comparison confirms that the only changed pre-existing `.tex` file is `Text/06_related_works.tex`.

The build uses the project's existing BibTeX behavior, which writes generated bibliography files in the Overleaf directory even when the PDF output is directed to this report's build folder. Those are generated artifacts; no other manuscript `.tex` source was edited.
