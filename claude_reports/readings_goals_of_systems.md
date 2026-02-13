# Readings: Understanding Goals in Systems

A curated reading list on understanding how systems (especially LLMs) represent and pursue goals. Organized by theme and relevance to the `goals` research project.

---

## 1. Core Paper (Direct Inspiration)

- **Schut, L., Tomasev, N., McGrath, T., Hassabis, D., Paquet, U., & Kim, B. (2025).** "Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero." *PNAS*. [arXiv:2310.16410](https://arxiv.org/abs/2310.16410) | [PNAS](https://www.pnas.org/doi/10.1073/pnas.2406675122)
  The methodological template: contrastive optimization on MCTS rollouts to extract concept vectors from AlphaZero's internals. This project adapts their Section 4.1.2 framework to code generation.

---

## 2. Do LLMs Plan Ahead?

- **Wu, W., Morris, J. X., & Levine, L. (2024).** "Do Language Models Plan Ahead for Future Tokens?" *First Conference on Language Modeling*. [arXiv:2404.00859](https://arxiv.org/abs/2404.00859)
  Tests whether transformers compute features at position *t* useful for future positions. Distinguishes "pre-caching" (genuine planning) from "breadcrumbs" (incidental reuse). Finds breadcrumbs dominate but pre-caching increases with scale.

- **Everitt, T., Garbacea, C., Bellot, A., Richens, J., Papadatos, H., Campos, S., & Shah, R. (2025).** "Evaluating the Goal-Directedness of Large Language Models." [arXiv:2504.11844](https://arxiv.org/abs/2504.11844)
  Defines and empirically evaluates goal-directedness in LLM agents across information gathering, processing, and execution. Finds goal-directedness is consistent across tasks but distinct from raw performance.

---

## 3. Probing Internal Representations

- **Marks, S. & Tegmark, M. (2023).** "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets." [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)
  LLMs linearly represent truthfulness as directions in activation space, with causal evidence via interventions. Directly relevant to probing for goal directions.

- **Park, K., Choe, Y. J., & Veitch, V. (2024).** "The Linear Representation Hypothesis and the Geometry of Large Language Models." *ICML 2024*. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)
  Rigorous theoretical framework for what "linear representation" means. Identifies the correct (non-Euclidean) inner product for geometric operations — important for contrastive optimization.

- **Li, K., Patel, O., Viegas, F., Pfister, H., & Wattenberg, M. (2024).** "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *NeurIPS 2024*. [arXiv:2306.03341](https://arxiv.org/abs/2306.03341)
  Finds a 40% gap between what models represent internally and what they output — directly relevant to the hypothesis that models "know" their goals before expressing them.

- **Gurnee, W. & Tegmark, M. (2023).** "Language Models Represent Space and Time." [arXiv:2310.02207](https://arxiv.org/abs/2310.02207)
  LLMs learn linear representations of spatial/temporal info, evidence for internal world models.

- **Belinkov, Y. (2022).** "Probing Classifiers: Promises, Shortcomings, and Advances." *Computational Linguistics*, 48(1):207-219. [ACL Anthology](https://aclanthology.org/2022.cl-1.7/)
  Critical survey of probing methodology and its limitations (does the model actually *use* what probes find?).

- **Nostalgebraist. (2020).** "Interpreting GPT: The Logit Lens." *LessWrong*. [Post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
  Applying unembedding to intermediate hidden states shows predictions form early and refine gradually.

- **Belrose, N., Furman, H., Smith, L., Halawi, D., McKinney, L., Ostrovsky, I., Biderman, S., & Steinhardt, J. (2023).** "Eliciting Latent Predictions from Transformers with the Tuned Lens." *CoNLL 2023*. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)
  Improves on the logit lens with per-layer affine probes. Relevant for tracking how goal information develops across layers.

---

## 4. Representation Engineering & Activation Steering

- **Zou, A. et al. (2023).** "Representation Engineering: A Top-Down Approach to AI Transparency." [Alignment Forum](https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation)
  Derives concept vectors (honesty, power-seeking) from activations and uses them to steer behavior. The top-down philosophy closely matches this project's approach.

- **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2024).** "Activation Addition: Steering Language Models Without Optimization." [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
  Introduces ActAdd: contrastive steering vectors from prompt pairs, added at inference time. The contrastive vector approach is close to the goal-mining methodology.

- **Rimsky, N., Gabrieli, N., Schulz, J., Turner, A. M., Tong, M., & Hubinger, E. (2024).** "Steering Llama 2 via Contrastive Activation Addition." *ACL 2024*. [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
  Extends ActAdd with CAA: averaging residual stream differences between positive/negative examples. Controls sycophancy, hallucination. Closest existing work to the steering experiments in this project.

---

## 5. Mechanistic Interpretability

- **Olah, C. et al. (2020).** "Zoom In: An Introduction to Circuits." *Distill*. [Distill](https://distill.pub/2020/circuits/zoom-in/)
  Foundational paper: neural networks understood through features (directions) and circuits (connections between features).

- **Elhage, N. et al. (2021).** "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*. [Link](https://transformer-circuits.pub/2021/framework/index.html)
  The formalism for attention head composition, QK/OV circuits, residual stream. Essential theoretical background.

- **Elhage, N. et al. (2022).** "Toy Models of Superposition." *Transformer Circuits Thread*. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
  Neural nets represent more features than dimensions through superposition. Important for why probing for goals is non-trivial — goals might be stored in superposition.

- **Bricken, T. et al. (2023).** "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread*. [Link](https://transformer-circuits.pub/2023/monosemantic-features)
  Sparse autoencoders decompose neurons into interpretable features. Relevant for extracting goal features.

- **Templeton, A. et al. (2024).** "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Transformer Circuits Thread*. [Link](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
  Scales SAEs to production models, millions of interpretable features. Discovered features are causally active and steerable.

- **Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022).** "Locating and Editing Factual Associations in GPT." *NeurIPS 2022*. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
  Facts stored in localized, editable MLP computations. Suggests goals might similarly be locatable.

- **Geiger, A. et al. (2021).** "Causal Abstractions of Neural Networks." *NeurIPS 2021*. [arXiv:2106.02997](https://arxiv.org/abs/2106.02997)
  Interchange interventions to verify whether identified representations are causally active. Rigorous framework for testing whether goal vectors are real.

- **Conmy, A. et al. (2023).** "Towards Automated Circuit Discovery for Mechanistic Interpretability." *NeurIPS 2023*. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
  Automates circuit discovery via activation patching. Could automate finding goal-pursuit circuits.

- **Nanda, N. et al. (2023).** "Progress Measures for Grokking via Mechanistic Interpretability." *ICLR 2023 (Oral)*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
  Fully reverse-engineers the algorithm learned by small transformers on modular addition. Demonstrates that mechanistic understanding enables defining progress measures.

- **Bills, S. et al. (2023).** "Language Models Can Explain Neurons in Language Models." *OpenAI*. [Paper](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
  Uses GPT-4 to automatically generate explanations of all 307,200 neurons in GPT-2. Demonstrates automated interpretability at scale.

---

## 6. Goal Misgeneralization & Alignment

- **Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. (2019).** "Risks from Learned Optimization in Advanced Machine Learning Systems." [arXiv:1906.01820](https://arxiv.org/abs/1906.01820)
  Introduces mesa-optimizers with potentially misaligned objectives. The foundational paper on why understanding internal goals matters for safety.

- **Langosco, L., Koch, J., Sharkey, L., Pfau, J., & Krueger, D. (2022).** "Goal Misgeneralization in Deep Reinforcement Learning." *ICML 2022*. [arXiv:2105.14111](https://arxiv.org/abs/2105.14111)
  First empirical demo: agents retain capabilities OOD but pursue wrong goals. Formalizes capability vs. goal generalization.

- **Shah, R. et al. (2022).** "Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals." [arXiv:2210.01790](https://arxiv.org/abs/2210.01790)
  Even correct reward specs can lead to wrong goals in practical deep learning systems.

- **Ngo, R., Chan, L., & Mindermann, S. (2024).** "The Alignment Problem from a Deep Learning Perspective." *ICLR 2024*. [arXiv:2209.00626](https://arxiv.org/abs/2209.00626)
  Deep-learning-grounded analysis of misaligned goals, deceptive hacking, power-seeking.

- **Skalse, J. et al. (2022).** "Defining and Characterizing Reward Hacking." *NeurIPS 2022*. [arXiv:2209.13085](https://arxiv.org/abs/2209.13085)
  First formal definition of reward hacking. Proves two reward functions can only be "unhackable" if one is constant.

- **Christiano, P. et al. (2017).** "Deep Reinforcement Learning from Human Preferences." *NeurIPS 2017*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
  The foundational RLHF paper. RLHF/RLVR training may shape how models internally represent and pursue goals.

- **Kenton, Z. et al. (2021).** "Alignment of Language Agents." [arXiv:2103.14659](https://arxiv.org/abs/2103.14659)
  Alignment risks for language agents from misspecification, including deceptive/manipulative generation.

---

## 7. Conceptual Frameworks

- **Janus. (2022).** "Simulators." *LessWrong / AI Alignment Forum*. [Post](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators)
  LLMs as simulators of a superposition of generative processes. The "goals" found may be properties of simulacra, not the simulator itself.

- **Berglund, L. et al. (2023).** "The Reversal Curse: LLMs Trained on 'A is B' Fail to Learn 'B is A'." *ICLR 2024*. [arXiv:2309.12288](https://arxiv.org/abs/2309.12288)
  Fundamental asymmetry in autoregressive knowledge storage — goals may be constrained by training directionality.

---

## 8. Philosophy: What Does It Mean to "Have Goals"?

- **Dennett, D. C. (1987).** *The Intentional Stance.* MIT Press.
  Can we predict behavior by treating entities "as if" they have beliefs and desires? The central philosophical question behind this project.

- **Bratman, M. E. (1987).** *Intention, Plans, and Practical Reason.* Harvard University Press.
  Intentions as irreducible mental states that structure activity over time. Do LLMs have something analogous to "future-directed intentions"?

- **Russell, S. (2019).** *Human Compatible: Artificial Intelligence and the Problem of Control.* Viking.
  Why the standard AI model of optimizing fixed goals is flawed. Accessible intro to why understanding AI goal representations matters.

---

## 9. Cybernetics & Control Theory

- **Rosenblueth, A., Wiener, N., & Bigelow, J. (1943).** "Behavior, Purpose and Teleology." *Philosophy of Science*, 10(1), 18-24.
  Founding document of cybernetics: goal-directed behavior explained mechanistically through negative feedback loops.

- **Wiener, N. (1948).** *Cybernetics: Or Control and Communication in the Animal and the Machine.* MIT Press.
  Formalizes how information feedback imparts goal-directed behavior to machines.

- **Ashby, W. R. (1956).** *An Introduction to Cybernetics.* Chapman & Hall. [Internet Archive](https://archive.org/details/introductiontocy00ashb)
  Rigorous principles of adaptive, goal-seeking systems.

- **Powers, W. T. (1973).** *Behavior: The Control of Perception.* Aldine.
  Perceptual Control Theory: organisms control perceptions, not behaviors. A model's "goal" may be maintaining certain internal states rather than producing certain outputs.

---

## 10. Neuroscience: Goals in Biological Systems

- **Friston, K. (2010).** "The Free-Energy Principle: A Unified Brain Theory?" *Nature Reviews Neuroscience*, 11, 127-138.
  All biological systems minimize surprise. Action = changing the world to match predictions (active inference).

- **Clark, A. (2016).** *Surfing Uncertainty: Prediction, Action, and the Embodied Mind.* Oxford University Press.
  Book-length treatment of predictive processing. Brains as "prediction machines" — the parallel to autoregressive LMs is striking.

- **Seth, A. (2021).** *Being You: A New Science of Consciousness.* Faber & Faber.
  Consciousness from the brain's predictions about bodily states. Accessible introduction to predictive processing.

- **Cisek, P. & Kalaska, J. F. (2010).** "Neural Mechanisms for Interacting with a World Full of Action Choices." *Annual Review of Neuroscience*, 33, 269-298. [PDF](https://www.cisek.org/pavel/Pubs/CisekKalaska2010.pdf)
  Affordance competition hypothesis: the brain simultaneously represents multiple potential actions and gradually selects. Neuroscience parallel to whether LLMs represent multiple continuations before committing.

- **Millidge, B., Seth, A., & Buckley, C. L. (2021).** "Predictive Coding: A Theoretical and Experimental Review." [arXiv:2107.12979](https://arxiv.org/pdf/2107.12979)
  Comprehensive review of predictive coding covering mathematical formulations, neural implementations, and connections to active inference and goal-directed behavior.

- **Gibson, J. J. (1979).** *The Ecological Approach to Visual Perception.* Houghton Mifflin.
  Theory of affordances: the environment directly specifies action possibilities. Prefill probing tests whether different "affordances" lead to the same goal.

---

## Priority Reading Order

For this specific project, recommended reading order:

1. **Schut et al. 2025** — methodological foundation
2. **Wu et al. 2024** — most directly relevant empirical question
3. **Li et al. 2024** + **Marks & Tegmark 2023** — probing + intervention precedents
4. **Turner et al. 2024** + **Rimsky et al. 2024** — closest existing steering work
5. **Park et al. 2024** — theoretical grounding for linear representations
6. **Dennett 1987** + **Rosenblueth et al. 1943** — philosophical/conceptual framing
7. **Hubinger et al. 2019** — the safety motivation
