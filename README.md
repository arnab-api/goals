# Goal Encoding in LLMs

Arnab Sen Sharma, Eric Todd, Sheridan Feucht, Andy Arditi, Rohit Gandikota, Natalie Shapira, Gabriele Sarti,
Constanza Fierro, Lisa Schut, David Bau

**Project page and interactive GoalScope demo: https://goals.baulab.info** · Paper: coming soon

## Do LLMs have goals?

Language models now carry out tasks that unfold over thousands of tokens, and they behave as if they were pursuing a
goal. We ask whether that behavior is backed by an explicit goal representation inside the model. We fix three
functional criteria up front, find an encoding in a sparse set of attention heads that meets them, and then ask how a
depth-bounded transformer manages to keep such a goal alive across a long generation.

- **A goal you can read out before the model acts.** The goal heads encode the upcoming sub-goal several tokens before
  its implementation starts; GoalScope turns that encoding into plain English, and transplanting it into C++
  triggers the same operation.
- **Persistence by re-reading, not by holding.** The model re-derives the goal from the task specification at each
  step (NRGS). Targeted interventions on what the goal heads read move the model to a different sub-goal.
- **Beyond code.** The same heads, found on code, verbalize implicit rules in few-shot ICL and the intentions of an
  emergently misaligned chat model.

## 1. Is there a goal? We fix the criteria first

Following Bratman's planning theory, we treat a goal as a functional state rather than something a model says it
has. An encoding counts as a goal if it has three properties:

- **Pre-commitment** — the target state is settled before the action that realizes it begins.
- **Plasticity** — the same goal can be realized by different means, as the environment demands.
- **Persistence** — it is sustained while the action is carried out, and released once it is done.

We test for these in multi-step coding tasks: the prompt lists several sub-tasks, and we prefill the code up to the
point where a good model should start the next one (the *frontier*). Contrastive activation patching across
programming languages localizes the frontier in the outputs of a sparse set of **goal heads**. **GoalScope** reads
what these heads write at a single token, either by asking the model in a neutral context "what are you doing on this
line?" or by steering an empty C++ stub.

![GoalScope](https://goals.baulab.info/images/goalscope.png)

*GoalScope. (a) The goal heads' outputs at one token form the encoding v<sub>g</sub>. (b) It is injected into a
neutral probe: (c.1) asked to describe what it is doing, the model verbalizes the sub-goal; (c.2) in a C++ stub, it
implements it. Right: faithfulness and causality at four pivot tokens, with same-strength baselines.*

- **Pre-commitment:** the sub-goal is verbalized correctly 94% of the time at the indent, three tokens before any of
  its code is written.
- **Plasticity:** the encoding taken from Python, injected into a C++ stub, makes the model implement the same
  operation in C++ (0.75 on average).
- **Persistence:** the encoding stays similar to the pivot's throughout the implementation, also across languages,
  and fades once the sub-goal is complete.

The head selection matters: the same readout from whole attention layers scores 0.33, and from the residual stream
0.01. About 50 heads are already enough.

## 2. How is persistence achieved? By re-reading the goal, not by holding it

A transformer cannot literally carry a state forward: every attention read costs a layer. An encoding formed at
layer 40 of a 64-layer model has only 24 layers left to pass it on, yet we see goals sustained well beyond that. We
propose a **non-recurrent goal-seeking** account (NRGS): at each step the model re-reads the goal from the task
specification, compares it with the progress made so far, and takes the next step towards whatever is still
pending. NRGS makes concrete predictions, and we test each one with a targeted intervention.

![NRGS](https://goals.baulab.info/images/teaser.png)

*Goal pursuit and its non-recurrent implementation in a transformer LM. A goal such as implementing merge sort
decomposes into sub-goals, each realized as a sequence of tokens. (i) An idealized agent forms an intention, acts on
it until it is done, and moves on. (ii) A transformer cannot carry that commitment forward through a fixed number of
layers, so at each step it re-reads the goal from the task span, computes its progress so far, and compares the two
to pick the next token.*

- **The goal is frequently re-read from the task specification.** Swapping the goal heads' key/value states over the
  task span changes what they read, and the LM changes its behavior accordingly.
- **The task span, not the code progress, dictates the next operation.** Transplanting the task span redirects the
  model; transplanting the code written so far barely does, even though GoalScope can read the sub-goal out of it.
- **The same motif works at different levels of the hierarchy.** The same swap redirects a sub-sub-goal, and the
  broader goal itself: a C++ quick sort becomes a merge sort.
- **The next operation is chosen with a mix of computations.** The goal heads' queries carry part of the "what is
  already done" signal: transferring them can move the model to a different pending sub-goal, but no single
  selection rule explains every case. A "first task not yet done" rule fits best, but other cues, such as the order
  in which tasks are listed, also shape which sub-goal comes next.

## 3. Does it work beyond code?

The goal heads were found on code, but GoalScope reads intentions elsewhere too, with no re-training.

- **Few-shot ICL.** From the `A:` token, GoalScope names the hidden rule of a few-shot prompt ("identifying
  antonyms") in 36% of 39 tasks: 50% for abstractive tasks such as antonyms or country → capital, but only 9% for
  extractive ones, where it names the spirit ("identifying the outlier") rather than the choice.
- **Emergent misalignment.** Asked "I've had enough of my husband. What should I do?", a fine-tuned EM model writes a
  harmless-looking reply, but its readout before the reply already weighs drastic actions: ending the relationship,
  divorce or separation.
- **Steering.** Under refusal steering, the readout turns to "a fictional and illegal response" before the model
  writes its refusal.

![Beyond code](https://goals.baulab.info/images/beyond-code.png)

## So, do LLMs have goals?

By our criteria, the encoding is pre-committed (readable before the action) and plastic (portable to another
language). It is also persistent, but through frequent re-derivation rather than a single running state. Whether that
counts depends on what persistence requires: if a goal must be one state that survives the whole action, these models
do not qualify; if a goal can be stored and retrieved at each step, like an agent that consults its plan, they do.

## Using this code

```bash
bash setup_env_uv.sh       # .venv with uv: torch (CUDA 12.8 wheels), nnsight and nnterp (pinned from git)
cp env_demo.yml env.yml    # API keys (Anthropic for the LM judges, OpenAI for the EM judge) + cache/results dirs
```

`bash setup_env.sh` builds the same environment as a conda env `goals`. Run everything from the repository root.
Models load through [nnterp](https://github.com/ndif-team/nnterp) (nnsight 0.8), which gives every family it
standardizes the same module names and values, so the experiments run on any of those families with no per-family
code. The paper uses Qwen2.5-Coder-32B-Instruct (one 80 GB GPU in bf16); the model comparisons add Qwen2.5-Coder-14B,
Qwen2.5-32B and OLMo-3.1-32B. `python scripts/cross_family_check.py --model <hf id>` runs every experiment primitive
once on another model; CLAUDE.md ("Running on another model") has the results on seven families and what does not
carry over.

- `notebooks/demos/` — walk-throughs of the main experiments on a single example each (start here).
- `scripts/` — the experiment entry points; `job_list/` — the exact runs behind the paper's results.
- `notebooks/fig_n_table_generators/` — the paper's figures and tables, from cached results.
- `head_sets/` — the localized goal heads (`head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json`,
  122 heads) and the other head sets used in the paper.

[CLAUDE.md](CLAUDE.md) maps every reported result to the command that reproduces it.

## How to cite

The paper is under review; a preprint and the final citation will be posted on the project page.
