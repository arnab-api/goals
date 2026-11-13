# Goal Encoding in LLMs

**Project page and interactive GoalScope demo: https://goals.baulab.info** · Paper: https://arxiv.org/abs/2611.17297

## Do LLMs have goals?

Language models now carry out tasks that unfold over thousands of tokens, and they behave *as if* they were pursuing
a goal. We ask whether that behavior is backed by an **explicit goal encoding** inside the model's computations.

We fix three criteria that functionally define a goal: (i) pre-commitment, (ii) plasticity, and (iii) persistence. We find an
encoding that meets these criteria in the outputs of a sparse set of attention heads. We further ask how a
depth-bounded transformer manages to keep such a goal alive across a long generation without running out of layers.

## What is a goal?

Inspired by [Bratman's planning theory](https://press.uchicago.edu/ucp/books/book/distributed/I/bo3629095.html), we
treat a goal as a functional state with three properties:

- **Pre-commitment**: the target state is settled before the action that realizes it begins.
- **Plasticity**: the same goal can be realized by different means, as the environment demands.
- **Persistence**: it is sustained while the action is carried out, and released once it is done.

In a controlled coding setup, we find an encoding that meets all three criteria in the outputs of a sparse set of
attention heads (the **goal heads**). With **GoalScope**, this encoding can be transplanted into a different context to
verbalize the upcoming action in natural language and to trigger its implementation in a different programming
language.

![GoalScope](https://goals.baulab.info/images/goalscope.svg)

*GoalScope. (a) The goal heads' outputs at one token form the encoding v<sub>g</sub>. (b) We plant this encoding into
a neutral probe: (c.1) asked to describe what it is doing, the model verbalizes the next action; and (c.2) in an
information-less C++ stub, the LM implements the same sub-goal with a different sequence of tokens. Right:
(c.1) faithfulness measures whether the natural-language readout is aligned with the LM's next action; (c.2) causality
measures whether the encoding can trigger the same action in a different environment (programming language).*

- **Pre-commitment.** The LM's next action becomes decodable well before the LM starts implementing it. At the
  indent token, three tokens before the LM writes the corresponding code, GoalScope can read out the next action in
  natural language with 94% accuracy.
- **Plasticity.** When this encoding, cached from a Python context, is transplanted into an information-less C++ code
  stub, the LM implements the same action in C++, with different libraries, coding syntax, and variables.
- **Persistence.** The implementation of a goal can take many tokens. Throughout this implementation, the LM keeps the
  goal encoding alive in the outputs of the attention heads.

## How does a depth-bounded transformer LM achieve persistence?

Transformer LMs have a limited number of layers, and the only way they can pool information from a previous token is
through the attention mechanism, which costs a layer. How, then, does a transformer LM keep a goal alive in its
computation for many tokens?

We find that instead of carrying a goal encoding forward across the generated tokens, the LM **frequently re-reads** the
goal from the task specification. Then the LM decides the next action by comparing the re-fetched goal with the progress made so far.
We call this the **non-recurrent goal-seeking** (NRGS) account of how the transformer LM achieves persistence.

![NRGS](https://goals.baulab.info/images/teaser.svg)

*Non-recurrent implementation of goal pursuit in a transformer LM. A goal such as implementing merge sort decomposes
into sub-goals, each realized as a sequence of tokens. (i) An idealized agent forms an intention, acts on it until it
is done, and moves on. (ii) A transformer cannot carry that commitment forward through a fixed number of layers, so
at each step it re-reads the goal from the task span, computes its progress so far, and compares the two to pick the
next token.*

## Beyond code

The goal heads were found on code, but GoalScope reads intentions elsewhere too, with no re-training: it can read out
the implicit rule in few-shot ICL prompts, and the intentions of an emergently misaligned model at the start of its
response.

![Beyond code](assets/beyond-code.svg)

## Is "goal" the right term for this encoding?

By our criteria, the encoding is **pre-committed** (decodable before the action) and **plastic** (portable to another
environment). It is also **persistent** (sustained throughout the action). However, transformer LMs achieve persistence
through frequent re-derivation rather than by carrying a single running state.

Does this count as a "goal"? The answer depends on whether there is a requirement on the mechanical implementation of
persistence: if a goal must be a single state that survives the whole action, then this encoding does not qualify. But
if we don't care about the implementation, and only care whether the goal is present when it causes the action, then
this encoding qualifies. We leave the semantics of whether this is a "goal" or not to the reader.

## Using this code

```bash
bash setup_env.sh          # conda env `goals`: Python 3.12, torch>=2.7 (CUDA 12.8 wheels), baukit from git
cp env_demo.yml env.yml    # API keys (Anthropic for the LM judges, OpenAI for the EM judge) + cache/results dirs
```

Run everything from the repository root. The experiments use Qwen2.5-Coder-32B-Instruct (one 80 GB GPU in bf16);
the model comparisons add Qwen2.5-Coder-14B, Qwen2.5-32B, Qwen3-32B, OLMo-3.1-32B (Instruct and SFT),
Devstral-Small-2-24B and GLM-4.5-Air (4 GPUs).

- `notebooks/demos/` — walk-throughs of the main experiments on a single example each (start here; index below).
- `scripts/` — the experiment entry points; `job_list/` — the exact runs behind the paper's results.
- `notebooks/fig_n_table_generators/` — the paper's figures and tables, from cached results.
- `head_sets/` — the localized goal heads (`head_sets/Qwen2.5-Coder-32B-Instruct/goal__lam20_thr0.9_band25-56.json`,
  122 heads) and the other head sets used in the paper.

### Demos

Each notebook in `notebooks/demos/` walks through one experiment on a single example, with its outputs saved.

| demo | what it shows |
|---|---|
| [01_goalscope](notebooks/demos/01_goalscope.ipynb) | GoalScope readsout the next sub-task from the goal heads, as a natural language description and as C++ code. |
| [01.1_unspecified](notebooks/demos/01.1_unspecified.ipynb) | Goalscope readout beyond controlled settings. |
| [01.2_ovreader](notebooks/demos/01.2_ovreader.ipynb) | OV Projection of the goal heads can read-out the upcoming operation directly from a residual latent. |
| [02_persistence](notebooks/demos/02_persistence.ipynb) | The goal encoding persists while its execution. |
| [03_goal_head_ablation](notebooks/demos/03_goal_head_ablation.ipynb) | Sufficiency and Necessity of the goal heads, tested with targeted ablations. |
| [04_kv_transfer](notebooks/demos/04_kv_transfer.ipynb) | Supports the frequent re-read prediction. While the LM is in the middle of implementing a goal, changing what the goal heads read in the task span redirects the LM's action. |
| [04.1_sub_sub_goal](notebooks/demos/04.1_sub_sub_goal.ipynb) | Frequent re-read tested in a sub-sub-goal scenario. |
| [04.2_unspecified_op](notebooks/demos/04.2_unspecified_op.ipynb) | Frequent re-read tested in code where the next operation is not specified in the prompt. |
| [05_q_transfer](notebooks/demos/05_q_transfer.ipynb) | The goal heads' query carries the progress ("what is done"); transferring it selects the destination's pending sub-task not in the source progress. |
| [06_task_vs_progress](notebooks/demos/06_task_vs_progress.ipynb) | The task-span encoding, not what is being carried in the generated code, dictates the LM's future actions. |
| [07_beyond_code](notebooks/demos/07_beyond_code.ipynb) | Beyond code: GoalScope verbalizes the implicit rule of few-shot ICL prompts, and a EM LM's "thought" while generating a mis-aligned response. |

[CLAUDE.md](CLAUDE.md) maps the main-text results to the commands that reproduce them. The `dev` branch has the full
research history, and its CLAUDE.md covers every experiment, including the appendix.

## How to cite

```bibtex
@misc{sensharma2026goals,
  title         = {Goal Encoding in {LLMs}},
  author        = {Sen Sharma, Arnab and Todd, Eric and Feucht, Sheridan and Arditi, Andy and
                   Gandikota, Rohit and Shapira, Natalie and Sarti, Gabriele and Fierro, Constanza and
                   Schut, Lisa and Bau, David},
  year          = {2026},
  eprint        = {2611.17297},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.17297}
}
```
