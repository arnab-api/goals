# Goal Encoding in LLMs

Arnab Sen Sharma, Eric Todd, Sheridan Feucht, Andy Arditi, Rohit Gandikota, Natalie Shapira, Gabriele Sarti,
Constanza Fierro, Lisa Schut, David Bau

**Project page and interactive GoalScope demo: https://goals.baulab.info** · Paper: coming soon

## Do LLMs have goals?

Language models now carry out tasks that unfold over thousands of tokens, and they behave as if they were pursuing a
goal. We ask whether that behavior is backed by an explicit goal encoding inside the model's computations. We fix
three criteria that functionally define a goal: (i) pre-commitment, (ii) plasticity, and (iii) persistence. We find an
encoding that meets these criteria in the outputs of a sparse set of attention heads. We further ask how a
depth-bounded transformer manages to keep such a goal alive across a long generation without running out of layers.

## What is a goal?

Inspired by [Bratman's planning theory](https://press.uchicago.edu/ucp/books/book/distributed/I/bo3629095.html), we
treat a goal as a functional state with three properties:

- **Pre-commitment**: the target state is settled before the action that realizes it begins.
- **Plasticity**: the same goal can be realized by different means, as the environment demands.
- **Persistence**: it is sustained while the action is carried out, and released once it is done.

In the LM's computations, we look for these criteria in a controlled setup with multi-step coding tasks: the prompt
lists several sub-tasks, and we prefill the code up to the point where a good model should start the next one (the
*frontier*). In this setup, we find an encoding that meets all three criteria in the outputs of a sparse set of
attention heads (the **goal heads**).

When we transplant the collective encoding of these heads into a neutral probe (**GoalScope**), the LM can verbalize
the upcoming action in natural language and implement it in a different programming language.

![GoalScope](https://goals.baulab.info/images/goalscope.svg)

*GoalScope. (a) The goal heads' outputs at one token form the encoding v<sub>g</sub>. (b) We plant this encoding into
a neutral probe: (c.1) asked to describe what it is doing, the model verbalizes the next action; and (c.2) in an
information-less C++ stub, the LM implements the same sub-goal with a different sequence of tokens. Right:
faithfulness (c.1) and causality (c.2) at four pre-implementation token positions.*

- **Pre-commitment.** The LM's next action becomes decodable well before the LM starts implementing it. At the
  indent token, three tokens before the LM writes the corresponding code, GoalScope can read out the next action in
  natural language with 94% accuracy.
- **Plasticity.** When this encoding, cached from a Python context, is transplanted into an information-less C++ code
  stub, the LM implements the same action in C++, with different libraries, coding syntax, and variables.
- **Persistence.** The code implementation of a sub-goal can take many tokens. Throughout this implementation, the LM
  keeps the sub-goal encoding alive in the same attention heads that first formed it.

## How does a depth-bounded transformer LM achieve persistence?

The only way a transformer LM can pool information from a previous token is through the attention mechanism, which
costs a layer. If the LM were to carry a running encoding of the goal from one token to the next, it would have only
a limited number of layers to do so. As an example, an encoding formed at layer 40 of a 64-layer model has only 24
layers left to pass it on, yet we see goals sustained well beyond that limit.

We find that instead of carrying a goal encoding forward, the LM frequently re-reads the goal from the task
specification. Then the LM decides the next action by comparing the re-fetched goal with the progress made so far.
We call this the **non-recurrent goal-seeking** (NRGS) account of how the transformer LM achieves persistence.

![NRGS](https://goals.baulab.info/images/teaser.svg)

*Non-recurrent implementation of goal pursuit in a transformer LM. A goal such as implementing merge sort decomposes
into sub-goals, each realized as a sequence of tokens. (i) An idealized agent forms an intention, acts on it until it
is done, and moves on. (ii) A transformer cannot carry that commitment forward through a fixed number of layers, so
at each step it re-reads the goal from the task span, computes its progress so far, and compares the two to pick the
next token.*

## Can GoalScope read out the LM's next action beyond code?

The goal heads were found on code, but GoalScope reads intentions elsewhere too, with no re-training: it can read out
the implicit rule in few-shot ICL prompts, and the intentions of an emergently misaligned model at the start of its
response.

![Beyond code](https://goals.baulab.info/images/beyond-code.svg)

## Is "goal" the right term for this encoding?

By our criteria, the encoding is pre-committed (decodable before the action) and plastic (portable to another
environment). It is also persistent, but through frequent re-derivation rather than a single running state. Whether
that counts as a "goal" depends on whether there is a requirement on the mechanical implementation of persistence: if
a goal must be a single state that survives the whole action, then this encoding does not qualify. But if we don't
care about the implementation, and only care whether the goal is present during the action, then this encoding
qualifies. We leave the semantics of whether this is a "goal" or not to the reader.

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
- Older research code on this branch: `src/diff/`, `scripts/diff/`, `notebooks/diff/` (earlier NRGS work),
  `scripts/clock/`, `notebooks/clock/` (clock / boundary heads), `notebooks/0*–16*` (first experiments),
  `agent_reports/` (write-ups of most experiments).

### Demos

Each notebook in `notebooks/demos/` walks through one experiment on a single example, with its outputs saved.

| demo | what it shows |
|---|---|
| [01_goalscope](notebooks/demos/01_goalscope.ipynb) | GoalScope readsout the next sub-task from the goal heads, as a natural language description and as C++ code. |
| [01.1_unspecified](notebooks/demos/01.1_unspecified.ipynb) | Goalscope readout beyond controlled settings. |
| [01.2_ovreader](notebooks/demos/01.2_ovreader.ipynb) | OV Projection of the goal heads can read-out the upcoming operation directly from a residual latent. |
| [01.4_unspec_caus_target](notebooks/demos/01.4_unspec_caus_target.ipynb) | GoalScope causality with a targeted C++ stub: from a shared base case, the goal heads' encoding of "implement <goal>" (the next step never written in the prompt) makes the stub implement that goal. |
| [02_persistence](notebooks/demos/02_persistence.ipynb) | The goal encoding persists while its execution. |
| [03_goal_head_ablation](notebooks/demos/03_goal_head_ablation.ipynb) | Sufficiency and Necessity of the goal heads, tested with targeted ablations. |
| [04_kv_transfer](notebooks/demos/04_kv_transfer.ipynb) | Supports the frequent re-read prediction. While the LM is in the middle of implementing a goal, changing what the goal heads read in the task span redirects the LM's action. |
| [04.1_sub_sub_goal](notebooks/demos/04.1_sub_sub_goal.ipynb) | Frequent re-read tested in a sub-sub-goal scenario. |
| [04.2_unspecified_op](notebooks/demos/04.2_unspecified_op.ipynb) | Frequent re-read tested in code where the next operation is not specified in the prompt. |
| [05_q_transfer](notebooks/demos/05_q_transfer.ipynb) | The goal heads' query carries the progress ("what is done"); transferring it selects the destination's pending sub-task not in the source progress. |
| [06_task_vs_progress](notebooks/demos/06_task_vs_progress.ipynb) | The task-span encoding, not what is being carried in the generated code, dictates the LM's future actions. |
| [07_beyond_code](notebooks/demos/07_beyond_code.ipynb) | Beyond code: GoalScope verbalizes the implicit rule of few-shot ICL prompts, and a EM LM's "thought" while generating a mis-aligned response. |

[CLAUDE.md](CLAUDE.md) maps every reported result to the command that reproduces it, and lists the unreported and
exploratory experiments on this branch with their entry points.

## How to cite

The paper is under review; a preprint and the final citation will be posted on the project page.
