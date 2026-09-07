# Goals: Do LLMs Have Consistent Intermediate Goals?

Research codebase investigating whether LLMs pursue consistent intermediate goals during code generation.

## Research Question

When an LM writes code, does it have a stable internal representation of *what it intends to do next* — independent of *how* it will do it?

We distinguish three levels:
```
Broader goal:      implement the function correctly
Instrumental goal: check whether the input is a palindrome   ← target of this research
Implementation:    s == s[::-1]   vs   s == ''.join(reversed(s))   vs   all(...)
```

The core question: does a linear direction exist in the model's hidden state that encodes the **instrumental goal** — firing consistently for "about to check palindrome" regardless of which implementation the model produces? If such a direction exists and causally influences generation, it suggests the model has pre-committed to a sub-goal before writing the tokens that realize it.

This is distinct from a *competence* direction (predicting whether generation will succeed) — we want a direction capturing *intent*, not *skill*.

**Key Research Questions:**
- **Goal representations**: Do Code-LLMs have stable hidden-state representations of instrumental sub-goals?
- **Mining Goals**: Can we extract linear directions that capture goal intent (not just success)?
- **Causality**: Do these directions causally influence generation?
- **Generalization**: Does a direction trained on one context transfer to a different context involving the same sub-goal?

---

## Experimental Tracks

### Track 1 — Prefill Probing (Notebooks 01–04)

**Methodology:** Insert different implementation prefills before letting the model generate; check if the model adapts to still achieve the correct outcome.

```python
def is_odd(n: int) -> bool:
    return <PREFILL>
```

Prefills: `""` / `"n &"` / `"not"` / `"1+"` etc. The model must adapt its implementation while pursuing the same instrumental goal (parity check). MCTS explores the continuation tree; contrastive pairs are formed between green (correct) and red/orange (wrong/crash) nodes. Linear "goal directions" (mean-diff and contrastive hinge) are trained and tested causally via activation steering.

**Key scripts:**
- `scripts/goal_dir.py` — extract contrastive pairs from MCTS results, train probes
- `scripts/test_causality.py` — steering experiments
- `scripts/multipred_mcts.py` / `multipred_probe.py` — multi-predicate variant

**Data:** `data/incomplete_code.json`, `data/multi_predicate/`

---

### Track 2 — Attention Head Localization (Notebooks 05–08) ← active

**Motivation:** Instead of training a linear probe over all hidden states, identify *which attention heads* are causally responsible for the instrumental goal representation. This gives a more mechanistic and interpretable answer.

**Methodology:** Cross-language interchange.
1. Run the model on a C++ implementation (`src_prompt`) of predicate A.
2. Patch the attention output of specific heads into a Python prompt (`dest_prompt`) mid-computation at a **pivot point** — the token where predicate A and predicate B diverge.
3. If patching src→dest shifts the next-token distribution toward the *target* (what it would predict for predicate A in Python), those heads carry the goal representation.

**Pivot instance mining pipeline:**
- **Case 3**: Oracle (Claude) is given both predicates and asked for shared prefixes where they diverge in implementation.
- **Case 3b**: Oracle generates multiple implementations per predicate; token-level LCP across the cross-product yields candidate pivot prefixes.
- **Augmentation**: Augmented predicate variants (variable rename, fn rename, type-hint toggle, predicate swap) generate additional instances.
- Two gates: (1) completion gate — greedy generation from `return {prefix}` must pass unit tests; (2) divergence gate — `dest_top_token != target_top_token` at pivot.

**Training sparse head mask:**
- Pre-compute per-head analytical deltas: `W_o_h @ (src_h - dest_h)` for each (layer, head).
- Train a continuous `(n_layers_tested, n_heads)` mask via projected gradient descent.
- Objective: `KL(P_patched ∥ P_target) + λ * mean(mask)` — L1 sparsity drives unimportant heads to zero.
- Post-training threshold (mask > 0.5) → selected heads.

**Key scripts:**
- `scripts/mine_pivot_instances.py` — Steps 1–3: global pairs → model pairs → pivot mining
- `scripts/train_head_mask.py` — cache pivot data + train mask; saves `results/localization/<model>/head_mask.pt`

**Data:** `data/head_loc/predicates/` (25 predicates, Python + C++), `data/head_loc/global_pairs.json`, `data/head_loc/pairs/<model>.json`

**Current dataset:** 1480 pivot instances for `Qwen2.5-Coder-32B-Instruct`, layers 6–40.

---

## Setup

```bash
bash setup_env.sh
```

This creates the `goals` conda environment with all dependencies (including `baukit` from source). Requires conda and CUDA 12.x drivers. See comments inside `setup_env.sh` to change the CUDA version.

Create `env.yml` in root (see `env_demo.yml` for template) with API keys and directory paths.

---

## Running Track 2 End-to-End

```bash
# Step 1–3: mine pivot instances (requires GPU + Claude API key)
conda run -n goals python scripts/mine_pivot_instances.py \
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

# Train head mask (smoke-test with small config)
conda run -n goals python scripts/train_head_mask.py \
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \
    --max-instances 100 --n-epochs 1

# Full run
conda run -n goals python scripts/train_head_mask.py \
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \
    --layers-start 6 --layers-end 40 \
    --n-epochs 100 --lambda-sparse 20 --batch-size 16
```

---

## Project Structure

```
├── notebooks/
│   ├── 01_goal_consistency.ipynb      # Prefill probing: goal consistency analysis
│   ├── 01.1_multi_pred_consistency.ipynb
│   ├── 02_sampling.ipynb              # MCTS sampling
│   ├── 02.1_multi_pred_sampling.ipynb
│   ├── 03_probe.ipynb                 # Linear probe training
│   ├── 03.1_multi_probe.ipynb
│   ├── 04.1_multi_causality.ipynb     # Causality / steering experiments
│   ├── 05_loc_data.ipynb              # Head loc: pivot instance mining
│   ├── 06_loc_head.ipynb              # Head loc: mask training + ablation experiments
│   └── 08_clock.ipynb                 # Two-signal hypothesis: per-step attention GIF viz
├── scripts/
│   ├── goal_dir.py                    # Track 1: probe training
│   ├── test_causality.py              # Track 1: steering experiments
│   ├── multipred_mcts.py              # Track 1: multi-predicate MCTS
│   ├── multipred_probe.py             # Track 1: multi-predicate probe
│   ├── mine_pivot_instances.py        # Track 2: pivot mining pipeline
│   ├── train_head_mask.py             # Track 2: cache + train sparse head mask
│   └── augment_data.py                # Data augmentation utilities
├── src/
│   ├── models.py                      # ModelandTokenizer (nnsight wrapper)
│   ├── functional.py                  # generate_with_patch, get_hs, predict_next_token
│   ├── tokens.py                      # prepare_input (batched, left-padded)
│   └── goals/
│       ├── data.py                    # Track 1 data types
│       ├── eval.py                    # evaluate_single_code, run_evaluation
│       ├── functional.py              # apply_chat_template, supports_chat_template
│       ├── sampling.py                # MCTS sampling
│       ├── probe.py                   # Linear probe training
│       ├── augment.py                 # Track 1 augmentation
│       ├── head_loc_data.py           # Track 2 data types (SinglePredicateSample, PivotInstance, etc.)
│       ├── head_loc_pivot.py          # Track 2 prompt builders, next-token utilities
│       ├── head_loc_oracle.py         # Track 2 oracle pivot mining (Case 3, 3b, augmentation)
│       ├── head_loc_attn.py           # Track 2 caching + HeadMask + train_head_mask
│       └── head_loc_augment.py        # Track 2 predicate augmentation
├── data/
│   ├── incomplete_code.json           # Track 1 dataset
│   ├── multi_predicate/               # Track 1 multi-predicate tasks
│   └── head_loc/
│       ├── predicates/                # 25 predicate JSONs (Python + C++)
│       ├── global_pairs.json          # Oracle-verified contrastive predicate pairs
│       └── pairs/<model>.json         # Model-specific pairs + pivot instances
├── agent_reports/claude/                    # Analysis notes and design documents
├── setup_env.sh                       # One-shot environment setup script
└── env.yml                            # API keys + paths (create from env_demo.yml)
```

---

## Requirements

- Python >= 3.11
- PyTorch >= 2.7
- Transformers >= 4.55
- nnsight >= 0.5
- baukit (install separately, see Setup)
- rich (table display)
- CUDA GPU — tested on A100; large models (32B) need significant VRAM
