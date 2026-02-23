# Goals: Do LLMs Have Consistent Intermediate Goals?

Research codebase investigating whether LLMs pursue consistent intermediate goals during code generation.

## Research Question

When an LM writes code, does it have a stable internal representation of *what it intends to do next* — independent of *how* it will do it?

We distinguish three levels:
```
Broader goal:      implement the function correctly
Instrumental goal: check whether the input is odd        ← target of this research
Implementation:    n % 2 != 0   vs   n & 1 == 1   vs   not n % 2
```

The core question is whether there exists a linear direction in the model's hidden state that encodes the **instrumental goal** — firing consistently for "about to check parity" regardless of which specific implementation the model produces. If such a direction exists and causally influences generation, it suggests the model has pre-committed to a sub-goal before writing the tokens that realize it.

This is distinct from a *competence* direction (predicting whether generation will succeed) — we want a direction that captures *intent*, not *skill*.

**Key Research Directions:**
- **Goal representations**: Do Code-LLMs have stable hidden-state representations of instrumental sub-goals?
- **Mining Goals**: Can we extract linear directions that capture goal intent, following Schut et al.?
- **Causality**: Do these directions causally influence generation, or are they merely correlated with outcome?
- **Generalization**: Does a direction trained on one context transfer to a different context involving the same sub-goal?

## Methodology: Prefill Probing

We use a **prefill probing** methodology:

1. Design simple code completion tasks with a unique correct solution (a single "goal")
2. Insert different "prefills" before letting the model generate
3. Check if the model adapts its generation to still achieve the correct outcome

**Example:**
```python
def is_odd(n: int) -> bool:
    return True if <PREFILL>
```

With prefills like:
- `""` (empty) → generates `n % 2 != 0 else False`
- `"n &"` → generates `n & 1 else False`
- `"1+"` → should generate something like `1 + n&1 - 1 else False` (adapt to the prefill while maintaining correctness)
- `"not"` → generates `not n % 2 == 0 else False`

If the model successfully adapts to different prefills while still producing correct code, it suggests the model is pursuing a consistent goal rather than generating token-by-token without foresight.

## Setup

```bash
conda env create -f conda_env.yml
pip install git+https://github.com/davidbau/baukit  # Required, not installed by conda
```

Create `env.yml` in root (see `env_demo.yml` for template) with API keys and directory paths.

## Usage

See `notebooks/00_goals.ipynb` for the main experimentation notebook. The notebook demonstrates:
- Loading code LLMs (OLMo, LLaMA, Qwen, etc.)
- Running code completion with various prefills
- Evaluating generated code against unit tests
- Analyzing model behavior across prefills

## Project Structure

```
├── notebooks/
│   └── 00_goals.ipynb      # Main experimentation notebook
├── src/
│   ├── models.py           # ModelandTokenizer wrapper for nnsight
│   ├── functional.py       # Generation and analysis functions
│   ├── tokens.py           # Tokenization utilities
│   └── ...
├── sample_eval.json        # Evaluation dataset with tasks and prefills
├── conda_env.yml           # Conda environment
└── env.yml                 # API keys and paths (create from env_demo.yml)
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.7
- Transformers >= 4.55
- nnsight >= 0.5
- CUDA-capable GPU (tested on A100)
