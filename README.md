# Goals: Do LLMs Have Consistent Intermediate Goals?

Research codebase investigating whether LLMs pursue consistent intermediate goals during code generation.

## Research Question

When an LM performs a task like writing code, does it have consistent intermediate goals? For example, when writing control-flow logic like `if b&1 == 0`, does the LM "know" it's going to write this logic ahead of time, or is it figuring it out token by token?

**Key Research Directions:**
- **Goals**: Do Code-LLMs pursue consistent goals during generation?
- **Mining Goals**: Can we extract/identify these goals using interpretability techniques (following Schut et al.)?
- **Superhuman Knowledge**: Does RLVR training allow models to discover non-conventional approaches?

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
