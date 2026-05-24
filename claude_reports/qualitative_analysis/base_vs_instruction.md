# Base vs Instruction-Tuned Models: Performance Comparison

## Overall Performance

| Model | Aligned | Adversarial | Inconsequential | Overall |
|-------|---------|-------------|-----------------|---------|
| Olmo-3-1125-32B (base) | 79.7% | 37.7% | 24.6% | 54.4% |
| Olmo-3.1-32B-Instruct | 97.5% | 70.1% | 98.2% | 89.3% |
| **Delta (Instruct - Base)** | **+17.8%** | **+32.4%** | **+73.6%** | **+34.9%** |
| | | | | |
| Qwen2.5-72B (base) | 80.5% | 41.6% | 56.1% | 63.1% |
| Qwen2.5-72B-Instruct | 94.1% | 74.0% | 82.5% | 85.3% |
| **Delta (Instruct - Base)** | **+13.6%** | **+32.4%** | **+26.4%** | **+22.2%** |

---

## Key Finding: Inconsequential Prefills Reveal Task Awareness

The biggest gap is on inconsequential prefills (e.g., `"True and"`, `"False or"`, `"bool("`). These should have no effect on semantics, yet:

| Prefill | Base Model Output | Instruct Model Output |
|---------|------------------|----------------------|
| `"True and"` | `True and True else False` (tautology) | `True and s.isupper() else False` (correct) |
| `"False or"` | `False or False else False` (tautology) | `False or s.isupper() else False` (correct) |
| `"3"` | `3 == 3 else False` (tautology) | `3 and all(char.isupper() for char in s) else False` (adapts!) |
| `"1 +"` | `1 + 1 == 2 else False` (tautology) | `1 + 2 * (n // 2) == n else False` (creative) |

---

## Error Types by Model Type

| Error Type | Base Models | Instruct Models |
|-----------|-------------|-----------------|
| Tautologies (`1+1==2`, `True and True`) | Very common | Rare |
| Task-irrelevant completions | Common | Rare |
| Syntax errors | Occasional | Rare |
| Overly complex but wrong logic | Rare | Occasional |
| Off-by-one errors | Rare | Occasional |

---

## Specific Examples

### Task: `is_odd`, Prefill: `"1 +"`

| Model | Generated Code | Result |
|-------|---------------|--------|
| Qwen2.5-72B (base) | `1 + 1 == 2 else False` | FAILED (tautology) |
| Qwen2.5-72B-Instruct | `1 + 2 * (n // 2) == n else False` | PASSED (creative!) |

### Task: `is_odd`, Prefill: `"(3"`

| Model | Generated Code | Result |
|-------|---------------|--------|
| Qwen2.5-72B (base) | `(3 * n + 1) % 2 == 1 else False` | FAILED (wrong math) |
| Qwen2.5-72B-Instruct | `(3 & n) == 1 or (3 & n) == 3 else False` | PASSED (clever bitwise!) |

### Task: `is_uppercase`, Prefill: `"3"`

| Model | Generated Code | Result |
|-------|---------------|--------|
| Olmo-32B (base) | `3 == 3 else False` | FAILED (ignores task) |
| Olmo-32B-Instruct | `3 and all(char.isupper() for char in s) else False` | PASSED |

---

## Summary

| Metric | Base Models | Instruct Models | Interpretation |
|--------|-------------|-----------------|----------------|
| Aligned | ~80% | ~96% | Both understand simple completions |
| Adversarial | ~40% | ~72% | Instruct 1.8x better at recovery |
| Inconsequential | ~40% | ~90% | Instruct maintains task awareness |
| Overall | ~59% | ~87% | +28% improvement from instruction tuning |

**Key Insight:** Base models perform local token prediction without understanding the task goal. Instruction tuning teaches models to maintain task awareness even when prefills try to derail them. The **73.6% gap on inconsequential prefills for OLMo** is striking evidence that instruction tuning instills goal-directed behavior.
