# LM Performance on Prefill Probing

## Overall Results (16 Models)

| Model | Aligned | Adversarial | Inconsequential | Overall |
|-------|---------|-------------|-----------------|---------|
| Olmo-3.1-32B-Instruct | 97.5% | 70.1% | 98.2% | 89.3% |
| Qwen2.5-Coder-32B-Instruct | 94.9% | 74.0% | 86.0% | 86.5% |
| Llama-3.3-70B-Instruct | 99.2% | 63.6% | 87.7% | 85.7% |
| Qwen2.5-72B-Instruct | 94.1% | 74.0% | 82.5% | 85.3% |
| CWM | 93.2% | 58.4% | 86.0% | 81.0% |
| GPT-OSS-20B | 94.9% | 51.9% | 89.5% | 80.6% |
| Qwen3-Coder-30B-A3B-Instruct | 90.7% | 53.2% | 87.7% | 78.6% |
| Qwen3-Coder-Next | 92.4% | 57.1% | 77.2% | 78.2% |
| DeepSeek-Coder-V2-Lite-Instruct | 90.7% | 41.6% | 84.2% | 74.2% |
| Qwen2.5-Coder-14B-Instruct | 91.5% | 49.4% | 68.4% | 73.4% |
| Qwen2.5-Coder-32B (base) | 89.8% | 48.1% | 71.9% | 73.0% |
| deepseek-coder-33b-base | 93.2% | 40.3% | 71.9% | 72.2% |
| Qwen2.5-72B (base) | 80.5% | 41.6% | 56.1% | 63.1% |
| starcoder2-15b | 89.8% | 39.0% | 35.1% | 61.9% |
| Olmo-3-1125-32B | 79.7% | 37.7% | 24.6% | 54.4% |
| Olmo-3.1-7B-RL-Zero-Code | 77.1% | 22.1% | 35.1% | 50.8% |

All models handle aligned prefills well (77-99%), but adversarial prefills expose significant fragility (22-74%).

**Key Observations:**
- Adversarial prefills cause a ~35 percentage point drop from aligned performance
- Top models on adversarial robustness: Qwen2.5-Coder-32B-Instruct & Qwen2.5-72B-Instruct (tied at 74%), Olmo-3.1-32B-Instruct (70%)
- Worst: Olmo-3.1-7B-RL-Zero-Code (22% adversarial) despite RL training
- New models CWM (58.4%) and Qwen3-Coder-Next (57.1%) are mid-pack on adversarial robustness

---

## Detailed Analysis

### Model Robustness Rankings (Adversarial Prefills)

| Rank | Model | Adversarial Pass Rate | Gap (Aligned - Adv) |
|------|-------|----------------------|---------------------|
| 1 | Qwen2.5-Coder-32B-Instruct | 74.0% | 20.9% (most robust) |
| 2 | Qwen2.5-72B-Instruct | 74.0% | 20.1% |
| 3 | Olmo-3.1-32B-Instruct | 70.1% | 27.4% |
| 4 | Llama-3.3-70B-Instruct | 63.6% | 35.6% |
| 5 | CWM | 58.4% | 34.8% |
| 6 | Qwen3-Coder-Next | 57.1% | 35.3% |
| 7 | Qwen3-Coder-30B-A3B-Instruct | 53.2% | 37.5% |
| 8 | GPT-OSS-20B | 51.9% | 43.0% |
| 9 | Qwen2.5-Coder-14B-Instruct | 49.4% | 42.1% |
| 10 | Qwen2.5-Coder-32B (base) | 48.1% | 41.7% |
| 11 | Qwen2.5-72B (base) | 41.6% | 38.9% |
| 12 | DeepSeek-Coder-V2-Lite-Instruct | 41.6% | 49.1% |
| 13 | deepseek-coder-33b-base | 40.3% | 52.9% |
| 14 | starcoder2-15b | 39.0% | 50.8% |
| 15 | Olmo-3-1125-32B | 37.7% | 42.0% |
| 16 | Olmo-3.1-7B-RL-Zero-Code | 22.1% | 55.0% (least robust) |

### Universal Breaking Prefills (ALL models fail)

| Prefill | Task | Why It Breaks |
|---------|------|---------------|
| `"1 +"` | is_even | Generates tautologies like `1 + 1 == 2` |
| `"(1 +"` | is_even | Same issue with parentheses |
| `"1 +"` | is_uppercase | Type confusion |
| `"not (1"` | is_odd | Logic inversion |
| `"3"` | is_palindrome | Numeric literal confuses semantics |

---

## Most Interesting Patterns

1. **Instruction-tuning significantly improves adversarial robustness:**
   - Qwen2.5-Coder-32B-Instruct (74.0%) vs Qwen2.5-Coder-32B base (48.1%) = **+25.9%**
   - Qwen2.5-72B-Instruct (74.0%) vs Qwen2.5-72B base (41.6%) = **+32.4%**
   - Both represent massive improvements from instruction-tuning

2. **RL training doesn't help robustness** — Olmo-7B-RL-Zero-Code is the WORST model (22.1% adversarial)

3. **Scale matters for robustness** within model families:
   - Qwen2.5-Coder: 14B-Instruct (49.4%) < 32B-Instruct (74.0%)
   - Olmo: 7B (22%) << 32B-Instruct (70%)

4. **Qwen2.5-Coder-32B-Instruct and Qwen2.5-72B-Instruct are tied** for best adversarial robustness:
   - Both at 74.0% adversarial pass rate
   - Smallest aligned-adversarial gaps (20.9% and 20.1% respectively)
   - Top 4 in overall performance (86.5% and 85.3%)

5. **New model observations:**
   - CWM (81.0% overall, 58.4% adversarial) shows strong general capability with moderate robustness
   - Qwen3-Coder-Next (78.2% overall, 57.1% adversarial) is comparable to CWM in robustness
   - Qwen2.5-72B base (63.1% overall, 41.6% adversarial) significantly underperforms its instruct variant
   - Olmo-3-1125-32B (54.4% overall, 37.7% adversarial, 24.6% inconsequential) struggles especially on inconsequential prefills

6. **Some models uniquely handle certain adversarial prefills:**
   - Qwen-72B-Instruct uniquely passes `"1 +"` on is_palindrome, is_digit_string
   - Llama-70B uniquely passes `"-n <"` on is_non_negative
   - OLMo-32B-Instruct uniquely passes `"len(lst) <"` on has_duplicates

---

## Research Implications

The 20-55% gap between aligned and adversarial performance across all 16 models confirms:

1. **Goals exist** — models succeed on aligned prefills (77-99%)
2. **Goals are fragile** — adversarial prefills break them (22-74%)
3. **Goal robustness correlates with scale and instruction-tuning**, not RL training
4. **Instruction-tuning on code** significantly strengthens goal representations
5. **Base vs instruct comparisons** within same architecture (Qwen2.5-72B, Qwen2.5-Coder-32B) show instruction-tuning adds 26-32% adversarial robustness

This suggests goals are represented but easily overridden by misleading local context — consistent with the **"weak goals" hypothesis**.
