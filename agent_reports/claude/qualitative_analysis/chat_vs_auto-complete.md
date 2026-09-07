# Chat Template vs Auto-Complete Prompting

## Performance Comparison

| Model | Chat Template | wo_chat (auto-complete) | Delta |
|-------|--------------|------------------------|-------|
| Llama-3.3-70B-Instruct | ~85% | ~60% | +25% |
| Qwen2.5-72B-Instruct | ~85% | ~55% | +30% |

**Winner:** Chat template is significantly better (~25-30% improvement)

---

## Error Types by Prompting Style

| Error Type | Chat Template | wo_chat |
|-----------|--------------|---------|
| Placeholder code (`_____`, `???`, `# your code here`) | Rare | Common |
| Tautologies (`1 + 1 == 2`, `True and False`) | Rare | Common |
| Natural language in code | Never | Occasional |
| Incomplete/unclosed parens | Rare | Common |
| Undefined variables | Rare | Occasional |
| Overly complex solutions | Occasional | Rare |
| Off-by-one/subtle logic errors | Occasional | Rare |

---

## Concrete Examples

### Task: `is_uppercase`, Prefill: `"all("`

**Chat template — PASSED:**
```python
all(65 <= ord(c) <= 90 for c in s) else False
```

**wo_chat — FAILED (placeholder):**
```python
all(_____) else False
```

### Task: `is_sorted`, Prefill: `""` (empty)

**Chat template — PASSED:**
```python
lst == sorted(lst) else False
```

**wo_chat — SYNTAX_ERROR:**
```python
# your code here  else False
```

### Task: `is_uppercase`, Prefill: `"1 +"`

**Chat template — FAILED (off-by-one):**
```python
1 + sum(1 for c in s if c.isupper()) == len(s) else False
```

**wo_chat — FAILED (tautology, ignores task entirely):**
```python
1 + 1 == 2 else False
```

---

## Key Insights

1. **Chat template provides task context** — Models understand they're completing a function with specific semantics
2. **wo_chat loses task awareness** — Models often generate syntactically plausible but semantically irrelevant code
3. **wo_chat failures are more "catastrophic"** — Placeholders, tautologies, syntax errors
4. **Chat failures are more "subtle"** — Logic errors, off-by-one, over-complexity

---

## Suggestions for Prompting Styles

1. **Always use chat template for instruction-tuned models** — They're trained to expect it

2. **For auto-complete style, consider:**
   - Adding explicit docstrings with examples in the prompt
   - Using fill-in-middle (FIM) tokens if the model supports them
   - Providing more context before the completion point

3. **For adversarial robustness research:**
   - wo_chat results actually reveal more about "raw" model behavior
   - Chat template may mask fragile goal representations with instruction-following

4. **Hybrid approach for research:**
   ```python
   # Instead of raw auto-complete:
   def is_odd(n: int) -> bool:
       """Return True if n is odd, False otherwise."""
       return True if <prefill>
   ```
   Adding docstrings may help wo_chat without full chat template overhead.
