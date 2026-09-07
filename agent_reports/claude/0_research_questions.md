# Research Questions & Next Steps

## Immediate Next Steps (Low-hanging fruit)

### 1. Quantify the "Goal Strength" Phenomenon

The data shows goals exist but compete with local context. Formalize this:

```python
# Define prefill categories by "adversarial strength"
prefill_categories = {
    "neutral": ["", "n", "s"],           # Don't push toward wrong answer
    "compatible": ["n &", "n %", "s =="], # Align with goal
    "adversarial": ["1+", "not 1"],       # Push away from goal
    "partial_method": ["s.up", "c.low"],  # Trap into typos
}

# Hypothesis: Pass rate should be: compatible > neutral > adversarial
# Test this systematically across all models/tasks
```

### 2. Expand the Evaluation Dataset

Current dataset has 14 tasks. Strengthen it:
- **More prefill types:** Add graduated difficulty (e.g., `"1"`, `"1+"`, `"1+0"`, `"1+0+"`)
- **More tasks:** Include tasks with multiple valid solutions (sorting algorithms) as control
- **Longer completions:** Current tasks are one-liners; test multi-statement functions

### 3. Control Experiment: Multi-Goal Tasks

Test the hypothesis that single-goal tasks show more consistent adaptation:

```python
# Single goal (current): "check if odd" - only one correct answer
# Multi-goal: "sort this list" - bubble, quick, merge all valid

# Prediction: Multi-goal tasks should show LESS consistent adaptation
# because there's no single goal to maintain
```

---

## Mechanistic Investigation

### 4. Probe for Goal Representations

Where are goals encoded? Use linear probing:

```python
# Collect hidden states at the completion point across tasks
# Train probes to predict: (a) task identity, (b) whether model will succeed

# Key positions to probe:
# - After function signature (does model know the goal here?)
# - After "return True if" (goal should be fully formed)
# - After prefill (is goal still intact or corrupted?)

def collect_goal_probing_data(mt, tasks, prefills):
    data = []
    for task in tasks:
        for prefill in prefills:
            prompt = task["instruction"] + task["code"] + prefill
            h_states = get_hs(mt, prompt, layers=range(mt.n_layer))

            # Also record: did model succeed on this?
            output = generate(mt, prompt)
            success = evaluate(output, task["unit_tests"])

            data.append({
                "task": task["function_name"],
                "prefill": prefill,
                "prefill_type": categorize_prefill(prefill),
                "h_states": h_states,
                "success": success
            })
    return data
```

**Key questions:**
- At which layer does task identity become linearly separable?
- Does probe accuracy drop after adversarial prefills?
- Is there a "goal subspace" that's consistent across prefills?

### 5. Activation Patching: Can Goals Be Transferred?

The critical mechanistic test:

```python
# Source: "is_odd" with neutral prefill (model will succeed)
# Target: "is_odd" with adversarial prefill "1+" (model will fail)

# Patch source activations into target
# If goals are localized, patching should "rescue" the target

def goal_transfer_experiment(mt, task, good_prefill, bad_prefill):
    # Get activations from successful context
    source_prompt = task["instruction"] + task["code"] + good_prefill
    source_h = get_hs(mt, source_prompt, layer=goal_layer, pos=-1)

    # Patch into failing context
    target_prompt = task["instruction"] + task["code"] + bad_prefill

    # Generate with patched activations
    output = generate_with_patch(
        mt, target_prompt,
        patches=[PatchSpec(location=(goal_layer, -1), patch=source_h)]
    )

    # Does patching rescue the generation?
    return evaluate(output, task["unit_tests"])
```

Sweep across layers to find where goals are stored.

### 6. Cross-Task Goal Patching

Even more interesting: can you transfer the "goal structure" between related tasks?

```python
# Source: is_odd with "n &" prefill -> generates "n & 1"
# Target: is_even with "n &" prefill -> should generate "n & 1 == 0"

# Hypothesis: The "oddness checking" goal might be related to "evenness checking"
# Patching might cause is_even to generate is_odd logic (or vice versa)
```

---

## Understanding the Competition

### 7. Logit Lens Analysis

When does the model "commit" to the wrong path?

```python
# For adversarial prefill "1+" on is_odd:
# Apply logit lens at each layer to see probability of goal-consistent tokens

def trace_goal_commitment(mt, prompt, goal_tokens, distractor_tokens):
    """Track when model commits to wrong path."""
    h_states = get_all_layer_states(mt, prompt)

    commitment_trace = []
    for layer, h in enumerate(h_states):
        logits = logit_lens(mt, h)

        goal_prob = sum(logits[t] for t in goal_tokens)
        distractor_prob = sum(logits[t] for t in distractor_tokens)

        commitment_trace.append({
            "layer": layer,
            "goal_prob": goal_prob,
            "distractor_prob": distractor_prob,
            "ratio": goal_prob / (distractor_prob + 1e-8)
        })

    return commitment_trace

# For is_odd with "1+" prefill:
# goal_tokens = tokens for "%", "&", "n"
# distractor_tokens = tokens for "=", "2", numbers
```

**Expected finding:** Early layers might still favor goal-consistent tokens, but later layers shift to distractor tokens. This would show WHERE the goal is lost.

### 8. Attention Pattern Analysis

What does the model attend to when it fails vs succeeds?

```python
# When model succeeds: likely attends to function name "is_odd", return type
# When model fails: might over-attend to the adversarial prefill

# Compare attention patterns between:
# - is_odd with "" prefill (success)
# - is_odd with "1+" prefill (failure)
```

---

## Strengthening Goals (Intervention Research)

### 9. Can We "Boost" the Goal Signal?

If goals are represented, can we amplify them?

```python
# Extract goal direction from successful completions
# Add it to failing completions

successful_h = get_hs(mt, "is_odd... n &", layer=L)  # Will succeed
failing_h = get_hs(mt, "is_odd... 1+", layer=L)      # Will fail

# Compute goal direction
goal_direction = successful_h - failing_h

# Intervene: add goal direction to failing context
boosted_h = failing_h + alpha * goal_direction
output = generate_with_patch(mt, failing_prompt, patch=boosted_h)
```

### 10. Prompt Engineering for Goal Robustness

Test if explicit goal statements help:

```python
# Current: "Write a function that checks if a number is odd"
# Enhanced: "Write a function that checks if a number is odd.
#           The function MUST return True for odd numbers and False for even numbers.
#           Regardless of how the code starts, ensure the logic checks oddness."

# Does explicit goal reinforcement improve adversarial prefill performance?
```

---

## Connecting to Schut et al.

### 11. Replicate Goal Mining

The research log mentions replicating Schut et al.'s recipe. Key steps:

1. **Collect trajectories:** Full generation traces with intermediate states
2. **Identify goal-relevant states:** States that predict successful completion
3. **Extract goal features:** PCA/clustering on goal-relevant states
4. **Validate:** Show these features transfer across contexts

---

## Suggested Priority Order

| Priority | Experiment | Why |
|----------|-----------|-----|
| 1 | Linear probing for goals (#4) | Foundational — establishes IF goals are represented |
| 2 | Logit lens analysis (#7) | Shows WHERE goals are lost |
| 3 | Activation patching (#5) | Proves goals are CAUSAL |
| 4 | Prefill categorization (#1) | Formalizes the behavioral findings |
| 5 | Cross-task patching (#6) | Tests goal generality |
| 6 | Goal boosting (#9) | Practical intervention |

---

## Concrete Deliverables

1. **Paper Figure 1:** Prefill difficulty gradient showing pass rates across categories
2. **Paper Figure 2:** Probing accuracy by layer showing where goals emerge
3. **Paper Figure 3:** Logit lens showing goal commitment trajectory
4. **Paper Figure 4:** Patching experiments showing goal transfer
5. **Main Claim:** *"LLMs encode task goals in middle layers, but these representations are fragile and can be overridden by adversarial local context"*
