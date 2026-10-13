# Query transfer changes gaze without a consistent three-contrast rule

## Experimental setup

We used three matched configurations (ALT1–ALT3) to distinguish four hypotheses about which donor task a transferred source query selects. Letters denote task identities, not variable names. Every source lists tasks [a, b, c, d]. Every recipient lists [a, x, y, z] and has completed only a; every key/value (KV) donor has also completed only a.

Notation: an apostrophe marks a completed task, and the arrow points to the next frontier (the first unfinished task). For example, `a'b'c → c` means a and b are completed and c is next. Letters retain task-list order, not code-writing order. The recipient is `a'xyz → x` in every setup; the arrows below describe each prompt’s own frontier before transfer, not the patched gaze winner.

| Contrast | Source → frontier | Source writing order | Donor → frontier |
| --- | --- | --- | --- |
| ALT1 | `a'b'cd' → c` | a, b, d | `a'bde → b` |
| ALT2 | `a'b'cd' → c` | a, b, d | `a'def → d` |
| ALT3 | `a'b'c'd → d` | a, b, c | `a'cbe → c` |

### Four hypotheses and their predictions

| Hypothesis | Selection rule | ALT1 (out of 128) | ALT2 (out of 128) | ALT3 (out of 128) | Complete matches (out of 128; KV + source query) |
| --- | --- | --- | --- | --- | --- |
| Set difference (SET_DIFF) | Choose the first donor task absent from the source’s completed set. | e 65 | e 56 | e 85 | 28 |
| Order ID | Use the source’s first unfinished task’s list position to select from the donor list. | d 57 | e 56 | e 85 | 11 |
| Completed count | Use the number of completed source tasks as a zero-based donor-list index: three completed tasks selects the fourth donor task. | e 65 | f 47 | e 85 | 16 |
| Induction (last-written successor) | Find the final written source task in the donor list and choose the following task. | e 65 | e 56 | b 35 | 15 |

Each ALT cell shows the predicted task and its individual success count out of 128 evaluations under KV transfer plus source-query replacement. Success means that task is the gaze winner among all four donor tasks, not merely that it beats the paired alternative. Counts are aggregated from the saved winner transitions; predictions shared by multiple hypotheses have identical counts and are not independent evidence for each hypothesis.

Complete matches count matched (ALT1, ALT2, ALT3) tuples in which all three predicted tasks are the gaze winners among the four donor tasks, under KV transfer plus source-query replacement. The denominator is 128 tuples, not individual runs; 58 tuples match none of the four hypotheses. These are descriptive pattern counts, not a formal hypothesis ranking. [Figure 2 shows their pointwise 95% bootstrap intervals.](#triplets)

These predictions determine the pairwise gaze comparisons. In ALT1, order ID maps the source frontier c’s third list position to the donor’s third task, d; set difference, completed count, and induction instead predict e. We therefore compare e versus d. In ALT2, completed count predicts f while the other three predict e, giving e versus f. In ALT3, induction predicts b while the other three predict e, giving e versus b. ALT1–ALT3 are contrastive setups, not hypotheses: a preference for e favors a shared prediction, not a unique hypothesis. Complete prediction triplets are written in ALT1, ALT2, ALT3 order: `eee`, `dee`, `efe`, and `eeb`, respectively. Here, induction refers specifically to the last-written-successor rule, not every possible induction mechanism.

### Shared intervention and readout

We compared two conditions in frozen `Qwen2.5-Coder-32B-Instruct` (BF16, eager attention, 250 fixed heads). Both transferred donor task-list KV into the recipient. The second additionally replaced the query before RoPE, only at the recipient’s final equals token, with the source query.

The panel contains 128 matched groups: 32 each for `int`, `str`, `list[int]`, and `list[str]`, giving 384 configurations across the three contrasts. Gaze is mean attention probability × effective-value L2 norm over donor-description tokens and heads, with full causal keys. Winners are chosen across all four donor tasks, not just the pair in the table. No training or token generation was involved; 10,000 linked whole-group bootstrap draws give pointwise 95% intervals. Further implementation details appear in Method and coverage.

## Results overview

Replacing the source query changes gaze, but none of the four hypotheses predicts the gaze winner in all three setups for a majority of the 128 groups. This is partial signal in a gaze readout, not identification of the model’s natural algorithm.

## Results by hypothesis

The results below refer to KV transfer plus source-query replacement. Pooled margins combine all 128 groups; brackets give pointwise 95% bootstrap intervals. A pairwise gaze preference and a complete three-setup match answer different questions: the former compares two tasks on average, while the latter requires the predicted task to win over all four donor tasks in every setup.

### Set difference: some matching groups, no consistent identification

**Prediction.** Select e in all three setups, giving `eee`. This requires selecting the first donor task absent from the source’s completed set; in ALT2, that is e before f.

**Results.** The complete `eee` pattern occurs in 28 of 128 groups. The pooled ALT3 comparison favors e over b, but order ID and completed count also predict e there. The ALT1 and ALT2 pooled intervals include zero, so those comparisons do not distinguish set difference from order ID and completed count, respectively.

**Verdict.** Some groups match set difference, but the evidence does not identify it as a consistent rule. Its complete pattern does not cover a majority, and the ALT3 preference is not unique evidence for set difference.

### Order ID: its distinguishing comparison is unresolved

**Prediction.** Select d, e, e across ALT1–ALT3, giving `dee`. ALT1 is the distinguishing setup: order ID predicts d, whereas all three other hypotheses predict e.

**Results.** The patched pooled ALT1 margin, e−d, is 0.006402 [−0.006476, 0.01938]. The complete `dee` pattern occurs in 11 of 128 groups. The average preference for e in ALT3 agrees with order ID, but also agrees with set difference and completed count.

**Verdict.** ALT1 does not establish an average preference for either prediction. This does not demonstrate equivalence or rule out order ID; it leaves the distinguishing comparison unresolved. Complete matches do not support it as a consistent rule across groups.

### Completed count: its distinguishing comparison is unresolved

**Prediction.** Select e, f, e across ALT1–ALT3, giving `efe`. ALT2 is the distinguishing setup: completed count predicts f, whereas all three other hypotheses predict e.

**Results.** The patched pooled ALT2 margin, e−f, is −0.0009371 [−0.01373, 0.01182]. The complete `efe` pattern occurs in 16 of 128 groups. The average preference for e in ALT3 agrees with completed count, but also agrees with set difference and order ID.

**Verdict.** ALT2 does not establish an average preference for either prediction. Its interval does not establish equivalence or rule out completed count. Complete matches do not support it as a consistent rule across groups.

### Induction: the ALT3 average favors the competing prediction

**Prediction.** Under the last-written-successor rule, select e, e, b across ALT1–ALT3, giving `eeb`. In ALT3, the final written source task is c; its successor in the donor list [a, c, b, e] is b. All three other hypotheses predict e.

**Results.** The patched pooled ALT3 margin, e−b, is 0.04119 [0.02765, 0.05498], favoring the prediction shared by set difference, order ID, and completed count over the induction prediction. The complete `eeb` pattern nevertheless occurs in 15 of 128 groups.

**Verdict.** The average pairwise gaze result weighs against this specific induction prediction in ALT3. It does not select among the other three hypotheses, exclude induction-like behavior in individual groups, or test every induction mechanism.

**Overall insight.** None of the four complete prediction patterns covers a majority, and 58 of 128 groups match none of them. These are descriptive matches and pairwise gaze comparisons, not a formal hypothesis ranking or identification of the model’s natural algorithm. [Figure 1 shows the comparisons by datatype](#margins); [Figure 2 shows complete-pattern counts](#triplets).

[Inspect all configuration scores](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/configurations.html) | [Inspect all cell means and transitions](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/cells.html) | [Inspect representative prompts and tokens](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/representatives.html)

## Supporting evidence across setups and datatypes

Figure 1 compares each hypothesis’s distinguishing prediction with the prediction shared by the other three: e−d tests order ID in ALT1, e−f tests completed count in ALT2, and e−b tests induction in ALT3. Positive margins favor the shared prediction; negative margins favor the singled-out hypothesis’s prediction. Each cell uses 32 matched groups, with KV-only and KV-plus-source-query mean margins and pointwise 95% intervals; zero marks equal pairwise gaze scores.

### Figure 1: Before and after margins in all 12 cells

Interactive chart omitted from Markdown. [Open standalone figure](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/figures/margins/margins.html).

Mean pairwise gaze margins before and after query replacement, with pointwise 95% intervals. ALT1 compares e with order ID’s d; ALT2 compares e with completed count’s f; ALT3 compares e with induction’s b. In each setup, e is predicted by the other three hypotheses.

A positive margin does not imply e beats all four donor tasks or uniquely support set difference. The hypothesis subsections above interpret the pooled comparisons; this figure preserves the variation across datatypes rather than treating each setup as a hypothesis.

## No proposal matches most complete groups

The SET_DIFF triplet eee occurs in 28 of 128 groups, while 58 match none of the four proposals. Counts below use complete ALT1/ALT2/ALT3 winners, with stratified whole-group bootstrap pointwise 95% intervals and denominator 128.

### Figure 2: Complete patched triplets

Interactive chart omitted from Markdown. [Open standalone figure](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/figures/triplets/triplets.html).

Patched triplet counts among 128 matched groups, with pointwise 95% intervals.

Triplets encode the winning donor role in ALT1, ALT2, ALT3 order. The four proposals are SET_DIFF `eee` (28), order ID `dee` (11), completed count `efe` (16), and induction (last-written successor) `eeb` (15); other contains 58 groups.

A complete match means that one hypothesis’s predicted task wins over all four donor tasks in each of the three setups within the same group. It is stricter than a positive pooled pairwise margin. These counts describe pattern matches, not posterior probabilities or a formal ranking of hypotheses.

## A set-difference match and a group outside all four hypotheses

The same transfer procedure can yield the complete set-difference prediction, eee, or a pattern outside all four proposals. These two examples are the first frozen groups meeting those respective post-analysis selection rules, shown at ALT1; the companion preserves all eight selected groups.

### int-group-01: eee

This group matches set difference’s complete prediction, eee. ALT1 winner: b → e; full triplet order is ALT1, ALT2, ALT3. ALT1 alone would not distinguish set difference from completed count or induction, which also predict e there.

The source and donor excerpts show actual instructions and completed code, with the common system prefix omitted. The token row below locates the first scored e-description token using the saved offsets and transfer map.

**Source query prompt, unpadded excerpt**

```python
Write a Python program that implements a function `func` which, given an integer `x`, computes:
(1) Absolute value, including zero.
(2) Magnitude of x with its lowest set bit cleared; zero stays zero.
(3) Largest decimal digit of abs(x); zero gives zero.
(4) Whether x is divisible by two; zero is even.
<|im_end|>
<|im_start|>assistant
<code>
def func(x: int):
    n = abs(x)
    u = abs(x) & (abs(x) - 1)
    s = x % 2 == 0
    t =
```

**KV donor prompt, unpadded excerpt**

```python
Write a Python program that implements a function `func` which, given an integer `x`, computes:
(1) Absolute value, including zero.
(2) Magnitude of x with its lowest set bit cleared; zero stays zero.
(3) Whether x is divisible by two; zero is even.
(4) Whether x is a nonnegative perfect square; zero is true.
<|im_end|>
<|im_start|>assistant
<code>
def func(x: int):
    n = abs(x)
    t =
```

ALT1 e score: 0.03483 → 0.1373. Donor token 107, ID 13139, offset [453, 461], text `" Whether"` → recipient position 107 (native token ID 8116).

[Inspect full prompt and token mapping](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/representatives.html#int-group-01)

### int-group-04: ddc

This group’s ddc triplet matches none of the four complete predictions. ALT1 winner: b → d; full triplet order is ALT1, ALT2, ALT3. Although d in ALT1 agrees with order ID, the full triplet does not match its dee prediction.

The source and donor excerpts show actual instructions and completed code, with the common system prefix omitted. The token row below locates the first scored e-description token using the saved offsets and transfer map.

**Source query prompt, unpadded excerpt**

```python
Write a Python program that implements a function `func` which, given an integer `x`, computes:
(1) Reverse decimal digits of abs(x), dropping leading zeros in the result.
(2) Magnitude of x with its lowest set bit cleared; zero stays zero.
(3) Square of the signed integer.
(4) Half of x rounded toward negative infinity.
<|im_end|>
<|im_start|>assistant
<code>
def func(x: int):
    w = int(str(abs(x))[::-1])
    q = abs(x) & (abs(x) - 1)
    h = x // 2
    s =
```

**KV donor prompt, unpadded excerpt**

```python
Write a Python program that implements a function `func` which, given an integer `x`, computes:
(1) Reverse decimal digits of abs(x), dropping leading zeros in the result.
(2) Magnitude of x with its lowest set bit cleared; zero stays zero.
(3) Half of x rounded toward negative infinity.
(4) Whether x is a nonnegative perfect square; zero is true.
<|im_end|>
<|im_start|>assistant
<code>
def func(x: int):
    w = int(str(abs(x))[::-1])
    s =
```

ALT1 e score: 0.04247 → 0.08011. Donor token 111, ID 13139, offset [491, 499], text `" Whether"` → recipient position 111 (native token ID 19).

[Inspect full prompt and token mapping](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/representatives.html#int-group-04)

## Method and coverage

The comparison holds donor KV transfer fixed and adds source-query replacement only at the recipient’s final equals token. All 384 configurations are retained: 128 matched groups × three contrasts, two primary conditions (768 observations), and four donor scores per condition (3072 rows).

For the configurations introduced above, the KV-only frontier targets are b, d, and c in ALT1, ALT2, ALT3 order.

### Design and readout

- **Model:** `Qwen2.5-Coder-32B-Instruct`, revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, BF16/eager, batch size 1.

- **Data:** 32 groups each for int, str, list[int], list[str], with three matched contrasts and 250 fixed heads.

- **Intervention:** Donor task-list KV at selected heads, donor K rotated at recipient positions, native unselected sibling heads, source Q before RoPE at final equals only.

- **Alignment:** Common newline-token task-list width across recipient and all three donors within each group; source unchanged.

- **Score and winner:** Saved FP32 mean attention × effective-value L2 norm across heads and donor-description tokens; argmax across all four tasks, exact ties retained.

- **Uncertainty:** 10,000 linked whole-group percentile bootstrap draws, seed 82391; pooled estimates preserve datatype counts.

- **Panel constraints:** All nine task IDs differ within each group; scalar b,c,d,e,f families are distinct while a,x,y,z may overlap, list groups use nine distinct families, and 16 earlier assignments are excluded.

- **Scope:** No training, generation, LLM judging, OV boost, or train/test benchmark split.

## Validation

Independent raw-array reaggregation passed, with maximum mean/sum discrepancy 5.123 × 10⁻⁷ against the saved FP32 scores (tolerance 10⁻⁶). Routing, query replacement, rotation and parity checks were exact; attention-output reconstruction passed its BF16 tolerance.

Logit parity was rechecked on four shared recipient inputs and all twelve pilot transfer configurations. Selected/unselected routing and attention-output reconstruction were checked throughout the grid.

There were no primary ties. Clean-recipient frontier gaze matched 124/128 inputs and clean-source frontier gaze matched 253/256 unique inputs; diagnostics did not filter the primary panel.

## Limitations

- Gaze does not measure execution correctness or identify a natural algorithm; no tokens were generated. Source-copy and original-recipient count targets are absent from the donor inventory.

- Groups, not task rows, are the resampling unit. Predicates recur and scalar/list family constraints differ, so these intervals do not establish fresh-family generalization.

- Intervals are pointwise, not multiplicity-adjusted. Numerical leads do not establish a ranking, and zero-count bootstrap intervals do not establish impossibility.

## Resources

### Models and checkpoints

- [`Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct): evaluated checkpoint

### Datasets

- [Frozen Python panel](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/panel.json): complete configuration, task definitions and heads

### Artifacts

- [Task scores CSV](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/task_scores.csv): all primary task scores

- [Task scores JSON](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/task_scores.json): full-precision primary scores and metadata

- [Configuration scores JSON](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/scores.json): primary and diagnostic condition scores

- [Configuration CSV](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/configurations.csv): full IDs, scores, margins and provenance

- [Reaggregation validation](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/validation.json): independent raw-array audit

- [Instrumentation validation](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/instrumentation_summary.json): routing, parity and reconstruction checks

- [Panel validation](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/analysis_evidence.zip): frozen panel coverage checks

- [Model provenance](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/model_provenance.json): checkpoint revision and file sizes

- [Storage manifest](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/storage.json): actual model, environment and array sizes

- [Representative examples JSON](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/data/representatives.json): all eight groups and complete token maps

- [Analysis evidence ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/analysis_evidence.zip): complete analysis outputs

- [Runnable code ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/gaze_grid_code.zip): experiment code, README and configuration

- [Token maps and checks ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/tokenmaps_checks.zip): all prompt mappings and instrumentation records

- [Integer arrays ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/raw_int.zip): lossless raw attention and effective-value norms

- [String arrays ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/raw_str.zip): lossless raw attention and effective-value norms

- [Integer-list arrays ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/raw_list_int.zip): lossless raw attention and effective-value norms

- [String-list arrays ZIP](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/downloads/raw_list_str.zip): lossless raw attention and effective-value norms

- [Margin figure](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/figures/margins/margins.html): standalone interactive plot

- [Triplet figure](https://u-42bd8c2570c6--researcher-david-bau.prod2-usw2.runtime.silico.run/api/workspaces/ws_01m2kvfmf8fg295e9rvce3vhe7/experiments/exp_01m4f3g9npexwv2rexndg2kcp7/app/st1.1791959252.86829dface68df3c463d84e63c6f44b070efcaeeaaa89205133a25efbf4a276d/@apprev/apprev_0c4f263c813201017b4f4eab7ccd27536f5ac2ec3806514b71a24e86ab56a305/figures/triplets/triplets.html): standalone interactive plot
