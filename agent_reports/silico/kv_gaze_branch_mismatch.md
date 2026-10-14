# KV-on gaze reports the wrong attention branch

## Scope and status

- Repository: `arnab-api/goals`.
- Reviewed commit: **`6a9f33ca8316a6023b38d0a7034232979faba8bd`**.
- Repository line numbers below refer to that commit, not an unverified newer upstream revision.
- Affected implementation: headwise KV transfer in `src/diff/functional.py`, consumed by `src/attention.py` and the gaze-based task selectors.
- Evidence: source inspection, plus a separately implemented and validated effective-branch observer in the completed ALT1 smoke, [exp:exp_01m4ehsf78efcs34hnm8ntwdw5].
- **This is a report, not an applied repository patch.** No repository files were modified and no new model run was performed to write it.

## 1. Executive finding

The headwise KV wrapper correctly routes donor-KV attention **outputs** into the selected query heads, but returns **recipient-branch attention weights** for those same heads. The value-weighting helper then uses **recipient-side cached values**, rather than the effective donor values.

Consequently, KV-on gaze is not a faithful measurement of the attention computation used by the transferred heads. In `gaze_select` and `gaze_select_restrict`, this is consequential beyond visualization: the score's argmax chooses a task, which subsequently conditions a probe or determines the restricted-generation whitelist.

This finding does **not** establish that KV transfer itself failed, that every selected task is wrong, or that all historical generation results are invalid. It establishes a mismatch between the routed computation and the diagnostic used to explain or select it. Its direction and magnitude must be measured, not assumed.

### Terminology

The code calls the recipient-side branch `out_clean`. That name must not be interpreted as an independent, entirely clean model run. Earlier-layer interventions can already have changed the residual stream and recipient-side projections. Here, **native/recipient branch** means the branch before the additional task-span KV replacement at the current layer.

## 2. Root cause, with exact locations

### 2.1 Outputs and returned weights refer to different computations

In `src/diff/functional.py::_make_headwise_forward`:

| Lines | Operation | Consequence |
|---|---|---|
| 637–641 | Compute Q/K/V projections and apply RoPE. | These are the current recipient-trajectory projections, including any active Q patch. |
| 642–648 | Update the cache with recipient-side K/V. | Donor K/V are not written into this cache. This is intentional routing behavior. |
| 650–660 | Compute `out_clean, w = eager(..., k, v, ...)`. | `w` describes the recipient branch. |
| 668–671 | Clone K/V and replace the donor span, producing `k2, v2`. | The second branch has different task-span keys and values. |
| 672–682 | Compute `out_patched, _ = eager(..., k2, v2, ...)`. | **The donor-branch attention weights are discarded.** |
| 683–684 | Replace only selected query-head outputs with `out_patched`. | The model uses the donor branch for those heads; other heads retain the recipient branch. |
| 687–688 | Return `self.o_proj(attn_output), w`. | **Returned attention remains recipient-branch attention, even for donor-routed heads.** |

The relevant logic, abridged without changing its meaning:

```python
out_clean, w = eager(self, q, k, v, attention_mask, ...)
# k2/v2 contain the donor task-span replacement.
out_patched, _ = eager(self, q, k2, v2, attention_mask, ...)
out = out_clean.clone()
out[:, :, gh, :] = out_patched[:, :, gh, :]
return self.o_proj(out.reshape(...)), w
```

Returning recipient weights does not undo the donor output routing. It makes the reported attention inconsistent with that routing.

### 2.2 Value weighting introduces a second mismatch

In `src/attention.py::get_attention_matrices`:

- Lines **131–136** request `output_attentions=True, use_cache=True` through the supplied patch interface.
- Lines **141–146** stack the returned attention matrices.
- Lines **148–160** obtain values from `output.past_key_values`.
- Lines **164–167** repeat the KV-head values across their corresponding query heads.
- Line **169** multiplies attention probabilities by those cached value norms.

Because the cache was updated before the donor clones were constructed, it contains recipient-side values, not the effective task-span values `v2` used by the selected heads.

For a selected head, the intended versus reported quantities are therefore:

| Quantity | Effective transferred computation | Legacy gaze |
|---|---|---|
| Attention | `A(q, k2)` | `A(q, k)` |
| Values in transferred span | `v2` | Cached `v` |
| Value-weighted attention | `A(q, k2) * norm(v2)` | `A(q, k) * norm(v)` |

**Correcting only the returned attention weights is insufficient.** Value-weighted gaze must also use the matching effective values.

## 3. Why ordinary gaze can be disabled while selection gaze still runs

There are two separate consumers.

### 3.1 Ordinary evaluation path

In `scripts/diff/diff_trigger_eval.py`:

1. The KV branch begins at **229**.
2. Lines **234–250** construct the KV donor, replace the recipient inventory, and pass `kv=kv_arg` into `run_q_transfer_intervention`.
3. Lines **258–270** save the KV results, including optional `cl_selected` and `cl_select_scores`.
4. Line **271** continues the loop, bypassing the ordinary gaze evaluation at **303–306**.

Inside `run_q_transfer_intervention`, `src/diff/functional.py:1269` also sets `compute_gaze = False` when KV is enabled. Under this normal KV path, `task_positions` becomes `None` at **1279–1290**, and the ordinary attention fields become `None` at **1465–1488**.

Thus, the ordinary `gaze.shifted` metric is not produced by this KV runner branch.

### 3.2 Internal gaze-selection path

That does **not** disable the closed-loop selector:

- `src/diff/functional.py:1271–1277` maps donor-task descriptions onto aligned recipient token positions.
- **1362–1368** enter the `gaze_select` / `gaze_select_restrict` branch and call `get_attention_matrices(..., value_weighted=True)` under KV and the source-query patch.
- **1370–1373** aggregate per-task scores and store `cl_select_scores`.
- **1382** chooses the maximum-scoring task.
- **1393–1395** construct the selected-task probe prompt and store `cl_selected`.
- For `gaze_select_restrict`, **1404–1418** use that choice to construct the attention whitelist for a fresh recipient run.

Therefore, **`compute_gaze=False` does not mean that no gaze calculation occurs**. The internal task selector computes its own gaze independently of the ordinary evaluation flag.

Similarly, `--no-oracle` disables the external judge, and `--no-readout` disables the optional GoalScope readout outputs; neither disables this selector. The script is not a gaze-only runner merely because those two flags are supplied.

## 4. Triggering and unaffected configurations

### 4.1 Configuration matrix

| Configuration | Ordinary gaze metric | Internal gaze selection | Exposure to this defect |
|---|---|---|---|
| KV off, ordinary Q transfer | Computed | None | Not affected by the donor/native branch mismatch. |
| KV off, `gaze_select` or `gaze_select_restrict` | Available separately | Computed from recipient attention | Not affected by this KV-specific mismatch. Other configuration issues are separate. |
| KV on, ordinary re-clamp | Skipped | None | The wrapper's returned weights are wrong for selected heads, but this runner path does not use them for gaze selection. Generation is not invalidated solely by this diagnostic bug. |
| KV on, `gaze_select` | Skipped | Computed and used to name the probe's task | **Affected.** |
| KV on, `gaze_select_restrict` | Skipped | Computed and used to choose a whitelist | **Affected.** |
| KV on, a manual `get_attention_matrices` call through `headwise_kv_patch` | Depends on caller | Depends on caller | **Affected for donor-routed query heads.** Unweighted attention is also wrong; value weighting adds the second mismatch. |

For partial or different patch/readout headsets, the affected rows are the query heads actually routed to donor K/V. Non-routed heads should retain their native-branch diagnostics. The issue is not restricted to the 250-head configuration, but that is the concrete job configuration reviewed here.

### 4.2 Concrete checked job configurations

Both files below contain 16 entries: two task orders × KV off/on × four input datatypes. Eight entries per file enable KV transfer:

- `job_list/diff_eval/diff_trigger_gaze_select_restrict.json`
- `job_list/diff_eval/diff_trigger_gaze_select_restrict_anc_eq.json`

In each file, KV-entry names occur at lines **19, 23, 27, 31, 51, 55, 59, 63**; each command is on the immediately following line. The first group uses `order_diff`, the second `order_matched`. These are configuration declarations, not evidence that every job completed.

Shared relevant settings:

```text
--closed-loop
--probe-prompt gaze_select_restrict
--kv-transfer
--pivot eq
--patch-heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json
--readout-heads head_sets/Qwen2.5-Coder-32B-Instruct/goal_topk/goal__lam20_top250.json
--alphas 2
--naming letters
--lang-transfers py-py
--n-samples 32
```

The `_anc_eq` file additionally supplies `--gsr-anchors eq`; the other file uses the default `indent var eq` anchors. Changing the later whitelist anchors does not repair the earlier task-selection measurement.

Representative affected job names:

```text
clgazerestrict_kv_od_goal250_a2_pypy_list_str
clgazerestrict_kv_om_goal250_a2_pypy_list_int
clgsreq_kv_od_goal250_a2_pypy_list_str
clgsreq_kv_om_goal250_a2_pypy_list_int
```

The head file contains 250 unique `(layer, query_head)` pairs; its SHA-256 is:

```text
6b7c52d21a3cfc75c718a6c618e8f02f15a3372d00434497fbf427923978a96b
```

### 4.3 Effective boost settings are phase-specific

Despite `--alphas 2` and the `a2` job labels:

- The batch **selector** uses `kv_ctx()` and `proj_patch_hooks(...)`, without `_boost_ctx()` (`functional.py:1364–1366`). Its effective OV multiplier is 1.
- The batch **restricted-generation** context contains KV and the whitelist, without a boost (`1415–1419`).
- The separate patched GoalScope interface does include `_boost_ctx()` (`1314–1322`, used at `1459`). Consequently, `ro_patched` is not a readout of the exact unboosted selector or restricted-generation state.

This phase mismatch is separate from the attention-branch bug. Enabling or disabling boost alone cannot fix the branch/value mismatch.

## 5. Notebook exposure

For `notebooks/diff/04.4_q_transfer_multipred.ipynb`, references use stable cell IDs and **1-based lines within each cell's source**, rather than JSON file lines.

| Cell ID | Source lines | Relevant behavior |
|---|---|---|
| `2027d37f` | 2; 41–48; 64; 88–90; 102–108 | Saved code resolves to 250 heads, `list[int]`, `USE_KV=False`, `BOOST_ALPHA=2.0`, pad alignment, and matching patch/gaze heads. |
| `312435a3` | 36–50 | `patched_ctx` activates `headwise_kv_patch` when `USE_KV=True`. |
| `68bb2f01` | 1–10 | `pivot_gaze` calls `get_attention_matrices(value_weighted=True)` and averages over description tokens. |
| `68bb2f01` | 17–19 | KV and query-patched gaze use `ro_iface(..., boost=False)`. |
| `8beff8f1` | 115–123 | The `gaze_select` probe obtains its task from `GAZE["patched"]`. |
| `c6284305` | 14–20 | `SPAN_SELECTED` is the argmax of the same gaze and determines `SPAN_KEEP`. |
| `c6284305` | 138–149 | The separate span-generation context optionally uses KV, whitelist, and its own OV boost. |

**The saved `USE_KV=False` configuration does not exercise this KV-specific defect.** Turning it on routes these notebook gaze calls through the defective wrapper. A notebook may contain stale outputs from different execution states; saved code alone is not proof of the configuration that produced every displayed output.

Notebook span generation currently specifies `OV_BOOST=2.0`, whereas its gaze remains unboosted. Even after correcting donor-branch measurement, unboosted gaze should not be described as an exact trace of boosted generation: earlier-layer boosting can change later-layer queries, keys, and values.

## 6. How donor labels can be attached to recipient-branch scores

The old script's KV setup is:

```text
p_src: A′ C B′  -> C
p_kv:  A′ B D   -> B
p_dst: A′ X Y   -> X
```

The alignment transfers a contiguous donor task-list window into recipient positions. `sel_items` then labels subsets of those positions with donor tasks such as `D`.

However, the returned `w` was calculated before the donor replacement. It still uses recipient-side keys at those positions. Reporting that score under label `D` therefore does not show that the effective computation attended to donor `D`.

The mapping is tokenwise, not a semantic one-to-one mapping between entire tasks: descriptions can have different lengths. It is therefore too strong to say every donor-`D` position necessarily corresponds exactly to recipient `Y`. The failure remains: the **score's keys/values and the score's donor-task labels belong to different branches**.

A toy illustration, not an empirical result: two one-token donor tasks have correct attention scores `(0.1, 0.9)`, but the corresponding recipient positions have `(0.8, 0.2)`, with all value norms equal to one. The legacy selector chooses the first task while the donor-branch selector chooses the second. Real rankings may instead agree; the code mismatch alone does not quantify their disagreement rate.

## 7. Task-length normalization is separate from this bug

Both old selector and corrected smoke use the intended mean-over-description aggregation:

\[
s_p = \frac{1}{|H|\,|J_p|}\sum_{(\ell,h)\in H}\sum_{j\in J_p}
A^{\mathrm{effective}}_{\ell,h,t,j}\,
\left\|V^{\mathrm{effective}}_{\ell,g(h),j}\right\|_2.
\]

- `functional.py:462–465` averages over the chosen heads.
- `functional.py:1370–1372` averages over each candidate description's tokens.
- Ordinary gaze evaluation uses the analogous span mean at `src/diff/eval.py:242–263`.
- The smoke computes the same aggregation in its `src/run_smoke.py:81–83`.

There is no explicit keyword, stop-word, saliency, or top-k filter. Attention and value norms weight contributions, but every included description token counts in the denominator. Softmax is over all available causal keys; task scores are not renormalized to sum to one.

Mean versus sum or keyword pooling is a **measurement-design choice**. None can repair using the wrong attention/value branch. If alternative aggregation is later evaluated, retain the original scoring results and identify the alternative analysis explicitly.

## 8. Correction used in the ALT1 smoke

The completed smoke uses an experiment-owned adaptation of the previously checked KV runtime. These paths are **relative to that experiment's code bundle**, not repository paths at the reviewed commit:

| Experiment file and lines | Correction/check |
|---|---|
| `src/interventions.py:151–175` | Retains `transferred_weights`, routes the same selected outputs, and sends the observer the matching `v2`. Returns `None` rather than presenting native weights as effective attention. |
| `src/observe.py:12–37` | Captures selected heads at the final query, associates them with their effective KV-head values, and checks output reconstruction and routing. |
| `src/run_smoke.py:81–83` | Computes the head/token mean of attention × value norm. |
| `src/run_smoke.py:145–158` | Uses that observer for KV-only and KV-plus-source-query measurements on the same padded recipient. |
| `src/run_smoke.py:160–170` | Compares capture on/off and self-transfer logits on one configuration per datatype. |

For `Qwen2.5-Coder-32B-Instruct`, grouped-query attention has 40 query heads and 8 KV heads, so query head \(h\) uses KV head \(\lfloor h/5\rfloor\). Only selected **query heads** should use the donor branch; nonselected siblings in the same group retain the recipient branch.

The observer receives:

```python
observer(self, transferred_weights, v2, transferred, clean, routed)
```

It reconstructs the final selected-head output from the recorded attention and effective value vectors. This directly checks that the diagnostic describes the computation whose output is routed into the model.

### Recorded validation

The smoke used the actual nonquantized model revision `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, BF16/eager attention, all 250 heads, eight `list[str]` and eight `list[int]` configurations, and no OV boost. It ran 86 prefill forwards and generated zero tokens.

| Check | Recorded result |
|---|---|
| Selected-head donor-output routing | Zero discrepancy |
| Unselected-head native-output preservation | Zero discrepancy relative to the native branch of the same intervened forward |
| Source-query replacement and recipient-position key rotation | Zero discrepancy |
| Attention row normalization | Maximum error 0.002892; tolerance 0.005 |
| FP32 `A @ V` reconstruction versus native BF16 head output | Maximum absolute error 0.02990; all elementwise checks pass with tolerance `0.03125 + 0.015625 * abs(output)` |
| Capture on/off and self-transfer logit parity | Zero discrepancy on string seed 1000 and integer seed 2000; not tested on every configuration |
| Independent raw-score reconstruction | All 256 scores checked; maximum discrepancy 4.068e-8 |

The corrected smoke selected `e/d/b` in **8/7/1** cases, versus `b` in all 16 KV-only cases. These are results for the new ALT1 panel, **not** a paired estimate of how much the bug changed historical selector accuracy. No full restricted generation was tested.

Evidence:

- Report: [exp:exp_01m4ehsf78efcs34hnm8ntwdw5].
- Runnable code: [file:artifact://researcher-david-bau/experiments/exp_01m4ehsf78efcs34hnm8ntwdw5/deliverables/alt1_gaze_code.zip].
- Scores and validation: [file:artifact://researcher-david-bau/experiments/exp_01m4ehsf78efcs34hnm8ntwdw5/deliverables/analysis_evidence.zip].
- Raw final-query rows and value norms: [file:artifact://researcher-david-bau/experiments/exp_01m4ehsf78efcs34hnm8ntwdw5/smoke/raw_rows.zip].

## 9. Recommended repository fix and acceptance checks

### Preferred approach: an explicit effective-attention observer

Expose a bounded diagnostic callback inside the headwise wrapper, where both branches and their values are available. Capture only the requested query positions/heads, task scores, or per-token weighted rows as needed. Keep output routing and generation-cache semantics unchanged. Handle no-KV forwards through an equivalent native observer.

This avoids pretending that the ordinary shared KV cache contains a single effective value tensor for all query heads. Selected and unselected query heads in the same GQA group can read different branches.

### Alternative: extend the attention return contract

If downstream tooling requires returned attention matrices:

1. Return native weights for non-routed query heads and donor weights for routed heads.
2. Supply corresponding effective value norms or an already value-weighted per-query-head diagnostic through an explicit interface.
3. Update `get_attention_matrices` to consume that interface rather than blindly weighting the ordinary cache.

**Do not merely change `return ..., w` to donor weights for every head. Do not overwrite the generation cache to make the visualization convenient.** Both would break the intended per-query-head routing or its diagnostic correspondence.

Suggested focused acceptance checks, separate from model-performance claims:

- Use contrasting donor/native keys and values so a wrong-branch observer is detectably wrong; self-transfer alone can hide this defect.
- Check selected and unselected query heads, including siblings sharing one KV head.
- Check raw attention and value-weighted attention separately.
- Reconstruct selected head output with effective `A @ V` and numerical tolerances appropriate to the runtime precision.
- Verify diagnostic capture leaves logits unchanged and hooks are removed afterward.
- Verify padded and unpadded donor-description mappings, excluding standalone padding from scores.
- Use the same aligned recipient input for KV-only and KV-plus-query comparisons.
- If recording decode gaze, capture during the actual cached forwards; do not substitute an unboosted full-prefix replay for a boosted trajectory.
- Version the corrected diagnostic/results and preserve old records. If historical records contain only native-branch weights and cache values, the missing donor probabilities generally cannot be recovered by simply reweighting them; the relevant forwards must be rerun or reconstructed from sufficiently complete saved projections.

## 10. Interpretation boundary

Confirmed: the checked repository returns mismatched diagnostics for donor-routed heads, and the corrected ALT1 observer measures the effective donor computation under its validated settings.

Not established: the fraction of historical selections that change after correction, corrected historical restricted-generation accuracy, or a unique set/count/order mechanism. The local correction does not repair historical results or the repository automatically.

**Bottom line:** fix attention-branch and value-source consistency before interpreting KV-on task gaze or using it as a selector. Keep task-length normalization and other aggregation choices as a separate, explicitly versioned analysis decision.

## Appendix: source fingerprints

```text
ad75666f4a68571549281a5eb712a74a64280a8bdbccd6703e35d33c21d5b238  src/diff/functional.py
1bbccb358a04a97470ef555a2132696be26b67af171cbf47f917e78934dcdffd  src/attention.py
e77ed33c7d15d672d4d391ff6eee07e0140423c0f5694bc3a66e7705eeca7f07  src/diff/eval.py
c4953346618cff2ceff9597b7655de7ccd14c1eed8d257acff5068ac51f0915a  scripts/diff/diff_trigger_eval.py
50de02f099e47680e0f7844047a216d699e2d9ff1de1e66982482d1e26e0801e  notebooks/diff/04.4_q_transfer_multipred.ipynb
```
