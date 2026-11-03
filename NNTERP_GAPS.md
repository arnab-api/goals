# nnterp / nnsight gaps found during the port

A running list of what was awkward, undocumented or impossible in nnterp or nnsight while porting this repo from
baukit hooks to nnterp. Every port PR appends its own section. Each entry says what happened, gives a one-line repro or
a pointer, and names the workaround used here, if any.

## Summary for upstream

Every proposal and bug of the sections below, each with a pointer to its entry (section and item: "NRGS 1" is item
1 of the NRGS section). Status: **fixed** (merged upstream), **filed** (an open issue), otherwise unfiled. Behaviour
notes that ask nothing of upstream (GoalScope 5, 7, 8; head-mask 4, 5; demos 1) are left out. "Head-mask" is the
head-mask training section.

**nnterp**

1. **fixed** (ndif-team/nnterp#124): the family lookup read the config without `cache_dir`, so a checkpoint in a
   non-default cache failed offline. Model layer 1.
2. A family-name string: `model.family` is a module, and `model.config.model_type` differs on multimodal wrappers.
   Model layer 5.
3. The norm gains as values or accessors: the pre-attention norm's effective gain (Gemma's `1 + weight`; nothing on
   post-norm blocks) and the post-attention norm's, which sets the direction a head writes into the stream on
   Gemma-2, OLMo-3 and GLM-4. GoalScope 4, demos 2.
4. A per-head K/V patch under GQA, `attn.patch_kv_for_heads(heads, positions, keys, values, rows)`, computed with the
   family's own attention arithmetic (Gemma-2's logit softcap); or per-head `attention_head_keys` /
   `attention_head_values` after `repeat_kv`. NRGS 1, demos 3.
5. The rotation's input as values, `attention_queries_prerope` / `attention_keys_prerope` (after any q / k norm), and
   `model.rotate(x, positions, layer=)` with the model's own rotary embedding. Without them a transplant to another
   position rotates projection outputs by hand, which misses Qwen3's and OLMo-3's key norms, fails on OLMo-3's
   per-layer rotary call and has nothing to read on Phi-3's fused `qkv_proj`. NRGS 2, demos 3.
6. Per-head projection weights (each kv head's W_V, each head's W_O slice) that also cover fused projections
   (Phi-3's `qkv_proj`). Demos 4.
7. A forward-order sort key for values and native submodules, `nnterp.forward_rank(model, path)`, so an intervention
   library orders its edits without encoding a family's forward. NRGS 3, last hooks 4.
8. Build the attention modules' `.source` at load under eager, so reading a child module's value and then an
   interior value in one trace does not fail on a fresh model. NRGS 5.

**nnsight**

1. **filed** (ndif-team/nnsight#822): the meta build (`dispatch=False`) loads the tokenizer without `cache_dir`.
   Model layer 2.
2. **unfiled** bug: `del x` in a trace body neither releases nor unbinds `x` (`Scope` has no `__delitem__`; the
   sibling store keeps the binding), so a later read still finds it and the tensor lives until the trace ends.
   Head-mask 2.
3. A step loop that may end early has no quiet form: `tracer.iter[:N]` cut short by EOS or a stopping criterion warns
   "was never reached", the same warning as a read-order bug. Proposal: "steps up to N, stop quietly". GoalScope 2,
   last hooks 1.
4. A persistent edit meant for every forward (`edit(inplace=True)` + an open `tracer.iter[:]`) warns on every run.
   Proposal: `edit(inplace=True, every_call=True)`, or no warning for an open loop inside an edit. Last hooks 2.
5. A replaced `forward` makes `.output` fail with an `OutOfOrderError` that does not name the cause (only `.source`
   warns). GoalScope 6, NRGS 6.

**Docs** (nnsight's, nnterp's, and the nnterp skill)

1. Subclassing `StandardizedTransformer` (plain attributes and class-level properties are safe). Model layer 4.
2. The nnterp skill's version line names a `0.8-refactor` / nnsight `dev` pair; nnterp 2.0.0rc1 + nnsight 0.8.2 on
   transformers 5.22.1 runs everything here. Model layer 6.
3. Composing a caller's intervention with a function's reads: the intervention in the input invoke, the reads in an
   empty second invoke. GoalScope 1.
4. `generate(do_sample=False)` keeps the checkpoint's `repetition_penalty` and extra EOS ids. GoalScope 3.
5. In-place edits of `attention_keys` / `attention_values` under `generate` persist in the KV cache. NRGS 1.
6. **unfiled, stale**: the cheat-sheet and `attention-interior.md` caveat that two invokes cannot both touch
   `attention_scores` / `attention_probabilities` no longer reproduces on nnsight 0.8.2. NRGS 4.
7. An envoy call outside a trace is a plain module call; nnterp's pages reach for `_module` instead. NRGS 7.
8. A stopping criterion ends a run early like EOS (nnterp's `generation.md` names only EOS). Last hooks 1; model
   layer 3 (the loop of traces threading `past_key_values` it replaced).
9. `proj.output = proj.output + f(proj.input)` reads `.output` first and fails; bind the input first. Last hooks 3.
10. Reading a value from inside an edit (append a saved value to a list made outside it). Last hooks 5.
11. An edit whose output is backpropagated must be an assignment, not an in-place write (nnterp's
    `attention-interior.md` shows only the in-place form). Head-mask 1.
12. The trace body runs under the caller's grad mode (`torch.enable_grad()` around `mt.trace` overrides a global
    `set_grad_enabled(False)`). Head-mask 3.

## Model layer (`port/model-layer`)

nnterp 2.0.0rc1, nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct.

1. **nnterp's family lookup ignored `cache_dir`** (FIXED upstream: ndif-team/nnterp#124, on `v2.0.0` at `feeb24c`;
   this repo pins that commit until a release carries it). `StandardizedTransformer` read the config to pick the family
   with only `revision` and `trust_remote_code`, so a checkpoint that lives only in a non-default cache (`env.yml`
   `HF_CACHE`) failed offline before any weights loaded. `_read_config` now takes the hub kwargs (`cache_dir`, `token`,
   `local_files_only`, ...) as its own parameters.
2. **nnsight's meta build ignores `cache_dir` for the tokenizer** (filed: ndif-team/nnsight#822). With `family=` passed (item 1), a load without
   `dispatch=True` still fails the same way: `TransformersModel._load_meta` builds its meta `pipeline(...)` without
   `cache_dir`, so the tokenizer is looked up in the default cache (`nnsight/modeling/transformers.py`, `_load_meta`).
   `ModelandTokenizer` always dispatches, so it does not hit this; a lazy (`dispatch=False`) load would.
3. **A loop of single traces that threads `past_key_values` is undocumented.** `scripts/evaluation/ablation_common.py`
   decodes one token per `mt.trace({"input_ids", "attention_mask", "past_key_values", "use_cache": True})` and saves
   `mt.output.past_key_values` (the live `DynamicCache`) for the next trace. It works on 0.8.2 (decode logits match a
   full forward to 1.4e-5), but neither nnsight's nor nnterp's docs cover passing a cache into a trace. The documented
   route, `model.generate` + `tracer.iter`, has no equivalent of the scripts' stop-at-newline rule short of a
   `StoppingCriteria` or truncating afterwards.
4. **Subclassing `StandardizedTransformer` is undocumented.** Every docs page constructs it directly; this repo needs a
   subclass to carry its legacy interface. A plain `super().__init__(...)` works (`Envoy.__setattr__` diverts only
   `nn.Module` values to the envoy tree, so plain attributes and class-level properties are safe), but that had to be
   read off `nnsight/intervention/envoy.py`.
5. **No family-name accessor.** `model.family` is a module; the string is `model.family.__name__.rsplit(".", 1)[-1]`
   (or `model.config.model_type`, which differs for multimodal wrappers). `ModelandTokenizer.family_name` wraps it.
6. **The nnterp skill's version line is stale.** `skills/plugins/nnterp/skills/nnterp/SKILL.md` says it was verified
   on nnterp `0.8-refactor` with nnsight `dev` at `d565f5f8` and transformers 5.17/5.19, and that nnterp needs that
   nnsight commit. The PyPI pair nnterp 2.0.0rc1 + nnsight 0.8.2 on transformers 5.22.1 runs everything this PR uses,
   including the dotted `o_proj` alias behind `attention_premix`.

## GoalScope read / steer (`port/goalscope`)

nnterp 2.0.0rc1, nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct (float32, eager) for
`test/test_goalscope_port.py`, and the 7B parity harness on one GPU.

1. **Composing a caller's intervention with a function's own reads needs two invokes, and nothing says so.** The
   ported readers take `interventions(mt)`, called inside their trace. Called in the same invoke as the reads, it may
   only touch blocks at or before the first read: an OV boost at layer 12 followed by a read at layer 4 raises
   `OutOfOrderError`. What works is the intervention in the invoke that carries the inputs and the reads in a second,
   empty invoke (`tracer.invoke()`), which sees the same rows. Each invoke keeps forward order on its own, and at one
   location the invoke written first is served first, so an edit of `attention_premix` at a block lands before the
   read of it there (`test_interventions_composition`). nnsight documents empty invokes for batch-wide reads and the
   same-location order only in passing (`references/batching.md`, the barrier section); nnterp's docs say nothing
   about composing interventions. The pattern also inherits the two-invoke bug on `attention_probabilities` /
   `attention_scores` (CLAUDE.md cheat-sheet): an intervention that edits the scores cannot be composed with
   `get_attention_matrices` this way.
2. **A per-step loop that may meet an early EOS has no quiet form.** `tracer.iter[:N]` with `N = max_new_tokens`
   is cut short, with a `UserWarning` ("was never reached: the loop asked for a step the run did not make"), whenever
   every row stops early, and drops what follows the loop in that invoke. `min_new_tokens=N` is not an option when
   the decoding must not change. Workaround (`generate_with_patch`, `headpatched_generate`): read `tracer.result`
   in its own empty invoke, and filter that one warning message around the call. The filter cannot tell this
   expected cut from a read-order bug, which produces the same warning (nnterp's `generation.md`); an option such as
   "steps up to N, stop quietly" would make the intent explicit.
3. **`mt.generate` brings the checkpoint's `generation_config`, and a fresh `GenerationConfig` does not reset it.**
   The old `headpatched_generate` was a hand-written argmax loop that stopped on `tokenizer.eos_token_id`. On
   `mt.generate(..., do_sample=False)` the Qwen2.5 checkpoints still apply `repetition_penalty` 1.1 and stop on both
   EOS ids (151645, 151643). transformers 5.22's `_prepare_generation_config` fills every unset field of a passed
   `GenerationConfig` from the model's, so the neutral settings must be passed explicitly:
   `do_sample=False, repetition_penalty=1.0, eos_token_id=mt.tokenizer.eos_token_id` (temperature / top-p / top-k
   are ignored under greedy). nnsight's and nnterp's generation pages say to pass `do_sample=False`; neither says the
   repetition penalty survives it. Not an nnterp bug, but the first trap a port of a manual decode loop meets.
4. **No accessor for the pre-attention norm's gain.** `OVReader` folds the gain of the norm the attention reads into
   each head's OV map. nnterp serves the normed stream (`attention_norm_output`) but not the gain; `input_layernorm`
   is a best-effort alias that does not exist on post-norm blocks (OLMo-2/3), and on Gemma the effective gain is
   `1 + weight`. Workaround: `getattr(mt.layers[l], "input_layernorm", None)` outside the trace, and a `ValueError`
   naming the family when it is missing (OLMo-3 is refused; it was an `AttributeError` before).
5. **`attention_output` is not the attention module's output on every family.** On Qwen2 / Llama it is (the
   `--source attn` readout baseline of `eval_readout_sources.py` reads bit-identical values); on OLMo-3 and GLM-4 it
   is the output of the norm after the attention, the tensor the block adds to the stream. That is the better
   definition of the paper's "attention output" baseline, and it changes what `--source attn` reads on those two
   families. The reported baseline (section 4.2) is on Qwen2.5-Coder-32B, so no reported number moves.
6. **A hook context still works around an nnsight trace, with one exception.** nnsight 0.8 installs its controller as
   the module's `forward`, so PyTorch pre-hooks run before `.input` is served (a read sees the pre-hook's edit) and
   forward hooks run after `.output` is served. NRGS's interventions stay PyTorch hooks in this PR and enter around the
   new traces through a deprecated `legacy_hooks` argument. The exception is `headwise_kv_patch`, which replaces the
   attention module's `forward`: for the duration nnsight's controller on that module is gone, so the module's own
   `.input` / `.output` / `.source` values (`attention_probabilities` among them) are not served, while its children's
   (`o_proj`, so `attention_premix`) still are. A read of `self_attn.output` then fails with `OutOfOrderError` ("the
   model already ran past it") and no hint at the cause; only a `.source` read, which re-instruments the module,
   adds nnsight's "`forward` was replaced" warning. None of the ported readers read the attention module's own values
   under that patch (the parity harness's `kv_transfer` read-outs read `attention_premix`).
7. **Position ids under left padding (a behaviour difference, not a gap).** An nnsight trace derives `position_ids`
   from the attention mask; the old hook paths called the raw HF module (`mt._model(**inputs)`), which numbers
   positions from the first pad. On a left-padded row the two agree only up to rounding of the RoPE angles
   (float32, 0.5B: 3.5e-5 on attention probabilities, 2.4e-7 on head steering vectors; 0.0 once the raw forward is
   given the mask-derived positions). Unpadded rows are bit-identical. The parity harness's two padded cases deviate
   from the nnsight 0.7 reference for the same reason (since PR 1, which moved the nnsight traces to 0.8).
8. **The old step counter in `generate_with_patch` (a behaviour difference).** The baukit version counted decode steps
   at the lexicographically smallest patched module name, so a steer band crossing a digit boundary (layers 8 and 12:
   `"model.layers.12" < "model.layers.8"`) patched the lower layers for one extra decode step under an integer
   `n_steer_tokens`. `tracer.iter` counts steps exactly, so such a band now gets what `n_steer_tokens` says
   (`test_old_sentinel_quirk` reproduces the old schedule and shows a text change). Reported runs use layers 35-45
   with `n_steer_tokens="all"`, which neither version miscounts.

## Head-mask training (`port/head-localization`)

nnterp 2.0.0rc2.dev7, nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct (float32,
eager) for `test/test_head_localization_port.py`, and Qwen2.5-Coder-7B-Instruct (bf16, eager) on one RTX A6000 for
the parity harness and the memory measurement. This is the one path that backpropagates through the model.

1. **An edit that must carry a gradient has to be an assignment; nnterp's page shows only the in-place form.**
   Writing the interpolation into the served value (`attn.attention_premix[:, -1] = alpha * m * src + (1 - m) * cur`)
   inside a trace whose output is backpropagated fails at `backward()` with torch's "modified by an inplace
   operation"; assigning a new tensor (`attn.attention_premix = torch.cat([...])`) works, and its loss and mask
   gradient are bit-equal to the old pre-hook version. nnsight's `references/gradients.md` says so ("Replace, don't
   write through, when the new value is computed from the old"). nnterp's `attention-interior.md` says "In-place
   edits reach the model; assignment flattens the heads back", with an in-place example only and nothing about
   gradients. Editing a few positions by assignment takes a `torch.cat` splice of the untouched slices and the edited
   positions (`_interpolate_heads`), which also keeps the untouched positions off the mask's gradient path.
2. **`del` in a trace body neither releases nor unbinds a name** (nnsight bug). `Scope.__setitem__`
   (`nnsight/tracing/util.py`) writes every binding to the block's dict and to the shared store of sibling blocks,
   and `Scope` defines no `__delitem__`, so `del x` removes only the first copy. A later read still finds `x`
   through `__missing__`: `with mt.trace(p): x = mt.layers[0].layer_output; del x; s = nnsight.save(tuple(x.shape))`
   returns the shape instead of raising `NameError`. The memory cost: the first port bound each layer's
   `attention_premix` to `x` in the trace body's layer loop, which kept the last one, one
   `[batch, seq, num_heads * head_dim]` tensor, alive until the trace ended, at the forward's memory peak (+4.8 MiB at
   7B batch 8, about 65 MB at 32B batch 64). `del x` did not help; `x = None` did, and so does passing the served
   value straight into a function (`attn.attention_premix = _interpolate_heads(attn.attention_premix, ...)`), which is
   what the port does. A forward pre-hook never had this, because its locals die when it returns.
3. **What grad mode the trace body runs under is undocumented.** It is the caller's: nnsight runs the trace body as
   greenlets on the calling thread, so `torch.enable_grad()` around `mt.trace` turns gradients on under the eval
   scripts' process-wide `torch.set_grad_enabled(False)`, and `torch.no_grad()` / `inference_mode` around it give a
   plain forward. `train_head_mask` now wraps its forward and backward in `torch.enable_grad()` (the old loop failed
   under the global switch with "element 0 of tensors does not require grad"). nnsight's gradients page covers only
   `generate`'s own `no_grad`.
4. **Training through a trace costs nothing measurable (feedback, not a gap).** 7B, one forward + backward over all
   28 layers at mask 0.5, the old pre-hooks against the trace (median of 5, interleaved): batch 8 peak
   3.5658 vs 3.5661 GiB, forward 0.121 vs 0.123 s, backward 0.134 vs 0.133 s; batch 16 peak 7.1732 vs 7.1739 GiB. A
   plain trace reading `mt.logits` has the same peak as the raw HF forward (+0.02 MiB), and nothing is held after the
   step under either version.
5. **Position ids under left padding, now on both halves of training (a behaviour difference, not a gap).** The
   source o_proj inputs and target log-probs already came from nnsight traces (mask-derived `position_ids`); the
   destination forward was a raw `mt._model(...)` call that numbered a padded row from its first pad. Both are traces
   now, so a padded destination row moves: on 7B batch 8 (15 pads) the mask gradient differs from the old one by
   2.4 % relative (1.4 % at batch 16); in the harness's `mask_grad_padded`, 0.19 %. Equal-length batches are
   bit-identical. Handing the old code nnsight's `position_ids` makes the two agree exactly on the 0.5B.

## NRGS interventions (`port/nrgs`)

nnterp 2.0.0rc2.dev7 (`feeb24c5`), nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct
(float32, eager; 14 query heads over 2 kv heads) for `test/test_nrgs_port.py`, and Coder-7B (bf16, one RTX A6000) for
the parity harness and the recompute variants below.

1. **No primitive for patching keys / values per query head under GQA.** The KV transfer needs SRC's keys and values
   on a span read by chosen query heads only, by the last query row only (the pivot on the prompt pass, then each new
   token), with the cache left clean. `attention_keys` / `attention_values` are served before `repeat_kv`
   (`num_kv_heads` wide), so an edit there reaches every query head of the group (the "GQA bleed": a kv-head-wide
   write at layer 6 of the 0.5B moves the unpatched head 3 next to the patched head 2 by 0.6, `test_gqa_bleed`), and
   every query row reads it. Under `generate` an in-place edit of them is also an edit of the `DynamicCache` (an
   in-place zero at step 0 is still there at step 1; assignment changes one forward only), which no nnterp page says.
   Editing the nested source after `repeat_kv` (`self_attn.source.attention_interface_1.source.<op>.output[:, qh,
   span]`) is per head but not per row, and is bound to transformers' op names. What the port does
   (`headwise_kv_patch`): per step and patched block, read `attention_queries` / `_keys` / `_values`, clone k and v,
   replace the span of the patched kv heads, recompute `softmax(q @ k.repeat_interleave(g)^T * attn._module.scaling)
   @ v.repeat_interleave(g)` (the "Recomputing by hand" recipe of `attention-interior.md`), mask the last row with the
   model's own mask read off `attention_scores[:, :, -1:] <= finfo.min`, and write `attention_head_outputs[:, -1,
   heads]`. Two traps on the way:
   - **Bit-exactness needs the forward's own GEMM shapes.** Recompute every query head and every row, keep the chosen
     heads' last row. Recomputing only the chosen heads, or only the last row, runs other GEMM shapes that round
     differently: on Coder-7B the patched heads' outputs then differ from eager attention by up to 0.53 at the prompt
     pass (last row only, V x4, 448 heads) and by 0.2-0.38 after a few decode steps (chosen heads only, 60 heads), in
     float32 on CPU by up to 5e-6. The full recompute is 0.0 on both, and costs what one more eager attention of the
     block costs.
   - **The mask is reachable only through the scores.** No per-block value serves the 4-D additive mask the interface
     receives, so a recompute takes it from the masked `attention_scores`.

   Proposal: a helper on `Attention`, usable in a trace or a `tracer.iter` step,

   ```python
   attn.patch_kv_for_heads(
       heads: Sequence[int],                    # query heads that read the patched keys / values
       positions: Sequence[int] | Tensor,       # key positions, absolute (into the cache on a decode step)
       keys: Tensor | None = None,              # [len(heads), len(positions), qk_head_dim], post-RoPE
       values: Tensor | None = None,            # [len(heads), len(positions), head_dim]
       rows: int | slice | Sequence[int] = -1,  # query rows that read them; default the last
   ) -> None
   ```

   after which those heads' `attention_head_outputs` at `rows` are what eager attention gives with the patched keys /
   values, and every other head, every other row and the cache are as computed (implemented as above). For the
   all-rows case, two forward-local values would make it a one-line edit: `attention_head_keys` /
   `attention_head_values`, `[batch, num_heads, seq, head_dim]`, the keys / values each query head reads (after
   `repeat_kv`), writable per head. And a gotcha in `attention-interior.md` / `generation.md`: in-place edits of
   `attention_keys` / `attention_values` under `generate` persist in the cache.
2. **Pre-RoPE vs post-RoPE for a transplant to another position.** nnterp serves queries and keys after RoPE,
   rotated for their own positions, and the projections before it (`self_attn.q_proj.output`, bias included). A
   transplant to another position has to enter before the rotation. The Q transfer writes the SRC pivot's `q_proj`
   output into the DST pivot's (`q_proj.output[:, pivot, h * head_dim:(h + 1) * head_dim]`), so the model rotates it
   at the DST pivot; `attention_queries` would carry the SRC pivot's rotation. The KV transfer cannot write pre-RoPE
   keys into the destination forward (that edits the cache and the whole group, item 1), so it rotates the SRC
   window's `k_proj` outputs at the DST positions by hand: `mt.model.rotary_emb(dummy, positions)` (a native path,
   an ordinary module call outside a trace) and the family module's `apply_rotary_pos_emb` (imported from
   `type(attn._module).__module__`). The shift is real: the SRC and DST windows start at the same position for py-py
   and cpp-cpp but not across languages (the C++ note is in the system prompt; e.g. positions 61 vs 72 for int
   py-cpp), which is half of the reported KV grid. Proposal: `model.rotate(x, positions)`, the model's own rotary
   embedding (partial, interleaved or half rotation, scaling) applied to `[..., seq, qk_head_dim]` at `positions`;
   and writable values at the rotation's input, `attention_queries_prerope` / `attention_keys_prerope` (`[batch,
   heads, seq, qk_head_dim]`, after any q/k norm), so a query transplant is `attn.attention_queries_prerope[:, heads,
   pivot] = src` with no `head_dim` slicing of a projection whose layout nnterp does not promise.
3. **Composing interventions needs a forward order; nnterp has it, and the first version of this port missed it.**
   An intervention made of edits in several blocks cannot run after another one: the patched GoalScope readout of
   the Q transfer combines a query patch and an OV boost over the same 250 heads (blocks 2-63), and the boost at
   block 30 after the query patch at block 60 raises `OutOfOrderError` in a trace, or inside `tracer.iter` binds the
   next step's occurrence without an error. `src.nrgs.functional.Intervention` sorts its edits by
   `mt.rank(name, layer)` when it is built (`mt.order(layer)` lists a block's values in the order the forward reaches
   them, measured by one probe scan per model, so Falcon's values-before-queries order and the hybrids come out
   right); the KV transfer reads queries, keys and values in that order too. The first version hand-wrote a stage
   table for the Llama / Qwen2 forward and proposed a `forward_rank` as a missing feature: `rank` has existed since
   2026-10-07, but it lived on the availability page and the API reference only, and neither nnterp's `CLAUDE.md`
   nor the nnterp skill routed to it. nnterp #125 and #126 add the routes. What still has no standard name is a
   projection's output (`self_attn.q_proj.output`, the Q patch's pre-RoPE write): it is ranked just before the
   block's first attention value, and goes away once `attention_queries_prerope` (nnterp #128) is in the pinned
   nnterp.
4. **The two-invoke limitation on `attention_scores` / `attention_probabilities` no longer reproduces.** nnterp's
   `CLAUDE.md` cheat-sheet and `attention-interior.md` say two invokes of one trace cannot both touch them
   (`TypeError`). On this stack every combination works, on a fresh model too: a scores edit in an input invoke with a
   read of the scores, the pattern or the head outputs in an empty invoke; two input invokes reading the same value.
   `test_scores_edit_composes_with_attention_matrices` passes `whitelist_head_attn` (a scores edit) to
   `get_attention_matrices`, and the pattern read in the second invoke carries it; the KV transfer reads
   `attention_scores` in the intervention invoke of `get_o_proj_inputs_batch`. The caveat in this repo's
   `get_attention_matrices` docstring is gone. Proposal: drop it from the nnterp docs, or name the nnsight version
   that fixed it.
5. **The fresh-model `.source` gotcha meets composed interventions.** `get_attention_matrices` under the query patch
   reads `q_proj.output` (intervention invoke) and the same module's `attention_values` / `attention_probabilities`
   (read invoke). On a fresh model the first such trace raises `OutOfOrderError` naming
   `attention_interface_1.fn.i0` (the documented gotcha; `q_proj.output` then `attention_scores` in one trace
   reproduces it). `proj_patch` and `headwise_kv_patch` touch `self_attn.source` when built, outside any trace.
   Reading the interface inputs and then `attention_scores` (the nested source) on a fresh model is fine. Proposal:
   build the attention modules' `.source` at load under eager, so no caller has to know.
6. **A swapped `forward` (PR 3, item 6) is gone.** The KV transfer swapped the attention modules' `forward` for its
   duration, which removed nnsight's controller from them; the recompute above needs no swap, and the `legacy_hooks`
   bridge PR 3 added for it is removed. What stays worth a fix in nnsight: a replaced `forward` makes `.output` fail
   with an `OutOfOrderError` that does not name the cause (only `.source` warns).
7. **An envoy call outside a trace is an ordinary module call, and only nnsight's docstring says so.** The port needs
   two module calls outside any trace: the rotary embedding (item 2) and the embedding lookup of the
   task-vs-progress embedding transplant (`mt.embed_tokens(src_ids)`). `Envoy.__call__` documents that this is a
   plain call; nnterp's pages reach for `_module` instead (`attn._module.scaling`), which suggests the private route.

## The last hooks (`port/last-hooks`)

nnterp 2.0.0rc2.dev7 (`feeb24c5`), nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct
(float32, eager) for `test/test_last_hooks_port.py`, and Coder-7B (bf16, one RTX A6000) for the parity harness.

1. **A stop rule other than EOS (model-layer item 3, resolved).** The ablation decode stops after a generated token
   whose text ends in `"\n"` (keeping it) and at EOS (dropping it). It now runs on `mt.generate` + `tracer.iter`
   with a transformers `StoppingCriteria` passed as `stopping_criteria=`; a stop it triggers cuts
   `tracer.iter[:max_new]` short exactly like an early EOS (the same "was never reached" warning, so GoalScope item
   2's own-invoke read and filter cover it), which nnsight's `docs/usage/iteration.md` states for `stop_strings`.
   The loop of single traces threading `past_key_values` is gone. nnterp's `generation.md` names only EOS among the
   things that end a run early; a stopping criterion belongs in that gotcha.
2. **A persistent edit on every forward warns on every run.** `mt.edit(inplace=True)` applies a value at its first
   occurrence, so an edit meant for every forward of a `generate` (the EM LoRA on each projection, every step) goes
   under an open `tracer.iter[:]`, as nnsight's `docs/usage/edit.md` shows. That loop ends each run by asking for a
   step that never comes, so every trace and every `generate` while the edit is installed emits
   `'...q_proj.input.i1' was never reached: the loop asked for a step the run did not make` (a single trace too).
   Workaround (`em_adapter`): filter that message while the adapter is on. Proposal: `model.edit(inplace=True,
   every_call=True)`, or no warning for an open loop inside an edit, where outrunning the run is the intent.
3. **A hook's `o + f(i)` does not port as one expression.** `proj.output = proj.output + s * ((proj.input @ A.T)
   @ B.T)` evaluates `.output` first and raises `OutOfOrderError` on `proj.input` ("the model already ran past it");
   `x = proj.input` has to come first. The forward-order rule is documented; this instance of it (Python's
   evaluation order inside one statement) is the first thing a forward-hook port writes. Proposal: a line next to
   the forward-order rule in nnsight's `access-and-modify.md`.
4. **Native submodules have no forward order either.** One edit block for a LoRA on q / k / v / gate / up / down of
   every block must visit them in call order; the adapter's module list (safetensors keys) is sorted by name, which
   puts `mlp.*` before `self_attn.*` and block 10 before block 2. Workaround: sort by (block, a hand-written
   projection order for the Llama / Qwen2 forward). NRGS item 3's `forward_rank` proposal would cover this if it
   accepted native module paths.
5. **An edit can observe another function's run.** The parity harness reads the step-0 logits of
   `head_additive_generate`'s own `generate` with an edit that appends `mt.logits[:, -1].save()` to a list made
   outside it; each run appends its value, readable after the run (bit-identical to the forward hook it replaces).
   Neither nnsight's `edit.md` nor nnterp's docs show reading a value from an edit.

## Demos, docs and the cross-family check (`port/demos-docs`)

nnterp 2.0.0rc2.dev7 (`feeb24c5`), nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; bf16, eager, one RTX A6000;
`scripts/cross_family_check.py` on Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, Olmo-3-7B-Instruct,
gemma-2-9b-it, Qwen3-8B, GLM-4-9B-0414 and Phi-4-mini-instruct (CLAUDE.md section 7 has the matrix).

1. **Every value the repo reads or edits is served on all seven families (feedback, not a gap).** `support()` reports
   the attention interior available on each under eager, and the edits of `layer_output`, `attention_premix`,
   `attention_scores` and `attention_head_outputs` work unchanged: GoalScope (both strategies), the attention
   patterns, the head vectors on a padded batch, the sink and blind-span ablations (a sunk head puts all its pivot-row
   mass on key 0, a blinded head none on the blocked keys), the OV boost and the temporal encodings run on all seven.
   Every failure is where the repo goes below nnterp's vocabulary: projection modules, a rotary embedding called by
   hand, attention written out by hand, norm parameters.
2. **Norm gains (GoalScope item 4, continued).** OVReader took `input_layernorm.weight` as the gain the attention's
   input is scaled by; Gemma's norm multiplies by `1 + weight`. Workaround: call the norm module on a constant vector
   outside the trace (`norm(torch.full((d,), 256.0))`: 1/rms is then an exact power of two that eps cannot move), which
   returns the gain the module applies, bit-equal to `weight` on all 28 Qwen2 layers and to `1 + weight` on all 42
   Gemma-2 layers. It assumes an RMS norm (a LayerNorm maps a constant to its bias). On post-norm and sandwich blocks
   the head's write passes a second norm after o_proj, whose gain sets its direction in the stream; neither gain is
   served.
3. **The KV recompute is not the model's attention where the family does more (NRGS items 1 and 2, continued).** The
   self-transplant check (DST's own window written back, V x1) leaves the logits bit-identical on Qwen2, Llama and
   GLM-4 (partial rotary included), and moves them on:
   - Gemma-2 by 0.25: the recompute has no attention logit softcap; with `attn_logit_softcapping` switched off in the
     model the change is 0.0, and the hand-rotated keys equal `attention_keys` exactly;
   - Qwen3 by 2.6: the keys pass `k_norm` between `k_proj` and RoPE, so the hand-rotated `k_proj` outputs differ from
     `attention_keys` by up to 19.7 (0.0 with the module's `k_norm` applied first).
   OLMo-3 does not get that far: its rotary embedding's forward takes `layer_type` (sliding or full), so
   `mt.model.rotary_emb(x, positions)` raises `TypeError`, and its keys pass a full-width key norm as well. A repo fix
   would mean a softcap branch, a `k_norm` lookup and a per-layer rotary call, each a family's arithmetic written
   again; `patch_kv_for_heads` with the family's own score function and `attention_keys_prerope` + `model.rotate`
   would make the transplant family-free.
4. **Fused projections (Phi-3).** Phi-3 has `qkv_proj` and no `q_proj` / `k_proj` / `v_proj`, so the Q transfer
   (writes `q_proj.output`), the KV transfer (reads `k_proj` / `v_proj` outputs) and OVReader (reads `v_proj.weight`)
   fail with `AttributeError` / a `ValueError` naming the module. nnterp serves Phi-3's queries, keys and values after
   the split and RoPE, but nothing before RoPE and no per-head weights. Proposal: item 5 of the summary, and per-head
   projection weights that also cover fused projections.

