# nnterp / nnsight gaps found during the port

A running list of what was awkward, undocumented or impossible in nnterp or nnsight while porting this repo from
baukit hooks to nnterp. Every port PR appends its own section. Each entry says what happened, gives a one-line repro or
a pointer, and names the workaround used here, if any.

## Summary for upstream

Every proposal and bug of the sections below, each with a pointer to its entry (section and item: "NRGS 1" is item
1 of the NRGS section). Status: **fixed** (merged upstream), **filed** (an open issue), otherwise unfiled. Behaviour
notes that ask nothing of upstream (GoalScope 5, 7, 8; head-mask 4, 5; demos 1; EM adapter 3) are left out. "Head-mask" is the
head-mask training section.

**nnterp**

1. **fixed** (ndif-team/nnterp#124): the family lookup read the config without `cache_dir`, so a checkpoint in a
   non-default cache failed offline. Model layer 1.
2. A family-name string: `model.family` is a module, and `model.config.model_type` differs on multimodal wrappers.
   Model layer 5.
3. **fixed** (ndif-team/nnterp#134) for the pre-attention norm: `mt.layers[l].attention_input_gain` is the gain
   between the stream and what the attention reads (`weight` on Qwen2 / Llama, `1 + weight` on Gemma, ones on a
   post-norm block), and OVReader folds it into each head's OV map. Not served yet (`attention_output_gain`, a
   follow-up to #134): the gain of the norm the attention's output passes before the stream (OLMo-2/3's post-norm,
   Gemma-2/3's and GLM-4's norm after the attention), which sets the direction a head writes into the stream there;
   OVReader refuses those blocks until it is. GoalScope 4, demos 2, head weights 4.
4. **fixed** (ndif-team/nnterp#127): nnterp serves `self_attn.interface`, the attention function the block called,
   bound to this call's mask, scale, softcap and sinks, to run on other queries, keys and values; goals' head-selective
   KV transfer calls it on copies of the keys and values and writes the chosen heads' last row. NRGS 1, demos 3.
5. **fixed** (ndif-team/nnterp#128): nnterp serves `attention_queries_prerope` / `attention_keys_prerope` (after any
   query / key norm, before the rotation, writable) and `self_attn.rotate(x, positions)`, the block's own rotary
   embedding; goals' Q transfer writes the queries before the rotary, and the KV transfer rotates SRC's keys before
   the rotary for the DST positions. NRGS 2, demos 3 and 4.
6. **fixed** (ndif-team/nnterp#134): per-head projection weights, `self_attn.value_weight(kv)` and
   `output_weight(j)`, in each family's own layout (Phi-3's fused `qkv_proj` included) and, under a PEFT LoRA, the
   weight the layer's forward applies. OVReader and the head vectors read them, so OVReader runs on Phi-3 and the EM
   adapter is never merged. Demos 4, EM adapter 4, head weights 1.
7. **filed** (ndif-team/nnterp#131): a forward-order rank for native submodule paths. nnterp ranks its values
   (`mt.rank(name, layer)`), which goals' `Intervention` sorts its edits by, but not `self_attn.q_proj` or
   `mlp.down_proj`. Goals needs none: every `Intervention` edit names an nnterp value, and the EM adapter is attached
   by nnsight, so its projections run in the model's own forward. NRGS 3, last hooks 4.
8. Build the attention modules' `.source` at load under eager, so reading a child module's value and then an
   interior value in one trace does not fail on a fresh model (goals reads no attention submodule in a trace). NRGS 5.
9. `self_attn.rotate` on a model spread over several GPUs: it applies the rotary embedding where the rotary module
   computed it, which under `device_map="auto"` can be another GPU than the block's. Proposal: move the embedding to
   `x`'s device, as the forward hands it to the block. NRGS 2. Fixed in nnterp #136 (v2.0.0 b9621bc).
10. `attention_input_gain` in float32: it is in the norm weight's dtype, so on a bf16 Gemma-2 it is `1 + weight`
   rounded to bf16, where the model multiplies by the float32 `1 + weight` (a float32 probe gives the exact gain on
   both parameterizations). Head weights 2.
11. A `support()` row for the per-head weights: `self_attn.no_head_weights()` is a method, so `mt.support()` does
    not list it, and a caller learns that a block has none from the method or the `Unavailable`. Head weights 3.

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
6. **fixed** (ndif-team/nnsight#824, on nnsight's dev branch, which goals pins): attaching an adapter (`peft=`,
   `load_adapter`) leaves every base parameter's `requires_grad` as it was, so `load_adapter(None)` gives back the
   base model's modules with their flags. peft's injection freezes the base, and a frozen base sends some bf16 matmuls
   to another kernel. EM adapter 2.
7. **fixed** (ndif-team/nnsight#824): `model.disable_adapters()` / `enable_adapters()` switch an attached adapter off
   and on without touching the module tree, and off is the base model bit for bit. Goals loads the 32B once with
   `peft=` and switches. EM adapter 1.

**Docs** (nnsight's, nnterp's, and the nnterp skill)

1. Subclassing `StandardizedTransformer` (plain attributes and class-level properties are safe). Model layer 4.
2. The nnterp skill's version line names a `0.8-refactor` / nnsight `dev` pair; nnterp 2.0.0rc1 + nnsight 0.8.2 on
   transformers 5.22.1 runs everything here. Model layer 6.
3. Composing a caller's intervention with a function's reads: the intervention in the input invoke, the reads in an
   empty second invoke. GoalScope 1.
4. `generate(do_sample=False)` keeps the checkpoint's `repetition_penalty` and extra EOS ids. GoalScope 3.
5. **fixed** (ndif-team/nnterp#127): nnterp's `attention-interior.md` and `generation.md` say that in-place edits of
   `attention_keys` / `attention_values` under `generate` persist in the KV cache. NRGS 1.
6. **fixed** (ndif-team/nnterp#125): the cheat-sheet and `attention-interior.md` say two invokes of one trace can both
   read and edit `attention_scores` / `attention_probabilities`. NRGS 4.
7. **fixed** (ndif-team/nnterp#127): nnterp's `vocabulary.md` says an envoy call outside a trace is a plain module
   call. Its "Recomputing by hand" section still reads `attn._module.scaling`. NRGS 7.
8. A stopping criterion ends a run early like EOS (nnterp's `generation.md` names only EOS). Last hooks 1; model
   layer 3 (the loop of traces threading `past_key_values` it replaced).
9. `proj.output = proj.output + f(proj.input)` reads `.output` first and fails; bind the input first. Last hooks 3.
10. Reading a value from inside an edit (append a saved value to a list made outside it). Last hooks 5.
11. An edit whose output is backpropagated must be an assignment, not an in-place write (nnterp's
    `attention-interior.md` shows only the in-place form). Head-mask 1.
12. The trace body runs under the caller's grad mode (`torch.enable_grad()` around `mt.trace` overrides a global
    `set_grad_enabled(False)`). Head-mask 3.
13. Reading an adapted weight: nnterp's per-head weights are the weights a LoRA layer applies (ndif-team/nnterp#134),
    so goals merges nothing. A caller who merges by hand meets two peft traps the docs could name: `unmerge()` in half
    precision does not give the base weight back bit for bit, and a merged layer whose adapter is off unmerges itself
    on its next forward (nnsight's "PEFT adapters" section and the model-editing skill's PEFT section). EM adapter 4.

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
4. **The pre-attention norm's gain is `attention_input_gain`** (ndif-team/nnterp#134; summary, nnterp 3). `OVReader`
   folds the gain of the norm the attention reads into each head's OV map. nnterp serves the normed stream
   (`attention_norm_output`) and its gain, `mt.layers[l].attention_input_gain`: the norm called on ±256 in
   alternating channels, which is `weight` on Qwen2 / Llama, `1 + weight` on Gemma, and ones on a post-norm block
   (OLMo-2/3), whose attention reads the stream itself. OVReader reads it outside the trace. A post-norm block also
   norms the attention's output before the stream, a second gain OVReader does not have, so OLMo-3 is still refused
   (head weights 4).
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

nnterp 2.0.0rc2.dev36 (`3fb2cc02`: v2.0.0 with #127 and #128; items 4 and 6 first seen on `feeb24c5`), nnsight
0.8.2, transformers 5.22.1, torch 2.11.0; CPU, Qwen/Qwen2.5-0.5B-Instruct (float32, eager; 14 query heads over 2 kv
heads) for `test/test_nrgs_port.py`, and Coder-7B (bf16, one RTX A6000) for the parity harness and the partial
recomputes in item 1.

1. **Patching keys / values per query head under GQA: nnterp serves `self_attn.interface`; goals uses it.** The KV
   transfer needs SRC's keys and values on a span read by chosen query heads only, by the last query row only (the
   pivot on the prompt pass, then each new token), with the cache left clean. `attention_keys` / `attention_values`
   are served before `repeat_kv` (`num_kv_heads` wide), so an edit there reaches every query head of the group (the
   "GQA bleed": a kv-head-wide write at layer 6 of the 0.5B moves the unpatched head 3 next to the patched head 2 by
   0.6, `test_gqa_bleed`), every query row reads it, and under `generate` an in-place edit of them is an edit of the
   `DynamicCache` (an in-place zero at step 0 is still there at step 1; assignment changes one forward only).
   nnterp serves `self_attn.interface` (ndif-team/nnterp#127): read with the queries, keys and values, it is the
   attention function the block called, bound to the rest of this call (this step's mask, the scale, Gemma-2's
   logit softcap, a sink, the sliding window), which runs on any queries, keys and values and returns
   `(head_outputs, pattern)` without writing anything. `headwise_kv_patch` is nnterp's head-selective K/V recipe
   (`docs/patterns/head-kv-patching.md`): per step and patched block, read `interface` and `attention_queries` /
   `_keys` / `_values`, clone k and v, replace the span of the patched kv heads, call `interface` on `q` and the
   copies, and write the chosen heads' last row of `attention_head_outputs` from the result. The call runs every
   query head and every row, the forward's own GEMM shapes, so the chosen heads' last row is bit for bit what the
   block computes over the replaced keys and values (the 7B parity `kv_transfer` case is bit-identical to the
   nnsight 0.7 reference, 609 checks; the Gemma-2 self-transplant leaves the logits unchanged, demos 3). Whole
   tensors matter: a computation of only the chosen heads, or only the last row, runs other shapes that round
   differently (on Coder-7B, bf16, up to 0.53 at the prompt pass and 0.2-0.38 after a few decode steps; in float32
   on CPU up to 5e-6), and nnterp's page says to pass whole tensors and slice the result. The cache behaviour of
   in-place K/V edits under `generate` is in nnterp's `attention-interior.md` and `generation.md` (#127).
2. **A transplant to another position: nnterp serves the queries and keys before the rotary embedding, and the
   block's own rotation; goals uses them.** `attention_queries` / `attention_keys` are rotated for their own
   positions, so a query or key moved to another position has to enter before the rotation. nnterp serves
   `attention_queries_prerope` / `attention_keys_prerope` (after any query / key norm, before the rotation, writable,
   under any `attn_implementation`) and `self_attn.rotate(x, positions)`, the block's own rotary embedding (its layer
   type, partial rotary and scaling) as an ordinary call outside a trace (ndif-team/nnterp#128). The Q transfer
   writes the SRC pivot's queries before the rotary into the DST pivot's (`attention_queries_prerope[:, heads,
   pivot] = src`, in place), and the block rotates them for the DST pivot. The KV transfer cannot write keys into the
   destination forward (that edits the cache and the whole group, item 1), so it rotates SRC's window of
   `attention_keys_prerope` for the DST positions with `rotate`, outside the trace, and passes the result to
   `interface`. The shift is real: the SRC and DST windows start at the same position for py-py and cpp-cpp but not
   across languages (the C++ note is in the system prompt; e.g. positions 61 vs 72 for int py-cpp), which is half of
   the reported KV grid. On Qwen2 the values before the rotary are the projections' outputs, heads split off, so
   the 7B parity `q_transfer` (523 checks) and `kv_transfer` cases are bit-identical to the reference; on Qwen3 and
   OLMo-3 they come after the query / key norms and on Phi-3 out of the fused `qkv_proj`, so the transplanted
   vectors are the model's own there too (demos 3 and 4). On a model spread over several GPUs (`device_map="auto"`)
   `rotate` fails: the model's rotary module can sit on another GPU than the block, its accelerate hook returns the
   cos / sin there, and `rotate` applies them to keys on the block's GPU (GLM-4.5-Air on four A100s: cuda:0 and
   cuda:3 in `apply_rotary_pos_emb`). The forward instead hands each block its embedding on the block's device. The
   KV transfer calls `rotate`'s two halves, `rotary_embedding` and `apply_rotary` on the embedding moved to the block
   (on one GPU the move is a no-op and the result is `rotate`'s, bit for bit).
3. **Composing interventions needs a forward order; nnterp has it (`mt.rank`).** An intervention made of edits in
   several blocks cannot run after another one: the patched GoalScope readout of the Q transfer combines a query
   patch and an OV boost over the same 250 heads (blocks 2-63), and the boost at block 30 after the query patch at
   block 60 raises `OutOfOrderError` in a trace, or inside `tracer.iter` binds the next step's occurrence without an
   error. `src.nrgs.functional.Intervention` sorts its edits by `mt.rank(name, layer)` when it is built
   (`mt.order(layer)` lists a block's values in the order the forward reaches them, measured by one probe scan per
   model). Every edit names the first nnterp value it touches (`self_attn.attention_queries_prerope` for the Q patch,
   `self_attn.attention_queries` and `self_attn.attention_head_outputs` for the KV transfer's call and write), so
   every edit has a rank and the order has no special case. `rank` is on nnterp's availability page and in the API
   reference; nnterp's `CLAUDE.md` and the nnterp skill route to it (nnterp #125, #126). It takes nnterp's values,
   not native module paths (`mt.rank("self_attn.q_proj.output", 6)` raises `KeyError`; ndif-team/nnterp#131 asks for
   them), and goals needs no native path ranked: the EM adapter is nnsight's (last hooks 4).
4. **The two-invoke limitation on `attention_scores` / `attention_probabilities` no longer reproduces.** nnterp's
   `CLAUDE.md` cheat-sheet and `attention-interior.md` say two invokes of one trace cannot both touch them
   (`TypeError`). On this stack every combination works, on a fresh model too: a scores edit in an input invoke with a
   read of the scores, the pattern or the head outputs in an empty invoke; two input invokes reading the same value.
   `test_scores_edit_composes_with_attention_matrices` passes `whitelist_head_attn` (a scores edit) to
   `get_attention_matrices`, and the pattern read in the second invoke carries it. The caveat in this repo's
   `get_attention_matrices` docstring is gone. Proposal: drop it from the nnterp docs, or name the nnsight version
   that fixed it.
5. **The fresh-model `.source` gotcha.** On a fresh model, the first trace that reads an attention block's child
   module (`q_proj.output`) and then an interior value of the same block (`attention_values`,
   `attention_scores`) raises `OutOfOrderError` naming `attention_interface_1.fn.i0`; nnterp's
   `attention-interior.md` names it (#127). Goals reads no attention submodule in a trace: the Q patch writes
   `attention_queries_prerope`, a value on the attention's own forward, and the KV transfer reads the interface's
   inputs, so neither touches `self_attn.source` ahead of a trace. On a fresh 0.5B, the gaze pass under the Q patch
   (the patch in the input invoke, the values and pattern read in a second invoke), a KV-transfer `greedy` and a
   Q patch + KV transfer + OV boost on one block each run as the first trace. Proposal, for callers that do read a
   submodule: build the attention modules' `.source` at load under eager, so no caller has to know.
6. **A swapped `forward` (PR 3, item 6) is gone.** The KV transfer needs no swap of the attention modules'
   `forward` (item 1), and the `legacy_hooks` bridge PR 3 added for it is removed. What stays worth a fix in nnsight:
   a replaced `forward` makes `.output` fail with an `OutOfOrderError` that does not name the cause (only `.source`
   warns).
7. **An envoy call outside a trace is an ordinary module call.** The task-vs-progress embedding transplant calls
   `mt.embed_tokens(src_ids)` outside any trace. `Envoy.__call__` documents that this is a plain call, and nnterp's
   `vocabulary.md` ("Calling a module outside a trace", #127) says so too; nnterp's "Recomputing by hand" section
   still reads the softmax scale off `attn._module.scaling`, a private route, where `interface` needs none.

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
   occurrence, so an edit meant for every forward of a `generate` (a LoRA written as output edits on each
   projection, every step) goes under an open `tracer.iter[:]`, as nnsight's `docs/usage/edit.md` shows. That loop
   ends each run by asking for a step that never comes, so every trace and every `generate` while the edit is
   installed emits `'...q_proj.input.i1' was never reached: the loop asked for a step the run did not make` (a single
   trace too). Goals has no such edit: the EM adapter is nnsight's (item 4). Proposal: `model.edit(inplace=True,
   every_call=True)`, or no warning for an open loop inside an edit, where outrunning the run is the intent.
3. **A hook's `o + f(i)` does not port as one expression.** `proj.output = proj.output + s * ((proj.input @ A.T)
   @ B.T)` evaluates `.output` first and raises `OutOfOrderError` on `proj.input` ("the model already ran past it");
   `x = proj.input` has to come first. The forward-order rule is documented; this instance of it (Python's
   evaluation order inside one statement) is the first thing a forward-hook port writes. Proposal: a line next to
   the forward-order rule in nnsight's `access-and-modify.md`.
4. **A LoRA adapter is nnsight's, so the order of the projections is not goals' concern.** nnsight's
   `TransformersModel` attaches a PEFT adapter in place, at load (`peft=`, a hub repo id or a local path) or on a
   loaded model (`load_adapter(path)`, and `load_adapter(None)` to detach it), and nnterp's `StandardizedTransformer`
   passes both through. Each targeted projection becomes peft's `lora.Linear` holding the original as `base_layer`,
   every native path and nnterp value names the same module as before (`support(layer=i)` is unchanged, `interface`
   reproduces `attention_head_outputs` bit for bit), and the adapter runs inside the model's own forward. nnsight's
   `docs/models/transformers-model.md` documents `peft=` and `load_adapter`. A rank for native module paths
   (`mt.rank("self_attn.q_proj.output", 6)` raises `KeyError`) is for callers whose own edits name native modules:
   ndif-team/nnterp#131.
5. **An edit can observe another function's run.** The parity harness reads the step-0 logits of
   `head_additive_generate`'s own `generate` with an edit that appends `mt.logits[:, -1].save()` to a list made
   outside it; each run appends its value, readable after the run (bit-identical to the forward hook it replaces).
   Neither nnsight's `edit.md` nor nnterp's docs show reading a value from an edit.

## Demos, docs and the cross-family check (`port/demos-docs`)

nnterp 2.0.0rc2.dev36 (`3fb2cc02`), nnsight 0.8.2, transformers 5.22.1, torch 2.11.0; bf16, eager, one RTX A6000;
`scripts/cross_family_check.py` on Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, Olmo-3-7B-Instruct,
gemma-2-9b-it, Qwen3-8B, GLM-4-9B-0414 and Phi-4-mini-instruct (CLAUDE.md section 7 has the matrix).

1. **Every value the repo reads or edits is served on all seven families (feedback, not a gap).** `support()` reports
   the attention interior available on each under eager, and the edits of `layer_output`, `attention_premix`,
   `attention_scores` and `attention_head_outputs` work unchanged: GoalScope (both strategies), the attention
   patterns, the head vectors on a padded batch, the sink and blind-span ablations (a sunk head puts all its pivot-row
   mass on key 0, a blinded head none on the blocked keys), the OV boost and the temporal encodings run on all seven.
   OVReader, which read projection weights and a norm gain below nnterp's vocabulary (items 2 and 4), reads nnterp's
   per-head weights (ndif-team/nnterp#134) and runs on Phi-3 now; it refuses OLMo-3, Gemma-2 and GLM-4, whose
   attention output passes a norm whose gain is not served (item 2).
2. **Norm gains (GoalScope item 4, continued).** OVReader's input gain is nnterp's `attention_input_gain`. In bf16 it
   is bit-equal to the weight on every block of Qwen2.5-Coder-7B, Llama-3.1-8B, Qwen3-8B, GLM-4-9B and Phi-4-mini,
   and ones on all 32 OLMo-3 blocks; on gemma-2-9b it is `1 + weight` rounded to bf16 (head weights 2). On post-norm
   and sandwich blocks (OLMo-3, Gemma-2, GLM-4) the head's write passes a second norm after o_proj, whose gain sets
   its direction in the stream; that gain is not served, so OVReader refuses those three (head weights 4).
3. **The KV and Q transfers are the model's own attention and queries on all seven families (NRGS items 1 and 2;
   feedback, not a gap).** The self-transplant check (DST's own window written back, V x1) leaves the logits
   bit-identical on every family: on Gemma-2 `self_attn.interface` carries the attention logit softcap (50); on Qwen3
   and OLMo-3 `attention_keys_prerope` comes after the key norm (Qwen3's per-head `k_norm`, OLMo-3's across all
   heads); on OLMo-3 `rotate` passes the block's layer type (sliding or full) to the rotary embedding; on Phi-3 the
   keys come out of the fused `qkv_proj` and `rotate` applies its partial rotary. The Q patch writes
   `attention_queries_prerope`, after the query norm, so the query the DST pivot attends with is the SRC pivot's query
   rotated for the DST pivot, bit for bit, on OLMo-3 and Qwen3 (one head of the middle block). A write at
   `q_proj.output` lands there on Qwen3, whose query norm is per head, and misses it by up to 0.44 (of 6.5) on OLMo-3,
   whose query norm spans all heads and so renormalizes the written head together with the DST heads around it.
4. **Fused projections (Phi-3).** Phi-3 has `qkv_proj` and no `q_proj` / `k_proj` / `v_proj`. nnterp's
   `value_weight(kv)` reads key/value head `kv`'s rows of `qkv_proj` (every query, every key, then every value;
   ndif-team/nnterp#134, summary item 6), so OVReader runs on Phi-3, as the Q and KV transfers do through nnterp's
   queries, keys and values. On Phi-4-mini's middle block, `value_weight(kv) @ attention_norm_output` is
   `attention_values[:, kv]` to bf16 rounding (0.03 at values up to 8.6; the next head's weight misses by 3.7 or
   more).

## The EM adapter on nnsight's adapter support (`port/em-adapter-peft`, `port/em-adapter-switch`; item 4 `port/head-weight-accessors`)

nnterp 2.0.0rc2.dev36 (`3fb2cc02`), nnsight's dev branch at `e09ab1c8` (0.8.0rc2.dev236, carries ndif-team/nnsight#824),
peft 0.21.2, transformers 5.22.1, torch 2.11.0; CPU, float32 Qwen/Qwen2.5-0.5B-Instruct for
`test/test_last_hooks_port.py`; Qwen/Qwen2.5-32B-Instruct with
`ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice` (bf16, eager, one A100-80GB) for the EM evaluation;
item 4 on nnterp `200ac05b` (ndif-team/nnterp#134).

1. **One model, the adapter switched off and on** (nnsight #824; summary, nnsight 7). The EM evaluation runs both
   sides on one loaded 32B model. It is loaded with the adapter attached (`peft=`, a local dir or a hub repo id, which
   `ModelandTokenizer` passes through nnterp to nnsight) and switched off at once (`mt.disable_adapters()`);
   `em_adapter` switches it on (`mt.enable_adapters()`) for the EM side and off again after it. The switch moves no
   module, so every native path and nnterp value names the same module in both states, and with the adapter off the
   model is the base model bit for bit: in test_em_adapter the logits, the generation and the source vectors of the
   model with its adapter off equal those of a model that never had one (0.5B, float32 on CPU; the same holds for the
   logits in bf16 on an A6000). `mt.adapters_enabled` holds the state. On the 32B, entering `em_adapter` (the
   switch and a probe forward) takes 0.10 s, where the detach and re-attach before (`load_adapter(None)`,
   `load_adapter(path)`) read the adapter's 1.07 GB again for every EM side (3.1 s).
2. **The base parameters keep their `requires_grad`** (nnsight #824; summary, nnsight 6). peft's injection sets
   `requires_grad=False` on every base parameter; nnsight's attach gives the flags back and the switch leaves them
   alone, so `em_adapter` has nothing to restore. The flags matter outside training: `F.linear` on an input it cannot
   flatten without a copy (an expanded batch) folds it when the weight requires grad and runs a batched matmul when it
   does not, which in bf16 gives other bits (one random case: up to 0.5). On the 32B a frozen base changes 2 of the
   1040 innocent readouts.
3. **A half-precision model runs the adapter in float32 (a behaviour difference).** nnsight's attach casts a
   half-precision adapter to float32, as `PeftModel` does, and the EM adapter is stored in float32 anyway; peft's
   `lora.Linear` computes the delta in float32 and casts the sum back. The hand-written edits rounded A and B to bf16
   and computed in bf16. In float32 the two are bit-identical (test_em_adapter). On the 32B in bf16, the EM logits on
   the four prefilled prompts differ by up to 1.0 (logits up to 34.5; KL at the last position at most 2e-3), and the
   greedy EM answer stays the same on one example and diverges at a near-tie on three (generated tokens 5, 16 and
   40). Computing the adapter as the old code did (A and B rounded to bf16, the q/k/v/gate/up/down deltas in bf16,
   the o_proj merge rounded twice: peft's `safe_merge`) gives the old logits and the old EM outputs bit for bit, so
   the adapter's precision is the whole difference.
   nnsight's `autocast_adapter_dtype=False` (#824) holds the adapter in the model's dtype instead; goals keeps the
   default.
4. **Nothing is merged: GoalScope's head weights are the adapted ones** (ndif-team/nnterp#134; summary, nnterp 6).
   `collect_head_steering_vecs` reads each head's W_O through `self_attn.output_weight(h)`, which on a lora layer
   whose adapter is on is the base weight plus the adapter's delta, in the base weight's dtype, as peft's `merge()`
   computes it: with the 32B's bf16 base and float32 adapter it equals the slice of the merged o_proj weight bit for
   bit at all 122 goal heads, and `value_weight` the merged v_proj's. So `em_adapter` only switches the adapter, the
   load-time o_proj copy is gone (3.1 GiB: peak 62.55 GiB against 65.68), and the EM side runs o_proj unmerged, as
   `PeftModel` does (`base(x) + B(A(x)) * s`), where #9 and #10 ran it merged. In bf16 the two differ by rounding: on
   the four prefilled prompts the EM logits move by up to 1.2 (logits up to 34.5; the adapter itself moves them by 33
   to 45), KL at the last position at most 3.8e-3, the size of item 3's float32-adapter change. That moves the EM
   side of the evaluation: on quickbuck and husband the generated ids are the same and the head vectors differ by up
   to 0.5 (of 40); on ruler and gender the greedy EM answer leaves #10's at generated token 28 and 16; 39 of the 188
   EM readouts are unchanged. With o_proj merged as #10 merged it and the head weights still read through the
   accessors, the whole evaluation is #10's bit for bit (all 1228 readouts), so the unmerged forward is the whole
   difference. The innocent side is unchanged, all 1040 readouts. The two traps of a hand merge (summary, docs 13)
   no longer arise: `unmerge()` in bf16 does not give the weight back (all 64 o_proj weights of the 32B off by up to
   9.8e-4 after a merge and unmerge), and a layer still marked merged when its adapter goes off unmerges itself on
   its next forward.

## Per-head weights from nnterp (`port/head-weight-accessors`)

nnterp 2.0.0rc2.dev54 (`200ac05b`: v2.0.0 with ndif-team/nnterp#134), nnsight's dev branch at `e09ab1c8`, peft 0.21.2,
transformers 5.22.1, torch 2.11.0; CPU, float32 Qwen/Qwen2.5-0.5B-Instruct for the A/B tests; bf16, eager, one RTX
A6000 for the seven cross-family models; one A100-80GB for the 32B EM evaluation.

1. **The accessors replace every weight read (feedback, not a gap).** OVReader's factors are
   `self_attn.output_weight(j)`, `value_weight(kv)` and the block's `attention_input_gain`, and
   `collect_head_steering_vecs` takes each head's W_O from `output_weight(h)`; nothing in `src/` or `scripts/` slices
   a native `.weight` any more. On Qwen2.5-0.5B the factors are `torch.equal` to the old slices and the gain to the
   constant-256 probe on all 24 blocks; OVReader runs on Phi-3, which it refused. Without an adapter the weights are
   views of the parameters; with the EM adapter on, each call computes its projection's whole adapted weight to
   return one head's slice, and OVReader's factors for the 122 heads take 0.43 s on the 32B.
2. **The gain is in the norm weight's dtype (summary, nnterp 9).** nnterp probes the norm with ±256 in the weight's
   dtype, so on a bf16 Gemma, whose norm computes `x * (1 + weight)` in float32, the gain comes back rounded to bf16.
   On gemma-2-9b it differs from the float32 `1 + weight` of the constant-256 probe on all 42 blocks, in 75856 of the
   150528 channels, by up to 3.9e-3 relative (one bf16 rounding); on the other six families it is bit-equal. A
   float32 probe, where the norm accepts one, would give the exact gain. Goals takes the accessor as it is: OVReader
   refuses Gemma-2 for its norm after the attention anyway (item 4), and the paper's families (Qwen2, Llama, OLMo-3,
   GLM-4) have no `1 + weight` norm.
3. **`no_head_weights` is not a `support()` row (summary, nnterp 10).** `self_attn.no_head_weights()` is a method,
   and `mt.support()` lists values only, so it has no `self_attn.no_head_weights` entry. OVReader lets nnterp's
   `Unavailable` surface: it is raised when the reader is built, outside any trace, and its message names the block
   and the reason, so a second check in goals would only repeat it.
4. **A norm after the attention: OVReader refuses the block (summary, nnterp 3).** OVReader's map ends at `W_O`, the
   head's write into the stream only where the block adds the attention's output as it is. OLMo-2/3 (post-norm) and
   Gemma-2/3 and GLM-4 (a norm after the attention as well as before it) norm that output first, which scales each
   head's write per channel by the norm's gain (the `1/rms` is shared by the heads and drops out of the unit writes).
   Without that gain the OV write's direction is wrong: in the audit of #134 the top-10 tokens of a head's OV write
   overlap those with the gain folded only 0.64-0.88 on Olmo-3-7B-Instruct and 0.575 on gemma-2-2b. nnterp does not
   serve the gain yet (`attention_output_gain` is the follow-up), so OVReader refuses such a block when it is built,
   naming the norm. It finds them from nnterp, without per-family code: on every family that norms the attention's
   output, nnterp reads `attention_output` (what the attention adds to the stream) at that norm's output,
   `../post_attention_layernorm.output` (`../post_self_attn_layernorm.output` on GLM-4), where elsewhere it is the
   attention's own output (or an op of the block's forward that multiplies it by a scalar, which leaves directions
   alone). The cross-family check's OVReader row fails on OLMo-3, Gemma-2 and GLM-4 with that reason; before, it ran
   on Gemma-2 and GLM-4 and silently missed the norm.
