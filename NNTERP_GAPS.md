# nnterp / nnsight gaps found during the port

A running list of what was awkward, undocumented or impossible in nnterp or nnsight while porting this repo from
baukit hooks to nnterp. Every port PR appends its own section. Each entry says what happened, gives a one-line repro or
a pointer, and names the workaround used here, if any.

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
