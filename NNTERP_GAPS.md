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
