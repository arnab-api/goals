# Parity harness

Reference outputs of the current (baukit hooks + nnsight 0.7) code on `Qwen/Qwen2.5-Coder-7B-Instruct`, and the
scripts that regenerate and compare them. A port of the interventions (to nnterp / nnsight 0.8) passes when
`compare.py` passes against `reference/Qwen2.5-Coder-7B-Instruct/`.

| file | what it is |
|---|---|
| `cases.py` | the fixed case set: which specs / instances / prompts each case uses, and the 7B-scaled constants |
| `run.py` | runs every case through the repo's public functions; writes `<case>.json`, `<case>.pt`, `env.json` |
| `compare.py` | compares two output directories case by case; exit status 1 on any failure |
| `heads_qwen25coder7b.json` | the 60-head set (`python test/parity/cases.py` checks it against `make_head_set()`) |
| `reference/Qwen2.5-Coder-7B-Instruct/` | the reference outputs (9.5 MB) |

## Run

```bash
# once: the model (about 15 GB); run.py itself sets HF_HUB_OFFLINE=1 and never downloads
hf download Qwen/Qwen2.5-Coder-7B-Instruct

# from the repo root, with env.yml present (cp env_demo.yml env.yml; set RESULTS_DIR to a scratch dir)
CUDA_VISIBLE_DEVICES=0 python test/parity/run.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out results_parity/new
python test/parity/compare.py test/parity/reference/Qwen2.5-Coder-7B-Instruct results_parity/new
```

`run.py` needs one GPU with about 20 GB free (bf16 weights 15.2 GB, eager attention, `device_map={"": 0}`) and takes
about 50 s on an RTX A6000 (model load from the page cache included). `--cases <name> ...` runs a subset. No case calls
the LM judge: `run.py` replaces `src.utils.oracle_llms.ask_claude` with a function that raises before any repo module
imports it, so an accidental judge call fails the case instead of reaching the API.

`compare.py <ref> <new>` applies the bf16 tolerances below; `compare.py <ref> <new> --exact` demands bit-identity.

## Reference environment

The old code does not run on nnsight 0.8 (`mt._model`, `scan=False`, `.source.attention_interface_0`), so the
environment pins `nnsight<0.8`. nnsight 0.7.0 does not cap transformers, and the old code needs transformers 5: the
attention forward that `headwise_kv_patch` swaps in (`src/nrgs/functional.py:336-358`) takes the plural
`past_key_values` keyword, and the authors' own notebook logs on `master` record transformers 5.3.0 with nnsight
0.6.2 and with torch 2.10. uv resolved transformers 5.22.1, which keeps the transformers version the same as the
port's environment, so differences in a port run come from the port rather than from transformers.

```bash
uv venv --python 3.12 .venv
printf 'nnsight>=0.5,<0.8\n' > /tmp/overrides.txt
uv pip install --python .venv/bin/python -r pyproject.toml --override /tmp/overrides.txt
```

| | |
|---|---|
| python | 3.12.13 |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| transformers | 5.22.1 |
| nnsight | 0.7.0 |
| baukit | git 9d51abd51ebf29769aecc38c4cbef459b731a36e |
| accelerate / tokenizers / safetensors / numpy | 1.15.0 / 0.23.3 / 0.8.0 / 2.5.4 |
| GPU | NVIDIA RTX A6000 (sm 8.6), driver 555.42.02 |
| model | `Qwen/Qwen2.5-Coder-7B-Instruct` snapshot c03e6d358207e414f1eca0bb1891e29f1db0e242, bf16, `attn_implementation="eager"` |
| TF32 matmul | off (PyTorch default) |
| repo | `nnterp` = `clean` at bbc4118 |

`reference/.../env.json` holds the same plus every installed package.

<details><summary><code>uv pip freeze</code></summary>

```
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anthropic==2.4.0
anyio==4.16.0
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
astor==0.8.1
asttokens==3.0.2
async-lru==2.4.0
attrs==26.1.0
babel==2.18.0
baukit @ git+https://github.com/davidbau/baukit@9d51abd51ebf29769aecc38c4cbef459b731a36e
beautifulsoup4==4.15.0
bidict==0.24.1
bleach==6.4.0
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
circuitsvis==1.43.3
click==8.5.0
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.4.0
cuda-bindings==12.9.10
cuda-pathfinder==1.8.5
cuda-toolkit==12.8.1
cycler==0.12.1
dataclasses-json==0.6.7
datasets==5.1.0
debugpy==1.8.22
defusedxml==0.7.1
dill==0.4.1
docstring-parser==0.18.0
executing==2.3.0
fastjsonschema==2.22.3
filelock==4.2.7
fonttools==4.68.0
fqdn==1.6.0
frozenlist==1.8.0
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpcore2==2.16.0
httpx==0.28.1
httpx2==2.16.0
huggingface-hub==2.4.0
idna==3.20
importlib-metadata==10.1.0
ipykernel==7.4.0
ipython==9.18.1
ipython-pygments-lexers==1.1.1
ipywidgets==8.1.9
isoduration==20.11.0
jedi==0.20.1
jinja2==3.1.6
jiter==0.18.0
json5==0.17.3
jsonpointer==3.8
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-builder==1.2.4
jupyter-client==8.10.0
jupyter-core==5.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter-server==2.21.1
jupyter-server-terminals==0.5.4
jupyterlab==4.6.5
jupyterlab-pygments==0.3.0
jupyterlab-server==2.28.1
jupyterlab-widgets==3.0.17
kiwisolver==1.5.1
lark==1.3.1
markdown-it-py==4.2.0
markupsafe==3.0.4
marshmallow==3.26.2
matplotlib==3.11.2
matplotlib-inline==0.2.2
mdurl==0.1.2
mistune==3.4.0
mpmath==1.3.0
multidict==7.2.1
multiprocess==0.70.19
mypy-extensions==1.1.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.7
nnsight==0.7.0
notebook-shim==0.2.4
numpy==2.5.4
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.19.0.56
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.28.9
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
opentelemetry-api==1.45.2
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pexpect==4.9.0
pillow==13.0.0
platformdirs==4.14.3
prometheus-client==0.26.0
prompt-toolkit==3.0.53
propcache==0.5.4
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pyarrow==26.0.0
pycparser==3.12
pydantic==2.14.2
pydantic-core==2.50.2
pygments==2.21.0
pyparsing==3.3.3
python-dateutil==4.0.0
python-engineio==4.15.1
python-json-logger==4.2.0
python-socketio==5.18.0
pyyaml==6.0.3
pyzmq==27.2.0
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rich==15.0.0
rpds-py==2026.9.1
safetensors==0.8.0
send2trash==2.1.0
sentencepiece==0.2.2
setuptools==81.0.0
shellingham==1.5.4
simple-websocket==1.1.0
six==1.17.0
sniffio==1.3.1
soupsieve==3.0.4
stack-data==0.6.3
sympy==1.14.0
terminado==0.18.1
tinycss2==1.5.1
tokenizers==0.23.3
toml==0.10.2
torch==2.11.0+cu128
torchvision==0.26.0+cu128
tornado==6.6
tqdm==4.70.1
traitlets==5.16.1
transformers==5.22.1
triton==3.6.0
truststore==0.10.5
typer==0.27.3
typing-extensions==4.16.0
typing-inspect==0.9.0
typing-inspection==0.4.4
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
widgetsnbextension==4.0.16
wsproto==1.3.2
xxhash==4.0.2
yarl==1.25.1
zipp==4.1.2
zstandard==0.25.0
```

</details>

Two runs of `run.py` in this environment on the same GPU are bit-identical in every case
(`compare.py --exact`: 15 / 15 pass, 3378 checks).

## Cases

Every generation and read-out is 32 new tokens. The 7B has 28 layers, so the 32B's layer constants are scaled:
steer layers 15-20 (32B: 35-45), OVReader probe layers 10 and 21, sufficiency `from_layer` 9 (32B: 20).

Two head sets ("arms"). `heads60` is the seeded random set of `heads_qwen25coder7b.json` (60 heads from layers 10-25).
It barely moves the 7B: sink ablation, OV boost and KV / Q transfer leave the greedy text unchanged with it. `band448`
(every head of layers 10-25) makes each of those interventions change the text, so a port that drops or misplaces an
intervention fails a string check rather than only a logit tolerance. Intervention cases run both arms.

| case | entry points | input | output |
|---|---|---|---|
| `hs_probs` | `get_hs`, `get_next_token_probs` | `list[str]_python_001` prompt | embedding / layer 0, 7, 14, 21, 27 / final-norm outputs at the last 2 positions; the full next-token distribution |
| `attention` | `get_attention_matrices` (plain, value-weighted) | `int_python_002` prompt (118 tokens) | layers 0 and 18 in full, the last query row of every layer, last-position logits |
| `head_vecs_single` | `collect_head_steering_vecs`, `get_o_proj_inputs_batch` (TokenizerOutput and list[str] paths) | `list[int]_python_001` prompt alone | steering vectors and o_proj inputs (16 head layers) at the last 5 positions; last-token o_proj inputs |
| `head_vecs_padded` | same | `int_python_001` + `list[int]_python_001` in one batch (123 / 119 tokens, 4 left pads) | same, **padded** |
| `gs_additive` | `apply_goalscope` additive_norm alpha 0.1 (faithfulness probe; causality probe on the first spec), `head_additive_generate` | `int_python_000`, `list[str]_python_000`: prompt, 32-token greedy completion, the 5 pivots | completions, read-outs per pivot and arm, steering vectors, step-0 logits of the steered probe; one `n_steer_tokens=2` arm |
| `gs_replace` | `apply_goalscope` replace alpha 1 (`headpatched_generate`), `HeadMask`, `forward_with_src_replacement` | `str_python_000` | read-outs per arm, source o_proj inputs, step-0 logits; one `n_steer_tokens=2` arm |
| `ov_reader` | `OVReader.apply_probe_on_prompt` alpha 0.15, `resid_ov_vec` | `int_python_000`, `eq` pivot | read-outs at probe layers 10 / 21 per arm, residual latents, OV vectors, step-0 logits |
| `temporal` | `temporal_persistence.build_probe_prompt`, `rollout_probe`, `locate_value`, `goal_encoding` | `multi_predicate_setups("int", [3, 4], seed 1123)[0]`, target t=1 | goal encodings at the `=` pivot + implementation tokens per arm; the oracle segment is replaced by the first generated line |
| `ablation_blind` | `frontier_block_positions`, `complement_heads`, `generate_blocking_span_cached` | `list[int]_python_000` | baseline / ablated (spare heads60, blind 472) / control generations |
| `ablation_sink` | `random_heads_like`, `sink_generate_cached` | `str_python_001` | baseline / ablated heads60 / control / ablated band448 generations |
| `ov_boost` | `build_obstacle_prompt`, `greedy_continue`, `boost_head_ov`, the script's `boosted_heads` and `run_obstacle` | `candidates()[0]` = list[str] `total_length`, obstacle `" 0 + "` | continuations (clean, x2 per arm, the script's path), step-0 logits of each path |
| `kv_transfer` | `sample_multi_instances`, `build_geometry`, `clean_runs`, `run_patched` (`readout=False`), `intervention_stack`, `cache_src_proj`, `goalscope_readout` | instance 0 of int py-py (seed 1123), V x1 and x4 | geometry, clean and patched generations per arm, step-0 logits, source k/v of 2 layers, read-outs at the pivot with the KV patch entered around the source trace (`iface` -> `legacy_hooks`) |
| `q_transfer` | `diff_trigger_setups`, `run_q_transfer_intervention` (`compute_readout=False`) | instance 0 of order_diff int py-py (cell seed as in the script) | every raw field (generations, gaze scores, selection, restrict set, pivot attention) per arm, the source query cache |
| `mask_grad_equal` | `get_next_token_probs_batch`, `get_o_proj_inputs_batch`, `forward_with_src_replacement`, `HeadMask` | Coder-32B pivot instances 0 and 24 (equal source / destination / target lengths, no padding) | one `train_head_mask` step at mask 0.5: mask gradient, KL and sparsity losses, patched logits (full) |
| `mask_grad_padded` | same | instances 0 and 52 (source, destination and target lengths all differ) | same, **padded** |

## Entry points `run.py` depends on

The port keeps these names and signatures, or changes `run.py` in the same PR (and says so):

- `src.nrgs.functional.setup(model_name, device_map)` returning a `Ctx` and binding the module global `mt`;
  `boost_head_ov(mt, heads, alpha)`; `goalscope_readout(tag, prompt, pivot, heads=, iface=, layers=, maxnew=, ctx=)`
  (`iface`: a zero-argument callable returning a hook context, passed on as `apply_goalscope(legacy_hooks=)`);
  `cache_src_proj(prompt, positions, layers, which, ctx=None)`;
  `run_q_transfer_intervention(mt, inst, patch_heads, readout_heads=, *, max_new, alpha, compute_readout, gsr_max_new)`.
- `src.nrgs.kv_transfer`: `sample_multi_instances`, `build_geometry(inst, *, span, ctx)`,
  `clean_runs(g, *, readout_heads, max_new, readout, ctx)`, `run_patched(g, *, goal_heads, readout_heads, v_scale,
  max_new, readout, ctx)`, `intervention_stack(g, *, goal_heads, v_scale, ctx)`, `geometry_info`, `inst_key`, `_ids`.
- `src.nrgs.data`: `multi_predicate_setups`, `diff_trigger_setups`, the module global `CPP_NOTE_IN_SYSTEM`.
- `src.goals.goalscope`: `apply_goalscope(prompt, *, mt, selected_heads, strategy, steer_layer_range, alpha,
  max_new_tokens, n_steer_tokens, goalscope_prompt, include_noncode, token_positions)`,
  `head_additive_generate(mt, dest_inputs, steering_vecs, layers, max_new_tokens, n_steer_tokens=None, alpha=0.1)`,
  `greedy_continue(mt, input_ids, attention_mask, max_new_tokens)`, `OVReader(mt, heads)` with
  `.apply_probe_on_prompt`, `.resid_ov_vec`, `.probe_inputs`, and `_build_goalscope_prompt` / `_build_causality_prompt`.
- `src.goals.head_loc_attn`: `HeadMask(n_layers_tested, n_heads)` with `.values`,
  `get_o_proj_inputs_batch(mt, prompts, layers, token_positions=None, interventions=None)`,
  `collect_head_steering_vecs(mt, src_tokenized, selected_heads, token_positions, interventions=None)`,
  `forward_with_src_replacement(mt, dest_inputs, src_head_inps, mask, mask_layers, alpha, patch_positions,
  return_past_key_values)`.
- `src.goals.head_loc_pivot`: `get_next_token_probs(mt, prompt)`, `get_next_token_probs_batch(mt, prompts)`,
  `build_cpp_prompt`, `build_python_prompt`; `src.goals.head_loc_data.HeadLocDataset.load(model_name=)`.
- `src.goals.temporal_persistence`: `build_probe_prompt`, `rollout_probe`, `_offsets`, `prompt_boundary`,
  `locate_value`, `goal_encoding(mt, text, positions, goal_heads)`, `context_key`, `N_COND`.
- `src.functional.get_hs(mt, input, locations, return_dict)`, with module names `model.embed_tokens`,
  `model.layers.{i}`, `model.norm`; `src.attention.get_attention_matrices(input, mt, value_weighted)` returning
  `.attention_matrices` (L, H, T, T), `.logits`, `.tokenized_prompt`; `src.tokens.prepare_input`.
- `scripts.evaluation.eval_goalscope`: `build_prompt`, `generate_completion`, `pivot_positions`, `frontier_impl_span`,
  `load_json_heads`, `load_locked_datapoints`, `ANCHOR_ORDER`, `ALL_DATATYPES`, `DEFAULT_DATASET`.
- `scripts.evaluation.ablation_common`: `complement_heads`, `random_heads_like`, `frontier_block_positions`,
  `generate_blocking_span_cached(mt, prompt_ids, block_heads, block_positions, q_start, max_new)`,
  `sink_generate_cached(mt, prompt_ids, ablated_heads, q_start, sink_pos, max_new)`.
- `scripts.evaluation.obstacle_persistence_eval.build_obstacle_prompt`;
  `scripts.evaluation.obstacle_persistence_ov_boost_eval`: `candidates`, `boosted_heads`, `run_obstacle`, `MAX_NEW`,
  `NUMERIC_OBSTACLE`.
- On the model object: `mt.tokenizer`, `mt._model` (an HF `Qwen2ForCausalLM` callable with `input_ids` /
  `attention_mask` and returning `.logits`; `.get_output_embeddings()` for the step-0 logit observer), `mt.n_layer`,
  `mt.n_heads`, `mt.layer_name_format`; `src.utils.oracle_llms.ask_claude` (replaced by the guard).

## Tolerances

Default mode compares across stacks. Relative error is Frobenius, `||new - ref|| / ||ref||`. A wrong position, head
slice, layer or scale gives relative errors of order 1; the bounds below admit kernel and op-order differences.

| kind | check | why |
|---|---|---|
| `act` (activations read from the model) | rel <= 2e-2 and cosine >= 0.9995 | bf16 activations; deep-layer outliers make max-abs meaningless |
| `vec` (steering, OV, goal-encoding vectors) | rel <= 2e-2 and cosine >= 0.9995 | fp32 sums of bf16 products |
| `attn` | max-abs <= 1e-2 | probabilities; the bf16 ulp just below 1.0 is 2^-8 = 0.0039 |
| `attn_vw` | rel <= 2e-2 | attention x value norm |
| `logits` | rel <= 2e-2 and max-abs <= 0.5 | bf16 logits of magnitude 10-40 have ulps of 0.06-0.25 |
| `probs` | max-abs <= 5e-3 | |
| `grad` (head-mask gradient) | rel <= 5e-2 and cosine >= 0.998 | a bf16 backward through 28 layers |
| top-50 rows | top-1 id equal, top-50 overlap >= 0.8, max abs value difference on shared ids <= 0.5 (logits) / 5e-3 (probs) | |
| floats in the json | abs <= 1e-3 + 2e-2 x abs(ref) | losses, gaze scores, cosines |
| strings | exact | a token-level summary of the first divergence is printed |

## Padding caveat

`head_vecs_padded` and `mask_grad_padded` run left-padded batches; compare.py reports their deviations as WARN, not
FAIL, in tolerance mode. In the old stack a plain forward on a left-padded batch (the nnsight traces of
`get_o_proj_inputs_batch` and `get_next_token_probs_batch`, the direct `mt._model(...)` call of
`forward_with_src_replacement`) passes no `position_ids`, so transformers numbers positions `0..T-1` and the pads of a
shorter row take the first positions: its real tokens sit at RoPE positions shifted by the pad count. HF `generate()`
instead derives positions from the attention mask. The new stack may handle this differently, and both are
defensible, so padded numerics are compared for information only. In the reference, the padded row's steering
vectors differ from the same prompt run alone (`head_vecs_single`) by 1.9 % relative
(`head_vecs_padded.json: _meta_padded_row_vs_alone_rel_err`; this includes the batch-shape change). The
equal-length gradient batch (`mask_grad_equal`) is the gating case for the head-mask path.

## Expected to differ by design

- The two padded cases (above).
- A port that changes how a greedy rollout decodes. `greedy_continue` (`src/goals/goalscope.py:32-72`) re-runs the
  whole sequence each step with no KV cache; `greedy` (`src/nrgs/functional.py:243-267`), `headpatched_generate` and
  the ablation generators use the cache; `head_additive_generate` goes through HF `generate()`, which applies the
  checkpoint's `generation_config` (repetition penalty 1.1 and both EOS ids 151645 / 151643 for this model) while the
  manual loops take a plain argmax and stop on `tokenizer.eos_token_id` only. Switching one decoder for another changes
  the numerics and, near ties, the text: every rollout (`gs_additive`, `gs_replace`, `ov_reader`, `temporal`,
  `ov_boost`) and every additive read-out would move. Keep the decoders as they are in the port and change them in a
  separate PR that regenerates the reference.
- The `*_default` read-out fields of `kv_transfer` are `null`: `clean_runs` / `run_patched` read out through
  `goalscope_readout`'s defaults, steer layers 35-45 and 128 tokens (`src/nrgs/functional.py:214-215`), which do not
  exist on a 28-layer model, so the case runs them with `readout=False` and calls `goalscope_readout` itself with the
  7B layers. The same 32B constants sit in `OVReader.apply_probe` (`src/goals/goalscope.py:391`) and in the OV-boost
  script's `GS` (`scripts/evaluation/obstacle_persistence_ov_boost_eval.py:61`).

## Old-code behavior the port must keep

file:line refers to `clean` at bbc4118.

- **Step counting differs by strategy.** additive_norm (baukit, `src/functional.py:121-150`) counts DECODE steps:
  the prefill is always patched and decode step k is patched while `k <= n_steer_tokens`, so `n_steer_tokens=2`
  patches 3 forward passes. replace (`headpatched_generate`, `src/goals/goalscope.py:138`) counts forward passes
  with the prefill as the first, so `n_steer_tokens=2` patches 2. `apply_goalscope`'s docstring
  (`src/goals/goalscope.py:480-483`) describes the second for both. Reported runs use `"all"`; the `band448_nsteer2`
  arms of `gs_additive` / `gs_replace` pin both (on this model n = 1, 2, 3, 4 all give different read-outs).
- **The baukit step sentinel is chosen by string sort** (`src/functional.py:124`): with steer layers that cross a
  digit boundary (e.g. 8-11) the sentinel is layer 10, so layers 8 and 9 see a stale counter and get one extra
  patched decode step under an integer `n_steer_tokens`. Not pinned (on the 7B the effect on the text is too small to
  test reliably); it matters only for a non-`"all"` `n_steer_tokens` with such a band.
- **additive_norm math and dtypes** (`src/goals/goalscope.py:217-218`, `src/functional.py:66-83`): the direction is
  normalized in fp32 on CPU, cast to bf16 at the hook; `||h||` is computed in fp32 and cast to bf16 before
  `h + scale * patch` in bf16; the edit is in place on each steer layer's OUTPUT at position -1, on the prefill and
  every decode step. The prefill is recognised by `seq_len > 1`.
- **replace splice** (`src/goals/head_loc_attn.py:750-800`): an o_proj forward PRE-hook; interpolation
  `alpha*m*src + (1-m)*cascade` in fp32, cast back to bf16; the mask lives on CPU and is moved per layer; patched
  K/V of later layers stay in the KV cache for the patched position.
- **KV transfer** (`src/nrgs/functional.py:289-427`): the attention modules' `forward` is replaced
  (`mod.forward = types.MethodType(...)`, line 423) for the duration of the context. Source keys are the pre-RoPE
  `k_proj` outputs, RoPE'd at the DESTINATION positions; only the last query row of each forward (the pivot, then each
  new token) reads the source span; the KV cache keeps the clean k/v. `v_scale` multiplies the transplanted values.
- **Sink / blind ablations** (`scripts/evaluation/ablation_common.py:119-238`) add a mask to the input of
  `.source.attention_interface_0.source.nn_functional_softmax_0` (nnsight `.source` on transformers' eager attention).
  Rows with absolute position `>= q_start` are hit (the last prompt row on the prefill, every decode step); generation
  stops after a token whose text ends in `"\n"` (that token is kept) or at EOS (dropped).
- **Q transfer / GSR** (`src/nrgs/functional.py:468-663`): `whitelist_head_attn` edits the 4-D additive
  `attention_mask` passed to each patched attention module (eager attention only), expanding it to all heads.
- **Module globals.** `src.nrgs.functional.setup()` binds `src.nrgs.functional.mt` (line 66); every helper called with
  `ctx=None` uses it; `q_transfer_eval.py` calls `run_q_transfer_intervention` without `ctx`, so every helper inside
  it (`materialize`, `token_span`, `greedy`, `cache_src_proj`, line 550 onward) reads that global.
  `scripts/nrgs/kv_transfer_eval.py:244` sets `src.nrgs.data.CPP_NOTE_IN_SYSTEM = True` for the whole process, which
  moves the C++ note into the system prompt of every later `materialize` call; `run.py` sets it only inside
  `kv_transfer` and restores it.
- **Global grad / parameter state.** The eval scripts call `torch.set_grad_enabled(False)` process-wide
  (e.g. `scripts/evaluation/eval_goalscope.py:522`); `train_head_mask` sets `requires_grad=False` on every model
  parameter and never restores it (`src/goals/head_loc_attn.py:1048`).
- **nnsight access order.** `get_o_proj_inputs_batch` sorts its layers (`src/goals/head_loc_attn.py:177`) because an
  nnsight 0.7 trace raises when a module is requested after a later one; `get_hs` does not sort
  (`src/functional.py:296-305`), so its callers must list locations in execution order (`hs_probs` does).
- **Accumulation order.** `collect_head_steering_vecs` multiplies each head's slice by its `W_O` columns in fp32 on the
  layer's device and sums on CPU in the order the heads appear in `selected_heads` (`src/goals/head_loc_attn.py:265-277`).
  Another order changes the fp32 bits (within tolerance) and, through the bf16 cast at the hook, can move a near-tied
  read-out token.
- **Tokenizer and padding.** `ModelandTokenizer` sets left padding and `pad_token = eos_token` (`src/models.py:47-49`);
  every negative index (`[:, -1]`) assumes left padding.
- **Two greedy decoders and HF generate** (see "Expected to differ by design").
