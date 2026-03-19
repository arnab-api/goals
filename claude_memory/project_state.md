# Project State — goals (as of 2026-03-19)

## Where We Are

Two parallel experimental tracks. **Track 2 (head localization) is the active focus.**

### Track 1 — Prefill Probing (complete / paused)
- Methodology: MCTS over code completions → contrastive pairs (green vs red/orange) → linear goal directions → causality via activation steering
- Scripts: `goal_dir.py`, `test_causality.py`, `multipred_mcts.py`, `multipred_probe.py`
- Status: experiments done; steering causality partially validated; known issue with nnsight `UnboundLocalError` at high coefficients (logged to `errors.log`)
- Remaining TODO: shared steering coeff selection strategy across prefills

### Track 2 — Attention Head Localization (active)
- Methodology: cross-language (C++ → Python) interchange at pivot points; learn sparse head mask minimizing KL to target distribution
- Data: 1480 pivot instances mined for Qwen2.5-Coder-32B-Instruct, stored in `data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json`
- Pivot mining: fully complete and committed
- Mask training script (`scripts/train_head_mask.py`): written and committed, **smoke-test pending** — has not yet been run end-to-end on the full dataset

## Pipeline Status (Track 2)

```
[DONE]  data/head_loc/predicates/          25 predicate JSONs (Python + C++)
[DONE]  data/head_loc/global_pairs.json    oracle-verified directed pairs
[DONE]  data/head_loc/pairs/Qwen2.5-Coder-32B-Instruct.json   1480 pivot instances
[WIP]   scripts/train_head_mask.py         written, smoke-test pending
[TODO]  results/localization/.../head_mask.pt   not yet generated
[TODO]  head analysis (which heads selected, do they generalize?)
```

## Next Steps

1. Run smoke-test of `train_head_mask.py`:
   ```bash
   conda run -n goals python scripts/train_head_mask.py \
       --model-key Qwen/Qwen2.5-Coder-32B-Instruct \
       --max-instances 100 --n-epochs 1
   ```
2. If smoke-test passes, run the full training:
   ```bash
   conda run -n goals python scripts/train_head_mask.py \
       --model-key Qwen/Qwen2.5-Coder-32B-Instruct \
       --layers-start 6 --layers-end 40 \
       --n-epochs 100 --lambda-sparse 20 --batch-size 16
   ```
3. Analyze selected heads in `06_loc_head.ipynb` — use `patched_generate` to qualitatively verify that patching those specific heads transfers the predicate intent.
4. Generalization test: does the mask trained on the full dataset work for held-out predicate pairs?

## Key Technical Decisions Made

- **Delta precomputation**: `precompute_deltas=True` is the default — fine for N=1480 (~42 GB CPU RAM for layers 6–40 on Qwen-32B). Machine has 1.5 TB RAM.
- **Mask parameterization**: raw `nn.Parameter` initialized to ones, box-clamped to [0,1] after each step. No sigmoid. Sparsity: `λ * mean(mask)` (scale-invariant). Threshold 0.5 for head selection.
- **KL direction**: `KL(P_patched ∥ P_target)` with `reduction="batchmean"`.
- **Combined batch padding**: src and dest prompts concatenated into one batch for o_proj extraction — ensures identical sequence lengths → no positional leakage in the delta.
- **Pivot instance format**: stored only the diverging prefix (`pivot_prefix`, `dest_top_token`, `target_top_token`) — predicate identity lives in the containing `PredicatePair`.
- **`--max-instances` flag**: added to `train_head_mask.py` for cheap smoke-tests without loading all data.

## Model / Hardware

- Primary model: `Qwen/Qwen2.5-Coder-32B-Instruct` (hidden=5120, heads=40, layers=64)
- Target layers: 6–40 (34 layers) — empirically chosen range
- Training machine: ~1.51 TB CPU RAM, multiple A100 GPUs
- Project is being migrated to a new machine — ensure `env.yml` is configured on arrival

## Commit History (recent)

```
7314e10  [WIP] add train_head_mask.py script (still testing end-to-end)
a459061  head_loc: unified pivot pipeline + scaled predicate dataset
1f009f6  add docstrings + trim tricky unit tests from head_loc predicates
873c448  [WIP] Case 3 oracle pivot mining + learned sparse head mask localization
8c2fb81  new dataset > head localization
```

## Quirks to Remember

- `highlight=False` required on any Rich `Console` printing tree nodes — otherwise ReprHighlighter colors token strings green
- Cached MCTSResult JSONs from before the LCA pairing change won't have `pos_path_tokens`/`neg_path_tokens` — re-run MCTS if using `--unique_positive_sampling` on old cache
- nnsight `UnboundLocalError` (Track 1): occurs at coeff > 0.1, layers 40-50; root cause is backend execution failure before `gen_out` is assigned; caught and logged
- `build_python_prompt` and `build_cpp_prompt` append a leading space before the pivot prefix — don't double-add a space when constructing prompts manually
