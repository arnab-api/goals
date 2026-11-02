"""The fixed parity case set: which inputs each case runs on, and the 7B-scaled constants.

Every input is drawn deterministically from data shipped in this repo:
``data/goalscope_eval.json`` (GoalScope / OVReader / ablation corpus), the inline banks of
``src/nrgs/data.py`` (KV / Q transfer, temporal persistence), ``scripts/evaluation/obstacle_persistence_prefills.py``
(OV boost) and ``data/head_loc/`` (head-mask gradient). Nothing here samples at run time except through the
repo's own seeded samplers, called with the same seeds the eval scripts use.

``run.py`` reads these constants; changing any of them invalidates the committed reference outputs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
HEADS_JSON = HERE / "heads_qwen25coder7b.json"

# ── head set ────────────────────────────────────────────────────────────────────────────────────
# The 7B has no localized goal heads. Parity needs a FIXED head set, not a meaningful one.
HEADS_SEED = 0
HEADS_N = 60
HEADS_LAYERS = range(10, 26)  # layers 10..25 of 28
HEADS_PER_LAYER = 28


def make_head_set() -> list[list[int]]:
    """60 (layer, head) pairs drawn with ``random.Random(0)`` from layers 10-25 x heads 0-27, sorted.
    ``heads_qwen25coder7b.json`` is the output of this function."""
    pool = [(l, h) for l in HEADS_LAYERS for h in range(HEADS_PER_LAYER)]
    picked = random.Random(HEADS_SEED).sample(pool, HEADS_N)
    return [[l, h] for l, h in sorted(picked)]


# A second, STRESS head set: every head of layers 10-25 (448 heads). The 60 random heads barely move the 7B's
# greedy output under most interventions (sink, OV boost, KV / Q transfer leave the text unchanged), which would let a
# port that drops or misplaces an intervention pass the string checks. With all 448 heads each intervention changes the
# greedy text, so the intervention cases run both sets ("arms").
STRESS_LAYERS = range(10, 26)


def stress_head_set() -> list[tuple[int, int]]:
    return [(l, h) for l in STRESS_LAYERS for h in range(HEADS_PER_LAYER)]


# ── budgets (kept small so the whole run takes minutes) ────────────────────────────────────────
GEN_TOKENS = 32  # every greedy rollout / continuation
READOUT_TOKENS = 32  # every GoalScope / OVReader read-out

# ── 7B-scaled layer constants (Coder-32B has 64 layers, the 7B has 28) ───────────────────────────
STEER_LAYER_RANGE = (15, 21)  # half-open, layers 15..20; the 32B uses (35, 46)
OV_PROBE_LAYERS = [10, 21]  # OVReader probe layers (residual after these layers)
ABLATION_FROM_LAYER = 9  # sufficiency blinds non-goal heads in layers >= this; 32B: 20 of 64 -> 9 of 28
HEADMASK_INIT = 0.5  # HeadMask value on every (layer, head) for the gradient case
HEADMASK_LAMBDA = 20.0  # the lam20 sparsity coefficient of scripts/train_head_mask.py

# ── GoalScope (scripts/evaluation/eval_goalscope.py) ───────────────────────────────────────────
GS_ADDITIVE_SPECS = ["int_python_000", "list[str]_python_000"]  # additive_norm, alpha 0.1
GS_ADDITIVE_ALPHA = 0.1
GS_CAUSALITY_SPEC = "int_python_000"  # same spec, the C++-stub (causality) probe
GS_REPLACE_SPEC = "str_python_000"  # replace strategy, alpha 1 (the model-axis runs)
GS_REPLACE_ALPHA = 1.0
# Step-counting arms (448-head set, first additive spec / the replace spec): every reported run uses
# n_steer_tokens="all"; these pin what an integer n_steer_tokens does today, which differs by strategy
# (additive_norm patches the prefill + 2 decode steps, replace patches 2 forward passes). On this model the read-outs
# for n_steer_tokens = 1, 2, 3, 4 all differ, so an off-by-one in a port changes the text.
GS_NSTEER = 2

# ── head-vector extraction (collect_head_steering_vecs / get_o_proj_inputs_batch) ───────────────
# Two prompts of DIFFERENT token lengths (123 and 119 tokens: the second row gets 4 left pads), and the second
# (shorter) one alone, so head_vecs_single is the unpadded run of the padded row.
HEADVEC_SPECS = ["int_python_001", "list[int]_python_001"]
HEADVEC_N_LAST = 5  # read the last 5 real tokens of each prompt (= the GoalScope pivots of a prompt)

# ── OVReader (scripts/evaluation/eval_ov_reader.py) ────────────────────────────────────────────
OV_SPEC = "int_python_000"  # probe at the `eq` pivot of this spec's rollout
OV_ALPHA = 0.15

# ── attention (src/attention.get_attention_matrices) ───────────────────────────────────────────
ATTN_SPEC = "int_python_002"
ATTN_SAVE_LAYERS = [0, 18]  # full (H, T, T) matrices saved for these two layers only

# ── ablations (scripts/evaluation/ablation_common.py) ──────────────────────────────────────────
ABL_BLIND_SPEC = "list[int]_python_000"  # sufficiency: generate_blocking_span_cached
ABL_SINK_SPEC = "str_python_001"  # necessity: sink_generate_cached
ABL_CONTROL_SEED = 123  # the ablation scripts' --seed default

# ── NRGS (scripts/nrgs/kv_transfer_eval.py, scripts/nrgs/q_transfer_eval.py) ──────────────────
KV_DATATYPE, KV_LANGS = "int", ("python", "python")  # py-py
KV_SEED, KV_N_SAMPLES, KV_INSTANCE = 1123, 32, 0  # the script's defaults; instance 0 of the draw
KV_V_SCALES = [1.0, 4.0]
KV_SAVE_LAYERS = 2  # cached source k/v saved for the first 2 patched layers
Q_DATATYPE, Q_VARIANT, Q_LANGS = "int", "order_diff", ("python", "python")
Q_SEED, Q_N_SAMPLES, Q_INSTANCE = 0, 32, 0
Q_ALPHA = 2.0  # the script's --alphas default (only used by the patched readout, which is off)

# ── OV boost (scripts/evaluation/obstacle_persistence_ov_boost_eval.py) ───────────────────────
OVB_CANDIDATE = 0  # index into obstacle_persistence_ov_boost_eval.candidates()
OVB_ALPHA = 2.0  # BOOST_ALPHA

# ── temporal persistence (src/goals/temporal_persistence.py) ──────────────────────────────────
TP_DATATYPE, TP_LANG = "int", "python"
TP_SEED, TP_N_TRIALS, TP_INSTANCE, TP_TARGET = 1123, 256, 0, 1  # build_records defaults; probe position t=1

# ── get_hs / get_next_token_probs ─────────────────────────────────────────────────────────────
HS_SPEC = "list[str]_python_001"
HS_LAYERS = [0, 7, 14, 21, 27]
HS_POSITIONS = [-1, -2]

# ── head-mask gradient (src/goals/head_loc_attn.forward_with_src_replacement) ──────────────────
# The 7B has no mined pairs file; train_head_mask.py falls back to the Coder-32B pool, so do we.
MASK_PAIRS_MODEL = "Qwen2.5-Coder-32B-Instruct"
MASK_SCAN = 400  # search the first 400 pivot instances
MASK_INCLUDE_DOCSTRING = True  # scripts/train_head_mask.py passes include_docstring=True


# ── selection helpers ─────────────────────────────────────────────────────────────────────────
def load_heads() -> list[tuple[int, int]]:
    return [(int(l), int(h)) for l, h in json.loads(HEADS_JSON.read_text())]


def spec_index() -> dict:
    """``{spec_id: QTransferInstance}`` over the locked GoalScope corpus, rebuilt the way every eval does."""
    from scripts.evaluation.eval_goalscope import ALL_DATATYPES, DEFAULT_DATASET, load_locked_datapoints

    pairs, _ = load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)
    return dict(pairs)


def kv_instance():
    import src.nrgs.kv_transfer as PP

    insts, seed = PP.sample_multi_instances(KV_DATATYPE, *KV_LANGS, n_samples=KV_N_SAMPLES, seed=KV_SEED)
    return insts[KV_INSTANCE], seed


def q_instance():
    """Instance ``Q_INSTANCE`` of the q-transfer cell, with the cell seed computed as ``run_config`` does."""
    import src.nrgs.data as DATA

    sl, dl = Q_LANGS
    tag = f"{Q_DATATYPE.replace('[', '(').replace(']', ')')}__{sl}-{dl}"
    cfg_seed = (Q_SEED + int(hashlib.md5(f"{Q_VARIANT}/{tag}".encode()).hexdigest()[:8], 16)) % (2**32)
    setups = DATA.diff_trigger_setups(Q_DATATYPE, variant=Q_VARIANT, n_samples=Q_N_SAMPLES, seed=cfg_seed,
                                      src_lang=sl, dst_lang=dl)
    return setups[Q_INSTANCE], cfg_seed


def tp_instance():
    import src.nrgs.data as D
    from src.goals import temporal_persistence as TP

    setups = D.multi_predicate_setups(TP_DATATYPE, n_conditions=list(TP.N_COND), n_samples=TP_N_TRIALS,
                                      seed=TP_SEED, naming="letters", src_lang=TP_LANG, dst_lang=TP_LANG)
    return setups[TP_INSTANCE]


def mask_instances(mt) -> dict:
    """Pick the head-mask-gradient batches from the Coder-32B pivot pool, deterministically:

    * ``equal``: the first pair (i < j, ordered by j then i) with DIFFERENT source and destination prompts
      whose source (C++), destination (Python) AND target (Python) prompts tokenize to the same length, so
      no batched forward in the case is padded;
    * ``padded``: instance 0 and the first later j whose source AND destination lengths both differ from
      instance 0's, so the batched source, destination and target forwards are all left-padded.

    Returns ``{"equal": [i, j], "padded": [i, j]}`` (indices into ``HeadLocDataset.pivot_instances``) plus
    the prompts, so the record shows exactly what ran."""
    from src.goals.head_loc_data import HeadLocDataset
    from src.goals.head_loc_pivot import build_cpp_prompt, build_python_prompt

    ds = HeadLocDataset.load(model_name=MASK_PAIRS_MODEL)
    pis = ds.pivot_instances[:MASK_SCAN]
    rows = []
    for pair, pi in pis:
        src_s = ds.get_sample(pair.source_predicate)
        dst_s = ds.get_sample(pair.destination_predicate)
        src = build_cpp_prompt(mt, src_s, include_docstring=MASK_INCLUDE_DOCSTRING)
        dst = build_python_prompt(mt, dst_s, pivot_prefix=pi.pivot_prefix, include_docstring=MASK_INCLUDE_DOCSTRING)
        targ = build_python_prompt(mt, src_s, pivot_prefix=pi.pivot_prefix, include_docstring=MASK_INCLUDE_DOCSTRING)
        n = lambda s: len(mt.tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
        rows.append(dict(src=src, dst=dst, targ=targ, lens=(n(src), n(dst), n(targ)),
                         pair=f"{pair.source_predicate}->{pair.destination_predicate}", prefix=pi.pivot_prefix))
    def _same_len(i, j):
        a, b = rows[i], rows[j]
        return a["lens"] == b["lens"] and a["src"] != b["src"] and a["dst"] != b["dst"]

    equal = next(([i, j] for j in range(len(rows)) for i in range(j) if _same_len(i, j)), None)
    padded = next(([0, j] for j in range(1, len(rows))
                   if rows[0]["lens"][0] != rows[j]["lens"][0] and rows[0]["lens"][1] != rows[j]["lens"][1]), None)
    return dict(equal=equal, padded=padded, rows=rows)


if __name__ == "__main__":
    # Regenerate (or check) the committed head set.
    import sys

    heads = make_head_set()
    if "--write" in sys.argv:
        HEADS_JSON.write_text(json.dumps(heads) + "\n")
        print(f"wrote {HEADS_JSON} ({len(heads)} heads)")
    else:
        same = HEADS_JSON.exists() and json.loads(HEADS_JSON.read_text()) == heads
        print(f"{HEADS_JSON.name}: {'matches' if same else 'DIFFERS FROM'} make_head_set()")
        sys.exit(0 if same else 1)
