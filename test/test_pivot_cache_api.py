"""Smoke-test for the updated PivotCache API.

Verifies:
1. cache_pivot_data populates src_sample / dest_sample
2. augment_pivot_caches works without pivot_instances / ds args
3. Augmented caches carry updated samples and are self-consistent
4. train_head_mask works without augment_instances / augment_ds

Run with:
    conda run -n goals python test/test_pivot_cache_api.py
"""

import sys
import random
import torch

sys.path.append(".")

from src.models import ModelandTokenizer
from src.goals.head_loc_data import HeadLocDataset
from src.goals.head_loc_attn import (
    augment_pivot_caches,
    cache_pivot_data,
    train_head_mask,
)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_KEY    = "Qwen/Qwen2.5-Coder-14B-Instruct"
MODEL_NAME   = "Qwen2.5-Coder-32B-Instruct"  # dataset exists only for 32B; model is 14B for speed
N_INSTANCES  = 10   # small slice for fast smoke-test
LAYERS       = list(range(6, 14))
BATCH_SIZE   = 4
N_EPOCHS     = 2

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading dataset ...")
ds = HeadLocDataset.load(model_name=MODEL_NAME)
pivot_instances = ds.pivot_instances[:N_INSTANCES]
assert len(pivot_instances) > 0, "No pivot instances found — run mine_pivot_instances.py first."

print(f"Loading model {MODEL_KEY} ...")
mt = ModelandTokenizer(
    model_key=MODEL_KEY,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

# ── Test 1: cache_pivot_data populates src_sample / dest_sample ──────────────
print("\n[1] cache_pivot_data — eager (precompute=True)")
pivot_caches_eager = cache_pivot_data(
    mt, pivot_instances, ds, layers=LAYERS,
    precompute=True, cache_batch_size=4,
)
c = pivot_caches_eager[0]
assert c.src_sample is not None, "src_sample not populated"
assert c.dest_sample is not None, "dest_sample not populated"
assert c.src_head_vals is not None, "src_head_vals missing in eager mode"
assert c.per_head_deltas is None, "per_head_deltas should always be None (new path)"
assert c.src_prompt is None, "src_prompt should be None in eager mode"
n_embd = mt.n_embd
assert c.src_head_vals.shape == (len(LAYERS), n_embd), (
    f"src_head_vals shape {c.src_head_vals.shape} != ({len(LAYERS)}, {n_embd})"
)
print(f"  src_sample.predicate_name  = {c.src_sample.predicate_name}")
print(f"  dest_sample.predicate_name = {c.dest_sample.predicate_name}")
print(f"  src_head_vals.shape        = {c.src_head_vals.shape}")
print("  PASS")

print("\n[2] cache_pivot_data — lazy (precompute=False)")
pivot_caches_lazy = cache_pivot_data(
    mt, pivot_instances, ds, layers=LAYERS,
    precompute=False, cache_batch_size=4,
)
c = pivot_caches_lazy[0]
assert c.src_sample is not None
assert c.dest_sample is not None
assert c.src_head_vals is None, "src_head_vals should be None in lazy mode"
assert c.per_head_deltas is None, "per_head_deltas should always be None (new path)"
assert c.src_prompt is not None, "src_prompt should be set in lazy mode"
print(f"  src_sample.predicate_name  = {c.src_sample.predicate_name}")
print(f"  src_prompt[:60]            = {c.src_prompt[:60]!r}")
print("  PASS")

# ── Test 2: augment_pivot_caches — no pivot_instances / ds args ──────────────
print("\n[3] augment_pivot_caches — no pivot_instances / ds")
rng = random.Random(42)
aug_caches = augment_pivot_caches(
    mt, pivot_caches_lazy, cache_batch_size=4, rng=rng,
)
assert len(aug_caches) == len(pivot_caches_lazy), "Length mismatch"
assert all(c.src_head_vals is None for c in aug_caches), "Augmented caches should be lazy"
assert all(c.per_head_deltas is None for c in aug_caches), "per_head_deltas should always be None"
assert all(c.src_prompt is not None for c in aug_caches), "src_prompt must be set"
assert all(c.src_sample is not None for c in aug_caches), "src_sample must be set"
assert all(c.dest_sample is not None for c in aug_caches), "dest_sample must be set"
print(f"  {len(aug_caches)} augmented caches, all lazy: PASS")

# Show a few augmented vs original dest_prompt diffs
for i in range(min(3, len(aug_caches))):
    orig = pivot_caches_lazy[i].dest_prompt[-80:].replace("\n", "↵")
    aug  = aug_caches[i].dest_prompt[-80:].replace("\n", "↵")
    changed = "CHANGED" if orig != aug else "same"
    print(f"  [{i}] dest_prompt tail ({changed})")
    if orig != aug:
        print(f"    orig: {orig!r}")
        print(f"    aug:  {aug!r}")

# ── Test 3: augmented caches can themselves be re-augmented ──────────────────
print("\n[4] re-augmenting augmented caches (self-contained)")
rng2 = random.Random(99)
re_aug = augment_pivot_caches(mt, aug_caches, cache_batch_size=4, rng=rng2)
assert len(re_aug) == len(aug_caches)
print("  PASS")

# ── Test 4: train_head_mask — no augment_instances / augment_ds ──────────────
print("\n[5] train_head_mask — augment=False (baseline)")
mask, history = train_head_mask(
    mt, pivot_caches_lazy, layers=LAYERS,
    n_epochs=N_EPOCHS, lr=1e-2, lambda_sparse=1.0,
    batch_size=BATCH_SIZE, checkpoint_every=N_EPOCHS,
)
assert len(history.loss_causal) == N_EPOCHS
assert len(history.val_recall)  == 1, f"Expected 1 val eval, got {len(history.val_recall)}"
print(f"  train loss (last): {history.loss_causal[-1]:.4f}")
print(f"  val recall@1..5:   {history.val_recall[-1]}")
print("  PASS")

print("\n[6] train_head_mask — augment=True")
mask_aug, history_aug = train_head_mask(
    mt, pivot_caches_lazy, layers=LAYERS,
    n_epochs=N_EPOCHS, lr=1e-2, lambda_sparse=1.0,
    batch_size=BATCH_SIZE, checkpoint_every=N_EPOCHS,
    augment=True, augment_seed=42, augment_cache_batch_size=4,
)
assert len(history_aug.loss_causal) == N_EPOCHS
assert len(history_aug.val_recall)  == 1
print(f"  train loss (last): {history_aug.loss_causal[-1]:.4f}")
print(f"  val recall@1..5:   {history_aug.val_recall[-1]}")
print("  PASS")

print("\n=== All tests passed ===")
