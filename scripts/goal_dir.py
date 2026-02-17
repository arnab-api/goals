"""
Goal direction extraction script.

Processes all samples from IncompleteCodeDataset:
  1. Runs MCTS per prefill (or reuses cached results)
  2. Collects contrastive pairs with train/test split
  3. Extracts latents (hidden states at last token)
  4. Trains mean-diff and contrastive goal directions per layer
  5. Saves directions, metrics, and configs

Usage:
    python -m scripts.goal_dir --model_name Qwen/Qwen2.5-Coder-32B-Instruct
    python -m scripts.goal_dir --model_name Qwen/Qwen2.5-Coder-14B-Instruct --sample_indices 0 1 5
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict

# Must be set before importing torch so CUDA sees only the requested devices.
# Overridden by --cuda_devices CLI arg in main().
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import torch  # noqa: E402

from src.functional import free_gpu_cache
from src.goals.data import IncompleteCodeDataset
from src.goals.eval import evaluate_mcts_tree
from src.goals.functional import (
    apply_chat_template,
    get_last_token_latents_batch,
    supports_chat_template,
)
from src.goals.probe import sample_multidepth_pairs
from src.goals.sampling import MCTSResult, MCTSSearch
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Sampling config defaults (edit here; not exposed as CLI args)
# ──────────────────────────────────────────────────────────────────────
SAMPLING_CONFIG = dict(
    min_depth=1,
    max_depth=10,
    min_green_confidence=0.5,
    max_red_in_positive=0.3,
    min_red_confidence=0.5,
    max_green_in_negative=0.1,
    max_uncertain=0.5,
)


# ══════════════════════════════════════════════════════════════════════
# Contrastive direction training
# ══════════════════════════════════════════════════════════════════════
@torch.enable_grad()
def learn_contrastive_direction(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    l1_lambda: float = 0.0,
    l2_lambda: float = 1e-3,
    margin: float = 1.0,
    lr: float = 1e-2,
    n_steps: int = 1000,
    patience: int = 50,
) -> tuple[torch.Tensor, dict]:
    """
    Learn a goal direction via the contrastive objective.

    Minimizes:  lambda1*||v||_1 + lambda2*||v||_2^2 + (1/N) sum_i max(0, margin - v.(z+_i - z-_i))
    """
    device = z_pos.device
    hidden_dim = z_pos.shape[1]

    delta = z_pos - z_neg  # (n_pairs, hidden_dim)

    v = torch.randn(hidden_dim, device=device, dtype=z_pos.dtype) * 0.01
    v.requires_grad_(True)

    optimizer = torch.optim.Adam([v], lr=lr)

    best_loss = float("inf")
    patience_counter = 0
    final_step = n_steps

    for step in range(n_steps):
        optimizer.zero_grad()

        scores = delta @ v
        hinge_loss = torch.clamp(margin - scores, min=0).mean()

        reg_loss = torch.zeros(1, device=device, dtype=z_pos.dtype)
        if l1_lambda > 0:
            reg_loss = reg_loss + l1_lambda * v.abs().sum()
        if l2_lambda > 0:
            reg_loss = reg_loss + l2_lambda * (v @ v)

        loss = hinge_loss + reg_loss.squeeze()

        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        if curr_loss < best_loss - 1e-6:
            best_loss = curr_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            final_step = step + 1
            break

    with torch.no_grad():
        v_final = v.detach()
        v_norm = v_final.norm()
        v_normalized = v_final / (v_norm + 1e-8)

        scores_final = delta @ v_normalized
        constraint_sat = (scores_final > 0).float().mean().item()
        sparsity = (v_normalized.abs() < 1e-2).float().mean().item()

    return v_normalized, {
        "final_loss": best_loss,
        "steps": final_step,
        "constraint_satisfaction": constraint_sat,
        "sparsity": sparsity,
        "v_norm_pre_normalize": v_norm.item(),
    }


# ══════════════════════════════════════════════════════════════════════
# MCTS: run or reuse
# ══════════════════════════════════════════════════════════════════════
def _mcts_config_matches(existing: dict, current: dict) -> tuple[bool, list[str]]:
    """Compare two MCTS config dicts. Returns (match, list of diff descriptions)."""
    diffs = []
    all_keys = set(existing) | set(current)
    for k in sorted(all_keys):
        v_old = existing.get(k)
        v_new = current.get(k)
        if v_old != v_new:
            diffs.append(f"  {k}: saved={v_old} vs current={v_new}")
    return len(diffs) == 0, diffs


def run_or_load_mcts(
    mt: ModelandTokenizer,
    sample,
    prefills: list[tuple[str, str]],
    prompts: list[str],
    mcts_dir: str,
    args,
) -> list[MCTSResult]:
    """
    Run MCTS for each prefill, or load cached results if available.
    Warns and prompts the user if cached configs differ from current args.
    """
    current_mcts_config = dict(
        iterations=args.mcts_iterations,
        max_depth=args.mcts_max_depth,
        top_k=args.mcts_top_k,
        prob_threshold=args.mcts_prob_threshold,
        depth_first=args.depth_first,
    )

    model_short_name = mt.name.split("/")[-1]
    function_name = sample.function_name or "func"

    # Check for existing results
    existing_jsons = {}
    if os.path.isdir(mcts_dir):
        for fname in os.listdir(mcts_dir):
            if fname.endswith(".json"):
                existing_jsons[fname] = os.path.join(mcts_dir, fname)

    # If we have cached results, check config compatibility
    reuse_cached = False
    if existing_jsons:
        # Load the first one to check config
        first_path = next(iter(existing_jsons.values()))
        first_result = MCTSResult.load(first_path)
        matches, diffs = _mcts_config_matches(
            first_result.mcts_config, current_mcts_config
        )

        if matches:
            logger.info(
                f"Found {len(existing_jsons)} cached MCTS results with matching config. Reusing."
            )
            reuse_cached = True
        else:
            print("\nWARNING: Found cached MCTS results with DIFFERENT config:")
            for d in diffs:
                print(d)
            if args.force_rerun_mcts:
                print("--force_rerun_mcts is set. Re-running MCTS.")
                reuse_cached = False
            elif not args.ask_override_permission:
                print(
                    "--ask_override_permission=False. Overriding with current params."
                )
                reuse_cached = False
            else:
                answer = (
                    input("Reuse cached results anyway? [y/N/q(uit)]: ").strip().lower()
                )
                if answer == "q":
                    print("Exiting.")
                    sys.exit(0)
                reuse_cached = answer == "y"

    # Load cached results
    if reuse_cached:
        all_results = []
        for fname in sorted(existing_jsons):
            all_results.append(MCTSResult.load(existing_jsons[fname]))
        print(
            f"Loaded {len(all_results)} cached MCTS results "
            f"(total pairs: {sum(r.total_pairs for r in all_results)})"
        )
        return all_results

    # Run MCTS from scratch
    os.makedirs(mcts_dir, exist_ok=True)
    all_results: list[MCTSResult] = []

    for prefill_idx, (prefill_val, prefill_type) in enumerate(prefills):
        print(f"\n{'=' * 60}")
        print(
            f"Prefill {prefill_idx}/{len(prefills)}: ({prefill_type}) {prefill_val!r}"
        )
        print(f"{'=' * 60}")

        prompt = prompts[prefill_idx]
        prefix = sample.code + prefill_val

        # 1) MCTS search
        search = MCTSSearch(
            mt=mt,
            prompt=prompt,
            max_depth=args.mcts_max_depth,
            top_k=args.mcts_top_k,
            prob_threshold=args.mcts_prob_threshold,
        )
        root = search.run(
            n_iterations=args.mcts_iterations, depth_first=args.depth_first
        )

        # 2) Evaluate tree
        counts = evaluate_mcts_tree(
            root=search.root,
            prefix=prefix,
            unit_tests=sample.unit_tests,
            function_name=function_name,
            terminal_only=False,
        )
        print(f"  Eval counts: {counts}")

        # 3) Sample contrastive pairs
        pairs_by_depth = sample_multidepth_pairs(root=search.root, **SAMPLING_CONFIG)
        n_pairs = sum(len(p) for p in pairs_by_depth.values())
        print(f"  Contrastive pairs: {n_pairs}")

        # 4) Serialize contrastive pairs (MCTSNode -> text paths)
        serialized_pairs = {}
        for depth, pairs in pairs_by_depth.items():
            serialized_pairs[str(depth)] = [
                {
                    "pos_path": pair.positive.get_path_text(),
                    "neg_path": pair.negative.get_path_text(),
                    "pos_probs": {
                        "green": pair.pos_probs.green,
                        "red": pair.pos_probs.red,
                        "uncertain": pair.pos_probs.uncertain,
                    },
                    "neg_probs": {
                        "green": pair.neg_probs.green,
                        "red": pair.neg_probs.red,
                        "uncertain": pair.neg_probs.uncertain,
                    },
                }
                for pair in pairs
            ]

        # 5) Build result
        result = MCTSResult(
            model_name=model_short_name,
            sample_idx=sample.idx if hasattr(sample, "idx") else 0,
            prefill=prefill_val,
            prefill_type=prefill_type,
            function_name=function_name,
            prompt=prompt,
            prefix=prefix,
            mcts_config=current_mcts_config,
            mcts_stats=search.get_stats(),
            eval_counts=counts,
            contrastive_pairs=serialized_pairs,
            sampling_config=SAMPLING_CONFIG,
        )
        all_results.append(result)

        # 6) Save to disk
        safe_prefill_name = prefill_val.replace(" ", "_").replace("/", "_") or "empty"
        result_path = os.path.join(
            mcts_dir, f"prefill_{prefill_idx}_{safe_prefill_name}.json"
        )
        result.save(result_path)

        # Also save tree HTML
        tree_path = os.path.join(
            mcts_dir, f"prefill_{prefill_idx}_{safe_prefill_name}_tree.html"
        )
        search.save_tree(tree_path, show_colors=True)

    print(f"\nDone! Saved {len(all_results)} results to {mcts_dir}")
    print(
        f"Total contrastive pairs across all prefills: "
        f"{sum(r.total_pairs for r in all_results)}"
    )
    return all_results


# ══════════════════════════════════════════════════════════════════════
# Contrastive pair collection + train/test split
# ══════════════════════════════════════════════════════════════════════
def collect_and_split_pairs(
    all_results: list[MCTSResult],
    unique_positive_sampling: bool,
    n_train_pairs: int | None,
    n_test_pairs: int | None,
) -> tuple[list[dict], list[int], list[int]]:
    """
    Flatten contrastive pairs from all MCTS results, apply tree-level
    train/test split, cap pairs, and filter to only used pairs.

    Returns:
        pair_records: filtered list of pair dicts (only used pairs)
        train_idx: indices into pair_records for training
        test_idx: indices into pair_records for testing
    """
    # --- 1) Collect all contrastive pairs, tagged by prefill_idx ---
    pair_records = []

    for prefill_idx, result in enumerate(all_results):
        for depth_str, pairs in result.contrastive_pairs.items():
            for pair in pairs:
                pair_records.append(
                    {
                        "prompt": result.prompt,
                        "prefix": result.prefix,
                        "pos_path": pair["pos_path"],
                        "neg_path": pair["neg_path"],
                        "depth": int(depth_str),
                        "prefill_idx": prefill_idx,
                        "prefill": result.prefill,
                        "prefill_type": result.prefill_type,
                    }
                )

    n_pairs_before_dedup = len(pair_records)
    print(f"Total contrastive pair records (Cartesian product): {n_pairs_before_dedup}")

    # --- 1b) Optional: unique positive sampling ---
    if unique_positive_sampling:
        random.seed(42)
        pos_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
        for i, rec in enumerate(pair_records):
            pos_groups[(rec["prefill_idx"], rec["pos_path"])].append(i)

        keep_indices = sorted(random.choice(indices) for indices in pos_groups.values())
        pair_records = [pair_records[i] for i in keep_indices]

        n_unique_pos = len(pos_groups)
        print(
            f"Unique positive sampling: {n_pairs_before_dedup} -> {len(pair_records)} pairs "
            f"({n_unique_pos} unique positive nodes, "
            f"{n_pairs_before_dedup / n_unique_pos:.1f}x avg Cartesian expansion)"
        )

    print(f"Pair records after dedup: {len(pair_records)}")

    # Show distribution by depth
    depth_counts = Counter(r["depth"] for r in pair_records)
    for d in sorted(depth_counts):
        print(f"  Depth {d}: {depth_counts[d]} pairs")

    # --- 2) Tree/prefill-level train/test split ---
    pairs_per_prefill: dict[int, int] = Counter(r["prefill_idx"] for r in pair_records)

    prefill_type_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, result in enumerate(all_results):
        if pairs_per_prefill.get(idx, 0) > 0:
            prefill_type_to_indices[result.prefill_type].append(idx)

    random.seed(42)
    test_prefill_set: set[int] = set()
    for ptype, indices in prefill_type_to_indices.items():
        if len(indices) == 0:
            print(f"  WARNING: no prefills with pairs for type '{ptype}', skipping")
            continue
        if len(indices) == 1:
            print(
                f"  WARNING: only 1 prefill with pairs for type '{ptype}' "
                f"(idx={indices[0]}), keeping in train"
            )
            continue
        # Prefer prefills that actually have contrastive pairs for test set
        candidates_with_pairs = [
            idx for idx in indices if pairs_per_prefill.get(idx, 0) > 0
        ]
        if len(candidates_with_pairs) >= 2:
            # Pick from candidates that have pairs, keeping at least 1 for train
            test_prefill_set.add(random.choice(candidates_with_pairs))
        elif len(candidates_with_pairs) == 1:
            # Only 1 prefill with pairs — keep for train, test will be empty for this type
            print(
                f"  WARNING: only 1 prefill with pairs for type '{ptype}' "
                f"(idx={candidates_with_pairs[0]}), keeping in train"
            )
        else:
            test_prefill_set.add(random.choice(indices))

    train_prefill_set = set(range(len(all_results))) - test_prefill_set

    train_idx = [
        i
        for i, rec in enumerate(pair_records)
        if rec["prefill_idx"] in train_prefill_set
    ]
    test_idx = [
        i
        for i, rec in enumerate(pair_records)
        if rec["prefill_idx"] in test_prefill_set
    ]

    # --- 2b) Cap training and test pairs ---
    random.seed(42)
    random.shuffle(train_idx)
    n_train_before_cap = len(train_idx)
    if n_train_pairs is not None and len(train_idx) > n_train_pairs:
        train_idx = train_idx[:n_train_pairs]

    random.seed(42)
    random.shuffle(test_idx)
    n_test_before_cap = len(test_idx)
    if n_test_pairs is not None and len(test_idx) > n_test_pairs:
        test_idx = test_idx[:n_test_pairs]

    # --- 2c) Filter pair_records to only used pairs ---
    used_idx_set = set(train_idx) | set(test_idx)
    used_idx_sorted = sorted(used_idx_set)

    old_to_new = {old: new for new, old in enumerate(used_idx_sorted)}
    pair_records = [pair_records[i] for i in used_idx_sorted]
    train_idx = sorted([old_to_new[i] for i in train_idx])
    test_idx = sorted([old_to_new[i] for i in test_idx])

    # --- Print split summary ---
    print(f"\n{'=' * 60}")
    print("Tree-level train/test split (no latent leakage)")
    if unique_positive_sampling:
        print("  + unique positive sampling (each pos node in 1 pair only)")
    if n_train_pairs is not None:
        print(f"  + train cap: {n_train_pairs} (from {n_train_before_cap} available)")
    else:
        print(f"  + train cap: None (using all {n_train_before_cap} available)")
    if n_test_pairs is not None:
        print(f"  + test cap:  {n_test_pairs} (from {n_test_before_cap} available)")
    else:
        print(f"  + test cap:  None (using all {n_test_before_cap} available)")
    print(f"{'=' * 60}")

    print(f"Train prefills ({len(train_prefill_set)}):")
    for idx in sorted(train_prefill_set):
        r = all_results[idx]
        n = pairs_per_prefill.get(idx, 0)
        print(f"  [{idx:>2}] ({r.prefill_type:>16}) {r.prefill!r:<15} -> {n} pairs")

    print(f"Test prefills ({len(test_prefill_set)}):")
    for idx in sorted(test_prefill_set):
        r = all_results[idx]
        n = pairs_per_prefill.get(idx, 0)
        print(f"  [{idx:>2}] ({r.prefill_type:>16}) {r.prefill!r:<15} -> {n} pairs")

    print(f"\nTrain pairs: {len(train_idx)}, Test pairs: {len(test_idx)}")
    print(
        f"Total pairs for latent extraction: {len(pair_records)} "
        f"(was {n_pairs_before_dedup} before filtering)"
    )
    if len(test_idx) == 0:
        logger.error(
            "0 test pairs! All test prefills likely had no valid contrastive pairs. "
            "This sample will be skipped."
        )

    return pair_records, train_idx, test_idx


# ══════════════════════════════════════════════════════════════════════
# Latent extraction
# ══════════════════════════════════════════════════════════════════════
def extract_latents(
    mt: ModelandTokenizer,
    pair_records: list[dict],
    layers: list[int],
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Extract last-token hidden states for all contrastive pairs.

    Returns:
        Z_pos_by_layer: dict layer -> (n_pairs, hidden_dim)
        Z_neg_by_layer: dict layer -> (n_pairs, hidden_dim)
    """
    n_pairs_total = len(pair_records)
    pos_texts = [rec["prompt"] + rec["pos_path"] for rec in pair_records]
    neg_texts = [rec["prompt"] + rec["neg_path"] for rec in pair_records]
    all_texts = pos_texts + neg_texts

    n_total = len(all_texts)
    print(
        f"Extracting latents for {n_total} texts ({n_pairs_total} pairs) "
        f"in batches of {batch_size}..."
    )

    latents_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for batch_start in range(0, n_total, batch_size):
        batch_texts = all_texts[batch_start : batch_start + batch_size]
        if batch_start % (batch_size * 10) == 0:
            print(
                f"  Batch {batch_start // batch_size + 1}"
                f"/{(n_total + batch_size - 1) // batch_size}..."
            )

        batch_hs = get_last_token_latents_batch(mt, batch_texts, layers=layers)
        for layer in layers:
            latents_by_layer[layer].append(batch_hs[layer])

    Z_pos_by_layer: dict[int, torch.Tensor] = {}
    Z_neg_by_layer: dict[int, torch.Tensor] = {}

    for layer in layers:
        all_l = torch.cat(latents_by_layer[layer], dim=0)
        Z_pos_by_layer[layer] = all_l[:n_pairs_total]
        Z_neg_by_layer[layer] = all_l[n_pairs_total:]

    del latents_by_layer

    sample_layer = layers[0]
    print(f"Per-layer tensors: {len(Z_pos_by_layer)} layers")
    print(
        f"  Layer {sample_layer}: Z_pos {Z_pos_by_layer[sample_layer].shape} "
        f"on {Z_pos_by_layer[sample_layer].device}"
    )

    return Z_pos_by_layer, Z_neg_by_layer


# ══════════════════════════════════════════════════════════════════════
# Direction training + evaluation
# ══════════════════════════════════════════════════════════════════════
def train_directions(
    Z_pos_by_layer: dict[int, torch.Tensor],
    Z_neg_by_layer: dict[int, torch.Tensor],
    train_idx: list[int],
    test_idx: list[int],
    layers: list[int],
    args,
) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    """
    Train mean-diff and contrastive directions for all layers.

    Returns:
        mean_diff_directions, mean_diff_train_accs, mean_diff_test_accs,
        contrastive_directions, contrastive_train_accs, contrastive_test_accs,
        contrastive_infos
    """
    # --- Mean-diff baseline ---
    mean_diff_directions: dict[int, torch.Tensor] = {}
    mean_diff_train_accs: dict[int, float] = {}
    mean_diff_test_accs: dict[int, float] = {}

    print(f"\nTraining mean-diff directions for {len(layers)} layers...")
    print(f"  Train pairs: {len(train_idx)}, Test pairs: {len(test_idx)}")

    for layer_idx in layers:
        z_pos_l = Z_pos_by_layer[layer_idx]
        z_neg_l = Z_neg_by_layer[layer_idx]

        z_pos_train = z_pos_l[train_idx]
        z_neg_train = z_neg_l[train_idx]

        v_l = z_pos_train.mean(dim=0) - z_neg_train.mean(dim=0)
        v_l = v_l / (v_l.norm() + 1e-8)
        mean_diff_directions[layer_idx] = v_l

        train_acc = (z_pos_train @ v_l > z_neg_train @ v_l).float().mean().item()
        mean_diff_train_accs[layer_idx] = train_acc

        if len(test_idx) > 0:
            z_pos_test = z_pos_l[test_idx]
            z_neg_test = z_neg_l[test_idx]
            test_acc = (z_pos_test @ v_l > z_neg_test @ v_l).float().mean().item()
        else:
            test_acc = float("nan")
        mean_diff_test_accs[layer_idx] = test_acc

    # --- Contrastive directions ---
    contrastive_directions: dict[int, torch.Tensor] = {}
    contrastive_train_accs: dict[int, float] = {}
    contrastive_test_accs: dict[int, float] = {}
    contrastive_infos: dict[int, dict] = {}

    reg_desc = []
    if args.l1_lambda > 0:
        reg_desc.append(f"l1={args.l1_lambda}")
    if args.l2_lambda > 0:
        reg_desc.append(f"l2={args.l2_lambda}")
    if not reg_desc:
        reg_desc.append("no regularization")
    print(
        f"\nLearning contrastive directions for {len(layers)} layers "
        f"({', '.join(reg_desc)}, margin={args.margin})..."
    )

    for layer_idx in layers:
        z_pos_l = Z_pos_by_layer[layer_idx]
        z_neg_l = Z_neg_by_layer[layer_idx]

        z_pos_train = z_pos_l[train_idx]
        z_neg_train = z_neg_l[train_idx]

        v_l, info = learn_contrastive_direction(
            z_pos_train,
            z_neg_train,
            l1_lambda=args.l1_lambda,
            l2_lambda=args.l2_lambda,
            margin=args.margin,
            lr=args.lr,
            n_steps=args.n_steps,
        )
        contrastive_directions[layer_idx] = v_l
        contrastive_infos[layer_idx] = info

        train_acc = (z_pos_train @ v_l > z_neg_train @ v_l).float().mean().item()
        contrastive_train_accs[layer_idx] = train_acc

        if len(test_idx) > 0:
            z_pos_test = z_pos_l[test_idx]
            z_neg_test = z_neg_l[test_idx]
            test_acc = (z_pos_test @ v_l > z_neg_test @ v_l).float().mean().item()
        else:
            test_acc = float("nan")
        contrastive_test_accs[layer_idx] = test_acc

        if layer_idx % 10 == 0:
            print(
                f"  Layer {layer_idx:>3}: test_acc={test_acc:.3f}, "
                f"steps={info['steps']}, sparsity={info['sparsity']:.3f}"
            )

    # --- Print comparison table ---
    print(f"\nComparison: Mean Difference vs Contrastive ({', '.join(reg_desc)})")
    print(
        f"{'Layer':>6} {'MeanDiff':>10} {'Contrastive':>12} "
        f"{'Delta':>8} {'Steps':>6} {'Sparsity':>9}"
    )
    print(
        f"{'-----':>6} {'--------':>10} {'-----------':>12} "
        f"{'-----':>8} {'-----':>6} {'--------':>9}"
    )
    for layer_idx in layers:
        md_acc = mean_diff_test_accs[layer_idx]
        ct_acc = contrastive_test_accs[layer_idx]
        delta = ct_acc - md_acc
        steps = contrastive_infos[layer_idx]["steps"]
        sparsity = contrastive_infos[layer_idx]["sparsity"]
        marker = " <--" if ct_acc >= 0.8 else ""
        print(
            f"{layer_idx:>6} {md_acc:>10.3f} {ct_acc:>12.3f} "
            f"{delta:>+8.3f} {steps:>6} {sparsity:>9.3f}{marker}"
        )

    return (
        mean_diff_directions,
        mean_diff_train_accs,
        mean_diff_test_accs,
        contrastive_directions,
        contrastive_train_accs,
        contrastive_test_accs,
        contrastive_infos,
    )


# ══════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════
def save_results(
    save_dir: str,
    mean_diff_directions: dict,
    mean_diff_train_accs: dict,
    mean_diff_test_accs: dict,
    contrastive_directions: dict,
    contrastive_train_accs: dict,
    contrastive_test_accs: dict,
    contrastive_infos: dict,
    args,
):
    """Save direction vectors, accuracies, and training configs."""
    os.makedirs(save_dir, exist_ok=True)

    # Direction vectors
    torch.save(
        contrastive_directions,
        os.path.join(save_dir, "contrastive_directions.pt"),
    )
    torch.save(
        mean_diff_directions,
        os.path.join(save_dir, "mean_diff_directions.pt"),
    )

    # Contrastive training info
    # Convert int keys to str for JSON serialization
    contrastive_infos_str = {str(k): v for k, v in contrastive_infos.items()}
    with open(os.path.join(save_dir, "contrastive_infos.json"), "w") as f:
        json.dump(contrastive_infos_str, f, indent=2)

    # Test accuracies
    test_accs = {
        "mean_diff_train": {str(k): v for k, v in mean_diff_train_accs.items()},
        "mean_diff_test": {str(k): v for k, v in mean_diff_test_accs.items()},
        "contrastive_train": {str(k): v for k, v in contrastive_train_accs.items()},
        "contrastive_test": {str(k): v for k, v in contrastive_test_accs.items()},
    }
    with open(os.path.join(save_dir, "accuracies.json"), "w") as f:
        json.dump(test_accs, f, indent=2)

    # Training config
    config = {
        "model_name": args.model_name,
        "layer_from": args.layer_from,
        "layer_to": args.layer_to,
        "layer_step": args.layer_step,
        "l1_lambda": args.l1_lambda,
        "l2_lambda": args.l2_lambda,
        "margin": args.margin,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "n_train_pairs": args.n_train_pairs,
        "n_test_pairs": args.n_test_pairs,
        "unique_positive_sampling": args.unique_positive_sampling,
        "batch_size": args.batch_size,
        "mcts_iterations": args.mcts_iterations,
        "mcts_max_depth": args.mcts_max_depth,
        "mcts_top_k": args.mcts_top_k,
        "mcts_prob_threshold": args.mcts_prob_threshold,
        "depth_first": args.depth_first,
        "sampling_config": SAMPLING_CONFIG,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved results to {save_dir}/")
    print("  contrastive_directions.pt")
    print("  mean_diff_directions.pt")
    print("  contrastive_infos.json")
    print("  accuracies.json")
    print("  config.json")


# ══════════════════════════════════════════════════════════════════════
# Process one sample (full pipeline)
# ══════════════════════════════════════════════════════════════════════
def process_sample(mt: ModelandTokenizer, sample, sample_idx: int, args):
    """Run the full goal direction extraction pipeline for one sample."""
    model_short_name = mt.name.split("/")[-1]
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.results_dir,
        model_short_name,
        f"sample_{sample_idx}",
    )
    mcts_dir = os.path.join(save_dir, "MCTS")

    print(f"\n{'#' * 70}")
    print(f"# Sample {sample_idx}: {sample.function_name or 'func'}")
    print(f"# Save dir: {save_dir}")
    print(f"{'#' * 70}")

    # --- Build prefills and prompts ---
    prefills = []
    for ptype, pfill_list in sample.prefills.items():
        prefills.extend([prefill, ptype] for prefill in pfill_list)

    if supports_chat_template(mt):
        prompts = [
            apply_chat_template(
                mt=mt,
                user_query=sample.instruction.strip(),
                incomplete_code=sample.code + prefill,
            )
            for prefill, ptype in prefills
        ]
    else:
        prompts = [
            sample.instruction + "<code>" + sample.code + prefill
            for (prefill, ptype) in prefills
        ]

    # --- 1) MCTS ---
    all_results = run_or_load_mcts(
        mt=mt,
        sample=sample,
        prefills=prefills,
        prompts=prompts,
        mcts_dir=mcts_dir,
        args=args,
    )

    if sum(r.total_pairs for r in all_results) == 0:
        logger.error(f"No contrastive pairs for sample {sample_idx}. Skipping.")
        return

    # --- 2) Collect pairs + train/test split ---
    pair_records, train_idx, test_idx = collect_and_split_pairs(
        all_results=all_results,
        unique_positive_sampling=args.unique_positive_sampling,
        n_train_pairs=args.n_train_pairs,
        n_test_pairs=args.n_test_pairs,
    )

    if len(train_idx) == 0:
        logger.error(f"No training pairs for sample {sample_idx}. Skipping.")
        return

    if len(test_idx) == 0:
        logger.error(
            f"No test pairs for sample {sample_idx} — all test prefills had no "
            f"valid contrastive pairs. Skipping."
        )
        return

    # --- 3) Determine layers ---
    layer_to = args.layer_to if args.layer_to is not None else mt.n_layer
    layers = list(range(args.layer_from, layer_to, args.layer_step))
    print(f"\nLayers to probe: {len(layers)} (from {layers[0]} to {layers[-1]})")

    # --- 4) Extract latents ---
    Z_pos_by_layer, Z_neg_by_layer = extract_latents(
        mt=mt,
        pair_records=pair_records,
        layers=layers,
        batch_size=args.batch_size,
    )

    # --- 5) Train directions ---
    (
        mean_diff_directions,
        mean_diff_train_accs,
        mean_diff_test_accs,
        contrastive_directions,
        contrastive_train_accs,
        contrastive_test_accs,
        contrastive_infos,
    ) = train_directions(
        Z_pos_by_layer=Z_pos_by_layer,
        Z_neg_by_layer=Z_neg_by_layer,
        train_idx=train_idx,
        test_idx=test_idx,
        layers=layers,
        args=args,
    )

    # --- 6) Save ---
    save_results(
        save_dir=save_dir,
        mean_diff_directions=mean_diff_directions,
        mean_diff_train_accs=mean_diff_train_accs,
        mean_diff_test_accs=mean_diff_test_accs,
        contrastive_directions=contrastive_directions,
        contrastive_train_accs=contrastive_train_accs,
        contrastive_test_accs=contrastive_test_accs,
        contrastive_infos=contrastive_infos,
        args=args,
    )

    # --- 7) Cleanup GPU memory ---
    del Z_pos_by_layer, Z_neg_by_layer
    del mean_diff_directions, contrastive_directions
    free_gpu_cache()
    print(f"GPU cache cleared for sample {sample_idx}.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract goal directions from all samples in IncompleteCodeDataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model name or path",
    )

    # --- Output ---
    parser.add_argument(
        "--results_dir",
        type=str,
        default="probe",
        help="Subdirectory under DEFAULT_RESULTS_DIR for saving results",
    )

    # --- Sample selection ---
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to process (default: all)",
    )

    # --- MCTS ---
    parser.add_argument("--mcts_iterations", type=int, default=1500)
    parser.add_argument("--mcts_max_depth", type=int, default=20)
    parser.add_argument("--mcts_top_k", type=int, default=10)
    parser.add_argument("--mcts_prob_threshold", type=float, default=0.01)
    parser.add_argument("--depth_first", action="store_true", default=True)
    parser.add_argument(
        "--no_depth_first",
        action="store_false",
        dest="depth_first",
        help="Use breadth-first MCTS instead of depth-first",
    )
    parser.add_argument(
        "--force_rerun_mcts",
        action="store_true",
        default=False,
        help="Re-run MCTS even if cached results exist with different config",
    )
    parser.add_argument(
        "--ask_override_permission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), prompt user before overriding cached MCTS results "
        "with different config. If False, override without asking.",
    )

    # --- Pair collection ---
    parser.add_argument(
        "--unique_positive_sampling",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_train_pairs",
        type=int,
        default=10_000,
        help="Max training pairs (None = all). Set to 0 for no cap.",
    )
    parser.add_argument(
        "--n_test_pairs",
        type=int,
        default=None,
        help="Max test pairs (None = all). Set to 0 for no cap.",
    )

    # --- Layers ---
    parser.add_argument("--layer_from", type=int, default=0)
    parser.add_argument(
        "--layer_to",
        type=int,
        default=None,
        help="Exclusive upper bound for layers (default: mt.n_layer)",
    )
    parser.add_argument("--layer_step", type=int, default=1)

    # --- Contrastive training ---
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--l2_lambda", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_steps", type=int, default=1000)

    # --- Latent extraction ---
    parser.add_argument("--batch_size", type=int, default=32)

    # --- Logging ---
    logging_utils.add_logging_args(parser)

    args = parser.parse_args()

    # Treat 0 as None for pair caps
    if args.n_train_pairs == 0:
        args.n_train_pairs = None
    if args.n_test_pairs == 0:
        args.n_test_pairs = None

    return args


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(42)

    # --- Print config ---
    print(f"\n{'=' * 70}")
    print("Goal Direction Extraction")
    print(f"{'=' * 70}")
    print(f"Model:           {args.model_name}")
    print(f"Results dir:     {env_utils.DEFAULT_RESULTS_DIR}/{args.results_dir}")
    print(
        f"MCTS:            iters={args.mcts_iterations}, depth={args.mcts_max_depth}, "
        f"top_k={args.mcts_top_k}, threshold={args.mcts_prob_threshold}"
    )
    print(
        f"Contrastive:     l1={args.l1_lambda}, l2={args.l2_lambda}, "
        f"margin={args.margin}, lr={args.lr}, steps={args.n_steps}"
    )
    print(
        f"Pairs:           train_cap={args.n_train_pairs}, test_cap={args.n_test_pairs}, "
        f"unique_pos={args.unique_positive_sampling}"
    )
    print(
        f"Layers:          from={args.layer_from}, to={args.layer_to or 'n_layer'}, "
        f"step={args.layer_step}"
    )
    print(f"Batch size:      {args.batch_size}")
    print(f"{'=' * 70}\n")

    # --- Load model ---
    logger.info(f"Loading model: {args.model_name}")
    mt = ModelandTokenizer(
        model_key=args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(f"Model loaded: {mt.name}, n_layer={mt.n_layer}, n_embd={mt.n_embd}")

    # --- Load dataset ---
    eval_dataset = IncompleteCodeDataset.load_dataset()
    logger.info(f"Dataset loaded: {len(eval_dataset)} samples")

    # --- Determine which samples to process ---
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        sample_indices = list(range(len(eval_dataset)))

    print(f"Processing {len(sample_indices)} samples: {sample_indices}\n")

    # --- Process each sample ---
    for sample_idx in sample_indices:
        sample = eval_dataset[sample_idx]
        process_sample(mt=mt, sample=sample, sample_idx=sample_idx, args=args)

    print(f"\n{'=' * 70}")
    print(f"All done! Processed {len(sample_indices)} samples.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
