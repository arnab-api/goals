"""Multi-predicate probe direction extraction script.

Loads pre-computed MCTS trees (produced by multipred_mcts.py) and trains
goal-direction probes for each TargetPredicate.

Run multipred_mcts.py FIRST to expand the trees.

Usage:
    python -m scripts.multipred_probe --model_name Qwen/Qwen2.5-Coder-32B-Instruct
    python -m scripts.multipred_probe --model_name Qwen/Qwen2.5-Coder-14B-Instruct \
        --predicate_indices 0 1

Pipeline per TargetPredicate:
  1. Load MCTSSearchResults_for_TargetPredicate from per-sample subdirs under
         {save_dir}/{model}/{predicate.name}/{p_minus_desc}/mcts.json
  1.5. Ensure P+ / P- anchor tokens are populated (oracle LM extraction if missing)
  2. Collect P+ / P- labeled nodes (anchor-token based labeling) PER SAMPLE
  3. Form cross-tree dual-window contrastive pairs PER SAMPLE (prevents cross-P- mixing)
  4. Write per-tree probe logs (labeled-node counts by depth / abs-position)
  5. Extract last-token latents for all pairs
  6. Train mean-diff (unit-norm) and contrastive (unit-norm) directions per layer
  7. Evaluate: constraint accuracy + ROC-AUC on test set
  8. Save directions, accuracies, contrastive infos, and run config

Output artifact structure
─────────────────────────
All paths are under {save_dir}/ (default: $RESULTS_DIR/multi_probe).

    {save_dir}/
      {model_name}/                    e.g. Qwen2.5-Coder-32B-Instruct/
        {predicate.name}/              e.g. is_palindrome/
          │
          ├── {p_minus_desc}/                (one dir per sample, written by multipred_mcts.py)
          │     ├── mcts.json                  Per-sample MCTS results (read-only here)
          │     ├── {prefill}.html              Color-coded MCTS tree
          │     └── probe_log_{prefill}.json   Per-tree log:
          │                                      eval_counts   (green/orange/red/total)
          │                                      mcts_stats
          │                                      n_labeled_nodes
          │                                      by_depth      {depth: {p_plus, p_minus}}
          │                                      by_abs_position {pos:  {p_plus, p_minus}}
          │
          ├── mean_diff_directions.pt    dict[layer_idx → unit-norm Tensor(hidden_dim,)]
          ├── contrastive_directions.pt  dict[layer_idx → unit-norm Tensor(hidden_dim,)]
          │
          ├── accuracies.json            Per-layer metrics (keys are str(layer_idx)):
          │                               mean_diff_train / mean_diff_test   (constraint acc)
          │                               contrastive_train / contrastive_test (constraint acc)
          │                               mean_diff_auc / contrastive_auc    (ROC-AUC, test)
          │
          ├── contrastive_infos.json     Per-layer training info:
          │                               steps (early-stop), best_loss, sparsity
          │
          └── config.json                Run config: model, probe hyperparams,
                                         contrastive pair config, layer range.
"""

import argparse
import json
import logging
import os
import pathlib
import random
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import numpy as np
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score

from src.functional import free_gpu_cache
from src.goals.anchor import extract_anchor_tokens
from src.goals.functional import get_last_token_latents_batch
from src.goals.predicate_data import MultiPredicateDataset, TargetPredicate
from src.goals.predicate_probe import (
    LabeledNode,
    collect_labeled_nodes,
    sample_multipred_pairs,
)
from src.goals.predicate_sampling import MCTSSearchResults_for_TargetPredicate
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Contrastive pair config  (edit here; not exposed as CLI args)
# ──────────────────────────────────────────────────────────────────────
CONTRASTIVE_PAIR_CONFIG = dict(
    min_p_green=0.5,  # filter nodes with low green probability mass
    allow_same_tree=False,  # cross-tree pairing only
    abs_pos_tolerance=0,  # exact abs-position match
)


# ══════════════════════════════════════════════════════════════════════
# Direction training
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
    """Hinge-loss contrastive direction with L1/L2 regularisation."""
    device = z_pos.device
    delta = z_pos - z_neg

    v = torch.randn(z_pos.shape[1], device=device, dtype=z_pos.dtype) * 0.01
    v.requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=lr)

    best_loss = float("inf")
    patience_counter = 0
    final_step = n_steps

    for step in range(n_steps):
        optimizer.zero_grad()
        margins = margin - (delta @ v)
        loss = torch.clamp(margins, min=0).mean()
        loss += l1_lambda * v.abs().mean() + l2_lambda * (v**2).mean()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                final_step = step + 1
                break

    with torch.no_grad():
        v_norm = v / (v.norm() + 1e-8)

    sparsity = (v_norm.abs() < 0.01).float().mean().item()
    return v_norm.detach(), {
        "steps": final_step,
        "best_loss": best_loss,
        "sparsity": sparsity,
    }


# ══════════════════════════════════════════════════════════════════════
# Helpers (path utilities, shared by save_tree_logs)
# ══════════════════════════════════════════════════════════════════════
def _safe_name(s: str, max_len: int = 40) -> str:
    """Convert an arbitrary string into a safe filesystem name."""
    out = s.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    out = "".join(c for c in out if c.isalnum() or c in "_-.")
    return out[:max_len] or "empty"


def _p_minus_dir_name(sample, sample_idx: int) -> str:
    """Derive a safe directory name from the sample's P- predicate description."""
    if sample.p_minus is not None:
        return _safe_name(sample.p_minus.description[:50])
    return f"sample{sample_idx}"


# ══════════════════════════════════════════════════════════════════════
# Step 1 — Load MCTS results
# ══════════════════════════════════════════════════════════════════════
def load_mcts_results(
    tp: TargetPredicate,
    predicate_dir: str,
) -> MCTSSearchResults_for_TargetPredicate:
    """Load cached MCTS results from per-sample subdirs under predicate_dir.

    Supports both:
      - New layout: {predicate_dir}/{p_minus_desc}/mcts.json  (per-sample)
      - Legacy layout: {predicate_dir}/trees/mcts.json  (single bundled file)

    Raises FileNotFoundError if no cache exists (run multipred_mcts.py first).
    """
    # Try new per-sample layout first
    if os.path.isdir(predicate_dir):
        sample_jsons = list(
            p
            for p in sorted(pathlib.Path(predicate_dir).glob("*/mcts.json"))
            # Exclude legacy trees/ subdirectory
            if p.parent.name != "trees"
        )
        if sample_jsons:
            tp_results = MCTSSearchResults_for_TargetPredicate.load(predicate_dir)
            n_trees = sum(len(sr._trees) for sr in tp_results.sample_results)
            logger.info(
                f"Loaded MCTS results for '{tp.name}': "
                f"{len(tp_results.sample_results)} samples, {n_trees} in-memory trees."
            )
            if n_trees == 0:
                logger.warning(
                    "No in-memory trees in cache — anchor labeling may fail. "
                    "Re-run multipred_mcts.py with --force_rerun_mcts to regenerate."
                )
            return tp_results

    # Fall back to legacy single-file layout
    legacy_path = os.path.join(predicate_dir, "trees", "mcts.json")
    if os.path.exists(legacy_path):
        tp_results = MCTSSearchResults_for_TargetPredicate.load(legacy_path)
        n_trees = sum(len(sr._trees) for sr in tp_results.sample_results)
        logger.info(
            f"Loaded MCTS results (legacy) for '{tp.name}': {n_trees} in-memory trees."
        )
        if n_trees == 0:
            logger.warning(
                "No in-memory trees in cache — anchor labeling may fail. "
                "Re-run multipred_mcts.py with --force_rerun_mcts to regenerate."
            )
        return tp_results

    raise FileNotFoundError(
        f"MCTS cache not found under {predicate_dir}\n"
        "Run multipred_mcts.py first to generate MCTS trees."
    )


# ══════════════════════════════════════════════════════════════════════
# Step 1.5 — Anchor tokens
# ══════════════════════════════════════════════════════════════════════
def ensure_anchor_tokens(
    tp: TargetPredicate,
    tp_results: MCTSSearchResults_for_TargetPredicate,
    force_reextract: bool = False,
) -> None:
    """Populate P+ / P- anchor tokens via oracle LM if any are missing, then save to disk.

    Anchor tokens live in:
        tp.p_plus.anchor_tokens               (shared across all samples)
        tp.samples[i].p_minus.anchor_tokens   (per-sample)

    If all are already populated and force_reextract=False, this is a no-op.
    On update, the TargetPredicate JSON is overwritten via tp.save().
    """
    needs = (
        force_reextract
        or not tp.p_plus.anchor_tokens
        or any(
            not (sample.p_minus and sample.p_minus.anchor_tokens)
            for sample in tp.samples
        )
    )
    if not needs:
        logger.info(
            f"Anchor tokens already populated for '{tp.name}' — skipping oracle."
        )
        return

    print(f"\nExtracting anchor tokens for '{tp.name}' via oracle LM...")
    p_plus_anchors, p_minus_per_sample = extract_anchor_tokens(
        tp=tp,
        completions_per_sample=tp_results.completions_per_sample,
    )

    tp.p_plus.anchor_tokens = sorted(p_plus_anchors)
    for i, sample in enumerate(tp.samples):
        if sample.p_minus is not None:
            sample.p_minus.anchor_tokens = sorted(p_minus_per_sample[i])

    tp.save()
    print(f"  P+ anchors:  {tp.p_plus.anchor_tokens}")
    for i, sample in enumerate(tp.samples):
        anchors = sample.p_minus.anchor_tokens if sample.p_minus else []
        print(f"  P- anchors (sample {i}): {anchors}")
    print(f"Saved anchor tokens → {tp._source_path}")


# ══════════════════════════════════════════════════════════════════════
# Step 2 — Per-tree probe logs
# ══════════════════════════════════════════════════════════════════════
def save_tree_logs(
    tp: TargetPredicate,
    tp_results: MCTSSearchResults_for_TargetPredicate,
    labeled_nodes: list[LabeledNode],
    predicate_dir: str,
) -> None:
    """Write per-run JSON logs (eval counts + labeled-node breakdown by depth / abs-position).

    Logs are written alongside MCTS artifacts:
        {predicate_dir}/{p_minus_desc}/probe_log_{prefill}.json
    """
    nodes_by_tree: dict[str, list[LabeledNode]] = defaultdict(list)
    for ln in labeled_nodes:
        nodes_by_tree[ln.tree_id].append(ln)

    for sample_idx, sr in enumerate(tp_results.sample_results):
        sample = tp.samples[sample_idx]
        p_minus_name = _p_minus_dir_name(sample, sample_idx)
        sample_dir = os.path.join(predicate_dir, p_minus_name)
        os.makedirs(sample_dir, exist_ok=True)

        for run in sr.runs:
            safe_pval = _safe_name(run.prefill) if run.prefill else "empty"

            tree_nodes = nodes_by_tree.get(run.prefix, [])

            by_depth: dict[int, dict[str, int]] = defaultdict(
                lambda: {"p_plus": 0, "p_minus": 0}
            )
            by_abs_pos: dict[int, dict[str, int]] = defaultdict(
                lambda: {"p_plus": 0, "p_minus": 0}
            )
            for ln in tree_nodes:
                by_depth[ln.depth][ln.label] += 1
                by_abs_pos[ln.abs_position][ln.label] += 1

            log = {
                "sample_idx": sample_idx,
                "function_name": sample.function_name,
                "prefill": run.prefill,
                "prefill_type": run.prefill_type,
                "eval_counts": run.eval_counts,
                "mcts_stats": run.mcts_stats,
                "n_labeled_nodes": len(tree_nodes),
                "by_depth": {str(d): v for d, v in sorted(by_depth.items())},
                "by_abs_position": {str(p): v for p, v in sorted(by_abs_pos.items())},
            }
            log_path = os.path.join(sample_dir, f"probe_log_{safe_pval}.json")
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

    logger.info(f"Per-tree logs written under {predicate_dir}/")


# ══════════════════════════════════════════════════════════════════════
# Step 3 — Contrastive pair collection + train/test split
# ══════════════════════════════════════════════════════════════════════
def collect_and_split_pairs(
    tp_results: MCTSSearchResults_for_TargetPredicate,
    tp: TargetPredicate,
    mt: ModelandTokenizer,
    args,
) -> tuple[list[LabeledNode], list[dict], list[int], list[int]]:
    """Collect labeled nodes, form cross-tree pairs, build pair records, split.

    Pairing is done PER SAMPLE (per P- companion) to prevent cross-P-
    contamination — nodes from different (P+, P-) pairs are never paired.
    """
    prefix_to_prompt = {
        run.prefix: run.prompt for sr in tp_results.sample_results for run in sr.runs
    }

    all_labeled: list[LabeledNode] = []
    pair_records: list[dict] = []

    for sample_idx, _sr in enumerate(tp_results.sample_results):
        # Collect labeled nodes for this sample only
        labeled_i = collect_labeled_nodes(
            tp_results=tp_results,
            tp=tp,
            mt=mt,
            sample_indices=[sample_idx],
            min_p_green=CONTRASTIVE_PAIR_CONFIG["min_p_green"],
        )
        all_labeled.extend(labeled_i)

        if not labeled_i:
            continue

        # Form pairs within this sample
        pairs_by_pos_i = sample_multipred_pairs(
            labeled_nodes=labeled_i,
            allow_same_tree=CONTRASTIVE_PAIR_CONFIG["allow_same_tree"],
            abs_pos_tolerance=CONTRASTIVE_PAIR_CONFIG["abs_pos_tolerance"],
        )

        for pairs in pairs_by_pos_i.values():
            for pair in pairs:
                pos_ln = pair.positive
                neg_ln = pair.negative
                pair_records.append(
                    {
                        "pos_text": prefix_to_prompt.get(pos_ln.tree_id, "")
                        + pos_ln.node.get_path_text(),
                        "neg_text": prefix_to_prompt.get(neg_ln.tree_id, "")
                        + neg_ln.node.get_path_text(),
                        "abs_position": pair.abs_position,
                        "pos_depth": pos_ln.depth,
                        "neg_depth": neg_ln.depth,
                        "pos_window": pos_ln.window,
                        "neg_window": neg_ln.window,
                        "pos_tree_id": pos_ln.tree_id,
                        "neg_tree_id": neg_ln.tree_id,
                        "pos_prefill_type": pos_ln.prefill_type,
                        "neg_prefill_type": neg_ln.prefill_type,
                        "sample_idx": pos_ln.sample_idx,
                    }
                )

    print(f"\nLabeled nodes: {len(all_labeled)}")
    p_plus_count = sum(1 for ln in all_labeled if ln.label == "p_plus")
    p_minus_count = sum(1 for ln in all_labeled if ln.label == "p_minus")
    print(f"  P+: {p_plus_count}  P-: {p_minus_count}")
    print(
        f"Contrastive pairs: {len(pair_records)} (paired per-sample to prevent cross-P- mixing)"
    )

    if not pair_records:
        return all_labeled, [], [], []

    # Train/test split by tree (prevents leakage)
    random.seed(42)
    all_tree_ids = list({rec["pos_tree_id"] for rec in pair_records})
    random.shuffle(all_tree_ids)
    n_train_trees = max(1, int(0.7 * len(all_tree_ids)))
    train_trees = set(all_tree_ids[:n_train_trees])
    test_trees = set(all_tree_ids[n_train_trees:])

    train_idx = [
        i for i, r in enumerate(pair_records) if r["pos_tree_id"] in train_trees
    ]
    test_idx = [i for i, r in enumerate(pair_records) if r["pos_tree_id"] in test_trees]

    if args.n_train_pairs and len(train_idx) > args.n_train_pairs:
        train_idx = random.sample(train_idx, args.n_train_pairs)
    if args.n_test_pairs and len(test_idx) > args.n_test_pairs:
        test_idx = random.sample(test_idx, args.n_test_pairs)

    print(
        f"Split: {len(train_idx)} train / {len(test_idx)} test  "
        f"({len(train_trees)} train trees / {len(test_trees)} test trees)"
    )
    return all_labeled, pair_records, train_idx, test_idx


# ══════════════════════════════════════════════════════════════════════
# Step 4 — Latent extraction
# ══════════════════════════════════════════════════════════════════════
def extract_latents(
    mt: ModelandTokenizer,
    pair_records: list[dict],
    layers: list[int],
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    n_pairs = len(pair_records)
    pos_texts = [r["pos_text"] for r in pair_records]
    neg_texts = [r["neg_text"] for r in pair_records]
    all_texts = pos_texts + neg_texts

    print(f"Extracting latents for {len(all_texts)} texts ({n_pairs} pairs)...")
    latents: dict[int, list[torch.Tensor]] = {li: [] for li in layers}

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        if i % (batch_size * 10) == 0:
            print(
                f"  Batch {i // batch_size + 1}/"
                f"{(len(all_texts) + batch_size - 1) // batch_size}..."
            )
        hs = get_last_token_latents_batch(mt, batch, layers=layers)
        for li in layers:
            latents[li].append(hs[li])

    Z_pos, Z_neg = {}, {}
    for li in layers:
        all_li = torch.cat(latents[li], dim=0)
        Z_pos[li] = all_li[:n_pairs]
        Z_neg[li] = all_li[n_pairs:]

    del latents
    print(f"Latents extracted. Shape (layer 0): {list(Z_pos[layers[0]].shape)}")
    return Z_pos, Z_neg


# ══════════════════════════════════════════════════════════════════════
# Step 5 — Train directions
# ══════════════════════════════════════════════════════════════════════
def train_directions(
    Z_pos: dict[int, torch.Tensor],
    Z_neg: dict[int, torch.Tensor],
    train_idx: list[int],
    test_idx: list[int],
    layers: list[int],
    args,
) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    # L2-normalize activations (unit-sphere) before training
    Z_pos = {
        li: z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8) for li, z in Z_pos.items()
    }
    Z_neg = {
        li: z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8) for li, z in Z_neg.items()
    }

    # ── Mean-diff ─────────────────────────────────────────────────────
    md_dirs: dict[int, torch.Tensor] = {}
    md_train_accs: dict[int, float] = {}
    md_test_accs: dict[int, float] = {}

    print(f"\nTraining mean-diff directions for {len(layers)} layers...")
    for li in layers:
        z_pos_tr = Z_pos[li][train_idx]
        z_neg_tr = Z_neg[li][train_idx]
        v = z_pos_tr.mean(0) - z_neg_tr.mean(0)
        v = v / (v.norm() + 1e-8)
        md_dirs[li] = v

        md_train_accs[li] = (z_pos_tr @ v > z_neg_tr @ v).float().mean().item()
        if test_idx:
            md_test_accs[li] = (
                (Z_pos[li][test_idx] @ v > Z_neg[li][test_idx] @ v)
                .float()
                .mean()
                .item()
            )
        else:
            md_test_accs[li] = float("nan")

    # ── Contrastive ───────────────────────────────────────────────────
    cd_dirs: dict[int, torch.Tensor] = {}
    cd_train_accs: dict[int, float] = {}
    cd_test_accs: dict[int, float] = {}
    cd_infos: dict[int, dict] = {}

    reg_parts = []
    if args.l1_lambda > 0:
        reg_parts.append(f"l1={args.l1_lambda}")
    if args.l2_lambda > 0:
        reg_parts.append(f"l2={args.l2_lambda}")
    if not reg_parts:
        reg_parts.append("no-reg")
    print(
        f"Training contrastive directions ({', '.join(reg_parts)}, margin={args.margin})..."
    )

    for li in layers:
        z_pos_tr = Z_pos[li][train_idx]
        z_neg_tr = Z_neg[li][train_idx]
        v, info = learn_contrastive_direction(
            z_pos_tr,
            z_neg_tr,
            l1_lambda=args.l1_lambda,
            l2_lambda=args.l2_lambda,
            margin=args.margin,
            lr=args.lr,
            n_steps=args.n_steps,
        )
        cd_dirs[li] = v
        cd_infos[li] = info

        cd_train_accs[li] = (z_pos_tr @ v > z_neg_tr @ v).float().mean().item()
        if test_idx:
            cd_test_accs[li] = (
                (Z_pos[li][test_idx] @ v > Z_neg[li][test_idx] @ v)
                .float()
                .mean()
                .item()
            )
        else:
            cd_test_accs[li] = float("nan")

        if li % 10 == 0:
            print(
                f"  Layer {li:>3}: cd_test={cd_test_accs[li]:.3f}  "
                f"steps={info['steps']}  sparsity={info['sparsity']:.3f}"
            )

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n{'Layer':>6} {'MeanDiff':>10} {'Contrastive':>12} {'Delta':>8}")
    print(f"{'-----':>6} {'--------':>10} {'-----------':>12} {'-----':>8}")
    for li in layers:
        delta = cd_test_accs[li] - md_test_accs[li]
        marker = " <--" if cd_test_accs[li] >= 0.8 else ""
        print(
            f"{li:>6} {md_test_accs[li]:>10.3f} {cd_test_accs[li]:>12.3f} {delta:>+8.3f}{marker}"
        )

    return (
        md_dirs,
        md_train_accs,
        md_test_accs,
        cd_dirs,
        cd_train_accs,
        cd_test_accs,
        cd_infos,
    )


# ══════════════════════════════════════════════════════════════════════
# Step 6 — Evaluate directions (AUC)
# ══════════════════════════════════════════════════════════════════════
def evaluate_directions(
    Z_pos: dict[int, torch.Tensor],
    Z_neg: dict[int, torch.Tensor],
    test_idx: list[int],
    md_dirs: dict[int, torch.Tensor],
    cd_dirs: dict[int, torch.Tensor],
    layers: list[int],
) -> dict[str, dict[int, float]]:
    """Compute ROC-AUC for mean-diff and contrastive directions on the test set."""
    if not test_idx:
        logger.warning("No test pairs — skipping AUC evaluation.")
        return {}

    test_idx_arr = np.array(test_idx)
    labels = np.concatenate([np.ones(len(test_idx)), np.zeros(len(test_idx))])

    md_aucs: dict[int, float] = {}
    cd_aucs: dict[int, float] = {}

    print(f"\n{'Layer':>6} {'MeanDiff AUC':>14} {'Contrastive AUC':>16}")
    print(f"{'-----':>6} {'------------':>14} {'---------------':>16}")

    for li in layers:
        z_pos = Z_pos[li][test_idx_arr].cpu().numpy()
        z_neg = Z_neg[li][test_idx_arr].cpu().numpy()
        all_z = np.concatenate([z_pos, z_neg], axis=0)
        norms = np.maximum(np.linalg.norm(all_z, axis=1, keepdims=True), 1e-8)
        all_z = all_z / norms

        v_md = md_dirs[li].cpu().numpy()
        v_cd = cd_dirs[li].cpu().numpy()

        md_aucs[li] = roc_auc_score(labels, all_z @ v_md)
        cd_aucs[li] = roc_auc_score(labels, all_z @ v_cd)

        if li % 8 == 0:
            print(f"{li:>6} {md_aucs[li]:>14.4f} {cd_aucs[li]:>16.4f}")

    return {"mean_diff_auc": md_aucs, "contrastive_auc": cd_aucs}


# ══════════════════════════════════════════════════════════════════════
# Step 7 — Save results
# ══════════════════════════════════════════════════════════════════════
def save_results(
    save_dir: str,
    md_dirs: dict,
    md_train_accs: dict,
    md_test_accs: dict,
    cd_dirs: dict,
    cd_train_accs: dict,
    cd_test_accs: dict,
    cd_infos: dict,
    auc_results: dict,
    args,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(md_dirs, os.path.join(save_dir, "mean_diff_directions.pt"))
    torch.save(cd_dirs, os.path.join(save_dir, "contrastive_directions.pt"))

    with open(os.path.join(save_dir, "contrastive_infos.json"), "w") as f:
        json.dump({str(k): v for k, v in cd_infos.items()}, f, indent=2)

    accs = {
        "mean_diff_train": {str(k): v for k, v in md_train_accs.items()},
        "mean_diff_test": {str(k): v for k, v in md_test_accs.items()},
        "contrastive_train": {str(k): v for k, v in cd_train_accs.items()},
        "contrastive_test": {str(k): v for k, v in cd_test_accs.items()},
    }
    for key, layer_dict in auc_results.items():
        accs[key] = {str(k): v for k, v in layer_dict.items()}
    with open(os.path.join(save_dir, "accuracies.json"), "w") as f:
        json.dump(accs, f, indent=2)

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
        "batch_size": args.batch_size,
        "contrastive_pair_config": CONTRASTIVE_PAIR_CONFIG,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved results to {save_dir}/")
    print("  mean_diff_directions.pt  contrastive_directions.pt")
    print("  contrastive_infos.json   accuracies.json   config.json")


# ══════════════════════════════════════════════════════════════════════
# Full pipeline for one TargetPredicate
# ══════════════════════════════════════════════════════════════════════
def process_predicate(
    mt: ModelandTokenizer,
    tp: TargetPredicate,
    predicate_idx: int,
    args,
) -> None:
    model_short = mt.name.split("/")[-1]
    probe_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.save_dir, model_short, tp.name
    )
    predicate_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.tree_dir, model_short, tp.name
    )

    print(f"\n{'#' * 70}")
    print(f"# Predicate [{predicate_idx}]: {tp.name}")
    print(f"# P+: {tp.p_plus.description}")
    print(f"# Samples: {len(tp.samples)}")
    print(f"# Save dir: {probe_dir}")
    print(f"{'#' * 70}")

    # ── 1) Load MCTS ──────────────────────────────────────────────────
    try:
        tp_results = load_mcts_results(tp=tp, predicate_dir=predicate_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    total_green = sum(
        run.eval_counts.get("green", 0)
        for sr in tp_results.sample_results
        for run in sr.runs
    )
    print(f"Total green leaves across all runs: {total_green}")
    if total_green == 0:
        logger.error(f"No green leaves for '{tp.name}'. Skipping.")
        return

    # ── 1.5) Ensure anchor tokens ─────────────────────────────────────
    ensure_anchor_tokens(tp, tp_results, args.force_reextract_anchors)

    # ── 2) Collect labeled nodes + contrastive pairs ──────────────────
    labeled, pair_records, train_idx, test_idx = collect_and_split_pairs(
        tp_results=tp_results, tp=tp, mt=mt, args=args
    )

    # ── 3) Per-tree probe logs ────────────────────────────────────────
    save_tree_logs(
        tp=tp,
        tp_results=tp_results,
        labeled_nodes=labeled,
        predicate_dir=predicate_dir,
    )

    if not pair_records:
        logger.error(
            f"No contrastive pairs for '{tp.name}'. Skipping direction training."
        )
        return

    if not train_idx:
        logger.error(f"No training pairs for '{tp.name}'. Skipping.")
        return

    # ── 4) Determine layers ───────────────────────────────────────────
    layer_to = args.layer_to if args.layer_to is not None else mt.n_layer
    layers = list(range(args.layer_from, layer_to, args.layer_step))
    print(f"\nLayers to probe: {len(layers)}  ({layers[0]}..{layers[-1]})")

    # ── 5) Extract latents ────────────────────────────────────────────
    Z_pos, Z_neg = extract_latents(
        mt=mt, pair_records=pair_records, layers=layers, batch_size=args.batch_size
    )

    # ── 6) Train directions ───────────────────────────────────────────
    (
        md_dirs,
        md_train_accs,
        md_test_accs,
        cd_dirs,
        cd_train_accs,
        cd_test_accs,
        cd_infos,
    ) = train_directions(
        Z_pos=Z_pos,
        Z_neg=Z_neg,
        train_idx=train_idx,
        test_idx=test_idx,
        layers=layers,
        args=args,
    )

    # ── 7) Evaluate (AUC) ─────────────────────────────────────────────
    auc_results = evaluate_directions(
        Z_pos=Z_pos,
        Z_neg=Z_neg,
        test_idx=test_idx,
        md_dirs=md_dirs,
        cd_dirs=cd_dirs,
        layers=layers,
    )

    # ── 8) Save ───────────────────────────────────────────────────────
    save_results(
        save_dir=probe_dir,
        md_dirs=md_dirs,
        md_train_accs=md_train_accs,
        md_test_accs=md_test_accs,
        cd_dirs=cd_dirs,
        cd_train_accs=cd_train_accs,
        cd_test_accs=cd_test_accs,
        cd_infos=cd_infos,
        auc_results=auc_results,
        args=args,
    )

    # ── 9) Cleanup ────────────────────────────────────────────────────
    del Z_pos, Z_neg, md_dirs, cd_dirs
    free_gpu_cache()
    print(f"GPU cache cleared for '{tp.name}'.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train goal-direction probes from pre-computed multi-predicate MCTS trees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct"
    )

    # Output
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(env_utils.DEFAULT_RESULTS_DIR, "multi_probe"),
        help="Root directory for saving probe directions and results.",
    )
    parser.add_argument(
        "--tree_dir",
        type=str,
        default=None,
        help=(
            "Root directory to load MCTS trees from "
            "(default: same as --save_dir). "
            "Useful when trees and directions live in separate locations. "
            "Trees are expected at {tree_dir}/{model}/{predicate.name}/{p_minus_desc}/mcts.json"
        ),
    )

    # Predicate selection
    parser.add_argument(
        "--predicate_indices",
        type=int,
        nargs="+",
        default=None,
        help="Indices of TargetPredicates to process (default: all)",
    )

    # Pair collection
    parser.add_argument(
        "--n_train_pairs",
        type=int,
        default=10_000,
        help="Max training pairs (0 = no cap)",
    )
    parser.add_argument(
        "--n_test_pairs", type=int, default=10_000, help="Max test pairs (0 = no cap)"
    )

    # Anchor tokens
    parser.add_argument(
        "--force_reextract_anchors",
        action="store_true",
        default=False,
        help="Re-extract anchor tokens even if already populated",
    )

    # Layers
    parser.add_argument("--layer_from", type=int, default=0)
    parser.add_argument(
        "--layer_to",
        type=int,
        default=None,
        help="Exclusive upper bound (default: mt.n_layer)",
    )
    parser.add_argument("--layer_step", type=int, default=1)

    # Contrastive training
    parser.add_argument("--l1_lambda", type=float, default=0.0)
    parser.add_argument("--l2_lambda", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n_steps", type=int, default=1000)

    # Latent extraction
    parser.add_argument("--batch_size", type=int, default=32)

    # Logging
    logging_utils.add_logging_args(parser)

    args = parser.parse_args()

    # Default tree_dir to save_dir
    if args.tree_dir is None:
        args.tree_dir = args.save_dir

    # Treat 0 as "no cap"
    if args.n_train_pairs == 0:
        args.n_train_pairs = None
    if args.n_test_pairs == 0:
        args.n_test_pairs = None

    return args


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(42)

    print(f"\n{'=' * 70}")
    print("Multi-Predicate Probe Direction Extraction")
    print(f"{'=' * 70}")
    print(f"Model:       {args.model_name}")
    print(f"Save dir:    {args.save_dir}")
    if args.tree_dir != args.save_dir:
        print(f"Tree dir:    {args.tree_dir}  (loading trees from here)")
    print(
        f"Contrastive: l1={args.l1_lambda}, l2={args.l2_lambda}, "
        f"margin={args.margin}, lr={args.lr}, steps={args.n_steps}"
    )
    print(f"Pairs:       train_cap={args.n_train_pairs}, test_cap={args.n_test_pairs}")
    print(
        f"Layers:      from={args.layer_from}, to={args.layer_to or 'n_layer'}, "
        f"step={args.layer_step}"
    )
    print(f"Batch size:  {args.batch_size}")
    print(f"{'=' * 70}\n")

    logger.info(f"Loading model: {args.model_name}")
    mt = ModelandTokenizer(
        model_key=args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(f"Model loaded: {mt.name}, n_layer={mt.n_layer}, n_embd={mt.n_embd}")

    dataset = MultiPredicateDataset.load_dataset()
    logger.info(f"Dataset: {len(dataset.groups)} target predicates")

    indices = (
        args.predicate_indices
        if args.predicate_indices is not None
        else list(range(len(dataset.groups)))
    )
    print(f"Processing {len(indices)} predicate(s): {indices}\n")

    for predicate_idx in indices:
        tp = dataset.groups[predicate_idx]
        process_predicate(mt=mt, tp=tp, predicate_idx=predicate_idx, args=args)

    print(f"\n{'=' * 70}")
    print(f"All done! Processed {len(indices)} predicate(s).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
