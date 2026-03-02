"""Multi-predicate MCTS expansion script.

For each TargetPredicate in MultiPredicateDataset:
  1. Runs MCTS per (sample × prefill), or reuses cached results
  2. Evaluates each tree with unit tests (green / orange / red)
  3. Saves color-coded HTML trees per (P-, prefill) combination
  4. Calls oracle LM to extract P+ / P- anchor tokens from green completions
  5. Writes populated anchor tokens back to the data JSON files

Run this script BEFORE multipred_probe.py.

Usage:
    python -m scripts.multipred_mcts --model_name Qwen/Qwen2.5-Coder-32B-Instruct
    python -m scripts.multipred_mcts --model_name Qwen/Qwen2.5-Coder-14B-Instruct \
        --predicate_indices 0 1 --force_rerun_mcts

Output artifact structure
─────────────────────────
All paths are under {save_dir}/ (default: $RESULTS_DIR/multi_probe).

    {save_dir}/
      {model_name}/                    e.g. Qwen2.5-Coder-32B-Instruct/
        {predicate.name}/              e.g. is_palindrome/
          trees/
            mcts.json                  Single bundled JSON: all samples × prefills,
                                       including serialized trees (root.to_dict()).
                                       Reused as cache on subsequent runs unless
                                       --force_rerun_mcts is set.
            {p_minus_desc}/            One dir per sample, named after P- description.
              {prefill}/               One dir per prefill (safe-name of prefill string).
                mcts_tree.html         Color-coded MCTS tree (green/orange/red).

    data/multi_predicate/{predicate.name}.json   ← anchor tokens written back here
"""

import argparse
import logging
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import torch  # noqa: E402

from src.goals.anchor import extract_anchor_tokens
from src.goals.eval import evaluate_mcts_tree
from src.goals.functional import apply_chat_template, supports_chat_template
from src.goals.predicate_data import MultiPredicateDataset, TargetPredicate
from src.goals.predicate_sampling import (
    MCTSRunResult,
    MCTSSearchResults_for_MultiPredicateSample,
    MCTSSearchResults_for_TargetPredicate,
)
from src.goals.sampling import MCTSNode, MCTSSearch
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# MCTS config defaults  (edit here; not exposed as CLI args)
# ──────────────────────────────────────────────────────────────────────
MCTS_DIVERSITY_CONFIG = dict(
    c_puct=1.0,
    temperature=1.0,
    min_children_up_to_depth=6,
    min_children=4,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _safe_name(s: str, max_len: int = 40) -> str:
    out = s.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    out = "".join(c for c in out if c.isalnum() or c in "_-.")
    return out[:max_len] or "empty"


def _p_minus_dir_name(sample, sample_idx: int) -> str:
    if sample.p_minus is not None:
        return _safe_name(sample.p_minus.description[:50])
    return f"sample{sample_idx}"


def _save_tree_html(root: MCTSNode, filepath: str) -> None:
    """Save a color-coded MCTS tree to HTML (no MCTSSearch instance needed)."""
    from rich.console import Console
    from rich.terminal_theme import MONOKAI

    lines: list[str] = []
    MCTSSearch._collect_tree_lines(root, "", True, None, True, lines)
    console = Console(
        record=True,
        highlight=False,
        width=310,
        force_jupyter=False,
        file=open(os.devnull, "w"),
    )
    console.print("\n".join(lines))
    html = console.export_html(theme=MONOKAI, inline_styles=True)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.debug(f"Tree HTML → {filepath}")


def _mcts_config_matches(existing: dict, current: dict) -> tuple[bool, list[str]]:
    diffs = []
    for k in sorted(set(existing) | set(current)):
        v_old, v_new = existing.get(k), current.get(k)
        if v_old != v_new:
            diffs.append(f"  {k}: saved={v_old!r} vs current={v_new!r}")
    return len(diffs) == 0, diffs


def _collect_green_leaves(root: MCTSNode) -> list[MCTSNode]:
    leaves: list[MCTSNode] = []

    def _walk(node: MCTSNode) -> None:
        if node.is_leaf():
            if node.color == "green":
                leaves.append(node)
            return
        for child in node.children:
            _walk(child)

    _walk(root)
    return leaves


# ══════════════════════════════════════════════════════════════════════
# Step 1 — MCTS
# ══════════════════════════════════════════════════════════════════════
def run_or_load_mcts(
    mt: ModelandTokenizer,
    tp: TargetPredicate,
    tree_root_dir: str,
    args,
) -> MCTSSearchResults_for_TargetPredicate:
    """Run MCTS for every (sample × prefill) of tp, or load from cache.

    Cache: {tree_root_dir}/mcts.json
    HTML trees: {tree_root_dir}/{p_minus_desc}/{safe_prefill}/mcts_tree.html
    """
    model_short = mt.name.split("/")[-1]
    cache_path = os.path.join(tree_root_dir, "mcts.json")

    current_config = dict(
        iterations=args.mcts_iterations,
        max_depth=args.mcts_max_depth,
        top_k=args.mcts_top_k,
        prob_threshold=args.mcts_prob_threshold,
        depth_first=args.depth_first,
        **MCTS_DIVERSITY_CONFIG,
    )

    # ── Try cache ──────────────────────────────────────────────────────
    reuse_cached = False
    if os.path.exists(cache_path) and not args.force_rerun_mcts:
        cached = MCTSSearchResults_for_TargetPredicate.load(cache_path)
        matches, diffs = _mcts_config_matches(cached.mcts_config, current_config)
        if matches:
            n_trees = sum(len(sr._trees) for sr in cached.sample_results)
            if n_trees > 0:
                logger.info(f"Cache hit for '{tp.name}' ({n_trees} trees).")
                reuse_cached = True
            else:
                logger.warning("Cached JSON has no in-memory trees — re-running MCTS.")
        else:
            print(f"\nWARNING: Cached MCTS config differs for '{tp.name}':")
            for d in diffs:
                print(d)
            if not args.ask_override_permission:
                print("--no-ask_override_permission set. Re-running MCTS.")
            else:
                ans = input("Reuse cached results anyway? [y/N/q]: ").strip().lower()
                if ans == "q":
                    sys.exit(0)
                reuse_cached = ans == "y"

    if reuse_cached:
        tp_results = cached
        # Fill in any missing HTML trees from in-memory cache
        for sample_idx, sr in enumerate(tp_results.sample_results):
            sample = tp.samples[sample_idx]
            p_minus_name = _p_minus_dir_name(sample, sample_idx)
            for run in sr.runs:
                safe_pval = _safe_name(run.prefill) if run.prefill else "empty"
                html_path = os.path.join(
                    tree_root_dir, p_minus_name, safe_pval, "mcts_tree.html"
                )
                if not os.path.exists(html_path):
                    root = sr._trees.get(run.prefix)
                    if root is not None:
                        _save_tree_html(root, html_path)
                    else:
                        logger.warning(
                            f"No in-memory tree for {run.prefix[:30]!r} — "
                            "re-run with --force_rerun_mcts to generate HTML."
                        )
        return tp_results

    # ── Run MCTS from scratch ──────────────────────────────────────────
    tp_results = MCTSSearchResults_for_TargetPredicate(
        target_predicate_name=tp.name,
        model_name=model_short,
        mcts_config=current_config,
    )

    for sample_idx, sample in enumerate(tp.samples):
        p_minus_name = _p_minus_dir_name(sample, sample_idx)
        print(f"\n{'=' * 70}")
        print(f"Sample {sample_idx}: fn={sample.function_name}  P-={p_minus_name}")
        print(f"{'=' * 70}")

        prefills: list[tuple[str, str]] = [
            (pval, ptype) for ptype, plist in sample.prefills.items() for pval in plist
        ]

        if supports_chat_template(mt):
            prompts = [
                apply_chat_template(
                    mt=mt,
                    user_query=sample.instruction.strip(),
                    incomplete_code=sample.code + pval,
                )
                for pval, _ in prefills
            ]
        else:
            prompts = [
                sample.instruction + "<code>" + sample.code + pval
                for pval, _ in prefills
            ]

        sr = MCTSSearchResults_for_MultiPredicateSample(sample=sample)

        for (pval, ptype), prompt in zip(prefills, prompts):
            prefix = sample.code + pval
            safe_pval = _safe_name(pval) if pval else "empty"
            print(f"\n  Prefill ({ptype}): {pval!r}")

            search = MCTSSearch(
                mt=mt,
                prompt=prompt,
                max_depth=args.mcts_max_depth,
                top_k=args.mcts_top_k,
                prob_threshold=args.mcts_prob_threshold,
                **MCTS_DIVERSITY_CONFIG,
            )
            search.run(n_iterations=args.mcts_iterations, depth_first=args.depth_first)

            counts = evaluate_mcts_tree(
                root=search.root,
                prefix=prefix,
                unit_tests=sample.unit_tests,
                function_name=sample.function_name or "func",
                terminal_only=True,
            )
            print(f"  Eval: {counts}")

            run = MCTSRunResult(
                prefill=pval,
                prefill_type=ptype,
                prompt=prompt,
                prefix=prefix,
                mcts_config=current_config,
                mcts_stats=search.get_stats(),
                eval_counts=counts,
                green_completions=[
                    node.get_path_text() for node in _collect_green_leaves(search.root)
                ],
                tree=search.root.to_dict(),
            )
            sr.runs.append(run)
            sr._trees[prefix] = search.root

            html_path = os.path.join(
                tree_root_dir, p_minus_name, safe_pval, "mcts_tree.html"
            )
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            search.save_tree(html_path, show_colors=True)
            logger.info(f"  HTML → {html_path}")

        tp_results.sample_results.append(sr)

    os.makedirs(tree_root_dir, exist_ok=True)
    tp_results.save(cache_path)
    print(f"\nSaved MCTSSearchResults → {cache_path}")
    return tp_results


# ══════════════════════════════════════════════════════════════════════
# Step 2 — Anchor tokens
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
# Full pipeline for one TargetPredicate
# ══════════════════════════════════════════════════════════════════════
def process_predicate(
    mt: ModelandTokenizer,
    tp: TargetPredicate,
    predicate_idx: int,
    args,
) -> None:
    model_short = mt.name.split("/")[-1]
    tree_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.save_dir, model_short, tp.name, "trees"
    )

    print(f"\n{'#' * 70}")
    print(f"# Predicate [{predicate_idx}]: {tp.name}")
    print(f"# P+: {tp.p_plus.description}")
    print(f"# Samples: {len(tp.samples)}")
    print(f"# Trees dir: {tree_dir}")
    print(f"{'#' * 70}")

    # 1) MCTS
    tp_results = run_or_load_mcts(mt=mt, tp=tp, tree_root_dir=tree_dir, args=args)

    total_green = sum(
        run.eval_counts.get("green", 0)
        for sr in tp_results.sample_results
        for run in sr.runs
    )
    print(f"\nTotal green leaves across all runs: {total_green}")
    if total_green == 0:
        logger.error(f"No green leaves for '{tp.name}'. Anchor extraction skipped.")
        return

    # 2) Anchor tokens
    ensure_anchor_tokens(
        tp=tp,
        tp_results=tp_results,
        force_reextract=args.force_reextract_anchors,
    )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCTS + anchor extraction for multi-predicate samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="multi_probe",
        help="Root output dir. Trees saved as {save_dir}/{model}/{predicate.name}/trees/",
    )
    parser.add_argument(
        "--predicate_indices",
        type=int,
        nargs="+",
        default=None,
        help="Indices of TargetPredicates to process (default: all)",
    )

    # MCTS
    parser.add_argument("--mcts_iterations", type=int, default=1500)
    parser.add_argument("--mcts_max_depth", type=int, default=32)
    parser.add_argument("--mcts_top_k", type=int, default=8)
    parser.add_argument("--mcts_prob_threshold", type=float, default=0.01)
    parser.add_argument("--depth_first", action="store_true", default=True)
    parser.add_argument(
        "--no_depth_first",
        action="store_false",
        dest="depth_first",
        help="Use breadth-first MCTS",
    )
    parser.add_argument(
        "--force_rerun_mcts",
        action="store_true",
        default=False,
        help="Re-run MCTS even if cached JSON exists",
    )
    parser.add_argument(
        "--ask_override_permission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt before overriding cached MCTS with a different config",
    )

    # Anchor tokens
    parser.add_argument(
        "--force_reextract_anchors",
        action="store_true",
        default=False,
        help="Re-extract anchor tokens even if already populated",
    )

    logging_utils.add_logging_args(parser)
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(42)

    print(f"\n{'=' * 70}")
    print("Multi-Predicate MCTS Expansion")
    print(f"{'=' * 70}")
    print(f"Model:    {args.model_name}")
    print(f"Save dir: {args.save_dir}")
    print(
        f"MCTS:     iters={args.mcts_iterations}, depth={args.mcts_max_depth}, "
        f"top_k={args.mcts_top_k}, threshold={args.mcts_prob_threshold}"
    )
    print(f"{'=' * 70}\n")

    logger.info(f"Loading model: {args.model_name}")
    mt = ModelandTokenizer(
        model_key=args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(f"Model loaded: {mt.name}, n_layer={mt.n_layer}")

    dataset = MultiPredicateDataset.load_dataset()
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
    print(f"Done. Processed {len(indices)} predicate(s).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
