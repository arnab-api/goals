"""
Test steerability of extracted goal directions.

For each sample, steers generation using contrastive and mean-diff directions
across a grid of coefficients. Evaluates pass rates and saves results.

Usage:
    python -m scripts.test_causality --model Qwen/Qwen2.5-Coder-32B-Instruct --goal_dir_path probe
    python -m scripts.test_causality --model Qwen/Qwen2.5-Coder-14B-Instruct --goal_dir_path probe --n_generations 50
"""

import argparse
import json
import logging
import os
import sys

# Must be set before importing torch so CUDA sees only the requested devices.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.functional import (  # noqa: E402
    PatchSpec,
    free_gpu_cache,
    generate_with_patch,
    get_hs,
)
from src.goals.data import Code, CodeGenerations, IncompleteCodeDataset  # noqa: E402
from src.goals.eval import run_evaluation  # noqa: E402
from src.goals.functional import (  # noqa: E402
    apply_chat_template,
    supports_chat_template,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Steering prefills (edit here to change; not exposed as CLI args)
# NOTE: coeff is tuned independently per sample AND per prefill.
# This is not ideal — see CLAUDE.md "Known Issues / TODOs".
# ──────────────────────────────────────────────────────────────────────
PREFILLS = ["(3", "1 -", "5", "3 +"]
# PREFILLS = ["1 -"]

# Coefficient grid search range
COEFF_GRID = list(np.arange(0.02, 0.21, 0.02))


# ══════════════════════════════════════════════════════════════════════
# Steering helpers
# ══════════════════════════════════════════════════════════════════════
def build_steer_patches(
    mt: ModelandTokenizer,
    directions: dict[int, torch.Tensor],
    hs: dict[tuple[str, int], torch.Tensor],
    layer_from: int,
    layer_to: int,
    coeff: float,
) -> list[PatchSpec]:
    """Build PatchSpec list for steering with given directions and coeff."""
    patches = []
    for layer_idx in range(layer_from, layer_to):
        layer_name = mt.layer_name_format.format(layer_idx)
        if layer_idx not in directions:
            continue
        h_orig = hs[(layer_name, -1)]
        goal_dir = directions[layer_idx].to(h_orig.device)
        scaled_goal_dir = goal_dir * coeff * (h_orig.norm() / goal_dir.norm())
        patches.append(
            PatchSpec(
                location=(layer_name, -1),
                patch=scaled_goal_dir,
                strategy="add",
                on_pass="all",
            )
        )
    return patches


def generate_and_evaluate(
    mt: ModelandTokenizer,
    sample,
    prefill: str,
    prompt: str,
    n_generations: int,
    patches: list[PatchSpec] | None = None,
    label: str = "",
) -> CodeGenerations:
    """Generate with optional patches, evaluate, and return the full CodeGenerations."""
    logger.debug(
        f"Generating {n_generations} samples{f' ({label})' if label else ''}..."
    )
    outputs = generate_with_patch(
        mt=mt,
        inputs=prompt,
        patches=patches,
        do_sample=True,
        n_gen_per_prompt=n_generations,
        remove_prefix=True,
        max_new_tokens=32,
    )
    logger.debug(f"Generation done. Evaluating {len(outputs)} outputs...")

    generations = CodeGenerations(
        sample=sample,
        generations=[
            Code(
                prefill=prefill,
                prefill_type="steering_eval",
                full_prompt=prompt,
                lm_generation=gen,
            )
            for gen in outputs
        ],
    )

    eval_results = run_evaluation(
        result=generations,
        function_name=sample.function_name,
        code_stop_tag="</code>",
    )

    pass_rate = eval_results.summary.pass_count / eval_results.summary.total
    logger.debug(
        f"Evaluation done: {eval_results.summary.pass_count}/{eval_results.summary.total} "
        f"passed ({pass_rate:.3f})"
    )

    return eval_results


def log_steering_error(
    e: Exception,
    error_log_path: str,
    model_name: str,
    sample_idx: int,
    prefill: str,
    coeff: float,
    layer_from: int,
    layer_to: int,
    direction_type: str,
) -> None:
    """Log a nnsight steering error with full config to errors.log and logger.error."""
    import traceback as tb_module

    error_config = {
        "model": model_name,
        "sample_idx": sample_idx,
        "prefill": prefill,
        "coeff": round(coeff, 6),
        "layer_from": layer_from,
        "layer_to": layer_to,
        "direction_type": direction_type,
        "error_type": type(e).__name__,
        "error_message": str(e),
    }

    logger.error(
        f"Steering error [{direction_type}] sample={sample_idx}, prefill={prefill!r}, "
        f"coeff={coeff:.3f}, layers=({layer_from},{layer_to}): "
        f"{type(e).__name__}: {e}"
    )

    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "a") as f:
        f.write(json.dumps(error_config) + "\n")
        f.write(tb_module.format_exc() + "\n---\n")


def save_eval_table(eval_results: CodeGenerations, filepath: str) -> None:
    """Save the generation-evaluation table as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(eval_results.to_dict(), f, indent=2)
    logger.debug(f"Saved eval table to {filepath}")


# ══════════════════════════════════════════════════════════════════════
# Process one sample
# ══════════════════════════════════════════════════════════════════════
def process_sample(
    mt: ModelandTokenizer,
    sample,
    sample_idx: int,
    goal_sample_dir: str,
    results_sample_dir: str,
    error_log_path: str,
    args,
) -> dict | None:
    """
    Run steering evaluation for one sample across all prefills and coefficients.

    Args:
        goal_sample_dir: Where to load directions from (goal_dir.py output).
        results_sample_dir: Where to save causality results.

    Returns a results dict for this sample, or None if directions not found.
    """
    # --- Load directions ---
    contrastive_path = os.path.join(goal_sample_dir, "contrastive_directions.pt")
    mean_diff_path = os.path.join(goal_sample_dir, "mean_diff_directions.pt")

    if not os.path.exists(contrastive_path) or not os.path.exists(mean_diff_path):
        logger.warning(
            f"Directions not found in {goal_sample_dir}. Skipping sample {sample_idx}."
        )
        return None

    logger.info(f"Loading contrastive directions from {contrastive_path}")
    contrastive_directions = torch.load(
        contrastive_path, weights_only=False, map_location="cuda:0"
    )
    logger.info(f"Loading mean-diff directions from {mean_diff_path}")
    mean_diff_directions = torch.load(
        mean_diff_path, weights_only=False, map_location="cuda:0"
    )

    layer_from = args.layer_from
    layer_to = args.layer_to if args.layer_to is not None else mt.n_layer
    logger.info(f"Steering layers: {layer_from} to {layer_to}")

    os.makedirs(results_sample_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Sample {sample_idx}: {sample.function_name or 'func'}")
    print(f'"{sample.code}"')
    print(f"# Directions from: {goal_sample_dir}")
    print(f"# Results to:      {results_sample_dir}")
    print(f"# Layers: {layer_from} to {layer_to}")
    print(f"# Prefills: {PREFILLS}")
    print(f"# Coeff grid: {[round(c, 3) for c in COEFF_GRID]}")
    print(f"{'#' * 70}")

    sample_results = {
        "sample_idx": sample_idx,
        "function_name": sample.function_name or "func",
        "layer_from": layer_from,
        "layer_to": layer_to,
        "n_generations": args.n_generations,
        "prefill_results": {},
    }

    for pfx_i, prefill in enumerate(PREFILLS):
        safe_prefill = prefill.replace(" ", "_").replace("/", "_") or "empty"
        prefill_dir = os.path.join(results_sample_dir, f"prefill_{safe_prefill}")
        os.makedirs(prefill_dir, exist_ok=True)

        logger.info(
            f"Sample {sample_idx}, prefill {pfx_i + 1}/{len(PREFILLS)}: {prefill!r}"
        )
        print(f"\n  Prefill: {prefill!r}")
        print(f"  {'-' * 50}")

        # --- Build prompt ---
        if supports_chat_template(mt):
            prompt = apply_chat_template(
                mt=mt,
                user_query=sample.instruction.strip(),
                incomplete_code=sample.code + prefill,
            )
        else:
            prompt = sample.instruction + "<code>" + sample.code + prefill

        # --- Get original hidden states for scaling ---
        logger.debug(f"Extracting hidden states for prefill {prefill!r}...")
        tokenized = prepare_input(prompts=prompt, tokenizer=mt)
        hs = get_hs(
            mt=mt,
            input=tokenized,
            locations=[
                (mt.layer_name_format.format(li), -1)
                for li in range(layer_from, layer_to)
            ],
        )

        # --- Baseline (no steering) ---
        logger.info(f"Running baseline (no steering) for prefill {prefill!r}...")
        baseline_eval = generate_and_evaluate(
            mt=mt,
            sample=sample,
            prefill=prefill,
            prompt=prompt,
            n_generations=args.n_generations,
            patches=None,
            label="baseline",
        )
        baseline_pass_rate = (
            baseline_eval.summary.pass_count / baseline_eval.summary.total
        )
        logger.info(f"Baseline pass rate: {baseline_pass_rate:.3f}")
        print(f"    Baseline pass rate: {baseline_pass_rate:.3f}")

        # Save baseline eval table
        save_eval_table(baseline_eval, os.path.join(prefill_dir, "baseline.json"))

        # --- Grid search over coeff ---
        contrastive_by_coeff = {}
        mean_diff_by_coeff = {}

        logger.info(
            f"Starting coeff grid search ({len(COEFF_GRID)} values) "
            f"for prefill {prefill!r}..."
        )
        for coeff_i, coeff in enumerate(COEFF_GRID):
            coeff_key = f"{coeff:.3f}"
            logger.info(
                f"Coeff {coeff_i + 1}/{len(COEFF_GRID)}: {coeff_key} "
                f"(prefill {prefill!r}, sample {sample_idx})"
            )

            # Contrastive steering
            contrastive_patches = build_steer_patches(
                mt=mt,
                directions=contrastive_directions,
                hs=hs,
                layer_from=layer_from,
                layer_to=layer_to,
                coeff=coeff,
            )
            try:
                contrastive_eval = generate_and_evaluate(
                    mt=mt,
                    sample=sample,
                    prefill=prefill,
                    prompt=prompt,
                    n_generations=args.n_generations,
                    patches=contrastive_patches,
                    label=f"contrastive coeff={coeff_key}",
                )
                contrastive_pass_rate = (
                    contrastive_eval.summary.pass_count / contrastive_eval.summary.total
                )
                contrastive_by_coeff[coeff_key] = contrastive_pass_rate
                save_eval_table(
                    contrastive_eval,
                    os.path.join(prefill_dir, f"contrastive_coeff_{coeff_key}.json"),
                )
            except Exception as e:
                log_steering_error(
                    e=e,
                    error_log_path=error_log_path,
                    model_name=args.model,
                    sample_idx=sample_idx,
                    prefill=prefill,
                    coeff=coeff,
                    layer_from=layer_from,
                    layer_to=layer_to,
                    direction_type="contrastive",
                )
                contrastive_pass_rate = None

            # Mean-diff steering
            mean_diff_patches = build_steer_patches(
                mt=mt,
                directions=mean_diff_directions,
                hs=hs,
                layer_from=layer_from,
                layer_to=layer_to,
                coeff=coeff,
            )
            try:
                mean_diff_eval = generate_and_evaluate(
                    mt=mt,
                    sample=sample,
                    prefill=prefill,
                    prompt=prompt,
                    n_generations=args.n_generations,
                    patches=mean_diff_patches,
                    label=f"mean_diff coeff={coeff_key}",
                )
                mean_diff_pass_rate = (
                    mean_diff_eval.summary.pass_count / mean_diff_eval.summary.total
                )
                mean_diff_by_coeff[coeff_key] = mean_diff_pass_rate
                save_eval_table(
                    mean_diff_eval,
                    os.path.join(prefill_dir, f"mean_diff_coeff_{coeff_key}.json"),
                )
            except Exception as e:
                log_steering_error(
                    e=e,
                    error_log_path=error_log_path,
                    model_name=args.model,
                    sample_idx=sample_idx,
                    prefill=prefill,
                    coeff=coeff,
                    layer_from=layer_from,
                    layer_to=layer_to,
                    direction_type="mean_diff",
                )
                mean_diff_pass_rate = None

            print(
                f"    coeff={coeff:.3f}: "
                f"contrastive={'ERROR' if contrastive_pass_rate is None else f'{contrastive_pass_rate:.3f}'}, "
                f"mean_diff={'ERROR' if mean_diff_pass_rate is None else f'{mean_diff_pass_rate:.3f}'}"
            )

        # --- Find best coeff for this prefill ---
        if contrastive_by_coeff:
            best_contrastive_coeff = max(
                contrastive_by_coeff, key=contrastive_by_coeff.get
            )
            best_contrastive_pass_rate = contrastive_by_coeff[best_contrastive_coeff]
        else:
            logger.warning(
                f"All contrastive coeff runs errored for prefill {prefill!r}."
            )
            best_contrastive_coeff = None
            best_contrastive_pass_rate = None

        if mean_diff_by_coeff:
            best_mean_diff_coeff = max(mean_diff_by_coeff, key=mean_diff_by_coeff.get)
            best_mean_diff_pass_rate = mean_diff_by_coeff[best_mean_diff_coeff]
        else:
            logger.warning(f"All mean_diff coeff runs errored for prefill {prefill!r}.")
            best_mean_diff_coeff = None
            best_mean_diff_pass_rate = None

        prefill_result = {
            "baseline_pass_rate": baseline_pass_rate,
            "contrastive_by_coeff": contrastive_by_coeff,
            "mean_diff_by_coeff": mean_diff_by_coeff,
            "best_contrastive_coeff": best_contrastive_coeff,
            "best_contrastive_pass_rate": best_contrastive_pass_rate,
            "best_mean_diff_coeff": best_mean_diff_coeff,
            "best_mean_diff_pass_rate": best_mean_diff_pass_rate,
        }
        sample_results["prefill_results"][prefill] = prefill_result

        if best_contrastive_pass_rate is not None:
            print(
                f"    Best contrastive: coeff={best_contrastive_coeff}, "
                f"pass_rate={best_contrastive_pass_rate:.3f}"
            )
        else:
            print("    Best contrastive: all runs errored.")

        if best_mean_diff_pass_rate is not None:
            print(
                f"    Best mean_diff:   coeff={best_mean_diff_coeff}, "
                f"pass_rate={best_mean_diff_pass_rate:.3f}"
            )
        else:
            print("    Best mean_diff:   all runs errored.")

    # --- Save aggregated per-sample results ---
    sample_agg_path = os.path.join(results_sample_dir, "sample_results.json")
    with open(sample_agg_path, "w") as f:
        json.dump(sample_results, f, indent=2)
    logger.info(f"Saved per-sample results to {sample_agg_path}")
    print(f"\n  Saved per-sample results to {sample_agg_path}")

    # --- Log summary for this sample ---
    print(f"\n  === Sample {sample_idx} Summary ===")
    for prefill, pr in sample_results["prefill_results"].items():
        c_rate = pr["best_contrastive_pass_rate"]
        m_rate = pr["best_mean_diff_pass_rate"]
        print(
            f"    {prefill!r:>8}: baseline={pr['baseline_pass_rate']:.3f}, "
            f"contrastive={'ERROR' if c_rate is None else f'{c_rate:.3f}'} "
            f"(coeff={pr['best_contrastive_coeff']}), "
            f"mean_diff={'ERROR' if m_rate is None else f'{m_rate:.3f}'} "
            f"(coeff={pr['best_mean_diff_coeff']})"
        )

    return sample_results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test steerability of extracted goal directions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--goal_dir_path",
        type=str,
        default="probe",
        help="Subdirectory under DEFAULT_RESULTS_DIR where goal directions are saved "
        "(i.e., the --results_dir used in goal_dir.py)",
    )
    parser.add_argument(
        "--causality_results_dir",
        type=str,
        default="causality",
        help="Subdirectory under DEFAULT_RESULTS_DIR for saving causality results",
    )
    parser.add_argument(
        "--layer_from",
        type=int,
        default=0,
        help="First layer to apply steering (inclusive)",
    )
    parser.add_argument(
        "--layer_to",
        type=int,
        default=None,
        help="Last layer to apply steering (exclusive). Default: mt.n_layer",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=100,
        help="Number of generations to sample per prompt",
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to process. Default: auto-discover from goal_dir_path",
    )

    logging_utils.add_logging_args(parser)

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    logging_utils.configure(args)
    set_seed(42)

    model_short_name = args.model.split("/")[-1]

    # Directory where goal directions were saved (by goal_dir.py)
    goal_model_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.goal_dir_path, model_short_name
    )
    # Directory where causality results will be saved
    results_model_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.causality_results_dir, model_short_name
    )
    os.makedirs(results_model_dir, exist_ok=True)

    # Shared error log for nnsight/steering failures across all samples
    error_log_path = os.path.join(results_model_dir, "errors.log")

    # --- Print config ---
    print(f"\n{'=' * 70}")
    print("Steering Causality Test")
    print(f"{'=' * 70}")
    print(f"Model:           {args.model}")
    print(f"Goal dir:        {goal_model_dir}")
    print(f"Results dir:     {results_model_dir}")
    print(f"Layers:          {args.layer_from} to {args.layer_to or 'n_layer'}")
    print(f"N generations:   {args.n_generations}")
    print(f"Prefills:        {PREFILLS}")
    print(f"Coeff grid:      {[round(c, 3) for c in COEFF_GRID]}")
    print(f"{'=' * 70}\n")

    # --- Discover sample directories from goal_dir output ---
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        if not os.path.isdir(goal_model_dir):
            logger.error(f"Goal model directory not found: {goal_model_dir}")
            sys.exit(1)
        sample_indices = sorted(
            int(d.split("_")[1])
            for d in os.listdir(goal_model_dir)
            if d.startswith("sample_")
            and os.path.isdir(os.path.join(goal_model_dir, d))
        )

    if not sample_indices:
        logger.error(f"No sample directories found in {goal_model_dir}")
        sys.exit(1)

    logger.info(f"Samples to process: {sample_indices}")

    # --- Load model ---
    logger.info(f"Loading model: {args.model}")
    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(f"Model loaded: {mt.name}, n_layer={mt.n_layer}, n_embd={mt.n_embd}")

    # --- Load dataset ---
    eval_dataset = IncompleteCodeDataset.load_dataset()
    logger.info(f"Dataset loaded: {len(eval_dataset)} samples")

    # --- Process each sample ---
    all_sample_results = {}

    for si, sample_idx in enumerate(sample_indices):
        logger.info(
            f"Processing sample {sample_idx} ({si + 1}/{len(sample_indices)})..."
        )
        sample = eval_dataset[sample_idx]
        goal_sample_dir = os.path.join(goal_model_dir, f"sample_{sample_idx}")
        results_sample_dir = os.path.join(results_model_dir, f"sample_{sample_idx}")

        result = process_sample(
            mt=mt,
            sample=sample,
            sample_idx=sample_idx,
            goal_sample_dir=goal_sample_dir,
            results_sample_dir=results_sample_dir,
            error_log_path=error_log_path,
            args=args,
        )

        if result is not None:
            all_sample_results[str(sample_idx)] = result

        # Free GPU cache between samples
        free_gpu_cache()
        logger.info(f"GPU cache cleared after sample {sample_idx}.")
        print(f"GPU cache cleared after sample {sample_idx}.")

    # --- Save aggregated model-level results ---
    aggregated_path = os.path.join(results_model_dir, "causality_results.json")
    with open(aggregated_path, "w") as f:
        json.dump(all_sample_results, f, indent=2)
    logger.info(f"Saved aggregated model-level results to {aggregated_path}")
    print(f"\nSaved aggregated results to {aggregated_path}")

    # --- Print final summary ---
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    for sidx, sr in all_sample_results.items():
        print(f"\nSample {sidx} ({sr['function_name']}):")
        for prefill, pr in sr["prefill_results"].items():
            c_rate = pr["best_contrastive_pass_rate"]
            m_rate = pr["best_mean_diff_pass_rate"]
            baseline = pr["baseline_pass_rate"]
            c_str = (
                f"ERROR (coeff={pr['best_contrastive_coeff']})"
                if c_rate is None
                else f"{c_rate:.3f} ({c_rate - baseline:+.3f}, coeff={pr['best_contrastive_coeff']})"
            )
            m_str = (
                f"ERROR (coeff={pr['best_mean_diff_coeff']})"
                if m_rate is None
                else f"{m_rate:.3f} ({m_rate - baseline:+.3f}, coeff={pr['best_mean_diff_coeff']})"
            )
            print(
                f"  {prefill!r:>8}: baseline={baseline:.3f} | "
                f"contrastive={c_str} | "
                f"mean_diff={m_str}"
            )

    print(f"\n{'=' * 70}")
    print(f"All done! Processed {len(all_sample_results)} samples.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
