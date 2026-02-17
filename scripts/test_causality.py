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

from src.functional import PatchSpec, free_gpu_cache, generate_with_patch, get_hs  # noqa: E402
from src.goals.data import Code, CodeGenerations, IncompleteCodeDataset  # noqa: E402
from src.goals.eval import run_evaluation  # noqa: E402
from src.goals.functional import apply_chat_template, supports_chat_template  # noqa: E402
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
        goal_dir = directions[layer_idx]
        scaled_goal_dir = goal_dir * coeff * (h_orig.norm() / goal_dir.norm())
        patches.append(
            PatchSpec(location=(layer_name, -1), patch=scaled_goal_dir, strategy="add")
        )
    return patches


def generate_and_evaluate(
    mt: ModelandTokenizer,
    sample,
    prefill: str,
    prompt: str,
    n_generations: int,
    patches: list[PatchSpec] | None = None,
) -> float:
    """Generate with optional patches and return pass rate."""
    outputs = generate_with_patch(
        mt=mt,
        inputs=prompt,
        patches=patches,
        patch_strategy="add" if patches else "replace",
        patch_at_all_generations=True if patches else False,
        do_sample=True,
        n_gen_per_prompt=n_generations,
        remove_prefix=True,
        max_new_tokens=32,
    )

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

    return eval_results.summary.pass_count / eval_results.summary.total


# ══════════════════════════════════════════════════════════════════════
# Process one sample
# ══════════════════════════════════════════════════════════════════════
def process_sample(
    mt: ModelandTokenizer,
    sample,
    sample_idx: int,
    sample_dir: str,
    args,
) -> dict | None:
    """
    Run steering evaluation for one sample across all prefills and coefficients.

    Returns a results dict for this sample, or None if directions not found.
    """
    # --- Load directions ---
    contrastive_path = os.path.join(sample_dir, "contrastive_directions.pt")
    mean_diff_path = os.path.join(sample_dir, "mean_diff_directions.pt")

    if not os.path.exists(contrastive_path) or not os.path.exists(mean_diff_path):
        logger.warning(
            f"Directions not found in {sample_dir}. Skipping sample {sample_idx}."
        )
        return None

    contrastive_directions = torch.load(contrastive_path, weights_only=False)
    mean_diff_directions = torch.load(mean_diff_path, weights_only=False)

    layer_from = args.layer_from
    layer_to = args.layer_to if args.layer_to is not None else mt.n_layer

    print(f"\n{'#' * 70}")
    print(f"# Sample {sample_idx}: {sample.function_name or 'func'}")
    print(f"# Directions from: {sample_dir}")
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

    for prefill in PREFILLS:
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
        tokenized = prepare_input(prompts=prompt, tokenizer=mt)
        hs = get_hs(
            mt=mt,
            input=tokenized,
            locations=[
                (mt.layer_name_format.format(l), -1)
                for l in range(layer_from, layer_to)
            ],
        )

        # --- Baseline (no steering) ---
        baseline_pass_rate = generate_and_evaluate(
            mt=mt,
            sample=sample,
            prefill=prefill,
            prompt=prompt,
            n_generations=args.n_generations,
            patches=None,
        )
        print(f"    Baseline pass rate: {baseline_pass_rate:.3f}")

        # --- Grid search over coeff ---
        contrastive_by_coeff = {}
        mean_diff_by_coeff = {}

        for coeff in COEFF_GRID:
            coeff_key = f"{coeff:.3f}"

            # Contrastive steering
            contrastive_patches = build_steer_patches(
                mt=mt,
                directions=contrastive_directions,
                hs=hs,
                layer_from=layer_from,
                layer_to=layer_to,
                coeff=coeff,
            )
            contrastive_pass_rate = generate_and_evaluate(
                mt=mt,
                sample=sample,
                prefill=prefill,
                prompt=prompt,
                n_generations=args.n_generations,
                patches=contrastive_patches,
            )
            contrastive_by_coeff[coeff_key] = contrastive_pass_rate

            # Mean-diff steering
            mean_diff_patches = build_steer_patches(
                mt=mt,
                directions=mean_diff_directions,
                hs=hs,
                layer_from=layer_from,
                layer_to=layer_to,
                coeff=coeff,
            )
            mean_diff_pass_rate = generate_and_evaluate(
                mt=mt,
                sample=sample,
                prefill=prefill,
                prompt=prompt,
                n_generations=args.n_generations,
                patches=mean_diff_patches,
            )
            mean_diff_by_coeff[coeff_key] = mean_diff_pass_rate

            print(
                f"    coeff={coeff:.3f}: "
                f"contrastive={contrastive_pass_rate:.3f}, "
                f"mean_diff={mean_diff_pass_rate:.3f}"
            )

        # --- Find best coeff for this prefill ---
        best_contrastive_coeff = max(contrastive_by_coeff, key=contrastive_by_coeff.get)
        best_mean_diff_coeff = max(mean_diff_by_coeff, key=mean_diff_by_coeff.get)

        prefill_result = {
            "baseline_pass_rate": baseline_pass_rate,
            "contrastive_by_coeff": contrastive_by_coeff,
            "mean_diff_by_coeff": mean_diff_by_coeff,
            "best_contrastive_coeff": best_contrastive_coeff,
            "best_contrastive_pass_rate": contrastive_by_coeff[best_contrastive_coeff],
            "best_mean_diff_coeff": best_mean_diff_coeff,
            "best_mean_diff_pass_rate": mean_diff_by_coeff[best_mean_diff_coeff],
        }
        sample_results["prefill_results"][prefill] = prefill_result

        print(
            f"    Best contrastive: coeff={best_contrastive_coeff}, "
            f"pass_rate={contrastive_by_coeff[best_contrastive_coeff]:.3f}"
        )
        print(
            f"    Best mean_diff:   coeff={best_mean_diff_coeff}, "
            f"pass_rate={mean_diff_by_coeff[best_mean_diff_coeff]:.3f}"
        )

    # --- Save per-sample results ---
    results_path = os.path.join(sample_dir, "causality_results.json")
    with open(results_path, "w") as f:
        json.dump(sample_results, f, indent=2)
    print(f"\n  Saved per-sample results to {results_path}")

    # --- Log aggregated summary for this sample ---
    print(f"\n  === Sample {sample_idx} Summary ===")
    for prefill, pr in sample_results["prefill_results"].items():
        print(
            f"    {prefill!r:>8}: baseline={pr['baseline_pass_rate']:.3f}, "
            f"contrastive={pr['best_contrastive_pass_rate']:.3f} "
            f"(coeff={pr['best_contrastive_coeff']}), "
            f"mean_diff={pr['best_mean_diff_pass_rate']:.3f} "
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
    model_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.goal_dir_path, model_short_name
    )

    # --- Print config ---
    print(f"\n{'=' * 70}")
    print("Steering Causality Test")
    print(f"{'=' * 70}")
    print(f"Model:           {args.model}")
    print(f"Goal dir:        {model_dir}")
    print(f"Layers:          {args.layer_from} to {args.layer_to or 'n_layer'}")
    print(f"N generations:   {args.n_generations}")
    print(f"Prefills:        {PREFILLS}")
    print(f"Coeff grid:      {[round(c, 3) for c in COEFF_GRID]}")
    print(f"{'=' * 70}\n")

    # --- Discover sample directories ---
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        # Auto-discover sample_* dirs
        if not os.path.isdir(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
        sample_indices = sorted(
            int(d.split("_")[1])
            for d in os.listdir(model_dir)
            if d.startswith("sample_") and os.path.isdir(os.path.join(model_dir, d))
        )

    if not sample_indices:
        logger.error(f"No sample directories found in {model_dir}")
        sys.exit(1)

    print(f"Samples to process: {sample_indices}\n")

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

    for sample_idx in sample_indices:
        sample = eval_dataset[sample_idx]
        sample_dir = os.path.join(model_dir, f"sample_{sample_idx}")

        result = process_sample(
            mt=mt,
            sample=sample,
            sample_idx=sample_idx,
            sample_dir=sample_dir,
            args=args,
        )

        if result is not None:
            all_sample_results[str(sample_idx)] = result

        # Free GPU cache between samples
        free_gpu_cache()
        print(f"GPU cache cleared after sample {sample_idx}.")

    # --- Save aggregated results ---
    aggregated_path = os.path.join(model_dir, "causality_results.json")
    with open(aggregated_path, "w") as f:
        json.dump(all_sample_results, f, indent=2)
    print(f"\nSaved aggregated results to {aggregated_path}")

    # --- Print final summary ---
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    for sidx, sr in all_sample_results.items():
        print(f"\nSample {sidx} ({sr['function_name']}):")
        for prefill, pr in sr["prefill_results"].items():
            delta_c = pr["best_contrastive_pass_rate"] - pr["baseline_pass_rate"]
            delta_m = pr["best_mean_diff_pass_rate"] - pr["baseline_pass_rate"]
            print(
                f"  {prefill!r:>8}: baseline={pr['baseline_pass_rate']:.3f} | "
                f"contrastive={pr['best_contrastive_pass_rate']:.3f} "
                f"({delta_c:+.3f}, coeff={pr['best_contrastive_coeff']}) | "
                f"mean_diff={pr['best_mean_diff_pass_rate']:.3f} "
                f"({delta_m:+.3f}, coeff={pr['best_mean_diff_coeff']})"
            )

    print(f"\n{'=' * 70}")
    print(f"All done! Processed {len(all_sample_results)} samples.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
