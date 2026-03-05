"""Test whether P+ goal directions improve pass rates on adversarial prefills.

For each TargetPredicate (from MultiPredicateDataset), steers generation on the
matched single-predicate IncompleteCodeSample using contrastive and mean-diff
directions trained by multipred_probe.py.  The claim: steering with the P+
direction "reminds" the model what instrumental goal it is pursuing, counteracting
misleading adversarial prefills.

Usage:
    python -m scripts.causality.adversarial_persistence \\
        --model_name Qwen/Qwen2.5-Coder-32B-Instruct \\
        --probe_dir multi_probe \\
        --results_dir causality
    python -m scripts.causality.adversarial_persistence \\
        --model_name Qwen/Qwen2.5-Coder-14B-Instruct \\
        --predicate_names is_palindrome validate_config \\
        --n_generations 50

Output directory / artifact schema
------------------------------------
All paths are relative to DEFAULT_RESULTS_DIR (from globals.yml / env.yml).

$RESULTS_DIR/
└── <results_dir>/                       # --results_dir  (default: "causality")
    └── <model_short_name>/              # last component of --model_name
        ├── errors.log                   # one JSON line per nnsight steering error
        ├── all_predicates_summary.json  # {predicate_name: <summary_dict>, ...}
        └── <predicate_name>/            # one directory per TargetPredicate
            ├── summary.json             # full results for this predicate (see below)
            └── prefill_<safe_prefill>/  # spaces and "/" replaced with "_"
                ├── baseline.json        # CodeGenerations for unsteered baseline
                └── layer_<layer_idx>/   # one directory per layer center in the grid
                    ├── contrastive_coeff_<coeff>.json
                    └── mean_diff_coeff_<coeff>.json

summary.json structure (also the per-predicate value inside all_predicates_summary.json):
{
  "predicate_name": str,
  "p_plus_description": str,
  "function_name": str,
  "n_generations": int,
  "layer_grid": {"from": int, "to": int, "step": int, "window": int},
  "prefill_results": {
    "<prefill>": {
      "baseline_pass_rate": float,
      "by_layer": {
        "<layer_idx>": {
          "window": [int, ...],            # actual layer indices steered
          "contrastive_by_coeff": {"0.020": float, ...},
          "mean_diff_by_coeff":   {"0.020": float, ...},
          "best_contrastive_coeff":      str | null,
          "best_contrastive_pass_rate":  float | null,
          "best_mean_diff_coeff":        str | null,
          "best_mean_diff_pass_rate":    float | null
        },
        ...
      },
      "best_contrastive": {"layer_idx": int, "window": [...], "coeff": str, "pass_rate": float} | null,
      "best_mean_diff":   {"layer_idx": int, "window": [...], "coeff": str, "pass_rate": float} | null
    },
    ...
  },
  "overall_best_contrastive": {"prefill": str, "layer_idx": int, "window": [...], "coeff": str, "pass_rate": float} | null,
  "overall_best_mean_diff":   {"prefill": str, "layer_idx": int, "window": [...], "coeff": str, "pass_rate": float} | null
}

AdversarialPersistenceResults (scripts/causality/results.py) can load summary.json
and provides query helpers: best_hparams(), best_hparams_for_prefill(),
best_coeff_for_layer(), best_layer_for_coeff(), results_for().
"""

import argparse
import json
import logging
import os
import signal
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
from src.goals.data import Code, CodeGenerations, IncompleteCodeDataset, SingleCodeEval  # noqa: E402
from src.goals.eval import evaluate_single_code, run_evaluation  # noqa: E402
from src.goals.functional import (  # noqa: E402
    apply_chat_template,
    supports_chat_template,
)
from src.goals.predicate_data import (  # noqa: E402
    MultiPredicateDataset,
    TargetPredicate,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, logging_utils  # noqa: E402
from src.utils.experiment_utils import set_seed  # noqa: E402

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Eval timeout helpers
# ══════════════════════════════════════════════════════════════════════
class _EvalTimeout(BaseException):
    """Private timeout sentinel — inherits BaseException, not Exception.

    This prevents it from being caught by the broad `except Exception` guards
    inside evaluate_single_code (one wrapping exec(), one wrapping func(input_val)
    in the unit test loop). If TimeoutError were used instead, it would be silently
    swallowed on the first unit test, the alarm would not be re-armed, and the
    second unit test's input() call would hang forever.
    """
    pass


def _log_eval_timeout(code: str, context: dict, log_path: str) -> None:
    """Append a timed-out code snippet + context to eval_timeouts.log (JSON lines)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {"code": code, **context}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.warning(f"Eval timeout: context={context}")


def _eval_with_timeout(
    full_code: str,
    unit_tests: list,
    function_name: str,
    timeout_secs: int,
    timeout_log: str | None,
    context: dict,
) -> SingleCodeEval:
    """Run evaluate_single_code with SIGALRM timeout. Linux main-thread only."""
    def _handler(signum, frame):
        raise _EvalTimeout(f"eval timed out after {timeout_secs}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_secs)
    try:
        result = evaluate_single_code(full_code, unit_tests, function_name)
    except _EvalTimeout:
        result = SingleCodeEval(
            code=full_code,
            verdict="TIMEOUT",
            warnings=[],
            unit_test_results=[],
            error_message=f"Timed out after {timeout_secs}s",
        )
        if timeout_log:
            _log_eval_timeout(full_code, context, timeout_log)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


def run_evaluation_timed(
    result: CodeGenerations,
    function_name: str | None,
    code_stop_tag: str,
    eval_timeout: int,
    eval_timeout_log: str | None,
    context: dict,
) -> CodeGenerations:
    """Like run_evaluation but with per-code SIGALRM timeout."""
    sample = result.sample
    base_code = sample.code
    unit_tests = list(sample.unit_tests)
    target_function = (
        function_name or result.function_name or sample.function_name or "func"
    )
    result.function_name = target_function

    for code_entry in result.generations:
        prefill = code_entry.prefill or ""
        out = code_entry.lm_generation
        if code_stop_tag:
            cur_stop_tag = code_stop_tag if code_stop_tag in out else "\n"
            out = out.split(cur_stop_tag)[0].rstrip()
        code_entry.lm_generation = out
        full_code = f"{base_code}{prefill}{out}"
        code_entry.source_code = full_code
        code_entry.function_name = target_function

        single_result = _eval_with_timeout(
            full_code, unit_tests, target_function,
            eval_timeout, eval_timeout_log, context,
        )
        single_result.prefill = (prefill, code_entry.prefill_type or "")
        single_result.output = out
        code_entry.eval = single_result

    pass_count = sum(
        1 for e in result.generations
        if e.eval and e.eval.verdict == "PASSED"
    )
    fail_count = len(result.generations) - pass_count
    from src.goals.data import EvaluationSummary
    result.summary = EvaluationSummary(
        pass_count=pass_count, fail_count=fail_count, total=len(result.generations)
    )
    return result


# Coefficient grid search range (matches test_causality.py)
COEFF_GRID = list(np.arange(0.02, 0.21, 0.02))

# Default adversarial prefills used across all predicates.
# Override with --prefills if needed.
ADVERSARIAL_PREFILLS = [
    "(3",
    "1 -",
    "1 +",
    "not",
    "not not",
    "5",
    "3 +",
    "0",
    "0 +",
]


# ══════════════════════════════════════════════════════════════════════
# Steering helpers (verbatim from test_causality.py)
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


def build_context_patches(
    mt: ModelandTokenizer,
    directions: dict[int, torch.Tensor],
    hs: dict[tuple[str, int], torch.Tensor],
    layer_from: int,
    layer_to: int,
    coeff: float,
    n_tokens: int,
) -> list[PatchSpec]:
    """Build on_pass='first' patches for all input context positions (0..n_tokens-2).

    Each position is scaled by its own hidden-state norm (not the last token's).
    Position n_tokens-1 (the last input token) is handled by build_steer_patches
    with on_pass='all', so it is excluded here to avoid double-application.
    """
    patches = []
    for layer_idx in range(layer_from, layer_to):
        layer_name = mt.layer_name_format.format(layer_idx)
        if layer_idx not in directions:
            continue
        goal_dir = directions[layer_idx]
        for token_idx in range(
            n_tokens - 1
        ):  # excludes last token (handled by build_steer_patches)
            h_tok = hs[(layer_name, token_idx)]
            goal_dir_dev = goal_dir.to(h_tok.device)
            scaled = goal_dir_dev * coeff * (h_tok.norm() / goal_dir_dev.norm())
            patches.append(
                PatchSpec(
                    location=(layer_name, token_idx),
                    patch=scaled,
                    strategy="add",
                    on_pass="first",
                )
            )
    return patches


@torch.no_grad()
def generate_and_evaluate(
    mt: ModelandTokenizer,
    sample,
    prefill: str,
    prompt: str,
    n_generations: int,
    patches: list[PatchSpec] | None = None,
    label: str = "",
    patch_interface: str = "baukit",
    eval_timeout: int = 5,
    eval_timeout_log: str | None = None,
) -> CodeGenerations:
    """Generate with optional patches, evaluate, and return the full CodeGenerations."""
    logger.debug(
        f"Generating {n_generations} samples{f' ({label})' if label else ''}..."
    )
    outputs = generate_with_patch(
        mt=mt,
        inputs=prompt,
        patches=patches,
        patch_interface=patch_interface,
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

    eval_context = {"label": label, "prefill": prefill}
    eval_results = run_evaluation_timed(
        result=generations,
        function_name=sample.function_name,
        code_stop_tag="</code>",
        eval_timeout=eval_timeout,
        eval_timeout_log=eval_timeout_log,
        context=eval_context,
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
    predicate_name: str,
    prefill: str,
    coeff: float,
    layer_idx: int,
    layer_from: int,
    layer_to: int,
    direction_type: str,
) -> None:
    """Log a nnsight steering error with full config to errors.log and logger.error."""
    import traceback as tb_module

    error_config = {
        "model": model_name,
        "predicate_name": predicate_name,
        "prefill": prefill,
        "coeff": round(coeff, 6),
        "layer_idx": layer_idx,
        "layer_from": layer_from,
        "layer_to": layer_to,
        "direction_type": direction_type,
        "error_type": type(e).__name__,
        "error_message": str(e),
    }

    logger.error(
        f"Steering error [{direction_type}] predicate={predicate_name}, prefill={prefill!r}, "
        f"coeff={coeff:.3f}, layer_idx={layer_idx}, layers=({layer_from},{layer_to}): "
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
# Process one TargetPredicate
# ══════════════════════════════════════════════════════════════════════
def process_predicate(
    mt: ModelandTokenizer,
    tp: TargetPredicate,
    predicate_idx: int,
    eval_dataset: IncompleteCodeDataset,
    probe_predicate_dir: str,
    results_predicate_dir: str,
    error_log_path: str,
    args,
) -> dict | None:
    """
    Run steering evaluation for one TargetPredicate across all adversarial
    prefills, layer centers, and coefficients.

    Args:
        probe_predicate_dir: Where to load directions from (multipred_probe.py output).
        results_predicate_dir: Where to save causality results.

    Returns a results dict for this predicate, or None if skipped.
    """
    # --- Match to single-predicate IncompleteCodeSample ---
    sample = eval_dataset.find_by_function_name(tp.name)
    if sample is None:
        logger.warning(
            f"No IncompleteCodeSample with function_name={tp.name!r}. Skipping."
        )
        return None

    # --- Determine adversarial prefills ---
    adversarial_prefills = (
        args.prefills if args.prefills is not None else ADVERSARIAL_PREFILLS
    )

    # --- Load directions ---
    contrastive_path = os.path.join(probe_predicate_dir, "contrastive_directions.pt")
    mean_diff_path = os.path.join(probe_predicate_dir, "mean_diff_directions.pt")

    if not os.path.exists(contrastive_path) or not os.path.exists(mean_diff_path):
        logger.warning(
            f"Directions not found in {probe_predicate_dir}. Skipping predicate {tp.name!r}."
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

    # --- Layer grid parameters ---
    layer_centers = list(range(args.layer_from, args.layer_to, args.layer_step))
    # Global hs range covers all possible window extents
    global_lo = max(0, args.layer_from - args.layer_window // 2)
    global_hi = min(mt.n_layer, args.layer_to + args.layer_window // 2)

    os.makedirs(results_predicate_dir, exist_ok=True)

    eval_timeout_log = os.path.join(
        os.path.dirname(error_log_path), "eval_timeouts.log"
    )

    print(f"\n{'#' * 70}")
    print(f"# Predicate {predicate_idx}: {tp.name!r}")
    print(f"# P+: {tp.p_plus.description}")
    print(f"# Sample: {sample.function_name}")
    print(f"# Directions from: {probe_predicate_dir}")
    print(f"# Results to:      {results_predicate_dir}")
    print(f"# Layer centers: {layer_centers} (window={args.layer_window})")
    print(f"# Adversarial prefills: {adversarial_prefills}")
    print(f"# Coeff grid: {[round(c, 3) for c in COEFF_GRID]}")
    print(f"{'#' * 70}")

    predicate_results = {
        "predicate_name": tp.name,
        "p_plus_description": tp.p_plus.description,
        "function_name": sample.function_name,
        "n_generations": args.n_generations,
        "layer_grid": {
            "from": args.layer_from,
            "to": args.layer_to,
            "step": args.layer_step,
            "window": args.layer_window,
        },
        "prefill_results": {},
    }

    for pfx_i, prefill in enumerate(adversarial_prefills):
        safe_prefill = prefill.replace(" ", "_").replace("/", "_") or "empty"
        prefill_dir = os.path.join(results_predicate_dir, f"prefill_{safe_prefill}")
        os.makedirs(prefill_dir, exist_ok=True)

        logger.info(
            f"Predicate {tp.name!r}, prefill {pfx_i + 1}/{len(adversarial_prefills)}: {prefill!r}"
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

        # --- Get hidden states once for all windows (global range) ---
        logger.debug(
            f"Extracting hidden states (layers {global_lo}-{global_hi}) for prefill {prefill!r}..."
        )
        tokenized = prepare_input(prompts=prompt, tokenizer=mt)
        n_tokens = tokenized.input_ids.shape[1]

        locations = [
            (mt.layer_name_format.format(i), -1) for i in range(global_lo, global_hi)
        ]

        if args.steer_context:
            locations += [
                (mt.layer_name_format.format(i), t)
                for i in range(global_lo, global_hi)
                for t in range(n_tokens-1)
            ]

        hs = get_hs(mt=mt, input=tokenized, locations=locations)

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
            eval_timeout_log=eval_timeout_log,
        )
        baseline_pass_rate = (
            baseline_eval.summary.pass_count / baseline_eval.summary.total
        )
        logger.info(f"Baseline pass rate: {baseline_pass_rate:.3f}")
        print(f"    Baseline pass rate: {baseline_pass_rate:.3f}")

        save_eval_table(baseline_eval, os.path.join(prefill_dir, "baseline.json"))
        del baseline_eval
        free_gpu_cache()

        # --- Grid search over layer centers and coefficients ---
        by_layer = {}

        for layer_idx in layer_centers:
            window_lo = max(0, layer_idx - args.layer_window // 2)
            window_hi = min(mt.n_layer - 1, layer_idx + args.layer_window // 2) + 1
            window_ints = list(range(window_lo, window_hi))
            layer_dir = os.path.join(prefill_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)

            logger.info(
                f"  Layer center {layer_idx}: window={window_ints}, "
                f"coeff grid ({len(COEFF_GRID)} values)"
            )
            print(f"\n    Layer {layer_idx} (window={window_ints}):")

            contrastive_by_coeff = {}
            mean_diff_by_coeff = {}

            for coeff_i, coeff in enumerate(COEFF_GRID):
                coeff_key = f"{coeff:.3f}"
                logger.info(
                    f"  Coeff {coeff_i + 1}/{len(COEFF_GRID)}: {coeff_key} "
                    f"(layer={layer_idx}, prefill={prefill!r})"
                )

                # Contrastive steering
                contrastive_patches = build_steer_patches(
                    mt=mt,
                    directions=contrastive_directions,
                    hs=hs,
                    layer_from=window_lo,
                    layer_to=window_hi,
                    coeff=coeff,
                )
                if args.steer_context:
                    contrastive_patches += build_context_patches(
                        mt=mt,
                        directions=contrastive_directions,
                        hs=hs,
                        layer_from=window_lo,
                        layer_to=window_hi,
                        coeff=coeff,
                        n_tokens=n_tokens,
                    )
                try:
                    contrastive_eval = generate_and_evaluate(
                        mt=mt,
                        sample=sample,
                        prefill=prefill,
                        prompt=prompt,
                        n_generations=args.n_generations,
                        patches=contrastive_patches,
                        label=f"contrastive layer={layer_idx} coeff={coeff_key}",
                        eval_timeout_log=eval_timeout_log,
                    )
                    contrastive_pass_rate = (
                        contrastive_eval.summary.pass_count
                        / contrastive_eval.summary.total
                    )
                    contrastive_by_coeff[coeff_key] = contrastive_pass_rate
                    save_eval_table(
                        contrastive_eval,
                        os.path.join(layer_dir, f"contrastive_coeff_{coeff_key}.json"),
                    )
                except Exception as e:
                    log_steering_error(
                        e=e,
                        error_log_path=error_log_path,
                        model_name=args.model_name,
                        predicate_name=tp.name,
                        prefill=prefill,
                        coeff=coeff,
                        layer_idx=layer_idx,
                        layer_from=window_lo,
                        layer_to=window_hi,
                        direction_type="contrastive",
                    )
                    contrastive_pass_rate = None
                free_gpu_cache()

                # Mean-diff steering
                mean_diff_patches = build_steer_patches(
                    mt=mt,
                    directions=mean_diff_directions,
                    hs=hs,
                    layer_from=window_lo,
                    layer_to=window_hi,
                    coeff=coeff,
                )
                if args.steer_context:
                    mean_diff_patches += build_context_patches(
                        mt=mt,
                        directions=mean_diff_directions,
                        hs=hs,
                        layer_from=window_lo,
                        layer_to=window_hi,
                        coeff=coeff,
                        n_tokens=n_tokens,
                    )
                try:
                    mean_diff_eval = generate_and_evaluate(
                        mt=mt,
                        sample=sample,
                        prefill=prefill,
                        prompt=prompt,
                        n_generations=args.n_generations,
                        patches=mean_diff_patches,
                        label=f"mean_diff layer={layer_idx} coeff={coeff_key}",
                        eval_timeout_log=eval_timeout_log,
                    )
                    mean_diff_pass_rate = (
                        mean_diff_eval.summary.pass_count / mean_diff_eval.summary.total
                    )
                    mean_diff_by_coeff[coeff_key] = mean_diff_pass_rate
                    save_eval_table(
                        mean_diff_eval,
                        os.path.join(layer_dir, f"mean_diff_coeff_{coeff_key}.json"),
                    )
                except Exception as e:
                    log_steering_error(
                        e=e,
                        error_log_path=error_log_path,
                        model_name=args.model_name,
                        predicate_name=tp.name,
                        prefill=prefill,
                        coeff=coeff,
                        layer_idx=layer_idx,
                        layer_from=window_lo,
                        layer_to=window_hi,
                        direction_type="mean_diff",
                    )
                    mean_diff_pass_rate = None
                free_gpu_cache()

                print(
                    f"      coeff={coeff:.3f}: "
                    f"contrastive={'ERROR' if contrastive_pass_rate is None else f'{contrastive_pass_rate:.3f}'}, "
                    f"mean_diff={'ERROR' if mean_diff_pass_rate is None else f'{mean_diff_pass_rate:.3f}'}"
                )

            # Best coeff for this layer center
            if contrastive_by_coeff:
                best_c_coeff = max(contrastive_by_coeff, key=contrastive_by_coeff.get)
                best_c_rate = contrastive_by_coeff[best_c_coeff]
            else:
                logger.warning(
                    f"All contrastive coeff runs errored for layer={layer_idx}, prefill={prefill!r}."
                )
                best_c_coeff = None
                best_c_rate = None

            if mean_diff_by_coeff:
                best_m_coeff = max(mean_diff_by_coeff, key=mean_diff_by_coeff.get)
                best_m_rate = mean_diff_by_coeff[best_m_coeff]
            else:
                logger.warning(
                    f"All mean_diff coeff runs errored for layer={layer_idx}, prefill={prefill!r}."
                )
                best_m_coeff = None
                best_m_rate = None

            by_layer[str(layer_idx)] = {
                "window": window_ints,
                "contrastive_by_coeff": contrastive_by_coeff,
                "mean_diff_by_coeff": mean_diff_by_coeff,
                "best_contrastive_coeff": best_c_coeff,
                "best_contrastive_pass_rate": best_c_rate,
                "best_mean_diff_coeff": best_m_coeff,
                "best_mean_diff_pass_rate": best_m_rate,
            }

            if best_c_rate is not None:
                print(
                    f"      Best contrastive: coeff={best_c_coeff}, pass_rate={best_c_rate:.3f}"
                )
            if best_m_rate is not None:
                print(
                    f"      Best mean_diff:   coeff={best_m_coeff}, pass_rate={best_m_rate:.3f}"
                )

        # --- Best layer for this prefill ---
        valid_c = {
            li: by_layer[li]["best_contrastive_pass_rate"]
            for li in by_layer
            if by_layer[li]["best_contrastive_pass_rate"] is not None
        }
        valid_m = {
            li: by_layer[li]["best_mean_diff_pass_rate"]
            for li in by_layer
            if by_layer[li]["best_mean_diff_pass_rate"] is not None
        }

        if valid_c:
            best_c_layer_str = max(valid_c, key=valid_c.get)
            bcd = by_layer[best_c_layer_str]
            best_contrastive = {
                "layer_idx": int(best_c_layer_str),
                "window": bcd["window"],
                "coeff": bcd["best_contrastive_coeff"],
                "pass_rate": bcd["best_contrastive_pass_rate"],
            }
        else:
            best_contrastive = None

        if valid_m:
            best_m_layer_str = max(valid_m, key=valid_m.get)
            bmd = by_layer[best_m_layer_str]
            best_mean_diff = {
                "layer_idx": int(best_m_layer_str),
                "window": bmd["window"],
                "coeff": bmd["best_mean_diff_coeff"],
                "pass_rate": bmd["best_mean_diff_pass_rate"],
            }
        else:
            best_mean_diff = None

        predicate_results["prefill_results"][prefill] = {
            "baseline_pass_rate": baseline_pass_rate,
            "by_layer": by_layer,
            "best_contrastive": best_contrastive,
            "best_mean_diff": best_mean_diff,
        }

        if best_contrastive is not None:
            print(
                f"\n    Best contrastive: layer={best_contrastive['layer_idx']}, "
                f"coeff={best_contrastive['coeff']}, pass_rate={best_contrastive['pass_rate']:.3f}"
            )
        else:
            print("\n    Best contrastive: all runs errored.")

        if best_mean_diff is not None:
            print(
                f"    Best mean_diff:   layer={best_mean_diff['layer_idx']}, "
                f"coeff={best_mean_diff['coeff']}, pass_rate={best_mean_diff['pass_rate']:.3f}"
            )
        else:
            print("    Best mean_diff:   all runs errored.")

    # --- Compute overall best across all prefills ---
    all_c_entries = [
        (pfx, pfr["best_contrastive"])
        for pfx, pfr in predicate_results["prefill_results"].items()
        if pfr["best_contrastive"] is not None
    ]
    all_m_entries = [
        (pfx, pfr["best_mean_diff"])
        for pfx, pfr in predicate_results["prefill_results"].items()
        if pfr["best_mean_diff"] is not None
    ]

    if all_c_entries:
        best_pfx, best_data = max(all_c_entries, key=lambda x: x[1]["pass_rate"])
        predicate_results["overall_best_contrastive"] = {
            "prefill": best_pfx,
            "layer_idx": best_data["layer_idx"],
            "window": best_data["window"],
            "coeff": best_data["coeff"],
            "pass_rate": best_data["pass_rate"],
        }
    else:
        predicate_results["overall_best_contrastive"] = None

    if all_m_entries:
        best_pfx, best_data = max(all_m_entries, key=lambda x: x[1]["pass_rate"])
        predicate_results["overall_best_mean_diff"] = {
            "prefill": best_pfx,
            "layer_idx": best_data["layer_idx"],
            "window": best_data["window"],
            "coeff": best_data["coeff"],
            "pass_rate": best_data["pass_rate"],
        }
    else:
        predicate_results["overall_best_mean_diff"] = None

    # --- Save per-predicate summary ---
    summary_path = os.path.join(results_predicate_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(predicate_results, f, indent=2)
    logger.info(f"Saved per-predicate summary to {summary_path}")
    print(f"\n  Saved per-predicate summary to {summary_path}")

    # --- Print summary for this predicate ---
    print(f"\n  === Predicate {tp.name!r} Summary ===")
    for prefill, pr in predicate_results["prefill_results"].items():
        bc = pr["best_contrastive"]
        bm = pr["best_mean_diff"]
        c_str = (
            f"{bc['pass_rate']:.3f} (layer={bc['layer_idx']}, coeff={bc['coeff']})"
            if bc is not None
            else "ERROR"
        )
        m_str = (
            f"{bm['pass_rate']:.3f} (layer={bm['layer_idx']}, coeff={bm['coeff']})"
            if bm is not None
            else "ERROR"
        )
        print(
            f"    {prefill!r:>12}: baseline={pr['baseline_pass_rate']:.3f}, "
            f"contrastive={c_str}, "
            f"mean_diff={m_str}"
        )

    return predicate_results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test adversarial-prefill persistence: steer with P+ directions from "
            "multipred_probe.py and measure pass-rate recovery on adversarial prefills."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="multi_probe",
        help=(
            "Subdirectory under DEFAULT_RESULTS_DIR where multipred_probe.py saved "
            "directions (mean_diff_directions.pt, contrastive_directions.pt, config.json)"
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="causality",
        help="Subdirectory under DEFAULT_RESULTS_DIR for saving causality results",
    )
    parser.add_argument(
        "--predicate_names",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific TargetPredicate names. Default: all predicates",
    )
    parser.add_argument(
        "--layer_from",
        type=int,
        default=30,
        help="First layer center to search (inclusive)",
    )
    parser.add_argument(
        "--layer_to",
        type=int,
        default=50,
        help="Last layer center to search (exclusive)",
    )
    parser.add_argument(
        "--layer_step",
        type=int,
        default=5,
        help="Step between layer centers",
    )
    parser.add_argument(
        "--layer_window",
        type=int,
        default=9,
        help="Window size passed to get_window (number of layers to steer)",
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=100,
        help="Number of generations to sample per prompt",
    )
    parser.add_argument(
        "--prefills",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override the default ADVERSARIAL_PREFILLS list (space-separated). "
            f"Default: {ADVERSARIAL_PREFILLS}"
        ),
    )

    parser.add_argument(
        "--steer_context",
        action="store_true",
        default=False,
        help=(
            "In addition to steering the last token at every generation step "
            "(on_pass='all'), also steer all input context positions during the "
            "prefill pass (on_pass='first') with per-token norm scaling."
        ),
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

    model_short_name = args.model_name.split("/")[-1]

    # Directory where probe directions were saved (by multipred_probe.py)
    probe_model_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.probe_dir, model_short_name
    )
    # Directory where causality results will be saved
    results_model_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR, args.results_dir, model_short_name
    )
    os.makedirs(results_model_dir, exist_ok=True)

    # Shared error log for nnsight/steering failures across all predicates
    error_log_path = os.path.join(results_model_dir, "errors.log")

    layer_centers = list(range(args.layer_from, args.layer_to, args.layer_step))
    active_prefills = (
        args.prefills if args.prefills is not None else ADVERSARIAL_PREFILLS
    )

    # --- Print config ---
    print(f"\n{'=' * 70}")
    print("Adversarial Persistence — P+ Direction Steering")
    print(f"{'=' * 70}")
    print(f"Model:           {args.model_name}")
    print(f"Probe dir:       {probe_model_dir}")
    print(f"Results dir:     {results_model_dir}")
    print(f"Layer centers:   {layer_centers} (window={args.layer_window})")
    print(f"N generations:   {args.n_generations}")
    print(f"Predicate filter:{args.predicate_names or 'all'}")
    print(f"Prefills:        {active_prefills}")
    print(f"Coeff grid:      {[round(c, 3) for c in COEFF_GRID]}")
    print(f"Steer context:   {args.steer_context}")
    print(f"{'=' * 70}\n")

    # --- Load MultiPredicateDataset ---
    logger.info("Loading MultiPredicateDataset...")
    multi_dataset = MultiPredicateDataset.load_dataset()
    logger.info(f"MultiPredicateDataset loaded: {len(multi_dataset)} predicates")

    # --- Filter predicates if requested ---
    predicates: list[TargetPredicate] = list(multi_dataset.groups)
    if args.predicate_names is not None:
        predicates = [tp for tp in predicates if tp.name in args.predicate_names]
        if not predicates:
            logger.error(
                f"No predicates matched --predicate_names={args.predicate_names}. "
                f"Available: {[tp.name for tp in multi_dataset.groups]}"
            )
            sys.exit(1)

    logger.info(f"Predicates to process: {[tp.name for tp in predicates]}")

    # --- Load model ---
    logger.info(f"Loading model: {args.model_name}")
    mt = ModelandTokenizer(
        model_key=args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    logger.info(f"Model loaded: {mt.name}, n_layer={mt.n_layer}, n_embd={mt.n_embd}")

    # --- Load single-predicate eval dataset ---
    eval_dataset = IncompleteCodeDataset.load_dataset()
    logger.info(f"IncompleteCodeDataset loaded: {len(eval_dataset)} samples")

    # --- Process each predicate ---
    all_predicate_results = {}

    for pred_i, tp in enumerate(predicates):
        logger.info(
            f"Processing predicate {tp.name!r} ({pred_i + 1}/{len(predicates)})..."
        )

        probe_predicate_dir = os.path.join(probe_model_dir, tp.name)
        results_predicate_dir = os.path.join(results_model_dir, tp.name)

        result = process_predicate(
            mt=mt,
            tp=tp,
            predicate_idx=pred_i,
            eval_dataset=eval_dataset,
            probe_predicate_dir=probe_predicate_dir,
            results_predicate_dir=results_predicate_dir,
            error_log_path=error_log_path,
            args=args,
        )

        if result is not None:
            all_predicate_results[tp.name] = result

        free_gpu_cache()
        logger.info(f"GPU cache cleared after predicate {tp.name!r}.")
        print(f"GPU cache cleared after predicate {tp.name!r}.")

    # --- Save aggregated model-level results ---
    aggregated_path = os.path.join(results_model_dir, "all_predicates_summary.json")
    with open(aggregated_path, "w") as f:
        json.dump(all_predicate_results, f, indent=2)
    logger.info(f"Saved aggregated model-level results to {aggregated_path}")
    print(f"\nSaved aggregated results to {aggregated_path}")

    # --- Print final summary ---
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    for pred_name, pr in all_predicate_results.items():
        print(f"\nPredicate {pred_name!r} ({pr['p_plus_description']}):")
        obc = pr.get("overall_best_contrastive")
        obm = pr.get("overall_best_mean_diff")
        if obc:
            print(
                f"  Overall best contrastive: prefill={obc['prefill']!r}, "
                f"layer={obc['layer_idx']}, coeff={obc['coeff']}, pass_rate={obc['pass_rate']:.3f}"
            )
        if obm:
            print(
                f"  Overall best mean_diff:   prefill={obm['prefill']!r}, "
                f"layer={obm['layer_idx']}, coeff={obm['coeff']}, pass_rate={obm['pass_rate']:.3f}"
            )
        for prefill, pfr in pr["prefill_results"].items():
            bc = pfr["best_contrastive"]
            bm = pfr["best_mean_diff"]
            baseline = pfr["baseline_pass_rate"]
            c_str = (
                f"{bc['pass_rate']:.3f} ({bc['pass_rate'] - baseline:+.3f}, "
                f"layer={bc['layer_idx']}, coeff={bc['coeff']})"
                if bc is not None
                else "ERROR"
            )
            m_str = (
                f"{bm['pass_rate']:.3f} ({bm['pass_rate'] - baseline:+.3f}, "
                f"layer={bm['layer_idx']}, coeff={bm['coeff']})"
                if bm is not None
                else "ERROR"
            )
            print(
                f"  {prefill!r:>12}: baseline={baseline:.3f} | "
                f"contrastive={c_str} | "
                f"mean_diff={m_str}"
            )

    print(f"\n{'=' * 70}")
    print(f"All done! Processed {len(all_predicate_results)} predicates.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
