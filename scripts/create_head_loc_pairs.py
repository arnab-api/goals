"""Populate data/head_loc/pairs.json with verified and KL-scored PredicatePair entries.

Two-phase script:

  Phase 1 (oracle verification, no GPU):
    Enumerates all same-type predicate pairs, asks Claude whether each pair is valid
    (same signature, meaningfully different logic), and writes PredicatePair entries
    with oracle_verified set.

  Phase 2 (KL computation, requires --model-key):
    For each oracle-verified pair, runs a forward pass at bare `return ` for both
    the destination and target Python prompts, computes KL(P_dest || P_target), and
    sets top_token_diverges_at_bare_return.

Usage:
    # Phase 1 only (no GPU)
    conda run -n goals python scripts/create_head_loc_pairs.py --verify-only

    # Both phases
    conda run -n goals python scripts/create_head_loc_pairs.py \\
        --model-key Qwen/Qwen2.5-Coder-14B-Instruct
"""

import argparse
import itertools
import logging
import sys

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

sys.path.append(".")

from src.goals.functional import apply_chat_template, supports_chat_template
from src.goals.head_loc_data import HeadLocDataset, PredicatePair, SinglePredicateSample
from src.tokens import prepare_input
from src.utils.oracle_llms import ask_claude

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
console = Console(width=120)

# ---------------------------------------------------------------------------
# Phase 1 — Oracle verification
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
Two Python predicates are being considered for a code-generation research experiment.

Predicate A:
  Name        : {name_a}
  Signature   : {sig_a}
  Description : {desc_a}

Predicate B:
  Name        : {name_b}
  Signature   : {sig_b}
  Description : {desc_b}

Answer YES if ALL of the following hold:
  1. Both predicates take exactly one argument of the same type.
  2. They implement meaningfully different logic (not mere inverses or trivial variants of each other).
Answer NO otherwise.

Reply with exactly one word: YES or NO."""


def _signature(sample: SinglePredicateSample) -> str:
    return f"{sample.function_name}({sample.variable_name}: {sample.variable_type}) -> bool"


def verify_pair_oracle(a: SinglePredicateSample, b: SinglePredicateSample) -> bool:
    prompt = VERIFY_PROMPT.format(
        name_a=a.predicate_name,
        sig_a=_signature(a),
        desc_a=f"see function name",
        name_b=b.predicate_name,
        sig_b=_signature(b),
        desc_b=f"see function name",
    )
    response = ask_claude(prompt, max_tokens=8, temperature=0.0, use_cache=True)
    return response.strip().upper().startswith("YES")


def run_oracle_verification(ds: HeadLocDataset) -> list[PredicatePair]:
    """Enumerate same-type pairs, verify each direction, return PredicatePair list."""
    # Group by variable_type
    by_type: dict[str, list[SinglePredicateSample]] = {}
    for s in ds:
        by_type.setdefault(s.variable_type, []).append(s)

    pairs: list[PredicatePair] = []
    for vtype, samples in by_type.items():
        for a, b in itertools.permutations(samples, 2):
            logger.info(f"Verifying  {a.predicate_name} → {b.predicate_name} ...")
            verified = verify_pair_oracle(a, b)
            pairs.append(
                PredicatePair(
                    source_predicate=a.predicate_name,
                    destination_predicate=b.predicate_name,
                    variable_type=vtype,
                    oracle_verified=verified,
                )
            )
            logger.info(f"  {'✓ YES' if verified else '✗ NO '}")
    return pairs


# ---------------------------------------------------------------------------
# Phase 2 — KL computation
# ---------------------------------------------------------------------------

GENERIC_INSTRUCTION = "Complete the Python function below."


def build_python_prompt(mt, sample: SinglePredicateSample) -> str:
    """Full model prompt ending at `return ` (trailing space = next token is the expression)."""
    code_at_pivot = sample.python_incomplete_code + " "
    if supports_chat_template(mt):
        return apply_chat_template(
            mt=mt,
            user_query=GENERIC_INSTRUCTION,
            incomplete_code=code_at_pivot,
        )
    return "<code>\n" + code_at_pivot


@torch.inference_mode()
def get_next_token_probs(mt, prompt: str) -> torch.Tensor:
    """Return full vocabulary probability distribution at the last token position."""
    inputs = prepare_input(prompts=[prompt], tokenizer=mt)
    with mt.trace(inputs, scan=False, validate=False):
        logits = mt.lm_head.output[0, -1, :].save()
    return torch.softmax(logits.float(), dim=-1).cpu()


def compute_kl(p_dest: torch.Tensor, p_target: torch.Tensor) -> float:
    """KL(P_dest || P_target): how far destination is from target (nats)."""
    return F.kl_div(
        input=torch.log(p_target + 1e-10),
        target=p_dest,
        reduction="sum",
    ).item()


def run_kl_computation(mt, ds: HeadLocDataset, pairs: list[PredicatePair]) -> None:
    """Fill in kl_at_bare_return and top_token_diverges_at_bare_return for each pair (in-place)."""
    # Cache prompts per predicate to avoid redundant forward passes
    prompt_cache: dict[str, str] = {}
    prob_cache: dict[str, torch.Tensor] = {}

    verified = [p for p in pairs if p.oracle_verified]
    # Collect all predicate names that appear in verified pairs
    needed = {p.source_predicate for p in verified} | {p.destination_predicate for p in verified}

    logger.info(f"Pre-computing next-token probs for {len(needed)} predicates ...")
    for name in sorted(needed):
        sample = ds.get_sample(name)
        if sample is None:
            logger.warning(f"  Sample not found: {name}")
            continue
        prompt = build_python_prompt(mt, sample)
        prompt_cache[name] = prompt
        probs = get_next_token_probs(mt, prompt)
        prob_cache[name] = probs
        top_tok = mt.tokenizer.decode([probs.argmax().item()])
        logger.info(f"  {name}: top token = {top_tok!r}  (p={probs.max().item():.3f})")

    for pair in verified:
        p_dest = prob_cache.get(pair.destination_predicate)
        p_target = prob_cache.get(pair.source_predicate)  # source = target predicate
        if p_dest is None or p_target is None:
            continue
        pair.kl_at_bare_return = round(compute_kl(p_dest, p_target), 4)
        pair.top_token_diverges_at_bare_return = (
            p_dest.argmax().item() != p_target.argmax().item()
        )


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def print_summary(pairs: list[PredicatePair], tokenizer=None, prob_cache: dict = None) -> None:
    table = Table(title="Predicate Pairs", show_lines=True)
    table.add_column("Source → Dest", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Verified", justify="center")
    table.add_column("KL", justify="right")
    table.add_column("Top tok diverges", justify="center")

    for p in pairs:
        table.add_row(
            f"{p.source_predicate} → {p.destination_predicate}",
            p.variable_type,
            "✓" if p.oracle_verified else "✗",
            f"{p.kl_at_bare_return:.4f}" if p.kl_at_bare_return else "—",
            "✓" if p.top_token_diverges_at_bare_return else "—",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create head_loc predicate pairs.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run only Phase 1 (oracle verification, no model needed).",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="HuggingFace model key for Phase 2 KL computation.",
    )
    args = parser.parse_args()

    if not args.verify_only and args.model_key is None:
        parser.error("Provide --model-key for Phase 2 KL computation, or use --verify-only.")

    ds = HeadLocDataset.load()
    if len(ds) == 0:
        logger.error("No predicate samples found. Run create_head_loc_predicates.py first.")
        sys.exit(1)

    # Phase 1
    logger.info("=== Phase 1: Oracle verification ===")
    pairs = run_oracle_verification(ds)
    ds.pairs = pairs
    ds.save_pairs()
    n_verified = sum(1 for p in pairs if p.oracle_verified)
    logger.info(f"Verified {n_verified}/{len(pairs)} pairs.")

    if args.verify_only:
        print_summary(pairs)
        return

    # Phase 2
    logger.info("=== Phase 2: KL computation ===")
    from src.models import ModelandTokenizer
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    run_kl_computation(mt, ds, pairs)
    ds.pairs = pairs
    ds.save_pairs()

    print_summary(pairs)


if __name__ == "__main__":
    main()
