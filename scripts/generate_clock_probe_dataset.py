"""Generate the labeled dataset for clock-probe training.

Runs the three-stage pipeline (augmentation → generation + eval → latent caching)
for every TargetPredicate in data/multi_predicate/, labels each passing sequence
with oracle-identified P+ / P- spans, and saves ProbeRecord objects.

Output layout:
    results/clock_probe/<model-short-name>/
        train<tag>/
            0000.pt         ← self-contained: metadata + masks + float16 tensors
            0001.pt  …
        test<tag>/
            0000.pt  …

Typical usage
─────────────
# Smoke test: 1 predicate, 1 original, no augmentation, layers 20-25, residual only
conda run -n goals python scripts/generate_clock_probe_dataset.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --smoke-test

# Full run (all layers, residual only — ~26 GB)
conda run -n goals python scripts/generate_clock_probe_dataset.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct

# Full run with MLP + attn (~77 GB)
conda run -n goals python scripts/generate_clock_probe_dataset.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct --include-mlp --include-attn

# Custom subset
conda run -n goals python scripts/generate_clock_probe_dataset.py \\
    --model-key Qwen/Qwen2.5-Coder-32B-Instruct \\
    --max-predicates 4 --n-p-minus-per-tp 1 \\
    --layers-start 20 --layers-end 40 --no-mlp --no-attn
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

sys.path.append(".")

from src.goals.clock_probe import (
    DEFAULT_HARMLESS_PREFILLS,
    ProbeRecord,
    augment_sample_for_probe,
    batch_generate,
    cache_latents_batch,
    evaluate_gen_ids,
    find_in_progress_mask_oracle,
)
from src.goals.functional import apply_chat_template
from src.goals.predicate_data import MultiPredicateDataset
from src.models import ModelandTokenizer
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.logging_utils import add_logging_args, configure

logger = logging.getLogger(__name__)
console = Console(width=120, highlight=False)

# Predicates held out for testing (not included in the train split).
# These match the notebook defaults but can be overridden via --test-predicates.
DEFAULT_TEST_PREDICATES = ["is_perfect_square", "has_unique_elements"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate clock-probe labeled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-key",
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="HuggingFace model key.",
    )

    # Data subset
    parser.add_argument(
        "--max-predicates",
        type=int,
        default=0,
        metavar="N",
        help="Process at most N predicates per split (0 = all).",
    )
    parser.add_argument(
        "--n-p-minus-per-tp",
        type=int,
        default=0,
        metavar="N",
        help="Max P- samples (originals) per target predicate. 0 = all available.",
    )
    parser.add_argument(
        "--test-predicates",
        nargs="+",
        default=DEFAULT_TEST_PREDICATES,
        metavar="NAME",
        help="Predicate names to hold out as the test split.",
    )
    parser.add_argument(
        "--prefills",
        nargs="+",
        default=None,
        metavar="PF",
        help=(
            "Harmless prefills to use (default: DEFAULT_HARMLESS_PREFILLS). "
            "Pass '' for the empty prefill."
        ),
    )

    # Generation
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=96,
        help="Maximum new tokens per generation.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=5,
        help="SIGALRM timeout (seconds) for unit-test evaluation.",
    )

    # Layer range
    parser.add_argument(
        "--layers-start",
        type=int,
        default=0,
        help="First layer index (inclusive).",
    )
    parser.add_argument(
        "--layers-end",
        type=int,
        default=-1,
        help="Last layer index (exclusive). -1 means all layers.",
    )

    # Activation types to cache
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="Skip caching MLP outputs (saves memory / time).",
    )
    parser.add_argument(
        "--no-attn",
        action="store_true",
        help="Skip caching attention outputs (saves memory / time).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to write <split>.pt files. "
            "Defaults to results/clock_probe/<model-short-name>/."
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        metavar="TAG",
        help=(
            "Optional tag appended to output filenames, e.g. 'smoke' → "
            "train_smoke.pt / test_smoke.pt."
        ),
    )

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable structural augmentation on the train split (only originals × prefills).",
    )

    # Batch sizes
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Number of prompts per batched generate call.",
    )
    parser.add_argument(
        "--lat-batch-size",
        type=int,
        default=16,
        metavar="N",
        help=(
            "Max sequences per batched latent-caching call. "
            "Keep smaller than --gen-batch-size to limit GPU peak memory "
            "during the nnsight trace."
        ),
    )

    # Smoke-test shortcut
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Shortcut for a quick end-to-end check: "
            "--max-predicates 1 --n-p-minus-per-tp 1 --no-augment "
            "--layers-start 20 --layers-end 26 --no-mlp --no-attn --tag smoke"
        ),
    )

    add_logging_args(parser)
    return parser


def apply_smoke_test_defaults(args: argparse.Namespace) -> None:
    """Override args with smoke-test values (only if not explicitly set)."""
    args.max_predicates = args.max_predicates or 1
    if args.n_p_minus_per_tp == 0:
        args.n_p_minus_per_tp = 1  # smoke: just 1 P- per TP
    args.layers_start = args.layers_start if args.layers_start != 0 else 0
    args.layers_end = args.layers_end if args.layers_end != -1 else -1
    args.no_mlp = True
    args.no_attn = True
    args.no_augment = True
    if not args.tag:
        args.tag = "smoke"


# ─────────────────────────────────────────────────────────────────────────────
# Incremental save helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_record_pt(record: ProbeRecord, idx: int, directory: Path) -> None:
    """Write one self-contained record file (metadata + masks + tensors)."""
    torch.save(
        {
            "predicate_name": record.predicate_name,
            "predicate_description": record.predicate_description,
            "prefill": record.prefill,
            "prompt": record.prompt,
            "prompt_len": record.latents.prompt_len,
            "prompt_ids": record.latents.prompt_ids,
            "gen_ids": record.latents.gen_ids,
            "full_text": record.latents.full_text,
            "code_start": record.latents.code_start,
            "p_plus_mask": record.p_plus_mask,
            "p_minus_mask": record.p_minus_mask,
            "residual": record.latents.residual,
            "code_residual": record.latents.code_residual,
            "mlp": record.latents.mlp,
            "code_mlp": record.latents.code_mlp,
            "attn": record.latents.attn,
            "code_attn": record.latents.code_attn,
        },
        directory / f"{idx:04d}.pt",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _chunk(lst: list, size: int):
    """Yield successive non-overlapping chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ─────────────────────────────────────────────────────────────────────────────
# Core dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_split(
    mt: ModelandTokenizer,
    multi_dataset: MultiPredicateDataset,
    predicate_names: list[str],
    layers: list[int],
    n_p_minus_per_tp: int,
    harmless_prefills: list[str],
    max_gen_tokens: int,
    eval_timeout: int,
    include_mlp: bool,
    include_attn: bool,
    out_dir: Path,
    augment: bool = True,
    gen_batch_size: int = 32,
    lat_batch_size: int = 16,
) -> list[ProbeRecord]:
    """Run the 3-stage pipeline for a list of predicates.

    Generation and latent caching are batched for throughput; oracle labeling
    and saving remain sequential.  Each passing record is written immediately
    as a self-contained .pt file so a mid-run crash loses at most the current
    in-progress batch, not entire completed predicates.

    Args:
        gen_batch_size: Prompts per batched generate call.
        lat_batch_size: Max sequences per batched latent-caching call (keep
            smaller than gen_batch_size to limit GPU peak memory during the
            nnsight trace).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ProbeRecord] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    pred_task = progress.add_task("predicates", total=len(predicate_names))

    with progress:
        for tp in multi_dataset.groups:
            if tp.name not in predicate_names:
                continue

            p_plus_desc = tp.p_plus.short_description or tp.name
            originals = [s for s in tp.samples if s.is_original]
            if n_p_minus_per_tp:
                originals = originals[:n_p_minus_per_tp]

            # Build all variants up-front so we know the total count
            all_variants: list[tuple] = []
            for orig_sample in originals:
                p_minus_desc = orig_sample.p_minus.description
                if augment:
                    vs = augment_sample_for_probe(
                        tp, orig_sample, harmless_prefills=harmless_prefills
                    )
                else:
                    vs = [(orig_sample, pf) for pf in harmless_prefills]
                all_variants.extend((aug, pf, p_minus_desc) for aug, pf in vs)

            pred_pass = pred_skip_eval = pred_skip_span = 0
            t0 = time.time()

            var_task = progress.add_task(
                f"  [cyan]{tp.name}[/cyan]", total=len(all_variants)
            )

            for gen_batch in _chunk(all_variants, gen_batch_size):
                # ── Build prompts ────────────────────────────────────────────
                batch_prompts: list[str] = [
                    apply_chat_template(
                        mt=mt,
                        user_query=aug_sample.instruction.strip(),
                        incomplete_code=aug_sample.code + prefill,
                    )
                    for aug_sample, prefill, _ in gen_batch
                ]

                # ── Stage 2a: Batched generation ─────────────────────────────
                progress.update(
                    var_task,
                    description=(
                        f"  [cyan]{tp.name}[/cyan]  "
                        f"pass=[green]{pred_pass}[/green] "
                        f"skip_eval=[yellow]{pred_skip_eval}[/yellow] "
                        f"no_span=[red]{pred_skip_span}[/red]  "
                        f"[dim]generating {len(gen_batch)}[/dim]"
                    ),
                )
                gen_results = batch_generate(mt, batch_prompts, max_gen_tokens)

                # ── Stage 2b: Sequential evaluation ─────────────────────────
                survivors: list[tuple] = []
                for (aug_sample, prefill, p_minus_desc), prompt, (
                    prompt_ids,
                    gen_ids,
                ) in zip(gen_batch, batch_prompts, gen_results):
                    passed = evaluate_gen_ids(
                        gen_ids=gen_ids,
                        sample=aug_sample,
                        prefill=prefill,
                        tokenizer=mt.tokenizer,
                        eval_timeout=eval_timeout,
                    )
                    if passed:
                        logger.info(
                            f"{tp.name} prefill={prefill!r}  gen_len={len(gen_ids)} eval=PASS"
                        )
                        survivors.append(
                            (
                                aug_sample,
                                prefill,
                                p_minus_desc,
                                prompt,
                                prompt_ids,
                                gen_ids,
                            )
                        )
                    else:
                        pred_skip_eval += 1
                    progress.advance(var_task)

                # ── Stage 3: Batched latent caching ──────────────────────────
                for lat_batch in _chunk(survivors, lat_batch_size):
                    seqs = [(item[4], item[5]) for item in lat_batch]
                    progress.update(
                        var_task,
                        description=(
                            f"  [cyan]{tp.name}[/cyan]  "
                            f"pass=[green]{pred_pass}[/green] "
                            f"skip_eval=[yellow]{pred_skip_eval}[/yellow] "
                            f"no_span=[red]{pred_skip_span}[/red]  "
                            f"[dim]caching {len(lat_batch)}[/dim]"
                        ),
                    )
                    latents_list = cache_latents_batch(
                        mt=mt,
                        sequences=seqs,
                        layers=layers,
                        include_mlp=include_mlp,
                        include_attn=include_attn,
                    )

                    # ── Oracle span labeling + incremental save ───────────────
                    for item, lat in zip(lat_batch, latents_list):
                        aug_sample, prefill, p_minus_desc, prompt, _, gen_ids = item

                        p_plus_mask, p_minus_mask = find_in_progress_mask_oracle(
                            latents=lat,
                            p_plus_description=p_plus_desc,
                            p_minus_description=p_minus_desc,
                            tokenizer=mt.tokenizer,
                            prefill=prefill,
                        )
                        if p_plus_mask.sum() == 0:
                            pred_skip_span += 1
                            logger.info(
                                f"No P+ span found for {tp.name} prefill={prefill!r}"
                            )
                            continue

                        logger.info(
                            f"{tp.name} prefill={prefill!r}  p_plus_tokens={int(p_plus_mask.sum())}/{len(gen_ids)}  p_minus={int(p_minus_mask.sum())}"
                        )

                        record = ProbeRecord(
                            predicate_name=tp.name,
                            predicate_description=p_plus_desc,
                            prefill=prefill,
                            prompt=prompt,
                            latents=lat,
                            p_plus_mask=p_plus_mask,
                            p_minus_mask=p_minus_mask,
                        )
                        _save_record_pt(record, len(records), out_dir)
                        records.append(record)
                        pred_pass += 1

            elapsed = time.time() - t0
            progress.update(var_task, visible=False)
            progress.advance(pred_task)

            logger.info(
                "%-30s  variants=%d  pass=%d  skip_eval=%d  no_span=%d  [%.0fs]",
                tp.name,
                len(all_variants),
                pred_pass,
                pred_skip_eval,
                pred_skip_span,
                elapsed,
            )
            logger.info("Running total: %d records saved to %s/", len(records), out_dir)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure(args)

    if args.smoke_test:
        logger.info("Smoke test mode enabled — applying smoke test defaults")
        apply_smoke_test_defaults(args)

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_key)
    mt = ModelandTokenizer(
        model_key=args.model_key,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model_short = args.model_key.split("/")[-1]
    logger.info(
        "Model loaded: %s  (%d layers, d_model=%d)", model_short, mt.n_layer, mt.n_embd
    )

    # ── Layers ───────────────────────────────────────────────────────────────
    layers_end = args.layers_end if args.layers_end != -1 else mt.n_layer
    layers = list(range(args.layers_start, layers_end))
    logger.info("Caching layers %d–%d (%d total)", layers[0], layers[-1], len(layers))

    # ── Dataset ──────────────────────────────────────────────────────────────
    multi_dataset = MultiPredicateDataset.load_dataset()
    all_names = [tp.name for tp in multi_dataset.groups]
    test_names = [n for n in args.test_predicates if n in all_names]
    train_names = [n for n in all_names if n not in test_names]

    if args.max_predicates:
        train_names = train_names[: args.max_predicates]
        test_names = test_names[: args.max_predicates]

    logger.info("Train predicates (%d): %s", len(train_names), ", ".join(train_names))
    logger.info("Test predicates  (%d): %s", len(test_names), ", ".join(test_names))

    # ── Prefills ─────────────────────────────────────────────────────────────
    harmless_prefills = (
        [pf.replace("\\t", "\t") for pf in args.prefills]
        if args.prefills is not None
        else DEFAULT_HARMLESS_PREFILLS
    )

    include_mlp = not args.no_mlp
    include_attn = not args.no_attn
    logger.info("Cache residual=True  mlp=%s  attn=%s", include_mlp, include_attn)

    # ── Output paths ─────────────────────────────────────────────────────────
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(DEFAULT_RESULTS_DIR) / "clock_probe" / model_short)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_suffix = f"_{args.tag}" if args.tag else ""

    train_dir = out_dir / f"train{tag_suffix}"
    test_dir = out_dir / f"test{tag_suffix}"

    # ── Generate train split ──────────────────────────────────────────────────
    console.rule("[bold]Train split")
    train_records = generate_split(
        mt=mt,
        multi_dataset=multi_dataset,
        predicate_names=train_names,
        layers=layers,
        n_p_minus_per_tp=args.n_p_minus_per_tp,
        harmless_prefills=harmless_prefills,
        max_gen_tokens=args.max_gen_tokens,
        eval_timeout=args.eval_timeout,
        include_mlp=include_mlp,
        include_attn=include_attn,
        out_dir=train_dir,
        augment=not args.no_augment,
        gen_batch_size=args.gen_batch_size,
        lat_batch_size=args.lat_batch_size,
    )
    logger.info("Train split: %d records → %s/", len(train_records), train_dir)

    # ── Generate test split ───────────────────────────────────────────────────
    console.rule("[bold]Test split")
    test_records = generate_split(
        mt=mt,
        multi_dataset=multi_dataset,
        predicate_names=test_names,
        layers=layers,
        n_p_minus_per_tp=args.n_p_minus_per_tp,
        harmless_prefills=harmless_prefills,
        max_gen_tokens=args.max_gen_tokens,
        eval_timeout=args.eval_timeout,
        include_mlp=include_mlp,
        include_attn=include_attn,
        out_dir=test_dir,
        augment=False,  # no augmentation on test split
        gen_batch_size=args.gen_batch_size,
        lat_batch_size=args.lat_batch_size,
    )
    logger.info("Test split:  %d records → %s/", len(test_records), test_dir)

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="Clock probe dataset summary", show_lines=False)
    table.add_column("Split", style="bold")
    table.add_column("Predicates", justify="right")
    table.add_column("Records", justify="right")
    table.add_column("Layers cached", justify="right")
    table.add_column("MLP", justify="center")
    table.add_column("Attn", justify="center")
    table.add_column("Directory")
    for split, names, recs, d in [
        ("train", train_names, train_records, train_dir),
        ("test", test_names, test_records, test_dir),
    ]:
        table.add_row(
            split,
            str(len(names)),
            str(len(recs)),
            f"{layers[0]}–{layers[-1]}",
            "✓" if include_mlp else "–",
            "✓" if include_attn else "–",
            str(d),
        )
    console.print(table)


if __name__ == "__main__":
    main()
