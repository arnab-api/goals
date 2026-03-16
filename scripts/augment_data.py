"""Augment data files with surface-level variants.

Usage:
    python scripts/augment_data.py --dataset multi_predicate [--dry-run] [--extract-descriptions]
    python scripts/augment_data.py --dataset incomplete_code [--dry-run]

Flags:
    --dry-run               Print counts but do not write to disk.
    --extract-descriptions  Populate short_description fields before generating
                            variants (required for swap_predicate_order; only
                            applies to multi_predicate).
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_multi_predicate(dry_run: bool, extract_descriptions: bool) -> None:
    from src.goals.augment import extract_short_descriptions, generate_variants
    from src.goals.predicate_data import MultiPredicateDataset

    ds = MultiPredicateDataset.load_dataset()
    total_orig = 0
    total_new = 0

    for tp in ds.groups:
        originals = [s for s in tp.samples if s.augmentation is None]
        total_orig += len(originals)

        if extract_descriptions:
            extract_short_descriptions(tp)
            logger.info(
                f"[{tp.name}] p_plus.short_description = {tp.p_plus.short_description!r}"
            )
            for i, s in enumerate(tp.samples):
                if s.augmentation is None and s.p_minus:
                    logger.info(
                        f"[{tp.name}] sample {i} p_minus.short_description = "
                        f"{s.p_minus.short_description!r}"
                    )
            if not dry_run:
                tp.save()

        new_samples = generate_variants(tp)
        total_new += len(new_samples)

        # Per-type breakdown
        type_counts: dict[str, int] = {}
        for s in new_samples:
            t = (s.augmentation or {}).get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        print(
            f"[{tp.name}] originals={len(originals)}  "
            f"new={len(new_samples)}  breakdown={type_counts}"
        )

        if not dry_run:
            tp.samples.extend(new_samples)
            tp.save()
            print(f"  → saved {tp._source_path}")

    print(f"\nTotal: {total_orig} originals → {total_new} new augmented samples")
    if dry_run:
        print("(DRY RUN — nothing written)")


def run_incomplete_code(dry_run: bool) -> None:
    from src.goals.augment import generate_variants_incomplete
    from src.goals.data import IncompleteCodeDataset
    from src.utils.env_utils import DEFAULT_DATA_DIR

    data_path = os.path.join(DEFAULT_DATA_DIR, "incomplete_code.json")
    ds = IncompleteCodeDataset.load_dataset(data_path)

    originals = [s for s in ds.samples if s.augmentation is None]
    all_new = []

    type_counts: dict[str, int] = {}
    for sample in originals:
        new_samples = generate_variants_incomplete(sample)
        all_new.extend(new_samples)
        for s in new_samples:
            t = (s.augmentation or {}).get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

    print(
        f"incomplete_code: originals={len(originals)}  "
        f"new={len(all_new)}  breakdown={type_counts}"
    )

    if not dry_run:
        ds.samples.extend(all_new)
        # IncompleteCodeDataset has no save() — write the raw list
        data = [s.to_dict() for s in ds.samples]
        with open(data_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"  → saved {data_path}")
    else:
        print("(DRY RUN — nothing written)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["multi_predicate", "incomplete_code"],
        required=True,
        help="Which dataset to augment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts but do not write to disk.",
    )
    parser.add_argument(
        "--extract-descriptions",
        action="store_true",
        help="Populate short_description fields before generating variants "
             "(multi_predicate only).",
    )
    args = parser.parse_args()

    if args.dataset == "multi_predicate":
        run_multi_predicate(
            dry_run=args.dry_run,
            extract_descriptions=args.extract_descriptions,
        )
    else:
        if args.extract_descriptions:
            logger.warning("--extract-descriptions ignored for incomplete_code dataset")
        run_incomplete_code(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
