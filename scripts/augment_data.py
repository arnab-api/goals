"""Augmentation preview / inspection tool.

Loads the dataset with online augmentation (in-memory) and prints counts and
sample examples.  Nothing is written to disk — JSON files remain clean.

Usage:
    python scripts/augment_data.py --dataset multi_predicate [--extract-descriptions]
    python scripts/augment_data.py --dataset incomplete_code
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_multi_predicate(extract_descriptions: bool) -> None:
    from src.goals.augment import extract_short_descriptions
    from src.goals.predicate_data import MultiPredicateDataset

    ds = MultiPredicateDataset.load_dataset(augment=True)
    total_orig = 0
    total_aug = 0

    for tp in ds.groups:
        originals = [s for s in tp.samples if s.is_original]
        augmented = [s for s in tp.samples if not s.is_original]
        total_orig += len(originals)
        total_aug += len(augmented)

        if extract_descriptions:
            logger.info(
                f"[{tp.name}] p_plus.short_description = {tp.p_plus.short_description!r}"
            )
            for i, s in enumerate(originals):
                if s.p_minus:
                    logger.info(
                        f"[{tp.name}] sample {i} p_minus.short_description = "
                        f"{s.p_minus.short_description!r}"
                    )
                logger.info(
                    f"[{tp.name}] sample {i} p_plus_short_description = "
                    f"{s.p_plus_short_description!r}"
                )

        type_counts: dict[str, int] = {}
        for s in augmented:
            aug_list = s.augmentation or []
            t = aug_list[-1].get("type", "unknown") if aug_list else "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1

        print(
            f"[{tp.name}] originals={len(originals)}  "
            f"augmented={len(augmented)}  breakdown={type_counts}"
        )

    print(f"\nTotal: {total_orig} originals, {total_aug} augmented (in-memory only)")
    print("(No data written to disk — augmentation is online/in-memory)")


def run_incomplete_code() -> None:
    from src.goals.data import IncompleteCodeDataset

    ds = IncompleteCodeDataset.load_dataset(augment=True)
    originals = [s for s in ds.samples if s.is_original]
    augmented = [s for s in ds.samples if not s.is_original]

    type_counts: dict[str, int] = {}
    for s in augmented:
        aug_list = s.augmentation or []
        t = aug_list[-1].get("type", "unknown") if aug_list else "unknown"
        type_counts[t] = type_counts.get(t, 0) + 1

    print(
        f"incomplete_code: originals={len(originals)}  "
        f"augmented={len(augmented)}  breakdown={type_counts}"
    )
    print("(No data written to disk — augmentation is online/in-memory)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["multi_predicate", "incomplete_code"],
        required=True,
        help="Which dataset to preview.",
    )
    parser.add_argument(
        "--extract-descriptions",
        action="store_true",
        help="Print short_description fields (multi_predicate only).",
    )
    args = parser.parse_args()

    if args.dataset == "multi_predicate":
        run_multi_predicate(extract_descriptions=args.extract_descriptions)
    else:
        if args.extract_descriptions:
            logger.warning("--extract-descriptions ignored for incomplete_code dataset")
        run_incomplete_code()


if __name__ == "__main__":
    main()
