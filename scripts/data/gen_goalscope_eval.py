#!/usr/bin/env python
"""Generate the LOCKED goalscope eval dataset -> data/goalscope_eval.json (source of truth).

QTASKS bank, OVReader prefill-to-frontier setup (var_eq stub, letters naming), with BALANCED frontier
coverage: every NON-trivial frontier (src.diff.data.TRIVIAL_FRONTIERS excluded) is the target exactly
`reps` times, each with a distinct prefill context. Frontier pool = non-trivial; context (prefill +
trailing predicates) may be any predicate, matching the frontier-only exclusion policy wired into
scripts/evaluation/eval_ov_reader.sample_datapoints.

Each datapoint is a model-agnostic RECIPE (ordered predicate ids + prefill count j); the full
QTransferInstance is rebuilt from QTASKS by id at load time (see
scripts.evaluation.eval_goalscope.load_locked_datapoints). No RNG at eval time — the dataset IS the corpus.

Usage:
    conda run -n goals python scripts/data/gen_goalscope_eval.py            # regenerate the default file
    conda run -n goals python scripts/data/gen_goalscope_eval.py --out /tmp/goalscope_eval.json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

import src.diff.data as DATA  # noqa: E402

DATATYPES = ["int", "list[int]", "str", "list[str]"]
N_CONDITIONS = [2, 3, 4]
DEFAULT_OUT = REPO_ROOT / "data" / "goalscope_eval.json"
SEED = 123
TARGET_N = 64  # aim per datatype; actual = reps * n_frontiers (exact balance)


def gen_datatype(dt, rng, target_n):
    bank = DATA.QTASKS[dt]
    pool = [p.id for p in bank.predicates]  # context pool (all predicates)
    frontiers = [
        p.id for p in DATA.nontrivial_frontiers(dt)
    ]  # eligible targets (non-trivial)
    var_name = bank.var_names[0]
    reps = max(1, round(target_n / len(frontiers)))
    recs, seen = [], set()
    for F in frontiers:
        others_all = [p for p in pool if p != F]
        made, attempts = 0, 0
        while made < reps and attempts < 4000:
            attempts += 1
            T = rng.choice(N_CONDITIONS)
            j = rng.randint(1, T - 1)  # frontier index; progress = subset[:j]
            others = rng.sample(others_all, T - 1)  # the other T-1 predicates
            subset = others[:j] + [F] + others[j:]  # F sits at index j
            progress = subset[:j]
            key = (F, tuple(progress))  # dedup on (frontier, progress)
            if key in seen:
                continue
            seen.add(key)
            recs.append(
                dict(
                    datatype=dt,
                    lang="python",
                    T=T,
                    j=j,
                    predicate_ids=subset,
                    progress_ids=progress,
                    frontier_id=F,
                    var_name=var_name,
                    presentation="listed",
                    naming="letters",
                )
            )
            made += 1
        if made < reps:
            print(f"  [warn] {dt}/{F}: only {made}/{reps} distinct contexts")
    for i, r in enumerate(recs):
        r["spec_id"] = f"{dt}_python_{i:03d}"
    return recs, reps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output json path")
    args = ap.parse_args()

    rng = random.Random(SEED)
    data, counts, reps_by = {}, {}, {}
    for dt in DATATYPES:
        recs, reps = gen_datatype(dt, rng, TARGET_N)
        data[dt] = recs
        counts[dt] = len(recs)
        reps_by[dt] = reps

    out = dict(
        meta=dict(
            description="Locked OVReader-setup corpus (QTASKS bank, prefill-to-frontier, var_eq stub, "
            "letters naming) with BALANCED frontier coverage: each non-trivial frontier is the "
            "target `reps` times with distinct prefill contexts. Rebuild from QTASKS by id.",
            bank="src.diff.data.QTASKS",
            generator="scripts/data/gen_goalscope_eval.py",
            balanced=True,
            reps_per_frontier=reps_by,
            n_conditions=N_CONDITIONS,
            seed=SEED,
            target_n=TARGET_N,
            lang="python",
            datatypes=DATATYPES,
            trivial_frontiers_excluded={
                k: sorted(v) for k, v in DATA.TRIVIAL_FRONTIERS.items()
            },
            counts=counts,
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
        datapoints=data,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    for dt in DATATYPES:
        fr = Counter(r["frontier_id"] for r in data[dt])
        print(
            f"  {dt:9}: {counts[dt]} datapoints | {len(fr)} frontiers x {reps_by[dt]} reps "
            f"| per-frontier counts: {sorted(set(fr.values()))}"
        )


if __name__ == "__main__":
    main()
