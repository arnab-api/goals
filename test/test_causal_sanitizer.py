"""How many existing causality successes does the trivial-read-out sanitizer compromise?

Re-grades the ALREADY-SAVED OV-reader causality records with
``eval_goalscope.is_trivial_causal_readout``: every record currently ``hard == "YES"`` whose
C++ read-out is a bare scalar sign/zero comparison (``value > 0`` / ``value == 0`` / ...) would be
forced to NO under the sanitizer. Focus on the upper stack (layer > 40), where genuine goal signal
is supposed to live — if too many of those successes are trivial, the layer had less real causality
than ``caus_all`` implied. No model is loaded (pure JSON re-grade).

Run:  conda run -n goals python test/test_causal_sanitizer.py
"""
import glob
import json
import sys
from pathlib import Path

sys.path.append(".")

from scripts.evaluation.eval_goalscope import is_trivial_causal_readout  # noqa: E402

ROOT = "results/evaluation/ov-reader/Qwen2.5-Coder-32B-Instruct"
LAYER_MIN = 40  # "> 40 layers"


def _lk(name):
    return 10**9 if name == "goalscope" else int(name)


def main():
    for dt in ("int", "list[int]"):
        print(f"===== dtype={dt} · compromised YES under trivial-read-out sanitizer =====")
        print(f"  {'layer':>9} {'YES':>5} {'trivial':>8} {'compromised':>12} {'YES_clean':>9}")
        tot_yes = tot_triv = 0
        for ldir in sorted((p for p in Path(ROOT).iterdir() if (p / "causality").exists()),
                           key=lambda p: _lk(p.name)):
            if isinstance(_lk(ldir.name), int) and _lk(ldir.name) <= LAYER_MIN:
                continue
            leaves = [x for x in (ldir / "causality").glob("*") if f"dtype={dt}__" in x.name]
            if not leaves:
                continue
            recs = [json.loads(l) for l in open(leaves[0] / "records.jsonl")]
            yes = [r for r in recs if r["hard"] == "YES"]
            triv = [r for r in yes if is_trivial_causal_readout(r["readout"])]
            n_yes, n_triv = len(yes), len(triv)
            frac = n_triv / n_yes if n_yes else float("nan")
            print(f"  {ldir.name:>9} {n_yes:>5} {n_triv:>8} {frac:>11.1%} {n_yes - n_triv:>9}")
            if ldir.name != "goalscope":
                tot_yes += n_yes
                tot_triv += n_triv
        frac = tot_triv / tot_yes if tot_yes else float("nan")
        print(f"  {'ALL>40':>9} {tot_yes:>5} {tot_triv:>8} {frac:>11.1%} {tot_yes - tot_triv:>9}")
        print()


if __name__ == "__main__":
    main()
