"""Judge the ICL GoalScope readouts (``icl_goalscope.py`` output) with a binary LM judge (Claude Sonnet 4.6, as in
App L.2.1) against each rule of the task: one rule for single-relation tasks, one per sub-task for composition /
superposition. Writes ``verdicts`` (eval positions) and ``all_verdicts`` (the site example) in place, as lists of
per-rule booleans, and prints the paper-style rates.

    python scripts/icl/icl_judge.py results/icl_goalscope/*.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from src.utils.oracle_llms import ask_claude  # noqa: E402

SYSTEM = "You are a strict grader. Answer with exactly one word: yes or no."
PROMPT = """A language model is answering few-shot questions that all follow a hidden rule. We asked the model to describe, in its own words, what task it is performing. Its description:
<description>{readout}</description>

Does the description identify this rule: "{rule}"?
Answer "yes" if it names this rule or an equivalent one (e.g. "finding opposites" for "the antonym of the word"), even if it also mentions other tasks alongside it. It need not match every detail of the rule (letter case, the exact kind of input or sequence, e.g. "the next number in a sequence" for "the next item in a sequence"); it must name the same core relation. Answer "no" if it only names different or vaguely related tasks (e.g. "identifying patterns", "answering questions"), or no task at all.
Answer "yes" or "no"."""


def judge(readout: str, rule: str) -> bool:
    if not readout.strip():
        return False
    a = ask_claude(PROMPT.format(readout=readout.strip(), rule=rule), max_tokens=300, temperature=0.0,
                   system_prompt=SYSTEM).strip().lower()
    words = [w.strip(".,!?\"'*:") for w in a.split()]
    last = next((w for w in reversed(words) if w in ("yes", "no")), "no")   # the judge may reason first
    return last == "yes"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for fp in args.files:
        d = json.load(open(fp))
        rules = d["rules"]
        with ThreadPoolExecutor(args.workers) as ex:
            for ev in d["prompts"]:
                ev["verdicts"] = list(ex.map(lambda r: [judge(r, rule) for rule in rules], ev["readouts"]))
                if "all_readouts" in ev:
                    ev["all_verdicts"] = list(ex.map(lambda r: [judge(r, rule) for rule in rules], ev["all_readouts"]))
        v = [x for ev in d["prompts"] for x in ev["verdicts"]]
        d["rate_any"] = sum(any(x) for x in v) / len(v)
        d["rate_all"] = sum(all(x) for x in v) / len(v)
        json.dump(d, open(fp, "w"), ensure_ascii=False, indent=1)
        print(f"{d['task']:<24} {d['setting']:<14} any {d['rate_any']:.2f}" + (f" · all {d['rate_all']:.2f}" if len(rules) > 1 else "")
              + f"  (n={len(v)})")


if __name__ == "__main__":
    main()
