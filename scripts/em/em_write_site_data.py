"""Write the EM tab of the demo site (goals-web: site/em.html <- site/demos/data/em/, via src.goals.site_data) from our App L.2.2 reproduction:
GoalScope on every response token (``em_goalscope_all_tokens.py``), each readout labelled benign / concerning by the
Claude oracle (``em_tone_judge.py``). Provisional until the co-author's script arrives
(agent_reports/claude/analysis/em-goalscope-reproduction.md); the page shows a banner saying so.

    python scripts/em/em_write_site_data.py [--dir results/em_goalscope_repro/all_tokens]
"""

import argparse
import json
import sys

sys.path.append(".")

from src.goals.site_data import rebuild_indexes, write_em_example  # noqa: E402

MODELS = {"innocent": "Qwen2.5-32B-Instruct", "em": "Qwen2.5-32B-Instruct_risky-financial-advice"}
ORDER = ["quickbuck", "husband", "ruler", "gender"]
BANNER = ("Provisional reproduction: read-outs from our rerun of the paper's EM experiment with some settings inferred "
          "(verbalizer system prompt, readout length). They are real model outputs but do not match the paper's "
          "figures verbatim. Benign / concerning labels are from an LLM judge (Claude Sonnet 4.6).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/em_goalscope_repro/all_tokens")
    args = ap.parse_args()
    for k, ex in enumerate(ORDER):
        d = json.load(open(f"{args.dir}/{ex}.json"))
        sides = {}
        for side in ("innocent", "em"):
            s = d["sides"][side]
            assert len(s["tones"]) == len(s["readouts"]) == len(s["positions"])
            # an unparsed judge answer ("?...") is shown as benign
            ro = {p: (r.strip(), "concerning" if t == "concerning" else "benign")
                  for p, r, t in zip(s["positions"], s["readouts"], s["tones"])}
            rows = [{"pos": i, "token": t, "patched_gen": ro[i][0] if i in ro else "", "scoped": i in ro,
                     **({"tone": ro[i][1]} if i in ro else {})} for i, t in enumerate(s["tokens"])]
            # pinned on open: the newline before the response and the last prefill token (the paper's two callouts)
            sides[side] = {"model": MODELS[side], "rows": rows, "continuation": "",
                           "pins": [d["n_prompt"] - 1, d["n_prefilled"] - 1]}
        n_c = {side: sum(t == "concerning" for t in d["sides"][side]["tones"]) for side in sides}
        n = {side: len(d["sides"][side]["tones"]) for side in sides}
        caption = (f"Concerning read-outs: innocent {n_c['innocent']}/{n['innocent']} tokens, "
                   f"EM {n_c['em']}/{n['em']} tokens.")
        meta = {"order": k, "reproduction": True, "banner": BANNER, "source": f"{args.dir}/{ex}.json", **d["settings"]}
        print(write_em_example(ex, title=d["question"], caption=caption, sides=sides, meta=meta), "·", caption)
    rebuild_indexes()


if __name__ == "__main__":
    main()
