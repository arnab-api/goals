"""Write the ICL tab of the demo site (goals-web: site/icl.html <- site/demos/data/icl/, via src.goals.site_data) from ``icl_goalscope.py`` +
``icl_judge.py``: per task, prompt #0 (unscoped context) and its greedy answer with a readout at every generated token; badge = the judge's verdict
(composition / superposition: "yes" when at least one sub-task is named); the percentage = the task's judged rate over
the paper's evaluation positions and 20 prompts (at least one sub-task for composition / superposition). Replaces the
placeholder examples. The percentage shown is the paper's per-task rate (``PAPER_RATE``), not our rerun's; our rates stay
in ``meta`` (``rate_A``, ``paper_protocol_rate_any``).

    python scripts/icl/icl_write_site_data.py [--dir results/icl_goalscope]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(".")

from src.goals.site_data import rebuild_indexes, write_icl_example  # noqa: E402
from scripts.icl.icl_goalscope import TASKS  # noqa: E402

CATEGORY = {t: (v[0] if v[0] != "single" else v[2]) for t, v in TASKS.items()}   # abstractive / extractive / ...
RULE = {  # shown on the page
    "antonym": "the antonym of the word", "country-capital": "the capital city of the country",
    "person-occupation": "the occupation of the person", "singular-plural": "the plural of the noun",
    "present-past": "the past tense of the verb", "english-spanish": "the Spanish translation of the word",
    "conll2003_person": "the person named in the sentence", "fruit_v_animal_3": "the fruit among the animals",
    "antonym-spanish": "the antonym of the word, then its Spanish translation",
    "plus-ten-times-two": "add ten, then multiply by two",
    "country-superposition": "the capital, the continent, or the name in upper case (mixed)",
}
# The page's percentage = the per-task rate reported in the paper (App L.2.1, fig:noncode_fv_eval_Atoken): judged rate
# at the "A" token of the last 3 demonstrations, 20 prompts x 3 readouts, 122 goal heads. Source: the co-authors'
# summary Overleaf-draft/REF/icl/goalheads_vs_fvheads_results_Atok.csv, column goal_122_thr0.9_Atok_hard_yes.
# Composition / superposition tasks here are our own rebuilds with no paper value -> no percentage.
PAPER_RATE = {
    "adjective_v_verb_3": 0.0, "ag_news": 0.1833, "animal_v_object_3": 0.0, "antonym": 0.9167, "capitalize": 0.0167,
    "capitalize_first_letter": 0.5, "capitalize_last_letter": 0.0667, "choose_first_of_3": 0.0,
    "choose_last_of_3": 0.0, "choose_middle_of_3": 0.0167, "color_v_animal_3": 0.0, "concept_v_object_3": 0.0667,
    "conll2003_location": 0.3833, "conll2003_organization": 0.0167, "conll2003_person": 0.6833,
    "country-capital": 0.9333, "country-currency": 0.7, "english-french": 0.1333, "english-german": 0.0833,
    "english-spanish": 0.05, "fruit_v_animal_3": 0.0, "landmark-country": 0.3333, "lowercase_first_letter": 0.7833,
    "lowercase_last_letter": 0.6167, "national_parks": 0.1, "next_item": 0.8833, "object_v_concept_3": 0.0,
    "park-country": 0.7667, "person-instrument": 0.3833, "person-occupation": 0.9333, "person-sport": 0.2167,
    "present-past": 0.6667, "prev_item": 0.7833, "product-company": 0.7333, "sentiment": 0.4167,
    "singular-plural": 0.8333, "synonym": 0.1833, "verb_v_adjective_3": 0.0, "word_length": 0.75,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/icl_goalscope")
    ap.add_argument("--site-dir", default="results/icl_goalscope/site")
    args = ap.parse_args()
    for site in sorted(Path(args.site_dir).glob("*.json")):
        sd = json.load(open(site))                  # 20 prompts x (A, :) tokens + the site example
        if "rate_any" not in sd:
            print("not judged yet:", site.name); continue
        fp = Path(args.dir) / site.name
        if sd["setting"] == "single":               # paper protocol = the "A" of the last 3 demonstrations; prompt
            v = [any(x) for ev in sd["prompts"]       # readouts don't depend on the generation, so take them from here
                 for p, x in zip(ev["positions"], ev["verdicts"])
                 if ev["tokens"][p] == "A" and p < ev["n_prompt"] - 2]
            d = {"rate_any": sum(v) / len(v), "rate_all": sum(v) / len(v)}
        else:                                       # composition / superposition: the separate paper-protocol run
            d = json.load(open(fp))
        # our rerun's judged rate at the "A" tokens (last 3 demonstrations, the query and every generated round; at
        # least one sub-task for composition / superposition); kept in meta, the page shows PAPER_RATE
        va = [any(x) for e in sd["prompts"] for p, x in zip(e["positions"], e["verdicts"]) if e["tokens"][p] == "A"]
        rate_a = sum(va) / len(va)
        ev = sd["prompts"][0]
        n = ev["n_prompt"]           # GoalScope on the prompt's last token + the LM's own generation; the rest is context
        shown = set(ev["positions"]) | {p for p in ev["all_positions"] if p >= n - 1}   # A/: of the last 3 demos + generation
        ro = {p: (r, "yes" if any(v) else "no")
              for p, r, v in zip(ev["all_positions"], ev["all_readouts"], ev["all_verdicts"]) if p in shown}
        gen_a = [p for p in range(n, len(ev["tokens"]) - 1) if ev["tokens"][p] == "A" and ev["tokens"][p + 1] == ":"]
        # pinned on open: the "A" and ":" before each of the first 2 answers the LM generates (the query's "A:" and the
        # first "A:" the model writes itself) -> 4 readouts
        pins = [n - 2, n - 1] + [q for a in gen_a[:1] for q in (a, a + 1)]
        # the page shows the last 3 demonstrations by default (a toggle shows all 10): start at the 8th demo's "Q"
        q_pos = [p for p in range(n - 1) if ev["tokens"][p] == "Q" and ev["tokens"][p + 1] == ":"]
        meta_view = {"collapse_from": q_pos[-4] if len(q_pos) >= 4 else 0, "n_demos": len(q_pos) - 1}
        rows = [{"pos": i, "token": t, "patched_gen": ro[i][0] if i in ro else "", "scoped": i in ro,
                 **({"verdict": ro[i][1]} if i in ro else {})} for i, t in enumerate(ev["tokens"])]
        meta = {"probe": sd["probe"], "probe_system": sd["probe_system"], "rules_judged": sd["rules"],
                "rate_A": rate_a, "n_A": len(va), "rate_A_colon_any": sd["rate_any"], "rate_A_colon_all": sd["rate_all"],
                "paper_protocol_rate_any": d["rate_any"], "paper_protocol_rate_all": d["rate_all"],
                "eval_positions": ev["positions"],
                "source": str(site), **sd["settings"], **meta_view,
                "accuracy_source": "paper (App L.2.1, A token, 20 prompts x 3)" if sd["task"] in PAPER_RATE else None}
        out = write_icl_example(f"{sd['task']}_0", task=sd["task"], category=CATEGORY.get(sd["task"], "abstractive"),
                                rule=RULE.get(sd["task"], sd["rules"][0]), rows=rows, pins=pins, continuation="",
                                accuracy=PAPER_RATE.get(sd["task"]), model=sd["settings"]["model"].split("/")[-1], meta=meta)
        print(out, f"· paper {PAPER_RATE.get(sd['task'])} · ours: A rate {rate_a:.2f} (A+: {sd['rate_any']:.2f}, paper protocol {d['rate_any']:.2f})")
    rebuild_indexes()


if __name__ == "__main__":
    main()
