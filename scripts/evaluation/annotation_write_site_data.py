"""Write the item bank of the human-annotation page (goals-web branch ``human-annotation``: site/annotate.html <-
site/demos/data/annotation/items.json). One item per (spec, pivot) for the pivots indent / var / eq of the paper's
main faithfulness run (Coder-32B, the 122 goal heads, additive_norm α 0.1, 253 specs): the read-out, the paper's
Claude verdict and the gpt-5 re-judge verdict (``rejudge_goalscope.py``), and each judge's reasoning
(``rejudge_goalscope.py --explain``: the same judge and rubric asked to justify before its verdict; the verdict shown
stays the original one). The page shows the spec from the Eval tab's
data (site/demos/data/faithfulness/<spec>.json, written by goalscope_eval_write_site_data.py) and samples items.

LM verdicts are mapped onto the page's four labels: HEAVY_LOOP -> MALFORMED (the page's MALFORMED includes short
repeated cycles); UNPARSED -> null (no verdict).

    python scripts/evaluation/annotation_write_site_data.py
"""
import argparse
import json
import re
import os
import sys
from collections import Counter

sys.path.append(".")

from src.goals.site_data import SITE  # noqa: E402
from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402

PIVOTS = ["indent", "var", "eq"]
REL = ("Qwen2.5-Coder-32B-Instruct/faithfulness/additive_norm/0.1/"
       "heads=goal__layers=35-46__nsteer=all__maxnew=64__span=frontier__dtype=all__nspec=253__nc=2-3-4__seed=123")
MAP = {"YES": "YES", "NO": "NO", "NO_INFO": "NO_INFO", "MALFORMED": "MALFORMED", "HEAVY_LOOP": "MALFORMED"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpt-judge", default="gpt-5")
    ap.add_argument("--out", default=os.path.join(SITE, "demos", "data", "annotation", "items.json"))
    args = ap.parse_args()
    src = os.path.join(DEFAULT_RESULTS_DIR, "evaluation", "goalscope", REL, "records.jsonl")
    gpt = os.path.join(DEFAULT_RESULTS_DIR, "evaluation", "goalscope_rejudge", REL, f"judge={args.gpt_judge}",
                       "records.jsonl")
    g = {}
    for l in open(gpt):
        r = json.loads(l)
        g[(r["spec"], r["anchor"])] = r["judge2_verdict"]
    why = {}
    for k, judge in (("claude", "claude-sonnet-4-6"), ("gpt", args.gpt_judge)):
        f = os.path.join(DEFAULT_RESULTS_DIR, "evaluation", "goalscope_rationale", REL, f"judge={judge}", "records.jsonl")
        for l in open(f):
            r = json.loads(l)
            txt = (r.get("judge2_response") or "") if r.get("judged") else "Not sent to the judge: the read-out is empty."
            why[(k, r["spec"], r["anchor"])] = re.sub(r"<verdict>.*?</verdict>", "", txt, flags=re.S).strip()
    items = []
    for l in open(src):
        r = json.loads(l)
        if r["anchor"] not in PIVOTS:
            continue
        items.append(dict(id=f"{r['spec']}|{r['anchor']}", spec=r["spec"], datatype=r["datatype"], pivot=r["anchor"],
                          frontier_id=r["frontier_id"], condition=r["condition"], readout=r["readout"], impl_span=r["impl_span"],
                          claude=MAP.get(r["hard"]), gpt=MAP.get(g.get((r["spec"], r["anchor"]))),
                          claude_why=why.get(("claude", r["spec"], r["anchor"])),
                          gpt_why=why.get(("gpt", r["spec"], r["anchor"]))))
    assert all(i["gpt"] is not None or g.get((i["spec"], i["pivot"])) == "UNPARSED" for i in items)
    meta = dict(model="Qwen2.5-Coder-32B-Instruct", heads=122, strategy="additive_norm", alpha=0.1, layers="35-45",
                judges={"claude": "claude-sonnet-4-6", "gpt": args.gpt_judge}, pivots=PIVOTS,
                n_specs=len({i["spec"] for i in items}))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(meta=meta, items=items), open(args.out, "w"), separators=(",", ":"))
    print(f"{len(items)} items, {meta['n_specs']} specs -> {args.out}  (reasoning: claude "
          f"{sum(bool(i['claude_why']) for i in items)}, gpt {sum(bool(i['gpt_why']) for i in items)})")
    print("claude:", Counter(i["claude"] for i in items), "\ngpt:", Counter(i["gpt"] for i in items))


if __name__ == "__main__":
    main()
