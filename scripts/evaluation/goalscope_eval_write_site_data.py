"""Write the "Faithfulness" tab of the demo site (goals-web: site/faithfulness.html <- site/demos/data/faithfulness/) from
the paper's main GoalScope faithfulness run (Coder-32B, the 122 goal heads, additive_norm α 0.1, layers 35-45, all 253
specs). Per spec: the task prompt, the assistant's code prefix with the four pre-implementation pivots
(prev_end, indent, var, eq) marked, the LM's own rollout of the frontier line, and at each pivot the GoalScope read-out
with the judge's verdict (the verdict that is in the paper) and the judge's reasoning.

The reasoning comes from ``rejudge_goalscope.py --judge claude-sonnet-4-6 --explain`` (same judge, same rubric, asked to
justify before the verdict; the reported rubric asks for the verdict tag only). When that reasoned verdict differs from
the reported one, both are kept and the page says so. Pure parsing (no model, no API): needs the Coder-32B tokenizer.

    python scripts/evaluation/goalscope_eval_write_site_data.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(".")

from src.goals.site_data import SITE  # noqa: E402
from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402

PIVOTS = ["prev_end", "indent", "var", "eq"]
MODEL = "Qwen2.5-Coder-32B-Instruct"
LEAF = "heads=goal__layers=35-46__nsteer=all__maxnew=64__span=frontier__dtype=all__nspec=253__nc=2-3-4__seed=123"
REL = f"{MODEL}/faithfulness/additive_norm/0.1/{LEAF}"
ASSISTANT = "<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rationale-judge", default="claude-sonnet-4-6")
    ap.add_argument("--out", default=os.path.join(SITE, "demos", "data", "faithfulness"))
    args = ap.parse_args()
    from transformers import AutoTokenizer

    src = os.path.join(DEFAULT_RESULTS_DIR, "evaluation", "goalscope", REL)
    rat = os.path.join(DEFAULT_RESULTS_DIR, "evaluation", "goalscope_rationale", REL, f"judge={args.rationale_judge}")
    samples = [json.loads(l) for l in open(os.path.join(src, "samples.jsonl"))]
    recs = defaultdict(dict)
    for l in open(os.path.join(src, "records.jsonl")):
        r = json.loads(l)
        recs[r["spec"]][r["anchor"]] = r
    why = {}
    if os.path.exists(os.path.join(rat, "records.jsonl")):
        for l in open(os.path.join(rat, "records.jsonl")):
            r = json.loads(l)
            why[(r["spec"], r["anchor"])] = r
    print(f"{len(samples)} specs; reasoning for {len(why)} records from {rat}")
    tok = AutoTokenizer.from_pretrained(f"Qwen/{MODEL}")
    os.makedirs(args.out, exist_ok=True)
    index, n_missing = [], 0
    for s in samples:
        sid, prompt = s["spec_id"], s["prompt"]
        piv = s["pivots"] if isinstance(s["pivots"], dict) else json.loads(s["pivots"].replace("'", '"'))
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        a0 = prompt.index(ASSISTANT) + len(ASSISTANT)
        user = prompt[prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n"): prompt.index("<|im_end|>\n<|im_start|>assistant")]
        at = {v: k for k, v in piv.items() if k in PIVOTS}
        assert piv["eq"] == len(offs) - 1, sid            # eq is the prompt's last token (rollout starts after it)
        code, cur = [], a0                                 # code prefix as segments: plain text / pivot tokens
        for i, (b, e) in enumerate(offs):
            if e <= a0 or i not in at:
                continue
            if b > cur:
                code.append({"text": prompt[cur:b]})
            code.append({"text": prompt[b:e], "pivot": at[i]})
            cur = e
        if cur < len(prompt):
            code.append({"text": prompt[cur:]})
        readouts = []
        for p in PIVOTS:
            r = recs[sid].get(p)
            if r is None:
                continue
            w = why.get((sid, p))
            if w is None and r.get("hard_prompt"):
                n_missing += 1
            readouts.append(dict(
                pivot=p, readout=r["readout"], verdict=r["hard"],
                reasoning=(w["judge2_response"] if w and w.get("judged") else None),
                reasoned_verdict=(w["judge2_verdict"] if w and w.get("judged") else None),
                judged=bool(r.get("hard_prompt")),
            ))
        ex = dict(id=sid, datatype=s["datatype"], frontier_id=s["frontier_id"], condition=recs[sid][PIVOTS[0]]["condition"],
                  user=user, code=code, rollout=s["completion"], impl_span=s["impl_span"], readouts=readouts)
        json.dump(ex, open(os.path.join(args.out, f"{sid}.json"), "w"), indent=1)
        index.append(dict(id=sid, datatype=s["datatype"], frontier_id=s["frontier_id"], condition=ex["condition"],
                          n_yes=sum(r["verdict"] == "YES" for r in readouts), n=len(readouts)))
    summ = json.load(open(os.path.join(src, "summary.json")))["by_anchor"]
    meta = dict(model=MODEL, heads=122, strategy="additive_norm", alpha=0.1, layers="35-45", judge="claude-sonnet-4-6",
                rationale_judge=args.rationale_judge, by_pivot={p: summ[p]["hard_yes"] for p in PIVOTS})
    json.dump(dict(meta=meta, examples=index), open(os.path.join(args.out, "index.json"), "w"), indent=1)
    print(f"wrote {len(index)} specs -> {args.out}  (judged records without reasoning: {n_missing})")


if __name__ == "__main__":
    main()
