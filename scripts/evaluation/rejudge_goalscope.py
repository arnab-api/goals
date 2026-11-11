"""Re-grade a finished GoalScope eval leaf with a different LLM judge (default: OpenAI gpt-5) and measure agreement
with the stored Claude verdicts. Offline: no model, no GPU, no re-generation. Each record's stored judge prompt
(``hard_prompt``) is sent with the same system prompt the run used (``FAITH_HARD_SYS`` / ``CAUSAL_HARD_SYS``, checked
against the leaf's ``hard_rubric_tag``) and parsed with the same ``<verdict>`` parser.

Records the original run never sent to a judge (empty read-out -> MALFORMED; causality read-outs the sanitizer marks as
a trivial sign/zero test -> NO) keep their deterministic verdict and are excluded from the agreement statistics.

The source leaf is only read. Output (resumable, one line per record):
    results/goalscope/<model>/<mode>/<strategy>/<alpha>/<leaf>/rejudge=<judge>/   (--explain: .../<leaf>/rationale=<judge>/)
        records.jsonl   source record fields + judge2_verdict / judge2_response
        summary.json    per-pivot hard_yes under both judges, agreement, Cohen's kappa, confusion

    python -m scripts.evaluation.rejudge_goalscope --mode faithfulness
    python -m scripts.evaluation.rejudge_goalscope --mode causality
    # judge reasoning (same rubric + "explain, then verdict"; same judge as the paper):
    python -m scripts.evaluation.rejudge_goalscope --mode faithfulness --judge claude-sonnet-4-6 --explain --max-tokens 1024
    # smoke: --limit 20 (writes judge=<judge>__limit=20, never the full output)
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(".")

from scripts.evaluation.eval_goalscope import (  # noqa: E402
    CAUSAL_HARD_SYS,
    FAITH_HARD_SYS,
    _parse_verdict,
)
from src.utils.env_utils import DEFAULT_RESULTS_DIR  # noqa: E402
from src.utils.oracle_llms import ask_claude, ask_gpt  # noqa: E402

# --explain: same rubric, but the judge writes a short justification before the <verdict> tag (the reported rubric
# forbids any text besides the tag, so the stored responses carry no reasoning). Only this sentence changes.
_VERDICT_ONLY = ("Put your final answer inside <verdict>{ans}</verdict> tags, with no other text or explanation, "
                 "to facilitate automated parsing.")
_EXPLAIN = ("First explain your judgment in 1-3 sentences (what in the description is or is not relevant to the code), "
            "then put your final answer inside <verdict>{ans}</verdict> tags, to facilitate automated parsing.")


def explain_variant(sys_prompt: str) -> str:
    assert sys_prompt.endswith(_VERDICT_ONLY), "rubric's output instruction changed; update _VERDICT_ONLY"
    return sys_prompt[: -len(_VERDICT_ONLY)] + _EXPLAIN

PIVOTS = ["prev_end", "indent", "var", "eq"]  # mean4 pivots (first_impl is never reported)
CORE_LEAF = (
    "heads=goal__layers=35-46__nsteer=all__maxnew=64__span=frontier__dtype=all__nspec=253__nc=2-3-4__seed=123"
)


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def summarize(recs: list[dict]) -> dict:
    def block(rs):
        j = [r for r in rs if r["judged"]]
        a = [r["hard"] == "YES" for r in j]
        b = [r["judge2_verdict"] == "YES" for r in j]
        return dict(
            n=len(rs),
            n_judged=len(j),
            hard_yes_claude=sum(r["hard"] == "YES" for r in rs) / max(len(rs), 1),
            hard_yes_judge2=sum(r["judge2_verdict"] == "YES" for r in rs) / max(len(rs), 1),
            agreement=sum(x == y for x, y in zip(a, b)) / max(len(j), 1),
            kappa=kappa(a, b),
            claude_yes_judge2_no=sum(x and not y for x, y in zip(a, b)),
            claude_no_judge2_yes=sum(y and not x for x, y in zip(a, b)),
        )

    by = {p: block([r for r in recs if r["anchor"] == p]) for p in sorted({r["anchor"] for r in recs})}
    four = [r for r in recs if r["anchor"] in PIVOTS]
    out = dict(
        by_anchor=by,
        mean4=dict(
            claude=sum(by[p]["hard_yes_claude"] for p in PIVOTS if p in by) / 4,
            judge2=sum(by[p]["hard_yes_judge2"] for p in PIVOTS if p in by) / 4,
        ),
        pooled4=block(four),
        by_datatype={d: block([r for r in four if r["datatype"] == d]) for d in sorted({r["datatype"] for r in four})},
        label_confusion=Counter(f"{r['hard']}->{r['judge2_verdict']}" for r in four if r["judged"]).most_common(),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["faithfulness", "causality"])
    ap.add_argument("--model", default="Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--strategy", default="additive_norm")
    ap.add_argument("--alpha", default="0.1")
    ap.add_argument("--leaf", default=CORE_LEAF)
    ap.add_argument("--judge", default="gpt-5", help="gpt-* / o* -> ask_gpt; claude-* -> ask_claude")
    ap.add_argument("--explain", action="store_true",
                    help="ask for a 1-3 sentence justification before the verdict (writes <leaf>/rationale=<judge>/)")
    ap.add_argument("--reasoning-effort", default=None, help="gpt-5 reasoning_effort (default: API default)")
    # gpt-5's max_completion_tokens also covers hidden reasoning; 512 (the Claude budget) can return empty content
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N records (separate output dir)")
    args = ap.parse_args()

    leaf = args.leaf + ("__stub=func" if args.mode == "causality" and args.leaf == CORE_LEAF else "")  # v2-probe causality leaf
    rel = os.path.join(args.model, args.mode, args.strategy, args.alpha, leaf)
    src = os.path.join(DEFAULT_RESULTS_DIR, "goalscope", rel)
    sys_prompt = CAUSAL_HARD_SYS if args.mode == "causality" else FAITH_HARD_SYS
    tag = json.load(open(os.path.join(src, "config.json")))["hard_rubric_tag"]
    assert hashlib.md5(sys_prompt.encode()).hexdigest()[:8] == tag, f"rubric changed since the run ({tag})"
    records = [json.loads(l) for l in open(os.path.join(src, "records.jsonl"))]
    if args.limit:
        records = records[: args.limit]

    if args.explain:
        sys_prompt = explain_variant(sys_prompt)
    jtag = f"judge={args.judge}" + (f"__effort={args.reasoning_effort}" if args.reasoning_effort else "")
    jtag += f"__limit={args.limit}" if args.limit else ""
    # side-runs live inside their GoalScope leaf: <leaf>/rejudge=<judge>/ or <leaf>/rationale=<judge>/
    out = os.path.join(src, ("rationale=" if args.explain else "rejudge=") + jtag[len("judge="):])
    assert os.path.realpath(out) != os.path.realpath(src)
    os.makedirs(out, exist_ok=True)
    rec_path = os.path.join(out, "records.jsonl")
    done = {}
    if os.path.exists(rec_path):  # resume: never rewrite finished lines
        for l in open(rec_path):
            r = json.loads(l)
            done[(r["spec"], r["anchor"])] = r
    todo = [r for r in records if (r["spec"], r["anchor"]) not in done]
    print(f"{src}\n-> {out}\n{len(records)} records, {len(done)} done, {len(todo)} to judge", flush=True)

    def work(r):
        r = dict(r)
        if r.get("hard_prompt"):
            if args.judge.startswith("claude"):
                resp = ask_claude(prompt=r["hard_prompt"], system_prompt=sys_prompt, max_tokens=args.max_tokens,
                                  model_name=args.judge, use_cache=True)
            else:
                resp = ask_gpt(prompt=r["hard_prompt"], system_prompt=sys_prompt, max_tokens=args.max_tokens,
                               model_name=args.judge, reasoning_effort=args.reasoning_effort, use_cache=True)
            r.update(judged=True, judge2_verdict=_parse_verdict(resp), judge2_response=resp)
        else:  # deterministic verdict in the original run (MALFORMED / sanitizer NO): carried over
            r.update(judged=False, judge2_verdict=r["hard"], judge2_response=None)
        return r

    with open(rec_path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done[(r["spec"], r["anchor"])] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    recs = [done[(r["spec"], r["anchor"])] for r in records]
    summ = summarize(recs)
    summ.update(source_leaf=src, judge=args.judge, explain=args.explain,
                system_prompt_tag=hashlib.md5(sys_prompt.encode()).hexdigest()[:8], reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens, hard_rubric_tag=tag, n_records=len(recs))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    p4 = summ["pooled4"]
    print(f"mean4 claude {summ['mean4']['claude']:.3f}  {args.judge} {summ['mean4']['judge2']:.3f}  |  "
          f"agreement {p4['agreement']:.3f}  kappa {p4['kappa']:.3f}  (n_judged {p4['n_judged']})")
    for p, b in summ["by_anchor"].items():
        print(f"  {p:10s} claude {b['hard_yes_claude']:.3f}  judge2 {b['hard_yes_judge2']:.3f}  "
              f"agree {b['agreement']:.3f}  kappa {b['kappa']:.3f}")


if __name__ == "__main__":
    main()
