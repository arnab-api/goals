"""Reproduce Fig 32 (App: readouts without enumerated sub-tasks): GoalScope read-outs while Qwen2.5-Coder-32B-Instruct
implements the stable marriage algorithm, with the figure's exact prompt and the prefill up to the function signature.
Generates greedily, compares the code with the figure, runs ``apply_goalscope`` with the paper / site config (122 goal
heads, additive_norm, alpha 0.1, steer layers 35-46, 64 new tokens), prints the read-outs at the figure's four marked
positions and saves the rows to ``results/unspecified_readouts/Qwen2.5-Coder-32B-Instruct/stable_marriage.json``. ``--write-web`` also replaces the
site's ``goals-web/site/data/<model>/ds_and_algo/stable_marriage.json`` (via ``src.goals.site_data``).

    python scripts/run_goalscope_fig32_stable_marriage.py [--write-web]
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.append(".")

import torch  # noqa: E402

SYSTEM = ("You are a helpful assistant proficient in programming. Always wrap your code inside <code></code> blocks. "
          "Just the code, no need to comment or explain.")
USER = "Implement the stable marriage algorithm in python"
# the prefill ends with the newline after the signature: without it the LM writes ":  \n" and a different program
PREFILL = "<code>\ndef stable_marriage(men_preferences, women_preferences):\n"

# the code shown in Fig 32 (after the prefill), transcribed from the figure
FIG_CODE = """    n = len(men_preferences)
    men_free = list(range(n))
    women_free = list(range(n))
    men_engaged_to = [None] * n
    women_engaged_to = [None] * n
    men_proposals = [0] * n

    while men_free:
        man = men_free.pop(0)
        woman = men_preferences[man][men_proposals[man]]
        men_proposals[man] += 1

        if women_engaged_to[woman] is None:
            women_engaged_to[woman] = man
            men_engaged_to[man] = woman
        else:
            current_guy = women_engaged_to[woman]
            if women_preferences[woman].index(man) < women_preferences[woman].index(current_guy):
                women_engaged_to[woman] = man
                men_engaged_to[man] = woman
                men_free.append(current_guy)
            else:
                men_free.append(man)

    return list(zip(men_engaged_to, women_engaged_to))
"""

# the figure's marked read-outs: (callout number, the line that follows the marked indentation, occurrence, text)
FIG_READOUTS = [
    (2, "man = men_free.pop(0)", 0, "I'm attempting to match an unmatched man with a woman he hasn't proposed to yet."),
    (1, "men_engaged_to[man] = woman", 0, "I'm assigning the current index to the variable representing the person being matched."),
    (3, "if women_preferences[woman].index(man)", 0, "I'm evaluating which option is preferred based on the current preferences list."),
    (4, "men_free.append(current_guy)", 0, "I'm updating the reference to point to a different object, effectively \"dumping\" the previous one."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--max-gen-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/unspecified_readouts/Qwen2.5-Coder-32B-Instruct/stable_marriage.json")
    ap.add_argument("--write-web", action="store_true")
    args = ap.parse_args()

    from src.goals.head_sets import load_json_heads
    from src.goals.goalscope import apply_goalscope
    from src.models import ModelandTokenizer
    from src.tokens import prepare_input

    model = args.model_key.split("/")[-1]
    heads_fp = str(sorted(Path("head_sets", model).glob("goal__lam20_thr0.9_band*.json"))[0])
    mt = ModelandTokenizer(model_key=args.model_key, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    heads = load_json_heads(heads_fp, mt.n_layer, mt.model.config.num_attention_heads)
    tok = mt.tokenizer

    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n"
              f"<|im_start|>assistant\n{PREFILL}")
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    with torch.no_grad():
        g = mt._model.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                               max_new_tokens=args.max_gen_tokens, do_sample=False)
    gen = tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    code = gen.split("</code>", 1)[0]
    match = code.rstrip() == FIG_CODE.rstrip()
    print(f"generation == figure: {match} (ratio {difflib.SequenceMatcher(None, code, FIG_CODE).ratio():.3f})")
    if not match:
        print("\n".join(difflib.unified_diff(FIG_CODE.splitlines(), code.splitlines(), "figure", "generated", lineterm="")))
    if "</code>" in gen:
        gen = gen.split("</code>", 1)[0] + "</code>"

    rows = apply_goalscope(prompt + gen, mt=mt, selected_heads=heads, strategy="additive_norm", alpha=0.1,
                           steer_layer_range=(35, 46), n_steer_tokens="all", max_new_tokens=64,
                           include_noncode=True)

    # locate each marked read-out: the indentation token right before the line's first character
    text, starts = "", []
    for r in rows:
        starts.append(len(text)); text += r["token"]
    marked = []
    for num, line, occ, ref in FIG_READOUTS:
        idx = -1
        for _ in range(occ + 1):
            idx = text.find(line, idx + 1)
        if idx < 0:
            print(f"({num}) line not found: {line!r}")
            continue
        j = max(i for i, s in enumerate(starts) if s < idx)    # the token holding the line's first char
        j -= rows[j]["token"].strip() != "" and text[starts[j]:idx].strip() == ""   # " man" -> its indentation
        r = rows[j]
        marked.append(dict(callout=num, line=line, pos=r["pos"], token=r["token"], readout=r["patched_gen"], figure=ref))
        print(f"\n({num}) before {line!r} · token {r['token']!r} (pos {r['pos']})\n  figure: {ref}\n  ours:   {r['patched_gen']!r}")

    meta = dict(prompt=prompt, generation=gen, heads=heads_fp, n_heads=len(heads), strategy="additive_norm", alpha=0.1,
                steer_layers=[35, 46], max_new_tokens=64, note="Fig 32 prompt (stable marriage), prefill up to the signature")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, generation_matches_figure=match, marked=marked, rows=rows),
                              ensure_ascii=False, indent=1, default=str))
    print(f"\nsaved {out}")
    if args.write_web:
        from src.goals.site_data import write_data
        print("web data:", write_data(model, "ds_and_algo", "stable_marriage", rows, meta=meta))


if __name__ == "__main__":
    main()
