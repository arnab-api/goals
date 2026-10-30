"""Site rates + examples for the ICL tab. Per task, the 20 prompts of ``icl_goalscope.py`` (same seed), each with a
greedy generation of up to 128 tokens cut after 5 rounds (the answer to the final query + 4 model-invented Q/A pairs).
Scored readouts (the site's rate): the "A" and ":" tokens of the last 3 demonstrations, of the query, and of every
generated round. Prompt #0 is the site example and also gets a readout at EVERY generated token. Same settings as ``icl_goalscope.py``.
Writes ``results/icl_goalscope/site/<task>.json`` in the same layout (one prompt), for ``icl_judge.py`` and
``icl_write_site_data.py``.

    python scripts/icl/icl_site_examples.py --tasks antonym country-capital
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import torch  # noqa: E402

from scripts.icl.icl_goalscope import HEADS, MODEL, PROBE, PROBE_SYS, READ_TOKENS, TASKS, build_prompt  # noqa: E402
from src.goals.goalscope import apply_goalscope  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils.typing import TokenizerOutput  # noqa: E402

MAX_GEN, N_ROUNDS = 128, 5


def cut_after_rounds(gen_tokens: list[str], n_rounds: int) -> int:
    """Number of generated tokens up to the end of the ``n_rounds``-th answer line (the first line is the answer to
    the final query; later rounds are "Q: ..." then "A: ..." lines). Keeps the line's newline out."""
    text = "".join(gen_tokens)
    lines, ends, pos = text.split("\n"), [], 0
    for i, line in enumerate(lines):
        pos += len(line)
        if i == 0 or line.lstrip().startswith("A:"):
            ends.append(pos)
        pos += 1
    if len(ends) < n_rounds:
        return len(gen_tokens)
    end, acc = ends[n_rounds - 1], 0
    for k, t in enumerate(gen_tokens):
        if acc >= end:
            return k
        acc += len(t)
    return len(gen_tokens)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--out-dir", default="results/icl_goalscope/site")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-prompts", type=int, default=20)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mt = ModelandTokenizer(model_key=MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    tok = mt.tokenizer
    heads = [tuple(h) for h in json.load(open(HEADS))]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    def probe_prompt(setting):
        return tok.apply_chat_template([{"role": "system", "content": PROBE_SYS}, {"role": "user", "content": PROBE[setting]},
                                        {"role": "assistant", "content": ""}], tokenize=False,
                                       add_generation_prompt=False, continue_final_message=True)

    for task in args.tasks:
        t0 = time.time()
        setting, rules, _ = TASKS[task]
        rng = random.Random(0)                                    # same prompts as icl_goalscope.py
        evals = []
        for k in range(args.n_prompts):
            text, demos = build_prompt(task, rng)
            inp = prepare_input(prompts=text, tokenizer=mt)
            n = inp["input_ids"].shape[1]
            with torch.no_grad():
                g = mt._model.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                       max_new_tokens=MAX_GEN, do_sample=False)[0].tolist()
            gen = [i for i in g[n:]]
            while gen and gen[-1] in eos:
                gen = gen[:-1]
            gen = gen[:cut_after_rounds([tok.decode([i]) for i in gen], N_ROUNDS)]
            ids = g[:n] + gen
            toks = [tok.decode([i]) for i in ids]
            # scored positions: the "A" and ":" of the last 3 demonstrations, of the query, and of every generated round
            a_prompt = [p for p in range(n - 1) if toks[p] == "A" and toks[p + 1] == ":"][-4:]
            a_gen = [p for p in range(n, len(ids) - 1) if toks[p] == "A" and toks[p + 1] == ":"]
            scored = sorted({q for a in a_prompt + a_gen for q in (a, a + 1)})
            # prompt #0 is the site example: also every generated token
            pos = sorted(set(scored) | (set(range(n - 1, len(ids))) if k == 0 else set()))
            src = TokenizerOutput(data=dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)))
            rows = apply_goalscope(src, mt=mt, selected_heads=heads, strategy="additive_norm", alpha=0.1,
                                   steer_layer_range=(35, 46), batch_size=args.batch, max_new_tokens=READ_TOKENS,
                                   n_steer_tokens="all", show=False, goalscope_prompt=probe_prompt(setting),
                                   token_positions=pos)
            by = {int(r["pos"]): (r.get("patched_gen") or "").strip() for r in rows}
            ev = dict(k=k, text=text, demos=demos, ids=ids, n_prompt=n, tokens=toks, generation=tok.decode(gen),
                      n_rounds=len(a_gen) + 1, positions=scored, readouts=[by[p] for p in scored])
            if k == 0:
                ev.update(all_positions=pos, all_readouts=[by[p] for p in pos])
            evals.append(ev)
        json.dump(dict(task=task, setting=setting, rules=rules, probe=PROBE[setting], probe_system=PROBE_SYS,
                       scored="the A and : of the last 3 demonstrations, of the query and of each generated round",
                       settings=dict(model=MODEL, heads=HEADS, n_heads=len(heads), strategy="additive_norm", alpha=0.1,
                                     layers=[35, 45], max_gen=MAX_GEN, n_rounds=N_ROUNDS, readout_tokens=READ_TOKENS,
                                     n_prompts=args.n_prompts),
                       prompts=evals), open(out_dir / f"{task}.json", "w"), ensure_ascii=False, indent=1)
        print(f"{task}: {args.n_prompts} prompts, {sum(len(e['positions']) for e in evals)} scored readouts "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
