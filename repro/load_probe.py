"""Load a model through the repo's ModelandTokenizer and time it on a paper prompt.

    CUDA_VISIBLE_DEVICES=<ids> .venv/bin/python repro/load_probe.py --model <hf id> [--out <json>]

Loads bf16 / eager / device_map="auto" (the scripts' load), prints what nnterp serves, the
per-GPU memory after load, and times on the first GoalScope corpus prompt (int_python_000,
eval_goalscope.build_prompt): one traced forward (warm, then the median of 5), a batch-16
traced forward, and a 64-token greedy generate (the frontier rollout's length). Writes the
numbers to --out. Read-only with respect to results/.
"""
import argparse
import json
import statistics
import sys
import time

sys.path.append(".")
import nnterp  # noqa: F401,E402  (nnterp before transformers.models.*)
import torch  # noqa: E402

from scripts.evaluation import eval_goalscope as EG  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402


def gpu_mem():
    return {i: round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def gpu_peak():
    return {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def timed(fn, n=5):
    fn()
    sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec = {"model": args.model}

    t = time.perf_counter()
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    rec["load_s"] = round(time.perf_counter() - t, 1)
    rec["module_class"] = type(mt._module).__name__
    rec["family"] = mt.family_name
    rec["sizes"] = dict(num_layers=mt.num_layers, num_heads=mt.num_heads, num_kv_heads=mt.num_kv_heads,
                        head_dim=mt.head_dim, hidden_size=mt.hidden_size)
    try:
        rec["config_num_attention_heads"] = mt.model.config.num_attention_heads
    except Exception as e:  # eval_goalscope.py reads this raw config field
        rec["config_num_attention_heads"] = f"ERROR {type(e).__name__}: {e}"
    rec["dtype"] = str(mt._module.dtype)
    rec["has_vision"] = any("vision" in n for n, _ in mt._module.named_modules())
    rec["n_params_B"] = round(sum(p.numel() for p in mt._module.parameters()) / 1e9, 2)
    rec["devices"] = sorted({str(p.device) for p in mt._module.parameters()})
    rec["mem_after_load_GiB"] = gpu_mem()
    rec["eos"] = [mt.tokenizer.eos_token, mt.tokenizer.eos_token_id]
    try:
        sup = mt.support()
        rec["support"] = str(sup)[:3000]
    except Exception as e:
        rec["support"] = f"ERROR {type(e).__name__}: {e}"
    print(json.dumps({k: v for k, v in rec.items() if k != "support"}, indent=1), flush=True)

    dps, _ = EG.load_locked_datapoints(EG.DEFAULT_DATASET,
                                       ["int"])
    sid, inst = dps[0]
    prompt, var = EG.build_prompt(mt, inst)
    rec["spec"] = sid
    rec["prompt_tail"] = prompt[-300:]
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    rec["prompt_tokens"] = int(inp["input_ids"].shape[1])
    L = mt.num_layers

    def fwd(batch):
        def f():
            with torch.no_grad(), mt.trace(batch):
                out = mt.layers[L // 2].layer_output.save()
            return out
        return f

    torch.cuda.reset_peak_memory_stats()
    rec["fwd1_s"] = round(timed(fwd(prompt)), 4)
    rec["fwd16_s"] = round(timed(fwd([prompt] * 16), n=3), 4)
    t = time.perf_counter()
    with torch.no_grad():
        out = EG.generate_completion(mt, prompt, 64)
    sync()
    rec["greedy64_nocache_s"] = round(time.perf_counter() - t, 2)
    rec["greedy64_text"] = out if isinstance(out, str) else str(out)[:400]
    t = time.perf_counter()
    with torch.no_grad(), mt.generate(prompt, max_new_tokens=64, do_sample=False) as tr:
        g = tr.result.save()
    sync()
    rec["generate64_cached_s"] = round(time.perf_counter() - t, 2)
    rec["generate64_text"] = mt.tokenizer.decode(g[0][rec["prompt_tokens"]:])
    rec["peak_GiB"] = gpu_peak()
    print(json.dumps(rec, indent=1, default=str), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=str)


if __name__ == "__main__":
    main()
