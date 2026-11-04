"""Time an evaluation script with the Claude judge stubbed: run it unchanged, with every ``ask_claude`` call
answered from the on-disk judge cache when the exact (system, prompt, max_tokens) is there, else by a stub
that never reaches the API, so the model side can be timed on a host without a CLAUDE_KEY.

    CUDA_VISIBLE_DEVICES=<ids> .venv/bin/python repro/run_stub_judge.py <module | path.py> [args...]

The stub answers ``<verdict>NO</verdict>`` (parsers read it as a miss), or, for the temporal-persistence
segmentation prompt, the first line of the model continuation it quotes, so the condition is still encoded.
Verdicts from stubbed calls are meaningless: run this only into a scratch RESULTS_DIR (the repro/scratch
worktree). With ``REPRO_JUDGE_STRICT=1`` a call that is not in the cache raises instead (cache-only runs whose
oracle calls are all cached, e.g. pivot mining, can then write real outputs). At exit one ``REPRO_STUB {...}`` json line on stderr: wall clock, judge cache hits / stubbed calls,
the seconds spent in them, and each visible GPU's peak allocation."""
import atexit
import hashlib
import json
import os
import re
import runpy
import sys
import time

sys.path.insert(0, ".")
import src.utils.oracle_llms as OL  # noqa: E402  (no torch import: --cuda pre-parsers still run first)

T0 = time.time()
STATS = {"hits": 0, "stubbed": 0, "judge_s": 0.0}


def _stub(prompt):
    m = re.search(r"MODEL CONTINUATION \(generated after the `=`\):\n```\n(.*?)\n```", prompt, re.S)
    if m:
        first = next((ln for ln in m.group(1).splitlines() if ln.strip()), "")
        return first.strip() or "0"
    return "<verdict>NO</verdict>"


def ask_claude(prompt, max_tokens=2048, temperature=0.6, use_cache=True, model_name="claude-sonnet-4-6",
               system_prompt="You are a helpful assistant."):
    t = time.time()
    h = hashlib.md5(f"{system_prompt}__{prompt}__{max_tokens=}".encode()).hexdigest()
    path = os.path.join(OL.CLAUDE_CACHE_DIR, f"{h}.json")
    try:
        with open(path) as f:
            out = json.load(f)["response"]
        STATS["hits"] += 1
    except (OSError, ValueError, KeyError):
        if os.environ.get("REPRO_JUDGE_STRICT") == "1":
            raise RuntimeError(f"judge call not in the cache ({h}) and REPRO_JUDGE_STRICT=1")
        out = _stub(prompt)
        STATS["stubbed"] += 1
    STATS["judge_s"] += time.time() - t
    return out


OL.ask_claude = ask_claude


def report():
    rec = {"wall_s": round(time.time() - T0, 1), **{k: round(v, 2) for k, v in STATS.items()}}
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            rec["peak_alloc_GiB"] = {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                                     for i in range(torch.cuda.device_count())}
    print("REPRO_STUB " + json.dumps(rec), file=sys.stderr, flush=True)


atexit.register(report)
target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
if target.endswith(".py"):
    runpy.run_path(target, run_name="__main__")
else:
    runpy.run_module(target, run_name="__main__", alter_sys=True)
