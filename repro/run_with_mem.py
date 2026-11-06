"""Run a repo module as __main__ and report, at exit, the wall clock and each visible GPU's peak allocation.

    CUDA_VISIBLE_DEVICES=<ids> .venv/bin/python repro/run_with_mem.py <module> [args...]

e.g. ``repro/run_with_mem.py scripts.train_head_mask --tag scratch_smoke --n-epochs 1``. The module runs
unchanged (``runpy.run_module(..., run_name="__main__")``); the report is one ``REPRO_MEM {...}`` json line on
stderr, printed also when the module raises (an OOM included)."""
import atexit
import json
import runpy
import sys
import time

sys.path.insert(0, ".")
T0 = time.time()


def report():
    try:
        import torch

        peak = {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
        reserved = {i: round(torch.cuda.max_memory_reserved(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
    except Exception as e:  # torch not imported / no CUDA
        peak = reserved = repr(e)
    print("REPRO_MEM " + json.dumps({"wall_s": round(time.time() - T0, 1), "peak_alloc_GiB": peak,
                                     "peak_reserved_GiB": reserved}), file=sys.stderr, flush=True)


atexit.register(report)
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
