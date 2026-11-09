"""Causality evals with the v2 C++ probe prompt, written to a SEPARATE results root.

    python scripts/evaluation/run_causal_v2.py <eval module> <its args...>
    e.g. python scripts/evaluation/run_causal_v2.py scripts.evaluation.eval_goalscope --mode causality --heads goal

v2 probe (vs the paper's ``bool check(auto value){`` stub):
    user:      "You are implementing a C++ function. Just write the function body, without any comments or
                explanations. Assume all the necessary headers are already included."
    assistant: "auto func (auto value){"
The trivial-readout sanitizer also strips the new stub name ``func`` (as it strips ``check``). Everything else
(heads, strategy, alpha, layers, judge rubric) is the target script's own. Results go to
``<repo>/results_causal_v2/`` under the usual leaf names, so the paper's ``results/`` is never touched.
"""
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())

import src.goals.goalscope as GS  # noqa: E402
from src.utils import env_utils  # noqa: E402

V2_PREFIX = "auto func (auto value){"
V2_INSTRUCTION = ("You are implementing a C++ function. Just write the function body, without any comments or "
                  "explanations. Assume all the necessary headers are already included.")
V2_RESULTS_DIR = os.path.join(env_utils.PROJECT_ROOT, "results_causal_v2")

module_name, args = sys.argv[1], sys.argv[2:]
assert "--mode" in args and args[args.index("--mode") + 1] == "causality", "v2 only changes the causality probe"

GS.CAUSAL_PREFIX, GS.CAUSAL_INSTRUCTION = V2_PREFIX, V2_INSTRUCTION
E = importlib.import_module("scripts.evaluation.eval_goalscope")
E.CAUSAL_PREFIX = V2_PREFIX
if "func" not in E._TRIVIAL_READOUT_TOKENS:
    E._TRIVIAL_READOUT_TOKENS = tuple(E._TRIVIAL_READOUT_TOKENS) + ("func",)
M = importlib.import_module(module_name)
if hasattr(M, "CAUSAL_PREFIX"):
    M.CAUSAL_PREFIX = V2_PREFIX              # the scripts import the name at module load

assert os.path.realpath(V2_RESULTS_DIR) != os.path.realpath(env_utils.DEFAULT_RESULTS_DIR)
env_utils.DEFAULT_RESULTS_DIR = V2_RESULTS_DIR
print(f"[causal v2] probe stub {V2_PREFIX!r}; results -> {V2_RESULTS_DIR}", flush=True)

sys.argv = [module_name] + args
M.main()
