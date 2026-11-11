"""``src.nrgs`` — tooling for the NRGS interventions / DIFF hypothesis (TASK − PROGRESS at the pivot).

Submodules
----------
functional       : interventions, generations, prompt builders, q/k/v transplant machinery and
                   readouts. Also hosts ``setup()`` / :class:`~src.nrgs.functional.Ctx` (the model).
data             : data generation (predicate banks, instances, prompt materialization).
hier_data        : the multi-hierarchy (filter-condition) instances.
eval             : the oracle judges.
kv_transfer      : the pivot-onwards KV transfer (geometry, runs, records).
task_vs_progress : which span (task / progress) the pivot reads the goal from.

Results: ``results/{kv_transfer,q_transfer,task_vs_progress}/<model>/...`` (results layout of 2026-11-11).

Import the submodule you need explicitly, e.g. ``import src.nrgs.functional as FUNC``.
"""
