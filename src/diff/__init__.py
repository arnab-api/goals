"""``src.diff`` — tooling for the DIFF hypothesis (TASK − PROGRESS at the pivot).

Submodules
----------
functional       : interventions, generations, prompt builders, q/k/v transplant machinery and
                   readouts. Also hosts ``setup()`` / :class:`~src.diff.functional.Ctx` (the model).
data             : data generation (predicate banks, instances, prompt materialization).
hier_data        : the multi-hierarchy (filter-condition) instances.
eval             : the oracle judges.
path_patch       : the pivot-onwards kv-transfer (geometry, runs, records).
task_vs_progress : which span (task / progress) the pivot reads the goal from.

Import the submodule you need explicitly, e.g. ``import src.diff.functional as FUNC``.
"""
