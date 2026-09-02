"""``src.diff`` — tooling for the DIFF hypothesis (TASK − PROGRESS at the pivot).

Submodules
----------
functional : interventions, generations, prompt builders, q/k/v transplant machinery,
             readouts, and debugging/visualization utilities. Also hosts ``setup()`` /
             :class:`~src.diff.functional.Ctx` (model + trained head sets).
data       : data generation for Spec-1 (query-transfer) and Spec-2 (kv-transfer).
eval       : evaluation utilities for the q-transfer and kv-transfer interventions.

Import the submodule you need explicitly, e.g. ``import src.diff.functional as P``.
Nothing is re-exported at the package level on purpose — ``functional`` binds a large
set of module globals via ``setup()``, so flattening them into the package namespace
would be confusing.
"""
