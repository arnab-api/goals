# Appendix details

Added the exact serialized source prompt and clean continuation for int_python_000; data tables by datatype, task length, and completed-work count; complete frontier-ID lists; exact faithfulness and causality system rubrics, user templates, and instantiated saved oracle messages; deterministic scoring exclusions; and five-model replication tables plus a per-pivot figure.

Data: data/goalscope_eval.json (253 specifications, 57 datatype-specific frontiers). Cross-model evaluation: 10 replacement runs, 1,265 records each, identical specification/pivot key sets. Oracle rubric MD5-prefix identifiers matched the runner constants for every selected run. Rates are recomputed from hard == YES, retaining all records in the denominator. Input hashes are recorded in manifest.json; per-model counts/rates are in model_rates.csv and distribution details in distribution.json.

Exact text payloads live in Overleaf-draft/Goals-Overleaf/Text/Appendix/prompts. The LaTeX listings soft-wrap for presentation; text files preserve raw prompt content and end-of-prefix whitespace. Source prompt/records were copied from the main 122-head additive_norm alpha=.1 run. No model or oracle requests were made.

Also corrected the main/GoalScope-appendix description of the judging target: it is the extracted actual frontier implementation, not the task's natural-language instruction. Replaced the previous short duplicate model table with the expanded Generalization appendix.

Validation: compiled with latexmk; no unresolved citations/references, duplicate labels, or overfull boxes. Visually inspected exact-prompt and system-rubric pages and the model plot. Existing unrelated appendix completion notes remain.
