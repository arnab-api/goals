# GoalScope re-evaluation manifest

_Generated 2026-08-17 22:46:34_  ·  34 runs under `results/evaluation/goalscope`.

Every run below is re-graded with the current `eval_goalscope` oracle rubric. `stored`/`current` are the `hard_rubric_tag` md5 prefixes; a run is a no-op when they match and it has already been backed up.

| model | mode | strategy | α | heads | n_heads | n_specs | records | stored→current | done |
|---|---|---|---|---|---|---|---|---|---|
| Olmo-3.1-32B-Instruct | causality | replace | 1.0 | goal | 130 | 512 | 10235 | 2d514490→2d514490 |  |
| Olmo-3.1-32B-Instruct | faithfulness | replace | 1.0 | goal | 130 | 512 | 10235 | 15dba839→15dba839 |  |
| Olmo-3.1-32B-Instruct-SFT | causality | replace | 1.0 | goal | 79 | 512 | 10235 | 2d514490→2d514490 |  |
| Olmo-3.1-32B-Instruct-SFT | faithfulness | replace | 1.0 | goal | 79 | 512 | 10235 | 15dba839→15dba839 |  |
| Qwen2.5-32B-Instruct | causality | replace | 1.0 | goal | 108 | 512 | 10235 | 2d514490→2d514490 |  |
| Qwen2.5-32B-Instruct | faithfulness | replace | 1.0 | goal | 108 | 512 | 10235 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-14B-Instruct | causality | additive_norm | 0.1 | goal | 88 | 256 | 5080 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-14B-Instruct | causality | replace | 1.0 | goal | 88 | 256 | 5080 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-14B-Instruct | faithfulness | additive_norm | 0.1 | goal | 88 | 256 | 5080 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-14B-Instruct | faithfulness | replace | 1.0 | goal | 88 | 256 | 5080 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive | 0.2 | goal | 122 | 512 | 10230 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive | 0.2 | goal | 122 | 8 | 180 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | goal | 122 | 512 | 10230 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam100_top100 | 100 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam100_top50 | 50 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam20_top150 | 150 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam20_top200 | 200 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam20_top250 | 250 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | additive_norm | 0.1 | json:goal__lam5_top300 | 300 | 256 | 5075 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | replace | 1.0 | goal | 122 | 512 | 10230 | 2d514490→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | causality | resid_norm | 0.1 | resid:L25-56:mean | None | 512 | 10230 | None→2d514490 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive | 0.2 | goal | 122 | 256 | 5075 | 2f3a61f8→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive | 0.2 | goal | 122 | 512 | 10230 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive | 0.2 | goal | 122 | 8 | 180 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | goal | 122 | 512 | 10230 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam100_top100 | 100 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam100_top50 | 50 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam20_top150 | 150 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam20_top200 | 200 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam20_top250 | 250 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | additive_norm | 0.1 | json:goal__lam5_top300 | 300 | 256 | 5075 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | replace | 1.0 | goal | 122 | 256 | 5075 | 2f3a61f8→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | replace | 1.0 | goal | 122 | 512 | 10230 | 15dba839→15dba839 |  |
| Qwen2.5-Coder-32B-Instruct | faithfulness | resid_norm | 0.1 | resid:L25-56:mean | None | 512 | 10230 | None→15dba839 |  |

**Total records:** 234,980 across 34 runs (oracle calls are fewer after de-duplicating identical read-outs).
