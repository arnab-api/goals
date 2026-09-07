# Frozen result tables
Snapshot: 2026-09-07T17:34:32.665668+00:00. Counts are records, not distinct independent experiments.

| Mechanism/setup | Run | Pivot | N | Causal (SRC or BOTH) | Patched readout = SRC | Patched readout = SRC or BOTH | Valid | Readout shift given valid |
|---|---|---|---:|---:|---:|---:|---:|---:|
| kv_transfer/dual_comp | `ph-goal_250__ro-goal_250__abl-off__a2__pad` | - | 299 | 96.66% | 93.98% | 94.65% | 93.98% | 95.37% |
| kv_transfer/dual_comp | `ph-goal__abl-33-43-restrict__a2__pad` | - | 448 | 73.88% | 52.90% | 54.91% | 94.42% | 55.79% |
| kv_transfer/dual_comp | `ph-goal__abl-33-43-zero__a2__pad` | - | 448 | 72.54% | 52.68% | 54.91% | 95.09% | 55.63% |
| kv_transfer/dual_comp | `ph-goal__abl-35-50__a1__pad` | - | 2 | 100.00% | 50.00% | 50.00% | 100.00% | 50.00% |
| kv_transfer/multi_pred | `ph-goal_250__ro-goal_250__abl-33-43-restrict__a2__pad` | var_eq | 512 | 98.83% | 96.09% | 96.09% | 96.68% | 96.77% |
| kv_transfer/multi_pred | `ph-goal_250__ro-goal_250__abl-off__a2__pad` | var_eq | 512 | 99.22% | 96.09% | 96.09% | 96.68% | 96.77% |
| kv_transfer/multi_pred | `ph-goal__abl-33-43-restrict__a2__pad` | indent | 256 | 58.59% | 44.14% | 49.22% | 87.89% | 51.11% |
| kv_transfer/multi_pred | `ph-goal__abl-33-43-restrict__a2__pad` | var_eq | 256 | 83.59% | 50.78% | 54.30% | 96.09% | 54.47% |
| kv_transfer/multi_pred | `ph-goal__abl-33-43-zero__a2__pad` | indent | 256 | 54.69% | 43.75% | 48.83% | 87.89% | 51.11% |
| kv_transfer/multi_pred | `ph-goal__abl-33-43-zero__a2__pad` | var_eq | 256 | 71.88% | 50.00% | 53.91% | 96.88% | 54.03% |
| q_transfer/filter | `ph-goal_250__ro-goal_250__a2` | - | 128 | 60.94% | 74.22% | 92.97% | 85.94% | 95.45% |
| q_transfer/filter | `ph-goal__a1` | - | 768 | 41.41% | 76.43% | 88.15% | 70.18% | 95.92% |
| q_transfer/filter | `ph-goal__a2` | - | 720 | 71.53% | 77.50% | 87.36% | 76.94% | 96.03% |
| q_transfer/multi_pred | `ph-goal__a1` | - | 1024 | 83.20% | 88.77% | 90.43% | 81.25% | 94.47% |
| q_transfer/multi_pred | `ph-goal__a2` | - | 1024 | 93.55% | 88.57% | 90.14% | 83.20% | 94.25% |
