# Closed-loop query transfer — regenerating the goal heads' query at every step

*Qwen2.5-Coder-32B-Instruct · 2026-10-05 → 2026-10-07 · notebook `notebooks/diff/04.4_q_transfer_multipred.ipynb` ·
batch `scripts/diff/diff_trigger_eval.py` (`--closed-loop`, `--probe-prompt`, `--kv-transfer`) · status: work in progress*

## 1. Motivation

The q-transfer result in the paper uses a **re-clamp**: the SRC pivot query of the goal heads is written into DST's
pivot, and then the *same* query is written again into the current token for the first `N_STEER − 1` decode steps.
If the goal heads encode the goal at several levels of granularity (NRGS), re-using one query at every generated token
does not make sense — the query should change token by token ("given this goal and what I have written, what do I
look for next?"), while the goal itself persists.

The **closed loop** keeps the goal fixed and *regenerates* the query at every step from it, then asks whether DST
still pursues the transferred sub-goal.

## 2. Setup

### Tasks (diff-trigger q-transfer; 04.3/04.4 notation, `'` = written)

| design | prompts | DST writes (clean) | prediction of the q-transfer |
|---|---|---|---|
| no kv (order-diff) | `p_src A'CB' → C` · `p_dst A'BD → B` | B | **D** (set progress and count agree) |
| no kv (order-matched) | `p_src A'B'C → C` · `p_dst A'BD → B` | B | D (also = "the 3rd-listed" positional copy) |
| kv (04.3 / 04.4) | `p_src A'CB' → C` · `p_dst A'XY → X` · `p_kv A'BD → B` | X | kv-only → **B** · kv+q: set progress → **D**, count → **Y** |

C (SRC's own target) is in neither p_dst nor p_kv. With kv, p_kv's task-list key+value on the goal heads replace
those on p_dst's task-list span (GQA-corrected head-wise routing, `pad` alignment: newline tokens spliced before the
shorter span), so the "which task is pending" computation happens inside the goal heads.

### Common configuration (= the paper's order-diff q-transfer arm)

Goal-250 heads (patch + read-out) · OV boost α = 2 on every patched forward · pivot = the frontier's ` =` · sequential
letter naming (`__seq`) · py→py · 32 instances per datatype × 4 datatypes. Judges: `judge_joint` (exclusive
B / C / D / neither) without kv; one exclusive verdict over {X, Y, B, C, D} with kv. Both grade only the FIRST
computation.

## 3. The interventions

**Re-clamp (baseline).** SRC pivot query → DST pivot (prefill), then re-clamped at the current token for 15 decode
steps (`N_STEER=16`).

**Closed loop (all variants).** Same prefill patch (→ t₁). Then at every step k: an auxiliary **probe** prompt is fed
the patched run's token t_k; its goal-head query at t_k (`q_caus`) replaces the patched run's goal-head query at t_k
(every other head keeps its own; kv + boost stay on); → t_{k+1}. 64 tokens, no statement cut-off. Variants differ in
the probe and in whether the goal encoding is injected:

| variant | probe | what conditions the probe |
|---|---|---|
| `stub` | information-less: DST's signature + `return` (or the bare indent) | the **goal encoding** (goal heads' OV write at the patched pivot), injected `additive_norm` α 0.1 at the probe pivot and every appended token |
| `gaze_select` | stub whose user turn names ONE task: the one the patched pivot gazes at most | text only (no injection) |
| `select-stub` | stub naming the task chosen by a k-way prompt ("Select one of the tasks below…", `I select (`), label logits steered with the goal encoding and averaged over all orderings | text only; the goal encoding is used only for the selection |
| `select-as-probe` | that k-way prompt itself, prefilled `I select (k)` + `<code>` + the stub | text only (list + decision); goal encoding only for the selection |

### Notebook-only variants that did not work (qualitative, 04.4)

All with `PROBE_PROMPT="stub"` unless stated; none sustained the transfer in single instances:

* goal encoding from a **restricted** run (goal heads attend only to the task span, ± the anchor tokens);
* goal encoding as **patched − base** (base = the same run without the SRC query) — none of patched / base / contrast
  read out as D under the paper read-out config;
* longer **steering windows** on the probe (`PROBE_STEER_START`), earlier steer layers (20–40);
* probe = stub **+ the task list** (`tasklist`), and the **self-probe** (the DST prompt itself, kv-patched, no q).

Diagnostic that motivated the selection variants: at the patched pivot the goal heads' gaze shifts to D, but **after
the first closed-loop token it returns to the native B**; under the re-clamp the gaze stays on D at every step
(notebook re-clamp-gaze cell). The regenerated query does not keep pointing at D.

## 4. Results

### 4.1 Order-diff (complete)

**No kv** — joint D / native B; *(sel)* = how often the probe's chosen task was D

| | int | str | list[int] | list[str] |
|---|---|---|---|---|
| re-clamp | **0.62** / 0.34 | **0.41** / 0.50 | **0.47** / 0.38 | **0.38** / 0.50 |
| closed loop · stub | 0.12 / 0.88 | 0.12 / 0.88 | 0.31 / 0.69 | 0.06 / 0.88 |
| closed loop · gaze_select | 0.25 / 0.72 (0.66) | 0.25 / 0.72 (0.62) | 0.41 / 0.50 (0.78) | 0.12 / 0.50 (0.66) |
| closed loop · select-stub | 0.16 / 0.78 (0.50) | 0.22 / 0.69 (0.25) | 0.41 / 0.53 (0.66) | 0.09 / 0.56 (0.47) |
| closed loop · select-as-probe | 0.22 / 0.69 (0.50) | 0.22 / 0.66 (0.25) | 0.41 / 0.53 (0.66) | 0.25 / 0.56 (0.47) |

**kv** — kv+q D (set) / Y (count) / B (kv-only); clean DST → X 0.97–1.00, kv-only → B 0.81–1.00 in every arm

| | int | str | list[int] | list[str] |
|---|---|---|---|---|
| kv re-clamp | **0.56** / 0 / 0.34 | **0.34** / 0.03 / 0.47 | **0.43** / 0 / 0.47 | **0.50** / 0 / 0.44 |
| kv closed loop · stub | 0.19 / 0 / 0.81 | 0.03 / 0 / 0.91 | 0.20 / 0 / 0.80 | 0.03 / 0 / 0.91 |
| kv closed loop · gaze_select | 0.31 / 0 / 0.66 (0.78) | 0.06 / 0.03 / 0.69 (0.75) | 0.43 / 0.03 / 0.50 (0.77) | 0.19 / 0 / 0.41 (0.69) |
| kv closed loop · select-stub | 0.22 / 0 / 0.66 (0.47) | 0.03 / 0.03 / 0.72 (0.16) | 0.47 / 0.03 / 0.50 (0.67) | 0.06 / 0 / 0.53 (0.41) |
| kv closed loop · select-as-probe | 0.25 / 0 / 0.66 (0.47) | 0.09 / 0.03 / 0.66 (0.16) | 0.40 / 0 / 0.50 (0.67) | 0.25 / 0 / 0.50 (0.41) |

(kv list[int]: n = 30 — two instances had no valid (X, Y).)

**Selection vs execution** — P(writes D | the probe's task is D) · P(writes D | another task), pooled over datatypes:

| | no kv | kv |
|---|---|---|
| gaze_select | 0.37 [87] · 0.02 [41] | 0.32 [94] · 0.03 [32] |
| select-stub | 0.43 [60] · 0.03 [68] | 0.40 [53] · 0.04 [73] |
| select-as-probe | **0.53** [60] · 0.04 [68] | **0.49** [53] · 0.07 [73] |

### 4.2 Order-matched (in progress, 2026-10-07 12:52)

No kv — joint D / B (sel): re-clamp int 0.66/0.31 · str 0.53/0.34 · list[int] 0.56/0.31 · list[str] 0.69/0.28;
closed loop · gaze_select int 0.47/0.44 (0.94) · str 0.47/0.47 (0.84) · list[int] 0.44/0.38 (0.91) · list[str]
0.28/0.38 (0.91). kv · gaze_select: int D 0.44 / Y 0 / B 0.50 (0.91) · str 0.44 / 0 / 0.53 (0.91). The select-stub /
select-as-probe arms are still queued (ETA ≈ 19:00–19:30). There is no kv re-clamp order-matched baseline yet.

### 4.3 Execution audit of the judged successes (order-diff)

The judges are lenient. Every record judged "D" was executed: its first generated expression was run on ~70–80
inputs per datatype and compared with the bank's reference implementation of D. *Plausible* = ≥ 60 % agreement
(allows edge-case conventions, e.g. `n % 10` vs `abs(n) % 10`); *degenerate* = < 60 %, crash, syntax error, or a
reference to a written variable (`a`, `b`).

| arm | judged D | degenerate among them | execution-plausible D (all instances) |
|---|---|---|---|
| re-clamp | 60 / 128 | **25 %** | **0.35** |
| closed loop · stub | 20 | 30 % | 0.11 |
| closed loop · gaze_select | 33 | 18 % | 0.21 |
| closed loop · select-stub | 28 | 21 % | 0.17 |
| closed loop · select-as-probe | 35 | **6 %** | 0.26 |
| kv re-clamp | 58 / 126 | **36 %** | **0.29** |
| kv closed loop · stub | 14 | 43 % | 0.06 |
| kv closed loop · gaze_select | 31 | 29 % | 0.17 |
| kv closed loop · select-stub | 24 | 21 % | 0.15 |
| kv closed loop · select-as-probe | 31 | 23 % | 0.19 |

Typical degenerate re-clamp "successes" mention D's ingredients without computing D: self-contradictory chains
(`len(nums) != len(set(nums)) - len(nums) != len(set(nums))` for has_duplicate), syntax errors
(`sum(sum(1 for c in w for c in w)`), the negation of the task (`n % 2 == 0` for is_odd), or D mixed with a written
variable (`… == len(s) and a > 0`).

## 5. Findings

1. **Regenerating the query from the goal encoding does not sustain the transfer.** The `stub` closed loop falls back
   to the native task (B, or B/kv-only with kv) in 69–91 % of instances; D drops from 0.38–0.62 (re-clamp) to
   0.06–0.31. Re-applying the SRC query at decode steps is what keeps DST on D.
2. **Text-conditioned probes recover part of it.** Naming the right sub-goal in the probe (`gaze_select`,
   `select-*`) roughly doubles D over `stub`, but stays below the re-clamp except on list[int].
3. **Two bottlenecks: selection and execution.** The patched pivot's gaze picks D 62–78 % of the time (order-diff;
   84–94 % order-matched); the goal-encoding k-way selection only 16–67 % (weakest on `str`). And even with D
   selected, the loop writes D in only 32–53 % of instances. `select-as-probe` executes best (0.53 / 0.49), plausibly
   because its probe sees the whole list and the decision.
4. **kv + q favours set progress over count** in every arm: Y (count) ≤ 0.03 everywhere, D up to 0.56.
5. **The re-clamp's numbers are inflated by degenerate code** (25 % / 36 % of its judged successes); the selection
   loops produce cleaner code (`select-as-probe` 6 % without kv). The `stub` loop is *not* cleaner (30 % / 43 %) —
   it succeeds less and its few successes are as degenerate as the re-clamp's, so regenerating the query alone does
   not fix the degeneracy. Measured by execution, the gap shrinks (no kv: re-clamp 0.35 vs select-as-probe 0.26;
   kv: 0.29 vs 0.19) but the re-clamp still leads.

## 6. Caveats

* **Judges** grade only the first computation and are not anchored on the current variable; they are lenient on
  degenerate code (§4.3). Not yet fixed — see the deferred judge issues (memory note).
* **Execution audit** thresholds (60 %), input pools and the "first line = the expression" extraction are my choices;
  some "wrong" verdicts may be reasonable variants (a different edge-case convention, a boolean vs a count).
* **Selection modes are text-conditioned:** `gaze_select` uses the gaze, not the goal encoding; the `select-*` modes
  use the goal encoding only to choose the task. None of the successful variants injects the goal encoding into the
  probe — which is the open TODO ("query-redirected goal encoding").
* n = 32 per cell; Coder-32B; py→py; one α / steering configuration.

## 7. Next steps

1. Finish order-matched (all arms) and add a **kv re-clamp order-matched** baseline.
2. An **execution-backed judge** as a separate metric, re-scored offline on every saved run (re-clamp and closed loop).
3. Combine **gaze's selection rate with `select-as-probe`'s execution** (a select-as-probe whose choice comes from the
   gaze), and close the TODO: make the goal encoding itself drive the probe (e.g. via the k-way selection with a
   stronger / per-layer encoding).

## Files

* Notebook: `notebooks/diff/04.4_q_transfer_multipred.ipynb` (closed loop, probe modes, selection cells (a)/(b),
  re-clamp gaze, knobs for kv / boost / restriction / sufficiency ablation)
* Library: `src/diff/functional.py` — `closed_loop_generate`, `closed_loop_select_scores`, `closed_loop_select_prompt`,
  `run_q_transfer_intervention(closed_loop=…, kv=…)`
* Batch: `scripts/diff/diff_trigger_eval.py` (`--closed-loop`, `--probe-prompt`, `--kv-transfer`); job lists
  `job_list/diff_eval/diff_trigger_orderdiff_{closed_loop,kv}.json`, `diff_trigger_closed_loop_probe_modes.json`
* Results: `results/diff/Qwen2.5-Coder-32B-Instruct/diff_trigger/ph-goal_250__a2__piv-eq{,__kv}{__cl,__cl-ppgaze,
  __cl-ppselstub,__cl-ppselprobe}__seq/` (re-clamp baselines: `…__piv-eq__seq/` order-diff, `…__piv-eq/` order-matched,
  `…__piv-eq__kv__seq/`)
