# Alternative hypotheses to SET_DIFF — test setups

*Qwen2.5-Coder-32B-Instruct · 2026-10-08 · source notes `logs/alt_hypothesis.md` · status: designs validated, not
yet implemented*

## Primary hypothesis and the hint against it

**SET_DIFF**: the goal heads' query carries SRC's progress (its written set W_s), and DST continues with
`L_dst − W_s`.

| | p_src | p_dst | no-op | SET_DIFF |
|---|---|---|---|---|
| order-diff | `a'cb' → c` | `a'bd → b` | b | d |
| order-matched | `a'b'c → c` | `a'bd → b` | b | d |

Order-matched gives higher intervention accuracy (re-clamp and gsr alike; gsr py→py B share 0.34 → 0.11), so the
query carries more than progress. Of the alternatives below, **only ORDER-ID changes its prediction between the two
variants**:

| hypothesis | order-diff | order-matched |
|---|---|---|
| ORDER-ID | c is 2nd → **b (= no-op)** | c is 3rd → **d** |
| COUNT | 2 done → 3rd = d | d |
| INDUCTION | just did b → after b = d | d |

COUNT and INDUCTION predict d in both variants, so they cannot explain the gap. The working guess is a mix of
SET_DIFF and position.

## Conventions

- `'` = already written. The task without `'` is the one SRC is about to write, its target.
- **ORDER-ID:** the task at SRC's target position.
- **COUNT:** L_dst[|W_s| + 1].
- **INDUCTION:** the task after SRC's last written task.
- **SRC's own target never appears in p_dst.** The earlier `c_in_dst` variant showed it gets selected most of the
  time, a literal copy.
- **Each setup tests one alternative against SET_DIFF.** Overlaps with the other alternatives are allowed; each is
  resolved by its own setup.

## Setups

### ALT1 — ORDER-ID: "I am about to do the n-th listed task"

p_src `a'b'cd' → c` · p_dst `a'bde → b`

| no-op | SET_DIFF | ORDER-ID |
|---|---|---|
| b | e = {a,b,d,e} − {a,b,d} | d (c is SRC's 3rd task) |

COUNT also predicts e; ALT2 separates it.

### ALT2 — COUNT: "n tasks done, so the (n+1)-th is next"

p_src `a'b'cd' → c` (same SRC as ALT1) · p_dst `a'def → d`

| no-op | SET_DIFF | COUNT |
|---|---|---|
| d | e | f (3 done → 4th) |

- **SET_DIFF here is {e, f}.** e assumes ties go to the first remaining task in list order.
- **e is overdetermined.** ORDER-ID, INDUCTION and "skip the native task" also predict it.
- So **f is clean evidence for COUNT; e is not clean evidence for SET_DIFF.**
- *Optional variant:* p_dst `a'bed → b` → no-op b, SET_DIFF {e} exactly, COUNT d. COUNT then lands on a task SRC has
  already written, which SET_DIFF rules out.

### ALT3 — INDUCTION: "what follows the task I just did"

p_src `a'b'c'd → d` · p_dst `a'cbe → c`

| no-op | SET_DIFF | INDUCTION |
|---|---|---|
| c | e = {a,c,b,e} − {a,b,c} | b (SRC just did c; c → b in DST) |

ORDER-ID and COUNT also predict e (both the 4th task).

## Confounds and controls

1. **The letter naming gives the count away.** The SRC pivot stub's letter is `chr('a' + |W_s|)`, which
   is `d =` in all three setups (3 written). The query at ` =` is read right after it, so a COUNT result could be read off the
   letter. The main order-diff/order-matched pair is unaffected (both `c =`). Control: a `--pivot indent` replicate
   (no stub), or random stub letters.
2. **"Skip the native task" fallback.** Any perturbation that suppresses no-op may land on the next listed task. This
   coincides with ORDER-ID in ALT1 (d) and with SET_DIFF in ALT2 (e). Control: transplant the query from an SRC whose
   tasks are all different from DST's, with the same |W_s| and target position. SET_DIFF then predicts nothing, while
   positional hypotheses and skip-next still predict L_dst[pos].
3. **Judge.** One exclusive verdict per setup over {no-op, SET_DIFF, alternative, C, neither}: a generalisation of
   `hyp_eval._letters_prompt`, as in the kv judge. Grade the first computation, plus the execution audit for Python
   destinations.

## Run plan

- **Arms:** the re-clamp and gsr. gsr's gaze selection directly reads which task the query points the goal heads at
  (P(D | selected D) ≈ 0.98 in the main grid).
- **Defaults:** same as the main grid: goal-250 heads, α = 2, ` =` pivot, `__seq`, 32 instances × 4 datatypes;
  start with py→py.
- **Implementation:**
  - a setup generator with role-labelled tasks (pairwise compatible, distinct families, as `_kv_roles` picks them).
    `DiffTriggerInstance._parts` already takes any per-side order and written set;
  - the generalised judge;
  - a `--variants alt1|alt2|alt3` entry in `scripts/diff/diff_trigger_eval.py`.
- **Order:** ALT1 → ALT2 → ALT3, then the different-tasks SRC control and the `--pivot indent` replicate of ALT2.
