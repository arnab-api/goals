# Figure 1: two interventions on one destination

Revisited Figure 1 of `Overleaf-draft/REF/papers/filter.pdf`. The central design choice follows that figure's direct activation-transfer arrows: show a donor state extracted from a particular token and placed into a particular destination site. Here there are two donors but only one displayed destination prompt.

Open `index.html` to compare:

- **F_shared_destination**: Q donor on the left, destination in the middle, KV donor on the right; three aligned result cards below. Recommended for a compact Figure 1.
- **G_converging_donors** (revised): stacked donors on the left, shared destination in the middle. The upper pair is **(a) task/goal transfer → (b) K/V-patched run**; the lower pair is **(c) progress transfer → (d) query-patched run**. Dashed links identify activation transplants, and separate solid links connect the affected destination sites to their corresponding results. The clean baseline sits above the destination. The previous PDF is preserved as `G_converging_donors_v1.pdf`.

Both have editable SVGs, vector PDFs and PNGs. Mathematics uses editable text with subscripts.

## Exact construction

| Context | Task set G | Completed work P | Next target |
|---|---|---|---|
| Destination | {T1, T2} | {T1} | T2 |
| Query donor | {T1, T2} | {T2} | T1 |
| KV donor | {T1, T3} | {T1} | T3 |

T1 is palindrome filtering, T2 is vowel-initial filtering, T3 is filtering for more than one distinct character (`len(set(w)) > 1`). Code assumes nonempty lowercase strings.

Every prompt ends at `if` in `return [w for w in res if`. The Q arrow identifies the corresponding goal-head query states at that token. The KV arrow identifies the two-condition task-description span; shaded text stays visibly unchanged in the destination, while its internal K/V is replaced. This is a schematic span mapping; tokenizer alignment is not shown.

The output cards depict three **separate** runs on the same destination: clean, Q-patched, and KV-patched. There is no jointly patched fourth condition. The head and token indices suppressed in the compact math should be defined in the caption/methods.

The Q intervention repeats palindrome filtering in a destination that has already performed it. This is intentional: it demonstrates importing a different progress state rather than editing the visible prefix. The KV intervention introduces a third predicate absent from the destination instructions. Outputs illustrate the user's proposed experiment; no new model generations were run.

## Draft caption

For the revised G layout, panel references are: (a) task K/V transplant, (b) its T3 continuation, (c) query transplant, (d) its T1 continuation. Use `src_kv` for the KV donor and `src_q` for the query donor. The clean baseline is unlettered.

**Two interventions on a shared destination separate progress from task requirements.** The destination requests palindrome filtering (T1) and vowel-initial filtering (T2), and has already implemented T1. Its clean continuation therefore implements T2. A query donor requests the same two filters but has implemented T2; transferring its goal-head query states at the pivot π redirects the destination toward T1. A task donor requests T1 and filtering for more than one distinct character (T3), and shares the destination's completed T1 computation; transferring its task-span keys and values redirects the destination toward T3. Both interventions act on the same destination prompt, whose text and code prefix remain fixed. G and P denote the task set and completed work. Highlighted tokens/spans identify activation extraction and replacement sites. The continuations are schematic predictions of the DIFF account.

For an empirical final version, replace the illustrative suffixes with recorded continuations and report head sets, intervention duration, span alignment, amplification and any other-head ablation in the caption or methods reference. GoalScope and cross-language transfer are omitted to keep this figure focused.

Regeneration:

```bash
python agent_reports/chatgpt/figure1_shared_destination/build.py
/usr/bin/python3 agent_reports/chatgpt/figure1_shared_destination/render_figures.py
```
