# GoalScope figure drafts

Open [the comparison page](index.html) or [the four-page PDF](all_options.pdf).

- A: concrete code-to-readout pipeline; recommended general methods figure.
- B: source/probe model stacks and extraction/injection distinction.
- C: independent snapshots as source computation progresses.
- D: shared destination, Q/KV interventions, aligned GoalScope readouts.

All verbalizations are illustrative. These are design drafts, not empirical result figures. Each final source token supplies one summed vector across selected heads/layers. The norm-relative additive strategy injects its direction repeatedly into a separate probe run. Replacement GoalScope is not depicted. Probe question is abbreviated; actual default wording: “What are you doing at this line of the code. Keep your answer as brief as possible.” The system prompt is “You are a helpful assistant proficient in programming.” DIFF defaults: alpha 0.1, layers 35–45, all readout steps. Source extraction layers need not match probe injection layers.

SVG text remains editable. PDF embeds fonts. Regenerate with `python agent_reports/chatgpt/goalscope_figure_options/generate.py` from the repository root. Requires matplotlib.
