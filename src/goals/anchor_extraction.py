"""Oracle-based extraction of P+ / P- anchor tokens from green MCTS completions.

Typical notebook usage (one sample, one prefill):

    from src.goals.anchor_extraction import get_green_completions, extract_anchor_tokens

    # Build completions dict for each sample (prefix → list of green completions)
    # One (root, prefix) pair per MCTS run; a sample can have multiple runs.
    completions_per_sample = []
    for sample_idx, sample in enumerate(tp.samples):
        sample_comps: dict[str, list[str]] = {}
        for prefix, root in roots_per_sample[sample_idx].items():
            comps = get_green_completions(root, prefix)
            sample_comps[prefix] = list(set(c.strip() for c in comps))
        completions_per_sample.append(sample_comps)

    p_plus_anchors, p_minus_anchors_per_sample = extract_anchor_tokens(
        tp, completions_per_sample, use_cache=True,
    )

    # Inspect, then set manually:
    print(p_plus_anchors)
    print(p_minus_anchors_per_sample)

    tp.p_plus.anchor_tokens = p_plus_anchors
    for i, sample in enumerate(tp.samples):
        if sample.p_minus:
            sample.p_minus.anchor_tokens = p_minus_anchors_per_sample[i]

    tp.save()
"""

import json
import logging
import re

from src.goals.predicate_data import MultiPredicateSample, TargetPredicate
from src.utils.oracle_llms import ask_claude

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_ANCHOR_PROMPT = """\
You are labeling branches in a code-generation search tree. A language model is completing a Python function stub that must implement two independent boolean predicates:

- **P+** (target predicate): {p_plus_description}
- **P-** (companion predicate): {p_minus_description}

Function stub:
```python
{code_stub}
```

One valid green completion (passes all unit tests):
```
return {completion}
```

**Task**: Identify *anchor tokens* — short Python code substrings that appear in the completion and unambiguously signal which predicate is being implemented at the branching point (right after `return`).

- **P+ anchor tokens**: the earliest, shortest substring(s) that uniquely indicate the model is implementing P+. Must NOT normally appear in a P--only implementation.
- **P- anchor tokens**: same for P-. Must NOT normally appear in a P+-only implementation.

Rules:
- Focus on substrings near the *start* of each predicate's code in the completion.
- Prefer short, distinctive substrings (2–8 characters typical, e.g. `len(`, `s ==`, `[::-1]`).
- Include 1–3 anchors per predicate — only the most distinctive ones.

Respond with ONLY a JSON object, no markdown fences, no explanation:
{{"p_plus_anchor_tokens": ["...", "..."], "p_minus_anchor_tokens": ["...", "..."]}}"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_prompt(
    sample: MultiPredicateSample,
    tp: TargetPredicate,
    completion: str,
) -> str:
    p_minus_desc = sample.p_minus.description if sample.p_minus else "(none)"
    return _ANCHOR_PROMPT.format(
        p_plus_description=tp.p_plus.description,
        p_minus_description=p_minus_desc,
        code_stub=sample.code.rstrip(),
        completion=completion.strip(),
    )


def _parse_response(response: str) -> tuple[list[str], list[str]]:
    """Extract anchor token lists from oracle response.

    Handles both raw JSON and JSON wrapped in markdown code fences.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in oracle response:\n{response}")
    data = json.loads(match.group(0))
    p_plus = data.get("p_plus_anchor_tokens", [])
    p_minus = data.get("p_minus_anchor_tokens", [])
    if not isinstance(p_plus, list) or not isinstance(p_minus, list):
        raise ValueError(f"Expected lists in oracle response, got: {data}")
    return [str(t) for t in p_plus], [str(t) for t in p_minus]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_green_completions(root, prefix: str) -> list[str]:
    """Return the completion text (after prefix) for every green leaf node.

    Args:
        root: Root MCTSNode after evaluate_mcts_tree has been called.
        prefix: The prompt string that was passed as the MCTS root context
                (typically instruction + code stub up to `return `).

    Returns:
        List of completion strings (text after prefix), one per green leaf.
    """
    results: list[str] = []

    def _walk(node) -> None:
        if node.is_leaf() and node.color == "green":
            full_text = prefix + node.get_path_text()
            results.append(full_text[len(prefix) :])
        for child in node.children:
            _walk(child)

    _walk(root)
    return results


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_anchor_tokens(
    tp: TargetPredicate,
    completions_per_sample: list[dict[str, list[str]]],
    use_cache: bool = True,
    temperature: float = 0.0,
) -> tuple[list[str], list[list[str]]]:
    """Ask oracle to identify P+ / P- anchor tokens from green completions.

    For each green completion, prompts the oracle (Claude) to identify the short
    code substrings that uniquely signal whether the model is implementing P+ or
    P- at the branching point of the generation tree.

    This function is **read-only** with respect to `tp` — it never modifies
    anchor_tokens.  The caller decides what to do with the returned lists.

    Args:
        tp: TargetPredicate (used for descriptions; not modified).
        completions_per_sample: completions_per_sample[i] is a dict mapping
            prefix → list of unique green completions for sample i.  Each
            (root, prefix) pair is a separate MCTS run; a sample may have
            multiple runs if you ran MCTS under several prefills.
            Build each entry using get_green_completions(root, prefix).
        use_cache: Whether to read/write cached oracle responses (default True).
        temperature: Sampling temperature for oracle (0 = deterministic JSON).

    Returns:
        (p_plus_anchors, p_minus_anchors_per_sample) — both as sorted lists.
        - p_plus_anchors: aggregated P+ anchors across all samples and prefixes.
        - p_minus_anchors_per_sample[i]: P- anchors for sample i.
    """
    if len(completions_per_sample) != len(tp.samples):
        raise ValueError(
            f"completions_per_sample has {len(completions_per_sample)} entries "
            f"but tp '{tp.name}' has {len(tp.samples)} samples."
        )

    all_p_plus: set[str] = set()
    p_minus_per_sample: list[set[str]] = [set() for _ in tp.samples]

    for sample_idx, (sample, prefix_to_comps) in enumerate(
        zip(tp.samples, completions_per_sample)
    ):
        if sample.p_minus is None:
            logger.warning(
                f"[{tp.name}] sample {sample_idx} has no p_minus — skipping."
            )
            continue

        for prefix, completions in prefix_to_comps.items():
            for completion in completions:
                if not completion.strip():
                    continue
                prompt = _build_prompt(sample, tp, completion)
                try:
                    response = ask_claude(
                        prompt=prompt,
                        max_tokens=256,
                        temperature=temperature,
                        use_cache=use_cache,
                    )
                    p_plus_new, p_minus_new = _parse_response(response)
                    all_p_plus.update(p_plus_new)
                    p_minus_per_sample[sample_idx].update(p_minus_new)
                    logger.debug(
                        f"[{tp.name}] sample={sample_idx} prefix={prefix[:30]!r}: "
                        f"p+={p_plus_new}  p-={p_minus_new}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{tp.name}] sample={sample_idx} prefix={prefix[:30]!r}: "
                        f"oracle call failed — {e}"
                    )

    p_plus_anchors = sorted(all_p_plus)
    p_minus_anchors = [sorted(s) for s in p_minus_per_sample]

    logger.info(f"[{tp.name}] p+ anchors: {p_plus_anchors}")
    for i, anchors in enumerate(p_minus_anchors):
        if anchors:
            logger.info(f"[{tp.name}] sample {i} p- anchors: {anchors}")

    return p_plus_anchors, p_minus_anchors
