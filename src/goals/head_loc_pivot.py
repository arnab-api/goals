"""Pivot prefix mining for head localization dataset.

Step 4 of the head localization pipeline: finds positions where dest and target
diverge at the first generated token, producing PivotInstance objects.

**Online-per-model design**: pivot instances are always mined fresh for the LM
being investigated. Do not cache results across models — the token-level divergence
criterion is model-specific. Call `find_pivot_instances(mt, ds)` for each new LM.

Two cases:
  Case 1 — bare return: divergence at `return ` with no prefix (pivot_prefix="").
  Case 2 — shared prefix: divergence at `return <shared_prefix>` where the shared
            prefix is the token-level LCP of passing generations from dest and target.

Selection criterion: ``dest_top_token != target_top_token`` at the pivot position.
This is stored as-is in PivotInstance (no KL, no float threshold) so that the
divergence evidence is interpretable and model-independent in format.

Docstring control: pass ``include_docstring=True`` to insert the sample's
``docstring`` field as a Python docstring between the def line and ``return``.
This gives the model a semantic hint and changes which token it predicts — useful
for studying how the docstring interacts with cross-predicate head patching.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter

import torch
import torch.nn.functional as F

from src.goals.eval import evaluate_mcts_tree
from src.goals.functional import apply_chat_template, supports_chat_template
from src.goals.head_loc_data import (
    HeadLocDataset,
    HeadLocMCTSResult,
    PivotInstance,
    PredicatePair,
)
from src.goals.sampling import MCTSSearch
from src.tokens import prepare_input
from src.utils.env_utils import DEFAULT_RESULTS_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

GENERIC_INSTRUCTION = "Complete the Python function below."
CPP_INSTRUCTION = "Complete the C++ function below."


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_cpp_prompt(
    mt,
    sample,
    include_docstring: bool = False,
) -> str:
    """Build the source C++ prompt ending at ``return ``.

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample whose ``cpp_incomplete_code`` ends with
            ``return`` in one-liner brace form (e.g. ``bool isPalindrome(string s) { return``).
        include_docstring: When True and ``sample.docstring`` is non-empty, appends
            the docstring as a C++ inline comment before ``return``.

    Returns:
        Full prompt string ready for tokenization.
    """
    code_at_pivot = sample.cpp_incomplete_code + " "
    if include_docstring and sample.docstring:
        code_at_pivot = code_at_pivot.rstrip() + f"  // {sample.docstring}\n    return "
    if supports_chat_template(mt):
        return apply_chat_template(mt, CPP_INSTRUCTION, code_at_pivot)
    return "<code>\n" + code_at_pivot


def build_python_prompt(
    mt,
    sample,
    pivot_prefix: str = "",
    include_docstring: bool = False,
) -> str:
    """Build the full prompt for a predicate sample ending at ``return <pivot_prefix>``.

    Args:
        mt: ModelandTokenizer instance.
        sample: SinglePredicateSample whose ``python_incomplete_code`` ends with
            ``return`` (no trailing space).
        pivot_prefix: Token string appended after ``return ``.  Empty string for
            the bare-return case (Case 1).
        include_docstring: When True and ``sample.docstring`` is non-empty, insert
            a Python docstring between the def line and ``return``.  This changes
            the model's prediction at the pivot — use consistently across dest and
            target within a single experiment.

    Returns:
        Full prompt string ready for tokenization.
    """
    code_at_pivot = sample.python_incomplete_code + " " + pivot_prefix

    if include_docstring and sample.docstring:
        lines = code_at_pivot.split("\n")
        def_idx = next(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("def ")), None
        )
        if def_idx is not None:
            indent = "    "
            lines.insert(def_idx + 1, f'{indent}"""{sample.docstring}"""')
            code_at_pivot = "\n".join(lines)

    if supports_chat_template(mt):
        return apply_chat_template(mt, GENERIC_INSTRUCTION, code_at_pivot)
    return "<code>\n" + code_at_pivot


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


@torch.inference_mode()
def get_next_token_probs(mt, prompt: str) -> torch.Tensor:
    """Return the softmax probability distribution over the full vocabulary for the
    next token after ``prompt``.

    Args:
        mt: ModelandTokenizer instance.
        prompt: Full prompt string ending at the pivot position.

    Returns:
        Float32 CPU tensor of shape ``(vocab_size,)`` with probabilities summing to 1.
    """
    inputs = prepare_input(prompts=[prompt], tokenizer=mt)
    with mt.trace(inputs, scan=False, validate=False):
        logits = mt.lm_head.output[0, -1, :].save()
    return torch.softmax(logits.float(), dim=-1).cpu()


def get_top_token(mt, probs: torch.Tensor) -> str:
    """Decode the argmax token from a probability distribution.

    Args:
        mt: ModelandTokenizer instance (for its tokenizer).
        probs: Vocabulary probability tensor, shape ``(vocab_size,)``.

    Returns:
        Decoded string for the highest-probability token.
    """
    token_id = int(probs.argmax().item())
    return mt.tokenizer.decode([token_id])


def compute_kl(p_dest: torch.Tensor, p_target: torch.Tensor) -> float:
    """Compute KL(P_dest || P_target) as a supplementary divergence measure.

    Note: KL is **not** stored in PivotInstance — it is used internally within
    Case 2 to rank candidate pivot prefixes.  The primary selection criterion for
    PivotInstance is token-level divergence (dest_top_token != target_top_token).

    Args:
        p_dest: Probability distribution for destination at pivot, shape ``(vocab_size,)``.
        p_target: Probability distribution for target at pivot, shape ``(vocab_size,)``.

    Returns:
        KL divergence as a scalar float.  Larger = stronger training signal.
    """
    return F.kl_div(torch.log(p_target + 1e-10), p_dest, reduction="sum").item()


# ---------------------------------------------------------------------------
# Case 1 — bare return
# ---------------------------------------------------------------------------


def case1_pivot_instances(
    mt,
    ds: HeadLocDataset,
    pairs: list[PredicatePair] | None = None,
    include_docstring: bool = False,
) -> list[tuple[PredicatePair, PivotInstance]]:
    """Mine Case 1 pivot instances: divergence at bare ``return `` (pivot_prefix="").

    For each pair, runs two forward passes to get the top predicted token for dest
    and target at the bare-return position.  Only pairs where the top tokens differ
    (``dest_top_token != target_top_token``) are returned as PivotInstances.

    This is an **online** operation — always computed fresh for the given ``mt``.
    No cached values from ``PredicatePair`` (e.g. ``kl_at_bare_return``) are used.

    Args:
        mt: ModelandTokenizer to probe.
        ds: HeadLocDataset providing samples and pairs.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        include_docstring: Passed to ``build_python_prompt``; inserts sample docstrings
            when True and the sample has a non-empty ``docstring`` field.

    Returns:
        List of (pair, PivotInstance) tuples where dest and target predict different top tokens.
    """
    target_pairs = pairs if pairs is not None else ds.pairs
    instances: list[tuple[PredicatePair, PivotInstance]] = []

    for pair in target_pairs:
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            logger.warning(
                f"Missing sample for pair "
                f"{pair.source_predicate} -> {pair.destination_predicate}"
            )
            continue

        try:
            dest_prompt = build_python_prompt(
                mt, dest_sample, pivot_prefix="", include_docstring=include_docstring
            )
            targ_prompt = build_python_prompt(
                mt, targ_sample, pivot_prefix="", include_docstring=include_docstring
            )
            p_dest = get_next_token_probs(mt, dest_prompt)
            p_targ = get_next_token_probs(mt, targ_prompt)
        except Exception as e:
            logger.error(
                f"Forward pass failed for pair "
                f"{pair.source_predicate}->{pair.destination_predicate}: {e}"
            )
            continue

        dest_tok = get_top_token(mt, p_dest)
        targ_tok = get_top_token(mt, p_targ)

        logger.debug(
            f"Case 1 | {pair.destination_predicate} top={dest_tok!r}"
            f" | {pair.source_predicate} top={targ_tok!r}"
            f" | diverge={dest_tok != targ_tok}"
        )

        if dest_tok != targ_tok:
            instances.append((
                pair,
                PivotInstance(pivot_prefix="", dest_top_token=dest_tok, target_top_token=targ_tok),
            ))

    logger.info(
        f"Case 1: {len(instances)} diverging pivot instances out of {len(target_pairs)} pairs"
    )
    return instances


# ---------------------------------------------------------------------------
# Case 2 helpers
# ---------------------------------------------------------------------------


def _extract_effective_completions(
    root,
    code_stop_tag: str = "</code>",
) -> list[str]:
    """Walk a post-evaluated MCTS tree and collect unique stripped green-leaf texts.

    For each green leaf:
    1. Get path_text via ``leaf.get_path_text()``.
    2. Strip stop tag: if ``code_stop_tag`` is in path_text, split on it and rstrip;
       otherwise split on the first newline and rstrip.
    3. Deduplicate and sort.

    Args:
        root: Root MCTSNode of a post-evaluated tree (colors already assigned).
        code_stop_tag: Stop tag to strip from generated text.

    Returns:
        Sorted list of unique non-empty effective completion strings.
    """
    leaves: list = []

    def _walk(node) -> None:
        if node.is_leaf():
            if node.color == "green":
                leaves.append(node)
            return
        for child in node.children:
            _walk(child)

    _walk(root)

    effective: list[str] = []
    for leaf in leaves:
        path_text = leaf.get_path_text()
        if code_stop_tag and code_stop_tag in path_text:
            path_text = path_text.split(code_stop_tag)[0].rstrip()
        else:
            path_text = path_text.split("\n")[0].rstrip()
        if path_text:
            effective.append(path_text)

    return sorted(set(effective))


def run_mcts_for_sample(
    mt,
    sample,
    include_docstring: bool = False,
    n_iterations: int = 200,
    max_depth: int = 20,
    top_k: int = 5,
    temperature: float = 1.0,
    c_puct: float = 1.0,
    min_children_up_to_depth: int = 3,
    min_children: int = 2,
    save_tree: bool = True,
    save_dir: PathLike | None = None,
) -> HeadLocMCTSResult:
    """Run MCTS for one SinglePredicateSample and return a HeadLocMCTSResult.

    Args:
        mt: ModelandTokenizer to run MCTS with.
        sample: SinglePredicateSample to explore.
        include_docstring: Passed to ``build_python_prompt``.
        n_iterations: Number of MCTS iterations.
        max_depth: Maximum tree depth.
        top_k: Top-k branching factor per expansion.
        temperature: Sampling temperature.
        c_puct: PUCT constant for MCTS.
        min_children_up_to_depth: Minimum number of children to expand up to a certain depth.
        min_children: Minimum number of children to expand.
        save_tree: If True, serialize the tree dict into the result; otherwise None.
        save_dir: If provided, also write ``<predicate_name>_tree.html`` and
            ``<predicate_name>_tree.json`` into this directory.

    Returns:
        HeadLocMCTSResult with green_completions populated.
    """
    prompt = build_python_prompt(
        mt, sample, pivot_prefix="", include_docstring=include_docstring
    )
    prefix = sample.python_incomplete_code + " "

    search = MCTSSearch(
        mt,
        prompt,
        max_depth=max_depth,
        top_k=top_k,
        temperature=temperature,
        c_puct=c_puct,
        min_children_up_to_depth=min_children_up_to_depth,
        min_children=min_children,
    )
    search.run(n_iterations=n_iterations, depth_first=True)

    mcts_config = {
        "n_iterations": n_iterations,
        "max_depth": max_depth,
        "top_k": top_k,
        "temperature": temperature,
        "c_puct": c_puct,
        "min_children_up_to_depth": min_children_up_to_depth,
        "min_children": min_children,
    }

    eval_counts = evaluate_mcts_tree(
        root=search.root,
        prefix=prefix,
        unit_tests=sample.unit_tests,
        function_name=sample.function_name or sample.predicate_name,
    )

    green_completions = _extract_effective_completions(search.root)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        html_path = os.path.join(save_dir, f"{sample.predicate_name}_tree.html")
        search.save_tree(html_path, show_colors=True)
        json_path = os.path.join(save_dir, f"{sample.predicate_name}_tree.json")
        with open(json_path, "w") as f:
            json.dump(search.root.to_dict(), f, indent=2)
        logger.info(f"  Saved tree → {html_path}, {json_path}")

    return HeadLocMCTSResult(
        predicate_name=sample.predicate_name,
        model_name=mt.name,
        prompt=prompt,
        prefix=prefix,
        mcts_config=mcts_config,
        mcts_stats=search.get_stats(),
        eval_counts=eval_counts,
        green_completions=green_completions,
        tree=search.root.to_dict() if save_tree else None,
    )


def run_mcts_for_dataset(
    mt,
    ds: HeadLocDataset,
    save_dir: PathLike | None = None,
    overwrite: bool = False,
    include_docstring: bool = False,
    n_iterations: int = 256,
    max_depth: int = 32,
    top_k: int = 5,
    temperature: float = 1.0,
    c_puct: float = 1.0,
    min_children_up_to_depth: int = 3,
    min_children: int = 2,
    save_tree: bool = True,
) -> dict[str, HeadLocMCTSResult]:
    """Run (or load cached) MCTS for all samples in a HeadLocDataset.

    Results are saved to ``results/localization/<model_short_name>/`` by default
    (computed from ``mt.name``).  Pass ``save_dir`` to override.  For each sample,
    also saves ``<predicate_name>_tree.html`` and ``<predicate_name>_tree.json``
    alongside the main JSON.

    Loads a cached result from disk unless ``overwrite=True``. Returns a dict
    keyed by predicate_name.

    Args:
        mt: ModelandTokenizer to run MCTS with.
        ds: HeadLocDataset whose samples are processed.
        save_dir: Directory for saving/loading results.  Defaults to
            ``{DEFAULT_RESULTS_DIR}/localization/{mt.name.split('/')[-1]}``.
        overwrite: If True, re-run MCTS even if a cached result exists.
        include_docstring: Passed to ``run_mcts_for_sample``.
        n_iterations: MCTS iterations per sample.
        max_depth: MCTS max depth.
        top_k: MCTS top-k branching.
        temperature: MCTS sampling temperature.
        save_tree: Whether to serialize the full tree dict into the result JSON.

    Returns:
        Dict mapping predicate_name → HeadLocMCTSResult for all samples.
    """
    if save_dir is None:
        model_short = mt.name.split("/")[-1]
        save_dir = os.path.join(
            DEFAULT_RESULTS_DIR, "localization", "mcts", model_short
        )

    logger.info(f"run_mcts_for_dataset: save_dir={save_dir}")
    results: dict[str, HeadLocMCTSResult] = {}

    for sample in ds.samples:
        cached = (
            None
            if overwrite
            else HeadLocDataset.load_mcts_result(sample.predicate_name, save_dir)
        )
        if cached is not None:
            logger.info(
                f"  {sample.predicate_name}: loaded from cache "
                f"(green={cached.eval_counts.get('green', 0)}"
                f"/{cached.eval_counts.get('total', 0)})"
            )
            results[sample.predicate_name] = cached
            continue

        logger.info(f"  {sample.predicate_name}: running MCTS ({n_iterations} iters)…")
        result = run_mcts_for_sample(
            mt,
            sample,
            include_docstring=include_docstring,
            n_iterations=n_iterations,
            max_depth=max_depth,
            top_k=top_k,
            temperature=temperature,
            save_tree=save_tree,
            save_dir=save_dir,
            c_puct=c_puct,
            min_children_up_to_depth=min_children_up_to_depth,
            min_children=min_children,
        )
        ds.save_mcts_result(result, save_dir)
        logger.info(
            f"  {sample.predicate_name}: done — "
            f"green={result.eval_counts.get('green', 0)}"
            f"/{result.eval_counts.get('total', 0)}, "
            f"unique_completions={len(result.green_completions)}"
        )
        results[sample.predicate_name] = result

    return results


def _lcp_tokens(mt, gen_a: str, gen_b: str) -> str:
    """Compute the token-level longest common prefix (LCP) of two generated strings.

    Tokenizes both strings independently, finds the longest prefix where all token
    ids match, then decodes back to a string.

    Args:
        mt: ModelandTokenizer instance (for its tokenizer).
        gen_a: First generated completion string.
        gen_b: Second generated completion string.

    Returns:
        The decoded LCP string, or ``""`` if the first tokens differ.
    """
    tok_a = mt.tokenizer.encode(gen_a, add_special_tokens=False)
    tok_b = mt.tokenizer.encode(gen_b, add_special_tokens=False)

    common_len = 0
    for ta, tb in zip(tok_a, tok_b):
        if ta != tb:
            break
        common_len += 1

    if common_len == 0:
        return ""

    return mt.tokenizer.decode(tok_a[:common_len])


def _find_stable_lcps(
    mt,
    dest_gens: list[str],
    targ_gens: list[str],
    min_stable_k: int,
) -> list[str]:
    """Find pairwise LCPs across (dest_gen, targ_gen) that appear at least
    ``min_stable_k`` times.

    A "stable" LCP is one that appears consistently across many (dest, target) generation
    pairs, indicating a genuine shared syntactic trajectory rather than a coincidental
    overlap.  Empty LCPs are excluded — they carry no pivot signal.

    Args:
        mt: ModelandTokenizer instance.
        dest_gens: List of passing completions for the destination predicate.
        targ_gens: List of passing completions for the target predicate.
        min_stable_k: Minimum number of (dest, target) pairs that must share an LCP
            for it to be considered stable.

    Returns:
        List of stable LCP strings, sorted by frequency descending.
    """
    lcp_counts: Counter[str] = Counter()

    for dg in dest_gens:
        for tg in targ_gens:
            lcp = _lcp_tokens(mt, dg, tg)
            if lcp:
                lcp_counts[lcp] += 1

    return [
        lcp
        for lcp, cnt in sorted(lcp_counts.items(), key=lambda x: -x[1])
        if cnt >= min_stable_k
    ]


# ---------------------------------------------------------------------------
# Case 2 — shared prefix
# ---------------------------------------------------------------------------


def case2_pivot_instances(
    mt,
    ds: HeadLocDataset,
    mcts_results: dict[str, HeadLocMCTSResult],
    pairs: list[PredicatePair] | None = None,
    min_stable_k: int = 1,
    include_docstring: bool = False,
) -> list[tuple[PredicatePair, PivotInstance]]:
    """Mine Case 2 pivot instances: divergence at ``return <shared_prefix>``.

    For each pair, reads pre-computed green_completions from ``mcts_results``,
    computes pairwise token-level LCPs, and keeps those appearing at least
    ``min_stable_k`` times.  For each stable LCP, runs a forward pass to check
    whether the top predicted tokens diverge at ``return <lcp>``.

    Selection criterion: ``dest_top_token != target_top_token`` at the pivot.

    This is an **online** operation — always computed fresh for the given ``mt``.
    The MCTS generation step is decoupled: run ``run_mcts_for_dataset`` first.

    Args:
        mt: ModelandTokenizer to probe.
        ds: HeadLocDataset providing samples and pairs.
        mcts_results: Pre-computed MCTS results keyed by predicate_name.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        min_stable_k: Minimum frequency for an LCP to be considered stable.
        include_docstring: Passed to ``build_python_prompt``; inserts sample docstrings
            when True and the sample has a non-empty ``docstring`` field.

    Returns:
        List of (pair, PivotInstance) tuples where dest and target predict different top tokens
        at a stable shared prefix.
    """
    target_pairs = pairs if pairs is not None else ds.pairs
    instances: list[tuple[PredicatePair, PivotInstance]] = []

    for pair in target_pairs:
        dest_sample = ds.get_sample(pair.destination_predicate)
        targ_sample = ds.get_sample(pair.source_predicate)
        if dest_sample is None or targ_sample is None:
            logger.warning(
                f"Missing sample for pair "
                f"{pair.source_predicate} -> {pair.destination_predicate}"
            )
            continue

        dest_result = mcts_results.get(pair.destination_predicate)
        targ_result = mcts_results.get(pair.source_predicate)

        if dest_result is None or targ_result is None:
            logger.warning(
                f"Missing MCTS result for pair "
                f"{pair.source_predicate} -> {pair.destination_predicate} — skipping"
            )
            continue

        dest_passing = dest_result.green_completions
        targ_passing = targ_result.green_completions

        logger.info(
            f"Case 2: pair {pair.destination_predicate} -> {pair.source_predicate}"
            f" | dest_green={len(dest_passing)} targ_green={len(targ_passing)}"
        )

        if not dest_passing or not targ_passing:
            logger.warning(
                f"  No passing completions — skipping pair "
                f"{pair.destination_predicate} -> {pair.source_predicate}"
            )
            continue

        stable_lcps = _find_stable_lcps(mt, dest_passing, targ_passing, min_stable_k)
        logger.info(f"  Found {len(stable_lcps)} stable LCPs: {stable_lcps}")

        for lcp in stable_lcps:
            dest_pivot_prompt = build_python_prompt(
                mt, dest_sample, pivot_prefix=lcp, include_docstring=include_docstring
            )
            targ_pivot_prompt = build_python_prompt(
                mt, targ_sample, pivot_prefix=lcp, include_docstring=include_docstring
            )

            try:
                p_dest = get_next_token_probs(mt, dest_pivot_prompt)
                p_targ = get_next_token_probs(mt, targ_pivot_prompt)
            except Exception as e:
                logger.error(f"  Forward pass failed for prefix {lcp!r}: {e}")
                continue

            dest_tok = get_top_token(mt, p_dest)
            targ_tok = get_top_token(mt, p_targ)

            logger.debug(
                f"  prefix={lcp!r} | dest_top={dest_tok!r} | targ_top={targ_tok!r}"
                f" | diverge={dest_tok != targ_tok}"
            )

            if dest_tok != targ_tok:
                instances.append((
                    pair,
                    PivotInstance(pivot_prefix=lcp, dest_top_token=dest_tok, target_top_token=targ_tok),
                ))

    logger.info(f"Case 2: {len(instances)} diverging pivot instances")
    return instances


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def find_pivot_instances(
    mt,
    ds: HeadLocDataset,
    mcts_results: dict[str, HeadLocMCTSResult] | None = None,
    pairs: list[PredicatePair] | None = None,
    include_case1: bool = True,
    include_case2: bool = True,
    include_case3: bool = False,
    include_case3b: bool = False,
    include_docstring: bool = False,
    # Case 2 params
    min_stable_k: int = 3,
    # Case 3 params
    n_oracle: int = 5,
    oracle_temperature: float = 0.5,
    max_completion_tokens: int = 32,
    # Case 3b params
    n_implementations: int = 8,
    min_prefix_len: int = 3,
) -> list[PivotInstance]:
    """Mine all pivot instances for the given model, combining Case 1, 2, and/or 3.

    This is the main entry point for Step 4.  Call this once per LM; results are
    model-specific and should not be reused across different models.

    Deduplicates by ``(destination_predicate, target_predicate, pivot_prefix)`` —
    the Case 1 bare-return instance is kept over any Case 2/3 duplicate if both are
    found (Case 1 comes first in the combined list).

    Args:
        mt: ModelandTokenizer to mine pivot instances for.
        ds: HeadLocDataset providing samples and pairs.
        mcts_results: Pre-computed MCTS results (required when ``include_case2=True``).
            Run ``run_mcts_for_dataset`` first to populate this.
        pairs: Subset of pairs to evaluate.  Defaults to ``ds.pairs``.
        include_case1: Whether to run Case 1 (bare-return divergence).
        include_case2: Whether to run Case 2 (MCTS shared-prefix divergence).
        include_case3: Whether to run Case 3 (oracle LLM shared-prefix divergence).
        include_case3b: Whether to run Case 3b (per-predicate oracle + mechanical LCP).
        include_docstring: Insert sample docstrings into prompts when True.
        min_stable_k: (Case 2) Minimum LCP frequency to be considered stable.
        n_oracle: (Case 3) Number of oracle implementations requested per predicate.
        oracle_temperature: (Case 3/3b) Oracle LLM sampling temperature.
        max_completion_tokens: (Case 3/3b) Max tokens for greedy generation in the completion gate.
        n_implementations: (Case 3b) Number of oracle implementations per predicate.
        min_prefix_len: (Case 3b) Minimum character length for a candidate LCP.

    Returns:
        Deduplicated list of (pair, PivotInstance) tuples where ``tokens_diverge`` is True.

    Raises:
        ValueError: If ``include_case2=True`` and ``mcts_results`` is None.
    """
    if include_case2 and mcts_results is None:
        raise ValueError(
            "mcts_results is required when include_case2=True. "
            "Run run_mcts_for_dataset() first."
        )

    all_instances: list[tuple[PredicatePair, PivotInstance]] = []

    if include_case1:
        all_instances.extend(
            case1_pivot_instances(
                mt, ds, pairs=pairs, include_docstring=include_docstring
            )
        )

    if include_case2:
        all_instances.extend(
            case2_pivot_instances(
                mt,
                ds,
                mcts_results=mcts_results,
                pairs=pairs,
                min_stable_k=min_stable_k,
                include_docstring=include_docstring,
            )
        )

    if include_case3:
        from src.goals.head_loc_oracle import case3_pivot_instances
        all_instances.extend(
            case3_pivot_instances(
                mt,
                ds,
                pairs=pairs,
                include_docstring=include_docstring,
                n_oracle=n_oracle,
                oracle_temperature=oracle_temperature,
                max_completion_tokens=max_completion_tokens,
            )
        )

    if include_case3b:
        from src.goals.head_loc_oracle import case3b_pivot_instances
        all_instances.extend(
            case3b_pivot_instances(
                mt,
                ds,
                pairs=pairs,
                include_docstring=include_docstring,
                n_implementations=n_implementations,
                oracle_temperature=oracle_temperature,
                min_prefix_len=min_prefix_len,
                max_completion_tokens=max_completion_tokens,
            )
        )

    # Deduplicate by (destination_predicate, source_predicate, pivot_prefix)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[tuple[PredicatePair, PivotInstance]] = []
    for pair, pi in all_instances:
        key = (pair.destination_predicate, pair.source_predicate, pi.pivot_prefix)
        if key not in seen:
            seen.add(key)
            deduped.append((pair, pi))

    logger.info(
        f"find_pivot_instances: {len(all_instances)} total -> {len(deduped)} after dedup"
    )
    return deduped
