"""Intent block labeling for IntentBlock trees.

Two backends, selected by whether ``mt`` is provided:

  mt=None (default)  — calls ``ask_claude`` from ``src.utils.oracle_llms``.
                       Responses are disk-cached automatically by that util.
  mt=ModelandTokenizer — runs the loaded LM.  ``label_all_blocks`` batches
                         prompts for speed using ``batch_generate``.

The oracle is asked to wrap its answer in ``<label>…</label>`` tags so the
response can be parsed unambiguously regardless of any preamble.

Public API
----------
``build_label_prompt(block, full_code)``         → prompt str
``label_intent_block(block, full_code, mt, …)``  → str  (sets block.label)
``label_all_blocks(root, full_code, mt, …)``     → None (in-place)
``save_labeled_tree(root, path, full_code, meta)``
``load_labeled_tree(path)``                      → (root, full_code, metadata)
``tree_to_dict(block)``                          → dict
``dict_to_tree(d, parent)``                      → IntentBlock
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from src.goals.scope_id import IntentBlock
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_MARKER = "# ← LABEL THIS"

_LABEL_SYSTEM_PROMPT = (
    "You are a code annotation assistant. "
    "Respond concisely and wrap your final answer in <label>…</label> tags."
)

_PROMPT_TEMPLATE = """\
You are annotating Python code. Label the marked block with a short phrase.

Rules:
  1. One short phrase, 3–8 words.
  2. Describe the block's own operation — what it does, not why it exists \
relative to the surrounding function or algorithm.
  3. Start with a verb: "initialize X", "check if Y", "return the result".
  4. Do not reference the enclosing function's purpose or the overall algorithm context.
  5. No punctuation at the end.
  6. Do not use variable names from the code. Describe the abstract operation instead: \
e.g. "dequeue the next element" not "pop vertex from queue".

The block to label is marked with "{marker}" in the code below.
It is a {kind} block starting at line {lineno}.

```python
{annotated_code}
```

Wrap your answer in <label>…</label> tags, e.g. <label>dequeue the next element</label>."""


def _annotate_code(block: IntentBlock, full_code: str) -> tuple[str, int]:
    """Return ``(annotated_code, lineno)`` with ``_MARKER`` on the opening line."""
    lines = full_code.splitlines(keepends=True)
    pos = 0
    opening_lineno = 1
    for i, line in enumerate(lines, 1):
        if pos <= block.char_start < pos + len(line):
            opening_lineno = i
            break
        pos += len(line)

    annotated = lines.copy()
    orig = annotated[opening_lineno - 1]
    stripped = orig.rstrip("\n")
    newline = orig[len(stripped) :]
    annotated[opening_lineno - 1] = stripped + "  " + _MARKER + newline

    return "".join(annotated).strip(), opening_lineno


def build_label_prompt(block: IntentBlock, full_code: str) -> str:
    """Return the user-message prompt for labeling ``block``.

    The full code is shown with ``# ← LABEL THIS`` on the opening line of the
    block, unambiguously identifying it even when the same snippet appears
    multiple times.

    Args:
        block:     The ``IntentBlock`` to label.
        full_code: Full Python source string the block belongs to.

    Returns:
        Prompt string (user message content, no system prompt included).
    """
    annotated_code, lineno = _annotate_code(block, full_code)
    return _PROMPT_TEMPLATE.format(
        marker=_MARKER,
        kind=block.kind,
        lineno=lineno,
        annotated_code=annotated_code,
    )


def _parse_label(text: str) -> str:
    """Extract the label from ``<label>…</label>`` tags.

    Falls back to the first non-empty line if no tags are found.
    """
    m = re.search(r"<label>(.*?)</label>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: first non-empty line, strip any echoed "Label:" prefix
    for line in text.splitlines():
        line = line.strip()
        if line:
            if line.lower().startswith("label:"):
                line = line[6:].strip()
            return line
    return text.strip()


# ---------------------------------------------------------------------------
# LM helpers
# ---------------------------------------------------------------------------


def _format_prompt_for_lm(prompt: str, mt) -> str:
    """Wrap ``prompt`` in the model's chat template if supported.

    Uses a simple user→assistant format with ``add_generation_prompt=True``
    so the model produces a fresh assistant turn containing the label.

    Args:
        prompt: User-message content (output of ``build_label_prompt``).
        mt:     ``ModelandTokenizer`` instance.

    Returns:
        Formatted string ready for tokenization.
    """
    from src.goals.functional import supports_chat_template

    if not supports_chat_template(mt):
        return prompt  # base model: pass raw prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return mt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        return mt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def _lm_labels_batch(
    prompts: list[str],
    mt,
    max_new_tokens: int,
    batch_size: int,
    show_progress: bool,
) -> list[str]:
    """Run batched LM generation for a list of prompts, return parsed labels.

    Uses ``batch_generate`` from ``clock_probe`` which handles left-padding
    and HF's ``.generate()``.  The ``<label>`` tags are parsed posthoc so no
    custom stop tokens are needed.

    Args:
        prompts:        Raw user-message prompts (output of ``build_label_prompt``).
        mt:             ``ModelandTokenizer`` instance.
        max_new_tokens: Max tokens to generate per sequence.
        batch_size:     Number of prompts per forward-pass batch.
        show_progress:  Show rich progress bar.

    Returns:
        List of label strings, one per prompt, in the same order.
    """
    from src.goals.clock_probe import batch_generate

    formatted = [_format_prompt_for_lm(p, mt) for p in prompts]
    labels: list[str] = []

    _chunks = [
        formatted[i : i + batch_size] for i in range(0, len(formatted), batch_size)
    ]

    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=False,
            )
            task_id = progress.add_task("labeling (LM)", total=len(prompts))
            with progress:
                for chunk in _chunks:
                    results = batch_generate(mt, chunk, max_new_tokens=max_new_tokens)
                    for _, gen_ids in results:
                        text = mt.tokenizer.decode(gen_ids, skip_special_tokens=True)
                        labels.append(_parse_label(text))
                    progress.advance(task_id, len(chunk))
            return labels
        except ImportError:
            pass

    for chunk in _chunks:
        results = batch_generate(mt, chunk, max_new_tokens=max_new_tokens)
        for _, gen_ids in results:
            text = mt.tokenizer.decode(gen_ids, skip_special_tokens=True)
            labels.append(_parse_label(text))

    return labels


# ---------------------------------------------------------------------------
# Public labeling functions
# ---------------------------------------------------------------------------


def label_intent_block(
    block: IntentBlock,
    full_code: str,
    mt=None,
    max_new_tokens: int = 60,
    claude_model: str = "claude-sonnet-4-6",
) -> str:
    """Label a single ``IntentBlock``, set ``block.label``, and return the label.

    Args:
        block:          Block to label.
        full_code:      Full Python source string.
        mt:             If ``None``, uses ``ask_claude`` (default).
                        If a ``ModelandTokenizer``, uses the loaded LM.
        max_new_tokens: Max tokens to generate (LM path only).
        claude_model:   Claude model name (Claude path only).

    Returns:
        The label string (also stored in ``block.label``).
    """
    prompt = build_label_prompt(block, full_code)

    if mt is None:
        from src.utils.oracle_llms import ask_claude

        response = ask_claude(
            prompt,
            max_tokens=max_new_tokens,
            model_name=claude_model,
            system_prompt=_LABEL_SYSTEM_PROMPT,
        )
        label = _parse_label(response)
    else:
        labels = _lm_labels_batch(
            [prompt],
            mt,
            max_new_tokens=max_new_tokens,
            batch_size=1,
            show_progress=False,
        )
        label = labels[0]

    block.label = label
    return label


def _collect_blocks(
    root: IntentBlock,
    skip_kinds: frozenset[str],
) -> list[IntentBlock]:
    """DFS; collect blocks whose kind is not in ``skip_kinds``."""
    result: list[IntentBlock] = []

    def _dfs(b: IntentBlock) -> None:
        if b.kind not in skip_kinds:
            result.append(b)
        for child in b.children:
            _dfs(child)

    _dfs(root)
    return result


def label_all_blocks(
    root: IntentBlock,
    full_code: str,
    mt=None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
    skip_kinds: Optional[set[str]] = None,
    skip_labeled: bool = True,
    show_progress: bool = True,
    claude_model: str = "claude-sonnet-4-6",
) -> None:
    """Label every block in the tree in-place.

    When ``mt`` is provided, prompts are batched for speed.
    When ``mt`` is ``None``, ``ask_claude`` is called per block (disk-cached).

    Args:
        root:          Root of the ``IntentBlock`` tree.
        full_code:     Full Python source string.
        mt:            ``None`` → Claude oracle; ``ModelandTokenizer`` → LM.
        batch_size:    Prompts per batch (LM path only).
        max_new_tokens: Max tokens to generate per label.
        skip_kinds:    Block kinds to skip.  Defaults to ``{"module"}``.
        skip_labeled:  Skip blocks that already have a label (resume support).
        show_progress: Show a rich progress bar.
        claude_model:  Claude model name (Claude path only).
    """
    _skip = frozenset(skip_kinds if skip_kinds is not None else {"module"})
    blocks = _collect_blocks(root, _skip)
    if skip_labeled:
        blocks = [b for b in blocks if b.label is None]

    if not blocks:
        logger.info("label_all_blocks: nothing to label")
        return

    # ── LM path: batched ─────────────────────────────────────────────────────
    if mt is not None:
        prompts = [build_label_prompt(b, full_code) for b in blocks]
        labels = _lm_labels_batch(
            prompts,
            mt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        for block, label in zip(blocks, labels):
            block.label = label
        return

    # ── Claude path: sequential (cached) ─────────────────────────────────────
    from src.utils.oracle_llms import ask_claude

    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=False,
            )
            task_id = progress.add_task("labeling (Claude)", total=len(blocks))
            with progress:
                for block in blocks:
                    try:
                        prompt = build_label_prompt(block, full_code)
                        response = ask_claude(
                            prompt,
                            max_tokens=max_new_tokens,
                            model_name=claude_model,
                            system_prompt=_LABEL_SYSTEM_PROMPT,
                        )
                        block.label = _parse_label(response)
                    except Exception as exc:
                        logger.warning("labeling failed for %r: %s", block, exc)
                        block.label = "(error)"
                    progress.advance(task_id)
            return
        except ImportError:
            pass

    for block in blocks:
        try:
            prompt = build_label_prompt(block, full_code)
            response = ask_claude(
                prompt,
                max_tokens=max_new_tokens,
                model_name=claude_model,
                system_prompt=_LABEL_SYSTEM_PROMPT,
            )
            block.label = _parse_label(response)
        except Exception as exc:
            logger.warning("labeling failed for %r: %s", block, exc)
            block.label = "(error)"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def tree_to_dict(block: IntentBlock) -> dict:
    """Recursively serialize an ``IntentBlock`` tree to a JSON-compatible dict."""
    return {
        "kind": block.kind,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "label": block.label,
        "children": [tree_to_dict(c) for c in block.children],
    }


def dict_to_tree(
    d: dict,
    parent: Optional[IntentBlock] = None,
) -> IntentBlock:
    """Recursively deserialize an ``IntentBlock`` tree from a dict."""
    block = IntentBlock(
        kind=d["kind"],
        char_start=d["char_start"],
        char_end=d["char_end"],
        label=d.get("label"),
        parent=parent,
    )
    block.children = [dict_to_tree(c, parent=block) for c in d.get("children", [])]
    return block


def save_labeled_tree(
    root: IntentBlock,
    path: PathLike,
    full_code: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Save a labeled ``IntentBlock`` tree to a JSON file.

    Stores ``"metadata"``, ``"full_code"`` (optional), and ``"tree"``.
    """
    data: dict = {
        "metadata": metadata or {},
        "tree": tree_to_dict(root),
    }
    if full_code is not None:
        data["full_code"] = full_code

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved labeled tree → %s", path)


def load_labeled_tree(
    path: PathLike,
) -> tuple[IntentBlock, Optional[str], dict]:
    """Load a labeled ``IntentBlock`` tree from a JSON file.

    Returns:
        ``(root, full_code, metadata)`` — ``full_code`` is ``None`` if not saved.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    root = dict_to_tree(data["tree"])
    full_code: Optional[str] = data.get("full_code")
    metadata: dict = data.get("metadata", {})
    return root, full_code, metadata
