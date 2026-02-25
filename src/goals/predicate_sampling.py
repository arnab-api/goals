"""
MCTS search result containers for the multi-predicate setup.

Hierarchy:
  MCTSRunResult                             one (sample, prefill) MCTS run
  MCTSSearchResults_for_MultiPredicateSample  all runs for one sample
  MCTSSearchResults_for_TargetPredicate       all samples for one TargetPredicate

All three are DataClassJsonMixin so they round-trip through JSON cleanly.
MCTSNode trees are NOT serialized (they hold GPU tensors / circular refs).
They are kept in `_trees: dict[str, MCTSNode]` on the sample-level object
(populated during a live session, lost on save/load).
"""

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin

from src.goals.predicate_data import MultiPredicateSample
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)


@dataclass
class MCTSRunResult(DataClassJsonMixin):
    """Serializable summary of one (sample × prefill) MCTS run.

    Fields:
        prefill:           The prefill string that was appended to sample.code.
        prefill_type:      E.g. "inconsequential", "aligned_p_plus", ...
        prompt:            Full prompt sent to the LM.
        prefix:            sample.code + prefill  (for eval and completion extraction).
        mcts_config:       Hyperparams used for this run.
        mcts_stats:        Tree statistics returned by MCTSSearch.get_stats().
        eval_counts:       {"green": N, "orange": N, "red": N, "total": N}
        green_completions: Unique stripped completion strings from green leaves.
    """

    prefill: str
    prefill_type: str
    prompt: str
    prefix: str
    mcts_config: dict[str, Any]
    mcts_stats: dict[str, Any]
    eval_counts: dict[str, int]
    green_completions: list[str]


@dataclass
class MCTSSearchResults_for_MultiPredicateSample(DataClassJsonMixin):
    """MCTS results for one MultiPredicateSample across all tested prefills.

    `runs` is the serializable list of MCTSRunResult (one per prefill run).
    `_trees` holds live MCTSNode roots keyed by `prefix`; it is populated
    during a session but is NOT persisted — re-run MCTS to repopulate after
    loading from disk.
    """

    sample: MultiPredicateSample
    runs: list[MCTSRunResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        # In-memory only: prefix → MCTSNode root.  Never serialized.
        self._trees: dict[str, Any] = {}

    @property
    def completions_by_prefix(self) -> dict[str, list[str]]:
        """Returns {prefix: green_completions} for use with extract_anchor_tokens."""
        return {run.prefix: run.green_completions for run in self.runs}


@dataclass
class MCTSSearchResults_for_TargetPredicate(DataClassJsonMixin):
    """MCTS results for all samples of one TargetPredicate.

    Can be saved as a single JSON file and loaded without re-running MCTS
    (trees are not restored, but all textual data is).
    """

    target_predicate_name: str
    model_name: str
    mcts_config: dict[str, Any]
    sample_results: list[MCTSSearchResults_for_MultiPredicateSample] = field(
        default_factory=list
    )

    @property
    def completions_per_sample(self) -> list[dict[str, list[str]]]:
        """List indexed by sample, each entry {prefix: [green_completions]}.

        Feeds directly into extract_anchor_tokens(tp, completions_per_sample).
        """
        return [sr.completions_by_prefix for sr in self.sample_results]

    def save(self, path: PathLike) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            f"Saved MCTSSearchResults for '{self.target_predicate_name}' to {path}"
        )

    @classmethod
    def load(cls, path: PathLike) -> "MCTSSearchResults_for_TargetPredicate":
        with open(path, "r") as f:
            result = cls.from_dict(json.load(f))
        logger.info(
            f"Loaded MCTSSearchResults for '{result.target_predicate_name}' "
            f"({len(result.sample_results)} sample(s)) from {path}"
        )
        return result
