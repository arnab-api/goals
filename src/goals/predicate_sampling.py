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
    tree: dict[str, Any] | None = field(default=None)
    """Serialized MCTSNode tree (root.to_dict()).  None if not saved."""


@dataclass
class MCTSSearchResults_for_MultiPredicateSample(DataClassJsonMixin):
    """MCTS results for one MultiPredicateSample across all tested prefills.

    `runs` is the serializable list of MCTSRunResult (one per prefill run).
    Each run optionally carries a serialized tree in `run.tree` (set via
    `root.to_dict()`).  `_trees` is the in-memory cache (prefix → MCTSNode)
    reconstructed automatically from `run.tree` on load, or populated live
    during an MCTS session.
    """

    sample: MultiPredicateSample
    runs: list[MCTSRunResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        # In-memory tree cache: prefix → MCTSNode root.
        # Populated here from serialized run.tree data (if present), or during a
        # live MCTS session.  Keyed by run.prefix, same as completions_by_prefix.
        from src.goals.sampling import MCTSNode  # lazy — avoids circular import at load time

        self._trees: dict[str, Any] = {
            run.prefix: MCTSNode.from_dict(run.tree)
            for run in self.runs
            if run.tree is not None
        }

    @property
    def completions_by_prefix(self) -> dict[str, list[str]]:
        """Returns {prefix: green_completions} for use with extract_anchor_tokens."""
        return {run.prefix: run.green_completions for run in self.runs}

    def save(self, path: PathLike) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved MCTSSearchResults for sample → {path}")

    @classmethod
    def load(cls, path: PathLike) -> "MCTSSearchResults_for_MultiPredicateSample":
        with open(path, "r") as f:
            result = cls.from_dict(json.load(f))
        logger.info(f"Loaded MCTSSearchResults for sample from {path}")
        return result


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

    def save(self, predicate_dir: PathLike) -> None:
        """Save per-sample results as individual mcts.json files under predicate_dir.

        Each sample is saved to {predicate_dir}/{subdir}/mcts.json where subdir
        is derived from the sample's P- description (matching the MCTS script's
        naming convention).
        """
        predicate_dir = pathlib.Path(predicate_dir)
        predicate_dir.mkdir(parents=True, exist_ok=True)
        for sr in self.sample_results:
            # Derive subdir name from sample P- description (same as _p_minus_dir_name)
            if sr.sample.p_minus is not None:
                desc = sr.sample.p_minus.description[:50]
                subdir = desc.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
                subdir = "".join(c for c in subdir if c.isalnum() or c in "_-.")
                subdir = subdir[:40] or "empty"
            else:
                subdir = f"sample{self.sample_results.index(sr)}"
            sample_path = predicate_dir / subdir / "mcts.json"
            sr.save(sample_path)
        logger.info(
            f"Saved MCTSSearchResults for '{self.target_predicate_name}' "
            f"({len(self.sample_results)} sample(s)) under {predicate_dir}"
        )

    @classmethod
    def load(cls, path: PathLike) -> "MCTSSearchResults_for_TargetPredicate":
        """Load from a predicate directory containing per-sample subdirs.

        Supports two layouts:
          - New (per-sample): path is a directory, scan */mcts.json
          - Legacy (single file): path is a single JSON file
        """
        path = pathlib.Path(path)

        # Legacy: single JSON file
        if path.is_file():
            with open(path, "r") as f:
                result = cls.from_dict(json.load(f))
            logger.info(
                f"Loaded MCTSSearchResults for '{result.target_predicate_name}' "
                f"({len(result.sample_results)} sample(s)) from {path}"
            )
            return result

        # New layout: path is predicate directory, scan subdirs for mcts.json
        if not path.is_dir():
            raise FileNotFoundError(f"MCTS results not found at {path}")

        sample_jsons = sorted(path.glob("*/mcts.json"))
        if not sample_jsons:
            raise FileNotFoundError(
                f"No per-sample mcts.json files found under {path}/*/mcts.json"
            )

        sample_results = []
        mcts_config: dict = {}
        model_name = ""
        predicate_name = ""

        for sp in sample_jsons:
            sr = MCTSSearchResults_for_MultiPredicateSample.load(sp)
            sample_results.append(sr)
            # Infer metadata from first run
            if sr.runs and not mcts_config:
                mcts_config = sr.runs[0].mcts_config

        # Infer predicate name from directory name
        predicate_name = path.name

        result = cls(
            target_predicate_name=predicate_name,
            model_name=model_name,
            mcts_config=mcts_config,
            sample_results=sample_results,
        )
        logger.info(
            f"Loaded MCTSSearchResults for '{predicate_name}' "
            f"({len(sample_results)} sample(s)) from {path}"
        )
        return result
