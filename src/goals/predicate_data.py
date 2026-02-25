import json
import logging
import os
import pathlib
from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin

from src.goals.data import IncompleteCodeSample
from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

MULTI_PREDICATE_DIR = os.path.join(DEFAULT_DATA_DIR, "multi_predicate")


@dataclass
class PredicateSpec(DataClassJsonMixin):
    """Specification for one predicate in a multi-predicate task.

    anchor_tokens: token substrings that uniquely identify the model is
    implementing this predicate (filled in post-hoc after inspecting MCTS trees).
    """

    description: str
    anchor_tokens: list[str] = field(default_factory=list)


@dataclass
class MultiPredicateSample(IncompleteCodeSample):
    """A code completion task paired with a specific P- companion.

    Extends IncompleteCodeSample with a P- predicate spec. P+ lives at the
    TargetPredicate level and is shared across all samples in that group.

    The pivot is the point in `code` where the model must choose which predicate
    to implement first (typically right after `return `).

    Prefills convention:
      - "aligned_p_plus":  nudge/force the P+-first branch
      - "aligned_p_minus": nudge/force the P--first branch
      - "adversarial":     unusual/tricky implementations
      - "inconsequential": model's natural preference (empty or neutral)
    """

    p_minus: PredicateSpec | None = None


@dataclass
class TargetPredicate(DataClassJsonMixin):
    """All samples sharing the same P+ (instrumental goal).

    One TargetPredicate corresponds to one .json file in data/multi_predicate/.
    The file is named after the predicate (e.g., is_palindrome.json) and that
    stem is stored in `name`.
    """

    p_plus: PredicateSpec
    samples: list[MultiPredicateSample] = field(default_factory=list)
    name: str = ""

    @staticmethod
    def load(path: PathLike) -> "TargetPredicate":
        path = pathlib.Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        tp = TargetPredicate.from_dict(data)
        if not tp.name:
            tp.name = path.stem
        logger.info(
            f"Loaded TargetPredicate '{tp.name}' with {len(tp.samples)} sample(s) from {path}"
        )
        return tp

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultiPredicateSample:
        return self.samples[idx]


@dataclass
class MultiPredicateDataset(DataClassJsonMixin):
    """Collection of TargetPredicates, one per P+ instrumental goal.

    Loaded from a directory of .json files (data/multi_predicate/ by default).
    Each .json file corresponds to one TargetPredicate.
    """

    groups: list[TargetPredicate] = field(default_factory=list)

    @staticmethod
    def load_dataset(
        dir_path: PathLike = MULTI_PREDICATE_DIR,
    ) -> "MultiPredicateDataset":
        dir_path = pathlib.Path(dir_path)
        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {dir_path}")
        groups = [TargetPredicate.load(p) for p in json_files]
        logger.info(
            f"Loaded MultiPredicateDataset: {len(groups)} target predicate(s) "
            f"from {dir_path}"
        )
        return MultiPredicateDataset(groups=groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> TargetPredicate:
        return self.groups[idx]

    def get_group(self, name: str) -> TargetPredicate:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"No TargetPredicate named '{name}'")
