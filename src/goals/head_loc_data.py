import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin

from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

HEAD_LOC_DATA_DIR = os.path.join(DEFAULT_DATA_DIR, "head_loc")
PIVOT_INSTANCES_FILENAME = "pivot_instances.json"


@dataclass
class SinglePredicateSample(DataClassJsonMixin):
    """One predicate function in both Python and C++, each trimmed at `return`.

    python_incomplete_code ends with `return` (no trailing space).
    cpp_incomplete_code ends with `return` in one-liner brace form:
        bool isPalindrome(string s) { return
    The space before the expression — and any pivot prefix — is appended externally.

    docstring is an optional one-line description (no triple-quotes), e.g.
    "Return True if s is a palindrome." When non-empty and include_docstring=True
    is passed to build_python_prompt, it is inserted as a docstring after the def
    line and before `return`, giving the model a semantic hint about the function's
    intent without changing the pivot position.
    """

    predicate_name: str          # canonical snake_case id, e.g. "is_palindrome"
    variable_type: str           # e.g. "str", "int", "list[int]"
    variable_name: str           # Python variable name, e.g. "s", "n", "lst"
    python_incomplete_code: str  # "def is_palindrome(s: str) -> bool:\n    return"
    cpp_incomplete_code: str     # "bool isPalindrome(string s) { return"
    unit_tests: list[tuple[Any, Any]]
    function_name: str = ""      # Python fn name; auto-set to predicate_name if empty
    docstring: str = ""          # optional semantic hint, e.g. "Return True if s is a palindrome."
    augmentation: list[dict] | None = None

    def __post_init__(self) -> None:
        if not self.function_name:
            self.function_name = self.predicate_name
        self.unit_tests = [
            t if isinstance(t, tuple) else tuple(t) for t in self.unit_tests
        ]

    @property
    def is_original(self) -> bool:
        return not self.augmentation


@dataclass
class PredicatePair(DataClassJsonMixin):
    """Pairing metadata between a source predicate (C++) and a destination predicate (Python).

    Both (A→B) and (B→A) directions are stored as separate entries since training
    signal quality may differ by direction.

    kl_at_bare_return and top_token_diverges_at_bare_return are populated by the
    pairing script (Step 3) which requires a model forward pass; stored as 0.0 / False
    until then.
    """

    source_predicate: str         # C++ source, e.g. "is_palindrome"
    destination_predicate: str    # Python destination, e.g. "is_lower"
    variable_type: str            # shared type constraint, e.g. "str"
    oracle_verified: bool = False
    kl_at_bare_return: float = 0.0
    top_token_diverges_at_bare_return: bool = False


@dataclass
class PivotInstance(DataClassJsonMixin):
    """A ready-to-train triple for the head localization experiment.

    Stores predicate name keys only (not inline sample objects) to avoid redundancy.
    Samples are resolved via HeadLocDataset.get_sample().

    The full incomplete code at pivot:
        sample.python_incomplete_code + " " + pivot_prefix
    (the space after `return` is prepended here, not stored in python_incomplete_code).

    pivot_prefix == "" for Case 1 (bare return divergence),
    non-empty for Case 2 (shared prefix of successful generations).

    Token divergence (dest_top_token != target_top_token) is the primary selection
    criterion. KL is not stored here because it is model-specific and can change
    across LMs; the top-token strings are more interpretable and capture the same
    signal. Pivot instances are mined online per model — do not cache across models.
    """

    source_predicate: str       # C++ sample whose head outputs are patched in
    destination_predicate: str  # Python sample receiving the intervention
    target_predicate: str       # Python sample to match (semantically == source_predicate)
    pivot_prefix: str           # shared prefix appended after "return "; "" for Case 1
    dest_top_token: str         # top-1 predicted token for destination at pivot
    target_top_token: str       # top-1 predicted token for target at pivot

    @property
    def tokens_diverge(self) -> bool:
        """True when dest and target predict different top tokens at the pivot."""
        return self.dest_top_token != self.target_top_token


@dataclass
class HeadLocMCTSResult(DataClassJsonMixin):
    """Serializable MCTS result for one SinglePredicateSample.

    green_completions: Unique "effective" passing completion strings — the
    raw suffix after `return ` with stop tokens stripped and deduplicated.
    This is sufficient for LCP computation; the tree is optional (debugging only).
    """

    predicate_name: str
    model_name: str
    prompt: str                   # full prompt sent to LM
    prefix: str                   # python_incomplete_code + " " (used in evaluate_mcts_tree)
    mcts_config: dict[str, Any]
    mcts_stats: dict[str, Any]
    eval_counts: dict[str, int]   # {"green": N, "orange": N, "red": N, "total": N}
    green_completions: list[str]  # unique stripped passing suffixes; empty list if none
    tree: dict[str, Any] | None = None  # root.to_dict(); None if save_tree=False


MCTS_RESULTS_SUBDIR = "mcts"


@dataclass
class HeadLocDataset:
    """Loads and indexes single-predicate samples and pairing metadata.

    Does not inherit DataClassJsonMixin — loaded from multiple files, not a single JSON.

    Directory layout (default: data/head_loc/):
        predicates/<predicate_name>.json   — one SinglePredicateSample per file
        pairs.json                         — {"pairs": [PredicatePair, ...]}
    """

    samples: list[SinglePredicateSample]
    pairs: list[PredicatePair]
    _sample_index: dict[str, SinglePredicateSample] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._sample_index = {s.predicate_name: s for s in self.samples}

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_sample(self, predicate_name: str) -> SinglePredicateSample | None:
        return self._sample_index.get(predicate_name)

    def get_pair(self, source: str, destination: str) -> PredicatePair | None:
        for p in self.pairs:
            if p.source_predicate == source and p.destination_predicate == destination:
                return p
        return None

    def pairs_for_predicate(self, predicate_name: str) -> list[PredicatePair]:
        """All pairs where predicate_name appears as source or destination."""
        return [
            p for p in self.pairs
            if p.source_predicate == predicate_name
            or p.destination_predicate == predicate_name
        ]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, dir_path: PathLike = HEAD_LOC_DATA_DIR) -> "HeadLocDataset":
        predicates_dir = os.path.join(dir_path, "predicates")
        pairs_path = os.path.join(dir_path, "pairs.json")

        samples = []
        if os.path.isdir(predicates_dir):
            for fname in sorted(os.listdir(predicates_dir)):
                if fname.endswith(".json"):
                    with open(os.path.join(predicates_dir, fname)) as f:
                        samples.append(SinglePredicateSample.from_dict(json.load(f)))

        pairs = []
        if os.path.isfile(pairs_path):
            with open(pairs_path) as f:
                data = json.load(f)
                pairs = [PredicatePair.from_dict(p) for p in data.get("pairs", [])]

        logger.info(f"Loaded {len(samples)} predicates, {len(pairs)} pairs from {dir_path}")
        return cls(samples=samples, pairs=pairs)

    def save_pairs(self, dir_path: PathLike = HEAD_LOC_DATA_DIR) -> None:
        """Persist updated pairs list (e.g. after KL values are filled in by Step 3)."""
        pairs_path = os.path.join(dir_path, "pairs.json")
        with open(pairs_path, "w") as f:
            json.dump({"pairs": [p.to_dict() for p in self.pairs]}, f, indent=2)
        logger.info(f"Saved {len(self.pairs)} pairs to {pairs_path}")

    def save_pivot_instances(
        self,
        pivot_instances: list[PivotInstance],
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> None:
        """Persist pivot instances to pivot_instances.json."""
        path = os.path.join(dir_path, PIVOT_INSTANCES_FILENAME)
        with open(path, "w") as f:
            json.dump([pi.to_dict() for pi in pivot_instances], f, indent=2)
        logger.info(f"Saved {len(pivot_instances)} pivot instances to {path}")

    def save_mcts_result(
        self,
        result: HeadLocMCTSResult,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> None:
        """Save a HeadLocMCTSResult to <dir_path>/<predicate_name>.json."""
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{result.predicate_name}.json")
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved MCTS result for {result.predicate_name!r} to {path}")

    @staticmethod
    def load_mcts_result(
        predicate_name: str,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> "HeadLocMCTSResult | None":
        """Load a HeadLocMCTSResult from disk. Returns None if file absent."""
        path = os.path.join(dir_path, f"{predicate_name}.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return HeadLocMCTSResult.from_dict(json.load(f))

    def load_all_mcts_results(
        self,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> "dict[str, HeadLocMCTSResult]":
        """Load all cached MCTS results for samples in this dataset.

        Iterates over ds.samples, attempts to load each, skips missing files.
        Returns {predicate_name: HeadLocMCTSResult}.
        """
        results: dict[str, HeadLocMCTSResult] = {}
        for sample in self.samples:
            result = self.load_mcts_result(sample.predicate_name, dir_path)
            if result is not None:
                results[sample.predicate_name] = result
        logger.info(
            f"Loaded {len(results)}/{len(self.samples)} MCTS results from {dir_path}"
        )
        return results

    @staticmethod
    def load_pivot_instances(
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> list[PivotInstance]:
        """Load pivot instances from pivot_instances.json. Returns [] if file absent."""
        path = os.path.join(dir_path, PIVOT_INSTANCES_FILENAME)
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            return [PivotInstance.from_dict(d) for d in json.load(f)]

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int | str) -> SinglePredicateSample:
        if isinstance(idx, str):
            return self._sample_index[idx]
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
