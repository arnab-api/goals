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
HEAD_LOC_PAIRS_SUBDIR = "pairs"
GLOBAL_PAIRS_FILENAME = "global_pairs.json"


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


@dataclass
class PivotInstance(DataClassJsonMixin):
    """A validated pivot prefix for a PredicatePair.

    Stores only the prefix-specific data — predicate identifiers are held by the
    containing ``PredicatePair``.  Access the pair to resolve samples via
    ``HeadLocDataset.get_sample(pair.destination_predicate)``.

    The full incomplete code at pivot:
        sample.python_incomplete_code + " " + pivot_prefix
    (the space after `return` is prepended here, not stored in python_incomplete_code).

    Token divergence (dest_top_token != target_top_token) is the primary selection
    criterion. Pivot instances are mined per model (``pairs/<model>.json``).
    """

    pivot_prefix: str       # shared prefix appended after "return "
    dest_top_token: str     # top-1 predicted token for destination at pivot
    target_top_token: str   # top-1 predicted token for target at pivot


@dataclass
class PredicatePair(DataClassJsonMixin):
    """Pairing metadata between a source predicate (C++) and a destination predicate (Python),
    together with all validated pivot prefixes for this pair.

    Both (A→B) and (B→A) directions are stored as separate entries since training
    signal quality may differ by direction.

    ``pivot_instances`` are mined per model and persisted here so that
    ``pairs/<model>.json`` is the single source of truth for both pairing metadata and
    pivots.

    kl_at_bare_return and top_token_diverges_at_bare_return are legacy fields that the
    saved pairs files still carry; they are never written (always 0.0 / False).
    """

    source_predicate: str         # C++ source, e.g. "is_palindrome"
    destination_predicate: str    # Python destination, e.g. "is_lower"
    variable_type: str            # shared type constraint, e.g. "str"
    oracle_verified: bool = False
    kl_at_bare_return: float = 0.0
    top_token_diverges_at_bare_return: bool = False
    pivot_instances: list[PivotInstance] = field(default_factory=list)


@dataclass
class HeadLocDataset:
    """Loads and indexes single-predicate samples and pairing metadata.

    Does not inherit DataClassJsonMixin — loaded from multiple files, not a single JSON.

    Directory layout (default: data/head_loc/):
        predicates/<predicate_name>.json   — one SinglePredicateSample per file
        global_pairs.json                  — oracle-verified pairs (model-agnostic)
        pairs/<model_name>.json            — {"pairs": [PredicatePair, ...]} with pivots
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

    @property
    def pivot_instances(self) -> list[tuple[PredicatePair, PivotInstance]]:
        """Flat list of (pair, pivot) across all pairs, in pair order."""
        return [(pair, pi) for pair in self.pairs for pi in pair.pivot_instances]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_name: str,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> "HeadLocDataset":
        """Load predicates and model-specific pairs.

        Args:
            model_name: Short model identifier, e.g. ``mt.name.split('/')[-1]``.
                Used to locate ``pairs/{model_name}.json``.
            dir_path: Root head_loc data directory.
        """
        predicates_dir = os.path.join(dir_path, "predicates")
        pairs_path = os.path.join(dir_path, HEAD_LOC_PAIRS_SUBDIR, f"{model_name}.json")

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

        logger.info(
            f"Loaded {len(samples)} predicates, {len(pairs)} pairs"
            f" from {pairs_path}"
        )
        return cls(samples=samples, pairs=pairs)

    def save_global_pairs(
        self,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> None:
        """Persist oracle-verified pairing metadata to ``global_pairs.json``.

        Only saves the model-agnostic fields (source, destination, variable_type,
        oracle_verified). Model-specific fields (kl, pivot_instances) are not
        included — those live in ``pairs/{model_name}.json``.
        """
        path = os.path.join(dir_path, GLOBAL_PAIRS_FILENAME)
        records = [
            {
                "source_predicate": p.source_predicate,
                "destination_predicate": p.destination_predicate,
                "variable_type": p.variable_type,
                "oracle_verified": p.oracle_verified,
            }
            for p in self.pairs
        ]
        with open(path, "w") as f:
            json.dump({"pairs": records}, f, indent=2)
        logger.info(f"Saved {len(records)} global pairs to {path}")

    @staticmethod
    def load_global_pairs(
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> list[PredicatePair]:
        """Load model-agnostic pairs from ``global_pairs.json``.

        Returns only oracle-verified pairs. Returns [] if file absent.
        """
        path = os.path.join(dir_path, GLOBAL_PAIRS_FILENAME)
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            data = json.load(f)
        pairs = [
            PredicatePair(
                source_predicate=p["source_predicate"],
                destination_predicate=p["destination_predicate"],
                variable_type=p["variable_type"],
                oracle_verified=p.get("oracle_verified", False),
            )
            for p in data.get("pairs", [])
            if p.get("oracle_verified", False)
        ]
        logger.info(f"Loaded {len(pairs)} oracle-verified global pairs from {path}")
        return pairs

    def initialize_model_pairs(
        self,
        model_name: str,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> None:
        """Bootstrap ``pairs/{model_name}.json`` from ``global_pairs.json``.

        Loads oracle-verified global pairs and writes them to the model-specific
        file with empty model fields (kl=0.0, top_token_diverges=False,
        pivot_instances=[]). Overwrites any existing file for this model.

        Call this once per new model before running Case 1 or pivot mining.
        """
        global_pairs = self.load_global_pairs(dir_path)
        if not global_pairs:
            raise FileNotFoundError(
                f"global_pairs.json not found or empty in {dir_path}. "
                "Run generate_global_pairs() first."
            )
        self.pairs = global_pairs
        self.save_pairs(model_name, dir_path)
        logger.info(
            f"Initialized pairs/{model_name}.json with "
            f"{len(global_pairs)} global pairs"
        )

    def save_pairs(
        self,
        model_name: str,
        dir_path: PathLike = HEAD_LOC_DATA_DIR,
    ) -> None:
        """Persist pairs (including pivot_instances) to ``pairs/{model_name}.json``."""
        pairs_dir = os.path.join(dir_path, HEAD_LOC_PAIRS_SUBDIR)
        os.makedirs(pairs_dir, exist_ok=True)
        pairs_path = os.path.join(pairs_dir, f"{model_name}.json")
        with open(pairs_path, "w") as f:
            json.dump({"pairs": [p.to_dict() for p in self.pairs]}, f, indent=2)
        logger.info(f"Saved {len(self.pairs)} pairs to {pairs_path}")

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
