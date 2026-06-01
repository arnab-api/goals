import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dataclasses_json import DataClassJsonMixin

from src.goals.data import IncompleteCodeSample
from src.utils.env_utils import DEFAULT_DATA_DIR
from src.utils.typing import PathLike

logger = logging.getLogger(__name__)

ListStyle = Literal["numbered", "bulleted"]
Modularity = Literal["modular", "monolithic"]


@dataclass
class MultiStepSample(IncompleteCodeSample):
    """A multi-step program-writing task for the clock-signal experiment.

    Extends IncompleteCodeSample. The canonical representation is `steps` —
    `instruction` and `code` are auto-rendered from `steps` + flags in
    __post_init__ so existing code that reads `sample.instruction` /
    `sample.code` keeps working. To re-render with different flags, call
    `render_instruction(...)` / `render_code_stub(...)` directly.

    Flags:
      list_style:  "numbered" | "bulleted"     — how the steps are listed
      modularity:  "modular"  | "monolithic"   — write helpers vs single function
    """

    # Override parent fields with empty defaults — they're auto-filled in __post_init__.
    instruction: str = ""
    code: str = ""

    # Override parent defaults: this dataset asks for no comments by default,
    # with a more direct phrasing than the parent's default no-comment line.
    without_comments: bool = True
    no_comment_str: str = (
        "Do not add any comments or docstrings. Only provide the code."
    )

    # New fields (canonical representation).
    name: str = ""
    description: str = ""  # metadata only — not rendered into the prompt
    steps: list[str] = field(default_factory=list)
    domain: str = "mixed"
    entry_function: str = "main"
    entry_signature: str = "def main():"

    # Render-time flags.
    list_style: ListStyle = "numbered"
    modularity: Modularity = "modular"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.instruction:
            self.instruction = self.render_instruction()
        if not self.code:
            self.code = self.render_code_stub()

    @property
    def full_instruction(self) -> str:
        # render_instruction already appends the no-comment suffix when
        # without_comments is True, so just return instruction as-is and avoid
        # the parent's double-append behavior.
        return self.render_instruction()

    def render_instruction(
        self,
        list_style: ListStyle | None = None,
        modularity: Modularity | None = None,
        without_comments: bool | None = None,
    ) -> str:
        ls = list_style or self.list_style
        mod = modularity or self.modularity
        wc = self.without_comments if without_comments is None else without_comments

        if ls == "numbered":
            marker = lambda i: f"{i + 1}."  # noqa: E731
        elif ls == "bulleted":
            marker = lambda i: "-"  # noqa: E731
        else:
            raise ValueError(f"unknown list_style={ls!r}")

        body = "\n".join(f"{marker(i)} {step}" for i, step in enumerate(self.steps))

        opener = "Write a python code that does the following:"

        if mod == "modular":
            mod_suffix = (
                "Implement each of the above steps as its own well-defined function."
            )
        elif mod == "monolithic":
            mod_suffix = (
                "Put all the logic inside a single function `main()` (taking no arguments) "
                "— do not define any helper functions. If a step requires reading from a "
                "file path, the command line, or standard input, do that inside `main()` as well."
            )
        else:
            raise ValueError(f"unknown modularity={mod!r}")

        parts = [opener, "", body, "", mod_suffix]
        if wc:
            parts.append(self.no_comment_str)

        return "\n".join(parts)

    def render_code_stub(self, modularity: Modularity | None = None) -> str:
        mod = modularity or self.modularity
        if mod == "monolithic":
            # Monolithic mode always uses `def main():` so terminal-IO steps are
            # consistent with the function's interface. Per-sample
            # `entry_function`/`entry_signature` are kept as documentation of
            # the natural library interface but are not rendered here.
            return "def main():\n"
        return ""

    def render(
        self,
        list_style: ListStyle | None = None,
        modularity: Modularity | None = None,
        without_comments: bool | None = None,
    ) -> tuple[str, str]:
        """Returns (instruction, code_stub) under the chosen flags."""
        return (
            self.render_instruction(
                list_style=list_style,
                modularity=modularity,
                without_comments=without_comments,
            ),
            self.render_code_stub(modularity=modularity),
        )


@dataclass
class MultiStepDataset(DataClassJsonMixin):
    samples: list[MultiStepSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx) -> MultiStepSample:
        return self.samples[idx]

    @staticmethod
    def load_dataset(
        path: PathLike = os.path.join(DEFAULT_DATA_DIR, "multistep", "seed.json"),
    ) -> "MultiStepDataset":
        with open(path, "r") as f:
            data = json.load(f)
        # Clear stale rendered fields so __post_init__ re-renders under the
        # sample's current list_style / modularity.
        for s in data.get("samples", []):
            s["instruction"] = ""
            s["code"] = ""
        ds = MultiStepDataset.from_dict(data)
        logger.info(f"loaded MultiStepDataset: {len(ds)} samples from {path}")
        return ds

    def save(self, path: PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Strip the auto-rendered fields before saving — they're derived.
        d = self.to_dict()
        for s in d.get("samples", []):
            s.pop("instruction", None)
            s.pop("code", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        logger.info(f"saved MultiStepDataset: {len(self)} samples → {path}")
