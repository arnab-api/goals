"""Task list + sub-goal descriptions for the composing-functions half of the GoalScope
sub-goal-coverage benchmark (`apoorvkh/composing-functions`, all 18 tasks — no cherry-picking,
per explicit instruction).

Each task composes two functions ``y = g(f(x))``. ``f_description``/``g_description`` are the two
sub-goals checked for presence in a verbalization (see `goalscope_subgoal_judge.py`) — written by
reading the dataset's own generator code (`composing_functions/tasks/{arithmetic,wikidata,
translation}.py`, vendored upstream), not guessed from the task name.
"""
from __future__ import annotations

import random

ALL_TASKS: list[str] = [
    # antonym -> translation
    "antonym-spanish",
    "antonym-german",
    "antonym-french",
    # wikidata two-hop relations
    "book-author-birthyear",
    "song-artist-birthyear",
    "landmark-country-capital",
    "park-country-capital",
    "movie-director-birthyear",
    "person-university-year",
    "person-university-founder",
    "product-company-ceo",
    "product-company-hq",
    # arithmetic / symbolic
    "plus-ten-times-two",
    "plus-hundred-times-two",
    "mod-twenty-times-two",
    "word-int-times-two",
    "word-substring-reverse",
    "rgb-rot120-name",
]

# (f_description, g_description) -- f is the first hop (x -> Fx), g is applied to Fx to get GFx.
TASK_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "antonym-spanish": ("find the antonym of the word", "translate the word into Spanish"),
    "antonym-german": ("find the antonym of the word", "translate the word into German"),
    "antonym-french": ("find the antonym of the word", "translate the word into French"),
    "book-author-birthyear": ("name the author of the book", "give the author's birth year"),
    "song-artist-birthyear": ("name the performing artist of the song", "give the artist's birth year"),
    "landmark-country-capital": ("name the country the landmark is in", "name that country's capital city"),
    "park-country-capital": ("name the country the park is in", "name that country's capital city"),
    "movie-director-birthyear": ("name the director of the movie", "give the director's birth year"),
    "person-university-year": ("name the university the person attended", "give the university's founding year"),
    "person-university-founder": ("name the university the person attended", "name the university's founder"),
    "product-company-ceo": ("name the company that makes the product", "name the company's CEO"),
    "product-company-hq": ("name the company that makes the product", "name the company's headquarters location"),
    "plus-ten-times-two": ("add ten to the number", "multiply the number by two"),
    "plus-hundred-times-two": ("add one hundred to the number", "multiply the number by two"),
    "mod-twenty-times-two": ("take the number mod twenty", "multiply the number by two"),
    "word-int-times-two": ("convert the number-word into its digit form", "double the number"),
    "word-substring-reverse": ("remove the last letter of the word", "reverse the letters of the word"),
    "rgb-rot120-name": ("rotate the RGB color's hue by 120 degrees", "name the closest CSS color for the rotated color"),
}


def task_subgoals(task_name: str) -> list[str]:
    """[f_description, g_description] -- the two presence-checked sub-goals for this task."""
    f_desc, g_desc = TASK_DESCRIPTIONS[task_name]
    return [f_desc, g_desc]


_HF_DATASET = None


def _load_hf_dataset():
    global _HF_DATASET
    if _HF_DATASET is None:
        from datasets import load_dataset
        _HF_DATASET = load_dataset("apoorvkh/composing-functions")["train"]
    return _HF_DATASET


def load_task_records(task_name: str, seed: int = 0) -> list[dict]:
    """All rows for `task_name`, shuffled with `seed` (deterministic)."""
    ds = _load_hf_dataset()
    recs = [r for r in ds if r["task"] == task_name]
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs


def to_demos(recs: list[dict]) -> list[dict]:
    """`build_icl_prompt`/`build_icl_prompt_with_spans` want {'input':..., 'output':...} dicts --
    the raw HF records use the paper's x/Fx/Gx/GFx/FGx column names."""
    return [{"input": r["x"], "output": r["GFx"]} for r in recs]
