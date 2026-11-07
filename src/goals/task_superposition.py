"""ICL task-superposition prompt builders (Track 2 · GoalScope OOD eval).

Vendored + adapted from the task-superposition paper's task generators:
  "Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks
  in Superposition" — https://github.com/edixiong/task-superposition

Families used by ``19_goalscope_out_of_domain.ipynb``:
  - country        — country -> {capital, continent, UPPERCASE}
  - country_meta   — country -> {capital, currency, Chinese name} (more visually/formally
                      distinct outputs than `country` — capital is a Latin city name, currency
                      a Latin phrase, Chinese name a different script entirely)
  - AB             — copy-A / copy-B / A+B digit arithmetic
  - add_translate  — A+B, with the sum spelled out in one of several languages
  - word           — word -> {first letter, last letter, FIRST LETTER, LAST LETTER}. The original
                      repo pulls words from the external ``random-word`` package (network calls,
                      flagged as slow in their README); here it's a static wordlist vendored to
                      ``data/task_superposition/util/wordlist.txt`` (filtered from the system
                      dictionary, same first-letter != last-letter constraint they use).

``country``/``country_meta`` are built via ``make_country_family(fields)`` (any subset of
{capital, continent, currency, upper, zh}); ``word`` via ``make_word_family(fields)`` (any subset
of {first_letter, last_letter, first_upper, last_upper}) — see those functions for other
combinations.

The core mechanic (unchanged from the paper): a few-shot prompt is built by
INTERLEAVING lines drawn from several distinct input->output tasks (a mixture,
so no single line disambiguates which task governs the whole prompt). The final
query line poses one input with no output; ``output_choices`` holds the
candidate completion under each task, aligned with ``task_labels``. The paper's
finding is that the model's next-token probability mass often SPLITS across
multiple candidates roughly matching the mixture weights, rather than
collapsing to one task — i.e. the model appears to pursue several instrumental
goals "in superposition." ``score_choices`` reproduces their per-candidate
teacher-forced scoring (``get_probs`` in the original ``run_experiments.py``)
so that GoalScope read-outs at the query pivot can be checked against it.

``build_superposition_prompt(..., order=...)`` supports two interleaving orders: ``"random"``
(the paper's default — the realized mixture is shuffled, unpredictable line-to-line) and
``"cyclic"`` (deterministic round-robin task0, task1, task2, ... — added here, not in the
original repo, to test whether GoalScope's read-out gets sharper/more confident when the
identity of the next task is positionally predictable rather than genuinely ambiguous).

``build_superposition_prompt(..., prompt_format=...)`` also supports two example/query renderings:
``"raw"`` (default — the paper's inline-arrow format, e.g. ``"45@67=112"``) and ``"qa"``
(function-vector-style ``Q:``/``A:`` format, mirroring ``word_pairs_to_prompt_data`` in
``existing_works/function_vectors/src/utils/prompt_utils.py``).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.models import ModelandTokenizer
from src.tokens import prepare_input

_UTIL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "task_superposition", "util"
)


def _load_dict(name: str) -> dict:
    return torch.load(os.path.join(_UTIL_DIR, f"{name}.pt"), weights_only=False)


COUNTRY_CAPITAL = _load_dict("country_capital_dict")
COUNTRY_CONTINENT = _load_dict("country_continent_dict")
COUNTRY_CURRENCY = _load_dict("country_currency_dict")
COUNTRY_ZH = _load_dict("country_eng_ch_dict")
LANG_NUM_WORDS = {lang: _load_dict(f"{lang}_dict") for lang in ["or", "en", "fr", "es", "de", "ko", "zh"]}

with open(os.path.join(_UTIL_DIR, "wordlist.txt")) as _f:
    WORDLIST = [w.strip() for w in _f if w.strip()]


# ---------------------------------------------------------------------------
# ExampleLine — an example's input/output split, plus the "raw" inline-arrow
# rendering (e.g. "45@67=112"). The split is what makes Q:/A: formatting
# possible; ``raw`` is kept so the paper's original format stays exactly
# reproducible and so dedup checks (below) can still substring-match against
# previously generated lines regardless of ``prompt_format``.
# ---------------------------------------------------------------------------

@dataclass
class ExampleLine:
    input: str
    output: str
    raw: str


def _prev_raw(examples: list[ExampleLine]) -> str:
    return "\n".join(e.raw for e in examples)


# ---------------------------------------------------------------------------
# Per-line generators — one call produces one few-shot example line
# ---------------------------------------------------------------------------

def _gen_copyA(examples: list[ExampleLine], low: int = 10, high: int = 99, symbol: str = "@", symbol2: str = "=") -> ExampleLine:
    A, B = random.randint(low, high), random.randint(low, high)
    inp, out = f"{A}{symbol}{B}", str(A)
    return ExampleLine(input=inp, output=out, raw=f"{inp}{symbol2}{out}")


def _gen_copyB(examples: list[ExampleLine], low: int = 10, high: int = 99, symbol: str = "@", symbol2: str = "=") -> ExampleLine:
    A, B = random.randint(low, high), random.randint(low, high)
    inp, out = f"{A}{symbol}{B}", str(B)
    return ExampleLine(input=inp, output=out, raw=f"{inp}{symbol2}{out}")


def _gen_AplusB(examples: list[ExampleLine], low: int = 10, high: int = 99, symbol: str = "@", symbol2: str = "=") -> ExampleLine:
    A, B = random.randint(low, high), random.randint(low, high)
    inp, out = f"{A}{symbol}{B}", str(A + B)
    return ExampleLine(input=inp, output=out, raw=f"{inp}{symbol2}{out}")


def _gen_AplusB_t(
    examples: list[ExampleLine], low: int = 10, high: int = 99, language: str = "en", symbol: str = "+", symbol2: str = "->"
) -> ExampleLine:
    A, B = random.randint(low, high), random.randint(low, high)
    inp, out = f"{A}{symbol}{B}", LANG_NUM_WORDS[language][A + B]
    return ExampleLine(input=inp, output=out, raw=f"{inp}{symbol2}{out}")


# ---------------------------------------------------------------------------
# Query-line generators — return (query_input, query_raw, output_choices).
# ``query_raw`` is the paper's inline format ending in the joiner symbol with
# no value yet (e.g. "45@67="); ``query_input`` is just the bare input, used
# to build the Q:/A: query line instead.
# ---------------------------------------------------------------------------

def _q_AB1(examples: list[ExampleLine], low: int = 10, high: int = 99, symbol: str = "@", symbol2: str = "=") -> tuple[str, str, list[str]]:
    while True:
        A, B = random.randint(low, high), random.randint(low, high)
        inp = f"{A}{symbol}{B}"
        if inp not in _prev_raw(examples):
            return inp, f"{inp}{symbol2}", [str(A), str(B), str(A + B)]


def _q_add_translate(
    examples: list[ExampleLine],
    low: int = 10,
    high: int = 99,
    lang_list: tuple[str, ...] = ("or", "en", "fr", "es"),
    symbol: str = "+",
    symbol2: str = "->",
) -> tuple[str, str, list[str]]:
    while True:
        A, B = random.randint(low, high), random.randint(low, high)
        inp = f"{A}{symbol}{B}"
        if inp not in _prev_raw(examples):
            return inp, f"{inp}{symbol2}", [LANG_NUM_WORDS[lang][A + B] for lang in lang_list]


# ---------------------------------------------------------------------------
# Family registry + prompt builder
# ---------------------------------------------------------------------------

@dataclass
class TaskFamily:
    name: str
    line_funcs: list[Callable]
    line_kwargs: list[dict]
    query_func: Callable
    query_kwargs: dict
    task_labels: list[str]  # human-readable name per task index, aligned to output_choices order


# ---------------------------------------------------------------------------
# country field system — build a TaskFamily from any subset of country meta-data fields
# ---------------------------------------------------------------------------

_COUNTRY_FIELD_VALUES: dict[str, Callable[[str], str]] = {
    "capital": lambda c: COUNTRY_CAPITAL[c],
    "continent": lambda c: COUNTRY_CONTINENT[c],
    "currency": lambda c: COUNTRY_CURRENCY[c],
    "upper": lambda c: c.upper(),
    "zh": lambda c: COUNTRY_ZH[c],
}
_COUNTRY_FIELD_LABELS: dict[str, str] = {
    "capital": "capital",
    "continent": "continent",
    "currency": "currency",
    "upper": "UPPERCASE",
    "zh": "Chinese name",
}


def _gen_country_field(field: str) -> Callable[..., ExampleLine]:
    value_fn = _COUNTRY_FIELD_VALUES[field]

    def _gen(examples: list[ExampleLine], symbol2: str = "->") -> ExampleLine:
        while True:
            country = random.choice(list(COUNTRY_CAPITAL))
            if country not in _prev_raw(examples):
                out = value_fn(country)
                return ExampleLine(input=country, output=out, raw=f"{country}{symbol2}{out}")

    return _gen


def _q_country_fields(fields: list[str]) -> Callable[..., tuple[str, str, list[str]]]:
    value_fns = [_COUNTRY_FIELD_VALUES[f] for f in fields]

    def _q(examples: list[ExampleLine], symbol2: str = "->") -> tuple[str, str, list[str]]:
        while True:
            country = random.choice(list(COUNTRY_CAPITAL))
            if country not in _prev_raw(examples):
                return country, f"{country}{symbol2}", [fn(country) for fn in value_fns]

    return _q


def make_country_family(fields: list[str], name: Optional[str] = None) -> TaskFamily:
    """Build a country TaskFamily over any subset of {capital, continent, currency, upper, zh}."""
    fields = list(fields)
    return TaskFamily(
        name=name or f"country[{'+'.join(fields)}]",
        line_funcs=[_gen_country_field(f) for f in fields],
        line_kwargs=[{}] * len(fields),
        query_func=_q_country_fields(fields),
        query_kwargs={},
        task_labels=[_COUNTRY_FIELD_LABELS[f] for f in fields],
    )


# ---------------------------------------------------------------------------
# word field system — build a TaskFamily from any subset of word transform fields
# ---------------------------------------------------------------------------

_WORD_FIELD_VALUES: dict[str, Callable[[str], str]] = {
    "first_letter": lambda w: w[0].lower(),
    "last_letter": lambda w: w[-1].lower(),
    "first_upper": lambda w: w[0].upper(),
    "last_upper": lambda w: w[-1].upper(),
}
_WORD_FIELD_LABELS: dict[str, str] = {
    "first_letter": "first letter",
    "last_letter": "last letter",
    "first_upper": "first letter (upper)",
    "last_upper": "last letter (upper)",
}


def _pick_word(examples: list[ExampleLine]) -> str:
    joined = _prev_raw(examples)
    while True:
        word = random.choice(WORDLIST)
        if word not in joined:
            return word


def _gen_word_field(field: str) -> Callable[..., ExampleLine]:
    value_fn = _WORD_FIELD_VALUES[field]

    def _gen(examples: list[ExampleLine], symbol2: str = "->") -> ExampleLine:
        word = _pick_word(examples)
        out = value_fn(word)
        return ExampleLine(input=word, output=out, raw=f"{word}{symbol2}{out}")

    return _gen


def _q_word_fields(fields: list[str]) -> Callable[..., tuple[str, str, list[str]]]:
    value_fns = [_WORD_FIELD_VALUES[f] for f in fields]

    def _q(examples: list[ExampleLine], symbol2: str = "->") -> tuple[str, str, list[str]]:
        word = _pick_word(examples)
        return word, f"{word}{symbol2}", [fn(word) for fn in value_fns]

    return _q


def make_word_family(fields: list[str], name: Optional[str] = None) -> TaskFamily:
    """Build a word TaskFamily over any subset of {first_letter, last_letter, first_upper, last_upper}."""
    fields = list(fields)
    return TaskFamily(
        name=name or f"word[{'+'.join(fields)}]",
        line_funcs=[_gen_word_field(f) for f in fields],
        line_kwargs=[{}] * len(fields),
        query_func=_q_word_fields(fields),
        query_kwargs={},
        task_labels=[_WORD_FIELD_LABELS[f] for f in fields],
    )


TASK_FAMILIES: dict[str, TaskFamily] = {
    "country": make_country_family(["capital", "continent", "upper"], name="country"),
    "country_meta": make_country_family(["capital", "currency", "zh"], name="country_meta"),
    "word": make_word_family(["first_letter", "last_letter", "first_upper", "last_upper"], name="word"),
    "AB": TaskFamily(
        name="AB",
        line_funcs=[_gen_copyA, _gen_copyB, _gen_AplusB],
        line_kwargs=[{"low": 10, "high": 99}] * 3,
        query_func=_q_AB1,
        query_kwargs={"low": 10, "high": 99},
        task_labels=["copy A", "copy B", "A+B"],
    ),
    "add_translate": TaskFamily(
        name="add_translate",
        line_funcs=[_gen_AplusB_t] * 4,
        line_kwargs=[
            {"low": 10, "high": 99, "language": lang, "symbol": "+", "symbol2": "->"}
            for lang in ["or", "en", "fr", "es"]
        ],
        query_func=_q_add_translate,
        query_kwargs={"low": 10, "high": 99, "lang_list": ["or", "en", "fr", "es"], "symbol": "+", "symbol2": "->"},
        task_labels=["Odia", "English", "French", "Spanish"],
    ),
    # Two 3-language variants of the paper's headline addition+translate task (Fig 1a/2a/Sec 3),
    # matching the "English/Spanish/French" and "French/German/Italian" combinations described in
    # the paper's text. Italian number-word data isn't vendored anywhere (confirmed against both
    # the original repo and the arXiv source: the paper's actual fr/de/it experiment, Sec 6/Fig 4b,
    # is a *different*, simpler task -- plain number->word translation with no addition at all) --
    # Spanish is substituted for Italian here to keep the addition+translate structure intact.
    "add_translate_efs": TaskFamily(
        name="add_translate_efs",
        line_funcs=[_gen_AplusB_t] * 3,
        line_kwargs=[
            {"low": 10, "high": 99, "language": lang, "symbol": "+", "symbol2": "->"}
            for lang in ["en", "fr", "es"]
        ],
        query_func=_q_add_translate,
        query_kwargs={"low": 10, "high": 99, "lang_list": ["en", "fr", "es"], "symbol": "+", "symbol2": "->"},
        task_labels=["English", "French", "Spanish"],
    ),
    "add_translate_dfs": TaskFamily(
        name="add_translate_dfs",
        line_funcs=[_gen_AplusB_t] * 3,
        line_kwargs=[
            {"low": 10, "high": 99, "language": lang, "symbol": "+", "symbol2": "->"}
            for lang in ["de", "fr", "es"]
        ],
        query_func=_q_add_translate,
        query_kwargs={"low": 10, "high": 99, "lang_list": ["de", "fr", "es"], "symbol": "+", "symbol2": "->"},
        task_labels=["German", "French", "Spanish"],
    ),
}


@dataclass
class SuperpositionPrompt:
    family: str
    prompt: str  # examples_str + query_str (raw completion prompt, no chat scaffolding)
    examples: list[str]  # rendered example lines, in ``prompt_format`` style
    task_indices: list[int]  # which task generated each example line (the realized mixture)
    query_str: str
    output_choices: list[str]
    task_labels: list[str]  # aligned to output_choices
    order: str = "random"
    prompt_format: str = "raw"


# Q:/A: formatting, mirroring ``word_pairs_to_prompt_data`` in
# ``existing_works/function_vectors/src/utils/prompt_utils.py``: prefixes carry no trailing
# space, and a space is prepended to each input/output value instead (so e.g. "Q:" + " 45@67").
_QA_PREFIXES = {"input": "Q:", "output": "A:"}
_QA_SEPARATORS = {"input": "\n", "output": "\n\n"}


def _format_qa_example(line: ExampleLine, prefixes: dict, separators: dict, prepend_space: bool) -> str:
    inp = f" {line.input}" if prepend_space else line.input
    out = f" {line.output}" if prepend_space else line.output
    return f"{prefixes['input']}{inp}{separators['input']}{prefixes['output']}{out}{separators['output']}"


def _format_qa_query(query_input: str, prefixes: dict, separators: dict, prepend_space: bool) -> str:
    inp = f" {query_input}" if prepend_space else query_input
    return f"{prefixes['input']}{inp}{separators['input']}{prefixes['output']}"


def build_superposition_prompt(
    family: str,
    num_examples: int = 15,
    distribution: Optional[list[float]] = None,
    seed: Optional[int] = None,
    order: str = "random",
    prompt_format: str = "raw",
    qa_prefixes: dict = _QA_PREFIXES,
    qa_separators: dict = _QA_SEPARATORS,
    qa_prepend_space: bool = True,
) -> SuperpositionPrompt:
    """Build one interleaved-mixture few-shot prompt + query for ``family``.

    ``distribution`` is the mixture weight per task index (default: uniform; ``"random"`` order only).
    ``order``:
      - ``"random"`` — the paper's default. The realized mixture (``distribution`` weighted) is
        shuffled, so which task governs the next line is unpredictable.
      - ``"cyclic"`` — deterministic round-robin task0, task1, task2, ... repeating. Not in the
        original repo. Makes the identity of the next task positionally PREDICTABLE, as a contrast
        condition: does GoalScope's read-out get sharper when the goal is inferable from position
        alone, vs. genuinely ambiguous under ``"random"``?

    ``prompt_format``:
      - ``"raw"`` (default) — the paper's inline-arrow format, e.g. ``"45@67=112"`` / ``"France->Paris"``.
      - ``"qa"`` — function-vector-style ``Q:``/``A:`` format (see ``word_pairs_to_prompt_data`` in
        ``existing_works/function_vectors/src/utils/prompt_utils.py``): each example line is
        ``"Q: {input}\\nA: {output}\\n\\n"`` and the query is ``"Q: {input}\\nA:"`` with no trailing
        value — the generation/scoring point. ``output_choices`` get a leading space to match
        (``qa_prepend_space``), consistent with the leading space on every example's output.
    """
    fam = TASK_FAMILIES[family]
    if seed is not None:
        random.seed(seed)
    n_tasks = len(fam.line_funcs)

    if order == "random":
        distribution = distribution or [1.0 / n_tasks] * n_tasks
        counts = [int(round(w * num_examples)) for w in distribution]
        counts[-1] += num_examples - sum(counts)  # fix rounding drift onto the last task
        task_indices = [i for i, c in enumerate(counts) for _ in range(c)]
        random.shuffle(task_indices)
    elif order == "cyclic":
        task_indices = [i % n_tasks for i in range(num_examples)]
    else:
        raise ValueError(f"unknown order: {order!r} (expected 'random' or 'cyclic')")

    examples: list[ExampleLine] = []
    for idx in task_indices:
        examples.append(fam.line_funcs[idx](examples, **fam.line_kwargs[idx]))

    query_input, query_raw, output_choices = fam.query_func(examples, **fam.query_kwargs)

    if prompt_format == "raw":
        examples_str = [e.raw for e in examples]
        query_str = query_raw
        prompt = "\n".join(examples_str) + "\n" + query_str
    elif prompt_format == "qa":
        examples_str = [_format_qa_example(e, qa_prefixes, qa_separators, qa_prepend_space) for e in examples]
        query_str = _format_qa_query(query_input, qa_prefixes, qa_separators, qa_prepend_space)
        prompt = "".join(examples_str) + query_str
        if qa_prepend_space:
            output_choices = [f" {c}" for c in output_choices]
    else:
        raise ValueError(f"unknown prompt_format: {prompt_format!r} (expected 'raw' or 'qa')")

    return SuperpositionPrompt(
        family=family,
        prompt=prompt,
        examples=examples_str,
        task_indices=task_indices,
        query_str=query_str,
        output_choices=output_choices,
        task_labels=list(fam.task_labels),
        order=order,
        prompt_format=prompt_format,
    )


@torch.no_grad()
def score_choices(mt: ModelandTokenizer, prompt: str, output_choices: list[str]) -> list[float]:
    """Teacher-forced P(choice | prompt) for each candidate completion (product over its tokens).

    Mirrors ``get_probs()`` in the original repo's ``run_experiments.py`` — this is the
    "superposition" evidence: multiple candidates getting non-trivial mass at once.
    """
    prompt_inputs = prepare_input(prompts=prompt, tokenizer=mt)
    n_prompt = prompt_inputs["input_ids"].shape[-1]

    probs_out = []
    for choice in output_choices:
        choice_inputs = prepare_input(prompts=prompt + choice, tokenizer=mt)
        ids = choice_inputs["input_ids"]
        logits = mt._model(**dict(choice_inputs)).logits.squeeze(0)[n_prompt - 1 : -1]
        target = ids.squeeze(0)[n_prompt:].unsqueeze(-1)
        probs = F.softmax(logits.float(), dim=-1)
        tok_probs = torch.gather(probs, dim=-1, index=target).squeeze(-1)
        probs_out.append(tok_probs.prod().item())
    return probs_out
