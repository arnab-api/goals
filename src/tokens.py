import logging
from typing import Any, Literal, Optional

import torch

from src.models import ModelandTokenizer, unwrap_tokenizer
from src.utils.tokenization_utils import set_padding_side
from src.utils.typing import Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


def chat_prompt(tokenizer: ModelandTokenizer | Tokenizer, messages: list[dict], **kwargs) -> str:
    """``messages`` rendered by the tokenizer's chat template, as text. ``kwargs`` go to ``apply_chat_template``
    (e.g. ``add_generation_prompt=False, continue_final_message=True`` to leave a prefilled assistant turn open).

    ``enable_thinking=False`` is passed to the template (Qwen3 reads it), and dropped if the call rejects the keyword.
    A template that refuses a system turn (Gemma's raises "System role not supported") gets the system text at the
    start of the first user turn instead, separated by a blank line; every other template sees ``messages`` as given."""
    tokenizer = unwrap_tokenizer(tokenizer)

    def render(msgs):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, chat_template_kwargs={"enable_thinking": False},
                                                 **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)

    try:
        return render(messages)
    except Exception as e:  # jinja2.TemplateError, raised by the template itself
        if not (len(messages) > 1 and messages[0]["role"] == "system" and messages[1]["role"] == "user"
                and "system" in str(e).lower()):
            raise
        system, user = messages[0], messages[1]
        return render([dict(user, content=f"{system['content']}\n\n{user['content']}"), *messages[2:]])


_TURN_TEXT = "Turn text 417."  # an assistant turn's content, to find what the template writes after it


def end_of_turn_token_id(tokenizer: ModelandTokenizer | Tokenizer) -> Optional[int]:
    """The chat template's end-of-turn token: the first special token after an assistant turn's text, in a
    user / assistant / user conversation rendered by :func:`chat_prompt` (``<|im_end|>`` on ChatML templates:
    Qwen2.5, Olmo-3; ``<|eot_id|>`` on Llama-3; ``<|user|>`` on GLM-4.5, whose turns end where the next one starts;
    ``</s>`` on Mistral). ``None`` when the tokenizer has no chat template, the template refuses the conversation,
    or it writes no special token there."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if not getattr(tokenizer, "chat_template", None):
        return None
    messages = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": _TURN_TEXT},
                {"role": "user", "content": "Thanks."}]
    try:
        text = chat_prompt(tokenizer, messages)
    except Exception:  # jinja2.TemplateError, raised by the template itself
        return None
    if _TURN_TEXT not in text:
        return None
    end = text.index(_TURN_TEXT) + len(_TURN_TEXT)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    special = {i for i, t in tokenizer.added_tokens_decoder.items() if t.special}
    return next((int(i) for i, (a, _) in zip(enc["input_ids"], enc["offset_mapping"]) if a >= end and i in special),
                None)


def prepare_input(
    prompts: str | list[str],
    tokenizer: ModelandTokenizer | Tokenizer,
    device: torch.device = "cpu",
    return_offsets_mapping=False,
    padding_side: Optional[Literal["left", "right"]] = None,
    **kwargs,
) -> TokenizerOutput:
    """Prepare input for the model."""
    if isinstance(tokenizer, ModelandTokenizer):
        # if tokenizer type is ModelandTokenizer, get device and ignore the passed device
        device = tokenizer.device

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts

    padding_side = padding_side or tokenizer.padding_side

    with set_padding_side(tokenizer, padding_side):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=return_offsets_mapping,
            **kwargs,
        )

    inputs = inputs.to(device)
    return inputs


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[ModelandTokenizer | Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    # logger.debug(f"Found substring in string {string.count(substring)} times")

    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # logger.debug(
    #     f"char range: [{char_start}, {char_end}] => `{string[char_start:char_end]}`"
    # )

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping[0]

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # logger.debug(f"{index=} | token range: [{token_char_start}, {token_char_end}]")
        if token_char_start == token_char_end:
            # Skip special tokens # ! Is this the proper way of doing this?
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                # if token_char_start <= char_start and token_char_end > char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                # if token_char_start < char_end and token_char_end > char_start:
                token_end = index
                break

    # print(f"{substring=}, {occurrence=} | {token_start=}, {token_end=}")
    assert token_start is not None, (
        "Are you working with Llama-3? Try passing the ModelandTokenizer object as the tokenizer"
    )
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)
