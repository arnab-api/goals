import hashlib
import json
import logging
import os
import time

from anthropic import Anthropic

import threading

from src.utils.env_utils import CLAUDE_CACHE_DIR, GPT_CACHE_DIR, load_env_var

logger = logging.getLogger(__name__)

MAX_TRIES = 5  # request attempts before giving up


def ask_claude(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    use_cache: bool = True,
    model_name: str = "claude-sonnet-4-6",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    ##################################################
    client = Anthropic(
        api_key=load_env_var("CLAUDE_KEY"),
    )
    ##################################################
    hash_val = hashlib.md5(
        f"{system_prompt}__{prompt}__{max_tokens=}".encode()
    ).hexdigest()
    cache_file = os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json")
    if use_cache and os.path.exists(cache_file):
        logger.debug(f"found cached claude response for {hash_val} - loading")
        try:
            with open(cache_file, "r") as f:
                return json.load(f)["response"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            # empty/partial/corrupt cache file (e.g. crash mid-write, or a
            # concurrent writer) — ignore it and re-request from the API.
            logger.warning(
                f"corrupt cache {cache_file} ({e}); ignoring and re-requesting"
            )

    request_kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }
    if "-5" not in model_name:
        request_kwargs["temperature"] = temperature
    else:
        logger.warning(
            f"temperature parameter is deprecated for {model_name} - ignoring"
        )

    response = None
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            msg = client.messages.create(**request_kwargs)
            # content may include non-text blocks (e.g. ThinkingBlock when extended
            # thinking is on, or tool_use); take the text block(s), not content[0].
            texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if not texts:
                kinds = [getattr(b, "type", "?") for b in msg.content]
                raise ValueError(
                    f"no text block in response (blocks={kinds}, stop_reason={msg.stop_reason}); "
                    "if thinking is enabled, raise max_tokens or disable thinking"
                )
            response = "".join(texts)
            break
        except Exception as e:  # network / API / malformed-response errors
            last_err = e
            logger.warning(
                f"claude request failed (attempt {attempt}/{MAX_TRIES}): {e}"
            )
            if attempt < MAX_TRIES:
                time.sleep(min(2 ** (attempt - 1), 30))  # 1,2,4,8s backoff
    if response is None:
        raise ConnectionError(
            f"claude request failed after {MAX_TRIES} attempts: {last_err}"
        ) from last_err

    # atomic write so a crash / concurrent writer never leaves a partial JSON
    # (which is what made the cache read above raise JSONDecodeError).
    tmp_file = f"{cache_file}.{os.getpid()}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": model_name,
                "hash": hash_val,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
            },
            f,
        )
    os.replace(tmp_file, cache_file)

    return response




_openai_client = None


def ask_gpt(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    use_cache: bool = True,
    model_name: str = "gpt-5",
    system_prompt: str = "You are a helpful assistant.",
    reasoning_effort: str | None = None,
) -> str:
    """OpenAI counterpart of :func:`ask_claude` (same cache / retry / atomic-write contract), cached in
    ``GPT_CACHE_DIR``. Unlike ``ask_claude`` the cache key includes ``model_name`` (and
    ``reasoning_effort``), so two GPT models never share verdicts. ``openai`` is imported lazily.
    GPT-5 / o-series reasoning models take ``max_completion_tokens`` (which also covers the hidden
    reasoning tokens, so give them headroom) and reject a custom ``temperature``."""
    global _openai_client
    from openai import OpenAI  # lazy: ask_claude stays usable without the openai package

    if _openai_client is None:
        _openai_client = OpenAI(api_key=load_env_var("OPENAI_KEY"))
    hash_val = hashlib.md5(
        f"{model_name}__{reasoning_effort}__{system_prompt}__{prompt}__{max_tokens=}".encode()
    ).hexdigest()
    cache_file = os.path.join(GPT_CACHE_DIR, f"{hash_val}.json")
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)["response"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"corrupt cache {cache_file} ({e}); ignoring and re-requesting")

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    reasoning_model = ("-5" in model_name) or model_name.startswith(("o1", "o3", "o4"))
    if reasoning_model:
        request_kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
    else:
        request_kwargs["max_tokens"] = max_tokens
        request_kwargs["temperature"] = temperature

    response = None
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            completion = _openai_client.chat.completions.create(**request_kwargs)
            response = completion.choices[0].message.content
            if not response:
                fr = completion.choices[0].finish_reason
                raise ValueError(
                    f"empty content (finish_reason={fr}); for a reasoning model raise max_tokens"
                )
            break
        except Exception as e:  # network / API / malformed-response errors
            last_err = e
            logger.warning(f"gpt request failed (attempt {attempt}/{MAX_TRIES}): {e}")
            if attempt < MAX_TRIES:
                time.sleep(min(2 ** (attempt - 1), 30))
    if response is None:
        raise ConnectionError(f"gpt request failed after {MAX_TRIES} attempts: {last_err}") from last_err

    # atomic write; tmp name carries pid AND thread id so concurrent threads never share a tmp file
    tmp_file = f"{cache_file}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": model_name,
                "reasoning_effort": reasoning_effort,
                "hash": hash_val,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
            },
            f,
        )
    os.replace(tmp_file, cache_file)
    return response
