import hashlib
import json
import logging
import os

from anthropic import Anthropic

from src.utils.env_utils import CLAUDE_CACHE_DIR, load_env_var

logger = logging.getLogger(__name__)


def ask_claude(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    use_cache: bool = True,
    model_name: str = "claude-sonnet-5",
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
    if use_cache:
        cache_file = os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json")
        if os.path.exists(cache_file):
            logger.debug(f"found cached claude response for {hash_val} - loading")
            with open(cache_file, "r") as f:
                json_data = json.load(f)
                return json_data["response"]
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
    response = client.messages.create(
        **request_kwargs
    )
    response = response.content[0].text

    with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "w") as f:
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

    return response


ASK_ORACLE_MODEL = {
    # "gpt": ask_gpt4,
    "claude": ask_claude
}
