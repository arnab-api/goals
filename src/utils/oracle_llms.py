import hashlib
import json
import logging
import os

from anthropic import Anthropic

from src.utils.env_utils import CLAUDE_CACHE_DIR, load_env_var

logger = logging.getLogger(__name__)


def ask_claude(
    prompt: str,
    max_tokens: int = 6000,
    temperature: float = 0.6,
    use_cache: bool = True,
) -> str:
    ##################################################
    client = Anthropic(
        api_key=load_env_var("CLAUDE_KEY"),
    )
    MODEL_NAME = "claude-sonnet-4-6"
    ##################################################
    hash_val = hashlib.md5(
        f"{prompt}__{temperature=}__{max_tokens=}".encode()
    ).hexdigest()
    if use_cache:
        cache_file = os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json")
        if os.path.exists(cache_file):
            logger.debug(f"found cached claude response for {hash_val} - loading")
            with open(cache_file, "r") as f:
                json_data = json.load(f)
                return json_data["response"]

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You are a helpful assistant.",
        messages=[
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
    )
    response = response.content[0].text

    with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "temperature": temperature,
            },
            f,
        )

    return response


ASK_ORACLE_MODEL = {
    # "gpt": ask_gpt4,
    "claude": ask_claude
}
