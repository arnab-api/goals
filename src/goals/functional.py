import logging

from src.models import ModelandTokenizer

logger = logging.getLogger(__name__)


def apply_chat_template(
    mt: ModelandTokenizer,
    user_query: str,
    incomplete_code: str,
    system_prompt: str = "You are a helpful assistant proficient in programming. When completing code, ensure proper syntax and logical correctness. Make sure to have all codes inside the <code> </code> tags.",
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_query,
        },
        {
            "role": "assistant",
            "content": f"<code>\n{incomplete_code}",
        },
    ]

    prompt = mt.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,  # Don't add a new assistant header
        continue_final_message=True,  # Don't close the assistant turn with <|eot_id|>
        tokenize=False,
    )
    return prompt


def supports_chat_template(mt: ModelandTokenizer) -> bool:
    """
    Heuristic to determine if the model supports chat templates.
    """
    instruct_signatures = [
        "instruct",
        "it",
        "chat",
        "oss",
        "zai-org/glm",
        "next",
        "cwm",
    ]
    return any(
        inst_signature in mt.name.lower() for inst_signature in instruct_signatures
    )
