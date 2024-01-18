import json
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def get_completion(
    context: dict, text: dict, *, _cache_file: str = "", extra: list[dict] | None = None
) -> str:
    _ = _cache_file
    client = OpenAI()

    prompt = (
        "You translate movies from informal Russian into informal Portuguese. "
        "You reply in json that has same number of phrases as the input."
    )

    msgs = [
        {
            "role": "system",
            "content": prompt,
        },
    ]
    if context:
        msgs.append(
            {
                "role": "system",
                "content": f"Conversational context to help you translate better: {json.dumps(context, ensure_ascii=False)}",
            }
        )

    msgs.append({"role": "user", "content": json.dumps(text, ensure_ascii=False)})

    if extra:
        msgs.extend(extra)

    chat_completion = client.chat.completions.create(
        messages=cast(list[ChatCompletionMessageParam], msgs),
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        seed=0,
    )

    return chat_completion.choices[0].message.content or ""


def fix_completion(
    context: dict, text: dict, wrong: dict, *, _cache_file: str = ""
) -> str:
    _ = _cache_file

    return get_completion(
        context,
        text,
        extra=[
            {"role": "system", "content": json.dumps(wrong, ensure_ascii=False)},
            {
                "role": "user",
                "content": f"Great, but fix this error now: returned translation should have keys: {list(text.keys())}",
            },
        ],
    )
