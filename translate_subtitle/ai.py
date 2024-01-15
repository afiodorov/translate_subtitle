import json
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from translate_subtitle.formatting import to_int_keys, to_str_keys


def get_completion(context: dict, text: dict, *, _cache_file: str = ""):
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
                "content": f"Conversational context to help you translate better: {json.dumps(context)}",
            }
        )

    msgs.append({"role": "user", "content": json.dumps(text)})

    chat_completion = client.chat.completions.create(
        messages=cast(list[ChatCompletionMessageParam], msgs),
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        seed=0,
    )

    return chat_completion.choices[0].message.content


def fix_completion(
    original: dict[int, str], translations: dict[int, str], *, _cache_file: str = ""
) -> str:
    _ = _cache_file
    o, t = to_str_keys(original), to_str_keys(translations)
    client = OpenAI()

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "patternProperties": {"^[0-9]+$": {"type": "string"}},
        "additionalProperties": False,
    }

    prompt = (
        f"You synchronise translated subtitles."
        f"Make sure there's a match between original & translated text so that the dialogue "
        f"is in sync."
        f"For example if you see 560: -Hello 561: -Mom in the original, but 560: -Привет Мама"
        f" in the translation, you can just split the sentence so that it is 560: Привет 561: Мама"
        f" in the translation."
        f"You need to make sure corrected output follows the following json schema. "
        f"Schema: {json.dumps(schema)}. Use utf-8 in the output. "
        f"Fix this error: output json doesn't have same keys as input json. "
        f"The output json should have following keys: {json.dumps(list(o.keys()))}. "
        f"just like the original. So make sure the output has {len(o.keys())} items. "
        f"Another program will use your output so make sure it's correct. "
        f"Here is the orignal text: {json.dumps(o)}. "
    )

    msgs = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    msgs.append({"role": "user", "content": json.dumps(t)})

    chat_completion = client.chat.completions.create(
        messages=cast(list[ChatCompletionMessageParam], msgs),
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        seed=0,
    )
    resp = str(chat_completion.choices[0].message.content)
    parsed = json.loads(resp)
    return json.dumps(to_int_keys(parsed), indent=4, ensure_ascii=False)
