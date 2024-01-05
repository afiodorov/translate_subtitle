import argparse
import json
from dataclasses import dataclass
from itertools import batched
from math import ceil
from pathlib import Path
from typing import cast, Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from translate_subtitle.cache import cache
from translate_subtitle.extract import extract_subtitles


@cache()
def get_completion(context: dict, text: dict):
    client = OpenAI()

    prompt = (
        "You translate movies from informal Russian into informal Portuguese. "
        "You reply in json."
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


@dataclass()
class Text:
    number: int
    time: str
    text: str


def is_valid_file(parser: argparse.ArgumentParser, arg) -> Path:
    path = Path(arg)
    if not path.is_file():
        parser.error(f"The file {arg} does not exist.")
    else:
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a subtitle")
    parser.add_argument(
        "-i",
        "--input",
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help="Input video file path",
    )

    args = parser.parse_args()
    input_path: Path = args.input
    extracted: Path = input_path.with_suffix(".srt")

    if not extracted.exists():
        extracted = extract_subtitles(input_path, extracted)

    text = extracted.read_text().strip().split("\n\n")

    subtitles = []
    for t in text:
        n, ts, *rest = t.split("\n")
        subtitles.append(Text(number=int(n), time=ts, text="\n".join(rest)))

    context = {}
    responses = []

    for b in tqdm(batched(subtitles, 20), total=ceil(len(subtitles) / 20)):
        text = {t.number: t.text for t in b}
        resp = get_completion(context, text)
        context = text
        responses.append(resp)

    translated: dict[int, Any] = {}
    for r in responses:
        parsed = json.loads(r)
        try:
            translated |= {int(x[0]): x[1] for x in parsed.items()}
        except:
            print(r)
            raise

    with input_path.with_suffix(".pt.srt").open("w") as f:
        for i, translation in enumerate(translated.values()):
            if isinstance(translation, dict):
                if "translated" in translation:
                    translation = translation["translated"]
                elif "text" in translation:
                    translation = translation["text"]
                elif "translation" in translation:
                    translation = translation["translation"]
                elif "pt" in translation:
                    translation = translation["pt"]
                else:
                    raise ValueError(f"unknown dict {translation}")

            if not isinstance(translation, str):
                raise ValueError(f"wrong type: {translation}")

            n = subtitles[i].number
            ts = subtitles[i].time
            print(n, file=f)
            print(ts, file=f)
            print(translation.replace("\r", "").replace("\n", "\r\n") + "\n", file=f)
