import argparse
import json
from dataclasses import dataclass
from itertools import batched
from math import ceil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from translate_subtitle.ai import fix_completion, get_completion
from translate_subtitle.cache import cache
from translate_subtitle.extract import extract_subtitles
from translate_subtitle.formatting import to_int_keys


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

    temp_dir = Path("/tmp") / input_path.with_suffix("").name
    get_completion = cache(temp_dir)(get_completion)
    fix_completion = cache(temp_dir)(fix_completion)

    if input_path.suffix == ".srt":
        extracted: Path = input_path
    else:
        extracted: Path = input_path.with_suffix(".srt")
        if not extracted.exists():
            extracted = extract_subtitles(input_path, extracted)

    text = extracted.read_text().strip().split("\n\n")

    subtitles = []
    for t in text:
        num, ts, *rest = t.split("\n")
        subtitles.append(Text(number=int(num), time=ts, text="\n".join(rest)))

    context = {}
    original = []
    responses = []

    for i, b in enumerate(
        tqdm(batched(subtitles, 20), total=ceil(len(subtitles) / 20))
    ):
        text = {t.number: t.text for t in b}
        resp = get_completion(context, text, _cache_file=f"batch_{i}.txt")
        context = text
        responses.append(resp)
        original.append(text)

    translated: dict[int, Any] = {}
    for i, (r, o) in enumerate(zip(responses, original)):
        try:
            translations = to_int_keys(json.loads(r))
        except ValueError as e:
            raise ValueError(f"invalid batch {i}") from e

        if not translations.keys() == o.keys():
            fixed = to_int_keys(
                json.loads(
                    fix_completion(o, translations, _cache_file=f"batch_{i}_fixed.txt")
                )
            )
            if fixed.keys() == o.keys():
                translations = fixed
            else:
                raise ValueError(f"not matching batch {i}")

        translated |= translations

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
