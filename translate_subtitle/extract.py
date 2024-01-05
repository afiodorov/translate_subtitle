import subprocess
from pathlib import Path


def extract_subtitles(input_path: Path, output_file_path: Path) -> Path:
    command: list[str] = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-map",
        "0:s:0",
        str(output_file_path),
    ]
    subprocess.run(command, check=True)
    return output_file_path
