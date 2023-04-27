import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import simple_parsing
import soundfile as sf
import utils
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from rich.console import Console

Metadatum = dict[str, Any]
Sample = dict[str, Any]


@dataclass
class Config:
    audio_format: str = "wav"
    remove_source: bool = False
    jobs: int = 1


console = Console()
config: Config = None


def audio_path_of(metadata_path: Path) -> Path:
    audio_path = Path(str(metadata_path).replace("라벨링", "원천"))
    return audio_path.with_suffix(f".{config.audio_format}")


def segment_path_of(metadata_path: Path, index: int) -> Path:
    segments_path = metadata_path.parent / metadata_path.stem
    segments_path = Path(str(segments_path).replace("라벨링", "원천"))
    return segments_path / f"{index}.{config.audio_format}"


def load_metadata(files_path: Path) -> Iterable[tuple[str, list[Metadatum]]]:
    metadata_paths = [*files_path.rglob("*.json")]

    def work(metadata_path: Path) -> tuple[str, list[Metadatum]]:
        metadata = json.loads(metadata_path.read_text())

        list_of_metadatum = []
        for index, utterance in enumerate(metadata["utterance"]):
            list_of_metadatum.append(
                {
                    "id": utterance["id"],
                    "path": segment_path_of(metadata_path, index),
                    "start": float(utterance["start"]),
                    "end": float(utterance["end"]),
                }
            )

        return (audio_path_of(metadata_path), list_of_metadatum)

    with joblib_progress("Loading metadata...", total=len(metadata_paths)):
        metadata = Parallel(n_jobs=config.jobs)(delayed(work)(path) for path in metadata_paths)

    return utils.flatten(metadata)


def split_audio(dataset_path: Path):
    files_path = dataset_path / "files"
    metadata_by_audio_path = dict(load_metadata(files_path))

    def work(audio_path: Path, metadata: list[Metadatum]):
        if not audio_path.exists():
            return

        segments_path = audio_path.parent / audio_path.stem
        segments_path.mkdir(parents=True, exist_ok=True)

        waveform, sample_rate = sf.read(audio_path)

        for metadatum in metadata:
            start_index = int(metadatum["start"] * sample_rate)
            end_index = int(metadatum["end"] * sample_rate)
            segment = waveform[start_index:end_index]
            if segment.size == 0:
                continue

            sf.write(metadatum["path"], segment, sample_rate)

        if config.remove_source:
            audio_path.unlink()

    with joblib_progress(total=len(metadata_by_audio_path)):
        Parallel(n_jobs=config.jobs)(
            delayed(work)(audio_path, metadata) for audio_path, metadata in metadata_by_audio_path.items()
        )


if __name__ == "__main__":
    config = simple_parsing.parse(Config)
    current_path = Path.cwd()

    console.log("[bold green]Splitting audio files...")
    split_audio(current_path)
