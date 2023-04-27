import json
import random
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

gender_table = {
    "남": "male",
    "여": "female",
}

background_table = {
    "배경 음악": "music",
    "잡음": "noise",
    "": None,
}


@dataclass
class Config:
    revisions_path: Path = Path("revisions.json")
    revise_train_split_only: bool = False
    audio_format: str = "wav"
    jobs: int = 1
    seed: int = 4909


console = Console()
config: Config = None


def split_of(audio_path: Path) -> Path:
    if "1.Training" in audio_path.parts:
        return "train"
    elif "2.Validation" in audio_path.parts:
        return "test"
    else:
        raise ValueError(f"Invalid path: {audio_path}")


def extract_speaker_by_id(metadata: Metadatum):
    return {
        speaker["id"]: {
            "id": speaker["id"],
            "name": speaker["name"],
            "age": utils.only_digits(speaker["age"]),
            "occupation": speaker["occupation"],
            "role": speaker["role"],
            "gender": gender_table[speaker["sex"]],
        }
        for speaker in metadata["speaker"]
    }


def segment_path_of(metadata_path: Path, index: int) -> Path:
    segments_path = metadata_path.parent / metadata_path.stem
    segments_path = Path(str(segments_path).replace("라벨링", "원천"))
    return segments_path / f"{index}.{config.audio_format}"


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    metadata_paths = [*files_path.rglob("*.json")]

    def work(metadata_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        metadata = json.loads(metadata_path.read_text())
        speaker_by_id = extract_speaker_by_id(metadata)

        list_of_metadatum = []
        for index, utterance in enumerate(metadata["utterance"]):
            speaker = speaker_by_id.get(
                utterance["speaker_id"],
                {
                    "id": None,
                    "name": None,
                    "age": None,
                    "occupation": None,
                    "role": None,
                    "gender": None,
                },
            )

            list_of_metadatum.append(
                {
                    "split": split_of(metadata_path),
                    "id": utterance["id"],
                    "path": segment_path_of(metadata_path, index),
                    "text": utterance["form"],
                    "background": background_table[utterance["environment"]],
                    "speaker_id": speaker["id"],
                    "speaker_name": speaker["name"],
                    "speaker_age": speaker["age"],
                    "speaker_occupation": speaker["occupation"],
                    "speaker_role": speaker["role"],
                    "speaker_gender": speaker["gender"],
                }
            )

        return list_of_metadatum

    with joblib_progress("Loading metadata...", total=len(metadata_paths)):
        metadata = Parallel(n_jobs=config.jobs)(delayed(work)(path) for path in metadata_paths)

    return utils.flatten(metadata)


def scan_dataset(dataset_path: Path) -> Iterable[Sample]:
    files_path = dataset_path / "files"
    metadata = [*load_metadata(files_path)]

    def work(metadatum: Metadatum) -> Optional[Union[list[Sample], Sample]]:
        if not metadatum["path"].exists():
            return None

        try:
            return {
                **metadatum,
                "path": str(metadatum["path"].relative_to(dataset_path)),
                "duration": utils.duration_of(metadatum["path"]),
                "sha1": utils.sha1(metadatum["path"]),
            }
        except Exception as exception:
            print(f"An error ocurred: {metadatum['path']} - {exception}")
            return None

    with joblib_progress("Processing samples...", total=len(metadata)):
        samples = Parallel(n_jobs=config.jobs)(delayed(work)(metadatum) for metadatum in metadata)

    return utils.flatten(samples)


def build_manifest(dataset_path: Path) -> pd.DataFrame:
    df = pd.DataFrame.from_records(scan_dataset(dataset_path))
    df = df[
        [
            "split",
            "id",
            "path",
            "text",
            "duration",
            "sha1",
            "background",
            "speaker_id",
            "speaker_name",
            "speaker_age",
            "speaker_role",
            "speaker_gender",
        ]
    ]

    revisions_path = current_path / config.revisions_path
    if revisions_path.exists():
        df = utils.revise_manifest(
            df, revisions_path, train_split_only=config.revise_train_split_only, jobs=config.jobs
        )

    df["split"] = pd.Categorical(df["split"], categories=["train", "dev", "test"], ordered=True)
    return df.sort_values(["split", "id"]).reset_index(drop=True)


if __name__ == "__main__":
    config = simple_parsing.parse(Config)
    current_path = Path.cwd()

    random.seed(config.seed)
    np.random.seed(config.seed)

    console.log("[bold green]Building manifest...")
    df = build_manifest(current_path)

    console.log("[bold green]Saving manifest...")
    manifest_path = current_path / "manifest.parquet"
    df.to_parquet(manifest_path)
