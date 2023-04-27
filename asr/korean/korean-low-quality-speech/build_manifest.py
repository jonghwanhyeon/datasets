import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import simple_parsing
import utils
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from rich.console import Console

Metadatum = dict[str, Any]
Sample = dict[str, Any]


@dataclass
class Config:
    revisions_path: Path = Path("revisions.json")
    revise_train_split_only: bool = False
    audio_format: str = "wav"
    jobs: int = 1
    seed: int = 4909


console = Console()
config: Config = None


gender_table = {
    "남": "male",
    "여": "female",
}


T = TypeVar("T")


def optional(value: str, factory: T = str) -> Optional[T]:
    value = value.strip()
    if value in ("", "None", "알수없음"):
        return None
    return factory(value)


def split_of(audio_path: Path) -> Path:
    if "1.Training" in audio_path.parts:
        return "train"
    elif "2.Validation" in audio_path.parts:
        return "test"
    else:
        raise ValueError(f"Invalid path: {audio_path}")


def id_of(dialog: dict[str, str]) -> str:
    return dialog["audioPath"].replace("/", "-").removesuffix(".wav")


def session_id_of(dialog: dict[str, str]) -> str:
    path = Path(dialog["audioPath"])
    return str(path.parent).replace("/", "-")


def audio_path_of(metadata_path: Path, dialog: dict[str, str]):
    split_path = metadata_path.parent.parent.parent.parent.parent
    audio_path = split_path / "원천데이터" / dialog["audioPath"].strip()
    return audio_path.with_suffix(f".{config.audio_format}")


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    group_paths = list(files_path.glob("*/라벨링데이터/*/*/*"))

    def work(group_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        metadata_path = group_path / f"{group_path.name}.json"
        metadata = json.loads(metadata_path.read_text())["dataSet"]

        category = metadata["typeInfo"]["category"]
        subcategory = metadata["typeInfo"]["subcategory"]

        speaker_by_id = {
            speaker["id"]: {
                "type": optional(speaker["type"]),
                "age": optional(utils.only_digits(speaker["age"]), int),
                "gender": gender_table.get(optional(speaker["gender"])),
                "residence": optional(speaker["residence"]),
                "telephone_network": optional(speaker["telephone_network"]),
            }
            for speaker in metadata["typeInfo"]["speakers"]
        }

        list_of_metadatum = []
        for dialog in metadata["dialogs"]:
            audio_path = audio_path_of(metadata_path, dialog)

            list_of_metadatum.append(
                {
                    "id": id_of(dialog),
                    "split": split_of(audio_path),
                    "path": audio_path,
                    "text": dialog["text"].strip(),
                    "session_id": session_id_of(dialog),
                    "category": category,
                    "subcategory": subcategory,
                    "speaker_type": speaker_by_id[dialog["speaker"]]["type"],
                    "speaker_age": speaker_by_id[dialog["speaker"]]["age"],
                    "speaker_gender": speaker_by_id[dialog["speaker"]]["gender"],
                    "speaker_residence": speaker_by_id[dialog["speaker"]]["residence"],
                    "speaker_telephone_network": speaker_by_id[dialog["speaker"]]["telephone_network"],
                }
            )

        return list_of_metadatum

    with joblib_progress("Loading metadata...", total=len(group_paths)):
        metadata = Parallel(n_jobs=config.jobs)(delayed(work)(path) for path in group_paths)

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
            "session_id",
            "category",
            "subcategory",
            "speaker_type",
            "speaker_age",
            "speaker_gender",
            "speaker_residence",
            "speaker_telephone_network",
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
