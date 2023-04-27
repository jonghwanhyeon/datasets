import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

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
    "m": "male",
    "f": "female",
}


def split_of(path: Path) -> Path:
    filename = str(path)
    if "train" in filename:
        return "train"
    elif "test" in filename:
        return "test"
    else:
        raise ValueError(f"Invalid path: {path}")


def load_speakers(audio_info_path: Path) -> dict[str, dict[str, str]]:
    with open(audio_info_path, "r") as input_file:
        reader = csv.DictReader(input_file, delimiter="|")
        return {
            row["SPEAKERID"]: {
                "name": row["NAME"].strip(),
                "gender": gender_table[row["SEX"].strip()],
            }
            for row in reader
        }


def load_texts(metadatum_path: Path) -> dict[str, str]:
    text_by_id = {}

    for line in utils.readlines(metadatum_path):
        id, text = line.split(" ", maxsplit=1)
        text_by_id[id] = text

    return text_by_id


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    speaker_by_id = load_speakers(files_path / "AUDIO_INFO")

    metadatum_paths = list(files_path.glob("*/*/*/*.txt"))

    def work(metadatum_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        speaker_path = metadatum_path.parent
        speaker_id = speaker_path.stem

        list_of_metadatum = []
        for text_id, text in load_texts(metadatum_path).items():
            list_of_metadatum.append(
                {
                    "split": split_of(metadatum_path),
                    "id": text_id,
                    "path": speaker_path / f"{text_id}.{config.audio_format}",
                    "text": text.strip(),
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_by_id[speaker_id]["name"],
                    "speaker_gender": speaker_by_id[speaker_id]["gender"],
                }
            )

        return list_of_metadatum

    with joblib_progress("Loading metadata...", total=len(metadatum_paths)):
        metadata = Parallel(n_jobs=config.jobs)(delayed(work)(path) for path in metadatum_paths)

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
            "speaker_id",
            "speaker_name",
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
