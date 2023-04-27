import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


kst = timezone(timedelta(hours=9))

gender_table = {
    "남": "male",
    "여": "female",
}


def split_of(path: Path) -> Path:
    if "Training" in path.parts:
        return "train"
    elif "Validation" in path.parts:
        return "test"
    else:
        raise ValueError(f"Invalid path: {path}")


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    group_paths = list(files_path.glob("*/*"))

    def work(group_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        list_of_metadatum = []
        for metadatum_path in group_path.rglob("*.json"):
            item = json.loads(metadatum_path.read_text())
            gender = gender_table[item["녹음자정보"]["gender"]]

            list_of_metadatum.append(
                {
                    "split": split_of(metadatum_path),
                    "id": metadatum_path.stem,
                    "path": metadatum_path.with_suffix(f".{config.audio_format}"),
                    "text": item["발화정보"]["stt"],
                    "recorded_at": datetime.strptime(item["발화정보"]["recrdDt"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=kst),
                    "script_set": item["발화정보"]["scriptSetNo"],
                    "environment": item["대화정보"]["recrdEnvrn"],
                    "city": item["대화정보"]["cityCode"],
                    "conversation_theme": item["대화정보"]["convrsThema"],
                    "speaker_id": f"{gender}-{item['녹음자정보']['age']}-{item['녹음자정보']['recorderId']}",
                    "raw_speaker_id": item["녹음자정보"]["recorderId"],
                    "speaker_gender": gender,
                    "speaker_age": item["녹음자정보"]["age"],
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
            "recorded_at",
            "script_set",
            "environment",
            "city",
            "conversation_theme",
            "speaker_id",
            "raw_speaker_id",
            "speaker_gender",
            "speaker_age",
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
