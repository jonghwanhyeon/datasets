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


def split_of(path: Path) -> Path:
    # TODO: Return split of given path
    raise NotImplementedError("split_of() is not implemented")


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    # TODO: List metadatum paths
    metadatum_paths = []

    def work(metadatum_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        # TODO: Return metadatum or list of metadatum
        # Type 1:
        #     return metadatum
        # Type 2:
        #     list_of_metadatum = []
        #     list_of_metadatum.append(metadatum)
        #     return list_of_metadatum
        raise NotImplementedError()

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
            # TODO: Add columns
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
