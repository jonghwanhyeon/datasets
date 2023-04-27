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
    if path.stem == "train":
        return "train"
    elif path.stem == "dev":
        return "dev"
    elif path.stem.startswith("eval"):
        return "test"
    else:
        raise ValueError(f"Invalid path: {path}")


def load_split_metadata(metadata_path: Path) -> Iterable[Metadatum]:
    lines = list(utils.readlines(metadata_path))

    def work(line: str) -> Metadatum:
        filename, transcript = line.split("::", maxsplit=1)
        audio_path = Path(filename.strip())
        audio_path = audio_path.with_suffix(f".{config.audio_format}")

        if audio_path.is_relative_to("KsponSpeech_eval/"):
            audio_path = audio_path.relative_to("KsponSpeech_eval/")

        return {
            "split": split_of(metadata_path),
            "id": audio_path.stem,
            "path": audio_path,
            "text": transcript.strip(),
        }

    with joblib_progress(f"Loading {metadata_path.name}", total=len(lines)):
        metadata = Parallel(n_jobs=config.jobs)(delayed(work)(line) for line in lines)

    return metadata


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    yield from load_split_metadata(files_path / "train.trn")
    yield from load_split_metadata(files_path / "dev.trn")
    yield from load_split_metadata(files_path / "eval_clean.trn")
    yield from load_split_metadata(files_path / "eval_other.trn")


def scan_dataset(dataset_path: Path) -> Iterable[Sample]:
    files_path = dataset_path / "files"
    metadata = [*load_metadata(files_path)]

    def work(metadatum: Metadatum) -> Optional[Union[list[Sample], Sample]]:
        path = files_path / metadatum["path"]
        if not path.exists():
            return None

        try:
            return {
                **metadatum,
                "path": str(path.relative_to(dataset_path)),
                "duration": utils.duration_of(path),
                "sha1": utils.sha1(path),
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
