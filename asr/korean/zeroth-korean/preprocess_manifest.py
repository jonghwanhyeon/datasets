import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import simple_parsing
import utils
from rich.console import Console
from simple_parsing import list_field


class OutlierDetector(Enum):
    TUKEY_FENCE = "tukey_fence"
    INTERVAL = "interval"


@dataclass
class Config:
    seed: int = 4909
    jobs: int = 1

    normalize_transcription: bool = True
    strip_annotations: bool = True

    original_text_should_contain: list[str] = list_field(
        # Hangul syllable
        r"[가-힣]",
    )
    original_text_should_not_contain: list[str] = list_field(
        # Hangul jamo
        r"[ㄱ-ㅎㅏ-ㅣ]",
    )

    text_should_contain: list[str] = list_field(
        # Hangul syllable
        r"[가-힣]",
    )
    text_should_not_contain: list[str] = list_field()

    remove_duplicates: bool = True

    exclude_character_rate_outliers: bool = True
    character_rate_outlier_detector: OutlierDetector = OutlierDetector.TUKEY_FENCE
    outlier_detector_tukey_fence_k: float = 1.5
    outlier_detector_fraction: float = 0.975

    min_duration: Optional[float] = None
    max_duration: Optional[float] = None


console = Console()
config: Config = None


def normalize_transcription(text: str) -> str:
    return text


def strip_annotations(text: str) -> str:
    # Convert English letters to uppercase
    stripped = text.upper()

    return stripped.strip()


def preprocess_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(df.columns.get_loc("text") + 1, "original_text", df["text"])

    def normalize_manifest(df_i: pd.DataFrame) -> pd.DataFrame:
        if config.normalize_transcription:
            df_i["text"] = df_i["text"].map(normalize_transcription)

        if config.strip_annotations:
            df_i["text"] = df_i["text"].map(strip_annotations)

        return df_i

    def filter_manifest(df_i: pd.DataFrame) -> pd.DataFrame:
        for pattern in config.original_text_should_contain:
            df_i = df_i[df_i["original_text"].str.contains(pattern)]

        for pattern in config.original_text_should_not_contain:
            df_i = df_i[~df_i["original_text"].str.contains(pattern)]

        for pattern in config.text_should_contain:
            df_i = df_i[df_i["text"].str.contains(pattern)]

        for pattern in config.text_should_not_contain:
            df_i = df_i[~df_i["text"].str.contains(pattern)]

        return df_i

    df = utils.parallel_map(normalize_manifest, df, jobs=config.jobs, description="Normalizing manifest...")

    df_train = df[df["split"] == "train"]
    df_dev = df[df["split"] == "dev"]
    df_test = df[df["split"] == "test"]

    df_train = utils.parallel_map(filter_manifest, df_train, jobs=config.jobs, description="Filtering manifest...")

    if config.remove_duplicates:
        df_train = df_train.drop_duplicates(subset="sha1")

    if config.exclude_character_rate_outliers:
        character_rate = df_train["text"].str.len() / df_train["duration"]

        if config.character_rate_outlier_detector == OutlierDetector.TUKEY_FENCE:
            range = utils.tukey_fence(character_rate, k=config.outlier_detector_tukey_fence_k)
        elif config.character_rate_outlier_detector == OutlierDetector.INTERVAL:
            range = utils.interval(character_rate, fraction=config.outlier_detector_fraction)

        df_outlier = df_train[~utils.between(character_rate, range)]
        df_train = df_train.drop(df_outlier.index)

    if config.min_duration is not None:
        df_matched = df_train[df_train["duration"] < config.min_duration]
        df_train = df_train.drop(df_matched.index)

    if config.max_duration is not None:
        df_matched = df_train[df_train["duration"] > config.max_duration]
        df_train = df_train.drop(df_matched.index)

    df = pd.concat([df_train, df_dev, df_test])
    return df.sort_values(["split", "id"]).reset_index(drop=True)


if __name__ == "__main__":
    config = simple_parsing.parse(Config)
    console.log(config)

    random.seed(config.seed)
    np.random.seed(config.seed)

    console.log("[bold green]Loading manifest...")
    dataset_path = Path(__file__).parent
    manifest_path = dataset_path / "manifest.parquet"
    df = pd.read_parquet(manifest_path)

    console.log("[bold green]Preprocessing manifest...")
    df = preprocess_manifest(df)

    console.log("[bold green]Saving clean manifest...")
    clean_manifest_path = dataset_path / "manifest-clean.parquet"
    df.to_parquet(clean_manifest_path)
