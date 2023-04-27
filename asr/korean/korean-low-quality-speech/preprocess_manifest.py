import random
import re
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
    prefer_abbreviation_as_orthographic: bool = True
    revise_number_spacing: bool = True

    original_text_should_contain: list[str] = list_field(
        # Hangul syllable
        r"[가-힣]",
    )
    original_text_should_not_contain: list[str] = list_field(
        # Hangul jamo
        r"[ㄱ-ㅎㅏ-ㅣ]",
        # Deidentification symbols
        r"(?i)[x×]{2,}",
        r"(?i)\([^)]*x[^)]*\)",
        # Speaker annotation
        r"^[MF]\s*\d+",
        # Mumbled
        r"\(\)",
    )

    text_should_contain: list[str] = list_field(
        # Hangul syllable
        r"[가-힣]",
    )
    text_should_not_contain: list[str] = list_field(
        # Ill-formed noise annotations
        # Typos of Korean as English (e.g., sp)
        r"[a-z]",
        # Phonetic transcription cannot contain digits
        r"[0-9]",
        # Unmatched annotations
        r"[()]",
    )

    remove_duplicates: bool = True

    exclude_character_rate_outliers: bool = True
    character_rate_outlier_detector: OutlierDetector = OutlierDetector.TUKEY_FENCE
    outlier_detector_tukey_fence_k: float = 1.5
    outlier_detector_fraction: float = 0.975

    exclude_session_related_to_outlier: bool = True

    min_duration: Optional[float] = None
    max_duration: Optional[float] = None


console = Console()
config: Config = None

number_spacer = utils.NumberSpacer()


def normalize_transcription(text: str) -> str:
    # ((11월)/(십일 월))
    normalized = re.sub(r"\(\(([^()]+)\)/\(([^()]+)\)\)", r"(\1)/(\2)", text)

    # (10000원)/(만 원))
    # Exception: ((11월)/(십일 월))
    normalized = re.sub(r"(?<!\()\(([^()]+)\)/\(([^()]+)\)\)", r"(\1)/(\2)", normalized)

    # )Q&A)/(큐앤에이)
    normalized = re.sub(r"\)([^()]+)\)/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (12월)/(십 이 월(
    normalized = re.sub(r"\(([^()]+)\)/\(([^()]+)\(", r"(\1)/(\2)", normalized)

    # (2)/(투_
    # (12월)/(십 이 월(
    # Drop: (1번)/(한 번0
    # - Due to (30킬로를)/(30키로를)
    normalized = re.sub(r"\(([^()]+)\)/\(([^()_]+)[(_]", r"(\1)/(\2)", normalized)

    # (1시) /(한 시)
    normalized = re.sub(r"\(([^()]+)\)\s+/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (PC))/(피씨)
    normalized = re.sub(r"\(([^()]+)\)\)/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (190000원)(/(십 구만 원)
    normalized = re.sub(r"\(([^()]+)\)\(/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (1시)/ (한 시)
    normalized = re.sub(r"\(([^()]+)\)/\s+\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (56.6%)/((오십 육 점 육 퍼센트)
    normalized = re.sub(r"\(([^()]+)\)/\(\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (430000원)/)(사십 삼만 원)
    normalized = re.sub(r"\(([^()]+)\)/\)\(([^()]+)\)", r"(\1)/(\2)", normalized)

    #  (1시/)/(한 시)
    normalized = re.sub(r"\(([^()]+)/\)/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (1시(/(한 시)
    normalized = re.sub(r"\(([^()]+)\(/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (2만4천원)/)이만 사천원)
    normalized = re.sub(r"\(([^()]+)\)/\)([^()]+)\)", r"(\1)/(\2)", normalized)

    # (10만/(십 만)
    normalized = re.sub(r"\(([^()/]+)/\(([^()]+)\)", r"(\1)/(\2)", normalized)

    # (1회)/일 회)
    normalized = re.sub(r"\(([^()]+)\)/([^()]+)\)", r"(\1)/(\2)", normalized)

    # (6조)(육 조)
    normalized = re.sub(r"\(([^()]+)\)\(([^()]+)\)", r"(\1)/(\2)", normalized)

    return normalized


def strip_annotations(text: str) -> str:
    def replacement(match: re.Match) -> str:
        orthographic, phonetic = match.group(1), match.group(2)

        if config.prefer_abbreviation_as_orthographic:
            if utils.is_abbreviation(orthographic):
                return orthographic

        if utils.has_digit(orthographic):
            phonetic = number_spacer.revise(phonetic)

        return phonetic

    # Use phonetic transcription except for abbreviation
    stripped = re.sub(r"\(([^()]+)\)/\(([^()]+)\)", replacement, text)

    # Remove annotations for noise
    stripped = utils.nested_sub(r"(?<!/)[blonuilp]\s*/|/\s*[blonuilp](?!/)", r"", stripped, flags=re.IGNORECASE)
    stripped = stripped.replace("(())", "")

    # Replace annotations for filler, ambiguous, repetitive words as whitespace
    stripped = re.sub(r"[/*+]", r" ", stripped)

    # Replace punctuation marks as whitespace
    stripped = re.sub(r"[!?]", r" ", stripped)
    ## Exception: 10.21 / 1,000
    stripped = re.sub(r"(?<!\d)[,.]|[,.](?!\d)", " ", stripped)
    ## Then, remove comma
    stripped = stripped.replace(",", "")

    # Replace hyphen-like as whitespace
    stripped = re.sub(r"[-_~]", r" ", stripped)

    # Replace special symbols as whitespace
    stripped = re.sub(r"[`]", r" ", stripped)

    # Remove special symbols
    stripped = re.sub(r"[\"'{}\]=\\]", r"", stripped)

    # Remove annotations for repetitive words (should be last)
    stripped = utils.nested_sub(r"\(([0-9A-Za-z가-힣 ]+)\)", r" \1 ", stripped)

    # Merge consecutive whitespaces
    stripped = re.sub(r"\s{2,}", " ", stripped)

    # Convert English letters to uppercase
    stripped = stripped.upper()

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

        if config.exclude_session_related_to_outlier:
            session_ids = set()
            for _, row in df_outlier.iterrows():
                session_ids.add(row["session_id"])

            df_related = df_train[df_train["session_id"].isin(session_ids)]
            df_outlier = pd.concat([df_outlier, df_related])

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
