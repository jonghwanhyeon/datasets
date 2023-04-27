import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional

import numpy as np
import pandas as pd
import parsekit as pk
import soundfile as sf
from joblib import Parallel, delayed
from joblib_progress import joblib_progress


class Range(NamedTuple):
    low: float
    high: float


class NumberSpacer:
    def __init__(self):
        self._ordinal_number_parser = self._build_ordinal_number_parser()
        self._cardinal_number_parser = self._build_cardinal_number_parser()

    def revise(self, text: str) -> str:
        revised = self._ordinal_number_parser.transform(text)
        revised = self._cardinal_number_parser.transform(revised)
        return re.sub(r"\s{2,}", " ", revised).strip()

    def _add_space(self, items: list[str]):
        return " ".join(["", *items])

    def _build_ordinal_number_parser(self) -> pk.Parser:
        digit = pk.regex(r"[공일이삼사오육칠팔구]")

        multiplier_1e1 = pk.combine(pk.optional(digit) + pk.literal("십"))
        multiplier_1e2 = pk.combine(pk.optional(digit) + pk.literal("백"))
        multiplier_1e3 = pk.combine(pk.optional(digit) + pk.literal("천"))
        multiplier = pk.optional_sequence(multiplier_1e3, multiplier_1e2, multiplier_1e1, at_least=1)

        number_1e0 = pk.flatten(pk.optional(multiplier) + pk.optional(digit))
        number_1e4 = pk.flatten(pk.optional(multiplier) + pk.combine(pk.optional(digit) + pk.literal("만")))
        number_1e8 = pk.flatten(multiplier + pk.combine(pk.optional(digit) + pk.literal("억")))
        number_1e12 = pk.flatten(multiplier + pk.combine(pk.optional(digit) + pk.literal("조")))
        number_1e16 = pk.flatten(multiplier + pk.combine(pk.optional(digit) + pk.literal("경")))
        number_1e20 = pk.flatten(multiplier + pk.combine(pk.optional(digit) + pk.literal("해")))

        number = pk.filter(
            pk.flatten(
                pk.optional_sequence(
                    number_1e20, number_1e16, number_1e12, number_1e8, number_1e4, number_1e0, at_least=1
                )
            )
        )
        return number.map(self._add_space)

    def _build_cardinal_number_parser(self) -> pk.Parser:
        number_1e0 = pk.regex(r"하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉")
        number_1e1 = pk.regex(r"열|스물|서른|마흔|쉰|예순|일흔|여든|아흔")

        number = pk.filter(pk.optional_sequence(number_1e1, number_1e0, at_least=1))
        return number.map(self._add_space)


def train_test_split_by_speaker(df: pd.DataFrame, fraction: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_by_speaker_id = {speaker_id: df for speaker_id, df in df.groupby("speaker_id")}

    speaker_ids = list(df_by_speaker_id.keys())
    random.shuffle(speaker_ids)

    pivot = int(len(df) * (1 - fraction))
    number_of_rows = 0
    for index, speaker_id in enumerate(speaker_ids):
        if number_of_rows > pivot:
            break
        number_of_rows += len(df_by_speaker_id[speaker_id])

    return (
        pd.concat([df_by_speaker_id[speaker_id] for speaker_id in speaker_ids[index:]], axis=0),
        pd.concat([df_by_speaker_id[speaker_id] for speaker_id in speaker_ids[:index]], axis=0),
    )


def revise_manifest(
    df: pd.DataFrame, revisions_path: Path, train_split_only: bool = False, jobs: int = 1
) -> pd.DataFrame:
    revisions = json.loads(revisions_path.read_text())

    if train_split_only:
        train_ids = set(df[df["split"] == "train"]["id"])
        revision_ids = set(revisions.keys())
        for id in revision_ids - train_ids:
            del revisions[id]

    def work(df_i: pd.DataFrame) -> pd.DataFrame:
        for id, revision in revisions.items():
            if revision["action"] == "replace":
                df_i.loc[df_i["id"] == id, "text"] = revision["text"]
            elif revision["action"] == "drop":
                df_i = df_i[df_i["id"] != id]
        return df_i

    return parallel_map(work, df, jobs=jobs, description="Revising manifest...")


def flatten(iterable: Iterable[Any]) -> Any:
    for item in iterable:
        if item is None:
            continue

        if isinstance(item, list):
            yield from item
        else:
            yield item


def readlines(path: Path, encoding: Optional[str] = None) -> Iterable[str]:
    with open(path, "r", encoding=encoding) as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                yield line


def only_digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def has_punctuation(text: str):
    # Dot and comma are an exception due to numbers
    return re.search(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ .,]", text) is not None


def has_digit(text: str):
    return re.search(r"[0-9]", text) is not None


def is_abbreviation(text: str):
    return re.search(r"^[A-Z]+$", text) is not None


def duration_of(audio_path: Path) -> float:
    info = sf.info(audio_path)
    if info.frames == sys.maxsize:
        raise ValueError(f"invalid audio file: {audio_path}")

    return info.frames / info.samplerate


def sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def nested_sub(pattern: str, replacement: str, text: str, flags: int = 0) -> str:
    previous = None
    current = text

    while previous != current:
        previous = current
        current = re.sub(pattern, replacement, current, flags=flags)

    return current


def tukey_fence(series: pd.Series, k: float = 1.5) -> Range:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return Range(q1 - (k * iqr), q3 + (k * iqr))


def interval(series: pd.Series, fraction: float) -> Range:
    return Range(series.quantile((1 - fraction) / 2), series.quantile((1 + fraction) / 2))


def between(series: pd.Series, range: Range) -> pd.Series:
    return (range.low < series) & (series < range.high)


def parallel_map(
    func: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame, jobs: int, description: Optional[str] = None
) -> pd.DataFrame:
    list_of_df = np.array_split(df, jobs)
    with joblib_progress(description=description, total=jobs):
        return pd.concat(Parallel(n_jobs=jobs)(delayed(func)(df_i) for df_i in list_of_df))
