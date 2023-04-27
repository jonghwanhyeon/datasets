import random
import re
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
from rich.progress import track
from simple_parsing.helpers import set_field

Metadatum = dict[str, Any]
Sample = dict[str, Any]


@dataclass
class Config:
    exclude_categories: set[str] = set_field("방송")
    revisions_path: Path = Path("revisions.json")
    revise_train_split_only: bool = False
    audio_format: str = "wav"
    jobs: int = 1
    seed: int = 4909


console = Console()
config: Config = None


table = {
    "category": {
        "01": "방송",
        "02": "취미",
        "03": "일상안부",
        "04": "기술",
        "05": "생활",
        "06": "날씨",
        "07": "경제",
        "08": "놀이",
        "09": "쇼핑",
    },
    "subcategory": {
        "방송": {
            "01": "드라마",
            "02": "영화",
            "03": "K-POP",
            "04": "시사교양",
            "05": "예능",
            "06": "연예인",
            "07": "회화",
            "08": "다큐",
            "09": "뉴스",
            "10": "스포츠",
            "11": "만화",
            "12": "여행",
            "13": "건강",
            "14": "역사",
            "15": "교육",
            "99": "기타",
        },
        "취미": {
            "01": "운동",
            "02": "공연",
            "03": "낚시",
            "04": "게임",
            "05": "여행",
            "06": "그림",
            "07": "음악",
            "08": "등산",
            "09": "독서",
            "10": "사진",
            "11": "음식",
            "12": "전시회",
            "13": "자동차",
            "99": "기타",
        },
        "일상안부": {
            "01": "자기소개",
            "02": "거주지정보",
            "03": "이성친구",
            "04": "학교생활",
            "05": "회사생활",
            "06": "기념일",
            "07": "안부인사",
            "08": "코로나",
            "99": "기타",
        },
        "기술": {
            "01": "4차산업",
            "02": "스마트폰",
            "03": "IT동향",
            "04": "인공지능",
            "05": "기술용어",
            "06": "자동차",
            "07": "게임",
            "99": "기타",
        },
        "생활": {
            "01": "형제",
            "02": "가족",
            "03": "생계",
            "04": "농사",
            "05": "밭일",
            "06": "소일거리",
            "07": "직장생활",
            "08": "추억",
            "09": "반려동물",
            "10": "음식",
            "11": "조리",
            "12": "건강",
            "99": "기타",
        },
        "날씨": {
            "01": "계절",
            "02": "황사",
            "03": "미세먼지",
            "04": "악취",
            "05": "온도",
            "06": "장마",
            "07": "폭설",
            "08": "혹서기",
            "09": "혹한기",
            "10": "눈",
            "11": "비",
            "12": "안개",
            "99": "기타",
        },
        "경제": {
            "01": "부동산",
            "02": "주식",
            "03": "경제지표",
            "04": "재테크",
            "99": "기타",
        },
        "놀이": {
            "01": "유치원생활",
            "02": "친구",
            "03": "엄마아빠",
            "04": "장난감",
            "05": "선생님",
            "99": "기타",
        },
        "쇼핑": {
            "01": "의류",
            "02": "전자기기",
            "03": "생활용품",
            "04": "악기",
            "05": "식품",
            "06": "소모품",
            "99": "기타",
        },
    },
    "gender": {
        "M": "male",
        "F": "female",
    },
    "generation": {
        "C": "유아",
        "T": "청소년",
        "A": "성인",
        "S": "노인",
        "Z": "기타",
    },
    "city": {
        "1": "서울경기",
        "2": "강원",
        "3": "충청",
        "4": "경상",
        "5": "전라",
        "6": "제주",
        "9": "기타",
    },
    "dialect": {
        "1": "서울경기",
        "2": "강원",
        "3": "충청",
        "4": "경상",
        "5": "전라",
        "6": "제주",
        "9": "기타",
    },
    "source": {
        "1": "방송",
        "2": "제작",
        "3": "크라우드소싱",
        "9": "기타",
    },
    "sound_quality": {
        "1": "정상",
        "2": "노이즈",
        "3": "잡음",
        "4": "원거리",
    },
}


def split_of(audio_path: Path) -> Path:
    if "1.Training" in audio_path.parts:
        return "train"
    elif "2.Validation" in audio_path.parts:
        return "test"
    else:
        raise ValueError(f"Invalid path: {audio_path}")


def fix_path(path: str) -> str:
    return path.replace("06.경제", "6.경제")


def id_of(audio_path: Path) -> str:
    if "1.Training" in audio_path.parts:
        prefix = "train"
    elif "2.Validation" in audio_path.parts:
        prefix = "test"
    else:
        raise ValueError(f"Invalid path: {audio_path}")

    return f"{prefix}-{audio_path.stem}"


def load_metadata(files_path: Path) -> Iterable[Metadatum]:
    group_paths = list(files_path.glob("*/1.라벨링데이터/*/*"))

    def work(group_path: Path) -> Optional[Union[list[Metadatum], Metadatum]]:
        metadata_path = group_path / f"{group_path.name}_metadata.txt"
        scripts_path = group_path / f"{group_path.name}_scripts.txt"

        metadatum_by_path = {}
        for line in utils.readlines(metadata_path):
            items = re.split(r"[ \t]*\|[ \t]*", line)
            path = fix_path(items[0].lstrip("/"))
            audio_path = (files_path / path).with_suffix(f".{config.audio_format}")

            category = table["category"][items[1]]
            if category in config.exclude_categories:
                continue

            metadatum_by_path[path] = {
                "id": id_of(audio_path),
                "split": split_of(audio_path),
                "path": audio_path,
                "category": category,
                "subcategory": table["subcategory"][category].get(items[2]),
                "speaker_gender": table["gender"].get(items[3]),
                "speaker_generation": table["generation"].get(items[4]),
                "speaker_residence": table["city"].get(items[5]),
                "speaker_dialect": table["dialect"].get(items[6]),
                "source": table["source"].get(items[7]),
                "sound_quality": table["sound_quality"].get(items[8]),
            }

        for line in utils.readlines(scripts_path):
            path, text = line.split("::", maxsplit=1)
            path = path.strip().lstrip("/")

            # To handle config.exclude_categories
            if path in metadatum_by_path:
                metadatum_by_path[path]["text"] = text.strip()

        return list(metadatum_by_path.values())

    for group_path in track(group_paths, "Loading metadata..."):
        yield from work(group_path)


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
            "category",
            "subcategory",
            "speaker_gender",
            "speaker_generation",
            "speaker_residence",
            "speaker_dialect",
            "source",
            "sound_quality",
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
