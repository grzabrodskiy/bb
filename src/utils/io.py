from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: object, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_records(items: Iterable[object]) -> list[dict]:
    records = []
    for item in items:
        if hasattr(item, "model_dump"):
            records.append(item.model_dump())
        elif hasattr(item, "dict"):
            records.append(item.dict())
        elif isinstance(item, dict):
            records.append(item)
        else:
            records.append({"value": item})
    return records


def write_parquet(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    df = pd.json_normalize(records)
    df.to_parquet(path, index=False)


def write_csv(records: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    df = pd.json_normalize(records)
    df.to_csv(path, index=False)
