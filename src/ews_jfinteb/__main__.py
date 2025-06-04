import random
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset, Dataset, DatasetDict


COL_TEXT: str = "text"
COL_LABEL: str = "label"
COLUMNS: list[str] = ["id", COL_TEXT, COL_LABEL]


def load_dataset_from_hf(tense: str, version: str, col_text: str) -> Dataset:
    splits: tuple[str] = ("train", "validation", "test")
    dsd: DatasetDict = load_dataset(
        "retarfi/economy-watchers-survey", tense, revision=version
    )
    ds: Dataset = concatenate_datasets([dsd[split] for split in splits])
    ds = ds.rename_column(col_text, COL_TEXT)
    ds = filter_short_text(ds, min_length=10)
    return ds


def filter_short_text(ds: Dataset, min_length: int) -> Dataset:
    return ds.filter(lambda x: len(x[COL_TEXT]) >= min_length)


def sample(lst: list[Any], num_samples: int, seed: int) -> list[Any]:
    assert len(lst) >= num_samples
    random.seed(seed)
    return random.sample(lst, num_samples)


def query_samples(
    ds: Dataset,
    labels: tuple[str, ...],
    num_each_label: int,
    seed: int,
) -> Dataset:
    df: Dataset = ds.to_pandas()
    dct_counter: dict[str, int] = df[COL_LABEL].value_counts().to_dict()
    assert all(dct_counter[label] >= num_each_label for label in labels)
    lst_idx: list[int] = []
    for label in labels:
        idx: list[int] = df[df[COL_LABEL] == label].index.tolist()
        idx_sample: list[int] = sample(idx, num_each_label, seed)
        lst_idx.extend(idx_sample)
    lst_idx.sort()
    ds_queried: Dataset = ds.select(lst_idx).select_columns(COLUMNS)
    return ds_queried


def reason(ds: Dataset, seed: int) -> Dataset:
    labels: tuple[str, ...] = (
        "来客数の動き",
        "販売量の動き",
        "お客様の様子",
        "受注量や販売量の動き",
        "単価の動き",
        "取引先の様子",
        "求人数の動き",
        "競争相手の様子",
        "受注価格や販売価格の動き",
        "周辺企業の様子",
        "求職者数の動き",
        "採用者数の動き",
        "雇用形態の様子",
    )
    ds_reason: Dataset = query_samples(
        ds.rename_column("判断の理由", COL_LABEL), labels, num_each_label=1000, seed=seed
    )
    return ds_reason


def horizon(
    ds_current: Dataset,
    ds_future: Dataset,
    seed: int,
) -> Dataset:
    num_each_label: int = 2000

    def _query_samples_horizon(ds: Dataset, label: str) -> Dataset:
        idx: list[int] = sorted(sample(range(len(ds)), num_each_label, seed))
        ds_q: Dataset = ds.select(idx)
        ds_q = ds_q.add_column(COL_LABEL, [label] * len(ds_q)).select_columns(COLUMNS)
        return ds_q

    ds_q: Dataset = concatenate_datasets(
        [
            _query_samples_horizon(ds_current, "現状"),
            _query_samples_horizon(ds_future, "先行き"),
        ]
    )
    return ds_q


def sentiment(ds_current: Dataset, ds_future: Dataset, seed: int) -> Dataset:
    num_each_label: int = 1000
    labels: tuple[str, ...] = ("◎", "○", "□", "▲", "×")
    ds_q_current: Dataset = ds_current.rename_column("景気の現状判断", COL_LABEL)
    ds_q_future: Dataset = ds_future.rename_column("景気の先行き判断", COL_LABEL)
    ds_q: Dataset = concatenate_datasets(
        [ds_q_current.select_columns(COLUMNS), ds_q_future.select_columns(COLUMNS)]
    )
    ds_q = query_samples(ds_q, labels, num_each_label=num_each_label, seed=seed)
    return ds_q


def main():
    ews_version: str = "2025.04.0"
    p_data: Path = Path(__file__).parents[2] / "hf-hub"
    p_data.mkdir(exist_ok=True)

    seed: int = 42

    ds_current: Dataset = load_dataset_from_hf(
        "current", ews_version, "追加説明及び具体的状況の説明"
    )
    ds_future: Dataset = load_dataset_from_hf(
        "future", ews_version, "景気の先行きに対する判断理由"
    )

    # clustering - reason
    ds_reason: Dataset = reason(ds_current, seed=seed)
    ds_reason.to_parquet(p_data / "reason.parquet")

    # classification- horizon
    ds_horizon: Dataset = horizon(ds_current, ds_future, seed=seed)
    ds_horizon.to_parquet(p_data / "horizon.parquet")

    # classification - sentiment
    ds_sentiment: Dataset = sentiment(ds_current, ds_future, seed=seed)
    ds_sentiment.to_parquet(p_data / "sentiment.parquet")


if __name__ == "__main__":
    main()
