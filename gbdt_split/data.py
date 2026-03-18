import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs

TOTAL_TIME_STEPS = 239
TARGET_TIME_STEPS = 229
BASE_FEATURE_COLS = [f"f{i}" for i in range(384)]
TIME_RANGES = [
    (0, 29),
    (30, 59),
    (60, 89),
    (90, 119),
    (120, 149),
    (150, 179),
    (180, 209),
    (210, 238),
]


def list_date_ids(parquet_dir: str) -> List[int]:
    date_ids: List[int] = []
    for file_name in os.listdir(parquet_dir):
        if not file_name.lower().endswith(".parquet"):
            continue
        stem = os.path.splitext(file_name)[0]
        if stem.isdigit():
            date_ids.append(int(stem))
    return sorted(date_ids)


def clean_day_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(cs.float().fill_nan(None))
    df = df.with_columns(cs.float().fill_null(cs.float().mean().over("timeid")))
    df = df.with_columns(cs.float().fill_null(0.0))
    df = df.with_columns(
        pl.when(cs.float().is_infinite()).then(0.0).otherwise(cs.float()).name.keep()
    )
    return df


def build_feature_frame(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    include_target: bool,
    target_col: str = "LabelA",
) -> Tuple[pl.DataFrame, List[str]]:
    df = clean_day_frame(df)
    count_expr = pl.col("stockid").count().over("timeid")
    rank_exprs = [
        (pl.col(col).rank().over("timeid") / count_expr).alias(col)
        for col in feature_cols
    ]
    df = df.with_columns(rank_exprs).sort(["dateid", "stockid", "timeid"])

    engineered_cols = list(feature_cols)
    selected_cols = ["stockid", "dateid", "timeid"] + engineered_cols
    if include_target:
        selected_cols.append(target_col)
    return df.select(selected_cols), engineered_cols


def load_time_range_data(
    parquet_dir: str,
    date_ids: Sequence[int],
    time_range: Tuple[int, int],
    feature_cols: Sequence[str],
    include_target: bool,
    target_col: str = "LabelA",
    train_only_rows: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[str]]:
    frames: List[pl.DataFrame] = []
    engineered_cols: Optional[List[str]] = None

    for date_id in date_ids:
        file_path = os.path.join(parquet_dir, f"{date_id}.parquet")
        if not os.path.exists(file_path):
            continue

        lf = pl.scan_parquet(file_path)
        lf = lf.filter(
            (pl.col("timeid") >= time_range[0]) & (pl.col("timeid") <= time_range[1])
        )
        if train_only_rows:
            lf = lf.filter(pl.col("timeid") < TARGET_TIME_STEPS)

        day_df = lf.collect()
        if day_df.height == 0:
            continue

        feature_df, engineered_cols = build_feature_frame(
            day_df,
            feature_cols=feature_cols,
            include_target=include_target,
            target_col=target_col,
        )
        frames.append(feature_df)

    merged = pl.concat(frames, how="vertical")
    ids = merged.select(["stockid", "dateid", "timeid"]).to_numpy()
    x = merged.select(engineered_cols).to_numpy().astype(np.float32, copy=False)
    y = None
    if include_target:
        y = merged.select(target_col).to_numpy().reshape(-1).astype(np.float32, copy=False)

    return x, y, ids, engineered_cols
