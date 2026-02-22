import os
import polars as pl
from tqdm import tqdm


def split_parquet_by_date(input_parquet: str, output_dir: str, date_col: str = "dateid", overwrite: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    # 先拿到所有 dateid
    date_ids = (
        pl.scan_parquet(input_parquet)
        .select(pl.col(date_col).unique())
        .collect()
        .to_series()
        .sort()
        .to_list()
    )

    print(f"[INFO] {input_parquet} -> {output_dir}, total dates: {len(date_ids)}")

    for d in tqdm(date_ids, desc=f"Splitting {os.path.basename(input_parquet)}"):
        out_file = os.path.join(output_dir, f"{int(d)}.parquet")

        if (not overwrite) and os.path.exists(out_file):
            continue

        df_day = (
            pl.scan_parquet(input_parquet)
            .filter(pl.col(date_col) == d)
            .collect()
        )

        if df_day.is_empty():
            continue

        df_day.write_parquet(out_file, compression="zstd")


if __name__ == "__main__":
    parquet_root = r"D:\kaggle_data"
    train_in = os.path.join(parquet_root, "train.parquet")
    test_in = os.path.join(parquet_root, "test.parquet")

    train_out = os.path.join(parquet_root, "train")
    test_out = os.path.join(parquet_root, "test")

    split_parquet_by_date(train_in, train_out, date_col="dateid", overwrite=False)
    split_parquet_by_date(test_in, test_out, date_col="dateid", overwrite=False)

    print("Done.")