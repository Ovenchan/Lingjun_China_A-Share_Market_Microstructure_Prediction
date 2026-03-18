import os

from gbdt_split import GBDTSplitConfig, predict_test_by_time, train_gbdt_by_time


def build_config() -> GBDTSplitConfig:
    parquet_root = "./data"
    return GBDTSplitConfig(
        train_dir=os.path.join(parquet_root, "train"),
        test_dir=os.path.join(parquet_root, "test"),
        model_prefix="./model_cat_time",
        submission_path="./submission_cat_time_split.csv",
        model_type="catboost",
    )


if __name__ == "__main__":
    config = build_config()
    train_gbdt_by_time(config)
    predict_test_by_time(config)
