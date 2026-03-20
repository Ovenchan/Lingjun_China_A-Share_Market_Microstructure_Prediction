import os

from mlp_split import MLPSplitConfig, finetune_mlp_by_time, predict_test_by_time


def build_config() -> MLPSplitConfig:
    parquet_root = "D:/kaggle_data"
    return MLPSplitConfig(
        train_dir=os.path.join(parquet_root, "train"),
        test_dir=os.path.join(parquet_root, "test"),
        model_prefix="./model_mlp_time",
        submission_path="./submission_mlp_time_split.csv",
        epochs=10,
        batch_size=65536,
        lr=1e-3,
        hidden_dims=(512, 256),
        leaky_relu_slope=0.1,
    )


if __name__ == "__main__":
    config = build_config()
    finetune_mlp_by_time(config, finetune_epochs=3, finetune_lr=1e-5)
    predict_test_by_time(config)