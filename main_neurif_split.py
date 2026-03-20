import os

from neurif_split import NeurIFSplitConfig, finetune_neurif_by_time, predict_test_by_time


def build_config() -> NeurIFSplitConfig:
    parquet_root = "D:/kaggle_data"
    return NeurIFSplitConfig(
        train_dir=os.path.join(parquet_root, "train"),
        test_dir=os.path.join(parquet_root, "test"),
        model_prefix="./model_neurif_time",
        submission_path="./submission_neurif_time_split.csv",
        feature_cols=[f"f{i}" for i in range(384)],
        target_col="LabelA",
        seq_len=10,
        output_dim=1,
        epochs=10,
        lr=1e-3,
        hidden_dim=64,
        num_factors=32,
        num_layers=2,
        dropout=0.2,
        lambda_1=1e-4,
        lambda_2=1e-4,
    )


if __name__ == "__main__":
    cfg = build_config()
    finetune_neurif_by_time(cfg, finetune_epochs=5, finetune_lr=1e-5)
    predict_test_by_time(cfg)
