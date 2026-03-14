import json
import os
import gc
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
import polars.selectors as cs

TOTAL_TIME_STEPS = 239
TARGET_TIME_STEPS = 229
BASE_FEATURE_COLS = [f"f{i}" for i in range(384)]
# 核心代码注释：定义需要切分的时间段，涵盖全天所有 timeid
# TIME_RANGES = [(0, 59), (60, 119), (120, 179), (180, 238)]
TIME_RANGES = [(0, 29), (30, 59), (60, 89), (90, 119), (120, 149), (150, 179), (180, 209), (210, 238)]


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
    
    # mean_exprs = [
    #     pl.col(col).mean().over("timeid").alias(f"{col}_time_mean")
    #     for col in feature_cols
    # ]
    # var_exprs = [
    #     pl.col(col).var(ddof=0).over("timeid").fill_null(0.0).alias(f"{col}_time_var")
    #     for col in feature_cols
    # ]
    rank_exprs = [
        (pl.col(col).rank().over("timeid") / count_expr).alias(col)
        for col in feature_cols
    ]

    # extra_exprs = [
    #     (pl.col("timeid") / float(TOTAL_TIME_STEPS - 1)).alias("timeid_norm"),
    #     (pl.col("stockid") / 499.0).alias("stockid_norm"),
    #     pl.col("exchangeid").cast(pl.Float32).alias("exchangeid_float"),
    # ]

    # df = df.with_columns(rank_exprs + mean_exprs + var_exprs + extra_exprs).sort(
    #     ["dateid", "stockid", "timeid"]
    # )
    df = df.with_columns(rank_exprs).sort(
        ["dateid", "stockid", "timeid"]
    )

    engineered_cols = (
        list(feature_cols)
        # + [f"{col}_time_mean" for col in feature_cols]
        # + [f"{col}_time_var" for col in feature_cols]
        # + ["timeid_norm", "stockid_norm", "exchangeid_float"]
    )

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
        
        # 核心代码注释：利用 scan_parquet 进行 Lazy 运算，仅读取所需时间段的数据
        # 这样可以将原来一整天的数据量缩减为原来的 1/4 左右，保证内存安全
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


@dataclass
class TargetTransformState:
    mean: Optional[float] = None
    std: Optional[float] = None


class TargetTransformer:
    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.mean_ = float(y.mean())
        self.std_ = float(y.std())
        if self.std_ < 1e-8:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        transformed = (y - self.mean_) / (self.std_ + 1e-8)
        return np.clip(transformed, -5.0, 5.0).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        return (y * self.std_ + self.mean_).astype(np.float32)

    def to_state(self) -> TargetTransformState:
        return TargetTransformState(mean=self.mean_, std=self.std_)

    @classmethod
    def from_state(cls, state: TargetTransformState) -> "TargetTransformer":
        transformer = cls()
        transformer.mean_ = state.mean
        transformer.std_ = state.std
        return transformer


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def save_transformer(transformer: TargetTransformer, path: str) -> None:
    state = transformer.to_state()
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(asdict(state), fp)


def load_transformer(path: str) -> TargetTransformer:
    with open(path, "r", encoding="utf-8") as fp:
        state = TargetTransformState(**json.load(fp))
    return TargetTransformer.from_state(state)


def train_gbdt_by_time(
    train_dir: str,
    feature_cols: Sequence[str],
    target_col: str,
    val_ratio: float,
    epochs: int,
    model_prefix: str,
):
    all_dates = list_date_ids(train_dir)
    split_idx = int(len(all_dates) * (1.0 - val_ratio))
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]

    print(f"Total dates: {len(all_dates)}, Train: {len(train_dates)}, Val: {len(val_dates)}")

    # params = {
    #     "objective": "regression",
    #     "metric": "rmse",
    #     "boosting_type": "gbdt",
    #     "learning_rate": 0.03,
    #     "num_leaves": 31,
    #     "max_depth": 8,
    #     "min_data_in_leaf": 256,
    #     "feature_fraction": 0.7,
    #     "bagging_fraction": 0.8,
    #     "bagging_freq": 1,
    #     "verbosity": -1,
    #     "num_threads": os.cpu_count() or 8,
    #     "seed": 42,
    # }

    # params = {
    #     "objective": "regression",
    #     "metric": "rmse",
    #     "boosting_type": "gbdt",
    #     'lambda_l1': 0.01, 
    #     'lambda_l2': 0.1, 
    #     'num_leaves': 128, 
    #     'bagging_freq': 7, 
    #     'max_depth': -1, 
    #     # 'max_bin': 501, 
    #     'n_estimators': 1000, 
    #     "learning_rate": 0.03,
    #     "min_data_in_leaf": 256,
    #     "feature_fraction": 0.7,
    #     "bagging_fraction": 0.8,
    #     "verbosity": -1,
    #     "num_threads": os.cpu_count() or 8,
    #     # "device": "gpu",
    #     "seed": 42,
    # }
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        'lambda_l1': 0.01, 
        'lambda_l2': 0.1, 
        'num_leaves': 128, 
        'bagging_freq': 7, 
        'max_depth': -1, 
        # 'max_bin': 501, 
        'n_estimators': 500, 
        "learning_rate": 0.03,
        "min_data_in_leaf": 256,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "verbosity": -1,
        "num_threads": os.cpu_count() or 8,
        # "device": "gpu",
        "seed": 42,
    }

    # params = {
    #     "objective": "regression",
    #     "metric": "rmse",
    #     "boosting_type": "gbdt",
    #     'lambda_l1': 0.001, 
    #     'lambda_l2': 0.001, 
    #     'num_leaves': 64, 
    #     'bagging_freq': 7, 
    #     'max_depth': -1, 
    #     # 'max_bin': 501, 
    #     'n_estimators': 500, 
    #     "learning_rate": 0.03,
    #     "min_data_in_leaf": 256,
    #     "feature_fraction": 0.7,
    #     "bagging_fraction": 0.8,
    #     "verbosity": -1,
    #     "num_threads": os.cpu_count() or 8,
    #     # "device": "gpu",
    #     "seed": 42,
    # }

    # 核心代码注释：外层循环遍历所有定义好的时段，为每个时段单独训练一个树模型与目标转换器
    for tr in TIME_RANGES:
        print(f"\n{'='*40}\nTraining model for Time Range: {tr[0]} - {tr[1]}\n{'='*40}")
        
        # 1. 独立加载该时段的训练集与验证集
        x_train, y_train_raw, _, eng_cols = load_time_range_data(
            train_dir, train_dates, tr, feature_cols, True, target_col, train_only_rows=True
        )
        x_val, y_val_raw, _, _ = load_time_range_data(
            train_dir, val_dates, tr, feature_cols, True, target_col, train_only_rows=True
        )

        # 2. 针对此时段的收益率分布，单独拟合 QuantileTransformer，剥离特定时段的重尾效应
        transformer = TargetTransformer().fit(y_train_raw)
        y_train = transformer.transform(y_train_raw)
        
        # 显式释放原始标签内存
        del y_train_raw 
        gc.collect()

        train_set = lgb.Dataset(x_train, label=y_train, feature_name=eng_cols, free_raw_data=False)
        # 验证集不进行 transform，保留原始刻度，以便直接评估最终的 R2
        val_set = lgb.Dataset(x_val, label=np.zeros_like(y_val_raw), reference=train_set, free_raw_data=False) 

        print(f"Loaded Train shape: {x_train.shape}, Val shape: {x_val.shape}")

        best_val_r2 = -np.inf
        booster = None

        for epoch in range(epochs):
            booster = lgb.train(
                params=params,
                train_set=train_set,
                num_boost_round=100, 
                init_model=booster,
                keep_training_booster=True,
            )
            
            # 使用训练出的 booster 预测验证集并还原分布计算 R2
            pred_val_norm = booster.predict(x_val)
            pred_val = transformer.inverse_transform(pred_val_norm)
            val_r2 = r2_score(y_val_raw, pred_val)
            print(f"Epoch {epoch + 1}/{epochs} - Val R2: {val_r2:.6f}")

            model_path = f"{model_prefix}_{tr[0]}_{tr[1]}.txt"
            trans_path = f"{model_prefix}_trans_{tr[0]}_{tr[1]}.json"

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                booster.save_model(model_path)
                save_transformer(transformer, trans_path)
                print(f"  [Saved] Best model for {tr} with R2: {val_r2:.6f}")
                
        # 训练结束后清理当前时段的内存
        del x_train, x_val, y_train, y_val_raw, train_set, val_set, booster
        gc.collect()


def predict_test_by_time(
    test_dir: str,
    feature_cols: Sequence[str],
    model_prefix: str,
    submission_path: str,
):
    test_dates = list_date_ids(test_dir)
    all_uids = []
    all_preds = []

    # 核心代码注释：推理时同样按时域读取预测，然后将各时段预测结果拼装
    for tr in TIME_RANGES:
        print(f"Predicting Time Range: {tr[0]} - {tr[1]}")
        model_path = f"{model_prefix}_{tr[0]}_{tr[1]}.txt"
        trans_path = f"{model_prefix}_trans_{tr[0]}_{tr[1]}.json"
        
        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping...")
            continue
            
        booster = lgb.Booster(model_file=model_path)
        transformer = load_transformer(trans_path)
        
        x_test, _, ids, _ = load_time_range_data(
            test_dir, test_dates, tr, feature_cols, False, train_only_rows=False
        )
        
        if x_test.shape[0] > 0:
            pred_norm = booster.predict(x_test)
            pred = transformer.inverse_transform(pred_norm)
            
            all_preds.append(pred)
            all_uids.extend([f"{int(s)}|{int(d)}|{int(t)}" for s, d, t in ids])
            print(f"  Predicted {len(pred)} rows.")

    if all_preds:
        final_preds = np.concatenate(all_preds)
        submission = pl.DataFrame({"Uid": all_uids, "prediction": final_preds})
        # 为保证与官方要求的完全对齐，可以可选地对 Uid 解析后排序
        submission.write_csv(submission_path)
        print(f"Saved complete submission to {submission_path}")


if __name__ == "__main__":
    parquet_root = "./data"
    train_dir = os.path.join(parquet_root, "train")
    test_dir = os.path.join(parquet_root, "test")

    model_prefix = "./model_lgbm_time"
    submission_path = "./submission_gbdt_time_split.csv"

    train_gbdt_by_time(
        train_dir=train_dir,
        feature_cols=BASE_FEATURE_COLS,
        target_col="LabelA",
        val_ratio=0.2,
        epochs=5,
        model_prefix=model_prefix,
    )

    predict_test_by_time(
        test_dir=test_dir,
        feature_cols=BASE_FEATURE_COLS,
        model_prefix=model_prefix,
        submission_path=submission_path,
    )
