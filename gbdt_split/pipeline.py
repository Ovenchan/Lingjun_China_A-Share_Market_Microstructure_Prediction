import gc
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
import polars as pl

from .backends import create_backend
from .data import BASE_FEATURE_COLS, TIME_RANGES, list_date_ids, load_time_range_data
from .transform import TargetTransformer, load_transformer, save_transformer


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


@dataclass
class GBDTSplitConfig:
    train_dir: str
    test_dir: str
    model_prefix: str = "./model_lgbm_time"
    submission_path: str = "./submission_gbdt_time_split.csv"
    feature_cols: Sequence[str] = field(default_factory=lambda: list(BASE_FEATURE_COLS))
    target_col: str = "LabelA"
    val_ratio: float = 0.2
    epochs: int = 10
    num_boost_round: int = 100
    model_type: str = "lightgbm"
    params: Dict[str, Any] | None = None

    def model_path(self, time_range: tuple[int, int], extension: str) -> str:
        return f"{self.model_prefix}_{time_range[0]}_{time_range[1]}{extension}"

    def transformer_path(self, time_range: tuple[int, int]) -> str:
        return f"{self.model_prefix}_trans_{time_range[0]}_{time_range[1]}.json"

    @staticmethod
    def default_model_params(model_type: str) -> Dict[str, Any]:
        normalized = model_type.lower()
        if normalized in {"lightgbm", "lgbm", "lgb"}:
            return {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "max_depth": 8,
                "min_data_in_leaf": 4096,
                "lambda_l1": 1,
                "lambda_l2": 1,
                "learning_rate": 0.015,
                "feature_fraction": 0.6,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "verbosity": -1,
                "num_threads": os.cpu_count() or 8,
                "seed": 42,
            }
        if normalized in {"xgboost", "xgb"}:
            return {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": 0.015,
                "max_depth": 8,
                "min_child_weight": 4096,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "lambda": 1.0,
                "alpha": 1.0,
                "seed": 42,
                "nthread": os.cpu_count() or 8,
                "verbosity": 0,
            }
        if normalized in {"catboost", "cat"}:
            return {
                "loss_function": "RMSE",
                "learning_rate": 0.015,
                "depth": 8,
                "l2_leaf_reg": 1.0,
                "subsample": 0.7,
                "random_seed": 42,
                "thread_count": os.cpu_count() or 8,
            }
        raise ValueError(f"Unsupported model_type: {model_type}")

    def get_model_params(self) -> Dict[str, Any]:
        if self.params is None:
            return self.default_model_params(self.model_type)
        return dict(self.params)


def train_gbdt_by_time(
    config: GBDTSplitConfig,
) -> None:
    backend = create_backend(config.model_type)
    params = config.get_model_params()

    all_dates = list_date_ids(config.train_dir)
    split_idx = int(len(all_dates) * (1.0 - config.val_ratio))
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]

    print(f"Total dates: {len(all_dates)}, Train: {len(train_dates)}, Val: {len(val_dates)}")

    for tr in TIME_RANGES:
        print(f"\n{'=' * 40}\nTraining model for Time Range: {tr[0]} - {tr[1]}\n{'=' * 40}")

        x_train, y_train_raw, _, eng_cols = load_time_range_data(
            config.train_dir,
            train_dates,
            tr,
            config.feature_cols,
            True,
            config.target_col,
            train_only_rows=True,
        )
        x_val, y_val_raw, _, _ = load_time_range_data(
            config.train_dir,
            val_dates,
            tr,
            config.feature_cols,
            True,
            config.target_col,
            train_only_rows=True,
        )

        transformer = TargetTransformer().fit(y_train_raw)
        y_train = transformer.transform(y_train_raw)
        del y_train_raw
        gc.collect()

        artifacts = backend.prepare_datasets(x_train, y_train, x_val, eng_cols)
        print(f"Loaded Train shape: {x_train.shape}, Val shape: {x_val.shape}")

        best_val_r2 = -np.inf
        booster = None
        model_path = config.model_path(tr, backend.model_extension)
        trans_path = config.transformer_path(tr)

        for epoch in range(config.epochs):
            booster = backend.train_epoch(
                params=params,
                artifacts=artifacts,
                booster=booster,
                num_boost_round=config.num_boost_round,
            )
            pred_val_norm = backend.predict(booster, x_val, eng_cols)
            pred_val = transformer.inverse_transform(pred_val_norm)
            val_r2 = r2_score(y_val_raw, pred_val)
            print(f"Epoch {epoch + 1}/{config.epochs} - Val R2: {val_r2:.6f}")

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                backend.save_model(booster, model_path)
                save_transformer(transformer, trans_path)
                print(f"  [Saved] Best model for {tr} with R2: {val_r2:.6f}")

        del x_train, x_val, y_train, y_val_raw, artifacts, booster
        gc.collect()


def predict_test_by_time(
    config: GBDTSplitConfig,
) -> None:
    backend = create_backend(config.model_type)
    test_dates = list_date_ids(config.test_dir)
    all_uids = []
    all_preds = []

    for tr in TIME_RANGES:
        print(f"Predicting Time Range: {tr[0]} - {tr[1]}")
        model_path = config.model_path(tr, backend.model_extension)
        trans_path = config.transformer_path(tr)

        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping...")
            continue

        booster = backend.load_model(model_path)
        transformer = load_transformer(trans_path)

        x_test, _, ids, eng_cols = load_time_range_data(
            config.test_dir,
            test_dates,
            tr,
            config.feature_cols,
            False,
            train_only_rows=False,
        )

        if x_test.shape[0] > 0:
            pred_norm = backend.predict(booster, x_test, eng_cols)
            pred = transformer.inverse_transform(pred_norm)
            all_preds.append(pred)
            all_uids.extend([f"{int(s)}|{int(d)}|{int(t)}" for s, d, t in ids])
            print(f"  Predicted {len(pred)} rows.")

    if all_preds:
        final_preds = np.concatenate(all_preds)
        submission = pl.DataFrame({"Uid": all_uids, "prediction": final_preds})
        submission.write_csv(config.submission_path)
        print(f"Saved complete submission to {config.submission_path}")
