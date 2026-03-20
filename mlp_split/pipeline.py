import gc
import os
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gbdt_split.data import BASE_FEATURE_COLS, TIME_RANGES, list_date_ids, load_time_range_data
from gbdt_split.transform import TargetTransformer, load_transformer, save_transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-12:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (512, 256), negative_slope: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPSplitConfig:
    train_dir: str
    test_dir: str
    model_prefix: str = "./model_mlp_time"
    submission_path: str = "./submission_mlp_time_split.csv"
    feature_cols: Sequence[str] = field(default_factory=lambda: list(BASE_FEATURE_COLS))
    target_col: str = "LabelA"
    val_ratio: float = 0.2
    epochs: int = 20
    batch_size: int = 65536
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden_dims: tuple[int, int] = (512, 256)
    leaky_relu_slope: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def model_path(self, time_range: tuple[int, int]) -> str:
        return f"{self.model_prefix}_{time_range[0]}_{time_range[1]}.pth"

    def transformer_path(self, time_range: tuple[int, int]) -> str:
        return f"{self.model_prefix}_trans_{time_range[0]}_{time_range[1]}.json"


def _build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _predict_in_batches(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            preds.append(yb)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_mlp_by_time(config: MLPSplitConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    all_dates = list_date_ids(config.train_dir)
    split_idx = int(len(all_dates) * (1.0 - config.val_ratio))
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]
    print(f"Total dates: {len(all_dates)}, Train: {len(train_dates)}, Val: {len(val_dates)}")

    for tr in TIME_RANGES:
        print(f"\n{'=' * 40}\nTraining MLP for Time Range: {tr[0]} - {tr[1]}\n{'=' * 40}")

        x_train, y_train_raw, _, _ = load_time_range_data(
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

        if x_train.shape[0] == 0 or x_val.shape[0] == 0:
            print("No train/val rows for this range, skipping...")
            continue

        transformer = TargetTransformer().fit(y_train_raw)
        y_train = transformer.transform(y_train_raw)

        print(f"Loaded Train shape: {x_train.shape}, Val shape: {x_val.shape}")

        model = MLPRegressor(
            input_dim=x_train.shape[1],
            hidden_dims=config.hidden_dims,
            negative_slope=config.leaky_relu_slope,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()

        train_loader = _build_loader(
            x_train,
            y_train.astype(np.float32, copy=False),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        best_val_r2 = -np.inf
        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)

        for epoch in range(config.epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_size = xb.shape[0]
                train_loss_sum += float(loss.detach().item()) * batch_size
                train_count += batch_size

            pred_val_norm = _predict_in_batches(model, x_val, config.batch_size, device)
            pred_val = transformer.inverse_transform(pred_val_norm)
            val_r2 = r2_score(y_val_raw, pred_val)
            mean_train_loss = train_loss_sum / max(1, train_count)
            print(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train MSE: {mean_train_loss:.6f} - Val R2: {val_r2:.6f}"
            )

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), model_path)
                save_transformer(transformer, trans_path)
                print(f"  [Saved] Best model for {tr} with R2: {val_r2:.6f}")

        del x_train, x_val, y_train_raw, y_val_raw, y_train, train_loader, model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def finetune_mlp_by_time(
    config: MLPSplitConfig,
    finetune_epochs: int = 5,
    finetune_lr: float = 1e-5,
) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    all_dates = list_date_ids(config.train_dir)
    split_idx = int(len(all_dates) * (1.0 - config.val_ratio))
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]
    print(f"Total dates: {len(all_dates)}, Train: {len(train_dates)}, Val: {len(val_dates)}")

    for tr in TIME_RANGES:
        print(f"\n{'=' * 40}\nFinetuning MLP for Time Range: {tr[0]} - {tr[1]}\n{'=' * 40}")

        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)
        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping...")
            continue
        if not os.path.exists(trans_path):
            print(f"  Transformer {trans_path} not found, skipping...")
            continue

        x_train, y_train_raw, _, _ = load_time_range_data(
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

        if x_train.shape[0] == 0 or x_val.shape[0] == 0:
            print("No train/val rows for this range, skipping...")
            continue

        transformer = load_transformer(trans_path)
        y_train = transformer.transform(y_train_raw)

        print(f"Loaded Train shape: {x_train.shape}, Val shape: {x_val.shape}")

        model = MLPRegressor(
            input_dim=x_train.shape[1],
            hidden_dims=config.hidden_dims,
            negative_slope=config.leaky_relu_slope,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()

        train_loader = _build_loader(
            x_train,
            y_train.astype(np.float32, copy=False),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        baseline_pred_norm = _predict_in_batches(model, x_val, config.batch_size, device)
        baseline_pred = transformer.inverse_transform(baseline_pred_norm)
        baseline_val_r2 = r2_score(y_val_raw, baseline_pred)
        best_val_r2 = baseline_val_r2
        print(f"Baseline Val R2: {baseline_val_r2:.6f}")

        for epoch in range(finetune_epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_size = xb.shape[0]
                train_loss_sum += float(loss.detach().item()) * batch_size
                train_count += batch_size

            pred_val_norm = _predict_in_batches(model, x_val, config.batch_size, device)
            pred_val = transformer.inverse_transform(pred_val_norm)
            val_r2 = r2_score(y_val_raw, pred_val)
            mean_train_loss = train_loss_sum / max(1, train_count)
            print(
                f"Epoch {epoch + 1}/{finetune_epochs} - "
                f"Train MSE: {mean_train_loss:.6f} - Val R2: {val_r2:.6f}"
            )

            if np.isfinite(val_r2) and val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), model_path)
                save_transformer(transformer, trans_path)
                print(f"  [Saved] Improved model for {tr} with R2: {val_r2:.6f}")

        print(f"Range {tr} summary - Baseline R2: {baseline_val_r2:.6f}, Best R2: {best_val_r2:.6f}")

        del x_train, x_val, y_train_raw, y_val_raw, y_train, train_loader, model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def predict_test_by_time(config: MLPSplitConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    test_dates = list_date_ids(config.test_dir)
    all_uids = []
    all_preds = []

    for tr in TIME_RANGES:
        print(f"Predicting Time Range: {tr[0]} - {tr[1]}")
        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)

        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping...")
            continue
        if not os.path.exists(trans_path):
            print(f"  Transformer {trans_path} not found, skipping...")
            continue

        x_test, _, ids, _ = load_time_range_data(
            config.test_dir,
            test_dates,
            tr,
            config.feature_cols,
            False,
            train_only_rows=False,
        )

        if x_test.shape[0] == 0:
            print("  No rows in this range, skipping...")
            continue

        model = MLPRegressor(
            input_dim=x_test.shape[1],
            hidden_dims=config.hidden_dims,
            negative_slope=config.leaky_relu_slope,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        transformer = load_transformer(trans_path)
        pred_norm = _predict_in_batches(model, x_test, config.batch_size, device)
        pred = transformer.inverse_transform(pred_norm)

        all_preds.append(pred)
        all_uids.extend([f"{int(s)}|{int(d)}|{int(t)}" for s, d, t in ids])
        print(f"  Predicted {len(pred)} rows.")

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_preds:
        final_preds = np.concatenate(all_preds)
        submission = pl.DataFrame({"Uid": all_uids, "prediction": final_preds})
        submission.write_csv(config.submission_path)
        print(f"Saved complete submission to {config.submission_path}")