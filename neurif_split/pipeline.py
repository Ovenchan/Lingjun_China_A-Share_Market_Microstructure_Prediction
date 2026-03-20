import gc
import os
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from gbdt_split.data import BASE_FEATURE_COLS, TARGET_TIME_STEPS, TIME_RANGES, list_date_ids
from gbdt_split.transform import TargetTransformer, load_transformer, save_transformer
from models.NeurIF import NeurIF


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


def _unwrap_model_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


@dataclass
class NeurIFSplitConfig:
    train_dir: str
    test_dir: str
    model_prefix: str = "./model_neurif_time"
    submission_path: str = "./submission_neurif_time_split.csv"
    feature_cols: Sequence[str] = field(default_factory=lambda: list(BASE_FEATURE_COLS))
    target_col: str = "LabelA"
    val_ratio: float = 0.2
    seq_len: int = 10
    output_dim: int = 1
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim: int = 64
    num_factors: int = 32
    num_layers: int = 2
    dropout: float = 0.2

    lambda_1: float = 1e-4
    lambda_2: float = 1e-4

    def model_path(self, time_range: tuple[int, int]) -> str:
        return f"{self.model_prefix}_{time_range[0]}_{time_range[1]}.pth"

    def transformer_path(self, time_range: tuple[int, int]) -> str:
        return f"{self.model_prefix}_trans_{time_range[0]}_{time_range[1]}.json"


def _load_single_day_sample(
    parquet_dir: str,
    date_id: int,
    time_range: tuple[int, int],
    feature_cols: Sequence[str],
    seq_len: int,
    include_target: bool,
    target_col: str,
    transformer: TargetTransformer | None,
    supervised_time_upper: int | None,
):
    file_path = os.path.join(parquet_dir, f"{int(date_id)}.parquet")
    if not os.path.exists(file_path):
        return None

    start_t, end_t = time_range
    df = pl.read_parquet(file_path)
    df = df.filter((pl.col("timeid") >= start_t) & (pl.col("timeid") <= end_t))
    if df.height == 0:
        return None

    df = df.fill_nan(0).fill_null(0)
    df = df.with_columns(
        pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
    )

    count_expr = pl.col("stockid").count().over("timeid")
    rank_exprs = [
        (pl.col(col).rank().over("timeid") / count_expr).alias(col)
        for col in feature_cols
    ]
    df = df.with_columns(rank_exprs).sort(["stockid", "timeid"])

    time_ids = (
        df.select("timeid")
        .unique()
        .sort("timeid")
        .to_numpy()
        .reshape(-1)
        .astype(np.int64, copy=False)
    )
    num_stocks = int(df.select("stockid").n_unique())
    time_steps = int(time_ids.shape[0])

    x_np = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    x_tensor = torch.from_numpy(x_np).reshape(num_stocks, time_steps, -1)
    x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)

    x_padded = F.pad(x_tensor, (0, 0, seq_len - 1, 0), value=0.0)
    x_windows = x_padded.unfold(1, seq_len, 1).permute(0, 1, 3, 2)

    if not include_target:
        ids = df.select(["stockid", "dateid", "timeid"]).to_numpy()
        return x_windows, ids

    y_raw_np = (
        df.select(target_col)
        .to_numpy()
        .reshape(-1)
        .astype(np.float32, copy=False)
        .reshape(num_stocks, time_steps)
    )

    if transformer is None:
        y_norm_np = y_raw_np
    else:
        y_norm_np = transformer.transform(y_raw_np.reshape(-1)).reshape(num_stocks, time_steps)

    if supervised_time_upper is None:
        mask_np = np.ones((num_stocks, time_steps), dtype=np.float32)
    else:
        valid_t = (time_ids < supervised_time_upper).astype(np.float32, copy=False)
        mask_np = np.broadcast_to(valid_t[None, :], (num_stocks, time_steps)).astype(np.float32, copy=False)

    y_norm = torch.from_numpy(y_norm_np)
    y_raw = torch.from_numpy(y_raw_np)
    mask = torch.from_numpy(mask_np)
    return x_windows, y_norm, y_raw, mask


def _load_range_samples_in_memory(
    parquet_dir: str,
    date_ids: Sequence[int],
    time_range: tuple[int, int],
    feature_cols: Sequence[str],
    seq_len: int,
    include_target: bool,
    target_col: str,
    transformer: TargetTransformer | None,
    supervised_time_upper: int | None,
):
    samples = []
    for date_id in date_ids:
        item = _load_single_day_sample(
            parquet_dir=parquet_dir,
            date_id=int(date_id),
            time_range=time_range,
            feature_cols=feature_cols,
            seq_len=seq_len,
            include_target=include_target,
            target_col=target_col,
            transformer=transformer,
            supervised_time_upper=supervised_time_upper,
        )
        if item is not None:
            samples.append(item)
    return samples


def _collect_targets_for_transformer(
    train_dir: str,
    train_dates: Sequence[int],
    time_range: tuple[int, int],
    target_col: str,
) -> np.ndarray:
    files = [os.path.join(train_dir, f"{int(d)}.parquet") for d in train_dates]
    if not files:
        return np.empty((0,), dtype=np.float32)

    lf = pl.scan_parquet(files)
    lf = lf.filter((pl.col("timeid") >= time_range[0]) & (pl.col("timeid") <= time_range[1]))
    lf = lf.filter(pl.col("timeid") < TARGET_TIME_STEPS)
    y = lf.select(target_col).collect().to_numpy().reshape(-1).astype(np.float32, copy=False)
    return y


def _build_model(config: NeurIFSplitConfig, time_steps: int) -> NeurIF:
    return NeurIF(
        input_dim=len(config.feature_cols),
        hidden_dim=config.hidden_dim,
        num_factors=config.num_factors,
        time_steps=time_steps,
        output_dim=config.output_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    weighted = err * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def _compute_neurif_loss(
    outputs: torch.Tensor,
    y_norm: torch.Tensor,
    x_batch: torch.Tensor,
    Lambda: torch.Tensor,
    F_prime: torch.Tensor,
    mask: torch.Tensor,
    config: NeurIFSplitConfig,
) -> torch.Tensor:
    pred = outputs[:, :, 0]
    loss_task = _masked_mse(pred, y_norm, mask)

    K = F_prime.shape[-1]
    F_T_F = torch.matmul(F_prime.T, F_prime)
    I_K = torch.eye(K, device=F_prime.device)
    loss_orth = torch.norm(F_T_F - I_K, p="fro")

    lambda_mean = Lambda.mean(dim=0, keepdim=True)
    lambda_dev = torch.norm(Lambda - lambda_mean, dim=-1)

    x_feat = x_batch.mean(dim=2)
    x_mean = x_feat.mean(dim=0, keepdim=True)
    x_dev = torch.norm(x_feat - x_mean, dim=-1)
    loss_inst = torch.mean((lambda_dev - x_dev) ** 2)

    return loss_task + config.lambda_1 * loss_orth + config.lambda_2 * loss_inst


def _evaluate_val_r2(
    model: nn.Module,
    val_samples,
    transformer: TargetTransformer,
    device: torch.device,
) -> float:
    model.eval()
    val_pred_norm_parts: list[np.ndarray] = []
    val_true_parts: list[np.ndarray] = []

    with torch.no_grad():
        for x, _, y_raw, mask in val_samples:
            x = x.to(device)
            outputs = _unwrap_model_output(model(x))[:, :, 0].detach().cpu().numpy()
            y_raw_np = y_raw.numpy()
            mask_np = mask.numpy() > 0.5

            if mask_np.any():
                val_pred_norm_parts.append(outputs[mask_np])
                val_true_parts.append(y_raw_np[mask_np])

    if not val_pred_norm_parts:
        return float("nan")

    pred_norm_all = np.concatenate(val_pred_norm_parts)
    true_all = np.concatenate(val_true_parts)
    pred_all = transformer.inverse_transform(pred_norm_all)
    return r2_score(true_all, pred_all)


def train_neurif_by_time(config: NeurIFSplitConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    all_dates = list_date_ids(config.train_dir)
    split_idx = int(len(all_dates) * (1.0 - config.val_ratio))
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]
    print(f"Total dates: {len(all_dates)}, Train: {len(train_dates)}, Val: {len(val_dates)}")

    for tr in TIME_RANGES:
        print(f"\n{'=' * 48}\nTraining NeurIF for Time Range: {tr[0]} - {tr[1]}\n{'=' * 48}")

        y_fit = _collect_targets_for_transformer(config.train_dir, train_dates, tr, config.target_col)
        if y_fit.size == 0:
            print("No train rows in supervised target region for this range, skipping...")
            continue

        transformer = TargetTransformer().fit(y_fit)

        train_samples = _load_range_samples_in_memory(
            parquet_dir=config.train_dir,
            date_ids=train_dates,
            time_range=tr,
            feature_cols=config.feature_cols,
            seq_len=config.seq_len,
            include_target=True,
            target_col=config.target_col,
            transformer=transformer,
            supervised_time_upper=TARGET_TIME_STEPS,
        )
        val_samples = _load_range_samples_in_memory(
            parquet_dir=config.train_dir,
            date_ids=val_dates,
            time_range=tr,
            feature_cols=config.feature_cols,
            seq_len=config.seq_len,
            include_target=True,
            target_col=config.target_col,
            transformer=transformer,
            supervised_time_upper=TARGET_TIME_STEPS,
        )

        if not train_samples or not val_samples:
            print("No usable train/val day samples for this range, skipping...")
            continue

        print(f"Loaded Train days: {len(train_samples)}, Val days: {len(val_samples)}")

        range_len = tr[1] - tr[0] + 1
        model = _build_model(config, time_steps=range_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        best_val_r2 = -np.inf
        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)

        for epoch in range(config.epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            shuffled_indices = np.random.permutation(len(train_samples))
            for idx in tqdm(shuffled_indices, desc=f"Train {tr} Epoch {epoch + 1}/{config.epochs}"):
                x, y_norm, _, mask = train_samples[int(idx)]

                x = x.to(device)
                y_norm = y_norm.to(device)
                mask = mask.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs, Lambda, F_prime = model(x)
                loss = _compute_neurif_loss(outputs, y_norm, x, Lambda, F_prime, mask, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                valid_n = int(mask.sum().item())
                train_loss_sum += float(loss.detach().item()) * max(1, valid_n)
                train_count += max(1, valid_n)

            mean_train_loss = train_loss_sum / max(1, train_count)

            model.eval()
            val_pred_norm_parts: list[np.ndarray] = []
            val_true_parts: list[np.ndarray] = []

            with torch.no_grad():
                for x, _, y_raw, mask in tqdm(val_samples, desc=f"Val   {tr} Epoch {epoch + 1}/{config.epochs}"):

                    x = x.to(device)
                    outputs = _unwrap_model_output(model(x))[:, :, 0].detach().cpu().numpy()
                    y_raw_np = y_raw.numpy()
                    mask_np = mask.numpy() > 0.5

                    if mask_np.any():
                        val_pred_norm_parts.append(outputs[mask_np])
                        val_true_parts.append(y_raw_np[mask_np])

            if val_pred_norm_parts:
                pred_norm_all = np.concatenate(val_pred_norm_parts)
                true_all = np.concatenate(val_true_parts)
                pred_all = transformer.inverse_transform(pred_norm_all)
                val_r2 = r2_score(true_all, pred_all)
            else:
                val_r2 = float("nan")

            print(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train MSE: {mean_train_loss:.6f} - Val R2: {val_r2:.6f}"
            )

            if np.isfinite(val_r2) and val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), model_path)
                save_transformer(transformer, trans_path)
                print(f"  [Saved] Best model for {tr} with R2: {val_r2:.6f}")

        del model, optimizer, train_samples, val_samples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def finetune_neurif_by_time(
    config: NeurIFSplitConfig,
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
        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)
        print(f"\n{'=' * 48}\nFinetuning NeurIF for Time Range: {tr[0]} - {tr[1]}\n{'=' * 48}")

        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping this range.")
            continue
        if not os.path.exists(trans_path):
            print(f"  Transformer {trans_path} not found, skipping this range.")
            continue

        transformer = load_transformer(trans_path)

        train_samples = _load_range_samples_in_memory(
            parquet_dir=config.train_dir,
            date_ids=train_dates,
            time_range=tr,
            feature_cols=config.feature_cols,
            seq_len=config.seq_len,
            include_target=True,
            target_col=config.target_col,
            transformer=transformer,
            supervised_time_upper=TARGET_TIME_STEPS,
        )
        val_samples = _load_range_samples_in_memory(
            parquet_dir=config.train_dir,
            date_ids=val_dates,
            time_range=tr,
            feature_cols=config.feature_cols,
            seq_len=config.seq_len,
            include_target=True,
            target_col=config.target_col,
            transformer=transformer,
            supervised_time_upper=TARGET_TIME_STEPS,
        )

        if not train_samples or not val_samples:
            print("  No usable train/val day samples for this range, skipping...")
            continue

        print(f"Loaded Train days: {len(train_samples)}, Val days: {len(val_samples)}")

        range_len = tr[1] - tr[0] + 1
        model = _build_model(config, time_steps=range_len).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

        baseline_val_r2 = _evaluate_val_r2(model, val_samples, transformer, device)
        best_val_r2 = baseline_val_r2
        print(f"Baseline Val R2: {baseline_val_r2:.6f}")

        for epoch in range(finetune_epochs):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            shuffled_indices = np.random.permutation(len(train_samples))
            for idx in tqdm(shuffled_indices, desc=f"Finetune {tr} Epoch {epoch + 1}/{finetune_epochs}"):
                x, y_norm, _, mask = train_samples[int(idx)]

                x = x.to(device)
                y_norm = y_norm.to(device)
                mask = mask.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs, Lambda, F_prime = model(x)
                loss = _compute_neurif_loss(outputs, y_norm, x, Lambda, F_prime, mask, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                valid_n = int(mask.sum().item())
                train_loss_sum += float(loss.detach().item()) * max(1, valid_n)
                train_count += max(1, valid_n)

            mean_train_loss = train_loss_sum / max(1, train_count)
            val_r2 = _evaluate_val_r2(model, val_samples, transformer, device)
            print(
                f"Epoch {epoch + 1}/{finetune_epochs} - "
                f"Train MSE: {mean_train_loss:.6f} - Val R2: {val_r2:.6f}"
            )

            if np.isfinite(val_r2) and val_r2 > best_val_r2:
                best_val_r2 = val_r2
                torch.save(model.state_dict(), model_path)
                print(f"  [Saved] Improved model for {tr} with R2: {val_r2:.6f}")

        print(f"Range {tr} summary - Baseline R2: {baseline_val_r2:.6f}, Best R2: {best_val_r2:.6f}")

        del model, optimizer, train_samples, val_samples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def predict_test_by_time(config: NeurIFSplitConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    test_dates = list_date_ids(config.test_dir)
    all_uids: list[str] = []
    all_preds: list[np.ndarray] = []

    for tr in TIME_RANGES:
        model_path = config.model_path(tr)
        trans_path = config.transformer_path(tr)

        print(f"Predicting NeurIF Time Range: {tr[0]} - {tr[1]}")
        if not os.path.exists(model_path):
            print(f"  Model {model_path} not found, skipping...")
            continue
        if not os.path.exists(trans_path):
            print(f"  Transformer {trans_path} not found, skipping...")
            continue

        transformer = load_transformer(trans_path)
        range_len = tr[1] - tr[0] + 1

        test_samples = _load_range_samples_in_memory(
            parquet_dir=config.test_dir,
            date_ids=test_dates,
            time_range=tr,
            feature_cols=config.feature_cols,
            seq_len=config.seq_len,
            include_target=False,
            target_col=config.target_col,
            transformer=None,
            supervised_time_upper=None,
        )

        if not test_samples:
            print("  No test day samples in this range, skipping...")
            continue

        model = _build_model(config, time_steps=range_len).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        pred_parts: list[np.ndarray] = []
        uid_parts: list[str] = []

        with torch.no_grad():
            for x, ids in tqdm(test_samples, desc=f"Test  {tr}"):

                x = x.to(device)
                outputs = _unwrap_model_output(model(x))[:, :, 0].detach().cpu().numpy().reshape(-1)
                pred = transformer.inverse_transform(outputs)

                pred_parts.append(pred)
                uid_parts.extend([f"{int(s)}|{int(d)}|{int(t)}" for s, d, t in ids])

        if pred_parts:
            all_preds.append(np.concatenate(pred_parts))
            all_uids.extend(uid_parts)
            print(f"  Predicted {len(uid_parts)} rows.")

        del model, test_samples
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_preds:
        final_preds = np.concatenate(all_preds)
        submission = pl.DataFrame({"Uid": all_uids, "prediction": final_preds})
        submission.write_csv(config.submission_path)
        print(f"Saved complete submission to {config.submission_path}")
    else:
        print("No predictions generated; submission is not written.")
