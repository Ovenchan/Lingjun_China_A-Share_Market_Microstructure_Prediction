import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
from tqdm import tqdm

from gbdt_split.backends import create_backend
from gbdt_split.data import TIME_RANGES, list_date_ids
from gbdt_split.transform import load_transformer
from models.MoE import FrozenExpertMoE
from models.NeurIF import NeurIF
from models.THGNN_V2 import QuantTHGNN


FEATURE_COLS = [f"f{i}" for i in range(384)]
TOTAL_TIME_STEPS = 239
TARGET_TIME_STEPS = 229
SEQ_LEN = 10
NUM_EXPERTS = 3
EXPERT_NAMES = ("NeurIF", "THGNN_V2", "XGB")


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


def clean_day_frame(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(cs.float().fill_nan(None))
    df = df.with_columns(cs.float().fill_null(cs.float().mean().over("timeid")))
    df = df.with_columns(cs.float().fill_null(0.0))
    df = df.with_columns(
        pl.when(cs.float().is_infinite()).then(0.0).otherwise(cs.float()).name.keep()
    )
    return df


def prepare_ranked_day_frame(file_path: str, include_target: bool) -> pl.DataFrame:
    df = pl.read_parquet(file_path)
    df = clean_day_frame(df)

    count_expr = pl.col("stockid").count().over("timeid")
    rank_exprs = [
        (pl.col(col).rank().over("timeid") / count_expr).alias(col)
        for col in FEATURE_COLS
    ]
    selected = ["stockid", "dateid", "timeid"] + FEATURE_COLS
    if include_target:
        selected.append("LabelA")

    return df.with_columns(rank_exprs).sort(["stockid", "timeid"]).select(selected)


def compute_label_a_stats(train_dir: str) -> tuple[float, float]:
    stats = (
        pl.scan_parquet(os.path.join(train_dir, "*.parquet"))
        .select(
            [
                pl.col("LabelA").mean().alias("mean_A"),
                pl.col("LabelA").std().alias("std_A"),
            ]
        )
        .collect()
    )
    mean_a = float(stats["mean_A"][0])
    std_a = float(stats["std_A"][0])
    if std_a < 1e-8:
        std_a = 1.0
    return mean_a, std_a


def build_windows_from_ranked_frame(frame: pl.DataFrame, seq_len: int) -> torch.Tensor:
    feature_data = frame.select(FEATURE_COLS).to_numpy().astype(np.float32, copy=False)
    num_stocks = frame.select("stockid").n_unique()
    x_tensor = torch.from_numpy(feature_data).float()
    x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
    x_tensor = x_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, -1)

    x_padded = torch.nn.functional.pad(x_tensor, (0, 0, seq_len - 1, 0), value=0.0)
    return x_padded.unfold(1, seq_len, 1).permute(0, 1, 3, 2).contiguous()


class DeepExperts:
    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.neurif = NeurIF(
            input_dim=len(FEATURE_COLS),
            hidden_dim=64,
            num_factors=32,
            time_steps=TOTAL_TIME_STEPS,
            output_dim=3,
            num_layers=2,
            dropout=0.2,
        ).to(device)
        self.thgnn = QuantTHGNN(
            input_dim=len(FEATURE_COLS),
            hidden_dim=64,
            num_layers=2,
            output_dim=3,
            dropout=0.3,
            gat_out_dim=8,
            pos_threshold=0.6,
            neg_threshold=0.4,
            graph_on="encoded",
            topk=25,
        ).to(device)

        self.neurif.load_state_dict(
            torch.load("./model_params_NeurIF_cyx.pth", map_location=device, weights_only=True)
        )
        self.thgnn.load_state_dict(
            torch.load("./model_params_THGNN_V2.pth", map_location=device, weights_only=True)
        )
        self.neurif.eval()
        self.thgnn.eval()

        for model in (self.neurif, self.thgnn):
            for param in model.parameters():
                param.requires_grad_(False)

    def _predict_model(self, model: nn.Module, windows: torch.Tensor, chunk_size: int) -> np.ndarray:
        preds = []
        num_stocks = windows.shape[0]

        with torch.no_grad():
            for start in range(0, num_stocks, chunk_size):
                end = min(start + chunk_size, num_stocks)
                x_chunk = windows[start:end].to(self.device, non_blocking=True)
                output = model(x_chunk)
                if isinstance(output, tuple):
                    output = output[0]
                pred = output[:, :, 0].detach().cpu().numpy()
                preds.append(pred)

        return np.concatenate(preds, axis=0).reshape(-1).astype(np.float32, copy=False)

    def predict_day(self, ranked_frame: pl.DataFrame, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
        windows = build_windows_from_ranked_frame(ranked_frame, seq_len=SEQ_LEN)
        neurif_pred = self._predict_model(self.neurif, windows, chunk_size)
        thgnn_pred = self._predict_model(self.thgnn, windows, chunk_size)
        return neurif_pred, thgnn_pred


class XGBSplitExpert:
    def __init__(self) -> None:
        self.backend = create_backend("xgb")
        self.models = {}
        self.transformers = {}
        for time_range in TIME_RANGES:
            model_path = f"./model_xgb_time_{time_range[0]}_{time_range[1]}.json"
            trans_path = f"./model_xgb_time_trans_{time_range[0]}_{time_range[1]}.json"
            self.models[time_range] = self.backend.load_model(model_path)
            self.transformers[time_range] = load_transformer(trans_path)

    def predict_day(self, ranked_frame: pl.DataFrame) -> np.ndarray:
        preds = np.zeros(ranked_frame.height, dtype=np.float32)
        time_array = ranked_frame["timeid"].to_numpy()
        feature_array = ranked_frame.select(FEATURE_COLS).to_numpy().astype(np.float32, copy=False)

        for time_range in TIME_RANGES:
            mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
            if not np.any(mask):
                continue
            preds[mask] = self.backend.predict(
                self.models[time_range],
                feature_array[mask],
                FEATURE_COLS,
            )

        return preds


@dataclass
class MoEConfig:
    train_dir: str = "./data/train"
    test_dir: str = "./data/test"
    cache_dir: str = "./moe_cache"
    moe_model_path: str = "./model_params_MoE.pth"
    submission_path: str = "./submission_moe.csv"
    feature_cols: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_COLS))
    val_ratio: float = 0.2
    epochs: int = 8
    batch_size: int = 65536
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, int] = (4, )
    dropout: float = 0.2
    deep_chunk_size: int = 64
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def gate_input_dim(self) -> int:
        return len(self.feature_cols) + NUM_EXPERTS


def get_split_dates(train_dir: str, val_ratio: float) -> tuple[list[int], list[int]]:
    all_dates = list_date_ids(train_dir)
    split_idx = int(len(all_dates) * (1.0 - val_ratio))
    return all_dates[:split_idx], all_dates[split_idx:]


def iter_cache_files(cache_dir: str, split_name: str) -> list[Path]:
    split_dir = Path(cache_dir) / split_name
    return sorted(split_dir.glob("*.npz"), key=lambda path: int(path.stem))


def save_day_cache(
    cache_path: Path,
    ranked_frame: pl.DataFrame,
    neurif_pred: np.ndarray,
    thgnn_pred: np.ndarray,
    xgb_pred: np.ndarray,
    include_target: bool,
    target_mean: float,
    target_std: float,
) -> None:
    ids = ranked_frame.select(["stockid", "dateid", "timeid"]).to_numpy().astype(np.int32, copy=False)
    features = ranked_frame.select(FEATURE_COLS).to_numpy().astype(np.float32, copy=False)
    expert_preds = np.stack([neurif_pred, thgnn_pred, xgb_pred], axis=1).astype(np.float32, copy=False)

    if include_target:
        mask = ids[:, 2] < TARGET_TIME_STEPS
        ids = ids[mask]
        features = features[mask]
        expert_preds = expert_preds[mask]
        target = ranked_frame["LabelA"].to_numpy().astype(np.float32, copy=False)[mask]
        target = ((target - target_mean) / (target_std + 1e-8)).astype(np.float32, copy=False)
        target = np.clip(target, -5.0, 5.0)
        gate_inputs = np.concatenate([features, expert_preds], axis=1).astype(np.float32, copy=False)
        np.savez_compressed(
            cache_path,
            ids=ids,
            gate_inputs=gate_inputs,
            expert_preds=expert_preds,
            target=target,
        )
        return

    gate_inputs = np.concatenate([features, expert_preds], axis=1).astype(np.float32, copy=False)
    np.savez_compressed(
        cache_path,
        ids=ids,
        gate_inputs=gate_inputs,
        expert_preds=expert_preds,
    )


def build_cache_for_dates(
    config: MoEConfig,
    split_name: str,
    date_ids: Iterable[int],
    data_dir: str,
    include_target: bool,
    overwrite: bool = False,
) -> None:
    split_dir = Path(config.cache_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    target_mean, target_std = compute_label_a_stats(config.train_dir)
    device = torch.device(config.device)
    deep_experts = DeepExperts(device=device)
    xgb_expert = XGBSplitExpert()

    for date_id in tqdm(list(date_ids), desc=f"Building {split_name} cache"):
        cache_path = split_dir / f"{date_id}.npz"
        if cache_path.exists() and not overwrite:
            continue

        file_path = os.path.join(data_dir, f"{date_id}.parquet")
        ranked_frame = prepare_ranked_day_frame(file_path, include_target=include_target)
        neurif_pred, thgnn_pred = deep_experts.predict_day(ranked_frame, chunk_size=config.deep_chunk_size)
        xgb_pred = xgb_expert.predict_day(ranked_frame)
        save_day_cache(
            cache_path=cache_path,
            ranked_frame=ranked_frame,
            neurif_pred=neurif_pred,
            thgnn_pred=thgnn_pred,
            xgb_pred=xgb_pred,
            include_target=include_target,
            target_mean=target_mean,
            target_std=target_std,
        )


def build_all_caches(config: MoEConfig, overwrite: bool = False) -> None:
    train_dates, val_dates = get_split_dates(config.train_dir, config.val_ratio)
    test_dates = list_date_ids(config.test_dir)

    build_cache_for_dates(config, "train", train_dates, config.train_dir, include_target=True, overwrite=overwrite)
    build_cache_for_dates(config, "val", val_dates, config.train_dir, include_target=True, overwrite=overwrite)
    build_cache_for_dates(config, "test", test_dates, config.test_dir, include_target=False, overwrite=overwrite)


def build_day_arrays(
    ranked_frame: pl.DataFrame,
    neurif_pred: np.ndarray,
    thgnn_pred: np.ndarray,
    xgb_pred: np.ndarray,
    include_target: bool,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    ids = ranked_frame.select(["stockid", "dateid", "timeid"]).to_numpy().astype(np.int32, copy=False)
    features = ranked_frame.select(FEATURE_COLS).to_numpy().astype(np.float32, copy=False)
    expert_preds = np.stack([neurif_pred, thgnn_pred, xgb_pred], axis=1).astype(np.float32, copy=False)

    if include_target:
        mask = ids[:, 2] < TARGET_TIME_STEPS
        ids = ids[mask]
        features = features[mask]
        expert_preds = expert_preds[mask]
        target = ranked_frame["LabelA"].to_numpy().astype(np.float32, copy=False)[mask]
        target = ((target - target_mean) / (target_std + 1e-8)).astype(np.float32, copy=False)
        target = np.clip(target, -5.0, 5.0)
        gate_inputs = np.concatenate([features, expert_preds], axis=1).astype(np.float32, copy=False)
        return ids, gate_inputs, target, expert_preds

    gate_inputs = np.concatenate([features, expert_preds], axis=1).astype(np.float32, copy=False)
    return ids, gate_inputs, None, expert_preds


def build_day_arrays_from_file(
    file_path: str,
    deep_experts: DeepExperts,
    xgb_expert: XGBSplitExpert,
    include_target: bool,
    deep_chunk_size: int,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    ranked_frame = prepare_ranked_day_frame(file_path, include_target=include_target)
    neurif_pred, thgnn_pred = deep_experts.predict_day(ranked_frame, chunk_size=deep_chunk_size)
    xgb_pred = xgb_expert.predict_day(ranked_frame)
    return build_day_arrays(
        ranked_frame=ranked_frame,
        neurif_pred=neurif_pred,
        thgnn_pred=thgnn_pred,
        xgb_pred=xgb_pred,
        include_target=include_target,
        target_mean=target_mean,
        target_std=target_std,
    )


def batch_predict_moe(
    moe: FrozenExpertMoE,
    gate_inputs: np.ndarray,
    expert_preds: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    moe.eval()

    with torch.no_grad():
        for start in range(0, gate_inputs.shape[0], batch_size):
            end = min(start + batch_size, gate_inputs.shape[0])
            xb = torch.from_numpy(gate_inputs[start:end]).to(device, non_blocking=True)
            eb = torch.from_numpy(expert_preds[start:end]).to(device, non_blocking=True)
            pred, _ = moe(xb, eb)
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def train_moe(config: MoEConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_files = iter_cache_files(config.cache_dir, "train")
    val_files = iter_cache_files(config.cache_dir, "val")

    if not train_files or not val_files:
        raise FileNotFoundError("MoE cache is missing. Run cache or all mode first.")

    moe = FrozenExpertMoE(
        gate_input_dim=config.gate_input_dim,
        num_experts=NUM_EXPERTS,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(moe.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val_r2 = -math.inf

    for epoch in range(config.epochs):
        moe.train()
        random.shuffle(train_files)
        train_loss_sum = 0.0
        train_count = 0

        for cache_file in tqdm(train_files, desc=f"Train epoch {epoch + 1}"):
            data = np.load(cache_file)
            gate_inputs = data["gate_inputs"]
            expert_preds = data["expert_preds"]
            target = data["target"]

            order = np.random.permutation(gate_inputs.shape[0])
            gate_inputs = gate_inputs[order]
            expert_preds = expert_preds[order]
            target = target[order]

            for start in range(0, gate_inputs.shape[0], config.batch_size):
                end = min(start + config.batch_size, gate_inputs.shape[0])
                xb = torch.from_numpy(gate_inputs[start:end]).to(device, non_blocking=True)
                eb = torch.from_numpy(expert_preds[start:end]).to(device, non_blocking=True)
                yb = torch.from_numpy(target[start:end]).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred, _ = moe(xb, eb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=5.0)
                optimizer.step()

                batch_n = end - start
                train_loss_sum += float(loss.detach().item()) * batch_n
                train_count += batch_n

        moe.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for cache_file in tqdm(val_files, desc=f"Validate epoch {epoch + 1}"):
                data = np.load(cache_file)
                pred = batch_predict_moe(
                    moe=moe,
                    gate_inputs=data["gate_inputs"],
                    expert_preds=data["expert_preds"],
                    batch_size=config.batch_size,
                    device=device,
                )
                val_preds.append(pred)
                val_targets.append(data["target"])

        y_val = np.concatenate(val_targets, axis=0)
        y_pred = np.concatenate(val_preds, axis=0)
        val_r2 = r2_score(y_val, y_pred)
        val_mse = float(np.mean((y_val - y_pred) ** 2))
        train_mse = train_loss_sum / max(1, train_count)
        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train MSE: {train_mse:.6f} - Val MSE: {val_mse:.6f} - Val R2: {val_r2:.6f}"
        )

        if np.isfinite(val_r2) and val_r2 > best_val_r2:
            best_val_r2 = val_r2
            target_mean, target_std = compute_label_a_stats(config.train_dir)
            torch.save(
                {
                    "state_dict": moe.state_dict(),
                    "gate_input_dim": config.gate_input_dim,
                    "num_experts": NUM_EXPERTS,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "expert_names": EXPERT_NAMES,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                config.moe_model_path,
            )
            print(f"  [Saved] Improved MoE checkpoint with Val R2: {val_r2:.6f}")


def train_moe_streaming(config: MoEConfig) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_dates, val_dates = get_split_dates(config.train_dir, config.val_ratio)
    target_mean, target_std = compute_label_a_stats(config.train_dir)
    deep_experts = DeepExperts(device=device)
    xgb_expert = XGBSplitExpert()

    moe = FrozenExpertMoE(
        gate_input_dim=config.gate_input_dim,
        num_experts=NUM_EXPERTS,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(moe.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val_r2 = -math.inf

    for epoch in range(config.epochs):
        moe.train()
        shuffled_train_dates = list(train_dates)
        random.shuffle(shuffled_train_dates)
        train_loss_sum = 0.0
        train_count = 0

        for date_id in tqdm(shuffled_train_dates, desc=f"Train epoch {epoch + 1}"):
            file_path = os.path.join(config.train_dir, f"{date_id}.parquet")
            _, gate_inputs, target, expert_preds = build_day_arrays_from_file(
                file_path=file_path,
                deep_experts=deep_experts,
                xgb_expert=xgb_expert,
                include_target=True,
                deep_chunk_size=config.deep_chunk_size,
                target_mean=target_mean,
                target_std=target_std,
            )

            order = np.random.permutation(gate_inputs.shape[0])
            gate_inputs = gate_inputs[order]
            expert_preds = expert_preds[order]
            target = target[order]

            for start in range(0, gate_inputs.shape[0], config.batch_size):
                end = min(start + config.batch_size, gate_inputs.shape[0])
                xb = torch.from_numpy(gate_inputs[start:end]).to(device, non_blocking=True)
                eb = torch.from_numpy(expert_preds[start:end]).to(device, non_blocking=True)
                yb = torch.from_numpy(target[start:end]).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred, _ = moe(xb, eb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=5.0)
                optimizer.step()

                batch_n = end - start
                train_loss_sum += float(loss.detach().item()) * batch_n
                train_count += batch_n

        moe.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for date_id in tqdm(val_dates, desc=f"Validate epoch {epoch + 1}"):
                file_path = os.path.join(config.train_dir, f"{date_id}.parquet")
                _, gate_inputs, target, expert_preds = build_day_arrays_from_file(
                    file_path=file_path,
                    deep_experts=deep_experts,
                    xgb_expert=xgb_expert,
                    include_target=True,
                    deep_chunk_size=config.deep_chunk_size,
                    target_mean=target_mean,
                    target_std=target_std,
                )
                pred = batch_predict_moe(
                    moe=moe,
                    gate_inputs=gate_inputs,
                    expert_preds=expert_preds,
                    batch_size=config.batch_size,
                    device=device,
                )
                val_preds.append(pred)
                val_targets.append(target)

        y_val = np.concatenate(val_targets, axis=0)
        y_pred = np.concatenate(val_preds, axis=0)
        val_r2 = r2_score(y_val, y_pred)
        val_mse = float(np.mean((y_val - y_pred) ** 2))
        train_mse = train_loss_sum / max(1, train_count)
        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train MSE: {train_mse:.6f} - Val MSE: {val_mse:.6f} - Val R2: {val_r2:.6f}"
        )

        if np.isfinite(val_r2) and val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(
                {
                    "state_dict": moe.state_dict(),
                    "gate_input_dim": config.gate_input_dim,
                    "num_experts": NUM_EXPERTS,
                    "hidden_dims": config.hidden_dims,
                    "dropout": config.dropout,
                    "expert_names": EXPERT_NAMES,
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
                config.moe_model_path,
            )
            print(f"  [Saved] Improved MoE checkpoint with Val R2: {val_r2:.6f}")


def load_trained_moe(config: MoEConfig) -> FrozenExpertMoE:
    device = torch.device(config.device)
    checkpoint = torch.load(config.moe_model_path, map_location=device, weights_only=True)
    moe = FrozenExpertMoE(
        gate_input_dim=int(checkpoint["gate_input_dim"]),
        num_experts=int(checkpoint["num_experts"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    moe.load_state_dict(checkpoint["state_dict"])
    moe.eval()
    return moe


def predict_test(config: MoEConfig) -> None:
    test_files = iter_cache_files(config.cache_dir, "test")
    if not test_files:
        raise FileNotFoundError("Test cache is missing. Run cache or all mode first.")

    device = torch.device(config.device)
    checkpoint = torch.load(config.moe_model_path, map_location=device, weights_only=True)
    moe = load_trained_moe(config)
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    with open(config.submission_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Uid", "prediction"])

        for cache_file in tqdm(test_files, desc="Predict test"):
            data = np.load(cache_file)
            pred = batch_predict_moe(
                moe=moe,
                gate_inputs=data["gate_inputs"],
                expert_preds=data["expert_preds"],
                batch_size=config.batch_size,
                device=device,
            )
            pred = (pred * target_std + target_mean).astype(np.float32, copy=False)
            ids = data["ids"]
            for (stockid, dateid, timeid), value in zip(ids, pred):
                writer.writerow([f"{int(stockid)}|{int(dateid)}|{int(timeid)}", float(value)])

    print(f"Saved submission to {config.submission_path}")


def predict_test_streaming(config: MoEConfig) -> None:
    device = torch.device(config.device)
    checkpoint = torch.load(config.moe_model_path, map_location=device, weights_only=True)
    moe = load_trained_moe(config)
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    deep_experts = DeepExperts(device=device)
    xgb_expert = XGBSplitExpert()
    test_dates = list_date_ids(config.test_dir)

    with open(config.submission_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Uid", "prediction"])

        for date_id in tqdm(test_dates, desc="Predict test"):
            file_path = os.path.join(config.test_dir, f"{date_id}.parquet")
            ids, gate_inputs, _, expert_preds = build_day_arrays_from_file(
                file_path=file_path,
                deep_experts=deep_experts,
                xgb_expert=xgb_expert,
                include_target=False,
                deep_chunk_size=config.deep_chunk_size,
                target_mean=target_mean,
                target_std=target_std,
            )
            pred = batch_predict_moe(
                moe=moe,
                gate_inputs=gate_inputs,
                expert_preds=expert_preds,
                batch_size=config.batch_size,
                device=device,
            )
            pred = (pred * target_std + target_mean).astype(np.float32, copy=False)
            for (stockid, dateid, timeid), value in zip(ids, pred):
                writer.writerow([f"{int(stockid)}|{int(dateid)}|{int(timeid)}", float(value)])

    print(f"Saved submission to {config.submission_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-expert MoE for NeurIF / THGNN_V2 / XGB.")
    parser.add_argument(
        "--mode",
        choices=("cache", "train", "predict", "all", "train-stream", "predict-stream", "all-stream"),
        default="all-stream",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MoEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    if args.mode in {"cache", "all"}:
        build_all_caches(config, overwrite=args.overwrite_cache)
    if args.mode in {"train", "all"}:
        train_moe(config)
    if args.mode in {"predict", "all"}:
        predict_test(config)
    if args.mode in {"train-stream", "all-stream"}:
        train_moe_streaming(config)
    if args.mode in {"predict-stream", "all-stream"}:
        predict_test_streaming(config)


if __name__ == "__main__":
    main()
