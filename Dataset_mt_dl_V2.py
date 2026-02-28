import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import os

def _list_date_ids_from_dir(parquet_dir):
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Directory not found: {parquet_dir}")

    date_ids = []
    for fn in os.listdir(parquet_dir):
        if fn.lower().endswith(".parquet"):
            stem = os.path.splitext(fn)[0]
            if stem.isdigit():
                date_ids.append(int(stem))

    date_ids.sort()
    return date_ids

class AShareParquetDataset(Dataset):
    def __init__(self, parquet_path, features_col, date_ids=None, target_col='LabelA', seq_len=5, label_means=None, label_stds=None):
        self.parquet_path = parquet_path   # 目录，如 ./data/train
        self.features = features_col
        self.target = target_col
        self.seq_len = seq_len

        if date_ids is None:
            try:
                self.date_ids = _list_date_ids_from_dir(self.parquet_path)
            except Exception as e:
                print(f"Error init dates: {e}")
                self.date_ids = []
        else:
            self.date_ids = sorted(date_ids)
            
        # === 新增：动态全局标签统计 ===
        # 如果外部传入了 means 和 stds (例如验证集使用训练集的参数)，则直接使用
        if label_means is not None and label_stds is not None:
            self.label_means = label_means
            self.label_stds = label_stds
        else:
            # 否则自行计算 (通常是 Train Dataset 实例化时触发)
            self._compute_label_stats()
            
    def _compute_label_stats(self):
        """利用 Polars 极速读取该 Dataset 包含的所有文件的全局均值和标准差"""
        print(f"Calculating global label stats for {len(self.date_ids)} days via Polars...")
        try:
            # 只选取当前 dataset 包含的文件，防止数据穿越
            files = [os.path.join(self.parquet_path, f"{int(d)}.parquet") for d in self.date_ids]
            
            # 使用 Lazy 模式并发扫描所有文件
            q = pl.scan_parquet(files)
            stats = q.select([
                pl.col("LabelA").mean().alias("mean_A"), pl.col("LabelA").std().alias("std_A"),
                pl.col("LabelB").mean().alias("mean_B"), pl.col("LabelB").std().alias("std_B"),
                pl.col("LabelC").mean().alias("mean_C"), pl.col("LabelC").std().alias("std_C"),
            ]).collect()
            
            self.label_means = [stats["mean_A"][0], stats["mean_B"][0], stats["mean_C"][0]]
            self.label_stds = [stats["std_A"][0], stats["std_B"][0], stats["std_C"][0]]
            
            print(f"Stats Computed -> Means: {self.label_means}")
            print(f"               -> Stds : {self.label_stds}")
        except Exception as e:
            print(f"Failed to compute label stats, using safe defaults. Error: {e}")
            self.label_means = [0.0, 0.0, 0.0]
            self.label_stds = [1.0, 1.0, 1.0]

    def __len__(self):
        return len(self.date_ids)

    def __getitem__(self, idx):
        current_date = self.date_ids[idx]
        
        try:
            # 1. 立即读取数据 (Eager Mode)
            day_file = os.path.join(self.parquet_path, f"{int(current_date)}.parquet")
            df = pl.read_parquet(day_file)

            if df.is_empty():
                raise ValueError(f"Date {current_date} is empty.")

            # 2. 数据清洗 
            df = df.fill_nan(0).fill_null(0)
            df = df.with_columns(
                pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            )
            
            # 3. 特征归一化 (Rank)
            count_expr = pl.col("stockid").count().over("timeid")
            rank_exprs = [
                (pl.col(feat).rank().over("timeid") / count_expr).alias(feat)
                for feat in self.features
            ]
            df = df.with_columns(rank_exprs).sort(["stockid", "timeid"])
            
            TOTAL_TIME_STEPS = 239 
            WINDOW_SIZE = self.seq_len    

            # 4. 转 Tensor 并 Reshape 
            feature_data = df.select(self.features).to_numpy().copy()
            num_stocks = df.select("stockid").n_unique()
            
            x_tensor = torch.from_numpy(feature_data).float()
            x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            x_tensor = x_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, -1)
            
            # 5. 时间维度前置 Padding 
            x_padded = F.pad(x_tensor, (0, 0, WINDOW_SIZE - 1, 0), value=0.0)
            
            # 6. 构建滑动窗口 
            x_windows = x_padded.unfold(1, WINDOW_SIZE, 1).permute(0, 1, 3, 2)
            
            # ========= 针对 Train Dataset 的 Label 处理 =========
            target_cols = ['LabelA', 'LabelB', 'LabelC']
            label_data = df.select(target_cols).to_numpy().copy()
            
            y_tensor = torch.from_numpy(label_data).float()
            
            # === 核心修改：利用广播机制进行 Z-Score 标准化 ===
            means_tensor = torch.tensor(self.label_means, dtype=torch.float32)
            stds_tensor = torch.tensor(self.label_stds, dtype=torch.float32)
            
            y_tensor = (y_tensor - means_tensor) / (stds_tensor + 1e-8)
            
            # 截断异常值，保护梯度 (限制在 5 个标准差以内)
            y_tensor = torch.clamp(y_tensor, min=-5.0, max=5.0)
            
            y_tensor = y_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, 3)
            
            return x_windows, y_tensor

        except Exception as e:
            print(f"Error loading train date {current_date}:")
            traceback.print_exc() 
            return torch.zeros(1, self.seq_len, len(self.features)), torch.zeros(1, self.seq_len, 3)
        

class AShareTestDataset(Dataset):
    def __init__(self, parquet_path, features_col, seq_len=5):
        self.parquet_path = parquet_path   # 测试集目录，如 ./data/test
        self.features = features_col
        self.seq_len = seq_len
        self.date_ids = []

        try:
            print("Scanning test date IDs...")
            self.date_ids = _list_date_ids_from_dir(self.parquet_path)
        except Exception as e:
            print(f"Error init test dates: {e}")

    def __len__(self):
        return len(self.date_ids)

    def __getitem__(self, idx):
        current_date = self.date_ids[idx]
        try:
            day_file = os.path.join(self.parquet_path, f"{int(current_date)}.parquet")
            df = pl.read_parquet(day_file)

            if df.is_empty():
                return None, None

            df = df.fill_nan(0).fill_null(0)
            df = df.with_columns(
                pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            )
            
            count_expr = pl.col("stockid").count().over("timeid")
            rank_exprs = [
                (pl.col(feat).rank().over("timeid") / count_expr).alias(feat)
                for feat in self.features
            ]
            df = df.with_columns(rank_exprs).sort(["stockid", "timeid"])
            
            id_info = df.select(["stockid", "dateid", "timeid"]).to_numpy()
        
            TOTAL_TIME_STEPS = 239 
            WINDOW_SIZE = self.seq_len    

            feature_data = df.select(self.features).to_numpy().copy()
            num_stocks = df.select("stockid").n_unique()
            
            x_tensor = torch.from_numpy(feature_data).float()
            x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            x_tensor = x_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, -1)
            
            x_padded = F.pad(x_tensor, (0, 0, WINDOW_SIZE - 1, 0), value=0.0)
            x_windows = x_padded.unfold(1, WINDOW_SIZE, 1).permute(0, 1, 3, 2)
            
            return x_windows, id_info

        except Exception as e:
            print(f"Error loading test date {current_date}:")
            traceback.print_exc()
            return None, None
        
def create_dataloaders(parquet_path, feature_cols, val_ratio=0.2, seq_len=5):
    print("Scanning date IDs for split...")
    all_dates = _list_date_ids_from_dir(parquet_path)

    total_dates = len(all_dates)
    split_idx = int(total_dates * (1 - val_ratio))

    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]

    print(f"Total Dates: {total_dates}")
    print(f"Train Dates: {len(train_dates)} ({train_dates[0]} -> {train_dates[-1]})")
    print(f"Val Dates:   {len(val_dates)} ({val_dates[0]} -> {val_dates[-1]})")

    # 1. 初始化训练集，它会自动计算全局均值和标准差
    train_dataset = AShareParquetDataset(parquet_path, feature_cols, date_ids=train_dates, seq_len=seq_len)
    
    # 2. 初始化验证集，【强制传入】训练集的统计量，防止数据穿越
    val_dataset = AShareParquetDataset(
        parquet_path, feature_cols, date_ids=val_dates, seq_len=seq_len,
        label_means=train_dataset.label_means,
        label_stds=train_dataset.label_stds
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 将统计量返回，方便后续推断使用
    return train_loader, val_loader, train_dataset.label_means, train_dataset.label_stds


def generate_submission_with_scale(model, test_parquet_path, feature_cols, seq_len=5, output_csv="submission.csv", train_parquet_path="./data/train"):
    """
    基于 Z-Score 的精确还原 (Inverse Transform)。
    需要提取训练集 LabelA 的真实均值和标准差，将预测结果从 N(0,1) 还原回原始物理分布。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(384)]
        
    print(f"\n[Inverse Transform] Extracting LabelA stats from {train_parquet_path}...")
    try:
        # 极速扫描获取 LabelA 的统计数据
        q = pl.scan_parquet(os.path.join(train_parquet_path, "*.parquet"))
        stats = q.select([
            pl.col("LabelA").mean().alias("mean_A"),
            pl.col("LabelA").std().alias("std_A")
        ]).collect()
        
        mean_A = stats["mean_A"][0]
        std_A = stats["std_A"][0]
        print(f"Extraction Successful -> Mean_A: {mean_A:.8f}, Std_A: {std_A:.8f}")
    except Exception as e:
        print(f"Extraction Failed: {e}. Falling back to default identity transform.")
        mean_A, std_A = 0.0, 1.0
    
    print(f"\nInitializing Test Dataset from {test_parquet_path}...")
    test_dataset = AShareTestDataset(test_parquet_path, feature_cols, seq_len=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    model.to(device)
    
    predictions = []
    row_ids = []
    
    print(f"Start Inference on {len(test_dataset)} days...")
    
    with torch.no_grad():
        for x, ids_numpy in tqdm(test_loader):
            if x is None: continue
            
            if x.shape[0] == 1:
                x = x.squeeze(0)
            
            if x.dim() < 3: continue

            num_stocks = x.shape[0]
            chunk_size = 64
            y_preds = []
            
            # --- 分块进行推理 ---
            for start_idx in range(0, num_stocks, chunk_size):
                x_chunk = x[start_idx : start_idx + chunk_size].to(device)
                
                y_pred_chunk = model(x_chunk)
                
                # 只提取 LabelA (第 0 维)
                y_pred_chunk_A = y_pred_chunk[:, :, 0] 
                
                y_preds.append(y_pred_chunk_A.cpu())
                
            # 拼接并展平
            y_pred = torch.cat(y_preds, dim=0)
            y_pred_np = y_pred.numpy().flatten()
            
            # === 精确数学还原 (Inverse Transform) ===
            # 将模型的 N(0,1) 预测值映射回真实收益率的波动率量级
            y_pred_np = (y_pred_np * std_A) + mean_A
            
            # === 处理 ID ===
            ids = ids_numpy.squeeze(0).numpy()
            current_ids = [
                f"{int(s)}|{int(d)}|{int(t)}" 
                for s, d, t in ids
            ]
            
            predictions.extend(y_pred_np)
            row_ids.extend(current_ids)
            
    # === 生成 CSV ===
    print("Writing submission file...")
    submission_df = pd.DataFrame({
        "row_id": row_ids,
        "prediction": predictions
    })
    
    submission_df.to_csv(output_csv, index=False, header=["Uid", "prediction"])
    print(f"Done! Saved to {output_csv} with {len(submission_df)} rows.")