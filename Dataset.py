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

class AShareParquetDataset(Dataset):
    def __init__(self, parquet_path, features_col, date_ids=None, target_col='LabelA', seq_len=5):
        self.parquet_path = parquet_path
        self.features = features_col
        self.target = target_col
        self.seq_len = seq_len
        
        # 获取日期列表
        if date_ids is None:
            try:
                # 这里只读取 dateid，开销很小，可以用 Lazy
                q = pl.scan_parquet(parquet_path).select("dateid").unique()
                self.date_ids = q.collect().to_series().sort().to_list()
            except Exception as e:
                print(f"Error init dates: {e}")
                self.date_ids = []
        else:
            self.date_ids = date_ids
        
    def __len__(self):
        return len(self.date_ids)

    def __getitem__(self, idx):
        current_date = self.date_ids[idx]
        
        try:
            # === 1. 立即读取数据 (Eager Mode) ===
            # 不要在这里写复杂的 with_columns，先把它读进内存
            # 这一步非常快，因为只读一天的数据
            df = (
                pl.scan_parquet(self.parquet_path)
                .filter(pl.col("dateid") == current_date)
                .collect()  # <--- 关键：立即执行，后续操作不再涉及 Lazy Graph
            )
            
            if df.is_empty():
                raise ValueError(f"Date {current_date} is empty.")

            # === 2. 数据清洗 (Eager 操作) ===
            # Eager 模式下，Polars 的操作非常稳定
            df = df.fill_nan(0).fill_null(0)
            
            # 处理 Inf
            df = df.with_columns(
                pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            )
            
            # === 3. 特征归一化 (Rank) ===
            # Eager 模式下，多列广播除法不会报错
            # 先计算分母（当天每个时间点的股票总数）
            count_expr = pl.col("stockid").count().over("timeid")
            
            # 批量生成 Rank 表达式
            rank_exprs = [
                (pl.col(feat).rank().over("timeid") / count_expr).alias(feat)
                for feat in self.features
            ]
            
            # 执行归一化
            df = df.with_columns(rank_exprs)
            
            # 排序
            df = df.sort(["stockid", "timeid"])
            
            # === 新增超参数 ===
            TOTAL_TIME_STEPS = 239 # 每天固定的 timeid 数量
            WINDOW_SIZE = self.seq_len    # 新的 seq_len (滑动窗口大小)

            # === 4. 转 Tensor ===
            feature_data = df.select(self.features).to_numpy()
            num_stocks = df.select("stockid").n_unique()
            
            x_tensor = torch.from_numpy(feature_data).float()
            x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            
            # 先恢复成全天的常规形状: [Stocks, 239, Features]
            x_tensor = x_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, -1)
            
            # === 5. 核心逻辑：时间维度前置 Padding ===
            # F.pad 对最后一个维度向前数。x_tensor 的维度是 (Stocks, Time, Features)
            # 我们要在 Time 维度 (dim=1) 的前面填充 WINDOW_SIZE - 1 个 0
            # 填充格式: (最后一维左填充, 最后一维右填充, 倒数第二维左填充, 倒数第二维右填充)
            x_padded = F.pad(x_tensor, (0, 0, WINDOW_SIZE - 1, 0), value=0.0)
            
            # === 6. 构建滑动窗口 (Unfold 魔法) ===
            # 在 Time 维度 (dim=1) 上滑动，窗口大小为 WINDOW_SIZE，步长为 1
            # 此时形状变为: [Stocks, 239, Features, WINDOW_SIZE]
            x_windows = x_padded.unfold(1, WINDOW_SIZE, 1)
            
            # 调整维度顺序以符合 LSTM/GRU 的 [Seq, Feature] 习惯
            # 最终形状: [Stocks, 239, WINDOW_SIZE, Features]
            x_windows = x_windows.permute(0, 1, 3, 2)
            
            # ========= 针对 Train Dataset 的 Label 处理 =========
            # 如果是 Train Dataset，对 y_tensor 也做相同的 Reshape (无需 Padding)
            label_data = df.select(self.target).to_numpy()
            y_tensor = torch.from_numpy(label_data).float()
            y_tensor = torch.clamp(y_tensor, min=-1.0, max=1.0)
            y_tensor = y_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, 1)
            return x_windows, y_tensor
            
            # ========= 针对 Test Dataset 的 ID 处理 =========
            # 如果是 Test Dataset，返回 x_windows 和 id_info 即可
            # return x_windows, id_info
            
            # # === 4. 转 Tensor ===
            # feature_data = df.select(self.features).to_numpy()
            # label_data = df.select(self.target).to_numpy()
            # num_stocks = df.select("stockid").n_unique()

            # x_tensor = torch.from_numpy(feature_data).float()
            # y_tensor = torch.from_numpy(label_data).float()

            # # Clamp 防止数值越界
            # x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            # y_tensor = torch.clamp(y_tensor, min=-1.0, max=1.0)

            # # Reshape
            # x_tensor = x_tensor.reshape(num_stocks, self.seq_len, -1)
            # y_tensor = y_tensor.reshape(num_stocks, self.seq_len, 1)
            
            # return x_tensor, y_tensor

        except Exception as e:
            # 打印详细报错，方便调试！
            print(f"Error loading train date {current_date}:")
            traceback.print_exc() 
            # 返回 0 占位符
            return torch.zeros(1, self.seq_len, len(self.features)), torch.zeros(1, self.seq_len, 1)
        
class AShareTestDataset(Dataset):
    def __init__(self, parquet_path, features_col, seq_len=5):
        self.parquet_path = parquet_path
        self.features = features_col
        self.seq_len = seq_len
        self.date_ids = []
        
        try:
            print("Scanning test date IDs...")
            q = pl.scan_parquet(parquet_path).select("dateid").unique()
            self.date_ids = q.collect().to_series().sort().to_list()
        except Exception as e:
            print(f"Error init test dates: {e}")

    def __len__(self):
        return len(self.date_ids)

    def __getitem__(self, idx):
        current_date = self.date_ids[idx]
        
        try:
            # === 1. 立即读取数据 (Eager Mode) ===
            df = (
                pl.scan_parquet(self.parquet_path)
                .filter(pl.col("dateid") == current_date)
                .collect() # <--- 关键：立即转化为 DataFrame
            )
            
            if df.is_empty():
                return None, None # 交给 collate_fn 处理

            # === 2. 清洗 ===
            df = df.fill_nan(0).fill_null(0)
            df = df.with_columns(
                pl.when(cs.float().is_infinite()).then(0).otherwise(cs.float()).name.keep()
            )
            
            # === 3. Rank 归一化 ===
            count_expr = pl.col("stockid").count().over("timeid")
            rank_exprs = [
                (pl.col(feat).rank().over("timeid") / count_expr).alias(feat)
                for feat in self.features
            ]
            df = df.with_columns(rank_exprs)
            
            # 排序
            df = df.sort(["stockid", "timeid"])
            
            # === 4. 提取数据 ===
            id_info = df.select(["stockid", "dateid", "timeid"]).to_numpy()
            # feature_data = df.select(self.features).to_numpy()
            # num_stocks = df.select("stockid").n_unique()
            
            # x_tensor = torch.from_numpy(feature_data).float()
            # x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            
            # x_tensor = x_tensor.reshape(num_stocks, self.seq_len, -1)
                
            # return x_tensor, id_info
        
            # === 新增超参数 ===
            TOTAL_TIME_STEPS = 239 # 每天固定的 timeid 数量
            WINDOW_SIZE = self.seq_len    # 新的 seq_len (滑动窗口大小)

            # === 4. 转 Tensor ===
            feature_data = df.select(self.features).to_numpy()
            num_stocks = df.select("stockid").n_unique()
            
            x_tensor = torch.from_numpy(feature_data).float()
            x_tensor = torch.clamp(x_tensor, min=0.0, max=1.0)
            
            # 先恢复成全天的常规形状: [Stocks, 239, Features]
            x_tensor = x_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, -1)
            
            # === 5. 核心逻辑：时间维度前置 Padding ===
            # F.pad 对最后一个维度向前数。x_tensor 的维度是 (Stocks, Time, Features)
            # 我们要在 Time 维度 (dim=1) 的前面填充 WINDOW_SIZE - 1 个 0
            # 填充格式: (最后一维左填充, 最后一维右填充, 倒数第二维左填充, 倒数第二维右填充)
            x_padded = F.pad(x_tensor, (0, 0, WINDOW_SIZE - 1, 0), value=0.0)
            
            # === 6. 构建滑动窗口 (Unfold 魔法) ===
            # 在 Time 维度 (dim=1) 上滑动，窗口大小为 WINDOW_SIZE，步长为 1
            # 此时形状变为: [Stocks, 239, Features, WINDOW_SIZE]
            x_windows = x_padded.unfold(1, WINDOW_SIZE, 1)
            
            # 调整维度顺序以符合 LSTM/GRU 的 [Seq, Feature] 习惯
            # 最终形状: [Stocks, 239, WINDOW_SIZE, Features]
            x_windows = x_windows.permute(0, 1, 3, 2)
            
            # ========= 针对 Train Dataset 的 Label 处理 =========
            # 如果是 Train Dataset，对 y_tensor 也做相同的 Reshape (无需 Padding)
            # label_data = df.select(self.target).to_numpy()
            # y_tensor = torch.from_numpy(label_data).float()
            # y_tensor = torch.clamp(y_tensor, min=-1.0, max=1.0)
            # y_tensor = y_tensor.reshape(num_stocks, TOTAL_TIME_STEPS, 1)
            # return x_windows, y_tensor
            
            # ========= 针对 Test Dataset 的 ID 处理 =========
            # 如果是 Test Dataset，返回 x_windows 和 id_info 即可
            return x_windows, id_info

        except Exception as e:
            print(f"Error loading test date {current_date}:")
            traceback.print_exc()
            return None, None
        
def create_dataloaders(parquet_path, feature_cols, val_ratio=0.2, seq_len = 5):
    # 1. 获取所有唯一的 dateid 并排序
    print("Scanning date IDs for split...")
    # 这里为了快，直接只读 dateid 列
    q = pl.scan_parquet(parquet_path).select("dateid").unique()
    all_dates = q.collect().to_series().sort().to_list()
    
    total_dates = len(all_dates)
    split_idx = int(total_dates * (1 - val_ratio))
    
    # 2. 按时间切分 (Chronological Split)
    train_dates = all_dates[:split_idx]
    val_dates = all_dates[split_idx:]
    
    print(f"Total Dates: {total_dates}")
    print(f"Train Dates: {len(train_dates)} ({train_dates[0]} -> {train_dates[-1]})")
    print(f"Val Dates:   {len(val_dates)} ({val_dates[0]} -> {val_dates[-1]})")
    
    # 3. 创建两个 Dataset 实例
    # 关键：将切分好的 dates 传进去
    train_dataset = AShareParquetDataset(parquet_path, feature_cols, date_ids=train_dates, seq_len=seq_len)
    val_dataset = AShareParquetDataset(parquet_path, feature_cols, date_ids=val_dates, seq_len=seq_len)
    
    # 4. 创建 DataLoader
    # num_workers=0 避免 Windows 下 Polars 多进程冲突
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def generate_submission(model, test_parquet_path, output_csv="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = [f"f{i}" for i in range(384)]
    
    print(f"Initializing Test Dataset from {test_parquet_path}...")
    # 关键：使用更新后的、带 Rank 归一化的 Dataset
    test_dataset = AShareTestDataset(test_parquet_path, feature_cols)
    
    # batch_size=1 代表一次处理一整天的数据
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval() # 开启评估模式
    model.to(device)
    
    predictions = []
    row_ids = []
    
    print(f"Start Inference on {len(test_dataset)} days...")
    
    with torch.no_grad(): # 关闭梯度计算
        for x, ids_numpy in tqdm(test_loader):
            if x is None: continue
            
            if x.shape[0] == 1:
                x = x.squeeze(0)
            
            if x.dim() < 3: continue

            num_stocks = x.shape[0]
            chunk_size = 64
            y_preds = []
            
            # --- 核心修改：分块进行推理 ---
            for start_idx in range(0, num_stocks, chunk_size):
                x_chunk = x[start_idx : start_idx + chunk_size].to(device)
                
                # 模型预测，预测后立即移回 CPU 以释放 GPU 显存
                y_pred_chunk = model(x_chunk)
                y_preds.append(y_pred_chunk.cpu())
                
            # 在 CPU 端拼接一整天的预测结果
            y_pred = torch.cat(y_preds, dim=0)
            
            # 展平为一维数组
            y_pred_np = y_pred.numpy().flatten()
            
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
    
    # 根据你的代码要求，header 设为 ["Uid", "prediction"]
    submission_df.to_csv(output_csv, index=False, header=["Uid", "prediction"])
    
    print(f"Done! Saved to {output_csv} with {len(submission_df)} rows.")