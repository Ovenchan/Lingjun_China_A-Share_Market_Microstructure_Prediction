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

class QuantGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1, dropout=0.5):
        super(QuantGRU, self).__init__()
        # LayerNorm 比 BatchNorm 更适合 RNN
        self.ln = nn.LayerNorm(input_dim)
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     # x: [Batch, Seq, Feat]
    #     x = self.ln(x)
    #     out, _ = self.gru(x)
    #     # 取所有时间步的输出，因为我们需要预测每一分钟
    #     out = self.dropout(out)
    #     out = self.fc(out)
    #     return out

    def forward(self, x):
        # 此时的 x 形状是滑动窗口格式: [Batch(Stocks), TimeSteps(239), WindowSize(10), Features(384)]
        batch_size, time_steps, window_size, features = x.shape
        
        # 1. 展平前两维，把所有的窗口变成独立的样本送给 GRU
        # 形状变为: [Stocks * 239, 10, 384]
        x = x.reshape(batch_size * time_steps, window_size, features)
        
        x = self.ln(x)
        out, _ = self.gru(x)
        
        # 2. Seq2One 核心：我们只需要这个窗口里【最后一个时间步】的输出状态
        # out 形状: [Stocks * 239, 10, HiddenDim] -> 取 -1 -> [Stocks * 239, HiddenDim]
        out = out[:, -1, :] 
        
        out = self.dropout(out)
        out = self.fc(out) # 形状: [Stocks * 239, 1]
        
        # 3. 还原回原始的日期维度结构，无缝衔接 Loss 计算
        # 形状还原为: [Stocks, 239, 1]
        out = out.reshape(batch_size, time_steps, 1)
        
        return out