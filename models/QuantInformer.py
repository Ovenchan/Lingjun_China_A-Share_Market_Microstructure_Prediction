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

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=3):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.head_dim = embed_dim // num_heads

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)
        k = min(self.k, L)
        topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(-1, topk_idx, topk_val)
        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, E)
        return self.out(out)


class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=3):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class QuantInformer(nn.Module):
    def __init__(self, input_dim=384, embed_dim=32, patch_size=2, output_dim=1, topk=3):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.patch_size = patch_size
        
        self.local = InformerBlock(embed_dim, k=topk)
        self.global_ = InformerBlock(embed_dim, k=topk)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # 适配 DataLoader 的 4D 输入: [Batch, TimeSteps, WindowSize, Features]
        batch_size, time_steps, window_size, features = x.shape
        
        # 将所有的独立滑动窗口展平为标准的 3D 张量输入
        x = x.reshape(batch_size * time_steps, window_size, features)
        
        B, T, D = x.shape
        x = self.embed(x)

        T_truncated = (T // self.patch_size) * self.patch_size
        if T_truncated == 0:
            raise ValueError(f"window_size({T}) 必须 >= patch_size({self.patch_size})")

        x = x[:, :T_truncated, :].contiguous()
        num_patches = T_truncated // self.patch_size

        # 原来是 view，改为 reshape（更安全）
        x = x.reshape(B * num_patches, self.patch_size, -1)
        x = self.local(x)
        x = x.mean(dim=1).reshape(B, num_patches, -1)
        x = self.global_(x)

        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))
        out = out.reshape(batch_size, time_steps, -1)
        return out
