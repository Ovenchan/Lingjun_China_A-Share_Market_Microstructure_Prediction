import torch
import torch.nn as nn
import math

class NeurIF(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=64, num_factors=10, time_steps=239, output_dim=1, num_layers=1, dropout=0.2):
        super(NeurIF, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_factors = num_factors # 原文中的 K 个潜因子 [cite: 145]
        self.time_steps = time_steps
        self.output_dim = output_dim
        
        # ==========================================
        # 1. 时序微观特征提取 (继承您的 GRU 逻辑)
        # ==========================================
        self.ln_input = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # ==========================================
        # 2. 局部因子载荷估计模块 (Local Factor Loading)
        # 对应原文中带有空间注意力(Spatial Attention)的模块 
        # ==========================================
        # 为了捕获横截面上的股票溢出效应/联动性
        self.spatial_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.ln_spatial = nn.LayerNorm(hidden_dim)
        
        # 将融合了空间特征的向量映射到因子载荷空间 (K维) [cite: 267]
        self.loading_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_factors)
        )
        
        # ==========================================
        # 3. 全局潜因子嵌入模块 (Global Factor Embedding)
        # 对应原文中随时间演变的系统性风险因子 [cite: 162, 163]
        # ==========================================
        # 初始化全局因子矩阵 F_init，大小为 [T, K]
        self.global_factor_init = nn.Parameter(torch.randn(time_steps, num_factors) * 0.02)
        
        # 注入时间位置编码并使用时间自注意力 [cite: 164, 202]
        self.temporal_attention = nn.MultiheadAttention(embed_dim=num_factors, num_heads=2, batch_first=True)
        self.ln_temporal = nn.LayerNorm(num_factors)

        # ==========================================
        # 4. 多维输出头
        # 与 QuantInformer 保持一致：支持可配置 output_dim，输出 [N, T, output_dim]
        # ==========================================
        self.output_head = nn.Linear(num_factors, output_dim)
        
    def forward(self, x):
        # x 形状: [N(Stocks), T(239), W(10), P(384)]
        N, T, W, P = x.shape
        
        # ---------------------------------------------------------
        # Step 1: GRU 特征提取 (将每个窗口压平处理)
        # ---------------------------------------------------------
        x_reshaped = x.reshape(N * T, W, P)
        x_reshaped = self.ln_input(x_reshaped)
        
        out, _ = self.gru(x_reshaped)
        
        # 取每个窗口的最后一步状态作为当期特征，形状: [N*T, hidden_dim]
        h_t = out[:, -1, :] 
        
        # 还原回时间和横截面维度: [T, N, hidden_dim]
        # 注意：这里把 T 放在前面，是为了方便在同一个时间点上算 N 只股票的 Spatial Attention
        h_t = h_t.reshape(N, T, self.hidden_dim).permute(1, 0, 2) 
        
        # ---------------------------------------------------------
        # Step 2: Spatial Attention -> 计算因子载荷 Lambda [cite: 245, 267]
        # ---------------------------------------------------------
        # 对每个时间步 t，计算 N 只股票之间的相互作用
        # 输入形状 [T, N, hidden_dim] 作为 batch (batch_size=T, seq_len=N)
        attn_out, _ = self.spatial_attention(h_t, h_t, h_t)
        
        # 残差连接 + LayerNorm [cite: 265]
        h_t = self.ln_spatial(h_t + attn_out)
        
        # 映射到因子载荷空间，再把维度换回 [N, T, K] [cite: 267]
        # Lambda 形状: [N, T, num_factors]
        Lambda = self.loading_mlp(h_t).permute(1, 0, 2) 
        
        # ---------------------------------------------------------
        # Step 3: Temporal Attention -> 计算全局潜因子 F [cite: 202, 209]
        # ---------------------------------------------------------
        # global_factor_init 形状: [T, num_factors]
        # 为了应用 MultiheadAttention，扩展为 [1, T, num_factors] (batch_size=1)
        F_t = self.global_factor_init.unsqueeze(0)
        
        # 让因子在时间维度上进行交互，捕捉宏观时序特征 [cite: 211]
        F_attn, _ = self.temporal_attention(F_t, F_t, F_t)
        
        # 残差连接 + LayerNorm -> 最终的精炼全局因子 F_prime [cite: 209]
        # 形状: [1, T, num_factors] -> 降维到 [T, num_factors]
        F_prime = self.ln_temporal(F_t + F_attn).squeeze(0)
        
        # ---------------------------------------------------------
        # Step 4: 多维收益预测 (Return Prediction)
        # ---------------------------------------------------------
        # 因子交互表征: [N, T, K]
        factor_feature = Lambda * F_prime.unsqueeze(0)

        # 输出头映射到目标维度: [N, T, output_dim]
        R_hat = self.output_head(factor_feature)
        
        # return 还可以把 Lambda 和 F_prime 传出去，用于计算损失函数的约束项
        return R_hat, Lambda, F_prime