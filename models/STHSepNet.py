import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveHypergraph(nn.Module):
    """
    量化适配版的自适应超图卷积层
    用于捕捉横截面(Stocks)维度的隐式关联与高阶交互
    """
    def __init__(self, feature_dim, hidden_dim, k_neighbors=3):
        super(AdaptiveHypergraph, self).__init__()
        self.k_neighbors = k_neighbors
        self.ffn = nn.Linear(feature_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x_cross_section):
        # x_cross_section: [TimeSteps, Stocks, FeatureDim]
        T, N, F_dim = x_cross_section.shape
        
        # 1. 节点特征映射
        x_enc = self.ffn(x_cross_section) # [T, N, HiddenDim]
        
        # 2. 动态自适应超图构建 (基于特征相似度)
        # 计算截面上 N 个股票的余弦相似度矩阵
        x_norm = F.normalize(x_enc, p=2, dim=-1)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) # [T, N, N]
        
        # 使用 Top-K (KNN) 构造超边关联矩阵 H_adp
        # 论文 Innovation 点：通过 KNN 动态捕捉 Spatial Drift (空间漂移)
        topk_vals, topk_idx = torch.topk(sim_matrix, k=self.k_neighbors, dim=-1)
        H_adp = torch.zeros_like(sim_matrix).scatter_(-1, topk_idx, 1.0)
        
        # 3. 超图信息聚合 (Node -> Hyperedge -> Node)
        # 简化的两阶段聚合: X_out = H_adp * W * X_enc
        hyperedge_features = torch.bmm(H_adp, x_enc) # [T, N, HiddenDim]
        out = self.activation(self.W(hyperedge_features))
        
        return out

class STHSepNet(nn.Module): 
    def __init__(self, input_dim=384, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.5, k_neighbors=3):
        super(STHSepNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # --- 1. Temporal Module (时间模块 - 替代 GRU) ---
        # 论文中使用轻量化 LLM，此处以 Transformer Encoder 替代作为时序 Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim*2, 
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_proj = nn.Linear(input_dim, hidden_dim)
        
        # --- 2. Spatial Module (空间模块 - 自适应超图) ---
        self.spatial_encoder = AdaptiveHypergraph(
            feature_dim=input_dim, 
            hidden_dim=hidden_dim, 
            k_neighbors=k_neighbors
        )
        
        # --- 3. Gated Fusion Module (门控融合模块) ---
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: [Stocks(N), TimeSteps(T=239), WindowSize(W=10), Features(F=384)]
        N, T, W, F_dim = x.shape
        
        # ==========================================
        # 分支 A: 独立的时间建模 (Temporal Dynamics)
        # ==========================================
        x_temporal = x.reshape(N * T, W, F_dim)
        # 提取时序特征，取序列的最后一个时间步 (Sequence-to-One)
        t_out = self.temporal_encoder(x_temporal)[:, -1, :] 
        t_out = self.temporal_proj(t_out) # [N * T, HiddenDim]
        t_out = t_out.reshape(N, T, self.hidden_dim)
        
        # ==========================================
        # 分支 B: 独立的空间建模 (Spatial Drift via Hypergraph)
        # ==========================================
        # 取窗口内均值或最后一个时间步的特征作为横截面空间特征
        x_spatial = x[:, :, -1, :].transpose(0, 1) # [T, N, F_dim]
        s_out = self.spatial_encoder(x_spatial)    # [T, N, HiddenDim]
        s_out = s_out.transpose(0, 1)              # 还原为 [N, T, HiddenDim]
        
        # ==========================================
        # 融合层: Gated Fusion (动态集成局部时序与全局横截面)
        # ==========================================
        fusion_input = torch.cat([t_out, s_out], dim=-1) # [N, T, HiddenDim * 2]
        gate_weight = self.gate(fusion_input)            # [N, T, HiddenDim]
        
        # 动态加权
        fused_out = gate_weight * t_out + (1 - gate_weight) * s_out
        
        # 输出层
        fused_out = self.dropout(fused_out)
        out = self.fc_out(fused_out) # 形状: [N, T, output_dim]
        
        return out