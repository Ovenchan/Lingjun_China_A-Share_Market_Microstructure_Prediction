import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedGATLayer(nn.Module):
    """
    批处理多头图注意力层 (复现论文 Temporal Graph Attention)
    支持直接处理三维张量 [T, N, Feature_Dim]，避免在 T 维度上写 For 循环
    """
    def __init__(self, in_features, out_features, num_heads=4, alpha=0.2):
        super(BatchedGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # 权重映射
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(1, 1, num_heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(1, 1, num_heads, out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: [T, N, in_features]
        # adj: [T, N, N]
        T, N, _ = h.size()
        
        # 节点特征线性变换 -> [T, N, num_heads, out_features]
        h_prime = self.W(h).view(T, N, self.num_heads, self.out_features)
        
        # 计算节点自注意力得分
        attn_src = (h_prime * self.a_src).sum(dim=-1, keepdim=True) # [T, N, num_heads, 1]
        attn_dst = (h_prime * self.a_dst).sum(dim=-1, keepdim=True) # [T, N, num_heads, 1]
        
        # 广播相加得到注意力矩阵 (利用 LeakyReLU)
        attn_src = attn_src.permute(0, 2, 1, 3) # [T, num_heads, N, 1]
        attn_dst = attn_dst.permute(0, 2, 3, 1) # [T, num_heads, 1, N]
        e = self.leakyrelu(attn_src + attn_dst) # [T, num_heads, N, N]
        
        # 掩码机制: 将没有连边(adj=0)的权重置为极小值
        adj_expanded = adj.unsqueeze(1) # [T, 1, N, N]
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        
        # 将注意力权重施加到节点特征上进行消息聚合
        h_prime_t = h_prime.permute(0, 2, 1, 3)  # [T, num_heads, N, out_features]
        out = torch.matmul(attention, h_prime_t) # [T, num_heads, N, out_features]
        
        # 拼接多个注意力头的输出并还原形状
        out = out.permute(0, 2, 1, 3).contiguous().view(T, N, self.num_heads * self.out_features)
        return out


class BatchedHeteroAttention(nn.Module):
    """
    语义异质图注意力 (复现论文 Heterogeneous Graph Attention)
    自适应调节: 自节点(Self)、正邻居(Pos)、负邻居(Neg) 三类特征的融合比重
    """
    def __init__(self, in_features, hidden_size):
        super(BatchedHeteroAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, inputs):
        # inputs 形状: [T, N, num_relations=3, in_features]
        w = self.project(inputs)          # [T, N, 3, 1] 评估每种图特征的重要性
        beta = torch.softmax(w, dim=2)    # [T, N, 3, 1] 归一化注意力权重
        
        # 异质信息加权求和
        out = (beta * inputs).sum(dim=2)  # [T, N, in_features]
        return out


class DynamicTHGNNSpatial(nn.Module):
    """
    将原生 AdaptiveHypergraph 升级为基于论文 THGNN 的异质时空图网络
    """
    def __init__(self, feature_dim, hidden_dim, k_neighbors=3, num_heads=4):
        super(DynamicTHGNNSpatial, self).__init__()
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        
        assert hidden_dim % num_heads == 0, "Hidden_dim must be divisible by num_heads"
        gat_out_dim = hidden_dim // num_heads
        
        self.ffn = nn.Linear(feature_dim, hidden_dim)
        
        # 1. 对应论文的两阶段 GAT (Pos 与 Neg)
        self.pos_gat = BatchedGATLayer(hidden_dim, gat_out_dim, num_heads)
        self.neg_gat = BatchedGATLayer(hidden_dim, gat_out_dim, num_heads)
        
        # 2. 独立全连接映射，用于保证异质对齐
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_neg = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. 对应论文的异质语义整合模块
        self.sem_gat = BatchedHeteroAttention(hidden_dim, hidden_dim)

    def forward(self, x_cross_section):
        # x_cross_section: [T, N, F_dim] 
        T, N, F_dim = x_cross_section.shape
        
        # 节点特征映射
        x_enc = self.ffn(x_cross_section) # [T, N, HiddenDim]
        
        # ====== 构造异质截面关联图 (Cosine Similarity) ======
        x_norm = F.normalize(x_enc, p=2, dim=-1)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2)) # [T, N, N]
        
        # 正向漂移图: 选取相似度最高的 Top-K 节点 (largest=True)
        _, topk_pos_idx = torch.topk(sim_matrix, k=self.k_neighbors, dim=-1, largest=True)
        pos_adj = torch.zeros_like(sim_matrix).scatter_(-1, topk_pos_idx, 1.0)
        
        # 负向反转图: 选取相似度最低的 Bottom-K 节点 (largest=False)
        _, topk_neg_idx = torch.topk(sim_matrix, k=self.k_neighbors, dim=-1, largest=False)
        neg_adj = torch.zeros_like(sim_matrix).scatter_(-1, topk_neg_idx, 1.0)
        
        # ====== 时序图消息传递 (TGA) ======
        pos_support = self.pos_gat(x_enc, pos_adj) # [T, N, HiddenDim]
        neg_support = self.neg_gat(x_enc, neg_adj) # [T, N, HiddenDim]
        
        # ====== 异质图信息聚合 (HGA) ======
        h_self = self.mlp_self(x_enc)
        h_pos = self.mlp_pos(pos_support)
        h_neg = self.mlp_neg(neg_support)
        
        # 将自特征、正邻居聚合、负邻居聚合堆叠为 [T, N, 3, HiddenDim]
        all_embedding = torch.stack((h_self, h_pos, h_neg), dim=2)
        
        # 分配语义权重，融合输出为 [T, N, HiddenDim]
        out = self.sem_gat(all_embedding) 
        
        return out


class THGNN(nn.Module): 
    def __init__(self, input_dim=384, hidden_dim=64, num_layers=1, output_dim=1, dropout=0.5, k_neighbors=3):
        super(THGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # --- 1. Temporal Module (时间模块) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim*2, 
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_proj = nn.Linear(input_dim, hidden_dim)
        
        # --- 2. Spatial Module 升级为 THGNN ---
        # 完全替换原有的 AdaptiveHypergraph
        self.spatial_encoder = DynamicTHGNNSpatial(
            feature_dim=input_dim, 
            hidden_dim=hidden_dim, 
            k_neighbors=k_neighbors,
            num_heads=4 # 可以根据您的隐藏层维度微调，保证能整除 hidden_dim
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
        # 分支 A: 独立的时间建模 
        # ==========================================
        x_temporal = x.reshape(N * T, W, F_dim)
        t_out = self.temporal_encoder(x_temporal)[:, -1, :] 
        t_out = self.temporal_proj(t_out) # [N * T, HiddenDim]
        t_out = t_out.reshape(N, T, self.hidden_dim)
        
        # ==========================================
        # 分支 B: 融合了 THGNN 的异质动态空间建模
        # ==========================================
        x_spatial = x[:, :, -1, :].transpose(0, 1) # [T, N, F_dim]
        # 送入 THGNN 空间模块，形状无缝衔接
        s_out = self.spatial_encoder(x_spatial)    # [T, N, HiddenDim]
        s_out = s_out.transpose(0, 1)              # 还原为 [N, T, HiddenDim]
        
        # ==========================================
        # 融合层
        # ==========================================
        fusion_input = torch.cat([t_out, s_out], dim=-1) # [N, T, HiddenDim * 2]
        gate_weight = self.gate(fusion_input)            # [N, T, HiddenDim]
        
        fused_out = gate_weight * t_out + (1 - gate_weight) * s_out
        fused_out = self.dropout(fused_out)
        out = self.fc_out(fused_out) # [N, T, output_dim]
        
        return out