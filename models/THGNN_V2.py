import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphAttnMultiHead(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, negative_slope=0.2, residual=True, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        self.weight = Parameter(torch.empty(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.empty(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.empty(num_heads, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.project = nn.Linear(in_features, num_heads * out_features) if residual else None
        self.bias = Parameter(torch.empty(1, num_heads * out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(-1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.weight_u, -stdv, stdv)
        nn.init.uniform_(self.weight_v, -stdv, stdv)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, adj):
        # x: [N, D], adj: [N, N]
        support = x @ self.weight                         # [N, H*Dout]
        support = support.view(-1, self.num_heads, self.out_features).permute(1, 0, 2)  # [H, N, Dout]

        f1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)  # [H, 1, N]
        f2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)  # [H, N, 1]
        logits = self.leaky_relu(f1 + f2)                # [H, N, N]

        # 非边位置置为极小值
        mask = adj.unsqueeze(0).bool()                   # [1, N, N]
        logits = logits.masked_fill(~mask, -1e9)

        attn = torch.softmax(logits, dim=-1)             # [H, N, N]
        out = torch.matmul(attn, support)                # [H, N, Dout]
        out = out.permute(1, 0, 2).reshape(x.size(0), -1)

        if self.bias is not None:
            out = out + self.bias
        if self.residual:
            out = out + self.project(x)

        return out


class GraphAttnSemIndividual(nn.Module):
    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x):
        # x: [N, 3, D]
        w = self.project(x)              # [N, 3, 1]
        beta = torch.softmax(w, dim=1)
        return (beta * x).sum(dim=1)     # [N, D]


class PairNorm(nn.Module):
    def __init__(self, eps=1e-6, scale=1.0):
        super().__init__()
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        # 简化版 PN-SI
        x = x - x.mean(dim=0, keepdim=True)
        norm = (x.pow(2).sum(dim=1, keepdim=True) + self.eps).sqrt()
        return self.scale * x / norm


class QuantTHGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        output_dim=1,
        dropout=0.5,
        gat_out_dim=16,
        num_heads=4,
        pos_threshold=0.5,
        neg_threshold=0.5,
        graph_on="encoded",   # "encoded" 或 "raw"
        topk=None             # 可选，控制每个节点最多连多少边
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.graph_on = graph_on
        self.topk = topk

        self.ln = nn.LayerNorm(input_dim)

        # 先保留你原来的时序编码骨架
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=gat_out_dim,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=gat_out_dim,
            num_heads=num_heads
        )

        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(gat_out_dim * num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(gat_out_dim * num_heads, hidden_dim)

        self.sem_gat = GraphAttnSemIndividual(hidden_dim, hidden_size=hidden_dim)
        self.pn = PairNorm()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def _build_adj_from_cosine(self, feat):
        """
        feat: [N, D]
        return:
            pos_adj, neg_adj: [N, N]
        """
        feat = F.normalize(feat, p=2, dim=-1)
        sim = feat @ feat.transpose(0, 1)   # [N, N]

        # 去掉自环前先保留单位阵，后面再加回
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)

        if self.topk is not None and self.topk < sim.size(0):
            # 正边 topk
            sim_pos = sim.masked_fill(eye.bool(), -1e9)
            pos_topk_val, pos_topk_idx = torch.topk(sim_pos, k=self.topk, dim=1)
            pos_adj = torch.zeros_like(sim)
            pos_adj.scatter_(1, pos_topk_idx, (pos_topk_val > self.pos_threshold).float())

            # 负边 topk：取最小的 k 个
            sim_neg = sim.masked_fill(eye.bool(), 1e9)
            neg_topk_val, neg_topk_idx = torch.topk(-sim_neg, k=self.topk, dim=1)
            real_neg_val = -neg_topk_val
            neg_adj = torch.zeros_like(sim)
            neg_adj.scatter_(1, neg_topk_idx, (real_neg_val < -self.neg_threshold).float())
        else:
            pos_adj = (sim > self.pos_threshold).float()
            neg_adj = (sim < -self.neg_threshold).float()

        # 加自环，避免某些节点无邻居
        pos_adj = torch.maximum(pos_adj, eye)
        neg_adj = torch.maximum(neg_adj, eye)

        return pos_adj, neg_adj

    def forward(self, x):
        """
        x: [Stocks, TimeSteps, WindowSize, Features]
        out: [Stocks, TimeSteps, 1]
        """
        n_stocks, time_steps, window_size, features = x.shape

        # [N, T, W, F] -> [N*T, W, F]
        x_flat = x.reshape(n_stocks * time_steps, window_size, features)
        x_flat = self.ln(x_flat)

        # 窗口级时序编码
        gru_out, _ = self.gru(x_flat)                  # [N*T, W, H]
        h = gru_out[:, -1, :]                         # [N*T, H]

        # 还原成 [N, T, H]
        h = h.view(n_stocks, time_steps, self.hidden_dim)

        outs = []

        for t in range(time_steps):
            h_t = h[:, t, :]                          # [N, H]

            # 用什么特征构图
            if self.graph_on == "raw":
                feat_t = x[:, t, :, :].reshape(n_stocks, -1)   # [N, W*F]
            else:
                feat_t = h_t                                   # [N, H]

            pos_adj, neg_adj = self._build_adj_from_cosine(feat_t)

            pos_h = self.pos_gat(h_t, pos_adj)         # [N, H2]
            neg_h = self.neg_gat(h_t, neg_adj)         # [N, H2]

            self_h = self.mlp_self(h_t)
            pos_h = self.mlp_pos(pos_h)
            neg_h = self.mlp_neg(neg_h)

            all_h = torch.stack([self_h, pos_h, neg_h], dim=1)   # [N, 3, H]
            all_h = self.sem_gat(all_h)                          # [N, H]
            all_h = self.pn(all_h)
            all_h = self.dropout(all_h)

            out_t = self.fc(all_h)                               # [N, 1]
            outs.append(out_t)

        out = torch.stack(outs, dim=1)                           # [N, T, 1]
        return out