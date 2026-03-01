import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerSelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                "norm": nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch, seq_len, dim)
        for layer in self.layers:
            attn_output, _ = layer["attn"](x, x, x)
            x = layer["norm"](x + attn_output)  # Add & Norm
        return x

class PointCloudTransformer(nn.Module):
    def __init__(self, ref_lidar_num,hidden_dim=64, mlp_hidden_dim=128, out_dim=1):
        super().__init__()
        self.ref_lidar_num = 2*ref_lidar_num+1
        # Project inputs to hidden dimension
        # 3-layer Attention blocks
        self.self_attn_A = MultiLayerSelfAttentionBlock(hidden_dim, num_layers=3)

        self.lstmA = torch.nn.LSTM(4, hidden_dim,3, batch_first=True)
        self.lstmC = torch.nn.LSTM(4, hidden_dim,3, batch_first=True)
        self.lstmB = nn.ModuleList([
            nn.LSTM(4, hidden_dim,3, batch_first=True) for _ in range(2*ref_lidar_num+1)
        ])
        self.linearA = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linearC = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_listB = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2*ref_lidar_num+1)
        ])
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_dim)
        )
        

    def forward(self, A, C, B):
        A_attn,_ = self.lstmA(A) 
        A_feat = self.linearA(A_attn[:, -1, :])
        C_attn,_ = self.lstmC(C) 
        C_feat = self.linearC(C_attn[:, -1, :])
        branches = torch.unbind(B, dim=1)
        features = []
        for i in range(self.ref_lidar_num):
            lstm_out, _ = self.lstmB[i](branches[i])  # [batch, 50, hidden_size]
            last_step = lstm_out[:, -1, :]  # 取最后一个时间步作为表示：[batch, hidden_size]
            fc_out = self.fc_listB[i](last_step)  # [batch, fc_output_size]
            features.append(fc_out)
        
        B_feat = torch.cat(features, dim=1)
        # Concatenate
        concat = torch.stack([A_feat, *features, C_feat], dim=1)  # (batch, 3*hidden_dim)
        concat = self.self_attn_A(concat)
        # MLP output
        concat = concat.mean(dim=1)
        out = self.mlp(concat)  # (batch, out_dim)
        return out


