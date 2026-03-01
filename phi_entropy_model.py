import torch
from torch import nn
import torch.nn.functional as F 
import math
import random
import os
import numpy as np
import torch
import numpy as np
import random
import os

seed = 42 
random.seed(seed) 
os.environ['PYTHONHASHSEED'] = str(seed) 
np.random.seed(seed) 
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False  # 关键改动！
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + x)  # 残差连接，无正则化

class phi_entropymodel(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=256, output_dim=3, num_res_blocks=3):
        super(phi_entropymodel, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def normal_cdf(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def skew_normal_cdf(self, x, mu, sigma, alpha):
        z = (x - mu) / sigma
        denom = torch.sqrt(1 + (math.pi / 8.0) * alpha * alpha)
        return 0.5 * (1 + torch.erf((z / denom) / math.sqrt(2)))

    def forward(self, x, x2):
        """
        x: shape [N, 49]   （前49维历史序列）
        x2: shape [N]      （当前值，用于求 PMF）
        """
        x = F.relu(self.input_fc(x))
        x1 = self.res_blocks(x)
        x_out = self.output_fc(x1)

        mu = x_out[:, 0]
        sigma = F.softplus(x_out[:, 1]) + 1e-2
        alpha = torch.tanh(x_out[:,2]) * 5   # 将偏度限制在 [-5,5]

        lower = self.skew_normal_cdf(x2 - 0.5, mu, sigma, alpha)
        upper = self.skew_normal_cdf(x2 + 0.5, mu, sigma, alpha)
        pmf = (upper - lower).clamp(min=1e-9)

        return pmf, mu, sigma, x1, alpha
