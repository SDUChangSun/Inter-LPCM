import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, V)
        # 汇聚成单个向量
        context = context.mean(dim=1)  # [batch, hidden_size]
        return context, attn_weights

 
 
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.batch = batch

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.attention = SelfAttention(hidden_size)

        self.linear1 = nn.Linear(hidden_size, 32) 
        self.linear2 = nn.Linear(32, output_size)

        self.linear3 = nn.Linear(input_size, 8)
        self.linear4 = nn.Linear(8, 4) 
        self.linear5 = nn.Linear(4, 1)

        self.act = nn.Sigmoid() 

    def forward(self, x, x2,x3):
        lstm_out, _ = self.lstm(x)  
        # lstm_out: [batch, seq_len, hidden_size]


        attn_out, attn_weights = self.attention(lstm_out)

        output = self.linear1(attn_out)
        output = self.linear2(output)

        x2_modified = x2.clone()
        x2_modified[:, 2] = (output / 100).squeeze()

        output2 = self.linear3(x2_modified)
        output2 = self.linear4(output2)
        output2 = self.linear5(output2)
        
        output2 = output2+x3.unsqueeze(1)
        #output2 = output2
        return output, output2
