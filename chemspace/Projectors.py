import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProjConfig:
  input_size  : int         = 384
  hidden_size : int         = 128
  output_size : int         = 256
  dropout_rate : float      = 0.2
  kernel_size : int         = 384
  activation_function : str = "relu"


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5), dim=-1)
        out = attn_weights @ v
        return out

    def init_weights(self):
        initrange = 0.1
        self.query.weight.data.uniform_(-initrange, initrange)
        self.query.bias.data.zero_()
        self.key.weight.data.uniform_(-initrange, initrange)
        self.key.bias.data.zero_()
        self.value.weight.data.uniform_(-initrange, initrange)
        self.value.bias.data.zero_()


class Projector(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.2,
                 kernel_size: Optional[int] = None,
                 *args,
                 **kwargs):
        super(Projector, self).__init__()

        # Deprecated code from when we were using a self-attention layer.
        # will be reimplemented in the future
        # if kernel_size:
        #     self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        #     # self.attention = SelfAttention(hidden_size, hidden_size)
        # else:
        #     self.attention = SelfAttention(input_size, hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU() 
        self.tanh = nn.Tanh() 
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(output_size)
        # self.init_weights()

    def forward(self, x):
        # if self.conv1:
        #     x = self.conv1(x)
        #     x = x.squeeze(-1)
        # x = self.attention(x)
        # out = self.fc1(x.flatten(1))
        out = self.fc1(x)
        out = self.dropout(self.relu(self.layer_norm1(out)))
        out = self.fc2(out)
        out = self.dropout(self.relu(self.layer_norm2(out)))
        out = self.fc3(out)
        out = self.dropout(self.relu(self.layer_norm3(out)))
        return self.tanh(out)

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()
        # self.attention.init_weights()

if __name__ == "__main__":
    config = ProjConfig(
        input_size=512,
        kernel_size=384,
        hidden_size=128,
        output_size=256,    
    )
    m = Projector(**vars(config))
    test = torch.randn(1,512,384)
    print(m(test).shape)

    test = torch.Tensor(2, 512, 384)
    P = Projector(**vars(config))
    print(P(test).shape)