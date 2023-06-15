import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

@dataclass
class ProjConfig:
  input_size  : int         = 384
  hidden_size : int         = 256
  output_size : int         = 256
  dropout_rate : float      = 0.2
  activation_function : str = "relu"


class Projector(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.2,
                 *args,
                 **kwargs
                 ):
        super(Projector, self).__init__()
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

if __name__ == "__main__":
    m = Projector(**vars(ProjConfig()))
    test = torch.randn(1,384)
    print(m(test).shape)