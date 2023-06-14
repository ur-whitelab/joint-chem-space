import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

@dataclass
class ProjConfig:
  input_size  : int         = 32
  hidden_size : int         = 16
  output_size : int         = 32
  activation_function : str = "relu"


class Projector(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 *args,
                 **kwargs
                 ):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.GELU() # do we want to expose this?

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out


def cos_sim(v1, v2):
  return torch.nn.functional.cosine_similarity(v1, v2)


def loss_function(zi, zj):
  def sigma(E):
    '''
    Part of the EBM-NCE loss function
    https://openreview.net/pdf?id=xQUe1pOKPam: page 23
    '''
    return 1 / (torch.exp(-E)+1)

  noise = torch.randn(32)
  return -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(zi, noise))).mean() ) + \
         -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(noise, zj))).mean() )
