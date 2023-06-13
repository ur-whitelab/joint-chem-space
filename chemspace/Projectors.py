import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

@dataclass
class ProjConfig:
  input_size  : int         = 384
  hidden_size : int         = 128
  output_size : int         = 32
  activation_function : str = "relu"


def cos_sim(v1: torch.Tensor, v2: torch.Tensor) -> float:
  return torch.nn.functional.cosine_similarity(v1, v2)


def loss_function(zi: torch.Tensor, zj: torch.Tensor) -> float:
  def sigma(E):
    '''
    Part of the EBM-NCE loss function
    https://openreview.net/pdf?id=xQUe1pOKPam: page 23
    '''
    return 1 / (torch.exp(-E)+1)

  noise = torch.randn(zi.shape)
  return -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(zi, noise))).mean() ) + \
         -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(noise, zj))).mean() )


class Projector(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 *args,
                 **kwargs) -> None:
        super(Projector, self).__init__()
        self.conv = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU() # do we want to expose this?

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = torch.squeeze(out, 1) 
        out = self.gelu(self.fc1(out))
        out = self.gelu(self.fc2(out))
        out = self.gelu(self.fc3(out))
        return out