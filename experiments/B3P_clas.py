import sys
sys.path.append("..")
sys.path.append(".")

from chemspace import ProjConfig, Projector, Encoder

import pandas as pd
import numpy as np
import torch
import torch.nn as nn



url = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv"

BBBP_df = pd.read_csv(url, sep="\t")
BBBP_df.replace(to_replace="BBB+", value=1, inplace=True)
BBBP_df.replace(to_replace="BBB-", value=0, inplace=True)
BBBP_df.rename(columns={"BBB+/BBB-": "BBB"}, inplace=True)

BBBP_df = BBBP_df[['SMILES', 'BBB']]


class B3P_classifier(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 hidden_size: int = 256,
                 output_size: int = 1):
        super(B3P_classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    


if __name__ == "__main__":
    model = B3P_classifier()
    E_sml = Encoder()
    P_sml = Projector(
        input_size=384, 
        hidden_size=256, 
        output_size=512
        ) #-> output [512, 512]
    P_sml.load_state_dict(torch.load("experiments/P_sml.pt"))

    model(P_sml(E_sml(["CCO", "COC"])))
