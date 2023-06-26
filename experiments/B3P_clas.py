import sys
sys.path.append("..")
sys.path.append(".")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chemspace import ProjConfig, Projector, Encoder

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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


class B3P_dataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __getitem__(self, index):
        sml = self.data["SMILES"][index]
        bbb = self.data["BBB"][index]
        return {
            "SMILES": sml,
            "BBB": bbb
        }

    def __len__(self):
        return len(self.data)


def test_step(model, criterion, batch):
    x = batch["SMILES"]
    y = batch["BBB"].to(torch.float32).unsqueeze(1)
    y_hat = model(x)
    loss = criterion(y_hat, y)
    return loss.item()

def train_step(model, optimizer, criterion, x, y):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, optimizer, criterion, dataloader, num_epochs=100):
    model.train()
    for i in range(1, num_epochs+1):
        losses = []
        for b, batch in enumerate(dataloader):
            print(f"Batch: {b}", end="\r", flush=True)
            x_emb = P_sml(E_sml(batch["SMILES"]))
            y = batch['BBB'].to(torch.float32).unsqueeze(1)
            loss = train_step(model, optimizer, criterion, x_emb, y)
            losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch: {i} \n\tLoss: {avg_loss}", flush=True)
        torch.save(model.state_dict(), "B3P_classifier.pt")

if __name__ == "__main__":
    model = B3P_classifier()
    E_sml = Encoder()
    P_sml = Projector(
        input_size=384, 
        hidden_size=256, 
        output_size=512
        ) #-> output [512, 512]
    P_sml.load_state_dict(torch.load("P_sml.pt"))
    
    E_sml.model.eval()
    P_sml.eval()

    for param in E_sml.model.parameters():
       param.requires_grad = False
    for param in P_sml.parameters():
      param.requires_grad = False
        

    url = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv"

    BBBP_df = pd.read_csv(url, sep="\t")
    BBBP_df.replace(to_replace="BBB+", value=1, inplace=True)
    BBBP_df.replace(to_replace="BBB-", value=0, inplace=True)
    BBBP_df.rename(columns={"BBB+/BBB-": "BBB"}, inplace=True)

    BBBP_df = BBBP_df[['SMILES', 'BBB']]
    BBBP_df = BBBP_df.sample(frac=0.1).reset_index(drop=True)
    BBBP_ds = B3P_dataset(BBBP_df)
    
    
    split = [0.8, 0.1, 0.1]
    print(sum(split), len(BBBP_ds))
    train_df, val_df, test_df = torch.utils.data.random_split(BBBP_ds, split)

    train_dataloader = DataLoader(train_df, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_df, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_df, batch_size=32, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    train(model, optimizer, criterion, train_dataloader, num_epochs=100)


    
