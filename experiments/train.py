import sys
sys.path.append("..")

from chemspace import ProjConfig, Projector, Encoder

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import TripletMarginLoss
import pandas as pd
from typing import Optional, Union

def cos_sim(v1, v2):
  return torch.nn.functional.cosine_similarity(v1, v2)

def loss_function(zi, zj):
  def sigma(E):
    '''
    Part of the EBM-NCE loss function
    https://openreview.net/pdf?id=xQUe1pOKPam: page 23
    '''
    return 1 / (torch.exp(-E)+1)

  noise = torch.randn(256)
  return -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(zi, noise))).mean() ) + \
         -0.5 * ( torch.log(sigma(cos_sim(zi,zj))).mean() + torch.log(sigma(1-cos_sim(noise, zj))).mean() ) 

def rmse(zi, zj):
  return torch.sqrt(torch.mean((zi-zj)**2))

@torch.jit.script
def distance(x, y):
    """
    Return distance between two vectors
    """
    return torch.norm(x-y, dim=1).mean()
    # return ((x-y) ** 2).sum() ** 0.5

@torch.jit.script
def triplet_loss_function(zA, zP, zN):
  """
  Implement triplet loss function
  Args:
    zA: anchor vector
    zP: positive comparison vector, same as anchor
    zN: negative comparison vector, different from anchor
  """
  alpha = 0.0
  return (distance(zA, zP) - distance(zA, zN) + alpha)

# Create a dummy dataset
class DummyDataset(Dataset):
    def __init__(self,
                 num_samples: int = 400
                ):
        self.num_samples = num_samples
        self.data = list(zip(
            torch.randn(num_samples, 32),
            torch.randn(num_samples, 16),
          ))

    def __getitem__(self, index):
        return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return self.num_samples

class PubChemDataset(Dataset):
    def __init__(self, 
                 path: str,
                 frac: Optional[Union[float, None]] = None
                ) -> None:
      
        self.data = pd.read_csv(path)
        if frac:
            self.data = self.data.sample(frac=frac).reset_index(drop=True)

    def __getitem__(self, index) -> tuple[str, str]:
        sml = self.data["SMILES"][index]
        desc = self.data["Description"][index]
        return {
            "sml": sml,
            "desc": desc
        }
    
    def __len__(self) -> int:
        return len(self.data)

def train(dataset: Dataset,
          num_epochs: int = 50,
          batch_size: int = 32,
          optimizer: optim = None,
          E_sml: Encoder = None,
          P_sml: Projector = None,
          E_desc: Encoder = None,
          P_desc: Projector = None, # Add encoders/projectors as a list? Addapt the loss to receive n encoders/projectors?
          ) -> None:
  loss_fxn = torch.nn.TripletMarginLoss(margin=1)
  for i in range(1, num_epochs+1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    same_entry_distances = []
    different_entry_distances = []
    for batch in dataloader:
        x_sml, x_desc = batch['sml'], batch['desc']
        z_sml = P_sml(E_sml(x_sml).to(device))
        z_desc = P_desc(E_desc(x_desc).to(device))

        loss = loss_fxn(z_sml, z_desc, z_desc.roll(3, 0))

        same_entry_distance = distance(z_sml, z_desc)
        different_entry_distance = distance(z_sml, z_desc.roll(3, 0))
        same_entry_distances.append(same_entry_distance.mean().item())
        different_entry_distances.append(different_entry_distance.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average distances

    avg_same_entry_distance = sum(same_entry_distances) / len(same_entry_distances)
    avg_different_entry_distance = sum(different_entry_distances) / len(different_entry_distances)

    if not i%1:
        print (f'Epoch: {i}\n\tLoss: {loss.item():.4f}')
        print(f"\tAverage distance for the same entries: {avg_same_entry_distance}")
        print(f"\tAverage distance for different entries: {avg_different_entry_distance}")


def test(dataset: Dataset,
          batch_size: int = 32,
          E_sml: Encoder = None,
          P_sml: Projector = None,
          E_desc: Encoder = None,
          P_desc: Projector = None, # Add encoders/projectors as a list? Addapt the loss to receive n encoders/projectors?
          ) -> None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    E_sml.model.eval()
    P_sml.eval()
    E_desc.model.eval()
    P_desc.eval()
    loss_fxn = torch.nn.TripletMarginLoss(margin=1)
    same_entry_distances = []
    different_entry_distances = []
    losses = []
    for batch in dataloader:
        x_sml, x_desc = batch[0], batch[1]
        z_sml = P_sml(E_sml(x_sml).to(device))
        z_desc = P_desc(E_desc(x_desc).to(device))

        loss = loss_fxn(z_sml, z_desc, z_desc.roll(3, 0))
        same_entry_distance = distance(z_sml, z_desc)
        different_entry_distance = distance(z_sml, z_desc.roll(3, 0))

        losses.append(loss.item())
        same_entry_distances.append(same_entry_distance.mean().item())
        different_entry_distances.append(different_entry_distance.mean().item())

    avg_loss = sum(losses) / len(losses)
    avg_same_entry_distance = sum(same_entry_distances) / len(same_entry_distances)
    avg_different_entry_distance = sum(different_entry_distances) / len(different_entry_distances)
    print(f'{20*"="} Test {20*"="}\n\tLoss: {loss.item():.4f}')
    print(f"\tAverage distance for the same entries: {avg_same_entry_distance}")
    print(f"\tAverage distance for different entries: {avg_different_entry_distance}")

if __name__ == "__main__":
    P_sml_config = ProjConfig(input_size=384)
    P_sml = Projector(**vars(P_sml_config))
    E_sml = Encoder()

    P_desc_config = ProjConfig(input_size=384)
    P_desc = Projector(**vars(P_desc_config))
    E_desc = Encoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PubChemDataset(path="../chemspace/Dataset/Data/PubChem.csv", frac=0.02)
    train_data, test_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)])
    print(len(train_data), len(test_data), len(val_data))

    # Start training
    optimizer = optim.Adam(list(P_sml.parameters()) + list(P_desc.parameters()), lr=0.005)
    train(train_data, optimizer=optimizer, E_sml=E_sml, P_sml=P_sml, E_desc=E_desc, P_desc=P_desc)

    # Start testing
    # test(dataset, E_sml=E_sml, P_sml=P_sml, E_desc=E_desc, P_desc=P_desc)

