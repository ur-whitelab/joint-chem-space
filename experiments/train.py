import sys
sys.path.append("..")

from chemspace import ProjConfig, Projector, Encoder, loss_function

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import pandas as pd

def rmse(zi, zj):
  return torch.sqrt(torch.mean((zi-zj)**2))


# Create a dummy dataset
class DummyDataset(Dataset):
    def __init__(self,
                 num_samples: int = 100
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
                 path: str
                ) -> None:
      self.data = pd.read_csv(path)

    def __getitem__(self, index) -> tuple[str, str]:
      return (self.data["SMILES"][index], self.data["Description"][index])
    
    def __len__(self) -> int:
      return len(self.data)

def train(dataset: Dataset,
          num_epochs: int = 5,
          batch_size: int = 32,
          optimizer: optim = None,
          E1: Encoder = None,
          P1: Projector = None,
          E2: Encoder = None,
          P2: Projector = None, # Add encoders/projectors as a list? Addapt the loss to receive n encoders/projectors?
          ) -> None:
  for i in range(1, num_epochs+1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:
      
      x_sml, x_desc = batch[0], batch[1]
      z_sml = P1(E1(x_sml).to(device))
      z_desc = P2(E2(x_desc).to(device))

      loss = loss_function(z_sml, z_desc)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if not i%1:
      print (f'Epoch: {i} == Loss: {loss.item():.4f}')

if __name__ == "__main__":
  # Create two dummy projectores
  P1config = ProjConfig(input_size=384)
  P1 = Projector(**vars(P1config))
  E1 = Encoder()

  P2config = ProjConfig(input_size=384)
  P2 = Projector(**vars(P2config))
  E2 = Encoder()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # dataset = DummyDataset()
  dataset = PubChemDataset(path="../Data/PubChem500.csv")

  #Quick test
  # Same entity
  k = 0
  z1 = P1(E1(dataset[k][0]))
  z2 = P2(E2(dataset[k][1]))
  print(f" RMSE for the same entry: {rmse(z1,z2)}")
  # different entity
  z1 = P1(E1(dataset[k][0]))
  z2 = P2(E2(dataset[k+88][1]))
  print(f" RMSE for different entries: {rmse(z1,z2)}")

  # Start training
  optimizer = optim.Adam(list(P1.parameters()) + list(P2.parameters()), lr=0.001)
  train(dataset, optimizer=optimizer, E1=E1, P1=P1, E2=E2, P2=P2)

# Same entity
k = 0
z1 = P1(E1(dataset[k][0]))
z2 = P2(E2(dataset[k][1]))
print(f" RMSE for the same entry: {rmse(z1,z2)}")

# different entity
z1 = P1(E1(dataset[k][0]))
z2 = P2(E2(dataset[k+88][1]))
print(f" RMSE for different entries: {rmse(z1,z2)}")