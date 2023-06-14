import sys
sys.path.append("..")

from chemspace import ProjConfig, Projector, Encoder, loss_function

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import TripletMarginLoss
import pandas as pd

def rmse(zi, zj):
  return torch.sqrt(torch.mean((zi-zj)**2))

def triplet_loss_function(zA, zP, zN):
  """
  Implement triplet loss function
  Args:
    zA: anchor vector
    zP: positive comparison vector, same as anchor
    zN: negative comparison vector, different from anchor
  """
  def distance(x, y):
      """
      Return distance between two vectors
      """
      return ((x-y) ** 2).sum() ** 0.5
   
  alpha = 0
  distances_term = (distance(zA, zP) - distance(zA, zN) + alpha)
  return max(distances_term, torch.zeros_like(distances_term))

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
          E_sml: Encoder = None,
          P_sml: Projector = None,
          E_desc: Encoder = None,
          P_desc: Projector = None, # Add encoders/projectors as a list? Addapt the loss to receive n encoders/projectors?
          ) -> None:
  loss_fxn = TripletMarginLoss(margin=0)
  for i in range(1, num_epochs+1):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:
      
      x_sml, x_desc = batch[0], batch[1]
      z_sml = P_sml(E_sml(x_sml).to(device))
      z_desc = P_desc(E_desc(x_desc).to(device))

      loss = loss_fxn(z_sml, z_desc, torch.randn(z_sml.shape))
      

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if not i%1:
      print (f'Epoch: {i} == Loss: {loss.item():.4f}')

if __name__ == "__main__":
  # Create two dummy projectores
  P_sml_config = ProjConfig(input_size=384)
  P_sml = Projector(**vars(P_sml_config))
  E_sml = Encoder()

  P_desc_config = ProjConfig(input_size=384)
  P_desc = Projector(**vars(P_desc_config))
  E_desc = Encoder()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # dataset = DummyDataset()
  dataset = PubChemDataset(path="../chemspace/Dataset/Data/PubChem.csv")

  #Quick test
  # Same entity
  k = 0
  z1 = P_sml(E_sml(dataset[k][0]))
  z2 = P_desc(E_desc(dataset[k][1]))
  print(f" RMSE for the same entry: {rmse(z1,z2)}")
  # different entity
  z1 = P_sml(E_sml(dataset[k][0]))
  z2 = P_desc(E_desc(dataset[k+88][1]))
  print(f" RMSE for different entries: {rmse(z1,z2)}")

  # Start training
  optimizer = optim.Adam(list(P_sml.parameters()) + list(P_desc.parameters()), lr=0.001)
  train(dataset, optimizer=optimizer, E_sml=E_sml, P_sml=P_sml, E_desc=E_desc, P_desc=P_desc)

# Same entity
k = 0
z1 = P_sml(E_sml(dataset[k][0]))
z2 = P_desc(E_desc(dataset[k][1]))
print(f" RMSE for the same entry: {rmse(z1,z2)}")

# different entity
z1 = P_sml(E_sml(dataset[k][0]))
z2 = P_desc(E_desc(dataset[k+88][1]))
print(f" RMSE for different entries: {rmse(z1,z2)}")