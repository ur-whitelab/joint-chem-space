import sys
sys.path.append("..")

from chemspace import ProjConfig, Projector, loss_function

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


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

def train(dataset: Dataset,
          num_epochs: int = 50,
          batch_size: int = 16,
          optimizer: optim = None,
          P1: Projector = None,
          P2: Projector = None, # Add projectors as a list? Addapt the loss to receive n projectors?
          ):
  for i in range(num_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch in dataloader:
      
      x1, x2 = batch[0].to(device), batch[1].to(device)
      z1 = P1(x1)
      z2 = P2(x2)

      loss = loss_function(z1, z2)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if not i%10:
      print (f'Epoch: {i} == Loss: {loss.item():.4f}')

if __name__ == "__main__":
  # Create two dummy projectores
  P1config = ProjConfig(input_size=32)
  P1 = Projector(**vars(P1config))

  P2config = ProjConfig(input_size=16)
  P2 = Projector(**vars(P2config))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Start training
  dataset = DummyDataset()
  optimizer = optim.Adam(list(P1.parameters()) + list(P2.parameters()), lr=0.001)
  train(dataset, optimizer=optimizer, P1=P1, P2=P2)

# Same entity
k = 0
z1 = P1(dataset[k][0])
z2 = P2(dataset[k][1])
print(f" RMSE for the same entry: {rmse(z1,z2)}")

# different entity
z1 = P1(dataset[k][0])
z2 = P2(dataset[k+88][1])
print(f" RMSE for different entries: {rmse(z1,z2)}")