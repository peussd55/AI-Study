import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision.datasets import MNIST

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device : ', DEVICE)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.x = data.data.float().unsqueeze(1) / 255.0
        self.y = data.targets
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index], 
                            dtype=torch.float32),\
               torch.tensor(self.y[index],
                            dtype=torch.float32)
path = 'c:/study25/_data/torch/'
mnist_raw = MNIST(path, train=True, download=True)

dataset = CustomDataset(mnist_raw)

loader = DataLoader(dataset, batch_size = 2, shuffle=True)

for batch_index, (xb,yb) in enumerate(loader):
    print('bath : ', batch_index)
    print('x : ', xb)
    print('y : ', yb)