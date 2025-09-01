import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tensorflow.python.keras.datasets import mnist

class CustomDataset(Dataset):
    def __init__(self,x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return torch.tensor(self.x, dtype=torch.float32),\
               torch.tensor(self.y, dtype=torch.float32)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset = CustomDataset(x_train, y_train)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_index, (xb,yb) in enumerate(loader):
    print('bath : ', batch_index)
    print('x : ', xb)
    print('y : ', yb)