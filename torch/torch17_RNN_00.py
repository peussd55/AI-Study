import numpy as np
import pandas as pd
import torch
import random 
seed = 2497
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#1 data
datasets = ([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
x = x.reshape(x.shape[0], x.shape[1], 1)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
print(x.shape, y.size())

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x,y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=False)

import torch.nn as nn
import torch.optim as optim
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size = 1,
            hidden_size = 32,
            num_layers = 1,
            batch_first = True
        )
        self.fc1 = nn.Linear(3*32, 16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x,_ = self.rnn(x)
        x = self.relu(x)
        x = x.reshape(-1,3*32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = RNN().to(DEVICE)

from torchinfo import summary
summary(model)