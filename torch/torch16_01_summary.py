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

###### prepare data
    ### load data
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5),(0.50))])
from torchvision.datasets import MNIST
path = 'c:/study25/_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

    ### make data loader
from torch.utils.data import DataLoader
BATCH_SIZE=32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

###### learn
    ### model
import torch.nn as nn
import torch.optim as optim
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Dropout(0.2),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32 , kernel_size=(3,3), stride =1),           # (n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                  # (n, 32, 12, 12)
            nn.Dropout(0.2),
        )        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride =1),           # (n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),                 # (n, 16, 5, 5)
            nn.Dropout(0.2),
        )   
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear((16*5*5), 64),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output = nn.Linear(32,10)
        
    def forward(self,x):
        x = self.hidden1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x
    
model = CNN(1).to(DEVICE)
    ### summary
from torchsummary import summary
summary(model, input_size=(1,56,56), device=str(DEVICE))
from torchinfo import summary
summary(model, (32,1,56,56))
summary(model)


exit()
    
    
    ### compile
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

    ### train
def train(model, crit_, optim_, loader):
    epoch_loss = 0
    for x_bch, y_bch in loader:
        x_bch, y_bch = x_bch.to(DEVICE), y_bch.to(DEVICE)
        optim_.zero_grad()
        hyp_ = model(x_bch)
        loss = crit_(hyp_, y_bch)
        loss.backward()
        optim_.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

epochs = 50
for epoch in range(1,epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('Epoch_{} Loss : {}'.format(epoch, loss))
    
###### eval / pred
    ### eval
def evaluate(model, crit_, loader):
    model.eval()
    epoch_loss = 0
    for x_bch, y_bch in loader:
        x_bch, y_bch = x_bch.to(DEVICE), y_bch.to(DEVICE)
        with torch.no_grad():
            hyp_ = model(x_bch)
            loss = crit_(hyp_, y_bch)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

f_loss = evaluate(model, criterion, test_loader)
print('f_loss : ', f_loss)