import numpy as np
import pandas as pd
import torch
import random
seed = 2497
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

###### prepare data
    ### load data
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(64), tr.ToTensor()])

from torchvision.datasets import CIFAR100
D_PATH = 'c:/study25/_data/cifar100'
train_dataset = CIFAR100(D_PATH, train=True, download=True,
                        transform = transf)
test_dataset = CIFAR100(D_PATH, train=False, download=True,
                       transform = transf)

    ### make tensor dataset
from torch.utils.data import TensorDataset, DataLoader
BATCH_SIZE = 100
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

###### learn
import torch.nn as nn
import torch.optim as optim
    ### functions for learn
class DNN(nn.Module):
    def __init__(self, num_features):
        super(DNN,self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(num_features,512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128,100),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output = nn.Linear(64,10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x
def train(model, crit_, optim_, loader):
    epoch_loss, epoch_acc = 0,0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optim_.zero_grad()
        hyp_ = model(x_batch)
        loss = crit_(hyp_, y_batch)
        loss.backward()
        optim_.step()
        epoch_loss += loss.item()
        
        y_pred = torch.argmax(hyp_,1)
        acc = (y_pred==y_batch).float().mean()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)
def evaluate(model, crit_, loader):
    epoch_loss, epoch_acc = 0,0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            hyp_ = model(x_batch)
            loss = crit_(hyp_, y_batch)
            epoch_loss += loss.item()
            
            y_pred = torch.argmax(hyp_,1)
            acc = (y_pred==y_batch).float().mean()
            epoch_acc += acc
        return epoch_loss / len(loader), epoch_acc / len(loader)
    
    ### model
model = DNN(3*64*64).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
    ### fit
EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch%1==0:print('epoch {} LOSS : {} \n          acc : {}'.format(epoch,loss[0],loss[1]))

### eval / pred
    ### eval
f_loss = evaluate(model, criterion, test_loader)
print('f_loss : {}\nf_acc : {}'.format(f_loss[0], f_loss[1]))
