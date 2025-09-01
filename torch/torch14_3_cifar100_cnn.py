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
PATH = 'c:/study25/_data/torch/'
train_dataset = CIFAR100(PATH, train=True, download=True,
                         transform = transf)
test_dataset = CIFAR100(PATH, train=False, download=True,
                        transform = transf)

from torch.utils.data import DataLoader
BATCH_SIZE = 100
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

###### learn
    ### functions for model
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, channel):
        super(CNN,self).__init__()
        self.CNN1 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.CNN2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.CNN3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.5)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.output = nn.Linear(256,100)
        
    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = self.CNN3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.output(x)
        return x
    
def train(model, crit_, optim_, loader):
    epoch_loss, epoch_acc = 0, 0
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
    epoch_loss, epoch_acc = 0, 0
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

model = CNN(3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

EPOCHS = 50
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch_{} LOSS : {} \n     acc : {}'.format(epoch, loss[0], loss[1]))

loss = evaluate(model, criterion, test_loader)
print('f_loss : {} \nf_acc : {}'.format(loss[0], loss[1]))