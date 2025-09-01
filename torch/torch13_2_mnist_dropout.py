from torchvision.datasets import MNIST
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import random

path ='./_data/torch/'

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

#################랜덤고정###################
seed=592
random.seed(seed) # 넘파이 랜검 고정 
torch.manual_seed(seed) # 토치 랜덤고정
torch.cuda.manual_seed(seed)  #토치 쿠다 시드 고정
##########################################

print(train_dataset)
# torch: 2.7.1+cu118 사용: cuda
# Dataset MNIST
#     Number of datapoints: 60000
#     Root location: ./_data/torch/
#     Split: Train
print(train_dataset[0])
#(<PIL.Image.Image image mode=L size=28x28 at 0x22703712CA0>, 5)

x_train, y_train= train_dataset.data/255., train_dataset.targets
x_test, y_test= train_dataset.data/255., train_dataset.targets

print(x_train)
print(y_train)

print(x_train.size(), y_train.size())
# torch.Size([60000, 28, 28]) torch.Size([60000])

print(np.min(x_train.numpy()), np.max(x_train.numpy()))

x_train, x_test=x_train.view(-1,28*28), x_test.reshape(-1,784)
print(x_train.shape, x_test.size())
# torch.Size([60000, 784]) torch.Size([60000, 784])


from torch.utils.data import TensorDataset  # x,y 데이터 합치기
from torch.utils.data import DataLoader # batch 정의!!

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader=DataLoader(train_set, batch_size=32, shuffle=True)
test_loader=DataLoader(test_set, batch_size=32, shuffle=False)

#2 모델

class DNN(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        # super(self, DNN).__init__()
        
        self.hidden_layers1 = nn.Sequential(
            nn.Linear(num_feature,128),
            nn.ReLU()
        )
        self.hidden_layers2 = nn.Sequential(
            nn.Linear(128,64),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.hidden_layers3 = nn.Sequential(
        nn.Linear(64,32),
        nn.Dropout(0.2),
        nn.ReLU()
        )
        self.hidden_layers4 = nn.Sequential(
        nn.Linear(32,16),
        nn.ReLU()
        )
        self.hidden_layers5 = nn.Sequential(
        nn.ReLU()
        )
        self.output_layer =nn.Linear(16,10)
    def forward (self, x):
        x=self.hidden_layers1(x)
        x=self.hidden_layers2(x)
        x=self.hidden_layers3(x)
        x=self.hidden_layers4(x)
        x=self.hidden_layers5(x)
        x=self.output_layer(x)
        return x
model=DNN(784).to(DEVICE)
        
#컴파일, 훈련
criterion=nn.CrossEntropyLoss()
optimzer=optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    epoch_loss=0
    epoch_acc=0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis=model(x_batch)
        loss=criterion(hypothesis,y_batch)
        
        loss.backward()
        optimizer.step()
        
        y_prd = torch.argmax(hypothesis,1)
        acc=(y_prd == y_batch).float().mean()

        epoch_loss += loss.item()
        epoch_acc+=acc
    return epoch_loss/len(loader), epoch_acc/len(loader)
epoch=10

for epoch in range(1,epoch):
    loss, acc = train(model, criterion, optimzer, train_loader)
    print(f'epoch:{epoch}, loss:{loss:.4f}, acc:{acc:.3f}')
    
def evalutate(model, criterion, loader):
    model.eval()
    epoch_loss=0
    epoch_acc=0

    with torch.no_grad():    
        for x_batch, y_batch in loader:
            x_batch, y_batch=x_batch.to(DEVICE),y_batch.to(DEVICE)
            
            hypothesis=model(x_batch)
            loss=criterion(hypothesis,y_batch)
            
            y_prd = torch.argmax(hypothesis,1)
            acc=(y_prd == y_batch).float().mean()

            epoch_loss += loss.item()
            epoch_acc+=acc
    return epoch_loss/len(loader), epoch_acc/len(loader)

epoch=10
for epoch in range(1,epoch):
    loss, acc = train(model, criterion, optimzer, train_loader)
    val_loss, val_acc=evalutate(model, criterion,test_loader)
    
    print(f'epoch:{epoch}, loss:{loss:.4f}, acc:{acc:.3f},val_loss:{val_loss:.4f}, val_acc:{val_acc:.3f}')

#4.평가, 예측
loss, acc = evalutate(model, criterion,test_loader)
print('loss:',loss)
print('acc:',acc)