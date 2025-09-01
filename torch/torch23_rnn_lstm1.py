from torchvision.datasets import FashionMNIST
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import random
#to.Tensor=토치텐더 바꾸기+minmaxSacler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용 DEVICE', DEVICE)

#################랜덤고정###################
seed=592
random.seed(seed) # 넘파이 랜검 고정 
torch.manual_seed(seed) # 토치 랜덤고정
torch.cuda.manual_seed(seed)  #토치 쿠다 시드 고정
##########################################

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([
                [1,2,3],    
                [2,3,4],
                [3,4,5],
                [4,5,6],
                [5,6,7],
                [6,7,8],
                [7,8,9],
            ])  # 7x3. timesteps은 3
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)
x=x.reshape(x.shape[0], x.shape[1],1)

print(x.shape) #(7, 3, 1)

x=torch.tensor(x,dtype=torch.float32).to(DEVICE)
y=torch.tensor(y,dtype=torch.float32).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader

train_set= TensorDataset(x,y)
train_loader=DataLoader(train_set, batch_size=2,shuffle=True)

aaa =iter(train_loader)
bbb=next(aaa)
print(bbb)

#2 모델 

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer1 =nn.LSTM(
            input_size=1, #feature 개수, 텐서플로워에서는 input_dim  
            hidden_size=32, #output_node갯수, 텐서플로에서는 unit 
            num_layers=1, #디폴트, rnn레이어의 갯수
            batch_first=True, # 디폴트 그냥 적음
            #원래 (n,3,1) false 옵션을 주면 (3,n,1)
            #그래서 다시 true 주면 원위치된다, 머리쓰기 귀찮으니까 그냥 이 옵션 반드시 넣는다.
        ) #(N,3,32)
        # self.fc1=nn.RNN(1,32,batch_first = True)
        self.fc1=nn.Linear(32, 16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,1)
        self.relu=nn.ReLU()
        
    def forward(self, x,h0, c0):
        h0=torch.zeros(1,x.size(0),32).to(DEVICE)
        c0=torch.zeros(1,x.size(0),32).to(DEVICE)
        
        x,(hn,cn)=self.lstm_layer1(x,(h0,c0))
        x= self.relu(x)
        
        # x=x.reshape(-1, 3*32)
        x=x[:,-1,:]
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x  

    
model=LSTM().to(DEVICE)      

# from torchsummary import summary

# summary(model,(2,1))
# exit()
#컴파일, 훈련
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.0001)

from sklearn.metrics import mean_squared_error

def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        hypothesis = model(x_batch)
        hypothesis = hypothesis.view(-1)  # 예측 값을 (batch_size,)로 변경
        loss = criterion(hypothesis, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            hypothesis = hypothesis.view(-1)  # 예측 값을 (batch_size,)로 변경
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            all_preds.extend(hypothesis.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    mse = mean_squared_error(all_labels, all_preds)
    return epoch_loss / len(loader), mse
