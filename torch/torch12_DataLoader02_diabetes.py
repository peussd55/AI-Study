#분류1

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import random

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch:", torch.__version__,'사용:', DEVICE)

#################랜덤고정###################
seed=592
random.seed(seed) # 넘파이 랜검 고정 
torch.manual_seed(seed) # 토치 랜덤고정
torch.cuda.manual_seed(seed)  #토치 쿠다 시드 고정
##########################################

datasets = load_diabetes()
x=datasets.data
y=datasets.target 

print(x.shape) #(20640, 8)

x_train, x_test, y_train, y_test=train_test_split(
    x,
    y,
    random_state=seed, 
    train_size=0.9
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train=torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test=torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train=torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_test=torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

print(x_train.shape, y_train.shape)
# torch.Size([132, 10]) torch.Size([132])

from torch.utils.data import TensorDataset  # x,y 데이터 합치기
from torch.utils.data import DataLoader # batch 정의!!

# 1.x,y 데이이터 합침

train_set = TensorDataset(x_train, y_train) # tuple 형태 임
test_set=TensorDataset(x_test,y_test)

print(train_set)

# 2. batch 정의

train_loader= DataLoader(train_set,batch_size=32,shuffle=True)
test_loader= DataLoader(test_set,batch_size=32,shuffle=False)

#모델
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # super(Model, self).__init__() #nn.Module에 있는 model과 self 다 사용
        ### 모델에 대한 정의 부분###
        self.liner1=nn.Linear(input_dim,64)
        self.liner2=nn.Linear(64,32)
        self.liner3=nn.Linear(32,32)
        self.liner4=nn.Linear(32,16)
        self.liner5=nn.Linear(16,output_dim)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        # self.sigmoid=nn.Sigmoid()
        
    def forward(self, x): # 정의 구현 
        x=self.liner1(x)
        x=self.relu(x)        
        x=self.liner2(x)  
        x=self.relu(x)        
        x=self.liner3(x) 
        x=self.relu(x)        
        x=self.liner4(x)
        x=self.relu(x)        
        x=self.liner5(x)
        # x=self.sigmoid(x)
        
        return x
    
model=Model(10,1).to(DEVICE)
        
criterion= nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer,loader):
    #model.train(),
    total_loss=0
    for x_batch, y_batch in loader:
        
        optimizer.zero_grad()
        hypothesis=model(x_batch)
        loss=criterion(hypothesis,y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)

epochs=100
for epoch in range(1,epochs+1):
    loss=train(model, criterion,optimizer,train_loader)
    print("epoch:{}, loss:{}".format(epoch, loss))

#평가, 예측
def evalute(model, criterion,loader):   
    model.eval()
    total_loss=0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_prd=model(x_batch)
            loss2=criterion(y_prd,y_batch)
            total_loss+=loss2.item()
    return total_loss/len(loader)
    
results = evalute(model,criterion,test_loader)
print("loss:", results)

y_prd=model(x_test)

# y_prd = np.round(y_prd.detach().cpu().numpy())

y_prd=y_prd.detach().cpu().numpy()
y_test=y_test.detach().cpu().numpy()

# R2 Score 계산
r2 = r2_score(y_test, y_prd)
print(f'R2 Score (Test Set): {r2:.4f}')

# MSE (sklearn 버전) 계산 - PyTorch MSE Loss 값과 동일해야 함
mse_sklearn = mean_squared_error(y_test, y_prd)
print(f'MSE (Sklearn): {mse_sklearn:.4f}')