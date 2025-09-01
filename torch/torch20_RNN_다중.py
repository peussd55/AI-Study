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

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 =nn.RNN(
            input_size=1, #feature 개수, 텐서플로워에서는 input_dim  
            hidden_size=32, #output_node갯수, 텐서플로에서는 unit 
            num_layers=1, #디폴트, rnn레이어의 갯수
            batch_first=True, # 디폴트 그냥 적음
            #원래 (n,3,1) false 옵션을 주면 (3,n,1)
            #그래서 다시 true 주면 원위치된다, 머리쓰기 귀찮으니까 그냥 이 옵션 반드시 넣는다.
        ) #(N,3,32)
        # self.fc1=nn.RNN(1,32,batch_first = True)
        self.rnn_layer2=nn.RNN(32,32,batch_first=True)
        self.fc1=nn.Linear(32, 16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,1)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        x,_=self.rnn_layer1(x)
        x= self.relu(x)
        x,_=self.rnn_layer2(x)
        x= self.relu(x)
        # x=x.reshape(-1, 3*32)
        x=x.view(-1,3*32)
        # x=x[:,-1,:]
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x  

    
model=RNN().to(DEVICE)      

from torchsummary import summary

summary(model,(2,1))
# exit()
#컴파일, 훈련
