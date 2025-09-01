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

path ='./_data/torch/'

train_dataset = FashionMNIST(path, train=True, download=True)
test_dataset = FashionMNIST(path, train=False, download=True)

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
# Dataset FashionMNIST
#     Number of datapoints: 60000
#     Root location: ./_data/torch/
#     Split: Train

x_train, y_train=train_dataset.data/255. ,train_dataset.targets
x_test,y_test=train_dataset.data/255.,test_dataset.targets

print(x_train.shape) #torch.Size([60000, 28, 28])
print(y_train.shape) #torch.Size([60000])

x_train,x_test=x_train.view(-1,28*28),x_test.view(-1,28*28)

print(x_train.shape, x_test.shape)
#torch.Size([60000, 784]) torch.Size([60000, 784])

from torch.utils.data import TensorDataset, DataLoader

train_set=TensorDataset(x_train,y_train)
test_set=TensorDataset(x_test,y_test)

trian_loader=DataLoader(train_set, batch_size=32, shuffle=True)
test_loader=DataLoader(test_set, batch_size=32, shuffle=False)

#모델

class DNN(nn.moudle):
    def __init__(self,num_feature):
        super().__init__()
        
        self.hiden_layer1=nn.Sequential(
            nn.Linear(num_feature,128),
            nn.ReLU()
        )
        self.hiden_layer2=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU()
        )